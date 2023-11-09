/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/RBFDivergenceReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"
#include <ADS/reconstructions.h>

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyMathOps.h"

#include "SAMRAIVectorReal.h"

#include <libmesh/explicit_system.h>

#include <utility>

namespace
{
static Timer* t_apply_reconstruction;
}

namespace ADS
{
RBFDivergenceReconstructions::RBFDivergenceReconstructions(std::string object_name, Pointer<Database> input_db)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_u_scr_var(new SideVariable<NDIM, double>(d_object_name + "::Q_scratch")),
      d_div_var(new CellVariable<NDIM, double>(d_object_name + "::Div"))
{
    d_rbf_stencil_size = input_db->getInteger("stencil_size");
    d_rbf_order = Reconstruct::string_to_enum<Reconstruct::RBFPolyOrder>(input_db->getString("rbf_order"));

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_u_scr_idx = var_db->registerVariableAndContext(
        d_u_scr_var, var_db->getContext(d_object_name + "::CTX"), std::ceil(0.5 * d_rbf_stencil_size));
    d_div_idx =
        var_db->registerVariableAndContext(d_div_var, var_db->getContext(d_object_name + "::CTX"), IntVector<NDIM>(2));

    IBTK_DO_ONCE(t_apply_reconstruction =
                     TimerManager::getManager()->getTimer("ADS::RBFDivergenceReconstruction::applyReconstruction()"););
    return;
} // RBFDivergenceReconstructions

RBFDivergenceReconstructions::~RBFDivergenceReconstructions()
{
    deallocateOperatorState();
    return;
} // ~RBFDivergenceReconstructions

void
RBFDivergenceReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
    ADS_TIMER_START(t_apply_reconstruction);
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    // TODO: What kind of physical boundary conditions should we use for advection?
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    ghost_cell_comps[0] =
        ITC(d_u_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, d_bc_coef);
    ghost_cell_comps[1] = ITC(d_cur_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    HierarchyGhostCellInterpolation hier_ghost_cells;
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(d_current_time);
    applyReconstructionLS(d_u_scr_idx, N_idx, path_idx);
    ADS_TIMER_STOP(t_apply_reconstruction);
}

void
RBFDivergenceReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                    double current_time,
                                                    double new_time)
{
    AdvectiveReconstructionOperator::allocateOperatorState(hierarchy, current_time, new_time);
    d_hierarchy = hierarchy;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_u_scr_idx)) level->allocatePatchData(d_u_scr_idx);
        if (!level->checkAllocated(d_div_idx)) level->allocatePatchData(d_div_idx);
    }
    d_is_allocated = true;
}

void
RBFDivergenceReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();
    if (!d_is_allocated) return;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_u_scr_idx)) level->deallocatePatchData(d_u_scr_idx);
    }
    d_is_allocated = false;
}

void
RBFDivergenceReconstructions::applyReconstructionLS(const int u_idx, const int div_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_ls_idx > 0);
    TBOX_ASSERT(d_new_ls_idx > 0);
#endif

    // Compute div(u) using centered differences. We'll interpolate this if we can, otherwise we'll use RBF-FD
    HierarchyMathOps hier_math_ops("hier_math_ops", d_hierarchy);
    hier_math_ops.div(d_div_idx, d_div_var, 1.0, u_idx, d_u_scr_var, nullptr, 0.0, true);
    // Fill ghost cells
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp = { ITC(d_div_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE") };
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(ghost_cell_comp, d_hierarchy);
    hier_ghost_fill.fillData(0.0);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            const hier::Index<NDIM>& idx_low = box.lower();

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CellData<NDIM, double>> div_data = patch->getPatchData(div_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);
            Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(d_new_ls_idx);
            Pointer<CellData<NDIM, double>> fd_div_data = patch->getPatchData(d_div_idx);

            auto within_regular_interpolant =
                [](const CellIndex<NDIM>& idx, NodeData<NDIM, double>& ls_data, const double ls_val) -> bool {
                for (int axis = 0; axis < NDIM; ++axis)
                {
                    SideIndex<NDIM> si(idx, axis, 0);
                    IntVector<NDIM> shft;
                    for (int i = -1; i <= 1; ++i)
                    {
                        shft(0) = i;
                        for (int j = -1; j <= 1; ++j)
                        {
                            shft(1) = j;
                            if (ADS::node_to_side(si + shft, ls_data) * ls_val < 0.0) return false;
                        }
                    }
                }
                return true;
            };

            div_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                // Only do things if ls is on the same value
                const double ls_val = ADS::node_to_cell(idx, *ls_new_data);
                IBTK::VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d);

                // Determine if we can use a standard FD stencil
                if (within_regular_interpolant(idx, *ls_data, ls_val))
                {
                    // Interpolate fd_div_data to x_loc.
                    for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d);
                    (*div_data)(idx) = Reconstruct::quadraticLagrangeInterpolantLimited(x_loc, idx, *fd_div_data);
                }
                else
                {
                    for (int d = 0; d < NDIM; ++d)
                    {
                        std::vector<VectorNd> X_pts;
                        std::vector<double> u_vals;
                        std::vector<SideIndex<NDIM>> test_idxs = { SideIndex<NDIM>(idx, d, 0),
                                                                   SideIndex<NDIM>(idx, d, 1) };
                        unsigned int i = 0;
                        while (X_pts.size() < d_rbf_stencil_size)
                        {
#ifndef NDEBUG
                            if (i >= test_idxs.size())
                            {
                                std::ostringstream err_msg;
                                err_msg << d_object_name
                                        << "::applyReconstruction(): Could not find enough cells to perform "
                                           "reconstruction.\n";
                                err_msg << "  Reconstructing on index: " << idx << " and level " << ln
                                        << " and patch num " << patch->getPatchNumber() << "\n";
                                err_msg << "  Reconstructing at point: " << x_loc.transpose() << "\n";
                                err_msg << "  ls value: " << ls_val << "\n";
                                err_msg << "  Searched " << i << " indices and found " << test_idxs.size()
                                        << " valid indices\n";
                                err_msg << "  Ls neighbor values: "
                                        << (*ls_new_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerLeft)) << " "
                                        << (*ls_new_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerRight)) << " "
                                        << (*ls_new_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperLeft)) << " "
                                        << (*ls_new_data)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperRight)) << "\n";
                                TBOX_ERROR(err_msg.str());
                            }
#endif
                            const SideIndex<NDIM>& test_idx = test_idxs[i];
                            if (ADS::node_to_side(test_idx, *ls_data) * ls_val > 0.0)
                            {
                                u_vals.push_back((*u_data)(test_idx));
                                VectorNd xpt;
                                for (int dd = 0; dd < NDIM; ++dd)
                                    xpt[dd] = static_cast<double>(test_idx(dd) - idx_low(dd)) + (dd == d ? 0.0 : 0.5);
                                X_pts.push_back(xpt);
                            }

                            // Add neighboring points to new_idxs. Prefer to add points in direction of axis.
                            IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
                            SideIndex<NDIM> idx_l(test_idx + l), idx_r(test_idx + r);
                            SideIndex<NDIM> idx_u(test_idx + u), idx_b(test_idx + b);
                            if (d == 0)
                            {
                                if (ADS::node_to_side(idx_l, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end()))
                                    test_idxs.push_back(idx_l);
                                if (ADS::node_to_side(idx_r, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end()))
                                    test_idxs.push_back(idx_r);

                                if (ADS::node_to_side(idx_u, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end()))
                                    test_idxs.push_back(idx_u);
                                if (ADS::node_to_side(idx_b, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end()))
                                    test_idxs.push_back(idx_b);
                            }
                            else
                            {
                                if (ADS::node_to_side(idx_u, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end()))
                                    test_idxs.push_back(idx_u);
                                if (ADS::node_to_side(idx_b, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end()))
                                    test_idxs.push_back(idx_b);

                                if (ADS::node_to_side(idx_l, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end()))
                                    test_idxs.push_back(idx_l);
                                if (ADS::node_to_side(idx_r, *ls_data) * ls_val > 0.0 &&
                                    (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end()))
                                    test_idxs.push_back(idx_r);
                            }
                            ++i;
                        }

                        // We have all the points. Now determine weights.
                        auto rbf = [](const double r) -> double { return r * r * r * r * r; };
                        auto L_rbf = [](const VectorNd& xi, const VectorNd& xj, void* ctx) -> double {
                            int d = *static_cast<int*>(ctx);
                            double r = (xi - xj).norm();
                            return 5.0 * (xi[d] - xj[d]) * r * r * r;
                        };

                        auto L_polys = [](const std::vector<VectorNd>& xpts,
                                          int poly_degree,
                                          double dx,
                                          VectorNd base_pt,
                                          void* ctx) -> VectorXd {
                            int d = *static_cast<int*>(ctx);
                            if (d == 0)
                                return PolynomialBasis::dPdxMonomials(xpts, poly_degree, dx, base_pt).transpose();
                            else
                                return PolynomialBasis::dPdyMonomials(xpts, poly_degree, dx, base_pt).transpose();
                        };

                        // Need to shift x_loc by idx_low
                        VectorNd new_x_loc;
                        for (int d = 0; d < NDIM; ++d) new_x_loc[d] = x_loc[d] - static_cast<double>(idx_low(d));
                        std::vector<double> wgts;
                        std::vector<double> dummy_dx = { 1.0, 1.0 };
                        Reconstruct::RBFFDReconstruct<VectorNd>(wgts,
                                                                new_x_loc,
                                                                X_pts,
                                                                d_rbf_order == Reconstruct::RBFPolyOrder::LINEAR ? 1 :
                                                                                                                   2,
                                                                dummy_dx.data(),
                                                                rbf,
                                                                L_rbf,
                                                                static_cast<void*>(&d),
                                                                L_polys,
                                                                static_cast<void*>(&d));
                        // Now perform reconstruction.
                        for (size_t i = 0; i < u_vals.size(); ++i)
                            (*div_data)(idx) = (*div_data)(idx) + u_vals[i] * wgts[i] / dx[d];
                    }
                }
            }
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
