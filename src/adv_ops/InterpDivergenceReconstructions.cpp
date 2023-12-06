/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/InterpDivergenceReconstructions.h"
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
InterpDivergenceReconstructions::InterpDivergenceReconstructions(std::string object_name, Pointer<Database> input_db)
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

    IBTK_DO_ONCE(t_apply_reconstruction = TimerManager::getManager()->getTimer(
                     "ADS::InterpDivergenceReconstruction::applyReconstruction()"););
    return;
} // InterpDivergenceReconstructions

InterpDivergenceReconstructions::~InterpDivergenceReconstructions()
{
    deallocateOperatorState();
    return;
} // ~InterpDivergenceReconstructions

void
InterpDivergenceReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
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
InterpDivergenceReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
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
InterpDivergenceReconstructions::deallocateOperatorState()
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
InterpDivergenceReconstructions::applyReconstructionLS(const int u_idx, const int div_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_ls_idx > 0);
    TBOX_ASSERT(d_new_ls_idx > 0);
#endif

    // Compute div(u) using centered differences. We'll interpolate this quantity.
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

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CellData<NDIM, double>> div_data = patch->getPatchData(div_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);
            Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(d_new_ls_idx);
            Pointer<CellData<NDIM, double>> fd_div_data = patch->getPatchData(d_div_idx);

            div_data->fillAll(0.0);

            auto within_lagrange_interpolant =
                [](const CellIndex<NDIM>& idx, NodeData<NDIM, double>& ls_data, const double ls_val) -> bool
            {
                Box<NDIM> box(idx, idx);
                box.grow(2);
                for (int axis = 0; axis < NDIM; ++axis)
                {
                    for (SideIterator<NDIM> si(box, axis); si; si++)
                    {
                        const SideIndex<NDIM>& sc = si();
                        if (ADS::node_to_side(sc, ls_data) * ls_val < 0.0) return false;
                    }
                }
                return true;
            };

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                // Only do things if ls is on the same value
                const double ls_val = ADS::node_to_cell(idx, *ls_new_data);
                IBTK::VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d);

                // Determine if we can use a standard FD stencil
                if (within_lagrange_interpolant(idx, *ls_data, ls_val))
                {
                    // Interpolate fd_div_data to x_loc.
                    (*div_data)(idx) = Reconstruct::quadratic_lagrange_interpolant_limited(x_loc, idx, *fd_div_data);
                }
                else
                {
                    // Use a finite difference stencil
                    try
                    {
                        (*div_data)(idx) = Reconstruct::radial_basis_function_reconstruction(
                            x_loc, ls_val, idx, *fd_div_data, *ls_data, patch, d_rbf_order, d_rbf_stencil_size);
                    }
                    catch (const std::runtime_error& e)
                    {
                        pout << e.what() << "\n";
                        TBOX_ERROR(d_object_name + "::applyReconstruction(): Could not perform reconstruction!\n");
                    }
                }
            }
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
