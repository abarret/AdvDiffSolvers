#include <ADS/InternalBdryFill.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/ls_functions.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/HierarchyMathOps.h>

#include <VisItDataWriter.h>

// Fortran routines
extern "C"
{
#if (NDIM == 2)
    void fast_sweep_2d_(double* U,
                        const int& U_gcw,
                        const int& ilower0,
                        const int& iupper0,
                        const int& ilower1,
                        const int& iupper1,
                        const double* dx,
                        int* v,
                        const int& v_gcw);
#endif
}

void
upwind_on_patch(Pointer<Patch<NDIM>> patch,
                const int Q_scr_idx,
                const int Q_idx,
                const int sc_idx,
                const int phi_idx,
                const double negative_fac,
                const int max_gcw,
                const double dt)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double min_ls = negative_fac * -1.0e-3 * (*std::min_element(dx, dx + NDIM));
    const double max_ls = negative_fac * max_gcw * (*std::min_element(dx, dx + NDIM));

    Pointer<CellData<NDIM, double>> Q_scr_data = patch->getPatchData(Q_scr_idx);
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
    Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(sc_idx);
    Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        const double ls = ADS::node_to_cell(idx, *phi_data);
        if ((negative_fac > 0.0 && ls > min_ls && ls < max_ls) || (negative_fac < 0.0 && ls < min_ls && ls > max_ls))
        {
            // Upwinding
            SideIndex<NDIM> idx_l(idx, 0, 0), idx_r(idx, 0, 1);
            SideIndex<NDIM> idx_b(idx, 1, 0), idx_u(idx, 1, 1);
            IntVector<NDIM> x(1, 0), y(0, 1);
            const double u_r = negative_fac * (*u_data)(idx_r);
            const double u_l = negative_fac * (*u_data)(idx_l);
            const double u_u = negative_fac * (*u_data)(idx_u);
            const double u_b = negative_fac * (*u_data)(idx_b);
            for (int d = 0; d < Q_data->getDepth(); ++d)
            {
                double diff = (std::max(u_r, 0.0) * ((*Q_scr_data)(idx, d) - (*Q_scr_data)(idx - x, d)) +
                               std::min(u_l, 0.0) * ((*Q_scr_data)(idx + x, d) - (*Q_scr_data)(idx, d))) /
                                  dx[0] +
                              (std::max(u_u, 0.0) * ((*Q_scr_data)(idx, d) - (*Q_scr_data)(idx - y, d)) +
                               std::min(u_b, 0.0) * ((*Q_scr_data)(idx + y, d) - (*Q_scr_data)(idx, d))) /
                                  dx[1];
                (*Q_data)(idx, d) = (*Q_scr_data)(idx, d) - dt * diff;
            }
        }
    }
}

namespace ADS
{
InternalBdryFill::InternalBdryFill(std::string object_name, Pointer<Database> input_db)
    : d_object_name(std::move(object_name))
{
    if (input_db)
    {
        d_tol = input_db->getDoubleWithDefault("tolerance", d_tol);
        d_max_iters = input_db->getIntegerWithDefault("max_iterations", d_max_iters);
        d_enable_logging = input_db->getBoolWithDefault("enable_logging", d_enable_logging);
        d_max_gcw = input_db->getIntegerWithDefault("max_gcw", d_max_gcw);
    }

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    std::string var_name = d_object_name + "::normal";
    if (var_db->checkVariableExists(var_name))
        d_sc_var = var_db->getVariable(var_name);
    else
        d_sc_var = new SideVariable<NDIM, double>(var_name);
    d_sc_idx = var_db->registerVariableAndContext(d_sc_var, var_db->getContext(d_object_name + "::CTX"));
}

InternalBdryFill::~InternalBdryFill()
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    var_db->removePatchDataIndex(d_sc_idx);
}

void
InternalBdryFill::advectInNormal(const std::vector<Parameters>& Q_params,
                                 const int phi_idx,
                                 Pointer<NodeVariable<NDIM, double>> phi_var,
                                 Pointer<PatchHierarchy<NDIM>> hierarchy,
                                 const double time)
{
    // Allocate velocity data
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    allocate_patch_data(d_sc_idx, hierarchy, time, coarsest_ln, finest_ln);
    // Compute the velocity for advection
    fillNormal(phi_idx, hierarchy, time);
    // Now fill in normal cells for each index
    bool converged = true;
    for (const auto& Q_param : Q_params)
    {
        converged = converged && doAdvectInNormal(Q_param, phi_idx, phi_var, hierarchy, time);
    }

    if (!converged)
    {
        writeVizFiles(Q_params, phi_idx, hierarchy, time, 0);
        if (d_error_on_non_convergence) TBOX_ERROR(d_object_name + "::advectInNormal(): Failed to converge!\n");
    }

    // Deallocate velocity data
    deallocate_patch_data(d_sc_idx, hierarchy, coarsest_ln, finest_ln);
}

void
InternalBdryFill::advectInNormal(const Parameters& Q_param,
                                 const int phi_idx,
                                 Pointer<NodeVariable<NDIM, double>> phi_var,
                                 Pointer<PatchHierarchy<NDIM>> hierarchy,
                                 const double time)
{
    std::vector<Parameters> Q_param_vec = { Q_param };
    advectInNormal(Q_param_vec, phi_idx, phi_var, hierarchy, time);
}

void
InternalBdryFill::fillNormal(const int phi_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, const double time)
{
    // First fill ghost cells for phi_idx.
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp = { ITC(phi_idx, "LINEAR_REFINE", false, "NONE", "LINEAR") };
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(ghost_cell_comp, hierarchy, coarsest_ln, finest_ln);
    hier_ghost_fill.fillData(time);
    auto fcn = [](Pointer<Patch<NDIM>> patch, const int phi_idx, const int sc_idx)
    {
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();

        Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
        Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(sc_idx);

        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (SideIterator<NDIM> si(patch->getBox(), axis); si; si++)
            {
                const SideIndex<NDIM>& idx = si();
                CellIndex<NDIM> idx_up = idx.toCell(1), idx_low = idx.toCell(0);
                VectorNd normal;
                if (axis == 0)
                {
                    normal(0) = ((*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::UpperRight)) +
                                 (*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::LowerRight)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_low, NodeIndex<NDIM>::UpperLeft)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_low, NodeIndex<NDIM>::LowerLeft))) /
                                (2.0 * dx[0]);
                    normal(1) = ((*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::UpperLeft)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::LowerLeft))) /
                                dx[1];
                    normal.normalize();
                    (*u_data)(idx) = normal(0);
                }
                else if (axis == 1)
                {
                    normal(0) = ((*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::LowerRight)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::LowerLeft))) /
                                dx[0];
                    normal(1) = ((*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::UpperRight)) +
                                 (*phi_data)(NodeIndex<NDIM>(idx_up, NodeIndex<NDIM>::UpperLeft)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_low, NodeIndex<NDIM>::LowerRight)) -
                                 (*phi_data)(NodeIndex<NDIM>(idx_low, NodeIndex<NDIM>::LowerLeft))) /
                                (2.0 * dx[1]);

                    normal.normalize();
                    (*u_data)(idx) = normal(1);
                }
                else
                    TBOX_ERROR("Unsupported dimension " << NDIM << "\n");
            }
        }
    };

    perform_on_patch_hierarchy(hierarchy, fcn, phi_idx, d_sc_idx);
}

bool
InternalBdryFill::doAdvectInNormal(const Parameters& Q_param,
                                   const int phi_idx,
                                   Pointer<NodeVariable<NDIM, double>> /*phi_var*/,
                                   Pointer<PatchHierarchy<NDIM>> hierarchy,
                                   const double time)
{
    const int Q_idx = Q_param.Q_idx;
    const Pointer<CellVariable<NDIM, double>>& Q_var = Q_param.Q_var;
    const bool negative_inside = Q_param.negative_inside;
    HierarchyMathOps hier_math_ops("hier_math_ops", hierarchy);
    const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    // Use a pseudo-time integration routine to advect
    // We need a scratch variable.
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_scr_idx = var_db->registerClonedPatchDataIndex(Q_var, Q_idx);
    allocate_patch_data(Q_scr_idx, hierarchy, time, coarsest_ln, finest_ln);

    // Now do time integration
    // We are using simple upwinding, speed has magnitude 1, so max CFL number should be 0.5.
    const double CFL_MAX = 0.5;
    // Determine the time step
    double dt = std::numeric_limits<double>::max();
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = hierarchy->getGridGeometry();
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(finest_ln);
    const IntVector<NDIM>& ratio_to_coarsest = level->getRatio();
    for (int d = 0; d < NDIM; ++d) dt = std::min(dt, grid_geom->getDx()[d] / ratio_to_coarsest[d]);
    dt *= CFL_MAX;
    // Final time
    int iter_num = 0;
    bool not_converged = true;
    double max_diff = std::numeric_limits<double>::max();

    const double negative_fac = negative_inside ? 1.0 : -1.0;
    // Iterate until we've hit a final time or we converged
    while (iter_num < d_max_iters && not_converged)
    {
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(hierarchy);
        hier_cc_data_ops.copyData(Q_scr_idx, Q_idx);

        // Fill ghost cells
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp{ ITC(Q_scr_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR") };
        HierarchyGhostCellInterpolation hier_ghost_fill;
        hier_ghost_fill.initializeOperatorState(ghost_cell_comp, hierarchy);
        hier_ghost_fill.fillData(time);

        perform_on_patch_hierarchy(
            hierarchy, upwind_on_patch, Q_scr_idx, Q_idx, d_sc_idx, phi_idx, negative_fac, d_max_gcw, dt);

        // Determine if we need another iteration
        hier_cc_data_ops.subtract(Q_scr_idx, Q_idx, Q_scr_idx);
        max_diff = hier_cc_data_ops.maxNorm(Q_scr_idx, wgt_cc_idx);
        if (max_diff <= d_tol) not_converged = false;
        ++iter_num;
    }

    if (d_enable_logging)
    {
        if (not_converged)
        {
            plog << d_object_name << ": After " << iter_num << " iterations, the solver failed to converge!\n";
            plog << d_object_name << ": Final residual tolerance was: " << max_diff << "\n";
            TBOX_WARNING("Failed to converge after " << iter_num << " iterations!\n");
        }
        else
        {
            plog << d_object_name << ": After " << iter_num << " iterations, the solver converged to a tolerance of "
                 << max_diff << "!\n";
        }
    }

    // Deallocate patch data and remove scratch index
    deallocate_patch_data(Q_scr_idx, hierarchy, coarsest_ln, finest_ln);
    var_db->removePatchDataIndex(Q_scr_idx);

    return !not_converged;
}

void
InternalBdryFill::writeVizFiles(const std::vector<Parameters>& Q_params,
                                const int phi_idx,
                                Pointer<PatchHierarchy<NDIM>> hierarchy,
                                const double time,
                                const int iter_num)
{
    pout << d_object_name << ": Writing viz files to folder: interval_fill_viz\n";
    VisItDataWriter<NDIM> viz_writer(d_object_name + "::VizWriter", "internal_fill_viz");
    viz_writer.registerPlotQuantity("PHI", "SCALAR", phi_idx);
    for (const auto& Q_param : Q_params)
    {
        const Pointer<CellVariable<NDIM, double>>& Q_var = Q_param.Q_var;
        const int Q_idx = Q_param.Q_idx;
        // Retrieve the depth. Note that this will probably fail if we are running in parallel (if a level has no
        // patches).
        Pointer<CellDataFactory<NDIM, double>> fac = Q_var->getPatchDataFactory();
        const int depth = fac->getDefaultDepth();

        switch (depth)
        {
        case 1:
            // Scalar
            viz_writer.registerPlotQuantity(Q_var->getName(), "SCALAR", Q_idx);
            break;
        case NDIM:
            // Vector
            viz_writer.registerPlotQuantity(Q_var->getName(), "VECTOR", Q_idx);
            break;
        case (NDIM * NDIM):
            // Tensor
            viz_writer.registerPlotQuantity(Q_var->getName(), "TENSOR", Q_idx);
            break;
        default:
            // Plot components separately
            for (int d = 0; d < depth; ++d)
                viz_writer.registerPlotQuantity(Q_var->getName() + "_" + std::to_string(d), "SCALAR", Q_idx, d);
            break;
        }
    }
    std::string sum_file_name = "summary.samrai";
    viz_writer.setSummaryFilename(sum_file_name);
    viz_writer.writePlotData(hierarchy, iter_num, time);
}
} // namespace ADS
