#include "ibamr/AdvDiffWavePropConvectiveOperator.h"
#include "ibamr/app_namespaces.h"

#include "QInitial.h"
#include "SAMRAIVectorReal.h"
#include "SemiLagrangianAdvIntegrator.h"
#include "utility_functions.h"

#include <Eigen/Dense>

namespace IBAMR
{
SemiLagrangianAdvIntegrator::SemiLagrangianAdvIntegrator(const std::string& object_name,
                                                         Pointer<Database> input_db,
                                                         bool register_for_restart)
    : AdvDiffHierarchyIntegrator(object_name, input_db, register_for_restart),
      d_path_var(new CellVariable<NDIM, double>(d_object_name + "::PathVar", NDIM)),
      d_vol_var(new CellVariable<NDIM, double>(d_object_name + "::VolVar")),
      d_ls_normal_var(new FaceVariable<NDIM, double>(d_object_name + "::NormalVar"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_path_idx = var_db->registerVariableAndContext(
        d_path_var, var_db->getContext(d_object_name + "::PathContext"), IntVector<NDIM>(4));
    d_xstar_idx = var_db->registerVariableAndContext(
        d_path_var, var_db->getContext(d_object_name + "::XStar"), IntVector<NDIM>(4));

    if (input_db)
    {
        d_max_iterations = input_db->getInteger("max_iterations");
        d_using_forward_integration = input_db->getBool("using_forward_integration");
    }
}

void
SemiLagrangianAdvIntegrator::registerTransportedQuantity(Pointer<CellVariable<NDIM, double>> Q_var, bool Q_output)
{
    AdvDiffHierarchyIntegrator::registerTransportedQuantity(Q_var, Q_output);
    // We need to register our own scratch variable since we need more ghost cells.
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scratch_idx = var_db->registerVariableAndContext(
        Q_var, var_db->getContext(d_object_name + "::BiggerScratch"), IntVector<NDIM>(4));
    d_Q_convec_oper = new AdvDiffWavePropConvectiveOperator(d_object_name + "::ConvectiveOp",
                                                            Q_var,
                                                            nullptr,
                                                            string_to_enum<ConvectiveDifferencingType>("ADVECTIVE"),
                                                            { nullptr });
    d_Q_R_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext(d_object_name + "::ADV_SCR"));
}

void
SemiLagrangianAdvIntegrator::registerLevelSetFunction(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                      Pointer<SetLSValue> ls_fcn)
{
    d_ls_var = ls_var;
    d_ls_fcn = ls_fcn;
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_ls_cur_idx = var_db->registerVariableAndContext(d_ls_var, getCurrentContext(), IntVector<NDIM>(1));
    d_ls_new_idx = var_db->registerVariableAndContext(d_ls_var, getNewContext(), IntVector<NDIM>(1));
    d_vol_idx = var_db->registerVariableAndContext(d_vol_var, getCurrentContext());
    d_ls_normal_idx = var_db->registerVariableAndContext(d_ls_normal_var, getCurrentContext());
}

void
SemiLagrangianAdvIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                           Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    d_hierarchy = hierarchy;
    d_gridding_alg = gridding_alg;

    AdvDiffHierarchyIntegrator::registerVariables();
    AdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    d_visit_writer->registerPlotQuantity("Volume", "SCALAR", d_vol_idx);

    d_integrator_is_initialized = true;
}

void
SemiLagrangianAdvIntegrator::initializeLevelDataSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                            const int ln,
                                                            const double data_time,
                                                            const bool /*can_be_refined*/,
                                                            bool initial_time,
                                                            Pointer<BasePatchLevel<NDIM>> old_level,
                                                            bool allocate_data)
{
    // Initialize level set
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_ls_cur_idx)) level->allocatePatchData(d_ls_cur_idx);
        if (!level->checkAllocated(d_ls_new_idx)) level->allocatePatchData(d_ls_new_idx);
        if (!level->checkAllocated(d_vol_idx)) level->allocatePatchData(d_vol_idx);
        if (!level->checkAllocated(d_ls_normal_idx)) level->allocatePatchData(d_ls_normal_idx);
    }

    d_ls_fcn->setDataOnPatchHierarchy(d_ls_cur_idx, d_ls_var, d_hierarchy, 0.0, false);
    d_ls_fcn->setDataOnPatchHierarchy(d_ls_new_idx, d_ls_var, d_hierarchy, 0.0, false);

    d_vol_fcn = new LSFindCellVolume(d_object_name + "::FindCellVolume", d_hierarchy);
    d_vol_fcn->updateVolumeAndArea(d_vol_idx, d_vol_var, IBTK::invalid_index, nullptr, d_ls_cur_idx, d_ls_var, false);

    for (const auto& Q_var : d_Q_var)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        Pointer<QInitial> Q_init = d_Q_init[Q_var];
        Q_init->setLSIndex(d_ls_cur_idx, d_vol_idx);
        Q_init->setDataOnPatchHierarchy(Q_idx, Q_var, d_hierarchy, 0.0);
    }
}

void
SemiLagrangianAdvIntegrator::applyGradientDetectorSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                              const int ln,
                                                              const double data_time,
                                                              const int tag_index,
                                                              const bool initial_time,
                                                              const bool /*uses_richardson_extrapolation_too*/)
{
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CellData<NDIM, int>> tag_data = patch->getPatchData(tag_index);
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_cur_idx);
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double ls = node_to_cell(idx, *ls_data);
            if (ls < 5.0 * dx[0]) (*tag_data)(idx) = 1;
        }
    }
}

int
SemiLagrangianAdvIntegrator::getNumberOfCycles() const
{
    return 1;
}

void
SemiLagrangianAdvIntegrator::regridHierarchyEndSpecialized()
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_path_idx)) level->allocatePatchData(d_path_idx);
        if (!level->checkAllocated(d_xstar_idx)) level->allocatePatchData(d_xstar_idx);
        if (!level->checkAllocated(d_ls_cur_idx)) level->allocatePatchData(d_ls_cur_idx);
        if (!level->checkAllocated(d_ls_new_idx)) level->allocatePatchData(d_ls_new_idx);
        if (!level->checkAllocated(d_vol_idx)) level->allocatePatchData(d_vol_idx);
        if (!level->checkAllocated(d_ls_normal_idx)) level->allocatePatchData(d_ls_normal_idx);
    }
}

void
SemiLagrangianAdvIntegrator::setupPlotDataSpecialized()
{
    IBTK_DO_ONCE(d_visit_writer->registerPlotQuantity("Path", "VECTOR", d_path_idx);
                 d_visit_writer->registerPlotQuantity("XStar", "VECTOR", d_path_idx);
                 d_visit_writer->registerPlotQuantity("LS current", "SCALAR", d_ls_cur_idx);
                 d_visit_writer->registerPlotQuantity("LS new", "SCALAR", d_ls_new_idx);
                 d_visit_writer->registerPlotQuantity("Q_scratch", "SCALAR", d_Q_scratch_idx););
}

void
SemiLagrangianAdvIntegrator::preprocessIntegrateHierarchy(const double current_time,
                                                          const double new_time,
                                                          const int num_cycles)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_scratch_data, current_time);
        level->allocatePatchData(d_new_data, new_time);
        if (!level->checkAllocated(d_Q_scratch_idx)) level->allocatePatchData(d_Q_scratch_idx, current_time);
        if (!level->checkAllocated(d_Q_R_idx)) level->allocatePatchData(d_Q_R_idx, current_time);
    }

    // set up convective operator
    SAMRAIVectorReal<NDIM, double> in(d_object_name + "::in", d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
    SAMRAIVectorReal<NDIM, double> out(d_object_name + "::out", d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
    in.addComponent(d_Q_var[0], d_Q_scratch_idx);
    out.addComponent(d_Q_var[0], d_Q_scratch_idx);
    d_Q_convec_oper->initializeOperatorState(in, out);
    d_Q_convec_oper->setSolutionTime(current_time);

    // Set level set
    d_ls_fcn->setDataOnPatchHierarchy(d_ls_cur_idx, d_ls_var, d_hierarchy, current_time, false);
    d_ls_fcn->setDataOnPatchHierarchy(d_ls_new_idx, d_ls_var, d_hierarchy, new_time, false);

    // Set velocities
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (const auto& u_var : d_u_var)
    {
        const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
        const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
        d_u_fcn[u_var]->setDataOnPatchHierarchy(
            u_cur_idx, u_var, d_hierarchy, current_time, false, 0, d_hierarchy->getFinestLevelNumber());
        d_u_fcn[u_var]->setDataOnPatchHierarchy(
            u_new_idx, u_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
    }

    AdvDiffHierarchyIntegrator::preprocessIntegrateHierarchy(current_time, new_time, num_cycles);
}

void
SemiLagrangianAdvIntegrator::integrateHierarchy(const double current_time, const double new_time, const int cycle_num)
{
    for (const auto& Q_var : d_Q_var)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
        const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
        advectionUpdate(Q_var, Q_cur_idx, Q_new_idx, current_time, new_time);
    }
    AdvDiffHierarchyIntegrator::integrateHierarchy(current_time, new_time, cycle_num);
}

void
SemiLagrangianAdvIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                                           const double new_time,
                                                           const bool skip_synchronize_new_state_data,
                                                           const int num_cycles)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        //        level->deallocatePatchData(d_Q_scratch_idx);
    }
    AdvDiffHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);
}

/////////////////////// PRIVATE ///////////////////////////////

void
SemiLagrangianAdvIntegrator::advectionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const int Q_cur_idx,
                                             const int Q_new_idx,
                                             const double current_time,
                                             const double new_time)
{
    const double dt = new_time - current_time;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    int coarsest_ln = 0;
    auto var_db = VariableDatabase<NDIM>::getDatabase();

    for (auto Q_var : d_Q_var)
    {
        Pointer<FaceVariable<NDIM, double>> u_var = d_Q_u_map[Q_var];
        const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
        // Integrate path
        integratePaths(d_path_idx, u_cur_idx, dt);

        // Fill in normal ghost cells for d_Q_scratch_idx
        fillNormalCells(Q_cur_idx, d_Q_scratch_idx, d_ls_cur_idx);

        // fill ghost cells
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(2);
        HierarchyGhostCellInterpolation hier_ghost_cells;
        ghost_cell_comps[0] = ITC(d_path_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR", true, nullptr);
        ghost_cell_comps[1] =
            ITC(d_Q_scratch_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);

        // Invert mapping to find \XX^\star
        // Note that for the Z_0 z-spline, \XX^\star = \xx^{n+1}
        invertMapping(d_path_idx, d_xstar_idx);

        // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
        d_vol_fcn->updateVolumeAndArea(d_vol_idx, d_vol_var, -1, nullptr, d_ls_new_idx, d_ls_var);
        evaluateMappingOnHierarchy(d_xstar_idx, d_Q_scratch_idx, Q_new_idx, d_vol_idx, /*order*/ 2);
    }
}

void
SemiLagrangianAdvIntegrator::fillNormalCells(const int Q_idx, const int Q_scr_idx, const int ls_idx)
{
    // Here we fill in d_Q_norm_idx by solving
    // dQ/dt - N * grad(Q) = 0

    // First fill in volume data
    d_vol_fcn->updateVolumeAndArea(d_vol_idx, d_vol_var, IBTK::invalid_index, nullptr, ls_idx, d_ls_var, false);

    // Find normal
    findLSNormal(ls_idx, d_ls_normal_idx);
    d_Q_convec_oper->setAdvectionVelocity(d_ls_normal_idx);
    double min_dx = std::numeric_limits<double>::max();

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_scr_idx);
            Pointer<CellData<NDIM, double>> Q_src_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
            Q_data->fill(0.0);
            // Fill cut cell with same cell average
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) > 0.0) (*Q_data)(idx) = (*Q_src_data)(idx);
                for (int d = 0; d < NDIM; ++d) min_dx = std::min(min_dx, dx[d]);
            }
        }
    }
    int iter = 0;
    double dt = 0.3 * min_dx; // CFL times dx
    double max_ls_length_frac = 10.0;
    while (iter++ < d_max_iterations)
    {
        d_Q_convec_oper->applyConvectiveOperator(Q_scr_idx, d_Q_R_idx);
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Box<NDIM>& box = patch->getBox();
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const dx = pgeom->getDx();
                Pointer<CellData<NDIM, double>> Q_src_data = patch->getPatchData(Q_idx);
                Pointer<CellData<NDIM, double>> Q_norm_data = patch->getPatchData(Q_scr_idx);
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
                Pointer<CellData<NDIM, double>> Q_R_data = patch->getPatchData(d_Q_R_idx);
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    double ls = node_to_cell(idx, *ls_data);
                    if (ls > (max_ls_length_frac * dx[0])) (*Q_R_data)(idx) = 0.0;
                }

                // Compute flux differencing
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double vol = 1.0 - (*vol_data)(idx);
                    if (vol > 0.0)
                    {
                        // We are in a cut cell or external cell, we need to do a flux differencing
                        (*Q_norm_data)(idx) -= dt * (*Q_R_data)(idx);
                    }
                }

                // Now we set cut cells to be the same as interior average
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double vol = 1.0 - (*vol_data)(idx);
                    if ((*vol_data)(idx) > 0.0 && (*vol_data)(idx) < 1.0) (*Q_norm_data)(idx) = (*Q_src_data)(idx);
                }
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::findLSNormal(const int ls_idx, const int ls_n_idx)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            Pointer<FaceData<NDIM, double>> N_data = patch->getPatchData(ls_n_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            for (FaceIterator<NDIM> fi(patch->getBox(), 0); fi; fi++)
            {
                const FaceIndex<NDIM>& idx = fi();
                double ls_top_right = (*ls_data)(NodeIndex<NDIM>(idx.toCell(1), IntVector<NDIM>(1, 1)));
                double ls_bottom_right = (*ls_data)(NodeIndex<NDIM>(idx.toCell(1), IntVector<NDIM>(1, 0)));
                double ls_top_left = (*ls_data)(NodeIndex<NDIM>(idx.toCell(0), IntVector<NDIM>(0, 1)));
                double ls_bottom_left = (*ls_data)(NodeIndex<NDIM>(idx.toCell(0), IntVector<NDIM>(0, 0)));
                (*N_data)(idx) = (ls_top_right + ls_bottom_right - ls_top_left - ls_bottom_left) / (4.0 * dx[0]);
            }
            for (FaceIterator<NDIM> fi(patch->getBox(), 1); fi; fi++)
            {
                const FaceIndex<NDIM>& idx = fi();
                double ls_top_right = (*ls_data)(NodeIndex<NDIM>(idx.toCell(1), IntVector<NDIM>(1, 1)));
                double ls_bottom_right = (*ls_data)(NodeIndex<NDIM>(idx.toCell(0), IntVector<NDIM>(1, 0)));
                double ls_top_left = (*ls_data)(NodeIndex<NDIM>(idx.toCell(1), IntVector<NDIM>(0, 1)));
                double ls_bottom_left = (*ls_data)(NodeIndex<NDIM>(idx.toCell(0), IntVector<NDIM>(0, 0)));
                (*N_data)(idx) = (ls_top_right + ls_top_left - ls_bottom_right - ls_bottom_left) / (4.0 * dx[1]);
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::integratePaths(const int path_idx, const int u_idx, const double dt)
{
    // Integrate path to find \xx^{n+1}
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
            Pointer<FaceData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();

                for (int d = 0; d < NDIM; ++d)
                {
                    const double u =
                        0.5 * ((*u_data)(FaceIndex<NDIM>(idx, d, 0)) + (*u_data)(FaceIndex<NDIM>(idx, d, 1)));
                    (*path_data)(idx, d) = (idx(d) + 0.5) + dt * u / dx[d] * (d_using_forward_integration ? 1.0 : -1.0);
                }
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::invertMapping(const int path_idx, const int xstar_idx)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(xstar_idx);

            if (!d_using_forward_integration)
            {
                xstar_data->copy(*path_data);
            }
            else
            {
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    MatrixNd grad_xi;
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> dir(0);
                        dir(d) = 1;
                        for (int dd = 0; dd < NDIM; ++dd)
                            grad_xi(d, dd) = 0.5 * ((*path_data)(idx + dir, dd) - (*path_data)(idx - dir, dd));
                    }
                    VectorNd XStar, xnp1, x;
                    for (int d = 0; d < NDIM; ++d)
                    {
                        x(d) = static_cast<double>(idx(d)) + 0.5;
                        xnp1(d) = (*path_data)(idx, d);
                    }
                    XStar = grad_xi.inverse() * (x - xnp1) + x;
                    for (int d = 0; d < NDIM; ++d) (*xstar_data)(idx, d) = XStar(d);
                }
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::evaluateMappingOnHierarchy(const int xstar_idx,
                                                        const int Q_cur_idx,
                                                        const int Q_new_idx,
                                                        const int vol_idx,
                                                        const int order)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(xstar_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_cur_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(Q_new_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    (*Q_new_data)(idx) = sumOverZSplines(x_loc, idx, *Q_cur_data, order);
                }
                else
                {
                    (*Q_new_data)(idx) = 0.0;
                }
            }
        }
    }
}

double
SemiLagrangianAdvIntegrator::sumOverZSplines(const IBTK::VectorNd& x_loc,
                                             const CellIndex<NDIM>& idx,
                                             const CellData<NDIM, double>& Q_data,
                                             const int order)
{
    double val = 0.0;
    Box<NDIM> box(idx, idx);
    box.grow(getSplineWidth(order) + 1);
    const Box<NDIM>& ghost_box = Q_data.getGhostBox();
    TBOX_ASSERT(ghost_box.contains(box));
    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx_c = ci();
        VectorNd xx;
        for (int d = 0; d < NDIM; ++d) xx(d) = idx_c(d) + 0.5;
        val += Q_data(idx_c) * evaluateZSpline(x_loc - xx, order);
    }
    return val;
}
} // namespace IBAMR
