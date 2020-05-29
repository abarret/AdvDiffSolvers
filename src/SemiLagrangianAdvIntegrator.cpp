#include "ibamr/AdvDiffCUIConvectiveOperator.h"
#include "ibamr/AdvDiffPPMConvectiveOperator.h"
#include "ibamr/AdvDiffWavePropConvectiveOperator.h"
#include "ibamr/app_namespaces.h"

#include "QInitial.h"
#include "SAMRAIVectorReal.h"
#include "SemiLagrangianAdvIntegrator.h"
#include "utility_functions.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace IBAMR
{
int SemiLagrangianAdvIntegrator::GHOST_CELL_WIDTH = 4;

SemiLagrangianAdvIntegrator::SemiLagrangianAdvIntegrator(const std::string& object_name,
                                                         Pointer<Database> input_db,
                                                         bool register_for_restart)
    : AdvDiffHierarchyIntegrator(object_name, input_db, register_for_restart),
      d_path_var(new CellVariable<NDIM, double>(d_object_name + "::PathVar", NDIM)),
      d_vol_var(new CellVariable<NDIM, double>(d_object_name + "::VolVar")),
      d_area_var(new CellVariable<NDIM, double>(d_object_name + "::AreaVar"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_path_idx = var_db->registerVariableAndContext(d_path_var, var_db->getContext(d_object_name + "::PathContext"));
    d_adv_data.setFlag(d_path_idx);

    if (input_db)
    {
        d_prescribe_ls = input_db->getBool("prescribe_level_set");
        d_min_ls_refine_factor = input_db->getDouble("min_ls_refine_factor");
        d_max_ls_refine_factor = input_db->getDouble("max_ls_refine_factor");
        d_least_squares_reconstruction_order =
            string_to_enum<LeastSquaresOrder>(input_db->getString("least_squares_order"));
        d_use_strang_splitting = input_db->getBool("use_strang_splitting");
    }
}

void
SemiLagrangianAdvIntegrator::registerTransportedQuantity(Pointer<CellVariable<NDIM, double>> Q_var, bool Q_output)
{
    AdvDiffHierarchyIntegrator::registerTransportedQuantity(Q_var, Q_output);
    // We need to register our own scratch variable since we need more ghost cells.
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scratch_idx = var_db->registerVariableAndContext(
        Q_var, var_db->getContext(d_object_name + "::BiggerScratch"), IntVector<NDIM>(GHOST_CELL_WIDTH));
    d_current_data.setFlag(d_Q_scratch_idx);
}

void
SemiLagrangianAdvIntegrator::registerLevelSetVariable(Pointer<CellVariable<NDIM, double>> ls_var)
{
    d_ls_cell_var = ls_var;
    d_ls_node_var = new NodeVariable<NDIM, double>(d_object_name + "::LSNodeVar");
}

void
SemiLagrangianAdvIntegrator::registerLevelSetVelocity(Pointer<FaceVariable<NDIM, double>> u_var)
{
    TBOX_ASSERT(std::find(d_u_var.begin(), d_u_var.end(), u_var) != d_u_var.end());
    d_ls_u_pair.first = d_ls_cell_var;
    d_ls_u_pair.second = u_var;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetResetFunction(Pointer<LSInitStrategy> ls_strategy)
{
    d_ls_strategy = ls_strategy;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetFunction(Pointer<CartGridFunction> ls_fcn)
{
    d_ls_fcn = ls_fcn;
}

void
SemiLagrangianAdvIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                           Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    if (d_integrator_is_initialized) return;
    d_hierarchy = hierarchy;
    d_gridding_alg = gridding_alg;
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();

    AdvDiffHierarchyIntegrator::registerVariables();
    registerVariable(d_ls_cell_cur_idx,
                     d_ls_cell_new_idx,
                     d_ls_cell_scr_idx,
                     d_ls_cell_var,
                     IntVector<NDIM>(GHOST_CELL_WIDTH),
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE");

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_ls_node_cur_idx = var_db->registerVariableAndContext(d_ls_node_var, getCurrentContext(), GHOST_CELL_WIDTH);
    d_ls_node_new_idx = var_db->registerVariableAndContext(d_ls_node_var, getNewContext(), GHOST_CELL_WIDTH);
    d_vol_cur_idx = var_db->registerVariableAndContext(d_vol_var, getCurrentContext(), GHOST_CELL_WIDTH);
    d_vol_new_idx = var_db->registerVariableAndContext(d_vol_var, getNewContext(), GHOST_CELL_WIDTH);
    d_area_idx = var_db->registerVariableAndContext(d_area_var, getCurrentContext(), GHOST_CELL_WIDTH);

    d_ls_data.setFlag(d_ls_node_cur_idx);
    d_ls_data.setFlag(d_ls_node_new_idx);
    d_ls_data.setFlag(d_vol_cur_idx);
    d_ls_data.setFlag(d_vol_new_idx);
    d_ls_data.setFlag(d_area_idx);

    AdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    d_visit_writer->registerPlotQuantity("Volume", "SCALAR", d_vol_new_idx);
    d_visit_writer->registerPlotQuantity("LS current", "SCALAR", d_ls_node_cur_idx);
    d_visit_writer->registerPlotQuantity("LS new", "SCALAR", d_ls_node_new_idx);
    d_visit_writer->registerPlotQuantity("Q_scratch", "SCALAR", d_Q_scratch_idx);
    d_visit_writer->registerPlotQuantity("LS Cell", "SCALAR", d_ls_cell_cur_idx);

    d_integrator_is_initialized = true;
}

void
SemiLagrangianAdvIntegrator::initializeLevelDataSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                            const int ln,
                                                            const double data_time,
                                                            const bool can_be_refined,
                                                            bool initial_time,
                                                            Pointer<BasePatchLevel<NDIM>> old_level,
                                                            bool allocate_data)
{
    AdvDiffHierarchyIntegrator::initializeLevelDataSpecialized(
        hierarchy, ln, data_time, can_be_refined, initial_time, old_level, allocate_data);
    // Initialize level set
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
    if (allocate_data) level->allocatePatchData(d_ls_data, data_time);

    if (initial_time)
    {
        pout << "Setting level set at initial time: " << initial_time << "\n";
        d_ls_fcn->setDataOnPatchLevel(d_ls_cell_cur_idx, d_ls_cell_var, level, data_time, initial_time);
        d_ls_fcn->setDataOnPatchLevel(d_ls_node_cur_idx, d_ls_node_var, level, data_time, initial_time);
        if (d_prescribe_ls)
        {
            d_ls_fcn->setDataOnPatchLevel(d_ls_node_new_idx, d_ls_node_var, level, data_time, initial_time);
            d_ls_fcn->setDataOnPatchLevel(d_ls_node_cur_idx, d_ls_node_var, level, data_time, initial_time);
        }
    }
    else if (d_prescribe_ls)
    {
        d_ls_fcn->setDataOnPatchLevel(d_ls_node_cur_idx, d_ls_node_var, level, data_time, initial_time);
        d_ls_fcn->setDataOnPatchLevel(d_ls_cell_cur_idx, d_ls_cell_var, level, data_time, initial_time);
    }
}

void
SemiLagrangianAdvIntegrator::applyGradientDetectorSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                              const int ln,
                                                              const double data_time,
                                                              const int tag_index,
                                                              const bool initial_time,
                                                              const bool uses_richardson_extrapolation_too)
{
    AdvDiffHierarchyIntegrator::applyGradientDetectorSpecialized(
        hierarchy, ln, data_time, tag_index, initial_time, uses_richardson_extrapolation_too);
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CellData<NDIM, int>> tag_data = patch->getPatchData(tag_index);
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_node_cur_idx);
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double ls = node_to_cell(idx, *ls_data);
            if (ls < d_max_ls_refine_factor * dx[0] && ls > d_min_ls_refine_factor * dx[0]) (*tag_data)(idx) = 1;
        }
    }
}

int
SemiLagrangianAdvIntegrator::getNumberOfCycles() const
{
    return 1;
}

void
SemiLagrangianAdvIntegrator::preprocessIntegrateHierarchy(const double current_time,
                                                          const double new_time,
                                                          const int num_cycles)
{
    AdvDiffHierarchyIntegrator::preprocessIntegrateHierarchy(current_time, new_time, num_cycles);
    const double dt = new_time - current_time;
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_scratch_data, current_time);
        level->allocatePatchData(d_new_data, new_time);
        level->allocatePatchData(d_adv_data, current_time);
    }

    // Update level set at current time
    if (d_prescribe_ls)
    {
        d_ls_fcn->setDataOnPatchHierarchy(d_ls_cell_cur_idx, d_ls_cell_var, d_hierarchy, current_time, false);
        d_ls_fcn->setDataOnPatchHierarchy(d_ls_node_cur_idx, d_ls_node_var, d_hierarchy, current_time, false);
    }
    else
    {
        d_ls_strategy->initializeLSData(d_ls_cell_cur_idx, d_hier_math_ops, d_integrator_step, current_time, false);
        // interpolate cell data to node data
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] = ITC(d_ls_cell_scr_idx,
                                  d_ls_cell_cur_idx,
                                  "CONSERVATIVE_LINEAR_REFINE",
                                  false,
                                  "CONSERVATIVE_COARSEN",
                                  "LINEAR",
                                  false,
                                  nullptr);
        Pointer<HierarchyGhostCellInterpolation> hier_ghost_cell = new HierarchyGhostCellInterpolation();
        hier_ghost_cell->initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        d_hier_math_ops->interp(
            d_ls_node_cur_idx, d_ls_node_var, false, d_ls_cell_scr_idx, d_ls_cell_var, hier_ghost_cell, current_time);
        hier_ghost_cell->deallocateOperatorState();
        ghost_cell_comps[0] = ITC(d_ls_node_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        hier_ghost_cell->initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        hier_ghost_cell->fillData(current_time);
    }

    // ensure volume and area indices are set to current values.
    d_vol_fcn->updateVolumeAndArea(
        d_vol_cur_idx, d_vol_var, d_area_idx, d_area_var, d_ls_node_cur_idx, d_ls_node_var, true);

    // Set velocities
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (const auto& u_var : d_u_var)
    {
        const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
        const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
        d_u_fcn[u_var]->setDataOnPatchHierarchy(
            u_cur_idx, u_var, d_hierarchy, current_time, false, coarsest_ln, finest_ln);
        d_u_fcn[u_var]->setDataOnPatchHierarchy(u_new_idx, u_var, d_hierarchy, new_time, false, coarsest_ln, finest_ln);
    }

    // Prepare diffusion
    int l = 0;
    for (const auto& Q_var : d_Q_var)
    {
        Pointer<CellVariable<NDIM, double>> Q_rhs_var = d_Q_Q_rhs_map[Q_var];
        Pointer<SideVariable<NDIM, double>> D_var = d_Q_diffusion_coef_variable[Q_var];
        Pointer<SideVariable<NDIM, double>> D_rhs_var = d_diffusion_coef_rhs_map[D_var];
        const double lambda = d_Q_damping_coef[Q_var];
        const std::vector<RobinBcCoefStrategy<NDIM>*> Q_bc_coef = d_Q_bc_coef[Q_var];

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int D_cur_idx =
            (D_var ? var_db->mapVariableAndContextToIndex(D_var, getCurrentContext()) : IBTK::invalid_index);
        const int D_scr_idx =
            (D_var ? var_db->mapVariableAndContextToIndex(D_var, getScratchContext()) : IBTK::invalid_index);
        const int D_rhs_scr_idx =
            (D_rhs_var ? var_db->mapVariableAndContextToIndex(D_rhs_var, getScratchContext()) : IBTK::invalid_index);

        // This should be changed for different time stepping for diffusion. Right now set at trapezoidal rule.
        double K = 0.5;

        PoissonSpecifications rhs_spec(d_object_name + "::rhs_spec" + Q_var->getName());
        PoissonSpecifications solv_spec(d_object_name + "::solv_spec" + Q_var->getName());

        const double dt_scale = d_use_strang_splitting ? 2.0 : 1.0;
        solv_spec.setCConstant(dt_scale / dt + K * lambda);
        rhs_spec.setCConstant(dt_scale / dt - (1.0 - K) * lambda);

        if (isDiffusionCoefficientVariable(Q_var))
        {
            d_hier_sc_data_ops->scale(D_scr_idx, -K, D_cur_idx);
            solv_spec.setDPatchDataId(D_scr_idx);
            d_hier_sc_data_ops->scale(D_rhs_scr_idx, 1.0 - K, D_cur_idx);
            rhs_spec.setDPatchDataId(D_rhs_scr_idx);
        }
        else
        {
            const double kappa = d_Q_diffusion_coef[Q_var];
            solv_spec.setDConstant(-K * kappa);
            rhs_spec.setDConstant((1.0 - K) * kappa);
        }

        // Initialize RHS Operator
        Pointer<LaplaceOperator> rhs_oper = d_helmholtz_rhs_ops[l];
        rhs_oper->setPoissonSpecifications(rhs_spec);
        rhs_oper->setPhysicalBcCoefs(Q_bc_coef);
        rhs_oper->setHomogeneousBc(false);
        rhs_oper->setSolutionTime(current_time);
        rhs_oper->setTimeInterval(current_time, new_time);
        if (d_helmholtz_rhs_ops_need_init[l])
        {
            if (d_enable_logging)
                plog << d_object_name << ": "
                     << "Initializing Helmholtz RHS operator for variable number " << l << "\n";
            rhs_oper->initializeOperatorState(*d_sol_vecs[l], *d_rhs_vecs[l]);
            d_helmholtz_rhs_ops_need_init[l] = false;
        }

        Pointer<PoissonSolver> helmholtz_solver = d_helmholtz_solvers[l];
        helmholtz_solver->setPoissonSpecifications(solv_spec);
        helmholtz_solver->setPhysicalBcCoefs(Q_bc_coef);
        helmholtz_solver->setHomogeneousBc(false);
        helmholtz_solver->setSolutionTime(new_time);
        helmholtz_solver->setTimeInterval(current_time, new_time);
        if (d_helmholtz_solvers_need_init[l])
        {
            if (d_enable_logging)
                plog << d_object_name << ": "
                     << "Initializing Helmholtz solver for variable number " << l << "\n";
            helmholtz_solver->initializeSolverState(*d_sol_vecs[l], *d_rhs_vecs[l]);
            d_helmholtz_solvers_need_init[l] = false;
        }
        l++;
    }
}

void
SemiLagrangianAdvIntegrator::integrateHierarchy(const double current_time, const double new_time, const int cycle_num)
{
    AdvDiffHierarchyIntegrator::integrateHierarchy(current_time, new_time, cycle_num);
    const double half_time = current_time + 0.5 * (new_time - current_time);
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (const auto& Q_var : d_Q_var)
    {
        const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());

        // Copy current data to scratch
        d_hier_cc_data_ops->copyData(Q_scr_idx, Q_cur_idx);

        // First do a diffusion update.
        // Note diffusion update fills in "New" context
        diffusionUpdate(
            Q_var, d_ls_node_cur_idx, d_vol_cur_idx, current_time, d_use_strang_splitting ? half_time : new_time);

        plog << d_object_name + "::integrateHierarchy() finished diffusion update for variable: " << Q_var->getName()
             << "\n";

        // TODO: Should we synchronize hierarchy?
    }

    if (d_prescribe_ls)
    {
        plog << d_object_name + "::integrateHierarchy() prescribing level set at time: " << new_time << "\n";
        d_ls_fcn->setDataOnPatchHierarchy(
            d_ls_cell_new_idx, d_ls_cell_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
        d_ls_fcn->setDataOnPatchHierarchy(
            d_ls_node_new_idx, d_ls_node_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
    }
    else
    {
        plog << d_object_name + "::integrateHierarchy() updating level set at time: " << new_time << "\n";
        // Now we integrate paths and update level set
        const int u_cur_idx = var_db->mapVariableAndContextToIndex(d_ls_u_pair.second, getCurrentContext());
        integratePaths(d_path_idx, u_cur_idx, new_time - current_time);

        // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
        evaluateMappingOnHierarchy(d_path_idx, d_ls_cell_scr_idx, d_ls_cell_new_idx, /*order*/ 2);

        // Synchronize hierarchy and interpolate to cell nodes
        // TODO: Should we synchronize hierarchy?
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] = ITC(d_ls_cell_scr_idx,
                                  d_ls_cell_new_idx,
                                  "CONSERVATIVE_LINEAR_REFINE",
                                  false,
                                  "CONSERVATIVE_COARSEN",
                                  "LINEAR",
                                  false,
                                  nullptr);
        Pointer<HierarchyGhostCellInterpolation> hier_ghost_cell = new HierarchyGhostCellInterpolation();
        hier_ghost_cell->initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        d_hier_math_ops->interp(
            d_ls_node_new_idx, d_ls_node_var, false, d_ls_cell_scr_idx, d_ls_cell_var, hier_ghost_cell, current_time);
        hier_ghost_cell->deallocateOperatorState();
        ghost_cell_comps[0] = ITC(d_ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        hier_ghost_cell->initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        hier_ghost_cell->fillData(current_time);
    }

    // Now do advective update for each variable
    for (const auto& Q_var : d_Q_var)
    {
        const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
        const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
        // Reset for next iteration
        // Copy new data to current and scratch.
        d_hier_cc_data_ops->copyData(Q_cur_idx, Q_new_idx);
        d_hier_cc_data_ops->copyData(Q_scr_idx, Q_new_idx);

        // Now update advection.
        advectionUpdate(Q_var, current_time, new_time);

        plog << d_object_name + "::integrateHierarchy() finished advection update for variable: " << Q_var->getName()
             << "\n";
    }

    if (d_use_strang_splitting)
    {
        for (const auto& Q_var : d_Q_var)
        {
            const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
            const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());

            // Copy current data to scratch
            d_hier_cc_data_ops->copyData(Q_cur_idx, Q_new_idx);
            d_hier_cc_data_ops->copyData(Q_scr_idx, Q_new_idx);

            d_vol_fcn->updateVolumeAndArea(
                d_vol_new_idx, d_vol_var, d_area_idx, d_area_var, d_ls_node_new_idx, d_ls_node_var, true);

            // First do a diffusion update.
            // Note diffusion update fills in "New" context
            diffusionUpdate(Q_var, d_ls_node_new_idx, d_vol_new_idx, half_time, new_time);

            plog << d_object_name + "::integrateHierarchy() finished diffusion update for variable: "
                 << Q_var->getName() << "\n";

            // TODO: Should we synchronize hierarchy?
        }
    }
}

void
SemiLagrangianAdvIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                                           const double new_time,
                                                           const bool skip_synchronize_new_state_data,
                                                           const int num_cycles)
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        d_hierarchy->getPatchLevel(ln)->deallocatePatchData(d_adv_data);

    AdvDiffHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);
}

void
SemiLagrangianAdvIntegrator::initializeCompositeHierarchyDataSpecialized(const double current_time,
                                                                         const bool initial_time)
{
    AdvDiffHierarchyIntegrator::initializeCompositeHierarchyDataSpecialized(current_time, initial_time);

    if (initial_time)
    {
        d_ls_fcn->setDataOnPatchHierarchy(d_ls_node_cur_idx, d_ls_node_var, d_hierarchy, 0.0);
        d_vol_fcn = new LSFindCellVolume(d_object_name + "::FindCellVolume", d_hierarchy);
        d_vol_fcn->updateVolumeAndArea(
            d_vol_cur_idx, d_vol_var, IBTK::invalid_index, nullptr, d_ls_node_cur_idx, d_ls_node_var, false);

        for (const auto& Q_var : d_Q_var)
        {
            auto var_db = VariableDatabase<NDIM>::getDatabase();
            const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            Pointer<QInitial> Q_init = d_Q_init[Q_var];
            Q_init->setLSIndex(d_ls_node_cur_idx, d_vol_cur_idx);
            Q_init->setDataOnPatchHierarchy(Q_idx, Q_var, d_hierarchy, 0.0);
        }
    }
}

void
SemiLagrangianAdvIntegrator::regridHierarchyEndSpecialized()
{
    // Force reinitialization of level set after regrid.
    if (!d_prescribe_ls) d_ls_strategy->setReinitializeLSData(true);
}

void
SemiLagrangianAdvIntegrator::resetTimeDependentHierarchyDataSpecialized(const double new_time)
{
    // Copy level set info
    d_hier_cc_data_ops->copyData(d_ls_cell_cur_idx, d_ls_cell_new_idx);

    AdvDiffHierarchyIntegrator::resetTimeDependentHierarchyDataSpecialized(new_time);
}

/////////////////////// PRIVATE ///////////////////////////////
void
SemiLagrangianAdvIntegrator::advectionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const double current_time,
                                             const double new_time)
{
    const double dt = new_time - current_time;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    int coarsest_ln = 0;
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    Pointer<FaceVariable<NDIM, double>> u_var = d_Q_u_map[Q_var];
    const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());

    // fill ghost cells
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    HierarchyGhostCellInterpolation hier_ghost_cells;
    ghost_cell_comps[0] =
        ITC(d_Q_scratch_idx, Q_cur_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
    ghost_cell_comps[1] = ITC(d_ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(current_time);
    hier_ghost_cells.deallocateOperatorState();
    d_vol_fcn->updateVolumeAndArea(
        d_vol_new_idx, d_vol_var, IBTK::invalid_index, nullptr, d_ls_node_new_idx, d_ls_node_var, true);
    // Integrate path
    integratePaths(d_path_idx, u_cur_idx, d_vol_new_idx, d_ls_node_new_idx, dt);

    ghost_cell_comps[0] =
        ITC(d_Q_scratch_idx, Q_cur_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
    ghost_cell_comps[1] = ITC(d_ls_node_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(current_time);
    hier_ghost_cells.deallocateOperatorState();
    d_vol_fcn->updateVolumeAndArea(
        d_vol_cur_idx, d_vol_var, IBTK::invalid_index, nullptr, d_ls_node_cur_idx, d_ls_node_var, true);

    // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
    evaluateMappingOnHierarchy(
        d_path_idx, d_Q_scratch_idx, d_vol_cur_idx, Q_new_idx, d_vol_new_idx, d_ls_node_cur_idx, /*order*/ 2);
}

void
SemiLagrangianAdvIntegrator::diffusionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const int ls_idx,
                                             const int vol_idx,
                                             const double current_time,
                                             const double new_time)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    // We assume scratch context is already filled correctly.
    const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    const size_t l = distance(d_Q_var.begin(), std::find(d_Q_var.begin(), d_Q_var.end(), Q_var));

    Pointer<LSCutCellLaplaceOperator> rhs_oper = d_helmholtz_rhs_ops[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(rhs_oper);
#endif
    rhs_oper->setLSIndices(ls_idx, d_ls_node_var, vol_idx, d_vol_var, d_area_idx, d_area_var);
    rhs_oper->setSolutionTime(current_time);
    rhs_oper->apply(*d_sol_vecs[l], *d_rhs_vecs[l]);

    Pointer<PETScKrylovPoissonSolver> Q_helmholtz_solver = d_helmholtz_solvers[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_helmholtz_solver);
#endif
    Pointer<LSCutCellLaplaceOperator> solv_oper = Q_helmholtz_solver->getOperator();
#if !defined(NDEBUG)
    TBOX_ASSERT(solv_oper);
#endif
    solv_oper->setLSIndices(ls_idx, d_ls_node_var, vol_idx, d_vol_var, d_area_idx, d_area_var);
    solv_oper->setSolutionTime(new_time);
    Q_helmholtz_solver->solveSystem(*d_sol_vecs[l], *d_rhs_vecs[l]);
    d_hier_cc_data_ops->copyData(Q_new_idx, Q_scr_idx);
    if (d_enable_logging)
    {
        plog << d_object_name << "::integrateHierarchy(): diffusion solve number of iterations = "
             << Q_helmholtz_solver->getNumIterations() << "\n";
        plog << d_object_name << "::integrateHierarchy(): diffusion solve residual norm        = "
             << Q_helmholtz_solver->getResidualNorm() << "\n";
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
                    (*path_data)(idx, d) = (idx(d) + 0.5) - dt * u / dx[d];
                }
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::integratePaths(const int path_idx,
                                            const int u_idx,
                                            const int vol_idx,
                                            const int ls_idx,
                                            const double dt)
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
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<FaceData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd com = find_cell_centroid(idx, *ls_data);
                for (int d = 0; d < NDIM; ++d)
                {
                    const double u =
                        0.5 * ((*u_data)(FaceIndex<NDIM>(idx, d, 0)) + (*u_data)(FaceIndex<NDIM>(idx, d, 1)));
                    (*path_data)(idx, d) = com(d) - dt * u / dx[d];
                }
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::evaluateMappingOnHierarchy(const int xstar_idx,
                                                        const int Q_cur_idx,
                                                        const int Q_new_idx,
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

            Q_new_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                IBTK::VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                (*Q_new_data)(idx) = sumOverZSplines(x_loc, idx, *Q_cur_data, order);
            }
        }
    }
}

void
SemiLagrangianAdvIntegrator::evaluateMappingOnHierarchy(const int xstar_idx,
                                                        const int Q_cur_idx,
                                                        const int vol_cur_idx,
                                                        const int Q_new_idx,
                                                        const int vol_new_idx,
                                                        const int ls_idx,
                                                        const int order)
{
#ifndef NDEBUG
    TBOX_ASSERT(vol_cur_idx > 0);
    TBOX_ASSERT(vol_new_idx > 0);
#endif
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(xstar_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_cur_idx);
            Pointer<CellData<NDIM, double>> vol_cur_data = patch->getPatchData(vol_cur_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(Q_new_idx);
            Pointer<CellData<NDIM, double>> vol_new_data = patch->getPatchData(vol_new_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);

            Q_new_data->fillAll(0.0);

            const int stencil_width = getSplineWidth(order);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_new_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    // Check if we can use z-spline
                    if (indexWithinWidth(stencil_width, idx, *vol_cur_data))
                        (*Q_new_data)(idx) = sumOverZSplines(x_loc, idx, *Q_cur_data, order);
                    else
                        (*Q_new_data)(idx) =
                            leastSquaresReconstruction(x_loc, idx, *Q_cur_data, *vol_cur_data, *ls_data, patch);
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

bool
SemiLagrangianAdvIntegrator::indexWithinWidth(const int stencil_width,
                                              const CellIndex<NDIM>& idx,
                                              const CellData<NDIM, double>& vol_data)
{
    bool withinWidth = true;
    Box<NDIM> check_box(idx, idx);
    check_box.grow(stencil_width);
    for (CellIterator<NDIM> i(check_box); i; i++)
    {
        const CellIndex<NDIM>& idx_c = i();
        if (vol_data(idx_c) < 1.0) withinWidth = false;
    }
    return withinWidth;
}

double
SemiLagrangianAdvIntegrator::leastSquaresReconstruction(IBTK::VectorNd x_loc,
                                                        const CellIndex<NDIM>& idx,
                                                        const CellData<NDIM, double>& Q_data,
                                                        const CellData<NDIM, double>& vol_data,
                                                        const NodeData<NDIM, double>& ls_data,
                                                        const Pointer<Patch<NDIM>>& patch)
{
    int size = 0;
    int box_size = 0;
    switch (d_least_squares_reconstruction_order)
    {
    case CONSTANT:
        size = 1;
        box_size = 1;
        break;
    case LINEAR:
        size = 1 + NDIM;
        box_size = 2;
        break;
    case QUADRATIC:
        size = 3 * NDIM;
        box_size = 3;
        break;
    case CUBIC:
        size = 10;
        box_size = 4;
        break;
    case UNKNOWN_ORDER:
        TBOX_ERROR("Unknown order.");
        break;
    }
    Box<NDIM> box(idx, idx);
    box.grow(box_size);
#ifndef NDEBUG
    TBOX_ASSERT(ls_data.getGhostBox().contains(box));
    TBOX_ASSERT(Q_data.getGhostBox().contains(box));
    TBOX_ASSERT(vol_data.getGhostBox().contains(box));
#endif

    const CellIndex<NDIM>& idx_low = patch->getBox().lower();

    std::vector<double> Q_vals;
    std::vector<VectorNd> X_vals;

    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx_c = ci();
        if (vol_data(idx_c) > 0.0)
        {
            // Use this point to calculate least squares reconstruction.
            // Find cell center
            VectorNd x_cent_c = find_cell_centroid(idx_c, ls_data);
            Q_vals.push_back(Q_data(idx_c));
            X_vals.push_back(x_cent_c);
        }
    }
    const int m = Q_vals.size();
    MatrixXd A(MatrixXd::Zero(m, size)), Lambda(MatrixXd::Zero(m, m));
    VectorXd U(VectorXd::Zero(m));
    for (size_t i = 0; i < Q_vals.size(); ++i)
    {
        U(i) = Q_vals[i];
        const VectorNd X = X_vals[i] - x_loc;
        Lambda(i, i) = std::sqrt(weight((X_vals[i] - x_loc).norm()));
        switch (d_least_squares_reconstruction_order)
        {
        case CUBIC:
            A(i, 9) = X[1] * X[1] * X[1];
            A(i, 8) = X[1] * X[1] * X[0];
            A(i, 7) = X[1] * X[0] * X[0];
            A(i, 6) = X[0] * X[0] * X[0];
            /* FALLTHROUGH */
        case QUADRATIC:
            A(i, 5) = X[1] * X[1];
            A(i, 4) = X[0] * X[1];
            A(i, 3) = X[0] * X[0];
            /* FALLTHROUGH */
        case LINEAR:
            A(i, 2) = X[1];
            A(i, 1) = X[0];
            /* FALLTHROUGH */
        case CONSTANT:
            A(i, 0) = 1.0;
            break;
        case UNKNOWN_ORDER:
            TBOX_ERROR("Unknown order.");
            break;
        }
    }

    VectorXd x = (Lambda * A).fullPivHouseholderQr().solve(Lambda * U);
    return x(0);
}
} // namespace IBAMR
