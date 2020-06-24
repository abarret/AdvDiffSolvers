#include "ibamr/AdvDiffCUIConvectiveOperator.h"
#include "ibamr/AdvDiffPPMConvectiveOperator.h"
#include "ibamr/AdvDiffWavePropConvectiveOperator.h"
#include "ibamr/app_namespaces.h"

#include "LS/QInitial.h"
#include "LS/SemiLagrangianAdvIntegrator.h"
#include "LS/utility_functions.h"

#include "SAMRAIVectorReal.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace LS
{
int SemiLagrangianAdvIntegrator::GHOST_CELL_WIDTH = 4;

SemiLagrangianAdvIntegrator::SemiLagrangianAdvIntegrator(const std::string& object_name,
                                                         Pointer<Database> input_db,
                                                         bool register_for_restart)
    : AdvDiffHierarchyIntegrator(object_name, input_db, register_for_restart),
      d_path_var(new CellVariable<NDIM, double>(d_object_name + "::PathVar", NDIM))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_path_idx = var_db->registerVariableAndContext(d_path_var, var_db->getContext(d_object_name + "::PathContext"));
    d_adv_data.setFlag(d_path_idx);

    if (input_db)
    {
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
    d_ls_cell_vars.push_back(ls_var);
    d_ls_node_vars.push_back(new NodeVariable<NDIM, double>(ls_var->getName() + "_NodeVar"));
    d_vol_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_VolVar"));
    d_area_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_AreaVar"));
    d_ls_fcn_map[ls_var] = nullptr;
    d_ls_strategy_map[ls_var] = nullptr;
    d_ls_use_fcn[ls_var] = false;
    d_ls_use_ls_for_tagging[ls_var] = true;
    d_ls_u_map[ls_var] = nullptr;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetVelocity(Pointer<CellVariable<NDIM, double>> ls_var,
                                                      Pointer<FaceVariable<NDIM, double>> u_var)
{
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    TBOX_ASSERT(std::find(d_u_var.begin(), d_u_var.end(), u_var) != d_u_var.end());
    d_ls_u_map[ls_var] = u_var;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetResetFunction(Pointer<CellVariable<NDIM, double>> ls_var,
                                                           Pointer<LSInitStrategy> ls_strategy)
{
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    d_ls_strategy_map[ls_var] = ls_strategy;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetFunction(Pointer<CellVariable<NDIM, double>> ls_var,
                                                      Pointer<CartGridFunction> ls_fcn)
{
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    d_ls_fcn_map[ls_var] = ls_fcn;
}

void
SemiLagrangianAdvIntegrator::restrictToLevelSet(Pointer<CellVariable<NDIM, double>> Q_var,
                                                Pointer<CellVariable<NDIM, double>> ls_var)
{
    TBOX_ASSERT(std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) != d_Q_var.end());
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    d_Q_ls_map[Q_var] = ls_var;
}

void
SemiLagrangianAdvIntegrator::useLevelSetFunction(Pointer<CellVariable<NDIM, double>> ls_var, const bool use_ls_function)
{
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    d_ls_use_fcn[ls_var] = use_ls_function;
}

void
SemiLagrangianAdvIntegrator::useLevelSetForTagging(Pointer<CellVariable<NDIM, double>> ls_var,
                                                   const bool use_ls_for_tagging)
{
    TBOX_ASSERT(std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_var) != d_ls_cell_vars.end());
    d_ls_use_ls_for_tagging[ls_var] = use_ls_for_tagging;
}

Pointer<NodeVariable<NDIM, double>>
SemiLagrangianAdvIntegrator::getLevelSetNodeVariable(Pointer<CellVariable<NDIM, double>> ls_c_var)
{
    const size_t l =
        distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_c_var));
    return d_ls_node_vars[l];
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
    for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
    {
        const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
        const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];

        int ls_cell_cur_idx, ls_cell_new_idx, ls_cell_scr_idx;
        int ls_node_cur_idx, ls_node_new_idx;
        registerVariable(ls_cell_cur_idx,
                         ls_cell_new_idx,
                         ls_cell_scr_idx,
                         ls_cell_var,
                         IntVector<NDIM>(GHOST_CELL_WIDTH),
                         "CONSERVATIVE_COARSEN",
                         "CONSERVATIVE_LINEAR_REFINE");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        ls_node_cur_idx = var_db->registerVariableAndContext(ls_node_var, getCurrentContext(), GHOST_CELL_WIDTH);
        ls_node_new_idx = var_db->registerVariableAndContext(ls_node_var, getNewContext(), GHOST_CELL_WIDTH);
        int vol_cur_idx = var_db->registerVariableAndContext(vol_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int vol_new_idx = var_db->registerVariableAndContext(vol_var, getNewContext(), GHOST_CELL_WIDTH);
        int area_cur_idx = var_db->registerVariableAndContext(area_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int area_new_idx = var_db->registerVariableAndContext(area_var, getNewContext(), GHOST_CELL_WIDTH);

        d_ls_data.setFlag(ls_node_cur_idx);
        d_ls_data.setFlag(ls_node_new_idx);
        d_ls_data.setFlag(vol_cur_idx);
        d_ls_data.setFlag(vol_new_idx);
        d_ls_data.setFlag(area_cur_idx);
        d_ls_data.setFlag(area_new_idx);

        const std::string& ls_name = ls_cell_var->getName();
        d_visit_writer->registerPlotQuantity(ls_name + "_Volume_previous", "SCALAR", vol_new_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_LS_previous", "SCALAR", ls_node_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_LS_currrent", "SCALAR", ls_node_new_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_Volume_current", "SCALAR", vol_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_LS_Cell", "SCALAR", ls_cell_cur_idx);
    }

    d_u_s_var = new SideVariable<NDIM, double>(d_object_name + "::USide");
    int u_s_idx;
    registerVariable(u_s_idx, d_u_s_var, IntVector<NDIM>(1), getScratchContext());

    d_vol_fcn = new LSFindCellVolume(d_object_name + "::VolumeFunction", hierarchy);

    AdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

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
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
        {
            const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
            const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
            const Pointer<CartGridFunction>& ls_fcn = d_ls_fcn_map[ls_cell_var];
            TBOX_ASSERT(ls_fcn);
            const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
            const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
            ls_fcn->setDataOnPatchLevel(ls_cell_cur_idx, ls_cell_var, level, data_time, initial_time);
            ls_fcn->setDataOnPatchLevel(ls_node_cur_idx, ls_node_var, level, data_time, initial_time);
        }
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
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CellData<NDIM, int>> tag_data = patch->getPatchData(tag_index);
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        for (const auto& ls_cell_var : d_ls_cell_vars)
        {
            if (!d_ls_use_ls_for_tagging[ls_cell_var]) continue;
            const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
            Pointer<CellData<NDIM, double>> ls_data = patch->getPatchData(ls_cell_cur_idx);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const double ls = (*ls_data)(idx);
                if (ls < d_max_ls_refine_factor * dx[0] && ls > d_min_ls_refine_factor * dx[0]) (*tag_data)(idx) = 1;
            }
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
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
    {
        const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
        const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
        const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
        const int ls_cell_scr_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getScratchContext());
        const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
        if (d_ls_use_fcn[ls_cell_var])
        {
            const Pointer<CartGridFunction>& ls_fcn = d_ls_fcn_map[ls_cell_var];
            ls_fcn->setDataOnPatchHierarchy(ls_cell_cur_idx, ls_cell_var, d_hierarchy, current_time, false);
            ls_fcn->setDataOnPatchHierarchy(ls_node_cur_idx, ls_node_var, d_hierarchy, current_time, false);
        }
        else
        {
            const Pointer<LSInitStrategy>& ls_strategy = d_ls_strategy_map[ls_cell_var];
            ls_strategy->initializeLSData(ls_cell_cur_idx, d_hier_math_ops, d_integrator_step, current_time, false);
            // interpolate cell data to node data
            using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<ITC> ghost_cell_comps(1);
            ghost_cell_comps[0] = ITC(ls_cell_scr_idx,
                                      ls_cell_cur_idx,
                                      "CONSERVATIVE_LINEAR_REFINE",
                                      false,
                                      "CONSERVATIVE_COARSEN",
                                      "LINEAR",
                                      false,
                                      nullptr);
            Pointer<HierarchyGhostCellInterpolation> hier_ghost_cell = new HierarchyGhostCellInterpolation();
            hier_ghost_cell->initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
            d_hier_math_ops->interp(
                ls_node_cur_idx, ls_node_var, false, ls_cell_scr_idx, ls_cell_var, hier_ghost_cell, current_time);
            hier_ghost_cell->deallocateOperatorState();
            ghost_cell_comps[0] = ITC(ls_node_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
            hier_ghost_cell->initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
            hier_ghost_cell->fillData(current_time);
        }

        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(d_vol_vars[l], getCurrentContext());
        const int area_cur_idx = var_db->mapVariableAndContextToIndex(d_area_vars[l], getCurrentContext());
        d_vol_fcn->updateVolumeAndArea(
            vol_cur_idx, d_vol_vars[l], area_cur_idx, d_area_vars[l], ls_node_cur_idx, ls_node_var, true);
    }

    // Set velocities
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

        const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_Q_ls_map[Q_var];
        const size_t l =
            distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_cell_var));
        const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
        // Fill ghost cells for ls_node_cur
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] = ITC(ls_node_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        HierarchyGhostCellInterpolation hier_ghost_cell;
        hier_ghost_cell.initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        hier_ghost_cell.fillData(current_time);

        d_vol_fcn->updateVolumeAndArea(
            vol_cur_idx, vol_var, area_cur_idx, area_var, ls_node_cur_idx, ls_node_var, true);

        // Copy current data to scratch
        d_hier_cc_data_ops->copyData(Q_scr_idx, Q_cur_idx);

        // First do a diffusion update.
        // Note diffusion update fills in "New" context
        diffusionUpdate(Q_var,
                        ls_node_cur_idx,
                        ls_node_var,
                        vol_cur_idx,
                        vol_var,
                        area_cur_idx,
                        area_var,
                        current_time,
                        d_use_strang_splitting ? half_time : new_time);

        plog << d_object_name + "::integrateHierarchy() finished diffusion update for variable: " << Q_var->getName()
             << "\n";

        // TODO: Should we synchronize hierarchy?
    }

    // Update Level sets
    for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
    {
        const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
        const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
        const int ls_cell_new_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getNewContext());
        const int ls_node_new_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getNewContext());
        if (d_ls_use_fcn[ls_cell_var])
        {
            plog << d_object_name + "::integrateHierarchy() prescribing level set " << ls_cell_var->getName()
                 << " at time " << new_time << "\n";
            const Pointer<CartGridFunction>& ls_fcn = d_ls_fcn_map[ls_cell_var];
            ls_fcn->setDataOnPatchHierarchy(
                ls_cell_new_idx, ls_cell_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
            ls_fcn->setDataOnPatchHierarchy(
                ls_node_new_idx, ls_node_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
        }
        else
        {
            const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
            const int ls_cell_scr_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getScratchContext());
            plog << d_object_name + "::integrateHierarchy() evolving level set " << ls_cell_var->getName()
                 << " at time " << new_time << "\n";
            const int u_cur_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_cell_var], getCurrentContext());
            const int u_s_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());
            using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<ITC> ghost_cell_comps(2);
            // Copy face data to side data
            copy_face_to_side(u_s_idx, u_cur_idx, d_hierarchy);
            ghost_cell_comps[0] =
                ITC(u_s_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr);
            ghost_cell_comps[1] = ITC(ls_cell_scr_idx,
                                      ls_cell_cur_idx,
                                      "CONSERVATIVE_LINEAR_REFINE",
                                      false,
                                      "NONE",
                                      "LINEAR",
                                      false,
                                      nullptr);
            Pointer<HierarchyGhostCellInterpolation> hier_ghost_cell = new HierarchyGhostCellInterpolation();
            hier_ghost_cell->initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
            hier_ghost_cell->fillData(current_time);
            hier_ghost_cell->deallocateOperatorState();
            integratePaths(d_path_idx, u_s_idx, new_time - current_time);

            // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next
            // iteration
            evaluateMappingOnHierarchy(d_path_idx, ls_cell_scr_idx, ls_cell_new_idx, /*order*/ 2);

            // Synchronize hierarchy and interpolate to cell nodes
            // TODO: Should we synchronize hierarchy?
            ghost_cell_comps.resize(1);
            ghost_cell_comps[0] = ITC(ls_cell_scr_idx,
                                      ls_cell_new_idx,
                                      "CONSERVATIVE_LINEAR_REFINE",
                                      false,
                                      "CONSERVATIVE_COARSEN",
                                      "LINEAR",
                                      false,
                                      nullptr);

            hier_ghost_cell->initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
            d_hier_math_ops->interp(
                ls_node_new_idx, ls_node_var, false, ls_cell_scr_idx, ls_cell_var, hier_ghost_cell, new_time);
            hier_ghost_cell->deallocateOperatorState();
            ghost_cell_comps[0] = ITC(ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
            hier_ghost_cell->initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
            hier_ghost_cell->fillData(current_time);
        }
        // Fill ghost cells and update volume
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] = ITC(ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        HierarchyGhostCellInterpolation hier_ghost_cell;
        hier_ghost_cell.initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        hier_ghost_cell.fillData(new_time);
        // Now update volume and area for new context
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
        const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
        d_vol_fcn->updateVolumeAndArea(
            vol_new_idx, vol_var, area_new_idx, area_var, ls_node_new_idx, ls_node_var, true);
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

            const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_Q_ls_map[Q_var];
            const size_t l =
                distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_cell_var));
            const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
            const int ls_node_new_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getNewContext());
            const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
            const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());

            // Copy current data to scratch
            d_hier_cc_data_ops->copyData(Q_cur_idx, Q_new_idx);
            d_hier_cc_data_ops->copyData(Q_scr_idx, Q_new_idx);

            // First do a diffusion update.
            // Note diffusion update fills in "New" context
            diffusionUpdate(
                Q_var, ls_node_new_idx, ls_node_var, vol_new_idx, vol_var, area_new_idx, area_var, half_time, new_time);

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
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        // Set initial level set data
        for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
        {
            const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
            const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];

            const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());

            d_ls_fcn_map[ls_cell_var]->setDataOnPatchHierarchy(ls_node_cur_idx,
                                                               ls_node_var,
                                                               d_hierarchy,
                                                               current_time,
                                                               initial_time,
                                                               0,
                                                               d_hierarchy->getFinestLevelNumber());
            d_vol_fcn->updateVolumeAndArea(
                vol_cur_idx, vol_var, IBTK::invalid_index, nullptr, ls_node_cur_idx, ls_node_var, false);
        }
        for (const auto& Q_var : d_Q_var)
        {
            const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_Q_ls_map[Q_var];
            const size_t l =
                distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_cell_var));
            const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            Pointer<QInitial> Q_init = d_Q_init[Q_var];
            TBOX_ASSERT(Q_init);
            Q_init->setLSIndex(ls_node_cur_idx, vol_cur_idx);
            Q_init->setDataOnPatchHierarchy(Q_idx, Q_var, d_hierarchy, 0.0);
        }
    }
}

void
SemiLagrangianAdvIntegrator::regridHierarchyEndSpecialized()
{
    // Force reinitialization of level set after regrid.
    for (const auto& ls_cell_var : d_ls_cell_vars)
    {
        if (!d_ls_use_fcn[ls_cell_var]) d_ls_strategy_map[ls_cell_var]->setReinitializeLSData(true);
    }
}

void
SemiLagrangianAdvIntegrator::resetTimeDependentHierarchyDataSpecialized(const double new_time)
{
    // Copy level set info
    for (const auto& ls_cell_var : d_ls_cell_vars)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
        const int ls_cell_new_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getNewContext());
        d_hier_cc_data_ops->copyData(ls_cell_cur_idx, ls_cell_new_idx);
    }

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
    const Pointer<FaceVariable<NDIM, double>>& u_var = d_Q_u_map[Q_var];
    const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
    const int u_s_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());

    const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_Q_ls_map[Q_var];
    const size_t l =
        distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_cell_var));
    const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
    const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
    const int ls_node_new_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getNewContext());
    const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
    const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
    const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());

    {
        copy_face_to_side(u_s_idx, u_new_idx, d_hierarchy);
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(2);
        HierarchyGhostCellInterpolation hier_ghost_cells;
        ghost_cell_comps[0] = ITC(ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        ghost_cell_comps[1] = ITC(u_s_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR");
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);
        // Integrate path
        integratePaths(d_path_idx, u_s_idx, vol_new_idx, ls_node_new_idx, dt);
    }

    {
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(2);
        ghost_cell_comps[0] =
            ITC(d_Q_scratch_idx, Q_cur_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
        ghost_cell_comps[1] = ITC(ls_node_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        HierarchyGhostCellInterpolation hier_ghost_cells;
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);

        d_vol_fcn->updateVolumeAndArea(vol_cur_idx, vol_var, -1, nullptr, ls_node_cur_idx, ls_node_var, true);

        // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
        evaluateMappingOnHierarchy(
            d_path_idx, d_Q_scratch_idx, vol_cur_idx, Q_new_idx, vol_new_idx, ls_node_cur_idx, /*order*/ 1);
    }
}

void
SemiLagrangianAdvIntegrator::diffusionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const int ls_idx,
                                             Pointer<NodeVariable<NDIM, double>> ls_var,
                                             const int vol_idx,
                                             Pointer<CellVariable<NDIM, double>> vol_var,
                                             const int area_idx,
                                             Pointer<CellVariable<NDIM, double>> area_var,
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
    rhs_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var);
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
    solv_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var);
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
            Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            TBOX_ASSERT(u_data);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd com = { idx(0) + 0.5, idx(1) + 0.5 };
                const VectorNd& u = findVelocity(idx, *u_data, com);
                for (int d = 0; d < NDIM; ++d)
                {
                    (*path_data)(idx, d) = (idx(d) + 0.5) - dt * u(d) / dx[d];
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
            Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            TBOX_ASSERT(u_data);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd com = find_cell_centroid(idx, *ls_data);
                const VectorNd& u = findVelocity(idx, *u_data, com);
                for (int d = 0; d < NDIM; ++d) (*path_data)(idx, d) = com(d) - dt * u(d) / dx[d];
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
        Lambda(i, i) = std::sqrt(weight(static_cast<double>((X_vals[i] - x_loc).norm())));
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

VectorNd
SemiLagrangianAdvIntegrator::findVelocity(const CellIndex<NDIM>& idx,
                                          const SideData<NDIM, double>& u_data,
                                          const VectorNd& x_pt)
{
    VectorNd u;
    // First x-axis:
    // Determine if we should use the "lower" or "upper" values.
    {
        VectorNd x_low;
        SideIndex<NDIM> idx_ll, idx_ul, idx_lu, idx_uu;
        if (x_pt(1) > (idx(1) + 0.5))
        {
            // Use "upper" points
            for (int d = 0; d < NDIM; ++d) x_low(d) = static_cast<double>(idx(d)) + (d == 1 ? 0.5 : 0.0);
            idx_ll = SideIndex<NDIM>(idx, 0, 0);
            idx_ul = SideIndex<NDIM>(idx, 0, 1);
            idx_lu = SideIndex<NDIM>(idx + IntVector<NDIM>(0, 1), 0, 0);
            idx_uu = SideIndex<NDIM>(idx + IntVector<NDIM>(0, 1), 0, 1);
        }
        else
        {
            // Use "lower" points
            for (int d = 0; d < NDIM; ++d) x_low(d) = static_cast<double>(idx(d)) - (d == 1 ? 0.5 : 0.0);
            idx_lu = SideIndex<NDIM>(idx, 0, 0);
            idx_uu = SideIndex<NDIM>(idx, 0, 1);
            idx_ll = SideIndex<NDIM>(idx - IntVector<NDIM>(0, 1), 0, 0);
            idx_ul = SideIndex<NDIM>(idx - IntVector<NDIM>(0, 1), 0, 1);
        }
        u(0) = u_data(idx_ll) + (u_data(idx_ul) - u_data(idx_ll)) * (x_pt(0) - x_low(0)) +
               (u_data(idx_lu) - u_data(idx_ll)) * (x_pt(1) - x_low(1)) +
               (u_data(idx_uu) - u_data(idx_lu) - u_data(idx_ul) + u_data(idx_ll)) * (x_pt(0) - x_low(0)) *
                   (x_pt(1) - x_low(1));
    }

    // Second y-axis
    {
        VectorNd x_low;
        SideIndex<NDIM> idx_ll, idx_ul, idx_lu, idx_uu;
        if (x_pt(0) > (idx(0) + 0.5))
        {
            // Use "upper" points
            for (int d = 0; d < NDIM; ++d) x_low(d) = static_cast<double>(idx(d)) + (d == 0 ? 0.5 : 0.0);
            idx_ll = SideIndex<NDIM>(idx, 1, 0);
            idx_lu = SideIndex<NDIM>(idx, 1, 1);
            idx_ul = SideIndex<NDIM>(idx + IntVector<NDIM>(1, 0), 1, 0);
            idx_uu = SideIndex<NDIM>(idx + IntVector<NDIM>(1, 0), 1, 1);
        }
        else
        {
            // Use "lower" points
            for (int d = 0; d < NDIM; ++d) x_low(d) = static_cast<double>(idx(d)) - (d == 0 ? 0.5 : 0.0);
            idx_ul = SideIndex<NDIM>(idx, 1, 0);
            idx_uu = SideIndex<NDIM>(idx, 1, 1);
            idx_ll = SideIndex<NDIM>(idx - IntVector<NDIM>(1, 0), 1, 0);
            idx_lu = SideIndex<NDIM>(idx - IntVector<NDIM>(1, 0), 1, 1);
        }
        u(1) = u_data(idx_ll) + (u_data(idx_ul) - u_data(idx_ll)) * (x_pt(0) - x_low(0)) +
               (u_data(idx_lu) - u_data(idx_ll)) * (x_pt(1) - x_low(1)) +
               (u_data(idx_uu) - u_data(idx_lu) - u_data(idx_ul) + u_data(idx_ll)) * (x_pt(0) - x_low(0)) *
                   (x_pt(1) - x_low(1));
    }

    return u;
}
} // namespace LS
