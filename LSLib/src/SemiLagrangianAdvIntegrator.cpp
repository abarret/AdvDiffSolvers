#include "ibamr/AdvDiffCUIConvectiveOperator.h"
#include "ibamr/AdvDiffPPMConvectiveOperator.h"
#include "ibamr/AdvDiffWavePropConvectiveOperator.h"
#include "ibamr/app_namespaces.h"

#include "LS/LSCartGridFunction.h"
#include "LS/SemiLagrangianAdvIntegrator.h"
#include "LS/utility_functions.h"

#include "SAMRAIVectorReal.h"
#include <tbox/Timer.h>
#include <tbox/TimerManager.h>

#include <Eigen/Core>
#include <Eigen/Dense>

extern "C"
{
#if (NDIM == 2)
    void integrate_paths_midpoint_(const double*,
                                   const int&,
                                   const double*,
                                   const double*,
                                   const int&,
                                   const double*,
                                   const double*,
                                   const int&,
                                   const double&,
                                   const double*,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const int&);
    void integrate_paths_forward_(const double*,
                                  const int&,
                                  const double*,
                                  const double*,
                                  const int&,
                                  const double&,
                                  const double*,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&);
    void integrate_paths_ls_midpoint_(const double*,
                                      const int&,
                                      const double*,
                                      const double*,
                                      const int&,
                                      const double*,
                                      const double*,
                                      const int&,
                                      const double*,
                                      const int&,
                                      const double&,
                                      const double*,
                                      const int&,
                                      const int&,
                                      const int&,
                                      const int&);
    void integrate_paths_ls_forward_(const double*,
                                     const int&,
                                     const double*,
                                     const double*,
                                     const int&,
                                     const double*,
                                     const int&,
                                     const double&,
                                     const double*,
                                     const int&,
                                     const int&,
                                     const int&,
                                     const int&);
#endif
}

namespace LS
{
namespace
{
static Timer* t_advective_step;
static Timer* t_diffusion_step;
static Timer* t_preprocess;
static Timer* t_postprocess;
static Timer* t_integrate_hierarchy;
static Timer* t_find_velocity;
static Timer* t_sum_zsplines;
static Timer* t_evaluate_mapping_vol;
static Timer* t_evaluate_mapping_ls;
static Timer* t_integrate_path_vol;
static Timer* t_integrate_path_ls;
static Timer* t_least_squares;

} // namespace
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
        d_adv_ts_type = string_to_enum<AdvectionTimeIntegrationMethod>(input_db->getString("advection_ts_type"));
        d_dif_ts_type = string_to_enum<DiffusionTimeIntegrationMethod>(input_db->getString("diffusion_ts_type"));
    }

    IBAMR_DO_ONCE(
        t_advective_step = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::advective_step");
        t_diffusion_step = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::diffusive_step");
        t_preprocess = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::preprocess");
        t_postprocess = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::postprocess");
        t_find_velocity = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::find_velocity");
        t_sum_zsplines = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::sum_zsplines");
        t_evaluate_mapping_vol =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::evaluate_mapping_vol");
        t_evaluate_mapping_ls =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::evaluate_mapping_ls");
        t_integrate_path_vol =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::integrage_path_vol");
        t_integrate_path_ls =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::integrate_path_ls");
        t_least_squares = TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::least_squares");
        t_integrate_hierarchy =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::integrate_hierarchy"););
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
    d_vol_wgt_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_VolWgtVar"));
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

Pointer<CellVariable<NDIM, double>>
SemiLagrangianAdvIntegrator::getAreaVariable(Pointer<CellVariable<NDIM, double>> ls_c_var)
{
    const size_t l =
        distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_c_var));
    return d_area_vars[l];
}

Pointer<CellVariable<NDIM, double>>
SemiLagrangianAdvIntegrator::getVolumeVariable(Pointer<CellVariable<NDIM, double>> ls_c_var)
{
    const size_t l =
        distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_c_var));
    return d_vol_vars[l];
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
        const Pointer<CellVariable<NDIM, double>>& vol_wgt_var = d_vol_wgt_vars[l];

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
        int vol_wgt_idx = var_db->registerVariableAndContext(vol_wgt_var, getCurrentContext());

        d_current_data.setFlag(ls_node_cur_idx);
        d_current_data.setFlag(vol_cur_idx);
        d_current_data.setFlag(area_cur_idx);
        d_current_data.setFlag(vol_wgt_idx);
        d_new_data.setFlag(ls_node_new_idx);
        d_new_data.setFlag(vol_new_idx);
        d_new_data.setFlag(area_new_idx);

        const std::string& ls_name = ls_cell_var->getName();
        d_visit_writer->registerPlotQuantity(ls_name + "_Node", "SCALAR", ls_node_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_volume", "SCALAR", vol_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_Cell", "SCALAR", ls_cell_cur_idx);
    }

    d_u_s_var = new SideVariable<NDIM, double>(d_object_name + "::USide");
    int u_new_idx, u_half_idx;
    registerVariable(u_new_idx, d_u_s_var, IntVector<NDIM>(1), getNewContext());
    registerVariable(u_half_idx, d_u_s_var, IntVector<NDIM>(1), getScratchContext());

    d_vol_fcn = new LSFindCellVolume(d_object_name + "::VolumeFunction", hierarchy);

    AdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    auto hier_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    d_hier_fc_data_ops =
        hier_ops_manager->getOperationsDouble(new FaceVariable<NDIM, double>("fc_var"), d_hierarchy, true);

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
    LS_TIMER_START(t_preprocess)
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
        double K = 0.0;
        switch (d_dif_ts_type)
        {
        case DiffusionTimeIntegrationMethod::BACKWARD_EULER:
            K = 1.0;
            break;
        case DiffusionTimeIntegrationMethod::TRAPEZOIDAL_RULE:
            K = 0.5;
            break;
        default:
            TBOX_ERROR(d_object_name << "::integrateHierarchy():\n"
                                     << "  unsupported diffusion time stepping type: "
                                     << enum_to_string<DiffusionTimeIntegrationMethod>(d_dif_ts_type) << "\n");
        }

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
            rhs_oper->initializeOperatorState(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
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
            helmholtz_solver->initializeSolverState(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
            d_helmholtz_solvers_need_init[l] = false;
        }
        l++;
    }
    LS_TIMER_STOP(t_preprocess);
}

void
SemiLagrangianAdvIntegrator::integrateHierarchy(const double current_time, const double new_time, const int cycle_num)
{
    LS_TIMER_START(t_integrate_hierarchy);
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
            const int u_new_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_cell_var], getNewContext());
            const int u_cur_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_cell_var], getCurrentContext());
            const int u_scr_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_cell_var], getScratchContext());
            const int u_s_half_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());
            const int u_s_new_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getNewContext());

            // Copy face data to side data
            copy_face_to_side(u_s_new_idx, u_new_idx, d_hierarchy);
            d_hier_fc_data_ops->linearSum(u_scr_idx, 0.5, u_new_idx, 0.5, u_cur_idx, true);
            copy_face_to_side(u_s_half_idx, u_scr_idx, d_hierarchy);

            using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<ITC> ghost_cell_comps(3);
            ghost_cell_comps[0] =
                ITC(u_s_new_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr);
            ghost_cell_comps[1] = ITC(
                u_s_half_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr);
            ghost_cell_comps[2] = ITC(ls_cell_scr_idx,
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
            integratePaths(d_path_idx, u_s_new_idx, u_s_half_idx, new_time - current_time);

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
    LS_TIMER_STOP(t_integrate_hierarchy);
}

void
SemiLagrangianAdvIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                                           const double new_time,
                                                           const bool skip_synchronize_new_state_data,
                                                           const int num_cycles)
{
    LS_TIMER_START(t_postprocess);
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        d_hierarchy->getPatchLevel(ln)->deallocatePatchData(d_adv_data);

    AdvDiffHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);
    LS_TIMER_STOP(t_postprocess);
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
            Pointer<LSCartGridFunction> Q_init = d_Q_init[Q_var];
            TBOX_ASSERT(Q_init);
            Q_init->setLSIndex(ls_node_cur_idx, vol_cur_idx);
            Q_init->setDataOnPatchHierarchy(Q_idx, Q_var, d_hierarchy, 0.0);
        }
    }
}

void
SemiLagrangianAdvIntegrator::regridHierarchyBeginSpecialized()
{
    updateWorkloadEstimates();
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
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    // Copy level set info
    for (size_t l = 0; l < d_ls_cell_vars.size(); ++l)
    {
        const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_cell_vars[l];
        const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
        const int ls_cell_new_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getNewContext());
        d_hier_cc_data_ops->copyData(ls_cell_cur_idx, ls_cell_new_idx);

        const Pointer<NodeVariable<NDIM, double>>& ls_node_var = d_ls_node_vars[l];
        const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getCurrentContext());
        const int ls_node_new_idx = var_db->mapVariableAndContextToIndex(ls_node_var, getNewContext());
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
                Pointer<NodeData<NDIM, double>> ls_cur_data = patch->getPatchData(ls_node_cur_idx);
                Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(ls_node_new_idx);
                ls_cur_data->copy(*ls_new_data);
            }
        }

        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
        d_hier_cc_data_ops->copyData(vol_cur_idx, vol_new_idx);

        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
        const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
        d_hier_cc_data_ops->copyData(area_cur_idx, area_new_idx);
    }

    AdvDiffHierarchyIntegrator::resetTimeDependentHierarchyDataSpecialized(new_time);
}

void
SemiLagrangianAdvIntegrator::resetHierarchyConfigurationSpecialized(
    const Pointer<BasePatchHierarchy<NDIM>> base_hierarchy,
    const int coarsest_ln,
    const int finest_ln)
{
    AdvDiffHierarchyIntegrator::resetHierarchyConfigurationSpecialized(base_hierarchy, coarsest_ln, finest_ln);
    d_hier_fc_data_ops->setPatchHierarchy(base_hierarchy);
    d_hier_fc_data_ops->resetLevels(0, finest_ln);

    // Reset solution and rhs vecs
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_sol_ls_vecs.resize(d_Q_var.size());
    d_rhs_ls_vecs.resize(d_Q_var.size());
    int l = 0;
    for (auto cit = d_Q_var.begin(); cit != d_Q_var.end(); ++cit, ++l)
    {
        const Pointer<CellVariable<NDIM, double>>& Q_var = *cit;
        const std::string& name = Q_var->getName();
        const int Q_scratch_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());

        const Pointer<CellVariable<NDIM, double>>& Q_rhs_var = d_Q_Q_rhs_map[Q_var];
        const int Q_rhs_scratch_idx = var_db->mapVariableAndContextToIndex(Q_rhs_var, getScratchContext());

        const Pointer<CellVariable<NDIM, double>> ls_c_var = d_Q_ls_map[Q_var];
        TBOX_ASSERT(ls_c_var);
        const size_t ll =
            std::distance(d_ls_cell_vars.begin(), std::find(d_ls_cell_vars.begin(), d_ls_cell_vars.end(), ls_c_var));
        const Pointer<CellVariable<NDIM, double>> vol_wgt_var = d_vol_wgt_vars[ll];
        const int vol_wgt_idx = var_db->mapVariableAndContextToIndex(vol_wgt_var, getCurrentContext());

        d_sol_ls_vecs[l] = new SAMRAIVectorReal<NDIM, double>(
            d_object_name + "::sol_vec::" + name, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        d_sol_ls_vecs[l]->addComponent(Q_var, Q_scratch_idx, vol_wgt_idx, d_hier_cc_data_ops);
        d_rhs_ls_vecs[l] = new SAMRAIVectorReal<NDIM, double>(
            d_object_name + "::rhs_vec::" + name, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        d_rhs_ls_vecs[l]->addComponent(Q_rhs_var, Q_rhs_scratch_idx, vol_wgt_idx, d_hier_cc_data_ops);
    }
}

void
SemiLagrangianAdvIntegrator::addWorkloadEstimate(Pointer<PatchHierarchy<NDIM>> hierarchy, const int workload_data_idx)
{
    plog << d_object_name << "::addWorkloadEstimate()"
         << "\n";
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> workload_data = patch->getPatchData(workload_data_idx);
            CellData<NDIM, double> temp_work_data(patch->getBox(), 1, 0);
            temp_work_data.fillAll(0.0);
            for (const auto& vol_var : d_vol_vars)
            {
                Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_var, getCurrentContext());
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if ((*vol_data)(idx) > 0.0) temp_work_data(idx) = 10.0;
                }
            }
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*workload_data)(idx) += temp_work_data(idx);
            }
        }
    }
}

/////////////////////// PRIVATE ///////////////////////////////
void
SemiLagrangianAdvIntegrator::advectionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const double current_time,
                                             const double new_time)
{
    LS_TIMER_START(t_advective_step);
    const double dt = new_time - current_time;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    int coarsest_ln = 0;
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    const Pointer<FaceVariable<NDIM, double>>& u_var = d_Q_u_map[Q_var];
    const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
    const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
    const int u_scr_idx = var_db->mapVariableAndContextToIndex(u_var, getScratchContext());
    const int u_s_new_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getNewContext());
    const int u_s_half_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());

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
        copy_face_to_side(u_s_new_idx, u_new_idx, d_hierarchy);
        d_hier_fc_data_ops->linearSum(u_scr_idx, 0.5, u_new_idx, 0.5, u_cur_idx, true);
        copy_face_to_side(u_s_half_idx, u_scr_idx, d_hierarchy);
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(3);
        HierarchyGhostCellInterpolation hier_ghost_cells;
        ghost_cell_comps[0] = ITC(ls_node_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        ghost_cell_comps[1] = ITC(u_s_new_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR");
        ghost_cell_comps[2] = ITC(u_s_half_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR");
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);
        // Integrate path
        integratePaths(d_path_idx, u_s_new_idx, u_s_half_idx, vol_new_idx, ls_node_new_idx, dt);
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
    LS_TIMER_STOP(t_advective_step);
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
    LS_TIMER_START(t_diffusion_step);
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    // We assume scratch context is already filled correctly.
    const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    const size_t l = distance(d_Q_var.begin(), std::find(d_Q_var.begin(), d_Q_var.end(), Q_var));

    const size_t ls_l =
        distance(d_ls_node_vars.begin(), std::find(d_ls_node_vars.begin(), d_ls_node_vars.end(), ls_var));
    const Pointer<CellVariable<NDIM, double>>& vol_wgt_var = d_vol_wgt_vars[ls_l];
    const int vol_wgt_idx = var_db->mapVariableAndContextToIndex(vol_wgt_var, getCurrentContext());
    const int wgt_idx = d_hier_math_ops->getCellWeightPatchDescriptorIndex();
    d_hier_cc_data_ops->multiply(vol_wgt_idx, vol_idx, wgt_idx);

    Pointer<LSCutCellLaplaceOperator> rhs_oper = d_helmholtz_rhs_ops[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(rhs_oper);
#endif
    rhs_oper->setTimeStepType(d_dif_ts_type);
    rhs_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var);
    rhs_oper->setSolutionTime(current_time);
    rhs_oper->apply(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);

    Pointer<PETScKrylovPoissonSolver> Q_helmholtz_solver = d_helmholtz_solvers[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_helmholtz_solver);
#endif
    Pointer<LSCutCellLaplaceOperator> solv_oper = Q_helmholtz_solver->getOperator();
#if !defined(NDEBUG)
    TBOX_ASSERT(solv_oper);
#endif
    solv_oper->setTimeStepType(d_dif_ts_type);
    solv_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var);
    solv_oper->setSolutionTime(new_time);
    Q_helmholtz_solver->solveSystem(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
    d_hier_cc_data_ops->copyData(Q_new_idx, Q_scr_idx);
    if (d_enable_logging)
    {
        plog << d_object_name << "::integrateHierarchy(): diffusion solve number of iterations = "
             << Q_helmholtz_solver->getNumIterations() << "\n";
        plog << d_object_name << "::integrateHierarchy(): diffusion solve residual norm        = "
             << Q_helmholtz_solver->getResidualNorm() << "\n";
    }
    LS_TIMER_STOP(t_diffusion_step);
}

void
SemiLagrangianAdvIntegrator::integratePaths(const int path_idx,
                                            const int u_new_idx,
                                            const int u_half_idx,
                                            const double dt)
{
    LS_TIMER_START(t_integrate_path_ls);
    // Integrate path to find \xx^{n+1}
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            const hier::Index<NDIM>& idx_low = box.lower();
            const hier::Index<NDIM>& idx_up = box.upper();

            Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
            Pointer<SideData<NDIM, double>> u_new_data = patch->getPatchData(u_new_idx);
            Pointer<SideData<NDIM, double>> u_half_data = patch->getPatchData(u_half_idx);
            TBOX_ASSERT(u_new_data && u_half_data);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            switch (d_adv_ts_type)
            {
            case AdvectionTimeIntegrationMethod::FORWARD_EULER:
                integrate_paths_forward_(path_data->getPointer(),
                                         path_data->getGhostCellWidth().max(),
                                         u_new_data->getPointer(0),
                                         u_new_data->getPointer(1),
                                         u_new_data->getGhostCellWidth().max(),
                                         dt,
                                         dx,
                                         idx_low(0),
                                         idx_low(1),
                                         idx_up(0),
                                         idx_up(1));
                break;
            case AdvectionTimeIntegrationMethod::MIDPOINT_RULE:
                integrate_paths_midpoint_(path_data->getPointer(),
                                          path_data->getGhostCellWidth().max(),
                                          u_new_data->getPointer(0),
                                          u_new_data->getPointer(1),
                                          u_new_data->getGhostCellWidth().max(),
                                          u_half_data->getPointer(0),
                                          u_half_data->getPointer(1),
                                          u_half_data->getGhostCellWidth().max(),
                                          dt,
                                          dx,
                                          idx_low(0),
                                          idx_low(1),
                                          idx_up(0),
                                          idx_up(1));
                break;
            default:
                TBOX_ERROR("UNKNOWN METHOD!\n");
            }
        }
    }
    LS_TIMER_STOP(t_integrate_path_ls);
}

void
SemiLagrangianAdvIntegrator::integratePaths(const int path_idx,
                                            const int u_new_idx,
                                            const int u_half_idx,
                                            const int vol_idx,
                                            const int ls_idx,
                                            const double dt)
{
    LS_TIMER_START(t_integrate_path_vol);
    // Integrate path to find \xx^{n+1}
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            const hier::Index<NDIM>& idx_low = box.lower();
            const hier::Index<NDIM>& idx_up = box.upper();

            Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<SideData<NDIM, double>> u_new_data = patch->getPatchData(u_new_idx);
            Pointer<SideData<NDIM, double>> u_half_data = patch->getPatchData(u_half_idx);
            TBOX_ASSERT(u_new_data && u_half_data);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            switch (d_adv_ts_type)
            {
            case AdvectionTimeIntegrationMethod::FORWARD_EULER:
                integrate_paths_ls_forward_(path_data->getPointer(),
                                            path_data->getGhostCellWidth().max(),
                                            u_new_data->getPointer(0),
                                            u_new_data->getPointer(1),
                                            u_new_data->getGhostCellWidth().max(),
                                            ls_data->getPointer(0),
                                            ls_data->getGhostCellWidth().max(),
                                            dt,
                                            dx,
                                            idx_low(0),
                                            idx_low(1),
                                            idx_up(0),
                                            idx_up(1));
                break;
            case AdvectionTimeIntegrationMethod::MIDPOINT_RULE:
                integrate_paths_ls_midpoint_(path_data->getPointer(),
                                             path_data->getGhostCellWidth().max(),
                                             u_new_data->getPointer(0),
                                             u_new_data->getPointer(1),
                                             u_new_data->getGhostCellWidth().max(),
                                             u_half_data->getPointer(0),
                                             u_half_data->getPointer(1),
                                             u_half_data->getGhostCellWidth().max(),
                                             ls_data->getPointer(),
                                             ls_data->getGhostCellWidth().max(),
                                             dt,
                                             dx,
                                             idx_low(0),
                                             idx_low(1),
                                             idx_up(0),
                                             idx_up(1));
                break;
            default:
                TBOX_ERROR("UNKNOWN METHOD!\n");
            }
        }
    }
    LS_TIMER_STOP(t_integrate_path_vol);
}

void
SemiLagrangianAdvIntegrator::evaluateMappingOnHierarchy(const int xstar_idx,
                                                        const int Q_cur_idx,
                                                        const int Q_new_idx,
                                                        const int order)
{
    LS_TIMER_START(t_evaluate_mapping_ls)
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
    LS_TIMER_STOP(t_evaluate_mapping_ls);
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
    LS_TIMER_START(t_evaluate_mapping_vol);
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
    LS_TIMER_STOP(t_evaluate_mapping_vol);
}

double
SemiLagrangianAdvIntegrator::sumOverZSplines(const IBTK::VectorNd& x_loc,
                                             const CellIndex<NDIM>& idx,
                                             const CellData<NDIM, double>& Q_data,
                                             const int order)
{
    LS_TIMER_START(t_sum_zsplines);
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
    LS_TIMER_STOP(t_sum_zsplines);
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
    LS_TIMER_START(t_least_squares);
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
    LS_TIMER_STOP(t_least_squares);
    return x(0);
}
} // namespace LS
