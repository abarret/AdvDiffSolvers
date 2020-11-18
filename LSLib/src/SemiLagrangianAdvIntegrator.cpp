#include "ibamr/AdvDiffCUIConvectiveOperator.h"
#include "ibamr/AdvDiffPPMConvectiveOperator.h"
#include "ibamr/AdvDiffWavePropConvectiveOperator.h"
#include "ibamr/app_namespaces.h"

#include "LS/LSCartGridFunction.h"
#include "LS/SBBoundaryConditions.h"
#include "LS/SemiLagrangianAdvIntegrator.h"
#include "LS/utility_functions.h"

#include "HierarchyDataOpsManager.h"
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
#if (NDIM == 3)
    void integrate_paths_midpoint_(const double*,
                                   const int&,
                                   const double*,
                                   const double*,
                                   const double*,
                                   const int&,
                                   const double*,
                                   const double*,
                                   const double*,
                                   const int&,
                                   const double&,
                                   const double*,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const int&);
    void integrate_paths_forward_(const double*,
                                  const int&,
                                  const double*,
                                  const double*,
                                  const double*,
                                  const int&,
                                  const double&,
                                  const double*,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&);
    void integrate_paths_ls_midpoint_(const double*,
                                      const int&,
                                      const double*,
                                      const double*,
                                      const double*,
                                      const int&,
                                      const double*,
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
                                      const int&,
                                      const int&,
                                      const int&);
    void integrate_paths_ls_forward_(const double*,
                                     const int&,
                                     const double*,
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
static Timer* t_rbf_reconstruct;
static Timer* t_find_cell_centroid;

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
        d_use_rbfs = input_db->getBool("use_rbfs");
        d_rbf_stencil_size = input_db->getInteger("rbf_stencil_size");
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
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::integrate_hierarchy");
        t_rbf_reconstruct =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::radial_basis_reconstruction");
        t_find_cell_centroid =
            TimerManager::getManager()->getTimer("LS::SemiLagrangianAdvIntegrator::find_cell_centroid"));
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
SemiLagrangianAdvIntegrator::registerLevelSetVariable(Pointer<NodeVariable<NDIM, double>> ls_var)
{
    d_ls_vars.push_back(ls_var);
    d_vol_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_VolVar"));
    d_area_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_AreaVar"));
    d_vol_wgt_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_VolWgtVar"));
    d_side_vars.push_back(new SideVariable<NDIM, double>(ls_var->getName() + "_SideVar"));
    d_ls_vol_fcn_map[ls_var] = nullptr;
    d_ls_use_ls_for_tagging[ls_var] = true;
    d_ls_u_map[ls_var] = nullptr;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetVelocity(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                      Pointer<FaceVariable<NDIM, double>> u_var)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    TBOX_ASSERT(std::find(d_u_var.begin(), d_u_var.end(), u_var) != d_u_var.end());
    d_ls_u_map[ls_var] = u_var;
}

void
SemiLagrangianAdvIntegrator::registerLevelSetVolFunction(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                         Pointer<LSFindCellVolume> ls_fcn)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_vol_fcn_map[ls_var] = ls_fcn;
}

void
SemiLagrangianAdvIntegrator::restrictToLevelSet(Pointer<CellVariable<NDIM, double>> Q_var,
                                                Pointer<NodeVariable<NDIM, double>> ls_var)
{
    TBOX_ASSERT(std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) != d_Q_var.end());
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_Q_ls_map[Q_var] = ls_var;
}

void
SemiLagrangianAdvIntegrator::useLevelSetForTagging(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                   const bool use_ls_for_tagging)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_use_ls_for_tagging[ls_var] = use_ls_for_tagging;
}

Pointer<CellVariable<NDIM, double>>
SemiLagrangianAdvIntegrator::getAreaVariable(Pointer<NodeVariable<NDIM, double>> ls_var)
{
    const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
    return d_area_vars[l];
}

Pointer<CellVariable<NDIM, double>>
SemiLagrangianAdvIntegrator::getVolumeVariable(Pointer<NodeVariable<NDIM, double>> ls_var)
{
    const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
    return d_vol_vars[l];
}

void
SemiLagrangianAdvIntegrator::registerSBIntegrator(Pointer<SBIntegrator> sb_integrator,
                                                  Pointer<NodeVariable<NDIM, double>> ls_var)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_sb_integrator_ls_map[sb_integrator] = ls_var;
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
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_wgt_var = d_vol_wgt_vars[l];
        const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        int ls_node_cur_idx = var_db->registerVariableAndContext(ls_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int ls_node_new_idx = var_db->registerVariableAndContext(ls_var, getNewContext(), GHOST_CELL_WIDTH);
        int vol_cur_idx = var_db->registerVariableAndContext(vol_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int vol_new_idx = var_db->registerVariableAndContext(vol_var, getNewContext(), GHOST_CELL_WIDTH);
        int area_cur_idx = var_db->registerVariableAndContext(area_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int area_new_idx = var_db->registerVariableAndContext(area_var, getNewContext(), GHOST_CELL_WIDTH);
        int vol_wgt_idx = var_db->registerVariableAndContext(vol_wgt_var, getCurrentContext());
        int side_cur_idx = var_db->registerVariableAndContext(side_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int side_new_idx = var_db->registerVariableAndContext(side_var, getNewContext(), GHOST_CELL_WIDTH);

        d_current_data.setFlag(ls_node_cur_idx);
        d_current_data.setFlag(vol_cur_idx);
        d_current_data.setFlag(area_cur_idx);
        d_current_data.setFlag(vol_wgt_idx);
        d_current_data.setFlag(side_cur_idx);
        d_new_data.setFlag(ls_node_new_idx);
        d_new_data.setFlag(vol_new_idx);
        d_new_data.setFlag(area_new_idx);
        d_new_data.setFlag(side_new_idx);

        const std::string& ls_name = ls_var->getName();
        d_visit_writer->registerPlotQuantity(ls_name + "_Node", "SCALAR", ls_node_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_volume", "SCALAR", vol_cur_idx);
        d_visit_writer->registerPlotQuantity(ls_name + "_area", "SCALAR", area_cur_idx);
    }

    d_u_s_var = new SideVariable<NDIM, double>(d_object_name + "::USide");
    int u_new_idx, u_half_idx;
    registerVariable(u_new_idx, d_u_s_var, IntVector<NDIM>(1), getNewContext());
    registerVariable(u_half_idx, d_u_s_var, IntVector<NDIM>(1), getScratchContext());

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
        for (size_t l = 0; l < d_ls_vars.size(); ++l)
        {
            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
            const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];

            const Pointer<LSFindCellVolume>& vol_fcn = d_ls_vol_fcn_map[ls_var];
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            const int side_cur_idx = var_db->mapVariableAndContextToIndex(side_var, getCurrentContext());
            vol_fcn->updateVolumeAreaSideLS(
                vol_cur_idx, vol_var, area_cur_idx, area_var, side_cur_idx, side_var, ls_cur_idx, ls_var, 0.0, false);
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
        for (const auto& ls_var : d_ls_vars)
        {
            if (!d_ls_use_ls_for_tagging[ls_var]) continue;
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_cur_idx);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const double ls = node_to_cell(idx, *ls_data);
                if (ls < d_max_ls_refine_factor * dx[0] && ls > d_min_ls_refine_factor * dx[0]) (*tag_data)(idx) = 1;
            }
        }
    }
}

int
SemiLagrangianAdvIntegrator::getNumberOfCycles() const
{
    return 2;
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
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];
        const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        const int side_cur_idx = var_db->mapVariableAndContextToIndex(side_var, getCurrentContext());
        const Pointer<LSFindCellVolume>& vol_fcn = d_ls_vol_fcn_map[ls_var];
        vol_fcn->updateVolumeAreaSideLS(vol_cur_idx,
                                        vol_var,
                                        area_cur_idx,
                                        area_var,
                                        side_cur_idx,
                                        side_var,
                                        ls_cur_idx,
                                        ls_var,
                                        current_time,
                                        true);
    }

    // Set velocities
    for (const auto& u_var : d_u_var)
    {
        const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
        const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
        if (d_u_fcn[u_var])
        {
            d_u_fcn[u_var]->setDataOnPatchHierarchy(
                u_cur_idx, u_var, d_hierarchy, current_time, false, coarsest_ln, finest_ln);
            d_u_fcn[u_var]->setDataOnPatchHierarchy(
                u_new_idx, u_var, d_hierarchy, new_time, false, coarsest_ln, finest_ln);
        }
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
        Pointer<LSCutCellLaplaceOperator> rhs_oper = d_helmholtz_rhs_ops[l];
        rhs_oper->setPoissonSpecifications(rhs_spec);
        rhs_oper->setPhysicalBcCoefs(Q_bc_coef);
        rhs_oper->setHomogeneousBc(false);
        rhs_oper->setSolutionTime(current_time);
        rhs_oper->setTimeInterval(current_time, new_time);

        Pointer<PoissonSolver> helmholtz_solver = d_helmholtz_solvers[l];
        helmholtz_solver->setPoissonSpecifications(solv_spec);
        helmholtz_solver->setPhysicalBcCoefs(Q_bc_coef);
        helmholtz_solver->setHomogeneousBc(false);
        helmholtz_solver->setSolutionTime(new_time);
        helmholtz_solver->setTimeInterval(current_time, new_time);
        l++;
    }

    for (auto& sb_ls_pair : d_sb_integrator_ls_map)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = sb_ls_pair.second;
        const unsigned int l = std::distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int vol_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        Pointer<SBIntegrator> sb_integrator = sb_ls_pair.first;
        sb_integrator->setLSData(ls_idx, vol_idx, d_hierarchy);
        sb_integrator->beginTimestepping(current_time, new_time);
    }
    LS_TIMER_STOP(t_preprocess);
}

void
SemiLagrangianAdvIntegrator::integrateHierarchy(const double current_time, const double new_time, const int cycle_num)
{
    AdvDiffHierarchyIntegrator::integrateHierarchy(current_time, new_time, cycle_num);
    if (cycle_num == 0) return;
    LS_TIMER_START(t_integrate_hierarchy);
    const double half_time = current_time + 0.5 * (new_time - current_time);
    auto var_db = VariableDatabase<NDIM>::getDatabase();

    for (auto& sb_ls_pair : d_sb_integrator_ls_map)
        sb_ls_pair.first->integrateHierarchy(getCurrentContext(), current_time, new_time);

    for (const auto& Q_var : d_Q_var)
    {
        const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());

        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
        const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];
        const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
        const int side_cur_idx = var_db->mapVariableAndContextToIndex(side_var, getCurrentContext());
        // Fill ghost cells for ls_node_cur
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] = ITC(ls_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        HierarchyGhostCellInterpolation hier_ghost_cell;
        hier_ghost_cell.initializeOperatorState(ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        hier_ghost_cell.fillData(current_time);

        // Copy current data to scratch
        d_hier_cc_data_ops->copyData(Q_scr_idx, Q_cur_idx);

        // First do a diffusion update.
        // Note diffusion update fills in "New" context
        diffusionUpdate(Q_var,
                        ls_cur_idx,
                        ls_var,
                        vol_cur_idx,
                        vol_var,
                        area_cur_idx,
                        area_var,
                        side_cur_idx,
                        side_var,
                        current_time,
                        d_use_strang_splitting ? half_time : new_time);

        plog << d_object_name + "::integrateHierarchy() finished diffusion update for variable: " << Q_var->getName()
             << "\n";

        // TODO: Should we synchronize hierarchy?
    }

    // Update Level sets
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
        const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];

        const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
        const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
        const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
        const int side_new_idx = var_db->mapVariableAndContextToIndex(side_var, getNewContext());

        const Pointer<LSFindCellVolume>& ls_fcn = d_ls_vol_fcn_map[ls_var];
        ls_fcn->updateVolumeAreaSideLS(
            vol_new_idx, vol_var, area_new_idx, area_var, side_new_idx, side_var, ls_new_idx, ls_var, new_time, true);
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

            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
            const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
            const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];
            const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
            const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
            const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
            const int side_new_idx = var_db->mapVariableAndContextToIndex(side_var, getNewContext());

            // Copy current data to scratch
            d_hier_cc_data_ops->copyData(Q_cur_idx, Q_new_idx);
            d_hier_cc_data_ops->copyData(Q_scr_idx, Q_new_idx);

            // First do a diffusion update.
            // Note diffusion update fills in "New" context
            diffusionUpdate(Q_var,
                            ls_new_idx,
                            ls_var,
                            vol_new_idx,
                            vol_var,
                            area_new_idx,
                            area_var,
                            side_new_idx,
                            side_var,
                            half_time,
                            new_time);

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

    for (auto& sb_ls_pair : d_sb_integrator_ls_map) sb_ls_pair.first->endTimestepping(current_time, new_time);

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
        for (size_t l = 0; l < d_ls_vars.size(); ++l)
        {
            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
            const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];

            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            const int area_cur_idx = var_db->mapVariableAndContextToIndex(area_var, getCurrentContext());
            const int side_cur_idx = var_db->mapVariableAndContextToIndex(side_var, getCurrentContext());

            d_ls_vol_fcn_map[ls_var]->updateVolumeAreaSideLS(
                vol_cur_idx, vol_var, area_cur_idx, area_var, side_cur_idx, side_var, ls_cur_idx, ls_var, 0.0, false);
        }
        for (const auto& Q_var : d_Q_var)
        {
            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
            const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
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
SemiLagrangianAdvIntegrator::resetTimeDependentHierarchyDataSpecialized(const double new_time)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    // Copy level set info
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
                Pointer<NodeData<NDIM, double>> ls_cur_data = patch->getPatchData(ls_cur_idx);
                Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(ls_new_idx);
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

        const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];
        const int side_cur_idx = var_db->mapVariableAndContextToIndex(side_var, getCurrentContext());
        const int side_new_idx = var_db->mapVariableAndContextToIndex(side_var, getNewContext());
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
                Pointer<SideData<NDIM, double>> side_cur_data = patch->getPatchData(side_cur_idx);
                Pointer<SideData<NDIM, double>> side_new_data = patch->getPatchData(side_new_idx);
                side_cur_data->copy(*side_new_data);
            }
        }
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

        const Pointer<NodeVariable<NDIM, double>> ls_var = d_Q_ls_map[Q_var];
        TBOX_ASSERT(ls_var);
        const size_t ll = std::distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
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

    const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
    const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
    const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
    const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
    const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
    const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
    const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());

    {
        copy_face_to_side(u_s_new_idx, u_new_idx, d_hierarchy);
        d_hier_fc_data_ops->linearSum(u_scr_idx, 0.5, u_new_idx, 0.5, u_cur_idx, true);
        copy_face_to_side(u_s_half_idx, u_scr_idx, d_hierarchy);
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(3);
        HierarchyGhostCellInterpolation hier_ghost_cells;
        ghost_cell_comps[0] = ITC(ls_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        ghost_cell_comps[1] = ITC(u_s_new_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR");
        ghost_cell_comps[2] = ITC(u_s_half_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR");
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);
        // Integrate path
        integratePaths(d_path_idx, u_s_new_idx, u_s_half_idx, vol_new_idx, ls_new_idx, dt);
    }

    {
        // TODO: What kind of physical boundary conditions should we use for advection?
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(2);
        ghost_cell_comps[0] =
            ITC(d_Q_scratch_idx, Q_cur_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
        ghost_cell_comps[1] = ITC(ls_cur_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
        HierarchyGhostCellInterpolation hier_ghost_cells;
        hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
        hier_ghost_cells.fillData(current_time);

        // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
        evaluateMappingOnHierarchy(
            d_path_idx, d_Q_scratch_idx, vol_cur_idx, Q_new_idx, vol_new_idx, ls_cur_idx, /*order*/ 1);
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
                                             const int side_idx,
                                             Pointer<SideVariable<NDIM, double>> side_var,
                                             const double current_time,
                                             const double new_time)
{
    LS_TIMER_START(t_diffusion_step);
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    // We assume scratch context is already filled correctly.
    const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    const int Q_rhs_scratch_idx = var_db->mapVariableAndContextToIndex(d_Q_Q_rhs_map[Q_var], getScratchContext());
    const size_t l = distance(d_Q_var.begin(), std::find(d_Q_var.begin(), d_Q_var.end(), Q_var));

    const size_t ls_l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
    const Pointer<CellVariable<NDIM, double>>& vol_wgt_var = d_vol_wgt_vars[ls_l];
    const int vol_wgt_idx = var_db->mapVariableAndContextToIndex(vol_wgt_var, getCurrentContext());
    const int wgt_idx = d_hier_math_ops->getCellWeightPatchDescriptorIndex();
    d_hier_cc_data_ops->multiply(vol_wgt_idx, vol_idx, wgt_idx);

    Pointer<LSCutCellLaplaceOperator> rhs_oper = d_helmholtz_rhs_ops[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(rhs_oper);
#endif
    rhs_oper->setTimeStepType(d_dif_ts_type);
    rhs_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var, side_idx, side_var);
    rhs_oper->setSolutionTime(current_time);
    if (d_helmholtz_rhs_ops_need_init[l])
    {
        if (d_enable_logging)
            plog << d_object_name << ": "
                 << "Initializing Helmholtz RHS operator for variable number " << l << "\n";
        rhs_oper->initializeOperatorState(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
        d_helmholtz_rhs_ops_need_init[l] = false;
    }
    rhs_oper->apply(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);

    if (d_Q_F_map[Q_var])
    {
        Pointer<CellVariable<NDIM, double>> F_var = d_Q_F_map[Q_var];
        const int F_scratch_idx = var_db->mapVariableAndContextToIndex(F_var, getScratchContext());
        const int F_new_idx = var_db->mapVariableAndContextToIndex(F_var, getNewContext());
        TBOX_ASSERT(d_F_fcn[F_var]);
        d_F_fcn[F_var]->setDataOnPatchHierarchy(F_scratch_idx, F_var, d_hierarchy, 0.5 * (current_time + new_time));
        d_hier_cc_data_ops->axpy(Q_rhs_scratch_idx, 1.0, F_scratch_idx, Q_rhs_scratch_idx);
        d_hier_cc_data_ops->copyData(F_new_idx, F_scratch_idx);
    }

    Pointer<PETScKrylovPoissonSolver> Q_helmholtz_solver = d_helmholtz_solvers[l];
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_helmholtz_solver);
#endif
    Pointer<LSCutCellLaplaceOperator> solv_oper = Q_helmholtz_solver->getOperator();
#if !defined(NDEBUG)
    TBOX_ASSERT(solv_oper);
#endif
    solv_oper->setTimeStepType(d_dif_ts_type);
    solv_oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var, side_idx, side_var);
    solv_oper->setSolutionTime(new_time);
    if (d_helmholtz_solvers_need_init[l])
    {
        if (d_enable_logging)
            plog << d_object_name << ": "
                 << "Initializing Helmholtz solver for variable number " << l << "\n";
        Q_helmholtz_solver->initializeSolverState(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
        d_helmholtz_solvers_need_init[l] = false;
    }
    Q_helmholtz_solver->solveSystem(*d_sol_ls_vecs[l], *d_rhs_ls_vecs[l]);
    d_hier_cc_data_ops->copyData(Q_new_idx, Q_scr_idx);
    if (d_enable_logging)
    {
        plog << d_object_name << "::integrateHierarchy(): diffusion solve number of iterations = "
             << Q_helmholtz_solver->getNumIterations() << "\n";
        plog << d_object_name << "::integrateHierarchy(): diffusion solve residual norm        = "
             << Q_helmholtz_solver->getResidualNorm() << "\n";
    }

    // De-initialize solver states
    if (!d_helmholtz_rhs_ops_need_init[l])
    {
        rhs_oper->deallocateOperatorState();
        d_helmholtz_rhs_ops_need_init[l] = true;
    }
    if (!d_helmholtz_solvers_need_init[l])
    {
        Q_helmholtz_solver->deallocateSolverState();
        d_helmholtz_solvers_need_init[l] = true;
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
#if (NDIM == 2)
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
#endif
#if (NDIM == 3)
                integrate_paths_forward_(path_data->getPointer(),
                                         path_data->getGhostCellWidth().max(),
                                         u_new_data->getPointer(0),
                                         u_new_data->getPointer(1),
                                         u_new_data->getPointer(2),
                                         u_new_data->getGhostCellWidth().max(),
                                         dt,
                                         dx,
                                         idx_low(0),
                                         idx_low(1),
                                         idx_low(2),
                                         idx_up(0),
                                         idx_up(1),
                                         idx_up(2));
#endif
                break;
            case AdvectionTimeIntegrationMethod::MIDPOINT_RULE:
#if (NDIM == 2)
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
#endif
#if (NDIM == 3)
                integrate_paths_midpoint_(path_data->getPointer(),
                                          path_data->getGhostCellWidth().max(),
                                          u_new_data->getPointer(0),
                                          u_new_data->getPointer(1),
                                          u_new_data->getPointer(2),
                                          u_new_data->getGhostCellWidth().max(),
                                          u_half_data->getPointer(0),
                                          u_half_data->getPointer(1),
                                          u_half_data->getPointer(2),
                                          u_half_data->getGhostCellWidth().max(),
                                          dt,
                                          dx,
                                          idx_low(0),
                                          idx_low(1),
                                          idx_low(2),
                                          idx_up(0),
                                          idx_up(1),
                                          idx_up(2));
#endif
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
#if (NDIM == 3)
            // Computing cell centroids in Fortran can be difficult. We'll precompute them here, and pass them along
            CellData<NDIM, double> centroid_data(box, NDIM, 0);
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                LS_TIMER_START(t_find_cell_centroid);
                VectorNd centroid = find_cell_centroid(idx, *ls_data);
                LS_TIMER_STOP(t_find_cell_centroid);
                for (int d = 0; d < NDIM; ++d) centroid_data(idx, d) = centroid[d];
            }
#endif

            switch (d_adv_ts_type)
            {
            case AdvectionTimeIntegrationMethod::FORWARD_EULER:
#if (NDIM == 2)
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
#endif
#if (NDIM == 3)
                integrate_paths_ls_forward_(path_data->getPointer(),
                                            path_data->getGhostCellWidth().max(),
                                            u_new_data->getPointer(0),
                                            u_new_data->getPointer(1),
                                            u_new_data->getPointer(2),
                                            u_new_data->getGhostCellWidth().max(),
                                            centroid_data.getPointer(),
                                            centroid_data.getGhostCellWidth().max(),
                                            dt,
                                            dx,
                                            idx_low(0),
                                            idx_low(1),
                                            idx_low(2),
                                            idx_up(0),
                                            idx_up(1),
                                            idx_up(2));
#endif
                break;
            case AdvectionTimeIntegrationMethod::MIDPOINT_RULE:
#if (NDIM == 2)
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
#endif
#if (NDIM == 3)
                integrate_paths_ls_midpoint_(path_data->getPointer(),
                                             path_data->getGhostCellWidth().max(),
                                             u_new_data->getPointer(0),
                                             u_new_data->getPointer(1),
                                             u_new_data->getPointer(2),
                                             u_new_data->getGhostCellWidth().max(),
                                             u_half_data->getPointer(0),
                                             u_half_data->getPointer(1),
                                             u_half_data->getPointer(2),
                                             u_half_data->getGhostCellWidth().max(),
                                             centroid_data.getPointer(),
                                             centroid_data.getGhostCellWidth().max(),
                                             dt,
                                             dx,
                                             idx_low(0),
                                             idx_low(1),
                                             idx_low(2),
                                             idx_up(0),
                                             idx_up(1),
                                             idx_up(2));
#endif
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
                            d_use_rbfs ?
                                radialBasisFunctionReconstruction(
                                    x_loc, idx, *Q_cur_data, *vol_cur_data, *ls_data, patch) :
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
SemiLagrangianAdvIntegrator::radialBasisFunctionReconstruction(IBTK::VectorNd x_loc,
                                                               const CellIndex<NDIM>& idx,
                                                               const CellData<NDIM, double>& Q_data,
                                                               const CellData<NDIM, double>& vol_data,
                                                               const NodeData<NDIM, double>& ls_data,
                                                               const Pointer<Patch<NDIM>>& patch)
{
    LS_TIMER_START(t_rbf_reconstruct);
    Box<NDIM> box(idx, idx);
    box.grow(d_rbf_stencil_size);
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
            LS_TIMER_START(t_find_cell_centroid);
            VectorNd x_cent_c = find_cell_centroid(idx_c, ls_data);
            LS_TIMER_STOP(t_find_cell_centroid);
            Q_vals.push_back(Q_data(idx_c));
            X_vals.push_back(x_cent_c);
        }
    }
    const int m = Q_vals.size();
    MatrixXd A(MatrixXd::Zero(m, m));
    MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
    VectorXd U(VectorXd::Zero(m + NDIM + 1));
    for (size_t i = 0; i < Q_vals.size(); ++i)
    {
        for (size_t j = 0; j < Q_vals.size(); ++j)
        {
            const VectorNd X = X_vals[i] - X_vals[j];
            A(i, j) = rbf(X.norm());
        }
        B(i, 0) = 1.0;
        for (int d = 0; d < NDIM; ++d) B(i, d + 1) = X_vals[i](d);
        U(i) = Q_vals[i];
    }

    MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
    final_mat.block(0, 0, m, m) = A;
    final_mat.block(0, m, m, NDIM + 1) = B;
    final_mat.block(m, 0, NDIM + 1, m) = B.transpose();

    VectorXd x = final_mat.fullPivHouseholderQr().solve(U);
    double val = 0.0;
    VectorXd rbf_coefs = x.block(0, 0, m, 1);
    VectorXd poly_coefs = x.block(m, 0, NDIM + 1, 1);
    VectorXd poly_vec = VectorXd::Ones(NDIM + 1);
    for (int d = 0; d < NDIM; ++d) poly_vec(d + 1) = x_loc(d);
    for (size_t i = 0; i < X_vals.size(); ++i)
    {
        val += rbf_coefs[i] * rbf((X_vals[i] - x_loc).norm());
    }
    val += poly_coefs.dot(poly_vec);
    LS_TIMER_STOP(t_rbf_reconstruct);
    return val;
}

double
SemiLagrangianAdvIntegrator::leastSquaresReconstruction(IBTK::VectorNd x_loc,
                                                        const CellIndex<NDIM>& idx,
                                                        const CellData<NDIM, double>& Q_data,
                                                        const CellData<NDIM, double>& vol_data,
                                                        const NodeData<NDIM, double>& ls_data,
                                                        const Pointer<Patch<NDIM>>& patch)
{
#if (NDIM == 3)
    TBOX_ERROR("MLS reconstruction not implemented for 3 spatial dimensions. Use RBF reconstruction.\n");
#endif
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
            LS_TIMER_START(t_find_cell_centroid);
            VectorNd x_cent_c = find_cell_centroid(idx_c, ls_data);
            LS_TIMER_STOP(t_find_cell_centroid);
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
