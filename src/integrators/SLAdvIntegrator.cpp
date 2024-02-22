#include "ADS/LSCartGridFunction.h"
#include "ADS/LSFromLevelSet.h"
#include "ADS/LinearReconstructions.h"
#include "ADS/PointwiseFunction.h"
#include "ADS/RBFReconstructions.h"
#include "ADS/SBBoundaryConditions.h"
#include "ADS/SLAdvIntegrator.h"
#include "ADS/ZSplineReconstructions.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

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
    void integrate_paths_midpoint_half_(const double*,
                                        const int&,
                                        const double*,
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
    void integrate_paths_midpoint_half_(const double*,
                                        const int&,
                                        const double*,
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
#endif
}

namespace ADS
{
namespace
{
static Timer* t_advective_step;
static Timer* t_preprocess;
static Timer* t_postprocess;
static Timer* t_integrate_hierarchy;
static Timer* t_find_velocity;
static Timer* t_evaluate_mapping_ls;
static Timer* t_integrate_path_vol;
static Timer* t_integrate_path_ls;
static Timer* t_find_cell_centroid;

} // namespace
int SLAdvIntegrator::GHOST_CELL_WIDTH = 4;

SLAdvIntegrator::SLAdvIntegrator(const std::string& object_name, Pointer<Database> input_db, bool register_for_restart)
    : AdvDiffHierarchyIntegrator(object_name, input_db, register_for_restart),
      d_path_var(new CellVariable<NDIM, double>(d_object_name + "::PathVar", NDIM)),
      d_Q_big_scr_var(new CellVariable<NDIM, double>(d_object_name + "::ExtrapVar"))
{
    if (input_db)
    {
        d_min_ls_refine_factor = input_db->getDouble("min_ls_refine_factor");
        d_max_ls_refine_factor = input_db->getDouble("max_ls_refine_factor");
        d_least_squares_reconstruction_order =
            Reconstruct::string_to_enum<Reconstruct::LeastSquaresOrder>(input_db->getString("least_squares_order"));
        d_adv_ts_type = string_to_enum<AdvectionTimeIntegrationMethod>(input_db->getString("advection_ts_type"));
        d_use_rbfs = input_db->getBool("use_rbfs");
        d_rbf_stencil_size = input_db->getIntegerWithDefault("rbf_stencil_size", d_rbf_stencil_size);
        d_mls_stencil_size = input_db->getIntegerWithDefault("mls_stencil_size", d_mls_stencil_size);
        d_rbf_poly_order =
            Reconstruct::string_to_enum<Reconstruct::RBFPolyOrder>(input_db->getString("rbf_poly_order"));
        d_default_adv_reconstruct_type =
            string_to_enum<AdvReconstructType>(input_db->getStringWithDefault("default_adv_reconstruct_type", "RBF"));
        d_num_cycles = input_db->getIntegerWithDefault("num_cycles", d_num_cycles);
    }

    IBAMR_DO_ONCE(
        t_advective_step = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::advective_step");
        t_preprocess = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::preprocess");
        t_postprocess = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::postprocess");
        t_find_velocity = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::find_velocity");
        t_evaluate_mapping_ls = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::evaluate_mapping_ls");
        t_integrate_path_vol = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::integrage_path_vol");
        t_integrate_path_ls = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::integrate_path_ls");
        t_integrate_hierarchy = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::integrate_hierarchy");
        t_find_cell_centroid = TimerManager::getManager()->getTimer("ADS::SLAdvIntegrator::find_cell_centroid"));
}

void
SLAdvIntegrator::registerGeneralBoundaryMeshMapping(const std::shared_ptr<GeneralBoundaryMeshMapping>& mesh_mapping)
{
    d_mesh_mapping = mesh_mapping;
}

void
SLAdvIntegrator::registerTransportedQuantity(Pointer<CellVariable<NDIM, double>> Q_var, bool Q_output)
{
    AdvDiffHierarchyIntegrator::registerTransportedQuantity(Q_var, Q_output);
    setDefaultReconstructionOperator(Q_var);
}

void
SLAdvIntegrator::registerLevelSetVariable(Pointer<NodeVariable<NDIM, double>> ls_var)
{
    d_ls_vars.push_back(ls_var);
    d_vol_vars.push_back(new CellVariable<NDIM, double>(ls_var->getName() + "_VolVar"));
    d_ls_vol_fcn_map[ls_var] = nullptr;
    d_ls_use_ls_for_tagging[ls_var] = true;
}

void
SLAdvIntegrator::registerLevelSetVolFunction(Pointer<NodeVariable<NDIM, double>> ls_var,
                                             Pointer<LSFindCellVolume> ls_fcn)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_vol_fcn_map[ls_var] = ls_fcn;
}

void
SLAdvIntegrator::restrictToLevelSet(Pointer<CellVariable<NDIM, double>> Q_var,
                                    Pointer<NodeVariable<NDIM, double>> ls_var)
{
    TBOX_ASSERT(std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) != d_Q_var.end());
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_Q_ls_map[Q_var] = ls_var;
}

void
SLAdvIntegrator::useLevelSetForTagging(Pointer<NodeVariable<NDIM, double>> ls_var, const bool use_ls_for_tagging)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_use_ls_for_tagging[ls_var] = use_ls_for_tagging;
}

Pointer<CellVariable<NDIM, double>>
SLAdvIntegrator::getVolumeVariable(Pointer<NodeVariable<NDIM, double>> ls_var)
{
    const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
    return d_vol_vars[l];
}

void
SLAdvIntegrator::registerAdvectionReconstruction(Pointer<CellVariable<NDIM, double>> Q_var,
                                                 std::shared_ptr<AdvectiveReconstructionOperator> reconstruct_op)
{
    TBOX_ASSERT(std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) != d_Q_var.end());
    d_Q_adv_reconstruct_map[Q_var] = std::move(reconstruct_op);
}

void
SLAdvIntegrator::registerDivergenceReconstruction(Pointer<FaceVariable<NDIM, double>> u_var,
                                                  std::shared_ptr<AdvectiveReconstructionOperator> reconstruct_op)
{
    TBOX_ASSERT(std::find(d_u_var.begin(), d_u_var.end(), u_var) != d_u_var.end());
    d_u_div_adv_ops_map[u_var] = std::move(reconstruct_op);
}

void
SLAdvIntegrator::setDiffusionCoefficient(Pointer<CellVariable<NDIM, double>> /*Q_var*/, const double /*D*/)
{
    TBOX_ERROR("Can not register a diffusion coefficient with the SLAdvIntegrator.\n");
}

void
SLAdvIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                               Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    if (d_integrator_is_initialized) return;
    plog << d_object_name + ": initializing Hierarchy integrator.\n";
    d_hierarchy = hierarchy;
    d_gridding_alg = gridding_alg;
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();

    AdvDiffHierarchyIntegrator::registerVariables();
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        int ls_node_cur_idx = var_db->registerVariableAndContext(ls_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int ls_node_new_idx = var_db->registerVariableAndContext(ls_var, getNewContext(), GHOST_CELL_WIDTH);
        int vol_cur_idx = var_db->registerVariableAndContext(vol_var, getCurrentContext(), GHOST_CELL_WIDTH);
        int vol_new_idx = var_db->registerVariableAndContext(vol_var, getNewContext(), GHOST_CELL_WIDTH);

        d_current_data.setFlag(ls_node_cur_idx);
        d_current_data.setFlag(vol_cur_idx);

        if (d_registered_for_restart)
        {
            var_db->registerPatchDataForRestart(ls_node_cur_idx);
            var_db->registerPatchDataForRestart(vol_cur_idx);
        }
        d_new_data.setFlag(ls_node_new_idx);
        d_new_data.setFlag(vol_new_idx);

        const std::string& ls_name = ls_var->getName();
        if (d_visit_writer)
        {
            d_visit_writer->registerPlotQuantity(ls_name + "_Node", "SCALAR", ls_node_cur_idx);
            d_visit_writer->registerPlotQuantity(ls_name + "_volume", "SCALAR", vol_cur_idx);
        }
    }

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_path_idx = var_db->registerVariableAndContext(d_path_var, var_db->getContext(d_object_name + "::PathContext"));
    d_adv_data.setFlag(d_path_idx);

    // We need to register our own scratch variable since we need more ghost cells.
    d_Q_big_scr_idx =
        var_db->registerVariableAndContext(d_Q_big_scr_var, getScratchContext(), IntVector<NDIM>(GHOST_CELL_WIDTH));
    d_scratch_data.setFlag(d_Q_big_scr_idx);

    // If any velocity fields are incompressible, then we also need a half path index.
    if (std::any_of(d_u_is_div_free.begin(),
                    d_u_is_div_free.end(),
                    [](const std::pair<Pointer<FaceVariable<NDIM, double>>, bool>& a) -> bool { return !a.second; }))
    {
        d_half_path_var = new CellVariable<NDIM, double>(d_object_name + "::HalfPath", NDIM);
        d_half_path_idx =
            var_db->registerVariableAndContext(d_half_path_var, var_db->getContext(d_object_name + "::PathContext"));
        d_adv_data.setFlag(d_half_path_idx);
    }

    d_u_s_var = new SideVariable<NDIM, double>(d_object_name + "::USide");
    int u_new_idx, u_half_idx;
    registerVariable(u_new_idx, d_u_s_var, IntVector<NDIM>(1), getNewContext());
    registerVariable(u_half_idx, d_u_s_var, IntVector<NDIM>(1), getScratchContext());

    d_u_div_var = new CellVariable<NDIM, double>(d_object_name + "::DivU");
    int div_u_scr_idx, div_u_cur_idx, div_u_new_idx;
    registerVariable(div_u_cur_idx, div_u_new_idx, div_u_scr_idx, d_u_div_var, IntVector<NDIM>(2));

    AdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    auto hier_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    d_hier_fc_data_ops =
        hier_ops_manager->getOperationsDouble(new FaceVariable<NDIM, double>("fc_var"), d_hierarchy, true);

    d_integrator_is_initialized = true;
}

void
SLAdvIntegrator::applyGradientDetectorSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                  const int ln,
                                                  const double data_time,
                                                  const int tag_index,
                                                  const bool initial_time,
                                                  const bool uses_richardson_extrapolation_too)
{
    plog << "Applying gradient detector for level " << ln << " at time " << data_time << "\n";
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

void
SLAdvIntegrator::initializeLevelDataSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                const int ln,
                                                const double data_time,
                                                const bool can_be_refined,
                                                bool initial_time,
                                                Pointer<BasePatchLevel<NDIM>> old_level,
                                                bool allocate_data)
{
    plog << d_object_name + ": initializing level data\n";
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
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());

            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_cur_idx);
                ls_data->fillAll(-1.0);
            }
        }
    }
}

int
SLAdvIntegrator::getNumberOfCycles() const
{
    return d_num_cycles;
}

void
SLAdvIntegrator::preprocessIntegrateHierarchy(const double current_time, const double new_time, const int num_cycles)
{
    ADS_TIMER_START(t_preprocess)
    AdvDiffHierarchyIntegrator::preprocessIntegrateHierarchy(current_time, new_time, num_cycles);
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_scratch_data, current_time);
        level->allocatePatchData(d_new_data, new_time);
        level->allocatePatchData(d_adv_data, current_time);
    }

    // Update level set at current time. Update the boundary mesh if necessary.
    if (d_mesh_mapping)
    {
        // TODO: This was placed here for restarts. We should only call reinitElementMappings() when required.
        plog << d_object_name + ": Initializing fe mesh mappings\n";
        for (const auto& fe_mesh_mapping : d_mesh_mapping->getMeshPartitioners())
        {
            fe_mesh_mapping->setPatchHierarchy(d_hierarchy);
            fe_mesh_mapping->reinitElementMappings();
        }
    }

    if (d_mesh_mapping) d_mesh_mapping->updateBoundaryLocation(current_time, false);
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
        const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());

        const Pointer<LSFindCellVolume>& vol_fcn = d_ls_vol_fcn_map[ls_var];

        vol_fcn->updateVolumeAreaSideLS(vol_cur_idx,
                                        vol_var,
                                        IBTK::invalid_index,
                                        nullptr,
                                        IBTK::invalid_index,
                                        nullptr,
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

    // Prepare advection
    for (const auto& Q_var : d_Q_var)
    {
        Pointer<NodeVariable<NDIM, double>> ls_var = d_Q_ls_map[Q_var];
        unsigned int l = std::distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
        Pointer<CellVariable<NDIM, double>> vol_var = d_vol_vars[l];

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
        const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
        const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());

        d_Q_adv_reconstruct_map[Q_var]->setLSData(ls_cur_idx, vol_cur_idx, ls_new_idx, vol_new_idx);
        d_Q_adv_reconstruct_map[Q_var]->setBoundaryConditions(d_Q_bc_coef[Q_var][0]);
        d_Q_adv_reconstruct_map[Q_var]->allocateOperatorState(d_hierarchy, current_time, new_time);
    }

    executePreprocessIntegrateHierarchyCallbackFcns(current_time, new_time, num_cycles);
    ADS_TIMER_STOP(t_preprocess);
}

void
SLAdvIntegrator::integrateHierarchySpecialized(const double current_time, const double new_time, const int cycle_num)
{
    AdvDiffHierarchyIntegrator::integrateHierarchySpecialized(current_time, new_time, cycle_num);
    ADS_TIMER_START(t_integrate_hierarchy);

    // Intentionally blank. Semi-Lagrangian methods require everything at the END of the timestep.
    // TODO: We need to be more careful about this. In particular, we should double check that "current" and "new" data
    // is still present at the end of posprocessIntegrateHierarchy().

    ADS_TIMER_STOP(t_integrate_hierarchy);
}

void
SLAdvIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                               const double new_time,
                                               const bool skip_synchronize_new_state_data,
                                               const int num_cycles)
{
    ADS_TIMER_START(t_postprocess);
    // We need to set velocity at final time...
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    for (const auto& u_var : d_u_var)
    {
        const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
        if (d_u_fcn.at(u_var))
        {
            d_u_fcn.at(u_var)->setDataOnPatchHierarchy(
                u_new_idx, u_var, d_hierarchy, new_time, false, 0, d_hierarchy->getFinestLevelNumber());
        }
    }
    // Update Level sets
    for (size_t l = 0; l < d_ls_vars.size(); ++l)
    {
        const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
        const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];

        const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
        const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());

        const Pointer<LSFindCellVolume>& ls_fcn = d_ls_vol_fcn_map[ls_var];
        // Update boundary mesh if necessary.
        if (d_mesh_mapping) d_mesh_mapping->updateBoundaryLocation(new_time, true);
        ls_fcn->updateVolumeAreaSideLS(vol_new_idx,
                                       vol_var,
                                       IBTK::invalid_index,
                                       nullptr,
                                       IBTK::invalid_index,
                                       nullptr,
                                       ls_new_idx,
                                       ls_var,
                                       new_time,
                                       true);
    }

    executePreAdvectionCallbacks(current_time, new_time);

    // Now do advective update for each variable
    for (const auto& Q_var : d_Q_var)
    {
        // Now update advection.
        advectionUpdate(Q_var, current_time, new_time);

        plog << d_object_name + "::integrateHierarchy() finished advection update for variable: " << Q_var->getName()
             << "\n";
    }

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        d_hierarchy->getPatchLevel(ln)->deallocatePatchData(d_adv_data);

    executePostprocessIntegrateHierarchyCallbackFcns(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);

    AdvDiffHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);
    ADS_TIMER_STOP(t_postprocess);
}

void
SLAdvIntegrator::initializeCompositeHierarchyDataSpecialized(const double current_time, const bool initial_time)
{
    AdvDiffHierarchyIntegrator::initializeCompositeHierarchyDataSpecialized(current_time, initial_time);
    plog << d_object_name + ": initializing composite Hierarchy data\n";
    if (initial_time)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        // Set initial level set data. Update boundary mesh if necessary.
        if (d_mesh_mapping)
        {
            plog << d_object_name + ": Initializing fe mesh mappings\n";
            for (const auto& fe_mesh_mapping : d_mesh_mapping->getMeshPartitioners())
            {
                fe_mesh_mapping->setPatchHierarchy(d_hierarchy);
                fe_mesh_mapping->reinitElementMappings();
            }
        }
        for (size_t l = 0; l < d_ls_vars.size(); ++l)
        {
            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];

            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());

            plog << d_object_name << ": initializing level set for: " << ls_var->getName() << "\n";

            d_ls_vol_fcn_map[ls_var]->updateVolumeAreaSideLS(vol_cur_idx,
                                                             vol_var,
                                                             IBTK::invalid_index,
                                                             nullptr,
                                                             IBTK::invalid_index,
                                                             nullptr,
                                                             ls_cur_idx,
                                                             ls_var,
                                                             0.0,
                                                             false);
        }
        for (const auto& Q_var : d_Q_var)
        {
            plog << d_object_name << ": Initializing data for variable: " << Q_var->getName() << "\n";
            const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
            const size_t l = distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const int ls_node_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            if (initial_time)
            {
                if (!d_Q_init.at(Q_var))
                {
                    // Just fill in zeros.
                    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
                    {
                        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
                        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                        {
                            Pointer<Patch<NDIM>> patch = level->getPatch(p());
                            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
                            Q_data->fillAll(0.0);
                        }
                    }
                }
                else
                {
                    Pointer<LSCartGridFunction> Q_init = d_Q_init[Q_var];
                    if (Q_init) Q_init->setLSIndex(ls_node_cur_idx, vol_cur_idx);
                    d_Q_init.at(Q_var)->setDataOnPatchHierarchy(Q_idx, Q_var, d_hierarchy, 0.0);
                }
            }
        }
    }
    plog << d_object_name << ": Finished initializing composite data\n";
}

void
SLAdvIntegrator::regridHierarchyEndSpecialized()
{
    if (d_mesh_mapping)
    {
        for (const auto& mesh_partitioner : d_mesh_mapping->getMeshPartitioners())
            mesh_partitioner->reinitElementMappings();
    }
    AdvDiffHierarchyIntegrator::regridHierarchyEndSpecialized();
}

void
SLAdvIntegrator::resetTimeDependentHierarchyDataSpecialized(const double new_time)
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
    }

    AdvDiffHierarchyIntegrator::resetTimeDependentHierarchyDataSpecialized(new_time);
}

void
SLAdvIntegrator::resetHierarchyConfigurationSpecialized(const Pointer<BasePatchHierarchy<NDIM>> base_hierarchy,
                                                        const int coarsest_ln,
                                                        const int finest_ln)
{
    AdvDiffHierarchyIntegrator::resetHierarchyConfigurationSpecialized(base_hierarchy, coarsest_ln, finest_ln);
    d_hier_fc_data_ops->setPatchHierarchy(base_hierarchy);
    d_hier_fc_data_ops->resetLevels(0, finest_ln);
}

void
SLAdvIntegrator::registerPreAdvectionUpdateFcnCallback(PreAdvectionCallbackFcnPtr fcn, void* ctx)
{
    d_preadvection_callback_fcns.push_back(fcn);
    d_preadvection_callback_ctxs.push_back(ctx);
}

/////////////////////// PRIVATE ///////////////////////////////
void
SLAdvIntegrator::advectionUpdate(Pointer<CellVariable<NDIM, double>> Q_var,
                                 const double current_time,
                                 const double new_time)
{
    plog << d_object_name << ": advecting " << Q_var->getName() << "\n";
    ADS_TIMER_START(t_advective_step);
    const double half_time = 0.5 * (new_time + current_time);
    const double dt = new_time - current_time;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    int coarsest_ln = 0;
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
    const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
    const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
    const Pointer<FaceVariable<NDIM, double>>& u_var = d_Q_u_map[Q_var];
    if (!u_var)
    {
        // no advection. Copy data to new context
        d_hier_cc_data_ops->copyData(Q_new_idx, Q_cur_idx);
        return;
    }
    const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
    const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
    const int u_scr_idx = var_db->mapVariableAndContextToIndex(u_var, getScratchContext());
    const int u_s_new_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getNewContext());
    const int u_s_half_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());

    const Pointer<NodeVariable<NDIM, double>>& ls_var = d_Q_ls_map[Q_var];
    const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());

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

    // If the velocity field is not divergence free, we need to reconstruct J*q where J = exp(-dt*div(u)) evaluated at
    // half point departure points.
    // TODO: Currently, we compute J = exp(-dt*div(u)) at half point departures and separately compute q at full point
    // departures. We should investigate if we can compute these at the same departure ponts while maintaining second
    // order accuracy.
    if (!d_u_is_div_free[u_var])
    {
        // Need to integrate the half path back as well
        integratePaths(d_path_idx, d_half_path_idx, u_s_new_idx, u_s_half_idx, dt);
        const int div_u_scr_idx = var_db->mapVariableAndContextToIndex(d_u_div_var, getScratchContext());
        const int div_u_idx = var_db->mapVariableAndContextToIndex(d_u_div_var, getCurrentContext());
        if (d_u_div_adv_ops_map.count(u_var) > 0)
        {
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            d_u_div_adv_ops_map[u_var]->setLSData(ls_cur_idx, -1, ls_new_idx, -1);
            d_u_div_adv_ops_map[u_var]->allocateOperatorState(d_hierarchy, d_integrator_time, d_integrator_time);
            d_u_div_adv_ops_map[u_var]->applyReconstruction(u_s_half_idx, div_u_scr_idx, d_half_path_idx);
            d_u_div_adv_ops_map[u_var]->deallocateOperatorState();
            PointwiseFunctions::ScalarFcn exp_fcn = [dt](const double Q, const VectorNd&, double) -> double
            { return std::exp(-dt * Q); };
            ADS::PointwiseFunction<PointwiseFunctions::ScalarFcn> exp_hier_fcn("Exp", exp_fcn);
            exp_hier_fcn.setDataOnPatchHierarchy(div_u_scr_idx, d_u_div_var, d_hierarchy, half_time);
        }
        else
        {
            // Note ghost cells have already been filled.
            d_hier_math_ops->div(div_u_idx,
                                 d_u_div_var,
                                 1.0,
                                 u_s_half_idx,
                                 d_u_s_var,
                                 nullptr /*hier_ghost_fill*/,
                                 half_time,
                                 true /*cf_bdry_synch*/);
            // Now compute exp of div_u_idx
            PointwiseFunctions::ScalarFcn exp_fcn = [dt](const double Q, const VectorNd&, double) -> double
            { return std::exp(-dt * Q); };
            ADS::PointwiseFunction<PointwiseFunctions::ScalarFcn> exp_hier_fcn("Exp", exp_fcn);
            exp_hier_fcn.setDataOnPatchHierarchy(div_u_idx, d_u_div_var, d_hierarchy, half_time);
            d_Q_adv_reconstruct_map[Q_var]->applyReconstruction(div_u_idx, div_u_scr_idx, d_half_path_idx);
        }

        // Now interpolate the data to \XX^\star
        d_Q_adv_reconstruct_map[Q_var]->applyReconstruction(Q_cur_idx, Q_scr_idx, d_path_idx);
        d_hier_cc_data_ops->multiply(Q_new_idx, div_u_scr_idx, Q_scr_idx);
    }
    else
    {
        integratePaths(d_path_idx, IBTK::invalid_index, u_s_new_idx, u_s_half_idx, dt);
        // We have xstar at each grid point. We need to evaluate our function at \XX^\star to update for next iteration
        d_Q_adv_reconstruct_map[Q_var]->applyReconstruction(Q_cur_idx, Q_new_idx, d_path_idx);
    }
    ADS_TIMER_STOP(t_advective_step);
}

void
SLAdvIntegrator::integratePaths(const int path_idx,
                                const int half_path_idx,
                                const int u_new_idx,
                                const int u_half_idx,
                                const double dt)
{
    ADS_TIMER_START(t_integrate_path_vol);
    // Integrate path to find \xx^{n+1}
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            const hier::Index<NDIM>& idx_low = box.lower();
            const hier::Index<NDIM>& idx_up = box.upper();

            Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> path_half_data =
                half_path_idx == IBTK::invalid_index ? nullptr : patch->getPatchData(half_path_idx);
            Pointer<SideData<NDIM, double>> u_new_data = patch->getPatchData(u_new_idx);
            Pointer<SideData<NDIM, double>> u_half_data = patch->getPatchData(u_half_idx);
            TBOX_ASSERT(u_new_data && u_half_data);
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            if (path_half_data)
            {
                switch (d_adv_ts_type)
                {
                case AdvectionTimeIntegrationMethod::MIDPOINT_RULE:
#if (NDIM == 2)
                    integrate_paths_midpoint_half_(path_data->getPointer(),
                                                   path_data->getGhostCellWidth().max(),
                                                   path_half_data->getPointer(),
                                                   path_half_data->getGhostCellWidth().max(),
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
                    integrate_paths_midpoint_half_(path_data->getPointer(),
                                                   path_data->getGhostCellWidth().max(),
                                                   path_half_data->getPointer(),
                                                   path_half_data->getGhostCellWidth().max(),
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
                    TBOX_ERROR("Invalid integration path "
                               << enum_to_string(d_adv_ts_type)
                               << ".\n Valid type for compressible flow is MIDPOINT_RULE\n");
                }
            }
            else
            {
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
                    TBOX_ERROR("Invalid integration path "
                               << enum_to_string(d_adv_ts_type)
                               << ".\n Valid types for incompressible flow are MIDPOINT_RULE and FORWARD_EULER\n");
                }
            }
        }
    }
    ADS_TIMER_STOP(t_integrate_path_vol);
}

void
SLAdvIntegrator::setDefaultReconstructionOperator(Pointer<CellVariable<NDIM, double>> Q_var)
{
    if (!d_Q_adv_reconstruct_map[Q_var])
    {
        switch (d_default_adv_reconstruct_type)
        {
        case AdvReconstructType::ZSPLINES:
            d_Q_adv_reconstruct_map[Q_var] =
                std::make_shared<ZSplineReconstructions>(Q_var->getName() + "::DefaultReconstruct", 2);
            break;
        case AdvReconstructType::RBF:
            d_Q_adv_reconstruct_map[Q_var] =
                std::make_shared<RBFReconstructions>(Q_var->getName() + "::DefaultReconstruct",
                                                     d_rbf_poly_order,
                                                     d_rbf_stencil_size,
                                                     false /*use_cut_cells*/);
            break;
        case AdvReconstructType::LINEAR:
            d_Q_adv_reconstruct_map[Q_var] =
                std::make_shared<LinearReconstructions>(Q_var->getName() + "::DefaultReconstruct");
            break;
        default:
            TBOX_ERROR("Unknown adv reconstruction type " << enum_to_string(d_default_adv_reconstruct_type) << "\n");
            break;
        }
    }
}

void
SLAdvIntegrator::executePreAdvectionCallbacks(double current_time, double new_time)
{
    for (size_t l = 0; l < d_preadvection_callback_ctxs.size(); ++l)
    {
        d_preadvection_callback_fcns[l](current_time, new_time, d_preadvection_callback_ctxs[l]);
    }
}
} // namespace ADS
