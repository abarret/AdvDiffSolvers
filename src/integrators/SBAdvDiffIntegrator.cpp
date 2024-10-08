#include "ibamr/config.h"

#include "ADS/SBAdvDiffIntegrator.h"
#include "ADS/SBBoundaryConditions.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include "ibamr/AdvDiffCUIConvectiveOperator.h"
#include "ibamr/AdvDiffPPMConvectiveOperator.h"
#include "ibamr/AdvDiffWavePropConvectiveOperator.h"

#include "HierarchyDataOpsManager.h"
#include "SAMRAIVectorReal.h"
#include <tbox/Timer.h>
#include <tbox/TimerManager.h>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ADS
{
namespace
{
static Timer* t_advective_step;
static Timer* t_diffusion_step;
static Timer* t_preprocess;
static Timer* t_postprocess;
static Timer* t_integrate_hierarchy;
static Timer* t_find_velocity;
static Timer* t_evaluate_mapping_ls;
static Timer* t_integrate_path_vol;
static Timer* t_integrate_path_ls;
static Timer* t_find_cell_centroid;

} // namespace

void
callback_fcn(double current_time, double new_time, int cycle_num, void* ctx)
{
    static_cast<SBAdvDiffIntegrator*>(ctx)->integrateHierarchy(current_time, new_time, 500);
}

SBAdvDiffIntegrator::SBAdvDiffIntegrator(const std::string& object_name,
                                         Pointer<Database> input_db,
                                         Pointer<IBHierarchyIntegrator> ib_integrator,
                                         bool register_for_restart)
    : LSAdvDiffIntegrator(object_name, input_db, register_for_restart)
{
    if (ib_integrator)
    {
        d_used_with_ib = true;
        ib_integrator->registerIntegrateHierarchyCallback(callback_fcn, static_cast<void*>(this));
    }

    IBAMR_DO_ONCE(
        t_advective_step = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::advective_step");
        t_diffusion_step = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::diffusive_step");
        t_preprocess = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::preprocess");
        t_postprocess = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::postprocess");
        t_find_velocity = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::find_velocity");
        t_evaluate_mapping_ls = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::evaluate_mapping_ls");
        t_integrate_path_vol = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::integrage_path_vol");
        t_integrate_path_ls = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::integrate_path_ls");
        t_integrate_hierarchy = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::integrate_hierarchy");
        t_find_cell_centroid = TimerManager::getManager()->getTimer("ADS::SBAdvDiffIntegrator::find_cell_centroid"));
}

void
SBAdvDiffIntegrator::registerLevelSetSBDataManager(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                   std::shared_ptr<SBSurfaceFluidCouplingManager> sb_data_manager)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_sb_data_manager_map[ls_var] = sb_data_manager;
}

void
SBAdvDiffIntegrator::registerLevelSetCutCellMapping(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                    std::shared_ptr<CutCellMeshMapping> cut_cell_mesh_mapping)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_ls_cut_cell_mapping_map[ls_var] = cut_cell_mesh_mapping;
}

void
SBAdvDiffIntegrator::registerSBIntegrator(Pointer<SBIntegrator> sb_integrator,
                                          Pointer<NodeVariable<NDIM, double>> ls_var)
{
    TBOX_ASSERT(std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) != d_ls_vars.end());
    d_sb_integrator_ls_map[sb_integrator] = ls_var;
}

int
SBAdvDiffIntegrator::getNumberOfCycles() const
{
    return 2;
}

void
SBAdvDiffIntegrator::integrateHierarchySpecialized(const double current_time,
                                                   const double new_time,
                                                   const int cycle_num)
{
    if (!(d_used_with_ib && cycle_num == 500))
        AdvDiffHierarchyIntegrator::integrateHierarchySpecialized(current_time, new_time, cycle_num);
    ADS_TIMER_START(t_integrate_hierarchy);
    const double half_time = current_time + 0.5 * (new_time - current_time);
    auto var_db = VariableDatabase<NDIM>::getDatabase();

    if (cycle_num == 0)
    {
        for (auto& sb_ls_pair : d_sb_integrator_ls_map)
        {
            plog << d_object_name << ": Integrating surface odes\n";
            const Pointer<NodeVariable<NDIM, double>>& ls_var = sb_ls_pair.second;
            const unsigned int l =
                std::distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            auto var_db = VariableDatabase<NDIM>::getDatabase();
            const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int vol_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            Pointer<SBIntegrator> sb_integrator = sb_ls_pair.first;
            sb_integrator->setLSData(ls_idx, vol_idx, d_hierarchy);
            sb_integrator->beginTimestepping(current_time, d_use_strang_splitting ? half_time : new_time);
            sb_integrator->integrateHierarchy(
                getCurrentContext(), current_time, d_use_strang_splitting ? half_time : new_time);
            sb_integrator->endTimestepping(current_time, d_use_strang_splitting ? half_time : new_time);
        }
        plog << d_object_name << ": Finished integrating surface odes\n";

        for (size_t l = 0; l < d_ls_vars.size(); ++l)
        {
            Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
            const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
            const int vol_cur_idx = var_db->mapVariableAndContextToIndex(vol_var, getCurrentContext());
            d_reconstruct_from_centroids_ls_map[ls_var]->clearCache();
            d_reconstruct_to_centroids_ls_map[ls_var]->clearCache();
            d_reconstruct_from_centroids_ls_map[ls_var]->setLSData(ls_cur_idx, vol_cur_idx);
            d_reconstruct_to_centroids_ls_map[ls_var]->setLSData(ls_cur_idx, vol_cur_idx);
        }

        for (const auto& Q_var : d_Q_var)
        {
            const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());

            // Should we be skipping this solve?
            if (!d_Q_using_diffusion_solve.at(Q_var))
            {
                const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
                d_hier_cc_data_ops->copyData(Q_new_idx, Q_cur_idx);
                continue;
            }

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
            hier_ghost_cell.initializeOperatorState(
                ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
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

            plog << d_object_name + "::integrateHierarchy() finished diffusion update for variable: "
                 << Q_var->getName() << "\n";

            // TODO: Should we synchronize hierarchy?
        }
        return;
    }

    // Note we need to do advection as the LAST step. If we are using an IB solver, we need to wait until the
    // integrateHierarchyCallback is called.
    if ((d_used_with_ib && cycle_num == 500) || (!d_used_with_ib && cycle_num == 1))
    {
        // We need to set velocity at final time...
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
            const Pointer<CellVariable<NDIM, double>>& area_var = d_area_vars[l];
            const Pointer<SideVariable<NDIM, double>>& side_var = d_side_vars[l];

            const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
            const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
            const int area_new_idx = var_db->mapVariableAndContextToIndex(area_var, getNewContext());
            const int side_new_idx = var_db->mapVariableAndContextToIndex(side_var, getNewContext());

            const Pointer<LSFindCellVolume>& ls_fcn = d_ls_vol_fcn_map[ls_var];
            if (d_ls_u_map[ls_var].getPointer() != nullptr)
            {
                plog << "Advecting level set: " << ls_var->getName() << "\n";
                // We need to advect the level set.
                const Pointer<CellVariable<NDIM, double>>& ls_cell_var = d_ls_ls_cell_map[ls_var];
                const int ls_cell_cur_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getCurrentContext());
                const int ls_cell_scr_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getScratchContext());
                const int ls_cell_new_idx = var_db->mapVariableAndContextToIndex(ls_cell_var, getNewContext());

                const int u_new_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_var], getNewContext());
                const int u_scr_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_var], getScratchContext());
                const int u_cur_idx = var_db->mapVariableAndContextToIndex(d_ls_u_map[ls_var], getCurrentContext());

                const int u_s_half_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getScratchContext());
                const int u_s_new_idx = var_db->mapVariableAndContextToIndex(d_u_s_var, getNewContext());

                // fill half time data for u_s
                copy_face_to_side(u_s_new_idx, u_new_idx, d_hierarchy);
                d_hier_fc_data_ops->linearSum(u_scr_idx, 0.5, u_new_idx, 0.5, u_cur_idx, true);
                copy_face_to_side(u_s_half_idx, u_scr_idx, d_hierarchy);

                using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
                std::vector<ITC> ghost_cell_comps(3);
                ghost_cell_comps[0] = ITC(
                    u_s_new_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr);
                ghost_cell_comps[1] = ITC(u_s_half_idx,
                                          "CONSERVATIVE_LINEAR_REFINE",
                                          false,
                                          "CONSERVATIVE_COARSEN",
                                          "LINEAR",
                                          false,
                                          nullptr);
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

                // Interpolate cell centered data back to cell nodes
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
                    ls_new_idx, ls_var, false, ls_cell_scr_idx, ls_cell_var, hier_ghost_cell, new_time);
                hier_ghost_cell->deallocateOperatorState();
                ghost_cell_comps[0] = ITC(ls_new_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
                hier_ghost_cell->initializeOperatorState(
                    ghost_cell_comps, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
                hier_ghost_cell->fillData(current_time);

                // Now set ls_fcn so that it doesn't update the level set
                ls_fcn->setLS(false);
            }

            if (d_mesh_mapping) d_mesh_mapping->updateBoundaryLocation(new_time, true);
            ls_fcn->updateVolumeAreaSideLS(vol_new_idx,
                                           vol_var,
                                           area_new_idx,
                                           area_var,
                                           side_new_idx,
                                           side_var,
                                           ls_new_idx,
                                           ls_var,
                                           new_time,
                                           true);
            // Update Jacobian if applicable
            for (const auto& ls_sb_data_manager_pair : d_ls_sb_data_manager_map)
            {
                ls_sb_data_manager_pair.second->updateJacobian();
            }
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

            plog << d_object_name + "::integrateHierarchy() finished advection update for variable: "
                 << Q_var->getName() << "\n";
        }

        if (d_use_strang_splitting)
        {
            for (const auto& ls_cut_cell_mapping_pair : d_ls_cut_cell_mapping_map)
            {
                ls_cut_cell_mapping_pair.second->generateCutCellMappings(
                    unique_ptr_vec_to_raw_ptr_vec(d_fe_hierarchy_mappings));
            }
            for (auto& sb_ls_pair : d_sb_integrator_ls_map)
            {
                const Pointer<NodeVariable<NDIM, double>>& ls_var = sb_ls_pair.second;
                const unsigned int l =
                    std::distance(d_ls_vars.begin(), std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var));
                const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
                auto var_db = VariableDatabase<NDIM>::getDatabase();
                const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
                const int vol_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
                Pointer<SBIntegrator> sb_integrator = sb_ls_pair.first;
                sb_integrator->setLSData(ls_idx, vol_idx, d_hierarchy);
                sb_integrator->beginTimestepping(half_time, new_time);
                sb_integrator->integrateHierarchy(getNewContext(), half_time, new_time);
                sb_integrator->endTimestepping(half_time, new_time);
            }

            for (size_t l = 0; l < d_ls_vars.size(); ++l)
            {
                Pointer<NodeVariable<NDIM, double>>& ls_var = d_ls_vars[l];
                const Pointer<CellVariable<NDIM, double>>& vol_var = d_vol_vars[l];
                const int ls_new_idx = var_db->mapVariableAndContextToIndex(ls_var, getNewContext());
                const int vol_new_idx = var_db->mapVariableAndContextToIndex(vol_var, getNewContext());
                d_reconstruct_from_centroids_ls_map[ls_var]->clearCache();
                d_reconstruct_to_centroids_ls_map[ls_var]->clearCache();
                d_reconstruct_from_centroids_ls_map[ls_var]->setLSData(ls_new_idx, vol_new_idx);
                d_reconstruct_to_centroids_ls_map[ls_var]->setLSData(ls_new_idx, vol_new_idx);
            }
            for (const auto& Q_var : d_Q_var)
            {
                const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
                const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
                const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());

                // Should we be skipping this solve? Note we don't copy data here because Q_new already has the correct
                // data.
                if (!d_Q_using_diffusion_solve.at(Q_var)) continue;

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
    }
    ADS_TIMER_STOP(t_integrate_hierarchy);
}

} // namespace ADS
