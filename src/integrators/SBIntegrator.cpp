#include "ADS/SBIntegrator.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include "ibtk/IndexUtilities.h"

#include "libmesh/elem_cutter.h"
#include "libmesh/explicit_system.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/transient_system.h"

#include <boost/multi_array.hpp>

namespace
{
static Timer* t_integrateHierarchy = nullptr;
} // namespace

namespace ADS
{
SBIntegrator::SBIntegrator(std::string object_name,
                           const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager)
    : d_object_name(std::move(object_name)), d_sb_data_manager(sb_data_manager)
{
    IBTK_DO_ONCE(t_integrateHierarchy =
                     TimerManager::getManager()->getTimer("ADS::SBIntegrator::integrateHierarchy()"););
}

void
SBIntegrator::setLSData(const int ls_idx, const int vol_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_ls_idx = ls_idx;
    d_vol_idx = vol_idx;
    d_hierarchy = hierarchy;
    d_sb_data_manager->setLSData(ls_idx, vol_idx, hierarchy);
}

void
SBIntegrator::integrateHierarchy(Pointer<VariableContext> ctx, const double current_time, const double new_time)
{
    ADS_TIMER_START(t_integrateHierarchy);
    for (unsigned int part = 0; part < d_sb_data_manager->getNumParts(); ++part)
    {
        FESystemManager& fe_sys_manager = d_sb_data_manager->getFESystemManager(part);
        auto fe_hierarchy_mapping = std::make_unique<FEToHierarchyMapping>(d_object_name + "::FEToHierarchyMapping",
                                                                           &fe_sys_manager,
                                                                           nullptr,
                                                                           d_hierarchy->getNumberOfLevels(),
                                                                           2 /*ghost width*/);
        fe_hierarchy_mapping->setPatchHierarchy(d_hierarchy);
        fe_hierarchy_mapping->reinitElementMappings(2 /*ghost width*/);
        for (unsigned int l = 0; l < d_sb_data_manager->getFLNames(part).size(); ++l)
        {
            d_sb_data_manager->interpolateToBoundary(
                d_sb_data_manager->getFLNames(part)[l], ctx, current_time, part, fe_hierarchy_mapping.get());
        }
        for (const auto& sf_name : d_sb_data_manager->getSFNames(part))
        {
            const ReactionFcnCtx& rcn_fcn_ctx = d_sb_data_manager->getSFReactionFcnCtxPair(sf_name, part);
            const double dt = new_time - current_time;
            // Solve ODE on surface
            EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
            auto& sf_base_sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
            DofMap& sf_base_dof_map = sf_base_sys.get_dof_map();
            NumericVector<double>* sf_base_cur_vec = sf_base_sys.solution.get();

            // Get the NumericVector for all associated systems.
            // TODO: We should check that the dof_maps are all the same, otherwise we need to grab them.
            std::vector<std::string> sf_names, fl_names;
            d_sb_data_manager->getSFCouplingLists(sf_name, sf_names, fl_names, part);
            std::vector<NumericVector<double>*> sf_cur_vecs, sf_old_vecs, fl_vecs;
            std::vector<DofMap*> sf_dof_maps, fl_dof_maps;
            for (const auto& fl_name : fl_names)
            {
                System& fl_sys = eq_sys->get_system(fl_name);
                fl_dof_maps.push_back(&fl_sys.get_dof_map());
                fl_vecs.push_back(fl_sys.solution.get());
            }
            for (const auto& sf_name : sf_names)
            {
                auto& sf_sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
                sf_dof_maps.push_back(&sf_sys.get_dof_map());
#if LIBMESH_VERSION_LESS_THAN(1, 7, 0)
                sf_cur_vecs.push_back(sf_sys.old_local_solution.get());
                sf_old_vecs.push_back(sf_sys.older_local_solution.get());
#else
                sf_cur_vecs.push_back(sf_sys.old_local_solution);
                sf_old_vecs.push_back(sf_sys.older_local_solution);
#endif
            }

            // Assume we are on the finest level.
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
            const std::vector<std::vector<Node*>>& active_patch_node_map =
                fe_hierarchy_mapping->getActivePatchNodeMap();
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const int patch_num = patch->getPatchNumber();
                const std::vector<Node*>& active_patch_nodes = active_patch_node_map[patch_num];

                for (const auto& node : active_patch_nodes)
                {
                    // Integrate solution to new value. Use Forward Euler
                    // TODO: We should do a better time stepping scheme. The simplest would be AB2.
                    std::vector<dof_id_type> fl_dofs, sf_dofs, sf_base_dofs;
                    std::vector<double> fl_vals, sf_cur_vals, sf_old_vals;
                    std::vector<double> sf_base_cur_vals, sf_base_new_vals;
                    for (unsigned int l = 0; l < sf_names.size(); ++l)
                    {
                        IBTK::get_nodal_dof_indices(*sf_dof_maps[l], node, 0, sf_dofs);
                        sf_cur_vals.push_back((*sf_cur_vecs[l])(sf_dofs[0]));
                        sf_old_vals.push_back((*sf_old_vecs[l])(sf_dofs[0]));
                    }
                    for (unsigned int l = 0; l < fl_names.size(); ++l)
                    {
                        IBTK::get_nodal_dof_indices(*fl_dof_maps[l], node, 0, fl_dofs);
                        fl_vals.push_back((*fl_vecs[l])(fl_dofs[0]));
                    }
                    IBTK::get_nodal_dof_indices(sf_base_dof_map, node, 0, sf_base_dofs);
                    double sf_cur_val = (*sf_base_cur_vec)(sf_base_dofs[0]);
                    sf_cur_val +=
                        dt * rcn_fcn_ctx.first(sf_cur_val, fl_vals, sf_cur_vals, current_time, rcn_fcn_ctx.second);
                    sf_base_cur_vec->set(sf_base_dofs[0], sf_cur_val);
                }
            }
            sf_base_cur_vec->close();
        }
    }
    ADS_TIMER_STOP(t_integrateHierarchy);
}

void
SBIntegrator::beginTimestepping(const double /*current_time*/, const double /*new_time*/)
{
    // Overwrite old surface concentration
    for (unsigned int part = 0; part < d_sb_data_manager->getNumParts(); ++part)
    {
        const FESystemManager& fe_sys_manager = d_sb_data_manager->getFESystemManager(part);
        EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
        for (const auto& sf_name : d_sb_data_manager->getSFNames(part))
        {
            TransientExplicitSystem& sf_sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
            *sf_sys.older_local_solution = *sf_sys.old_local_solution;
            *sf_sys.old_local_solution = *sf_sys.current_local_solution;
            sf_sys.update();
        }
    }

    // Update Jacobian
    d_sb_data_manager->updateJacobian();
}

void
SBIntegrator::endTimestepping(const double /*current_time*/, const double /*new_time*/)
{
    for (unsigned int part = 0; part < d_sb_data_manager->getNumParts(); ++part)
    {
        const FESystemManager& fe_sys_manager = d_sb_data_manager->getFESystemManager(part);
        EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
        for (const auto& sf_name : d_sb_data_manager->getSFNames(part))
        {
            TransientExplicitSystem& sf_sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
            sf_sys.update();
        }
    }
}
} // namespace ADS
