#include "ADS/SBBoundaryConditions.h"
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
static Timer* t_applyBoundaryCondition = nullptr;
static Timer* t_allocateOperatorState = nullptr;
static Timer* t_deallocateOperatorState = nullptr;
} // namespace

namespace ADS
{
SBBoundaryConditions::SBBoundaryConditions(const std::string& object_name,
                                           const std::string& fl_name,
                                           const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                                           const Pointer<CutCellMeshMapping>& cut_cell_mesh_mapping)
    : LSCutCellBoundaryConditions(object_name),
      d_sb_data_manager(sb_data_manager),
      d_cut_cell_mapping(cut_cell_mesh_mapping),
      d_fl_name(fl_name)
{
    IBTK_DO_ONCE(t_applyBoundaryCondition =
                     TimerManager::getManager()->getTimer("ADS::SBBoundaryConditions::applyBoundaryCondition()");
                 t_allocateOperatorState =
                     TimerManager::getManager()->getTimer("ADS::SBBoundaryConditions::allocateOperatorState()");
                 t_deallocateOperatorState =
                     TimerManager::getManager()->getTimer("ADS::SBBoundaryConditions::deallocateOperatorState()"););
}

void
SBBoundaryConditions::setFluidContext(Pointer<VariableContext> ctx)
{
    d_ctx = ctx;
}

void
SBBoundaryConditions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy, double time)
{
    ADS_TIMER_START(t_allocateOperatorState);
    LSCutCellBoundaryConditions::allocateOperatorState(hierarchy, time);

    TBOX_ASSERT(d_ctx);

    // Interpolate to boundary
    std::vector<std::string> fl_names, sf_names;
    for (unsigned int part = 0; part < d_sb_data_manager->getNumParts(); ++part)
    {
        d_sb_data_manager->getFLCouplingLists(d_fl_name, sf_names, fl_names, part);
        for (const auto& fl_name : fl_names)
        {
            d_sb_data_manager->interpolateToBoundary(fl_name, d_ctx, time, part);
        }
    }

    d_cut_cell_mapping->initializeObjectState(hierarchy);
    d_cut_cell_mapping->generateCutCellMappings();
    ADS_TIMER_STOP(t_allocateOperatorState);
}

void
SBBoundaryConditions::deallocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy, double time)
{
    ADS_TIMER_START(t_deallocateOperatorState);
    LSCutCellBoundaryConditions::deallocateOperatorState(hierarchy, time);
    d_cut_cell_mapping->deinitializeObjectState();
    ADS_TIMER_STOP(t_deallocateOperatorState);
}

void
SBBoundaryConditions::applyBoundaryCondition(Pointer<CellVariable<NDIM, double>> Q_var,
                                             const int Q_idx,
                                             Pointer<CellVariable<NDIM, double>> R_var,
                                             const int R_idx,
                                             Pointer<PatchHierarchy<NDIM>> hierarchy,
                                             const double time)
{
    ADS_TIMER_START(t_applyBoundaryCondition);
    TBOX_ASSERT(d_ls_var && d_vol_var && d_area_var);
    TBOX_ASSERT(d_ls_idx > 0 && d_vol_idx > 0 && d_area_idx > 0);

    for (unsigned int part = 0; part < d_sb_data_manager->getNumParts(); ++part)
    {
        const std::string& sys_name = d_sb_data_manager->interpolateToBoundary(d_fl_name, Q_idx, time, part);
        TBOX_ASSERT(d_fl_name == sys_name);

        const double sgn = d_D / std::abs(d_D);
        double pre_fac = sgn * (d_ts_type == ADS::DiffusionTimeIntegrationMethod::TRAPEZOIDAL_RULE ? 0.5 : 1.0);
        if (d_D == 0.0) pre_fac = 0.0;

        FEToHierarchyMapping& fe_hierarchy_mapping = d_sb_data_manager->getFEToHierarchyMapping(part);
        EquationSystems* eq_sys = fe_hierarchy_mapping.getFESystemManager().getEquationSystems();

        System& X_system = eq_sys->get_system(fe_hierarchy_mapping.getCoordsSystemName());
        DofMap& X_dof_map = X_system.get_dof_map();
        FEType X_fe_type = X_dof_map.variable_type(0);
        NumericVector<double>* X_vec = X_system.solution.get();
        auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
        TBOX_ASSERT(X_petsc_vec != nullptr);
        const double* const X_local_soln = X_petsc_vec->get_array_read();
        FEDataManager::SystemDofMapCache& X_dof_map_cache =
            *fe_hierarchy_mapping.getFESystemManager().getDofMapCache(fe_hierarchy_mapping.getCoordsSystemName());

        System& J_sys = eq_sys->get_system(d_sb_data_manager->getJacobianName());
        DofMap& J_dof_map = J_sys.get_dof_map();
        NumericVector<double>* J_vec = J_sys.solution.get();
        auto J_petsc_vec = dynamic_cast<PetscVector<double>*>(J_vec);
        TBOX_ASSERT(J_petsc_vec != nullptr);
        const double* const J_local_soln = J_petsc_vec->get_array_read();

        std::vector<std::string> fl_names, sf_names;
        d_sb_data_manager->getFLCouplingLists(sys_name, sf_names, fl_names, part);
        std::vector<NumericVector<double>*> fl_vecs, sf_vecs;
        std::vector<DofMap*> fl_dof_maps, sf_dof_maps;
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
            sf_vecs.push_back(sf_sys.old_local_solution.get());
#else
            sf_vecs.push_back(sf_sys.old_local_solution);
#endif
        }

        // Get base system
        System& Q_sys = eq_sys->get_system(sys_name);
        DofMap& Q_dof_map = Q_sys.get_dof_map();
        FEType Q_fe_type = Q_dof_map.variable_type(0);
        NumericVector<double>* Q_vec = Q_sys.solution.get();
        TBOX_ASSERT(Q_fe_type == X_fe_type);

        std::unique_ptr<FEBase> fe = FEBase::build(eq_sys->get_mesh().mesh_dimension(), X_fe_type);
        std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, eq_sys->get_mesh().mesh_dimension(), THIRD);
        fe->attach_quadrature_rule(qrule.get());
        const std::vector<std::vector<double>>& phi = fe->get_phi();
        const std::vector<double>& JxW = fe->get_JxW();

        // Only changes are needed where the structure lives
        const int level_num = fe_hierarchy_mapping.getFinestPatchLevelNumber();
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(level_num);
        const Pointer<CartesianGridGeometry<NDIM>> grid_geom = level->getGridGeometry();
        VectorValue<double> n;
        IBTK::Point x_min, x_max;
        const BdryConds& bdry_reac_fcns = d_sb_data_manager->getFLBdryConditionFcns(sys_name, part);
        ReactionFcn a_fcn = std::get<0>(bdry_reac_fcns);
        ReactionFcn g_fcn = std::get<1>(bdry_reac_fcns);
        void* fcn_ctx = std::get<2>(bdry_reac_fcns);

        const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_elem_map =
            d_cut_cell_mapping->getIdxCutCellElemsMap(level_num);
        unsigned int local_patch_num = 0;

        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
            for (const auto& idx_cut_cell_elem_pair_vec : idx_cut_cell_elem_map[local_patch_num])
            {
                const CellIndex<NDIM>& idx = idx_cut_cell_elem_pair_vec.first.d_idx;
                Pointer<CellData<NDIM, double>> R_data = patch->getPatchData(R_idx);
                Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
                Pointer<CellData<NDIM, double>> area_data = patch->getPatchData(d_area_idx);
                Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);

                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const dx = pgeom->getDx();
                for (const auto& cut_cell_elem : idx_cut_cell_elem_pair_vec.second)
                {
                    const Elem* const old_elem = cut_cell_elem.d_parent_elem;
                    const std::unique_ptr<Elem>& new_elem = cut_cell_elem.d_elem;
                    const std::vector<std::pair<libMesh::Point, int>>& intersection_side_vec =
                        cut_cell_elem.d_intersection_side_vec;
                    std::vector<libMesh::Point> intersections(intersection_side_vec.size());
                    for (unsigned int k = 0; k < intersection_side_vec.size(); ++k)
                        intersections[k] = intersection_side_vec[k].first;

                    std::vector<dof_id_type> fl_dofs, sf_dofs, Q_dofs, J_dofs;
                    boost::multi_array<double, 2> x_node;
                    boost::multi_array<double, 1> Q_node, J_node;
                    std::vector<boost::multi_array<double, 1>> fl_node(fl_names.size()), sf_node(sf_names.size());
                    std::vector<double> sf_vals(sf_names.size()), fl_vals(fl_names.size());

                    Q_dof_map.dof_indices(old_elem, Q_dofs);
                    IBTK::get_values_for_interpolation(Q_node, *Q_vec, Q_dofs);

                    const auto& X_dof_indices = X_dof_map_cache.dof_indices(old_elem);
                    IBTK::get_values_for_interpolation(x_node, *X_petsc_vec, X_local_soln, X_dof_indices);

                    J_dof_map.dof_indices(old_elem, J_dofs);
                    IBTK::get_values_for_interpolation(J_node, *J_petsc_vec, J_local_soln, J_dofs);
                    for (unsigned int l = 0; l < fl_names.size(); ++l)
                    {
                        fl_dof_maps[l]->dof_indices(old_elem, fl_dofs);
                        IBTK::get_values_for_interpolation(fl_node[l], *fl_vecs[l], fl_dofs);
                    }
                    for (unsigned int l = 0; l < sf_names.size(); ++l)
                    {
                        sf_dof_maps[l]->dof_indices(old_elem, sf_dofs);
                        IBTK::get_values_for_interpolation(sf_node[l], *sf_vecs[l], sf_dofs);
                    }
                    // We need to interpolate our solution to the new element's nodes
                    std::array<double, 2> Q_soln_on_new_elem, J_soln_on_new_elem;
                    std::vector<std::array<double, 2>> sf_soln_on_new_elem(sf_names.size()),
                        fl_soln_on_new_elem(fl_names.size());
                    fe->reinit(old_elem, &intersections);
                    for (unsigned int l = 0; l < 2; ++l)
                    {
                        Q_soln_on_new_elem[l] = IBTK::interpolate(l, Q_node, phi);
                        for (unsigned int k = 0; k < sf_names.size(); ++k)
                            sf_soln_on_new_elem[k][l] = IBTK::interpolate(l, sf_node[k], phi);
                        for (unsigned int k = 0; k < fl_names.size(); ++k)
                            fl_soln_on_new_elem[k][l] = IBTK::interpolate(l, fl_node[k], phi);
                        J_soln_on_new_elem[l] = IBTK::interpolate(l, J_node, phi);
                    }
                    // Then we need to integrate
                    fe->reinit(new_elem.get());
                    double a = 0.0, g = 0.0;
                    double area = 0.0;
                    for (unsigned int qp = 0; qp < JxW.size(); ++qp)
                    {
                        double Q_val = 0.0, J_val = 0.0;
                        std::fill(sf_vals.begin(), sf_vals.end(), 0.0);
                        std::fill(fl_vals.begin(), fl_vals.end(), 0.0);
                        for (int n = 0; n < 2; ++n)
                        {
                            Q_val += Q_soln_on_new_elem[n] * phi[n][qp];
                            J_val += J_soln_on_new_elem[n] * phi[n][qp];
                            for (unsigned int l = 0; l < fl_names.size(); ++l)
                                fl_vals[l] += fl_soln_on_new_elem[l][n] * phi[n][qp];
                            for (unsigned int l = 0; l < sf_names.size(); ++l)
                                sf_vals[l] += sf_soln_on_new_elem[l][n] * phi[n][qp];
                        }
                        a += a_fcn(Q_val, fl_vals, sf_vals, time, fcn_ctx) * J_val * JxW[qp];
                        area += JxW[qp];
                        if (!d_homogeneous_bdry) g += g_fcn(Q_val, fl_vals, sf_vals, time, fcn_ctx) * J_val * JxW[qp];
                    }

                    double cell_volume = dx[0] * dx[1] * (*vol_data)(idx);
                    if (cell_volume <= 0.0)
                    {
                        plog << "Found intersection with zero cell volume.\n";
                        plog << "On index: " << idx << "\n";
                        plog << "Ignoring contribution.\n";
                        continue;
                    }
                    if (!d_homogeneous_bdry) (*R_data)(idx) += pre_fac * g / cell_volume;
                    (*R_data)(idx) -= pre_fac * a / cell_volume;
                }
            }
        }
        X_petsc_vec->restore_array();
        J_petsc_vec->restore_array();
    }
    ADS_TIMER_STOP(t_applyBoundaryCondition);
}
} // namespace ADS
