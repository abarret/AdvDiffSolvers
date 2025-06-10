#include "ADS/app_namespaces.h"
#include "ADS/reconstructions.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/libmesh_utilities.h"
#include <ibtk/LData.h>

#include "libmesh/boundary_info.h"
#include "libmesh/enum_xdr_mode.h"
#include "libmesh/explicit_system.h"

// Local includes
#include "IBBoundaryMeshMapping.h"

IBBoundaryMeshMapping::IBBoundaryMeshMapping(std::string object_name,
                                             Pointer<Database> input_db,
                                             MeshBase* mesh,
                                             LDataManager* data_manager,
                                             const int level_num,
                                             Pointer<PatchHierarchy<NDIM>> hierarchy,
                                             const std::string& restart_read_dirname,
                                             unsigned int restart_restore_number)
    : GeneralBoundaryMeshMapping(std::move(object_name), input_db, mesh, restart_read_dirname, restart_restore_number),
      d_base_data_manager(data_manager),
      d_part_nums({ 0 }),
      d_level_num(level_num),
      d_hierarchy(hierarchy)
{
    // intentionally blank
}

IBBoundaryMeshMapping::IBBoundaryMeshMapping(std::string object_name,
                                             Pointer<Database> input_db,
                                             const std::vector<MeshBase*>& meshes,
                                             LDataManager* data_manager,
                                             const int level_num,
                                             std::vector<int> part_nums,
                                             Pointer<PatchHierarchy<NDIM>> hierarchy,
                                             const std::string& restart_read_dirname,
                                             unsigned int restart_restore_number)
    : GeneralBoundaryMeshMapping(std::move(object_name),
                                 input_db,
                                 meshes,
                                 restart_read_dirname,
                                 restart_restore_number),
      d_base_data_manager(data_manager),
      d_part_nums(std::move(part_nums)),
      d_level_num(level_num),
      d_hierarchy(hierarchy)
{
    // intentionally blank
}

void
IBBoundaryMeshMapping::updateBoundaryLocation(const double time, unsigned int part, const bool end_of_timestep)
{
    int base_part = d_part_nums[part];
    const std::pair<int, int>& idx_pair = d_base_data_manager->getLagrangianStructureIndexRange(base_part, d_level_num);
    const std::vector<LNode*>& nodes = d_base_data_manager->getLMesh(d_level_num)->getLocalNodes();

    Pointer<LData> X_data;
    X_data = d_base_data_manager->getLData(d_base_data_manager->POSN_DATA_NAME, d_level_num);
    const double* X_vals;
    int ierr = VecGetArrayRead(X_data->getVec(), &X_vals);
    IBTK_CHKERRQ(ierr);

    System& X_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_coords_sys_name);
    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();
    NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();

    System& dX_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_disp_sys_name);
    NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

    for (const auto& node : nodes)
    {
        const int lag_idx = node->getLagrangianIndex();
        const int petsc_idx = node->getLocalPETScIndex();
        const int bdry_node_id = lag_idx;
        Node* bdry_node = d_bdry_meshes[part]->node_ptr(bdry_node_id);
        std::vector<dof_id_type> X_bdry_dof_indices;
        for (int d = 0; d < NDIM; ++d)
        {
            X_bdry_dof_map.dof_indices(bdry_node, X_bdry_dof_indices, d);
            X_bdry_vec->set(X_bdry_dof_indices[0], X_vals[petsc_idx * NDIM + d]);
            dX_bdry_vec->set(X_bdry_dof_indices[0], X_vals[petsc_idx * NDIM + d] - (*bdry_node)(d));
        }
    }

    X_bdry_vec->close();
    dX_bdry_vec->close();
    X_bdry_sys.update();
    dX_bdry_sys.update();

    ierr = VecRestoreArrayRead(X_data->getVec(), &X_vals);
    IBTK_CHKERRQ(ierr);
    return;
}

void
IBBoundaryMeshMapping::initializeBoundaryLocation(const double time, const unsigned int part)
{
    // Here we assume the boundary mesh is in the correct initial location, so we set the X and dX to be the mesh
    // position and 0 respectively.
    System& X_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_coords_sys_name);
    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();
    NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();

    System& dX_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_disp_sys_name);
    NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

    const MeshBase& mesh = d_bdry_eq_sys_vec[part]->get_mesh();
    auto node_it = mesh.local_nodes_begin();
    auto node_end = mesh.local_nodes_end();
    for (; node_it != node_end; ++node_it)
    {
        const Node* bdry_node = *node_it;
        std::vector<dof_id_type> X_dof_idx;
        for (int d = 0; d < NDIM; ++d)
        {
            X_bdry_dof_map.dof_indices(bdry_node, X_dof_idx, d);
            X_bdry_vec->set(X_dof_idx[0], (*bdry_node)(d));
            dX_bdry_vec->set(X_dof_idx[0], 0.0);
        }
    }

    X_bdry_vec->close();
    dX_bdry_vec->close();
    X_bdry_sys.update();
    dX_bdry_sys.update();
    return;
}
