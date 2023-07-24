#include "ADS/VolumeBoundaryMeshMapping.h"
#include "ADS/app_namespaces.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/libmesh_utilities.h"

#include "libmesh/boundary_info.h"
#include "libmesh/enum_xdr_mode.h"
#include "libmesh/explicit_system.h"

namespace ADS
{
VolumeBoundaryMeshMapping::VolumeBoundaryMeshMapping(std::string object_name,
                                                     Pointer<Database> input_db,
                                                     MeshBase* mesh,
                                                     FEDataManager* fe_data_manager,
                                                     std::vector<std::set<boundary_id_type>> bdry_ids,
                                                     const std::string& restart_read_dirname,
                                                     unsigned int restart_restore_number)
    : GeneralBoundaryMeshMapping(std::move(object_name), input_db, mesh, restart_read_dirname, restart_restore_number),
      d_base_fe_data_managers({ fe_data_manager }),
      d_bdry_ids_vec(std::move(bdry_ids)),
      d_vol_id_vec({ 0 })
{
    // intentionally blank
}

VolumeBoundaryMeshMapping::VolumeBoundaryMeshMapping(std::string object_name,
                                                     Pointer<Database> input_db,
                                                     const std::vector<MeshBase*>& meshes,
                                                     const std::vector<FEDataManager*>& fe_data_managers,
                                                     std::vector<std::set<boundary_id_type>> bdry_ids,
                                                     std::vector<unsigned int> parts,
                                                     const std::string& restart_read_dirname,
                                                     unsigned int restart_restore_number)
    : GeneralBoundaryMeshMapping(std::move(object_name),
                                 input_db,
                                 meshes,
                                 restart_read_dirname,
                                 restart_restore_number),
      d_base_fe_data_managers(fe_data_managers),
      d_bdry_ids_vec(std::move(bdry_ids)),
      d_vol_id_vec(std::move(parts))
{
    // intentionally blank
}

void
VolumeBoundaryMeshMapping::updateBoundaryLocation(const double time,
                                                  const unsigned int part,
                                                  const bool end_of_timestep)
{
    FEDataManager* fe_data_manager = d_base_fe_data_managers[d_vol_id_vec[part]];
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();

    System& X_system = eq_sys->get_system(fe_data_manager->getCurrentCoordinatesSystemName());
    const DofMap& X_dof_map = X_system.get_dof_map();
    NumericVector<double>* X_vec;
    if (!end_of_timestep)
        X_vec = X_system.solution.get();
    else
        X_vec = &X_system.get_vector("new");

    System& X_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_coords_sys_name);
    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();
    NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();

    System& dX_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_disp_sys_name);
    NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

    std::map<dof_id_type, dof_id_type> node_id_map;
    std::map<dof_id_type, unsigned char> side_id_map;
    d_base_meshes[d_vol_id_vec[part]]->boundary_info->get_side_and_node_maps(
        *d_bdry_meshes[part], node_id_map, side_id_map);
    auto node_it = d_bdry_meshes[part]->local_nodes_begin();
    auto node_end = d_bdry_meshes[part]->local_nodes_end();
    for (; node_it != node_end; ++node_it)
    {
        Node* node = *node_it;
        dof_id_type bdry_node_id = node->id();
        // TODO: This is potentially expensive. We should cache our own map between bdry nodes and volumetric nodes.
        auto vol_iter = std::find_if(
            node_id_map.begin(), node_id_map.end(), [bdry_node_id](const std::pair<dof_id_type, dof_id_type>& obj) {
                return obj.second == bdry_node_id;
            });
        dof_id_type vol_node_id = vol_iter->first;
        // Grab current position of volumetric mesh.
        std::vector<dof_id_type> X_dof_indices, X_bdry_dof_indices;
        for (int d = 0; d < NDIM; ++d)
        {
            X_dof_map.dof_indices(d_base_meshes[d_vol_id_vec[part]]->node_ptr(vol_node_id), X_dof_indices, d);
            X_bdry_dof_map.dof_indices(node, X_bdry_dof_indices, d);
            X_bdry_vec->set(X_bdry_dof_indices[0], (*X_vec)(X_dof_indices[0]));
            dX_bdry_vec->set(X_bdry_dof_indices[0], (*X_vec)(X_dof_indices[0]) - (*node)(d));
        }
    }
    X_bdry_vec->close();
    dX_bdry_vec->close();
    X_bdry_sys.update();
    dX_bdry_sys.update();
    return;
}

void
VolumeBoundaryMeshMapping::buildBoundaryMesh()
{
    unsigned int num_parts = d_vol_id_vec.size();
    d_bdry_meshes.resize(num_parts);
    for (unsigned int part = 0; part < d_vol_id_vec.size(); ++part)
    {
        unsigned int vol_part = d_vol_id_vec[part];
        d_vol_id_vec[part] = vol_part;
        auto bdry_mesh = std::make_unique<BoundaryMesh>(d_base_meshes[vol_part]->comm(),
                                                        d_base_meshes[vol_part]->spatial_dimension() - 1);
        d_base_meshes[vol_part]->boundary_info->sync(d_bdry_ids_vec[part], *bdry_mesh);
        d_bdry_meshes[part] = std::move(bdry_mesh);
    }
}
} // namespace ADS
