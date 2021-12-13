#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/app_namespaces.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/libmesh_utilities.h"

#include "libmesh/enum_xdr_mode.h"
#include "libmesh/explicit_system.h"

namespace ADS
{
GeneralBoundaryMeshMapping::GeneralBoundaryMeshMapping(std::string object_name,
                                                       Pointer<Database> input_db,
                                                       MeshBase* bdry_mesh,
                                                       const std::string& restart_read_dirname,
                                                       const unsigned int restart_restore_number)
    : d_object_name(std::move(object_name))
{
    d_bdry_meshes.push_back(bdry_mesh);
    d_own_bdry_mesh.push_back(0);
    commonConstructor(input_db, restart_read_dirname, restart_restore_number);
}

GeneralBoundaryMeshMapping::GeneralBoundaryMeshMapping(std::string object_name,
                                                       Pointer<Database> input_db,
                                                       const std::vector<MeshBase*>& bdry_mesh,
                                                       const std::string& restart_read_dirname,
                                                       const unsigned int restart_restore_number)
    : d_object_name(std::move(object_name))
{
    for (unsigned int part = 0; part < bdry_mesh.size(); ++part)
    {
        d_bdry_meshes.push_back(bdry_mesh[part]);
        d_own_bdry_mesh.push_back(0);
    }
    commonConstructor(input_db, restart_read_dirname, restart_restore_number);
}

GeneralBoundaryMeshMapping::GeneralBoundaryMeshMapping(std::string object_name) : d_object_name(std::move(object_name))
{
    // intentionally blank
}

GeneralBoundaryMeshMapping::~GeneralBoundaryMeshMapping()
{
    // Clean up allocated memory if we need to
    for (unsigned int part = 0; part < d_bdry_meshes.size(); ++part)
    {
        if (d_own_bdry_mesh[part])
        {
            delete d_bdry_meshes[part];
            d_bdry_meshes[part] = nullptr;
            d_own_bdry_mesh[part] = 0;
        }
    }
}

void
GeneralBoundaryMeshMapping::commonConstructor(Pointer<Database> input_db,
                                              std::string restart_read_dirname,
                                              unsigned int restart_restore_number)
{
    const bool from_restart = RestartManager::getManager()->isFromRestart();
    unsigned int num_parts = d_bdry_meshes.size();
    d_bdry_meshes.resize(num_parts);
    d_bdry_eq_sys_vec.resize(num_parts);
    d_fe_data.resize(num_parts);
    for (unsigned int part = 0; part < num_parts; ++part)
    {
        d_bdry_eq_sys_vec[part] = std::move(libmesh_make_unique<EquationSystems>(*d_bdry_meshes[part]));
        d_fe_data[part] = std::make_shared<FEData>(
            d_object_name + "::FEData::" + std::to_string(part), *d_bdry_eq_sys_vec[part], true);

        if (from_restart)
        {
            const std::string& file_name = get_libmesh_restart_file_name(
                restart_read_dirname, d_object_name, restart_restore_number, part, d_libmesh_restart_file_extension);
            const XdrMODE xdr_mode = (d_libmesh_restart_file_extension == "xdr" ? DECODE : READ);
            const int read_mode =
                EquationSystems::READ_HEADER | EquationSystems::READ_DATA | EquationSystems::READ_ADDITIONAL_DATA;
            d_bdry_eq_sys_vec[part]->read(file_name, xdr_mode, read_mode, /*partition_agnostic*/ true);
        }
        else
        {
            auto& X_sys = d_bdry_eq_sys_vec[part]->add_system<ExplicitSystem>(d_coords_sys_name);
            for (unsigned int d = 0; d < NDIM; ++d) X_sys.add_variable("X_" + std::to_string(d), FEType());
            auto& dX_sys = d_bdry_eq_sys_vec[part]->add_system<ExplicitSystem>(d_disp_sys_name);
            for (unsigned int d = 0; d < NDIM; ++d) dX_sys.add_variable("dX_" + std::to_string(d), FEType());
            X_sys.assemble_before_solve = false;
            X_sys.assemble();
            dX_sys.assemble_before_solve = false;
            dX_sys.assemble();
        }
        d_bdry_mesh_partitioners.push_back(
            std::make_shared<FEMeshPartitioner>(d_object_name + "::FEMeshPartitioner::" + std::to_string(part),
                                                input_db,
                                                input_db->getInteger("max_level"),
                                                IntVector<NDIM>(0),
                                                d_fe_data[part],
                                                d_coords_sys_name));
    }
    return;
}

void
GeneralBoundaryMeshMapping::updateBoundaryLocation(const double time, const bool end_of_timestep)
{
    for (unsigned int part = 0; part < d_bdry_meshes.size(); ++part)
        updateBoundaryLocation(time, part, end_of_timestep);
    return;
}

void
GeneralBoundaryMeshMapping::updateBoundaryLocation(const double time,
                                                   const unsigned int part,
                                                   const bool end_of_timestep)
{
    // Set the X system to the element location.
    System& X_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_coords_sys_name);
    System& dX_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_disp_sys_name);
    NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();
    NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();

    auto node_it = d_bdry_meshes[part]->local_nodes_begin();
    const auto node_end = d_bdry_meshes[part]->local_nodes_end();
    for (; node_it != node_end; ++node_it)
    {
        Node* node = *node_it;
        std::vector<dof_id_type> X_bdry_dof_indices;
        for (int d = 0; d < NDIM; ++d)
        {
            X_bdry_dof_map.dof_indices(node, X_bdry_dof_indices, d);
            X_bdry_vec->set(X_bdry_dof_indices[0], (*node)(d));
            dX_bdry_vec->set(X_bdry_dof_indices[0], 0.0);
        }
    }

    X_bdry_vec->close();
    dX_bdry_vec->close();
    X_bdry_sys.update();
    dX_bdry_sys.update();
}

void
GeneralBoundaryMeshMapping::initializeEquationSystems()
{
    const bool from_restart = RestartManager::getManager()->isFromRestart();
    if (from_restart) return;
    for (unsigned int part = 0; part < d_bdry_meshes.size(); ++part)
    {
        d_bdry_eq_sys_vec[part]->init();
    }
    updateBoundaryLocation(0.0, false);
    return;
}

void
GeneralBoundaryMeshMapping::writeFEDataToRestartFile(const std::string& restart_dump_dirname,
                                                     unsigned int time_step_number)
{
    for (unsigned int part = 0; part < d_bdry_eq_sys_vec.size(); ++part)
    {
        const std::string& file_name = get_libmesh_restart_file_name(
            restart_dump_dirname, d_object_name, time_step_number, part, d_libmesh_restart_file_extension);
        const XdrMODE xdr_mode = (d_libmesh_restart_file_extension == "xdr" ? ENCODE : WRITE);
        const int write_mode = EquationSystems::WRITE_DATA | EquationSystems::WRITE_ADDITIONAL_DATA;
        d_bdry_eq_sys_vec[part]->write(file_name, xdr_mode, write_mode, /*partition_agnostic*/ true);
    }
}
} // namespace ADS