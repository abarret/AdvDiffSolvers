#include <ibtk/IBTK_MPI.h>

#include <ADS/GlobalIndexing.h>
#include <ADS/app_namespaces.h>

namespace ADS
{
GlobalIndexing::GlobalIndexing(std::string object_name,
                               Pointer<PatchHierarchy<NDIM>> hierarchy,
                               std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                               int gcw)
    : d_object_name(std::move(object_name)),
      d_hierarchy(hierarchy),
      d_eul_idx_var(new CellVariable<NDIM, int>(d_object_name + "::IdxVar")),
      d_fe_mesh_partitioner(std::move(fe_mesh_partitioner))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_eul_idx_idx = var_db->registerVariableAndContext(
        d_eul_idx_var, var_db->getContext(d_object_name + "::EulIdx"), IntVector<NDIM>(gcw));
}

GlobalIndexing::~GlobalIndexing()
{
    clearDOFs();
}

void
GlobalIndexing::clearDOFs()
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->deallocatePatchData(d_eul_idx_idx);
    }
}

void
GlobalIndexing::setupDOFs()
{
    clearDOFs();
    // First count all the DOFs
    int eul_local_dofs = 0;
    int lag_local_dofs = 0;
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_eul_idx_idx);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(d_eul_idx_idx);
            const int depth = idx_data->getDepth();
            eul_local_dofs += depth * CellGeometry<NDIM>::toCellBox(box).size();
        }
    }

    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const System& sys = eq_sys->get_system(d_sys_x_name);
    const DofMap& dof_map = sys.get_dof_map();
    lag_local_dofs = dof_map.n_local_dofs();

    const int mpi_size = IBTK_MPI::getNodes();
    const int mpi_rank = IBTK_MPI::getRank();
    d_eul_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(eul_local_dofs, d_eul_dofs_per_proc.data());
    d_lag_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(lag_local_dofs, d_lag_dofs_per_proc.data());
    d_petsc_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(eul_local_dofs + lag_local_dofs, d_petsc_dofs_per_proc.data());

    const int local_dof_offset =
        std::accumulate(d_petsc_dofs_per_proc.begin(), d_petsc_dofs_per_proc.begin() + mpi_rank, 0);

    // Now we actually assign PETSc dofs.
    // Start with SAMRAI points
    int counter = local_dof_offset;
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(d_eul_idx_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*idx_data)(idx) = counter++;
            }
        }

        // Communicate ghost DOF indices.
        RefineAlgorithm<NDIM> ghost_fill_alg;
        ghost_fill_alg.registerRefine(d_eul_idx_idx, d_eul_idx_idx, d_eul_idx_idx, nullptr);
        ghost_fill_alg.createSchedule(level)->fillData(0.0);
    }

    // Now the libMesh dofs
    int lag_offset = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.begin() + mpi_rank, 0);
    int tot_lag_dofs = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.end(), 0);
    std::vector<int> libmesh_dofs(tot_lag_dofs, 0), petsc_dofs(tot_lag_dofs, 0);
    const MeshBase& mesh = eq_sys->get_mesh();
    MeshBase::const_node_iterator el_end = mesh.local_nodes_end();
    MeshBase::const_node_iterator el_it = mesh.local_nodes_begin();
    int lag_counter = lag_offset;
    for (; el_it != el_end; ++el_it)
    {
        const Node* const node = *el_it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, dofs);
        for (const auto& dof : dofs)
        {
            libmesh_dofs[lag_counter] = dof;
            petsc_dofs[lag_counter++] = counter++;
        }
    }

    // Now communicate ghost DOF data.
    IBTK_MPI::sumReduction(libmesh_dofs.data(), tot_lag_dofs, IBTK_MPI::getCommunicator());
    IBTK_MPI::sumReduction(petsc_dofs.data(), tot_lag_dofs, IBTK_MPI::getCommunicator());
    for (size_t i = 0; i < tot_lag_dofs; ++i)
    {
        d_lag_petsc_dof_map[libmesh_dofs[i]] = petsc_dofs[i];
    }
}
} // namespace ADS
