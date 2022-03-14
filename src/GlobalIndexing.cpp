#include <ADS/GlobalIndexing.h>
#include <ADS/app_namespaces.h>

#include <ibtk/IBTK_MPI.h>

#include <RefineAlgorithm.h>

namespace ADS
{
GlobalIndexing::GlobalIndexing(std::string object_name,
                               Pointer<PatchHierarchy<NDIM>> hierarchy,
                               std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                               std::string sys_name,
                               std::shared_ptr<GhostPoints> ghost_pts,
                               int gcw,
                               unsigned int depth)
    : d_object_name(std::move(object_name)),
      d_hierarchy(hierarchy),
      d_eul_idx_var(new CellVariable<NDIM, int>(d_object_name + "::IdxVar"), depth),
      d_fe_mesh_partitioner(std::move(fe_mesh_partitioner)),
      d_sys_name(std::move(sys_name)),
      d_ghost_pts(std::move(ghost_pts)),
      d_depth(depth)
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
    int ghost_local_dofs = 0;
    // Eulerian
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

    // Lagrangian
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    System& sys = eq_sys->get_system(d_sys_name);
    TBOX_ASSERT(sys.n_vars() == d_depth);
    const DofMap& dof_map = sys.get_dof_map();
    lag_local_dofs = dof_map.n_local_dofs();

    // Ghost
    ghost_local_dofs = d_ghost_pts->getLocalNumGhostNodes() * d_depth;

    const int mpi_size = IBTK_MPI::getNodes();
    const int mpi_rank = IBTK_MPI::getRank();
    d_eul_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(eul_local_dofs, d_eul_dofs_per_proc.data());
    d_lag_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(lag_local_dofs, d_lag_dofs_per_proc.data());
    d_ghost_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(ghost_local_dofs, d_ghost_dofs_per_proc.data());
    d_petsc_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(eul_local_dofs + lag_local_dofs + ghost_local_dofs, d_petsc_dofs_per_proc.data());

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
    size_t lag_offset = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.begin() + mpi_rank, 0);
    size_t tot_lag_dofs = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.end(), 0);
    std::vector<int> libmesh_dofs(tot_lag_dofs, 0), petsc_dofs(tot_lag_dofs, 0);
    const MeshBase& mesh = eq_sys->get_mesh();
    MeshBase::const_node_iterator el_end = mesh.local_nodes_end();
    MeshBase::const_node_iterator el_it = mesh.local_nodes_begin();
    size_t lag_counter = lag_offset;
    for (; el_it != el_end; ++el_it)
    {
        const Node* const node = *el_it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, dofs, 0);

        for (const auto& dof : dofs)
        {
            libmesh_dofs[lag_counter] = dof;
            petsc_dofs[lag_counter++] = counter++;
        }
    }

    // Communicate ghost DOF data.
    IBTK_MPI::sumReduction(libmesh_dofs.data(), tot_lag_dofs);
    IBTK_MPI::sumReduction(petsc_dofs.data(), tot_lag_dofs);
    for (size_t i = 0; i < tot_lag_dofs; ++i) d_lag_petsc_dof_map[libmesh_dofs[i]] = petsc_dofs[i];

    // Finally, ghost DOFs
    size_t ghost_offset = std::accumulate(d_ghost_dofs_per_proc.begin(), d_ghost_dofs_per_proc.begin() + mpi_rank, 0);
    size_t tot_ghost_dofs = std::accumulate(d_ghost_dofs_per_proc.begin(), d_ghost_dofs_per_proc.end(), 0);
    std::vector<int> ghost_dofs(tot_ghost_dofs, 0);
    petsc_dofs.resize(tot_ghost_dofs, 0);
    size_t ghost_counter = ghost_offset;
    const std::vector<std::unique_ptr<Node>>& eul_ghost_nodes = d_ghost_pts->getEulerianGhostNodes();
    const std::vector<std::unique_ptr<Node>>& lag_ghost_nodes = d_ghost_pts->getLagrangianGhostNodes();
    for (const auto& eul_ghost_node : eul_ghost_nodes)
    {
        ghost_dofs[ghost_counter] = eul_ghost_node->id();
        petsc_dofs[ghost_counter++] = counter++;
    }
    for (const auto& lag_ghost_node : lag_ghost_nodes)
    {
        ghost_dofs[ghost_counter] = lag_ghost_node->id();
        petsc_dofs[ghost_counter++] = counter++;
    }

    // Communicate ghost DOF data.
    IBTK_MPI::sumReduction(ghost_dofs.data(), tot_ghost_dofs);
    IBTK_MPI::sumReduction(petsc_dofs.data(), tot_ghost_dofs);
    for (size_t i = 0; i < tot_ghost_dofs; ++i) d_ghost_petsc_dof_map[ghost_dofs[i]] = petsc_dofs[i];
}
} // namespace ADS
