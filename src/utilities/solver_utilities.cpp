#include <ibtk/IBTK_CHKERRQ.h>
#include <ibtk/IBTK_MPI.h>

#include <ADS/app_namespaces.h>
#include <ADS/solver_utilities.h>
#include <tbox/Pointer.h>

#include <libmesh/numeric_vector.h>
#include <libmesh/system.h>

#include <PatchLevel.h>

namespace ADS
{
void
copyDataToPetsc(Vec& petsc_vec,
                const SAMRAIVectorReal<NDIM, double>& x_eul_vec,
                Pointer<PatchHierarchy<NDIM>> hierarchy,
                const System& x_lag_sys,
                int eul_map_idx,
                const std::map<int, int>& lag_dof_map,
                const std::vector<int>& /*dofs_per_proc*/)
{
    // We are assuming the PETSc Vec has been allocated correctly.
    int ierr;
    // Start with Eulerian data
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(hierarchy->getFinestLevelNumber());
#ifndef NDEBUG
    if (x_eul_vec.getNumberOfComponents() > 1)
        TBOX_ERROR("copyDataToPetsc: x_eul_vec has " << x_eul_vec.getNumberOfComponents()
                                                     << ". This function only knows how to copy vectors with depth 1.");
#endif
    for (int comp = 0; comp < x_eul_vec.getNumberOfComponents(); ++comp)
    {
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(eul_map_idx);
            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_eul_vec.getComponentDescriptorIndex(comp));
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                int petsc_idx = (*idx_data)(idx, comp);
                ierr = VecSetValue(petsc_vec, petsc_idx, (*x_data)(idx), INSERT_VALUES);
                IBTK_CHKERRQ(ierr);
            }
        }
    }

    // Now the Lag data
    // Note the corresponding PETSc index is the libMesh index plus the number of SAMRAI indices
    const MeshBase& mesh = x_lag_sys.get_mesh();
    NumericVector<double>* x_lag_vec = x_lag_sys.solution.get();
    const DofMap& dof_map = x_lag_sys.get_dof_map();
    auto it = mesh.local_nodes_begin();
    auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, dofs);
        for (const auto& dof : dofs)
        {
            const int petsc_dof = lag_dof_map.at(dof);
            ierr = VecSetValue(petsc_vec, petsc_dof, (*x_lag_vec)(dof), INSERT_VALUES);
            IBTK_CHKERRQ(ierr);
        }
    }
    x_lag_vec->close();
    ierr = VecAssemblyBegin(petsc_vec);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyEnd(petsc_vec);
    IBTK_CHKERRQ(ierr);
}

void
copyDataFromPetsc(Vec& petsc_vec,
                  const SAMRAIVectorReal<NDIM, double>& x_eul_vec,
                  Pointer<PatchHierarchy<NDIM>> hierarchy,
                  System& x_lag_sys,
                  int eul_map_idx,
                  const std::map<int, int>& lag_dof_map,
                  const std::vector<int>& dofs_per_proc)
{
    // We are assuming the PETSc Vec has been allocated correctly.
    int ierr;
    const double* x_petsc_data;
    ierr = VecGetArrayRead(petsc_vec, &x_petsc_data);
    IBTK_CHKERRQ(ierr);
    // Start with Eulerian data
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(hierarchy->getFinestLevelNumber());
    const int mpi_rank = IBTK_MPI::getRank();
    const int local_offset = std::accumulate(dofs_per_proc.begin(), dofs_per_proc.begin() + mpi_rank, 0);
    for (int comp = 0; comp < x_eul_vec.getNumberOfComponents(); ++comp)
    {
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(eul_map_idx);
            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_eul_vec.getComponentDescriptorIndex(comp));
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                int petsc_idx = (*idx_data)(idx, comp);
                (*x_data)(idx) = x_petsc_data[petsc_idx - local_offset];
            }
        }
    }

    // Now the Lag data
    const MeshBase& mesh = x_lag_sys.get_mesh();
    NumericVector<double>* x_lag_vec = x_lag_sys.solution.get();
    const DofMap& dof_map = x_lag_sys.get_dof_map();
    auto it = mesh.local_nodes_begin();
    auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, dofs);
        for (const auto& dof : dofs)
        {
            int petsc_dof = lag_dof_map.at(dof);
            x_lag_vec->set(dof, x_petsc_data[petsc_dof - local_offset]);
        }
    }
    x_lag_vec->close();
    x_lag_sys.update();
    ierr = VecRestoreArrayRead(petsc_vec, &x_petsc_data);
    IBTK_CHKERRQ(ierr);
}
} // namespace ADS
