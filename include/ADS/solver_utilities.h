#ifndef included_ADS_solver_utilities
#define included_ADS_solver_utilities

#include <ADS/ConditionCounter.h>

#include <libmesh/dof_map.h>

#include <petscvec.h>

#include <SAMRAIVectorReal.h>

#include <map>
#include <vector>

namespace ADS
{
void copyDataToPetsc(Vec& petsc_vec,
                     const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x_eul_vec,
                     SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                     const libMesh::System& x_lag_sys,
                     int eul_map_idx,
                     const std::map<int, int>& lag_dof_map,
                     const std::vector<int>& dofs_per_proc);

void copyDataFromPetsc(Vec& petsc_vec,
                       const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x_eul_vec,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       libMesh::System& x_lag_sys,
                       int eul_map_idx,
                       const std::map<int, int>& lag_dof_map,
                       const std::vector<int>& dofs_per_proc);

void copyDataFromPetsc(Vec& petsc_vec,
                       const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x_eul_vec,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       libMesh::System& x_lag_sys,
                       const ConditionCounter& cc);
} // namespace ADS

#endif // #ifndef included_ADS_solver_utilities
