/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_GlobalIndexing
#define included_ADS_GlobalIndexing

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/FEMeshPartitioner.h>
#include <ADS/RBFFDWeightsCache.h>

#include <mpi.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * GlobalIndexing is a class that manages a mapping between SAMRAI and libMesh local partitioning and a global PETSc
 * partitioning. This class makes it easier to set up PETSc structures that contain degrees of freedom from both libMesh
 * and SAMRAI data structures.
 */
class GlobalIndexing
{
public:
    GlobalIndexing(std::string object_name,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                   std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner);

    virtual ~GlobalIndexing();

    inline int getEulerianMap()
    {
        return d_eul_idx_idx;
    }

    inline const std::map<int, int>& getLagrangianMap()
    {
        return d_lag_petsc_dof_map;
    }

    inline const std::vector<int>& getDofsPerProc()
    {
        return d_petsc_dofs_per_proc;
    }

    virtual void setupDOFs();

    virtual void clearDOFs();

protected:
    std::string d_object_name;

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    std::vector<int> d_eul_dofs_per_proc;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, int>> d_eul_idx_var;
    int d_eul_idx_idx = IBTK::invalid_index;

    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;
    std::string d_sys_x_name, d_sys_b_name;
    std::map<int, int> d_lag_petsc_dof_map;
    std::vector<int> d_lag_dofs_per_proc;

    std::vector<int> d_petsc_dofs_per_proc;
};

} // namespace ADS

#endif // included_ADS_GlobalIndexing
