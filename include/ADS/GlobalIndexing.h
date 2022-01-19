/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_GlobalIndexing
#define included_ADS_GlobalIndexing

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/FEMeshPartitioner.h>

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
    /*!
     * \brief GlobalIndexing constructor. Sets up an Eulerian patch data index with ghost cell width specified. Requires
     * the mesh partitioner object for the boundary mesh.
     *
     * \note This leaves the object in an uninitialized state. Users must call setupDOFs() before an index mapping is
     * created.
     */
    GlobalIndexing(std::string object_name,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                   std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                   int gcw = 1);

    /*!
     * \brief Destructor that by default calls clearDOFs().
     */
    virtual ~GlobalIndexing();

    /*!
     * \brief Return the eulerian map that maps SAMRAI indices to global indices.
     *
     * \note This patch data index has ghost cell width equal to that specified in the constructor.
     */
    inline int getEulerianMap()
    {
        return d_eul_idx_idx;
    }

    /*!
     * \brief Return the map between libMesh dof's and the global dof's.
     */
    inline const std::map<int, int>& getLagrangianMap()
    {
        return d_lag_petsc_dof_map;
    }

    /*!
     * \brief Return the number of dof's per processor.
     */
    inline const std::vector<int>& getDofsPerProc()
    {
        return d_petsc_dofs_per_proc;
    }

    /*!
     * \brief Setup the mapping between SAMRAI and Lagrangian indices to global indices. This sets up a default mapping
     * that lists all Eulerian dof's and then all libMesh dof's.
     */
    virtual void setupDOFs();

    /*!
     * \brief Reset the object and leave it in an uninitialized state.
     */
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
