/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_GlobalIndexing
#define included_ADS_GlobalIndexing

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/FDPoint.h>
#include <ADS/FEMeshPartitioner.h>
#include <ADS/GhostPoints.h>

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
     * the mesh partitioner object for the boundary mesh. Additional ghost points may be registered.
     *
     * \note This leaves the object in an uninitialized state. Users must call setupDOFs() before an index mapping is
     * created.
     */
    GlobalIndexing(std::string object_name,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                   std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                   std::string sys_name,
                   std::shared_ptr<GhostPoints> ghost_pts,
                   int gcw = 1,
                   unsigned int depth = 1);

    /*!
     * \brief Destructor that by default calls clearDOFs().
     */
    virtual ~GlobalIndexing();

    /*!
     * \brief Return the eulerian map that maps SAMRAI indices to global indices.
     *
     * \note This patch data index has ghost cell width equal to that specified in the constructor.
     */
    inline int getEulerianMap() const
    {
        return d_eul_idx_idx;
    }

    /*!
     * \brief Return the map between libMesh dof's and the global dof's.
     */
    inline const std::map<int, int>& getLagrangianMap() const
    {
        return d_lag_petsc_dof_map;
    }

    /*!
     * \brief Return the map between the ghost dof's and the global dof's.
     */
    inline const std::map<int, int>& getGhostMap() const
    {
        return d_ghost_petsc_dof_map;
    }

    /*!
     * \brief Return the number of dof's per processor.
     */
    inline const std::vector<int>& getDofsPerProc() const
    {
        return d_petsc_dofs_per_proc;
    }

    /*!
     * \brief Setup the mapping between SAMRAI and Lagrangian indices to global indices. This sets up a default mapping
     * that lists all Eulerian dof's, then all libMesh dof's, and finally all the ghost indices.
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
    std::string d_sys_name;
    std::map<int, int> d_lag_petsc_local_dof_map;
    std::map<int, int> d_lag_petsc_dof_map;
    std::vector<int> d_lag_dofs_per_proc;

    std::shared_ptr<GhostPoints> d_ghost_pts;
    std::map<int, int> d_ghost_petsc_local_dof_map;
    std::map<int, int> d_ghost_petsc_dof_map;
    std::vector<int> d_ghost_dofs_per_proc;

    std::vector<int> d_petsc_dofs_per_proc;

    unsigned int d_depth = 1;
};

inline int
getDofIndex(const FDPoint& pt,
            SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
            const libMesh::DofMap& dof_map,
            const GlobalIndexing& cache)
{
    if (pt.isIdx())
    {
        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellData<NDIM, int>> idx_data = patch->getPatchData(cache.getEulerianMap());
        return (*idx_data)(pt.getIndex());
    }
    else if (pt.isNode())
    {
        const std::map<int, int>& lag_petsc_dof_map = cache.getLagrangianMap();
        std::vector<libMesh::dof_id_type> dof_idxs;
        dof_map.dof_indices(pt.getNode(), dof_idxs);
        return lag_petsc_dof_map.at(dof_idxs[0]);
    }
    else if (pt.isGhost())
    {
        const std::map<int, int>& ghost_petsc_dof_map = cache.getGhostMap();
        return ghost_petsc_dof_map.at(pt.getGhostPoint()->getId());
    }
    TBOX_ERROR("Should not reach this statement\n");
    return IBTK::invalid_index;
}
} // namespace ADS

#endif // included_ADS_GlobalIndexing
