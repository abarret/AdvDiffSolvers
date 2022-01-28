#ifndef included_ADS_GhostPoints
#define included_ADS_GhostPoints
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * GhostPoints maintains a list of ghost nodes useful in solving RBF-FD discretized linear systems. It maintains ghost
 * points outside the SAMRAI domain as well as inside the finite element mesh. The width of ghost cells is given in the
 * constructor.
 */
class GhostPoints
{
public:
    /*!
     * \brief Constructor that takes in a boundary mesh. Note that GeneralBoundaryMeshMapping assumes ownership of the
     * mesh.
     */
    GhostPoints(std::string object_name,
                SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner);

    /*!
     * \brief Default constructor.
     */
    GhostPoints() = delete;

    /*!
     * \brief Default deconstructor.
     */
    virtual ~GhostPoints() = default;

    /*!
     * \brief Deleted copy constructor.
     */
    GhostPoints(const GhostPoints& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    GhostPoints& operator=(const GhostPoints& that) = delete;

    /*!
     * \name Get the FEMeshPartitioner.
     */
    //\{
    virtual inline std::shared_ptr<FEMeshPartitioner>& getMeshPartitioner()
    {
        return d_fe_mesh_partitioner;
    }
    //\}

    /*!
     * \brief Find the normals of the mesh.
     */
    void findNormals();

    /*!
     * Update the location of the boundary mesh. An optional argument is provided if the location of the structure is
     * needed at the end of the timestep. By default, this function loops over parts and calls the part specific
     * function.
     */
    virtual void updateGhostNodeLocations(double time, bool end_of_timestep = false);

    virtual void setupGhostNodes();

    inline const std::vector<std::unique_ptr<libMesh::Node>>& getEulerianGhostNodes()
    {
        return d_eul_ghost_nodes;
    };

    inline const std::vector<std::unique_ptr<libMesh::Node>>& getLagrangianGhostNodes()
    {
        return d_lag_ghost_nodes;
    };

    inline size_t getLocalNumGhostNodes()
    {
        return d_local_dofs;
    }

    inline size_t getGlobalNumGhostNodes()
    {
        return d_global_dofs;
    }

protected:
    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;

    std::vector<std::unique_ptr<libMesh::Node>> d_eul_ghost_nodes, d_lag_ghost_nodes;

    std::string d_normal_sys_name = "normal";
    double d_ds = std::numeric_limits<double>::quiet_NaN();
    size_t d_local_dofs = 0;
    size_t d_global_dofs = 0;

    // Prescribed geometry
    IBTK::VectorNd d_com;

private:
    void commonConstructor(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);
};

} // namespace ADS
#endif
