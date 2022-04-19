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
 * GhostPoint is a point that lies outside the physical domain. Ghost points can contain knowledge of the "parent node"
 * which is either the libMesh node or SAMRAI index that this ghost point is generated from.
 */
class GhostPoint
{
public:
    GhostPoint(const IBTK::VectorNd& x, size_t id) : d_x(x), d_id(id)
    {
        // intentionally blank
    }

    ~GhostPoint() = default;

    inline const IBTK::VectorNd& getX() const
    {
        return d_x;
    }

    inline size_t getId() const
    {
        return d_id;
    }

    inline double operator()(unsigned int d) const
    {
        return d_x(d);
    }

private:
    IBTK::VectorNd d_x;
    size_t d_id;
};

/*!
 * GhostPoints maintains a list of ghost nodes useful in solving RBF-FD discretized linear systems. It maintains ghost
 * points outside the SAMRAI domain as well as inside the finite element mesh. The width of ghost cells is given in the
 * constructor.
 *
 * \note This is currently specialized to use a geometry of a circle (or sphere in 3D), see findNormals().
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
    virtual inline const std::shared_ptr<FEMeshPartitioner>& getMeshPartitioner() const
    {
        return d_fe_mesh_partitioner;
    }
    //\}

    /*!
     * \brief Find the normals of the mesh. This is currently specialized to find the normals of a circle (or sphere in
     * 3D). An accurate computation of the normals is required for other structures.
     */
    void findNormals();

    /*!
     * Update the location of the ghost nodes. By default, this clears the ghost nodes and recreates them via
     * setupGhostNodes(). Note the normals should be updated prior to calling this function.
     */
    virtual void updateGhostNodeLocations(double time);

    /*!
     * \brief Setup the ghost nodes.
     *
     * Two types of ghost nodes are created. Eulerian ghost nodes are one grid cell outside the physical domain.
     * Lagrangian ghost nodes are created at a specified distance from boundary nodes. This relies on an accurate normal
     * to move them, which is computed in findNormals(). Note the normals should be updated prior to calling this
     * function.
     */
    virtual void setupGhostNodes();

    /*!
     * \brief Return the Eulerian ghost nodes.
     */
    inline const std::vector<GhostPoint>& getEulerianGhostNodes() const
    {
        return d_eul_ghost_nodes;
    };

    /*!
     * \brief Return the Lagrangian ghost nodes.
     */
    inline const std::vector<GhostPoint>& getLagrangianGhostNodes() const
    {
        return d_lag_ghost_nodes;
    };

    /*!
     * \brief Return the number of local ghost nodes.
     */
    inline size_t getLocalNumGhostNodes() const
    {
        return d_local_dofs;
    }

    /*!
     * \brief Return the global number of ghost nodes. This is just the sum of all local dofs.
     */
    inline size_t getGlobalNumGhostNodes() const
    {
        return d_global_dofs;
    }

    /*!
     * \brief Return the system name for the normals.
     */
    inline const std::string& getNormalSysName() const
    {
        return d_normal_sys_name;
    }

protected:
    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;

    std::vector<GhostPoint> d_eul_ghost_nodes, d_lag_ghost_nodes;

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
