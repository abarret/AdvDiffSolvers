/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_FESystemManager
#define included_ADS_FESystemManager

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ibtk/FEDataManager.h"
#include "ibtk/ibtk_enums.h"
#include "ibtk/ibtk_utilities.h"

#include "CellVariable.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "RefineSchedule.h"
#include "VariableContext.h"
#include "tbox/Pointer.h"
#include "tbox/Serializable.h"

#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/equation_systems.h"
#include "libmesh/linear_solver.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/quadrature.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/system.h"

IBTK_DISABLE_EXTRA_WARNINGS
#include <boost/multi_array.hpp>
IBTK_ENABLE_EXTRA_WARNINGS

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class FEMeshPartitioner coordinates data required for
 * Lagrangian-Eulerian interaction between a Lagrangian finite element (FE)
 * mesh. In particular, the FEData member object stores the necessary finite
 * element data while this class stores additional data dependent on the
 * Eulerian grid.
 *
 * <h2>Parameters read from the input database</h2>
 *
 * <code>node_outside_patch_check</code>: parameter controlling how this class
 * responds to mesh nodes outside the finest patch level. In all cases, for
 * backwards compatibility, nodes that are outside the computational domain are
 * permitted and are ignored by this check. Possible values are:
 * <ol>
 *   <li><code>node_outside_permit</code>: Permit nodes to be outside the finest
 *   patch level.</li>
 *   <li><code>node_outside_warn</code>: Permit nodes to be outside the finest
 *   patch level, but log a warning whenever this is detected.
 *   <li><code>node_outside_error</code>: Abort the program when nodes are detected
 *   outside the finest patch level.
 * </ol>
 * The default value is <code>node_outside_error</code>.
 *
 * <code>subdomain_ids_on_levels</code>: a database correlating libMesh subdomain
 * IDs to patch levels. A possible value for this is
 * @code
 * subdomain_ids_on_levels
 * {
 *   level_1 = 4
 *   level_3 = 10, 12, 14
 *   level_5 = 1000, 1003, 1006, 1009
 * }
 * @endcode
 * This particular input will associate all elements with subdomain id 4 with
 * patch level 1, all elements with subdomain ids 10, 12, or 14 with patch level
 * 3, etc. All unspecified subdomain ids will be associated with the finest
 * patch level. All inputs in this database for levels finer than the finest
 * level are ignored (e.g., if the maximum patch level number is 4, then the
 * values given in the example for level 5 ultimately end up on level 4).
 * <em>This feature is experimental</em>: at the current time it is known that
 * it produces some artifacts at the coarse-fine interface, but that these
 * generally don't effect the overall solution quality.
 *
 * <h2>Parameters effecting workload estimate calculations</h2>
 * FEMeshPartitioner can estimate the amount of work done in IBFE calculations
 * (such as FEMeshPartitioner::spread). Since most calculations use a variable
 * number of quadrature points on each libMesh element this estimate can vary
 * quite a bit over different Eulerian cells corresponding to a single
 * mesh. The current implementation estimates the workload on each cell of the
 * background Eulerian grid by applying a background value representing the
 * work on the Eulerian cell itself and a weight times the number of
 * quadrature points on that cell. These values are set at the time of object
 * construction through the FEMeshPartitioner::WorkloadSpec object, which contains
 * reasonable defaults.
 *
 * \note Multiple FEMeshPartitioner objects may be instantiated simultaneously.
 */
class FESystemManager
{
public:
    /*!
     * \brief Constructor, where the FEData object owned by this class may be
     * co-owned by other objects.
     */
    FESystemManager(std::string object_name,
                    const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db,
                    std::shared_ptr<IBTK::FEData> fe_data,
                    std::string coords_sys_name);

    /*!
     * \brief The FEMeshPartitioner destructor cleans up any allocated data objects.
     */
    ~FESystemManager() = default;

    /*!
     * Alias FEData::SystemDofMapCache for backwards compatibility.
     */
    using SystemDofMapCache = IBTK::FEData::SystemDofMapCache;

    /*!
     * \brief The name of the equation system which stores the spatial position
     * data. The actual string is stored by FEData.
     *
     * \note The default value for this string is "coordinates system".
     */
    std::string COORDINATES_SYSTEM_NAME = "coordinates system";

    const std::string& getCoordsSystemName() const;

    /*!
     * \return A pointer to the equations systems object that is associated with
     * the FEData object.
     */
    libMesh::EquationSystems* getEquationSystems() const;

    /*!
     * \return The DofMapCache for a specified system.
     */
    SystemDofMapCache* getDofMapCache(const std::string& system_name);

    /*!
     * \return The DofMapCache for a specified system.
     */
    SystemDofMapCache* getDofMapCache(unsigned int system_num);

    /*!
     * \brief Enable or disable logging.
     *
     * @note This is usually set by the IBFEMethod which owns the current
     * FEMeshPartitioner, which reads the relevant boolean from the database.
     */
    void setLoggingEnabled(bool enable_logging = true);

    /*!
     * \brief Determine whether logging is enabled or disabled.
     */
    bool getLoggingEnabled() const;

    /*!
     * \return A pointer to the unghosted solution vector associated with the
     * specified system.
     */
    libMesh::NumericVector<double>* getSolutionVector(const std::string& system_name) const;

    /*!
     * \return A pointer to the unghosted coordinates (nodal position) vector.
     */
    libMesh::NumericVector<double>* getCoordsVector() const;

    /*!
     * \return The shared pointer to the object managing the Lagrangian data.
     */
    std::shared_ptr<IBTK::FEData>& getFEData();

    /*!
     * \return The shared pointer to the object managing the Lagrangian data.
     */
    const std::shared_ptr<IBTK::FEData>& getFEData() const;

    /*!
     * \return Pointer to vector representation of diagonal L2 mass matrix.
     */
    libMesh::NumericVector<double>* buildDiagonalL2MassMatrix(const std::string& system_name);

    /*!
     * \brief Set U to be the L2 projection of F.
     */
    bool computeL2Projection(libMesh::NumericVector<double>& U,
                             libMesh::NumericVector<double>& F,
                             const std::string& system_name,
                             bool consistent_mass_matrix = true,
                             bool close_U = true,
                             bool close_F = true,
                             double tol = 1.0e-6,
                             unsigned int max_its = 100);

    /*!
     * Clear all cached data that depends on the Eulerian data partitioning.
     *
     * This can include system dof caches (i.e. if one node moves from one processor to another).
     *
     * If this class is used with multiple patch hierarchies, too much use of this function could destroy efficiency.
     *
     * @note We may need to set up the caches to be dependent on which hierarchy is being used with this class. This may
     * require running a vector of FEData objects.
     */
    void clearPatchHierarchyDependentData();

protected:
    /*!
     * FEData object that contains the libMesh data structures.
     *
     * @note multiple FEMeshPartitioner objects may use the same FEData object,
     * usually combined with different hierarchies.
     */
    std::shared_ptr<IBTK::FEData> d_fe_data;

    /*!
     * FEProjector object that handles L2 projection functionality.
     */
    std::shared_ptr<IBTK::FEProjector> d_fe_projector;

    /*!
     * IB ghosted diagonal mass matrix representations.
     */
    std::map<std::string, std::unique_ptr<libMesh::PetscVector<double>>> d_L2_proj_matrix_diag_ghost;

private:
    /*!
     * Reinitialize IB ghosted DOF data structures for the specified system.
     */
    void reinitializeIBGhostedDOFs(const std::string& system_name);

    /*!
     * The object name is used as a handle to databases stored in restart files
     * and for error reporting purposes.
     */
    std::string d_object_name;

    /*!
     * Whether or not to log data to the screen: see
     * FEMeshPartitioner::setLoggingEnabled() and
     * FEMeshPartitioner::getLoggingEnabled().
     *
     * @note This is usually set by IBFEMethod, which reads the relevant
     * boolean from the database.
     */
    bool d_enable_logging = false;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_ADS_FESystemManager
