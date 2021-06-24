// ---------------------------------------------------------------------
//
// Copyright (c) 2011 - 2021 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_LS_FEMeshPartitioner
#define included_LS_FEMeshPartitioner

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/FEDataManager.h"
#include "ibtk/ibtk_enums.h"
#include "ibtk/ibtk_utilities.h"
#include <ibtk/config.h>

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

namespace LS
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
class FEMeshPartitioner
{
public:
    /*!
     * \brief Constructor, where the FEData object owned by this class may be
     * co-owned by other objects.
     */
    FEMeshPartitioner(std::string object_name,
                      const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db,
                      const int max_levels,
                      SAMRAI::hier::IntVector<NDIM> ghost_width,
                      std::shared_ptr<IBTK::FEData> fe_data,
                      std::string coords_sys_name);

    /*!
     * \brief The FEMeshPartitioner destructor cleans up any allocated data objects.
     */
    ~FEMeshPartitioner() = default;

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
     * \name Methods to set and get the patch hierarchy and range of patch
     * levels associated with this manager class.
     */
    //\{

    /*!
     * \brief Reset patch hierarchy over which operations occur.
     *
     * The patch hierarchy must be fully set up (i.e., contain all the levels it
     * is expected to have) at the point this function is called. If you need to
     * tag cells for refinement to create the initial hierarchy then use
     * applyGradientDetector, which does not use the stored patch hierarchy.
     */
    void setPatchHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    /*!
     * \brief Get the patch hierarchy used by this object.
     */
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> getPatchHierarchy() const;

    /*!
     * Get the coarsest patch level number on which elements are assigned.
     */
    int getCoarsestPatchLevelNumber() const;

    /*!
     * Get the finest patch level number on which elements are assigned.
     */
    int getFinestPatchLevelNumber() const;

    /*!
     * \return The ghost cell width used for quantities that are to be
     * interpolated from the Cartesian grid to the FE mesh.
     */
    const SAMRAI::hier::IntVector<NDIM>& getGhostCellWidth() const;

    /*!
     * \return A const reference to the map from local patch number to local
     * active elements.
     */
    const std::vector<std::vector<libMesh::Elem*>>& getActivePatchElementMap() const;

    /*!
     * \return A const reference to the map from local patch number to local
     * active nodes.
     *
     * \note The local active nodes are the nodes of the local active elements.
     */
    const std::vector<std::vector<libMesh::Node*>>& getActivePatchNodeMap() const;

    /*!
     * \brief Reinitialize the mappings from elements to Cartesian grid patches.
     */
    void reinitElementMappings();

    /*!
     * \return A pointer to the unghosted solution vector associated with the
     * specified system.
     */
    libMesh::NumericVector<double>* getSolutionVector(const std::string& system_name) const;

    /*!
     * \return A pointer to the ghosted solution vector associated with the
     * specified system. The vector contains positions for values in the
     * relevant IB ghost region which are populated if @p localize_data is
     * <code>true</code>.
     *
     * @note The vector returned by pointer is owned by this class (i.e., no
     * copying is done).
     *
     * @deprecated Use buildIBGhostedVector() instead which clones a vector
     * with the same ghost region.
     */
    libMesh::NumericVector<double>* buildGhostedSolutionVector(const std::string& system_name,
                                                               bool localize_data = true);

    /*!
     * \return A pointer to the unghosted coordinates (nodal position) vector.
     */
    libMesh::NumericVector<double>* getCoordsVector() const;

    /*!
     * \return A pointer to the ghosted coordinates (nodal position) vector.
     *
     * @deprecated Use buildIBGhostedVector() instead.
     */
    libMesh::NumericVector<double>* buildGhostedCoordsVector(bool localize_data = true);

    /*!
     * \return The shared pointer to the object managing the Lagrangian data.
     */
    std::shared_ptr<IBTK::FEData>& getFEData();

    /*!
     * \return The shared pointer to the object managing the Lagrangian data.
     */
    const std::shared_ptr<IBTK::FEData>& getFEData() const;

protected:
    /*!
     * FEData object that contains the libMesh data structures.
     *
     * @note multiple FEMeshPartitioner objects may use the same FEData object,
     * usually combined with different hierarchies.
     */
    std::shared_ptr<IBTK::FEData> d_fe_data;

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    FEMeshPartitioner() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    FEMeshPartitioner(const FEMeshPartitioner& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    FEMeshPartitioner& operator=(const FEMeshPartitioner& that) = delete;

    /*!
     * Collect all of the active elements which are located within a local
     * Cartesian grid patch grown by a ghost width of 1 (like
     * IBTK::LEInteractor::getMinimumGhostWidth(), we assume that IB points
     * are allowed to move no more than one cell width between regridding
     * operations).
     *
     * The parameters refer to the levels of different objects:
     * <ol>
     *   <li>@p level_number - the level number in the patch hierarchy on which
     *     we are identifying intersections.</li>
     *   <li>@p coarsest_elem_ln - The minimum level number of elements we should
     *     consider (see the main documentation of this class for an explanation
     *     on how elements are assigned to particular levels)</li>
     *   <li>@p finest_elem_ln - The maximum level number of elements we should
     *     consider.</li>
     * </ol>
     *
     * All three parameters are necessary because we use this function both to
     * tag cells for refinement (i.e., we want to refine cells containing
     * elements on levels higher than the present level) and to do IB
     * calculations (where all three numbers will be the same).
     *
     * In this method, the determination as to whether an element is local or
     * not is based on the position of the bounding box of the element.
     */
    void collectActivePatchElements(std::vector<std::vector<libMesh::Elem*>>& active_patch_elems,
                                    int level_number,
                                    int coarsest_elem_ln,
                                    int finest_elem_ln);

    /*!
     * Collect all of the nodes of the active elements that are located within a
     * local Cartesian grid patch grown by the specified ghost cell width.
     */
    void collectActivePatchNodes(std::vector<std::vector<libMesh::Node*>>& active_patch_nodes,
                                 const std::vector<std::vector<libMesh::Elem*>>& active_patch_elems);

    /*!
     * Store the association between subdomain ids and patch levels.
     */
    IBTK::SubdomainToPatchLevelTranslation d_level_lookup;

    /*!
     * Get the patch level on which an element lives.
     */
    int getPatchLevel(const libMesh::Elem* elem) const;

    /*!
     * Collect all ghost DOF indices for the specified collection of elements.
     */
    void collectGhostDOFIndices(std::vector<unsigned int>& ghost_dofs,
                                const std::vector<libMesh::Elem*>& active_elems,
                                const std::string& system_name);

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

    /*!
     * Grid hierarchy information.
     */
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    /*!
     * Maximum possible level number in the patch hierarchy.
     */
    int d_max_level_number = IBTK::invalid_level_number;

    /*!
     * after reassociating patches with elements a node may still lie
     * outside all patches on the finest level in unusual circumstances
     * (like when the parent integrator class does not regrid sufficiently
     * frequently and has more than one patch level). This enum controls
     * what we do when this problem is detected.
     */
    IBTK::NodeOutsidePatchCheckType d_node_patch_check = IBTK::NODE_OUTSIDE_ERROR;

    /*!
     * SAMRAI::hier::IntVector object which determines the required ghost cell
     * width of this class.
     */
    const SAMRAI::hier::IntVector<NDIM> d_ghost_width;

    /*!
     * SAMRAI::hier::IntVector object which determines how many ghost cells we
     * should enlarge a patch by when associating an element with a patch. An
     * element is associated with a patch when its bounding box (defined as
     * the bounding box of both its nodes and quadrature points) intersects
     * the bounding box (including ghost cells) of that patch.
     *
     * @note At the present time this is always 1, which matches the
     * assumption made by IBTK::LEInteractor::getMinimumGhostWidth().
     */
    const SAMRAI::hier::IntVector<NDIM> d_associated_elem_ghost_width = SAMRAI::hier::IntVector<NDIM>(1);

    /*!
     * Data to manage mappings between mesh elements and grid patches.
     */
    std::vector<std::vector<std::vector<libMesh::Elem*>>> d_active_patch_elem_map;
    std::vector<std::vector<std::vector<libMesh::Node*>>> d_active_patch_node_map;
    std::map<std::string, std::vector<unsigned int>> d_active_patch_ghost_dofs;
    std::vector<libMesh::Elem*> d_active_elems;

    /*!
     * Ghost vectors for the various equation systems.
     */
    std::map<std::string, std::unique_ptr<libMesh::NumericVector<double>>> d_system_ghost_vec;

    /*!
     * Exemplar relevant IB-ghosted vectors for the various equation
     * systems. These vectors are cloned for fast initialization in
     * buildIBGhostedVector.
     */
    std::map<std::string, std::unique_ptr<libMesh::PetscVector<double>>> d_system_ib_ghost_vec;
};
} // namespace LS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_LS_FEMeshPartitioner
