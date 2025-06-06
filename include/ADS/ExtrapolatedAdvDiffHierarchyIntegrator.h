#ifndef included_ADS_ExtrapolatedAdvDiffHierarchyIntegrator
#define included_ADS_ExtrapolatedAdvDiffHierarchyIntegrator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include <ADS/FEToHierarchyMapping.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/LSFindCellVolume.h>

#include "ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h"
#include "ibamr/ibamr_utilities.h"

#include <tbox/Database.h>
#include <tbox/Pointer.h>

#include <CellVariable.h>

#include <map>
#include <set>
#include <string>
/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class ExtrapolatedAdvDiffHierarchyIntegrator is a specialization of the class
 * AdvDiffSemiImplicitHierarchyIntegrator.
 *
 * This class extrapolates concentration fields in the normal direction across
 * immersed boundaries by solving an advection equation in which the velocity is given by the normal of the signed
 * distance function. In preprocessIntegrateHierarchy(), this class computes the level set corresponding to the immersed
 * boundary, converts the level set into a signed distance, and advects concentrations using the normal field. During
 * postprocessIntegrateHierarchy(), this class zeros values that appear on the unphysical side of the boundary.
 *
 * \Note This class should not be used with level sets that touch the physical boundary. Ghost cells are not currently
 * set in a consistent way at physical boundaries.
 *
 */
class ExtrapolatedAdvDiffHierarchyIntegrator : public IBAMR::AdvDiffSemiImplicitHierarchyIntegrator
{
public:
    /*!
     * The constructor for class ExtrapolatedAdvDiffHierarchyIntegrator sets
     * some default values, reads in configuration information from input and
     * restart databases, and registers the integrator object with the restart
     * manager when requested.
     *
     * The input database is also searched for "num_cells_to_extrap," which is the number of cells in which
     * concentrations will be extrapolated to. By default, this value is 2.
     */
    ExtrapolatedAdvDiffHierarchyIntegrator(const std::string& object_name,
                                           SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                                           bool register_for_restart = true);

    ~ExtrapolatedAdvDiffHierarchyIntegrator() = default;

    /*!
     * \brief Registers a mesh mapping object with the integrator. This is used to determine the location of the
     * structure before the level set is calculated.
     */
    void setMeshMapping(std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping);

    /*!
     * Return pointers to the FEToHierarchyMapping objects used by this class. Note that these are not set-up until the
     * hierarchy is created, so this returns an empty vector prior to initializePatchHierarchy() is called.
     */
    std::vector<FEToHierarchyMapping*> getFEHierarchyMappings();

    /*!
     * \brief Register an advected concentration field. Can also set the default reset value for unphysical cell
     * indices.
     *
     * If the reset_val is a NaN, then no reset is done and the transported quantity is advected on both sides of the
     * interface.
     */
    void registerTransportedQuantity(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                     double reset_val = std::numeric_limits<double>::quiet_NaN(),
                                     bool output_var = true);
    using IBAMR::AdvDiffSemiImplicitHierarchyIntegrator::registerTransportedQuantity;

    /*!
     * \brief Set the reset value for the specified advected variable.
     *
     * Note that this provides an alternative way to set the reset value after a variable has been registered for
     * transport. An error occurs if the variable is not already registered with the integrator.
     */
    void setResetValue(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var, double reset_val);

    /*!
     * \brief Registers a node centered level set variable with the integrator. A level set function must be supplied
     * that can compute the level set.
     *
     * Concentration fields can be restricted to this level set by the call restrictToLevelSet().
     */
    void registerLevelSetVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                  SAMRAI::tbox::Pointer<LSFindCellVolume> ls_fcn);

    /*!
     * \brief Restricts a given advected concentration to the provided level set. The level set must be registered prior
     * to calling this function.
     */
    void restrictToLevelSet(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                            SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * \brief Initialize this object for use by creating all items needed by the integrator.
     *
     * All registering of objects must be done before this function is called. This function is automatically called by
     * initializePatchHierarchy().
     */
    void
    initializeHierarchyIntegrator(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                  SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm<NDIM>> gridding_alg) override;

    /*!
     * Prepare to advance the data from current_time to new_time.
     *
     * Extrapolates concentrations fields in the normal direction according to their restricted level set.
     */
    void preprocessIntegrateHierarchy(double current_time, double new_time, int num_cycles = 1) override;

    /*!
     * Perform advection and diffusion of integrated quantities.
     *
     * Effectively the same as AdvDiffSemiImplicitHierarchyIntegrator, but uses extrapolated quantities to integrate.
     */
    void integrateHierarchySpecialized(double current_time, double new_time, int cycle_num = 0) override;

    /*!
     * Clean up data following call(s) to integrateHierarchy().
     *
     * Zeros out data in concentration fields outside of their physical regime (where the level set is positive).
     */
    void postprocessIntegrateHierarchy(double current_time,
                                       double new_time,
                                       bool skip_synchronize_new_state_data,
                                       int num_cycles = 1) override;

protected:
    void initializeCompositeHierarchyDataSpecialized(double current_time, bool initial_time) override;

private:
    // List of level set variables.
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>> d_ls_vars;

    // Map between the level set variable and the list of concentration fields that are restricted to that level set.
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>,
             std::set<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>>
        d_ls_Q_map;

    // Map between level set variable and the function that computes the level set.
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>, SAMRAI::tbox::Pointer<LSFindCellVolume>>
        d_ls_fcn_map;

    // Scratch variable for determining indices where signed distance function is needed.
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, int>> d_valid_var;
    int d_valid_idx = IBTK::invalid_index;
    std::shared_ptr<GeneralBoundaryMeshMapping> d_mesh_mapping;
    std::vector<std::unique_ptr<FEToHierarchyMapping>> d_fe_hierarchy_mapping;

    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>, bool> d_Q_reset_map;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>, double> d_Q_reset_val_map;

    // Number of cells to fill in level set. This controls the number of cells that concentrations are extrapolated.
    int d_num_cells_to_extrap = 2;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_IBAMR_ExtrapolatedAdvDiffHierarchyIntegrator
