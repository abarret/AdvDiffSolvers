#ifndef included_ADS_SLAdvIntegrator
#define included_ADS_SLAdvIntegrator

#include "ibamr/config.h"

#include "ADS/AdvectiveReconstructionOperator.h"
#include "ADS/LSFindCellVolume.h"
#include "ADS/MLSReconstructCache.h"
#include "ADS/RBFReconstructCache.h"
#include "ADS/SBIntegrator.h"
#include "ADS/VolumeBoundaryMeshMapping.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "ibamr/AdvDiffHierarchyIntegrator.h"
#include "ibamr/LSInitStrategy.h"

#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/PoissonSolver.h"

namespace ADS
{
/*!
 * \brief Class SLAdvIntegrator is a concrete implementation of IBAMR::AdvDiffHierarchyIntegrator that provides an
 * interface for solvin the advection equation with a semi-Lagrangian method.
 *
 * Users should register an AdvectiveReconstructionOperator for each advected quantity that instructs the integrator how
 * to reconstruct the quantity at points on the grid. Currently, advected quantites must be associated with level sets,
 * so a level set variable and an LSFindVolume function must be registered.
 *
 * This class differs from LSAdvDiffIntegrator in that SLAdvIntegrator does not include a diffusion component. Calls to
 * setDiffusionCoefficient() result in an unrecoverable error. Additionally, SLAdvIntegrator associates all degrees of
 * freedom with cell centers, regardless of the level set value. This is in contrast to LSAdvDiffIntegrator, which
 * associates all degrees of freedom to cut cell centroids.
 *
 * An important parameter read from the input database is 'num_cycles.' Because IBAMR currently requires the advection
 * diffusion integrator to use the same number of cycles, this number needs to match the cycle number for the INS
 * integrator. Note this integrator doesn't do anything during a cycle: everything is done at the end of a timestep.
 */
class SLAdvIntegrator : public IBAMR::AdvDiffHierarchyIntegrator
{
public:
    SLAdvIntegrator(const std::string& object_name,
                    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                    bool register_for_restart = true);

    ~SLAdvIntegrator() = default;

    /*!
     * Register a cell-centered quantity to be advected and diffused by the
     * hierarchy integrator. Can optionally turn off outputting the quantity.
     *
     * Data management for the registered quantity will be handled by the
     * hierarchy integrator.
     */
    void registerTransportedQuantity(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                     const bool Q_output = true) override;

    /*!
     * Register a GeneralBoundaryMeshMapping with the hierarchy integrator. This is important if the level set function
     * computes using the boundary mesh.
     */
    virtual void registerGeneralBoundaryMeshMapping(const std::shared_ptr<GeneralBoundaryMeshMapping>& mesh_mapping);

    /*!
     * Register a level set variable. Advected quantities should be restricted to the level set by the call
     * restrictToLevelSet.
     */
    virtual void registerLevelSetVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * Register a level set volume function that can compute the level set and volume fraction on the patch hierarchy.
     *
     * If ls_var has not been previously registered, this results in a unrecoverable error when debugging is enabled.
     */
    virtual void registerLevelSetVolFunction(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                             SAMRAI::tbox::Pointer<LSFindCellVolume> vol_fcn);

    /*!
     * Restrict the advected quantity Q_var to the level set ls_var.
     *
     * If Q_var or ls_var has not been registered with the integrator, this function call results in an unrecoverable
     * error when debugging is enabled.
     */
    virtual void restrictToLevelSet(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * If use_ls_for_tagging is set to true, the provided level set variable will be used to tag grid cells for
     * refinement.
     */
    virtual void useLevelSetForTagging(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                       bool use_ls_for_tagging);

    /*!
     * This function call results in an unrecoverable error. Note that it masks a base class function that is not
     * virtual.
     */
    void setDiffusionCoefficient(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var, double D);

    /*!
     * Return the volume variable used by the integrator.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>
    getVolumeVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * Register an AdvectiveReconstructionOperator to be used to reconstruct Q_var. If this function is not called, the
     * default operator is used to construct Q_var.
     *
     * If Q_var has not been registered with the integrator, this function call results in an unrecoverable error when
     * debugging is enabled.
     */
    void registerAdvectionReconstruction(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                         std::shared_ptr<AdvectiveReconstructionOperator> reconstruct_op);

    /*!
     * Register an AdvectiveReconstructionOperator to be used to reconstruct the divergence of the velocity field. If
     * this function is not called, we use centered differences to compute div(u) and the default reconstruction
     * operator to reconstruct div(u).
     *
     * If u_var has not been registered with the integrator, this function call results in an unrecoverable error when
     * debugging is enabled.
     *
     * If u_var is registered, but the velocity is set to be divergence free, this operator is not used.
     *
     * Note this operator will be given a side centered velocity field, and expects the results stored in a cell
     * centered field.
     */
    void registerDivergenceReconstruction(SAMRAI::tbox::Pointer<SAMRAI::pdat::FaceVariable<NDIM, double>> u_var,
                                          std::shared_ptr<AdvectiveReconstructionOperator> reconstruct_op);

    /*!
     * Initialize the variables, basic communications algorithms, solvers, and
     * other data structures used by this time integrator object.
     *
     * This method is called automatically by initializePatchHierarchy() prior
     * to the construction of the patch hierarchy.  It is also possible for
     * users to make an explicit call to initializeHierarchyIntegrator() prior
     * to calling initializePatchHierarchy().
     */
    void
    initializeHierarchyIntegrator(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                  SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm<NDIM>> gridding_alg) override;

    /*!
     * Apply the gradient detector to determine which cells need to be refined. Refines cells based on the level set
     * value.
     *
     * \see useLevelSetForTagging
     */
    void applyGradientDetectorSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                          int ln,
                                          double error_data_time,
                                          int tag_index,
                                          bool initial_time,
                                          bool uses_richardson_extrapolation_too) override;

    /*!
     * Initialize the data on the hierarchy. This is called automatically upon hierarchy creation.
     */
    void initializeLevelDataSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                        int level_number,
                                        double init_data_time,
                                        bool can_be_refined,
                                        bool initial_time,
                                        SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<NDIM>> old_level,
                                        bool allocate_data) override;

    /*!
     * Returns the number of cycles to perform for the present time step.
     */
    int getNumberOfCycles() const override;

    /*!
     * Prepare to advance the data from current_time to new_time.
     */
    void preprocessIntegrateHierarchy(double current_time, double new_time, int num_cycles = 1) override;

    /*!
     * Synchronously advance each level in the hierarchy over the given time
     * increment.
     */
    void integrateHierarchy(double current_time, double new_time, int cycle_num = 0) override;

    /*!
     * Clean up data following call(s) to integrateHierarchy().
     */
    void postprocessIntegrateHierarchy(double current_time,
                                       double new_time,
                                       bool skip_synchronize_new_state_data,
                                       int num_cycles = 1) override;

protected:
    void initializeCompositeHierarchyDataSpecialized(double current_time, bool initial_time) override;
    void regridHierarchyEndSpecialized() override;
    void resetTimeDependentHierarchyDataSpecialized(double new_time) override;
    void
    resetHierarchyConfigurationSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> base_hierarchy,
                                           int coarsest_ln,
                                           int finest_ln) override;

    /*!
     * Perform the advection solve. Integrates particle paths backwards in time, then interpolates the solution to these
     * points.
     */
    virtual void advectionUpdate(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                 double current_time,
                                 double new_time);

    /*!
     * Integrates the paths backward in time using the provided velocity patch indices. If half_path_idx is a valid
     * patch index, then the departure points are computed at half time points as well.
     */
    virtual void integratePaths(int path_idx, int half_path_idx, int u_new_idx, int u_half_idx, double dt);

    void setDefaultReconstructionOperator(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var);

    SAMRAI::hier::ComponentSelector d_adv_data;

    // Level set information
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>> d_ls_vars;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>> d_vol_vars;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>, bool> d_ls_use_ls_for_tagging;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>,
             SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>>
        d_Q_ls_map;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>, SAMRAI::tbox::Pointer<LSFindCellVolume>>
        d_ls_vol_fcn_map;

    std::shared_ptr<GeneralBoundaryMeshMapping> d_mesh_mapping;

    // Advection reconstruction information
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>,
             std::shared_ptr<AdvectiveReconstructionOperator>>
        d_Q_adv_reconstruct_map;
    AdvReconstructType d_default_adv_reconstruct_type = AdvReconstructType::RBF;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_u_s_var;

    // Divergence variable for compressible velocity fields
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_u_div_var;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::FaceVariable<NDIM, double>>,
             std::shared_ptr<AdvectiveReconstructionOperator>>
        d_u_div_adv_ops_map;

    // patch data for particle trajectories
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_path_var;
    int d_path_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_half_path_var;
    int d_half_path_idx = IBTK::invalid_index;
    // Scratch data for when we need more ghost cells.
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_big_scr_var;
    int d_Q_big_scr_idx = IBTK::invalid_index;

    SAMRAI::hier::ComponentSelector d_ls_data;

    static int GHOST_CELL_WIDTH;

    double d_min_ls_refine_factor = std::numeric_limits<double>::quiet_NaN();
    double d_max_ls_refine_factor = std::numeric_limits<double>::quiet_NaN();
    Reconstruct::LeastSquaresOrder d_least_squares_reconstruction_order = Reconstruct::LeastSquaresOrder::UNKNOWN_ORDER;
    AdvectionTimeIntegrationMethod d_adv_ts_type = AdvectionTimeIntegrationMethod::UNKNOWN_METHOD;

    SAMRAI::tbox::Pointer<SAMRAI::math::HierarchyFaceDataOpsReal<NDIM, double>> d_hier_fc_data_ops;

private:
    // Two is a good number for the current default INSStaggeredHierarchyIntegrator.
    int d_num_cycles = 2;
    bool d_use_rbfs = false;
    unsigned int d_rbf_stencil_size = 8;
    unsigned int d_mls_stencil_size = 8;
    Reconstruct::RBFPolyOrder d_rbf_poly_order = Reconstruct::RBFPolyOrder::UNKNOWN_ORDER;
}; // Class SLAdvIntegrator
} // namespace ADS

#endif
