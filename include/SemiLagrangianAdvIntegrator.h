

#include "IBAMR_config.h"

#include "ibamr/AdvDiffHierarchyIntegrator.h"

#include "LSFindCellVolume.h"
#include "SetLSValue.h"

namespace IBAMR
{
class SemiLagrangianAdvIntegrator : public AdvDiffHierarchyIntegrator
{
public:
    SemiLagrangianAdvIntegrator(const std::string& object_name,
                                SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                                bool register_for_restart = false);

    ~SemiLagrangianAdvIntegrator() = default;

    /*!
     * Register a cell-centered quantity to be advected and diffused by the
     * hierarchy integrator. Can optionally turn off outputting the quantity.
     *
     * Data management for the registered quantity will be handled by the
     * hierarchy integrator.
     */
    void registerTransportedQuantity(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                     const bool Q_output = true) override;

    void registerLevelSetFunction(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                  SAMRAI::tbox::Pointer<SetLSValue> ls_fcn);

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

    void initializeLevelDataSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                        int level_number,
                                        double init_data_time,
                                        bool can_be_refined,
                                        bool initial_time,
                                        SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<NDIM>> old_level,
                                        bool allocate_data) override;

    void applyGradientDetectorSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                          int ln,
                                          double error_data_time,
                                          int tag_index,
                                          bool initial_time,
                                          bool uses_richardson_extrapolation_too) override;

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
    void regridHierarchyEndSpecialized() override;

    void setupPlotDataSpecialized() override;

private:
    void integratePaths(int path_idx, int u_idx, double dt);

    void invertMapping(int path_idx, int xstar_idx);

    void evaluateMappingOnHierarchy(int xstar_idx, int Q_cur_idx, int Q_new_idx, int vol_idx, int order);

    void fillNormalCells(int Q_idx, int Q_scr_idx, int ls_idx);

    void findLSNormal(int ls_idx, int ls_n_idx);

    double sumOverZSplines(const IBTK::VectorNd& x_loc,
                           const SAMRAI::pdat::CellIndex<NDIM>& idx,
                           const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                           const int order);
    inline double evaluateZSpline(const IBTK::VectorNd x, const int order)
    {
        double val = 1.0;
        for (int d = 0; d < NDIM; ++d)
        {
            val *= ZSpline(x(d), order);
        }
        return val;
    }

    inline int getSplineWidth(const int order)
    {
        return order + 1;
    }

    inline double ZSpline(double x, const int order)
    {
        x = abs(x);
        switch (order)
        {
        case 0:
            if (x < 1.0)
                return 1.0 - x;
            else
                return 0.0;
        case 1:
            if (x < 1.0)
                return 1.0 - 2.5 * x * x + 1.5 * x * x * x;
            else if (x < 2.0)
                return 0.5 * (2.0 - x) * (2.0 - x) * (1.0 - x);
            else
                return 0.0;
        case 2:
            if (x < 1.0)
                return 1.0 - 15.0 / 12.0 * x * x - 35.0 / 12.0 * x * x * x + 63.0 / 12.0 * x * x * x * x -
                       25.0 / 12.0 * x * x * x * x * x;
            else if (x < 2.0)
                return -4.0 + 75.0 / 4.0 * x - 245.0 / 8.0 * x * x + 545.0 / 24.0 * x * x * x -
                       63.0 / 8.0 * x * x * x * x + 25.0 / 24.0 * x * x * x * x * x;
            else if (x < 3.0)
                return 18.0 - 153.0 / 4.0 * x + 255.0 / 8.0 * x * x - 313.0 / 24.0 * x * x * x +
                       21.0 / 8.0 * x * x * x * x - 5.0 / 24.0 * x * x * x * x * x;
            else
                return 0.0;
        default:
            TBOX_ERROR("Unavailable order: " << order << "\n");
            return 0.0;
        }
    }

    // patch data for particle trajectories
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_path_var;
    int d_path_idx = IBTK::invalid_index;
    int d_xstar_idx = IBTK::invalid_index;

    int d_Q_scratch_idx = IBTK::invalid_index;

    // Level set information
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_ls_var;
    int d_ls_cur_idx = IBTK::invalid_index, d_ls_new_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_vol_var;
    int d_vol_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::FaceVariable<NDIM, double>> d_ls_normal_var;
    int d_ls_normal_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SetLSValue> d_ls_fcn;
    SAMRAI::tbox::Pointer<LSFindCellVolume> d_vol_fcn;

    bool d_using_forward_integration = false;

}; // Class SemiLagrangianAdvIntegrator
} // Namespace IBAMR
