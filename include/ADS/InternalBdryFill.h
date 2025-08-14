#ifndef included_ADS_InternalBdryFill
#define included_ADS_InternalBdryFill
#include <ibtk/ibtk_utilities.h>

#include <PatchHierarchy.h>
#include <RobinBcCoefStrategy.h>

namespace ADS
{
/*!
 * Class InternalBdryFill fills in cells in "unphysical" regimes with values that extrapolated by performing constant
 * advection in the normal direction. The normal direction is computed from a signed distance function. The advection
 * step is done via simple upwinding, with values in the "physical" regime remaining constant. In effect, this performs
 * a normal constant extrapolation.
 *
 * If this class fails to find a steady state, visualization files will be written. Check the log file for the specific
 * folder name.
 *
 * Note: Care must be taken when using this class with level sets near the physical boundary. Ghost cells can easily be
 * filled in ways that are inconsistent with this class.
 */
class InternalBdryFill
{
public:
    /*!
     * The constructor reads the optional input database to set the following values:
     * -- tolerance: The advection is marked as completed if the difference between solutions in successive steps is
     * less than the tolerance. Default value is 1.0e-5.
     * -- max_iterations: The maximum number of pseudo-time integration steps. Default value is 1000.
     * -- enable_logging: If true, this class prints data about convergence to the log file. Default value is true.
     * -- max_gcw: The maximum width of cells to fill. Default value is std::numeric_limits<int>::max(). Note that if
     * the largest grid cell spacing is greater than one, overflow may occur which could potentially slow down
     * computations.
     */
    InternalBdryFill(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db = nullptr);

    /*!
     * Deleted default constructor.
     */
    InternalBdryFill() = delete;

    /*!
     * Destructor removes scratch patch indices from the VariableDatabase.
     */
    ~InternalBdryFill() = default;

    /*!
     * Helper struct that contains data required for advectInNormal to work correctly.
     *
     * If negative_inside = true, then where the level set is negative is treated as the interior.
     */
    struct Parameters
    {
        Parameters(int Q_idx,
                   SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                   bool negative_inside = true)
            : Q_idx(Q_idx), Q_var(Q_var), negative_inside(negative_inside)
        {
        }
        int Q_idx;
        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var;
        bool negative_inside = true;
    };

    /*!
     * Advect each concentration in the normal direction. Computes the normal direction from the signed distance
     * function in phi_idx.
     *
     * If negative_inside is false, advectInNormal will fill in values where the level set phi_idx is positive.
     */
    void advectInNormal(const std::vector<Parameters>& Q_params,
                        int phi_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        double time);

    /*!
     * Advect the given concentration in the normal direction. Computes the normal direction from the signed distance
     * function in phi_idx.
     *
     * If negative_inside is false, advectInNormal will fill in values where the level set phi_idx is positive.
     */
    void advectInNormal(const Parameters& Q_params,
                        int phi_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        double time);

private:
    /*!
     * Compute the normal from the provided signed distance function. Stores the normal in d_sc_idx.
     */
    void fillNormal(int phi_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy, double time);

    /*!
     * Advect the concentration field in the normal direction. This assumes the correct velocity is stored in d_sc_idx.
     */
    bool doAdvectInNormal(const Parameters& Q_param,
                          int phi_idx,
                          SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          double time);

    /*!
     * Write visualization files of the advected quantities.
     */
    void writeVizFiles(const std::vector<Parameters>& Q_params,
                       int phi_idx,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       double time,
                       const int inter_num);

    std::string d_object_name;
    double d_tol = 1.0e-5;
    int d_max_iters = 1000;
    bool d_enable_logging = true;
    bool d_error_on_non_convergence = true;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_sc_var;
    int d_sc_idx = IBTK::invalid_index;

    unsigned int d_max_gcw = std::numeric_limits<int>::max();
};
} // namespace ADS
#endif
