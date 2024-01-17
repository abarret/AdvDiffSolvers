#ifndef included_ADS_InternalBdryFill
#define included_ADS_InternalBdryFill
#include <ibtk/ibtk_utilities.h>

#include <PatchHierarchy.h>
#include <VisItDataWriter.h>

namespace ADS
{
/*!
 * Class InternalBdryFill fills in cells in "unphysical" regimes with values that extrapolated by performing constant
 * advection in the normal direction. The normal direction is computed from a signed distance function. The advection
 * step is done via simple upwinding, with values in the "physical" regime remaining constant. In effect, this performs
 * a normal constant extrapolation.
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
     */
    InternalBdryFill(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db = nullptr);

    /*!
     * Deleted default constructor.
     */
    InternalBdryFill() = delete;

    /*!
     * Destructor removes scratch patch indices from the VariableDatabase.
     */
    ~InternalBdryFill();

    /*!
     * Advect each concentration in the normal direction. Computes the normal direction from the signed distance
     * function in phi_idx.
     */
    void advectInNormal(
        const std::vector<std::pair<int, SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>>& Q_vars,
        int phi_idx,
        SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
        double time);
    /*!
     * Advect the given concentration in the normal direction. Computes the normal direction from the signed distance
     * function in phi_idx.
     */
    void advectInNormal(int Q_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
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
    void doAdvectInNormal(int Q_idx,
                          SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                          int phi_idx,
                          SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          double time);

    std::string d_object_name;
    double d_tol = 1.0e-5;
    int d_max_iters = 1000;
    bool d_enable_logging = true;
    bool d_error_on_non_convergence = false;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_sc_var;
    int d_sc_idx = IBTK::invalid_index;
};
} // namespace ADS
#endif
