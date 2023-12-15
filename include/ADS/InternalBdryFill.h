#ifndef included_ADS_InternalBdryFill
#define included_ADS_InternalBdryFill
#include <ibtk/ibtk_utilities.h>

#include <PatchHierarchy.h>
#include <VisItDataWriter.h>

namespace ADS
{
/*!
 * Class ReinitailizeLevelSet generates a signed distance function from a level set using a fast sweeping algorithm to
 * solve the Eikonal equation.
 *
 * This class uses routines based on those found in FastSweepingLSMethod in IBAMR.
 */
class InternalBdryFill
{
public:
    InternalBdryFill(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * Deleted default constructor.
     */
    InternalBdryFill() = delete;

    ~InternalBdryFill();

    /*!
     * Advect the given concentration in the normal direction. Needs a signed distance function
     */
    void advectInNormal(int Q_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                        int phi_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        double time);

private:
    void fillNormal(int phi_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    std::string d_object_name;
    double d_tol = 1.0e-5;
    int d_max_iters = 500;
    bool d_enable_logging = true;
    bool d_error_on_non_convergence = true;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_sc_var;
    int d_sc_idx = IBTK::invalid_index;
};
} // namespace ADS
#endif
