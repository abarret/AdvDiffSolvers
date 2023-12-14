#ifndef included_ADS_ReinitializeLevelSet
#define included_ADS_ReinitializeLevelSet
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
class ReinitializeLevelSet
{
public:
    ReinitializeLevelSet(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * Deleted default constructor.
     */
    ReinitializeLevelSet() = delete;

    ~ReinitializeLevelSet();

    /*!
     * \brief Compute the signed distance function phi_idx. Assumes the initial level set is provided in phi_idx.
     *
     * The input fixed_idx should correspond to a node centered, integer valued quantity. The value on a specified index
     * informs the method whether to adjust the signed distance function:
     *   -- A value of 0 means the signed distance function should be changed on this cell.
     *   -- A value of 1 means the signed distance function should not be changed but can be used to compute
     * derivatives. This should be used to indicate correct signed distance values.
     *   -- A value of 2 means the signed distance function should not be changed and should not be used to compute
     * derivatives. This should be used to indicate that the signed distance function is not needed on this node.
     *
     *
     * phi_idx and fixed_idx must have at least one layer of ghost cells and should correspond to node centered data.
     *
     * TODO This function fills in physical boundary conditions using extrapolation.
     */
    void computeSignedDistanceFunction(int phi_idx,
                                       SAMRAI::pdat::NodeVariable<NDIM, double>& phi_var,
                                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                       double time,
                                       int fixed_idx);

    /*!
     * \brief Compute the signed distance function phi_idx. Assumes the initial level set is provided in phi_idx.
     *
     * This function assumes incorrect signed distance functions are EXACTLY equal to the (signed) value of
     * value_to_be_changed (note that exact equality is checked, not floating point equality). This will compute the
     * signed distance function in all cells that are exactly equal to the signed value of value_to_be_changed. All
     * other values are assumed to be fixed and correct.
     *
     * phi_idx must have at least one layer of ghost cells and should correspond to node centered data.
     *
     * TODO This function fills in physical boundary conditions using extrapolation
     */
    void computeSignedDistanceFunction(int phi_idx,
                                       SAMRAI::pdat::NodeVariable<NDIM, double>& phi_var,
                                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                       double time,
                                       double value_to_be_changed);

private:
    void doSweep(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy, int phi_idx, int fixed_idx);

    void doSweepOnPatch(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, int phi_idx, int fixed_idx);

    std::string d_object_name;
    double d_tol = 1.0e-3;
    int d_max_iters = 100;
    bool d_enable_logging = true;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, int>> d_nc_var;
    int d_nc_idx = IBTK::invalid_index;
};
} // namespace ADS
#endif
