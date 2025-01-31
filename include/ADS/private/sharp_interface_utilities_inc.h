#ifndef included_ADS_sharp_interface_utilities_inc
#define included_ADS_sharp_interface_utilities_inc

#include <ADS/sharp_interface_utilities.h>

#include <CartesianPatchGeometry.h>

namespace ADS
{
namespace sharp_interface
{
inline void
apply_laplace_operator_on_patch(SAMRAI::hier::Patch<NDIM>& patch,
                                SAMRAI::pdat::CellData<NDIM, double>& u_data,
                                SAMRAI::pdat::CellData<NDIM, double>& b_data,
                                SAMRAI::pdat::CellData<NDIM, int>& i_data,
                                const ImagePointWeightsMap& img_wgts,
                                const std::vector<ImagePointData>& img_data_vec,
                                std::function<double(const IBTK::VectorNd& x)> bdry_fcn)
{
    SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch.getPatchGeometry();
    const double* const dx = pgeom->getDx();
    for (SAMRAI::pdat::CellIterator<NDIM> ci(patch.getBox()); ci; ci++)
    {
        const SAMRAI::pdat::CellIndex<NDIM>& idx = ci();
        const int idx_val = i_data(idx);
        if (idx_val == FLUID)
        {
            for (int d = 0; d < NDIM; ++d)
            {
                SAMRAI::hier::IntVector<NDIM> one(0);
                one(d) = 1;
                b_data(idx) += (u_data(idx + one) - 2.0 * u_data(idx) + u_data(idx - one)) / (dx[d] * dx[d]);
            }
        }
        else if (idx_val == INVALID)
        {
            b_data(idx) = 0.0;
        }
    }

    // Now boundary conditions
    for (const auto& img_data : img_data_vec)
    {
        const SAMRAI::pdat::CellIndex<NDIM>& gp_idx = img_data.d_gp_idx;
        b_data(gp_idx) = u_data(gp_idx);
        auto gp_patch_pair = std::make_pair(gp_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>(&patch, false));
        for (unsigned int i = 0; i < ImagePointWeights::s_num_pts; ++i)
        {
            const SAMRAI::pdat::CellIndex<NDIM>& idx = img_wgts.at(gp_patch_pair).d_idxs[i];
            const double wgt = img_wgts.at(gp_patch_pair).d_weights[i];
            b_data(gp_idx) += u_data(idx) * wgt;
        }
        const IBTK::VectorNd& bp_loc = img_data.d_bp_location;
        b_data(gp_idx) = b_data(gp_idx) - 2.0 * bdry_fcn(bp_loc);
    }
}
} // namespace sharp_interface
} // namespace ADS

#endif
