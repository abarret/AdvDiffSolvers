#ifndef included_ADS_reconstructions
#define included_ADS_reconstructions

#include <ibamr/config.h>

#include "ibtk/ibtk_utilities.h"

#include "CellData.h"
#include "CellIndex.h"
#include "NodeData.h"

namespace Reconstruct
{
double sumOverZSplines(const IBTK::VectorNd& x_loc,
                       const SAMRAI::pdat::CellIndex<NDIM>& idx,
                       const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                       const int order);

bool indexWithinWidth(int stencil_width,
                      const SAMRAI::pdat::CellIndex<NDIM>& idx,
                      const SAMRAI::pdat::CellData<NDIM, double>& vol_data);

double evaluateZSpline(const IBTK::VectorNd x, const int order);

int getSplineWidth(int order);
double ZSpline(double x, int order);

/*!
 * \brief Routine for converting strings to enums.
 */
template <typename T>
inline T
string_to_enum(const std::string& /*val*/)
{
    TBOX_ERROR("UNSUPPORTED ENUM TYPE\n");
    return -1;
}

/*!
 * \brief Routine for converting enums to strings.
 */
template <typename T>
inline std::string
enum_to_string(T /*val*/)
{
    TBOX_ERROR("UNSUPPORTED ENUM TYPE\n");
    return "UNKNOWN";
}

enum class LeastSquaresOrder
{
    CONSTANT,
    LINEAR,
    QUADRATIC,
    CUBIC,
    UNKNOWN_ORDER = -1
};

template <>
inline LeastSquaresOrder
string_to_enum<LeastSquaresOrder>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "CONSTANT") == 0) return LeastSquaresOrder::CONSTANT;
    if (strcasecmp(val.c_str(), "LINEAR") == 0) return LeastSquaresOrder::LINEAR;
    if (strcasecmp(val.c_str(), "QUADRATIC") == 0) return LeastSquaresOrder::QUADRATIC;
    if (strcasecmp(val.c_str(), "CUBIC") == 0) return LeastSquaresOrder::CUBIC;
    return LeastSquaresOrder::UNKNOWN_ORDER;
}

template <>
inline std::string
enum_to_string<LeastSquaresOrder>(LeastSquaresOrder val)
{
    if (val == LeastSquaresOrder::CONSTANT) return "CONSTANT";
    if (val == LeastSquaresOrder::LINEAR) return "LINEAR";
    if (val == LeastSquaresOrder::QUADRATIC) return "QUADRATIC";
    if (val == LeastSquaresOrder::CUBIC) return "CUBIC";
    return "UNKNOWN_ORDER";
}

enum class RBFPolyOrder
{
    LINEAR,
    QUADRATIC,
    UNKNOWN_ORDER
};

template <>
inline RBFPolyOrder
string_to_enum<RBFPolyOrder>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "LINEAR") == 0) return RBFPolyOrder::LINEAR;
    if (strcasecmp(val.c_str(), "QUADRATIC") == 0) return RBFPolyOrder::QUADRATIC;
    return RBFPolyOrder::UNKNOWN_ORDER;
}

template <>
inline std::string
enum_to_string<RBFPolyOrder>(RBFPolyOrder val)
{
    if (val == RBFPolyOrder::LINEAR) return "LINEAR";
    if (val == RBFPolyOrder::QUADRATIC) return "QUADRATIC";
    return "UNKNOWN_ORDER";
}

/*!
 * \brief Polyharmonic rbf.
 */
static inline double
rbf(double r)
{
    return r;
}

/*!
 * \brief MLS weight function: Gaussian-like kernel.
 */
static inline double
mls_weight(double r)
{
    return std::exp(-r * r);
}

/*!
 * Reconstruct the data at position x_loc, using a stencil centered at idx. Only uses points that have a non-zero volume
 * fraction in vol_data. The reconstruction uses a polyharmonic spline fit.
 *
 * x_loc must be given in index space.
 */
double radialBasisFunctionReconstruction(IBTK::VectorNd x_loc,
                                         const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                         const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                         const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                         const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                         const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                         const RBFPolyOrder order,
                                         const unsigned int stencil_size);

/*!
 * Reconstruct the data at position x_loc, using a stencil centered at idx. Only uses points that have a non-zero volume
 * fraction in vol_data. The reconstruction uses a least squares polynomial fit.
 */
double leastSquaresReconstruction(IBTK::VectorNd x_loc,
                                  const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                  const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                  const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                  const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                  LeastSquaresOrder order);

/*!
 * Reconstruct data at position x_loc given the lower cell index idx_ll using bilinear interpolation.
 */
double bilinearReconstruction(const IBTK::VectorNd& x_loc,
                              const IBTK::VectorNd& x_ll,
                              const SAMRAI::pdat::CellIndex<NDIM>& idx_ll,
                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              const double* const dx);

/*!
 * Reconstruct the data at position x_loc, using a stencil centered at idx. Only uses points that share the same sign of
 * the level set. The reconstruction uses a polyharmonic spline fit.
 *
 * x_loc must be given in index space.
 */
double radialBasisFunctionReconstruction(IBTK::VectorNd x_loc,
                                         double ls_val,
                                         const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                         const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                         const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                         const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                         const RBFPolyOrder order,
                                         const unsigned int stencil_size);

/*!
 * Reconstruct the data at position x_loc, using a stencil centered at x_loc. Uses the provided positions and values.
 *
 * x_loc must be given in physical space.
 */
double radialBasisFunctionReconstruction(const IBTK::VectorNd& x_loc,
                                         const std::vector<IBTK::VectorNd>& X_pts,
                                         const std::vector<double>& Q_vals,
                                         const RBFPolyOrder order);

/*!
 * Compute finite-difference weights using the points in fd_pts evaluated at the point base_pt. The action of the
 * operator needs to be supplied to both the radial basis function and the polynomials.
 */
template <class Point>
void
RBFFDReconstruct(std::vector<double>& wgts,
                 const Point& base_pt,
                 const std::vector<Point>& fd_pts,
                 const int poly_degree,
                 const double* const dx,
                 std::function<double(double)> rbf,
                 std::function<double(const Point&, const Point&, void*)> L_rbf,
                 void* rbf_ctx,
                 std::function<IBTK::VectorXd(const std::vector<Point>&, int, double, const Point&, void*)> L_polys,
                 void* poly_ctx);
/*!
 * Determine the WENO5 stencil is completely within the physical domain. We assume the stencil is centered at idx.
 *
 * Note: ls_data is assumed to have sufficient ghost width for the stencil.
 */
bool within_weno_stencil(const SAMRAI::pdat::CellIndex<NDIM>& idx, const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);

/*!
 * Compute the WENO5 interpolant at x using a stencil centered at idx and given the patch data Q_data.
 *
 * This performs interpolation in a dimension by dimension basis. First we interpolate in the y-direction to compute
 * values at ?. Then we interpolate in the x-direction to compute the value at o. If NDIM = 3, this process is prepended
 * by interpolations in the z-direction
 * -------------------------------
 * |     |     |     |     |     |
 * |  x  |  x  |  x  |  x  |  x  |
 * |     |     |     |     |     |
 * -------------------------------
 * |     |     |     |     |     |
 * |  x  |  x  |  x  |  x  |  x  |
 * |     |     |     |     |     |
 * -------------------------------
 * |  ?  |  ?  | o?  |  ?  |  ?  |
 * |  x  |  x  |  x  |  x  |  x  |
 * |     |     |     |     |     |
 * -------------------------------
 * |     |     |     |     |     |
 * |  x  |  x  |  x  |  x  |  x  |
 * |     |     |     |     |     |
 * -------------------------------
 * |     |     |     |     |     |
 * |  x  |  x  |  x  |  x  |  x  |
 * |     |     |     |     |     |
 * -------------------------------
 *
 * We assume Q_data has sufficient ghost width for the entire stencil.
 *
 * The point x should be given in index space.
 */
double weno5(const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
             const SAMRAI::pdat::CellIndex<NDIM>& idx,
             const IBTK::VectorNd& x);

/*!
 * Compute the WENO5 interpolant evaluated at xi given an array of Q values. The parameter xi should be between -0.5 and
 * 0.5 for accuracy. Note if |xi| > 0.5, then the stencil should be shifted so that |xi| <= 0.5.
 *
 * The type Array needs to be of length 5 with a defined operator[] that starts it's index at zero. We assume that Q[2]
 * is the center of the stencil, Q[0] is "below" the center, and Q[5] is "above" the center.
 */
template <typename Array>
double weno5(const Array& Q, double xi);

/*!
 * Compute the WENO interpolant given the candidate stencils f, smoothness indicators si, and smooth weights w.
 *
 * The type Array must be compatibe with standard operations on containers, i.e. it needs cbegin(), cend(), begin(), and
 * end() functions, and a copy constructor.
 */
template <typename Array>
double weno(const Array& f, const Array& si, const Array& w);
} // namespace Reconstruct

#include <ADS/private/reconstructions_inc.h>
#endif
