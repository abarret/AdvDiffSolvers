#ifndef included_ADS_reconstructions
#define included_ADS_reconstructions

#include <ibamr/config.h>

#include "ibtk/ibtk_utilities.h"

#include "CellData.h"
#include "CellIndex.h"
#include "NodeData.h"

namespace ADS
{
namespace Reconstruct
{
double sum_over_z_splines(const IBTK::VectorNd& x_loc,
                          const SAMRAI::pdat::CellIndex<NDIM>& idx,
                          const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                          const int order);

bool index_within_width(int stencil_width,
                        const SAMRAI::pdat::CellIndex<NDIM>& idx,
                        const SAMRAI::pdat::CellData<NDIM, double>& vol_data);

double evaluate_z_spline(const IBTK::VectorNd x, const int order);

int get_spline_width(int order);
double z_spline(double x, int order);

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
inline std::string enum_to_string(T /*val*/)
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
 * Use a flood filling algorithm to find neighboring points to a given index. Uses the value of the level set to
 * determine whether indices point to the same side.
 *
 * Note that fill_pts does not need to be empty. This function will append fill_pts until it's size is equal to
 * stencil_size.
 *
 * If compiled with debugging flags, throws a runtime_error if the flood filling algorithm could not find the requested
 * number of points.
 */
void flood_fill_for_points(std::vector<SAMRAI::pdat::CellIndex<NDIM>>& fill_pts,
                           const SAMRAI::pdat::CellIndex<NDIM>& idx,
                           const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                           double ls,
                           size_t stencil_size);

/*!
 * Use a flood filling algorithm to find neighboring points to a given side index. Uses the value of the level set to
 * determine whether indices point to same side.
 *
 * The cell index idx must contain sides on each axis that match the sign of ls.
 *
 * Note that fill_pts does not need to be empty. This function will append fill_pts until it's size is equal to
 * stencil_size.
 *
 * If compiled with debugging flags, throws a runtime_error if the flood filling algorithm could not find the requested
 * number of points.
 */
void flood_fill_for_points(std::vector<SAMRAI::pdat::SideIndex<NDIM>>& fill_pts,
                           const SAMRAI::pdat::CellIndex<NDIM>& idx,
                           const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                           double ls,
                           const int axis,
                           size_t stencil_size);

/*!
 * Reconstruct the data at position x_loc, using a stencil centered at idx. Only uses points that have a non-zero volume
 * fraction in vol_data. The reconstruction uses a polyharmonic spline fit.
 *
 * x_loc must be given in index space.
 */
double radial_basis_function_reconstruction(IBTK::VectorNd x_loc,
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
double least_squares_reconstruction(IBTK::VectorNd x_loc,
                                    const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                    const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                    const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                    const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                    const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                    LeastSquaresOrder order);

/*!
 * Reconstruct data at position x_loc given the lower cell index idx_ll using bilinear interpolation.
 */
double bilinear_reconstruction(const IBTK::VectorNd& x_loc,
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
double radial_basis_function_reconstruction(IBTK::VectorNd x_loc,
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
double radial_basis_function_reconstruction(const IBTK::VectorNd& x_loc,
                                            const std::vector<IBTK::VectorNd>& X_pts,
                                            const std::vector<double>& Q_vals,
                                            const RBFPolyOrder order);

/*!
 * Compute finite-difference weights using the points in fd_pts evaluated at the point base_pt. The action of the
 * operator needs to be supplied to both the radial basis function and the polynomials.
 */
template <class Point>
void
RBFFD_reconstruct(std::vector<double>& wgts,
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
 * Compute the quadratic Lagrange interpolant to the location x using an interpolant centered at idx.
 *
 * Note that x must be given in index space.
 */
double quadratic_lagrange_interpolant(IBTK::VectorNd x,
                                      const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                      const SAMRAI::pdat::CellData<NDIM, double>& Q_data);

/*!
 * Compute the quadratic Lagrange interpolant to the location x using an interpolant centered at idx. Limit the
 * interpolant if the reconstructed value falls outside the neighboring values.
 *
 * Note that x must be given in index space.
 */
double quadratic_lagrange_interpolant_limited(IBTK::VectorNd x,
                                              const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data);

/*!
 * Compute the divergence of a side centered velocity field using a finite difference stencil centered at idx.
 *
 * Note that x must be given in index space.
 *
 * The sign of ls determines the side from which we grab points.
 *
 * The cell index idx must contain sides on each axis that match the sign of ls.
 *
 * Uses the RBF r^5 with polynomials appended up to order RBFPolyOrder.
 */
double divergence(const IBTK::VectorNd& x,
                  const SAMRAI::pdat::CellIndex<NDIM>& idx,
                  double ls,
                  const SAMRAI::pdat::SideData<NDIM, double>& u_data,
                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                  RBFPolyOrder poly_order,
                  int stencil_size,
                  const double* const dx);
} // namespace Reconstruct
} // namespace ADS
#include <ADS/private/reconstructions_inc.h>
#endif
