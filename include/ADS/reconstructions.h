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

static inline double
rbf(double r)
{
    return r;
}

static inline double
mls_weight(double r)
{
    return std::exp(-r * r);
}

double radialBasisFunctionReconstruction(IBTK::VectorNd x_loc,
                                         const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                         const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                         const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                         const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                         const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                         const RBFPolyOrder order,
                                         const unsigned int stencil_size);

double leastSquaresReconstruction(IBTK::VectorNd x_loc,
                                  const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                  const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                  const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                  const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                                  LeastSquaresOrder order);

double bilinearReconstruction(const IBTK::VectorNd& x_loc,
                              const IBTK::VectorNd& x_ll,
                              const SAMRAI::pdat::CellIndex<NDIM>& idx_ll,
                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              const double* const dx);

template <class Point, class MultiArray>
std::vector<IBTK::VectorNd>
shift_and_scale_pts(const std::vector<Point>& pts, IBTK::VectorNd shift, const MultiArray scale)
{
    std::vector<IBTK::VectorNd> new_pts(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
    {
        const Point& pt = pts[i];
        IBTK::VectorNd new_vec;
        for (unsigned int d = 0; d < NDIM; ++d) new_vec[d] = (pt(d) - shift(d)) / scale[d];
        new_pts[i] = new_vec;
    }
    return new_pts;
}

} // namespace Reconstruct
#endif
