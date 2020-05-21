#ifndef included_utility_functions
#define included_utility_functions

#include "LSFindCellVolume.h"
#include "SetLSValue.h"

#include "Variable.h"
#include "tbox/MathUtilities.h"

namespace IBAMR
{
inline double
length_fraction(const double dx, const double phi_l, const double phi_u)
{
    double L = 0.0;
    if (phi_l < 0.0 && phi_u > 0.0)
    {
        L = phi_l / (phi_l - phi_u);
    }
    else if (phi_l > 0.0 && phi_u < 0.0)
    {
        L = phi_u / (phi_u - phi_l);
    }
    else if (phi_l < 0.0 && phi_u < 0.0)
    {
        L = 1.0;
    }
    return L * dx;
}

inline VectorNd
midpoint_value(const VectorNd& pt0, const double& phi0, const VectorNd& pt1, const double& phi1)
{
    return pt0 * phi1 / (phi1 - phi0) - pt1 * phi0 / (phi1 - phi0);
}

inline IBTK::VectorNd
find_cell_centroid(const CellIndex<NDIM>& idx, const NodeData<NDIM, double>& ls_data)
{
    IBTK::VectorNd center;
    center.setZero();
    std::vector<IBTK::VectorNd> X_pts;
    double phi_ll = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 0))) + 1.0e-10;
    double phi_lr = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 0))) + 1.0e-10;
    double phi_ur = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 1))) + 1.0e-10;
    double phi_ul = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 1))) + 1.0e-10;
    if ((phi_ll < 0.0 && phi_lr < 0.0 && phi_ur < 0.0 && phi_lr < 0.0) ||
        (phi_ll > 0.0 && phi_lr > 0.0 && phi_ur > 0.0 && phi_lr > 0.0))
    {
        // Not a cut cell. Center is idx
        center(0) = idx(0) + 0.5;
        center(1) = idx(1) + 0.5;
    }
    else
    {
        // Loop over nodes and edges and find points
        if (phi_ll < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1)));
        if (phi_ll * phi_ul < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1) - phi_ll / (phi_ul - phi_ll)));
        if (phi_ul < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1) + 1.5));
        if (phi_ul * phi_ur < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - phi_ul / (phi_ur - phi_ul), idx(1) + 1.0));
        if (phi_ur < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1) + 1.0));
        if (phi_ur * phi_lr < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1) - phi_lr / (phi_ur - phi_lr)));
        if (phi_lr < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1)));
        if (phi_lr * phi_ll < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - phi_ll / (phi_lr - phi_ll), idx(1)));

        double signed_area = 0.0;
        for (size_t i = 0; i < X_pts.size(); ++i)
        {
            const IBTK::VectorNd& X = X_pts[i];
            const IBTK::VectorNd& X_n = X_pts[(i + 1) % X_pts.size()];
            center += (X + X_n) * (X(0) * X_n(1) - X_n(0) * X(1));
            signed_area += 0.5 * (X(0) * X_n(1) - X_n(0) * X(1));
        }
        center /= 6.0 * signed_area;
    }

    return center;
}

inline double
node_to_cell(const CellIndex<NDIM>& idx, NodeData<NDIM, double>& ls_data)
{
    NodeIndex<NDIM> idx_ll(idx, IntVector<NDIM>(0, 0));
    NodeIndex<NDIM> idx_lu(idx, IntVector<NDIM>(0, 1));
    NodeIndex<NDIM> idx_ul(idx, IntVector<NDIM>(1, 0));
    NodeIndex<NDIM> idx_uu(idx, IntVector<NDIM>(1, 1));
    double ls_ll = ls_data(idx_ll), ls_lu = ls_data(idx_lu);
    double ls_ul = ls_data(idx_ul), ls_uu = ls_data(idx_uu);
    return 0.25 * (ls_ll + ls_lu + ls_ul + ls_uu);
}

enum LeastSquaresOrder
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
    if (strcasecmp(val.c_str(), "CONSTANT") == 0) return CONSTANT;
    if (strcasecmp(val.c_str(), "LINEAR") == 0) return LINEAR;
    if (strcasecmp(val.c_str(), "QUADRATIC") == 0) return QUADRATIC;
    if (strcasecmp(val.c_str(), "CUBIC") == 0) return CUBIC;
    return UNKNOWN_ORDER;
};

template <>
inline std::string
enum_to_string<LeastSquaresOrder>(LeastSquaresOrder val)
{
    if (val == CONSTANT) return "CONSTANT";
    if (val == LINEAR) return "LINEAR";
    if (val == QUADRATIC) return "QUADRATIC";
    if (val == CUBIC) return "CUBIC";
    return "UNKNOWN_ORDER";
};

} // namespace IBAMR
#endif
