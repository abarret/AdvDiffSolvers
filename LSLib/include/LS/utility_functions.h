#ifndef included_LS_utility_functions
#define included_LS_utility_functions

#include "LS/LSFindCellVolume.h"
#include "LS/SetLSValue.h"

#include "Variable.h"
#include "tbox/MathUtilities.h"

namespace LS
{
static double s_eps = 1.0e-12;
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

    double phi_ll = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 0)));
    if (std::abs(phi_ll) < s_eps) phi_ll = phi_ll < 0.0 ? -s_eps : s_eps;
    double phi_ul = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 0)));
    if (std::abs(phi_ul) < s_eps) phi_ul = phi_ul < 0.0 ? -s_eps : s_eps;
    double phi_uu = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 1)));
    if (std::abs(phi_uu) < s_eps) phi_uu = phi_uu < 0.0 ? -s_eps : s_eps;
    double phi_lu = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 1)));
    if (std::abs(phi_lu) < s_eps) phi_lu = phi_lu < 0.0 ? -s_eps : s_eps;
    if ((phi_ll < 0.0 && phi_ul < 0.0 && phi_uu < 0.0 && phi_lu < 0.0) ||
        (phi_ll > 0.0 && phi_ul > 0.0 && phi_uu > 0.0 && phi_lu > 0.0))
    {
        // Not a cut cell. Center is idx
        center(0) = idx(0) + 0.5;
        center(1) = idx(1) + 0.5;
    }
    else
    {
        // Loop over nodes and edges and find points
        if (phi_ll < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1)));
        if (phi_ll * phi_lu < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1) - phi_ll / (phi_lu - phi_ll)));
        if (phi_lu < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0), idx(1) + 1.0));
        if (phi_lu * phi_uu < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - phi_lu / (phi_uu - phi_lu), idx(1) + 1.0));
        if (phi_uu < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1) + 1.0));
        if (phi_uu * phi_ul < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1) - phi_ul / (phi_uu - phi_ul)));
        if (phi_ul < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 1.0, idx(1)));
        if (phi_ul * phi_ll < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - phi_ll / (phi_ul - phi_ll), idx(1)));

        double signed_area = 0.0;
        for (size_t i = 0; i < X_pts.size(); ++i)
        {
            const IBTK::VectorNd& X = X_pts[i];
            const IBTK::VectorNd& X_n = X_pts[(i + 1) % X_pts.size()];
            center += (X + X_n) * (X(0) * X_n(1) - X_n(0) * X(1));
            signed_area += 0.5 * (X(0) * X_n(1) - X_n(0) * X(1));
        }
        center /= 6.0 * signed_area;
        if (std::abs(signed_area) < 1.0e-8)
        {
            // Degenerate polygon. Switch to average of vertices.
            center.setZero();
            for (const auto& X : X_pts)
            {
                center += X;
            }
            center /= static_cast<double>(X_pts.size());
        }
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
}

template <>
inline std::string
enum_to_string<LeastSquaresOrder>(LeastSquaresOrder val)
{
    if (val == CONSTANT) return "CONSTANT";
    if (val == LINEAR) return "LINEAR";
    if (val == QUADRATIC) return "QUADRATIC";
    if (val == CUBIC) return "CUBIC";
    return "UNKNOWN_ORDER";
}

enum class AdvectionTimeIntegrationMethod
{
    FORWARD_EULER,
    MIDPOINT_RULE,
    UNKNOWN_METHOD
};

template <>
inline AdvectionTimeIntegrationMethod
string_to_enum<AdvectionTimeIntegrationMethod>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "FORWARD_EULER") == 0) return AdvectionTimeIntegrationMethod::FORWARD_EULER;
    if (strcasecmp(val.c_str(), "MIDPOINT_RULE") == 0) return AdvectionTimeIntegrationMethod::MIDPOINT_RULE;
    return AdvectionTimeIntegrationMethod::UNKNOWN_METHOD;
}

template <>
inline std::string
enum_to_string<AdvectionTimeIntegrationMethod>(AdvectionTimeIntegrationMethod val)
{
    if (val == AdvectionTimeIntegrationMethod::FORWARD_EULER) return "FORWARD_EULER";
    if (val == AdvectionTimeIntegrationMethod::MIDPOINT_RULE) return "MIDPOINT_RULE";
    return "UNKNOWN_METHOD";
}

enum class DiffusionTimeIntegrationMethod
{
    BACKWARD_EULER,
    TRAPEZOIDAL_RULE,
    UNKNOWN_METHOD
};

template <>
inline DiffusionTimeIntegrationMethod
string_to_enum<DiffusionTimeIntegrationMethod>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "BACKWARD_EULER") == 0) return DiffusionTimeIntegrationMethod::BACKWARD_EULER;
    if (strcasecmp(val.c_str(), "TRAPEZOIDAL_RULE") == 0) return DiffusionTimeIntegrationMethod::TRAPEZOIDAL_RULE;
    return DiffusionTimeIntegrationMethod::UNKNOWN_METHOD;
}

template <>
inline std::string
enum_to_string<DiffusionTimeIntegrationMethod>(DiffusionTimeIntegrationMethod val)
{
    if (val == DiffusionTimeIntegrationMethod::BACKWARD_EULER) return "BACKWARD_EULER";
    if (val == DiffusionTimeIntegrationMethod::TRAPEZOIDAL_RULE) return "TRAPEZOIDAL_RULE";
    return "UNKNOWN_METHOD";
}

inline void
copy_face_to_side(const int u_s_idx, const int u_f_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> s_data = patch->getPatchData(u_s_idx);
            Pointer<FaceData<NDIM, double>> f_data = patch->getPatchData(u_f_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SideIterator<NDIM> si(patch->getBox(), axis); si; si++)
                {
                    const SideIndex<NDIM>& s_idx = si();
                    FaceIndex<NDIM> f_idx(s_idx.toCell(0), axis, 1);
                    (*s_data)(s_idx) = (*f_data)(f_idx);
                }
            }
        }
    }
}

struct PatchIndexPair
{
public:
    PatchIndexPair(const Pointer<Patch<NDIM>>& patch, const CellIndex<NDIM>& idx) : d_idx(idx)
    {
        d_patch_num = patch->getPatchNumber();
        const Box<NDIM>& box = patch->getBox();
        const hier::Index<NDIM>& idx_low = box.lower();
        const hier::Index<NDIM>& idx_up = box.upper();
        int num_x = idx_up(0) - idx_low(0) + 1;
        d_global_idx = idx(0) - idx_low(0) + num_x * (idx(1) - idx_low(1) + 1);
#if (NDIM == 3)
        int num_y = idx_up(1) - idx_low(0) + 1;
        d_global_idx += num_x * num_y * (idx(2) - idx_low(2));
#endif
    }

    bool operator<(const PatchIndexPair& b) const
    {
        bool less_than_b = false;
        if (d_patch_num < b.d_patch_num)
        {
            // We're on a smaller patch
            less_than_b = true;
        }
        else if (d_patch_num == b.d_patch_num && d_global_idx < b.d_global_idx)
        {
            // Our global index is smaller than b but on the same patch
            less_than_b = true;
        }
        return less_than_b;
    }

    CellIndex<NDIM> d_idx;
    int d_patch_num = -1;
    int d_global_idx = -1;
};
#define LS_TIMER_START(timer) timer->start();

#define LS_TIMER_STOP(timer) timer->stop();

} // namespace LS
#endif
