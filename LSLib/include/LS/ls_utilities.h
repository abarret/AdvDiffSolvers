#ifndef LSLIB_INCLUDE_LS_LS_UTILITIES
#define LSLIB_INCLUDE_LS_LS_UTILITIES
#include "IBAMR_config.h"

#include "CellData.h"
#include "FaceData.h"
#include "SideData.h"
#include "tbox/Pointer.h"

#include "libmesh/elem.h"
#include "libmesh/point.h"

namespace LS
{
using ReactionFcn =
    std::function<double(double, const std::vector<double>&, const std::vector<double>&, double, void*)>;
using ReactionFcnCtx = std::pair<ReactionFcn, void*>;
using BdryConds = std::tuple<ReactionFcn, ReactionFcn, void*>;

inline void
copy_face_to_side(const int u_s_idx,
                  const int u_f_idx,
                  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
            SAMRAI::tbox::Pointer<SAMRAI::pdat::SideData<NDIM, double>> s_data = patch->getPatchData(u_s_idx);
            SAMRAI::tbox::Pointer<SAMRAI::pdat::FaceData<NDIM, double>> f_data = patch->getPatchData(u_f_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SAMRAI::pdat::SideIterator<NDIM> si(patch->getBox(), axis); si; si++)
                {
                    const SAMRAI::pdat::SideIndex<NDIM>& s_idx = si();
                    SAMRAI::pdat::FaceIndex<NDIM> f_idx(s_idx.toCell(0), axis, 1);
                    (*s_data)(s_idx) = (*f_data)(f_idx);
                }
            }
        }
    }
}

struct PatchIndexPair
{
public:
    PatchIndexPair(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                   const SAMRAI::pdat::CellIndex<NDIM>& idx)
        : d_idx(idx)
    {
        d_patch_num = patch->getPatchNumber();
        const SAMRAI::hier::Box<NDIM>& box = patch->getBox();
        const SAMRAI::hier::Index<NDIM>& idx_low = box.lower();
        const SAMRAI::hier::Index<NDIM>& idx_up = box.upper();
        int num_x = idx_up(0) - idx_low(0) + 1;
        d_global_idx = idx(0) - idx_low(0) + num_x * (idx(1) - idx_low(1) + 1);
#if (NDIM == 3)
        int num_y = idx_up(1) - idx_low(1) + 1;
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

    SAMRAI::pdat::CellIndex<NDIM> d_idx;
    int d_patch_num = -1;
    int d_global_idx = -1;
};

struct CutCellElems
{
public:
    CutCellElems(libMesh::Elem* parent_elem, std::vector<libMesh::Point> intersections)
        : d_parent_elem(parent_elem), d_intersections(intersections)
    {
        d_elem = libMesh::Elem::build(d_parent_elem->type());
        for (size_t n = 0; n < d_intersections.size(); ++n)
        {
            d_nodes.push_back(libMesh::Node::build(d_intersections[n], n));
            d_elem->set_id(0);
            d_elem->set_node(n) = d_nodes[n].get();
        }
    }

    libMesh::Elem* d_parent_elem;
    std::unique_ptr<libMesh::Elem> d_elem;
    std::vector<std::unique_ptr<libMesh::Node>> d_nodes;
    std::vector<libMesh::Point> d_intersections;
};

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
} // namespace LS
#endif /* LSLIB_INCLUDE_LS_LS_UTILITIES_H_ */
