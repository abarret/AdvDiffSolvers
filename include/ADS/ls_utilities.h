#ifndef included_ADS_utilities
#define included_ADS_utilities
#include "ibamr/config.h"

#include "CellData.h"
#include "FaceData.h"
#include "Patch.h"
#include "SideData.h"
#include "tbox/Pointer.h"

#include "libmesh/elem.h"
#include "libmesh/point.h"

#include <functional>

namespace ADS
{
using ReactionFcn =
    std::function<double(double, const std::vector<double>&, const std::vector<double>&, double, void*)>;
using ReactionFcnCtx = std::pair<ReactionFcn, void*>;
using BdryConds = std::tuple<ReactionFcn, ReactionFcn, void*>;
using MappingFcn = std::function<void(libMesh::Node*, libMesh::Elem*, libMesh::Point& x_cur)>;

struct IndexList
{
public:
    IndexList(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch, const SAMRAI::pdat::CellIndex<NDIM>& idx)
        : d_idx(idx), d_patch(patch)
    {
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

    bool operator<(const IndexList& b) const
    {
        bool less_than_b = false;
        if (d_patch->getPatchNumber() < b.d_patch->getPatchNumber())
            less_than_b = true;
        else if (b.d_patch->getPatchNumber() == d_patch->getPatchNumber() && d_global_idx < b.d_global_idx)
        {
            less_than_b = true;
        }
        return less_than_b;
    }
    int d_global_idx = -1;
    SAMRAI::pdat::CellIndex<NDIM> d_idx;
    SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> d_patch;
};

struct CutCellElems
{
public:
    CutCellElems(libMesh::Elem* parent_elem,
                 std::vector<std::pair<libMesh::Point, int>> intersections,
                 unsigned int part = 0)
        : d_parent_elem(parent_elem), d_intersection_side_vec(intersections), d_part(part)
    {
        d_elem = libMesh::Elem::build(d_parent_elem->type());
        for (size_t n = 0; n < d_intersection_side_vec.size(); ++n)
        {
            d_nodes.push_back(libMesh::Node::build(d_intersection_side_vec[n].first, n));
            d_elem->set_id(0);
            d_elem->set_node(n) = d_nodes[n].get();
        }
        for (size_t n = 0; n < d_parent_cur_pts.size(); ++n) d_parent_cur_pts[n] = d_parent_elem->point(n);
    }

    libMesh::Elem* d_parent_elem = nullptr;
    std::array<libMesh::Point, 2> d_parent_cur_pts;
    std::unique_ptr<libMesh::Elem> d_elem;
    std::vector<std::unique_ptr<libMesh::Node>> d_nodes;
    std::vector<std::pair<libMesh::Point, int>> d_intersection_side_vec;
    unsigned int d_part = 0;
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

enum class AdvReconstructType
{
    ZSPLINES,
    RBF,
    LINEAR,
    UNKNOWN_TYPE
};
template <>
inline AdvReconstructType
string_to_enum<AdvReconstructType>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "ZSPLINES") == 0) return AdvReconstructType::ZSPLINES;
    if (strcasecmp(val.c_str(), "RBF") == 0) return AdvReconstructType::RBF;
    if (strcasecmp(val.c_str(), "LINEAR") == 0) return AdvReconstructType::LINEAR;
    return AdvReconstructType::UNKNOWN_TYPE;
}

template <>
inline std::string
enum_to_string<AdvReconstructType>(AdvReconstructType val)
{
    if (val == AdvReconstructType::ZSPLINES) return "ZSPLINES";
    if (val == AdvReconstructType::RBF) return "RBF";
    if (val == AdvReconstructType::LINEAR) return "LINEAR";
    return "UNKNOWN_TYPE";
}
} // namespace ADS
#endif /* included_LS_utilities */
