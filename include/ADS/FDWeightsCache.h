/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_FDWeightsCache
#define included_ADS_FDWeightsCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ADS/FEMeshPartitioner.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/dof_map.h"
#include "libmesh/node.h"

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * Struct FDCachedPoint generalizes the notion of a degree of freedom. It is either a SAMRAI index, a libMesh Node, or a
 * boundary Node. This allows for efficient caching of points, and quick lookups of individual points.
 */
struct FDCachedPoint
{
public:
    /*!
     * \brief Constructor that makes a SAMRAI point.
     */
    FDCachedPoint(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch,
                  const SAMRAI::pdat::CellIndex<NDIM>& idx)
        : d_idx(idx)
    {
        SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const SAMRAI::hier::Index<NDIM>& idx_low = patch->getBox().lower();
        const SAMRAI::hier::Index<NDIM>& idx_up = patch->getBox().upper();
        for (unsigned int d = 0; d < NDIM; ++d)
            d_pt[d] = xlow[d] + dx[d] * (static_cast<double>(d_idx(d) - idx_low(d)) + 0.5);
        for (unsigned int d = 0; d < (NDIM - 1); ++d) d_max_idx[d] = idx_up[d] - idx_low[d] + 1;
    }

    /*!
     * \brief Constructor that makes either a libmesh or boundary point.
     */
    FDCachedPoint(const IBTK::VectorNd& pt, libMesh::Node* node, bool bdry_pt)
        : d_pt(pt), d_node(node), d_bdry_pt(bdry_pt)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that leaves the object in an uninitialized state.
     *
     * \note This is used for compatibility with STL containers.
     */
    FDCachedPoint() : d_empty(true)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that takes in a point. Leaves in an uninitialized state, but able to compare distances.
     *
     * \note This is used for creation of KD-trees.
     */
    FDCachedPoint(const std::vector<double>& pt) : d_empty(true)
    {
        for (unsigned int d = 0; d < NDIM; ++d) d_pt[d] = pt[d];
    }

    /*!
     * \brief Compute the distance between a point and the cached point.
     */
    double dist(const IBTK::VectorNd& x) const
    {
        return (d_pt - x).norm();
    }

    /*!
     * \brief Compute the distance between another cached point and this point.
     */
    double dist(const FDCachedPoint& pt) const
    {
        return (d_pt - pt.getVec()).norm();
    }

    /*!
     * \brief Return a copy of the ith point of the point.
     */
    double operator()(const size_t i) const
    {
        return d_pt(i);
    }

    /*!
     * \brief Return a copy of the ith point of the point.
     */
    double operator[](const size_t i) const
    {
        return d_pt[i];
    }

    /*!
     * \brief Print out the information of this point.
     */
    friend std::ostream& operator<<(std::ostream& out, const FDCachedPoint& pt)
    {
        out << "   location: " << pt.d_pt.transpose() << "\n";
        out << "   is located on boundary: " << pt.d_bdry_pt << "\n";
        if (pt.isNode())
            out << "   node id: " << pt.d_node->id();
        else if (!pt.isEmpty())
            out << "   idx:     " << pt.d_idx;
        else
            out << "   pt is neither node nor index";
        return out;
    }

    /*!
     * \brief Comparison operator. Used for compatibility with STL containers.
     */
    friend bool operator==(const FDCachedPoint& lhs, const FDCachedPoint& rhs)
    {
        if (lhs.isNode() && rhs.isNode())
        {
            return (lhs.d_bdry_pt == rhs.d_bdry_pt) && (lhs.d_node == rhs.d_node);
        }
        else if (lhs.isIdx() && rhs.isIdx())
        {
            return lhs.d_idx == rhs.d_idx;
        }
        else
        {
            return false;
        }
    }

    /*!
     * \brief Comparison operator. Used for compatibility with STL containers.
     *
     * The order of indexes is as follows:
     * SAMRAI < libMesh < bdry < empty
     */
    friend bool operator<(const FDCachedPoint& lhs, const FDCachedPoint& rhs)
    {
        if (lhs.isIdx())
        {
            if (!rhs.isIdx()) return true;
#if NDIM == 2
            int l_idx = lhs.d_idx(0) + lhs.d_idx(1) * lhs.d_max_idx[0];
            int r_idx = rhs.d_idx(0) + rhs.d_idx(1) * rhs.d_max_idx[0];
#endif
#if NDIM == 3
            int l_idx =
                lhs.d_idx(0) + lhs.d_idx(1) * lhs.d_max_idx[0] + lhs.d_idx(2) * lhs.d_max_idx[0] * lhs.d_max_idx[1];
            int r_idx =
                rhs.d_idx(0) + rhs.d_idx(1) * rhs.d_max_idx[0] + rhs.d_idx(2) * rhs.d_max_idx[0] * rhs.d_max_idx[1];
#endif
            return l_idx < r_idx;
        }
        else if (lhs.isNode())
        {
            if (rhs.isEmpty()) return true;
            if (rhs.isIdx()) return false;
            if (rhs.isBdry()) return true;
            return lhs.d_node->id() < rhs.d_node->id();
        }
        else if (lhs.isBdry())
        {
            if (rhs.isEmpty()) return true;
            if (!rhs.isBdry()) return false;
            return lhs.d_node->id() < rhs.d_node->id();
        }
        return false;
    }

    /*!
     * \brief Is this point actually initialized?
     */
    bool isEmpty() const
    {
        return d_empty;
    }

    /*!
     * \brief Is this point a libMesh node?
     */
    bool isNode() const
    {
        return !d_bdry_pt && d_node != nullptr;
    }

    /*!
     * \brief Is this point a SAMRAI point?
     */
    bool isIdx() const
    {
        return !isNode() && !isEmpty();
    }

    /*!
     * \brief Is this point a boundary node?
     */
    bool isBdry() const
    {
        return d_bdry_pt;
    }

    /*!
     * \brief Get the node pointer.
     */
    const libMesh::Node* const getNode() const
    {
        if (d_bdry_pt || !isNode()) TBOX_ERROR("Not at a node\n");
        return d_node;
    }

    /*!
     * \brief Get the SAMRAI index
     */
    const SAMRAI::pdat::CellIndex<NDIM>& getIndex() const
    {
        if (isNode()) TBOX_ERROR("At at node\n");
        if (d_empty) TBOX_ERROR("Not a point\n");
        return d_idx;
    }

    /*!
     * \brief Get the physical location of the point.
     */
    const IBTK::VectorNd& getVec() const
    {
        return d_pt;
    }

private:
    IBTK::VectorNd d_pt;
    libMesh::Node* d_node = nullptr;
    SAMRAI::pdat::CellIndex<NDIM> d_idx;
    std::array<int, NDIM - 1> d_max_idx;
    bool d_empty = false;
    bool d_bdry_pt = false;
};

/*!
 * \brief Class FDWeightsCache is a class that caches finite difference weights for general operators. This class
 * requires that finite difference weights be registered. No calculations are done by this class.
 */
class FDWeightsCache
{
public:
    /*!
     * \brief Constructor
     */
    FDWeightsCache(std::string object_name);

    /*!
     * \brief Destructor.
     */
    virtual ~FDWeightsCache();

    virtual void cachePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                            const FDCachedPoint& pt,
                            const std::vector<FDCachedPoint>& fd_pts,
                            const std::vector<double>& fd_weights);

    const std::map<FDCachedPoint, std::vector<double>>&
    getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);
    const std::map<FDCachedPoint, std::vector<FDCachedPoint>>&
    getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);
    const std::set<FDCachedPoint>& getRBFFDBasePoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    const std::vector<double>& getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                               const FDCachedPoint& pt);
    const std::vector<FDCachedPoint>& getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                                     const FDCachedPoint& pt);
    bool isRBFFDBasePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDCachedPoint& pt);

    virtual void clearCache();

    /*!
     * Debugging functions
     */
    virtual void printPtMap(std::ostream& os, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

protected:
    std::string d_object_name;

    // Weight and point information
    using PtVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::set<FDCachedPoint>>;
    using PtPairVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::map<FDCachedPoint, std::vector<FDCachedPoint>>>;
    using WeightVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::map<FDCachedPoint, std::vector<double>>>;
    PtVecMap d_base_pt_set;
    PtPairVecMap d_pair_pt_map;
    WeightVecMap d_pt_weight_map;

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    FDWeightsCache() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    FDWeightsCache(const FDWeightsCache& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    FDWeightsCache& operator=(const FDWeightsCache& that) = delete;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_FDWeightsCache
