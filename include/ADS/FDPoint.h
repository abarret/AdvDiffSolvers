/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_FDPoint
#define included_ADS_FDPoint

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/GhostPoints.h>

#include <ibtk/ibtk_utilities.h>

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "CellIndex.h"
#include "IntVector.h"
#include "Patch.h"
#include "tbox/Pointer.h"

#include "libmesh/node.h"

#include <string>
#include <vector>

namespace ADS
{
/*
 * Helper function to get the physical location of a patch and cell index pair.
 */
inline IBTK::VectorNd
getPt(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch, const SAMRAI::pdat::CellIndex<NDIM>& idx)
{
    SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const SAMRAI::hier::Index<NDIM>& idx_low = patch->getBox().lower();
    IBTK::VectorNd x;
    for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
    return x;
}

/*!
 * The class FDPoint generalizes the notion of a degree of freedom. It is either a SAMRAI index, a libMesh Node, or a
 * boundary Node. This allows for efficient caching of points, and quick lookups of individual points.
 */
class FDPoint
{
public:
    /*!
     * \brief Constructor that makes a SAMRAI point.
     */
    FDPoint(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch, const SAMRAI::pdat::CellIndex<NDIM>& idx)
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
    FDPoint(const IBTK::VectorNd& pt, libMesh::Node* node) : d_pt(pt), d_node(node)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that makes a ghost point.
     */
    FDPoint(GhostPoint* ghost_point) : d_pt(ghost_point->getX()), d_ghost_point(ghost_point)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that leaves the object in an uninitialized state.
     *
     * \note This is used for compatibility with STL containers.
     */
    FDPoint() : d_empty(true)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that takes in a point. Leaves in an uninitialized state, but able to compare distances.
     *
     * \note This is used for creation of KD-trees.
     */
    FDPoint(const std::vector<double>& pt) : d_empty(true)
    {
        for (unsigned int d = 0; d < NDIM; ++d) d_pt[d] = pt[d];
    }

    /*!
     * \brief Compute the distance between a point and this FDPoint.
     */
    double dist(const IBTK::VectorNd& x) const
    {
        return (d_pt - x).norm();
    }

    /*!
     * \brief Compute the distance between another FDPoint and this FDPoint.
     */
    double dist(const FDPoint& pt) const
    {
        return (d_pt - pt.getVec()).norm();
    }

    /*!
     * \brief Return a copy of the ith point of the FDPoint.
     */
    double operator()(const size_t i) const
    {
        return d_pt(i);
    }

    /*!
     * \brief Return a reference of the ith point of the FDPoint.
     */
    double& operator()(const size_t i)
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
    friend std::ostream& operator<<(std::ostream& out, const FDPoint& pt)
    {
        out << "   location: " << pt.d_pt.transpose() << "\n";
        if (pt.isNode())
            out << "   node id: " << pt.d_node->id();
        else if (pt.isIdx())
            out << "   idx:     " << pt.d_idx;
        else if (pt.isGhost())
            out << "   ghost id: " << pt.d_ghost_point->getId();
        else
            out << "   pt is neither node nor index";
        return out;
    }

    /*!
     * \brief Comparison operator. Used for compatibility with STL containers.
     */
    friend bool operator==(const FDPoint& lhs, const FDPoint& rhs)
    {
        if (lhs.isNode() && rhs.isNode())
        {
            return lhs.d_node == rhs.d_node;
        }
        else if (lhs.isIdx() && rhs.isIdx())
        {
            return lhs.d_idx == rhs.d_idx;
        }
        else if (lhs.isGhost() && rhs.isGhost())
        {
            return lhs.d_ghost_point == rhs.d_ghost_point;
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
     * SAMRAI < libMesh < ghost < empty
     */
    friend bool operator<(const FDPoint& lhs, const FDPoint& rhs)
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
            if (rhs.isGhost()) return true;
            return lhs.d_node->id() < rhs.d_node->id();
        }
        else if (lhs.isGhost())
        {
            if (rhs.isEmpty()) return true;
            if (!rhs.isGhost()) return false;
            return lhs.d_ghost_point->getId() < rhs.d_ghost_point->getId();
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
        return d_node != nullptr;
    }

    /*!
     * \brief Is this point a SAMRAI point?
     */
    bool isIdx() const
    {
        return !isNode() && !isGhost() && !isEmpty();
    }

    /*!
     * \brief Is this point a boundary node?
     */
    bool isGhost() const
    {
        return d_ghost_point != nullptr;
    }

    /*!
     * \brief Get the node pointer.
     */
    const libMesh::Node* const getNode() const
    {
        if (!isNode()) TBOX_ERROR("Not at a node\n");
        return d_node;
    }

    /*!
     * \brief Get the SAMRAI index
     */
    const SAMRAI::pdat::CellIndex<NDIM>& getIndex() const
    {
        if (!isIdx()) TBOX_ERROR("Not an index\n");
        return d_idx;
    }

    /*!
     * \brief Get the ghost point
     */
    const GhostPoint* getGhostPoint() const
    {
        if (!isGhost()) TBOX_ERROR("Not a ghost point!\n");
        return d_ghost_point;
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
    GhostPoint* d_ghost_point = nullptr;
    SAMRAI::pdat::CellIndex<NDIM> d_idx;
    std::array<int, NDIM - 1> d_max_idx;
    bool d_empty = false;
};
} // namespace ADS
#endif
