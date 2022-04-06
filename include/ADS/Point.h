/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_Point
#define included_ADS_Point

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
 * The class Point generalizes the notion of a point. It encapsulates the IBTK::VectorNd class.
 */
class Point
{
public:
    /*!
     * \brief Constructor that makes a SAMRAI point.
     */
    Point(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch, const SAMRAI::pdat::CellIndex<NDIM>& idx)
    {
        SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        d_pt = getPt(patch, idx);
    }

    /*!
     * \brief Constructor that makes either a libmesh or boundary point.
     */
    Point(const IBTK::VectorNd& pt) : d_pt(pt)
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that leaves the object in an uninitialized state.
     *
     * \note This is used for compatibility with STL containers.
     */
    Point()
    {
        // intentionally blank
    }

    /*!
     * \brief Constructor that takes in a point. Leaves in an uninitialized state, but able to compare distances.
     *
     * \note This is used for creation of KD-trees.
     */
    Point(const std::vector<double>& pt)
    {
        for (unsigned int d = 0; d < NDIM; ++d) d_pt[d] = pt[d];
    }

    /*!
     * \brief Virtual destructor.
     */
    virtual ~Point() = default;

    /*!
     * \brief Compute the distance between a point and a IBTK::VectorNd.
     */
    virtual double dist(const IBTK::VectorNd& x) const
    {
        return (d_pt - x).norm();
    }

    /*!
     * \brief Compute the distance between another Point and this Point.
     */
    virtual double dist(const Point& pt) const
    {
        return (d_pt - pt.d_pt).norm();
    }

    /*!
     * \brief Return a copy of the ith point of the Point.
     */
    virtual double operator()(const size_t i) const
    {
        return d_pt(i);
    }

    /*!
     * \brief Return a reference of the ith point of the Point.
     */
    virtual double& operator()(const size_t i)
    {
        return d_pt(i);
    }

    /*!
     * \brief Return a copy of the ith point of the point.
     */
    virtual double operator[](const size_t i) const
    {
        return d_pt[i];
    }

    /*!
     * \brief Minus operator with another Point.
     */
    virtual Point operator-(const Point& pt)
    {
        return Point(d_pt - pt.d_pt);
    }

    /*!
     * \brief Plus operator with another Point.
     */
    virtual Point operator+(const Point& pt)
    {
        return Point(d_pt + pt.d_pt);
    }

    /*!
     * \brief Dot product between two points.
     */
    virtual double dot(const Point& pt) const
    {
        return d_pt.dot(pt.d_pt);
    }

    /*!
     * \brief Squared norm of a point.
     */
    virtual double squared_norm() const
    {
        return d_pt.squaredNorm();
    }

    /*!
     * \brief Norm of a point.
     */
    virtual double norm() const
    {
        return d_pt.norm();
    }

    /*!
     * \brief Get the physical location of the point.
     */
    virtual const IBTK::VectorNd& getVec() const
    {
        return d_pt;
    }

protected:
    IBTK::VectorNd d_pt;
};
} // namespace ADS
#endif
