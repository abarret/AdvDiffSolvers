/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_RBFPoint
#define included_ADS_RBFPoint

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/GhostPoints.h>
#include <ADS/Point.h>

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
/*!
 * The class RBFPoint generalizes the notion of a degree of freedom. It is a Point that also has an index. The index
 * allows us to "sort" points.
 */
class RBFPoint : public Point
{
public:
    /*!
     * \brief Constructor that creates an RBFPoint.
     */
    RBFPoint(const IBTK::VectorNd& pt, const int id = -1) : Point(pt), d_id(id)
    {
        // intentionally blank.
    }

    RBFPoint(const Point& pt, const int id = -1) : Point(pt), d_id(id)
    {
        // intentionally blank.
    }

    int getId() const
    {
        return d_id;
    }

    /*!
     * \brief Print out the information of this point.
     */
    friend std::ostream& operator<<(std::ostream& out, const RBFPoint& pt)
    {
        out << "   location: " << pt.d_pt.transpose() << "\n";
        out << "   id:       " << pt.d_id << "\n";
        return out;
    }

    /*!
     * \brief Comparison operator. Used for compatibility with STL containers.
     */
    friend bool operator==(const RBFPoint& lhs, const RBFPoint& rhs)
    {
        return lhs.d_id == rhs.d_id;
    }

    /*!
     * \brief Comparison operator. Used for compatibility with STL containers.
     *
     * The order of indexes is as follows:
     * SAMRAI < libMesh < ghost < empty
     */
    friend bool operator<(const RBFPoint& lhs, const RBFPoint& rhs)
    {
        return lhs.d_id < rhs.d_id;
    }

private:
    int d_id = -1;
};
} // namespace ADS
#endif
