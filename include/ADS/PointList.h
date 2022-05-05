/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_PointList
#define included_ADS_PointList

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/RBFPoint.h>

#include <mpi.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
enum PointState
{
    ACTIVE,
    INACTIVE,
    BOUNDARY,
    GHOST,
    UNKNOWN_STATE = -1
};

std::string
enum_to_string(const PointState& state)
{
    if (state == ACTIVE) return "ACTIVE";
    if (state == INACTIVE) return "INACTIVE";
    if (state == BOUNDARY) return "BOUNDARY";
    if (state == GHOST) return "GHOST";
    return "UNKNOWN STATE";
}

PointState
string_to_enum(const std::string& str)
{
    if (strcasecmp(str.c_str(), "ACTIVE")) return ACTIVE;
    if (strcasecmp(str.c_str(), "INACTIVE")) return INACTIVE;
    if (strcasecmp(str.c_str(), "BOUNDARY")) return BOUNDARY;
    if (strcasecmp(str.c_str(), "GHOST")) return GHOST;
    return UNKNOWN_STATE;
}

/*!
 * PointList is a class that manages a list of points. Points are tracked as "active", "inactive", "boundary", or
 * "ghost." Points are assigned a degree of freedom.
 */
class PointList
{
public:
    /*!
     * \brief PointList constructor. This leaves the object without any points registered.
     */
    PointList(std::string object_name);

    /*!
     * \brief Destructor that by default calls clearDOFs().
     */
    virtual ~PointList();

    /*!
     * \brief Reset the object and leave it in an uninitialized state.
     */
    virtual void clearDOFs();

    /*!
     * \brief Insert the pts into the point list with the specified state.
     *
     * \note An optional parameter is included to update the counts of points on other processors.
     */
    template <typename Iter>
    void insertPoints(Iter it, Iter end, const PointState& state, bool update_counts = true);

    /*!
     * \brief Add one point to the point list with the specified state.
     *
     * \note This function does not update the number of points on other processors. Users should call updatePtCount()
     * after all points are added.
     */
    virtual void insertPoint(const Point& pt, const PointState& state);

    /*!
     * \brief Return all the points owned by this object.
     */
    virtual const std::set<RBFPoint>& getPts() const;

    /*!
     * \brief Return the map between point and the corresponding state.
     */
    virtual const std::map<RBFPoint, PointState>& getState() const;

    /*!
     * \brief Return the state for the corresponding point.
     *
     * \note This does no error checking when debugging is disabled.
     */
    virtual int getState(const RBFPoint& pt) const;

    /*!
     * \brief Checks whether the specified point is in the point list.
     */
    virtual bool isValidPt(const RBFPoint& pt) const;

    /*!
     * \brief Return the number of points held on each processor.
     */
    const std::vector<unsigned int>& getPtsPerProc() const;

    /*!
     * \brief Print the point list to the specified ostream.
     */
    void printPointList(std::ostream& out, int rank = 0);

    /*!
     * \brief Update the point count on all processors. This requires and All to all communication.
     */
    virtual void updatePtCount(unsigned int num_pts);

protected:
    std::string d_object_name;
    std::set<RBFPoint> d_pts;
    std::map<RBFPoint, PointState> d_state;
    std::vector<unsigned int> d_num_pts_per_proc;
};
} // namespace ADS

#include <ADS/private/PointList_inc.h>
#endif // included_ADS_PointList
