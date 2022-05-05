#include <ADS/PointList.h>

#include <ibtk/IBTK_MPI.h>

namespace ADS
{
template <typename Iter>
inline void
PointList::insertPoints(Iter it, Iter end, const PointState& state, bool update_counts)
{
    for (; it != end; ++it) insertPoint(*it, state);
    if (update_counts) updatePtCount(d_pts.size());
}

inline void
PointList::insertPoint(const Point& pt, const PointState& state)
{
    int rank = IBTK::IBTK_MPI::getRank();
    RBFPoint new_pt(pt, d_num_pts_per_proc[rank]++);
    d_pts.insert(new_pt);
    d_state[new_pt] = state;
}

inline const std::set<RBFPoint>&
PointList::getPts() const
{
    return d_pts;
}

inline const std::map<RBFPoint, PointState>&
PointList::getState() const
{
    return d_state;
}

inline int
PointList::getState(const RBFPoint& pt) const
{
#ifndef NDEBUG
    TBOX_ASSERT(isValidPt(pt));
#endif
    return d_state.at(pt);
}

inline bool
PointList::isValidPt(const RBFPoint& pt) const
{
    return std::find(d_pts.begin(), d_pts.end(), pt) != d_pts.end();
}

inline const std::vector<unsigned int>&
PointList::getPtsPerProc() const
{
    return d_num_pts_per_proc;
}

} // namespace ADS
