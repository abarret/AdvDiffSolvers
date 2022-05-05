#include <ADS/PointList.h>
#include <ADS/app_namespaces.h>

#include <ibtk/IBTK_MPI.h>

namespace ADS
{
PointList::PointList(std::string object_name) : d_object_name(std::move(object_name))
{
    d_num_pts_per_proc.resize(IBTK_MPI::getNodes());
}

PointList::~PointList()
{
    clearDOFs();
}

void
PointList::clearDOFs()
{
    d_pts.clear();
    d_state.clear();
}

void
PointList::printPointList(std::ostream& out, int rank)
{
    if (IBTK_MPI::getRank() == rank)
    {
        out << "Printing point list for object " << d_object_name << "\n";
        for (const auto& pt : d_pts)
        {
            out << "  pt    " << pt.getVec().transpose() << "\n";
            out << "  id    " << pt.getId() << "\n";
            out << "  state " << enum_to_string(d_state.at(pt)) << "\n";
        }
    }
}

void
PointList::updatePtCount(const unsigned int num_pts)
{
    const int mpi_size = IBTK_MPI::getNodes();
    d_num_pts_per_proc.resize(mpi_size);
    IBTK_MPI::allGather(num_pts, d_num_pts_per_proc.data());
}
} // namespace ADS
