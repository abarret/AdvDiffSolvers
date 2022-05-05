#include <ibamr/config.h>

#include <ADS/Point.h>
#include <ADS/PointList.h>
#include <ADS/app_namespaces.h>

#include <ibtk/IBTKInit.h>

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize PETSc, MPI, and SAMRAI.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);

    // Suppress a warning
    SAMRAI::tbox::Logger::getInstance()->setWarning(false);

    { // cleanup dynamically allocated objects prior to shutdown
        // Generate points
        std::array<int, NDIM> num_pts_per_dir;
        num_pts_per_dir.fill(10);
        int num_pts = 1;
        for (int d = 0; d < NDIM; ++d) num_pts *= num_pts_per_dir[d];

        // Now generate points
        VectorNd xlow(VectorNd::Zero());
        VectorNd xup(VectorNd::Ones());

        std::vector<VectorNd> pt_list;
        pt_list.reserve(num_pts);
        VectorNd pt = xlow;
        for (int x = 0; x < num_pts_per_dir[0]; ++x)
        {
            pt[0] = xlow[0] + x * (xup[0] - xlow[0]) / static_cast<double>(num_pts_per_dir[0]);
            for (int y = 0; y < num_pts_per_dir[1]; ++y)
            {
                pt[1] = xlow[1] + y * (xup[1] - xlow[1]) / static_cast<double>(num_pts_per_dir[1]);
#if (NDIM == 2)
                pt_list.push_back(pt);
#endif
#if (NDIM == 3)
                for (int z = 0; z < num_pts_per_dir[2]; ++z)
                {
                    pt[2] = xlow[2] + z * (xup[2] - xlow[2]) / static_cast<double>(num_pts_per_dir[2]);
                    pt_list.push_back(pt);
                }
#endif
            }
        }

        PointList point_list("PointList");
        point_list.insertPoints(pt_list.begin(), pt_list.end(), ACTIVE, true);

        std::ofstream out;
        out.open("output");
        point_list.printPointList(out);
        out.close();

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
