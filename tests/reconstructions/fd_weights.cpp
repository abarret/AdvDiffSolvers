#include <ibamr/config.h>

#include <ADS/Point.h>
#include <ADS/app_namespaces.h>
#include <ADS/reconstructions.h>

#include <ibtk/IBTKInit.h>
#include <ibtk/ibtk_utilities.h>

#include <random>
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
    {
        auto f = [](const VectorNd& x) -> double {
            return std::sin(2.0 * M_PI * x(0)) * std::sin(2.0 * M_PI * x(1))
#if (NDIM == 3)
                   * std::sin(2.0 * M_PI * x(2))
#endif
                ;
        };

        auto Lf = [](const VectorNd& x) -> double {
#if (NDIM == 2)
            return (1.0 - 8.0 * M_PI * M_PI) * std::sin(2.0 * M_PI * x(0)) * std::sin(2.0 * M_PI * x(1));
#endif
#if (NDIM == 3)
            return (1.0 - 12.0 * M_PI * M_PI) * std::sin(2.0 * M_PI * x(0)) * std::sin(2.0 * M_PI * x(1)) *
                   std::sin(2.0 * M_PI * x(2));
#endif
        };
        // RBF
        auto rbf = [](const double r) -> double { return PolynomialBasis::pow(r, 5.0); };

        auto L_rbf = [](const ADS::Point& pt0, const ADS::Point& pti, void*) -> double {
#if (NDIM == 2)
            return PolynomialBasis::pow((pt0 - pti).norm(), 5.0) + 25.0 * PolynomialBasis::pow((pt0 - pti).norm(), 4.0);
#endif
#if (NDIM == 3)
            return PolynomialBasis::pow((pt0 - pti).norm(), 5.0) + 30.0 * PolynomialBasis::pow((pt0 - pti).norm(), 4.0);
#endif
        };

        auto L_poly = [](std::vector<ADS::Point> pts, int deg, double ds, const ADS::Point& shft, void*) -> VectorXd {
            return (PolynomialBasis::formMonomials(pts, deg, ds, shft) +
                    PolynomialBasis::laplacianMonomials(pts, deg, ds, shft))
                .transpose();
        };

        VectorNd base_pt(VectorNd::Zero());
        base_pt(0) = 0.2;
        base_pt(1) = 0.9;
        std::vector<VectorNd> base_fd_pts;
#if (NDIM == 2)
        int num_pts = 15;
#endif
#if (NDIM == 3)
        int num_pts = 22;
#endif
        // Read in list of points from input file
        ifstream infile(argv[1]);
        base_fd_pts.push_back(base_pt);
        for (int j = 0; j < (num_pts - 1); ++j)
        {
            VectorNd x;
            for (int d = 0; d < NDIM; ++d) infile >> x[d];
            base_fd_pts.push_back(x);
        }
        infile.close();
        ofstream outfile("output");

        unsigned int n_runs = 10;
        std::vector<double> hh = { 0.1 };
        std::vector<double> errs(n_runs);
        for (size_t i = 1; i < n_runs; ++i) hh.push_back(hh[i - 1] * 0.5);
        for (size_t i = 0; i < n_runs; ++i)
        {
            double h = hh[i];
            // Generate set of points.
            std::vector<double> dx;
            for (size_t d = 0; d < NDIM; ++d) dx.push_back(h);

            std::vector<ADS::Point> fd_pts(num_pts);
            for (int j = 0; j < num_pts; ++j) fd_pts[j] = ADS::Point(h * (base_fd_pts[j] - base_pt) + base_pt);

            std::vector<double> wgts;
            Reconstruct::RBFFD_reconstruct<ADS::Point>(
                wgts, ADS::Point(base_pt), fd_pts, 3, dx.data(), rbf, L_rbf, nullptr, L_poly, nullptr);

            double approx = 0.0;
            for (size_t i = 0; i < wgts.size(); ++i) approx += f(fd_pts[i].getVec()) * wgts[i];
            double exact = Lf(base_pt);
            errs[i] = std::abs(exact - approx);
            outfile << "exact: " << exact << " approx: " << approx << " error: " << errs[i]
                    << " rel error: " << errs[i] / exact << "\n";
        }
        outfile.close();
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
