#ifndef ADS_reconstructions_inc
#define ADS_reconstructions_inc

#include <ADS/PolynomialBasis.h>
#include <ADS/ls_functions.h>
#include <ADS/reconstructions.h>

#include <Eigen/Dense>

namespace Reconstruct
{
template <class Point>
void
RBFFDReconstruct(std::vector<double>& wgts,
                 const Point& base_pt,
                 const std::vector<Point>& fd_pts,
                 const int poly_degree,
                 const double* const dx,
                 std::function<double(double)> rbf,
                 std::function<double(const Point&, const Point&, void*)> L_rbf,
                 void* rbf_ctx,
                 std::function<IBTK::VectorXd(const std::vector<Point>&, int, double, const Point&, void*)> L_polys,
                 void* poly_ctx)
{
    const int stencil_size = fd_pts.size();
    IBTK::MatrixXd A(IBTK::MatrixXd::Zero(stencil_size, stencil_size));
    IBTK::MatrixXd B = ADS::PolynomialBasis::formMonomials(fd_pts, poly_degree, dx[0], base_pt);
    const int poly_size = B.cols();
    IBTK::VectorXd U(IBTK::VectorXd::Zero(stencil_size + poly_size));
    for (int i = 0; i < stencil_size; ++i)
    {
        const Point& pti = fd_pts[i];
        for (int j = 0; j < stencil_size; ++j)
        {
            const Point& ptj = fd_pts[j];
            A(i, j) = rbf((pti - ptj).norm());
        }
        // We're in the bulk
        U(i) = L_rbf(base_pt, pti, rbf_ctx);
    }
    std::vector<Point> base_pt_vec = { base_pt };
    IBTK::VectorXd Ulow = L_polys(base_pt_vec, poly_degree, dx[0], base_pt, poly_ctx);
    U.block(stencil_size, 0, Ulow.rows(), 1) = Ulow;
    IBTK::MatrixXd final_mat(IBTK::MatrixXd::Zero(stencil_size + poly_size, stencil_size + poly_size));
    final_mat.block(0, 0, stencil_size, stencil_size) = A;
    final_mat.block(0, stencil_size, stencil_size, poly_size) = B;
    final_mat.block(stencil_size, 0, poly_size, stencil_size) = B.transpose();
    final_mat.block(stencil_size, stencil_size, poly_size, poly_size).setZero();

    IBTK::VectorXd x = final_mat.colPivHouseholderQr().solve(U);

    // Now cache the stencil
    const IBTK::VectorXd& weights = x.block(0, 0, stencil_size, 1);
    wgts.resize(stencil_size);
    for (int i = 0; i < stencil_size; ++i) wgts[i] = weights[i];
}

inline bool
within_weno_stencil(const SAMRAI::pdat::CellIndex<NDIM>& idx, const SAMRAI::pdat::NodeData<NDIM, double>& ls_data)
{
    // WENO stencil is a 5 by 5 grid.
    SAMRAI::hier::Box<NDIM> box(idx, idx);
    box.grow(2);
    for (SAMRAI::pdat::CellIterator<NDIM> ci(box); ci; ci++)
    {
        const SAMRAI::pdat::CellIndex<NDIM>& i = ci();
        const double ls_val = ADS::node_to_cell(i, ls_data);
        if (ls_val * ADS::node_to_cell(idx, ls_data) < 0.0) return false;
    }
    return true;
}

inline double
weno5(const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
      const SAMRAI::pdat::CellIndex<NDIM>& idx,
      const IBTK::VectorNd& x)
{
    std::array<double, 5> Q_x_vals;
    for (int i = -2; i <= 2; ++i)
    {
        SAMRAI::pdat::CellIndex<NDIM> center_idx;
        center_idx(0) = idx(0) + i;
        std::array<double, 5> Q_y_vals;
        for (int j = -2; j <= 2; ++j)
        {
            center_idx(1) = idx(1) + j;
#if (NDIM == 3)
            std::array<double, 5> Q_z_vals;
            for (int k = -2; k <= 2; ++k)
            {
                center_idx(2) = idx(2) + k;
                Q_z_vals[k + 2] = Q_data(center_idx);
            }
            Q_y_vals[j + 2] = Reconstruct::weno5(Q_z_vals, x[2] - (static_cast<double>(idx(2)) + 0.5));
#endif
#if (NDIM == 2)
            Q_y_vals[j + 2] = Q_data(center_idx);
#endif
        }
        Q_x_vals[i + 2] = Reconstruct::weno5(Q_y_vals, x[1] - (static_cast<double>(idx(1)) + 0.5));
    }
    return Reconstruct::weno5(Q_x_vals, x[0] - (static_cast<double>(idx(0)) + 0.5));
}

template <typename Array>
double
weno5(const Array& Q, double xi)
{
    // Candidate interpolants
    std::array<double, 3> f = {
        0.5 * xi * (xi + 1.0) * Q[0] - xi * (2.0 + xi) * Q[1] + 0.5 * (xi + 1.0) * (xi + 2.0) * Q[2],
        Q[1] * 0.5 * (xi - 1.0) * xi + Q[2] * (1.0 - xi) * (1.0 + xi) + Q[3] * 0.5 * xi * (1.0 + xi),
        Q[2] * 0.5 * (-2 + xi) * (-1.0 + xi) + Q[3] * (2 - xi) * xi + 0.5 * (xi - 1.0) * xi * Q[4]
    };
    // Smoothness indicators
    std::array<double, 3> is = { 1.0 / 3.0 *
                                     (10.0 * Q[2] * Q[2] + 4.0 * Q[0] * Q[0] + Q[0] * (11 * Q[2] - 19 * Q[1]) -
                                      31 * Q[2] * Q[1] + 25 * Q[1] * Q[1]),
                                 1.0 / 3.0 *
                                     (13.0 * Q[2] * Q[2] + 4 * Q[1] * Q[1] - 13 * Q[2] * Q[3] + 4 * Q[3] * Q[3] +
                                      Q[1] * (-13 * Q[2] + 5 * Q[3])),
                                 1.0 / 3.0 *
                                     (10 * Q[2] * Q[2] + 25 * Q[3] * Q[3] + 11 * Q[2] * Q[4] + 4 * Q[4] * Q[4] -
                                      Q[3] * (31 * Q[2] + 19 * Q[4])) };
    // Compute weights
    std::array<double, 3> omega_bar = { 1.0 / 12.0 * (2.0 - 3 * xi + xi * xi),
                                        -1.0 / 6.0 * (-2.0 + xi) * (2.0 + xi),
                                        1.0 / 12.0 * (1.0 + xi) * (2.0 + xi) };

    return weno(f, is, omega_bar);
}

template <typename Array>
double
weno(const Array& f, const Array& si, const Array& w_bar)
{
    // The overall accuracy of the interpolant seems very sensitive to the value of eps. Using too small of a value
    // gives extremely biased stencils. This suggests that the way we compute smoothness indicators is probably not
    // correct. We should look into that.
    static double eps = 1.0e-6;
    Array alpha = f;
    std::transform(w_bar.cbegin(),
                   w_bar.cend(),
                   si.cbegin(),
                   alpha.begin(),
                   [](const double& w_bar, const double& si) -> double { return w_bar / std::pow(si + eps, 2.0); });

    double alpha_sum = std::accumulate(alpha.cbegin(), alpha.cend(), 0.0);
    Array w = f;
    std::transform(alpha.cbegin(),
                   alpha.cend(),
                   w.begin(),
                   [alpha_sum](const double& alpha) -> double { return alpha / alpha_sum; });

    // Improve accuracy of weights (following the approach of Henrick, Aslam, and Powers).
    std::transform(
        w.cbegin(),
        w.cend(),
        w_bar.cbegin(),
        w.begin(),
        [](const double& w, const double& w_bar) -> double
        { return w * (w_bar + w_bar * w_bar - 3.0 * w_bar * w + w * w) / (w_bar * w_bar + w * (1.0 - 2.0 * w_bar)); });
    // normalize new weights
    double w_sum = std::accumulate(w.cbegin(), w.cend(), 0.0);
    std::for_each(w.begin(), w.end(), [w_sum](double& w) -> void { w /= w_sum; });

    // Compute interpolant
    return std::inner_product(w.cbegin(), w.cend(), f.cbegin(), 0.0);
}

} // namespace Reconstruct

#endif
