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
} // namespace Reconstruct

#endif
