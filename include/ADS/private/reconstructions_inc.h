#ifndef ADS_reconstructions_inc
#define ADS_reconstructions_inc

#include <ADS/PolynomialBasis.h>
#include <ADS/ls_functions.h>
#include <ADS/reconstructions.h>

#include <Eigen/Dense>

#include <exception>

namespace ADS
{
namespace Reconstruct
{
inline void
flood_fill_for_points(std::vector<SAMRAI::pdat::CellIndex<NDIM>>& fill_pts,
                      const SAMRAI::pdat::CellIndex<NDIM>& idx,
                      const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                      const double ls,
                      const size_t stencil_size)
{
    // Flood fill for Eulerian points
    std::vector<SAMRAI::pdat::CellIndex<NDIM>> test_idxs = { idx };
    unsigned int i = 0;
    while (fill_pts.size() < stencil_size)
    {
#ifndef NDEBUG
        if (i >= test_idxs.size())
        {
            std::ostringstream err_msg;
            err_msg << "Could not find enough cells to fill stencil.\n";
            err_msg << "  Starting at base point " << idx << "\n";
            err_msg << "  ls value: " << ls << "\n";
            err_msg << "  Searched " << i << " indices and found " << fill_pts.size() << " total pts\n";
            throw std::runtime_error(err_msg.str());
        }
#endif
        const SAMRAI::pdat::CellIndex<NDIM>& new_idx = test_idxs[i];
        // Add new idx to list of X_vals
        if (ADS::node_to_cell(new_idx, ls_data) * ls > 0.0) fill_pts.push_back(new_idx);

        // Add neighboring points to new_idxs.
        SAMRAI::hier::IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
        SAMRAI::pdat::CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
        SAMRAI::pdat::CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
        if (ADS::node_to_cell(idx_l, ls_data) * ls > 0.0 &&
            (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end()))
            test_idxs.push_back(idx_l);
        if (ADS::node_to_cell(idx_r, ls_data) * ls > 0.0 &&
            (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end()))
            test_idxs.push_back(idx_r);
        if (ADS::node_to_cell(idx_u, ls_data) * ls > 0.0 &&
            (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end()))
            test_idxs.push_back(idx_u);
        if (ADS::node_to_cell(idx_b, ls_data) * ls > 0.0 &&
            (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end()))
            test_idxs.push_back(idx_b);
        ++i;
    }
}

inline void
flood_fill_for_points(std::vector<SAMRAI::pdat::SideIndex<NDIM>>& fill_pts,
                      const SAMRAI::pdat::CellIndex<NDIM>& idx,
                      const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                      const double ls,
                      const int axis,
                      const size_t stencil_size)
{
    // Flood fill for Eulerian points
    std::vector<SAMRAI::pdat::SideIndex<NDIM>> test_idxs = { SAMRAI::pdat::SideIndex<NDIM>(idx, axis, 0),
                                                             SAMRAI::pdat::SideIndex<NDIM>(idx, axis, 1) };
    unsigned int i = 0;
    while (fill_pts.size() < stencil_size)
    {
#ifndef NDEBUG
        if (i >= test_idxs.size())
        {
            std::ostringstream err_msg;
            err_msg << "Could not find enough cells to fill stencil.\n";
            err_msg << "  Starting at base point " << idx << "\n";
            err_msg << "  ls value: " << ls << "\n";
            err_msg << "  Searched " << i << " indices and found " << fill_pts.size() << " total pts\n";
            throw std::runtime_error(err_msg.str());
        }
#endif
        const SAMRAI::pdat::SideIndex<NDIM>& new_idx = test_idxs[i];
        // Add new idx to list of X_vals
        if (ADS::node_to_side(new_idx, ls_data) * ls > 0.0) fill_pts.push_back(new_idx);

        // Add neighboring points to new_idxs.
        // Prefer to add points preferentially depending on the axis
        SAMRAI::hier::IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
        SAMRAI::pdat::SideIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
        SAMRAI::pdat::SideIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
        if (axis == 0)
        {
            // Add x points first
            if (ADS::node_to_side(idx_l, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end()))
                test_idxs.push_back(idx_l);
            if (ADS::node_to_side(idx_r, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end()))
                test_idxs.push_back(idx_r);
            if (ADS::node_to_side(idx_u, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end()))
                test_idxs.push_back(idx_u);
            if (ADS::node_to_side(idx_b, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end()))
                test_idxs.push_back(idx_b);
        }
        else
        {
            // Add y points first.
            if (ADS::node_to_side(idx_u, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_u) == test_idxs.end()))
                test_idxs.push_back(idx_u);
            if (ADS::node_to_side(idx_b, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_b) == test_idxs.end()))
                test_idxs.push_back(idx_b);
            if (ADS::node_to_side(idx_l, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_l) == test_idxs.end()))
                test_idxs.push_back(idx_l);
            if (ADS::node_to_side(idx_r, ls_data) * ls > 0.0 &&
                (std::find(test_idxs.begin(), test_idxs.end(), idx_r) == test_idxs.end()))
                test_idxs.push_back(idx_r);
        }
        ++i;
    }
}

template <class Point>
void
RBFFD_reconstruct(std::vector<double>& wgts,
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

inline double
quadratic_lagrange_interpolant(IBTK::VectorNd x,
                               const SAMRAI::pdat::CellIndex<NDIM>& idx,
                               const SAMRAI::pdat::CellData<NDIM, double>& Q_data)
{
    for (int d = 0; d < NDIM; ++d) x[d] -= static_cast<double>(idx(d)) + 0.5;
    SAMRAI::hier::IntVector<NDIM> one_x(1, 0), one_y(0, 1);
    return Q_data(idx) * (x[0] - 1.0) * (x[0] + 1.0) * (x[1] - 1.0) * (x[1] + 1.0) -
           Q_data(idx + one_x) * 0.5 * x[0] * (x[0] + 1.0) * (x[1] - 1.0) * (x[1] + 1.0) -
           Q_data(idx - one_x) * 0.5 * x[0] * (x[0] - 1.0) * (x[1] - 1.0) * (x[1] + 1.0) -
           Q_data(idx + one_y) * 0.5 * x[1] * (x[1] + 1.0) * (x[0] - 1.0) * (x[0] + 1.0) -
           Q_data(idx - one_y) * 0.5 * x[1] * (x[1] - 1.0) * (x[0] - 1.0) * (x[0] + 1.0);
}

inline double
quadratic_lagrange_interpolant_limited(IBTK::VectorNd x,
                                       const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                       const SAMRAI::pdat::CellData<NDIM, double>& Q_data)
{
    double Q = quadratic_lagrange_interpolant(x, idx, Q_data);

    SAMRAI::hier::IntVector<NDIM> one_x(1, 0), one_y(0, 1);
    SAMRAI::pdat::CellIndex<NDIM> ll;
    for (int d = 0; d < NDIM; ++d) ll(d) = std::round(x[d]) - 1.0;
    double q00 = Q_data(ll);
    double q10 = Q_data(ll + one_x);
    double q01 = Q_data(ll + one_y);
    double q11 = Q_data(ll + one_x + one_y);
    if (Q > std::max({ q00, q10, q01, q11 }) || Q < std::min({ q00, q10, q01, q11 }))
    {
        // Need to potentially "reshift" x if it's the below idx.
        for (int d = 0; d < NDIM; ++d) x[d] = x[d] - (static_cast<double>(ll(d)) + 0.5);
        Q = Q_data(ll) * (x[0] - 1.0) * (x[1] - 1.0) - Q_data(ll + one_y) * x[1] * (x[0] - 1.0) -
            Q_data(ll + one_x) * x[0] * (x[1] - 1.0) + Q_data(ll + one_x + one_y) * x[0] * x[1];
    }

    return Q;
}

inline double
divergence(const IBTK::VectorNd& x,
           const SAMRAI::pdat::CellIndex<NDIM>& idx,
           const double ls,
           const SAMRAI::pdat::SideData<NDIM, double>& u_data,
           const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
           const RBFPolyOrder poly_order,
           const int stencil_size,
           const double* const dx)
{
    double div = 0.0;
    for (int d = 0; d < NDIM; ++d)
    {
        // Need to grab indices, use flood filling
        std::vector<SAMRAI::pdat::SideIndex<NDIM>> idxs;
        flood_fill_for_points(idxs, idx, ls_data, ls, d, stencil_size);

        std::vector<IBTK::VectorNd> X_pts;
        std::vector<double> u_vals;
        for (const auto& si : idxs)
        {
            u_vals.push_back(u_data(si));
            IBTK::VectorNd xpt;
            for (int i = 0; i < NDIM; ++i) xpt[i] = static_cast<double>(si(i)) + (d == i ? 0.0 : 0.5);
            X_pts.push_back(xpt);
        }

        // We have all the points. Now determine weights.
        auto rbf = [](const double r) -> double { return r * r * r * r * r; };
        auto L_rbf = [](const IBTK::VectorNd& xi, const IBTK::VectorNd& xj, void* ctx) -> double
        {
            int d = *static_cast<int*>(ctx);
            double r = (xi - xj).norm();
            return 5.0 * (xi[d] - xj[d]) * r * r * r;
        };

        auto L_polys = [](const std::vector<IBTK::VectorNd>& xpts,
                          int poly_degree,
                          double dx,
                          IBTK::VectorNd base_pt,
                          void* ctx) -> IBTK::VectorXd
        {
            int d = *static_cast<int*>(ctx);
            if (d == 0)
                return PolynomialBasis::dPdxMonomials(xpts, poly_degree, dx, base_pt).transpose();
            else
                return PolynomialBasis::dPdyMonomials(xpts, poly_degree, dx, base_pt).transpose();
        };

        // Need to shift x_loc by idx_low
        std::vector<double> wgts;
        std::vector<double> dummy_dx = { 1.0, 1.0 };
        Reconstruct::RBFFD_reconstruct<IBTK::VectorNd>(wgts,
                                                       x,
                                                       X_pts,
                                                       poly_order == Reconstruct::RBFPolyOrder::LINEAR ? 1 : 2,
                                                       dummy_dx.data(),
                                                       rbf,
                                                       L_rbf,
                                                       static_cast<void*>(&d),
                                                       L_polys,
                                                       static_cast<void*>(&d));
        // Now perform reconstruction.
        for (size_t i = 0; i < u_vals.size(); ++i) div += u_vals[i] * wgts[i] / dx[d];
    }
    return div;
}

} // namespace Reconstruct
} // namespace ADS
#endif
