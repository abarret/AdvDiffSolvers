/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "NodeData.h"

namespace ADS
{
namespace Reconstruct
{
double
sum_over_z_splines(const IBTK::VectorNd& x_loc,
                   const CellIndex<NDIM>& idx,
                   const CellData<NDIM, double>& Q_data,
                   const int order)
{
    double val = 0.0;
    Box<NDIM> box(idx, idx);
    box.grow(get_spline_width(order) + 1);
    const Box<NDIM>& ghost_box = Q_data.getGhostBox();
    TBOX_ASSERT(ghost_box.contains(box));
    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx_c = ci();
        VectorNd xx;
        for (int d = 0; d < NDIM; ++d) xx(d) = idx_c(d) + 0.5;
        val += Q_data(idx_c) * evaluate_z_spline(x_loc - xx, order);
    }
    return val;
}

bool
index_within_width(const int stencil_width, const CellIndex<NDIM>& idx, const CellData<NDIM, double>& vol_data)
{
    bool withinWidth = true;
    Box<NDIM> check_box(idx, idx);
    check_box.grow(stencil_width);
    for (CellIterator<NDIM> i(check_box); i; i++)
    {
        const CellIndex<NDIM>& idx_c = i();
        if (vol_data(idx_c) < 1.0) withinWidth = false;
    }
    return withinWidth;
}

double
evaluate_z_spline(const VectorNd x, const int order)
{
    double val = 1.0;
    for (int d = 0; d < NDIM; ++d)
    {
        val *= z_spline(x(d), order);
    }
    return val;
}

int
get_spline_width(const int order)
{
    return order + 1;
}

double
z_spline(double x, const int order)
{
    x = std::fabs(x);
    switch (order)
    {
    case 0:
        if (x < 1.0)
            return 1.0 - x;
        else
            return 0.0;
    case 1:
        if (x < 1.0)
            return 1.0 - 2.5 * x * x + 1.5 * x * x * x;
        else if (x < 2.0)
            return 0.5 * (2.0 - x) * (2.0 - x) * (1.0 - x);
        else
            return 0.0;
    case 2:
        if (x < 1.0)
            return 1.0 - 15.0 / 12.0 * x * x - 35.0 / 12.0 * x * x * x + 63.0 / 12.0 * x * x * x * x -
                   25.0 / 12.0 * x * x * x * x * x;
        else if (x < 2.0)
            return -4.0 + 75.0 / 4.0 * x - 245.0 / 8.0 * x * x + 545.0 / 24.0 * x * x * x - 63.0 / 8.0 * x * x * x * x +
                   25.0 / 24.0 * x * x * x * x * x;
        else if (x < 3.0)
            return 18.0 - 153.0 / 4.0 * x + 255.0 / 8.0 * x * x - 313.0 / 24.0 * x * x * x +
                   21.0 / 8.0 * x * x * x * x - 5.0 / 24.0 * x * x * x * x * x;
        else
            return 0.0;
    default:
        TBOX_ERROR("Unavailable order: " << order << "\n");
        return 0.0;
    }
}

double
radial_basis_function_reconstruction(IBTK::VectorNd x_loc,
                                     const CellIndex<NDIM>& idx,
                                     const CellData<NDIM, double>& Q_data,
                                     const CellData<NDIM, double>& vol_data,
                                     const NodeData<NDIM, double>& ls_data,
                                     const Pointer<Patch<NDIM>>& patch,
                                     const RBFPolyOrder order,
                                     const unsigned int stencil_size)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();

    const CellIndex<NDIM>& idx_low = patch->getBox().lower();

    for (int d = 0; d < NDIM; ++d) x_loc[d] = xlow[d] + dx[d] * (x_loc[d] - static_cast<double>(idx_low(d)));

    // If we use a linear polynomial, include 6 closest points.
    // If we use a quadratic polynomial, include 14 closest points.
    // Use flooding to find points
    std::vector<CellIndex<NDIM>> new_idxs = { idx };
    std::vector<VectorNd> X_vals;
    std::vector<double> Q_vals;
    unsigned int i = 0;
    while (X_vals.size() < stencil_size)
    {
#ifndef NDEBUG
        TBOX_ASSERT(i < new_idxs.size());
#endif
        CellIndex<NDIM> new_idx = new_idxs[i];
        // Add new idx to list of X_vals
        if (vol_data(new_idx) > 0.0)
        {
            Q_vals.push_back(Q_data(new_idx));
            VectorNd x_cent_c = ADS::find_cell_centroid(new_idx, ls_data);
            for (int d = 0; d < NDIM; ++d)
                x_cent_c[d] = xlow[d] + dx[d] * (x_cent_c[d] - static_cast<double>(idx_low(d)));
            X_vals.push_back(x_cent_c);
        }
        // Add neighboring points to new_idxs
        IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
        CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
        CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
        if (vol_data(idx_l) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_l) == new_idxs.end()))
            new_idxs.push_back(idx_l);
        if (vol_data(idx_r) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_r) == new_idxs.end()))
            new_idxs.push_back(idx_r);
        if (vol_data(idx_u) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_u) == new_idxs.end()))
            new_idxs.push_back(idx_u);
        if (vol_data(idx_b) > 0.0 && (std::find(new_idxs.begin(), new_idxs.end(), idx_b) == new_idxs.end()))
            new_idxs.push_back(idx_b);
        ++i;
    }

    return radial_basis_function_reconstruction(x_loc, X_vals, Q_vals, order);
}

double
least_squares_reconstruction(IBTK::VectorNd x_loc,
                             const CellIndex<NDIM>& idx,
                             const CellData<NDIM, double>& Q_data,
                             const CellData<NDIM, double>& vol_data,
                             const NodeData<NDIM, double>& ls_data,
                             const Pointer<Patch<NDIM>>& patch,
                             LeastSquaresOrder order)
{
#if (NDIM == 3)
    TBOX_ERROR("MLS reconstruction not implemented for 3 spatial dimensions. Use RBF reconstruction.\n");
#endif
    int size = 0;
    int box_size = 0;
    switch (order)
    {
    case LeastSquaresOrder::CONSTANT:
        size = 1;
        box_size = 1;
        break;
    case LeastSquaresOrder::LINEAR:
        size = 1 + NDIM;
        box_size = 2;
        break;
    case LeastSquaresOrder::QUADRATIC:
        size = 3 * NDIM;
        box_size = 3;
        break;
    case LeastSquaresOrder::CUBIC:
        size = 10;
        box_size = 4;
        break;
    case LeastSquaresOrder::UNKNOWN_ORDER:
        TBOX_ERROR("Unknown order.");
        break;
    }
    Box<NDIM> box(idx, idx);
    box.grow(box_size);
#ifndef NDEBUG
    TBOX_ASSERT(ls_data.getGhostBox().contains(box));
    TBOX_ASSERT(Q_data.getGhostBox().contains(box));
    TBOX_ASSERT(vol_data.getGhostBox().contains(box));
#endif

    std::vector<double> Q_vals;
    std::vector<VectorNd> X_vals;

    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx_c = ci();
        if (vol_data(idx_c) > 0.0)
        {
            // Use this point to calculate least squares reconstruction.
            // Find cell center
            VectorNd x_cent_c = ADS::find_cell_centroid(idx_c, ls_data);
            Q_vals.push_back(Q_data(idx_c));
            X_vals.push_back(x_cent_c);
        }
    }
    const int m = Q_vals.size();
    MatrixXd A(MatrixXd::Zero(m, size)), Lambda(MatrixXd::Zero(m, m));
    VectorXd U(VectorXd::Zero(m));
    for (size_t i = 0; i < Q_vals.size(); ++i)
    {
        U(i) = Q_vals[i];
        const VectorNd X = X_vals[i] - x_loc;
        Lambda(i, i) = std::sqrt(mls_weight(static_cast<double>((X_vals[i] - x_loc).norm())));
        switch (order)
        {
        case LeastSquaresOrder::CUBIC:
            A(i, 9) = X[1] * X[1] * X[1];
            A(i, 8) = X[1] * X[1] * X[0];
            A(i, 7) = X[1] * X[0] * X[0];
            A(i, 6) = X[0] * X[0] * X[0];
            /* FALLTHROUGH */
        case LeastSquaresOrder::QUADRATIC:
            A(i, 5) = X[1] * X[1];
            A(i, 4) = X[0] * X[1];
            A(i, 3) = X[0] * X[0];
            /* FALLTHROUGH */
        case LeastSquaresOrder::LINEAR:
            A(i, 2) = X[1];
            A(i, 1) = X[0];
            /* FALLTHROUGH */
        case LeastSquaresOrder::CONSTANT:
            A(i, 0) = 1.0;
            break;
        case LeastSquaresOrder::UNKNOWN_ORDER:
            TBOX_ERROR("Unknown order.");
            break;
        }
    }

    VectorXd x = (Lambda * A).fullPivHouseholderQr().solve(Lambda * U);
    return x(0);
}

double
bilinear_reconstruction(const VectorNd& x_loc,
                        const VectorNd& x_ll,
                        const CellIndex<NDIM>& idx_ll,
                        const CellData<NDIM, double>& Q_data,
                        const double* const dx)
{
    double q00 = Q_data(idx_ll);
    double q01 = Q_data(idx_ll + IntVector<NDIM>(0, 1));
    double q10 = Q_data(idx_ll + IntVector<NDIM>(1, 0));
    double q11 = Q_data(idx_ll + IntVector<NDIM>(1, 1));
    return q00 + (q10 - q00) * (x_loc[0] - x_ll[0]) / dx[0] + (q01 - q00) * (x_loc[1] - x_ll[1]) / dx[1] +
           (q11 - q10 - q01 + q00) * (x_loc[1] - x_ll[1]) * (x_loc[0] - x_ll[0]) / (dx[0] * dx[1]);
}

double
radial_basis_function_reconstruction(IBTK::VectorNd x_loc,
                                     const double ls_val,
                                     const CellIndex<NDIM>& idx,
                                     const CellData<NDIM, double>& Q_data,
                                     const NodeData<NDIM, double>& ls_data,
                                     const Pointer<Patch<NDIM>>& patch,
                                     const RBFPolyOrder order,
                                     const unsigned int stencil_size)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();

    const CellIndex<NDIM>& idx_low = patch->getBox().lower();

    // Convert to current coordinates
    for (int d = 0; d < NDIM; ++d) x_loc[d] = xlow[d] + dx[d] * (x_loc[d] - static_cast<double>(idx_low(d)));

    // If we use a linear polynomial, include 6 closest points.
    // If we use a quadratic polynomial, include 14 closest points.
    // Use flooding to find points
    std::vector<CellIndex<NDIM>> stencil_idxs;
    try
    {
        flood_fill_for_points(stencil_idxs, idx, ls_data, ls_val, stencil_size);
    }
    catch (const std::runtime_error& e)
    {
        pout << e.what() << "\n";
        TBOX_ERROR("radialBasisFunctionReconstruction(): Could not find enough cells to perform reconstruction\n");
    }
    std::vector<VectorNd> X_vals;
    std::vector<double> Q_vals;
    for (const auto& idx : stencil_idxs)
    {
        Q_vals.push_back(Q_data(idx));
        VectorNd x_cent_c;
        for (int d = 0; d < NDIM; ++d) x_cent_c[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
        X_vals.push_back(x_cent_c);
    }

    return radial_basis_function_reconstruction(x_loc, X_vals, Q_vals, order);
}

double
radial_basis_function_reconstruction(const IBTK::VectorNd& x_loc,
                                     const std::vector<IBTK::VectorNd>& X_pts,
                                     const std::vector<double>& Q_vals,
                                     const RBFPolyOrder order)
{
    int poly_size = 0;
    switch (order)
    {
    case RBFPolyOrder::LINEAR:
        poly_size = NDIM + 1;
        break;
    case RBFPolyOrder::QUADRATIC:
        poly_size = 2 * NDIM + 2;
        break;
    default:
        TBOX_ERROR("Unknown polynomial order: " << ADS::enum_to_string(order) << "\n");
    }

    const int m = Q_vals.size();
    MatrixXd A(MatrixXd::Zero(m, m));
    MatrixXd B(MatrixXd::Zero(m, poly_size));
    VectorXd U(VectorXd::Zero(m + poly_size));
    for (size_t i = 0; i < Q_vals.size(); ++i)
    {
        for (size_t j = 0; j < Q_vals.size(); ++j)
        {
            const VectorNd X = X_pts[i] - X_pts[j];
            A(i, j) = rbf(X.norm());
        }
        B(i, 0) = 1.0;
        for (int d = 0; d < NDIM; ++d) B(i, d + 1) = X_pts[i](d);
        if (order == RBFPolyOrder::QUADRATIC)
        {
            B(i, NDIM + 1) = X_pts[i](0) * X_pts[i](0);
            B(i, NDIM + 2) = X_pts[i](1) * X_pts[i](1);
            B(i, NDIM + 3) = X_pts[i](0) * X_pts[i](1);
        }
        U(i) = Q_vals[i];
    }

    MatrixXd final_mat(MatrixXd::Zero(m + poly_size, m + poly_size));
    final_mat.block(0, 0, m, m) = A;
    final_mat.block(0, m, m, poly_size) = B;
    final_mat.block(m, 0, poly_size, m) = B.transpose();

    VectorXd x = final_mat.fullPivHouseholderQr().solve(U);
    double val = 0.0;
    VectorXd rbf_coefs = x.block(0, 0, m, 1);
    VectorXd poly_coefs = x.block(m, 0, poly_size, 1);
    VectorXd poly_vec = VectorXd::Ones(poly_size);
    for (int d = 0; d < NDIM; ++d) poly_vec(d + 1) = x_loc(d);
    if (order == RBFPolyOrder::QUADRATIC)
    {
        poly_vec(NDIM + 1) = x_loc(0) * x_loc(0);
        poly_vec(NDIM + 2) = x_loc(1) * x_loc(1);
        poly_vec(NDIM + 3) = x_loc(0) * x_loc(1);
    }
    for (size_t i = 0; i < X_pts.size(); ++i)
    {
        val += rbf_coefs[i] * rbf((X_pts[i] - x_loc).norm());
    }
    val += poly_coefs.dot(poly_vec);
    return val;
}

} // namespace Reconstruct
} // namespace ADS
