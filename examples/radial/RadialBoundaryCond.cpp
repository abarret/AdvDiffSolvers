#include "LS/utility_functions.h"

#include "RadialBoundaryCond.h"

namespace
{
static Timer* s_apply_timer = nullptr;
}
RadialBoundaryCond::RadialBoundaryCond(const std::string& object_name, Pointer<Database> input_db)
    : LSCutCellBoundaryConditions(object_name)
{
    d_a = input_db->getDouble("a_coef");
    d_D_coef = input_db->getDouble("D");
    d_R = input_db->getDouble("R");
    input_db->getDoubleArray("Center", d_center.data(), NDIM);
    d_vel.resize(NDIM);
    input_db->getDoubleArray("velocity", d_vel.data(), NDIM);
    IBAMR_DO_ONCE(s_apply_timer =
                      TimerManager::getManager()->getTimer("LS::RadialBoundaryCond::applyBoundaryCondition"));
}

void
RadialBoundaryCond::applyBoundaryCondition(Pointer<CellVariable<NDIM, double>> Q_var,
                                           const int Q_idx,
                                           Pointer<CellVariable<NDIM, double>> R_var,
                                           const int R_idx,
                                           Pointer<PatchHierarchy<NDIM>> hierarchy,
                                           const double time)
{
    LS_TIMER_START(s_apply_timer);
    TBOX_ASSERT(d_ls_var && d_vol_var && d_area_var);
    TBOX_ASSERT(d_ls_idx > 0 && d_vol_idx > 0 && d_area_idx > 0);

    auto g = [this](VectorNd x, double t) -> double {
        x = x - d_center;
        for (int d = 0; d < NDIM; ++d) x(d) -= t * d_vel[d];
        VectorNd n = x.normalized();
        return std::exp(-2.0 * d_D_coef * M_PI * M_PI * t) *
               (-d_a * std::cos(M_PI * x(0)) * std::cos(M_PI * x(1)) +
                d_D_coef * M_PI * n(0) * std::cos(M_PI * x(1)) * std::sin(M_PI * x(0)) +
                d_D_coef * M_PI * n(1) * std::cos(M_PI * x(0)) * std::sin(M_PI * x(1)));
    };

    const double sgn = d_D / std::abs(d_D);
    double pre_fac = sgn * (d_ts_type == LS::DiffusionTimeIntegrationMethod::TRAPEZOIDAL_RULE ? 0.5 : 1.0);
    if (d_D == 0.0) pre_fac = 0.0;

    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = box.lower();

            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> R_data = patch->getPatchData(R_idx);
            Pointer<CellData<NDIM, double>> area_data = patch->getPatchData(d_area_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const double cell_volume = dx[0] * dx[1] *
#if (NDIM == 3)
                                           dx[2] *
#endif
                                           (*vol_data)(idx);
                const double area = (*area_data)(idx);
                if (area > 0.0)
                {
                    TBOX_ASSERT(cell_volume > 0.0);
                    for (int l = 0; l < Q_data->getDepth(); ++l)
                    {
                        if (!d_homogeneous_bdry)
                        {
                            // Find midpoint on line
                            NodeIndex<NDIM> idx_ll(idx, NodeIndex<NDIM>::LowerLeft);
                            NodeIndex<NDIM> idx_ul(idx, NodeIndex<NDIM>::LowerRight);
                            NodeIndex<NDIM> idx_uu(idx, NodeIndex<NDIM>::UpperRight);
                            NodeIndex<NDIM> idx_lu(idx, NodeIndex<NDIM>::UpperLeft);
                            VectorNd x_ll, x_ul, x_uu, x_lu;
                            for (int d = 0; d < NDIM; ++d)
                            {
                                x_ll(d) = xlow[d] + dx[d] * static_cast<double>(idx_ll(d) - idx_low(d));
                                x_ul(d) = xlow[d] + dx[d] * static_cast<double>(idx_ul(d) - idx_low(d));
                                x_uu(d) = xlow[d] + dx[d] * static_cast<double>(idx_uu(d) - idx_low(d));
                                x_lu(d) = xlow[d] + dx[d] * static_cast<double>(idx_lu(d) - idx_low(d));
                            }
                            double phi_ll = (*ls_data)(idx_ll);
                            double phi_ul = (*ls_data)(idx_ul);
                            double phi_uu = (*ls_data)(idx_uu);
                            double phi_lu = (*ls_data)(idx_lu);
                            if (std::abs(phi_ll) < 1.0e-12) phi_ll = std::copysign(1.0e-12, phi_ll);
                            if (std::abs(phi_ul) < 1.0e-12) phi_ul = std::copysign(1.0e-12, phi_ul);
                            if (std::abs(phi_uu) < 1.0e-12) phi_uu = std::copysign(1.0e-12, phi_uu);
                            if (std::abs(phi_lu) < 1.0e-12) phi_lu = std::copysign(1.0e-12, phi_lu);
                            std::vector<VectorNd> x_pts;
                            if (phi_ll * phi_ul < 0.0) x_pts.push_back(LS::midpoint_value(x_ll, phi_ll, x_ul, phi_ul));
                            if (phi_ul * phi_uu < 0.0) x_pts.push_back(LS::midpoint_value(x_ul, phi_ul, x_uu, phi_uu));
                            if (phi_uu * phi_lu < 0.0) x_pts.push_back(LS::midpoint_value(x_uu, phi_uu, x_lu, phi_lu));
                            if (phi_lu * phi_ll < 0.0) x_pts.push_back(LS::midpoint_value(x_lu, phi_lu, x_ll, phi_ll));
                            TBOX_ASSERT(x_pts.size() == 2);
                            VectorNd X_mid = 0.5 * (x_pts[0] + x_pts[1]);
                            double g_val = g(X_mid, time);
                            (*R_data)(idx, l) += pre_fac * g_val * area / cell_volume;
                        }
                        (*R_data)(idx, l) -= pre_fac * d_a * (*Q_data)(idx, l) * area / cell_volume;
                    }
                }
            }
        }
    }
    LS_TIMER_STOP(s_apply_timer);
}
