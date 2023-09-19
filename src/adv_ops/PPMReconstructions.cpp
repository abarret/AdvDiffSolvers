/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/PPMReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include <ibtk/PatchMathOps.h>

#include "SAMRAIVectorReal.h"

#include <libmesh/explicit_system.h>

#include <utility>

namespace
{
template <typename T>
double
sgn(T val)
{
    return static_cast<double>((T(0) - val) - (val < T(0)));
}
} // namespace

namespace ADS
{
PPMReconstructions::PPMReconstructions(std::string object_name)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), 2);
    return;
} // PPMReconstructions

PPMReconstructions::~PPMReconstructions()
{
    deallocateOperatorState();
    return;
} // ~PPMReconstructions

void
PPMReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    // TODO: What kind of physical boundary conditions should we use for advection?
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    ghost_cell_comps[0] =
        ITC(d_Q_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, d_bc_coef);
    ghost_cell_comps[1] = ITC(d_cur_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    HierarchyGhostCellInterpolation hier_ghost_cells;
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(d_current_time);
    applyReconstructionLS(d_Q_scr_idx, N_idx, path_idx);
}

void
PPMReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy, double current_time, double new_time)
{
    AdvectiveReconstructionOperator::allocateOperatorState(hierarchy, current_time, new_time);
    d_hierarchy = hierarchy;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_Q_scr_idx)) level->allocatePatchData(d_Q_scr_idx);
    }
    d_is_allocated = true;
}

void
PPMReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();
    if (!d_is_allocated) return;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }
    d_is_allocated = false;
}

void
PPMReconstructions::applyReconstructionLS(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_ls_idx > 0);
    TBOX_ASSERT(d_new_ls_idx > 0);
#endif

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = box.lower();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);

            // BDS bilinear
#if (0)
            Pointer<NodeData<NDIM, double>> corner_vals = new NodeData<NDIM, double>(box, 1 /*depth*/, 0 /*ghosts*/);
            PatchMathOps patch_ops;
            patch_ops.interp(corner_vals, Q_cur_data, patch, false);
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d) - (static_cast<double>(idx(d)) + 0.5);
                const double LL = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerLeft));
                const double LH = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperLeft));
                const double RL = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerRight));
                const double RH = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperRight));

                const double sx = ((RH + RL) - (LH + LL)) / 2.0;
                const double sy = ((LH + RH) - (LL + RL)) / 2.0;
                const double sxy = ((RH - RL) - (LH - LL));

                const double shat = (*Q_cur_data)(idx);

                (*Q_new_data)(idx) = shat + sxy * x_loc[0] * x_loc[1] + sx * x_loc[0] + sy * x_loc[1];
            }
#endif
            // Lagrange quadratic
#if (0)
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d) - (static_cast<double>(idx(d)) + 0.5);
                IntVector<NDIM> one_x(1, 0), one_y(0, 1);
                (*Q_new_data)(idx) =
                    (*Q_cur_data)(idx) * (x_loc[0] - 1.0) * (x_loc[0] + 1.0) * (x_loc[1] - 1.0) * (x_loc[1] + 1.0) -
                    (*Q_cur_data)(idx + one_x) * 0.5 * x_loc[0] * (x_loc[0] + 1.0) * (x_loc[1] - 1.0) *
                        (x_loc[1] + 1.0) -
                    (*Q_cur_data)(idx - one_x) * 0.5 * x_loc[0] * (x_loc[0] - 1.0) * (x_loc[1] - 1.0) *
                        (x_loc[1] + 1.0) -
                    (*Q_cur_data)(idx + one_y) * 0.5 * x_loc[1] * (x_loc[1] + 1.0) * (x_loc[0] - 1.0) *
                        (x_loc[0] + 1.0) -
                    (*Q_cur_data)(idx - one_y) * 0.5 * x_loc[1] * (x_loc[1] - 1.0) * (x_loc[0] - 1.0) *
                        (x_loc[0] + 1.0);
            }
#endif
            // BDS Quadratic
#if (1)
            Pointer<NodeData<NDIM, double>> corner_vals = new NodeData<NDIM, double>(box, 1 /*depth*/, 0 /*ghosts*/);
            PatchMathOps patch_ops;
            patch_ops.interp(corner_vals, Q_cur_data, patch, false);
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x_loc;
                // Shift position so that (0, 0) is the center of the cell
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d) - (static_cast<double>(idx(d)) + 0.5);
                const double LL = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerLeft));
                const double LH = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperLeft));
                const double RL = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::LowerRight));
                const double RH = (*corner_vals)(NodeIndex<NDIM>(idx, NodeIndex<NDIM>::UpperRight));
                const double qij = (*Q_cur_data)(idx);

                double sx = ((RH + RL) - (LH + LL)) / 2.0;
                double sy = ((LH + RH) - (LL + RL)) / 2.0;
                double sxy = ((RH - RL) - (LH - LL));
                double sxx = 0.5 * (1.0 / (8.0) *
                                    (-(*Q_cur_data)(idx - IntVector<NDIM>(2, 0)) +
                                     12.0 * (*Q_cur_data)(idx - IntVector<NDIM>(1, 0)) - 22.0 * (*Q_cur_data)(idx) +
                                     12.0 * (*Q_cur_data)(idx + IntVector<NDIM>(1, 0)) -
                                     (*Q_cur_data)(idx + IntVector<NDIM>(2, 0))));
                double syy = 0.5 * (1.0 / (8.0) *
                                    (-(*Q_cur_data)(idx - IntVector<NDIM>(0, 2)) +
                                     12.0 * (*Q_cur_data)(idx - IntVector<NDIM>(0, 1)) - 22.0 * (*Q_cur_data)(idx) +
                                     12.0 * (*Q_cur_data)(idx + IntVector<NDIM>(0, 1)) -
                                     (*Q_cur_data)(idx + IntVector<NDIM>(0, 2))));
                double shat = qij - 1.0 / 12.0 * (sxx + syy);

                auto p = [&shat, &sx, &sy, &sxy, &sxx, &syy](const double x, const double y) -> double
                { return shat + sx * x + sy * y + sxy * x * y + sxx * x * x + syy * y * y; };

                // Returns true if ref is the largest or smallest of the values
                auto bds_check =
                    [](const double ref, const double LL, const double LH, const double RL, const double RH) -> bool {
                    return (ref > std::max({ LL, LH, RL, RH }) || ref < std::min({ LL, LH, RL, RH }));
                };
#if (1)
                (*Q_new_data)(idx) = p(x_loc[0], x_loc[1]);
#else
                // Now limit the reconstruction. This occurs in three steps.
                // Skip step 1. Limiting the linear terms with semi-Lagrangian is bad.
                // Step 2.
                double cmp = std::min(std::abs(sx + sxy * 0.5), std::abs(sx - sxy * 0.5));
                bool t1xx = false, t2xx = false, t1yy = false, t2yy = false;
                t1xx = ((sx + 0.5 * sxy) * (sx - 0.5 * sxy)) < 0.0;
                t2xx = !t1xx && cmp < std::abs(sxx);
                t1yy = ((sy + 0.5 * sxy) * (sy - 0.5 * sxy)) < 0.0;
                t2yy = !t1yy && cmp < std::abs(syy);
                if (t1xx && (t1yy || t2yy))
                    sxx = 0.0;
                else if (t2xx && (t1yy || t2yy))
                    sxx = sgn(sxx) * cmp;
                if (t1yy && (t1xx || t2xx))
                    syy = 0.0;
                else if (t2yy && (t1xx || t2xx))
                    syy = sgn(syy) * cmp;
                // Readjust constant
                shat = qij - 1.0 / 12.0 * (sxx + syy);
                // Now check that reconstructed polynomial lies in bounds
                double LL_new = p(-0.5, -0.5), LH_new = p(-0.5, 0.5), RL_new = p(0.5, -0.5), RH_new = p(0.5, 0.5);
                if (!bds_check(LL_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0))) &&
                    !bds_check(LH_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 1))) &&
                    !bds_check(RL_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 0))) &&
                    !bds_check(RH_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 1))))
                {
                    // Our corners are within the neighboring averages. Check that we don't have an extremum on the cell
                    // sides
                    bool done_limiting = true;
                    if (std::abs(sx + sxy * 0.5) < std::abs(sxx))
                    {
                        double x_extr = -(sx + sxy * 0.5) / (2.0 * sxx);
                        if (x_extr > -0.5 && x_extr < 0.5) done_limiting = false;
                    }
                    if (std::abs(sx - sxy * 0.5) < std::abs(sxx))
                    {
                        double x_extr = -(sx - sxy * 0.5) / (2.0 * sxx);
                        if (x_extr > -0.5 && x_extr < 0.5) done_limiting = false;
                    }
                    if (std::abs(sy + sxy * 0.5) < std::abs(syy))
                    {
                        double y_extr = -(sy + sxy * 0.5) / (2.0 * syy);
                        if (y_extr > -0.5 && y_extr < 0.5) done_limiting = false;
                    }
                    if (std::abs(sy - sxy * 0.5) < std::abs(syy))
                    {
                        double y_extr = -(sy + sxy * 0.5) / (2.0 * syy);
                        if (y_extr > -0.5 && y_extr < 0.5) done_limiting = false;
                    }

                    if (done_limiting)
                    {
                        pout << "Step 2 limited reconstruction on index " << idx << "\n";
                        (*Q_new_data)(idx) = p(x_loc[0], x_loc[1]);
                        continue;
                    }
                }

                if ((sx + sxy * 0.5) * (sx - sxy * 0.5) < 0.0)
                    sxx = 0.0;
                else if (cmp < std::abs(sxx))
                    sxx = sgn(sxx) * cmp;
                if ((sy + sxy * 0.5) * (sy - sxy * 0.5) < 0.0)
                    syy = 0.0;
                else if (cmp < std::abs(syy))
                    syy = sgn(syy) * cmp;

                // Check bounds again.
                LL_new = p(-0.5, -0.5);
                LH_new = p(-0.5, 0.5);
                RL_new = p(0.5, -0.5);
                RH_new = p(0.5, 0.5);
                if (!bds_check(LL_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0))) &&
                    !bds_check(LH_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(-1, 1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 1))) &&
                    !bds_check(RL_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, -1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 0))) &&
                    !bds_check(RH_new,
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(0, 1)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 0)),
                               (*Q_cur_data)(idx + IntVector<NDIM>(1, 1))))
                {
                    pout << "Step 2 limited reconstruction on index " << idx << "\n";
                    (*Q_new_data)(idx) = p(x_loc[0], x_loc[1]);
                    continue;
                }

                // Step 3. Limit quadratic profile close to discontinuities
                // Skip limiting the linear terms.
                // Now limit sxx
                double cmp = std::min(std::abs(sx + sxy * 0.5), std::abs(sx - sxy * 0.5));
                if ((sx + sxy * 0.5) * (sx - sxy * 0.5) < 0.0)
                    sxx = 0.0;
                else if (cmp < std::abs(sxx))
                    sxx = sgn(sxx) * cmp;
                if ((sy + sxy * 0.5) * (sy - sxy * 0.5) < 0.0)
                    syy = 0.0;
                else if (cmp < std::abs(syy))
                    syy = sgn(syy) * cmp;
                shat = qij - 1.0 / 12.0 * (sxx + syy);
                // Test bounds
                double LL_new = p(-0.5, -0.5);
                double LH_new = p(-0.5, 0.5);
                double RL_new = p(0.5, -0.5);
                double RH_new = p(0.5, 0.5);
                if (bds_check(LL_new,
                              (*Q_cur_data)(idx + IntVector<NDIM>(-1, -1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 0))) &&
                    bds_check(LH_new,
                              (*Q_cur_data)(idx + IntVector<NDIM>(-1, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(-1, 1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 1))) &&
                    bds_check(RL_new,
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, -1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(1, -1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(1, 0))) &&
                    bds_check(RH_new,
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(0, 1)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(1, 0)),
                              (*Q_cur_data)(idx + IntVector<NDIM>(1, 1))))
                {
                    sxx = 0.0;
                    syy = 0.0;
                    shat = qij;
                }
                (*Q_new_data)(idx) = p(x_loc[0], x_loc[1]);
#endif
            }
#endif
// PPM
#if (0)
            SideData<NDIM, double> Q_side_data(box, 1 /*depth*/, 0 /*ghosts*/);
            // First compute side interpolant values
            int axis = 0;
            {
                for (SideIterator<NDIM> si(box, axis); si; si++)
                {
                    const SideIndex<NDIM>& idx = si();
                    IntVector<NDIM> one = 0;
                    one(axis) = 1;
                    Q_side_data(idx) =
                        9.0 / 16.0 * ((*Q_cur_data)(idx.toCell(0)) + (*Q_cur_data)(idx.toCell(1))) -
                        1.0 / 16.0 * ((*Q_cur_data)(idx.toCell(1) + one) + (*Q_cur_data)(idx.toCell(0) - one));
                }
            }

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                pout << "Reconstructing on index " << idx << "\n";
                VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d);
                pout << "Physical location: " << x_loc.transpose() << "\n";
                double xi = 2.0 * (x_loc[0] - (idx(0) + 0.5));
                pout << "xi = " << xi << "\n";

                // Construct a quadratic interpolant.
                double alj = Q_side_data(SideIndex<NDIM>(idx, 0, 0)), arj = Q_side_data(SideIndex<NDIM>(idx, 0, 1));
                double aj = (*Q_cur_data)(idx);
                pout << "Pre-limited: " << alj << ", " << arj << "\n";
                auto minmod = [](const double a, const double b) -> double
                {
                    if (a * b > 0 && std::abs(a) <= std::abs(b))
                        return a;
                    else if (a * b > 0 && std::abs(b) <= std::abs(a))
                        return b;
                    else
                        return 0.0;
                };

                double ap1 = (*Q_cur_data)(idx + IntVector<NDIM>(1, 0));
                double am1 = (*Q_cur_data)(idx - IntVector<NDIM>(1, 0));
                if (((ap1 - aj) * (aj - am1)) <= 0.0)
                {
                    alj = aj;
                    arj = aj;
                }
                else
                {
                    if (((aj - alj) * (alj - am1)) <= 0.0)
                    {
                        double sig_r = 2.0 * (ap1 - aj) / dx[0];
                        double sig_c = 2.0 * (ap1 - am1) / (4.0 * dx[0]);
                        double sig_l = 2.0 * (aj - am1) / dx[0];
                        double sig = minmod(sig_c, minmod(sig_r, sig_l));
                        alj = aj - 0.5 * dx[0] * sig;
                    }
                    if (((ap1 - arj) * (arj - aj)) <= 0.0)
                    {
                        double sig_r = 2.0 * (ap1 - aj) / dx[0];
                        double sig_c = 2.0 * (ap1 - am1) / (4.0 * dx[0]);
                        double sig_l = 2.0 * (aj - am1) / dx[0];
                        double sig = minmod(sig_c, minmod(sig_r, sig_l));
                        arj = aj + 0.5 * dx[0] * sig;
                    }
                }

                double a0 = 1.5 * aj - 0.25 * alj - 0.25 * arj;
                double a1 = 0.5 * alj - 0.5 * arj;
                double a2 = -1.5 * aj + 0.75 * alj + 0.75 * arj;

                double xit = -0.5 * a1 / a2;
                if (xit <= 0.0 && xit >= -1.0)
                    arj = 3.0 * aj - 2.0 * alj;
                else if (xit >= 0.0 && xit <= 1.0)
                    alj = 3.0 * aj - 2.0 * arj;

                a0 = 1.5 * aj - 0.25 * alj - 0.25 * arj;
                a1 = 0.5 * alj - 0.5 * arj;
                a2 = -1.5 * aj + 0.75 * alj + 0.75 * arj;

                (*Q_new_data)(idx) = a0 + a1 * xi + a2 * xi * xi;
            }
#endif
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
