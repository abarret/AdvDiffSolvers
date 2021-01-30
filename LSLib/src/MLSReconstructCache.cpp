#include "LS/MLSReconstructCache.h"

namespace LS
{
MLSReconstructCache::MLSReconstructCache(int ls_idx,
                                         int vol_idx,
                                         Pointer<PatchHierarchy<NDIM>> hierarchy,
                                         bool use_centroids)
    : ReconstructCache(ls_idx, vol_idx, hierarchy, use_centroids)
{
    // intentionally blank
}

void
MLSReconstructCache::cacheData()
{
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    // Free any preallocated matrices
    for (std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_matrix_map : d_qr_matrix_vec)
        qr_matrix_map.clear();

    // allocate matrix data
    d_qr_matrix_vec.resize(finest_ln + 1);

    int size = NDIM + 1;

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_map = d_qr_matrix_vec[ln];
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // We are on a cut cell. We need to interpolate to cell center or cell centroid
                    VectorNd x_loc;
                    if (d_use_centroids)
                    {
                        x_loc = find_cell_centroid(idx, *ls_data);
                    }
                    else
                    {
                        for (int d = 0; d < NDIM; ++d) x_loc(d) = static_cast<double>(idx(d)) + 0.5;
                    }
                    Box<NDIM> box(idx, idx);
                    box.grow(d_stencil_size);
#ifndef NDEBUG
                    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
                    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif

                    const CellIndex<NDIM>& idx_low = patch->getBox().lower();
                    std::vector<VectorNd> X_vals;

                    for (CellIterator<NDIM> ci(box); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx_c = ci();
                        if ((*vol_data)(idx_c) > 0.0)
                        {
                            // Use this point to calculate least squares reconstruction.
                            VectorNd x_cent_c;
                            if (d_use_centroids)
                            {
                                x_cent_c = find_cell_centroid(idx_c, *ls_data);
                            }
                            else
                            {
                                for (int d = 0; d < NDIM; ++d) x_cent_c[d] = static_cast<double>(idx_c[d]) + 0.5;
                            }
                            X_vals.push_back(x_cent_c);
                        }
                    }
                    const int m = X_vals.size();
                    MatrixXd A(MatrixXd::Zero(m, size)), Lambda(MatrixXd::Zero(m, m));
                    for (size_t i = 0; i < X_vals.size(); ++i)
                    {
                        const VectorNd X = X_vals[i] - x_loc;
                        A(i, 2) = X[1];
                        A(i, 1) = X[0];
                        A(i, 0) = 0.0;
                    }
                    PatchIndexPair p_idx = PatchIndexPair(patch, idx);
#ifndef NDEBUG
                    if (qr_map.find(p_idx) == qr_map.end())
                        qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(Lambda * A);
                    else
                        TBOX_WARNING("Already had a QR decomposition in place");
#else
                    qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(Lambda * A);
#endif
                }
            }
        }
    }
    d_update_weights = false;
}

double
MLSReconstructCache::reconstructOnIndex(VectorNd x_loc,
                                        const hier::Index<NDIM>& idx,
                                        const CellData<NDIM, double>& Q_data,
                                        Pointer<Patch<NDIM>> patch)
{
    if (d_update_weights) cacheData();
    const int ln = patch->getPatchLevelNumber();
    Box<NDIM> box(idx, idx);
    box.grow(d_stencil_size);
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
    Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
#ifndef NDEBUG
    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
    TBOX_ASSERT(Q_data.getGhostBox().contains(box));
    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();
    for (int d = 0; d < NDIM; ++d) x_loc[d] = idx_low(d) + (x_loc[d] - xlow[d]) / dx[d];

    std::vector<double> Q_vals;
    std::vector<VectorNd> X_vals;

    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx_c = ci();
        if ((*vol_data)(idx_c) > 0.0)
        {
            // Use this point to calculate least squares reconstruction.
            // Find cell center
            VectorNd x_cent_c;
            if (d_use_centroids)
            {
                x_cent_c = find_cell_centroid(idx_c, *ls_data);
            }
            else
            {
                for (int d = 0; d < NDIM; ++d) x_cent_c[d] = static_cast<double>(idx_c[d]) + 0.5;
            }
            Q_vals.push_back(Q_data(idx_c));
            X_vals.push_back(x_cent_c);
        }
    }
    const int m = Q_vals.size();
    VectorXd U(VectorXd::Zero(m));
    for (size_t i = 0; i < Q_vals.size(); ++i) U(i) = Q_vals[i];
    VectorXd x = d_qr_matrix_vec[ln][PatchIndexPair(patch, idx)].solve(U);
    return x(0);
}
} // namespace LS
