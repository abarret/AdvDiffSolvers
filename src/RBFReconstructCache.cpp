#include "CCAD/RBFReconstructCache.h"
#include "CCAD/app_namespaces.h"
#include "CCAD/ls_functions.h"
#include "CCAD/reconstructions.h"

namespace CCAD
{
RBFReconstructCache::RBFReconstructCache(int ls_idx,
                                         int vol_idx,
                                         Pointer<PatchHierarchy<NDIM>> hierarchy,
                                         const int stencil_size,
                                         bool use_centroids)
    : ReconstructCache(ls_idx, vol_idx, hierarchy, stencil_size, use_centroids)
{
    // intentionally blank
}

RBFReconstructCache::RBFReconstructCache(const int stencil_size) : ReconstructCache(stencil_size)
{
    // intentionall blank
}

void
RBFReconstructCache::cacheData()
{
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    // Free any preallocated matrices
    for (auto& qr_matrix_map_vec : d_qr_matrix_vec)
        for (auto& qr_matrix_map : qr_matrix_map_vec) qr_matrix_map.clear();
    for (std::map<IndexList, std::vector<hier::Index<NDIM>>>& idx_map : d_reconstruct_idxs_map_vec) idx_map.clear();

    // allocate matrix data
    d_qr_matrix_vec.resize(finest_ln + 1);
    d_reconstruct_idxs_map_vec.resize(finest_ln + 1);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        std::vector<std::map<IndexList, FullPivHouseholderQR<MatrixXd>>>& qr_map_vec = d_qr_matrix_vec[ln];
        qr_map_vec.resize(level->getNumberOfPatches());
        std::map<IndexList, std::vector<hier::Index<NDIM>>>& idx_map = d_reconstruct_idxs_map_vec[ln];
        unsigned int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            std::map<IndexList, FullPivHouseholderQR<MatrixXd>>& qr_map = qr_map_vec[local_patch_num];
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // Use flooding to find group of cells to use for interpolation.
                    std::vector<CellIndex<NDIM>> new_idxs = { idx };
                    IndexList p_idx = IndexList(patch, idx);
                    std::vector<VectorNd> X_vals;
                    unsigned int i = 0;
                    while (idx_map[p_idx].size() < d_stencil_size)
                    {
#ifndef NDEBUG
                        TBOX_ASSERT(i < new_idxs.size());
#endif
                        CellIndex<NDIM> new_idx = new_idxs[i];
                        // Add new_idx to idx_map
                        idx_map[p_idx].push_back(new_idx);
                        VectorNd x_cent;
                        if (d_use_centroids)
                            x_cent = find_cell_centroid(new_idx, *ls_data);
                        else
                            for (int d = 0; d < NDIM; ++d) x_cent[d] = static_cast<double>(new_idx(d)) + 0.5;
                        X_vals.push_back(x_cent);
                        // Add Neighboring points to new_idxs
                        IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
                        CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
                        CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
                        if ((*vol_data)(idx_l) > 0.0 &&
                            (std::find(new_idxs.begin(), new_idxs.end(), idx_l) == new_idxs.end()))
                            new_idxs.push_back(idx_l);
                        if ((*vol_data)(idx_r) > 0.0 &&
                            (std::find(new_idxs.begin(), new_idxs.end(), idx_r) == new_idxs.end()))
                            new_idxs.push_back(idx_r);
                        if ((*vol_data)(idx_u) > 0.0 &&
                            (std::find(new_idxs.begin(), new_idxs.end(), idx_u) == new_idxs.end()))
                            new_idxs.push_back(idx_u);
                        if ((*vol_data)(idx_b) > 0.0 &&
                            (std::find(new_idxs.begin(), new_idxs.end(), idx_b) == new_idxs.end()))
                            new_idxs.push_back(idx_b);
                        ++i;
                    }
                    const int m = X_vals.size();
                    MatrixXd A(MatrixXd::Zero(m, m));
                    MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
                    for (size_t i = 0; i < X_vals.size(); ++i)
                    {
                        for (size_t j = 0; j < X_vals.size(); ++j)
                        {
                            const VectorNd X = X_vals[i] - X_vals[j];
                            const double phi = Reconstruct::rbf(X.norm());
                            A(i, j) = phi;
                        }
                        B(i, 0) = 1.0;
                        for (int d = 0; d < NDIM; ++d) B(i, d + 1) = X_vals[i](d);
                    }

                    MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
                    final_mat.block(0, 0, m, m) = A;
                    final_mat.block(0, m, m, NDIM + 1) = B;
                    final_mat.block(m, 0, NDIM + 1, m) = B.transpose();
#ifndef NDEBUG
                    if (qr_map.find(p_idx) == qr_map.end())
                        qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
                    else
                        TBOX_WARNING("Already had a QR decomposition in place");
#else
                    qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
#endif
                }
            }
        }
    }
    d_update_weights = false;
}

double
RBFReconstructCache::reconstructOnIndex(VectorNd x_loc,
                                        const hier::Index<NDIM>& idx,
                                        const CellData<NDIM, double>& Q_data,
                                        Pointer<Patch<NDIM>> patch)
{
    if (d_update_weights) cacheData();
    const int ln = patch->getPatchLevelNumber();
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
    Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();
    for (int d = 0; d < NDIM; ++d) x_loc[d] = idx_low(d) + (x_loc[d] - xlow[d]) / dx[d];

    std::vector<double> Q_vals;
    std::vector<VectorNd> X_vals;

    IndexList pi_pair(patch, idx);
    const std::vector<hier::Index<NDIM>>& idx_vec = d_reconstruct_idxs_map_vec[ln][pi_pair];
    // Get local patch index
    // TODO: rearrange data structures to avoid this
    unsigned int patch_num = 0;
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> cur_patch = level->getPatch(p());
        if (cur_patch == patch) break;
    }
    std::map<IndexList, FullPivHouseholderQR<MatrixXd>>& qr_map = d_qr_matrix_vec[ln][patch_num];

    for (const auto& idx_c : idx_vec)
    {
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
    const int m = Q_vals.size();
    VectorXd U(VectorXd::Zero(m + NDIM + 1));
    for (size_t i = 0; i < Q_vals.size(); ++i) U(i) = Q_vals[i];

    VectorXd x1 = qr_map[pi_pair].solve(U);
    VectorXd rbf_coefs = x1.block(0, 0, m, 1);
    VectorXd poly_coefs = x1.block(m, 0, NDIM + 1, 1);
    VectorXd poly_vec = VectorXd::Ones(NDIM + 1);
    for (int d = 0; d < NDIM; ++d) poly_vec(d + 1) = x_loc(d);
    double val = 0.0;
    for (size_t i = 0; i < X_vals.size(); ++i)
    {
        val += rbf_coefs[i] * Reconstruct::rbf((X_vals[i] - x_loc).norm());
    }
    val += poly_coefs.dot(poly_vec);
    return val;
}
} // namespace CCAD
