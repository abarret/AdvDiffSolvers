#include "CCAD/ReconstructCache.h"

#include "ibamr/app_namespaces.h"

namespace CCAD
{
ReconstructCache::ReconstructCache(int ls_idx, int vol_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, bool use_centroids)
    : d_hierarchy(hierarchy), d_use_centroids(use_centroids), d_vol_idx(vol_idx), d_ls_idx(ls_idx)
{
    // intentionally blank
}

void
ReconstructCache::clearCache()
{
    for (auto& qr_matrix_map : d_qr_matrix_vec) qr_matrix_map.clear();
    d_qr_matrix_vec.clear();
    d_update_weights = true;
}

void
ReconstructCache::setLSData(const int ls_idx, const int vol_idx)
{
    d_vol_idx = vol_idx;
    d_ls_idx = ls_idx;
    clearCache();
    d_update_weights = true;
}

void
ReconstructCache::setPatchHierarchy(Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_hierarchy = hierarchy;
    clearCache();
    d_update_weights = true;
}
} // namespace CCAD
