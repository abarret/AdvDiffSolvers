#ifndef included_CCAD_RBFReconstructCache
#define included_CCAD_RBFReconstructCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/ReconstructCache.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/ibtk_utilities.h"

#include "CellData.h"
#include "CellIndex.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "tbox/Pointer.h"

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace CCAD
{
/*!
 * \brief Class RBFReconstructCache caches the data necessary to form RBF reconstructions of data.
 */
class RBFReconstructCache : public ReconstructCache
{
public:
    RBFReconstructCache() = delete;

    RBFReconstructCache(int stencil_size);

    RBFReconstructCache(int ls_idx,
                        int vol_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        int d_stencil_size = 8,
                        bool use_centroids = true);

    ~RBFReconstructCache() = default;

    /*!
     * \brief Deleted copy constructor.
     */
    RBFReconstructCache(const RBFReconstructCache& from) = delete;

    void cacheData() override;

    double reconstructOnIndex(IBTK::VectorNd x_loc,
                              const SAMRAI::hier::Index<NDIM>& idx,
                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) override;

private:
    std::vector<std::map<IndexList, std::vector<SAMRAI::hier::Index<NDIM>>>> d_reconstruct_idxs_map_vec;
};
} // namespace CCAD
#endif
