#ifndef included_ADS_MLSReconstructCache
#define included_ADS_MLSReconstructCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/ReconstructCache.h"

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

namespace ADS
{
/*!
 * \brief Class MLSReconstructCache caches the data necessary to form RBF reconstructions of data.
 */
class MLSReconstructCache : public ReconstructCache
{
public:
    MLSReconstructCache() = delete;

    MLSReconstructCache(int stencil_size);

    MLSReconstructCache(int ls_idx,
                        int vol_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        int stencil_size = 8,
                        bool use_centroids = true);

    ~MLSReconstructCache() = default;

    /*!
     * \brief Deleted copy constructor.
     */
    MLSReconstructCache(const MLSReconstructCache& from) = delete;

    void cacheData() override;

    double reconstructOnIndex(IBTK::VectorNd x_loc,
                              const SAMRAI::hier::Index<NDIM>& idx,
                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);
};
} // namespace ADS
#endif
