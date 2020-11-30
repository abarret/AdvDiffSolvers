#ifndef included_RBFReconstructCache
#define included_RBFReconstructCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/ibtk_utilities.h"

#include "LS/utility_functions.h"

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

namespace LS
{
/*!
 * \brief Class RBFReconstructCache caches the data necessary to form RBF reconstructions of data.
 */
class RBFReconstructCache
{
public:
    RBFReconstructCache();

    RBFReconstructCache(int ls_idx,
                        int vol_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        bool use_centroids = true);

    ~RBFReconstructCache() = default;

    void constructTimers();

    /*!
     * \brief Deleted copy constructor.
     */
    RBFReconstructCache(const RBFReconstructCache& from) = delete;

    void clearCache();

    void setLSData(int ls_idx, int vol_idx);
    void setPatchHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);
    inline void setStencilWidth(int stencil_size)
    {
        d_update_weights = true;
        d_stencil_size = stencil_size;
    }

    inline int getStencilWidth()
    {
        return d_stencil_size;
    }

    inline void setUseCentroids(bool use_centroids)
    {
        d_use_centroids = use_centroids;
        d_update_weights = true;
    }

    void cacheRBFData();

    double reconstructOnIndex(IBTK::VectorNd x_loc,
                              const hier::Index<NDIM>& idx,
                              const CellData<NDIM, double>& Q_data,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

private:
    int d_stencil_size = 2;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    std::vector<std::map<PatchIndexPair, Eigen::FullPivHouseholderQR<MatrixXd>>> d_qr_matrix_vec;
    bool d_update_weights = true;
    bool d_use_centroids = true;
    int d_vol_idx = IBTK::invalid_index, d_ls_idx = IBTK::invalid_index;
};
} // namespace LS
#endif
