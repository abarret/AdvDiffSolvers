#ifndef included_ADS_BoundaryReconstructCache
#define included_ADS_BoundaryReconstructCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/FEToHierarchyMapping.h"
#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/ls_utilities.h"

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
 * \brief Class BoundaryReconstructCache caches the data necessary to form RBF reconstructions of data.
 *
 * This class has key differences from ReconstructCache (and subclasses) in that this class is designed to hold weights
 * for specific points associated with a finite element mesh. Meanwhile, ReconstructCache holds reconstructions for
 * specific cell indices in the patch hierarchy. This class should be used to store interpolation weights for repeated
 * interpolations.
 *
 * By default, this class assumes that negative level set values correspond to the "appropriate" side of the interface.
 * This can be changed with the setSign() function.
 *
 * This class should be invalidated with clearCache() or reconstructed when either the patch hierarchy changes OR the
 * interpolated points move.
 */
class BoundaryReconstructCache
{
public:
    /*!
     * \brief Constructor that only initializes the stencil size.
     *
     * The hierarchy, level set index, and mesh mapping must be set via functions before this class can be useable.
     */
    BoundaryReconstructCache(int stencil_size);

    /*!
     * \brief Constructor that initializes and sets variables. Interpolation stencils are not created during
     * construction.
     */
    BoundaryReconstructCache(int ls_idx,
                             SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                             std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping,
                             int stencil_size = 8);

    /*!
     * \brief Default deconstructor.
     */
    ~BoundaryReconstructCache() = default;

    /*!
     * \brief WeightStruct is a light-weight struct that maintains a paired vector of cell indices and weights for
     * finite differences.
     */
    struct WeightStruct
    {
        /*!
         * \brief Constructor.
         */
        WeightStruct(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                     std::vector<std::pair<SAMRAI::pdat::CellIndex<NDIM>, double>> idx_wgt_pair_vec)
            : d_patch(patch), d_idx_wgt_pair_vec(std::move(idx_wgt_pair_vec)){};

        /*!
         * Constructor that takes in a patch, index list, and weight list. Creates a paired vector between corresponding
         * vlaues in the index and weight lists.
         */
        WeightStruct(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                     const std::vector<SAMRAI::pdat::CellIndex<NDIM>>& idx_list,
                     const std::vector<double>& wgt_list)
            : d_patch(patch)
        {
#ifndef NDEBUG
            TBOX_ASSERT(idx_list.size() == wgt_list.size());
#endif
            d_idx_wgt_pair_vec.resize(idx_list.size());
            for (size_t i = 0; i < idx_list.size(); ++i)
                d_idx_wgt_pair_vec[i] = std::make_pair(idx_list[i], wgt_list[i]);
        }

        /*!
         * \brief Default constructors
         */
        ///\{
        WeightStruct() = default;
        WeightStruct(WeightStruct&&) = default;
	WeightStruct(const WeightStruct&) = default;
        WeightStruct& operator=(const WeightStruct&) = default;
        WeightStruct& operator=(WeightStruct&&) = default;
        ~WeightStruct() = default;
        ///\}

        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> d_patch;
        std::vector<std::pair<SAMRAI::pdat::CellIndex<NDIM>, double>> d_idx_wgt_pair_vec;
    };

    /*!
     * \brief Clears the cached interpolation stencils.
     */
    void clearCache();

    /*!
     * \brief Set the level set data. Also clears the cache.
     */
    void setLSData(int ls_idx);

    /*!
     * \brief Set the sign of the level set to refer to the "appropriate" side of the interface. Also clears the cache.
     *
     * If use_positive == true, then level set values with positive values are treated as the "appropriate" side of the
     * interface.
     */
    void setSign(bool use_positive);

    /*!
     * \brief Set the patch hierarchy. Also clears the cache.
     */
    void setPatchHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    /*!
     * \brief Set the mesh mapping. Also clears the cache.
     */
    void setMeshMapping(std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping);

    /*!
     * \brief Set the stencil width. Also clears the cache.
     */
    inline void setStencilWidth(int stencil_size)
    {
        d_update_weights = true;
        d_stencil_size = stencil_size;
        clearCache();
    }

    /*!
     * \brief Return the size of the stencil.
     */
    inline int getStencilWidth()
    {
        return d_stencil_size;
    }

    /*!
     * \brief Initialize and create the interpolation stencils for each node in the mesh mapping.
     *
     * This function is called by reconstruct() when weights have not been created.
     */
    void cacheData();

    /*!
     * \brief Reconstruct Q in Q_idx at the node in node_id for the given part.
     *
     * If the cache has not been constructed, this function will call cacheData(). Otherwise, it uses the interpolation
     * stencils previously computed.
     */
    double reconstruct(int part, int node_id, int Q_idx);

    /*!
     * \brief Reconstruct Q in Q_idx at the node in node_id for the given part.
     *
     * Note this const call will not call cacheData() (as cacheData() is not const). It is assumed that all data is
     * already cached before this call occurs.
     */
    double reconstruct(int part, int node_id, int Q_idx) const;

    /*!
     * \brief Reconstruct Q in Q_idx at all the nodes for the given system and variable name on the given part.
     *
     * If the cache has not been constructed, this function will call cacheData(). Otherwise, it uses the interpolation
     * stencils previously computed.
     */
    void reconstruct(int part, const std::string& Q_str, int Q_idx);

    /*!
     * \brief Reconstruct Q in Q_idx at all the nodes for the given system and variable name on all the parts.
     *
     * This loops through all the parts and calls reconstruct(part, Q_star, Q_idx);
     */
    void reconstruct(const std::string& Q_str, int Q_idx);

    /*!
     * \brief Returns the weights and indices for interpolating at node with node_id for the given part.
     */
    const WeightStruct& getWeightStruct(int part, int node_id) const;

private:
    unsigned int d_stencil_size = 8;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    std::shared_ptr<GeneralBoundaryMeshMapping> d_mesh_mapping;
    std::vector<std::unique_ptr<FEToHierarchyMapping>> d_fe_hierarchy_mappings;

    /*!
     * Key for map is <part, node_id>.
     */
    std::map<std::pair<int, int>, WeightStruct> d_weights_map;
    bool d_update_weights = true;
    int d_ls_idx = IBTK::invalid_index;
    double d_sign = -1.0;
};
} // namespace ADS
#endif
