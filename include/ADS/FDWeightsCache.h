/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_FDWeightsCache
#define included_ADS_FDWeightsCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/FDPoint.h>
#include <ADS/FEMeshPartitioner.h>

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/dof_map.h"
#include "libmesh/node.h"

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class FDWeightsCache is a class that caches finite difference weights for general operators. This class
 * requires that finite difference weights be registered. No calculations are done by this class.
 *
 * Points are stored on a patch by patch basis. If you need that points on a patch, you need a pointer to the patch
 * object.
 *
 * TODO: FD weights should not be attached to a given point. As this is written, we can not store more weights than
 * points in the overall mesh. This means we can really only generate square matrices with this cache.
 */
class FDWeightsCache
{
public:
    /*!
     * \brief Constructor
     */
    FDWeightsCache(std::string object_name);

    /*!
     * \brief Destructor.
     */
    virtual ~FDWeightsCache();

    /*!
     * \brief Cache FD weights for a given point for a given patch.
     */
    virtual void cachePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                            const FDPoint& pt,
                            const std::vector<FDPoint>& fd_pts,
                            const std::vector<double>& fd_weights);

    /*!
     * \brief Get the map between a point and it's list of FD weights.
     */
    const std::multimap<FDPoint, std::vector<double>>&
    getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    /*!
     * \brief Get the map between a point and it's list of FD points.
     */
    const std::multimap<FDPoint, std::vector<FDPoint>>&
    getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    /*!
     * \brief Get the set of base points that have cached FD weights.
     */
    const std::set<FDPoint>& getRBFFDBasePoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    /*!
     * \brief Get the vector of FD weights for a given patch and point pair.
     *
     * \note This function returns a copy of the weights.
     */
    std::vector<std::vector<double>> getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                                     const FDPoint& pt);

    /*!
     * \brief Get the vector of FD points for a given patch and point pair.
     *
     * \note This function returns a copy of the points.
     */
    std::vector<std::vector<FDPoint>> getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                                     const FDPoint& pt);

    /*!
     * \brief Determine if this patch and point pair has associated FD weights
     */
    bool isBasePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDPoint& pt);

    /*!
     * \brief Clear the cache.
     */
    virtual void clearCache();

    /*!
     * \brief Eliminate a point and it's associated weights from the cache.
     */
    void clearPoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDPoint& pt);

    /*!
     * Debugging function. Prints all the cached points and their associated FD points to the given output.
     */
    virtual void printPtMap(std::ostream& os, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

protected:
    std::string d_object_name;

    // Weight and point information
    using PtVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::set<FDPoint>>;
    using PtPairVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::multimap<FDPoint, std::vector<FDPoint>>>;
    using WeightVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::multimap<FDPoint, std::vector<double>>>;
    PtVecMap d_base_pt_set;
    PtPairVecMap d_pair_pt_map;
    WeightVecMap d_pt_weight_map;

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    FDWeightsCache() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    FDWeightsCache(const FDWeightsCache& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    FDWeightsCache& operator=(const FDWeightsCache& that) = delete;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_FDWeightsCache
