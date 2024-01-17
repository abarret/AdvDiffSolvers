#ifndef included_ADS_ads_utilities
#define included_ADS_ads_utilities

#include <ComponentSelector.h>
#include <PatchHierarchy.h>

#include <set>

namespace ADS
{
/*!
 * \brief Allocate patch data on the specified levels and at the specified time.
 *
 * Note data should be deallocated manually to avoid memory leaks
 */
//\{
void allocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                         const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                         double time,
                         int coarsest_ln,
                         int finest_ln);
void allocate_patch_data(const std::set<int>& idxs,
                         const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                         double time,
                         int coarsest_ln,
                         int finest_ln);
void allocate_patch_data(const int idx,
                         const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                         double time,
                         int coarsest_ln,
                         int finest_ln);
//\}

/*!
 * \brief Deallocate patch data on the specified levels.
 */
//\{
void deallocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                           const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                           int coarsest_ln,
                           int finest_ln);
void deallocate_patch_data(const std::set<int>& idxs,
                           SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                           int coarsest_ln,
                           int finest_ln);
void deallocate_patch_data(const int idx,
                           const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                           int coarsest_ln,
                           int finest_ln);
//\}

/*!
 * \brief Function that performs an action on every patch in the patch hierarchy
 */
template <typename... Args>
void perform_on_patch_hierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                std::function<void(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>, Args...)> fcn,
                                Args... args);

/*!
 * Function that swaps patch data.
 */
void swap_patch_data(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const int data1_idx, const int data2_idx);
} // namespace ADS

#include <ADS/private/ads_utilities_inc.h>
#endif
