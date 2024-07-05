#ifndef included_ADS_ads_utilities
#define included_ADS_ads_utilities

#include <ADS/FEToHierarchyMapping.h>

#include <ibamr/ibamr_utilities.h>

#include <CellVariable.h>
#include <ComponentSelector.h>
#include <PatchHierarchy.h>

#include <functional>
#include <map>
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
                           const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
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

/*!
 * Function that copies patch data.
 */
void copy_patch_data(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const int dst_idx, const int src_idx);

/*!
 * Functions that resets unphysical values
 */
///\{
void reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                             int ls_idx,
                             int dst_idx,
                             int src_idx,
                             double reset_val,
                             bool use_negative);
void reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                             int ls_idx,
                             SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                             SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                             double reset_val = 0.0,
                             bool use_negative = true);
void reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                             int ls_idx,
                             std::set<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& Q_vars,
                             SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                             double reset_val = 0.0,
                             bool use_negative = true);
void reset_unphysical_values(
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
    const int ls_idx,
    const std::set<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& Q_vars,
    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
    const std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>, double>& reset_map,
    const bool use_negative = true);
///\}

template <typename T>
std::vector<T*> unique_ptr_vec_to_raw_ptr_vec(const std::vector<std::unique_ptr<T>>& vec);

std::vector<FESystemManager*> get_system_managers(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings);

} // namespace ADS

#include <ADS/private/ads_utilities_inc.h>
#endif
