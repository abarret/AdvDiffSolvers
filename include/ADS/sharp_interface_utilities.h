#ifndef included_ads_sharp_interface_utilities
#define included_ads_sharp_interface_utilities

#include <ADS/CutCellMeshMapping.h>

#include <ibtk/ibtk_utilities.h>

#include <PatchHierarchy.h>
namespace ADS
{
namespace sharp_interface
{
using PointType = int;
static constexpr PointType FLUID = 0;
static constexpr PointType GHOST = 1;
static constexpr PointType INVALID = -1;

/*
 * Classify each point in the domain as either FLUID, GHOST, or INVALID.
 *
 * @param[out] i_idx: patch index that corresponds to CellData<NDIM, int>. Should have one layer of ghost cells.
 * @param[in] ls_idx: patch index that corresponds to NodeData<NDIM, double> level set data. Should have at least one
 * layer of ghost cells already filled.
 */
void classify_points(int i_idx,
                     int ls_idx,
                     SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                     double time,
                     int coarsest_ln = IBTK::invalid_level_number,
                     int finest_ln = IBTK::invalid_level_number);

/*!
 * Classify each point in the domain as either FLUID, GHOST, or INVALID.
 *
 * @param[out] i_idx: patch index that corresponds to CellData<NDIM, int>. Must have at least one layer of ghost cells.
 * @param[in] cut_cell_mapping: Cut cell mapping object that maps cut cells to patch indices.
 * @param[in] use_inside: If true, uses the normal as calculated by the routine. Otherwise, reverse the normal.
 * @param[in] reverse_normal: Optionally reverse the normal for specific parts of the mesh.
 * @param[in] norm_reverse_domain_ids: Optionally reverses particular domain id's of the parts of the mesh.
 */
///\{
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            const std::vector<int>& reverse_normal,
                            const std::vector<std::set<int>>& norm_reverse_domain_ids,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            const std::vector<int>& reverse_normal,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            const std::vector<std::set<int>>& norm_reverse_domain_ids,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
///\}

/*!
 * Use a flood filling algorithm to fill to mark cells as FLUID cells.
 *
 * i_idx must contain a layer of FLUID and GHOST cells to delineate the location of the boundary.
 */
void fill_interior_points(const int i_idx,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          int coarsest_ln = IBTK::invalid_level_number,
                          int finest_ln = IBTK::invalid_level_number);

/*!
 * Trim cells in i_idx to finish marking GHOST and INVALID cells. This ensures that cells marked as GHOST directly touch
 * a cell marked as FLUID.
 */
void trim_classified_points(int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
} // namespace sharp_interface
} // namespace ADS
#endif
