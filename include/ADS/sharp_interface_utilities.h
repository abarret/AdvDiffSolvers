#ifndef included_ads_sharp_interface_utilities
#define included_ads_sharp_interface_utilities

#include <ADS/CutCellMeshMapping.h>
#include <ADS/FEMeshPartitioner.h>

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

struct ImagePointData
{
public:
    static constexpr int s_num_pts = 4;
    // Physical data
    IBTK::VectorNd d_bp_location;
    IBTK::VectorNd d_ip_location;
    IBTK::VectorNd d_normal;

    // Lag structure information
    libMesh::Elem* d_parent_elem = nullptr;
    unsigned int d_part = 0;

    // Hierarchy data
    SAMRAI::pdat::CellIndex<NDIM> d_ip_idx, d_gp_idx;
};

struct ImagePointWeights
{
public:
    static constexpr int s_num_pts = 4;
    ImagePointWeights(std::array<double, s_num_pts> weights, std::array<SAMRAI::pdat::CellIndex<NDIM>, s_num_pts> idxs)
        : d_weights(std::move(weights)), d_idxs(std::move(idxs))
    {
    }
    std::array<double, s_num_pts> d_weights;
    std::array<SAMRAI::pdat::CellIndex<NDIM>, s_num_pts> d_idxs;
};

struct Compare
{
    bool operator()(const SAMRAI::pdat::CellIndex<NDIM>& a, const SAMRAI::pdat::CellIndex<NDIM>& b) const
    {
        if (a(0) < b(0))
            return true;
        else if (a(1) < b(1))
            return true;
#if (NDIM == 3)
        else if (a(2) < b(2))
            return true;
#endif
        else
            return false;
    }
};
using ImagePointWeightsMap = std::map<SAMRAI::pdat::CellIndex<NDIM>, ImagePointWeights, Compare>;

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

/*!
 * Determine the location of the image points on the provided level number.
 *
 * Returns the locations of all the image points in a vector of vectors. The outer vector consists of the local patch
 * numbers, the inner vector consists of the image point data for all points on the patch.
 */
std::vector<std::vector<ImagePointData>>
find_image_points(int i_idx,
                  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                  int ln,
                  const std::vector<std::shared_ptr<FEMeshPartitioner>>& mesh_partitioners);

/*!
 * Compute the weights for interpolation at each image point on a given level number. Returns a vector of maps where
 * each key is the cell index of the ghost point and the value is the weights.
 *
 * Note that some points in the stencil for an image point can include ghost points, including the ghost point that
 * corresponds to the image point.
 */
std::vector<ImagePointWeightsMap>
find_image_point_weights(int i_idx,
                         SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                         const std::vector<std::vector<ImagePointData>>& img_data_vec_vec,
                         int ln);
} // namespace sharp_interface
} // namespace ADS
#endif
