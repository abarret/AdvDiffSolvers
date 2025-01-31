#ifndef included_ads_sharp_interface_utilities
#define included_ads_sharp_interface_utilities

#include <ADS/CutCellMeshMapping.h>
#include <ADS/FEToHierarchyMapping.h>

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

/*!
 * Structure for storing information about image points.
 *
 * We store the physical location of the image point, the boundary point, and the normal.
 *
 * For finite element structures, we include information about the parent element on which the boundary point is
 * located, as well as the part in the finite element structure.
 *
 * The hierarchy data stores the cell index in which the image point and the ghost point are located.
 *
 * Note that no information concerning the patch or patch level is stored, therefore this structure is only intended to
 * be used on a fixed patch and patch level.
 */
struct ImagePointData
{
public:
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

/*!
 * Structure for storing the weights used to interpolate to the image point. We store both the weights and the cell
 * indices.
 *
 * Note that this structure has no information on whether those indices are ghost cells or fluid cells.
 *
 * Note that no information concerning the patch or patch level is stored, therefore this structure is only intended to
 * be used on a fixed patch and patch level.
 */
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

/*!
 * Functor used to compare two cell indices.
 *
 * TODO: I don't think this is a partial ordering of the indices.
 */
struct Compare
{
    using key_type = std::pair<SAMRAI::pdat::CellIndex<NDIM>, SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>;
    bool operator()(const key_type& a, const key_type& b) const
    {
        if (a.second->getPatchNumber() < b.second->getPatchNumber())
            return true;
        else if (a.second->getPatchNumber() > b.second->getPatchNumber())
            return false;
        // Compute "global index" on the patch
        const SAMRAI::hier::Index<NDIM>& idx_low = a.second->getBox().lower();
        const SAMRAI::hier::Index<NDIM>& idx_up = a.second->getBox().upper();
        int a_global = 0, b_global = 0;
        int shft = 1;
        for (int d = 0; d < NDIM; ++d)
        {
            a_global += shft * (a.first(d) - idx_low(d));
            b_global += shft * (b.first(d) - idx_low(d));
            shft *= idx_up(d) - idx_low(d);
        }

        if (a_global < b_global)
            return true;
        else
            return false;
    }
};

/*!
 * Alias for a map between cell indices and their weights, using a Compare functor.
 */
using ImagePointWeightsMap = std::map<Compare::key_type, ImagePointWeights, Compare>;

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
                            std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            const std::vector<int>& reverse_normal,
                            const std::vector<std::set<int>>& norm_reverse_domain_ids,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                            SAMRAI::tbox::Pointer<CutCellMeshMapping> cut_cell_mapping,
                            const std::vector<int>& reverse_normal,
                            bool use_inside = true,
                            int coarsest_ln = IBTK::invalid_level_number,
                            int finest_ln = IBTK::invalid_level_number);
void classify_points_struct(const int i_idx,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
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
///\{
std::vector<std::vector<ImagePointData>>
find_image_points(int i_idx,
                  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                  int ln,
                  const std::vector<std::unique_ptr<FEToHierarchyMapping>>& mesh_mapping);

std::vector<std::vector<ImagePointData>>
find_image_points(int i_idx,
                  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                  int ln,
                  const std::vector<FEToHierarchyMapping*>& mesh_mapping);
///\}

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

/*!
 * Fill the interface ghost cell with the boundary condition, given the image point weights. Note that the data stored
 * in Q_idx should already have patch and physical ghost cells filled.
 *
 * TODO: We need some sane way to set the boundary condition, probably with a system in the EquationSystems object
 */
void fill_ghost_cells(int i_idx,
                      int Q_idx,
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                      const std::vector<std::vector<ImagePointData>>& img_data_vec_vec,
                      const std::vector<ImagePointWeightsMap>& img_wgts_vec,
                      int ln,
                      std::function<double(const IBTK::VectorNd& x)> bdry_fcn);

/*!
 * Apply the sharp interface Laplacian and boundary condition operator on a given patch
 */
void apply_laplacian_on_patch(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                              const ImagePointWeightsMap& img_wgts,
                              SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              SAMRAI::pdat::CellData<NDIM, double>& R_data,
                              SAMRAI::pdat::CellData<NDIM, int>& i_data);
} // namespace sharp_interface
} // namespace ADS
#endif
