#ifndef included_ADS_SharpInterfaceGhostFill
#define included_ADS_SharpInterfaceGhostFill

#include <ADS/CutCellVolumeMeshMapping.h>
#include <ADS/sharp_interface_utilities.h>

namespace ADS
{

namespace sharp_interface
{
/*!
 * SharpInterfaceGhostFill is a class that stores and updates image point and image point weight data on a given patch
 * hierarchy.
 *
 * Lazily computes the image points and image point weights.
 *
 * If the structure moves or image points and image point weights need to be invalidated, call invalidateLevelData. This
 * prevents data from being deallocated, and is instead overwritten.
 */
class SharpInterfaceGhostFill
{
public:
    /*!
     * Constructor that sets up the object, but leaves it in an empty state.
     *
     * Note that coarsest_ln and finest_ln should be set to be valid for all levels on which image point data will be
     * needed.
     */
    SharpInterfaceGhostFill(std::string object_name,
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                            SAMRAI::tbox::Pointer<CutCellVolumeMeshMapping> cut_cell_mapping,
                            int coarsest_ln = IBTK::invalid_index,
                            int finest_ln = IBTK::invalid_index);

    /*!
     * Default destructor that does nothing interesting.
     */
    ~SharpInterfaceGhostFill() = default;

    /*!
     * Optional calls to set structure specific information:
     * @param[in] use_inside: If true, uses the normal as calculated by the routine. Otherwise, reverse the normal.
     * @param[in] reverse_normal: Optionally reverse the normal for specific parts of the mesh.
     * @param[in] norm_reverse_domain_ids: Optionally reverses the particular domain id's of the parts of the mesh.
     *
     * Note that any of these vectors can be empty. If these functions are not set, then use_inside is set to TRUE.
     */
    ///\{
    void setStructureInformation(bool use_inside,
                                 std::vector<int> reverse_normals,
                                 std::vector<std::set<int>> norm_reverse_domain_ids);
    void setStructureInformation(bool use_inside, std::vector<int> reverse_normals);
    void setStructureInformation(bool use_inside, std::vector<std::set<int>> norm_reverse_domain_ids);
    void setStructureInformation(bool use_inside);
    ///\}

    /*!
     * Return the image point data for the specified level. If that data has not been computed, compute the image
     * points.
     *
     * If the level number is outside the range of [coarsest_ln, finest_ln] as specified in the constructor, throws a
     * std::range_error exception.
     */
    const std::vector<std::vector<ImagePointData>>& getImagePointData(int ln);

    /*!
     * Return the image point weights for the specified level. If that data has not been computed, compute the image
     * point weights.
     *
     * If the level number is outside the range of [coarsest_ln, finest_ln] as specified in the constructor, throws a
     * std::range_error exception.
     */
    const std::vector<ImagePointWeightsMap>& getImagePointWeights(int ln);

    /*!
     * Return the cell variable used to classify the points.
     */
    inline SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, int>> getIndexVariable() const
    {
        return d_i_var;
    }

    /*!
     * Return the patch data index that is used to classify points. If that data has not been computed, classify the
     * points.
     */
    int getIndexPatchIndex();

    /*!
     * Invalidates the image points, image point weights, and classified points on the given patch level.
     *
     * Note that on the next iteration, points will be reclassified on all levels, but image points and weights will
     * only be recomputed on the invalidated level.
     */
    void invalidateLevelData(int ln);

private:
    /*!
     * Return a bool and error message if the level number is an invalid number.
     */
    std::tuple<bool, std::string> isValidLevelNumber(int ln);

    /*!
     * Allocate and classify points as FLUID, SOLID, or GHOST.
     */
    void classifyPoints();

    /*!
     * Compute the location of all the image points on the provided level.
     */
    void generateImagePoints(int ln);

    /*!
     * Compute the corresponding interpolation weights of all the image points on the provided level.
     */
    void generateImagePointWeights(int ln);

    std::string d_object_name;

    // Hierarchy Information
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    // Structure information
    SAMRAI::tbox::Pointer<CutCellVolumeMeshMapping> d_cut_cell_mapping;
    bool d_use_inside = true;
    std::vector<int> d_reverse_normal;
    std::vector<std::set<int>> d_norm_reverse_domain_ids;

    // Index label variable
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, int>> d_i_var;
    int d_i_idx = IBTK::invalid_index;

    // Image point data
    std::vector<std::vector<std::vector<ImagePointData>>> d_img_pt_data_level_vec;
    std::vector<std::vector<ImagePointWeightsMap>> d_img_pt_wgts_level_vec;

    // Do image point data need regenerating?
    bool d_classify_points = true;
    std::vector<bool> d_generate_image_points, d_generate_image_point_weights;
};

} // namespace sharp_interface
} // namespace ADS
#endif
