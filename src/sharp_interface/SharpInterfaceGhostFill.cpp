#include <ADS/SharpInterfaceGhostFill.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include <stdexcept>

namespace ADS
{

namespace
{
static Timer* t_classify_points;
static Timer* t_generate_image_points;
static Timer* t_generate_image_point_weights;
} // namespace
namespace sharp_interface
{
SharpInterfaceGhostFill::SharpInterfaceGhostFill(std::string object_name,
                                                 std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                                                 Pointer<CutCellMeshMapping> cut_cell_mapping,
                                                 int coarsest_ln,
                                                 int finest_ln)
    : d_object_name(std::move(object_name)),
      d_fe_hierarchy_mappings(std::move(fe_hierarchy_mappings)),
      d_cut_cell_mapping(cut_cell_mapping)
{
    // Grab the patch hierarchy
    Pointer<PatchHierarchy<NDIM>> hierarchy = d_fe_hierarchy_mappings[0]->getPatchHierarchy();
    // Set level numbers
    d_coarsest_ln = coarsest_ln < 0 ? 0 : coarsest_ln;
    d_finest_ln = finest_ln < 0 ? hierarchy->getFinestLevelNumber() : finest_ln;

    // Reserve space for data structures.
    size_t num_levels = d_finest_ln - d_coarsest_ln + 1;
    d_img_pt_data_level_vec.resize(num_levels);
    d_img_pt_wgts_level_vec.resize(num_levels);
    d_generate_image_points.resize(num_levels, true);
    d_generate_image_point_weights.resize(num_levels, true);

    const unsigned int num_parts = d_fe_hierarchy_mappings.size();
    d_norm_reverse_domain_ids.resize(num_parts);
    d_reverse_normal.resize(num_parts);

    // Set up index labeling
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    std::string var_name = d_object_name + "::IndexVar";
    if (var_db->checkVariableExists(var_name))
        d_i_var = var_db->getVariable(d_object_name + "::IndexVar");
    else
        d_i_var = new CellVariable<NDIM, int>(d_object_name + "::IndexVar");
    d_i_idx =
        var_db->registerVariableAndContext(d_i_var, var_db->getContext(d_object_name + "::CTX"), IntVector<NDIM>(1));

    IBTK_DO_ONCE(
        t_classify_points = TimerManager::getManager()->getTimer("ADS::SharpInterfaceGhostFill::classifyPoints()");
        t_generate_image_points =
            TimerManager::getManager()->getTimer("ADS::SharpInterfaceGhostFill::generateImagePoints()");
        t_generate_image_point_weights =
            TimerManager::getManager()->getTimer("ADS::SharpInterfaceGhostFill::generateImagePointWeights()"););
}

void
SharpInterfaceGhostFill::setStructureInformation(bool use_inside,
                                                 std::vector<int> reverse_normals,
                                                 std::vector<std::set<int>> norm_reverse_domain_ids)
{
    d_use_inside = use_inside;
    d_reverse_normal = reverse_normals;
    d_norm_reverse_domain_ids = norm_reverse_domain_ids;
}

void
SharpInterfaceGhostFill::setStructureInformation(bool use_inside, std::vector<int> reverse_normals)
{
    setStructureInformation(use_inside, reverse_normals, {});
}

void
SharpInterfaceGhostFill::setStructureInformation(bool use_inside, std::vector<std::set<int>> norm_reverse_domain_ids)
{
    setStructureInformation(use_inside, {}, norm_reverse_domain_ids);
}

void
SharpInterfaceGhostFill::setStructureInformation(bool use_inside)
{
    setStructureInformation(use_inside, {}, {});
}

const std::vector<std::vector<ImagePointData>>&
SharpInterfaceGhostFill::getImagePointData(const int ln)
{
    auto [valid_ln, error_msg] = isValidLevelNumber(ln);
    if (!valid_ln) throw range_error(error_msg);

    if (d_generate_image_points[ln - d_coarsest_ln]) generateImagePoints(ln);

    return d_img_pt_data_level_vec[ln - d_coarsest_ln];
}

const std::vector<ImagePointWeightsMap>&
SharpInterfaceGhostFill::getImagePointWeights(const int ln)
{
    auto [valid_ln, error_msg] = isValidLevelNumber(ln);
    if (!valid_ln) throw range_error(error_msg);

    if (d_generate_image_point_weights[ln - d_coarsest_ln]) generateImagePointWeights(ln);

    return d_img_pt_wgts_level_vec[ln - d_coarsest_ln];
}

int
SharpInterfaceGhostFill::getIndexPatchIndex()
{
    if (d_classify_points) classifyPoints();

    return d_i_idx;
}

std::tuple<bool, std::string>
SharpInterfaceGhostFill::isValidLevelNumber(const int ln)
{
    if (ln < d_coarsest_ln || ln > d_finest_ln)
    {
        std::string error_msg =
            d_object_name + "::getImagePointData(): Requested level number is outside range: " + std::to_string(ln) +
            "\n  Valid level numbers are " + std::to_string(d_coarsest_ln) + " to " + std::to_string(d_finest_ln) +
            "\n";
        return { false, error_msg };
    }
    return { true, "" };
}

void
SharpInterfaceGhostFill::classifyPoints()
{
    ADS_TIMER_START(t_classify_points);
    Pointer<PatchHierarchy<NDIM>> hierarchy = d_fe_hierarchy_mappings[0]->getPatchHierarchy();
    allocate_patch_data(d_i_idx, hierarchy, 0.0, d_coarsest_ln, d_finest_ln);
    classify_points_struct(d_i_idx,
                           hierarchy,
                           d_fe_hierarchy_mappings,
                           d_cut_cell_mapping,
                           d_reverse_normal,
                           d_norm_reverse_domain_ids,
                           d_use_inside,
                           d_coarsest_ln,
                           d_finest_ln);
    d_classify_points = false;
    ADS_TIMER_STOP(t_classify_points);
}

void
SharpInterfaceGhostFill::generateImagePoints(const int ln)
{
    ADS_TIMER_START(t_generate_image_points);
    if (d_classify_points) classifyPoints();
    Pointer<PatchHierarchy<NDIM>> hierarchy = d_fe_hierarchy_mappings[0]->getPatchHierarchy();
    d_img_pt_data_level_vec[ln] = find_image_points(d_i_idx, hierarchy, ln, d_fe_hierarchy_mappings);
    d_generate_image_points[ln] = false;
    ADS_TIMER_STOP(t_generate_image_points);
}

void
SharpInterfaceGhostFill::generateImagePointWeights(const int ln)
{
    ADS_TIMER_START(t_generate_image_point_weights);
    if (d_generate_image_points[ln]) generateImagePoints(ln);
    Pointer<PatchHierarchy<NDIM>> hierarchy = d_fe_hierarchy_mappings[0]->getPatchHierarchy();
    d_img_pt_wgts_level_vec[ln] = find_image_point_weights(d_i_idx, hierarchy, getImagePointData(ln), ln);
    d_generate_image_point_weights[ln] = false;
    ADS_TIMER_STOP(t_generate_image_point_weights);
}
} // namespace sharp_interface
} // namespace ADS
