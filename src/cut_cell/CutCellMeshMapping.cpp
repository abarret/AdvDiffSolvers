#include "ADS/CutCellMeshMapping.h"
#include "ADS/app_namespaces.h"

#include "ibtk/IndexUtilities.h"

namespace ADS
{
CutCellMeshMapping::CutCellMeshMapping(std::string object_name,
                                       Pointer<Database> input_db,
                                       const unsigned int num_parts)
    : d_object_name(std::move(object_name)), d_num_parts(num_parts)
{
    if (input_db) d_perturb_nodes = input_db->getBool("perturb_nodes");
}

CutCellMeshMapping::~CutCellMeshMapping()
{
    if (d_is_initialized) deinitializeObjectState();
}

void
CutCellMeshMapping::initializeObjectState(Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    if (d_is_initialized) deinitializeObjectState();
    d_hierarchy = hierarchy;

    // Reset mappings
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    d_idx_cut_cell_elems_map_vec.resize(finest_ln + 1);
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        d_idx_cut_cell_elems_map_vec[ln].resize(level->getProcessorMapping().getNumberOfLocalIndices());
    }

    d_is_initialized = true;
}

void
CutCellMeshMapping::deinitializeObjectState()
{
    d_idx_cut_cell_elems_map_vec.clear();

    d_is_initialized = false;
}
} // namespace ADS
