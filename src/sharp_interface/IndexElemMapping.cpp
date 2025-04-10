#include "ADS/IndexElemMapping.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"

#include "libmesh/explicit_system.h"

namespace ADS
{
namespace
{
template <typename T>
std::pair<CellIndex<NDIM>, T>&
get_elem(std::vector<std::pair<CellIndex<NDIM>, T>>& vec, const CellIndex<NDIM>& idx)
{
    auto it =
        std::find_if(vec.begin(), vec.end(), [&idx](const std::pair<CellIndex<NDIM>, T>& i) { return i.first == idx; });
    if (it == vec.end())
    {
        vec.emplace_back(idx, T());
        return vec.back();
    }
    else
    {
        return *it;
    }
}
} // namespace

IndexElemMapping::IndexElemMapping(std::string object_name, Pointer<Database> input_db)
    : d_object_name(std::move(object_name))
{
    d_perturb_nodes = input_db->getBool("perturb_nodes");
}

IndexElemMapping::IndexElemMapping(std::string object_name, const bool perturb_nodes)
    : d_object_name(std::move(object_name)), d_perturb_nodes(perturb_nodes)
{
    // intentionally blank
}

void
IndexElemMapping::generateCellElemMappingOnHierarchy(const std::vector<FEDataManager*>& fe_data_managers)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = fe_data_managers.front()->getPatchHierarchy();
    int coarsest_ln = 0;
    int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        generateCellElemMapping(fe_data_managers, ln);
    }
}

void
IndexElemMapping::generateCellElemMappingOnHierarchy(const std::vector<FEToHierarchyMapping*>& fe_hier_managers)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = fe_hier_managers.front()->getPatchHierarchy();
    int coarsest_ln = 0;
    int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        generateCellElemMapping(fe_hier_managers, ln);
    }
}

void
IndexElemMapping::generateCellElemMapping(const std::vector<FEDataManager*>& fe_data_managers, int ln)
{
    ln = ln == IBTK::invalid_level_number ? fe_data_managers.front()->getFinestPatchLevelNumber() : ln;
    if (d_elems_cached_per_level.size() > 0 && d_elems_cached_per_level[ln]) clearCache(ln);
    initializeObjectOnLevel(fe_data_managers[0]->getPatchHierarchy(), ln);
    for (size_t part = 0; part < fe_data_managers.size(); ++part)
    {
        FEDataManager* fe_data_manager = fe_data_managers[part];
        EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
        System& X_sys = eq_sys->get_system(fe_data_manager->getCurrentCoordinatesSystemName());
        FEDataManager::SystemDofMapCache* X_dof_map_cache =
            fe_data_manager->getDofMapCache(fe_data_manager->getCurrentCoordinatesSystemName());
        const std::vector<std::vector<Elem*>>& active_patch_elem_map = fe_data_manager->getActivePatchElementMap();
        generateCellElemMapping(
            X_sys, X_dof_map_cache, active_patch_elem_map, fe_data_manager->getPatchHierarchy(), ln, part);
    }
    d_elems_cached_per_level[ln] = true;
}

void
IndexElemMapping::generateCellElemMapping(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings, int ln)
{
    ln = ln == IBTK::invalid_level_number ? fe_hierarchy_mappings.front()->getPatchHierarchy()->getFinestLevelNumber() :
                                            ln;
    if (d_elems_cached_per_level.size() > 0 && d_elems_cached_per_level[ln]) clearCache(ln);

    initializeObjectOnLevel(fe_hierarchy_mappings[0]->getPatchHierarchy(), ln);
    for (size_t part = 0; part < fe_hierarchy_mappings.size(); ++part)
    {
        FEToHierarchyMapping* fe_hierarchy_mapping = fe_hierarchy_mappings[part];
        FESystemManager& fe_sys_manager = fe_hierarchy_mapping->getFESystemManager();
        EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
        System& X_sys = eq_sys->get_system(fe_hierarchy_mapping->getCoordsSystemName());
        FEDataManager::SystemDofMapCache* X_dof_map_cache =
            fe_sys_manager.getDofMapCache(fe_hierarchy_mapping->getCoordsSystemName());
        const std::vector<std::vector<Elem*>>& active_patch_elem_map = fe_hierarchy_mapping->getActivePatchElementMap();
        generateCellElemMapping(
            X_sys, X_dof_map_cache, active_patch_elem_map, fe_hierarchy_mapping->getPatchHierarchy(), ln, part);
    }
    d_elems_cached_per_level[ln] = true;
}

void
IndexElemMapping::clearCache(int ln)
{
    if (ln == IBTK::invalid_level_number)
    {
        d_idx_cell_elem_vec_vec.clear();
        d_elems_cached_per_level.clear();
    }
    else
    {
        d_idx_cell_elem_vec_vec[ln].clear();
        d_elems_cached_per_level[ln] = false;
    }
}

void
IndexElemMapping::initializeObjectOnLevel(Pointer<PatchHierarchy<NDIM>> hierarchy, int ln)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    ln = ln == IBTK::invalid_level_number ? finest_ln : ln;
    d_idx_cell_elem_vec_vec.resize(finest_ln + 1);
    d_elems_cached_per_level.resize(finest_ln + 1);

    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    d_idx_cell_elem_vec_vec[ln].resize(level->getProcessorMapping().getNumberOfLocalIndices());
    d_elems_cached_per_level[ln] = false;
}

void
IndexElemMapping::generateCellElemMapping(System& X_sys,
                                          FEDataManager::SystemDofMapCache* X_dof_map_cache,
                                          const std::vector<std::vector<Elem*>>& active_patch_elem_map,
                                          Pointer<PatchHierarchy<NDIM>> hierarchy,
                                          const int level_num,
                                          const int part)
{
    NumericVector<double>* X_vec = X_sys.solution.get();
    auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
    TBOX_ASSERT(X_petsc_vec != nullptr);
    const double* const X_local_soln = X_petsc_vec->get_array_read();

    // Only changes are needed where the structure lives
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(level_num);
    const Pointer<CartesianGridGeometry<NDIM>> grid_geom = level->getGridGeometry();
    VectorValue<double> n;
    IBTK::Point x_min, x_max;

    std::vector<std::vector<std::pair<CellIndex<NDIM>, std::vector<ElemData>>>>& idx_cell_elem_vec_vec =
        d_idx_cell_elem_vec_vec[level_num];
    idx_cell_elem_vec_vec.resize(level->getProcessorMapping().getNumberOfLocalIndices());

    unsigned int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const std::vector<Elem*>& patch_elems = active_patch_elem_map[local_patch_num];
        const size_t num_active_patch_elems = patch_elems.size();
        if (num_active_patch_elems == 0) continue;

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const x_lower = pgeom->getXLower();
        const double* const dx = pgeom->getDx();
        const hier::Index<NDIM>& patch_lower = patch->getBox().lower();

        std::vector<dof_id_type> fl_dofs, sf_dofs, Q_dofs;
        boost::multi_array<double, 2> x_node;
        boost::multi_array<double, 1> Q_node;
        for (const auto& elem : patch_elems)
        {
            const auto& X_dof_indices = X_dof_map_cache->dof_indices(elem);
            IBTK::get_values_for_interpolation(x_node, *X_petsc_vec, X_local_soln, X_dof_indices);
            const unsigned int n_node = elem->n_nodes();
            std::vector<libMesh::Point> X_node_cache(n_node), x_node_cache(n_node);
            x_min = IBTK::Point::Constant(std::numeric_limits<double>::max());
            x_max = IBTK::Point::Constant(-std::numeric_limits<double>::max());
            for (unsigned int k = 0; k < n_node; ++k)
            {
                X_node_cache[k] = elem->point(k);
                libMesh::Point& x = x_node_cache[k];
                for (unsigned int d = 0; d < NDIM; ++d) x(d) = x_node[k][d];
                // Perturb the mesh so that we keep FE nodes away from cell edges / nodes
                // Therefore we don't have to worry about nodes being on a cell edge
                if (d_perturb_nodes)
                {
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        const int i_s =
                            static_cast<int>(std::round(((x(d) - x_lower[d]) / dx[d]) - 0.5)) + patch_lower[d];

                        for (int shift = 0; shift <= 2; ++shift)
                        {
                            const double x_s =
                                x_lower[d] + dx[d] * (static_cast<double>(i_s - patch_lower[d]) + 0.5 * shift);
                            const double tol = 1.0e-8 * dx[d];
                            if (x(d) <= x_s) x(d) = std::min(x_s - tol, x(d));
                            if (x(d) >= x_s) x(d) = std::max(x_s + tol, x(d));
                        }
                    }
                }

                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    x_min[d] = std::min(x_min[d], x(d));
                    x_max[d] = std::max(x_max[d], x(d));
                }
                elem->point(k) = x;
            }

            // Check if all indices are the same
            // Simple check if element is completely within grid cell.
            std::vector<hier::Index<NDIM>> elem_idx_nodes(n_node);
            for (unsigned int k = 0; k < n_node; ++k)
            {
                const Node& node = elem->node_ref(k);
                elem_idx_nodes[k] = IndexUtilities::getCellIndex(&node(0), grid_geom, level->getRatio());
            }
            if (std::adjacent_find(elem_idx_nodes.begin(),
                                   elem_idx_nodes.end(),
                                   std::not_equal_to<hier::Index<NDIM>>()) == elem_idx_nodes.end())
            {
                // Element is entirely contained in cell.
                // Store element and continue to next element
                // Create copy of element
                std::pair<CellIndex<NDIM>, std::vector<ElemData>>& cell_elem_vec =
                    get_elem(idx_cell_elem_vec_vec[local_patch_num], elem_idx_nodes[0]);
                std::vector<libMesh::Point> pts;
                for (int n = 0; n < elem->n_nodes(); ++n) pts.push_back(elem->point(n));
                cell_elem_vec.second.emplace_back(pts, elem, part);
                // Reset element
                // Restore element's original positions
                for (unsigned int k = 0; k < n_node; ++k) elem->point(k) = X_node_cache[k];
                continue;
            }
            Box<NDIM> box(IndexUtilities::getCellIndex(&x_min[0], grid_geom, level->getRatio()),
                          IndexUtilities::getCellIndex(&x_max[0], grid_geom, level->getRatio()));
            box.grow(1);
            box = box * patch->getBox();

            // We have the bounding box of the element. Now loop over coordinate directions and look for
            // intersections with the background grid.
            for (BoxIterator<NDIM> b(box); b; b++)
            {
                const hier::Index<NDIM>& i_c = b();
                // We have the index of the box. Search for intersections with the cell. Note we've taken care of the
                // case in which an element is completely contained within a cell.
#if (NDIM == 2)
                for (int upper_lower = 0; upper_lower < 2; ++upper_lower)
                {
                    for (int axis = 0; axis < NDIM; ++axis)
                    {
                        VectorValue<double> q;
                        q((axis + 1) % NDIM) = dx[(axis + 1) % NDIM];
                        libMesh::Point r;
                        for (int d = 0; d < NDIM; ++d)
                            r(d) = x_lower[d] + dx[d] * (static_cast<double>(i_c(d) - patch_lower[d]) +
                                                         (d == axis ? (upper_lower == 1 ? 1.0 : 0.0) : 0.5));

                        libMesh::Point p;
                        if (find_intersection(p, elem, r, q))
                        {
                            // We found an intersection, store it in the vector.
                            std::pair<CellIndex<NDIM>, std::vector<ElemData>>& cell_elem_vec =
                                get_elem(idx_cell_elem_vec_vec[local_patch_num], i_c);
                            std::vector<libMesh::Point> pts;
                            for (unsigned int n = 0; n < elem->n_nodes(); ++n) pts.push_back(elem->point(n));
                            cell_elem_vec.second.emplace_back(pts, elem, part);
                        }
                    }
                }
#endif
#if (NDIM == 3)
                libMesh::Point r;
                for (int d = 0; d < NDIM; ++d)
                    r(d) = x_lower[d] + dx[d] * (static_cast<double>(i_c(d) - patch_lower(d)) + 0.5);
                std::vector<double> h = { dx[0] * 0.5, dx[1] * 0.5, dx[2] * 0.5 };
                if (find_intersection(r, h.data(), elem))
                {
                    // We found an intersection, store it in the vector.
                    std::pair<CellIndex<NDIM>, std::vector<ElemData>>& cell_elem_vec =
                        get_elem(idx_cell_elem_vec_vec[local_patch_num], i_c);
                    std::vector<libMesh::Point> pts;
                    for (unsigned int n = 0; n < elem->n_nodes(); ++n) pts.push_back(elem->point(n));
                    cell_elem_vec.second.emplace_back(pts, elem, part);
                }
#endif
            }
            // Restore element's original positions
            for (unsigned int k = 0; k < n_node; ++k) elem->point(k) = X_node_cache[k];
        }
    }
    X_petsc_vec->restore_array();
}
} // namespace ADS
