#include "ADS/CutCellMeshMapping.h"
#include "ADS/app_namespaces.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"

#include "libmesh/explicit_system.h"

namespace ADS
{
CutCellMeshMapping::CutCellMeshMapping(std::string object_name, Pointer<Database> input_db)
    : d_object_name(std::move(object_name))
{
    d_perturb_nodes = input_db->getBool("perturb_nodes");
}

CutCellMeshMapping::CutCellMeshMapping(std::string object_name, const bool perturb_nodes)
    : d_object_name(std::move(object_name)), d_perturb_nodes(perturb_nodes)
{
    // intentionally blank
}

void
CutCellMeshMapping::generateCutCellMappingsOnHierarchy(const std::vector<FEDataManager*>& fe_data_managers)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = fe_data_managers.front()->getPatchHierarchy();
    int coarsest_ln = 0;
    int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        generateCutCellMappings(fe_data_managers, ln);
    }
}

void
CutCellMeshMapping::generateCutCellMappingsOnHierarchy(const std::vector<FEToHierarchyMapping*>& fe_hier_managers)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = fe_hier_managers.front()->getPatchHierarchy();
    int coarsest_ln = 0;
    int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        generateCutCellMappings(fe_hier_managers, ln);
    }
}

void
CutCellMeshMapping::generateCutCellMappings(const std::vector<FEDataManager*>& fe_data_managers, int ln)
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
        generateCutCellMappings(
            X_sys, X_dof_map_cache, active_patch_elem_map, fe_data_manager->getPatchHierarchy(), ln, part);
    }
    d_elems_cached_per_level[ln] = true;
}

void
CutCellMeshMapping::generateCutCellMappings(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings, int ln)
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
        generateCutCellMappings(
            X_sys, X_dof_map_cache, active_patch_elem_map, fe_hierarchy_mapping->getPatchHierarchy(), ln, part);
    }
    d_elems_cached_per_level[ln] = true;
}

void
CutCellMeshMapping::clearCache(int ln)
{
    if (ln == IBTK::invalid_level_number)
    {
        d_idx_cut_cell_elems_map_vec.clear();
        d_elems_cached_per_level.clear();
    }
    else
    {
        d_idx_cut_cell_elems_map_vec[ln].clear();
        d_elems_cached_per_level[ln] = false;
    }
}

void
CutCellMeshMapping::initializeObjectOnLevel(Pointer<PatchHierarchy<NDIM>> hierarchy, int ln)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    ln = ln == IBTK::invalid_level_number ? finest_ln : ln;
    d_idx_cut_cell_elems_map_vec.resize(finest_ln + 1);
    d_elems_cached_per_level.resize(finest_ln + 1);

    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    d_idx_cut_cell_elems_map_vec[ln].resize(level->getProcessorMapping().getNumberOfLocalIndices());
    d_elems_cached_per_level[ln] = false;
}

void
CutCellMeshMapping::generateCutCellMappings(System& X_sys,
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

    std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_map_vec =
        d_idx_cut_cell_elems_map_vec[level_num];
    idx_cut_cell_map_vec.resize(level->getProcessorMapping().getNumberOfLocalIndices());

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
                        const int i_s = std::floor((x(d) - x_lower[d]) / dx[d]) + patch_lower[d];
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
                IndexList p_idx(patch, CellIndex<NDIM>(elem_idx_nodes[0]));
                // Create copy of element
                idx_cut_cell_map_vec[local_patch_num][p_idx].push_back(CutCellElems(
                    elem, { std::make_pair(elem->point(0), -1), std::make_pair(elem->point(1), -1) }, part));
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
                // We have the index of the box. Each box should have zero or two intersections
                std::vector<std::pair<libMesh::Point, int>> intersection_side_vec(0);
                for (int upper_lower = 0; upper_lower < 2; ++upper_lower)
                {
                    for (int axis = 0; axis < NDIM; ++axis)
                    {
                        VectorValue<double> q;
#if (NDIM == 2)
                        q((axis + 1) % NDIM) = dx[(axis + 1) % NDIM];
#endif
                        libMesh::Point r;
                        for (int d = 0; d < NDIM; ++d)
                            r(d) = x_lower[d] + dx[d] * (static_cast<double>(i_c(d) - patch_lower[d]) +
                                                         (d == axis ? (upper_lower == 1 ? 1.0 : 0.0) : 0.5));

                        libMesh::Point p;
                        // An element may intersect zero or one times with a cell edge.
                        if (findIntersection(p, elem, r, q))
                            intersection_side_vec.push_back(std::make_pair(p, upper_lower + axis * NDIM));
                    }
                }

                // An element may have zero, one, or two intersections with a cell.
                // Note we've already accounted for when an element is contained within a cell.
                if (intersection_side_vec.size() == 0) continue;
                TBOX_ASSERT(intersection_side_vec.size() <= 2);
                if (intersection_side_vec.size() == 1)
                {
                    for (unsigned int k = 0; k < n_node; ++k)
                    {
                        libMesh::Point xn;
                        for (int d = 0; d < NDIM; ++d) xn(d) = elem->point(k)(d);
                        const hier::Index<NDIM>& n_idx =
                            IndexUtilities::getCellIndex(&xn(0), grid_geom, level->getRatio());
                        if (n_idx == i_c)
                        {
                            // Check if we already have this point accounted for. Note this can happen when a node
                            // is EXACTLY on a cell face or node.
                            if (intersection_side_vec[0].first == xn) continue;
                            intersection_side_vec.push_back(std::make_pair(xn, -1));
                            break;
                        }
                    }
                }
                // At this point, if we still only have one intersection point, our node is on a face, and we can
                // skip this index.
                if (intersection_side_vec.size() == 1) continue;
                TBOX_ASSERT(intersection_side_vec.size() == 2);
                IndexList p_idx(patch, CellIndex<NDIM>(i_c));
                // Create a new element
                idx_cut_cell_map_vec[local_patch_num][p_idx].push_back(CutCellElems(elem, intersection_side_vec, part));
            }
            // Restore element's original positions
            for (unsigned int k = 0; k < n_node; ++k) elem->point(k) = X_node_cache[k];
        }
    }
    X_petsc_vec->restore_array();
}

bool
CutCellMeshMapping::findIntersection(libMesh::Point& p, Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q)
{
    bool found_intersection = false;
    switch (elem->type())
    {
    case libMesh::EDGE2:
    {
        // Use linear interpolation
        // Plane through r in q direction:
        // p = r + t * q
        // Plane through two element points p0, p1
        // p = 0.5*(1+u)*p0 + 0.5*(1-u)*p1
        // Set equal and solve for u and t.
        // Note that since q is aligned with a grid axis, we can solve for u first, then find t later
        // Solve for u via a * u + b = 0
        // with a = 0.5 * (p0 - p1)
        //      b = 0.5 * (p0 + p1) - r
        const libMesh::Point& p0 = elem->point(0);
        const libMesh::Point& p1 = elem->point(1);
        const int search_dir = q(0) == 0.0 ? 1 : 0;
        const int trans_dir = (search_dir + 1) % NDIM;
        double a = 0.5 * (p0(trans_dir) - p1(trans_dir));
        double b = 0.5 * (p0(trans_dir) + p1(trans_dir)) - r(trans_dir);
        const double u = -b / a;
        // Determine if this intersection is on the interior of the element
        // This means that u is between -1 and 1
        if (u >= -1.0 && u <= 1.0)
        {
            // Now determine if intersection occurs on axis
            // This amounts to t being between -0.5 and 0.5
            double p_search = 0.5 * p0(search_dir) * (1.0 + u) + 0.5 * (1.0 - u) * p1(search_dir);
            double t = (p_search - r(search_dir)) / q(search_dir);
            if (t >= -0.5 && t <= 0.5)
            {
                // We've found an intersection on this axis
                p = 0.5 * (1.0 + u) * p0 + 0.5 * (1.0 - u) * p1;
                found_intersection = true;
            }
        }
        break;
    }
    default:
        TBOX_ERROR("Unknown element.\n");
    }
    return found_intersection;
}
} // namespace ADS
