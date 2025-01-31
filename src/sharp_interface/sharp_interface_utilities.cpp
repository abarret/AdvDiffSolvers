#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/sharp_interface_utilities.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/IndexUtilities.h>

#include <queue>

namespace ADS
{
namespace sharp_interface
{
// Local helper functions
bool
is_ghost_point(const SAMRAI::pdat::CellIndex<NDIM>& idx, const SAMRAI::pdat::CellData<NDIM, int>& i_data)
{
    return i_data(idx) == GHOST;
}

template <typename T>
T
get_cell_center_location(const CellIndex<NDIM>& idx,
                         const double* const dx,
                         const double* const xlow,
                         const hier::Index<NDIM>& idx_low)
{
    T x;
    for (int d = 0; d < NDIM; ++d) x(d) = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
    return x;
}

VectorNd
point_to_vec(const libMesh::Point& pt)
{
    VectorNd vec;
    for (int d = 0; d < NDIM; ++d) vec[d] = pt(d);
    return vec;
}

double
project_onto_element(libMesh::Point& n, libMesh::Point& P, const Elem* elem, const libMesh::Point& X)
{
    switch (elem->type())
    {
    case libMesh::EDGE2:
    {
        // Project X onto the line given by elem: p(t) = elem(0) + t * (elem(1) - elem(0)).
        // Closest point to element is:
        //      {elem(0) if t < 0
        //  P = {p(t)    if 0 <= t <= 1
        //      {elem(1) if t > 1
        // vector n is given by n = P - X. This is the normal vector if 0 <= t <= 1.
        const double t =
            (X - elem->point(0)) * (elem->point(1) - elem->point(0)) / (elem->point(1) - elem->point(0)).norm_sq();
        if (t < 0)
            P = elem->point(0);
        else if (t > 1)
            P = elem->point(1);
        else
            P = elem->point(0) + t * (elem->point(1) - elem->point(0));
        n = P - X;
        return t;
    }
    break;
    default:
        TBOX_ERROR("Unsupported element type\n");
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void
classify_points(const int i_idx,
                const int ls_idx,
                Pointer<PatchHierarchy<NDIM>> hierarchy,
                double time,
                int coarsest_ln,
                int finest_ln)
{
    coarsest_ln = coarsest_ln == IBTK::invalid_level_number ? 0 : coarsest_ln;
    finest_ln = finest_ln == IBTK::invalid_level_number ? hierarchy->getFinestLevelNumber() : finest_ln;

    // We do this in several steps:
    // 1. Set all points as either "fluid points" ( = fp) or "invalid points" ( = invalid)
    // 2. Fill ghost cells *NOTE WE ASSUME THERE ARE ENOUGH CELLS ON A LEVEL THAT FLUID POINTS TO NOT TOUCH
    // CF-INTERFACES*
    // 3. If an invalid point touches a fluid point, it becomes a ghost point ( = gp).
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<CellData<NDIM, int>> i = patch->getPatchData(i_idx);

            // First pass
            for (CellIterator<NDIM> ci(i->getGhostBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                double ls_val = ADS::node_to_cell(idx, *ls_data);
                // Fluid point
                if (ls_val < 0.0)
                    (*i)(idx) = FLUID;
                else
                    (*i)(idx) = INVALID;
            }
        }
    }

    trim_classified_points(i_idx, hierarchy, coarsest_ln, finest_ln);
}

void
classify_points_struct(const int i_idx,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<int>& reverse_normal,
                       const std::vector<std::set<int>>& norm_reverse_domain_ids,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    coarsest_ln = coarsest_ln == IBTK::invalid_level_number ? 0 : coarsest_ln;
    finest_ln = finest_ln == IBTK::invalid_level_number ? hierarchy->getFinestLevelNumber() : finest_ln;

    cut_cell_mapping->generateCutCellMappingsOnHierarchy(fe_hierarchy_mappings);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_map_vec =
            cut_cell_mapping->getIdxCutCellElemsMap(ln);

        unsigned int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            if (idx_cut_cell_map_vec.size() > local_patch_num)
            {
                const std::map<IndexList, std::vector<CutCellElems>>& idx_cut_cell_map =
                    idx_cut_cell_map_vec[local_patch_num];
                Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);
                i_data->fillAll(INVALID);

                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const xlow = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();

                // Loop through all cut cells in map
                for (const auto& idx_elem_vec_pair : idx_cut_cell_map)
                {
                    const CellIndex<NDIM>& idx = idx_elem_vec_pair.first.d_idx;
                    const std::vector<CutCellElems>& cut_cell_elem_vec = idx_elem_vec_pair.second;

                    // Determine whether the cell center is on the "inside" or the "outside." Label it as either "FLUID"
                    // or "GHOST" First determine the normal for the element. Warning: we need the normal to be
                    // consistent between parent and child elements.
                    std::vector<IBTK::Vector3d> elem_normals;
                    for (const auto& cut_cell_elem : cut_cell_elem_vec)
                    {
                        // TODO: Works for 2d. For 3d, we need a consistent way to traverse nodes to get 2 "consistently
                        // pointing" tangential vectors.
#if (NDIM == 2)
                        // Note we use the parent element to calculate normals to preserve directions.
                        Vector3d v, w;
                        const std::array<libMesh::Point, 2>& parent_pts = cut_cell_elem.d_parent_cur_pts;
                        const unsigned int part = cut_cell_elem.d_part;
                        const unsigned int domain_id = cut_cell_elem.d_parent_elem->subdomain_id();
                        v << parent_pts[0](0), parent_pts[0](1), parent_pts[0](2);
                        w << parent_pts[1](0), parent_pts[1](1), parent_pts[1](2);

                        Vector3d e3 = Vector3d::UnitZ();
                        if (!use_inside) e3 *= -1.0;
                        if (norm_reverse_domain_ids[part].find(domain_id) != norm_reverse_domain_ids[part].end() ||
                            reverse_normal[part])
                            e3 *= -1.0;
                        Vector3d n = (w - v).cross(e3);
                        elem_normals.push_back(n);
#endif
#if (NDIM == 3)
                        TBOX_ERROR("classify_points(): NOT SETUP FOR 3D!\n");
#endif
                    }

                    // Loop through indices in small area around the element.
                    Box<NDIM> new_box(idx, idx);
                    new_box.grow(1);
                    for (CellIterator<NDIM> ci_2(new_box); ci_2; ci_2++)
                    {
                        const CellIndex<NDIM>& idx_2 = ci_2();

                        // Project this point onto each element. Find the minimum distance
                        Vector3d P;
                        for (int d = 0; d < NDIM; ++d) P(d) = static_cast<double>(idx_2(d) - idx_low(d)) + 0.5;
                        Vector3d avg_proj, avg_unit_normal;
                        avg_proj.setZero();
                        avg_unit_normal.setZero();
                        double min_dist = std::numeric_limits<double>::max();
                        int num_min = 0;
                        for (unsigned int i = 0; i < elem_normals.size(); ++i)
                        {
                            const std::unique_ptr<Elem>& elem = cut_cell_elem_vec[i].d_elem;
                            const Vector3d& n = elem_normals[i];
                            Vector3d v, w;
                            // Put these points in "index space"
                            v << (elem->point(0)(0) - xlow[0]) / dx[0], (elem->point(0)(1) - xlow[1]) / dx[1], 0.0;
                            w << (elem->point(1)(0) - xlow[0]) / dx[0], (elem->point(1)(1) - xlow[1]) / dx[1], 0.0;
                            const double t = std::max(0.0, std::min(1.0, (P - v).dot(w - v) / (v - w).squaredNorm()));
                            const Vector3d proj = v + t * (w - v);
                            const double dist = (proj - P).norm();
                            if (dist < min_dist)
                            {
                                min_dist = dist;
                                avg_proj = proj;
                                avg_unit_normal = n;
                                num_min = 1;
                            }
                            else if (IBTK::rel_equal_eps(dist, min_dist))
                            {
                                avg_proj += proj;
                                avg_unit_normal += n;
                                ++num_min;
                            }
                        }
                        avg_proj /= static_cast<double>(num_min);
                        avg_unit_normal /= static_cast<double>(num_min);
                        avg_unit_normal.normalize();
                        (*i_data)(idx_2) = (avg_unit_normal.dot(P - avg_proj) <= 0.0) ? FLUID : GHOST;
                    }
                }
            }
        }
    }

    fill_interior_points(i_idx, hierarchy, coarsest_ln, finest_ln);
    trim_classified_points(i_idx, hierarchy, coarsest_ln, finest_ln);
}

void
classify_points_struct(const int i_idx,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = fe_hierarchy_mappings.size();
    std::vector<int> reverse_normal(num_parts, 0);
    std::vector<std::set<int>> norm_reverse_domain_ids(num_parts);
    classify_points_struct(i_idx,
                           hierarchy,
                           fe_hierarchy_mappings,
                           cut_cell_mapping,
                           reverse_normal,
                           norm_reverse_domain_ids,
                           use_inside,
                           coarsest_ln,
                           finest_ln);
}
void
classify_points_struct(const int i_idx,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<int>& reverse_normal,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = fe_hierarchy_mappings.size();
    std::vector<std::set<int>> norm_reverse_domain_ids(num_parts);
    classify_points_struct(i_idx,
                           hierarchy,
                           fe_hierarchy_mappings,
                           cut_cell_mapping,
                           reverse_normal,
                           norm_reverse_domain_ids,
                           use_inside,
                           coarsest_ln,
                           finest_ln);
}
void
classify_points_struct(const int i_idx,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FEToHierarchyMapping*> fe_hierarchy_mappings,
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<std::set<int>>& norm_reverse_domain_ids,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = fe_hierarchy_mappings.size();
    std::vector<int> reverse_normal(num_parts, 0);
    classify_points_struct(i_idx,
                           hierarchy,
                           fe_hierarchy_mappings,
                           cut_cell_mapping,
                           reverse_normal,
                           norm_reverse_domain_ids,
                           use_inside,
                           coarsest_ln,
                           finest_ln);
}

void
fill_interior_points(const int i_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, int coarsest_ln, int finest_ln)
{
    coarsest_ln = coarsest_ln == IBTK::invalid_level_number ? 0 : coarsest_ln;
    finest_ln = finest_ln == IBTK::invalid_level_number ? hierarchy->getFinestLevelNumber() : finest_ln;

    // Note: We assume that each level has correctly specified points, so we only fill ghost cells using the current
    // level
    RefineAlgorithm<NDIM> ghost_fill_alg;
    ghost_fill_alg.registerRefine(i_idx, i_idx, i_idx, nullptr);
    std::vector<Pointer<RefineSchedule<NDIM>>> ghost_fill_scheds(finest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        ghost_fill_scheds[ln] = ghost_fill_alg.createSchedule(hierarchy->getPatchLevel(ln));
    }

    // We do a flood fill algorithm.
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        std::vector<int> patch_filled_vec(level->getNumberOfPatches());
        unsigned int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);
            CellData<NDIM, int> idx_touched(box, 1, i_data->getGhostCellWidth());
            idx_touched.fillAll(0);
            std::queue<CellIndex<NDIM>> idx_queue;
            // We need a place to start our flood fill
            bool found_pt = false;
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*i_data)(idx) == FLUID)
                {
                    idx_queue.push(idx);
                    found_pt = true;
                }
            }
            // If we've found a point, this patch will be filled.
            patch_filled_vec[patch_num] = found_pt ? 1 : 0;
            // We have our starting point. Now, loop through queue
            while (idx_queue.size() > 0)
            {
                const CellIndex<NDIM>& idx = idx_queue.front();
                // If this point is invalid, it is a fluid cell
                if (idx_touched(idx) == 0 && ((*i_data)(idx) == INVALID || (*i_data)(idx) == FLUID))
                {
                    idx_touched(idx) = 1;
                    (*i_data)(idx) = FLUID;
                    // Add neighboring points if they haven't been touched
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> shft(0);
                        shft(d) = 1;
                        CellIndex<NDIM> idx_up = idx + shft, idx_low = idx - shft;
                        if (box.contains(idx_up) && (*i_data)(idx_up) == INVALID && idx_touched(idx_up) == 0)
                            idx_queue.push(idx_up);
                        if (box.contains(idx_low) && (*i_data)(idx_low) == INVALID && idx_touched(idx_low) == 0)
                            idx_queue.push(idx_low);
                    }
                }
                idx_queue.pop();
            }
        }

        // At this point, if there's any box that hasn't been filled, then it's either entirely inside or outside.
        // Fill ghost cells, then check the ghost boxes. If there's a FLUID cell in the ghost box, then the entire patch
        // is FLUID.
        int num_negative_found = 1;
        while (num_negative_found > 0)
        {
            num_negative_found = 0;
            ghost_fill_scheds[ln]->fillData(0.0);
            patch_num = 0;
            for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
            {
                bool found_negative = false;
                if (patch_filled_vec[patch_num] == 1) continue;
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);
                for (CellIterator<NDIM> ci(i_data->getGhostBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if (patch->getBox().contains(idx)) continue;
                    if ((*i_data)(idx) == FLUID)
                    {
                        found_negative = true;
                        break;
                    }
                }
                if (found_negative)
                {
                    i_data->fillAll(FLUID, i_data->getGhostBox());
                    ++num_negative_found;
                    patch_filled_vec[patch_num] = 1;
                }
            }
            num_negative_found = IBTK_MPI::sumReduction(num_negative_found);
        }
    }
}

void
trim_classified_points(int i_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, int coarsest_ln, int finest_ln)
{
    coarsest_ln = coarsest_ln == IBTK::invalid_level_number ? 0 : coarsest_ln;
    finest_ln = finest_ln == IBTK::invalid_level_number ? hierarchy->getFinestLevelNumber() : finest_ln;

    // First, ensure that all ghost cells are filled on the given patch levels.
    Pointer<RefineAlgorithm<NDIM>> refine_alg = new RefineAlgorithm<NDIM>();
    refine_alg->registerRefine(i_idx, i_idx, i_idx, Pointer<RefineOperator<NDIM>>());
    std::vector<Pointer<RefineSchedule<NDIM>>> refine_scheds(finest_ln + 1);
    for (int dst_ln = coarsest_ln; dst_ln <= finest_ln; ++dst_ln)
    {
        refine_scheds[dst_ln] = refine_alg->createSchedule(hierarchy->getPatchLevel(dst_ln), nullptr);
        refine_scheds[dst_ln]->fillData(0.0);
    }

    // Now find the ghost points
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> i = patch->getPatchData(i_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*i)(idx) == INVALID || (*i)(idx) == GHOST)
                {
                    // Check immediate neighbors for a fluid point
                    bool found = false;
                    CellIndex<NDIM> nghbr;
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> shft(0);
                        shft(d) = 1;
                        if ((*i)(idx + shft) == FLUID || (*i)(idx - shft) == FLUID)
                        {
                            (*i)(idx) = GHOST;
                            found = true;
                        }
                    }
                    // If we are not a ghost cell, we are an invalid cell.
                    if (!found) (*i)(idx) = INVALID;
                }
            }
        }
    }
}

std::vector<std::vector<ImagePointData>>
find_image_points(const int i_idx,
                  Pointer<PatchHierarchy<NDIM>> hierarchy,
                  const int ln,
                  const std::vector<std::unique_ptr<FEToHierarchyMapping>>& fe_hierarchy_mappings)
{
    return find_image_points(i_idx, hierarchy, ln, unique_ptr_vec_to_raw_ptr_vec(fe_hierarchy_mappings));
}

std::vector<std::vector<ImagePointData>>
find_image_points(const int i_idx,
                  Pointer<PatchHierarchy<NDIM>> hierarchy,
                  const int ln,
                  const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings)
{
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    std::vector<std::vector<ImagePointData>> ip_data_vec_vec;
    ip_data_vec_vec.resize(level->getNumberOfPatches());

    int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const hier::Index<NDIM>& idx_low = patch->getBox().lower();

        Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);
#ifndef NDEBUG
        TBOX_ASSERT(i_data);
#endif
        std::vector<ImagePointData>& ip_data_vec = ip_data_vec_vec[local_patch_num];

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            if (!is_ghost_point(idx, *i_data)) continue;

            libMesh::Point gp_x = get_cell_center_location<libMesh::Point>(idx, dx, xlow, idx_low);

            ImagePointData ip;
            // First we determine which NODE is closest, that will tell us which ELEMENT to search for boundary
            // intercept points.
            unsigned int min_part, min_node_id;
            double min_dist = std::numeric_limits<double>::max();
            for (unsigned int part = 0; part < fe_hierarchy_mappings.size(); ++part)
            {
                const auto& hierarchy_mapping = fe_hierarchy_mappings[part];
                const DofMap& dof_map = hierarchy_mapping->getFESystemManager()
                                            .getEquationSystems()
                                            ->get_system(hierarchy_mapping->getCoordsSystemName())
                                            .get_dof_map();
                NumericVector<double>* X_vec = hierarchy_mapping->getCoordsVector();
                const std::vector<Node*>& patch_nodes = hierarchy_mapping->getActivePatchNodeMap(ln)[local_patch_num];
                for (const auto& node : patch_nodes)
                {
                    libMesh::Point X;
                    std::vector<dof_id_type> X_dofs;
                    dof_map.dof_indices(node, X_dofs);
                    X_vec->get(X_dofs, &X(0));

                    // Compute the distance
                    double dist = (X - gp_x).norm();
                    if (dist < min_dist)
                    {
                        min_part = part;
                        min_node_id = node->id();
                        min_dist = dist;
                    }
                }
            }

            // We know which NODE is closest. Now find all elements that have that node.
            const auto& hierarchy_mapping = fe_hierarchy_mappings[min_part];
            std::vector<Elem*> elems;
            auto elem_it =
                hierarchy_mapping->getFESystemManager().getEquationSystems()->get_mesh().local_elements_begin();
            const auto elem_it_end =
                hierarchy_mapping->getFESystemManager().getEquationSystems()->get_mesh().local_elements_end();
            for (; elem_it != elem_it_end; ++elem_it)
            {
                Elem* elem = *elem_it;
                for (unsigned int k = 0; k < elem->n_nodes(); ++k)
                {
                    if (elem->node_id(k) == min_node_id) elems.push_back(elem);
                }
            }

            FEDataManager::SystemDofMapCache* X_dof_map_cache =
                hierarchy_mapping->getFESystemManager().getDofMapCache(hierarchy_mapping->getCoordsSystemName());
            NumericVector<double>* X_vec = hierarchy_mapping->getCoordsVector();
            auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
#ifndef NDEBUG
            TBOX_ASSERT(X_petsc_vec != nullptr);
#endif
            const double* const X_soln = X_petsc_vec->get_array_read();

            min_dist = std::numeric_limits<double>::max();
            libMesh::Point bp_x;
            libMesh::Point vec;
            // Now find the boundary point. It will be the location with the MINIMUM distance to the ghost point.
            for (const auto& elem : elems)
            {
                // Cache the element's reference configuration and determine current positions
                std::vector<libMesh::Point> X_node_cache(elem->n_nodes());
                boost::multi_array<double, 2> x_node;
                const auto& X_dof_indices = X_dof_map_cache->dof_indices(elem);
                IBTK::get_values_for_interpolation(x_node, *X_petsc_vec, X_soln, X_dof_indices);
                for (unsigned int k = 0; k < elem->n_nodes(); ++k)
                {
                    X_node_cache[k] = elem->point(k);
                    libMesh::Point X;
                    for (unsigned int d = 0; d < NDIM; ++d) X(d) = x_node[k][d];
                    elem->point(k) = X;
                }

                // Project the gp onto the element
                libMesh::Point n, P;
                project_onto_element(n, P, elem, gp_x);
                if ((P - gp_x).norm() < min_dist)
                {
                    min_dist = (P - gp_x).norm();
                    ip.d_bp_location = point_to_vec(P);
                    ip.d_normal = point_to_vec(n);
                    ip.d_ip_location = point_to_vec(gp_x + 2 * n);
                    ip.d_parent_elem = elem;
                    ip.d_part = min_part;
                    ip.d_gp_idx = idx;
                    ip.d_ip_idx = IBTK::IndexUtilities::getCellIndex(ip.d_ip_location, pgeom, patch->getBox());
                }

                // Reset the element's position
                for (unsigned int k = 0; k < elem->n_nodes(); ++k) elem->point(k) = X_node_cache[k];
            }

            ip_data_vec.push_back(std::move(ip));
            X_petsc_vec->restore_array();
        }
    }
    return ip_data_vec_vec;
}

std::vector<ImagePointWeightsMap>
find_image_point_weights(int i_idx,
                         Pointer<PatchHierarchy<NDIM>> hierarchy,
                         const std::vector<std::vector<ImagePointData>>& img_data_vec_vec,
                         int ln)
{
    std::vector<ImagePointWeightsMap> ip_weights_vec(img_data_vec_vec.size());
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const std::vector<ImagePointData>& img_data_vec = img_data_vec_vec[local_patch_num];
        if (img_data_vec.size() == 0) continue;
        ImagePointWeightsMap& ip_weights = ip_weights_vec[local_patch_num];

        Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const hier::Index<NDIM>& idx_low = patch->getBox().lower();

        for (const auto& img_data : img_data_vec)
        {
            const CellIndex<NDIM>& gp_idx = img_data.d_gp_idx;
            const CellIndex<NDIM>& ip_idx = img_data.d_ip_idx;
#ifndef NDEBUG
            TBOX_ASSERT((*i_data)(gp_idx) == GHOST);
            TBOX_ASSERT((*i_data)(ip_idx) == FLUID || (*i_data)(ip_idx) == GHOST);
#endif
            // Interpolate to the image point. Note that if we encounter the ghost cell we are solving for, we use the
            // boundary condition value. Find the image point in index space
            const VectorNd& x_ip = img_data.d_ip_location;
            VectorNd x_idx_space;
            for (int d = 0; d < NDIM; ++d)
                x_idx_space[d] = (x_ip[d] - xlow[d]) / dx[d] + static_cast<double>(idx_low(d));
            // Determine weights of all the neighboring corners of x_ip.
            // Determine the bottom left of bounding box (0,0).
            CellIndex<NDIM> bl_idx;
            for (int d = 0; d < NDIM; ++d) bl_idx[d] = std::ceil(x_idx_space[d] - 0.5) - 1;
            // Now loop over all edges, determine our stencil
            constexpr int num_pts = ImagePointWeights::s_num_pts;
            CellIndex<NDIM> test_idx;
            std::array<VectorNd, num_pts> x_idxs;
            std::array<CellIndex<NDIM>, num_pts> idxs;
            int i = 0;
            for (int x = 0; x <= 1; ++x)
            {
                test_idx(0) = bl_idx(0) + x;
                for (int y = 0; y <= 1; ++y)
                {
                    test_idx(1) = bl_idx(1) + y;
                    if (test_idx == gp_idx)
                    {
                        // We've encountered the ghost cell, use the boundary condition.
                        x_idxs[i] = img_data.d_bp_location;
                    }
                    else
                    {
                        VectorNd x;
                        for (int d = 0; d < NDIM; ++d)
                            x[d] = xlow[d] + dx[d] * (static_cast<double>(test_idx(d) - idx_low(d)) + 0.5);
                        x_idxs[i] = x;
                    }
                    idxs[i++] = test_idx;
                }
            }

            VectorXd b = VectorXd::Zero(num_pts);
            b(0) = 1.0;
            MatrixXd A = MatrixXd::Ones(num_pts, num_pts);
            for (int i = 0; i < num_pts; ++i)
            {
                A(1, i) = x_idxs[i](0) - x_ip(0);
                A(2, i) = x_idxs[i](1) - x_ip(1);
                A(3, i) = (x_idxs[i](0) - x_ip(0)) * (x_idxs[i](1) - x_ip(1));
            }
            VectorXd w = A.lu().solve(b);
            std::array<double, num_pts> weights;
            std::copy(w.data(), w.data() + num_pts, weights.begin());
            ip_weights.insert(std::make_pair(std::make_pair(gp_idx, patch), ImagePointWeights(weights, idxs)));
        }
    }
    return ip_weights_vec;
}

void
fill_ghost_cells(int i_idx,
                 int Q_idx,
                 SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                 const std::vector<std::vector<ImagePointData>>& img_data_vec_vec,
                 const std::vector<ImagePointWeightsMap>& img_wgts_vec,
                 int ln,
                 std::function<double(const VectorNd&)> bdry_fcn)
{
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(i_idx);
        Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
        const unsigned int depth = Q_data->getDepth();

        const std::vector<ImagePointData>& img_data_vec = img_data_vec_vec[local_patch_num];
        const ImagePointWeightsMap& img_wgt_map = img_wgts_vec[local_patch_num];
        for (const auto& img_data : img_data_vec)
        {
            const CellIndex<NDIM>& gp_idx = img_data.d_gp_idx;
            const CellIndex<NDIM>& ip_idx = img_data.d_ip_idx;

            auto gp_patch_pair = std::make_pair(gp_idx, patch);
#ifndef NDEBUG
            TBOX_ASSERT(img_wgt_map.count(gp_patch_pair) > 0);
#endif
            for (unsigned int d = 0; d < depth; ++d) (*Q_data)(gp_idx, d) = 0.0;
            constexpr int num_pts = ImagePointWeights::s_num_pts;
            for (int i = 0; i < num_pts; ++i)
            {
                const CellIndex<NDIM>& idx = img_wgt_map.at(gp_patch_pair).d_idxs[i];
                const double wgt = img_wgt_map.at(gp_patch_pair).d_weights[i];
                if (idx != gp_idx)
                {
                    for (unsigned int d = 0; d < depth; ++d)
                    {
                        (*Q_data)(gp_idx, d) += (*Q_data)(idx, d) * wgt;
                    }
                }
                else
                {
                    for (unsigned int d = 0; d < depth; ++d)
                    {
                        (*Q_data)(gp_idx, d) += wgt * bdry_fcn(img_data.d_bp_location);
                    }
                }
            }

            // Now we fill in the ghost cell using the boundary condition
            for (unsigned int d = 0; d < depth; ++d)
                (*Q_data)(gp_idx, d) = 2.0 * bdry_fcn(img_data.d_bp_location) - (*Q_data)(gp_idx, d);
        }
    }
}

void
apply_laplacian_on_patch(Pointer<Patch<NDIM>> patch,
                         const ImagePointWeightsMap& img_wgts,
                         CellData<NDIM, double>& Q_data,
                         CellData<NDIM, double>& R_data,
                         CellData<NDIM, int>& i_data)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    R_data.fillAll(0.0);

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        const int idx_val = i_data(idx);
        if (idx_val == FLUID)
        {
            for (int d = 0; d < NDIM; ++d)
            {
                IntVector<NDIM> one(0);
                one(d) = 1;
                R_data(idx) += (Q_data(idx + one) - 2.0 * Q_data(idx) + Q_data(idx - one)) / (dx[d] * dx[d]);
            }
        }
        else if (idx_val == GHOST)
        {
            R_data(idx) = Q_data(idx);
            const ImagePointWeights& wgts = img_wgts.at(std::make_pair(idx, patch));
            for (int i = 0; i < wgts.s_num_pts; ++i) R_data(idx) += wgts.d_weights[i] * Q_data(wgts.d_idxs[i]);
        }
    }
}
} // namespace sharp_interface
} // namespace ADS
