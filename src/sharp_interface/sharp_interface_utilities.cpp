#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/sharp_interface_utilities.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/IBTK_MPI.h>

#include <queue>

namespace ADS
{
namespace sharp_interface
{
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
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<int>& reverse_normal,
                       const std::vector<std::set<int>>& norm_reverse_domain_ids,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    coarsest_ln = coarsest_ln == IBTK::invalid_level_number ? 0 : coarsest_ln;
    finest_ln = finest_ln == IBTK::invalid_level_number ? hierarchy->getFinestLevelNumber() : finest_ln;

    // If reverse mappings was not set, we need something else

    cut_cell_mapping->initializeObjectState(hierarchy);
    cut_cell_mapping->generateCutCellMappings();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_map_vec =
            cut_cell_mapping->getIdxCutCellElemsMap(ln);

        unsigned int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
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
                        double sgn = (avg_unit_normal.dot(P - avg_proj) <= 0.0) ? -1.0 : 1.0;
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
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = cut_cell_mapping->getNumParts();
    std::vector<int> reverse_normal(num_parts, 0);
    std::vector<std::set<int>> norm_reverse_domain_ids(num_parts);
    classify_points_struct(i_idx,
                           hierarchy,
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
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<int>& reverse_normal,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = cut_cell_mapping->getNumParts();
    std::vector<std::set<int>> norm_reverse_domain_ids(num_parts);
    classify_points_struct(i_idx,
                           hierarchy,
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
                       Pointer<CutCellMeshMapping> cut_cell_mapping,
                       const std::vector<std::set<int>>& norm_reverse_domain_ids,
                       bool use_inside,
                       int coarsest_ln,
                       int finest_ln)
{
    unsigned int num_parts = cut_cell_mapping->getNumParts();
    std::vector<int> reverse_normal(num_parts, 0);
    classify_points_struct(i_idx,
                           hierarchy,
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
} // namespace sharp_interface
} // namespace ADS
