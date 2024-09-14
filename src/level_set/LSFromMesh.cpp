#include "ADS/LSFromMesh.h"
#include "ADS/ads_utilities.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include "ibtk/IBTK_MPI.h"

#include "RefineAlgorithm.h"

#include <algorithm>
#include <queue>

namespace
{
static Timer* t_updateVolumeAreaSideLS = nullptr;
} // namespace

namespace ADS
{
LSFromMesh::LSFromMesh(std::string object_name,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FESystemManager*> fe_sys_managers,
                       const Pointer<CutCellMeshMapping>& cut_cell_mesh_mapping,
                       bool use_inside /* = true*/)
    : LSFindCellVolume(std::move(object_name), hierarchy),
      d_use_inside(use_inside),
      d_fe_sys_managers(std::move(fe_sys_managers)),
      d_cut_cell_mesh_mapping(cut_cell_mesh_mapping),
      d_sgn_nc_var(new NodeVariable<NDIM, double>(d_object_name + "::SGN_NC_VAR")),
      d_sgn_cc_var(new CellVariable<NDIM, double>(d_object_name + "::SGN_CC_VAR"))
{
    commonConstructor();
    return;
} // Constructor

LSFromMesh::LSFromMesh(std::string object_name,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       std::vector<FEDataManager*> fe_data_managers,
                       const Pointer<CutCellMeshMapping>& cut_cell_mesh_mapping,
                       bool use_inside /* = true*/)
    : LSFindCellVolume(std::move(object_name), hierarchy),
      d_use_inside(use_inside),
      d_fe_data_managers(std::move(fe_data_managers)),
      d_cut_cell_mesh_mapping(cut_cell_mesh_mapping),
      d_sgn_nc_var(new NodeVariable<NDIM, double>(d_object_name + "::SGN_NC_VAR")),
      d_sgn_cc_var(new CellVariable<NDIM, double>(d_object_name + "::SGN_CC_VAR"))
{
    commonConstructor();
    return;
} // Constructor

void
LSFromMesh::commonConstructor()
{
    IBAMR_DO_ONCE(t_updateVolumeAreaSideLS =
                      TimerManager::getManager()->getTimer("ADS::LSFromMesH::updateVolumeAreaSideLS()"););
    const unsigned int num_parts = getNumParts();
    d_norm_reverse_domain_ids.resize(num_parts);
    d_norm_reverse_elem_ids.resize(num_parts);
    d_reverse_normal.resize(num_parts, 0);

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_sgn_nc_idx = var_db->registerVariableAndContext(d_sgn_nc_var, var_db->getContext(d_object_name + "::Context"), 1);
    d_sgn_cc_idx = var_db->registerVariableAndContext(d_sgn_cc_var, var_db->getContext(d_object_name + "::Context"), 1);
}

size_t
LSFromMesh::getNumParts()
{
    return std::max(d_fe_sys_managers.size(), d_fe_data_managers.size());
}

void
LSFromMesh::doUpdateVolumeAreaSideLS(int vol_idx,
                                     Pointer<CellVariable<NDIM, double>> vol_var,
                                     int area_idx,
                                     Pointer<CellVariable<NDIM, double>> area_var,
                                     int side_idx,
                                     Pointer<SideVariable<NDIM, double>> side_var,
                                     int phi_idx,
                                     Pointer<hier::Variable<NDIM>> phi_var,
                                     double data_time,
                                     bool extended_box)
{
    ADS_TIMER_START(t_updateVolumeAreaSideLS);
    Pointer<NodeVariable<NDIM, double>> phi_nc_var = phi_var;
    Pointer<CellVariable<NDIM, double>> phi_cc_var = phi_var;
    if (phi_nc_var)
        doUpdateVolumeAreaSideLSNode(
            vol_idx, vol_var, area_idx, area_var, side_idx, side_var, phi_idx, phi_nc_var, data_time, extended_box);
    else if (phi_cc_var)
        doUpdateLSCell(phi_idx, phi_cc_var, data_time, extended_box);
    else
        TBOX_ERROR(d_object_name + "::doUpdateVolumeAreaSideLS(): Not a valid data centering\n");
    ADS_TIMER_STOP(t_updateVolumeAreaSideLS);
}

void
LSFromMesh::doUpdateVolumeAreaSideLSNode(int vol_idx,
                                         Pointer<CellVariable<NDIM, double>> vol_var,
                                         int area_idx,
                                         Pointer<CellVariable<NDIM, double>> area_var,
                                         int side_idx,
                                         Pointer<SideVariable<NDIM, double>> side_var,
                                         int phi_idx,
                                         Pointer<NodeVariable<NDIM, double>> phi_var,
                                         double data_time,
                                         bool extended_box)
{
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            phi_data->fillAll(static_cast<double>(ln + 2));
        }
    }
    HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(d_hierarchy, 0, finest_ln);
    if (vol_idx != IBTK::invalid_index) hier_cc_data_ops.setToScalar(vol_idx, 0.0, false);
    if (area_idx != IBTK::invalid_index) hier_cc_data_ops.setToScalar(area_idx, 0.0, false);
    HierarchySideDataOpsReal<NDIM, double> hier_sc_data_ops(d_hierarchy, 0, finest_ln);
    if (side_idx != IBTK::invalid_index) hier_sc_data_ops.setToScalar(side_idx, 0.0, false);

    if (d_fe_sys_managers.size() > 0)
    {
        std::vector<std::unique_ptr<FEToHierarchyMapping>> fe_hier_mappings(getNumParts());
        for (size_t part = 0; part < getNumParts(); ++part)
        {
            static IntVector<NDIM> gcw = 1;
            fe_hier_mappings[part] =
                std::make_unique<FEToHierarchyMapping>(d_object_name + "::FEMapping_" + std::to_string(part),
                                                       d_fe_sys_managers[part],
                                                       nullptr,
                                                       d_hierarchy->getNumberOfLevels(),
                                                       gcw);
            fe_hier_mappings[part]->setPatchHierarchy(d_hierarchy);
            fe_hier_mappings[part]->reinitElementMappings(gcw);
        }
        d_cut_cell_mesh_mapping->generateCutCellMappings(unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings));
    }
    else
    {
        d_cut_cell_mesh_mapping->generateCutCellMappings(d_fe_data_managers);
    }
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_map_vec =
            d_cut_cell_mesh_mapping->getIdxCutCellElemsMap(ln);

        unsigned int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            if (idx_cut_cell_map_vec.size() > local_patch_num)
            {
                const std::map<IndexList, std::vector<CutCellElems>>& idx_cut_cell_map =
                    idx_cut_cell_map_vec[local_patch_num];
                Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);

                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const x_low = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();

                // Loop through all cut cells in map
                for (const auto& idx_elem_vec_pair : idx_cut_cell_map)
                {
                    const CellIndex<NDIM>& idx = idx_elem_vec_pair.first.d_idx;
                    const std::vector<CutCellElems>& cut_cell_elem_vec = idx_elem_vec_pair.second;

                    // Let's find distances from background cell nodes to structure
                    // Determine normal for elements
                    // Warning, we need the normal to be consistent between parent and child elements.
                    std::vector<IBTK::Vector3d> elem_normals;
                    for (const auto& cut_cell_elem : cut_cell_elem_vec)
                    {
                        // Note we use the parent element to calculate normals to preserve directions
                        Vector3d v, w;
                        const std::array<libMesh::Point, 2>& parent_pts = cut_cell_elem.d_parent_cur_pts;
                        const unsigned int part = cut_cell_elem.d_part;
                        v << parent_pts[0](0), parent_pts[0](1), parent_pts[0](2);
                        w << parent_pts[1](0), parent_pts[1](1), parent_pts[1](2);
                        const unsigned int domain_id = cut_cell_elem.d_parent_elem->subdomain_id();
                        Vector3d e3 = Vector3d::UnitZ();
                        if (!d_use_inside) e3 *= -1.0;
                        if (d_norm_reverse_domain_ids[part].find(domain_id) != d_norm_reverse_domain_ids[part].end() ||
                            d_norm_reverse_elem_ids[part].find(cut_cell_elem.d_parent_elem->id()) !=
                                d_norm_reverse_elem_ids[part].end() ||
                            d_reverse_normal[part])
                        {
                            e3 *= -1.0;
                        }
                        Vector3d n = (w - v).cross(e3);
                        elem_normals.push_back(n);
                    }
                    // Determine distances to nodes
                    for (int x = 0; x < 2; ++x)
                    {
                        for (int y = 0; y < 2; ++y)
                        {
                            Vector3d P = Vector3d::Zero();
                            for (int d = 0; d < NDIM; ++d)
                                P(d) = static_cast<double>(idx(d) - idx_low(d)) + (d == 0 ? x : y);
                            // Project P onto element
                            Vector3d avg_proj, avg_unit_normal;
                            avg_proj.setZero();
                            avg_unit_normal.setZero();
                            double min_dist = std::numeric_limits<double>::max();
                            int num_min = 0;
                            // Loop through all elements and calculate the smallest distance
                            for (unsigned int i = 0; i < elem_normals.size(); ++i)
                            {
                                const std::unique_ptr<Elem>& elem = cut_cell_elem_vec[i].d_elem;
                                const Vector3d& n = elem_normals[i];
                                Vector3d v, w;
                                v << (elem->point(0)(0) - x_low[0]) / dx[0], (elem->point(0)(1) - x_low[1]) / dx[1],
                                    0.0;
                                w << (elem->point(1)(0) - x_low[0]) / dx[0], (elem->point(1)(1) - x_low[1]) / dx[1],
                                    0.0;
                                const double t =
                                    std::max(0.0, std::min(1.0, (P - v).dot(w - v) / (v - w).squaredNorm()));
                                const Vector3d proj = v + t * (w - v);
                                VectorNd x_proj;
                                for (int d = 0; d < NDIM; ++d) x_proj[d] = x_low[d] + dx[d] * (proj(d) - idx_low(d));
                                const double dist = (proj - P).norm();
                                if (dist < min_dist)
                                {
                                    min_dist = dist;
                                    avg_proj = proj;
                                    avg_unit_normal = n;
                                    num_min = 1;
                                }
                                else if (MathUtilities<double>::equalEps(dist, min_dist))
                                {
                                    avg_proj += proj;
                                    avg_unit_normal += n;
                                    ++num_min;
                                }
                            }
                            avg_proj /= static_cast<double>(num_min);
                            avg_unit_normal /= static_cast<double>(num_min);
                            avg_unit_normal.normalize();

                            Vector3d phys_vec = Vector3d::Zero();
                            for (unsigned int d = 0; d < NDIM; ++d) phys_vec(d) = dx[d] * (P - avg_proj)[d];
                            double dist_phys = phys_vec.norm();
                            double sgn = (avg_unit_normal.dot(P - avg_proj) <= 0.0 ? -1.0 : 1.0);
                            NodeIndex<NDIM> n_idx(idx, IntVector<NDIM>(x, y));
                            (*phi_data)(n_idx) =
                                dist_phys < std::abs((*phi_data)(n_idx)) ? (dist_phys * sgn) : (*phi_data)(n_idx);
                            // Truncate distances that are too small
                            // TODO: find a better way to do this.
                            if ((*phi_data)(n_idx) < 0.0 && (*phi_data)(n_idx) > -1.0e-6) (*phi_data)(n_idx) = 0.0;
                        }
                    }
                }
            }
        }

        // Fill in physical boundary cells
        if (d_bdry_fcn)
        {
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(phi_idx);

                // Loop over boundary boxes
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const xlow = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();
                std::vector<Box<NDIM>> fill_boxes;
                for (int d = 1; d <= NDIM; ++d)
                {
                    const tbox::Array<BoundaryBox<NDIM>>& bdry_boxes = pgeom->getCodimensionBoundaries(d);
                    for (int i = 0; i < bdry_boxes.size(); ++i)
                    {
                        const BoundaryBox<NDIM>& bdry_box = bdry_boxes[i];
                        const int location_index = bdry_box.getLocationIndex();
                        const int axis = location_index % 2;
                        const int upper_lower = location_index / 2;
                        if (pgeom->getTouchesRegularBoundary(axis, upper_lower))
                        {
                            fill_boxes.push_back(
                                pgeom->getBoundaryFillBox(bdry_box, patch->getBox(), ls_data->getGhostCellWidth()));
                        }
                    }
                }

                for (const auto& box : fill_boxes)
                {
                    for (NodeIterator<NDIM> ni(box); ni; ni++)
                    {
                        const NodeIndex<NDIM>& idx = ni();
                        if ((*ls_data)(idx) == static_cast<double>(ln + 2))
                        {
                            // Change this value
                            VectorNd X_loc;
                            for (int d = 0; d < NDIM; ++d)
                                X_loc[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));
                            double ls_val = (*ls_data)(idx);
                            d_bdry_fcn(X_loc, ls_val);
                            (*ls_data)(idx) = ls_val;
                        }
                    }
                }
            }
        }
    }

    // Now update the LS away from the interface using a flood filling algorithm.
    updateLSAwayFromInterfaceNode(phi_idx);

    // Synchronize the hierarchy.
    {
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp(1);
        ghost_cell_comp[0] = ITC(phi_idx, "LINEAR_REFINE", false, "CONSTANT_COARSEN");
        HierarchyGhostCellInterpolation ghost_cells;
        ghost_cells.initializeOperatorState(ghost_cell_comp, d_hierarchy, 0, finest_ln);
        ghost_cells.fillData(data_time);
    }

    // Finally, find the volume/area/side lengths using the computed level set.
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        double tot_area = 0.0;
        double tot_vol = 0.0;
        double min_vol = std::numeric_limits<double>::max();
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            Pointer<CellData<NDIM, double>> area_data;
            if (area_idx != IBTK::invalid_index) area_data = patch->getPatchData(area_idx);
            Pointer<CellData<NDIM, double>> vol_data;
            if (vol_idx != IBTK::invalid_index) vol_data = patch->getPatchData(vol_idx);
            Pointer<SideData<NDIM, double>> side_data;
            if (side_idx != IBTK::invalid_index) side_data = patch->getPatchData(side_idx);

            const Box<NDIM>& box = extended_box ? phi_data->getGhostBox() : patch->getBox();
            const hier::Index<NDIM>& patch_lower = box.lower();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const double cell_volume = dx[0] * dx[1];
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - patch_lower(d));
                std::pair<double, double> vol_area_pair = find_volume_and_area(x, dx, phi_data, idx);
                double volume = vol_area_pair.first;
                double area = vol_area_pair.second;
                if (area_idx != IBTK::invalid_index)
                {
                    (*area_data)(idx) = area;
                    if (patch->getBox().contains(idx)) tot_area += area;
                }
                if (vol_idx != IBTK::invalid_index)
                {
                    (*vol_data)(idx) = volume / cell_volume;
                    if (patch->getBox().contains(idx)) tot_vol += volume;
                    min_vol = volume > 0.0 ? std::min(min_vol, volume) : min_vol;
                }

                if (side_idx != IBTK::invalid_index)
                {
                    for (int f = 0; f < 2; ++f)
                    {
#if (NDIM == 2)
                        double L = length_fraction(1.0,
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 0))),
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 1))));
#endif
#if (NDIM == 3)
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
#endif
                        (*side_data)(SideIndex<NDIM>(idx, 0, f)) = L;
                    }
                    for (int f = 0; f < 2; ++f)
                    {
#if (NDIM == 2)
                        double L = length_fraction(1.0,
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, f))),
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, f))));
#endif
#if (NDIM == 3)
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
#endif
                        (*side_data)(SideIndex<NDIM>(idx, 1, f)) = L;
                    }
#if (NDIM == 3)
                    for (int f = 0; f < 2; ++f)
                    {
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
                        (*side_data)(SideIndex<NDIM>(idx, 2, f)) = L;
                    }
#endif
                }
            }
        }
        tot_area = SAMRAI_MPI::sumReduction(tot_area);
        tot_vol = SAMRAI_MPI::sumReduction(tot_vol);
        min_vol = SAMRAI_MPI::minReduction(min_vol);
        plog << "Minimum volume on level:     " << ln << " is: " << std::setprecision(12) << min_vol << "\n";
        plog << "Total area found on level:   " << ln << " is: " << std::setprecision(12) << tot_area << "\n";
        plog << "Total volume found on level: " << ln << " is: " << std::setprecision(12) << tot_vol << "\n";
    }
    ADS_TIMER_STOP(t_updateVolumeAreaSideLS);
}

void
LSFromMesh::doUpdateLSCell(const int phi_idx,
                           Pointer<CellVariable<NDIM, double>> phi_var,
                           const double data_time,
                           const bool extended_box)
{
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            const Pointer<Patch<NDIM>>& patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            phi_data->fillAll(static_cast<double>(ln + 2));
        }
    }

    if (d_fe_sys_managers.size() > 0)
    {
        std::vector<std::unique_ptr<FEToHierarchyMapping>> fe_hier_mappings(getNumParts());
        for (size_t part = 0; part < getNumParts(); ++part)
        {
            static IntVector<NDIM> gcw = 1;
            fe_hier_mappings[part] =
                std::make_unique<FEToHierarchyMapping>(d_object_name + "::FEMapping_" + std::to_string(part),
                                                       d_fe_sys_managers[part],
                                                       nullptr,
                                                       d_hierarchy->getNumberOfLevels(),
                                                       gcw);
            fe_hier_mappings[part]->setPatchHierarchy(d_hierarchy);
            fe_hier_mappings[part]->reinitElementMappings(gcw);
        }
        d_cut_cell_mesh_mapping->generateCutCellMappings(unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings));
    }
    else
    {
        d_cut_cell_mesh_mapping->generateCutCellMappings(d_fe_data_managers);
    }
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& idx_cut_cell_map_vec =
            d_cut_cell_mesh_mapping->getIdxCutCellElemsMap(ln);

        unsigned int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            if (idx_cut_cell_map_vec.size() > local_patch_num)
            {
                const std::map<IndexList, std::vector<CutCellElems>>& idx_cut_cell_map =
                    idx_cut_cell_map_vec[local_patch_num];
                Pointer<CellData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);

                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const x_low = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();

                // Loop through all cut cells in map
                for (const auto& idx_elem_vec_pair : idx_cut_cell_map)
                {
                    const CellIndex<NDIM>& idx = idx_elem_vec_pair.first.d_idx;
                    const std::vector<CutCellElems>& cut_cell_elem_vec = idx_elem_vec_pair.second;

                    // Let's find distances from background cell nodes to structure
                    // Determine normal for elements
                    // Warning, we need the normal to be consistent between parent and child elements.
                    std::vector<IBTK::Vector3d> elem_normals;
                    for (const auto& cut_cell_elem : cut_cell_elem_vec)
                    {
                        // Note we use the parent element to calculate normals to preserve directions
                        Vector3d v, w;
                        const std::array<libMesh::Point, 2>& parent_pts = cut_cell_elem.d_parent_cur_pts;
                        const unsigned int part = cut_cell_elem.d_part;
                        v << parent_pts[0](0), parent_pts[0](1), parent_pts[0](2);
                        w << parent_pts[1](0), parent_pts[1](1), parent_pts[1](2);
                        const unsigned int domain_id = cut_cell_elem.d_parent_elem->subdomain_id();
                        Vector3d e3 = Vector3d::UnitZ();
                        if (!d_use_inside) e3 *= -1.0;
                        if (d_norm_reverse_domain_ids[part].find(domain_id) != d_norm_reverse_domain_ids[part].end() ||
                            d_norm_reverse_elem_ids[part].find(cut_cell_elem.d_parent_elem->id()) !=
                                d_norm_reverse_elem_ids[part].end() ||
                            d_reverse_normal[part])
                        {
                            e3 *= -1.0;
                        }
                        Vector3d n = (w - v).cross(e3);
                        elem_normals.push_back(n);
                    }

                    Box<NDIM> cc_box(idx, idx);
                    cc_box.grow(1);
                    for (CellIterator<NDIM> ci(cc_box); ci; ci++)
                    {
                        const CellIndex<NDIM>& new_idx = ci();
                        Vector3d P = Vector3d::Zero();
                        for (int d = 0; d < NDIM; ++d) P(d) = static_cast<double>(new_idx(d) - idx_low(d)) + 0.5;
                        // Project P onto element
                        Vector3d avg_proj, avg_unit_normal;
                        avg_proj.setZero();
                        avg_unit_normal.setZero();
                        double min_dist = std::numeric_limits<double>::max();
                        int num_min = 0;
                        // Loop through all elements and calculate the smallest distance
                        for (unsigned int i = 0; i < elem_normals.size(); ++i)
                        {
                            const std::unique_ptr<Elem>& elem = cut_cell_elem_vec[i].d_elem;
                            const Vector3d& n = elem_normals[i];
                            Vector3d v, w;
                            v << (elem->point(0)(0) - x_low[0]) / dx[0], (elem->point(0)(1) - x_low[1]) / dx[1], 0.0;
                            w << (elem->point(1)(0) - x_low[0]) / dx[0], (elem->point(1)(1) - x_low[1]) / dx[1], 0.0;
                            const double t = std::max(0.0, std::min(1.0, (P - v).dot(w - v) / (v - w).squaredNorm()));
                            const Vector3d proj = v + t * (w - v);
                            VectorNd x_proj;
                            for (int d = 0; d < NDIM; ++d) x_proj[d] = x_low[d] + dx[d] * (proj(d) - idx_low(d));
                            const double dist = (proj - P).norm();
                            if (dist < min_dist)
                            {
                                min_dist = dist;
                                avg_proj = proj;
                                avg_unit_normal = n;
                                num_min = 1;
                            }
                            else if (MathUtilities<double>::equalEps(dist, min_dist))
                            {
                                avg_proj += proj;
                                avg_unit_normal += n;
                                ++num_min;
                            }
                        }
                        avg_proj /= static_cast<double>(num_min);
                        avg_unit_normal /= static_cast<double>(num_min);
                        avg_unit_normal.normalize();

                        Vector3d phys_vec = Vector3d::Zero();
                        for (unsigned int d = 0; d < NDIM; ++d) phys_vec(d) = dx[d] * (P - avg_proj)[d];
                        double dist_phys = phys_vec.norm();
                        double sgn = (avg_unit_normal.dot(P - avg_proj) <= 0.0 ? -1.0 : 1.0);
                        (*phi_data)(new_idx) =
                            dist_phys < std::abs((*phi_data)(new_idx)) ? (dist_phys * sgn) : (*phi_data)(new_idx);
                        // Truncate distances that are too small
                        // TODO: find a better way to do this.
                        if ((*phi_data)(new_idx) < 0.0 && (*phi_data)(new_idx) > -1.0e-6) (*phi_data)(new_idx) = 0.0;
                    }
                }
            }
        }

        // Fill in physical boundary cells
        if (d_bdry_fcn)
        {
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> ls_data = patch->getPatchData(phi_idx);

                // Loop over boundary boxes
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const xlow = pgeom->getXLower();
                const double* const dx = pgeom->getDx();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();
                std::vector<Box<NDIM>> fill_boxes;
                for (int d = 1; d <= NDIM; ++d)
                {
                    const tbox::Array<BoundaryBox<NDIM>>& bdry_boxes = pgeom->getCodimensionBoundaries(d);
                    for (int i = 0; i < bdry_boxes.size(); ++i)
                    {
                        const BoundaryBox<NDIM>& bdry_box = bdry_boxes[i];
                        const int location_index = bdry_box.getLocationIndex();
                        const int axis = location_index % 2;
                        const int upper_lower = location_index / 2;
                        if (pgeom->getTouchesRegularBoundary(axis, upper_lower))
                        {
                            fill_boxes.push_back(
                                pgeom->getBoundaryFillBox(bdry_box, patch->getBox(), ls_data->getGhostCellWidth()));
                        }
                    }
                }

                for (const auto& box : fill_boxes)
                {
                    for (CellIterator<NDIM> ni(box); ni; ni++)
                    {
                        const CellIndex<NDIM>& idx = ni();
                        if ((*ls_data)(idx) == static_cast<double>(ln + 2))
                        {
                            // Change this value
                            VectorNd X_loc;
                            for (int d = 0; d < NDIM; ++d)
                                X_loc[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
                            double ls_val = (*ls_data)(idx);
                            d_bdry_fcn(X_loc, ls_val);
                            (*ls_data)(idx) = ls_val;
                        }
                    }
                }
            }
        }
    }

    // Now update the LS away from the interface using a flood filling algorithm.
    updateLSAwayFromInterfaceCell(phi_idx);

    // Synchronize the hierarchy.
    {
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp(1);
        ghost_cell_comp[0] = ITC(phi_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN");
        HierarchyGhostCellInterpolation ghost_cells;
        ghost_cells.initializeOperatorState(ghost_cell_comp, d_hierarchy, 0, finest_ln);
        ghost_cells.fillData(data_time);
    }
    return;
}

void
LSFromMesh::updateLSAwayFromInterfaceNode(const int phi_idx)
{
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    HierarchyNodeDataOpsReal<NDIM, double> hier_nc_data_ops(d_hierarchy, 0, finest_ln);
    Pointer<PatchLevel<NDIM>> finest_level = d_hierarchy->getPatchLevel(finest_ln);
    // Now we need to update the sign of phi_data.
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        d_hierarchy->getPatchLevel(ln)->allocatePatchData(d_sgn_nc_idx);
    }
    hier_nc_data_ops.copyData(d_sgn_nc_idx, phi_idx, false);
    flood_fill_for_LS(d_sgn_nc_idx, d_sgn_nc_var, static_cast<double>(finest_ln + 2), finest_level);
    // At this point, the finest level has been filled in. We now need to fill in coarser levels
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();
    Pointer<CoarsenOperator<NDIM>> coarsen_op = grid_geom->lookupCoarsenOperator(d_sgn_nc_var, "CONSTANT_COARSEN");
    Pointer<CoarsenAlgorithm<NDIM>> coarsen_alg = new CoarsenAlgorithm<NDIM>();
    coarsen_alg->registerCoarsen(d_sgn_nc_idx, d_sgn_nc_idx, coarsen_op);
    std::vector<Pointer<CoarsenSchedule<NDIM>>> coarsen_scheds(finest_ln + 1);
    for (int ln = finest_ln; ln > 0; --ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        Pointer<PatchLevel<NDIM>> coarser_level = d_hierarchy->getPatchLevel(ln - 1);
        coarsen_scheds[ln] = coarsen_alg->createSchedule(coarser_level, level);
        coarsen_scheds[ln]->coarsenData();
        flood_fill_for_LS(d_sgn_nc_idx, d_sgn_nc_var, static_cast<double>(ln + 1), coarser_level);
    }

    hier_nc_data_ops.copyData(phi_idx, d_sgn_nc_idx, false);
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        d_hierarchy->getPatchLevel(ln)->deallocatePatchData(d_sgn_nc_idx);
    }
}

void
LSFromMesh::updateLSAwayFromInterfaceCell(const int phi_idx)
{
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(d_hierarchy, 0, finest_ln);
    Pointer<PatchLevel<NDIM>> finest_level = d_hierarchy->getPatchLevel(finest_ln);
    // Now we need to update the sign of phi_data.
    for (int ln = 0; ln <= finest_ln; ++ln) d_hierarchy->getPatchLevel(ln)->allocatePatchData(d_sgn_cc_idx);
    hier_cc_data_ops.copyData(d_sgn_cc_idx, phi_idx, false);
    flood_fill_for_LS(d_sgn_cc_idx, d_sgn_cc_var, static_cast<double>(finest_ln + 2), finest_level);
    // At this point, the finest level has been filled in. We now need to fill in coarser levels
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();
    Pointer<CoarsenOperator<NDIM>> coarsen_op = grid_geom->lookupCoarsenOperator(d_sgn_cc_var, "CONSERVATIVE_COARSEN");
    Pointer<CoarsenAlgorithm<NDIM>> coarsen_alg = new CoarsenAlgorithm<NDIM>();
    coarsen_alg->registerCoarsen(d_sgn_cc_idx, d_sgn_cc_idx, coarsen_op);
    std::vector<Pointer<CoarsenSchedule<NDIM>>> coarsen_scheds(finest_ln + 1);
    for (int ln = finest_ln; ln > 0; --ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        Pointer<PatchLevel<NDIM>> coarser_level = d_hierarchy->getPatchLevel(ln - 1);
        coarsen_scheds[ln] = coarsen_alg->createSchedule(coarser_level, level);
        coarsen_scheds[ln]->coarsenData();
        flood_fill_for_LS(d_sgn_cc_idx, d_sgn_cc_var, static_cast<double>(ln + 1), coarser_level);
    }

    hier_cc_data_ops.copyData(phi_idx, d_sgn_cc_idx, false);
    for (int ln = 0; ln <= finest_ln; ++ln) d_hierarchy->getPatchLevel(ln)->deallocatePatchData(d_sgn_cc_idx);
}

} // namespace ADS
