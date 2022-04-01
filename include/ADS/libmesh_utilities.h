#ifndef included_ADS_libmesh_utilities
#define included_ADS_libmesh_utilities
#include "ibamr/config.h"

#include <ibtk/FEDataManager.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>

#include <tbox/Pointer.h>

#include <libmesh/elem.h>

#include <CartesianGridGeometry.h>
#include <CellData.h>
#include <FaceData.h>
#include <Patch.h>
#include <SideData.h>

namespace ADS
{
inline void
collectGhostDOFIndices(std::vector<unsigned int>& ghost_dofs,
                       const std::set<libMesh::Node*>& active_nodes,
                       const std::string& sys_name,
                       const std::shared_ptr<IBTK::FEData>& fe_data)
{
    libMesh::System& system = fe_data->getEquationSystems()->get_system(sys_name);
    const unsigned int sys_num = system.number();
    const libMesh::DofMap& dof_map = system.get_dof_map();
    const unsigned int first_local_dof = dof_map.first_dof();
    const unsigned int end_local_dof = dof_map.end_dof();

    // Record the local DOFs associated with the active local nodes.
    std::set<unsigned int> ghost_dof_set;
    for (const auto& node : active_nodes)
    {
        // DOFs associated with the element
        for (unsigned int var_num = 0; var_num < node->n_vars(sys_num); ++var_num)
        {
            if (node->n_dofs(sys_num, var_num) > 0)
            {
                const unsigned int dof_index = node->dof_number(sys_num, var_num, 0);
                if (dof_index < first_local_dof || dof_index >= end_local_dof) ghost_dof_set.insert(dof_index);
            }
        }
    }

    // Now fill in the vector
    ghost_dofs.clear();
    ghost_dofs.insert(ghost_dofs.end(), ghost_dof_set.begin(), ghost_dof_set.end());
}

/*!
 * \brief Associates all the active elements with a patch.
 *
 * \note This assumes that all
 */
inline void
collectActivePatchElements(std::vector<std::vector<libMesh::Elem*>>& active_patch_elems,
                           const std::shared_ptr<IBTK::FEData>& fe_data,
                           const std::string& coords_sys_name,
                           SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level,
                           const SAMRAI::hier::IntVector<NDIM>& ghost_width)
{
    // Get the necessary FE data.
    const libMesh::MeshBase& mesh = fe_data->getEquationSystems()->get_mesh();
    libMesh::System& X_system = fe_data->getEquationSystems()->get_system(coords_sys_name);

    // Setup data structures used to assign elements to patches.
    const SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianGridGeometry<NDIM>> grid_geom = level->getGridGeometry();
    const int num_local_patches = level->getProcessorMapping().getNumberOfLocalIndices();
    std::vector<std::set<libMesh::Elem*>> local_patch_elems(num_local_patches);
    active_patch_elems.resize(num_local_patches);

    // We associate an element with a Cartesian grid patch if the element's
    // bounding box (which is computed based on the bounds of quadrature
    // points) intersects the patch interior grown by the specified ghost cell
    // width.
    double dx_0 = std::numeric_limits<double>::max();
    for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
        const SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        dx_0 = std::min(dx_0, *std::min_element(pgeom->getDx(), pgeom->getDx() + NDIM));
    }
    dx_0 = IBTK::IBTK_MPI::minReduction(dx_0);
    TBOX_ASSERT(dx_0 != std::numeric_limits<double>::max());

    // be a bit paranoid by computing bounding boxes for elements as the union
    // of the bounding box of the nodes and the bounding box of the quadrature
    // points:
    const std::vector<IBTK::libMeshWrappers::BoundingBox> local_bboxes =
        IBTK::get_local_element_bounding_boxes(mesh, X_system);

    const std::vector<IBTK::libMeshWrappers::BoundingBox> global_bboxes =
        IBTK::get_global_element_bounding_boxes(mesh, local_bboxes);

    int local_patch_num = 0;
    for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        std::set<libMesh::Elem*>& elems = local_patch_elems[local_patch_num];
        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
        const SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        // TODO: reimplement this with an rtree description of SAMRAI's patches
        IBTK::libMeshWrappers::BoundingBox patch_bbox;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            patch_bbox.first(d) = pgeom->getXLower()[d] - dx[d] * ghost_width[d];
            patch_bbox.second(d) = pgeom->getXUpper()[d] + dx[d] * ghost_width[d];
        }
        for (unsigned int d = NDIM; d < LIBMESH_DIM; ++d)
        {
            patch_bbox.first(d) = 0.0;
            patch_bbox.second(d) = 0.0;
        }

        auto el_it = mesh.active_elements_begin();
        for (const IBTK::libMeshWrappers::BoundingBox& bbox : global_bboxes)
        {
#if LIBMESH_VERSION_LESS_THAN(1, 2, 0)
            if (bbox.intersect(patch_bbox)) elems.insert(*el_it);
#else
            if (bbox.intersects(patch_bbox)) elems.insert(*el_it);
#endif
            ++el_it;
        }
    }

    // Set the active patch element data.
    local_patch_num = 0;
    for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        const std::set<libMesh::Elem*>& local_elems = local_patch_elems[local_patch_num];
        std::vector<libMesh::Elem*>& active_elems = active_patch_elems[local_patch_num];
        active_elems.resize(local_elems.size());
        std::copy(local_elems.begin(), local_elems.end(), active_elems.begin());
    }

    return;
} // collectActivePatchElements

inline void
collectActivePatchNodes(std::vector<std::vector<libMesh::Node*>>& active_patch_nodes,
                        const std::vector<std::vector<libMesh::Elem*>>& active_patch_elems,
                        const std::shared_ptr<IBTK::FEData>& fe_data)
{
    const libMesh::MeshBase& mesh = fe_data->getEquationSystems()->get_mesh();
    const unsigned int num_local_patches = active_patch_elems.size();
    active_patch_nodes.resize(num_local_patches);
    for (unsigned int k = 0; k < num_local_patches; ++k)
    {
        std::set<libMesh::dof_id_type> active_node_ids;
        for (const auto& elem : active_patch_elems[k])
        {
            for (unsigned int n = 0; n < elem->n_nodes(); ++n)
            {
                active_node_ids.insert(elem->node_id(n));
            }
        }
        const unsigned int num_active_nodes = active_node_ids.size();
        active_patch_nodes[k].reserve(num_active_nodes);
        for (const auto& active_node_id : active_node_ids)
        {
            active_patch_nodes[k].push_back(const_cast<libMesh::Node*>(mesh.node_ptr(active_node_id)));
        }
    }
    return;
}
} // namespace ADS
#endif /* included_libmesh_utilities */
