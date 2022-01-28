#include "ADS/GhostPoints.h"
#include "ADS/app_namespaces.h"

#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/libmesh_utilities.h"

#include "libmesh/enum_xdr_mode.h"
#include "libmesh/explicit_system.h"

#include <utility>
namespace ADS
{
GhostPoints::GhostPoints(std::string object_name,
                         Pointer<Database> input_db,
                         Pointer<PatchHierarchy<NDIM>> hierarchy,
                         std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner)
    : d_object_name(std::move(object_name)),
      d_hierarchy(hierarchy),
      d_fe_mesh_partitioner(std::move(fe_mesh_partitioner))
{
    commonConstructor(input_db);
}

void
GhostPoints::commonConstructor(Pointer<Database> input_db)
{
    EquationSystems* bdry_eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    auto& N_sys = bdry_eq_sys->add_system<ExplicitSystem>(d_normal_sys_name);
    for (unsigned int d = 0; d < NDIM; ++d) N_sys.add_variable("N_" + std::to_string(d), FEType());
    N_sys.assemble_before_solve = false;
    N_sys.assemble();

    d_ds = input_db->getDouble("ds");
    input_db->getDoubleArray("com", d_com.data(), NDIM);
    return;
}

void
GhostPoints::updateGhostNodeLocations(const double time, const bool end_of_timestep)
{
    // TODO: We should move points, but for now clear them, and set them up again
    d_eul_ghost_nodes.clear();
    d_lag_ghost_nodes.clear();
    setupGhostNodes();
}

void
GhostPoints::findNormals()
{
    // This is not entirely clear how to do. For now we use prescribed geometries.
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    auto& N_sys = eq_sys->get_system<ExplicitSystem>(d_normal_sys_name);
    const DofMap& N_dof_map = N_sys.get_dof_map();
    NumericVector<double>* N_vec = N_sys.solution.get();

    const MeshBase& mesh = eq_sys->get_mesh();
    auto it = mesh.local_nodes_begin();
    auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const n = *it;
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = (*n)(d)-d_com(d);
        x.normalize();
        std::vector<dof_id_type> dofs;
        N_dof_map.dof_indices(n, dofs);
        for (int d = 0; d < NDIM; ++d) N_vec->set(dofs[d], x[d]);
    }
    N_vec->close();
    N_sys.update();
}

void
GhostPoints::setupGhostNodes()
{
    d_local_dofs = 0;
    // First we do Eulerian boundary nodes.
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = box.lower();
            // Determine if this patch touches a physical boundary.
            if (!pgeom->getTouchesRegularBoundary()) continue;
            // Now we loop over the boundary and create ghost nodes for that boundary
            const tbox::Array<BoundaryBox<NDIM>>& bdry_boxes = pgeom->getCodimensionBoundaries(1);
            for (int i = 0; i < bdry_boxes.size(); ++i)
            {
                const Box<NDIM>& fill_box = pgeom->getBoundaryFillBox(bdry_boxes[i], patch->getBox(), 1);
                for (CellIterator<NDIM> ci(fill_box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    // Now we are looping over the boundary box. Create a node.
                    libMesh::Point pt;
                    for (int d = 0; d < NDIM; ++d)
                        pt(d) = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
                    d_eul_ghost_nodes.push_back(Node::build(pt, d_local_dofs++));
                }
            }
        }
    }

    // Now we do Lagrangian points
    EquationSystems* bdry_eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const MeshBase& bdry_mesh = bdry_eq_sys->get_mesh();
    System& X_sys = bdry_eq_sys->get_system(d_fe_mesh_partitioner->COORDINATES_SYSTEM_NAME);
    const DofMap& X_dof_map = X_sys.get_dof_map();
    NumericVector<double>* X_vec = X_sys.current_local_solution.get();
    System& N_sys = bdry_eq_sys->get_system(d_normal_sys_name);
    NumericVector<double>* N_vec = N_sys.current_local_solution.get();
    const DofMap& N_dof_map = N_sys.get_dof_map();

    // Loop through boundary nodes, and move in normal direction
    const auto ni_end = bdry_mesh.local_nodes_end();
    auto ni = bdry_mesh.local_nodes_begin();
    for (; ni != ni_end; ++ni)
    {
        const Node* const node = *ni;
        libMesh::Point pt;
        std::vector<dof_id_type> X_dofs, N_dofs;
        X_dof_map.dof_indices(node, X_dofs);
        N_dof_map.dof_indices(node, N_dofs);
        for (unsigned int d = 0; d < NDIM; ++d) pt(d) = (*X_vec)(X_dofs[d]);
        // Now move ds in normal direction
        for (unsigned int d = 0; d < NDIM; ++d) pt(d) += d_ds * (*N_vec)(N_dofs[d]);
        // Now add this point to the list of Lagrangian ghost nodes
        d_lag_ghost_nodes.push_back(Node::build(pt, d_local_dofs++));
    }

    d_global_dofs = IBTK_MPI::sumReduction(static_cast<int>(d_local_dofs));
}
} // namespace ADS
