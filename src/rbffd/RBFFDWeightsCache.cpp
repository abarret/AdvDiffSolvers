// ---------------------------------------------------------------------
//
// Copyright (c) 2021 - 2021 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/KDTree.h"
#include "ADS/PolynomialBasis.h"
#include "ADS/RBFFDWeightsCache.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"
#include "ADS/reconstructions.h"
#include <ADS/libmesh_utilities.h>

#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/IBTK_CHKERRQ.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/ibtk_utilities.h"

#include "CellVariable.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "PoissonSpecifications.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Timer.h"

#include <Eigen/Dense>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Timers.
static Timer* t_apply;
static Timer* t_initialize_operator_state;
static Timer* t_deallocate_operator_state;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

RBFFDWeightsCache::RBFFDWeightsCache(std::string object_name,
                                     std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                                     Pointer<PatchHierarchy<NDIM>> hierarchy,
                                     Pointer<Database> input_db)
    : FDWeightsCache(std::move(object_name)), d_hierarchy(hierarchy), d_fe_mesh_partitioner(fe_mesh_partitioner)
{
    d_poly_degree = input_db->getInteger("polynomial_degree");
    d_dist_to_bdry = input_db->getDouble("dist_to_bdry");
    d_eps = input_db->getDouble("eps");
    d_stencil_size = input_db->getInteger("stencil_size");
    d_num_ghost_cells = input_db->getIntegerWithDefault("num_ghost_cells", d_num_ghost_cells);
    if (d_num_ghost_cells < d_stencil_size)
        TBOX_WARNING(
            "Number of ghost cells is less than stencil size. This could force one-sided stencils near patch "
            "boundaries.\n");
    // Setup Timers.
    IBTK_DO_ONCE(t_apply = TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::apply()");
                 t_initialize_operator_state =
                     TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::initializeOperatorState()");
                 t_deallocate_operator_state =
                     TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::deallocateOperatorState()"););
    return;
}

RBFFDWeightsCache::~RBFFDWeightsCache()
{
    clearCache();
    return;
}

void
RBFFDWeightsCache::clearCache()
{
    FDWeightsCache::clearCache();
    d_weights_found = false;
}

void
RBFFDWeightsCache::sortLagDOFsToCells()
{
    // Fill ghost cells for level set.
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps;
    ghost_cell_comps.push_back(ITC(d_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR"));
    HierarchyGhostCellInterpolation ghost_cell_fill;
    ghost_cell_fill.initializeOperatorState(ghost_cell_comps, d_hierarchy);
    ghost_cell_fill.fillData(0.0);

    // Clear old data structures.
    d_idx_node_vec.clear();
    d_base_pt_set.clear();
    d_pair_pt_map.clear();
    // Assume structure is on finest level
    int ln = d_hierarchy->getFinestLevelNumber();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);

    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const System& sys = eq_sys->get_system(d_fe_mesh_partitioner->COORDINATES_SYSTEM_NAME);
    const DofMap& dof_map = sys.get_dof_map();
    NumericVector<double>* X_vec = sys.current_local_solution.get();

    if (d_fe_mesh_partitioner->getGhostCellWidth().max() < d_num_ghost_cells)
        TBOX_WARNING(
            "   FEMeshPartitioner has less ghost cell width than the SAMRAI patch data.\n   This can result in one "
            "sided stencils.\n");
    NumericVector<double>* X_ghosted_vec = d_fe_mesh_partitioner->buildGhostedCoordsVector();

    const std::vector<std::vector<Node*>>& active_nodes_map = d_fe_mesh_partitioner->getActivePatchNodeMap();

    // Loop through nodes
    unsigned int patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const std::vector<Node*>& active_nodes = active_nodes_map[patch_num];
        for (const auto& node : active_nodes)
        {
            VectorNd node_pt;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                std::vector<dof_id_type> dofs;
                dof_map.dof_indices(node, dofs, d);
                node_pt[d] = (*X_ghosted_vec)(dofs[0]);
            }
            const CellIndex<NDIM>& idx =
                IndexUtilities::getCellIndex(node_pt, d_hierarchy->getGridGeometry(), level->getRatio());
            if (patch->getBox().contains(idx)) d_idx_node_vec[patch.getPointer()].push_back(node);
        }
    }
    // At this point, each node is associated with a patch.

    // We now find the nearest neighbor of every point that needs an RBF reconstruction
    // We do this by finding a KD tree
    patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        // We are on a patch that has points. We need to form a KD tree.
        // TODO: Need to determine when we need to use RBFs
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        Box<NDIM> ghost_box = patch->getBox();
        ghost_box.grow(ls_data->getGhostCellWidth());

        // First start by collecting all points into a vector
        std::vector<FDPoint> pts;
        for (CellIterator<NDIM> ci(ghost_box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double ls_val = ADS::node_to_cell(idx, *ls_data);
            if (ls_val < -d_eps) pts.push_back(FDPoint(patch, idx));
        }
        // Now add in lagrangian nodes
        for (const auto& node : active_nodes_map[patch_num])
        {
            VectorNd node_pt;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                std::vector<dof_id_type> dofs;
                dof_map.dof_indices(node, dofs, d);
                node_pt[d] = (*X_ghosted_vec)(dofs[0]);
            }
            pts.push_back(FDPoint(node_pt, node));
        }

        // Now create KD tree
        tree::KDTree<FDPoint> tree(pts);
        // We have a tree, now we need to find closest points for each point.
        // Start with Eulerian points
        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double ls = ADS::node_to_cell(idx, *ls_data);
            if (ls < -d_eps && std::abs(ls) < d_dist_to_bdry)
            {
                std::vector<int> idx_vec;
                std::vector<double> distance_vec;
                FDPoint base_pt(patch, idx);
                d_base_pt_set[patch.getPointer()].insert(base_pt);
                std::vector<FDPoint> fd_pts;
                fd_pts.reserve(d_stencil_size);
#if (1)
                // Use KNN search.
                tree.knnSearch(FDPoint(patch, idx), d_stencil_size, idx_vec, distance_vec);
#else
                // Use a bounding box search
                VectorNd bbox;
                bbox(0) = 2.0 * dx[0];
                bbox(1) = 2.0 * dx[1];
                tree.cuboid_query(FDPoint(patch, idx), bbox, idx_vec, distance_vec);
#endif
                // Add these points to the vector
                for (const auto& idx_in_pts : idx_vec) fd_pts.push_back(pts[idx_in_pts]);
                d_pair_pt_map[patch.getPointer()][base_pt] = fd_pts;
            }
        }
        // Now do Lagrangian points
        for (const auto& node : d_idx_node_vec[patch.getPointer()])
        {
            VectorNd node_pt;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                std::vector<dof_id_type> dofs;
                dof_map.dof_indices(node, dofs, d);
                node_pt[d] = (*X_vec)(dofs[0]);
            }
            std::vector<int> idx_vec;
            std::vector<double> distance_vec;
            FDPoint base_pt(node_pt, node);
            d_base_pt_set[patch.getPointer()].insert(base_pt);
            std::vector<FDPoint> fd_pts;
            fd_pts.reserve(d_stencil_size);
            tree.knnSearch(FDPoint(node_pt, node), d_stencil_size, idx_vec, distance_vec);
            // Add these points to the vector
            for (const auto& idx_in_pts : idx_vec) fd_pts.push_back(pts[idx_in_pts]);
            d_pair_pt_map[patch.getPointer()][base_pt] = std::move(fd_pts);
        }
    }
}

void
RBFFDWeightsCache::findRBFFDWeights()
{
    sortLagDOFsToCells();
    d_pt_weight_map.clear();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    auto Lrbf_fcn = [this](const FDPoint& pti, const FDPoint& ptj, void*) -> double
    { return d_Lrbf_fcn((pti - ptj).norm()); };
    auto Lpoly_fcn =
        [this](const std::vector<FDPoint>& pts, int poly_deg, double ds, const FDPoint& base_pt, void*) -> VectorXd
    { return d_poly_fcn(pts, poly_deg, ds, base_pt).transpose(); };
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        if (d_base_pt_set[patch.getPointer()].size() == 0) continue;
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        // First loop through Cartesian grid cells.
        Pointer<CellData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        // Note Lagrangian data are located in d_aug_(x|b)_vec

        // All data have been sorted. We need to loop through d_base_pt_vec.
        for (const auto& base_pt : d_base_pt_set[patch.getPointer()])
        {
            const std::vector<FDPoint>& pt_vec = d_pair_pt_map.at(patch.getPointer()).at(base_pt);
            std::vector<double> wgts;
            Reconstruct::RBFFD_reconstruct<FDPoint>(
                wgts, base_pt, pt_vec, d_poly_degree, dx, d_rbf_fcn, Lrbf_fcn, nullptr, Lpoly_fcn, nullptr);
            d_pt_weight_map[patch.getPointer()][base_pt] = std::move(wgts);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
