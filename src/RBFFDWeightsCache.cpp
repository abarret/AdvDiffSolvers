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

unsigned int RBFFDWeightsCache::s_num_ghost_cells = 3;
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
    d_idx_node_ghost_vec.clear();
    d_base_pt_set.clear();
    d_pair_pt_map.clear();
    // Assume structure is on finest level
    int ln = d_hierarchy->getFinestLevelNumber();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);

    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const MeshBase& mesh = eq_sys->get_mesh();
    const System& sys = eq_sys->get_system(d_fe_mesh_partitioner->COORDINATES_SYSTEM_NAME);
    const DofMap& dof_map = sys.get_dof_map();
    NumericVector<double>* X_vec = sys.current_local_solution.get();

    // Loop through nodes
    auto it_end = mesh.nodes_end();
    for (auto it = mesh.nodes_begin(); it != it_end; ++it)
    {
        Node* const node = *it;
        // Get CellIndex of the node
        VectorNd node_pt;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            std::vector<dof_id_type> dofs;
            dof_map.dof_indices(node, dofs, d);
            node_pt[d] = (*X_vec)(dofs[0]);
        }
        const CellIndex<NDIM>& idx =
            IndexUtilities::getCellIndex(node_pt, d_hierarchy->getGridGeometry(), level->getRatio());
        // Sort nodes into patches
        // We also need ghost nodes to include in our KD tree
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            if (box.contains(idx)) d_idx_node_vec[patch.getPointer()].push_back(node);
            Box<NDIM> ghost_box = box;
            ghost_box.grow(IntVector<NDIM>(s_num_ghost_cells));
            if (ghost_box.contains(idx)) d_idx_node_ghost_vec[patch.getPointer()].push_back(node);
        }
    }
    // At this point, each node is associated with a patch
    // We now find the nearest neighbor of every point that needs an RBF reconstruction
    // We do this by finding a KD tree
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        // We are on a patch that has points. We need to form a KD tree.
        // TODO: Need to determine when we need to use RBFs
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        Box<NDIM> ghost_box = patch->getBox();
        ghost_box.grow(ls_data->getGhostCellWidth());

        // First start by collecting all points into a vector
        std::vector<FDCachedPoint> pts;
        for (CellIterator<NDIM> ci(ghost_box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double ls_val = ADS::node_to_cell(idx, *ls_data);
            if (ls_val < -d_eps) pts.push_back(FDCachedPoint(patch, idx));
        }
        // Now add in the points in d_idx_node_vec
        for (const auto& node : d_idx_node_ghost_vec[patch.getPointer()])
        {
            VectorNd node_pt;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                std::vector<dof_id_type> dofs;
                dof_map.dof_indices(node, dofs, d);
                node_pt[d] = (*X_vec)(dofs[0]);
            }
            pts.push_back(FDCachedPoint(node_pt, node, false));
        }

        // Now create KD tree
        tree::KDTree<FDCachedPoint> tree(pts);
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
                FDCachedPoint base_pt(patch, idx);
                d_base_pt_set[patch.getPointer()].insert(base_pt);
                d_pair_pt_map[patch.getPointer()][base_pt].reserve(d_stencil_size);
#if (1)
                // Use KNN search.
                tree.knnSearch(FDCachedPoint(patch, idx), d_stencil_size, idx_vec, distance_vec);
#else
                // Use a bounding box search
                VectorNd bbox;
                bbox(0) = 2.0 * dx[0];
                bbox(1) = 2.0 * dx[1];
                tree.cuboid_query(FDCachedPoint(patch, idx), bbox, idx_vec, distance_vec);
#endif
                // Add these points to the vector
                for (const auto& idx_in_pts : idx_vec)
                    d_pair_pt_map[patch.getPointer()][base_pt].push_back(pts[idx_in_pts]);
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
            FDCachedPoint base_pt(node_pt, node, false);
            d_base_pt_set[patch.getPointer()].insert(base_pt);
            d_pair_pt_map[patch.getPointer()][base_pt].reserve(d_stencil_size);
            tree.knnSearch(FDCachedPoint(node_pt, node, false), d_stencil_size, idx_vec, distance_vec);
            // Add these points to the vector
            for (const auto& idx_in_pts : idx_vec)
                d_pair_pt_map[patch.getPointer()][base_pt].push_back(pts[idx_in_pts]);
        }
    }
}

void
RBFFDWeightsCache::findRBFFDWeights()
{
    sortLagDOFsToCells();
    d_pt_weight_map.clear();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
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
            const std::vector<FDCachedPoint>& pt_vec = d_pair_pt_map[patch.getPointer()][base_pt];
            const std::vector<VectorNd>& shft_vec = Reconstruct::shift_pts(pt_vec, IBTK::VectorNd::Zero());
            d_pt_weight_map[patch.getPointer()][base_pt].reserve(pt_vec.size());
            // Note if we use a KNN search, interp_size is fixed.
            const int interp_size = pt_vec.size();
#if !defined(NDEBUG)
            TBOX_ASSERT(interp_size == d_stencil_size);
#endif
            MatrixXd A(MatrixXd::Zero(interp_size, interp_size));
            MatrixXd B = PolynomialBasis::formMonomials(shft_vec, d_poly_degree, dx[0], base_pt.getVec());
            const int poly_size = B.cols();
            VectorXd U(VectorXd::Zero(interp_size + poly_size));
            VectorNd pt0 = base_pt.getVec();
            for (int i = 0; i < interp_size; ++i)
            {
                VectorNd pti = pt_vec[i].getVec();
                for (int j = 0; j < interp_size; ++j)
                {
                    VectorNd ptj = pt_vec[j].getVec();
                    A(i, j) = d_rbf_fcn((pti - ptj).norm());
                }
                // Determine rhs
                U(i) = d_Lrbf_fcn((pt0 - pti).norm());
            }
            // Add quadratic polynomials
            std::vector<VectorNd> zeros = { base_pt.getVec() };
            MatrixXd Ulow = d_poly_fcn(zeros, d_poly_degree, dx[0], base_pt.getVec());
            U.block(interp_size, 0, Ulow.cols(), 1) = Ulow.transpose();
            MatrixXd final_mat(MatrixXd::Zero(interp_size + poly_size, interp_size + poly_size));
            final_mat.block(0, 0, interp_size, interp_size) = A;
            final_mat.block(0, interp_size, interp_size, poly_size) = B;
            final_mat.block(interp_size, 0, poly_size, interp_size) = B.transpose();

            VectorXd x = final_mat.colPivHouseholderQr().solve(U);
            // Now cache FD stencil
            VectorXd weights = x.block(0, 0, interp_size, 1);
            for (int i = 0; i < interp_size; ++i) d_pt_weight_map[patch.getPointer()][base_pt].push_back(weights(i));
        }
    }
}
//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
