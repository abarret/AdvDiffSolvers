/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/LagrangeStructureReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"

#include "SAMRAIVectorReal.h"

#include <libmesh/explicit_system.h>

#include <utility>

namespace
{
static Timer* t_apply_reconstruction;
}

namespace ADS
{
LagrangeStructureReconstructions::LagrangeStructureReconstructions(std::string object_name, Pointer<Database> input_db)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    d_rbf_stencil_size = input_db->getInteger("stencil_size");
    d_rbf_order = Reconstruct::string_to_enum<Reconstruct::RBFPolyOrder>(input_db->getString("rbf_order"));
    d_low_cutoff = input_db->getDoubleWithDefault("low_cutoff", d_low_cutoff);
    d_high_cutoff = input_db->getDoubleWithDefault("high_cutoff", d_high_cutoff);
    d_default_value = input_db->getDoubleWithDefault("default_value", d_default_value);

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(
        d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), std::ceil(0.5 * d_rbf_stencil_size));
    IBTK_DO_ONCE(t_apply_reconstruction = TimerManager::getManager()->getTimer(
                     "ADS::LagrangeStructureReconstruction::applyReconstruction()"););
    return;
} // LagrangeStructureReconstructions

LagrangeStructureReconstructions::~LagrangeStructureReconstructions()
{
    deallocateOperatorState();
    return;
} // ~LagrangeStructureReconstructions

void
LagrangeStructureReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
    ADS_TIMER_START(t_apply_reconstruction);
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    // TODO: What kind of physical boundary conditions should we use for advection?
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    ghost_cell_comps[0] =
        ITC(d_Q_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, d_bc_coef);
    ghost_cell_comps[1] = ITC(d_cur_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    HierarchyGhostCellInterpolation hier_ghost_cells;
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(d_current_time);
    applyReconstructionLS(d_Q_scr_idx, N_idx, path_idx);
    ADS_TIMER_STOP(t_apply_reconstruction);
}

void
LagrangeStructureReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                        double current_time,
                                                        double new_time)
{
    AdvectiveReconstructionOperator::allocateOperatorState(hierarchy, current_time, new_time);
    d_hierarchy = hierarchy;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_Q_scr_idx)) level->allocatePatchData(d_Q_scr_idx);
    }
    d_is_allocated = true;
}

void
LagrangeStructureReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();
    if (!d_is_allocated) return;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }
    d_is_allocated = false;
}

void
LagrangeStructureReconstructions::setCutCellMapping(Pointer<CutCellVolumeMeshMapping> cut_cell_mapping)
{
    d_cut_cell_mapping = std::move(cut_cell_mapping);
}

void
LagrangeStructureReconstructions::setInsideQSystemName(std::string Q_sys_name)
{
    d_Q_in_sys_name = std::move(Q_sys_name);
}

void
LagrangeStructureReconstructions::setOutsideQSystemName(std::string Q_sys_name)
{
    d_Q_out_sys_name = std::move(Q_sys_name);
}

void
LagrangeStructureReconstructions::setReconstructionOutside(const bool reconstruct_outside)
{
    d_reconstruct_outside = reconstruct_outside;
}

void
LagrangeStructureReconstructions::applyReconstructionLS(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_ls_idx > 0);
    TBOX_ASSERT(d_new_ls_idx > 0);
#endif

    // Grab the position and information
    const std::vector<std::shared_ptr<FEMeshPartitioner>>& mesh_partitioners =
        d_cut_cell_mapping->getMeshPartitioners();
    const int num_parts = d_cut_cell_mapping->getNumParts();
    std::vector<NumericVector<double>*> X_vecs(num_parts, nullptr), Q_in_vecs(num_parts, nullptr),
        Q_out_vecs(num_parts, nullptr);
    std::vector<DofMap*> X_dof_map_vecs(num_parts, nullptr), Q_in_dof_map_vecs(num_parts, nullptr),
        Q_out_dof_map_vecs(num_parts, nullptr);
    for (int part = 0; part < num_parts; ++part)
    {
        EquationSystems* eq_sys = mesh_partitioners[part]->getEquationSystems();

        auto& X_sys = eq_sys->get_system<ExplicitSystem>(mesh_partitioners[part]->COORDINATES_SYSTEM_NAME);
        X_vecs[part] = X_sys.current_local_solution.get();
        X_dof_map_vecs[part] = &X_sys.get_dof_map();

        auto& Q_in_sys = eq_sys->get_system<ExplicitSystem>(d_Q_in_sys_name);
        Q_in_vecs[part] = Q_in_sys.current_local_solution.get();
        Q_in_dof_map_vecs[part] = &Q_in_sys.get_dof_map();

        if (d_reconstruct_outside)
        {
            auto& Q_out_sys = eq_sys->get_system<ExplicitSystem>(d_Q_out_sys_name);
            Q_out_vecs[part] = Q_out_sys.current_local_solution.get();
            Q_out_dof_map_vecs[part] = &Q_out_sys.get_dof_map();
        }
    }

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            const hier::Index<NDIM>& idx_low = box.lower();

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);
            Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(d_new_ls_idx);

            Q_new_data->fillAll(0.0);

            auto within_lagrange_interpolant = [](const CellIndex<NDIM>& idx, NodeData<NDIM, double>& ls_data) -> bool
            {
                Box<NDIM> box(idx, idx);
                const double ls_val = ADS::node_to_cell(idx, ls_data);
                box.grow(1);
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& cc = ci();
                    if (ADS::node_to_cell(cc, ls_data) * ls_val < 0.0) return false;
                }
                return true;
            };

            // Grab the cut cell and element mappings
            const std::map<IndexList, std::vector<CutCellElems>>& cut_cell_map =
                d_cut_cell_mapping->getIdxCutCellElemsMap(ln)[patch_num];

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                IBTK::VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);

                const double ls_val = ADS::node_to_cell(idx, *ls_new_data);

                // If we are in the bulk, we can use a regular polynomial interpolant.
                if (d_reconstruct_outside || ls_val < 0.0)
                {
                    if (within_lagrange_interpolant(idx, *ls_data))
                    {
                        (*Q_new_data)(idx) = Reconstruct::quadraticLagrangeInterpolantLimited(x_loc, idx, *Q_cur_data);
                    }
                    else if (cut_cell_map.count(IndexList(patch, idx)) > 0 &&
                             (ls_val < 0.0 || !d_Q_out_sys_name.empty()))
                    {
                        // Our reconstruction can use boundary data.
                        // Need to determine closest points. If we are on a cut cell, we use the parent element's nodes
                        // in the stencil
                        // List of points and values
                        std::vector<VectorNd> X_pts;
                        std::vector<double> Q_vals;

                        const std::vector<CutCellElems>& cut_cell_elems = cut_cell_map.at(IndexList(patch, idx));
                        for (const auto& cut_cell_elem : cut_cell_elems)
                        {
                            const Elem* elem = cut_cell_elem.d_parent_elem;
                            const int part = cut_cell_elem.d_part;

                            // Grab the position of the nodes
                            for (unsigned int node_num = 0; node_num < elem->n_nodes(); ++node_num)
                            {
                                const Node* node = elem->node_ptr(node_num);
                                VectorNd X_pt;
                                std::vector<dof_id_type> dofs;
                                X_dof_map_vecs[part]->dof_indices(node, dofs);
                                for (int d = 0; d < NDIM; ++d) X_pt[d] = (*X_vecs[part])(dofs[d]);
                                X_pts.push_back(X_pt);
                                if (ls_val < 0.0)
                                {
                                    Q_in_dof_map_vecs[part]->dof_indices(node, dofs);
                                    Q_vals.push_back((*Q_in_vecs[part])(dofs[0]));
                                }
                                else
                                {
                                    Q_out_dof_map_vecs[part]->dof_indices(node, dofs);
                                    Q_vals.push_back((*Q_out_vecs[part])(dofs[0]));
                                }
                            }
                        }

                        // We have the points on the structure that we are using to reconstruct the function. Grab the
                        // rest from the Cartesian grid.
                        for (int d = 0; d < NDIM; ++d)
                            x_loc[d] = xlow[d] + dx[d] * (x_loc[d] - static_cast<double>(idx_low(d)));

                        // Flood fill for Eulerian points
                        std::vector<CellIndex<NDIM>> idx_vec;
                        try
                        {
                            Reconstruct::floodFillForPoints(
                                idx_vec, idx, *ls_data, ls_val, d_rbf_stencil_size - X_pts.size());
                        }
                        catch (const std::runtime_error& e)
                        {
                            pout << e.what() << "\n";
                            TBOX_ERROR(d_object_name + "::applyReconstruction(): Could not perform reconstruction!\n");
                        }
                        for (const auto& idx : idx_vec)
                        {
                            Q_vals.push_back((*Q_cur_data)(idx));
                            VectorNd x_cent_c;
                            for (int d = 0; d < NDIM; ++d)
                                x_cent_c[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
                            X_pts.push_back(x_cent_c);
                        }

                        // Now reconstruct the function
                        (*Q_new_data)(idx) =
                            Reconstruct::radialBasisFunctionReconstruction(x_loc, X_pts, Q_vals, d_rbf_order);
                    }
                    else
                    {
                        // A node doesn't touch this cell. Just use normal interpolation.
                        (*Q_new_data)(idx) = Reconstruct::radialBasisFunctionReconstruction(
                            x_loc, ls_val, idx, *Q_cur_data, *ls_data, patch, d_rbf_order, d_rbf_stencil_size);
                    }
                }
                else
                {
                    (*Q_new_data)(idx) = d_default_value;
                }

                // Cutoff solution if applicable
                (*Q_new_data)(idx) = std::min(std::max((*Q_new_data)(idx), d_low_cutoff), d_high_cutoff);
            }
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
