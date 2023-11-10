/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/RBFStructureReconstructions.h"
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
RBFStructureReconstructions::RBFStructureReconstructions(std::string object_name, Pointer<Database> input_db)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    d_rbf_stencil_size = input_db->getInteger("stencil_size");
    d_rbf_order = Reconstruct::string_to_enum<Reconstruct::RBFPolyOrder>(input_db->getString("rbf_order"));
    d_low_cutoff = input_db->getDoubleWithDefault("low_cutoff", d_low_cutoff);
    d_high_cutoff = input_db->getDoubleWithDefault("high_cutoff", d_high_cutoff);

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(
        d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), std::ceil(0.5 * d_rbf_stencil_size));
    IBTK_DO_ONCE(t_apply_reconstruction =
                     TimerManager::getManager()->getTimer("ADS::RBFStructureReconstruction::applyReconstruction()"););
    return;
} // RBFStructureReconstructions

RBFStructureReconstructions::~RBFStructureReconstructions()
{
    deallocateOperatorState();
    return;
} // ~RBFStructureReconstructions

void
RBFStructureReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
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
RBFStructureReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
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
RBFStructureReconstructions::deallocateOperatorState()
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
RBFStructureReconstructions::setCutCellMapping(Pointer<CutCellVolumeMeshMapping> cut_cell_mapping)
{
    d_cut_cell_mapping = std::move(cut_cell_mapping);
}

void
RBFStructureReconstructions::setQSystemName(std::string Q_sys_name)
{
    d_Q_sys_name = std::move(Q_sys_name);
}

void
RBFStructureReconstructions::applyReconstructionLS(const int Q_idx, const int N_idx, const int path_idx)
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
    std::vector<NumericVector<double>*> X_vecs(num_parts, nullptr), Q_vecs(num_parts, nullptr);
    std::vector<DofMap*> X_dof_map_vecs(num_parts, nullptr), Q_dof_map_vecs(num_parts, nullptr);
    for (int part = 0; part < num_parts; ++part)
    {
        EquationSystems* eq_sys = mesh_partitioners[part]->getEquationSystems();

        auto& X_sys = eq_sys->get_system<ExplicitSystem>(mesh_partitioners[part]->COORDINATES_SYSTEM_NAME);
        X_vecs[part] = X_sys.current_local_solution.get();
        X_dof_map_vecs[part] = &X_sys.get_dof_map();

        auto& Q_sys = eq_sys->get_system<ExplicitSystem>(d_Q_sys_name);
        Q_vecs[part] = Q_sys.current_local_solution.get();
        Q_dof_map_vecs[part] = &Q_sys.get_dof_map();
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

            // Grab the cut cell and element mappings
            const std::map<IndexList, std::vector<CutCellElems>>& cut_cell_map =
                d_cut_cell_mapping->getIdxCutCellElemsMap(ln)[patch_num];

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if (ADS::node_to_cell(idx, *ls_new_data) < 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);

                    // Need to determine closest points. If we are on a cut cell, we use the parent element's nodes in
                    // the stencil
                    if (cut_cell_map.count(IndexList(patch, idx)) > 0)
                    {
                        const double ls_new_val = node_to_cell(idx, *ls_new_data);
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
                                Q_dof_map_vecs[part]->dof_indices(node, dofs);
                                Q_vals.push_back((*Q_vecs[part])(dofs[0]));
                            }
                        }

                        // We have the points on the structure that we are using to reconstruct the function. Grab the
                        // rest from the Cartesian grid.
                        for (int d = 0; d < NDIM; ++d)
                            x_loc[d] = xlow[d] + dx[d] * (x_loc[d] - static_cast<double>(idx_low(d)));

                        // Flood fill for Eulerian points
                        std::vector<CellIndex<NDIM>> new_idxs = { idx };
                        unsigned int i = 0;
                        while (X_pts.size() < d_rbf_stencil_size)
                        {
#ifndef NDEBUG
                            TBOX_ASSERT(i < new_idxs.size());
#endif
                            const CellIndex<NDIM>& new_idx = new_idxs[i];
                            // Add new idx to list of X_vals
                            if (ADS::node_to_cell(new_idx, *ls_data) * ls_new_val > 0.0)
                            {
                                Q_vals.push_back((*Q_cur_data)(new_idx));
                                VectorNd x_cent_c;
                                for (int d = 0; d < NDIM; ++d)
                                    x_cent_c[d] =
                                        xlow[d] + dx[d] * (static_cast<double>(new_idx(d) - idx_low(d)) + 0.5);
                                X_pts.push_back(x_cent_c);
                            }

                            // Add neighboring points to new_idxs.
                            IntVector<NDIM> l(-1, 0), r(1, 0), b(0, -1), u(0, 1);
                            CellIndex<NDIM> idx_l(new_idx + l), idx_r(new_idx + r);
                            CellIndex<NDIM> idx_u(new_idx + u), idx_b(new_idx + b);
                            if (ADS::node_to_cell(idx_l, *ls_data) * ls_new_val > 0.0 &&
                                (std::find(new_idxs.begin(), new_idxs.end(), idx_l) == new_idxs.end()))
                                new_idxs.push_back(idx_l);
                            if (ADS::node_to_cell(idx_r, *ls_data) * ls_new_val > 0.0 &&
                                (std::find(new_idxs.begin(), new_idxs.end(), idx_r) == new_idxs.end()))
                                new_idxs.push_back(idx_r);
                            if (ADS::node_to_cell(idx_u, *ls_data) * ls_new_val > 0.0 &&
                                (std::find(new_idxs.begin(), new_idxs.end(), idx_u) == new_idxs.end()))
                                new_idxs.push_back(idx_u);
                            if (ADS::node_to_cell(idx_b, *ls_data) * ls_new_val > 0.0 &&
                                (std::find(new_idxs.begin(), new_idxs.end(), idx_b) == new_idxs.end()))
                                new_idxs.push_back(idx_b);
                            ++i;
                        }

                        // Now reconstruct the function
                        (*Q_new_data)(idx) =
                            Reconstruct::radial_basis_function_reconstruction(x_loc, X_pts, Q_vals, d_rbf_order);
                    }
                    else
                    {
                        // A node doesn't touch this cell. Just use normal interpolation.
                        (*Q_new_data)(idx) =
                            Reconstruct::radial_basis_function_reconstruction(x_loc,
                                                                              ADS::node_to_cell(idx, *ls_new_data),
                                                                              idx,
                                                                              *Q_cur_data,
                                                                              *ls_data,
                                                                              patch,
                                                                              d_rbf_order,
                                                                              d_rbf_stencil_size);
                    }
                }
                else
                {
                    (*Q_new_data)(idx) = 0.0;
                }

                // Cutoff solution if applicable
                (*Q_new_data)(idx) = std::min(std::max((*Q_new_data)(idx), d_low_cutoff), d_high_cutoff);
            }
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
