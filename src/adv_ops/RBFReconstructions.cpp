/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/RBFReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"

#include "SAMRAIVectorReal.h"

#include <utility>

namespace ADS
{
RBFReconstructions::RBFReconstructions(std::string object_name,
                                       Reconstruct::RBFPolyOrder rbf_order,
                                       int stencil_size,
                                       bool use_cut_cells)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_rbf_order(rbf_order),
      d_rbf_stencil_size(stencil_size),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch")),
      d_use_cut_cells(use_cut_cells)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(
        d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), std::ceil(0.5 * d_rbf_stencil_size));
    return;
} // RBFReconstructions

RBFReconstructions::~RBFReconstructions()
{
    deallocateOperatorState();
    return;
} // ~RBFReconstructions

void
RBFReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
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

    if (d_use_cut_cells)
        applyReconstructionCutCell(d_Q_scr_idx, N_idx, path_idx);
    else
        applyReconstructionLS(d_Q_scr_idx, N_idx, path_idx);
}

void
RBFReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy, double current_time, double new_time)
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
RBFReconstructions::deallocateOperatorState()
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
RBFReconstructions::applyReconstructionCutCell(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_vol_idx > 0);
    TBOX_ASSERT(d_new_vol_idx > 0);
#endif

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> vol_cur_data = patch->getPatchData(d_cur_vol_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<CellData<NDIM, double>> vol_new_data = patch->getPatchData(d_new_vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);

            Q_new_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_new_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    (*Q_new_data)(idx) = Reconstruct::radialBasisFunctionReconstruction(
                        x_loc, idx, *Q_cur_data, *vol_cur_data, *ls_data, patch, d_rbf_order, d_rbf_stencil_size);
                }
                else
                {
                    (*Q_new_data)(idx) = 0.0;
                }
            }
        }
    }
}

void
RBFReconstructions::applyReconstructionLS(const int Q_idx, const int N_idx, const int path_idx)
{
    pout << "Reconstructing using level set only.\n";
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_ls_idx > 0);
    TBOX_ASSERT(d_new_ls_idx > 0);
#endif

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);
            Pointer<NodeData<NDIM, double>> ls_new_data = patch->getPatchData(d_new_ls_idx);

            Q_new_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if (ADS::node_to_cell(idx, *ls_new_data) < 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    (*Q_new_data)(idx) =
                        Reconstruct::radialBasisFunctionReconstruction(x_loc,
                                                                       ADS::node_to_cell(idx, *ls_new_data),
                                                                       idx,
                                                                       *Q_cur_data,
                                                                       *ls_data,
                                                                       patch,
                                                                       d_rbf_order,
                                                                       d_rbf_stencil_size);
                }
                else
                {
                    (*Q_new_data)(idx) = 0.0;
                }
            }
        }
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
