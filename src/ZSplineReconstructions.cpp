/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/ZSplineReconstructions.h"
#include "CCAD/ls_functions.h"

#include "ibamr/namespaces.h" // IWYU pragma: keep

#include "ibtk/HierarchyGhostCellInterpolation.h"

#include "SAMRAIVectorReal.h"

#include <utility>

namespace CCAD
{
ZSplineReconstructions::ZSplineReconstructions(std::string object_name, int z_spline_order)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_order(z_spline_order),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), 4);
    return;
} // ZSplineReconstructions

ZSplineReconstructions::~ZSplineReconstructions()
{
    deallocateOperatorState();
    return;
} // ~ZSplineReconstructions

void
ZSplineReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
{
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
#ifndef NDEBUG
    TBOX_ASSERT(d_cur_vol_idx > 0);
    TBOX_ASSERT(d_new_vol_idx > 0);
#endif

    // TODO: What kind of physical boundary conditions should we use for advection?
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comps(2);
    ghost_cell_comps[0] =
        ITC(d_Q_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, nullptr);
    ghost_cell_comps[1] = ITC(d_cur_ls_idx, "LINEAR_REFINE", false, "NONE", "LINEAR");
    HierarchyGhostCellInterpolation hier_ghost_cells;
    hier_ghost_cells.initializeOperatorState(ghost_cell_comps, d_hierarchy, coarsest_ln, finest_ln);
    hier_ghost_cells.fillData(d_current_time);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(d_Q_scr_idx);
            Pointer<CellData<NDIM, double>> vol_cur_data = patch->getPatchData(d_cur_vol_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<CellData<NDIM, double>> vol_new_data = patch->getPatchData(d_new_vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);

            Q_new_data->fillAll(0.0);

            const int stencil_width = Reconstruct::getSplineWidth(d_order);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_new_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    // Check if we can use z-spline
                    if (Reconstruct::indexWithinWidth(stencil_width, idx, *vol_cur_data))
                        (*Q_new_data)(idx) = Reconstruct::sumOverZSplines(x_loc, idx, *Q_cur_data, d_order);
                    else
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
ZSplineReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
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
}

void
ZSplineReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }
}
} // namespace CCAD

//////////////////////////////////////////////////////////////////////////////
