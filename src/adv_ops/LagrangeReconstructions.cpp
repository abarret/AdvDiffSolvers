/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/LagrangeReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include <ibtk/IndexUtilities.h>

#include "SAMRAIVectorReal.h"

#include <utility>

namespace ADS
{
LagrangeReconstructions::LagrangeReconstructions(std::string object_name)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), 4);
    return;
} // LagrangeReconstructions

LagrangeReconstructions::~LagrangeReconstructions()
{
    deallocateOperatorState();
    return;
} // ~LagrangeReconstructions

void
LagrangeReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
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
        ITC(d_Q_scr_idx, Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "CONSTANT", true, d_bc_coef);
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
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> xstar_data = patch->getPatchData(path_idx);
            Pointer<CellData<NDIM, double>> Q_cur_data = patch->getPatchData(d_Q_scr_idx);
            Pointer<CellData<NDIM, double>> vol_cur_data = patch->getPatchData(d_cur_vol_idx);
            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(N_idx);
            Pointer<CellData<NDIM, double>> vol_new_data = patch->getPatchData(d_new_vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_cur_ls_idx);

            Q_new_data->fillAll(0.0);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x_loc;
                for (int d = 0; d < NDIM; ++d) x_loc[d] = (*xstar_data)(idx, d);
                (*Q_new_data)(idx) = Reconstruct::quadratic_lagrange_interpolant(x_loc, idx, *Q_cur_data);
            }
        }
    }
}

void
LagrangeReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
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
LagrangeReconstructions::deallocateOperatorState()
{
    AdvectiveReconstructionOperator::deallocateOperatorState();

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }
}
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
