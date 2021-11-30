/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/LinearReconstructions.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"

#include "SAMRAIVectorReal.h"

#include <utility>

namespace ADS
{
LinearReconstructions::LinearReconstructions(std::string object_name)
    : AdvectiveReconstructionOperator(std::move(object_name)),
      d_Q_scr_var(new CellVariable<NDIM, double>(d_object_name + "::Q_scratch"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_scr_var, var_db->getContext(d_object_name + "::CTX"), 4);
    return;
} // LinearReconstructions

LinearReconstructions::~LinearReconstructions()
{
    deallocateOperatorState();
    return;
} // ~LinearReconstructions

void
LinearReconstructions::applyReconstruction(const int Q_idx, const int N_idx, const int path_idx)
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
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();

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
                if ((*vol_new_data)(idx) > 0.0)
                {
                    IBTK::VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = (*xstar_data)(idx, d);
                    // Find Node index closest to point. Note that this location corresponds to the Cell index that is
                    // the "lower left" of the box centered at nodal index.
                    CellIndex<NDIM> idx_ll;
                    VectorNd x_ll;
                    for (int d = 0; d < NDIM; ++d) x_ll[d] = std::round(x_loc[d]) - 0.5;
                    for (int d = 0; d < NDIM; ++d) idx_ll(d) = static_cast<int>(x_ll[d]);
                    // Check if we can use bi-linear interpolation, i.e. check if all neighboring cells are "full" cells
                    bool use_bilinear = true;
                    for (int x = -1; x <= 2; ++x)
                        for (int y = -1; y <= 2; ++y)
                            use_bilinear = use_bilinear && (*vol_cur_data)(idx_ll + IntVector<NDIM>(x, y)) == 1.0;
                    std::vector<double> temp_dx = { 1.0, 1.0 };
                    if (use_bilinear)
                        (*Q_new_data)(idx) =
                            Reconstruct::bilinearReconstruction(x_loc, x_ll, idx_ll, *Q_cur_data, temp_dx.data());
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
LinearReconstructions::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> hierarchy,
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
LinearReconstructions::deallocateOperatorState()
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
