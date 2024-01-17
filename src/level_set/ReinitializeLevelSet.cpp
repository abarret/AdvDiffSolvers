#include <ADS/ReinitializeLevelSet.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/HierarchyMathOps.h>

// Fortran routines
extern "C"
{
#if (NDIM == 2)
    void fast_sweep_2d_(double* U,
                        const int& U_gcw,
                        const int& ilower0,
                        const int& iupper0,
                        const int& ilower1,
                        const int& iupper1,
                        const double* dx,
                        int* v,
                        const int& v_gcw);
#endif
}

namespace ADS
{
ReinitializeLevelSet::ReinitializeLevelSet(std::string object_name, Pointer<Database> input_db)
    : d_object_name(std::move(object_name))
{
    if (input_db)
    {
        d_tol = input_db->getDoubleWithDefault("tolerance", d_tol);
        d_max_iters = input_db->getIntegerWithDefault("max_iterations", d_max_iters);
        d_enable_logging = input_db->getBoolWithDefault("enable_logging", d_enable_logging);
    }

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    std::string var_name = d_object_name + "::fixed_vals";
    if (var_db->checkVariableExists(var_name))
        d_nc_var = var_db->getVariable(var_name);
    else
        d_nc_var = new NodeVariable<NDIM, int>(var_name);
    d_nc_idx = var_db->registerVariableAndContext(d_nc_var, var_db->getContext(d_object_name + "::CTX"));
}

ReinitializeLevelSet::~ReinitializeLevelSet()
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    var_db->removePatchDataIndex(d_nc_idx);
}

void
ReinitializeLevelSet::computeSignedDistanceFunction(const int phi_idx,
                                                    NodeVariable<NDIM, double>& phi_var,
                                                    Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                    const double time,
                                                    const double value_to_be_changed)
{
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    allocate_patch_data(d_nc_idx, hierarchy, time, coarsest_ln, finest_ln);

    // Now determine which cells should be fixed.
    auto fcn = [](Pointer<Patch<NDIM>> patch, const int phi_idx, const int nc_idx, const double value_to_be_changed)
    {
        Pointer<NodeData<NDIM, double>> ls_vals = patch->getPatchData(phi_idx);
        Pointer<NodeData<NDIM, int>> fixed_vals = patch->getPatchData(nc_idx);

        for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            if (std::abs((*ls_vals)(idx)) == value_to_be_changed)
                (*fixed_vals)(idx) = 0;
            else
                (*fixed_vals)(idx) = 1;
        }
    };
    perform_on_patch_hierarchy(hierarchy, fcn, phi_idx, d_nc_idx, value_to_be_changed);

    computeSignedDistanceFunction(phi_idx, phi_var, hierarchy, time, d_nc_idx);

    // Now deallocate data
    deallocate_patch_data(d_nc_idx, hierarchy, coarsest_ln, finest_ln);
}

void
ReinitializeLevelSet::computeSignedDistanceFunction(const int phi_idx,
                                                    NodeVariable<NDIM, double>& phi_var,
                                                    Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                    const double time,
                                                    const int fixed_idx)
{
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    // Create a cloned patch index for iteration count
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    Pointer<NodeVariable<NDIM, double>> phi_var_ptr(&phi_var, false);
    const int phi_old_idx = var_db->registerClonedPatchDataIndex(phi_var_ptr, phi_idx);
    // Create a ghost filling object.
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_comp{ ITC(phi_idx, "LINEAR_REFINE", false, "CONSTANT_COARSEN") };
    HierarchyGhostCellInterpolation ghost_fill;
    ghost_fill.initializeOperatorState(ghost_comp, hierarchy, coarsest_ln, finest_ln);

    HierarchyNodeDataOpsReal<NDIM, double> hier_nc_data_ops(hierarchy);

    // Allocate temporary data
    allocate_patch_data(phi_old_idx, hierarchy, time, coarsest_ln, finest_ln);

    double max_norm = std::numeric_limits<double>::max();
    int iter_num = 0;
    while (max_norm > d_tol && iter_num < d_max_iters)
    {
        // Prepare for a new iteration. Copy data into old index
        hier_nc_data_ops.copyData(phi_old_idx, phi_idx);

        // Fill ghost cells
        ghost_fill.fillData(time);

        // Now perform a sweep
        doSweep(hierarchy, phi_idx, fixed_idx);

        // Compute residual
        hier_nc_data_ops.subtract(phi_old_idx, phi_idx, phi_old_idx);
        max_norm = hier_nc_data_ops.maxNorm(phi_old_idx);
        if (d_enable_logging)
        {
            plog << "After iteration " << iter_num << "\n";
            plog << "  Residual: " << max_norm << "\n";
        }
        ++iter_num;
    }

    // Deallocate temporary data
    deallocate_patch_data(phi_old_idx, hierarchy, coarsest_ln, finest_ln);

    // Remove scratch index from database
    var_db->removePatchDataIndex(phi_old_idx);
    return;
}

void
ReinitializeLevelSet::doSweep(Pointer<PatchHierarchy<NDIM>> hierarchy, const int phi_idx, const int fixed_idx)
{
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);

        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            doSweepOnPatch(patch, phi_idx, fixed_idx);
        }
    }
    return;
}

void
ReinitializeLevelSet::doSweepOnPatch(Pointer<Patch<NDIM>> patch, const int phi_idx, const int fixed_idx)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();

    const hier::Index<NDIM>& idx_low = patch->getBox().lower();
    const hier::Index<NDIM>& idx_up = patch->getBox().upper();

    Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
    Pointer<NodeData<NDIM, int>> fixed_data = patch->getPatchData(fixed_idx);

    // Perform a single sweep.
#if (NDIM == 2)
    fast_sweep_2d_(phi_data->getPointer(),
                   phi_data->getGhostCellWidth().max(),
                   idx_low(0),
                   idx_up(0),
                   idx_low(1),
                   idx_up(1),
                   dx,
                   fixed_data->getPointer(),
                   fixed_data->getGhostCellWidth().max());
#endif
#if (NDIM == 3)
    NULL_USE(dx);
    NULL_USE(idx_low);
    NULL_USE(idx_up);
    TBOX_ERROR("3D version not implemented yet!\n");
#endif
}
} // namespace ADS
