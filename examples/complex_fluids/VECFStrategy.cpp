#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include "ibtk/ibtk_utilities.h"

#include "CellData.h"
#include "CellIterator.h"
#include "Patch.h"
#include "tbox/Database.h"

// Local includes
#include "VECFStrategy.h"

// Namespace
namespace ADS
{
VECFStrategy::VECFStrategy(const std::string& object_name,
                           Pointer<INSStaggeredHierarchyIntegrator> ins_integrator,
                           Pointer<CellVariable<NDIM, double>> zb_var,
                           Pointer<AdvDiffHierarchyIntegrator> zb_integrator,
                           const double C8,
                           const double beta)
    : CFStrategy(object_name),
      d_object_name(object_name),
      d_zb_var(zb_var),
      d_zb_integrator(zb_integrator),
      d_EE_var(new CellVariable<NDIM, double>(d_object_name + "::E_var", NDIM * (NDIM + 1) / 2)),
      d_ins_integrator(ins_integrator),
      d_C8(C8),
      d_beta(beta)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_EE_idx = var_db->registerVariableAndContext(d_EE_var, var_db->getContext("CTX"), 0);
    return;
} // Constructor

void
VECFStrategy::computeStress(int stress_idx,
                            Pointer<CellVariable<NDIM, double>> /*stress_var*/,
                            Pointer<PatchHierarchy<NDIM>> hierarchy,
                            double data_time)
{
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int zb_cur_idx = var_db->mapVariableAndContextToIndex(d_zb_var, d_zb_integrator->getCurrentContext());
    const int zb_scr_idx = var_db->mapVariableAndContextToIndex(d_zb_var, d_zb_integrator->getScratchContext());
    bool deallocate_after = !d_zb_integrator->isAllocatedPatchData(zb_scr_idx, coarsest_ln, finest_ln);
    if (deallocate_after) d_zb_integrator->allocatePatchData(zb_scr_idx, data_time, coarsest_ln, finest_ln);

    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp{ ITC(zb_scr_idx,
                                          zb_cur_idx,
                                          "CONSERVATIVE_LINEAR_REFINE",
                                          true,
                                          "NONE",
                                          "LINEAR",
                                          false,
                                          d_zb_integrator->getPhysicalBcCoefs(d_zb_var)) };
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(ghost_cell_comp, hierarchy, coarsest_ln, finest_ln);
    hier_ghost_fill.fillData(data_time);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CellData<NDIM, double>> stress_data = patch->getPatchData(stress_idx);
            Pointer<CellData<NDIM, double>> bond_data = patch->getPatchData(zb_scr_idx);

            // Note that stress_data has three ghost cells, but we really only need to fill one.
            for (CellIterator<NDIM> ci(bond_data->getGhostBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const double zb = (*bond_data)(idx);
                (*stress_data)(idx, 0) = (*stress_data)(idx, 0) + d_C8 * zb;
                (*stress_data)(idx, 1) = (*stress_data)(idx, 1) + d_C8 * zb;
            }
        }
    }

    if (deallocate_after) d_zb_integrator->deallocatePatchData(zb_scr_idx, coarsest_ln, finest_ln);
}

void
VECFStrategy::computeRelaxation(const int R_idx,
                                Pointer<CellVariable<NDIM, double>> /*R_var*/,
                                int C_idx,
                                Pointer<CellVariable<NDIM, double>> /*C_var*/,
                                TensorEvolutionType evolve_type,
                                Pointer<PatchHierarchy<NDIM>> hierarchy,
                                double data_time)
{
#ifndef NDEBUG
    // This only works if we evolve the stress tensor. TODO: Figure out how to evolve the square root or logarithm (need
    // an SPD version of the stress, e.g. conformation tensor).
    TBOX_ASSERT(evolve_type == TensorEvolutionType::STANDARD);
#endif
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int zb_idx = var_db->mapVariableAndContextToIndex(d_zb_var, d_zb_integrator->getCurrentContext());

    Pointer<SideVariable<NDIM, double>> u_var = d_ins_integrator->getVelocityVariable();
    const int u_cur_idx = var_db->mapVariableAndContextToIndex(u_var, d_ins_integrator->getCurrentContext());
    const int u_scr_idx = var_db->mapVariableAndContextToIndex(u_var, d_ins_integrator->getScratchContext());

    bool deallocate_after = !d_ins_integrator->isAllocatedPatchData(u_scr_idx, coarsest_ln, finest_ln);
    if (deallocate_after) d_ins_integrator->allocatePatchData(u_scr_idx, data_time, coarsest_ln, finest_ln);

    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp{ ITC(u_scr_idx,
                                          u_cur_idx,
                                          "CONSERVATIVE_LINEAR_REFINE",
                                          true,
                                          "NONE",
                                          "LINEAR",
                                          false,
                                          d_ins_integrator->getVelocityBoundaryConditions()) };
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(ghost_cell_comp, hierarchy, coarsest_ln, finest_ln);
    hier_ghost_fill.fillData(data_time);

    allocate_patch_data(d_EE_idx, hierarchy, data_time, coarsest_ln, finest_ln);
    HierarchyMathOps hier_math_ops(d_object_name + "::HierMathOps", hierarchy, coarsest_ln, finest_ln);
    hier_math_ops.strain_rate(d_EE_idx, d_EE_var, u_scr_idx, u_var, nullptr, data_time);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CellData<NDIM, double>> R_data = patch->getPatchData(R_idx);
            Pointer<CellData<NDIM, double>> C_data = patch->getPatchData(C_idx);
            Pointer<CellData<NDIM, double>> zb_data = patch->getPatchData(zb_idx);
            Pointer<CellData<NDIM, double>> E_data = patch->getPatchData(d_EE_idx);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const double zb = (*zb_data)(idx);
                (*R_data)(idx, 0) = d_C8 * zb * (*E_data)(idx, 0) - d_beta * ((*C_data)(idx, 0) + d_C8 * zb);
                (*R_data)(idx, 1) = d_C8 * zb * (*E_data)(idx, 1) - d_beta * ((*C_data)(idx, 1) + d_C8 * zb);
                (*R_data)(idx, 2) = d_C8 * zb * (*E_data)(idx, 2) - d_beta * ((*C_data)(idx, 2));
            }
        }
    }

    if (deallocate_after) d_ins_integrator->deallocatePatchData(u_scr_idx, coarsest_ln, finest_ln);
    deallocate_patch_data(d_EE_idx, hierarchy, coarsest_ln, finest_ln);
} // setDataOnPatch

} // namespace ADS
