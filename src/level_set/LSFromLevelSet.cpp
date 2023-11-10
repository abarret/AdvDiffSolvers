#include "ibtk/config.h"

#include "ADS/LSFromLevelSet.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

IBTK_DISABLE_EXTRA_WARNINGS
#include <boost/multi_array.hpp>
IBTK_ENABLE_EXTRA_WARNINGS

namespace ADS
{
LSFromLevelSet::LSFromLevelSet(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy)
    : LSFindCellVolume(std::move(object_name), hierarchy)
{
    // intentionally blank
    return;
} // Constructor

void
LSFromLevelSet::registerLSFcn(Pointer<CartGridFunction> ls_fcn)
{
    d_ls_fcn = ls_fcn;
}

void
LSFromLevelSet::doUpdateVolumeAreaSideLS(int vol_idx,
                                         Pointer<CellVariable<NDIM, double>> /*vol_var*/,
                                         int area_idx,
                                         Pointer<CellVariable<NDIM, double>> /*area_var*/,
                                         int side_idx,
                                         Pointer<SideVariable<NDIM, double>> /*side_var*/,
                                         int phi_idx,
                                         Pointer<Variable<NDIM>> phi_var,
                                         double data_time,
                                         bool extended_box)
{
    int coarsest_ln = 0, finest_ln = d_hierarchy->getFinestLevelNumber();

    TBOX_ASSERT(phi_var);
    TBOX_ASSERT(phi_idx != invalid_index);

    if (d_set_ls)
    {
        pout << "Setting Level set at time " << data_time << "\n";
        d_ls_fcn->setDataOnPatchHierarchy(phi_idx, phi_var, d_hierarchy, data_time);
    }
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp(1);
    ghost_cell_comp[0] = ITC(phi_idx, "LINEAR_REFINE", false, "CONSTANT_COARSEN", "LINEAR");
    HierarchyGhostCellInterpolation ghost_cells;
    ghost_cells.initializeOperatorState(ghost_cell_comp, d_hierarchy, coarsest_ln, finest_ln);
    ghost_cells.fillData(data_time);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        double tot_area = 0.0;
        double tot_vol = 0.0;
        double min_vol = 1.0;
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            Pointer<CellData<NDIM, double>> area_data;
            if (area_idx != IBTK::invalid_index) area_data = patch->getPatchData(area_idx);
            Pointer<CellData<NDIM, double>> vol_data;
            if (vol_idx != IBTK::invalid_index) vol_data = patch->getPatchData(vol_idx);
            Pointer<SideData<NDIM, double>> side_data;
            if (side_idx != IBTK::invalid_index) side_data = patch->getPatchData(side_idx);

            // Skip this code if we aren't computing area, volume, or cell side lengths
            if (area_idx == IBTK::invalid_index && vol_idx == IBTK::invalid_index && side_idx == IBTK::invalid_index)
                continue;

            // This code only works if phi is node centered. Since we are evaluating a prescribed level set function, we
            // should be able to create a scratch node centered level set for volume, area, and side computations.
            TBOX_ASSERT(phi_data);

            const Box<NDIM>& box = extended_box ? phi_data->getGhostBox() : patch->getBox();
            const hier::Index<NDIM>& patch_lower = box.lower();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const double cell_volume = dx[0] * dx[1];
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - patch_lower(d));
                std::pair<double, double> vol_area_pair = find_volume_and_area(x, dx, phi_data, idx);
                double volume = vol_area_pair.first;
                double area = vol_area_pair.second;
                if (area_idx != IBTK::invalid_index)
                {
                    (*area_data)(idx) = area;
                    if (patch->getBox().contains(idx)) tot_area += area;
                }
                if (vol_idx != IBTK::invalid_index)
                {
                    (*vol_data)(idx) = volume / cell_volume;
                    if (patch->getBox().contains(idx)) tot_vol += volume;
                    min_vol = volume > 0.0 ? std::min(min_vol, volume) : min_vol;
                }

                if (side_idx != IBTK::invalid_index)
                {
                    for (int f = 0; f < 2; ++f)
                    {
#if (NDIM == 2)
                        double L = length_fraction(1.0,
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 0))),
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 1))));
#endif
#if (NDIM == 3)
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
#endif
                        (*side_data)(SideIndex<NDIM>(idx, 0, f)) = L;
                    }
                    for (int f = 0; f < 2; ++f)
                    {
#if (NDIM == 2)
                        double L = length_fraction(1.0,
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, f))),
                                                   (*phi_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, f))));
#endif
#if (NDIM == 3)
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
#endif
                        (*side_data)(SideIndex<NDIM>(idx, 1, f)) = L;
                    }
#if (NDIM == 3)
                    for (int f = 0; f < 2; ++f)
                    {
                        double L = 0.0;
                        TBOX_ERROR("3D Not implemented yet.\n");
                        (*side_data)(SideIndex<NDIM>(idx, 2, f)) = L;
                    }
#endif
                }
            }
        }
        tot_area = SAMRAI_MPI::sumReduction(tot_area);
        tot_vol = SAMRAI_MPI::sumReduction(tot_vol);
        min_vol = SAMRAI_MPI::minReduction(min_vol);
        plog << "Minimum volume on level:     " << ln << " is: " << std::setprecision(12) << min_vol << "\n";
        plog << "Total area found on level:   " << ln << " is: " << std::setprecision(12) << tot_area << "\n";
        plog << "Total volume found on level: " << ln << " is: " << std::setprecision(12) << tot_vol << "\n";
    }
}
} // namespace ADS
