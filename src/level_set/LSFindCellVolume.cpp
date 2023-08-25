#include "ibamr/config.h"

#include "ADS/LSFindCellVolume.h"
#include "ADS/app_namespaces.h"

namespace ADS
{
LSFindCellVolume::LSFindCellVolume(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy)
    : d_object_name(std::move(object_name)), d_hierarchy(hierarchy)
{
    // intentionally blank
    return;
} // Constructor

void
LSFindCellVolume::updateVolumeAreaSideLS(const int vol_idx,
                                         Pointer<CellVariable<NDIM, double>> vol_var,
                                         const int area_idx,
                                         Pointer<CellVariable<NDIM, double>> area_var,
                                         const int side_idx,
                                         Pointer<SideVariable<NDIM, double>> side_var,
                                         const int phi_idx,
                                         Pointer<Variable<NDIM>> phi_var,
                                         const double data_time,
                                         const bool extended_box)
{
    doUpdateVolumeAreaSideLS(
        vol_idx, vol_var, area_idx, area_var, side_idx, side_var, phi_idx, phi_var, data_time, extended_box);
}
} // namespace ADS
