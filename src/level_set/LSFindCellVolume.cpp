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
} // namespace ADS
