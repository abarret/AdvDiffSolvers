#include "ibamr/config.h"

#include "CCAD/LSFindCellVolume.h"

#include "ibtk/DebuggingUtilities.h"
#include "ibtk/app_namespaces.h"

namespace CCAD
{
LSFindCellVolume::LSFindCellVolume(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy)
    : d_object_name(std::move(object_name)), d_hierarchy(hierarchy)
{
    // intentionally blank
    return;
} // Constructor
} // namespace CCAD
