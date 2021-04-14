#include "IBAMR_config.h"

#include "ibtk/DebuggingUtilities.h"
#include "ibtk/app_namespaces.h"

#include "LS/LSFindCellVolume.h"

namespace LS
{
LSFindCellVolume::LSFindCellVolume(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy)
    : d_object_name(std::move(object_name)), d_hierarchy(hierarchy)
{
    // intentionally blank
    return;
} // Constructor
} // namespace LS
