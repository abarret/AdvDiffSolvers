/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/AdvectiveReconstructionOperator.h"
#include "CCAD/app_namespaces.h" // IWYU pragma: keep

#include "SAMRAIVectorReal.h"

#include <utility>

namespace CCAD
{
AdvectiveReconstructionOperator::AdvectiveReconstructionOperator(std::string object_name)
    : d_object_name(std::move(object_name))
{
    // intentionally blank
    return;
} // AdvectiveReconstructionOperator

AdvectiveReconstructionOperator::~AdvectiveReconstructionOperator()
{
    deallocateOperatorState();
    return;
} // ~AdvectiveReconstructionOperator

void
AdvectiveReconstructionOperator::setCurrentLSData(const int ls_idx, const int vol_idx)
{
    d_cur_ls_idx = ls_idx;
    d_cur_vol_idx = vol_idx;
    return;
}

void
AdvectiveReconstructionOperator::setNewLSData(const int ls_idx, const int vol_idx)
{
    d_new_ls_idx = ls_idx;
    d_new_vol_idx = vol_idx;
    return;
}

void
AdvectiveReconstructionOperator::setLSData(const int ls_cur_idx,
                                           const int vol_cur_idx,
                                           const int ls_new_idx,
                                           const int vol_new_idx)
{
    setCurrentLSData(ls_cur_idx, vol_cur_idx);
    setNewLSData(ls_new_idx, vol_new_idx);
    return;
}

void
AdvectiveReconstructionOperator::allocateOperatorState(Pointer<PatchHierarchy<NDIM>> /*hierarchy*/,
                                                       double current_time,
                                                       double new_time)
{
    if (d_is_allocated) deallocateOperatorState();
    d_current_time = current_time;
    d_new_time = new_time;
}

void
AdvectiveReconstructionOperator::deallocateOperatorState()
{
    d_is_allocated = false;
}
} // namespace CCAD

//////////////////////////////////////////////////////////////////////////////
