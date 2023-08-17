#include "ibamr/config.h"

#include "ADS/app_namespaces.h"

#include "LSFcn.h"

namespace ADS
{

LSFcn::LSFcn(string object_name) : CartGridFunction(std::move(object_name))
{
    // intentionally blank
    return;
} // LSFcn

void
LSFcn::setDataOnPatch(const int data_idx,
                      Pointer<hier::Variable<NDIM>> /*var*/,
                      Pointer<Patch<NDIM>> patch,
                      const double data_time,
                      const bool /*initial_time*/,
                      Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<NodeData<NDIM, double>> ls_n_data = patch->getPatchData(data_idx);
    ls_n_data->fillAll(-1.0);
    return;
} // setDataOnPatch
} // namespace ADS
