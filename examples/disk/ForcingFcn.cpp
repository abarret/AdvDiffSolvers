// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2019 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <IBAMR_config.h>

#include "ForcingFcn.h"

#include <SAMRAI_config.h>

#include <array>

namespace LS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

ForcingFcn::ForcingFcn(const string& object_name, Pointer<Database> input_db) : LSCartGridFunction(object_name)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
#endif

    // Initialize object with data read from the input database.
    d_D = input_db->getDouble("d");
    return;
} // ForcingFcn

ForcingFcn::~ForcingFcn()
{
    // intentionally blank
    return;
} // ~ForcingFcn

void
ForcingFcn::setDataOnPatch(const int data_idx,
                           Pointer<Variable<NDIM>> /*var*/,
                           Pointer<Patch<NDIM>> patch,
                           const double /*data_time*/,
                           const bool initial_time,
                           Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> F_data = patch->getPatchData(data_idx);
    F_data->fillAll(0.0);
    if (initial_time) return;
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(F_data && vol_data);
#endif

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        if ((*vol_data)(idx) > 0.0) (*F_data)(idx) = -2.0;
    }
    return;
} // setDataOnPatch

} // namespace LS
