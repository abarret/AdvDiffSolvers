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

#include "LS/utility_functions.h"

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
    d_k_on = input_db->getDouble("k_on");
    d_k_off = input_db->getDouble("k_off");
    input_db->getDoubleArray("center", d_cent.data(), NDIM);
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
                           const double time,
                           const bool initial_time,
                           Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> F_data = patch->getPatchData(data_idx);
    F_data->fillAll(0.0);
    if (initial_time) return;
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
    Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(F_data && vol_data && ls_data);
#endif
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        if ((*vol_data)(idx) > 0.0)
        {
            double denom = -2.0 * d_D - d_k_on * (1.0 + (time - 1.0) * time);
            VectorNd X = LS::find_cell_centroid(idx, *ls_data);
            for (int d = 0; d < NDIM; ++d) X[d] = xlow[d] + dx[d] * X[d];
            X -= d_cent;
            const double r2 = X.squaredNorm();
            (*F_data)(idx) = r2 * d_k_on * (2.0 * time - 1.0) *
                                 (d_k_off * time * (time - 1.0) + d_k_on * (1.0 + (time - 1.0) * time)) /
                                 (denom * denom) +
                             r2 * (d_k_off * (time - 1.0) + d_k_off * time + d_k_on * (2.0 * time - 1.0)) / denom -
                             4.0 * d_D * (d_k_off * (time - 1.0) * time + d_k_on * (1.0 + (time - 1.0) * time)) / denom;
        }
    }
    return;
} // setDataOnPatch

} // namespace LS
