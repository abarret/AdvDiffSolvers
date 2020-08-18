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

#include "InsideLSFcn.h"

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// PUBLIC ///////////////////////////////////////

InsideLSFcn::InsideLSFcn(const string& object_name, Pointer<Database> input_db)
    : CartGridFunction(object_name), d_object_name(object_name)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
#endif
    d_y_low = input_db->getDouble("y_low");
    d_y_up = input_db->getDouble("y_up");
    d_L = input_db->getDouble("l");
    d_theta = input_db->getDouble("theta");
    return;
} // InsideLSFcn

void
InsideLSFcn::setDataOnPatch(const int data_idx,
                            Pointer<hier::Variable<NDIM>> /*var*/,
                            Pointer<Patch<NDIM>> patch,
                            const double data_time,
                            const bool /*initial_time*/,
                            Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> ls_c_data = patch->getPatchData(data_idx);
    Pointer<NodeData<NDIM, double>> ls_n_data = patch->getPatchData(data_idx);

    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const x_low = pgeom->getXLower();
    MatrixNd Q;
    Q(0, 0) = Q(1, 1) = std::cos(d_theta);
    Q(0, 1) = std::sin(d_theta);
    Q(1, 0) = -Q(0, 1);

    const Box<NDIM>& box = patch->getBox();
    const hier::Index<NDIM>& idx_low = box.lower();

    if (ls_c_data)
    {
        for (CellIterator<NDIM> ci(box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();

            VectorNd x_pt;
            for (int d = 0; d < NDIM; ++d)
                x_pt(d) = x_low[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
            x_pt = Q * x_pt;
            (*ls_c_data)(idx) = std::max(x_pt(1) - d_y_up, d_y_low - x_pt(1));
        }
    }
    else if (ls_n_data)
    {
        for (NodeIterator<NDIM> ci(box); ci; ci++)
        {
            const NodeIndex<NDIM>& idx = ci();

            VectorNd x_pt;
            for (int d = 0; d < NDIM; ++d) x_pt(d) = x_low[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));
            x_pt = Q * x_pt;
            (*ls_n_data)(idx) = std::max(x_pt(1) - d_y_up, d_y_low - x_pt(1));
        }
    }
    else
    {
        TBOX_ERROR("Should not get here.");
    }
    return;
} // setDataOnPatch

//////////////////////////////////////////////////////////////////////////////
