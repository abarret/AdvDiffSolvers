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

#include "SetLSValue.h"

#include "utility_functions.h"
/////////////////////////////// INCLUDES /////////////////////////////////////

#include <IBAMR_config.h>

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// STATIC ///////////////////////////////////////

/////////////////////////////// PUBLIC ///////////////////////////////////////

SetLSValue::SetLSValue(const string& object_name, Pointer<GridGeometry<NDIM>> grid_geom, Pointer<Database> input_db)
    : CartGridFunction(object_name), d_object_name(object_name), d_grid_geom(grid_geom)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
    TBOX_ASSERT(d_grid_geom);
#endif

    // Initialize object with data read from the input database.
    getFromInput(input_db);
    return;
} // SetLSValue

SetLSValue::~SetLSValue()
{
    // intentionally blank
    return;
} // ~SetLSValue

void
SetLSValue::setDataOnPatch(const int data_idx,
                           Pointer<hier::Variable<NDIM>> /*var*/,
                           Pointer<Patch<NDIM>> patch,
                           const double data_time,
                           const bool /*initial_time*/,
                           Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> ls_c_data = patch->getPatchData(data_idx);
    Pointer<NodeData<NDIM, double>> ls_n_data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(ls_c_data || ls_n_data);
#endif
    const Box<NDIM>& patch_box = d_extended_box ? ls_n_data->getGhostBox() : patch->getBox();
    const hier::Index<NDIM>& patch_lower = patch->getBox().lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const x_lower = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    VectorNd X;

    if (d_interface_type == "ANNULUS")
    {
        if (ls_c_data)
        {
            for (CellIterator<NDIM> ic(patch_box); ic; ic++)
            {
                const CellIndex<NDIM>& i = ic();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * (static_cast<double>(i(d) - patch_lower(d)) + 0.5);
                }
                X = X - d_center;
                double r = X.norm();
                if (r < d_R1)
                {
                    (*ls_c_data)(i) = d_R1 - r;
                }
                else if (r > d_R2)
                {
                    (*ls_c_data)(i) = r - d_R2;
                }
                else
                {
                    (*ls_c_data)(i) = std::max(d_R1 - r, r - d_R2);
                }
            }
        }
        else if (ls_n_data)
        {
            for (NodeIterator<NDIM> in (patch_box); in; in++)
            {
                const NodeIndex<NDIM>& i = in();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * static_cast<double>(i(d) - patch_lower(d));
                }
                X = X - d_center;
                double r = X.norm();
                if (r < d_R1)
                {
                    (*ls_n_data)(i) = d_R1 - r;
                }
                else if (r > d_R2)
                {
                    (*ls_n_data)(i) = r - d_R2;
                }
                else
                {
                    (*ls_n_data)(i) = std::max(d_R1 - r, r - d_R2);
                }
            }
        }
    }
    else if (d_interface_type == "DISK")
    {
        if (ls_c_data)
        {
            for (CellIterator<NDIM> ic(patch_box); ic; ic++)
            {
                const CellIndex<NDIM>& i = ic();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * (static_cast<double>(i(d) - patch_lower(d)) + 0.5) - data_time * d_U[d];
                }
                X = X - d_center;
                double r = X.norm();
                (*ls_c_data)(i) = r - d_R1;
            }
        }
        else if (ls_n_data)
        {
            for (NodeIterator<NDIM> ni(patch_box); ni; ni++)
            {
                const NodeIndex<NDIM>& i = ni();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * static_cast<double>(i(d) - patch_lower(d)) - data_time * d_U[d];
                }
                X = X - d_center;
                double r = X.norm();
                (*ls_n_data)(i) = r - d_R1;
            }
        }
    }
    else if (d_interface_type == "RADIAL")
    {
        if (ls_c_data)
        {
            for (CellIterator<NDIM> ic(patch_box); ic; ic++)
            {
                const CellIndex<NDIM>& i = ic();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * (static_cast<double>(i(d) - patch_lower(d)) + 0.5);
                }
                X = X - d_disk_center;
                double theta = std::atan2(X(1), X(0));
                MatrixNd Q;
                Q(0, 0) = Q(1, 1) = std::cos(theta - d_v * data_time);
                Q(0, 1) = std::sin(theta - d_v * data_time);
                Q(1, 0) = -Q(0, 1);
                X = Q.transpose() * X;
                double r = X.norm();
                (*ls_c_data)(i) = r - d_R1;
            }
        }
        else if (ls_n_data)
        {
            for (NodeIterator<NDIM> ni(patch_box); ni; ni++)
            {
                const NodeIndex<NDIM>& i = ni();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * static_cast<double>(i(d) - patch_lower(d));
                }
                double theta = std::atan2(static_cast<double>(X(1)), static_cast<double>(X(0)));
                MatrixNd Q;
                Q(0, 0) = Q(1, 1) = std::cos(d_v * data_time);
                Q(0, 1) = std::sin(d_v * data_time);
                Q(1, 0) = -Q(0, 1);
                X = Q * X - d_disk_center;
                double r = X.norm();
                (*ls_n_data)(i) = r - d_R1;
            }
        }
    }
    else if (d_interface_type == "CHANNEL")
    {
        MatrixNd Q;
        Q(0, 0) = cos(d_theta);
        Q(0, 1) = -sin(d_theta);
        Q(1, 0) = sin(d_theta);
        Q(1, 1) = cos(d_theta);
        if (ls_c_data)
        {
            for (CellIterator<NDIM> ic(patch_box); ic; ic++)
            {
                const CellIndex<NDIM>& i = ic();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * (static_cast<double>(i(d) - patch_lower(d)) + 0.5);
                }
                VectorNd X_rot = Q.transpose() * (X - d_channel_center);
                double distance_l = std::abs(-0.375 - X_rot(1));
                double distance_u = std::abs(0.375 - X_rot(1));
                if (X_rot(1) > -0.375 && X_rot(1) < 0.375)
                {
                    (*ls_c_data)(i) = -std::min(distance_l, distance_u);
                }
                else
                {
                    (*ls_c_data)(i) = std::min(distance_l, distance_u);
                }
            }
        }
        else if (ls_n_data)
        {
            for (NodeIterator<NDIM> in (patch_box); in; in++)
            {
                const NodeIndex<NDIM>& i = in();
                for (int d = 0; d < NDIM; ++d)
                {
                    X[d] = x_lower[d] + dx[d] * static_cast<double>(i(d) - patch_lower(d));
                }
                VectorNd X_rot = Q.transpose() * (X - d_channel_center);
                double distance_l = std::abs(-0.375 - X_rot(1));
                double distance_u = std::abs(0.375 - X_rot(1));
                if (X_rot(1) > -0.375 && X_rot(1) < 0.375)
                {
                    (*ls_n_data)(i) = -std::min(distance_l, distance_u);
                }
                else
                {
                    (*ls_n_data)(i) = std::min(distance_l, distance_u);
                }
            }
        }
    }
    else
    {
        TBOX_ERROR(d_object_name << "::initializeDataOnPatch()\n"
                                 << "  invalid initialization type " << d_interface_type << "\n");
    }
    return;
} // setDataOnPatch

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

void
SetLSValue::getFromInput(Pointer<Database> db)
{
    if (db)
    {
        d_interface_type = db->getStringWithDefault("interface_type", d_interface_type);

        if (d_interface_type == "ANNULUS")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            d_R2 = db->getDoubleWithDefault("R2", d_R2);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
        }
        else if (d_interface_type == "CHANNEL")
        {
            d_theta = db->getDoubleWithDefault("Theta", d_theta);
            d_y_p = db->getDoubleWithDefault("Y_p", d_y_p);
            d_y_n = db->getDoubleWithDefault("Y_n", d_y_n);
        }
        else if (d_interface_type == "DISK")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
            if (db->keyExists("Vel")) db->getDoubleArray("Vel", d_U.data(), NDIM);
        }
        else if (d_interface_type == "RADIAL")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_disk_center.data(), NDIM);
            if (db->keyExists("Rot_center")) db->getDoubleArray("Rot_center", d_rot_center.data(), NDIM);
            d_v = db->getDouble("v");
        }
        else
        {
            TBOX_ERROR(d_object_name << "::getFromInput()\n"
                                     << "  invalid initialization type " << d_interface_type << "\n");
        }
    }
    return;
} // getFromInput

//////////////////////////////////////////////////////////////////////////////
