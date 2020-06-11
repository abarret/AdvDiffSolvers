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

#include "QInitial.h"

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <IBAMR_config.h>

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// STATIC ///////////////////////////////////////

/////////////////////////////// PUBLIC ///////////////////////////////////////

QInitial::QInitial(const string& object_name, Pointer<GridGeometry<NDIM>> grid_geom, Pointer<Database> input_db)
    : CartGridFunction(object_name), d_grid_geom(grid_geom)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
    TBOX_ASSERT(d_grid_geom);
#endif

    // Initialize object with data read from the input database.
    getFromInput(input_db);
    return;
} // QInitial

QInitial::~QInitial()
{
    // intentionally blank
    return;
} // ~QInitial

void
QInitial::setDataOnPatchHierarchy(const int data_idx,
                                  Pointer<Variable<NDIM>> var,
                                  Pointer<PatchHierarchy<NDIM>> hierarchy,
                                  const double data_time,
                                  const bool /*initial_time*/,
                                  int coarsest_ln,
                                  int finest_ln)
{
    coarsest_ln = coarsest_ln < 0 ? 0 : coarsest_ln;
    finest_ln = finest_ln < 0 ? hierarchy->getFinestLevelNumber() : finest_ln;

    auto integrator = IntegrateFunction::getIntegrator();

    if (d_init_type == "ANNULUS")
    {
        auto fcn = [this](VectorNd X, double t) -> double {
            auto w = [](double theta, double D) -> double {
                return 0.5 * (std::erf((M_PI / 6.0 - theta) / std::sqrt(4.0 * D)) +
                              std::erf((M_PI / 6.0 + theta) / std::sqrt(4.0 * D)));
            };
            X = X - d_center;
            double r = X.norm();
            if (r >= d_R1 && r <= d_R2)
            {
                double theta = std::atan2(X(1), X(0));
                return w(theta - M_PI / 2.0, d_D);
            }
            else
            {
                return 0.0;
            }
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    else if (d_init_type == "DISK")
    {
        auto fcn = [this](VectorNd X, double t) -> double {
            X = X - d_center;
            for (int d = 0; d < NDIM; ++d) X(d) -= t * d_vel[d];
            double r = X.norm();
            return r <= d_R1 ? 1.0 : 0.0;
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    else if (d_init_type == "CHANNEL")
    {
        auto fcn = [this](VectorNd X, double t) -> double {
            auto w = [](double xi, double D, double t) -> double {
                return 0.5 * (std::erf((0.25 - (xi + 0.8)) / std::sqrt(4.0 * D * (1 + t))) +
                              std::erf((0.25 + xi + 0.8) / std::sqrt(4.0 * D * (1 + t))));
            };
            MatrixNd Q;
            Q(0, 0) = cos(d_theta);
            Q(0, 1) = -sin(d_theta);
            Q(1, 0) = sin(d_theta);
            Q(1, 1) = cos(d_theta);
            X[0] -= cos(M_PI / 12.0) * t;
            X[1] -= sin(M_PI / 12.0) * t;
            X = Q.transpose() * (X - d_channel_center) + d_channel_center;
            if ((X(1) - d_channel_center(1)) > 0.375 || (X(1) - d_channel_center(1)) < -0.375)
                return 0.0;
            else
                return w(X(0) - d_channel_center(0), d_D, t);
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    else if (d_init_type == "RADIAL")
    {
        auto fcn = [this](VectorNd X, double t) -> double {
            auto w = [](double r, double D, double t) -> double {
                return 10 * std::exp(-r * r / (4.0 * D * (t + 0.5))) / (4.0 * D * (t + 0.5));
            };
            X = X - d_center;
            for (int d = 0; d < NDIM; ++d) X(d) -= t * d_vel[d];
            double r = X.norm();
            return w(r, d_D, t);
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    else if (d_init_type == "ROTATIONAL")
    {
        auto fcn = [this](VectorNd X, double t) -> double {
            auto w = [](double r, double D, double t) -> double {
                return 10.0 * std::exp(-r * r / (4.0 * D * (t + 0.5))) / (4.0 * D * (t + 0.5));
            };
            MatrixNd Q;
            Q(0, 0) = Q(1, 1) = std::cos(d_v * t);
            Q(0, 1) = std::sin(d_v * t);
            Q(1, 0) = -Q(0, 1);
            X = Q * X;
            X(0) -= 2.0;
            double r = X.norm();
            return w(r, d_D, t);
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    else if (d_init_type == "CONSTANT")
    {
        auto fcn = [](VectorNd X, double t) -> double {
            return 10.0 * std::exp(-std::sqrt(std::pow(X(0) + 1.5, 2.0) + std::pow(X(1), 2.0)));
        };
        integrator->integrateFcnOnPatchHierarchy(hierarchy, d_ls_idx, data_idx, fcn, data_time);
    }
    for (int ln = coarsest_ln; ln <= finest_ln; ln++)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) > 0.0)
                {
                    (*Q_data)(idx) /= (*vol_data)(idx)*dx[0] * dx[1];
                }
            }
        }
    }
}

void
QInitial::setDataOnPatch(const int data_idx,
                         Pointer<Variable<NDIM>> /*var*/,
                         Pointer<Patch<NDIM>> patch,
                         const double /*data_time*/,
                         const bool /*initial_time*/,
                         Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_data);
#endif

    Q_data->fillAll(0.0);
    return;
} // setDataOnPatch

void
QInitial::updateAverageToTotal(Pointer<CellVariable<NDIM, double>> Q_var,
                               Pointer<VariableContext> Q_ctx,
                               Pointer<PatchHierarchy<NDIM>> hierarchy,
                               const int vol_idx)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, Q_ctx);

    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*Q_data)(idx) *= (*vol_data)(idx);
            }
        }
    }
    return;
}

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

void
QInitial::getFromInput(Pointer<Database> db)
{
    if (db)
    {
        d_init_type = db->getStringWithDefault("init_type", d_init_type);

        if (d_init_type == "ANNULUS")
        {
            d_D = db->getDoubleWithDefault("D", d_D);
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            d_R2 = db->getDoubleWithDefault("R2", d_R2);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
        }
        else if (d_init_type == "CHANNEL")
        {
            d_D = db->getDoubleWithDefault("D", d_D);
            d_theta = db->getDoubleWithDefault("Theta", d_theta);
            d_y_p = db->getDoubleWithDefault("Y_p", d_y_p);
            d_y_n = db->getDoubleWithDefault("Y_n", d_y_n);
        }
        else if (d_init_type == "DISK")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
            d_vel.resize(NDIM);
            db->getDoubleArray("velocity", d_vel.data(), NDIM);
        }
        else if (d_init_type == "RADIAL")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
            d_vel.resize(NDIM);
            db->getDoubleArray("velocity", d_vel.data(), NDIM);
            d_D = db->getDoubleWithDefault("D", d_D);
        }
        else if (d_init_type == "ROTATIONAL")
        {
            d_R1 = db->getDoubleWithDefault("R1", d_R1);
            if (db->keyExists("Center")) db->getDoubleArray("Center", d_center.data(), NDIM);
            d_v = db->getDouble("v");
            d_D = db->getDoubleWithDefault("D", d_D);
        }
        else if (d_init_type == "CONSTANT")
        {
            // intentionally blank.
        }
        else
        {
            TBOX_ERROR(d_object_name << "::getFromInput()\n"
                                     << "  invalid initialization type " << d_init_type << "\n");
        }
    }
    return;
} // getFromInput

//////////////////////////////////////////////////////////////////////////////
