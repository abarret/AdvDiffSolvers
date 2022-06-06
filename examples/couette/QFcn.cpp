#include "ibamr/config.h"

#include "ADS/app_namespaces.h"

#include "QFcn.h"

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// PUBLIC ///////////////////////////////////////

QFcn::QFcn(const string& object_name, Pointer<Database> input_db) : LSCartGridFunction(object_name)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
#endif

    // Initialize object with data read from the input database.
    getFromInput(input_db);
    return;
} // QFcn

QFcn::~QFcn()
{
    // intentionally blank
    return;
} // ~QFcn

void
QFcn::setDataOnPatch(const int data_idx,
                     Pointer<Variable<NDIM>> var,
                     Pointer<Patch<NDIM>> patch,
                     const double data_time,
                     const bool initial_time,
                     Pointer<PatchLevel<NDIM>> level)
{
    if (initial_time) return;
    auto fcn = [this](VectorNd X, double t) -> double {
        auto w = [](double r, double D, double t) -> double {
            if (r < 1.0)
                return std::pow(std::cos(M_PI * r) + 1.0, 2.0);
            else
                return 0.0;
        };
        X = X - d_com;
        double r = X.norm();
        return w(r, d_D, t);
    };
    auto integrator = IntegrateFunction::getIntegrator();
    integrator->integrateFcnOnPatch(patch, d_ls_idx, data_idx, fcn, data_time, level);
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        if ((*vol_data)(idx) > 0.0) (*Q_data)(idx) /= (*vol_data)(idx)*dx[0] * dx[1];
    }
    return;
} // setDataOnPatch

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

void
QFcn::getFromInput(Pointer<Database> db)
{
    d_D = db->getDouble("D");
    db->getDoubleArray("com", d_com.data(), NDIM);
    d_R = db->getDouble("r");
    return;
} // getFromInput
