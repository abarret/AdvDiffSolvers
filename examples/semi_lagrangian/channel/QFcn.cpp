#include "ibamr/config.h"

#include "ADS/app_namespaces.h"

#include "QFcn.h"

#include <SAMRAI_config.h>

#include <array>

namespace ADS
{

/////////////////////////////// PUBLIC ///////////////////////////////////////

QFcn::QFcn(std::string object_name, Pointer<Database> input_db) : LSCartGridFunction(std::move(object_name))
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
#endif

    // Initialize object with data read from the input database.
    getFromInput(input_db);
    return;
} // QFcn

void
QFcn::setDataOnPatch(const int data_idx,
                     Pointer<Variable<NDIM>> var,
                     Pointer<Patch<NDIM>> patch,
                     const double data_time,
                     const bool initial_time,
                     Pointer<PatchLevel<NDIM>> level)
{
    auto fcn = [this](VectorNd X, double t) -> double
    {
        auto w = [this](double r, double D, double t) -> double
        {
            if (r < d_R)
                return std::pow(std::cos(M_PI * r / d_R) + 1.0, 2.0);
            else
                return 0.0;
        };
        X = X - d_com;
        double r = X.norm();
        return w(r, d_D, t);
    };

    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);

    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
        // Shift this point to it's reference configuration
        x[0] -= data_time * x[1] * (1.0 - x[1]);
        (*Q_data)(idx) = fcn(x, data_time);
    }
    return;
} // setDataOnPatch

/////////////////////////////// PRIVATE //////////////////////////////////////

void
QFcn::getFromInput(Pointer<Database> db)
{
    d_D = db->getDouble("D");
    db->getDoubleArray("com", d_com.data(), NDIM);
    d_R = db->getDouble("r");
    return;
} // getFromInput
} // namespace ADS
