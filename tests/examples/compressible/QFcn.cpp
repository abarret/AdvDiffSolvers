#include "ibamr/config.h"

#include "ADS/app_namespaces.h"

#include "QFcn.h"

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// PUBLIC ///////////////////////////////////////

QFcn::QFcn(string object_name) : CartGridFunction(std::move(object_name))
{
    // intentionally blank
    return;
} // QFcn

void
QFcn::setDataOnPatch(const int data_idx,
                     Pointer<hier::Variable<NDIM>> var,
                     Pointer<Patch<NDIM>> patch,
                     const double t,
                     const bool initial_time,
                     Pointer<PatchLevel<NDIM>> level)
{
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();

    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
        (*Q_data)(idx) = std::pow(std::cos(M_PI * (x[0])) * std::cos(M_PI * (x[1])), 2.0);
    }
    return;
} // setDataOnPatch
