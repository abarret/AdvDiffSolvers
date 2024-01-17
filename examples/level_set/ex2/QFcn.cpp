#include "ibamr/config.h"

#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include "QFcn.h"

#include <SAMRAI_config.h>

#include <array>

/////////////////////////////// PUBLIC ///////////////////////////////////////
QFcn::QFcn(const string& object_name, Pointer<Database> input_db) : CartGridFunction(object_name)
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

    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
        (*Q_data)(idx) = setVal(x);
    }
    return;
} // setDataOnPatch

double
QFcn::setVal(VectorNd x)
{
    // Shift x to disk
    VectorNd x_ref = x - d_cent;
    // Compute r
    double r = x_ref.norm();
    // If r is within disk AND inside channel, set to smooth function, otherwise constant.
    if (r <= d_R && x[1] <= (d_alpha / (2.0 * M_PI) * (1.0 + d_gamma * std::sin(2.0 * M_PI * (x[0])))))
        return std::max(std::cos(M_PI * r / (2.0 * d_R)), 0.0);
    else
        return 0.0;
}

void
QFcn::getFromInput(Pointer<Database> db)
{
    d_R = db->getDouble("r");
    d_alpha = db->getDouble("alpha");
    d_gamma = db->getDouble("gamma");
    db->getDoubleArray("center", d_cent.data(), NDIM);
    return;
} // getFromInput
