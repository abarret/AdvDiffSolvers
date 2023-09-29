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
    std::function<double(const VectorNd&, double t)> fcn;
    switch (d_fcn_type)
    {
    case FcnType::SINE:
        fcn = [this](const VectorNd& x, double t) -> double
        {
            auto w = [](double r, double t) -> double
            {
                if (r < 1.0)
                    return std::pow(std::cos(M_PI * r) + 1.0, 2.0);
                else
                    return 0.0;
            };
            VectorNd X = x - d_com;
            double r = X.norm();
            return w(r, t);
        };
        break;
    case FcnType::TRIANGLE:
        fcn = [](const VectorNd& x, double t) -> double { return 2.0 * std::abs(x[0] - std::floor(x[0] + 0.5)); };
        break;
    case FcnType::DISK:
        fcn = [this](const VectorNd& x, double t) -> double
        {
            auto w = [](double r, double t) -> double
            {
                if (r < 0.5)
                    return 1.0;
                else
                    return 0.0;
            };
            VectorNd X = x - d_com;
            double r = X.norm();
            return w(r, t);
        };
        break;
    default:
        TBOX_ERROR("Unknown type.");
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
    db->getDoubleArray("com", d_com.data(), NDIM);
    std::string fcn_type = db->getString("type");
    if (fcn_type.compare("SINE") == 0) d_fcn_type = FcnType::SINE;
    if (fcn_type.compare("TRIANGLE") == 0) d_fcn_type = FcnType::TRIANGLE;
    if (fcn_type.compare("DISK") == 0) d_fcn_type = FcnType::DISK;
    return;
} // getFromInput
} // namespace ADS
