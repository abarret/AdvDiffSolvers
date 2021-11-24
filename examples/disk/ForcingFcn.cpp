#include "ibamr/config.h"

#include "CCAD/app_namespaces.h"
#include "CCAD/ls_functions.h"

#include "ForcingFcn.h"

#include <SAMRAI_config.h>

#include <array>

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
    d_sf_max = input_db->getDouble("sf_max");
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
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();

    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        if ((*vol_data)(idx) > 0.0)
        {
            VectorNd X = find_cell_centroid(idx, *ls_data);
            for (int d = 0; d < NDIM; ++d) X[d] = xlow[d] + dx[d] * (X[d] - idx_low(d));
            X -= d_cent;
            const double r2 = X.squaredNorm();
            (*F_data)(idx) = 4.0 * d_D * time * (time - 1) - (2.0 * time - 1) * r2;
        }
    }
    return;
} // setDataOnPatch
