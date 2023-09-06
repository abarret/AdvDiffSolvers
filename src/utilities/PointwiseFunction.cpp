#include "ibamr/config.h"

#include "ADS/PointwiseFunction.h"
#include "ADS/app_namespaces.h"

#include <CartesianPatchGeometry.h>
#include <SAMRAI_config.h>

namespace ADS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////
template <typename F>
PointwiseFunction<F>::PointwiseFunction(string object_name, F f) : CartGridFunction(std::move(object_name)), d_f(f)
{
    // intentionally blank
    return;
} // PointwiseFunction

template <>
void
PointwiseFunction<std::function<double(double, const VectorNd&, double)>>::setDataOnPatch(
    const int data_idx,
    Pointer<Variable<NDIM>> var,
    Pointer<Patch<NDIM>> patch,
    const double data_time,
    const bool initial_time,
    Pointer<PatchLevel<NDIM>> /*patch_level*/)
{
    const Box<NDIM>& patch_box = patch->getBox();
    const hier::Index<NDIM>& idx_low = patch_box.lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const xlow = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    // Set the data in the patch.
    Pointer<PatchData<NDIM>> data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(data);
#endif
    Pointer<CellData<NDIM, double>> cc_data = data;
    Pointer<NodeData<NDIM, double>> nc_data = data;
    if (cc_data)
    {
        for (CellIterator<NDIM> ci(patch_box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();

            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
            (*cc_data)(idx) = d_f((*cc_data)(idx), x, data_time);
        }
    }
    else if (nc_data)
    {
        for (NodeIterator<NDIM> ni(patch_box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));
            (*nc_data)(idx) = d_f((*nc_data)(idx), x, data_time);
        }
    }
    else
    {
        TBOX_ERROR("PointwiseFunction::setDataOnPatch(): unsupported patch data type encountered\n");
    }
}

template <>
void
PointwiseFunction<std::function<VectorNd(const VectorNd&, const VectorNd&, double)>>::setDataOnPatch(
    const int data_idx,
    Pointer<Variable<NDIM>> var,
    Pointer<Patch<NDIM>> patch,
    const double data_time,
    const bool initial_time,
    Pointer<PatchLevel<NDIM>> /*patch_level*/)
{
    const Box<NDIM>& patch_box = patch->getBox();
    const hier::Index<NDIM>& idx_low = patch_box.lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const xlow = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    // Set the data in the patch.
    Pointer<PatchData<NDIM>> data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(data);
#endif
    Pointer<CellData<NDIM, double>> cc_data = data;
    Pointer<NodeData<NDIM, double>> nc_data = data;
    if (cc_data)
    {
        for (CellIterator<NDIM> ci(patch_box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();

            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
            // Vector values
            Eigen::Map<const VectorNd> cc_val(&(*cc_data)(idx));
            VectorNd ret_val = d_f(cc_val, x, data_time);
            for (int d = 0; d < NDIM; ++d) (*cc_data)(idx, d) = ret_val[d];
        }
    }
    else if (nc_data)
    {
        for (NodeIterator<NDIM> ni(patch_box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));
            // Vector values
            Eigen::Map<const VectorNd> nc_val(&(*nc_data)(idx));
            VectorNd ret_val = d_f(nc_val, x, data_time);
            for (int d = 0; d < NDIM; ++d) (*nc_data)(idx, d) = ret_val[d];
        }
    }
    else
    {
        TBOX_ERROR("PointwiseFunction::setDataOnPatch(): unsupported patch data type encountered\n");
    }
}

template <>
void
PointwiseFunction<std::function<VectorXd(const VectorXd&, const VectorNd&, double)>>::setDataOnPatch(
    const int data_idx,
    Pointer<Variable<NDIM>> var,
    Pointer<Patch<NDIM>> patch,
    const double data_time,
    const bool initial_time,
    Pointer<PatchLevel<NDIM>> /*patch_level*/)
{
    const Box<NDIM>& patch_box = patch->getBox();
    const hier::Index<NDIM>& idx_low = patch_box.lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const xlow = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    // Set the data in the patch.
    Pointer<PatchData<NDIM>> data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(data);
#endif
    Pointer<CellData<NDIM, double>> cc_data = data;
    Pointer<NodeData<NDIM, double>> nc_data = data;
    if (cc_data)
    {
        const int depth = cc_data->getDepth();
        for (CellIterator<NDIM> ci(patch_box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();

            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
            Eigen::Map<const VectorXd> cc_val(&(*cc_data)(idx), depth);
            VectorXd ret_val = d_f(cc_val, x, data_time);
            for (int d = 0; d < depth; ++d) (*cc_data)(idx, d) = ret_val[d];
        }
    }
    else if (nc_data)
    {
        const int depth = nc_data->getDepth();
        for (NodeIterator<NDIM> ni(patch_box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));
            Eigen::Map<const VectorXd> nc_val(&(*nc_data)(idx), depth);
            VectorXd ret_val = d_f(nc_val, x, data_time);
            for (int d = 0; d < depth; ++d) (*nc_data)(idx, d) = ret_val[d];
        }
    }
    else
    {
        TBOX_ERROR("PointwiseFunction::setDataOnPatch(): unsupported patch data type encountered\n");
    }
}

template <>
void
PointwiseFunction<std::function<double(double, const VectorNd&, double, int)>>::setDataOnPatch(
    const int data_idx,
    Pointer<Variable<NDIM>> var,
    Pointer<Patch<NDIM>> patch,
    const double data_time,
    const bool initial_time,
    Pointer<PatchLevel<NDIM>> /*patch_level*/)
{
    const Box<NDIM>& patch_box = patch->getBox();
    const hier::Index<NDIM>& idx_low = patch_box.lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const xlow = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    // Set the data in the patch.
    Pointer<PatchData<NDIM>> data = patch->getPatchData(data_idx);
#if !defined(NDEBUG)
    TBOX_ASSERT(data);
#endif
    Pointer<FaceData<NDIM, double>> fc_data = data;
    Pointer<SideData<NDIM, double>> sc_data = data;
    Pointer<EdgeData<NDIM, double>> ec_data = data;
    if (sc_data)
    {
        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (SideIterator<NDIM> si(patch_box, axis); si; si++)
            {
                const SideIndex<NDIM>& idx = si();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d)
                    x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + (d == axis ? 0.0 : 0.5));

                (*sc_data)(idx) = d_f((*sc_data)(idx), x, data_time, axis);
            }
        }
    }
    else if (fc_data)
    {
        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (FaceIterator<NDIM> fi(patch_box, axis); fi; fi++)
            {
                const FaceIndex<NDIM>& idx = fi();
                const CellIndex<NDIM>& cell_idx = idx.toCell(1);
                VectorNd x;
                for (int d = 0; d < NDIM; ++d)
                    x[d] = xlow[d] + dx[d] * (static_cast<double>(cell_idx(d) - idx_low(d)) + (d == axis ? 0.0 : 0.5));

                (*fc_data)(idx) = d_f((*fc_data)(idx), x, data_time, axis);
            }
        }
    }
    else if (ec_data)
    {
        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (EdgeIterator<NDIM> ei(patch_box, axis); ei; ei++)
            {
                const EdgeIndex<NDIM>& idx = ei();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d)
                    x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + (d == axis ? 0.5 : 0.0));

                (*ec_data)(idx) = d_f((*ec_data)(idx), x, data_time, axis);
            }
        }
    }
    else
    {
        TBOX_ERROR("PointwiseFunction::setDataOnPatch(): unsupported patch data type encountered\n");
    }
}

// Instantiate valid templates
template class PointwiseFunction<std::function<double(double, const VectorNd&, double)>>;
template class PointwiseFunction<std::function<VectorNd(const VectorNd&, const VectorNd&, double)>>;
template class PointwiseFunction<std::function<VectorXd(const VectorXd&, const VectorNd&, double)>>;
template class PointwiseFunction<std::function<double(double, const VectorNd&, double, int)>>;

} // namespace ADS
