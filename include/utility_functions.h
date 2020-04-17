#ifndef included_utility_functions
#define included_utility_functions

#include "LSFindCellVolume.h"
#include "SetLSValue.h"

#include "Variable.h"
#include "tbox/MathUtilities.h"

namespace IBAMR
{
static int Q_cloned_idx = IBTK::invalid_index;
inline double
length_fraction(const double dx, const double phi_l, const double phi_u)
{
    double L = 0.0;
    if (phi_l < 0.0 && phi_u > 0.0)
    {
        L = phi_l / (phi_l - phi_u);
    }
    else if (phi_l > 0.0 && phi_u < 0.0)
    {
        L = phi_u / (phi_u - phi_l);
    }
    else if (phi_l < 0.0 && phi_u < 0.0)
    {
        L = 1.0;
    }
    return L * dx;
}

inline VectorNd
midpoint_value(const VectorNd& pt0, const double& phi0, const VectorNd& pt1, const double& phi1)
{
    return pt0 * phi1 / (phi1 - phi0) - pt1 * phi0 / (phi1 - phi0);
}

inline IBTK::VectorNd
find_cell_centroid(const CellIndex<NDIM>& idx, const NodeData<NDIM, double>& ls_data)
{
    IBTK::VectorNd center;
    center.setZero();
    std::vector<IBTK::VectorNd> X_pts;
    double phi_ll = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 0))) + 1.0e-10;
    double phi_lr = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 0))) + 1.0e-10;
    double phi_ur = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, 1))) + 1.0e-10;
    double phi_ul = ls_data(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, 1))) + 1.0e-10;
    if ((phi_ll < 0.0 && phi_lr < 0.0 && phi_ur < 0.0 && phi_lr < 0.0) ||
        (phi_ll > 0.0 && phi_lr > 0.0 && phi_ur > 0.0 && phi_lr > 0.0))
    {
        // Not a cut cell. Center is idx
        center(0) = idx(0);
        center(1) = idx(1);
    }
    else
    {
        // Loop over nodes and edges and find points
        if (phi_ll < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - 0.5, idx(1) - 0.5));
        if (phi_ll * phi_ul < 0.0)
            X_pts.push_back(IBTK::VectorNd(idx(0) - 0.5, idx(1) - 0.5 - phi_ll / (phi_ul - phi_ll)));
        if (phi_ul < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) - 0.5, idx(1) + 0.5));
        if (phi_ul * phi_ur < 0.0)
            X_pts.push_back(IBTK::VectorNd(idx(0) - 0.5 - phi_ul / (phi_ur - phi_ul), idx(1) + 0.5));
        if (phi_ur < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 0.5, idx(1) + 0.5));
        if (phi_ur * phi_lr < 0.0)
            X_pts.push_back(IBTK::VectorNd(idx(0) + 0.5, idx(1) - 0.5 - phi_lr / (phi_ur - phi_lr)));
        if (phi_lr < 0.0) X_pts.push_back(IBTK::VectorNd(idx(0) + 0.5, idx(1) - 0.5));
        if (phi_lr * phi_ll < 0.0)
            X_pts.push_back(IBTK::VectorNd(idx(0) - 0.5 - phi_ll / (phi_lr - phi_ll), idx(1) - 0.5));

        double signed_area = 0.0;
        for (size_t i = 0; i < X_pts.size(); ++i)
        {
            const IBTK::VectorNd& X = X_pts[i];
            const IBTK::VectorNd& X_n = X_pts[(i + 1) % X_pts.size()];
            center += (X + X_n) * (X(0) * X_n(1) - X_n(0) * X(1));
            signed_area += 0.5 * (X(0) * X_n(1) - X_n(0) * X(1));
        }
        center /= 6.0 * signed_area;
    }

    return center;
}

struct SetLSAndVolume
{
    SetLSAndVolume(int ls_idx,
                   int vol_idx,
                   int area_idx,
                   SAMRAI::tbox::Pointer<SetLSValue> set_ls_val,
                   SAMRAI::tbox::Pointer<LSFindCellVolume> find_vol)
        : d_ls_idx(ls_idx), d_vol_idx(vol_idx), d_area_idx(area_idx), d_set_ls_val(set_ls_val), d_find_vol(find_vol)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        var_db->mapIndexToVariable(d_ls_idx, d_ls_var);
        var_db->mapIndexToVariable(d_vol_idx, d_vol_var);
        var_db->mapIndexToVariable(d_area_idx, d_area_var);
    }

    void updateLSAndVol(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                        double data_time,
                        bool initial_time)
    {
        d_set_ls_val->setDataOnPatchHierarchyWithGhosts(d_ls_idx, d_ls_var, hierarchy, data_time, initial_time);
        d_find_vol->updateVolumeAndArea(d_vol_idx, d_vol_var, d_area_idx, d_area_var, d_ls_idx, d_ls_var, true);
    }

private:
    Pointer<SAMRAI::hier::Variable<NDIM>> d_ls_var, d_vol_var, d_area_var;
    int d_ls_idx = IBTK::invalid_index, d_vol_idx = IBTK::invalid_index, d_area_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SetLSValue> d_set_ls_val;
    SAMRAI::tbox::Pointer<LSFindCellVolume> d_find_vol;
};

inline void
regridHierarchyCallback(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                        const double data_time,
                        const bool initial_time,
                        void* ctx)
{
    auto setLSAndVol = static_cast<SetLSAndVolume*>(ctx);
    setLSAndVol->updateLSAndVol(hierarchy, data_time, initial_time);
}

inline bool
compare(double a, double b, double eps = SAMRAI::tbox::MathUtilities<double>::getEpsilon())
{
    double diff = std::fabs(a - b);
    a = std::fabs(a);
    b = std::fabs(b);
    const double larger = a > b ? a : b;
    if (diff <= (larger * eps)) return true;
    return false;
}

inline double
node_to_cell(const CellIndex<NDIM>& idx, NodeData<NDIM, double>& ls_data)
{
    NodeIndex<NDIM> idx_ll(idx, IntVector<NDIM>(0, 0));
    NodeIndex<NDIM> idx_lu(idx, IntVector<NDIM>(0, 1));
    NodeIndex<NDIM> idx_ul(idx, IntVector<NDIM>(1, 0));
    NodeIndex<NDIM> idx_uu(idx, IntVector<NDIM>(1, 1));
    double ls_ll = ls_data(idx_ll), ls_lu = ls_data(idx_lu);
    double ls_ul = ls_data(idx_ul), ls_uu = ls_data(idx_uu);
    return 0.25 * (ls_ll + ls_lu + ls_ul + ls_uu);
}

} // namespace IBAMR
#endif
