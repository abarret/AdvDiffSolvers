#ifndef included_ADS_ads_utilities_inc
#define included_ADS_ads_utilities_inc
#include <ADS/ads_utilities.h>
#include <ADS/ls_functions.h>

#include <PatchLevel.h>

namespace ADS
{
inline void
allocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                    const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                    const double time,
                    int coarsest_ln,
                    int finest_ln)
{
    coarsest_ln = coarsest_ln < 0 ? 0 : coarsest_ln;
    finest_ln = finest_ln < 0 ? hierarchy->getFinestLevelNumber() : finest_ln;
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        level->allocatePatchData(comp, time);
    }
}

inline void
allocate_patch_data(const std::set<int>& idxs,
                    const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                    const double time,
                    const int coarsest_ln,
                    const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comp;
    for (const auto& idx : idxs) comp.setFlag(idx);
    allocate_patch_data(comp, hierarchy, time, coarsest_ln, finest_ln);
}

inline void
allocate_patch_data(const int idx,
                    const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                    const double time,
                    const int coarsest_ln,
                    const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comp;
    comp.setFlag(idx);
    allocate_patch_data(comp, hierarchy, time, coarsest_ln, finest_ln);
}

inline void
deallocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                      const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                      int coarsest_ln,
                      int finest_ln)
{
    coarsest_ln = coarsest_ln < 0 ? 0 : coarsest_ln;
    finest_ln = finest_ln < 0 ? hierarchy->getFinestLevelNumber() : finest_ln;
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        level->deallocatePatchData(comp);
    }
}

inline void
deallocate_patch_data(const std::set<int>& idxs,
                      const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                      const int coarsest_ln,
                      const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comps;
    for (const auto& idx : idxs) comps.setFlag(idx);

    deallocate_patch_data(comps, hierarchy, coarsest_ln, finest_ln);
}

inline void
deallocate_patch_data(const int idx,
                      const SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>>& hierarchy,
                      const int coarsest_ln,
                      const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comps;
    comps.setFlag(idx);
    deallocate_patch_data(comps, hierarchy, coarsest_ln, finest_ln);
}

template <class F, typename... Args>
void
perform_on_patch_hierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy, F fcn, Args... args)
{
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
            fcn(patch, args...);
        }
    }
}

inline void
swap_patch_data(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const int data1_idx, const int data2_idx)
{
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchData<NDIM>> data1 = patch->getPatchData(data1_idx);
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchData<NDIM>> data2 = patch->getPatchData(data2_idx);
#ifndef NDEBUG
    // Ensure that data1 and data2 encapsulate same space
    TBOX_ASSERT(data1->getBox() == data2->getBox());
#endif

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchDataFactory<NDIM>> factory =
        patch->getPatchDescriptor()->getPatchDataFactory(data1_idx);
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchData<NDIM>> temp_data = factory->allocate(data1->getBox());
    temp_data->copy(*data1);
    data1->copy(*data2);
    data2->copy(*temp_data);
}

inline void
copy_patch_data(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const int dst_idx, const int src_idx)
{
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchData<NDIM>> dst = patch->getPatchData(dst_idx);
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchData<NDIM>> src = patch->getPatchData(src_idx);
#ifndef NDEBUG
    // Ensure that data1 and data2 encapsulate same space
    TBOX_ASSERT(dst->getBox() == src->getBox());
#endif
    dst->copy(*src);
}

inline void
reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        const int ls_idx,
                        const int dst_idx,
                        const int src_idx,
                        double reset_val,
                        const bool use_negative)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
            SAMRAI::tbox::Pointer<SAMRAI::pdat::CellData<NDIM, double>> dst_data = patch->getPatchData(dst_idx);
            SAMRAI::tbox::Pointer<SAMRAI::pdat::CellData<NDIM, double>> src_data = patch->getPatchData(src_idx);
#ifndef NDEBUG
            TBOX_ASSERT(dst_data->getDepth() == src_data->getDepth());
#endif
            SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            for (SAMRAI::pdat::CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const SAMRAI::pdat::CellIndex<NDIM>& idx = ci();
                if ((use_negative ? 1.0 : -1.0) * node_to_cell(idx, *ls_data) > 0.0)
                {
                    for (int d = 0; d < dst_data->getDepth(); ++d) (*dst_data)(idx, d) = reset_val;
                }
                else
                {
                    for (int d = 0; d < dst_data->getDepth(); ++d) (*dst_data)(idx, d) = (*src_data)(idx, d);
                }
            }
        }
    }
}

inline void
reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        const int ls_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                        double reset_val,
                        const bool use_negative)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (SAMRAI::hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch = level->getPatch(p());
            SAMRAI::tbox::Pointer<SAMRAI::pdat::CellData<NDIM, double>> Q_data = patch->getPatchData(Q_var, ctx);
            SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            for (SAMRAI::pdat::CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const SAMRAI::pdat::CellIndex<NDIM>& idx = ci();
                if ((use_negative ? 1.0 : -1.0) * node_to_cell(idx, *ls_data) > 0.0)
                {
                    for (int d = 0; d < Q_data->getDepth(); ++d) (*Q_data)(idx, d) = reset_val;
                }
            }
        }
    }
}

inline void
reset_unphysical_values(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                        const int ls_idx,
                        const std::set<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& Q_vars,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                        double reset_val,
                        const bool use_negative)
{
    for (const auto& Q_var : Q_vars) reset_unphysical_values(hierarchy, ls_idx, Q_var, ctx, reset_val, use_negative);
}

inline void
reset_unphysical_values(
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
    const int ls_idx,
    const std::set<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& Q_vars,
    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
    const std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>, double>& reset_map,
    const bool use_negative)
{
    for (const auto& Q_var : Q_vars)
        reset_unphysical_values(hierarchy, ls_idx, Q_var, ctx, reset_map.at(Q_var), use_negative);
}

} // namespace ADS
#endif
