#ifndef included_ADS_ads_utilities_inc
#define included_ADS_ads_utilities_inc
#include <ADS/ads_utilities.h>

#include <PatchLevel.h>

namespace ADS
{
inline void
allocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
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
                    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                    const double time,
                    const int coarsest_ln,
                    const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comps;
    for (const auto& idx : idxs) comps.setFlag(idx);

    allocate_patch_data(comps, hierarchy, time, coarsest_ln, finest_ln);
}

inline void
allocate_patch_data(const int idx,
                    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                    const double time,
                    const int coarsest_ln,
                    const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comps;
    comps.setFlag(idx);
    allocate_patch_data(comps, hierarchy, time, coarsest_ln, finest_ln);
}

inline void
deallocate_patch_data(const SAMRAI::hier::ComponentSelector& comp,
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
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
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                      const int coarsest_ln,
                      const int finest_ln)
{
    SAMRAI::hier::ComponentSelector comps;
    for (const auto& idx : idxs) comps.setFlag(idx);

    deallocate_patch_data(comps, hierarchy, coarsest_ln, finest_ln);
}

inline void
deallocate_patch_data(const int idx,
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
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

} // namespace ADS
#endif
