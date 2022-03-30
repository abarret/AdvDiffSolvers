#include <ADS/ConditionCounter.h>
#include <ADS/app_namespaces.h>

#include <ibtk/IBTK_MPI.h>

#include <RefineAlgorithm.h>

namespace ADS
{
ConditionCounter::ConditionCounter(std::string object_name) : d_object_name(std::move(object_name))
{
    // intentionally blank
}

ConditionCounter::ConditionCounter(std::string object_name,
                                   Pointer<PatchHierarchy<NDIM>> hierarchy,
                                   const FDWeightsCache& fd_cache)
    : d_object_name(std::move(object_name))
{
    addCondition(hierarchy, fd_cache);
    updateGlobalConditionCount();
}

void
ConditionCounter::clearConditions()
{
    d_fd_condition_map.clear();
    d_conditions_per_proc.clear();
}

unsigned int
ConditionCounter::addCondition(Patch<NDIM>* patch, FDPoint pt)
{
    int rank = IBTK_MPI::getRank();
    d_fd_condition_map[patch].insert(std::make_pair(pt, d_conditions_per_proc[rank]));
    return d_conditions_per_proc[rank]++;
}

void
ConditionCounter::addCondition(Pointer<PatchHierarchy<NDIM>> hierarchy, const FDWeightsCache& fd_cache)
{
    for (int ln = 0; ln <= hierarchy->getPatchLevel(ln); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const std::set<FDPoint>& fd_pts = fd_cache.getRBFFDBasePoints(patch);
            for (const auto& fd_pt : fd_pts)
            {
                addCondition(patch.getPointer(), fd_pt);
            }
        }
    }
}

void
ConditionCounter::updateGlobalConditionCount()
{
    const int rank = IBTK_MPI::getRank();
    const int nodes = IBTK_MPI::getNodes();
    for (int i = 0; i < nodes; ++i)
    {
        if (i == rank) continue;
        d_conditions_per_proc[i] = 0;
    }
    IBTK_MPI::sumReduction(d_conditions_per_proc.data(), nodes);
}
} // namespace ADS
