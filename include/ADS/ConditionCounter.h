/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_ConditionCounter
#define included_ADS_ConditionCounter

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/FDPoint.h>
#include <ADS/FDWeightsCache.h>

#include <mpi.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * ConditionCounter is a class that counts the number of conditions being applied to points. This is useful when we want
 * to enforce multiple conditions at a single location (e.g boundary condition and PDE operator).
 *
 * This class maintains a map between finite difference points and their global condition number. Note that each point
 * can contain multiple conditions.
 */
class ConditionCounter
{
public:
    /*!
     * \brief ConditionCounter constructor. Does nothing interesting. Leaves the object in an uninitialized state.
     */
    ConditionCounter(std::string object_name);

    /*!
     * \brief ConditionCounter constructor. Assigns conditions to each finite difference weight stored in the cache.
     */
    ConditionCounter(std::string object_name,
                     SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                     const FDWeightsCache& fd_cache);

    /*!
     * \brief Default destructor.
     */
    ~ConditionCounter() = default;

    /*!
     * \brief Get the number of conditions per processor.
     */
    const std::vector<unsigned int>& getNumConditionsPerProc()
    {
        return d_conditions_per_proc;
    }

    /*!
     * \brief Get the map between FD points and the global condition index.
     */
    const std::multimap<FDPoint, unsigned int>& getFDConditionMapPatch(SAMRAI::hier::Patch<NDIM>* p)
    {
        return d_fd_condition_map.at(p);
    }
    const std::map<SAMRAI::hier::Patch<NDIM>*, std::multimap<FDPoint, unsigned int>>& getFDConditionMap()
    {
        return d_fd_condition_map;
    }

    /*!
     * \brief Clear the condition counter
     */
    void clearConditions();

    /*!
     * \brief Add conditions based on those stored in the cache
     */
    void addCondition(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                      const FDWeightsCache& fd_cache);

    /*!
     * \brief Add a condition that depends on the point pt. Returns the index of the condition.
     */
    unsigned int addCondition(SAMRAI::hier::Patch<NDIM>*, FDPoint pt);

    /*!
     * \brief Update the global condition count. This assigns unique global conditions.
     *
     * \note This requires an all-to-all communication pattern so should only be done after all conditions are
     * registered.
     */
    void updateGlobalConditionCount();

private:
    std::string d_object_name;

    // Map between a finite difference point and it's global condition number.
    std::map<SAMRAI::hier::Patch<NDIM>*, std::multimap<FDPoint, unsigned int>> d_fd_condition_map;
    std::vector<unsigned int> d_conditions_per_proc;
};
} // namespace ADS

#endif // included_ADS_ConditionCounter
