#ifndef included_ADS_SBIntegrator
#define included_ADS_SBIntegrator

#include "ADS/RBFReconstructCache.h"
#include "ADS/SBSurfaceFluidCouplingManager.h"

#include "ibtk/FEDataManager.h"

#include "CellData.h"
#include "NodeData.h"

#include "libmesh/equation_systems.h"
#include <libmesh/mesh.h>

namespace ADS
{
class SBIntegrator
{
public:
    using ReactionFcn =
        std::function<double(double, const std::vector<double>&, const std::vector<double>&, double, void*)>;

    SBIntegrator(std::string object_name,
                 const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_sf_fl_coupling_manager);

    /*!
     * \brief Deleted default constructor.
     */
    SBIntegrator() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    SBIntegrator(const SBIntegrator& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    SBIntegrator& operator=(const SBIntegrator& that) = delete;

    void setLSData(int ls_idx, int vol_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    void
    integrateHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx, double current_time, double new_time);

    void beginTimestepping(double current_time, double new_time);
    void endTimestepping(double current_time, double new_time);

private:
    std::string d_object_name;

    std::shared_ptr<SBSurfaceFluidCouplingManager> d_sb_data_manager;

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    int d_vol_idx = IBTK::invalid_index, d_ls_idx = IBTK::invalid_index;
};

} // namespace ADS
#endif
