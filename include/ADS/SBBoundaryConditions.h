#ifndef included_ADS_SBBoundaryConditions
#define included_ADS_SBBoundaryConditions

#include "ADS/LSCutCellBoundaryConditions.h"
#include "ADS/RBFReconstructCache.h"
#include "ADS/SBSurfaceFluidCouplingManager.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/equation_systems.h"
#include <libmesh/mesh.h>

namespace ADS
{
class SBBoundaryConditions : public LSCutCellBoundaryConditions
{
public:
    using ReactionFcn =
        std::function<double(double, const std::vector<double>&, const std::vector<double>&, double, void*)>;

    SBBoundaryConditions(const std::string& object_name,
                         const std::string& fl_name,
                         const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                         const SAMRAI::tbox::Pointer<CutCellMeshMapping>& cut_cell_mesh_mapping,
                         const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings);

    ~SBBoundaryConditions() = default;

    /*!
     * \brief Deleted default constructor.
     */
    SBBoundaryConditions() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    SBBoundaryConditions(const SBBoundaryConditions& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    SBBoundaryConditions& operator=(const SBBoundaryConditions& that) = delete;

    void setFluidContext(SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx);

    void allocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                               double time) override;
    void deallocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                 double time) override;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                double time) override;

private:
    double d_D_coef = std::numeric_limits<double>::quiet_NaN();

    std::shared_ptr<SBSurfaceFluidCouplingManager> d_sb_data_manager;
    SAMRAI::tbox::Pointer<CutCellMeshMapping> d_cut_cell_mapping;

    std::string d_fl_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> d_ctx;

    std::vector<FEToHierarchyMapping*> d_fe_hierarchy_mappings;
};

} // namespace ADS
#endif
