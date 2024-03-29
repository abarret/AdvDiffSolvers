#ifndef included_ADS_SBAdvDiffIntegrator
#define included_ADS_SBAdvDiffIntegrator

#include "ibamr/config.h"

#include "ADS/AdvectiveReconstructionOperator.h"
#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/LSAdvDiffIntegrator.h"
#include "ADS/LSCutCellLaplaceOperator.h"
#include "ADS/LSFindCellVolume.h"
#include "ADS/MLSReconstructCache.h"
#include "ADS/RBFReconstructCache.h"
#include "ADS/SBIntegrator.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "ibamr/AdvDiffHierarchyIntegrator.h"
#include "ibamr/IBHierarchyIntegrator.h"
#include "ibamr/LSInitStrategy.h"

#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/PoissonSolver.h"

namespace ADS
{
void callback_fcn(double current_time, double new_time, int cycle_num, void* ctx);

class SBAdvDiffIntegrator : public LSAdvDiffIntegrator
{
public:
    SBAdvDiffIntegrator(const std::string& object_name,
                        SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                        SAMRAI::tbox::Pointer<IBAMR::IBHierarchyIntegrator> ib_integrator = nullptr,
                        bool register_for_restart = true);

    ~SBAdvDiffIntegrator() = default;

    /*!
     * Register a cell-centered quantity to be advected and diffused by the
     * hierarchy integrator. Can optionally turn off outputting the quantity.
     *
     * Data management for the registered quantity will be handled by the
     * hierarchy integrator.
     */

    static void callbackIntegrateHierarchy(double current_time, double neW_time, int cycle_num, void* ctx);

    void registerLevelSetSBDataManager(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                       std::shared_ptr<SBSurfaceFluidCouplingManager> sb_data_manager);

    void registerLevelSetCutCellMapping(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                                        std::shared_ptr<CutCellMeshMapping> cut_cell_mesh_mapping);

    void registerSBIntegrator(SAMRAI::tbox::Pointer<SBIntegrator> sb_integrator,
                              SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * Returns the number of cycles to perform for the present time step.
     */
    int getNumberOfCycles() const override;

    /*!
     * Synchronously advance each level in the hierarchy over the given time
     * increment.
     */
    void integrateHierarchySpecialized(double current_time, double new_time, int cycle_num = 0) override;

protected:

    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>,
             std::shared_ptr<SBSurfaceFluidCouplingManager>>
        d_ls_sb_data_manager_map;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>, std::shared_ptr<CutCellMeshMapping>>
        d_ls_cut_cell_mapping_map;
    std::map<SAMRAI::tbox::Pointer<SBIntegrator>, SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>>>
        d_sb_integrator_ls_map;

private:
    bool d_use_rbfs = false;
    unsigned int d_rbf_stencil_size = 2;
    Reconstruct::RBFPolyOrder d_rbf_poly_order = Reconstruct::RBFPolyOrder::UNKNOWN_ORDER;
    bool d_used_with_ib = false;
}; // Class SBAdvDiffIntegrator

} // namespace ADS

#endif
