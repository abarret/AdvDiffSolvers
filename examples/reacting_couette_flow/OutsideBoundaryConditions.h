#ifndef included_OutsideBoundaryConditions
#define included_OutsideBoundaryConditions

#include "LS/LSCutCellBoundaryConditions.h"

class OutsideBoundaryConditions : public LS::LSCutCellBoundaryConditions
{
public:
    OutsideBoundaryConditions(const std::string& object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> in_var,
                              SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> integrator);

    ~OutsideBoundaryConditions();

    /*!
     * \brief Deleted default constructor.
     */
    OutsideBoundaryConditions() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    OutsideBoundaryConditions(const OutsideBoundaryConditions& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    OutsideBoundaryConditions& operator=(const OutsideBoundaryConditions& that) = delete;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                double time);

private:
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_in_var;
    SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> d_integrator;

    double d_k1 = std::numeric_limits<double>::quiet_NaN();
};
#endif
