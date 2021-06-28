#ifndef included_InsideBoundaryConditions
#define included_InsideBoundaryConditions

#include "CCAD/LSCutCellBoundaryConditions.h"

#include "ibamr/AdvDiffHierarchyIntegrator.h"

class InsideBoundaryConditions : public CCAD::LSCutCellBoundaryConditions
{
public:
    InsideBoundaryConditions(const std::string& object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> out_var,
                             SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> integrator);

    ~InsideBoundaryConditions();

    /*!
     * \brief Deleted default constructor.
     */
    InsideBoundaryConditions() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    InsideBoundaryConditions(const InsideBoundaryConditions& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    InsideBoundaryConditions& operator=(const InsideBoundaryConditions& that) = delete;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                double time);

private:
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_out_var;
    SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> d_integrator;

    double d_k1 = std::numeric_limits<double>::quiet_NaN();
};
#endif
