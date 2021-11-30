#ifndef included_RadialBoundaryCond
#define included_RadialBoundaryCond

#include "ADS/LSCutCellBoundaryConditions.h"

class RadialBoundaryCond : public ADS::LSCutCellBoundaryConditions
{
public:
    RadialBoundaryCond(const std::string& object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    ~RadialBoundaryCond() = default;

    /*!
     * \brief Deleted default constructor.
     */
    RadialBoundaryCond() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    RadialBoundaryCond(const RadialBoundaryCond& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    RadialBoundaryCond& operator=(const RadialBoundaryCond& that) = delete;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                double time);

private:
    double d_a = std::numeric_limits<double>::quiet_NaN();
    double d_D_coef = std::numeric_limits<double>::quiet_NaN();
    double d_R = std::numeric_limits<double>::quiet_NaN();
};
#endif
