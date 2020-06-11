#ifndef included_LS_LSCutCellBoundaryConditions
#define included_LS_LSCutCellBoundaryConditions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "LS/LSFindCellVolume.h"
#include "LS/SetLSValue.h"

#include "CellVariable.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableContext.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include "libmesh/elem.h"
#include "libmesh/utility.h"

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace LS
{
class LSCutCellBoundaryConditions
{
public:
    LSCutCellBoundaryConditions(const std::string& object_name);

    ~LSCutCellBoundaryConditions() = default;

    /*!
     * \brief Deleted default constructor.
     */
    LSCutCellBoundaryConditions() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    LSCutCellBoundaryConditions(const LSCutCellBoundaryConditions& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    LSCutCellBoundaryConditions& operator=(const LSCutCellBoundaryConditions& that) = delete;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx) = 0;

    void setLSData(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                   int ls_idx,
                   SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> vol_var,
                   int vol_idx,
                   SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> area_var,
                   int area_idx);

private:
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_ls_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_vol_var, d_area_var;

    int d_ls_idx = IBTK::invalid_index, d_vol_idx = IBTK::invalid_index, d_area_idx = IBTK::invalid_index;
};

} // namespace LS

#endif
