#ifndef included_LS_LSCutCellLaplaceOperator
#define included_LS_LSCutCellLaplaceOperator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "LS/LSCutCellBoundaryConditions.h"
#include "LS/LSFindCellVolume.h"
#include "LS/SetLSValue.h"
#include "LS/utility_functions.h"

#include "CellVariable.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableContext.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace LS
{
/*!
 * \brief Class LSCutCellLaplaceOperator is a concrete LaplaceOperator which implements
 * a globally second-order accurate cell-centered finite difference
 * discretization of a scalar elliptic operator of the form \f$ L = C I + \nabla
 * \cdot D \nabla\f$.
 */
class LSCutCellLaplaceOperator : public IBTK::LaplaceOperator
{
public:
    /*!
     * \brief Constructor for class LSCutCellLaplaceOperator initializes the operator
     * coefficients and boundary conditions to default values.
     */
    LSCutCellLaplaceOperator(const std::string& object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             bool homogeneous_bc = true);

    /*!
     * \brief Destructor.
     */
    ~LSCutCellLaplaceOperator();

    /*!
     * \name Linear operator functionality.
     */
    //\{

    /*!
     * \brief Compute y=Ax.
     *
     * Before calling this function, the form of the vectors x and y should be
     * set properly by the user on all patch interiors on the range of levels
     * covered by the operator.  All data in these vectors should be allocated.
     * Thus, the user is responsible for managing the storage for the vectors.
     *
     * Conditions on arguments:
     * - vectors must have same hierarchy
     * - vectors must have same variables (except that x \em must
     * have enough ghost cells for computation of Ax).
     *
     * \note In general, the vectors x and y \em cannot be the same.
     *
     * Upon return from this function, the y vector will contain the result of
     * the application of A to x.
     *
     * initializeOperatorState must be called prior to any calls to
     * applyOperator.
     *
     * \see initializeOperatorState
     *
     * \param x input
     * \param y output: y=Ax
     */
    void apply(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
               SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& y) override;

    /*!
     * \brief Compute hierarchy-dependent data required for computing y=Ax (and
     * y=A'x).
     *
     * \param in input vector
     * \param out output vector
     *
     * \see KrylovLinearSolver::initializeSolverState
     */
    void initializeOperatorState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& in,
                                 const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& out) override;

    /*!
     * \brief Remove all hierarchy-dependent data computed by
     * initializeOperatorState().
     *
     * Remove all hierarchy-dependent data set by initializeOperatorState().  It
     * is safe to call deallocateOperatorState() even if the state is already
     * deallocated.
     *
     * \see initializeOperatorState
     * \see KrylovLinearSolver::deallocateSolverState
     */
    void deallocateOperatorState() override;

    void cacheLeastSquaresData();

    inline void setLSIndices(int ls_idx,
                             Pointer<NodeVariable<NDIM, double>> ls_var,
                             int vol_idx,
                             Pointer<CellVariable<NDIM, double>> vol_var,
                             int area_idx,
                             Pointer<CellVariable<NDIM, double>> area_var)
    {
        d_ls_idx = ls_idx;
        d_ls_var = ls_var;
        d_vol_idx = vol_idx;
        d_vol_var = vol_var;
        d_area_idx = area_idx;
        d_area_var = area_var;
        d_update_weights = true;
    }

    inline void setBoundaryConditionOperator(SAMRAI::tbox::Pointer<LSCutCellBoundaryConditions> bdry_conds)
    {
        d_update_weights = true;
        d_bdry_conds = bdry_conds;
    }

    inline void setTimeStepType(DiffusionTimeIntegrationMethod ts_type)
    {
        d_ts_type = ts_type;
    }

    //\}

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    LSCutCellLaplaceOperator() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    LSCutCellLaplaceOperator(const LSCutCellLaplaceOperator& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    LSCutCellLaplaceOperator& operator=(const LSCutCellLaplaceOperator& that) = delete;

    /*!
     * \brief Computes action of Helmholtz operator.
     */
    void computeHelmholtzAction(const CellData<NDIM, double>& Q_data,
                                CellData<NDIM, double>& R_data,
                                const Patch<NDIM>& patch);

    void extrapolateToCellCenters(int Q_idx, int R_idx);

    inline double weight(const double r)
    {
        return std::exp(-r * r);
    }

    // Operator parameters.
    int d_ncomp = 0;

    // Cached communications operators.
    std::vector<HierarchyGhostCellInterpolation::InterpolationTransactionComponent> d_transaction_comps;
    SAMRAI::tbox::Pointer<HierarchyGhostCellInterpolation> d_hier_bdry_fill;

    // Scratch data.
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_x, d_b;

    // Hierarchy configuration.
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    // Area and volume data
    SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> d_vol_var, d_area_var;
    int d_vol_idx = IBTK::invalid_index, d_area_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> d_ls_var;
    int d_ls_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<LSCutCellBoundaryConditions> d_bdry_conds;

    bool d_robin_bdry = false;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_var;
    int d_Q_scr_idx = IBTK::invalid_index;

    std::vector<std::map<PatchIndexPair, Eigen::FullPivHouseholderQR<MatrixXd>>> d_qr_matrix_vec;
    bool d_update_weights = true;
    bool d_cache_bdry;

    DiffusionTimeIntegrationMethod d_ts_type = DiffusionTimeIntegrationMethod::UNKNOWN_METHOD;
};
} // namespace LS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_LS_LSCutCellLaplaceOperator
