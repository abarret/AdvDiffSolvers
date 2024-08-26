#ifndef included_ADS_CCSharpInterfaceLaplaceOperator
#define included_ADS_CCSharpInterfaceLaplaceOperator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/SharpInterfaceGhostFill.h"
#include "ADS/sharp_interface_utilities.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "CellVariable.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableContext.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
namespace sharp_interface
{
/*!
 * \brief Class CCSharpInterfaceLaplaceOperator is a concrete LaplaceOperator which implements
 * a globally second-order accurate cell-centered finite difference
 * discretization of a scalar elliptic operator of the form \f$ L = C I + \nabla
 * \cdot D \nabla\f$.
 */
class CCSharpInterfaceLaplaceOperator : public IBTK::LaplaceOperator
{
public:
    /*!
     * \brief Constructor for class CCSharpInterfaceLaplaceOperator initializes the operator
     * coefficients and boundary conditions to default values.
     */
    CCSharpInterfaceLaplaceOperator(const std::string& object_name,
                                    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                                    SharpInterfaceGhostFill& ghost_fill,
                                    std::function<double(const IBTK::VectorNd&)> bdry_fcn,
                                    bool homogeneous_bc = true);

    /*!
     * \brief Destructor.
     */
    ~CCSharpInterfaceLaplaceOperator();

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

    //\}

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    CCSharpInterfaceLaplaceOperator() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    CCSharpInterfaceLaplaceOperator(const CCSharpInterfaceLaplaceOperator& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    CCSharpInterfaceLaplaceOperator& operator=(const CCSharpInterfaceLaplaceOperator& that) = delete;

    /*!
     * \brief Computes action of Helmholtz operator.
     */
    void applyOnLevel(const int Q_idx,
                      const int R_idx,
                      const int ln,
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    // Operator parameters.
    int d_ncomp = 0;

    // Cached communications operators.
    std::vector<IBTK::HierarchyGhostCellInterpolation::InterpolationTransactionComponent> d_transaction_comps;
    SAMRAI::tbox::Pointer<IBTK::HierarchyGhostCellInterpolation> d_hier_bdry_fill;

    // Scratch data.
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_x, d_b;

    // Hierarchy configuration.
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    SharpInterfaceGhostFill* d_ghost_fill;

    std::function<double(const IBTK::VectorNd&)> d_bdry_fcn;
};
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_CCSharpInterfaceLaplaceOperator
