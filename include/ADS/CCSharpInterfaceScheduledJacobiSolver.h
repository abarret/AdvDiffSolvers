#ifndef included_ADS_CCSharpInterfaceScheduledJacobiSolver
#define included_ADS_CCSharpInterfaceScheduledJacobiSolver

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/SharpInterfaceGhostFill.h"
#include "ADS/sharp_interface_utilities.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LinearSolver.h"
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
 * \brief Class CCSharpInterfaceScheduledJacobiSolver is an implementation of the Scheduled Relaxed Jacobi (SRJ) method
 * to solver the sharp interface Poisson Equation.
 *
 * The input database is searched for the following items
 * - num_levels: integer consisting of number of levels of relaxation
 * - Q: array of num_levels integers that are the number of iterations of relaxation
 * - w: array of num_levels doubles that are the relaxation weights
 *
 * Optimal weights and iterations depend on the grid spacing. We suggest the following for num_levels = 4. Optimal
 * weights for other levels can be found here: https://doi.org/10.1016/j.jcp.2014.06.010
 *
 *
 * N = 16:
 *   Q = {1, 2, 8, 20}
 *   w = {80.154, 17.217, 2.6201, 0.62230}
 * N = 32:
 *   Q = {1, 3, 14, 46}
 *   w = {289.46, 40.791, 4.0877, 0.66277}
 * N = 64:
 *   Q = {1, 5, 26, 114}
 *   w = {1029.4, 95.007, 6.3913, 0.70513}
 * N = 128:
 *   Q = {1, 7, 50, 285}
 *   w = {3596.4, 217.80, 9.9666, 0.74755}
 * N = 256:
 *   Q = {1, 9, 86, 664}
 *   w = {12329, 492.05, 15.444, 0.78831}
 * N = 512:
 *   Q = {1, 12, 155, 1650}
 *   w = {41459, 1096.3, 23.730, 0.82597}
 *
 * Note that weights for smaller values of N are still valid for larger values of N.
 */
class CCSharpInterfaceScheduledJacobiSolver : public IBTK::LinearSolver
{
public:
    /*!
     * \brief Constructor for class CCSharpInterfaceScheduledJacobiSolver initializes the operator
     * coefficients and boundary conditions to default values.
     */
    CCSharpInterfaceScheduledJacobiSolver(const std::string& object_name,
                                          SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                                          SharpInterfaceGhostFill& ghost_fill,
                                          std::function<double(const IBTK::VectorNd&)> bdry_fcn,
                                          bool homogeneous_bc = true);

    /*!
     * \brief Destructor.
     */
    ~CCSharpInterfaceScheduledJacobiSolver();

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
    bool solveSystem(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
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
    void initializeSolverState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& in,
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
    void deallocateSolverState() override;

    //\}

private:
    /*!
     * \brief Computes one Jacobi iteration with the specified relaxation parameter w.
     */
    void relaxOnHierarchy(int x_idx,
                          int y_idx,
                          double w,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          int coarsest_ln,
                          int finest_ln);

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

    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_bdry_coefs;

    std::function<double(const IBTK::VectorNd&)> d_bdry_fcn;

    std::vector<std::pair<double, int>> d_schedule;
    int d_max_iterations = -1;
};
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_CCSharpInterfaceScheduledJacobiSolver
