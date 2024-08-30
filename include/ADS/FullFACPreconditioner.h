#ifndef included_ADS_FullFACPreconditioner
#define included_ADS_FullFACPreconditioner

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ibtk/LinearSolver.h"
#include "ibtk/ibtk_enums.h"
#include <ibtk/FACPreconditioner.h>
#include <ibtk/FACPreconditionerStrategy.h>

#include "Box.h"
#include "GriddingAlgorithm.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "Variable.h"
#include "tbox/Pointer.h"

#include <string>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class FACPreconditioner is a concrete LinearSolver for implementing
 * FAC (multilevel multigrid) preconditioners.
 *
 * This class is similar to the SAMRAI class SAMRAI::solv::FACPreconditioner,
 * except that this class has been optimized for the case in which the solver is
 * to be used as a single-pass preconditioner, especially for the case in which
 * pre-smoothing sweeps are not needed.  This class is not suitable for use as a
 * stand-alone solver; rather, it is intended to be used in conjunction with an
 * iterative Krylov method.
 *
 * Sample parameters for initialization from database (and their default
 * values): \verbatim

 cycle_type = "V_CYCLE"  // see setMGCycleType()
 num_pre_sweeps = 0      // see setNumPreSmoothingSweeps()
 num_post_sweeps = 2     // see setNumPostSmoothingSweeps()
 enable_logging = FALSE  // see setLoggingEnabled()
 \endverbatim
*/
class FullFACPreconditioner : public IBTK::FACPreconditioner
{
public:
    /*!
     * Constructor.
     */
    FullFACPreconditioner(std::string object_name,
                          SAMRAI::tbox::Pointer<IBTK::FACPreconditionerStrategy> fac_strategy,
                          SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                          const std::string& default_options_prefix);

    /*!
     * Destructor.
     */
    ~FullFACPreconditioner();

    /*!
     * \name Linear solver functionality.
     */
    //\{
    /*!
     * \brief Solve the linear system of equations \f$Ax=b\f$ for \f$x\f$.
     *
     * Before calling solveSystem(), the form of the solution \a x and
     * right-hand-side \a b vectors must be set properly by the user on all
     * patch interiors on the specified range of levels in the patch hierarchy.
     * The user is responsible for all data management for the quantities
     * associated with the solution and right-hand-side vectors.  In particular,
     * patch data in these vectors must be allocated prior to calling this
     * method.
     *
     * \param x solution vector
     * \param b right-hand-side vector
     *
     * <b>Conditions on Parameters:</b>
     * - vectors \a x and \a b must have same patch hierarchy
     * - vectors \a x and \a b must have same structure, depth, etc.
     *
     * \note The vector arguments for solveSystem() need not match those for
     * initializeSolverState().  However, there must be a certain degree of
     * similarity, including:\par
     * - hierarchy configuration (hierarchy pointer and range of levels)
     * - number, type and alignment of vector component data
     * - ghost cell widths of data in the solution \a x and right-hand-side \a b
     *   vectors
     *
     * \note The solver need not be initialized prior to calling solveSystem();
     * however, see initializeSolverState() and deallocateSolverState() for
     * opportunities to save overhead when performing multiple consecutive
     * solves.
     *
     * \see initializeSolverState
     * \see deallocateSolverState
     *
     * \return \p true if the solver converged to the specified tolerances, \p
     * false otherwise
     */
    bool solveSystem(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                     SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& b) override;

    /*!
     * \brief Compute hierarchy dependent data required for solving \f$Ax=b\f$.
     *
     * By default, the solveSystem() method computes some required hierarchy
     * dependent data before solving and removes that data after the solve.  For
     * multiple solves that use the same hierarchy configuration, it is more
     * efficient to:
     *
     * -# initialize the hierarchy-dependent data required by the solver via
     *    initializeSolverState(),
     * -# solve the system one or more times via solveSystem(), and
     * -# remove the hierarchy-dependent data via deallocateSolverState().
     *
     * Note that it is generally necessary to reinitialize the solver state when
     * the hierarchy configuration changes.
     *
     * When linear operator or preconditioner objects have been registered with
     * this class via setOperator() and setPreconditioner(), they are also
     * initialized by this member function.
     *
     * \param x solution vector
     * \param b right-hand-side vector
     *
     * <b>Conditions on Parameters:</b>
     * - vectors \a x and \a b must have same patch hierarchy
     * - vectors \a x and \a b must have same structure, depth, etc.
     *
     * \note The vector arguments for solveSystem() need not match those for
     * initializeSolverState().  However, there must be a certain degree of
     * similarity, including:\par
     * - hierarchy configuration (hierarchy pointer and range of levels)
     * - number, type and alignment of vector component data
     * - ghost cell widths of data in the solution \a x and right-hand-side \a b
     *   vectors
     *
     * \note It is safe to call initializeSolverState() when the state is
     * already initialized.  In this case, the solver state is first deallocated
     * and then reinitialized.
     *
     * \see deallocateSolverState
     */
    void initializeSolverState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                               const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& b) override;

    /*!
     * \brief Remove all hierarchy dependent data allocated by
     * initializeSolverState().
     *
     * When linear operator or preconditioner objects have been registered with
     * this class via setOperator() and setPreconditioner(), they are also
     * deallocated by this member function.
     *
     * \note It is safe to call deallocateSolverState() when the solver state is
     * already deallocated.
     *
     * \see initializeSolverState
     */
    void deallocateSolverState() override;

    //\}

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> getDenseHierarchy()
    {
        return d_dense_hierarchy;
    }

    /*!
     * Allocate and transfer data from the base hierarchy given to the object via initializeOperatorState() to the dense
     * hierarchy owned by this object. If specified, this will also deallocate data from the dense hierarchy when
     * deallocateSolverState() is called.
     */
    //\{
    void transferToDense(int idx, bool deallocate_data = true);
    void transferToDense(std::set<int> idxs, bool deallocate_data = true);
    //\}

protected:
private:
    void generateDenseHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> base_hierarchy);

    void transferToDense(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& base_x,
                         const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& dense_x);

    void transferToBase(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& base_x,
                        const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& dense_x);

    int d_multigrid_max_levels = -1;
    std::string d_coarsening_operator;

    // These are used to do the actual multigrid steps
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_dense_hierarchy;
    SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm<NDIM>> d_grid_alg;

    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_u, d_f;

    std::map<SAMRAI::hier::Variable<NDIM>*, SAMRAI::tbox::Pointer<SAMRAI::math::HierarchyDataOpsReal<NDIM, double>>>
        d_var_op_map;

    std::set<int> d_allocated_idxs;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_IBTK_FACPreconditioner
