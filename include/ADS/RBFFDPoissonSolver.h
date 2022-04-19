/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_RBFPoissonSolver
#define included_ADS_RBFPoissonSolver

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include <ADS/ConditionCounter.h>
#include <ADS/FEMeshPartitioner.h>
#include <ADS/GhostPoints.h>
#include <ADS/GlobalIndexing.h>
#include <ADS/PETScAugmentedLinearOperator.h>
#include <ADS/RBFFDWeightsCache.h>

#include "ibtk/PoissonSolver.h"

#include "IntVector.h"
#include "MultiblockDataTranslator.h"
#include "SAMRAIVectorReal.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"

#include "petscksp.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscvec.h"

#include <mpi.h>

#include <iosfwd>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class RBFPoissonSolver provides an interface to PETSc to solve a Poisson problem on a complex domain. This
 * class generates a matrix corresponding to an RBF-FD discretization of the Poisson operator.
 *
 * This class has both Eulerian and Lagrangian degrees of freedom. On initialization of the solver, a PETSc ordering of
 * the two descriptions is created. Data from the two descriptions is copied to the PETSc representation on calls to
 * solveSystem(). Data is then copied back into the two separate descriptions.
 *
 * PETSc is developed in the Mathematics and Computer Science (MCS) Division at
 * Argonne National Laboratory (ANL).  For more information about PETSc, see <A
 * HREF="http://www.mcs.anl.gov/petsc">http://www.mcs.anl.gov/petsc</A>.
 */
class RBFFDPoissonSolver : public IBTK::PoissonSolver
{
public:
    /*!
     * \brief Constructor for a concrete KrylovLinearSolver that employs the
     * PETSc KSP solver framework.
     */
    RBFFDPoissonSolver(std::string object_name,
                       SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                       std::string sys_x_name,
                       std::string sys_b_name,
                       std::string default_options_prefix,
                       MPI_Comm petsc_comm = PETSC_COMM_WORLD);

    /*!
     * \brief Destructor.
     */
    ~RBFFDPoissonSolver();

    /*!
     * \brief Set the level set index.
     */
    inline void setLSIdx(const int ls_idx)
    {
        d_ls_idx = ls_idx;
    }

    /*!
     * \brief Set the KSP type.
     */
    void setKSPType(const std::string& ksp_type);

    /*!
     * \brief Set the options prefix used by this PETSc solver object.
     */
    void setOptionsPrefix(const std::string& options_prefix);

    /*!
     * \name Functions to access the underlying PETSc objects.
     */
    //\{

    /*!
     * \brief Get the PETSc KSP object.
     */
    const KSP& getPETScKSP() const;

    /*!
     * \brief Get the augmented solution vector.
     */
    const Vec& getAugmentedVec() const;
    //\}

    /*!
     * \name PoissonSolver solver functionality.
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

    inline Mat& getMat()
    {
        return d_petsc_mat;
    }
    inline Vec& getRHS()
    {
        return d_petsc_b;
    }
    inline Vec& getX()
    {
        return d_petsc_x;
    }

    inline const std::shared_ptr<GhostPoints>& getGhostPoints() const
    {
        return d_ghost_pts;
    }

    inline const std::shared_ptr<GlobalIndexing>& getGlobalIndexing() const
    {
        return d_index_ptr;
    }

    inline const std::unique_ptr<ConditionCounter>& getBulkConditionMap() const
    {
        return d_cc_bulk;
    }

    inline const std::unique_ptr<ConditionCounter>& getBdryConditionMap() const
    {
        return d_cc_bdry;
    }

    inline const std::unique_ptr<FDWeightsCache>& getBulkWeights() const
    {
        return d_bulk_weights;
    }

    inline const std::unique_ptr<FDWeightsCache>& getBdryWeights() const
    {
        return d_bdry_weights;
    }

    void writeMatToFile(const std::string& filename);

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    RBFFDPoissonSolver() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    RBFFDPoissonSolver(const RBFFDPoissonSolver& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    RBFFDPoissonSolver& operator=(const RBFFDPoissonSolver& that) = delete;

    /*!
     * \brief Common routine used by all class constructors.
     */
    void commonConstructor(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Set up the Mat and Vec objects. We preallocate the matrix and fill in the weights. We only preallocate the
     * Vec components.
     */
    void setupMatrixAndVec();

    void findFDWeights();

    bool d_reinitializing_solver = false;

    // KSP data
    std::string d_ksp_type;
    bool d_initial_guess_nonzero = true;
    std::string d_options_prefix;
    MPI_Comm d_petsc_comm;

    // PETSc indexing data

    // Eulerian data
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_ls_idx = IBTK::invalid_index;

    // Information of the Lagrangian mesh and it's degrees of freedom.
    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;
    std::string d_sys_x_name, d_sys_b_name, d_sys_n_name;

    // Data structures for entire system, Eulerian + augmented dofs.
    KSP d_petsc_ksp = nullptr;
    Mat d_petsc_mat = nullptr;
    Vec d_petsc_x = nullptr, d_petsc_b = nullptr;

    // DOFs and FD weights
    std::shared_ptr<GhostPoints> d_ghost_pts;
    std::shared_ptr<GlobalIndexing> d_index_ptr;
    std::unique_ptr<FDWeightsCache> d_bulk_weights, d_bdry_weights;
    std::unique_ptr<ConditionCounter> d_cc_bulk, d_cc_bdry;

    // Functions for RBF-FD
    std::function<double(double)> d_rbf;
    std::function<double(const FDPoint&, const FDPoint&, void*)> d_lap_rbf, d_bdry_rbf;
    std::function<IBTK::VectorXd(const std::vector<FDPoint>&, int, double, const FDPoint&, void*)> d_lap_polys,
        d_bdry_polys;

    int d_stencil_size = -1;
    int d_poly_degree = -1;
    double d_eps = std::numeric_limits<double>::quiet_NaN();
    double d_switch_to_rbffd_dist = std::numeric_limits<double>::quiet_NaN();

    double d_A = std::numeric_limits<double>::quiet_NaN(), d_B = std::numeric_limits<double>::quiet_NaN();
    double d_C = std::numeric_limits<double>::quiet_NaN(), d_D = std::numeric_limits<double>::quiet_NaN();
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_RBFFDPoissonSolver
