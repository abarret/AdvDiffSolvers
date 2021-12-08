/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_PETScPoissonLinearSolver
#define included_ADS_PETScPoissonLinearSolver

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ibtk/PoissonSolver.h"

#include "IntVector.h"
#include "MultiblockDataTranslator.h"
#include "SAMRAIVectorReal.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"
#include <ADS/FEMeshPartitioner.h>
#include <ADS/PETScAugmentedLinearOperator.h>

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
 * \brief Class PETScPoissonSolver provides an interface to PETSc to solve a Poisson problem on a complex domain. This
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
class PETScPoissonSolver : public IBTK::PoissonSolver
{
public:
    /*!
     * \brief Constructor for a concrete KrylovLinearSolver that employs the
     * PETSc KSP solver framework.
     */
    PETScPoissonSolver(std::string object_name,
                       SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                       std::string default_options_prefix,
                       MPI_Comm petsc_comm = PETSC_COMM_WORLD);

    /*!
     * \brief Destructor.
     */
    ~PETScPoissonSolver();

    void setLagrangianMeshData(std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner, std::string sys_name);
    void setEulerianMeshData(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

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

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    PETScPoissonSolver() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    PETScPoissonSolver(const PETScPoissonSolver& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    PETScPoissonSolver& operator=(const PETScPoissonSolver& that) = delete;

    /*!
     * \brief Common routine used by all class constructors.
     */
    void commonConstructor();

    void copyDataToPetsc(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                         const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& b);

    void copyDataFromPetsc(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                           SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& b);

    /*!
     * \brief Set up a pairing between SAMRAI data ordering to PETSc data ordering.
     * We count the number of DOFs for each processor, assign each cell index a PETSc index, and fill in ghost cells.
     */
    void setupEulDOFs(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x);
    /*!
     * \brief Set up a pairing between libMesh data ordering to PETSc data ordering.
     * We count the number of DOFs for each processor and assign each node a PETSc index.
     */
    void setupLagDOFs();

    /*!
     * \brief Set up the Mat and Vec objects. We preallocate the matrix and fill in the weights. We only preallocate the
     * Vec components.
     */
    void setupMatrixAndVec();

    bool d_reinitializing_solver = false;

    // KSP data
    std::string d_ksp_type;
    bool d_initial_guess_nonzero = true;
    std::string d_options_prefix;
    MPI_Comm d_petsc_comm;

    // PETSc indexing data

    // Eulerian data
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    std::vector<int> d_eul_dofs_per_proc;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, int>> d_eul_idx_var;
    int d_eul_idx_idx = IBTK::invalid_index;

    // Information of the Lagrangian mesh and it's degrees of freedom.
    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;
    std::string d_sys_name;
    std::vector<int> d_lag_dofs_per_proc;

    // Data structures for entire system, Eulerian + augmented dofs.
    KSP d_petsc_ksp = nullptr;
    Mat d_petsc_mat = nullptr;
    Vec d_petsc_x = nullptr, d_petsc_b = nullptr;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_PETScPoissonSolver
