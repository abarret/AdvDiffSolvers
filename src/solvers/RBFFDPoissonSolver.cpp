/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/PolynomialBasis.h"
#include "ADS/RBFFDPoissonSolver.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"
#include <ADS/KDTree.h>
#include <ADS/reconstructions.h>
#include <ADS/solver_utilities.h>

#include "ibtk/GeneralSolver.h"
#include "ibtk/IBTK_CHKERRQ.h"
#include "ibtk/IBTK_MPI.h"
#include "ibtk/KrylovLinearSolver.h"
#include "ibtk/LinearOperator.h"
#include "ibtk/PETScMatLOWrapper.h"
#include "ibtk/PETScPCLSWrapper.h"
#include "ibtk/PETScSAMRAIVectorReal.h"
#include "ibtk/solver_utilities.h"

#include "Box.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"
#include "tbox/Timer.h"
#include "tbox/Utilities.h"

#include "petscksp.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscvec.h"
#include <petsclog.h>

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Timers.
static Timer* t_solve_system;
static Timer* t_initialize_solver_state;
static Timer* t_deallocate_solver_state;
static Timer* t_setup_matrix_and_vectors;
static Timer* t_find_rbffd_weights;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

RBFFDPoissonSolver::RBFFDPoissonSolver(std::string object_name,
                                       Pointer<Database> input_db,
                                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                                       std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                                       std::string sys_x_name,
                                       std::string sys_b_name,
                                       std::string default_options_prefix,
                                       MPI_Comm petsc_comm)
    : d_ksp_type(KSPGMRES),
      d_options_prefix(std::move(default_options_prefix)),
      d_petsc_comm(petsc_comm),
      d_hierarchy(hierarchy),
      d_fe_mesh_partitioner(std::move(fe_mesh_partitioner)),
      d_sys_x_name(std::move(sys_x_name)),
      d_sys_b_name(std::move(sys_b_name))
{
    // Setup default values.
    GeneralSolver::init(std::move(object_name), /*homogeneous_bc*/ false);
    d_max_iterations = 10000;
    d_abs_residual_tol = 1.0e-50;
    d_rel_residual_tol = 1.0e-5;
    d_enable_logging = true;

    d_A = input_db->getDouble("a");
    d_B = input_db->getDouble("b");
    d_C = input_db->getDouble("c");
    d_D = input_db->getDouble("d");

    const double& A = d_A;
    const double& B = d_B;
    const double& C = d_C;
    const double& D = d_D;

    d_bulk_weights = std::make_unique<FDWeightsCache>(d_object_name + "::bulk_wgts");
    d_bdry_weights = std::make_unique<FDWeightsCache>(d_object_name + "::bdry_wgts");
    // Make a function for the weights
    d_rbf = [](const double r) -> double { return PolynomialBasis::pow(r, 5); };
    d_lap_rbf = [&C, &D](const FDPoint& x, const FDPoint& x0, void*) -> double {
        double r = 1.0;
        for (int d = 0; d < NDIM; ++d) r += (x(d) - x0(d)) * (x(d) - x0(d));
        r = std::sqrt(r);
#if (NDIM == 2)
        return C * PolynomialBasis::pow(r, 5) + D * 25.0 * PolynomialBasis::pow(r, 4);
#endif
#if (NDIM == 3)
        return C * PolynomialBasis::pow(r, 5) + D * 30.0 * PolynomialBasis::pow(r, 4);
#endif
    };

    d_lap_polys = [&C,
                   &D](const std::vector<FDPoint>& vec, int degree, double ds, const FDPoint& shft, void*) -> VectorXd {
        VectorXd ret = (C * PolynomialBasis::formMonomials(vec, degree, ds, shft) +
                        D * PolynomialBasis::laplacianMonomials(vec, degree, ds, shft))
                           .transpose();
        return ret;
    };

    d_bdry_rbf = [&A, &B](const FDPoint& x, const FDPoint& x0, void* ctx) -> double {
        const IBTK::VectorNd& n = *(static_cast<IBTK::VectorNd*>(ctx));
        double r = 1.0;
        for (int d = 0; d < NDIM; ++d) r += (x(d) - x0(d)) * (x(d) - x0(d));
        r = std::sqrt(r);
        return A * PolynomialBasis::pow(r, 5) + B * PolynomialBasis::pow(r, 3) * 5 * (x - x0).dot(n);
    };

    d_bdry_polys =
        [&A, &B](const std::vector<FDPoint>& vec, int degree, double ds, const FDPoint& shft, void* ctx) -> MatrixXd {
        const IBTK::VectorNd& n = *(static_cast<IBTK::VectorNd*>(ctx));
        return A * PolynomialBasis::formMonomials(vec, degree, ds, shft) +
               n(0) * PolynomialBasis::dPdxMonomials(vec, degree, ds, shft) +
               n(1) * PolynomialBasis::dPdyMonomials(vec, degree, ds, shft)
#if (NDIM == 3)
               + n(2) * PolynomialBasis::dPdzMonomials(vec, degree, ds, shft)
#endif
            ;
    };

    // Common constructor functionality.
    commonConstructor(input_db);
    return;
} // RBFPoissonSolver()

RBFFDPoissonSolver::~RBFFDPoissonSolver()
{
    if (d_is_initialized) deallocateSolverState();

    // Delete allocated PETSc solver components.
    int ierr;
    if (d_petsc_mat)
    {
        ierr = MatDestroy(&d_petsc_mat);
        IBTK_CHKERRQ(ierr);
        d_petsc_mat = nullptr;
    }
    return;
} // ~RBFPoissonSolver()

void
RBFFDPoissonSolver::setOptionsPrefix(const std::string& options_prefix)
{
    d_options_prefix = options_prefix;
    return;
} // setOptionsPrefix

const KSP&
RBFFDPoissonSolver::getPETScKSP() const
{
    return d_petsc_ksp;
} // getPETScKSP

bool
RBFFDPoissonSolver::solveSystem(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& b)
{
    ADS_TIMER_START(t_solve_system);
    int ierr;

    // Initialize the solver, when necessary.
    const bool deallocate_after_solve = !d_is_initialized;
    if (deallocate_after_solve) initializeSolverState(x, b);
#if !defined(NDEBUG)
    TBOX_ASSERT(d_petsc_ksp);
#endif

    // Indices should already be set up. Copy data to PETSc data structures
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    System& x_sys = eq_sys->get_system(d_sys_x_name);
    System& b_sys = eq_sys->get_system(d_sys_b_name);
    copy_data_to_petsc(d_petsc_x,
                       x,
                       d_hierarchy,
                       x_sys,
                       d_index_ptr->getEulerianMap(),
                       d_index_ptr->getLagrangianMap(),
                       d_index_ptr->getDofsPerProc());
    copy_data_to_petsc(d_petsc_b,
                       b,
                       d_hierarchy,
                       b_sys,
                       d_index_ptr->getEulerianMap(),
                       d_index_ptr->getLagrangianMap(),
                       d_index_ptr->getDofsPerProc());

    // Now solve the system
    ierr = KSPSolve(d_petsc_ksp, d_petsc_b, d_petsc_x);
    IBTK_CHKERRQ(ierr);

    // Get iterations count and residual norm.
    ierr = KSPGetIterationNumber(d_petsc_ksp, &d_current_iterations);
    IBTK_CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(d_petsc_ksp, &d_current_residual_norm);
    IBTK_CHKERRQ(ierr);

    // Determine the convergence reason.
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(d_petsc_ksp, &reason);
    IBTK_CHKERRQ(ierr);
    const bool converged = (static_cast<int>(reason) > 0);
    reportPETScKSPConvergedReason(d_object_name, reason, plog);
    plog << "converegd after " << d_current_iterations << " iterations with residual norm " << d_current_residual_norm
         << "\n";

    // Copy data back to other representations
    copy_data_from_petsc(d_petsc_x,
                         x,
                         d_hierarchy,
                         x_sys,
                         d_index_ptr->getEulerianMap(),
                         d_index_ptr->getLagrangianMap(),
                         d_index_ptr->getDofsPerProc());

    // Deallocate the solver, when necessary.
    if (deallocate_after_solve) deallocateSolverState();
    ADS_TIMER_STOP(t_solve_system);
    return converged;
} // solveSystem

void
RBFFDPoissonSolver::initializeSolverState(const SAMRAIVectorReal<NDIM, double>& x,
                                          const SAMRAIVectorReal<NDIM, double>& b)
{
    ADS_TIMER_START(t_initialize_solver_state);

#if !defined(NDEBUG)
    if (d_hierarchy->getNumberOfLevels() != 1)
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  patch hierarchy must have one level\n");
#endif

    int ierr;

// Rudimentary error checking.
#if !defined(NDEBUG)
    if (x.getNumberOfComponents() != b.getNumberOfComponents())
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  vectors must have the same number of components" << std::endl);
    }

    const Pointer<PatchHierarchy<NDIM>>& patch_hierarchy = x.getPatchHierarchy();
    if (patch_hierarchy != b.getPatchHierarchy())
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  vectors must have the same hierarchy" << std::endl);
    }

    const int coarsest_ln = x.getCoarsestLevelNumber();
    if (coarsest_ln < 0)
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  coarsest level number must not be negative" << std::endl);
    }
    if (coarsest_ln != b.getCoarsestLevelNumber())
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  vectors must have same coarsest level number" << std::endl);
    }

    const int finest_ln = x.getFinestLevelNumber();
    if (finest_ln < coarsest_ln)
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  finest level number must be >= coarsest level number" << std::endl);
    }
    if (finest_ln != b.getFinestLevelNumber())
    {
        TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                 << "  vectors must have same finest level number" << std::endl);
    }

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        if (!patch_hierarchy->getPatchLevel(ln))
        {
            TBOX_ERROR(d_object_name << "::initializeSolverState()\n"
                                     << "  hierarchy level " << ln << " does not exist" << std::endl);
        }
    }
#endif
    // Deallocate the solver state if the solver is already initialized.
    if (d_is_initialized)
    {
        d_reinitializing_solver = true;
        deallocateSolverState();
    }

    ierr = KSPCreate(d_petsc_comm, &d_petsc_ksp);
    IBTK_CHKERRQ(ierr);

    // Set the KSP options from the PETSc options database.
    if (d_options_prefix != "")
    {
        ierr = KSPSetOptionsPrefix(d_petsc_ksp, d_options_prefix.c_str());
        IBTK_CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(d_petsc_ksp);
    IBTK_CHKERRQ(ierr);

    // Reset the member state variables to correspond to the values used by the
    // KSP object.  (Command-line options always take precedence.)
    const char* ksp_type;
    ierr = KSPGetType(d_petsc_ksp, &ksp_type);
    IBTK_CHKERRQ(ierr);
    d_ksp_type = ksp_type;
    PetscBool initial_guess_nonzero;
    ierr = KSPGetInitialGuessNonzero(d_petsc_ksp, &initial_guess_nonzero);
    IBTK_CHKERRQ(ierr);
    d_initial_guess_nonzero = (initial_guess_nonzero == PETSC_TRUE);
    ierr = KSPGetTolerances(d_petsc_ksp, &d_rel_residual_tol, &d_abs_residual_tol, nullptr, &d_max_iterations);
    IBTK_CHKERRQ(ierr);

    // Setup DOF indexing
    d_ghost_pts->findNormals();
    d_ghost_pts->updateGhostNodeLocations(0.0);
    d_index_ptr->setupDOFs();
    // Get RBF-FD weights
    findFDWeights();

    // Now count conditions
    d_cc_bulk = std::unique_ptr<ConditionCounter>(
        new ConditionCounter(d_object_name + "::CC_Bulk", d_hierarchy, *d_bulk_weights));
    d_cc_bdry = std::unique_ptr<ConditionCounter>(
        new ConditionCounter(d_object_name + "::CC_Bdry", d_hierarchy, *d_bdry_weights));
    // Now create matrix
    setupMatrixAndVec();

    // Register matrix with KSP
    ierr = KSPSetOperators(d_petsc_ksp, d_petsc_mat, d_petsc_mat);

    // Indicate that the solver is initialized.
    d_reinitializing_solver = false;
    d_is_initialized = true;

    ADS_TIMER_STOP(t_initialize_solver_state);
    return;
} // initializeSolverState

void
RBFFDPoissonSolver::deallocateSolverState()
{
    if (!d_is_initialized) return;

    ADS_TIMER_START(t_deallocate_solver_state);
    int ierr;

    // Delete the solution and rhs vectors.
    ierr = VecDestroy(&d_petsc_x);
    IBTK_CHKERRQ(ierr);
    d_petsc_x = nullptr;

    ierr = VecDestroy(&d_petsc_b);
    IBTK_CHKERRQ(ierr);
    d_petsc_b = nullptr;

    // Destroy the matrix
    ierr = MatDestroy(&d_petsc_mat);
    IBTK_CHKERRQ(ierr);
    d_petsc_mat = nullptr;

    // Destroy the KSP solver.
    ierr = KSPDestroy(&d_petsc_ksp);
    IBTK_CHKERRQ(ierr);
    d_petsc_ksp = nullptr;

    // Indicate that the solver is NOT initialized.
    d_is_initialized = false;

    ADS_TIMER_STOP(t_deallocate_solver_state);
    return;
} // deallocateSolverState

/////////////////////////////// PRIVATE //////////////////////////////////////

void
RBFFDPoissonSolver::commonConstructor(Pointer<Database> input_db)
{
    // Get values from the input database.
    if (input_db)
    {
        if (input_db->keyExists("options_prefix")) d_options_prefix = input_db->getString("options_prefix");
        if (input_db->keyExists("max_iterations")) d_max_iterations = input_db->getInteger("max_iterations");
        if (input_db->keyExists("abs_residual_tol")) d_abs_residual_tol = input_db->getDouble("abs_residual_tol");
        if (input_db->keyExists("rel_residual_tol")) d_rel_residual_tol = input_db->getDouble("rel_residual_tol");
        if (input_db->keyExists("ksp_type")) d_ksp_type = input_db->getString("ksp_type");
        if (input_db->keyExists("initial_guess_nonzero"))
            d_initial_guess_nonzero = input_db->getBool("initial_guess_nonzero");
        if (input_db->keyExists("enable_logging")) d_enable_logging = input_db->getBool("enable_logging");
        d_poly_degree = input_db->getInteger("poly_degree");
        d_stencil_size = input_db->getInteger("stencil_size");
        d_eps = input_db->getDouble("eps");
        d_switch_to_rbffd_dist = input_db->getDouble("rbffd_dist");
    }

    d_ghost_pts =
        std::make_shared<GhostPoints>(d_object_name + "::GhostPoints", input_db, d_hierarchy, d_fe_mesh_partitioner);
    d_index_ptr = std::make_shared<GlobalIndexing>(d_object_name + "::GlobalIndexing",
                                                   d_hierarchy,
                                                   d_fe_mesh_partitioner,
                                                   d_sys_x_name,
                                                   d_ghost_pts,
                                                   d_poly_degree);

    // Setup Timers.
    IBTK_DO_ONCE(
        t_solve_system = TimerManager::getManager()->getTimer("ADS::RBFPoissonSolver::solveSystem()");
        t_initialize_solver_state =
            TimerManager::getManager()->getTimer("ADS::RBFPoissonSolver::initializeSolverState()");
        t_deallocate_solver_state =
            TimerManager::getManager()->getTimer("ADS::RBFPoissonSolver::deallocateSolverState()");
        t_setup_matrix_and_vectors = TimerManager::getManager()->getTimer("ADS::RBFPoissonSolver::setupMatrixAndVec()");
        t_find_rbffd_weights = TimerManager::getManager()->getTimer("ADS::RBFPoissonSolver::findRBFFDWeights()"););
    return;
} // common_ctor

void
RBFFDPoissonSolver::setupMatrixAndVec()
{
    ADS_TIMER_START(t_setup_matrix_and_vectors);
    // At this point we have the vectors and DOFs labeled. We need to create the matrix and vectors.
    // Determine the non-zero structure of the matrix.
    const std::vector<int>& dofs_per_proc = d_index_ptr->getDofsPerProc();
    const std::vector<unsigned int>& bulk_eqs_per_proc = d_cc_bulk->getNumConditionsPerProc();
    const std::vector<unsigned int>& bdry_eqs_per_proc = d_cc_bdry->getNumConditionsPerProc();
    const int mpi_rank = IBTK_MPI::getRank();

    const int dofs_local = dofs_per_proc[mpi_rank];
    const int dofs_lower = std::accumulate(dofs_per_proc.begin(), dofs_per_proc.begin() + mpi_rank, 0);
    const int dofs_upper = dofs_lower + dofs_local;
    const int dofs_total = std::accumulate(dofs_per_proc.begin(), dofs_per_proc.end(), 0);

    const int bulk_eqs_local = bulk_eqs_per_proc[mpi_rank];
    const int bulk_eqs_total = std::accumulate(bulk_eqs_per_proc.begin(), bulk_eqs_per_proc.end(), 0);

    const int bdry_eqs_local = bdry_eqs_per_proc[mpi_rank];
    const int bdry_eqs_total = std::accumulate(bdry_eqs_per_proc.begin(), bdry_eqs_per_proc.end(), 0);

    const int eqs_local = bulk_eqs_local + bdry_eqs_local;
    const int eqs_total = bdry_eqs_total + bulk_eqs_total;

    int ierr = VecCreateMPI(d_petsc_comm, dofs_local, dofs_total, &d_petsc_x);
    IBTK_CHKERRQ(ierr);
    ierr = VecCreateMPI(d_petsc_comm, eqs_local, eqs_total, &d_petsc_b);
    IBTK_CHKERRQ(ierr);

    // We'll need the DofMap
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const System& sys = eq_sys->get_system(d_sys_x_name);
    const DofMap& dof_map = sys.get_dof_map();

    // d_nnz is local dofs
    // o_nnz is dofs on different processors.
    std::vector<int> d_nnz(dofs_local, 0), o_nnz(dofs_local, 0);
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            // We should already have cached the FD weights
            // Start with bulk pts
            if (d_bulk_weights->patchHasPts(patch))
            {
                const std::set<FDPoint>& bulk_pts = d_bulk_weights->getRBFFDBasePoints(patch);
                const std::map<FDPoint, std::vector<FDPoint>>& rbf_pts = d_bulk_weights->getRBFFDPoints(patch);
                const std::map<FDPoint, unsigned int>& eq_nums = d_cc_bulk->getFDConditionMapPatch(patch.getPointer());

                for (const auto& base_pt : bulk_pts)
                {
                    const int dof_index = getDofIndex(base_pt, patch, dof_map, *d_index_ptr);
                    // Ensure that it is local
                    if (dofs_lower <= dof_index && dof_index < dofs_upper)
                    {
                        const unsigned int eq_num = eq_nums.at(base_pt);
                        const std::vector<FDPoint>& pt_vec = rbf_pts.at(base_pt);
                        for (const auto& pt : pt_vec)
                        {
                            const int rbf_dof_index = getDofIndex(pt, patch, dof_map, *d_index_ptr);
                            if (dofs_lower <= rbf_dof_index && rbf_dof_index < dofs_upper)
                                d_nnz[eq_num] += 1;
                            else
                                o_nnz[eq_num] += 1;
                        }
                        // Ensure we haven't overcounted
                        d_nnz[eq_num] = std::min(dofs_local, d_nnz[eq_num]);
                        o_nnz[eq_num] = std::min(dofs_total - dofs_local, o_nnz[eq_num]);
                    }
                }
            }

            // Now bdry pts
            if (d_bdry_weights->patchHasPts(patch))
            {
                const std::set<FDPoint>& bdry_pts = d_bdry_weights->getRBFFDBasePoints(patch);
                const std::map<FDPoint, std::vector<FDPoint>>& rbf_pts = d_bdry_weights->getRBFFDPoints(patch);
                const std::map<FDPoint, unsigned int>& eq_nums = d_cc_bdry->getFDConditionMapPatch(patch.getPointer());

                for (const auto& base_pt : bdry_pts)
                {
                    const int dof_index = getDofIndex(base_pt, patch, dof_map, *d_index_ptr);
                    // Ensure that it is local
                    if (dofs_lower <= dof_index && dof_index < dofs_upper)
                    {
                        // Bdry eq_num is the local bulk number + bdry_number
                        const unsigned int eq_num = eq_nums.at(base_pt) + bulk_eqs_local;
                        const std::vector<FDPoint>& pt_vec = rbf_pts.at(base_pt);
                        for (const auto& pt : pt_vec)
                        {
                            const int rbf_dof_index = getDofIndex(pt, patch, dof_map, *d_index_ptr);
                            if (dofs_lower <= rbf_dof_index && rbf_dof_index < dofs_upper)
                                d_nnz[eq_num] += 1;
                            else
                                o_nnz[eq_num] += 1;
                        }
                        // Ensure we haven't overcounted
                        d_nnz[eq_num] = std::min(dofs_local, d_nnz[eq_num]);
                        o_nnz[eq_num] = std::min(dofs_total - dofs_local, o_nnz[eq_num]);
                    }
                }
            }
        }
    }
    // We can now preallocate the matrix
    ierr = MatCreateAIJ(d_petsc_comm,
                        eqs_local,
                        dofs_local,
                        PETSC_DETERMINE,
                        PETSC_DETERMINE,
                        0,
                        d_nnz.data(),
                        0,
                        o_nnz.data(),
                        &d_petsc_mat);
    IBTK_CHKERRQ(ierr);

    // Now we need to fill in the coefficients
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            // The points should be cached.
            // Start with bulk equations
            if (d_bulk_weights->patchHasPts(patch))
            {
                const std::set<FDPoint>& bulk_pts = d_bulk_weights->getRBFFDBasePoints(patch);
                const std::map<FDPoint, std::vector<FDPoint>>& rbf_pts = d_bulk_weights->getRBFFDPoints(patch);
                const std::map<FDPoint, std::vector<double>>& rbf_weights = d_bulk_weights->getRBFFDWeights(patch);
                const std::map<FDPoint, unsigned int>& eq_nums = d_cc_bulk->getFDConditionMapPatch(patch.getPointer());

                for (const auto& base_pt : bulk_pts)
                {
                    const std::vector<FDPoint>& pt_vec = rbf_pts.at(base_pt);
                    const std::vector<double>& wgt_vec = rbf_weights.at(base_pt);
                    const unsigned int eq_num = eq_nums.at(base_pt);
                    int stencil_size = pt_vec.size();
                    std::vector<int> mat_cols(stencil_size);

                    // Get dof index for this point.
                    const int dof_index = getDofIndex(base_pt, patch, dof_map, *d_index_ptr);
                    // Ensure that it is local
                    if (dofs_lower <= dof_index && dof_index < dofs_upper)
                    {
                        unsigned int stencil_idx = 0;
                        for (const auto& pt : pt_vec)
                        {
                            const int rbf_dof_index = getDofIndex(pt, patch, dof_map, *d_index_ptr);
                            mat_cols[stencil_idx++] = rbf_dof_index;
                        }
                        ierr = MatSetValues(d_petsc_mat,
                                            1,
                                            reinterpret_cast<const PetscInt*>(&eq_num),
                                            stencil_size,
                                            mat_cols.data(),
                                            wgt_vec.data(),
                                            INSERT_VALUES);
                        IBTK_CHKERRQ(ierr);
                    }
                }
            }

            // Now bdry equations
            if (d_bdry_weights->patchHasPts(patch))
            {
                const std::set<FDPoint>& bdry_pts = d_bdry_weights->getRBFFDBasePoints(patch);
                const std::map<FDPoint, std::vector<FDPoint>>& rbf_pts = d_bdry_weights->getRBFFDPoints(patch);
                const std::map<FDPoint, std::vector<double>>& rbf_weights = d_bdry_weights->getRBFFDWeights(patch);
                const std::map<FDPoint, unsigned int>& eq_nums = d_cc_bdry->getFDConditionMapPatch(patch.getPointer());

                for (const auto& base_pt : bdry_pts)
                {
                    const std::vector<FDPoint>& pt_vec = rbf_pts.at(base_pt);
                    const std::vector<double>& wgt_vec = rbf_weights.at(base_pt);
                    // Bdry eq_num is the local bulk number + bdry_number
                    const unsigned int eq_num = eq_nums.at(base_pt) + bulk_eqs_local;
                    int stencil_size = pt_vec.size();
                    std::vector<int> mat_cols(stencil_size);
                    // Get dof index for this point.
                    const int dof_index = getDofIndex(base_pt, patch, dof_map, *d_index_ptr);
                    // Ensure that it is local
                    if (dofs_lower <= dof_index && dof_index < dofs_upper)
                    {
                        unsigned int stencil_idx = 0;
                        for (const auto& pt : pt_vec)
                        {
                            const int rbf_dof_index = getDofIndex(pt, patch, dof_map, *d_index_ptr);
                            mat_cols[stencil_idx++] = rbf_dof_index;
                        }
                        ierr = MatSetValues(d_petsc_mat,
                                            1,
                                            reinterpret_cast<const PetscInt*>(&eq_num),
                                            stencil_size,
                                            mat_cols.data(),
                                            wgt_vec.data(),
                                            INSERT_VALUES);
                        IBTK_CHKERRQ(ierr);
                    }
                }
            }
        }
    }
    // Assemble the matrix
    ierr = MatAssemblyBegin(d_petsc_mat, MAT_FINAL_ASSEMBLY);
    IBTK_CHKERRQ(ierr);
    ierr = MatAssemblyEnd(d_petsc_mat, MAT_FINAL_ASSEMBLY);
    IBTK_CHKERRQ(ierr);
    ADS_TIMER_STOP(t_setup_matrix_and_vectors);
}

void
RBFFDPoissonSolver::findFDWeights()
{
    ADS_TIMER_START(t_find_rbffd_weights);
    // Add in libmesh nodes
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const MeshBase& mesh = eq_sys->get_mesh();
    const System& sys = eq_sys->get_system(d_fe_mesh_partitioner->COORDINATES_SYSTEM_NAME);
    const System& n_sys = eq_sys->get_system(d_ghost_pts->getNormalSysName());
    const DofMap& dof_map = sys.get_dof_map();
    const DofMap& n_dof_map = n_sys.get_dof_map();
    NumericVector<double>* X_vec = d_fe_mesh_partitioner->buildGhostedCoordsVector(true);
    NumericVector<double>* N_vec =
        d_fe_mesh_partitioner->buildGhostedSolutionVector(d_ghost_pts->getNormalSysName(), true);
    int ghost_idx = 0;

    // Loop through all the points and find FD weights
    // We do this on a patch by patch basis
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        std::vector<std::vector<Node*>> nodes = d_fe_mesh_partitioner->getActivePatchNodeMap();
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            std::vector<FDPoint> global_fd_points;
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if (node_to_cell(idx, *ls_data) < -d_eps) global_fd_points.push_back(FDPoint(patch, idx));
            }

            if (ln == d_hierarchy->getFinestLevelNumber())
            {
                for (const auto& node : nodes[ln])
                {
                    VectorNd node_pt = VectorNd::Zero();
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        std::vector<dof_id_type> dofs;
                        dof_map.dof_indices(node, dofs, d);
                        for (const auto dof : dofs) node_pt[d] += (*X_vec)(dof);
                    }
                    global_fd_points.push_back(FDPoint(node_pt, node));
                }
            }

            // Now ghost nodes
            const std::vector<GhostPoint>& eul_ghost_nodes = d_ghost_pts->getEulerianGhostNodes();
            for (const auto& ghost_node : eul_ghost_nodes) global_fd_points.push_back(FDPoint(&ghost_node));
            const std::vector<GhostPoint>& lag_ghost_nodes = d_ghost_pts->getLagrangianGhostNodes();
            for (const auto& ghost_node : lag_ghost_nodes) global_fd_points.push_back(FDPoint(&ghost_node));

            // Now build tree
            tree::KDTree<FDPoint> tree(global_fd_points);

            // Find the closest points for the FD
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                FDPoint base_pt(patch, idx);
                std::vector<FDPoint> fd_pts;
                std::vector<double> wgts;

                if (node_to_cell(idx, *ls_data) < -d_switch_to_rbffd_dist)
                {
                    // For efficiency, we reduce to standard laplacian away from boundary.
                    std::vector<int> idx_vec;
                    std::vector<double> distance_vec;
#if (NDIM == 2)
                    tree.knnSearch(base_pt, 5, idx_vec, distance_vec);
#endif
#if (NDIM == 3)
                    tree.knnSearch(base_pt, 7, idx_vec, distance_vec);
#endif
                    for (const auto& idx : idx_vec) fd_pts.push_back(global_fd_points[idx]);
                    Reconstruct::RBFFD_reconstruct<FDPoint>(
                        wgts, base_pt, fd_pts, d_poly_degree, dx, d_rbf, d_lap_rbf, nullptr, d_lap_polys, nullptr);
                }
                else if (node_to_cell(idx, *ls_data) < -d_eps)
                {
                    std::vector<int> idx_vec;
                    std::vector<double> distance_vec;
                    tree.knnSearch(base_pt, d_stencil_size, idx_vec, distance_vec);
                    for (const auto& idx : idx_vec) fd_pts.push_back(global_fd_points[idx]);
                    // Now compute finite difference weights with these points.
                    Reconstruct::RBFFD_reconstruct<FDPoint>(
                        wgts, base_pt, fd_pts, d_poly_degree, dx, d_rbf, d_lap_rbf, nullptr, d_lap_polys, nullptr);
                }
                else
                {
                    fd_pts = { base_pt };
                    wgts = { 1.0 };
                }

                // Now cache the wgts
                d_bulk_weights->cachePoint(patch, base_pt, fd_pts, wgts);
            }

            // Now loop over the boundary, if it exists
            if (pgeom->getTouchesRegularBoundary())
            {
                // We touch a regular boundary, now get that boundary
                const tbox::Array<BoundaryBox<NDIM>>& bdry_boxes = pgeom->getCodimensionBoundaries(1);
                const double* const xlow = pgeom->getXLower();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();
                for (int i = 0; i < bdry_boxes.size(); ++i)
                {
                    const int axis = bdry_boxes[i].getLocationIndex() / 2;
                    VectorNd normal(VectorNd::Zero());
                    normal(axis) = bdry_boxes[i].getLocationIndex() % 2 == 0 ? -1.0 : 1.0;
                    for (CellIterator<NDIM> ci(bdry_boxes[i].getBox()); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx = ci();
                        std::vector<double> x(NDIM);
                        for (int d = 0; d < NDIM; ++d)
                            x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) +
                                                      (axis == d ? (0.5 - 0.5 * normal[axis]) : 0.5));
                        FDPoint pt(x);
                        std::vector<int> idxs;
                        std::vector<double> distances;
                        std::vector<FDPoint> fd_pts;
                        tree.knnSearch(pt, d_stencil_size, idxs, distances);

                        for (const auto& idx : idxs) fd_pts.push_back(global_fd_points[idx]);

                        // Now compute finite difference weights with these points.
                        std::vector<double> wgts;
                        Reconstruct::RBFFD_reconstruct<FDPoint>(wgts,
                                                                pt,
                                                                fd_pts,
                                                                d_poly_degree,
                                                                dx,
                                                                d_rbf,
                                                                d_lap_rbf,
                                                                static_cast<void*>(&normal),
                                                                d_lap_polys,
                                                                static_cast<void*>(&normal));
                        // TODO: This is a potentially dangerous hack. FD weights must be associated with an FDPoint,
                        // but we have no FDPoint's that live on the Eulerian boundary. The interior FDPoint can't be
                        // the base pt, because then we would use the corner FDPoints multiple times. Instead, we
                        // associate a unique, possibly unrelated GhostPoint with the FD weights.
                        //
                        // TODO: This will break if the structure is near the computational boundary.
                        d_bdry_weights->cachePoint(patch, FDPoint(&eul_ghost_nodes[ghost_idx]), fd_pts, wgts);
                        ghost_idx++;
                    }
                }
            }

            if (ln == d_hierarchy->getFinestLevelNumber())
            {
                // Note now we only loop over local nodes
                auto it = mesh.local_nodes_begin();
                const auto it_end = mesh.local_nodes_end();
                for (; it != it_end; it++)
                {
                    const Node* node = *it;
                    std::vector<dof_id_type> n_dofs, x_dofs;
                    dof_map.dof_indices(node, x_dofs);
                    n_dof_map.dof_indices(node, n_dofs);
                    IBTK::VectorNd x, n;
                    for (int d = 0; d < NDIM; ++d)
                    {
                        x[d] = (*X_vec)(x_dofs[d]);
                        n[d] = (*N_vec)(n_dofs[d]);
                    }
                    FDPoint base_pt(x, node);

                    std::vector<int> idxs;
                    std::vector<double> distances;
                    std::vector<FDPoint> fd_pts;

                    tree.knnSearch(base_pt, d_stencil_size, idxs, distances);
                    for (const auto idx : idxs) fd_pts.push_back(global_fd_points[idx]);

                    // Now find the finite difference. These are all boundary points, so create FD points that enforce
                    // both the boundary condition and PDE Now compute finite difference weights with these points.
                    std::vector<double> bdry_wgts, bulk_wgts;
                    Reconstruct::RBFFD_reconstruct<FDPoint>(
                        bulk_wgts, base_pt, fd_pts, d_poly_degree, dx, d_rbf, d_lap_rbf, nullptr, d_lap_polys, nullptr);
                    Reconstruct::RBFFD_reconstruct<FDPoint>(bdry_wgts,
                                                            base_pt,
                                                            fd_pts,
                                                            d_poly_degree,
                                                            dx,
                                                            d_rbf,
                                                            d_lap_rbf,
                                                            static_cast<void*>(&n),
                                                            d_lap_polys,
                                                            static_cast<void*>(&n));
                    d_bulk_weights->cachePoint(patch, base_pt, fd_pts, bulk_wgts);
                    d_bdry_weights->cachePoint(patch, base_pt, fd_pts, bdry_wgts);
                }
            }
        }
    }
    ADS_TIMER_STOP(t_find_rbffd_weights);
}

void
RBFFDPoissonSolver::writeMatToFile(const std::string& filename)
{
    std::ofstream mat_file;
    mat_file.open(filename);

    int num_rows, num_cols;
    int ierr = MatGetSize(d_petsc_mat, &num_rows, &num_cols);
    IBTK_CHKERRQ(ierr);
    mat_file << std::to_string(num_rows) << " " << std::to_string(num_cols) << "\n";
    // Now loop through rows of matrix
    for (int row = 0; row < num_rows; ++row)
    {
        int num_cols;
        const int* col_idxs = nullptr;
        const double* weights = nullptr;
        ierr = MatGetRow(d_petsc_mat, row, &num_cols, &col_idxs, &weights);
        IBTK_CHKERRQ(ierr);
        // Now print out columns
        for (int col = 0; col < num_cols; ++col)
        {
            mat_file << std::to_string(col_idxs[col]) << " " << std::to_string(weights[col]) << " ";
        }
        mat_file << "\n";
        MatRestoreRow(d_petsc_mat, row, &num_cols, &col_idxs, &weights);
        IBTK_CHKERRQ(ierr);
    }
    mat_file.close();

    // Now output rhs
    std::ofstream rhs_file;
    rhs_file.open("rhs");
    int vec_size;
    double* vec_vals;
    ierr = VecGetSize(d_petsc_b, &vec_size);
    IBTK_CHKERRQ(ierr);
    rhs_file << std::to_string(vec_size) << "\n";
    ierr = VecGetArray(d_petsc_b, &vec_vals);
    IBTK_CHKERRQ(ierr);
    for (int i = 0; i < vec_size; ++i) rhs_file << std::to_string(vec_vals[i]) << "\n";
    ierr = VecRestoreArray(d_petsc_b, &vec_vals);
    IBTK_CHKERRQ(ierr);
    rhs_file.close();

    // Now output sol
    std::ofstream sol_file;
    sol_file.open("sol");
    ierr = VecGetSize(d_petsc_x, &vec_size);
    IBTK_CHKERRQ(ierr);
    sol_file << std::to_string(vec_size) << "\n";
    ierr = VecGetArray(d_petsc_x, &vec_vals);
    IBTK_CHKERRQ(ierr);
    for (int i = 0; i < vec_size; ++i) sol_file << std::to_string(vec_vals[i]) << "\n";
    ierr = VecRestoreArray(d_petsc_b, &vec_vals);
    sol_file.close();
    IBTK_CHKERRQ(ierr);
    return;
}

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
