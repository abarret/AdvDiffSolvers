/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/PETScPoissonSolver.h"
#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

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
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

PETScPoissonSolver::PETScPoissonSolver(std::string object_name,
                                       Pointer<Database> input_db,
                                       std::string default_options_prefix,
                                       MPI_Comm petsc_comm)
    : d_ksp_type(KSPGMRES),
      d_options_prefix(std::move(default_options_prefix)),
      d_petsc_comm(petsc_comm),
      d_eul_idx_var(new CellVariable<NDIM, int>(object_name + "::IdxVar"))
{
    // Setup default values.
    GeneralSolver::init(std::move(object_name), /*homogeneous_bc*/ false);
    d_max_iterations = 10000;
    d_abs_residual_tol = 1.0e-50;
    d_rel_residual_tol = 1.0e-5;
    d_enable_logging = true;

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
    }

    // Common constructor functionality.
    commonConstructor();
    return;
} // PETScPoissonSolver()

PETScPoissonSolver::~PETScPoissonSolver()
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
} // ~PETScPoissonSolver()

void
PETScPoissonSolver::setLagrangianMeshData(std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner, std::string sys_name)
{
    d_fe_mesh_partitioner = std::move(fe_mesh_partitioner);
    d_sys_name = std::move(sys_name);
}

void
PETScPoissonSolver::setEulerianMeshData(Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_hierarchy = hierarchy;
    // Register eulerian patch data index if not already done
    if (d_eul_idx_idx != IBTK::invalid_index)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        d_eul_idx_idx = var_db->registerVariableAndContext(
            d_eul_idx_var, var_db->getContext(d_object_name + "::EulIdx"), IntVector<NDIM>(2));
    }
}

void
PETScPoissonSolver::setOptionsPrefix(const std::string& options_prefix)
{
    d_options_prefix = options_prefix;
    return;
} // setOptionsPrefix

const KSP&
PETScPoissonSolver::getPETScKSP() const
{
    return d_petsc_ksp;
} // getPETScKSP

bool
PETScPoissonSolver::solveSystem(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& b)
{
    ADS_TIMER_START(t_solve_system);
    int ierr;

    // Initialize the solver, when necessary.
    const bool deallocate_after_solve = !d_is_initialized;
    if (deallocate_after_solve) initializeSolverState(x, b);
#if !defined(NDEBUG)
    TBOX_ASSERT(d_petsc_ksp);
#endif

    // Indicies should already be set up. Copy data to PETSc data structures
    copyDataToPetsc(x, b);

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

    // Copy data back to other representations
    copyDataFromPetsc(x, b);

    // Deallocate the solver, when necessary.
    if (deallocate_after_solve) deallocateSolverState();
    ADS_TIMER_STOP(t_solve_system);
    return converged;
} // solveSystem

void
PETScPoissonSolver::initializeSolverState(const SAMRAIVectorReal<NDIM, double>& x,
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

    // Setup Eulerian DOF indexing
    setupEulDOFs(x);
    // Setup Lagrangian DOF indexing
    setupLagDOFs();
    // Now create matrix
    setupMatrix();

    // Register matrix with KSP
    ierr = KSPSetOperators(d_petsc_ksp, d_petsc_mat, d_petsc_mat);

    // Indicate that the solver is initialized.
    d_reinitializing_solver = false;
    d_is_initialized = true;

    ADS_TIMER_STOP(t_initialize_solver_state);
    return;
} // initializeSolverState

void
PETScPoissonSolver::deallocateSolverState()
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
PETScPoissonSolver::commonConstructor()
{
    // Setup Timers.
    IBTK_DO_ONCE(t_solve_system = TimerManager::getManager()->getTimer("ADS::PETScPoissonSolver::solveSystem()");
                 t_initialize_solver_state =
                     TimerManager::getManager()->getTimer("ADS::PETScPoissonSolver::initializeSolverState()");
                 t_deallocate_solver_state =
                     TimerManager::getManager()->getTimer("ADS::PETScPoissonSolver::deallocateSolverState()"););
    return;
} // common_ctor

void
PETScPoissonSolver::copyDataToPetsc(const SAMRAIVectorReal<NDIM, double>& x, const SAMRAIVectorReal<NDIM, double>& b)
{
    // We are assuming the PETSc Vec has been allocated correctly.
    int ierr;
    // Start with Eulerian data
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    for (unsigned int comp = 0; comp <= x.getNumberOfComponents(); ++comp)
    {
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(d_eul_idx_idx);
            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x.getComponentDescriptorIndex(comp));
            Pointer<CellData<NDIM, double>> b_data = patch->getPatchData(b.getComponentDescriptorIndex(comp));
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                int petsc_idx = (*idx_data)(idx, comp);
                ierr = VecSetValue(d_petsc_x, petsc_idx, (*x_data)(idx), INSERT_VALUES);
                IBTK_CHKERRQ(ierr);
                ierr = VecSetValue(d_petsc_b, petsc_idx, (*b_data)(idx), INSERT_VALUES);
                IBTK_CHKERRQ(ierr);
            }
        }
    }

    // Now the Lag data
    // Note the corresponding PETSc index is the libMesh index plus the number of SAMRAI indices
    int eul_offset = std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.end(), 0);
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const System& sys = eq_sys->get_system(d_sys_name);
    const DofMap& dof_map = sys.get_dof_map();
    NumericVector<double>* x_vec = sys.current_local_solution.get();
    NumericVector<double>* b_vec = sys.current_local_solution.get();
    const MeshBase& mesh = eq_sys->get_mesh();
    auto it = mesh.local_nodes_begin();
    auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, &dofs);
        for (const auto& dof : dofs)
        {
            ierr = VecSetValue(d_petsc_x, dof + eul_offset, (*x_vec)(dof), INSERT_VALUES);
            IBTK_CHKERRQ(ierr);
            ierr = VecSetValue(d_petsc_b, dof + eul_offset, (*b_vec)(dof), INSERT_VALUES);
            IBTK_CHKERRQ(ierr);
        }
    }
    x_vec->close();
    b_vec->close();
    ierr = VecAssemblyBegin(d_petsc_x);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyEnd(d_petsc_x);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyBegin(d_petsc_b);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyEnd(d_petsc_b);
    IBTK_CHKERRQ(ierr);
}

void
PETScPoissonSolver::copyDataFromPetsc(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& b)
{
    // We are assuming the PETSc Vec has been allocated correctly.
    int ierr;
    const double* x_data;
    const double* b_data;
    ierr = VecGetArrayRead(d_petsc_x, &x_data);
    IBTK_CHKERRQ(ierr);
    ierr = VecGetArrayRead(d_petsc_b, &b_data);
    IBTK_CHKERRQ(ierr);
    // Start with Eulerian data
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    const int mpi_rank = IBTK_MPI::getNodes();
    const int local_eul_offset =
        std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.begin() + mpi_rank, 0);
    for (unsigned int comp = 0; comp <= x.getNumberOfComponents(); ++comp)
    {
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, int>> idx_data = patch->getPatchData(d_eul_idx_idx);
            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x.getComponentDescriptorIndex(comp));
            Pointer<CellData<NDIM, double>> b_data = patch->getPatchData(b.getComponentDescriptorIndex(comp));
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                int petsc_idx = (*idx_data)(idx, comp);
                (*x_data)(idx) = x_data[petsc_idx - local_eul_offset];
                (*b_data)(idx) = b_data[petsc_idx - local_eul_offset];
            }
        }
    }

    // Now the Lag data
    // Note the corresponding PETSc index is the libMesh index plus the number of SAMRAI indices
    int eul_offset = std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.end(), 0);
    const int local_lag_offset =
        std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.begin() + mpi_rank, 0);
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const System& sys = eq_sys->get_system(d_sys_name);
    const DofMap& dof_map = sys.get_dof_map();
    NumericVector<double>* x_vec = sys.current_local_solution.get();
    NumericVector<double>* b_vec = sys.current_local_solution.get();
    const MeshBase& mesh = eq_sys->get_mesh();
    auto it = mesh.local_nodes_begin();
    auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;
        std::vector<dof_id_type> dofs;
        dof_map.dof_indices(node, &dofs);
        for (const auto& dof : dofs)
        {
            (*x_vec)(dof) = x_data[dof + eul_offset - local_lag_offset];
            (*b_vec)(dof) = b_data[dof + eul_offset - local_lag_offset];
        }
    }
    x_vec->close();
    b_vec->close();
    sys.update();
    ierr = VecRestoreArrayRead(d_petsc_x, &x_data);
    IBTK_CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(d_petsc_b, &b_data);
    IBTK_CHKERRQ(ierr);
}

void
PETScPoissonSolver::setupEulDOFs(const SAMRAIVectorReal<NDIM, double>& x)
{
    // TODO: This function assumes there's only one level in the hierarchy
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    // Determine the number of local DOFs.
    int local_dof_count = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& patch_box = patch->getBox();
        Pointer<CellData<NDIM, int>> dof_index_data = patch->getPatchData(d_eul_idx_idx);
        const int depth = dof_index_data->getDepth();
        local_dof_count += depth * CellGeometry<NDIM>::toCellBox(patch_box).size();
    }

    // Determine the number of DOFs local to each MPI process and compute the
    // local DOF index offset.
    const int mpi_size = IBTK_MPI::getNodes();
    const int mpi_rank = IBTK_MPI::getRank();
    d_eul_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(local_dof_count, &d_eul_dofs_per_proc[0]);
    const int local_dof_offset =
        std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.begin() + mpi_rank, 0);

    // Assign local DOF indices.
    int counter = local_dof_offset;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& patch_box = patch->getBox();
        Pointer<CellData<NDIM, int>> dof_index_data = patch->getPatchData(d_eul_idx_idx);
        dof_index_data->fillAll(-1);
        const int depth = dof_index_data->getDepth();
        for (Box<NDIM>::Iterator b(CellGeometry<NDIM>::toCellBox(patch_box)); b; b++)
        {
            const CellIndex<NDIM>& i = b();
            for (int d = 0; d < depth; ++d)
            {
                (*dof_index_data)(i, d) = counter++;
            }
        }
    }

    // Communicate ghost DOF indices.
    RefineAlgorithm<NDIM> ghost_fill_alg;
    ghost_fill_alg.registerRefine(d_eul_idx_idx, d_eul_idx_idx, d_eul_idx_idx, nullptr);
    ghost_fill_alg.createSchedule(level)->fillData(0.0);
}

void
PETScPoissonSolver::setupLagDOFs()
{
    EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
    const MeshBase& mesh = eq_sys->get_mesh();

    const System& sys = eq_sys->get_system(d_sys_name);
    const DofMap& dof_map = sys.get_dof_map();

    // For now, we just use libMesh's partitioning.
    // TODO: This could give very poor scaling if libMesh's partitioning is significantly different than SAMRAI's.
    int local_dofs = dof_map.n_local_dofs();
    const int mpi_size = IBTK_MPI::getNodes();
    const int mpi_rank = IBTK_MPI::getRank();
    d_lag_dofs_per_proc.resize(mpi_size, 0);
    IBTK_MPI::allGather(local_dofs, &d_lag_dofs_per_proc[0]);
}

void
PETScPoissonSolver::setupMatrixAndVec()
{
    // At this point we have the vectors and DOFs labeled. We need to create the matrix.

    // Determine the non-zero structure of the matrix.
    const int mpi_rank = IBTK_MPI::getRank();
    const int eul_local = d_eul_dofs_per_proc[mpi_rank];
    const int eul_lower = std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.begin() + mpi_rank, 0);
    const int eul_upper = eul_lower + eul_local;
    const int eul_total = std::accumulate(d_eul_dofs_per_proc.begin(), d_eul_dofs_per_proc.end(), 0);

    const int lag_local = d_lag_dofs_per_proc[mpi_rank];
    const int lag_lower = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.begin() + mpi_rank, 0);
    const int lag_upper = lag_lower + lag_local;
    const int lag_total = std::accumulate(d_lag_dofs_per_proc.begin(), d_lag_dofs_per_proc.end(), 0);

    // First the Eulerian degrees of freedom.
}

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
