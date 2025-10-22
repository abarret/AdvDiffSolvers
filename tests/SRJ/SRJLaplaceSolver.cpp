/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/app_namespaces.h" // IWYU pragma: keep

#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "CellDataFactory.h"
#include "CellVariable.h"
#include "IntVector.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "PoissonSpecifications.h"
#include "SAMRAIVectorReal.h"
#include "SRJLaplaceSolver.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"
#include "tbox/Utilities.h"

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Types of refining and coarsening to perform prior to setting coarse-fine
// boundary and physical boundary ghost cell values.
static const std::string DATA_REFINE_TYPE = "CONSERVATIVE_LINEAR_REFINE";
static const std::string DATA_COARSEN_TYPE = "CUBIC_COARSEN";

// Type of extrapolation to use at physical boundaries.
static const std::string BDRY_EXTRAP_TYPE = "LINEAR";

// Whether to enforce consistent interpolated values at Type 2 coarse-fine
// interface ghost cells.
static const bool CONSISTENT_TYPE_2_BDRY = false;

static Timer* t_apply;
} // namespace

namespace sharp_interface
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

SRJLaplaceSolver::SRJLaplaceSolver(const std::string& object_name,
                                   Pointer<Database> input_db,
                                   const bool homogeneous_bc)
{
    d_object_name = object_name;
    d_max_iterations = input_db->getInteger("num_iterations");
    int num_levels = input_db->getInteger("num_levels");
    d_Q.resize(num_levels);
    d_omega.resize(num_levels);
    input_db->getIntegerArray("Q", d_Q.data(), num_levels);
    input_db->getDoubleArray("omega", d_omega.data(), num_levels);
    IBTK_DO_ONCE(t_apply = TimerManager::getManager()->getTimer("ADS::SRJLaplaceSolver::apply()"););
    return;
} // SRJLaplaceSolver()

SRJLaplaceSolver::~SRJLaplaceSolver()
{
    if (d_is_initialized) deallocateSolverState();
    return;
} // ~SRJLaplaceSolver()

bool
SRJLaplaceSolver::solveSystem(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    ADS_TIMER_START(t_apply);
    if (d_initial_guess_nonzero) x.setToScalar(0.0);
#if !defined(NDEBUG)
    TBOX_ASSERT(d_is_initialized);
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        Pointer<CellVariable<NDIM, double>> x_cc_var = x.getComponentVariable(comp);
        Pointer<CellVariable<NDIM, double>> y_cc_var = y.getComponentVariable(comp);
        if (!x_cc_var || !y_cc_var)
        {
            TBOX_ERROR(d_object_name << "::apply()\n"
                                     << "  encountered non-cell centered vector components" << std::endl);
        }
        Pointer<CellDataFactory<NDIM, double>> x_factory = x_cc_var->getPatchDataFactory();
        Pointer<CellDataFactory<NDIM, double>> y_factory = y_cc_var->getPatchDataFactory();
        TBOX_ASSERT(x_factory);
        TBOX_ASSERT(y_factory);
        const unsigned int x_depth = x_factory->getDefaultDepth();
        const unsigned int y_depth = y_factory->getDefaultDepth();
        TBOX_ASSERT(x_depth == y_depth);
    }
#endif
    // Setup the interpolation transaction information.
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    d_transaction_comps.clear();
    InterpolationTransactionComponent component(x.getComponentDescriptorIndex(0),
                                                DATA_REFINE_TYPE,
                                                false,
                                                DATA_COARSEN_TYPE,
                                                BDRY_EXTRAP_TYPE,
                                                CONSISTENT_TYPE_2_BDRY,
                                                d_bdry_coefs);
    d_transaction_comps.push_back(component);

    // Initialize the interpolation operators.
    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(d_transaction_comps, d_hierarchy, d_coarsest_ln, d_finest_ln);

    Pointer<SAMRAIVectorReal<NDIM, double>> y_clone = y.cloneVector("y_clone");
    y_clone->allocateVectorData(d_solution_time);

    int iter_num = 0;
    while (iter_num < d_max_iterations)
    {
        pout << "Iteration " << iter_num << "\n";
        // On a given iteration, we do a Jacobi iteration with a given relaxation factor a specified number of times.
        for (size_t i = 0; i < d_Q.size(); ++i)
        {
            double w = d_omega[i]; // Relaxation factor
            int Q = d_Q[i];        // Number of steps with relaxation factor

            pout << "Relaxing " << Q << " times with factor " << w << "\n";

            // Fill ghost cells
            d_hier_bdry_fill->fillData(d_solution_time);
            for (int comp = 0; comp < x.getNumberOfComponents(); ++comp)
            {
                for (int j = 0; j < Q; ++j)
                {
                    relaxOnHierarchy(x.getComponentDescriptorIndex(comp),
                                     y.getComponentDescriptorIndex(comp),
                                     w,
                                     d_hierarchy,
                                     d_coarsest_ln,
                                     d_finest_ln);
                }
            }
        }
        ++iter_num;

        //        d_laplace_op->apply(x, *y_clone);
    }

    y_clone->deallocateVectorData();
    ADS_TIMER_STOP(t_apply);
    return true;
} // apply

void
SRJLaplaceSolver::initializeSolverState(const SAMRAIVectorReal<NDIM, double>& in,
                                        const SAMRAIVectorReal<NDIM, double>& out)
{
    // Deallocate the operator state if the operator is already initialized.
    if (d_is_initialized) deallocateSolverState();

    // Setup solution and rhs vectors.
    d_x = in.cloneVector(in.getName());
    d_b = out.cloneVector(out.getName());
    d_x->allocateVectorData(d_solution_time);
    d_b->allocateVectorData(d_solution_time);

    // Setup operator state.
    d_hierarchy = in.getPatchHierarchy();
    d_coarsest_ln = in.getCoarsestLevelNumber();
    d_finest_ln = in.getFinestLevelNumber();

    d_ncomp = in.getNumberOfComponents();

#if !defined(NDEBUG)
    TBOX_ASSERT(d_hierarchy == out.getPatchHierarchy());
    TBOX_ASSERT(d_coarsest_ln == out.getCoarsestLevelNumber());
    TBOX_ASSERT(d_finest_ln == out.getFinestLevelNumber());
    TBOX_ASSERT(d_ncomp == out.getNumberOfComponents());
#endif

    if (!d_hier_math_ops_external)
    {
        d_hier_math_ops =
            new HierarchyMathOps(d_object_name + "::HierarchyMathOps", d_hierarchy, d_coarsest_ln, d_finest_ln);
    }
    else
    {
#if !defined(NDEBUG)
        TBOX_ASSERT(d_hier_math_ops);
#endif
    }

    d_laplace_op = std::make_unique<CCLaplaceOperator>("laplace_op");
    d_poisson_spec.setCConstant(0.0);
    d_poisson_spec.setDConstant(1.0);
    d_laplace_op->setPoissonSpecifications(d_poisson_spec);
    d_laplace_op->initializeOperatorState(in, out);

    // Indicate the operator is initialized.
    d_is_initialized = true;
    return;
} // initializeOperatorState

void
SRJLaplaceSolver::deallocateSolverState()
{
    if (!d_is_initialized) return;

    // Deallocate the interpolation operators.
    d_hier_bdry_fill->deallocateOperatorState();
    d_hier_bdry_fill.setNull();
    d_transaction_comps.clear();

    // Deallocate hierarchy math operations object.
    if (!d_hier_math_ops_external) d_hier_math_ops.setNull();

    // Delete the solution and rhs vectors.
    d_x->deallocateVectorData();
    d_x->freeVectorComponents();
    d_x.setNull();

    d_b->deallocateVectorData();
    d_b->freeVectorComponents();
    d_b.setNull();

    // Indicate that the operator is NOT initialized.
    d_is_initialized = false;
    return;
} // deallocateOperatorState

/////////////////////////////// PRIVATE //////////////////////////////////////

void
SRJLaplaceSolver::relaxOnHierarchy(const int x_idx,
                                   const int y_idx,
                                   const double w,
                                   Pointer<PatchHierarchy<NDIM>> hierarchy,
                                   const int coarsest_ln,
                                   const int finest_ln)
{
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);

        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            double denom = 0.0;
            for (int d = 0; d < NDIM; ++d) denom -= 2.0 / (dx[d] * dx[d]);

            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_idx);
            Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y_idx);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                double new_val =
                    0.25 * ((*x_data)(idx + IntVector<NDIM>(1, 0)) + (*x_data)(idx - IntVector<NDIM>(1, 0)) +
                            (*x_data)(idx + IntVector<NDIM>(0, 1)) + (*x_data)(idx - IntVector<NDIM>(0, 1)) -
                            (*y_data)(idx)*dx[0] * dx[1]);
                (*x_data)(idx) = w * new_val + (1.0 - w) * (*x_data)(idx);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
