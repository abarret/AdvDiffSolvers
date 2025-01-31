/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/CCSharpInterfaceScheduledJacobiSolver.h"
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
static Timer* t_find_schedule;
} // namespace

namespace sharp_interface
{

std::vector<std::pair<double, int>>
find_relaxation_schedule(const int N, std::vector<double> w, std::vector<int> Q)
{
    ADS_TIMER_START(t_find_schedule);
    // Total number of iterations
    int M = std::accumulate(Q.begin(), Q.end(), 0);
    std::vector<std::pair<double, int>> schedule;
    // We will at most have M components of the schedule, but sqrt(M) is a good estimate
    schedule.reserve(std::sqrt(M));
    const double kmin = std::pow(std::sin(0.5 * M_PI / static_cast<double>(N)), 2.0);
    const int num_wave_lengths = std::floor(2.0 / kmin);
    Eigen::ArrayXd wave_lengths = Eigen::ArrayXd::Zero(num_wave_lengths);
    for (int i = 0; i < num_wave_lengths; ++i) wave_lengths[i] = kmin * static_cast<double>(i + 1);
    Eigen::ArrayXd G = Eigen::ArrayXd::Ones(num_wave_lengths);

    // First do the largest relaxation
    int min_location = 0;
    schedule.push_back(std::make_pair(w[min_location], 1));
    Q[min_location] -= 1;
    G = (G * (1.0 - w[min_location] * wave_lengths).abs()).eval();

    // Loop through the iteration schedule
    for (int i = 0; i < M; ++i)
    {
        // Determine which relaxation factor reduces the largest relaxation time
        Eigen::Index index;
        G.matrix().maxCoeff(&index);
        double optimal_weight = 1.0 / (wave_lengths[index]);

        double min_dist = std::numeric_limits<double>::max();
        for (int location = 0; location < w.size(); ++location)
        {
            if (Q[location] == 0) continue;
            double dist = std::abs(optimal_weight - w[location]);
            if (dist < min_dist)
            {
                // Found new minimum.
                min_location = location;
                min_dist = dist;
            }
        }

        // Use this relaxation factor.
        if (schedule.back().first == w[min_location])
            schedule.back().second += 1;
        else
            schedule.push_back(std::make_pair(w[min_location], 1));
        Q[min_location] -= 1;
        G = (G * (1.0 - w[min_location] * wave_lengths).abs()).eval();
    }

    ADS_TIMER_STOP(t_find_schedule);
    return schedule;
}

/////////////////////////////// PUBLIC ///////////////////////////////////////

CCSharpInterfaceScheduledJacobiSolver::CCSharpInterfaceScheduledJacobiSolver(
    const std::string& object_name,
    Pointer<Database> input_db,
    SharpInterfaceGhostFill& ghost_fill,
    std::function<double(const VectorNd&)> bdry_fcn,
    const bool homogeneous_bc)
    : d_ghost_fill(&ghost_fill), d_bdry_fcn(bdry_fcn)
{
    IBTK_DO_ONCE(t_apply = TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceScheduledJacobiSolver::apply()");
                 t_find_schedule = TimerManager::getManager()->getTimer(
                     "ADS::CCSharpInterfaceScheduledJacobiSolver::find_schedule()"););

    d_object_name = object_name;
    d_max_iterations = input_db->getInteger("num_iterations");
    int num_levels = input_db->getInteger("num_levels");
    int N = input_db->getInteger("n");
    std::vector<int> Q(num_levels);
    std::vector<double> omega(num_levels);
    input_db->getIntegerArray("Q", Q.data(), num_levels);
    input_db->getDoubleArray("omega", omega.data(), num_levels);
    d_schedule = find_relaxation_schedule(N, omega, Q);
    return;
} // CCSharpInterfaceScheduledJacobiSolver()

CCSharpInterfaceScheduledJacobiSolver::~CCSharpInterfaceScheduledJacobiSolver()
{
    if (d_is_initialized) deallocateSolverState();
    return;
} // ~CCSharpInterfaceScheduledJacobiSolver()

bool
CCSharpInterfaceScheduledJacobiSolver::solveSystem(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    ADS_TIMER_START(t_apply);
    x.setToScalar(0.0);
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

    int iter_num = 0;
    while (iter_num < d_max_iterations)
    {
        // On a given iteration, we do a Jacobi iteration with a given relaxation factor a specified number of times.
        for (const auto& w_Q_pair : d_schedule)
        {
            const double w = w_Q_pair.first;
            const int Q = w_Q_pair.second; // Number of steps with relaxation factor

            // Fill ghost cells
            d_hier_bdry_fill->fillData(d_solution_time);
            for (int comp = 0; comp < x.getNumberOfComponents(); ++comp)
            {
                for (int j = 0; j < Q; ++j)
                    relaxOnHierarchy(x.getComponentDescriptorIndex(comp),
                                     y.getComponentDescriptorIndex(comp),
                                     w,
                                     d_hierarchy,
                                     d_coarsest_ln,
                                     d_finest_ln);
            }
        }
        ++iter_num;
    }

    ADS_TIMER_STOP(t_apply);
    return true;
} // apply

void
CCSharpInterfaceScheduledJacobiSolver::initializeSolverState(const SAMRAIVectorReal<NDIM, double>& in,
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

    // Setup the interpolation transaction information.
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    d_transaction_comps.clear();
    InterpolationTransactionComponent component(d_x->getComponentDescriptorIndex(0),
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

    // Indicate the operator is initialized.
    d_is_initialized = true;
    return;
} // initializeOperatorState

void
CCSharpInterfaceScheduledJacobiSolver::deallocateSolverState()
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
CCSharpInterfaceScheduledJacobiSolver::relaxOnHierarchy(const int x_idx,
                                                        const int y_idx,
                                                        const double w,
                                                        Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                        const int coarsest_ln,
                                                        const int finest_ln)
{
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        const std::vector<ImagePointWeightsMap>& img_wgts_vec = d_ghost_fill->getImagePointWeights(ln);
        int local_patch_num = 0;

        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            double denom = 0.0;
            for (int d = 0; d < NDIM; ++d) denom -= 2.0 / (dx[d] * dx[d]);

            Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_idx);
            Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y_idx);
            CellData<NDIM, double> temp_data(patch->getBox(), x_data->getDepth(), IntVector<NDIM>(0));

            Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(d_ghost_fill->getIndexPatchIndex());
            const ImagePointWeightsMap& img_wgts = img_wgts_vec[local_patch_num];

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                const int idx_val = (*i_data)(idx);
                double new_val = 0.0;
                if (idx_val == FLUID)
                {
                    double val = (*y_data)(idx);
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> one(0);
                        one(d) = 1;
                        val -= ((*x_data)(idx + one) + (*x_data)(idx - one)) / (dx[d] * dx[d]);
                    }
                    new_val = val / denom;
                }
                else if (idx_val == GHOST)
                {
                    const ImagePointWeights& wgts = img_wgts.at(std::make_pair(idx, patch));
                    double val = (*y_data)(idx);
                    double gp_wgt = 1.0;
                    for (int i = 0; i < wgts.s_num_pts; ++i)
                    {
                        if (wgts.d_idxs[i] == idx)
                        {
                            gp_wgt += wgts.d_weights[i];
                        }
                        else
                        {
                            val -= wgts.d_weights[i] * (*x_data)(wgts.d_idxs[i]);
                        }
                    }
                    new_val = val / gp_wgt;
                }
                else if (idx_val == INVALID)
                {
                    new_val = 0.0;
                }
                temp_data(idx) = w * new_val + (1.0 - w) * (*x_data)(idx);
            }

            // Copy over temp data to x_data
            x_data->copy(temp_data);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
