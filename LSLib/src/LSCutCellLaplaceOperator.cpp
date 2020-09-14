/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/DebuggingUtilities.h"
#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"
#include "ibtk/namespaces.h" // IWYU pragma: keep

#include "LS/LSCutCellLaplaceOperator.h"
#include "LS/utility_functions.h"

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

#include <ostream>
#include <string>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace LS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Number of ghosts cells used for each variable quantity.
static const int CELLG = 1;

// Types of refining and coarsening to perform prior to setting coarse-fine
// boundary and physical boundary ghost cell values.
static const std::string DATA_REFINE_TYPE = "CONSERVATIVE_LINEAR_REFINE";
static const std::string DATA_COARSEN_TYPE = "CUBIC_COARSEN";

// Type of extrapolation to use at physical boundaries.
static const std::string BDRY_EXTRAP_TYPE = "LINEAR";

// Whether to enforce consistent interpolated values at Type 2 coarse-fine
// interface ghost cells.
static const bool CONSISTENT_TYPE_2_BDRY = false;

static Timer* t_cache_l2;
static Timer* t_compute_helmholtz;
static Timer* t_extrapolate;
static Timer* t_apply;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

LSCutCellLaplaceOperator::LSCutCellLaplaceOperator(const std::string& object_name,
                                                   Pointer<Database> input_db,
                                                   const bool homogeneous_bc)
    : LaplaceOperator(object_name, homogeneous_bc), d_Q_var(new CellVariable<NDIM, double>(d_object_name + "::ScrVar"))
{
    // Setup the operator to use default scalar-valued boundary conditions.
    setPhysicalBcCoef(nullptr);

    d_robin_bdry = input_db->getBoolWithDefault("robin_boundary", d_robin_bdry);
    d_cache_bdry = input_db->getBool("cache_boundary");
    d_using_rbf = input_db->getBool("using_rbf");

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_var, var_db->getContext(d_object_name + "::SCRATCH"), CELLG);

    IBAMR_DO_ONCE(t_cache_l2 =
                      TimerManager::getManager()->getTimer("LS::LSCutCellLaplaceOperator::cacheLeastSquaresData()");
                  t_compute_helmholtz =
                      TimerManager::getManager()->getTimer("LS::LSCutCellLaplaceOperator::computeHelmholtzAction()");
                  t_extrapolate =
                      TimerManager::getManager()->getTimer("LS::LSCutCellLaplaceOperator::extrapolateToCellCenters()");
                  t_apply = TimerManager::getManager()->getTimer("LS::LSCutCellLaplaceOperator::apply()"););
    return;
} // LSCutCellLaplaceOperator()

LSCutCellLaplaceOperator::~LSCutCellLaplaceOperator()
{
    if (d_is_initialized) deallocateOperatorState();
    return;
} // ~LSCutCellLaplaceOperator()

void
LSCutCellLaplaceOperator::apply(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    LS_TIMER_START(t_apply);
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
        if (x_depth != d_bc_coefs.size() || y_depth != d_bc_coefs.size())
        {
            TBOX_ERROR(d_object_name << "::apply()\n"
                                     << "  each vector component must have data depth == " << d_bc_coefs.size() << "\n"
                                     << "  since d_bc_coefs.size() == " << d_bc_coefs.size() << std::endl);
        }
    }
#endif

    // Loop over comp data
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        {
            using InterpolationTransactionComponent =
                HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<InterpolationTransactionComponent> transaction_comps;
            InterpolationTransactionComponent x_component(x.getComponentDescriptorIndex(comp),
                                                          DATA_REFINE_TYPE,
                                                          false,
                                                          DATA_COARSEN_TYPE,
                                                          BDRY_EXTRAP_TYPE,
                                                          CONSISTENT_TYPE_2_BDRY,
                                                          d_bc_coefs);
            transaction_comps.push_back(x_component);
            HierarchyGhostCellInterpolation hier_ghost_cell;
            hier_ghost_cell.initializeOperatorState(transaction_comps, d_hierarchy);
            hier_ghost_cell.setHomogeneousBc(d_homogeneous_bc);
            hier_ghost_cell.fillData(d_solution_time);
        }

        extrapolateToCellCenters(x.getComponentDescriptorIndex(comp), d_Q_scr_idx);
        d_hier_bdry_fill->setHomogeneousBc(d_homogeneous_bc);
        d_hier_bdry_fill->fillData(d_solution_time);

        // Compute the action of the operator.
        for (int comp = 0; comp < d_ncomp; ++comp)
        {
            Pointer<CellVariable<NDIM, double>> y_cc_var = y.getComponentVariable(comp);
            const int y_idx = y.getComponentDescriptorIndex(comp);
            for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
                for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(d_Q_scr_idx);
                    Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y_idx);
                    computeHelmholtzAction(*x_data, *y_data, *patch);
                }
            }

            if (d_robin_bdry)
            {
                TBOX_ASSERT(d_bdry_conds);
                d_bdry_conds->setTimeStepType(d_ts_type);
                d_bdry_conds->setLSData(d_ls_var, d_ls_idx, d_vol_var, d_vol_idx, d_area_var, d_area_idx);
                d_bdry_conds->setHomogeneousBdry(d_homogeneous_bc);
                d_bdry_conds->setDiffusionCoefficient(d_poisson_spec.getDConstant());
                d_bdry_conds->applyBoundaryCondition(
                    d_Q_var, d_Q_scr_idx, y_cc_var, y_idx, d_hierarchy, d_solution_time);
            }
        }
    }
    LS_TIMER_STOP(t_apply);
    return;
} // apply

void
LSCutCellLaplaceOperator::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& in,
                                                  const SAMRAIVectorReal<NDIM, double>& out)
{
    // Deallocate the operator state if the operator is already initialized.
    if (d_is_initialized) deallocateOperatorState();

    // Setup solution and rhs vectors.
    d_x = in.cloneVector(in.getName());
    d_b = out.cloneVector(out.getName());

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
    InterpolationTransactionComponent component(
        d_Q_scr_idx, DATA_REFINE_TYPE, false, DATA_COARSEN_TYPE, BDRY_EXTRAP_TYPE, CONSISTENT_TYPE_2_BDRY, d_bc_coefs);
    d_transaction_comps.push_back(component);

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_Q_scr_idx)) level->allocatePatchData(d_Q_scr_idx);
    }

    // Initialize the interpolation operators.
    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(d_transaction_comps, d_hierarchy, d_coarsest_ln, d_finest_ln);
    d_update_weights = true;

    if (d_bdry_conds) d_bdry_conds->allocateOperatorState(d_hierarchy, d_current_time);

    // Indicate the operator is initialized.
    d_is_initialized = true;
    return;
} // initializeOperatorState

void
LSCutCellLaplaceOperator::deallocateOperatorState()
{
    if (!d_is_initialized) return;

    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_Q_scr_idx)) level->deallocatePatchData(d_Q_scr_idx);
    }

    // Deallocate the interpolation operators.
    d_hier_bdry_fill->deallocateOperatorState();
    d_hier_bdry_fill.setNull();
    d_transaction_comps.clear();

    // Deallocate hierarchy math operations object.
    if (!d_hier_math_ops_external) d_hier_math_ops.setNull();

    // Delete the solution and rhs vectors.
    d_x->freeVectorComponents();
    d_x.setNull();

    d_b->freeVectorComponents();
    d_b.setNull();

    // Free any preallocated matrices
    for (std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_matrix_map : d_qr_matrix_vec)
        qr_matrix_map.clear();
    d_update_weights = true;

    if (d_bdry_conds) d_bdry_conds->deallocateOperatorState(d_hierarchy, d_current_time);

    // Indicate that the operator is NOT initialized.
    d_is_initialized = false;
    return;
} // deallocateOperatorState

void
LSCutCellLaplaceOperator::cacheLeastSquaresData()
{
    LS_TIMER_START(t_cache_l2);
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    // Free any preallocated matrices
    for (std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_matrix_map : d_qr_matrix_vec)
        qr_matrix_map.clear();

    // allocate matrix data
    d_qr_matrix_vec.resize(finest_ln + 1);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_map = d_qr_matrix_vec[ln];
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // We are on a cut cell. We need to interpolate to cell center
                    VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = static_cast<double>(idx(d)) + 0.5;
                    int size = 1 + NDIM;
                    int box_size = 1;
                    Box<NDIM> box(idx, idx);
                    box.grow(box_size);
#ifndef NDEBUG
                    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
                    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif

                    const CellIndex<NDIM>& idx_low = patch->getBox().lower();
                    std::vector<VectorNd> X_vals;

                    for (CellIterator<NDIM> ci(box); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx_c = ci();
                        if ((*vol_data)(idx_c) > 0.0)
                        {
                            // Use this point to calculate least squares reconstruction.
                            VectorNd x_cent_c = find_cell_centroid(idx_c, *ls_data);
                            X_vals.push_back(x_cent_c);
                        }
                    }
                    const int m = X_vals.size();
                    MatrixXd A(MatrixXd::Zero(m, size));
                    for (size_t i = 0; i < X_vals.size(); ++i)
                    {
                        const VectorNd X = X_vals[i] - x_loc;
                        double w = std::sqrt(weight(static_cast<double>(X.norm())));
                        A(i, 2) = w * X[1];
                        A(i, 1) = w * X[0];
                        A(i, 0) = w * 1.0;
                    }
                    PatchIndexPair p_idx = PatchIndexPair(patch, idx);

#ifndef NDEBUG
                    if (qr_map.find(p_idx) == qr_map.end())
                        qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(A);
                    else
                        TBOX_WARNING("Already had a QR decomposition in place");
#else
                    qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(A);
#endif
                }
            }
        }
    }
    d_update_weights = false;
    LS_TIMER_STOP(t_cache_l2);
}

void
LSCutCellLaplaceOperator::cacheRBFData()
{
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    // Free any preallocated matrices
    for (std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_matrix_map : d_qr_matrix_vec)
        qr_matrix_map.clear();

    // allocate matrix data
    d_qr_matrix_vec.resize(finest_ln + 1);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_map = d_qr_matrix_vec[ln];
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            const Box<NDIM>& box = patch->getBox();
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // We are on a cut cell. We need to interpolate to cell center
                    VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = static_cast<double>(idx(d)) + 0.5;
                    int box_size = 1;
                    Box<NDIM> box(idx, idx);
                    box.grow(box_size);
#ifndef NDEBUG
                    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
                    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif

                    const CellIndex<NDIM>& idx_low = patch->getBox().lower();
                    std::vector<VectorNd> X_vals;

                    for (CellIterator<NDIM> ci(box); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx_c = ci();
                        if ((*vol_data)(idx_c) > 0.0)
                        {
                            // Use this point to calculate least squares reconstruction.
                            VectorNd x_cent_c = find_cell_centroid(idx_c, *ls_data);
                            X_vals.push_back(x_cent_c);
                        }
                    }
                    const int m = X_vals.size();
                    MatrixXd A(MatrixXd::Zero(m, m));
                    MatrixXd B(MatrixXd::Zero(m, NDIM + 1));
                    for (size_t i = 0; i < X_vals.size(); ++i)
                    {
                        for (size_t j = 0; j < X_vals.size(); ++j)
                        {
                            const VectorNd X = X_vals[i] - X_vals[j];
                            const double phi = rbf(X.norm());
                            A(i, j) = phi;
                        }
                        B(i, 0) = 1.0;
                        for (int d = 0; d < NDIM; ++d) B(i, d + 1) = X_vals[i](d);
                    }
                    PatchIndexPair p_idx = PatchIndexPair(patch, idx);

                    MatrixXd final_mat(MatrixXd::Zero(m + NDIM + 1, m + NDIM + 1));
                    final_mat.block(0, 0, m, m) = A;
                    final_mat.block(0, m, m, NDIM + 1) = B;
                    final_mat.block(m, 0, NDIM + 1, m) = B.transpose();

#ifndef NDEBUG
                    if (qr_map.find(p_idx) == qr_map.end())
                        qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
                    else
                        TBOX_WARNING("Already had a QR decomposition in place");
#else
                    qr_map[p_idx] = FullPivHouseholderQR<MatrixXd>(final_mat);
#endif
                }
            }
        }
    }
    d_update_weights = false;
}

/////////////////////////////// PRIVATE //////////////////////////////////////

void
LSCutCellLaplaceOperator::computeHelmholtzAction(const CellData<NDIM, double>& Q_data,
                                                 CellData<NDIM, double>& R_data,
                                                 const Patch<NDIM>& patch)
{
    LS_TIMER_START(t_compute_helmholtz);
    const Box<NDIM>& box = patch.getBox();
    const Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch.getPatchGeometry();
    const double* const dx = pgeom->getDx();

    const double C = d_poisson_spec.getCConstant();
    const double D =
        d_poisson_spec.dIsConstant() ? d_poisson_spec.getDConstant() : std::numeric_limits<double>::quiet_NaN();

    Pointer<NodeData<NDIM, double>> phi_n_data = patch.getPatchData(d_ls_idx);
    Pointer<FaceData<NDIM, double>> D_data =
        d_poisson_spec.dIsConstant() ? nullptr : patch.getPatchData(d_poisson_spec.getDPatchDataId());
    Pointer<CellData<NDIM, double>> area_data = patch.getPatchData(d_area_idx);
    Pointer<CellData<NDIM, double>> vol_data = patch.getPatchData(d_vol_idx);

    IntVector<NDIM> xp(1, 0), yp(0, 1);
#if (NDIM == 3)
    IntVector<NDIM> zp(0, 0, 1);
#endif
    for (CellIterator<NDIM> ci(box); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        double cell_volume = (*vol_data)(idx)*dx[0] * dx[1];
        if (MathUtilities<double>::equalEps(cell_volume, 0.0))
        {
            for (unsigned int l = 0; l < d_bc_coefs.size(); ++l) R_data(idx, l) = 0.0;
            continue;
        }
        double Q_avg = (Q_data)(idx);
        for (unsigned int l = 0; l < d_bc_coefs.size(); ++l)
        {
            R_data(idx, l) = 0.0;
            // TODO: For NDIM == 3 Update length_fraction to calculate partial face areas instead of
            // lengths.
            // Loop through X faces
            for (int f = 0; f < 2; ++f)
            {
                const int sgn = f == 0 ? -1 : 1;
                double L = length_fraction(dx[1],
                                           (*phi_n_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 0))),
                                           (*phi_n_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(f, 1))));
                double Q_next = Q_data(idx + xp * sgn);
                double vol_next = (*vol_data)(idx + xp * sgn);
                if (MathUtilities<double>::equalEps(vol_next, 0.0)) continue;
                double dudx = (Q_next - Q_avg) / dx[0];
                R_data(idx, l) += D * L * dudx / cell_volume;
            }
            // Loop through Y faces
            for (int f = 0; f < 2; ++f)
            {
                const int sgn = f == 0 ? -1 : 1;
                double L = length_fraction(dx[0],
                                           (*phi_n_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(0, f))),
                                           (*phi_n_data)(NodeIndex<NDIM>(idx, IntVector<NDIM>(1, f))));
                double Q_next = Q_data(idx + yp * sgn);
                double vol_next = (*vol_data)(idx + yp * sgn);
                if (MathUtilities<double>::equalEps(vol_next, 0.0)) continue;
                double dudy = (Q_next - Q_avg) / dx[1];
                R_data(idx, l) += D * L * dudy / cell_volume;
            }
#if (NDIM == 3)
            // Loop through Z faces
            TBOX_ERROR("3 Spatial dimensions not currently supported");
#endif
            // Add C constant
            R_data(idx, l) += C * Q_data(idx, l);
        }
    }
    LS_TIMER_STOP(t_compute_helmholtz);
    return;
}

void
LSCutCellLaplaceOperator::extrapolateToCellCenters(const int Q_idx, const int R_idx)
{
    LS_TIMER_START(t_extrapolate);
    if (d_update_weights)
    {
        if (d_using_rbf)
            cacheRBFData();
        else
            cacheLeastSquaresData();
    }
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        std::map<PatchIndexPair, FullPivHouseholderQR<MatrixXd>>& qr_matrix_map = d_qr_matrix_vec[ln];
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();

            Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
            Pointer<CellData<NDIM, double>> R_data = patch->getPatchData(R_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

            R_data->copy(*Q_data);

            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // We are on a cut cell. We need to interpolate to cell center
                    VectorNd x_loc;
                    for (int d = 0; d < NDIM; ++d) x_loc(d) = static_cast<double>(idx(d)) + 0.5;
                    int box_size = 1;
                    Box<NDIM> box(idx, idx);
                    box.grow(box_size);
#ifndef NDEBUG
                    TBOX_ASSERT(ls_data->getGhostBox().contains(box));
                    TBOX_ASSERT(Q_data->getGhostBox().contains(box));
                    TBOX_ASSERT(vol_data->getGhostBox().contains(box));
#endif

                    const CellIndex<NDIM>& idx_low = patch->getBox().lower();

                    std::vector<double> Q_vals;
                    std::vector<VectorNd> X_vals;

                    for (CellIterator<NDIM> ci(box); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx_c = ci();
                        if ((*vol_data)(idx_c) > 0.0)
                        {
                            // Use this point to calculate least squares reconstruction.
                            // Find cell center
                            VectorNd x_cent_c = find_cell_centroid(idx_c, *ls_data);
                            Q_vals.push_back((*Q_data)(idx_c));
                            X_vals.push_back(x_cent_c);
                        }
                    }
                    const int m = Q_vals.size();
                    VectorXd U(VectorXd::Zero(m + (d_using_rbf ? NDIM + 1 : 0)));
                    for (size_t i = 0; i < Q_vals.size(); ++i)
                    {
                        if (d_using_rbf)
                        {
                            U(i) = Q_vals[i];
                        }
                        else
                        {
                            VectorNd X = X_vals[i] - x_loc;
                            U(i) = Q_vals[i] * std::sqrt(weight(static_cast<double>(X.norm())));
                        }
                    }

                    VectorXd x1 = qr_matrix_map[PatchIndexPair(patch, idx)].solve(U);
                    if (d_using_rbf)
                    {
                        VectorXd rbf_coefs = x1.block(0, 0, m, 1);
                        VectorXd poly_coefs = x1.block(m, 0, NDIM + 1, 1);
                        Vector3d poly_vec = { 1.0, x_loc(0), x_loc(1) };
                        double val = 0.0;
                        for (size_t i = 0; i < X_vals.size(); ++i)
                        {
                            val += rbf_coefs[i] * rbf((X_vals[i] - x_loc).norm());
                        }
                        val += poly_coefs.dot(poly_vec);
                        (*R_data)(idx) = val;
                    }
                    else
                    {
                        (*R_data)(idx) = x1(0);
                    }
                }
            }
        }
    }
    LS_TIMER_STOP(t_extrapolate);
}

//////////////////////////////////////////////////////////////////////////////

} // namespace LS

//////////////////////////////////////////////////////////////////////////////
