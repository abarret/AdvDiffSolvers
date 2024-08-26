/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/CCSharpInterfaceLaplaceOperator.h"
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
} // namespace

namespace sharp_interface
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

CCSharpInterfaceLaplaceOperator::CCSharpInterfaceLaplaceOperator(const std::string& object_name,
                                                                 Pointer<Database> input_db,
                                                                 SharpInterfaceGhostFill& ghost_fill,
                                                                 std::function<double(const VectorNd&)> bdry_fcn,
                                                                 const bool homogeneous_bc)
    : LaplaceOperator(object_name, homogeneous_bc), d_ghost_fill(&ghost_fill), d_bdry_fcn(bdry_fcn)
{
    // Setup the operator to use default scalar-valued boundary conditions.
    setPhysicalBcCoef(nullptr);

    IBTK_DO_ONCE(t_apply = TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceLaplaceOperator::apply()"););
    return;
} // CCSharpInterfaceLaplaceOperator()

CCSharpInterfaceLaplaceOperator::~CCSharpInterfaceLaplaceOperator()
{
    if (d_is_initialized) deallocateOperatorState();
    return;
} // ~CCSharpInterfaceLaplaceOperator()

void
CCSharpInterfaceLaplaceOperator::apply(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    ADS_TIMER_START(t_apply);
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

    std::function<double(const VectorNd&)> bdry_fcn = d_homogeneous_bc ? [](const VectorNd&) -> double { return 0.0; } :
                                                                         d_bdry_fcn;

    // Loop over comp data
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<InterpolationTransactionComponent> transaction_comps;
        InterpolationTransactionComponent x_component(d_x->getComponentDescriptorIndex(comp),
                                                      x.getComponentDescriptorIndex(comp),
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

        // Fill in ghost cells
        hier_ghost_cell.fillData(d_solution_time);

        Pointer<CellVariable<NDIM, double>> y_cc_var = y.getComponentVariable(comp);
        const int y_idx = y.getComponentDescriptorIndex(comp);
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            applyOnLevel(d_x->getComponentDescriptorIndex(comp), y_idx, ln, d_hierarchy);
        }
    }
    ADS_TIMER_STOP(t_apply);
    return;
} // apply

void
CCSharpInterfaceLaplaceOperator::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& in,
                                                         const SAMRAIVectorReal<NDIM, double>& out)
{
    // Deallocate the operator state if the operator is already initialized.
    if (d_is_initialized) deallocateOperatorState();

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
                                                d_bc_coefs);
    d_transaction_comps.push_back(component);

    // Initialize the interpolation operators.
    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(d_transaction_comps, d_hierarchy, d_coarsest_ln, d_finest_ln);

    // Indicate the operator is initialized.
    d_is_initialized = true;
    return;
} // initializeOperatorState

void
CCSharpInterfaceLaplaceOperator::deallocateOperatorState()
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
CCSharpInterfaceLaplaceOperator::applyOnLevel(const int Q_idx,
                                              const int R_idx,
                                              const int ln,
                                              Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    const std::vector<ImagePointWeightsMap>& img_wgts_vec = d_ghost_fill->getImagePointWeights(ln);
    int local_patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();

        Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
        Pointer<CellData<NDIM, double>> R_data = patch->getPatchData(R_idx);
        R_data->fillAll(0.0);
        Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(d_ghost_fill->getIndexPatchIndex());
        const ImagePointWeightsMap& img_wgts = img_wgts_vec[local_patch_num];

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const int idx_val = (*i_data)(idx);
            if (idx_val == FLUID)
            {
                for (int d = 0; d < NDIM; ++d)
                {
                    IntVector<NDIM> one(0);
                    one(d) = 1;
                    (*R_data)(idx) +=
                        ((*Q_data)(idx + one) - 2.0 * (*Q_data)(idx) + (*Q_data)(idx - one)) / (dx[d] * dx[d]);
                }
            }
            else if (idx_val == GHOST)
            {
                (*R_data)(idx) = (*Q_data)(idx);
                const ImagePointWeights& wgts = img_wgts.at(std::make_pair(idx, patch));
                for (int i = 0; i < wgts.s_num_pts; ++i)
                    (*R_data)(idx) += wgts.d_weights[i] * (*Q_data)(wgts.d_idxs[i]);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
