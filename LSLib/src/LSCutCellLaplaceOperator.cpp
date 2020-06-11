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
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

LSCutCellLaplaceOperator::LSCutCellLaplaceOperator(const std::string& object_name,
                                                   Pointer<Database> input_db,
                                                   const bool homogeneous_bc)
    : LaplaceOperator(object_name, homogeneous_bc)
{
    // Setup the operator to use default scalar-valued boundary conditions.
    setPhysicalBcCoef(nullptr);

    d_robin_bdry = input_db->getBoolWithDefault("robin_boundary", d_robin_bdry);
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

    // Simultaneously fill ghost cell values for all components.
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<InterpolationTransactionComponent> transaction_comps;
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        InterpolationTransactionComponent x_component(x.getComponentDescriptorIndex(comp),
                                                      DATA_REFINE_TYPE,
                                                      true,
                                                      DATA_COARSEN_TYPE,
                                                      BDRY_EXTRAP_TYPE,
                                                      CONSISTENT_TYPE_2_BDRY,
                                                      d_bc_coefs,
                                                      d_fill_pattern);
        transaction_comps.push_back(x_component);
    }

    d_hier_bdry_fill->resetTransactionComponents(transaction_comps);
    d_hier_bdry_fill->setHomogeneousBc(d_homogeneous_bc);
    d_hier_bdry_fill->fillData(d_solution_time);
    d_hier_bdry_fill->resetTransactionComponents(d_transaction_comps);

    // Compute the action of the operator.
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        Pointer<CellVariable<NDIM, double>> x_cc_var = x.getComponentVariable(comp);
        Pointer<CellVariable<NDIM, double>> y_cc_var = y.getComponentVariable(comp);
        const int x_idx = x.getComponentDescriptorIndex(comp);
        const int y_idx = y.getComponentDescriptorIndex(comp);
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_idx);
                Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y_idx);
                computeHelmholtzAction(*x_data, *y_data, *patch);
            }
        }
    }
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
    d_fill_pattern = nullptr;
    if (d_poisson_spec.dIsConstant())
    {
        d_fill_pattern = new CellNoCornersFillPattern(CELLG, false, false, true);
    }
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    d_transaction_comps.clear();
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        InterpolationTransactionComponent component(d_x->getComponentDescriptorIndex(comp),
                                                    DATA_REFINE_TYPE,
                                                    false,
                                                    DATA_COARSEN_TYPE,
                                                    BDRY_EXTRAP_TYPE,
                                                    CONSISTENT_TYPE_2_BDRY,
                                                    d_bc_coefs,
                                                    d_fill_pattern);
        d_transaction_comps.push_back(component);
    }

    // Initialize the interpolation operators.
    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(d_transaction_comps, d_hierarchy, d_coarsest_ln, d_finest_ln);

    // Indicate the operator is initialized.
    d_is_initialized = true;
    return;
} // initializeOperatorState

void
LSCutCellLaplaceOperator::deallocateOperatorState()
{
    if (!d_is_initialized) return;

    // Deallocate the interpolation operators.
    d_hier_bdry_fill->deallocateOperatorState();
    d_hier_bdry_fill.setNull();
    d_transaction_comps.clear();
    d_fill_pattern.setNull();

    // Deallocate hierarchy math operations object.
    if (!d_hier_math_ops_external) d_hier_math_ops.setNull();

    // Delete the solution and rhs vectors.
    d_x->freeVectorComponents();
    d_x.setNull();

    d_b->freeVectorComponents();
    d_b.setNull();

    // Indicate that the operator is NOT initialized.
    d_is_initialized = false;
    return;
} // deallocateOperatorState

/////////////////////////////// PRIVATE //////////////////////////////////////

void
LSCutCellLaplaceOperator::computeHelmholtzAction(const CellData<NDIM, double>& Q_data,
                                                 CellData<NDIM, double>& R_data,
                                                 const Patch<NDIM>& patch)
{
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
    // Fix up for boundary conditions
    if (d_robin_bdry)
    {
        for (CellIterator<NDIM> ci(box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double area = (*area_data)(idx);
            double cell_volume = (*vol_data)(idx)*dx[0] * dx[1];
            if (MathUtilities<double>::equalEps(cell_volume, 0.0))
            {
                cell_volume = dx[0] * dx[1];
            }
            if (!MathUtilities<double>::equalEps(D, 0.0))
            {
                // set g and a
                double a = 1.0;
                double R = 1.0;
                double D_val = 0.1;
                double g = 5.0 * (a - R + 2.0 * a * d_solution_time) /
                           (D_val * std::exp(R * R / (2.0 * D_val + 4.0 * D_val * d_solution_time)) *
                            (1.0 + 2.0 * d_solution_time) * (1.0 + 2.0 * d_solution_time));
                const double sgn = D / std::abs(D);
                if (area > 0.0)
                {
                    for (unsigned int l = 0; l < d_bc_coefs.size(); ++l)
                    {
                        if (!d_homogeneous_bc) R_data(idx, l) += 0.5 * sgn * g * area / cell_volume;
                        R_data(idx, l) -= 0.5 * sgn * a * Q_data(idx, l) * area / cell_volume;
                    }
                }
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////

} // namespace LS

//////////////////////////////////////////////////////////////////////////////
