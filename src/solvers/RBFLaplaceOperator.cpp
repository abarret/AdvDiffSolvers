// ---------------------------------------------------------------------
//
// Copyright (c) 2021 - 2021 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/KDTree.h"
#include "ADS/PolynomialBasis.h"
#include "ADS/RBFLaplaceOperator.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"
#include "ADS/reconstructions.h"

#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/IBTK_CHKERRQ.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/ibtk_utilities.h"

#include "CellVariable.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "PoissonSpecifications.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Timer.h"

#include <Eigen/Dense>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Types of refining and coarsening to perform prior to setting coarse-fine
// boundary and physical boundary ghost cell values.
static const std::string DATA_REFINE_TYPE = "NONE";
static const bool USE_CF_INTERPOLATION = true;
static const std::string DATA_COARSEN_TYPE = "CUBIC_COARSEN";

// Type of extrapolation to use at physical boundaries.
static const std::string BDRY_EXTRAP_TYPE = "LINEAR";

// Whether to enforce consistent interpolated values at Type 2 coarse-fine
// interface ghost cells.
static const bool CONSISTENT_TYPE_2_BDRY = false;

// Timers.
static Timer* t_apply;
static Timer* t_initialize_operator_state;
static Timer* t_deallocate_operator_state;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

RBFLaplaceOperator::RBFLaplaceOperator(const std::string& object_name,
                                       std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                                       const std::string& sys_name,
                                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                                       Pointer<Database> input_db)
    : PETScAugmentedLinearOperator(object_name, false),
      d_hierarchy(hierarchy),
      d_fe_mesh_partitioner(fe_mesh_partitioner),
      d_sys_name(sys_name)
{
    d_fd_weights =
        libmesh_make_unique<RBFFDWeightsCache>(d_object_name + "::Weights", fe_mesh_partitioner, d_hierarchy, input_db);
    d_dist_to_bdry = input_db->getDouble("dist_to_bdry");
    d_C = input_db->getDouble("C");
    d_D = input_db->getDouble("D");
    // Make a function for the weights
    d_rbf = [](const double r) -> double { return PolynomialBasis::pow(r, 5); };
    d_lap_rbf = [](const double r) -> double {
#if (NDIM == 2)
        // return PolynomialBasis::pow(r, 5) + 25.0 * PolynomialBasis::pow(r, 4);
        return 25.0 * PolynomialBasis::pow(r, 4);
#endif
#if (NDIM == 3)
        // return PolynomialBasis::pow(r, 5) + 30.0 * PolynomialBasis::pow(r, 4);
        return 30.0 * PolynomialBasis::pow(r, 4);
#endif
    };

    d_polys = [this](const std::vector<VectorNd>& vec, int degree) -> MatrixXd {
        // return d_C * PolynomialBasis::formMonomials(vec, degree) - d_D * PolynomialBasis::laplacianMonomials(vec,
        // degree);
        return d_D * PolynomialBasis::laplacianMonomials(vec, degree);
    };

    d_fd_weights->registerPolyFcn(d_polys, d_rbf, d_lap_rbf);
    // Setup Timers.
    IBTK_DO_ONCE(t_apply = TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::apply()");
                 t_initialize_operator_state =
                     TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::initializeOperatorState()");
                 t_deallocate_operator_state =
                     TimerManager::getManager()->getTimer("IBTK::LaplaceOperator::deallocateOperatorState()"););
    return;
} // LaplaceOperator()

RBFLaplaceOperator::~RBFLaplaceOperator()
{
    if (d_is_initialized)
    {
        this->deallocateOperatorState();
    }
    return;
} // ~LaplaceOperator()

void
RBFLaplaceOperator::setupBeforeApply(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
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
#if (0)
        if (x_depth != d_bc_coefs.size() || y_depth != d_bc_coefs.size())
        {
            TBOX_ERROR(d_object_name << "::apply()\n"
                                     << "  each vector component must have data depth == " << d_bc_coefs.size() << "\n"
                                     << "  since d_bc_coefs.size() == " << d_bc_coefs.size() << std::endl);
        }
#endif
    }
#endif

    // Simultaneously fill ghost cell values for all components.
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<InterpolationTransactionComponent> transaction_comps;
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        InterpolationTransactionComponent x_component(x.getComponentDescriptorIndex(comp),
                                                      DATA_REFINE_TYPE,
                                                      USE_CF_INTERPOLATION,
                                                      DATA_COARSEN_TYPE,
                                                      BDRY_EXTRAP_TYPE,
                                                      CONSISTENT_TYPE_2_BDRY);
        transaction_comps.push_back(x_component);
    }
    d_hier_bdry_fill->resetTransactionComponents(transaction_comps);
    d_hier_bdry_fill->setHomogeneousBc(d_homogeneous_bc);
    d_hier_bdry_fill->fillData(d_solution_time);
    d_hier_bdry_fill->resetTransactionComponents(d_transaction_comps);
}

void
RBFLaplaceOperator::fillBdryConds()
{
    // For this case, the boundary points are degrees of freedom (despite using Dirichlet bcs)
    // The action of this operator should be the identity
    int ierr = VecCopy(d_aug_x_vec, d_aug_y_vec);
    IBTK_CHKERRQ(ierr);
}

void
RBFLaplaceOperator::apply(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    IBTK_TIMER_START(t_apply);

    setupBeforeApply(x, y);
    fillBdryConds();
    // Compute the action of the operator.
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            for (int comp = 0; comp < d_ncomp; ++comp)
            {
                Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x.getComponentDescriptorIndex(comp));
                Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y.getComponentDescriptorIndex(comp));

                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double ls_val = ADS::node_to_cell(idx, *ls_data);
                    // Check if ls_val is within some specified tolerance of the boundary
                    if (ls_val < 0.0 && std::abs(ls_val) >= d_dist_to_bdry)
                    {
                        // We want to use standard finite differences.
                        double lap = 0.0;
                        for (int dir = 0; dir < NDIM; ++dir)
                        {
                            IntVector<NDIM> dirs(0);
                            dirs(dir) = 1;
                            lap += ((*x_data)(idx + dirs) - 2.0 * (*x_data)(idx) + (*x_data)(idx - dirs)) /
                                   (dx[dir] * dx[dir]);
                        }
                        (*y_data)(idx) = d_C * (*x_data)(idx)-d_D * lap;
                    }
                    else
                    {
                        // Not part of the FD domain. Copy a random value. Note that we're applying RBF-FD weights
                        // later, so this value can be overwritten.
                        (*y_data)(idx) = (*x_data)(idx);
                    }
                }
            }
        }
    }

    applyToLagDOFs(x.getComponentDescriptorIndex(0), y.getComponentDescriptorIndex(0));
    IBTK_TIMER_STOP(t_apply);
    return;
} // apply

void
RBFLaplaceOperator::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& in,
                                            const SAMRAIVectorReal<NDIM, double>& out)
{
    IBTK_TIMER_START(t_initialize_operator_state);

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

    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    d_transaction_comps.clear();
    for (int comp = 0; comp < d_ncomp; ++comp)
    {
        InterpolationTransactionComponent component(d_x->getComponentDescriptorIndex(comp),
                                                    DATA_REFINE_TYPE,
                                                    USE_CF_INTERPOLATION,
                                                    DATA_COARSEN_TYPE,
                                                    BDRY_EXTRAP_TYPE,
                                                    CONSISTENT_TYPE_2_BDRY);
        d_transaction_comps.push_back(component);
    }

    // Initialize the interpolation operators.
    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(d_transaction_comps, d_hierarchy, d_coarsest_ln, d_finest_ln);

    d_fd_weights->setLS(d_ls_idx);
    d_fd_weights->findRBFFDWeights();

    // Clone Lag Vector
    int ierr = VecDuplicate(d_aug_x_vec, &d_aug_y_vec);
    IBTK_CHKERRQ(ierr);

    // Indicate the operator is initialized.
    d_is_initialized = true;

    IBTK_TIMER_STOP(t_initialize_operator_state);
    return;
} // initializeOperatorState

void
RBFLaplaceOperator::deallocateOperatorState()
{
    if (!d_is_initialized) return;

    IBTK_TIMER_START(t_deallocate_operator_state);

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

    d_fd_weights->clearCache();

    IBTK_TIMER_STOP(t_deallocate_operator_state);
    return;
} // deallocateOperatorState

/////////////////////////////// PRIVATE //////////////////////////////////////
void
RBFLaplaceOperator::applyToLagDOFs(const int x_idx, const int y_idx)
{
    // Now apply the RBF reconstruction operator near the boundary mesh
    // Assume structure is on finest level
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    unsigned int patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const std::vector<UPoint>& base_pts = d_fd_weights->getRBFFDBasePoints(patch);
        if (base_pts.empty()) continue;
        const std::vector<std::vector<UPoint>>& rbf_pts = d_fd_weights->getRBFFDPoints(patch);
        const std::vector<std::vector<double>>& weights = d_fd_weights->getRBFFDWeights(patch);

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        Pointer<CellData<NDIM, double>> x_data = patch->getPatchData(x_idx);
        Pointer<CellData<NDIM, double>> y_data = patch->getPatchData(y_idx);
        for (size_t idx = 0; idx < base_pts.size(); ++idx)
        {
            const UPoint& pt = base_pts[idx];
            if (pt.isNode())
                continue;
            const int interp_size = weights[idx].size();
            double lap = 0.0;
            for (int i = 0; i < interp_size; ++i)
            {
                double w = weights[idx][i];
#if (NDIM == 2)
                double denom = dx[0] * dx[1];
#endif
#if (NDIM == 3)
                // TODO: Determine the correct scaling. This is determined by how we scale points when calculating
                // weights.
                double denom = dx[0] * dx[1];
#endif
                lap += w * getSolVal(rbf_pts[idx][i], *x_data, d_aug_x_vec) / denom;
            }
            //            setSolVal(lap, pt, *y_data, d_aug_y_vec);
            setSolVal(d_C * getSolVal(pt, *x_data, d_aug_x_vec) - lap, pt, *y_data, d_aug_y_vec);
        }
    }
    // We've set values, so we need to assemble the vector
    int ierr = VecAssemblyBegin(d_aug_y_vec);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyEnd(d_aug_y_vec);
    IBTK_CHKERRQ(ierr);
}

double
RBFLaplaceOperator::getSolVal(const UPoint& pt, const CellData<NDIM, double>& Q_data, Vec& vec) const
{
    double val = 0.0;
    if (pt.isNode())
    {
        // We're on a node. Need to grab value from augmented vec
        std::vector<unsigned int> dof_indices;
        EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
        const System& sys = eq_sys->get_system(d_sys_name);
        const DofMap& dof_map = sys.get_dof_map();
        dof_map.dof_indices(pt.getNode(), dof_indices);
        auto idxs = reinterpret_cast<PetscInt*>(dof_indices.data());
        int ierr = VecGetValues(vec, 1, idxs, &val);
        IBTK_CHKERRQ(ierr);
    }
#ifndef NDEBUG
    else if (pt.isEmpty())
    {
        TBOX_ERROR("Point should not be empty");
    }
#endif
    else
    {
        val = Q_data(pt.getIndex());
    }
    return val;
}

void
RBFLaplaceOperator::setSolVal(const double val, const UPoint& pt, CellData<NDIM, double>& Q_data, Vec& vec) const
{
    if (pt.isNode())
    {
        // We're on a node. Need to grab value from augmented vec
        std::vector<unsigned int> dof_indices;
        EquationSystems* eq_sys = d_fe_mesh_partitioner->getEquationSystems();
        const System& sys = eq_sys->get_system(d_sys_name);
        const DofMap& dof_map = sys.get_dof_map();
        dof_map.dof_indices(pt.getNode(), dof_indices);
        int ierr = VecSetValue(vec, dof_indices[0], val, INSERT_VALUES);
        IBTK_CHKERRQ(ierr);
    }
#ifndef NDEBUG
    else if (pt.isEmpty())
    {
        TBOX_ERROR("Point should not be empty");
    }
#endif
    else
    {
        Q_data(pt.getIndex()) = val;
    }
}
//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
