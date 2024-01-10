#include <ADS/ExtrapolatedAdvDiffHierarchyIntegrator.h>
#include <ADS/InternalBdryFill.h>
#include <ADS/ReinitializeLevelSet.h>
#include <ADS/ads_utilities.h>

#include "ibamr/ConvectiveOperator.h"
#include "ibamr/ibamr_enums.h"
#include "ibamr/ibamr_utilities.h"
#include "ibamr/namespaces.h" // IWYU pragma: keep

#include "ibtk/IBTK_MPI.h"
#include <ibtk/LaplaceOperator.h>
#include <ibtk/PoissonSolver.h>

#include "CartesianGridGeometry.h"
#include "CartesianPatchGeometry.h"
#include "CellVariable.h"
#include "FaceData.h"
#include "FaceVariable.h"
#include "GriddingAlgorithm.h"
#include "HierarchyCellDataOpsReal.h"
#include "HierarchyDataOpsManager.h"
#include "HierarchyFaceDataOpsReal.h"
#include "HierarchySideDataOpsReal.h"
#include "IntVector.h"
#include "MultiblockDataTranslator.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "PoissonSpecifications.h"
#include "SideVariable.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "tbox/MathUtilities.h"
#include "tbox/PIO.h"
#include "tbox/RestartManager.h"
#include "tbox/Utilities.h"

#include <algorithm>
#include <ostream>
#include <utility>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{

/////////////////////////////// PUBLIC ///////////////////////////////////////

ExtrapolatedAdvDiffHierarchyIntegrator::ExtrapolatedAdvDiffHierarchyIntegrator(const std::string& object_name,
                                                                               Pointer<Database> input_db,
                                                                               bool register_for_restart)
    : AdvDiffSemiImplicitHierarchyIntegrator(object_name, input_db, register_for_restart),
      d_valid_var(new NodeVariable<NDIM, int>(d_object_name + "::ValidVar"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_valid_idx = var_db->registerVariableAndContext(d_valid_var, var_db->getContext(d_object_name + "::SCR"));

    if (input_db) d_default_reset_val = input_db->getDoubleWithDefault("reset_value", d_default_reset_val);
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::setMeshMapping(std::shared_ptr<GeneralBoundaryMeshMapping> mesh_mapping)
{
    d_mesh_mapping = mesh_mapping;
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::registerTransportedQuantity(Pointer<CellVariable<NDIM, double>> Q_var,
                                                                    const double reset_val,
                                                                    const bool output_var)
{
    AdvDiffSemiImplicitHierarchyIntegrator::registerTransportedQuantity(Q_var, output_var);
    d_Q_reset_val_map[Q_var] = reset_val;
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::registerLevelSetVariable(Pointer<NodeVariable<NDIM, double>> ls_var,
                                                                 Pointer<LSFindCellVolume> ls_fcn)
{
    if (ls_var)
    {
        if (std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) == d_ls_vars.end())
            d_ls_vars.push_back(ls_var);
        else
            TBOX_WARNING(d_object_name + "::registerLevelSetVariable(): Level set already registered. Skipping.\n");
    }
    else
    {
        TBOX_ERROR(d_object_name + "::registerLevelSetVariable(): Not a valid node centered variable!\n");
    }

#ifndef NDEBUG
    if (!ls_fcn) TBOX_ERROR(d_object_name + "::setLevelSetFunction(): Not a valid level set function!\n");
#endif

    d_ls_fcn_map[ls_var] = ls_fcn;
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::restrictToLevelSet(Pointer<CellVariable<NDIM, double>> Q_var,
                                                           Pointer<NodeVariable<NDIM, double>> ls_var)
{
#ifndef NDEBUG
    if (!ls_var) TBOX_ERROR(d_object_name + "::restrictToLevelSet(): Not a valid node centered variable!\n");
    if (!Q_var) TBOX_ERROR(d_object_name + "::restrictToLevelSet(): Not a valid cell centered variable!\n");
    if (std::find(d_ls_vars.begin(), d_ls_vars.end(), ls_var) == d_ls_vars.end())
        TBOX_ERROR(d_object_name + "::restrictToLevelSet(): Level set not registered!\n");
    if (std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) == d_Q_var.end())
        TBOX_ERROR(d_object_name + "::restrictToLevelSet(): Advected variable not registered!\n");
#endif

    d_ls_Q_map[ls_var].insert(Q_var);
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                                      Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    AdvDiffSemiImplicitHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    // Register and set up level set variables
    for (const auto& ls_var : d_ls_vars)
    {
        static const IntVector<NDIM> ghosts = 1;
        int ls_idx;
        registerVariable(ls_idx, ls_var, ghosts, getCurrentContext());
    }

    // Set the default reconstruction values (if they are not set)
    for (const auto& Q_var : d_Q_var)
    {
        if (d_Q_reset_val_map.count(Q_var) == 0) d_Q_reset_val_map[Q_var] = d_default_reset_val;
    }

    // Register scratch index
    registerVariable(d_valid_idx, d_valid_var, 0 /*ghosts*/, getScratchContext());
}

void
ExtrapolatedAdvDiffHierarchyIntegrator::preprocessIntegrateHierarchy(const double current_time,
                                                                     const double new_time,
                                                                     const int num_cycles)
{
    AdvDiffHierarchyIntegrator::preprocessIntegrateHierarchy(current_time, new_time, num_cycles);

    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    const double dt = new_time - current_time;
    const bool initial_time = IBTK::rel_equal_eps(d_integrator_time, d_start_time);
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();

    // Indicate that all solvers need to be reinitialized if the current
    // timestep size is different from the previous one.
    const bool dt_change = initial_time || !IBTK::abs_equal_eps(dt, d_dt_previous[0]);
    if (dt_change)
    {
        std::fill(d_helmholtz_solvers_need_init.begin(), d_helmholtz_solvers_need_init.end(), true);
        std::fill(d_helmholtz_rhs_ops_need_init.begin(), d_helmholtz_rhs_ops_need_init.end(), true);
        d_coarsest_reset_ln = 0;
        d_finest_reset_ln = finest_ln;
    }

    // Allocate the scratch and new data.
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_scratch_data, current_time);
        level->allocatePatchData(d_new_data, new_time);
    }

    // Update the advection velocity.
    for (const auto& u_var : d_u_var)
    {
        const int u_current_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
        const int u_scratch_idx = var_db->mapVariableAndContextToIndex(u_var, getScratchContext());
        const int u_new_idx = var_db->mapVariableAndContextToIndex(u_var, getNewContext());
        if (d_u_fcn[u_var])
        {
            d_u_fcn[u_var]->setDataOnPatchHierarchy(u_current_idx, u_var, d_hierarchy, current_time);
            d_u_fcn[u_var]->setDataOnPatchHierarchy(u_new_idx, u_var, d_hierarchy, new_time);
        }
        else
        {
            d_hier_fc_data_ops->copyData(u_new_idx, u_current_idx);
        }
        d_hier_fc_data_ops->linearSum(u_scratch_idx, 0.5, u_current_idx, 0.5, u_new_idx);
    }

    // Update the diffusion coefficient
    for (const auto& D_var : d_diffusion_coef_var)
    {
        Pointer<CartGridFunction> D_fcn = d_diffusion_coef_fcn[D_var];
        if (D_fcn)
        {
            const int D_current_idx = var_db->mapVariableAndContextToIndex(D_var, getCurrentContext());
            D_fcn->setDataOnPatchHierarchy(D_current_idx, D_var, d_hierarchy, current_time);
        }
    }

    // Fill in "unphysical" cells.
    // First, compute the level set
    for (const auto& fe_mesh_mapping : d_mesh_mapping->getMeshPartitioners())
    {
        fe_mesh_mapping->setPatchHierarchy(d_hierarchy);
        fe_mesh_mapping->reinitElementMappings();
    }
    for (const auto& ls_var : d_ls_vars)
    {
        const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());
        d_ls_fcn_map.at(ls_var)->updateVolumeAreaSideLS(IBTK::invalid_index,
                                                        nullptr,
                                                        IBTK::invalid_index,
                                                        nullptr,
                                                        IBTK::invalid_index,
                                                        nullptr,
                                                        ls_idx,
                                                        ls_var,
                                                        current_time);
        // Now compute the signed distance function from the level set
        // First, mark valid cells. Cells to change are marked with 0, valid cells that should not be changed are marked
        // with 1. Cells that should be ignored are marked with 2.
        auto fcn = [](Pointer<Patch<NDIM>> patch, const int ls_idx, const int valid_idx)
        {
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<NodeData<NDIM, int>> valid_data = patch->getPatchData(valid_idx);

            Box<NDIM> ghost_node_box = NodeGeometry<NDIM>::toNodeBox(valid_data->getGhostBox());

            for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();

                if (std::abs((*ls_data)(idx)) < 1.0)
                    (*valid_data)(idx) = 1;
                else
                    (*valid_data)(idx) = 2;
            }

            // Now only compute level set in nearby indices of structure
            for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();

                if ((*valid_data)(idx) == 1)
                {
                    Box<NDIM> region(idx, idx);
                    region.grow(10);
                    for (NodeIterator<NDIM> ni2(region); ni2; ni2++)
                    {
                        const NodeIndex<NDIM>& idx2 = ni2();
                        if (ghost_node_box.contains(idx2) && (*valid_data)(idx2) != 1) (*valid_data)(idx2) = 0;
                    }
                }
            }
        };
        perform_on_patch_hierarchy(d_hierarchy, fcn, ls_idx, d_valid_idx);

        // Generate signed distance function from level set.
        ReinitializeLevelSet ls_method("LS", nullptr);
        ls_method.computeSignedDistanceFunction(ls_idx, *ls_var, d_hierarchy, current_time, d_valid_idx);

        for (const auto& Q_var : d_ls_Q_map[ls_var])
        {
            const int Q_cur_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            const int Q_scr_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
            d_hier_cc_data_ops->copyData(Q_scr_idx, Q_cur_idx);
            InternalBdryFill advect_in_norm("InternalFill", nullptr);
            advect_in_norm.advectInNormal(Q_scr_idx, Q_var, ls_idx, ls_var, d_hierarchy, current_time);
            d_hier_cc_data_ops->copyData(Q_cur_idx, Q_scr_idx);
        }
    }

    // Setup the operators and solvers and compute the right-hand-side terms.
    unsigned int l = 0;
    for (auto cit = d_Q_var.begin(); cit != d_Q_var.end(); ++cit, ++l)
    {
        Pointer<CellVariable<NDIM, double>> Q_var = *cit;
        Pointer<CellVariable<NDIM, double>> Q_rhs_var = d_Q_Q_rhs_map[Q_var];
        Pointer<SideVariable<NDIM, double>> D_var = d_Q_diffusion_coef_variable[Q_var];
        Pointer<SideVariable<NDIM, double>> D_rhs_var = d_diffusion_coef_rhs_map[D_var];
        IBAMR::TimeSteppingType diffusion_time_stepping_type = d_Q_diffusion_time_stepping_type[Q_var];
        const double lambda = d_Q_damping_coef[Q_var];
        const std::vector<RobinBcCoefStrategy<NDIM>*>& Q_bc_coef = d_Q_bc_coef[Q_var];

        const int Q_current_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
        const int Q_scratch_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
        const int Q_new_idx = var_db->mapVariableAndContextToIndex(Q_var, getNewContext());
        const int Q_rhs_scratch_idx = var_db->mapVariableAndContextToIndex(Q_rhs_var, getScratchContext());
        const int D_current_idx = (D_var ? var_db->mapVariableAndContextToIndex(D_var, getCurrentContext()) : -1);
        const int D_scratch_idx = (D_var ? var_db->mapVariableAndContextToIndex(D_var, getScratchContext()) : -1);
        const int D_rhs_scratch_idx =
            (D_rhs_var ? var_db->mapVariableAndContextToIndex(D_rhs_var, getScratchContext()) : -1);

        // Setup the problem coefficients for the linear solve for Q(n+1).
        double K = 0.0;
        switch (diffusion_time_stepping_type)
        {
        case BACKWARD_EULER:
            K = 1.0;
            break;
        case FORWARD_EULER:
            K = 0.0;
            break;
        case TRAPEZOIDAL_RULE:
            K = 0.5;
            break;
        default:
            TBOX_ERROR(d_object_name << "::integrateHierarchy():\n"
                                     << "  unsupported diffusion time stepping type: "
                                     << IBAMR::enum_to_string<IBAMR::TimeSteppingType>(diffusion_time_stepping_type)
                                     << " \n"
                                     << "  valid choices are: BACKWARD_EULER, "
                                        "FORWARD_EULER, TRAPEZOIDAL_RULE\n");
        }
        PoissonSpecifications solver_spec(d_object_name + "::solver_spec::" + Q_var->getName());
        PoissonSpecifications rhs_op_spec(d_object_name + "::rhs_op_spec::" + Q_var->getName());
        solver_spec.setCConstant(1.0 / dt + K * lambda);
        rhs_op_spec.setCConstant(1.0 / dt - (1.0 - K) * lambda);
        if (isDiffusionCoefficientVariable(Q_var))
        {
            // set -K*kappa in solver_spec
            d_hier_sc_data_ops->scale(D_scratch_idx, -K, D_current_idx);
            solver_spec.setDPatchDataId(D_scratch_idx);
            // set (1.0-K)*kappa in rhs_op_spec
            d_hier_sc_data_ops->scale(D_rhs_scratch_idx, (1.0 - K), D_current_idx);
            rhs_op_spec.setDPatchDataId(D_rhs_scratch_idx);
        }
        else
        {
            const double kappa = d_Q_diffusion_coef[Q_var];
            solver_spec.setDConstant(-K * kappa);
            rhs_op_spec.setDConstant(+(1.0 - K) * kappa);
        }

        // Initialize the RHS operator and compute the RHS vector.
        Pointer<LaplaceOperator> helmholtz_rhs_op = d_helmholtz_rhs_ops[l];
        helmholtz_rhs_op->setPoissonSpecifications(rhs_op_spec);
        helmholtz_rhs_op->setPhysicalBcCoefs(Q_bc_coef);
        helmholtz_rhs_op->setHomogeneousBc(false);
        helmholtz_rhs_op->setSolutionTime(current_time);
        helmholtz_rhs_op->setTimeInterval(current_time, new_time);
        if (d_helmholtz_rhs_ops_need_init[l])
        {
            if (d_enable_logging)
            {
                plog << d_object_name << ": "
                     << "Initializing Helmholtz RHS operator for variable number " << l << "\n";
            }
            helmholtz_rhs_op->initializeOperatorState(*d_sol_vecs[l], *d_rhs_vecs[l]);
            d_helmholtz_rhs_ops_need_init[l] = false;
        }
        d_hier_cc_data_ops->copyData(Q_scratch_idx, Q_current_idx, false);
        helmholtz_rhs_op->apply(*d_sol_vecs[l], *d_rhs_vecs[l]);

        // Initialize the linear solver.
        Pointer<PoissonSolver> helmholtz_solver = d_helmholtz_solvers[l];
        helmholtz_solver->setPoissonSpecifications(solver_spec);
        helmholtz_solver->setPhysicalBcCoefs(Q_bc_coef);
        helmholtz_solver->setHomogeneousBc(false);
        helmholtz_solver->setSolutionTime(new_time);
        helmholtz_solver->setTimeInterval(current_time, new_time);
        if (d_helmholtz_solvers_need_init[l])
        {
            if (d_enable_logging)
            {
                plog << d_object_name << ": "
                     << "Initializing Helmholtz solvers for variable number " << l << "\n";
            }
            helmholtz_solver->initializeSolverState(*d_sol_vecs[l], *d_rhs_vecs[l]);
            d_helmholtz_solvers_need_init[l] = false;
        }

        // Account for the convective difference term.
        Pointer<FaceVariable<NDIM, double>> u_var = d_Q_u_map[Q_var];
        if (u_var)
        {
            Pointer<CellVariable<NDIM, double>> N_var = d_Q_N_map[Q_var];
            Pointer<CellVariable<NDIM, double>> N_old_var = d_Q_N_old_map[Q_var];
            IBAMR::TimeSteppingType convective_time_stepping_type = d_Q_convective_time_stepping_type[Q_var];
            if (getIntegratorStep() == 0 && is_multistep_time_stepping_type(convective_time_stepping_type))
            {
                convective_time_stepping_type = d_Q_init_convective_time_stepping_type[Q_var];
            }
            if ((num_cycles == 1) &&
                (convective_time_stepping_type == MIDPOINT_RULE || convective_time_stepping_type == TRAPEZOIDAL_RULE))
            {
                TBOX_ERROR(
                    d_object_name << "::preprocessIntegrateHierarchy():\n"
                                  << "  time stepping type: "
                                  << IBAMR::enum_to_string<IBAMR::TimeSteppingType>(convective_time_stepping_type)
                                  << " requires num_cycles > 1.\n"
                                  << "  at current time step, num_cycles = " << num_cycles << "\n");
            }
            if (d_Q_convective_op_needs_init[Q_var])
            {
                d_Q_convective_op[Q_var]->initializeOperatorState(*d_sol_vecs[l], *d_rhs_vecs[l]);
                d_Q_convective_op_needs_init[Q_var] = false;
            }
            const int u_current_idx = var_db->mapVariableAndContextToIndex(u_var, getCurrentContext());
            d_Q_convective_op[Q_var]->setAdvectionVelocity(u_current_idx);
            const int Q_current_idx = var_db->mapVariableAndContextToIndex(Q_var, getCurrentContext());
            const int Q_scratch_idx = var_db->mapVariableAndContextToIndex(Q_var, getScratchContext());
            const int N_scratch_idx = var_db->mapVariableAndContextToIndex(N_var, getScratchContext());
            d_hier_cc_data_ops->copyData(Q_scratch_idx, Q_current_idx);
            d_Q_convective_op[Q_var]->setSolutionTime(current_time);
            d_Q_convective_op[Q_var]->applyConvectiveOperator(Q_scratch_idx, N_scratch_idx);
            const int N_old_new_idx = var_db->mapVariableAndContextToIndex(N_old_var, getNewContext());
            d_hier_cc_data_ops->copyData(N_old_new_idx, N_scratch_idx);
            if (convective_time_stepping_type == FORWARD_EULER)
            {
                d_hier_cc_data_ops->axpy(Q_rhs_scratch_idx, -1.0, N_scratch_idx, Q_rhs_scratch_idx);
            }
            else if (convective_time_stepping_type == TRAPEZOIDAL_RULE)
            {
                d_hier_cc_data_ops->axpy(Q_rhs_scratch_idx, -0.5, N_scratch_idx, Q_rhs_scratch_idx);
            }
        }

        // Set the initial guess.
        d_hier_cc_data_ops->copyData(Q_new_idx, Q_current_idx);
    }

    // Execute any registered callbacks.
    executePreprocessIntegrateHierarchyCallbackFcns(current_time, new_time, num_cycles);
    return;
} // preprocessIntegrateHierarchy

void
ExtrapolatedAdvDiffHierarchyIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                                                      const double new_time,
                                                                      const bool skip_synchronize_new_state_data,
                                                                      const int num_cycles)
{
    // Clear out any unphysical values
    d_mesh_mapping->updateBoundaryLocation(new_time, true);
    for (const auto& ls_var : d_ls_vars)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, getCurrentContext());

        d_ls_fcn_map[ls_var]->updateVolumeAreaSideLS(IBTK::invalid_index,
                                                     nullptr,
                                                     IBTK::invalid_index,
                                                     nullptr,
                                                     IBTK::invalid_index,
                                                     nullptr,
                                                     ls_idx,
                                                     ls_var,
                                                     new_time);

        auto fcn = [](Pointer<Patch<NDIM>> patch,
                      const int ls_idx,
                      const std::set<Pointer<CellVariable<NDIM, double>>>& Q_vars,
                      Pointer<VariableContext> ctx,
                      const std::map<Pointer<CellVariable<NDIM, double>>, double>& reset_map)
        {
            for (const auto& Q_var : Q_vars)
            {
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(Q_var, ctx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if (ADS::node_to_cell(idx, *ls_data) > 0.0) (*Q_new_data)(idx) = reset_map.at(Q_var);
                }
            }
        };
        perform_on_patch_hierarchy(d_hierarchy, fcn, ls_idx, d_ls_Q_map[ls_var], getNewContext(), d_Q_reset_val_map);
    }
    AdvDiffSemiImplicitHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);
    return;
} // postprocessIntegrateHierarchy

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
