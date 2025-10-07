// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2024 by the IBAMR developers
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

#include "ADS/AdvDiffImplicitIntegrator.h"
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include "ibamr/ibamr_utilities.h"

#include "ibtk/IBTK_MPI.h"

#include "CellVariable.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "tbox/Database.h"
#include "tbox/PIO.h"

#include <Eigen/LU>

#include <algorithm>
#include <ostream>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

// Number of ghosts cells used for each variable quantity.
static const int CELLG = 1;

/////////////////////////////// PUBLIC ///////////////////////////////////////

AdvDiffImplicitIntegrator::AdvDiffImplicitIntegrator(const std::string& object_name,
                                                     Pointer<Database> input_db,
                                                     bool register_for_restart)
    : AdvDiffSemiImplicitHierarchyIntegrator(object_name, input_db, register_for_restart)
{
    d_implicit_ts_type =
        IBAMR::string_to_enum<TimeSteppingType>(input_db->getStringWithDefault("implicit_ts_type", "BACKWARD_EULER"));
    switch (d_implicit_ts_type)
    {
    case TimeSteppingType::BACKWARD_EULER:
    case TimeSteppingType::TRAPEZOIDAL_RULE:
        // Do Nothing
        break;
    default:
        TBOX_ERROR("Invalid time stepping type " + IBAMR::enum_to_string(d_implicit_ts_type) +
                   ". Valid options are BACKWARD_EULER and TRAPEZOIDAL_RULE");
    }

    d_tol_for_newton = input_db->getDoubleWithDefault("tol_for_newton", d_tol_for_newton);
    d_max_iterations = input_db->getIntegerWithDefault("max_iterations", d_max_iterations);
    return;
} // AdvDiffImplicitIntegrator

void
AdvDiffImplicitIntegrator::setImplicitVariable(Pointer<CellVariable<NDIM, double>> Q_var)
{
    if (std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) == d_Q_var.end())
    {
        TBOX_ERROR(d_object_name + "::setImplicitVariable(): Variable " + Q_var->getName() +
                   " is not a tranported quantity!\n");
    }
    if (std::find(d_Q_implicit_vars.begin(), d_Q_implicit_vars.end(), Q_var) == d_Q_implicit_vars.end())
        d_Q_implicit_vars.push_back(Q_var);
    else
        TBOX_WARNING(d_object_name + "::setImplicitVariable(): Variable " + Q_var->getName() +
                     " was already registered as implicit. Ignoring\n");
}

void
AdvDiffImplicitIntegrator::setImplicitDependentVariable(Pointer<CellVariable<NDIM, double>> Q_var)
{
    if (std::find(d_Q_var.begin(), d_Q_var.end(), Q_var) == d_Q_var.end())
    {
        TBOX_ERROR(d_object_name + "::setImplicitDependentVariable(): Variable " + Q_var->getName() +
                   " is not a tranported quantity!\n");
    }
    if (std::find(d_Q_implicit_dependent_vars.begin(), d_Q_implicit_dependent_vars.end(), Q_var) ==
        d_Q_implicit_dependent_vars.end())
        d_Q_implicit_dependent_vars.push_back(Q_var);
    else
        TBOX_WARNING(d_object_name + "::setImplicitDependentVariable(): Variable " + Q_var->getName() +
                     " was already registered as dependent. Ignoring\n");
}

void
AdvDiffImplicitIntegrator::setImplicitDependentVariable(Pointer<CellVariable<NDIM, double>> Q_var,
                                                        std::shared_ptr<CartGridFunction> Q_fcn)
{
    if (std::find(d_Q_implicit_dependent_vars.begin(), d_Q_implicit_dependent_vars.end(), Q_var) ==
        d_Q_implicit_dependent_vars.end())
    {
        d_Q_implicit_dependent_vars.push_back(Q_var);
        d_Q_implicit_dependent_var_fcn_map[Q_var] = Q_fcn;
    }
    else
    {
        TBOX_WARNING(d_object_name + "::setImplicitDependentVariable(): Variable " + Q_var->getName() +
                     " was already registered as dependent. Ignoring\n");
    }
}

void
AdvDiffImplicitIntegrator::registerImplicitStrategy(std::shared_ptr<AdvDiffImplicitIntegratorStrategy> strategy)
{
    TBOX_ASSERT(strategy);
    d_implicit_strategy = std::move(strategy);
}

void
AdvDiffImplicitIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                         Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    if (d_integrator_is_initialized) return;

    // Do some error checking.
    for (const auto& Q_var : d_Q_implicit_vars)
    {
        if (d_Q_F_map.count(Q_var) > 0 && d_Q_F_map.at(Q_var))
        {
            TBOX_WARNING("Variable " << Q_var->getName()
                                     << " had a source variable associated with it but is listed as an implicit "
                                        "variable. Removing the source variable!\n");
            d_Q_F_map.erase(Q_var);
        }
    }

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_implicit_ctx = var_db->getContext(d_object_name + "::IMPLICIT");
    for (const auto& Q_var_fcn_pair : d_Q_implicit_dependent_var_fcn_map)
    {
        Pointer<CellVariable<NDIM, double>> Q_var = Q_var_fcn_pair.first;
        int Q_imp_idx = var_db->registerVariableAndContext(Q_var, d_implicit_ctx);
        d_implicit_comps.setFlag(Q_imp_idx);
    }

    // Perform hierarchy initialization operations common to all implementations
    // of AdvDiffHierarchyIntegrator.
    AdvDiffSemiImplicitHierarchyIntegrator::initializeHierarchyIntegrator(hierarchy, gridding_alg);

    // Indicate that the integrator has been initialized.
    d_integrator_is_initialized = true;
    return;
} // initializeHierarchyIntegrator

void
AdvDiffImplicitIntegrator::integrateHierarchySpecialized(const double current_time,
                                                         const double new_time,
                                                         const int cycle_num)
{
    AdvDiffSemiImplicitHierarchyIntegrator::integrateHierarchySpecialized(current_time, new_time, cycle_num);

    doImplicitUpdate(current_time, new_time);
    return;
} // integrateHierarchy

/////////////////////////////// PROTECTED ////////////////////////////////////

void
AdvDiffImplicitIntegrator::doImplicitUpdate(const double current_time, const double new_time)
{
    // Now compute the implicit update for all implicit variables
    if (d_Q_implicit_vars.size() == 0) return;
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();

    ADS::allocate_patch_data(d_implicit_comps, d_hierarchy, current_time, coarsest_ln, finest_ln);
    const double dt = new_time - current_time;

    // Precompute any necessary quantities
    for (const auto& Q_var_fcn_pair : d_Q_implicit_dependent_var_fcn_map)
    {
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var_fcn_pair.first, d_implicit_ctx);
        Q_var_fcn_pair.second->setDataOnPatchHierarchy(Q_idx, Q_var_fcn_pair.first, d_hierarchy, new_time);
    }

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);

        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            // Note that the NEW context should be filled at this point
            Pointer<VariableContext> ctx = getNewContext();
            std::vector<Pointer<CellData<NDIM, double>>> Q_cell_data(d_Q_implicit_vars.size());
            std::vector<Pointer<CellData<NDIM, double>>> Q_extra_data(d_Q_implicit_dependent_vars.size());
            int sys_size = 0;
            for (size_t i = 0; i < d_Q_implicit_vars.size(); ++i)
            {
                Q_cell_data[i] = patch->getPatchData(d_Q_implicit_vars[i], ctx);
                sys_size += Q_cell_data[i]->getDepth();
            }

            int extra_size = 0;
            for (size_t i = 0; i < d_Q_implicit_dependent_vars.size(); ++i)
            {
                auto Q_var = d_Q_implicit_dependent_vars[i];
                if (d_Q_implicit_dependent_var_fcn_map.count(Q_var) > 0)
                {
                    Q_extra_data[i] = patch->getPatchData(Q_var, d_implicit_ctx);
                }
                else
                {
                    Q_extra_data[i] = patch->getPatchData(Q_var, ctx);
                }
                extra_size += Q_extra_data[i]->getDepth();
            }

            // Now loop through cell indices and perform Newton iterations
            Eigen::VectorXd U(sys_size), U_cur(sys_size);
            Eigen::VectorXd U_new(sys_size);
            Eigen::VectorXd F(sys_size), Fn(sys_size);
            Eigen::MatrixXd J(sys_size, sys_size);
            Eigen::VectorXd Q(extra_size);
            // We need an identity for the Jacobian
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(sys_size, sys_size);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
#ifndef NDEBUG
                F.setConstant(sys_size, std::numeric_limits<double>::quiet_NaN());
                J.setConstant(sys_size, sys_size, std::numeric_limits<double>::quiet_NaN());
#endif
                const CellIndex<NDIM>& idx = ci();
                // Fill with the initial guess
                for (size_t i = 0, ii = 0; i < d_Q_implicit_vars.size(); ++i)
                {
                    for (int d = 0; d < Q_cell_data[i]->getDepth(); ++d)
                    {
                        U[ii++] = (*Q_cell_data[i])(idx, d);
                    }
                }

                for (size_t i = 0, ii = 0; i < d_Q_implicit_dependent_vars.size(); ++i)
                {
                    for (int d = 0; d < Q_extra_data[i]->getDepth(); ++d)
                    {
                        Q[ii++] = (*Q_extra_data[i])(idx, d);
                    }
                }

                U_cur = U;
                if (d_implicit_ts_type == IBAMR::TimeSteppingType::TRAPEZOIDAL_RULE)
                    d_implicit_strategy->computeFunction(Fn, U, current_time, Q);
                // Iterator
                bool converged = false;
                for (int iter = 0; iter < d_max_iterations && !converged; ++iter)
                {
                    // Compute F
                    d_implicit_strategy->computeFunction(F, U, new_time, Q);

                    // Compute Jacobian
                    d_implicit_strategy->computeJacobian(J, U, new_time, Q);

                    if (d_implicit_ts_type == IBAMR::TimeSteppingType::BACKWARD_EULER)
                    {
                        F = (U - U_cur) / dt - F;
                        J = I / dt - J;
                    }
                    else if (d_implicit_ts_type == IBAMR::TimeSteppingType::TRAPEZOIDAL_RULE)
                    {
                        F = (U - U_cur) / dt - 0.5 * (F + Fn);
                        J = I / dt - 0.5 * J;
                    }
                    else
                    {
                        TBOX_ERROR("Invalid time stepping type " << IBAMR::enum_to_string(d_implicit_ts_type));
                    }

                    // Now update solution by solving system.
                    U_new = U - J.fullPivLu().solve(F);
                    // Check if we have converged
                    if ((U_new - U).norm() < d_tol_for_newton)
                    {
                        converged = true;
                    }
                    // Update for next iteration
                    U = U_new;
                }
                if (!converged)
                {
                    TBOX_ERROR("On index " << idx << " the Newton solver failed to converge after " << d_max_iterations
                                           << "iterations\n");
                }

                // Fill in new data
                for (size_t i = 0, ii = 0; i < d_Q_implicit_vars.size(); ++i)
                {
                    for (int d = 0; d < Q_cell_data[i]->getDepth(); ++d)
                    {
                        (*Q_cell_data[i])(idx, d) = U[ii++];
                    }
                }
            }
        }
    }

    ADS::deallocate_patch_data(d_implicit_comps, d_hierarchy, coarsest_ln, finest_ln);
}

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
