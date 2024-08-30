// ---------------------------------------------------------------------
//
// Copyright (c) 2017 - 2020 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <ADS/FullFACPreconditioner.h>
#include <ADS/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/CCLaplaceOperator.h>
#include <ibtk/CCPoissonSolverManager.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/PETScKrylovPoissonSolver.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

// Local includes
#include "full_fac_preconditioner/CCPointFACPreconditionerStrategy.h"
#include "full_fac_preconditioner/CCPointRelaxationFACOperator.h"

#include "full_fac_preconditioner/CCPointFACPreconditionerStrategy.cpp"
#include "full_fac_preconditioner/CCPointRelaxationFACOperator.cpp"

/*******************************************************************************
 * For each run, the input filename must be given on the command line.  In all *
 * cases, the command line is:                                                 *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    PetscOptionsSetValue(nullptr, "-solver_ksp_rtol", "1.0e-12");

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "output");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", NULL, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Create variables and register them with the variable database.
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("context");

        // State variable
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");

        // RHS variable
        Pointer<CellVariable<NDIM, double>> rhs_var = new CellVariable<NDIM, double>("rhs");

        // Error terms.
        Pointer<CellVariable<NDIM, double>> err_var = new CellVariable<NDIM, double>("err");

        // Exact function
        Pointer<CellVariable<NDIM, double>> exa_var = new CellVariable<NDIM, double>("exact");

        // Register patch data indices...
        const int Q_idx = var_db->registerVariableAndContext(Q_var, ctx, IntVector<NDIM>(1));
        const int rhs_idx = var_db->registerVariableAndContext(rhs_var, ctx, IntVector<NDIM>(1));
        const int err_idx = var_db->registerVariableAndContext(err_var, ctx, IntVector<NDIM>(1));
        const int exa_idx = var_db->registerVariableAndContext(exa_var, ctx, 0);

        // Register variables for plotting.
        // Uncomment to draw data
// #define DRAW_DATA
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        TBOX_ASSERT(visit_data_writer);
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_idx);
        visit_data_writer->registerPlotQuantity("RHS", "SCALAR", rhs_idx);
        visit_data_writer->registerPlotQuantity("error", "SCALAR", err_idx);
        visit_data_writer->registerPlotQuantity("exact", "SCALAR", exa_idx);
#endif

        // Create the grid
        gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
        int tag_buffer = 1;
        int level_number = 0;
        bool done = false;
        while (!done && (gridding_algorithm->levelCanBeRefined(level_number)))
        {
            gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, tag_buffer);
            done = !patch_hierarchy->finerLevelExists(level_number);
            ++level_number;
        }

        // Allocate data on each level of the patch hierarchy.
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(Q_idx, 0.0);
            level->allocatePatchData(rhs_idx, 0.0);
            level->allocatePatchData(err_idx, 0.0);
            level->allocatePatchData(exa_idx, 0.0);
        }

        // Setup vector objects.
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> Q_vec("Q", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> rhs_vec("rhs", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> err_vec("err", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());

        Q_vec.addComponent(Q_var, Q_idx, wgt_idx);
        rhs_vec.addComponent(rhs_var, rhs_idx, wgt_idx);
        err_vec.addComponent(err_var, err_idx, wgt_idx);

        Q_vec.setToScalar(0.0);
        rhs_vec.setToScalar(0.0);
        err_vec.setToScalar(0.0);

        // Setup solution and RHS functions
        muParserCartGridFunction Q_fcn("Q", app_initializer->getComponentDatabase("Q"), grid_geometry);
        muParserCartGridFunction rhs_fcn("rhs", app_initializer->getComponentDatabase("RHS"), grid_geometry);
        Q_fcn.setDataOnPatchHierarchy(exa_idx, exa_var, patch_hierarchy, 0.0);
        Q_fcn.setDataOnPatchHierarchy(err_idx, err_var, patch_hierarchy, 0.0);
        rhs_fcn.setDataOnPatchHierarchy(rhs_idx, rhs_var, patch_hierarchy, 0.0);

        // Setup Poisson specifications
        PoissonSpecifications poisson_spec("poisson_spec");
        const double D = input_db->getDouble("D");
        const double C = input_db->getDouble("C");
        poisson_spec.setDConstant(D);
        poisson_spec.setCConstant(C);

        // Setup the Poisson solver
        Pointer<CCLaplaceOperator> laplace_op = new CCLaplaceOperator("laplace_op");
        laplace_op->setPoissonSpecifications(poisson_spec);
        laplace_op->initializeOperatorState(Q_vec, rhs_vec);

        Pointer<PETScKrylovPoissonSolver> poisson_solver = new PETScKrylovPoissonSolver(
            "PoissonSolver", app_initializer->getComponentDatabase("PoissonSolver"), "solver_");
        poisson_solver->setOperator(laplace_op);
        poisson_solver->setPoissonSpecifications(poisson_spec);

        // Now create a preconditioner
        Pointer<FACPreconditionerStrategy> poisson_strategy =
            new CCPointRelaxationFACOperator("PoissonPrecondStrategy",
                                             app_initializer->getComponentDatabase("PoissonPrecondStrategy"),
                                             "solver_precond_");
        Pointer<FACPreconditioner> poisson_precond =
            new FullFACPreconditioner("PoissonPrecond",
                                      poisson_strategy,
                                      app_initializer->getComponentDatabase("PoissonPrecond"),
                                      "solver_precond_");
        if (input_db->getBool("USE_PRECOND")) poisson_solver->setPreconditioner(poisson_precond);

        // Apply the operator
        poisson_solver->initializeSolverState(Q_vec, rhs_vec);
        poisson_solver->solveSystem(Q_vec, rhs_vec);
        poisson_solver->deallocateSolverState();

        // Compute error and print error norms.
        err_vec.subtract(Pointer<SAMRAIVectorReal<NDIM, double>>(&Q_vec, false),
                         Pointer<SAMRAIVectorReal<NDIM, double>>(&err_vec, false));
        pout << "|e|_oo = " << err_vec.maxNorm() << "\n";
        pout << "|e|_2  = " << err_vec.L2Norm() << "\n";
        pout << "|e|_1  = " << err_vec.L1Norm() << "\n";

        // Output data for plotting.
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif

        // Deallocate level data
        // Allocate data on each level of the patch hierarchy.
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(Q_idx);
            level->deallocatePatchData(rhs_idx);
            level->deallocatePatchData(err_idx);
            level->deallocatePatchData(exa_idx);
        }
    } // cleanup dynamically allocated objects prior to shutdown
} // main
