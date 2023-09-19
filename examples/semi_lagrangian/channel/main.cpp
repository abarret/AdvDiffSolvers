#include "ibamr/config.h"

#include "ADS/LSFromLevelSet.h"
#include "ADS/SLAdvIntegrator.h"
#include <ADS/PPMReconstructions.h>
#include <ADS/WENOReconstructions.h>
#include <ADS/app_namespaces.h>

#include <ibamr/RelaxationLSMethod.h>

#include "ibtk/CartGridFunctionSet.h"
#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>

#include "BergerRigoutsos.h"
#include "CartesianGridGeometry.h"
#include "LoadBalancer.h"
#include "SAMRAI_config.h"
#include "StandardTagAndInitialize.h"
#include "tbox/Pointer.h"

#include <petscsys.h>

#include <memory>
#include <utility>

// Local includes
#include "LSFcn.h"
#include "QFcn.h"

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize PETSc, MPI, and SAMRAI.
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    SAMRAI_MPI::setCommunicator(PETSC_COMM_WORLD);
    SAMRAI_MPI::setCallAbortInSerialInsteadOfExit();
    SAMRAIManager::startup();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const std::string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int dump_postproc_interval = app_initializer->getPostProcessingDataDumpInterval();
        const std::string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<SLAdvIntegrator> time_integrator = new SLAdvIntegrator(
            "LSAdvDiffIntegrator", app_initializer->getComponentDatabase("LSAdvDiffIntegrator"), false);

        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Setup the advection velocity.
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("u");
        Pointer<CartGridFunction> u_fcn;
        Pointer<CartGridFunction> u_adv_fcn;
        time_integrator->registerAdvectionVelocity(u_var);
        time_integrator->setAdvectionVelocityIsDivergenceFree(u_var, true);
        u_fcn = new muParserCartGridFunction(
            "UFunction", app_initializer->getComponentDatabase("UFunction"), grid_geometry);
        time_integrator->setAdvectionVelocityFunction(u_var, u_fcn);

        // Setup the level set function
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        time_integrator->registerLevelSetVariable(ls_var);
        Pointer<LSFcn> ls_fcn = new LSFcn("LSFcn");
        Pointer<LSFromLevelSet> vol_in_fcn = new LSFromLevelSet("LS", patch_hierarchy);
        vol_in_fcn->registerLSFcn(ls_fcn);
        time_integrator->registerLevelSetVolFunction(ls_var, vol_in_fcn);

        // Setup advected quantity
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CartGridFunction> Q_init = new QFcn("QInit", app_initializer->getComponentDatabase("QInitial"));
        const bool periodic_domain = grid_geometry->getPeriodicShift().min() > 0;
        std::vector<RobinBcCoefStrategy<NDIM>*> Q_bcs(1);
        if (!periodic_domain)
        {
            const std::string Q_bcs_name = "Q_bcs";
            Q_bcs[0] =
                new muParserRobinBcCoefs(Q_bcs_name, app_initializer->getComponentDatabase(Q_bcs_name), grid_geometry);
        }

        time_integrator->registerTransportedQuantity(Q_var);
        time_integrator->setAdvectionVelocity(Q_var, u_var);
        time_integrator->setInitialConditions(Q_var, Q_init);
        time_integrator->setPhysicalBcCoef(Q_var, Q_bcs[0]);
        time_integrator->restrictToLevelSet(Q_var, ls_var);

        if (input_db->getBool("USE_WENO"))
        {
            auto adv_ops = std::make_shared<WENOReconstructions>("adv_ops");
            time_integrator->registerAdvectionReconstruction(Q_var, adv_ops);
        }
        else if (input_db->getBool("USE_PPM"))
        {
            auto adv_ops = std::make_shared<PPMReconstructions>("adv_ops");
            time_integrator->registerAdvectionReconstruction(Q_var, adv_ops);
        }

        // Set up visualization plot file writer.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }

        // Register a drawing variable with the data writer
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<CellVariable<NDIM, double>> u_draw_var = new CellVariable<NDIM, double>("U", NDIM);
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw_var, var_db->getContext("Draw"));
        visit_data_writer->registerPlotQuantity("U", "VECTOR", u_draw_idx);

        // Set up error variable
        Pointer<CellVariable<NDIM, double>> Q_exa_var = new CellVariable<NDIM, double>("Q_exa");
        Pointer<CellVariable<NDIM, double>> Q_err_var = new CellVariable<NDIM, double>("Q_err");
        const int Q_exa_idx = var_db->registerVariableAndContext(Q_exa_var, var_db->getContext("CTX"));
        const int Q_err_idx = var_db->registerVariableAndContext(Q_err_var, var_db->getContext("CTX"));

        visit_data_writer->registerPlotQuantity("Q_exact", "SCALAR", Q_exa_idx);
        visit_data_writer->registerPlotQuantity("Q_error", "SCALAR", Q_err_idx);

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Close the restart manager.
        RestartManager::getManager()->closeRestartFile();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        double dt = time_integrator->getMaximumTimeStepSize();

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            time_integrator->setupPlotData();
            time_integrator->allocatePatchData(u_draw_idx, loop_time);
            u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, loop_time);
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            time_integrator->deallocatePatchData(u_draw_idx);
        }

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        while (!MathUtilities<double>::equalEps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            const int coarsest_ln = 0;
            const int finest_ln = patch_hierarchy->getFinestLevelNumber();
            time_integrator->allocatePatchData(Q_exa_idx, loop_time, coarsest_ln, finest_ln);
            time_integrator->allocatePatchData(Q_err_idx, loop_time, coarsest_ln, finest_ln);
            Q_init->setDataOnPatchHierarchy(
                Q_exa_idx, Q_exa_var, patch_hierarchy, loop_time, false, coarsest_ln, finest_ln);
            HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy);
            const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, time_integrator->getCurrentContext());
            hier_cc_data_ops.subtract(Q_err_idx, Q_exa_idx, Q_idx);
            HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy, coarsest_ln, finest_ln);
            hier_math_ops.setPatchHierarchy(patch_hierarchy);
            const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
            pout << "Computing error norms\n";
            pout << " L1-norm:  " << hier_cc_data_ops.L1Norm(Q_err_idx, wgt_cc_idx) << "\n";
            pout << " L2-norm:  " << hier_cc_data_ops.L2Norm(Q_err_idx, wgt_cc_idx) << "\n";
            pout << " max-norm: " << hier_cc_data_ops.maxNorm(Q_err_idx, wgt_cc_idx) << "\n";

            // At specified intervals, write visualization and restart files,
            // and print out timer data.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && uses_visit && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                time_integrator->setupPlotData();
                time_integrator->allocatePatchData(u_draw_idx, loop_time);
                u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, loop_time);
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                time_integrator->deallocatePatchData(u_draw_idx);
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
                TimerManager::getManager()->resetAllTimers();
            }

            time_integrator->deallocatePatchData(Q_exa_idx, coarsest_ln, finest_ln);
            time_integrator->deallocatePatchData(Q_err_idx, coarsest_ln, finest_ln);
        }

        if (!periodic_domain) delete Q_bcs[0];
    } // cleanup dynamically allocated objects prior to shutdown

    SAMRAIManager::shutdown();
    PetscFinalize();
    return 0;
} // main
