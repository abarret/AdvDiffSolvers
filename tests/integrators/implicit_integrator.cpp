#include "ADS/AdvDiffImplicitIntegrator.h"
#include <ADS/PointwiseFunction.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <Eigen/Dense>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>

class ImplicitIntegratorStrategy : public ADS::AdvDiffImplicitIntegratorStrategy
{
public:
    ImplicitIntegratorStrategy(double kappa) : AdvDiffImplicitIntegratorStrategy(), kappa(kappa)
    {
        // intentionally blank
    }

    void computeJacobian(MatrixXd& J, const VectorXd& U, const double time, const IBTK::VectorXd&) override
    {
#ifndef NDEBUG
        TBOX_ASSERT(J.rows() == 2);
        TBOX_ASSERT(J.cols() == 2);
        TBOX_ASSERT(U.rows() == 2);
#endif
        J(0, 0) = -kappa;
        J(0, 1) = 0.0;
        J(1, 0) = kappa;
        J(1, 1) = -kappa;
    }

    void computeFunction(VectorXd& F, const VectorXd& U, const double time, const IBTK::VectorXd&) override
    {
#ifndef NDEBUG
        TBOX_ASSERT(F.rows() == 2);
        TBOX_ASSERT(U.rows() == 2);
#endif
        F[0] = -kappa * U[0];
        F[1] = kappa * U[0] - kappa * U[1];
    }

private:
    double kappa = std::numeric_limits<double>::quiet_NaN();
};

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

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "stokes.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<AdvDiffImplicitIntegrator> adv_diff_integrator = new AdvDiffImplicitIntegrator(
            "AdvDiffIntegrator", app_initializer->getComponentDatabase("AdvDiffIntegrator"), true);
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               adv_diff_integrator,
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

        // Set up visualizations
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        adv_diff_integrator->registerVisItDataWriter(visit_data_writer);

        Pointer<CellVariable<NDIM, double>> z_var = new CellVariable<NDIM, double>("Z");
        Pointer<CellVariable<NDIM, double>> g_var = new CellVariable<NDIM, double>("g");
        adv_diff_integrator->registerTransportedQuantity(z_var);
        adv_diff_integrator->registerTransportedQuantity(g_var);
        adv_diff_integrator->setImplicitVariable(z_var);
        adv_diff_integrator->setImplicitVariable(g_var);

        const double kappa = input_db->getDouble("KAPPA");
        auto implicit_strat = std::make_unique<ImplicitIntegratorStrategy>(kappa);
        adv_diff_integrator->registerImplicitStrategy(std::move(implicit_strat));
        const double z_init = input_db->getDouble("Z_INIT");
        const double g_init = input_db->getDouble("G_INIT");

        Pointer<CartGridFunction> z_init_fcn = new ADS::PointwiseFunction<ADS::PointwiseFunctions::ScalarFcn>(
            "Z_INIT", [&z_init](double, const VectorNd& x, double) -> double { return z_init; });
        Pointer<CartGridFunction> g_init_fcn = new ADS::PointwiseFunction<ADS::PointwiseFunctions::ScalarFcn>(
            "G_INIT", [&g_init](double, const VectorNd& x, double) -> double { return g_init; });
        adv_diff_integrator->setInitialConditions(z_var, z_init_fcn);
        adv_diff_integrator->setInitialConditions(g_var, g_init_fcn);

        // Initialize the INS integrator
        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int z_exa_idx = var_db->registerVariableAndContext(z_var, var_db->getContext("EXACT"));
        const int z_err_idx = var_db->registerVariableAndContext(z_var, var_db->getContext("ERROR"));
        const int z_idx = var_db->mapVariableAndContextToIndex(z_var, adv_diff_integrator->getCurrentContext());
        PointwiseFunction<PointwiseFunctions::ScalarFcn> z_exa_fcn(
            "Z_EXACT",
            [&kappa, &z_init](double, const VectorNd& x, double t) -> double { return z_init * std::exp(-kappa * t); });
        visit_data_writer->registerPlotQuantity("Z_EXACT", "SCALAR", z_exa_idx);
        visit_data_writer->registerPlotQuantity("Z_ERROR", "SCALAR", z_err_idx);

        const int g_exa_idx = var_db->registerVariableAndContext(g_var, var_db->getContext("EXACT"));
        const int g_err_idx = var_db->registerVariableAndContext(g_var, var_db->getContext("ERROR"));
        const int g_idx = var_db->mapVariableAndContextToIndex(g_var, adv_diff_integrator->getCurrentContext());
        PointwiseFunction<PointwiseFunctions::ScalarFcn> g_exa_fcn(
            "G_EXACT",
            [&kappa, &z_init, &g_init](double, const VectorNd& x, double t) -> double
            { return (kappa * z_init * t + g_init) * std::exp(-kappa * t); });
        visit_data_writer->registerPlotQuantity("G_EXACT", "SCALAR", g_exa_idx);
        visit_data_writer->registerPlotQuantity("G_ERROR", "SCALAR", g_err_idx);

        // Get some time stepping information.
        unsigned int iteration_num = adv_diff_integrator->getIntegratorStep();
        double loop_time = adv_diff_integrator->getIntegratorTime();
        double time_end = adv_diff_integrator->getEndTime();
        double dt = 0.0;

        input_db->printClassData(plog);
        app_initializer.setNull();

        // Visualization files info.
        double viz_dump_time_interval = input_db->getDouble("VIZ_DUMP_TIME_INTERVAL");
        double next_viz_dump_time = 0.0;
        // At specified intervals, write visualization files
        if (IBTK::abs_equal_eps(loop_time, next_viz_dump_time, 0.1 * dt) || loop_time >= next_viz_dump_time)
        {
            // Compute errors
            ADS::allocate_patch_data({ z_exa_idx, z_err_idx, g_exa_idx, g_err_idx },
                                     patch_hierarchy,
                                     loop_time,
                                     0,
                                     patch_hierarchy->getFinestLevelNumber());
            HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
            HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy);
            z_exa_fcn.setDataOnPatchHierarchy(z_exa_idx, z_var, patch_hierarchy, loop_time);
            hier_cc_data_ops.subtract(z_err_idx, z_exa_idx, z_idx);
            g_exa_fcn.setDataOnPatchHierarchy(g_exa_idx, g_var, patch_hierarchy, loop_time);
            hier_cc_data_ops.subtract(g_err_idx, g_exa_idx, g_idx);
            const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
            pout << "Errors in z at time " << loop_time << "\n";
            pout << "  L1-norm:  " << hier_cc_data_ops.L1Norm(z_err_idx, wgt_idx) << "\n";
            pout << "  L2-norm:  " << hier_cc_data_ops.L2Norm(z_err_idx, wgt_idx) << "\n";
            pout << "  max-norm: " << hier_cc_data_ops.maxNorm(z_err_idx, wgt_idx) << "\n";
            pout << "Errors in g at time " << loop_time << "\n";
            pout << "  L1-norm:  " << hier_cc_data_ops.L1Norm(g_err_idx, wgt_idx) << "\n";
            pout << "  L2-norm:  " << hier_cc_data_ops.L2Norm(g_err_idx, wgt_idx) << "\n";
            pout << "  max-norm: " << hier_cc_data_ops.maxNorm(g_err_idx, wgt_idx) << "\n";
            pout << "\nWriting visualization files...\n\n";
            adv_diff_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            next_viz_dump_time += viz_dump_time_interval;
            ADS::deallocate_patch_data({ z_exa_idx, z_err_idx, g_exa_idx, g_err_idx },
                                       patch_hierarchy,
                                       0,
                                       patch_hierarchy->getFinestLevelNumber());
        }
        // Main time step loop
        while (!IBTK::rel_equal_eps(loop_time, time_end) && adv_diff_integrator->stepsRemaining())
        {
            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = adv_diff_integrator->getMaximumTimeStepSize();
            adv_diff_integrator->advanceHierarchy(dt);

            loop_time += dt;

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            iteration_num += 1;
            // At specified intervals, write visualization files
            if (IBTK::abs_equal_eps(loop_time, next_viz_dump_time, 0.1 * dt) || loop_time >= next_viz_dump_time)
            {
                // Compute errors
                ADS::allocate_patch_data({ z_exa_idx, z_err_idx, g_exa_idx, g_err_idx },
                                         patch_hierarchy,
                                         loop_time,
                                         0,
                                         patch_hierarchy->getFinestLevelNumber());
                HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
                HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy);
                z_exa_fcn.setDataOnPatchHierarchy(z_exa_idx, z_var, patch_hierarchy, loop_time);
                hier_cc_data_ops.subtract(z_err_idx, z_exa_idx, z_idx);
                g_exa_fcn.setDataOnPatchHierarchy(g_exa_idx, g_var, patch_hierarchy, loop_time);
                hier_cc_data_ops.subtract(g_err_idx, g_exa_idx, g_idx);
                const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
                pout << "Errors in z at time " << loop_time << "\n";
                pout << "  L1-norm:  " << hier_cc_data_ops.L1Norm(z_err_idx, wgt_idx) << "\n";
                pout << "  L2-norm:  " << hier_cc_data_ops.L2Norm(z_err_idx, wgt_idx) << "\n";
                pout << "  max-norm: " << hier_cc_data_ops.maxNorm(z_err_idx, wgt_idx) << "\n";
                pout << "Errors in g at time " << loop_time << "\n";
                pout << "  L1-norm:  " << hier_cc_data_ops.L1Norm(g_err_idx, wgt_idx) << "\n";
                pout << "  L2-norm:  " << hier_cc_data_ops.L2Norm(g_err_idx, wgt_idx) << "\n";
                pout << "  max-norm: " << hier_cc_data_ops.maxNorm(g_err_idx, wgt_idx) << "\n";
                pout << "\nWriting visualization files...\n\n";
                adv_diff_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                next_viz_dump_time += viz_dump_time_interval;
                ADS::deallocate_patch_data({ z_exa_idx, z_err_idx, g_exa_idx, g_err_idx },
                                           patch_hierarchy,
                                           0,
                                           patch_hierarchy->getFinestLevelNumber());
            }

            if (dump_timer_data && (iteration_num % timer_dump_interval == 0))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
        }
    } // cleanup dynamically allocated objects prior to shutdown
} // main
