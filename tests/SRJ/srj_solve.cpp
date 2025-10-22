#include <ibamr/config.h>

#include <ADS/CCSharpInterfaceFACPreconditionerStrategy.h>
#include <ADS/CCSharpInterfaceLaplaceOperator.h>
#include <ADS/CCSharpInterfaceScheduledJacobiSolver.h>
#include <ADS/CutCellMeshMapping.h>
#include <ADS/FullFACPreconditioner.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/SharpInterfaceGhostFill.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/sharp_interface_utilities.h>

#include "ibtk/CartGridFunctionSet.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/CCLaplaceOperator.h>
#include <ibtk/CCPoissonSolverManager.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/PETScKrylovLinearSolver.h>
#include <ibtk/PETScKrylovPoissonSolver.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include "tbox/Pointer.h"

#include <libmesh/boundary_mesh.h>
#include <libmesh/communicator.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>

// Local includes
#include "SRJLaplaceSolver.cpp"

double
x_fcn(double, const VectorNd& x, double)
{
    return std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
}

double
bdry_fcn(const VectorNd& x)
{
    return x_fcn(0.0, x, 0.0);
}

double
lap_fcn(double, const VectorNd& x, double)
{
    return -8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
}

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
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    PetscOptionsSetValue(nullptr, "-solver_ksp_rtol", "1.0e-12");

    // Suppress a warning
    SAMRAI::tbox::Logger::getInstance()->setWarning(false);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");
        const string reaction_exodus_filename = app_initializer->getExodusIIFilename("reaction");

        // Get various standard options set in the input file.
        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
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

        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", NULL, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        Pointer<CellVariable<NDIM, double>> X_var = new CellVariable<NDIM, double>("X");
        Pointer<CellVariable<NDIM, double>> err_var = new CellVariable<NDIM, double>("Error");
        Pointer<CellVariable<NDIM, double>> Y_var = new CellVariable<NDIM, double>("Y");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int X_idx = var_db->registerVariableAndContext(X_var, ctx, IntVector<NDIM>(1));
        const int err_idx = var_db->registerVariableAndContext(err_var, ctx, IntVector<NDIM>(1));
        const int Y_idx = var_db->registerVariableAndContext(Y_var, ctx, IntVector<NDIM>(1));
        std::set<int> idx_set{ X_idx, err_idx, Y_idx };
        ComponentSelector idxs;
        for (const auto& idx : idx_set) idxs.setFlag(idx);

        gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
        int tag_buffer = 1;
        int ln = 0;
        bool done = false;
        while (!done && (gridding_algorithm->levelCanBeRefined(ln)))
        {
            gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, tag_buffer);
            done = !patch_hierarchy->finerLevelExists(ln);
            ++ln;
        }

        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, 0.0, coarsest_ln, finest_ln);

        // Uncomment to draw data.
#define DRAW_DATA 1
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("X", "SCALAR", X_idx);
        visit_data_writer->registerPlotQuantity("Y", "SCALAR", Y_idx);
        visit_data_writer->registerPlotQuantity("err", "SCALAR", err_idx);
#endif

        PointwiseFunction<PointwiseFunctions::ScalarFcn> X_fcn("X_fcn", x_fcn);
        X_fcn.setDataOnPatchHierarchy(err_idx, err_var, patch_hierarchy, 0.0);
        PointwiseFunction<PointwiseFunctions::ScalarFcn> Y_fcn("Y_fcn", lap_fcn);
        Y_fcn.setDataOnPatchHierarchy(Y_idx, Y_var, patch_hierarchy, 0.0);

        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        hier_math_ops.resetLevels(coarsest_ln, finest_ln);
        const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> x_vec("X_vec", patch_hierarchy, coarsest_ln, finest_ln),
            y_vec("Y_vec", patch_hierarchy, coarsest_ln, finest_ln);
        x_vec.addComponent(X_var, X_idx, wgt_cc_idx);
        x_vec.setToScalar(0.0);
        y_vec.addComponent(Y_var, Y_idx, wgt_cc_idx);

        sharp_interface::SRJLaplaceSolver solver("SRJSolver", input_db->getDatabase("SRJSolver"));

        solver.initializeSolverState(x_vec, y_vec);

        solver.solveSystem(x_vec, y_vec);

        // Compute error
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
        hier_cc_data_ops.subtract(err_idx, err_idx, X_idx);
        pout << "Error in X:\n"
             << "  L1-norm:  " << std::setprecision(10) << hier_cc_data_ops.L1Norm(err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << std::setprecision(10) << hier_cc_data_ops.L2Norm(err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << std::setprecision(10) << hier_cc_data_ops.maxNorm(err_idx, wgt_cc_idx) << "\n";

        HierarchyCellDataOpsReal<NDIM, int> hier_cc_int_data_ops(patch_hierarchy, coarsest_ln, finest_ln);

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif

        // Deallocate data
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
