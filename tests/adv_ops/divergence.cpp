#include <ibamr/config.h>

#include <ADS/LSFromLevelSet.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/RBFDivergenceReconstructions.h>
#include <ADS/app_namespaces.h>

#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

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
            "StandardTagAndInitialize", nullptr, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Create Eulerian boundary condition specification objects.
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, static_cast<RobinBcCoefStrategy<NDIM>*>(nullptr));

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<SideVariable<NDIM, double>> u_var = new SideVariable<NDIM, double>("u");
        Pointer<CellVariable<NDIM, double>> div_var = new CellVariable<NDIM, double>("div");
        Pointer<CellVariable<NDIM, double>> path_var = new CellVariable<NDIM, double>("path", NDIM);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(4));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(4));
        const int u_idx = var_db->registerVariableAndContext(u_var, ctx, IntVector<NDIM>(0));
        const int div_idx = var_db->registerVariableAndContext(div_var, ctx, IntVector<NDIM>(0));
        const int div_err_idx = var_db->registerVariableAndContext(div_var, var_db->getContext("ERR"));
        const int path_idx = var_db->registerVariableAndContext(path_var, ctx);

        ComponentSelector comps;
        comps.setFlag(ls_idx);
        comps.setFlag(vol_idx);
        comps.setFlag(u_idx);
        comps.setFlag(div_idx);
        comps.setFlag(div_err_idx);
        comps.setFlag(path_idx);

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

// Uncomment to draw data.
// #define DRAW_DATA 1
#ifdef DRAW_DATA
        std::unique_ptr<ExodusII_IO> reaction_exodus_io(new ExodusII_IO(*meshes[REACTION_MESH_ID]));
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("div", "SCALAR", div_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(comps);
        }

        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    for (int d = 0; d < NDIM; ++d) (*path_data)(idx, d) = static_cast<double>(idx(d)) + 0.5;
                }
            }
        }

        LSFromLevelSet ls_vol_fcn("ls", patch_hierarchy);
        Pointer<CartGridFunction> ls_hier_fcn =
            new muParserCartGridFunction("ls_fcn", input_db->getDatabase("ls_fcn"), grid_geometry);
        ls_vol_fcn.registerLSFcn(ls_hier_fcn);
        ls_vol_fcn.updateVolumeAreaSideLS(
            vol_idx, vol_var, IBTK::invalid_index, nullptr, IBTK::invalid_index, nullptr, ls_idx, ls_var, 0.0);

        HierarchyMathOps hier_math_ops("hier_ops", patch_hierarchy);
        hier_math_ops.setPatchHierarchy(patch_hierarchy);
        const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy);

        muParserCartGridFunction u_fcn("u", input_db->getDatabase("u"), grid_geometry);
        muParserCartGridFunction div_fcn("div", input_db->getDatabase("div"), grid_geometry);

        u_fcn.setDataOnPatchHierarchy(u_idx, u_var, patch_hierarchy, 0.0);

        RBFDivergenceReconstructions div_ops("div_ops", input_db->getDatabase("div_ops"));
        div_ops.setLSData(ls_idx, vol_idx, ls_idx, vol_idx);
        div_ops.allocateOperatorState(patch_hierarchy, 0.0, 0.0);
        div_ops.applyReconstruction(u_idx, div_idx, path_idx);
        div_ops.deallocateOperatorState();

        // Compute error
        div_fcn.setDataOnPatchHierarchy(div_err_idx, div_var, patch_hierarchy, 0.0);
        hier_cc_data_ops.subtract(div_err_idx, div_err_idx, div_idx);
        pout << "Computing error in divergence interpolation:\n";
        pout << " L1-norm:  " << hier_cc_data_ops.L1Norm(div_err_idx, wgt_idx) << "\n";
        pout << " L2-norm:  " << hier_cc_data_ops.L2Norm(div_err_idx, wgt_idx) << "\n";
        pout << " max-norm: " << hier_cc_data_ops.maxNorm(div_err_idx, wgt_idx) << "\n";

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
#endif

        // Deallocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(comps);
        }

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
