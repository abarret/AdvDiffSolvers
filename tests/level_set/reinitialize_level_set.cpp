#include <ibamr/config.h>

#include <ADS/PointwiseFunction.h>
#include <ADS/ReinitializeLevelSet.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include "ibtk/CartGridFunctionSet.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
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

        const double dx = input_db->getDouble("DX");

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

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");
        Pointer<NodeVariable<NDIM, double>> phi_var = new NodeVariable<NDIM, double>("phi_var");
        Pointer<NodeVariable<NDIM, double>> exact_var = new NodeVariable<NDIM, double>("exact_var");
        Pointer<NodeVariable<NDIM, double>> error_var = new NodeVariable<NDIM, double>("error_var");
        Pointer<NodeVariable<NDIM, int>> fixed_var = new NodeVariable<NDIM, int>("fixed");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(2));
        const int phi_idx = var_db->registerVariableAndContext(phi_var, ctx, IntVector<NDIM>(2));
        const int exact_idx = var_db->registerVariableAndContext(exact_var, ctx, IntVector<NDIM>(0));
        const int error_idx = var_db->registerVariableAndContext(error_var, ctx, IntVector<NDIM>(0));
        const int fixed_idx = var_db->registerVariableAndContext(fixed_var, ctx, IntVector<NDIM>(1));
        std::set<int> idx_set{ ls_idx, phi_idx, exact_idx, error_idx, fixed_idx };
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

// Uncomment to draw data.
// #define DRAW_DATA 1
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("phi", "SCALAR", phi_idx);
        visit_data_writer->registerPlotQuantity("exact", "SCALAR", exact_idx);
        visit_data_writer->registerPlotQuantity("error", "SCALAR", error_idx);
        visit_data_writer->registerPlotQuantity("fixed", "SCALAR", fixed_idx);
#endif

        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, 0.0, coarsest_ln, finest_ln);

        // Fill level set data.
        auto ls_ghost_box_fcn =
            [dx](Pointer<Patch<NDIM>> patch, const int ls_idx, const int phi_idx, const int fixed_idx)
        {
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const h = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();

            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            Pointer<NodeData<NDIM, int>> fixed_data = patch->getPatchData(fixed_idx);
            for (NodeIterator<NDIM> ni(fixed_data->getGhostBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + h[d] * (static_cast<double>(idx(d) - idx_low(d)));
                const double r = x.norm();
                static const double R = 1.0;
                if (r < R)
                {
                    if (r < (R - 4.0 * dx))
                    {
                        (*ls_data)(idx) = (*phi_data)(idx) = -1.0;
                        (*fixed_data)(idx) = 0;
                    }
                    else
                    {
                        (*ls_data)(idx) = (*phi_data)(idx) = r - R;
                        (*fixed_data)(idx) = 1;
                    }
                }
                else
                {
                    if (r > (R + 4.0 * dx))
                    {
                        (*ls_data)(idx) = (*phi_data)(idx) = 1.0;
                        (*fixed_data)(idx) = 0;
                    }
                    else
                    {
                        (*ls_data)(idx) = (*phi_data)(idx) = r - R;
                        (*fixed_data)(idx) = 1;
                    }
                }
            }

            // Specify "boundary" values as invalid (we are using periodic boundaries for a problem that is not
            // periodic).
            const int ln = patch->getPatchLevelNumber();
            if (ln == 0)
            {
                Box<NDIM> shrunk_box = patch->getBox();
                shrunk_box.grow(-1);
                for (NodeIterator<NDIM> ni(fixed_data->getGhostBox()); ni; ni++)
                {
                    const NodeIndex<NDIM>& idx = ni();
                    if (!shrunk_box.contains(idx)) (*fixed_data)(idx) = 2;
                }
            }
        };
        perform_on_patch_hierarchy(patch_hierarchy, ls_ghost_box_fcn, ls_idx, phi_idx, fixed_idx);

        // Set exact
        PointwiseFunctions::ScalarFcn exact_pt_fcn = [](double, const VectorNd& x, double) -> double
        {
            const double r = x.norm();
            static const double R = 1.0;
            return r - R;
        };
        PointwiseFunction<PointwiseFunctions::ScalarFcn> exact_fcn("exact", exact_pt_fcn);
        exact_fcn.setDataOnPatchHierarchy(exact_idx, exact_var, patch_hierarchy, 0.0);

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif
        // Now transform level set into signed distance function
        ReinitializeLevelSet find_level_set("ReinitLS", input_db->getDatabase("ReinitLS"));
        find_level_set.computeSignedDistanceFunction(phi_idx, *phi_var, patch_hierarchy, 0.0, fixed_idx);

        // Determine error
        HierarchyNodeDataOpsReal<NDIM, double> hier_nc_data_ops(patch_hierarchy);
        hier_nc_data_ops.subtract(error_idx, exact_idx, phi_idx);
        plog << "Error: " << hier_nc_data_ops.max(error_idx) << "\n";

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
#endif

        // Deallocate data
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
