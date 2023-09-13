#include <ibamr/config.h>

#include <ADS/PointwiseFunction.h>
#include <ADS/app_namespaces.h>
#include <ADS/reconstructions.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/ibtk_utilities.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <random>
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
    IBTKInit init(argc, argv, MPI_COMM_WORLD);
    {
        auto f = [](const VectorNd& x) -> double
        {
            return std::pow(std::sin(2.0 * M_PI * x(0)) * std::sin(2.0 * M_PI * x(1))
#if (NDIM == 3)
                                * std::sin(2.0 * M_PI * x(2))
#endif
                                ,
                            1.0);
        };

        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));

        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", nullptr, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Create variables
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CellVariable<NDIM, double>> path_var = new CellVariable<NDIM, double>("Path", NDIM);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("Original"), IntVector<NDIM>(4));
        const int Q_err_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("Error"));
        const int Q_exa_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("Exact"));
        const int Q_int_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("Interp"));
        const int path_idx = var_db->registerVariableAndContext(path_var, var_db->getContext("Path"));
        ComponentSelector comps;
        comps.setFlag(Q_idx);
        comps.setFlag(Q_err_idx);
        comps.setFlag(Q_exa_idx);
        comps.setFlag(Q_int_idx);
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

        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();

        // Allocate data
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(comps, 0.0);
        }

        srand(0);
        // Fill in path data
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);

                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    // Add a random number [-0.5, 0.5] to the cell center
                    for (int d = 0; d < NDIM; ++d)
                        (*path_data)(idx, d) = static_cast<double>(idx(d)) + 0.5 + (d == 0 ? -0.25 : 0.25);
                }
            }
        }

        // Now fill in exact values
        PointwiseFunctions::ScalarFcn Q_fcn = [f](const double /*Q*/, const VectorNd& X, const double /*t*/) -> double
        { return f(X); };
        PointwiseFunction<PointwiseFunctions::ScalarFcn> Q_hier_fcn("PointwiseFunction", Q_fcn);
        Q_hier_fcn.setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, 0.0);

        // Fill in ghost cells
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp = { ITC(Q_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE") };
        HierarchyGhostCellInterpolation ghost_fill;
        ghost_fill.initializeOperatorState(ghost_cell_comp, patch_hierarchy, coarsest_ln, finest_ln);
        ghost_fill.fillData(0.0);

        // Now interpolate the data
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
                Pointer<CellData<NDIM, double>> Q_int_data = patch->getPatchData(Q_int_idx);
                Pointer<CellData<NDIM, double>> Q_exa_data = patch->getPatchData(Q_exa_idx);
                Pointer<CellData<NDIM, double>> Q_err_data = patch->getPatchData(Q_err_idx);
                Pointer<CellData<NDIM, double>> path_data = patch->getPatchData(path_idx);

                const hier::Index<NDIM>& idx_low = patch->getBox().lower();
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const xlow = pgeom->getXLower();
                const double* const dx = pgeom->getDx();

                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    pout << "On index " << idx << "\n";
                    VectorNd x;
                    for (int d = 0; d < NDIM; ++d) x[d] = (*path_data)(idx, d);

                    (*Q_int_data)(idx) = Reconstruct::weno5(*Q_data, idx, x);
                    pout << "Interpolated value: " << (*Q_int_data)(idx) << "\n";

                    // Exact value
                    for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (x[d] - static_cast<double>(idx_low(d)));
                    (*Q_exa_data)(idx) = f(x);
                    pout << "Exact value: " << (*Q_exa_data)(idx) << "\n";

                    // Error
                    (*Q_err_data)(idx) = (*Q_exa_data)(idx) - (*Q_int_data)(idx);
                    pout << "Error: " << (*Q_err_data)(idx) << "\n";
                }
            }
        }

        // Plot error.
// #define USE_VISIT
#ifdef USE_VISIT
        VisItDataWriter<NDIM> visit_writer("viz writer", "viz");
        visit_writer.registerPlotQuantity("Q", "SCALAR", Q_idx);
        visit_writer.registerPlotQuantity("Q_interp", "SCALAR", Q_int_idx);
        visit_writer.registerPlotQuantity("Q_error", "SCALAR", Q_err_idx);
        visit_writer.registerPlotQuantity("Q_exact", "SCALAR", Q_exa_idx);
        visit_writer.writePlotData(patch_hierarchy, 0, 0.0);
#endif

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
