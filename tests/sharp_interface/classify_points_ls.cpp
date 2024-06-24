#include <ibamr/config.h>

#include <ADS/PointwiseFunction.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/sharp_interface_utilities.h>

#include "ibtk/CartGridFunctionSet.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include "tbox/Pointer.h"

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

enum class InterfaceType
{
    DISK,
    PERISTALSIS,
    CHANNEL,
    SPHERE,
    UNKNOWN
};

std::string
enum_to_string(InterfaceType e)
{
    switch (e)
    {
    case InterfaceType::DISK:
        return "DISK";
        break;
    case InterfaceType::PERISTALSIS:
        return "PERISTALSIS";
        break;
    case InterfaceType::CHANNEL:
        return "CHANNEL";
        break;
    case InterfaceType::SPHERE:
        return "SPHERE";
        break;
    default:
        return "UNKNOWN";
        break;
    }
}

InterfaceType
string_to_enum(const std::string& str)
{
    if (strcasecmp(str.c_str(), "DISK") == 0) return InterfaceType::DISK;
    if (strcasecmp(str.c_str(), "PERISTALSIS") == 0) return InterfaceType::PERISTALSIS;
    if (strcasecmp(str.c_str(), "CHANNEL") == 0) return InterfaceType::CHANNEL;
    if (strcasecmp(str.c_str(), "SPHERE") == 0) return InterfaceType::SPHERE;
    return InterfaceType::UNKNOWN;
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

        Pointer<NodeVariable<NDIM, double>> phi_var = new NodeVariable<NDIM, double>("phi_var");
        Pointer<CellVariable<NDIM, int>> pt_type_var = new CellVariable<NDIM, int>("pt_type");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int phi_idx = var_db->registerVariableAndContext(phi_var, ctx, IntVector<NDIM>(2));
        const int pt_type_idx = var_db->registerVariableAndContext(pt_type_var, ctx, IntVector<NDIM>(1));
        std::set<int> idx_set{ phi_idx, pt_type_idx };
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

        InterfaceType interface = string_to_enum(input_db->getString("INTERFACE_TYPE"));

// Uncomment to draw data.
#define DRAW_DATA 1
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("Pt_type", "SCALAR", pt_type_idx);
        visit_data_writer->registerPlotQuantity("phi", "SCALAR", phi_idx);
#endif

        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, 0.0, coarsest_ln, finest_ln);

        // Fill level set data.
        if (interface == InterfaceType::DISK)
        {
            auto ls_ghost_box_fcn = [](Pointer<Patch<NDIM>> patch, const int phi_idx)
            {
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const dx = pgeom->getDx();
                const double* const xlow = pgeom->getXLower();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();

                Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
                for (NodeIterator<NDIM> ni(phi_data->getGhostBox()); ni; ni++)
                {
                    const NodeIndex<NDIM>& idx = ni();
                    VectorNd x;
                    for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)));
                    const double r = x.norm();
                    static const double R = 1.0;
                    (*phi_data)(idx) = r - R;
                }
            };
            perform_on_patch_hierarchy(patch_hierarchy, ls_ghost_box_fcn, phi_idx);
        }
        else if (interface == InterfaceType::CHANNEL)
        {
            double theta = input_db->getDouble("THETA");
            double ylow = input_db->getDouble("YLOW");
            double yup = input_db->getDouble("YUP");
            auto ls_ghost_box_fcn = [&theta, &ylow, &yup](Pointer<Patch<NDIM>> patch, const int phi_idx)
            {
                MatrixNd Q;
                Q(0, 0) = Q(1, 1) = std::cos(theta);
                Q(0, 1) = std::sin(theta);
                Q(1, 0) = -std::sin(theta);
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const dx = pgeom->getDx();
                const double* const xlow = pgeom->getXLower();
                const hier::Index<NDIM>& idx_low = patch->getBox().lower();

                Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
                for (NodeIterator<NDIM> ni(phi_data->getGhostBox()); ni; ni++)
                {
                    const NodeIndex<NDIM>& idx = ni();
                    VectorNd x;
                    for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)));
                    x = Q * x;
                    (*phi_data)(idx) = std::max(x[1] - yup, ylow - x[1]);
                }
            };
            perform_on_patch_hierarchy(patch_hierarchy, ls_ghost_box_fcn, phi_idx);
        }

        sharp_interface::classify_points(pt_type_idx, phi_idx, patch_hierarchy, 0.0);

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif

        // Count number of points
        int fp_num = 0, gp_num = 0, invalid_num = 0;
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, int>> pt_type = patch->getPatchData(pt_type_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    switch ((*pt_type)(idx))
                    {
                    case sharp_interface::FLUID:
                        ++fp_num;
                        break;
                    case sharp_interface::GHOST:
                        ++gp_num;
                        break;
                    case sharp_interface::INVALID:
                        ++invalid_num;
                        break;
                    default:
                        TBOX_ERROR("UNKNOWN_TYPE");
                    }
                }
            }
        }
        plog << "Num fluid points:   " << fp_num << "\n";
        plog << "Num ghost points:   " << gp_num << "\n";
        plog << "Num invalid points: " << invalid_num << "\n";

        // Deallocate data
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
