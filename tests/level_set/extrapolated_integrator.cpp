#include <ibamr/config.h>

#include <ADS/ExtrapolatedAdvDiffHierarchyIntegrator.h>
#include <ADS/LSFromLevelSet.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include "ibtk/CartGridFunctionSet.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include "tbox/Pointer.h"

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

class LSFcn : public IBTK::CartGridFunction
{
public:
    LSFcn(std::string object_name, Pointer<Database> db) : CartGridFunction(std::move(object_name))
    {
        d_R1 = db->getDouble("r1");
        d_R2 = db->getDouble("r2");
    }

    bool isTimeDependent() const override
    {
        return true;
    }

    void setDataOnPatch(const int data_idx,
                        Pointer<hier::Variable<NDIM>> /*var*/,
                        Pointer<Patch<NDIM>> patch,
                        const double data_time,
                        const bool /*initial_time*/,
                        Pointer<PatchLevel<NDIM>> /*level*/) override
    {
        Pointer<NodeData<NDIM, double>> ls_n_data = patch->getPatchData(data_idx);

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();

        const Box<NDIM>& box = patch->getBox();
        const hier::Index<NDIM>& idx_low = box.lower();

        for (NodeIterator<NDIM> ni(box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();

            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)));
            const double r = x.norm();
            const double ls_disk = std::max(r - d_R2, d_R1 - r);
            (*ls_n_data)(idx) = ls_disk;
        }
    }

private:
    double d_R1, d_R2;
};

class QFcn : public IBTK::CartGridFunction
{
public:
    QFcn(std::string object_name, Pointer<Database> db) : CartGridFunction(std::move(object_name))
    {
        d_outer_rad = db->getDouble("outer_rad");
        db->getDoubleArray("cent_1", d_cent_1.data(), NDIM);
        db->getDoubleArray("cent_2", d_cent_2.data(), NDIM);
    }

    bool isTimeDependent() const override
    {
        return true;
    }

    void setDataOnPatch(const int Q_idx,
                        Pointer<hier::Variable<NDIM>> /*var*/,
                        Pointer<Patch<NDIM>> patch,
                        const double data_time,
                        const bool /*initial_time*/,
                        Pointer<PatchLevel<NDIM>> /*level*/) override
    {
        Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();

        const Box<NDIM>& box = patch->getBox();
        const hier::Index<NDIM>& idx_low = box.lower();

        for (CellIterator<NDIM> ci(box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();

            VectorNd x;
            for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
            const double r1 = (x - d_cent_1).norm();
            const double r2 = (x - d_cent_2).norm();
            if (r1 < 1.0 && x.norm() < d_outer_rad)
                (*Q_data)(idx) = std::pow(std::cos(M_PI * r1) + 1.0, 2.0);
            else if (r2 < 1.0 && x.norm() > d_outer_rad)
                (*Q_data)(idx) = std::pow(std::cos(M_PI * r2) + 1.0, 2.0);
            else
                (*Q_data)(idx) = 0.0;
        }
    }

private:
    double d_outer_rad;
    VectorNd d_cent_1, d_cent_2;
};

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
        Pointer<ExtrapolatedAdvDiffHierarchyIntegrator> adv_diff_integrator =
            new ExtrapolatedAdvDiffHierarchyIntegrator("AdvDiffIntegrator", input_db->getDatabase("AdvDiffIntegrator"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               adv_diff_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        Pointer<NodeVariable<NDIM, double>> phi_var = new NodeVariable<NDIM, double>("phi_var");
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("U");
        Pointer<NodeVariable<NDIM, double>> u_draw_var = new NodeVariable<NDIM, double>("U_DRAW", NDIM);

        adv_diff_integrator->registerTransportedQuantity(Q_var);

        std::vector<RobinBcCoefStrategy<NDIM>*> Q_bc_coefs(1, nullptr);
        std::vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, nullptr);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw_var, ctx);
        std::set<int> idx_set{ u_draw_idx };
        ComponentSelector idxs;
        for (const auto& idx : idx_set) idxs.setFlag(idx);

// Uncomment to draw data.
#define DRAW_DATA 1
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("U", "VECTOR", u_draw_idx);
        adv_diff_integrator->registerVisItDataWriter(visit_data_writer);
#endif

        Pointer<CartGridFunction> ls_fcn = new LSFcn("LSFcn", input_db->getDatabase("LSFcn"));
        Pointer<LSFromLevelSet> phi_fcn = new LSFromLevelSet("PhiFcn", patch_hierarchy);
        phi_fcn->registerLSFcn(ls_fcn);
        adv_diff_integrator->registerLevelSetVariable(phi_var, phi_fcn);
        adv_diff_integrator->restrictToLevelSet(Q_var, phi_var);

        Pointer<CartGridFunction> Q_init_fcn = new QFcn("QFcn", input_db->getDatabase("QFcn"));
        adv_diff_integrator->setInitialConditions(Q_var, Q_init_fcn);

        Pointer<CartGridFunction> u_fcn =
            new muParserCartGridFunction("UFcn", input_db->getDatabase("UFcn"), grid_geometry);
        adv_diff_integrator->registerAdvectionVelocity(u_var);
        adv_diff_integrator->setAdvectionVelocityFunction(u_var, u_fcn);
        adv_diff_integrator->setAdvectionVelocity(Q_var, u_var);

        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        double t = input_db->getDouble("T_START");
        double dt = input_db->getDouble("DT");
        const double t_final = input_db->getDouble("T_FINAL");
        double next_draw_time = t;
        const double draw_freq = input_db->getDouble("DRAW_FREQ");

        int iter_num = 0;
        while (!IBTK::rel_equal_eps(t, t_final) && adv_diff_integrator->stepsRemaining())
        {
            pout << "Simulation time is " << t << "\n\n";

#ifdef DRAW_DATA
            if (t >= next_draw_time)
            {
                pout << "\nWriting visualization files...\n\n";
                // Allocate data
                const int coarsest_ln = 0;
                const int finest_ln = patch_hierarchy->getFinestLevelNumber();
                allocate_patch_data(idxs, patch_hierarchy, t, coarsest_ln, finest_ln);
                u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, t);
                adv_diff_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iter_num, t);
                deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
                next_draw_time += draw_freq;
            }
#endif

            dt = adv_diff_integrator->getMaximumTimeStepSize();
            adv_diff_integrator->advanceHierarchy(dt);
            t += dt;
            ++iter_num;

            pout << "Finished time step\n\n";
        }

#ifdef DRAW_DATA
        pout << "\nWriting visualization files...\n\n";
        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, t, coarsest_ln, finest_ln);
        u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, t);
        adv_diff_integrator->setupPlotData();
        visit_data_writer->writePlotData(patch_hierarchy, iter_num, t);
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
        next_draw_time += draw_freq;
#endif

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
