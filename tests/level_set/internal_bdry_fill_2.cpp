#include <ibamr/config.h>

#include <ADS/ExtrapolatedConvectiveOperator.h>
#include <ADS/InternalBdryFill.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>

#include <ibamr/AdvDiffConvectiveOperatorManager.h>

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

        for (NodeIterator<NDIM> ni(box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();

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
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CellVariable<NDIM, double>> N_var = new CellVariable<NDIM, double>("N");
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("U");
        Pointer<NodeVariable<NDIM, double>> u_draw_var = new NodeVariable<NDIM, double>("U_DRAW", NDIM);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        Pointer<VariableContext> cur_ctx = var_db->getContext("CURRENT");
        Pointer<VariableContext> new_ctx = var_db->getContext("NEW");
        Pointer<VariableContext> scr_ctx = var_db->getContext("SCRATCH");
        const int phi_idx = var_db->registerVariableAndContext(phi_var, ctx, IntVector<NDIM>(2));
        const int Q_cur_idx = var_db->registerVariableAndContext(Q_var, cur_ctx, IntVector<NDIM>(1));
        const int Q_new_idx = var_db->registerVariableAndContext(Q_var, new_ctx, IntVector<NDIM>(0));
        const int Q_scr_idx = var_db->registerVariableAndContext(Q_var, scr_ctx, IntVector<NDIM>(1));
        const int N_idx = var_db->registerVariableAndContext(N_var, ctx);
        const int N_old_idx = var_db->registerVariableAndContext(N_var, cur_ctx);
        const int u_idx = var_db->registerVariableAndContext(u_var, ctx, IntVector<NDIM>(1));
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw_var, ctx);
        std::set<int> idx_set{ phi_idx, Q_cur_idx, Q_new_idx, Q_scr_idx, N_idx, N_old_idx, u_idx, u_draw_idx };
        ComponentSelector idxs;
        for (const auto& idx : idx_set) idxs.setFlag(idx);

        std::vector<RobinBcCoefStrategy<NDIM>*> Q_bc_coefs(1, nullptr);
        std::vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, nullptr);

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
#define DRAW_DATA 1
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_cur_idx);
        visit_data_writer->registerPlotQuantity("phi", "SCALAR", phi_idx);
        visit_data_writer->registerPlotQuantity("N", "SCALAR", N_idx);
        visit_data_writer->registerPlotQuantity("U", "VECTOR", u_draw_idx);
#endif

        double t = input_db->getDouble("T_START");
        double dt = input_db->getDouble("DT");
        const double t_final = input_db->getDouble("T_FINAL");
        double next_draw_time = t;
        const double draw_freq = input_db->getDouble("DRAW_FREQ");

        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, t, coarsest_ln, finest_ln);

        // Fill level set data.
        LSFcn ls_fcn("LSFcn", input_db->getDatabase("LSFcn"));
        ls_fcn.setDataOnPatchHierarchy(phi_idx, phi_var, patch_hierarchy, t);

        // Fill initial conditions
        QFcn Q_fcn("QFcn", input_db->getDatabase("QFcn"));
        Q_fcn.setDataOnPatchHierarchy(Q_cur_idx, Q_var, patch_hierarchy, t);

        // Velocity function
        muParserCartGridFunction u_fcn("UFcn", input_db->getDatabase("UFcn"), grid_geometry);

        // Setup SAMRAI vectors
        SAMRAIVectorReal<NDIM, double> q_vec("Q", patch_hierarchy, coarsest_ln, finest_ln);
        SAMRAIVectorReal<NDIM, double> n_vec("N", patch_hierarchy, coarsest_ln, finest_ln);
        q_vec.addComponent(Q_var, Q_scr_idx);
        n_vec.addComponent(N_var, N_idx);

        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
        ExtrapolatedConvectiveOperator convec_op("ExtrapOp",
                                                 Q_var,
                                                 input_db->getDatabase("ConvecOp"),
                                                 ConvectiveDifferencingType::CONSERVATIVE,
                                                 Q_bc_coefs,
                                                 5);

        int iter_num = 0;
        do
        {
            pout << "Simulation time is " << t << "\n\n";

#ifdef DRAW_DATA
            if (t >= next_draw_time)
            {
                pout << "Writing visualization files at time " << t << "\n";
                visit_data_writer->writePlotData(patch_hierarchy, iter_num, t);
                next_draw_time += draw_freq;
            }
#endif

            dt = std::min(dt, t_final - t);
            pout << "Time step size: " << dt << "\n";

            // Set level set and velocity
            ls_fcn.setDataOnPatchHierarchy(phi_idx, phi_var, patch_hierarchy, t);
            u_fcn.setDataOnPatchHierarchy(u_idx, u_var, patch_hierarchy, t);
            u_fcn.setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, t);
            hier_cc_data_ops.copyData(Q_scr_idx, Q_cur_idx);

            convec_op.setSolutionTime(t);
            convec_op.setAdvectionVelocity(u_idx);
            convec_op.setTimeInterval(t, t + dt);
            convec_op.setLSData(phi_idx, phi_var);
            convec_op.initializeOperatorState(q_vec, n_vec);
            convec_op.apply(q_vec, n_vec);

            if (iter_num == 0)
            {
                // Now perform forward Euler step
                pout << "Completing forward Euler update\n";
                hier_cc_data_ops.linearSum(Q_new_idx, 1.0, Q_cur_idx, -1.0 * dt, N_idx);
            }
            else
            {
                pout << "Completing AB2 step\n";
                hier_cc_data_ops.linearSum(Q_new_idx, 1.0, Q_cur_idx, -3.0 * dt / 2.0, N_idx);
                hier_cc_data_ops.linearSum(Q_new_idx, 1.0, Q_new_idx, 0.5 * dt, N_old_idx);
            }

            // Update time
            t += dt;

            // Update for next time step
            hier_cc_data_ops.copyData(Q_cur_idx, Q_new_idx);
            hier_cc_data_ops.copyData(N_old_idx, N_idx);
            convec_op.deallocateOperatorState();

            pout << "Finished time step\n\n";

        } while (t < t_final && ++iter_num);

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, ++iter_num, t);
#endif

        // Deallocate data
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
