#include "ibamr/config.h"

#include <ADS/CutCellMeshMapping.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/LSCutCellLaplaceOperator.h>
#include <ADS/LSFromLevelSet.h>
#include <ADS/LSFromMesh.h>
#include <ADS/SBAdvDiffIntegrator.h>
#include <ADS/SBBoundaryConditions.h>
#include <ADS/SBIntegrator.h>
#include <ADS/app_namespaces.h>

#include <ibamr/FESurfaceDistanceEvaluator.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/IBFESurfaceMethod.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/RelaxationLSMethod.h>

#include "ibtk/CartGridFunctionSet.h"
#include "ibtk/IBTK_MPI.h"
#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"

#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>
#include <utility>

// Local includes
#include "QFcn.h"

static double k_on, k_off, sf_max;
double
a_fcn(double Q_bdry, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double time, void* ctx)
{
    return k_on * (sf_max - sf_vals[0]) * Q_bdry;
}

double
g_fcn(double Q_bdry, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double time, void* ctx)
{
    return k_off * sf_vals[0];
}

double
sf_ode(double q, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double time, void* ctx)
{
    return k_on * (sf_max - q) * fl_vals[0] - k_off * q;
}

void checkConservation(FESystemManager& fe_sys_manager,
                       const std::string& sys_name,
                       const int Q_idx,
                       const int vol_idx,
                       Pointer<PatchHierarchy<NDIM>> hierarchy,
                       const double time);

double y_low, y_up, theta, L;

void
bdry_fcn(const IBTK::VectorNd& x, double& ls_val)
{
    // Rotate point
    MatrixNd rot;
    rot(0, 0) = rot(1, 1) = cos(theta);
    rot(1, 0) = -sin(theta);
    rot(0, 1) = sin(theta);
    VectorNd cent;
    cent(1) = 0.5 * (y_low + y_up);
    cent(0) = 0.0;
    VectorNd xnew = rot * (x - cent) + cent;
    ls_val = std::max<double>(y_low - xnew(1), xnew(1) - y_up);
    // Now find distance from y_low and y_up.
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
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

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
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
        const string lower_exodus_filename = app_initializer->getExodusIIFilename("lower");
        const string upper_exodus_filename = app_initializer->getExodusIIFilename("upper");
        const string reaction_exodus_filename = app_initializer->getExodusIIFilename("reaction");

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

        // Create a simple FE mesh.
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;
        theta = input_db->getDouble("THETA"); // channel angle
        L = input_db->getDouble("LX");
        y_low = input_db->getDouble("Y_LOW");
        y_up = input_db->getDouble("Y_UP");
        k_on = input_db->getDouble("K_ON");
        k_off = input_db->getDouble("K_OFF");
        sf_max = input_db->getDouble("SF_MAX");

        string elem_type = input_db->getString("ELEM_TYPE");
        const int second_order_mesh = (input_db->getString("elem_order") == "SECOND");
        string bdry_elem_type = second_order_mesh ? "EDGE3" : "EDGE2";

        Mesh lower_mesh_bdry(init.comm(), NDIM);
        MeshTools::Generation::build_line(lower_mesh_bdry,
                                          static_cast<int>(ceil(L / (ds + 1))),
                                          0.0 - 0.5 * ds,
                                          L + 0.5 * ds,
                                          Utility::string_to_enum<ElemType>(bdry_elem_type));
        for (MeshBase::node_iterator it = lower_mesh_bdry.nodes_begin(); it != lower_mesh_bdry.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) = y_low + std::tan(theta) * X(0);
        }
        lower_mesh_bdry.set_spatial_dimension(NDIM);
        lower_mesh_bdry.prepare_for_use();

        Mesh upper_mesh_bdry(init.comm(), NDIM);
        MeshTools::Generation::build_line(upper_mesh_bdry,
                                          static_cast<int>(ceil(L / (ds + 1))),
                                          0.0 - 0.5 * ds,
                                          L + 0.5 * ds,
                                          Utility::string_to_enum<ElemType>(bdry_elem_type));
        for (MeshBase::node_iterator it = upper_mesh_bdry.nodes_begin(); it != upper_mesh_bdry.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) = y_up + std::tan(theta) * X(0);
        }
        upper_mesh_bdry.set_spatial_dimension(NDIM);
        upper_mesh_bdry.prepare_for_use();

        Mesh reaction_mesh(init.comm(), NDIM);
        const double reaction_fraction = input_db->getDouble("REACT_FRAC");
        MeshTools::Generation::build_line(reaction_mesh,
                                          static_cast<int>(ceil(reaction_fraction * L / ds)),
                                          0.4 * L,
                                          0.4 * L + L * reaction_fraction,
                                          Utility::string_to_enum<ElemType>(bdry_elem_type));
        for (MeshBase::node_iterator it = reaction_mesh.nodes_begin(); it != reaction_mesh.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) = y_up + std::tan(theta) * X(0);
        }
        reaction_mesh.set_spatial_dimension(NDIM);
        reaction_mesh.prepare_for_use();

        static const int LOWER_MESH_ID = 0;
        static const int UPPER_MESH_ID = 1;
        static const int REACTION_MESH_ID = 2;
        vector<MeshBase*> meshes(3);
        meshes[LOWER_MESH_ID] = &lower_mesh_bdry;
        meshes[UPPER_MESH_ID] = &upper_mesh_bdry;
        meshes[REACTION_MESH_ID] = &reaction_mesh;

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<SBAdvDiffIntegrator> adv_diff_integrator = new SBAdvDiffIntegrator(
            "LSAdvDiffIntegrator", app_initializer->getComponentDatabase("LSAdvDiffIntegrator"));

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

        // Create Eulerian boundary condition specification objects.
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, static_cast<RobinBcCoefStrategy<NDIM>*>(NULL));
        const bool periodic_domain = grid_geometry->getPeriodicShift().min() > 0;

        // Setup velocity
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("U");
        adv_diff_integrator->registerAdvectionVelocity(u_var);

        Pointer<CartGridFunction> u_fcn =
            new muParserCartGridFunction("UFcn", app_initializer->getComponentDatabase("UFcn"), grid_geometry);
        adv_diff_integrator->setAdvectionVelocityFunction(u_var, u_fcn);

        // Setup mesh mapping
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), meshes);
        mesh_mapping->initializeEquationSystems();

        // Setup cut cell mapping
        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellMeshMapping("CutCellMapping", app_initializer->getComponentDatabase("CutCellMapping"));
        Pointer<CutCellMeshMapping> rcn_cut_cell_mapping =
            new CutCellMeshMapping("CutCellMapping", app_initializer->getComponentDatabase("CutCellMapping"));

        // Setup the level set function
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        adv_diff_integrator->registerLevelSetVariable(ls_var);
        Pointer<LSFromMesh> vol_fcn = new LSFromMesh("LSFromMesh",
                                                     patch_hierarchy,
                                                     mesh_mapping->getSystemManagers({ LOWER_MESH_ID, UPPER_MESH_ID }),
                                                     cut_cell_mapping);
        vol_fcn->registerBdryFcn(bdry_fcn);
        vol_fcn->registerReverseNormal(UPPER_MESH_ID);
        adv_diff_integrator->registerLevelSetVolFunction(ls_var, vol_fcn);
        adv_diff_integrator->registerGeneralBoundaryMeshMapping(mesh_mapping);

        // Setup advected quantity
        Pointer<CellVariable<NDIM, double>> Q_in_var = new CellVariable<NDIM, double>("Q_in");
        Pointer<CartGridFunction> Q_in_init = new QFcn("QInit", app_initializer->getComponentDatabase("QInitial"));
        std::vector<RobinBcCoefStrategy<NDIM>*> Q_in_bcs(1);
        if (!periodic_domain)
        {
            const std::string Q_bcs_name = "Q_bcs";
            Q_in_bcs[0] =
                new muParserRobinBcCoefs(Q_bcs_name, app_initializer->getComponentDatabase(Q_bcs_name), grid_geometry);
        }

        adv_diff_integrator->registerTransportedQuantity(Q_in_var);
        adv_diff_integrator->setAdvectionVelocity(Q_in_var, u_var);
        adv_diff_integrator->setInitialConditions(Q_in_var, Q_in_init);
        adv_diff_integrator->setPhysicalBcCoef(Q_in_var, Q_in_bcs[0]);
        adv_diff_integrator->setDiffusionCoefficient(Q_in_var, input_db->getDoubleWithDefault("D_coef", 0.0));
        adv_diff_integrator->restrictToLevelSet(Q_in_var, ls_var);

        auto sb_data_manager =
            std::make_shared<SBSurfaceFluidCouplingManager>("SBDataManager",
                                                            app_initializer->getComponentDatabase("SBDataManager"),
                                                            &mesh_mapping->getSystemManager(REACTION_MESH_ID));
        sb_data_manager->registerFluidConcentration(Q_in_var);
        std::string sf_name = "SurfaceConcentration";
        sb_data_manager->registerSurfaceConcentration(sf_name);
        sb_data_manager->registerFluidSurfaceDependence(sf_name, Q_in_var);
        sb_data_manager->registerSurfaceReactionFunction(sf_name, sf_ode, nullptr);
        sb_data_manager->registerFluidBoundaryCondition(Q_in_var, a_fcn, g_fcn, nullptr);
        sb_data_manager->initializeFEData();

        Pointer<SBIntegrator> sb_integrator = new SBIntegrator("SBIntegrator", sb_data_manager);
        adv_diff_integrator->registerSBIntegrator(sb_integrator, ls_var);
        adv_diff_integrator->registerLevelSetSBDataManager(ls_var, sb_data_manager);

        // Set up diffusion operators
        Pointer<LSCutCellLaplaceOperator> rhs_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInRHSOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        Pointer<LSCutCellLaplaceOperator> sol_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        adv_diff_integrator->setHelmholtzRHSOperator(Q_in_var, rhs_in_oper);
        Pointer<PETScKrylovPoissonSolver> Q_in_helmholtz_solver = new PETScKrylovPoissonSolver(
            "PoissonSolver", app_initializer->getComponentDatabase("PoissonSolver"), "poisson_solve_");
        Q_in_helmholtz_solver->setOperator(sol_in_oper);
        adv_diff_integrator->setHelmholtzSolver(Q_in_var, Q_in_helmholtz_solver);

        // Create scratch index for SurfaceBoundaryReactions
        auto var_db = VariableDatabase<NDIM>::getDatabase();

        // Set up visualization plot file writer.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            adv_diff_integrator->registerVisItDataWriter(visit_data_writer);
        }
        std::unique_ptr<ExodusII_IO> lower_exodus_io(
            uses_exodus ? new ExodusII_IO(*mesh_mapping->getBoundaryMesh(LOWER_MESH_ID)) : NULL);
        std::unique_ptr<ExodusII_IO> upper_exodus_io(
            uses_exodus ? new ExodusII_IO(*mesh_mapping->getBoundaryMesh(UPPER_MESH_ID)) : NULL);
        std::unique_ptr<ExodusII_IO> reaction_exodus_io(
            uses_exodus ? new ExodusII_IO(*mesh_mapping->getBoundaryMesh(REACTION_MESH_ID)) : NULL);
        mesh_mapping->initializeFEData();
        // Initialize hierarchy configuration and data on all patches.
        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Create boundary operators. This must be done after the patch hierarchy is created.
        Pointer<SBBoundaryConditions> bdry_conditions =
            new SBBoundaryConditions("SBBoundaryConditions",
                                     sb_data_manager->getFLName(Q_in_var),
                                     sb_data_manager,
                                     rcn_cut_cell_mapping,
                                     { adv_diff_integrator->getFEHierarchyMappings()[REACTION_MESH_ID] });
        bdry_conditions->setFluidContext(adv_diff_integrator->getCurrentContext());
        rhs_in_oper->setBoundaryConditionOperator(bdry_conditions);
        sol_in_oper->setBoundaryConditionOperator(bdry_conditions);

        Pointer<CellVariable<NDIM, double>> u_draw = new CellVariable<NDIM, double>("UDraw", NDIM);
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw, var_db->getContext("Scratch"));
        visit_data_writer->registerPlotQuantity("velocity", "VECTOR", u_draw_idx);

        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_in_var, adv_diff_integrator->getCurrentContext());

        // Close the restart manager.
        RestartManager::getManager()->closeRestartFile();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        double dt = adv_diff_integrator->getMaximumTimeStepSize();

        // Write out initial visualization data.
        EquationSystems* lower_equation_systems = mesh_mapping->getSystemManager(LOWER_MESH_ID).getEquationSystems();
        EquationSystems* upper_equation_systems = mesh_mapping->getSystemManager(UPPER_MESH_ID).getEquationSystems();
        EquationSystems* reaction_eq_sys = mesh_mapping->getSystemManager(REACTION_MESH_ID).getEquationSystems();
        int iteration_num = adv_diff_integrator->getIntegratorStep();
        double loop_time = adv_diff_integrator->getIntegratorTime();
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            adv_diff_integrator->setupPlotData();
            adv_diff_integrator->allocatePatchData(u_draw_idx, 0.0);
            u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw, patch_hierarchy, loop_time);
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            adv_diff_integrator->deallocatePatchData(u_draw_idx);
            if (uses_exodus)
            {
                lower_exodus_io->write_timestep(
                    lower_exodus_filename, *lower_equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                upper_exodus_io->write_timestep(
                    upper_exodus_filename, *upper_equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                reaction_exodus_io->write_timestep(
                    reaction_exodus_filename, *reaction_eq_sys, iteration_num / viz_dump_interval + 1, loop_time);
            }
            const int vol_idx = var_db->mapVariableAndContextToIndex(adv_diff_integrator->getVolumeVariable(ls_var),
                                                                     adv_diff_integrator->getCurrentContext());
            checkConservation(
                mesh_mapping->getSystemManager(REACTION_MESH_ID), sf_name, Q_idx, vol_idx, patch_hierarchy, loop_time);
        }

        // Main time step loop.
        double loop_time_end = adv_diff_integrator->getEndTime();
        while (!MathUtilities<double>::equalEps(loop_time, loop_time_end) && adv_diff_integrator->stepsRemaining())
        {
            iteration_num = adv_diff_integrator->getIntegratorStep();
            loop_time = adv_diff_integrator->getIntegratorTime();

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

            // At specified intervals, write visualization and restart files,
            // and print out timer data.
            iteration_num += 1;
            const bool last_step = !adv_diff_integrator->stepsRemaining();
            if (dump_viz_data && uses_visit && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                adv_diff_integrator->setupPlotData();
                adv_diff_integrator->allocatePatchData(u_draw_idx, 0.0);
                u_fcn->setDataOnPatchHierarchy(u_draw_idx, u_draw, patch_hierarchy, loop_time);
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                adv_diff_integrator->deallocatePatchData(u_draw_idx);
                if (uses_exodus)
                {
                    lower_exodus_io->write_timestep(lower_exodus_filename,
                                                    *lower_equation_systems,
                                                    iteration_num / viz_dump_interval + 1,
                                                    loop_time);
                    upper_exodus_io->write_timestep(upper_exodus_filename,
                                                    *upper_equation_systems,
                                                    iteration_num / viz_dump_interval + 1,
                                                    loop_time);
                    reaction_exodus_io->write_timestep(
                        reaction_exodus_filename, *reaction_eq_sys, iteration_num / viz_dump_interval + 1, loop_time);
                }
                const int vol_idx = var_db->mapVariableAndContextToIndex(adv_diff_integrator->getVolumeVariable(ls_var),
                                                                         adv_diff_integrator->getCurrentContext());
                checkConservation(mesh_mapping->getSystemManager(REACTION_MESH_ID),
                                  sf_name,
                                  Q_idx,
                                  vol_idx,
                                  patch_hierarchy,
                                  loop_time);
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
        }

        if (!periodic_domain) delete Q_in_bcs[0];
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
checkConservation(FESystemManager& fe_sys_manager,
                  const std::string& sys_name,
                  const int Q_idx,
                  const int vol_idx,
                  Pointer<PatchHierarchy<NDIM>> hierarchy,
                  const double time)
{
    double surface_amount = 0.0, fluid_amount = 0.0;
    EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
    System& Q_sys = eq_sys->get_system(sys_name);
    DofMap& Q_dof_map = Q_sys.get_dof_map();
    NumericVector<double>* Q_vec = Q_sys.solution.get();

    const MeshBase& mesh = eq_sys->get_mesh();
    std::unique_ptr<FEBase> fe = FEBase::build(mesh.mesh_dimension(), Q_dof_map.variable_type(0));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, mesh.mesh_dimension(), THIRD);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<std::vector<double>>& phi = fe->get_phi();
    const std::vector<double>& JxW = fe->get_JxW();

    std::vector<dof_id_type> Q_dof_indices;
    boost::multi_array<double, 1> Q_nodes;
    auto it = mesh.local_elements_begin();
    const auto& it_end = mesh.local_elements_end();
    for (; it != it_end; ++it)
    {
        const auto elem = *it;
        Q_dof_map.dof_indices(elem, Q_dof_indices);
        IBTK::get_values_for_interpolation(Q_nodes, *Q_vec, Q_dof_indices);
        fe->reinit(elem);
        for (size_t qp = 0; qp < JxW.size(); ++qp)
        {
            double Q = 0.0;
            for (int n = 0; n < 2; ++n) Q += Q_nodes[n] * phi[n][qp];
            surface_amount += Q * JxW[qp];
        }
    }

    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(hierarchy->getFinestLevelNumber());
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& box = patch->getBox();
        Pointer<CellData<NDIM, double>> q_data = patch->getPatchData(Q_idx);
        Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);

        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();

        for (CellIterator<NDIM> ci(box); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            fluid_amount += (*q_data)(idx) * (*vol_data)(idx)*dx[0] * dx[1];
        }
    }

    fluid_amount = SAMRAI_MPI::sumReduction(fluid_amount);
    surface_amount = SAMRAI_MPI::sumReduction(surface_amount);

    if (SAMRAI_MPI::getRank() == 0)
    {
        plog << "Total amount on surface: " << surface_amount << "\n";
        plog << "Total amount in fluid:   " << fluid_amount << "\n";
        plog << "Total amount total:      " << surface_amount + fluid_amount << "\n";
        std::ofstream output;
        output.open("TotalAmount", time == 0.0 ? std::ofstream::out : std::ofstream::app);
        output << time << " " << surface_amount << " " << fluid_amount << " " << surface_amount + fluid_amount << "\n";
    }
}
