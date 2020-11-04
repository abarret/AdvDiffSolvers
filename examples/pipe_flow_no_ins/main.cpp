// Config files
#include <IBAMR_config.h>
#include <IBTK_config.h>

#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include "tbox/Pointer.h"

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <memory>
#include <utility>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/FESurfaceDistanceEvaluator.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/IBFESurfaceMethod.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/RelaxationLSMethod.h>

#include "ibtk/CartGridFunctionSet.h"
#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>

#include "LS/LSCutCellLaplaceOperator.h"
#include "LS/QInitial.h"

#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>

#include <ibtk/IBTKInit.h>

#include "LS/SemiLagrangianAdvIntegrator.h"

#include "InsideLSFcn.h"
#include "QFcn.h"
#include "SurfaceBoundaryReactions.h"

using namespace LS;

struct LocateInterface
{
public:
    LocateInterface(Pointer<CellVariable<NDIM, double>> ls_var,
                    Pointer<AdvDiffHierarchyIntegrator> integrator,
                    Pointer<CartGridFunction> ls_fcn)
        : d_ls_var(ls_var), d_integrator(integrator), d_ls_fcn(ls_fcn)
    {
        // intentionally blank
    }
    void resetData(const int D_idx, Pointer<HierarchyMathOps> hier_math_ops, const double time, const bool initial_time)
    {
        Pointer<PatchHierarchy<NDIM>> hierarchy = hier_math_ops->getPatchHierarchy();
        if (initial_time)
        {
            d_ls_fcn->setDataOnPatchHierarchy(D_idx, d_ls_var, hierarchy, time, initial_time);
        }
        else
        {
            auto var_db = VariableDatabase<NDIM>::getDatabase();
            const int ls_cur_idx = var_db->mapVariableAndContextToIndex(d_ls_var, d_integrator->getCurrentContext());
            HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(hierarchy, 0, hierarchy->getFinestLevelNumber());
            hier_cc_data_ops.copyData(D_idx, ls_cur_idx);
        }
    }

private:
    Pointer<CellVariable<NDIM, double>> d_ls_var;
    Pointer<AdvDiffHierarchyIntegrator> d_integrator;
    Pointer<CartGridFunction> d_ls_fcn;
};

void
locateInterface(const int D_idx,
                SAMRAI::tbox::Pointer<IBTK::HierarchyMathOps> hier_math_ops,
                const double time,
                const bool initial_time,
                void* ctx)
{
    auto interface = (static_cast<LocateInterface*>(ctx));
    interface->resetData(D_idx, hier_math_ops, time, initial_time);
}

void postprocess_data(Pointer<PatchHierarchy<NDIM>> hierarchy,
                      Pointer<SemiLagrangianAdvIntegrator> integrator,
                      Pointer<CellVariable<NDIM, double>> Q_in_var,
                      int iteration_num,
                      double loop_time,
                      const std::string& dirname);

void
synchFluidConcentration(const int Q_in_idx,
                        Pointer<CellVariable<NDIM, double>> Q_in_var,
                        Pointer<HierarchyIntegrator> integrator,
                        Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(Q_in_idx)) level->allocatePatchData(Q_in_idx);
    }

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int Q_true_idx = var_db->mapVariableAndContextToIndex(Q_in_var, integrator->getCurrentContext());
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> Q_in_data = patch->getPatchData(Q_in_idx);
            Pointer<CellData<NDIM, double>> Q_true_data = patch->getPatchData(Q_true_idx);
            Q_in_data->copy(*Q_true_data);
        }
    }
}

static std::map<std::pair<Elem*, int>, libMesh::Point> s_elem_point_cache;

void
checkConservation(FEDataManager* fe_data_manager, const std::string& sys_name, const int Q_idx, const int vol_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, const double time);

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
        const string vol_mesh_file_name = app_initializer->getExodusIIFilename("vol");

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
        // Create a simple FE mesh.
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;
        double theta = input_db->getDouble("THETA"); // channel angle
        double L = input_db->getDouble("LX");
        double y_low = input_db->getDouble("Y_LOW");
        double y_up = input_db->getDouble("Y_UP");

        string IB_delta_function = input_db->getString("IB_DELTA_FUNCTION");
        string elem_type = input_db->getString("ELEM_TYPE");
        const int second_order_mesh = (input_db->getString("elem_order") == "SECOND");
        string bdry_elem_type = second_order_mesh ? "EDGE3" : "EDGE2";
        static const bool use_boundary_mesh = true;

        Mesh lower_mesh_bdry(init.comm(), NDIM);
        MeshTools::Generation::build_line(
            lower_mesh_bdry, static_cast<int>(ceil(L / ds)), 0.0, L, Utility::string_to_enum<ElemType>(bdry_elem_type));
        for (MeshBase::node_iterator it = lower_mesh_bdry.nodes_begin(); it != lower_mesh_bdry.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) = y_low + std::tan(theta) * X(0);
        }
        lower_mesh_bdry.set_spatial_dimension(NDIM);
        lower_mesh_bdry.prepare_for_use();

        Mesh upper_mesh_bdry(init.comm(), NDIM);
        MeshTools::Generation::build_line(
            upper_mesh_bdry, static_cast<int>(ceil(L / ds)), 0.0, L, Utility::string_to_enum<ElemType>(bdry_elem_type));
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
                                          0.1 * L,
                                          0.1 * L + L * reaction_fraction,
                                          Utility::string_to_enum<ElemType>(bdry_elem_type));
        for (MeshBase::node_iterator it = reaction_mesh.nodes_begin(); it != reaction_mesh.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) = y_up + std::tan(theta) * X(0);
        }
        reaction_mesh.set_spatial_dimension(NDIM);
        reaction_mesh.prepare_for_use();

        Mesh vol_vol_mesh(init.comm(), NDIM);
        MeshTools::Generation::build_cube(vol_vol_mesh,
                                          static_cast<int>(ceil(L / ds)),
                                          static_cast<int>(ceil(1.0 / dx)),
                                          0,
                                          0.0,
                                          L,
                                          y_low,
                                          y_up,
                                          Utility::string_to_enum<ElemType>(bdry_elem_type));
        BoundaryMesh vol_mesh(init.comm(), vol_vol_mesh.mesh_dimension() - 1);
        vol_vol_mesh.boundary_info->sync(vol_mesh);

        for (MeshBase::node_iterator it = vol_mesh.nodes_begin(); it != vol_mesh.nodes_end(); ++it)
        {
            Node* n = *it;
            libMesh::Point& X = *n;
            X(1) += std::tan(theta) * X(0);
        }
        const MeshBase::const_element_iterator end_el = vol_mesh.elements_end();
        for (MeshBase::const_element_iterator el = vol_mesh.elements_begin(); el != end_el; ++el)
        {
            Elem* const elem = *el;
            for (unsigned int side = 0; side < elem->n_sides(); ++side)
            {
                BoundaryInfo* boundary_info = vol_mesh.boundary_info.get();
                const libMesh::Point& p = elem->point(side);
                if (p(0) == 0.0 || p(0) == 8.0)
                {
                    boundary_info->add_side(elem, side, 0);
                }
                if (boundary_info->has_boundary_id(elem, side, 0) || boundary_info->has_boundary_id(elem, side, 1))
                {
                    s_elem_point_cache[std::make_pair(elem, side)] = elem->point(side);
                }
            }
        }
        vol_mesh.set_spatial_dimension(NDIM);
        vol_mesh.prepare_for_use();

        static const int LOWER_MESH_ID = 0;
        static const int UPPER_MESH_ID = 1;
        static const int REACTION_MESH_ID = 2;
        static const int VOL_MESH_ID = 3;
        vector<MeshBase*> meshes(4);
        meshes[LOWER_MESH_ID] = &lower_mesh_bdry;
        meshes[UPPER_MESH_ID] = &upper_mesh_bdry;
        meshes[REACTION_MESH_ID] = &reaction_mesh;
        meshes[VOL_MESH_ID] = &vol_mesh;

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<IBFESurfaceMethod> ib_method_ops =
            new IBFESurfaceMethod("IBFESurfaceMethod",
                                  app_initializer->getComponentDatabase("IBFESurfaceMethod"),
                                  meshes,
                                  app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"));
        Pointer<SemiLagrangianAdvIntegrator> adv_diff_integrator = new SemiLagrangianAdvIntegrator(
            "SemiLagrangianAdvIntegrator",
            app_initializer->getComponentDatabase("AdvDiffSemiImplicitHierarchyIntegrator"),
            false);

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

        ib_method_ops->initializeFEEquationSystems();
        // Create Eulerian boundary condition specification objects.
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, static_cast<RobinBcCoefStrategy<NDIM>*>(NULL));
        const bool periodic_domain = grid_geometry->getPeriodicShift().min() > 0;

        // Setup velocity
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("U");
        adv_diff_integrator->registerAdvectionVelocity(u_var);

        Pointer<CartGridFunction> u_fcn =
            new muParserCartGridFunction("UFcn", app_initializer->getComponentDatabase("UFcn"), grid_geometry);
        adv_diff_integrator->setAdvectionVelocityFunction(u_var, u_fcn);

        // Setup the level set function
        Pointer<CellVariable<NDIM, double>> ls_in_cell_var = new CellVariable<NDIM, double>("LS_In");
        adv_diff_integrator->registerLevelSetVariable(ls_in_cell_var);
        adv_diff_integrator->registerLevelSetVelocity(ls_in_cell_var, u_var);
        bool use_ls_fcn = input_db->getBool("USING_LS_FCN");
        Pointer<InsideLSFcn> ls_fcn =
            new InsideLSFcn("InsideLSFcn", app_initializer->getComponentDatabase("InsideLSFcn"));
        adv_diff_integrator->registerLevelSetFunction(ls_in_cell_var, ls_fcn);
        adv_diff_integrator->useLevelSetFunction(ls_in_cell_var, use_ls_fcn);
        LocateInterface interface_in(ls_in_cell_var, adv_diff_integrator, ls_fcn);
        Pointer<RelaxationLSMethod> ls_in_ops =
            new RelaxationLSMethod("RelaxationLSMethod", app_initializer->getComponentDatabase("RelaxationLSMethod"));
        ls_in_ops->registerInterfaceNeighborhoodLocatingFcn(&locateInterface, static_cast<void*>(&interface_in));
        std::vector<RobinBcCoefStrategy<NDIM>*> ls_bcs(1);
        if (!periodic_domain)
        {
            const std::string ls_bcs_name = "LSBcCoefs";
            ls_bcs[0] = new muParserRobinBcCoefs(
                ls_bcs_name, app_initializer->getComponentDatabase(ls_bcs_name), grid_geometry);
            ls_in_ops->registerPhysicalBoundaryCondition(ls_bcs[0]);
        }
        adv_diff_integrator->registerLevelSetResetFunction(ls_in_cell_var, ls_in_ops);
        Pointer<NodeVariable<NDIM, double>> ls_in_node_var =
            adv_diff_integrator->getLevelSetNodeVariable(ls_in_cell_var);

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
        adv_diff_integrator->restrictToLevelSet(Q_in_var, ls_in_cell_var);

        // Set up diffusion operators
        Pointer<LSCutCellLaplaceOperator> rhs_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInRHSOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        Pointer<LSCutCellLaplaceOperator> sol_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        // Create boundary operators
        Pointer<SurfaceBoundaryReactions> surface_bdry_reactions =
            new SurfaceBoundaryReactions("SurfaceBoundaryReactions",
                                         app_initializer->getComponentDatabase("SurfaceBoundaryReactions"),
                                         &reaction_mesh,
                                         ib_method_ops->getFEDataManager(REACTION_MESH_ID));
        rhs_in_oper->setBoundaryConditionOperator(surface_bdry_reactions);
        sol_in_oper->setBoundaryConditionOperator(surface_bdry_reactions);
        adv_diff_integrator->setHelmholtzRHSOperator(Q_in_var, rhs_in_oper);
        Pointer<PETScKrylovPoissonSolver> Q_in_helmholtz_solver = new PETScKrylovPoissonSolver(
            "PoissonSolver", app_initializer->getComponentDatabase("PoissonSolver"), "poisson_solve_");
        Q_in_helmholtz_solver->setOperator(sol_in_oper);
        adv_diff_integrator->setHelmholtzSolver(Q_in_var, Q_in_helmholtz_solver);

        // Create scratch index for SurfaceBoundaryReactions
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_in_idx =
            var_db->registerVariableAndContext(Q_in_var, var_db->getContext("Scratch"), IntVector<NDIM>(2));


        // Set up visualization plot file writer.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            adv_diff_integrator->registerVisItDataWriter(visit_data_writer);
        }
        libMesh::UniquePtr<ExodusII_IO> lower_exodus_io(uses_exodus ? new ExodusII_IO(*meshes[LOWER_MESH_ID]) : NULL);
        libMesh::UniquePtr<ExodusII_IO> upper_exodus_io(uses_exodus ? new ExodusII_IO(*meshes[UPPER_MESH_ID]) : NULL);
        libMesh::UniquePtr<ExodusII_IO> reaction_exodus_io(uses_exodus ? new ExodusII_IO(*meshes[REACTION_MESH_ID]) :
                                                                         NULL);
        libMesh::UniquePtr<ExodusII_IO> vol_mesh_io(uses_exodus ? new ExodusII_IO(*meshes[VOL_MESH_ID]) : NULL);

        // Register a drawing variable with the data writer
        const int Q_scr_idx =
            var_db->registerVariableAndContext(Q_in_var, var_db->getContext("SCRATCH"), IntVector<NDIM>(4));

        Pointer<CellVariable<NDIM, double>> dist_var = new CellVariable<NDIM, double>("distance");
        Pointer<CellVariable<NDIM, double>> n_var = new CellVariable<NDIM, double>("num_elements");
        const int dist_idx =
            var_db->registerVariableAndContext(dist_var, var_db->getContext("SCRATCH"), IntVector<NDIM>(1));
        const int n_idx = var_db->registerVariableAndContext(n_var, var_db->getContext("Scratch"), IntVector<NDIM>(1));
        visit_data_writer->registerPlotQuantity("num_elements", "SCALAR", n_idx);
        visit_data_writer->registerPlotQuantity("distance", "SCALAR", dist_idx);

        FESurfaceDistanceEvaluator surface_distance_eval(
            "FESurfaceDistanceEvaulator", patch_hierarchy, vol_vol_mesh, vol_mesh, 1, true);

        ib_method_ops->initializeFEData();
        // Initialize hierarchy configuration and data on all patches.
        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);
        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_in_var, adv_diff_integrator->getCurrentContext());

        surface_bdry_reactions->fillInitialCondition();

        for (unsigned int part = 0; part < meshes.size(); ++part)
        {
            ib_method_ops->getFEDataManager(part)->setPatchHierarchy(patch_hierarchy);
            ib_method_ops->getFEDataManager(part)->reinitElementMappings();
        }

        // Close the restart manager.
        RestartManager::getManager()->closeRestartFile();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        double dt = adv_diff_integrator->getMaximumTimeStepSize();

        // Write out initial visualization data.
        EquationSystems* lower_equation_systems = ib_method_ops->getFEDataManager(0)->getEquationSystems();
        EquationSystems* upper_equation_systems = ib_method_ops->getFEDataManager(1)->getEquationSystems();
        EquationSystems* reaction_eq_sys = ib_method_ops->getFEDataManager(2)->getEquationSystems();
        EquationSystems* vol_eq_sys = ib_method_ops->getFEDataManager(3)->getEquationSystems();
        FEDataManager* vol_fe_manager = ib_method_ops->getFEDataManager(3);
        int iteration_num = adv_diff_integrator->getIntegratorStep();
        double loop_time = adv_diff_integrator->getIntegratorTime();
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            adv_diff_integrator->setupPlotData();

            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            surface_bdry_reactions->fillExactSolution(loop_time);
            if (uses_exodus)
            {
                lower_exodus_io->write_timestep(
                    lower_exodus_filename, *lower_equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                upper_exodus_io->write_timestep(
                    upper_exodus_filename, *upper_equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                reaction_exodus_io->write_timestep(
                    reaction_exodus_filename, *reaction_eq_sys, iteration_num / viz_dump_interval + 1, loop_time);
            }
            const int vol_idx = var_db->mapVariableAndContextToIndex(
                adv_diff_integrator->getVolumeVariable(ls_in_cell_var), adv_diff_integrator->getCurrentContext());
            checkConservation(ib_method_ops->getFEDataManager(REACTION_MESH_ID), SurfaceBoundaryReactions::s_surface_sys_name, Q_idx, vol_idx, patch_hierarchy, loop_time);
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
            synchFluidConcentration(Q_in_idx, Q_in_var, adv_diff_integrator, patch_hierarchy);
            surface_bdry_reactions->beginTimestepping(loop_time, loop_time + dt);
            const int ls_idx = var_db->mapVariableAndContextToIndex(
                adv_diff_integrator->getLevelSetNodeVariable(ls_in_cell_var), adv_diff_integrator->getCurrentContext());
            const int vol_idx = var_db->mapVariableAndContextToIndex(
                adv_diff_integrator->getVolumeVariable(ls_in_cell_var), adv_diff_integrator->getCurrentContext());
            surface_bdry_reactions->setLSData(adv_diff_integrator->getLevelSetNodeVariable(ls_in_cell_var),
                                              ls_idx,
                                              adv_diff_integrator->getVolumeVariable(ls_in_cell_var),
                                              vol_idx,
                                              nullptr,
                                              -1);
            surface_bdry_reactions->updateSurfaceConcentration(Q_in_idx, loop_time, loop_time + dt, patch_hierarchy);
            surface_bdry_reactions->endTimestepping(loop_time, loop_time + dt);
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
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                surface_bdry_reactions->fillExactSolution(loop_time);
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
                checkConservation(ib_method_ops->getFEDataManager(REACTION_MESH_ID), SurfaceBoundaryReactions::s_surface_sys_name, Q_idx, vol_idx, patch_hierarchy, loop_time);
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
            if (dump_postproc_data && (iteration_num % dump_postproc_interval == 0 || last_step))
            {
                auto var_db = VariableDatabase<NDIM>::getDatabase();
                const int Q_in_idx =
                    var_db->mapVariableAndContextToIndex(Q_in_var, adv_diff_integrator->getCurrentContext());
                const int ls_in_n_idx =
                    var_db->mapVariableAndContextToIndex(ls_in_node_var, adv_diff_integrator->getCurrentContext());
                const int vol_in_idx = var_db->mapVariableAndContextToIndex(
                    adv_diff_integrator->getVolumeVariable(ls_in_cell_var), adv_diff_integrator->getCurrentContext());
                const int area_in_idx = var_db->mapVariableAndContextToIndex(
                    adv_diff_integrator->getAreaVariable(ls_in_cell_var), adv_diff_integrator->getCurrentContext());
                postprocess_data(patch_hierarchy,
                                 adv_diff_integrator,
                                 Q_in_var,
                                 iteration_num,
                                 loop_time,
                                 postproc_data_dump_dirname);
            }
        }

        if (!periodic_domain) delete Q_in_bcs[0];
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
postprocess_data(Pointer<PatchHierarchy<NDIM>> hierarchy,
                 Pointer<SemiLagrangianAdvIntegrator> integrator,
                 Pointer<CellVariable<NDIM, double>> Q_in_var,
                 const int iteration_num,
                 const double loop_time,
                 const std::string& dirname)
{
    std::string file_name = dirname + "/hier_data.";
    char temp_buf[128];
    sprintf(temp_buf, "%05d.samrai.%05d", iteration_num, SAMRAI_MPI::getRank());
    file_name += temp_buf;
    Pointer<HDFDatabase> hier_db = new HDFDatabase("hier_db");
    hier_db->create(file_name);
    ComponentSelector hier_data;
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    hier_data.setFlag(var_db->mapVariableAndContextToIndex(Q_in_var, integrator->getCurrentContext()));
    hierarchy->putToDatabase(hier_db->putDatabase("PatchHierarchy"), hier_data);
    hier_db->putDouble("loop_time", loop_time);
    hier_db->putInteger("iteration_num", iteration_num);
    hier_db->close();
}

void
checkConservation(FEDataManager* fe_data_manager, const std::string& sys_name, const int Q_idx, const int vol_idx, Pointer<PatchHierarchy<NDIM>> hierarchy, const double time)
{
    double surface_amount = 0.0, fluid_amount = 0.0;
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
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
        for (int qp = 0; qp < JxW.size(); ++qp)
        {
            double Q = 0.0;
            for (int n = 0; n < 2; ++n)
                Q += Q_nodes[n] * phi[n][qp];
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
            fluid_amount += (*q_data)(idx) * (*vol_data)(idx) * dx[0] * dx[1];
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

void
updateVolumeMesh(Mesh& vol_mesh, EquationSystems* vol_eq_sys, FEDataManager* vol_fe_manager)
{
    System& X_sys = vol_eq_sys->get_system(vol_fe_manager->COORDINATES_SYSTEM_NAME);
    System& X_mapping_sys = vol_eq_sys->get_system("IB coordinate mapping system");
    DofMap& X_dof_map = X_sys.get_dof_map();
    DofMap& X_mapping_map = X_mapping_sys.get_dof_map();
    NumericVector<double>* X_vec = X_sys.solution.get();
    NumericVector<double>* X_map_vec = X_mapping_sys.solution.get();
    // Loop through lower and upper mesh
    const MeshBase::const_element_iterator end_el = vol_mesh.elements_end();
    for (MeshBase::const_element_iterator el = vol_mesh.elements_begin(); el != end_el; ++el)
    {
        Elem* const elem = *el;
        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            BoundaryInfo* boundary_info = vol_mesh.boundary_info.get();
            std::vector<dof_id_type> X_dofs;
            if (boundary_info->has_boundary_id(elem, side, 0))
            {
                Node* node = elem->node_ptr(side);
                for (int d = 0; d < NDIM; ++d)
                {
                    IBTK::get_nodal_dof_indices(X_dof_map, node, d, X_dofs);
                    X_vec->set(X_dofs[0], s_elem_point_cache[std::make_pair(elem, side)](d));
                    IBTK::get_nodal_dof_indices(X_mapping_map, node, d, X_dofs);
                    X_map_vec->set(X_dofs[0], 0.0);
                }
            }
            else
            {
                Node* node = elem->node_ptr(side);
                libMesh::Point& p = elem->point(side);
                for (int d = 0; d < NDIM; ++d)
                {
                    IBTK::get_nodal_dof_indices(X_dof_map, node, d, X_dofs);
                    double X_val;
                    X_vec->get(X_dofs, &X_val);
                    p(d) = X_val;
                    IBTK::get_nodal_dof_indices(X_mapping_map, node, d, X_dofs);
                    X_map_vec->set(X_dofs[0], 0.0);
                }
            }
        }
    }
    X_vec->close();
    X_map_vec->close();
    vol_mesh.prepare_for_use();
}