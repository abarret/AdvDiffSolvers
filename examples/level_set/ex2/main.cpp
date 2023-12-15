#include <ADS/CutCellVolumeMeshMapping.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/LSFromMesh.h>
#include <ADS/LagrangeStructureReconstructions.h>
#include <ADS/RBFStructureReconstructions.h>
#include <ADS/SLAdvIntegrator.h>
#include <ADS/app_namespaces.h>

#include <ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h>
#include <ibamr/CFINSForcing.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBLagrangianForceStrategySet.h>
#include <ibamr/IBMethod.h>
#include <ibamr/IBRedundantInitializer.h>
#include <ibamr/IBStandardForceGen.h>
#include <ibamr/IBStandardInitializer.h>
#include <ibamr/IBTargetPointForceSpec.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/FEDataManager.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/IndexUtilities.h>
#include <ibtk/LData.h>
#include <ibtk/LDataManager.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <libmesh/edge_edge2.h>
#include <libmesh/exodusII.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh_base.h>
#include <libmesh/mesh_tools.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <array>
#include <memory>

// Local includes
#include "IBBoundaryMeshMapping.h"
#include "LSCUIConvectiveOperator.h"
#include "QFcn.h"

int finest_ln;
std::array<int, NDIM> N;
double alpha = 0.0;
double g = 0.0;
double upper_perim = 0.0;
double lower_perim = 0.0;
double MFAC = 0.0;
double dx = 0.0;
double L = 0.0;
double K = 0.0;
std::vector<int> num_nodes_per_struct;
std::vector<double> ds;
double time_for_ls = 0.0;

VectorNd
upper_channel(const double s, const double t)
{
    VectorNd x;
    x(0) = s;
    x(1) = alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (s - t)));
    return x;
}

VectorNd
lower_channel(const double s, const double t)
{
    VectorNd x;
    x(0) = s;
    x(1) = -alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (s - t)));
    return x;
}
void
generate_structure(const unsigned int& strct_num,
                   const int& ln,
                   int& num_vertices,
                   std::vector<IBTK::Point>& vertex_posn,
                   void* /*ctx*/)
{
    if (ln != finest_ln)
    {
        num_vertices = 0;
        vertex_posn.resize(num_vertices);
    }
    if (strct_num == 0)
    {
        // Generating upper level of channel.
        // Determine lag grid spacing.
        ds[strct_num] = MFAC * upper_perim * dx;
        num_vertices = std::ceil(L / ds[strct_num]);
        vertex_posn.resize(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            VectorNd x = upper_channel(ds[strct_num] * (static_cast<double>(i) + 0.15), 0.0);
            vertex_posn[i] = x;
        }
    }
    else if (strct_num == 1)
    {
        // Generating lower level of channel.
        ds[strct_num] = MFAC * lower_perim * dx;
        num_vertices = std::ceil(L / ds[strct_num]);
        vertex_posn.resize(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            VectorNd x = lower_channel(ds[strct_num] * (static_cast<double>(i) + 0.15), 0.0);
            vertex_posn[i] = x;
        }
    }
    num_nodes_per_struct[strct_num] = num_vertices;
    return;
}

void
generate_tethers(const unsigned int& strct_num,
                 const int& ln,
                 std::multimap<int, IBRedundantInitializer::TargetSpec>& tg_pt_spec,
                 void* /*ctx*/)
{
    if (ln != finest_ln) return;
    for (int k = 0; k < num_nodes_per_struct[strct_num]; ++k)
    {
        IBRedundantInitializer::TargetSpec e;
        e.stiffness = K / ds[strct_num];
        e.damping = 0.0;
        tg_pt_spec.insert(std::make_pair(k, e));
    }
}

void
move_tethers(LDataManager* data_manager, const double time)
{
    const std::pair<int, int>& upper_lag_idxs = data_manager->getLagrangianStructureIndexRange(0, finest_ln);

    // Update both local and ghost nodes.
    Pointer<LMesh> mesh = data_manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();
    const std::vector<LNode*>& ghost_nodes = mesh->getGhostNodes();
    nodes.insert(nodes.end(), ghost_nodes.begin(), ghost_nodes.end());

    // Also need reference information.
    Pointer<LData> X_ref_data = data_manager->getLData(data_manager->INIT_POSN_DATA_NAME, finest_ln);
    double* X_ref_vals = X_ref_data->getVecArray()->data();

    for (const auto node : nodes)
    {
        const auto& force_spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!force_spec) continue;
        const int lag_idx = node->getLagrangianIndex();
        const int petsc_idx = node->getLocalPETScIndex();
        IBTK::Point& X_target = force_spec->getTargetPointPosition();
        // Detect with side of channel we are on.
        if (upper_lag_idxs.first <= lag_idx && lag_idx < upper_lag_idxs.second)
            X_target = upper_channel(X_ref_vals[petsc_idx * NDIM], time);
        else
            X_target = lower_channel(X_ref_vals[petsc_idx * NDIM], time);
    }

    X_ref_data->restoreArrays();
}

double
Q_fcn(const VectorNd& X, double time)
{
    return (X[1] <= (alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (X[0])))) && X[1] >= 0.1 && X[0] > 0.1 &&
            X[0] <= 0.4) ?
               0.5 :
               0.05;
}

void
ls_bdry_fcn(const VectorNd& X, double& ls_val)
{
    double yup = upper_channel(X[0], time_for_ls)[1];
    double ylow = lower_channel(X[0], time_for_ls)[1];
    if (X[1] <= yup && X[1] >= ylow)
        ls_val = -1.0;
    else
        ls_val = 1.0;
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
    IBTKInit init(argc, argv);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<INSStaggeredHierarchyIntegrator> ins_integrator = new INSStaggeredHierarchyIntegrator(
            "INSIntegrator", app_initializer->getComponentDatabase("INSIntegrator"), false);
        Pointer<IBMethod> ib_ops = new IBMethod("IBMethod", app_initializer->getComponentDatabase("IBMethod"));
        Pointer<IBExplicitHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_ops,
                                              ins_integrator);
        Pointer<SLAdvIntegrator> sl_integrator =
            new SLAdvIntegrator("AdvDiffIntegrator",
                                app_initializer->getComponentDatabase("AdvDiffIntegrator"),
                                true /*register_for_restart*/);
        Pointer<AdvDiffSemiImplicitHierarchyIntegrator> adv_diff_integrator =
            new AdvDiffSemiImplicitHierarchyIntegrator(
                "AdvDiffSemiImplicitIntegrator",
                app_initializer->getComponentDatabase("AdvDiffSemiImplicitIntegrator"),
                true /*register_for_restart*/);
        ins_integrator->registerAdvDiffHierarchyIntegrator(sl_integrator);
        ins_integrator->registerAdvDiffHierarchyIntegrator(adv_diff_integrator);
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
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

        // Configure the IB solver.
        Pointer<IBRedundantInitializer> ib_initializer = new IBRedundantInitializer(
            "IBNRedundantInitializer", app_initializer->getComponentDatabase("IBRedundantInitializer"));
        std::vector<std::string> struct_list = { "upper", "lower" };
        ds.resize(2);
        num_nodes_per_struct.resize(2);
        N[0] = N[1] = input_db->getInteger("NFINEST");
        finest_ln = input_db->getInteger("MAX_LEVELS") - 1;
        alpha = input_db->getDouble("ALPHA");
        K = input_db->getDouble("K_TETHER");
        g = input_db->getDouble("GAMMA");
        L = input_db->getDouble("L");
        MFAC = input_db->getDouble("MFAC");
        dx = L / N[0];
        upper_perim = 1.02423522856; // alpha * (L*M_PI + g * std::sin(L*M_PI)*std::sin(L*M_PI)) / (2.0 * M_PI * M_PI);
        lower_perim = upper_perim;

        // Set up advected quantity
        Pointer<CellVariable<NDIM, double>> Q_sl_var = new CellVariable<NDIM, double>("Q_sl");
        sl_integrator->registerTransportedQuantity(Q_sl_var, true);
        sl_integrator->setAdvectionVelocity(Q_sl_var, ins_integrator->getAdvectionVelocityVariable());
        Pointer<QFcn> Q_init_fcn = new QFcn("LSInit", input_db->getDatabase("LSInit"));
        sl_integrator->setInitialConditions(Q_sl_var, Q_init_fcn);

        Pointer<CellVariable<NDIM, double>> Q_fv_var = new CellVariable<NDIM, double>("Q_fv");
        adv_diff_integrator->registerTransportedQuantity(Q_fv_var, true);
        adv_diff_integrator->setAdvectionVelocity(Q_fv_var, ins_integrator->getAdvectionVelocityVariable());
        adv_diff_integrator->setInitialConditions(Q_fv_var, Q_init_fcn);

        ib_initializer->setStructureNamesOnLevel(finest_ln, struct_list);
        ib_initializer->registerInitStructureFunction(generate_structure);
        ib_initializer->registerInitTargetPtFunction(generate_tethers);
        ib_ops->registerLInitStrategy(ib_initializer);
        Pointer<IBStandardForceGen> ib_spring_forces = new IBStandardForceGen();
        ib_ops->registerIBLagrangianForceFunction(ib_spring_forces);

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        Pointer<LSiloDataWriter> data_writer =
            new LSiloDataWriter("DataWriter", input_db->getDatabase("Main")->getString("viz_dump_dirname"), false);
        if (uses_visit)
        {
            ib_initializer->registerLSiloDataWriter(data_writer);
            ib_ops->registerLSiloDataWriter(data_writer);
            time_integrator->registerVisItDataWriter(visit_data_writer);
            sl_integrator->registerVisItDataWriter(visit_data_writer);
        }

        // Create Eulerian initial condition specification objects.  These
        // objects also are used to specify exact solution values for error
        // analysis.
        Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
            "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
        ins_integrator->registerVelocityInitialConditions(u_init);

        Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
            "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
        ins_integrator->registerPressureInitialConditions(p_init);

        // Generate finite element structure.
        libMesh::Mesh upper_mesh(init.getLibMeshInit().comm(), NDIM - 1),
            lower_mesh(init.getLibMeshInit().comm(), NDIM - 1);

        // Note we can not copy the mesh from the LDataManager because we need to setup mesh points before the
        // LDataManager is populated. Instead, we construct the mesh using the same functions used to populate the
        // LDataManager.
        {
            // Lower mesh
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(1, finest_ln, num_vertices, vertex_posn, nullptr);
            lower_mesh.reserve_nodes(num_vertices + 2);
            lower_mesh.reserve_elem(num_vertices + 1);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                lower_mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0), node_num);
            }

            // Add extra node on the LEFT boundary
            VectorNd x = lower_channel(0.0, 0.0);
            lower_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

            // Add extra node on the RIGHT boundary
            x = lower_channel(1.0, 0.0);
            lower_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices + 1);

            // Generate elements
            for (int i = 0; i < num_vertices - 1; ++i)
            {
                Elem* elem = lower_mesh.add_elem(new libMesh::Edge2());
                elem->set_node(0) = lower_mesh.node_ptr(i);
                elem->set_node(1) = lower_mesh.node_ptr(i + 1);
            }

            // Last element is from node num_vertices to node 0
            Elem* elem = lower_mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = lower_mesh.node_ptr(num_vertices);
            elem->set_node(1) = lower_mesh.node_ptr(0);
            // Last element is from node num_vertices to node 0
            elem = lower_mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = lower_mesh.node_ptr(num_vertices - 1);
            elem->set_node(1) = lower_mesh.node_ptr(num_vertices + 1);
        }
        {
            // Upper mesh
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(0, finest_ln, num_vertices, vertex_posn, nullptr);
            upper_mesh.reserve_nodes(num_vertices + 2);
            upper_mesh.reserve_elem(num_vertices + 1);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                upper_mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0), node_num);
            }

            // Add extra node on the LEFT boundary
            VectorNd x = upper_channel(0.0, 0.0);
            upper_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

            // Add extra node on the RIGHT boundary
            x = upper_channel(1.0, 0.0);
            upper_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices + 1);

            // Generate elements
            for (int i = 0; i < num_vertices - 1; ++i)
            {
                Elem* elem = upper_mesh.add_elem(new libMesh::Edge2());
                elem->set_node(0) = upper_mesh.node_ptr(i);
                elem->set_node(1) = upper_mesh.node_ptr(i + 1);
            }

            // Last element is from node num_vertices to node 0
            Elem* elem = upper_mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = upper_mesh.node_ptr(num_vertices);
            elem->set_node(1) = upper_mesh.node_ptr(0);
            // Last element is from node num_vertices to node 0
            elem = upper_mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = upper_mesh.node_ptr(num_vertices - 1);
            elem->set_node(1) = upper_mesh.node_ptr(num_vertices + 1);
        }

        lower_mesh.prepare_for_use();
        upper_mesh.prepare_for_use();

        // Generate mesh mappings and level set information.
        std::vector<MeshBase*> meshes = { &lower_mesh, &upper_mesh };
        std::vector<int> part_nums = { 1, 0 };
        LDataManager* ib_manager = ib_ops->getLDataManager();
        auto mesh_mapping = std::make_shared<IBBoundaryMeshMapping>(
            "BoundaryMeshMapping", input_db->getDatabase("MeshMapping"), meshes, ib_manager, finest_ln, part_nums);
        mesh_mapping->initializeEquationSystems();

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellVolumeMeshMapping("CutCellMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         mesh_mapping->getMeshPartitioners());
        Pointer<LSFromMesh> vol_fcn = new LSFromMesh("LSFromMesh", patch_hierarchy, cut_cell_mapping, true);
        vol_fcn->registerBdryFcn(ls_bdry_fcn);
        vol_fcn->registerNormalReverseDomainId(0, 1);

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");

        // setup the LSAdvDiffIntegrator if necessary.
        sl_integrator->registerLevelSetVariable(ls_var);
        sl_integrator->registerLevelSetVolFunction(ls_var, vol_fcn);
        sl_integrator->restrictToLevelSet(Q_sl_var, ls_var);
        sl_integrator->registerGeneralBoundaryMeshMapping(mesh_mapping);

        // Setup systems.
        const std::string Q_exact_str = "Q_EXACT";
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = mesh_mapping->getMeshPartitioner(part);
            EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();
            auto& Q_exact_sys = eq_sys->add_system<ExplicitSystem>(Q_exact_str);
            Q_exact_sys.add_variable(Q_exact_str);
            Q_exact_sys.assemble_before_solve = false;
            Q_exact_sys.assemble();
        }

        std::string interp_type = input_db->getString("INTERP_TYPE");
        pout << "Using interp type " << interp_type << "\n";
        if (interp_type.compare("RBF") == 0)
        {
            auto adv_op_reconstruct =
                std::make_shared<RBFStructureReconstructions>("RBFReconstruct", input_db->getDatabase("AdvOps"));
            adv_op_reconstruct->setCutCellMapping(cut_cell_mapping);
            adv_op_reconstruct->setQSystemName(Q_exact_str);
            sl_integrator->registerAdvectionReconstruction(Q_sl_var, adv_op_reconstruct);
        }
        else if (interp_type.compare("LAGRANGE") == 0)
        {
            auto adv_op_reconstruct =
                std::make_shared<LagrangeStructureReconstructions>("RBFReconstruct", input_db->getDatabase("AdvOps"));
            adv_op_reconstruct->setCutCellMapping(cut_cell_mapping);
            adv_op_reconstruct->setInsideQSystemName(Q_exact_str);
            adv_op_reconstruct->setReconstructionOutside(false);
            sl_integrator->registerAdvectionReconstruction(Q_sl_var, adv_op_reconstruct);
        }

        // setup the AdvDiffIntegrator
        Pointer<LSCUIConvectiveOperator> convec_op =
            new LSCUIConvectiveOperator("LSCUI",
                                        Q_fv_var,
                                        nullptr,
                                        ConvectiveDifferencingType::ADVECTIVE,
                                        adv_diff_integrator->getPhysicalBcCoefs(Q_fv_var));
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        adv_diff_integrator->setConvectiveOperator(Q_fv_var, convec_op);

        mesh_mapping->initializeFEData();

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, sl_integrator->getCurrentContext());
        convec_op->setLSData(ls_var, ls_idx, patch_hierarchy);

        // Set the exact solution
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = mesh_mapping->getMeshPartitioner(part);
            EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();
            auto& Q_exact_sys = eq_sys->get_system<ExplicitSystem>(Q_exact_str);
            NumericVector<double>* Q_vec = Q_exact_sys.solution.get();
            const DofMap& dof_map = Q_exact_sys.get_dof_map();

            const MeshBase& mesh = eq_sys->get_mesh();
            auto it = mesh.local_nodes_begin();
            auto it_end = mesh.local_nodes_end();
            for (; it != it_end; ++it)
            {
                const Node* const node = *it;
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = (*node)(d);
                std::vector<dof_id_type> dof;
                dof_map.dof_indices(node, dof);
                Q_vec->set(dof[0], Q_init_fcn->setVal(x));
            }

            Q_vec->close();
            Q_exact_sys.update();
        }

        // Create mesh visualization.
        auto lower_exodus_io = std::make_unique<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(0));
        auto upper_exodus_io = std::make_unique<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(1));

        // Deallocate initialization objects.
        ib_ops->freeLInitStrategy();
        ib_initializer.setNull();
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        std::string filename = "amounts";
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            time_integrator->setupPlotData();
            mesh_mapping->updateBoundaryLocation(loop_time);
            for (auto& mesh_partitioner : mesh_mapping->getMeshPartitioners())
            {
                mesh_partitioner->setPatchHierarchy(patch_hierarchy);
                mesh_partitioner->reinitElementMappings(1);
            }
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            data_writer->writePlotData(iteration_num, loop_time);
            lower_exodus_io->write_timestep("lower.ex2",
                                            *mesh_mapping->getMeshPartitioner(0)->getEquationSystems(),
                                            iteration_num / viz_dump_interval + 1,
                                            loop_time);
            upper_exodus_io->write_timestep("upper.ex2",
                                            *mesh_mapping->getMeshPartitioner(1)->getEquationSystems(),
                                            iteration_num / viz_dump_interval + 1,
                                            loop_time);
        }

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;
        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();
            move_tethers(ib_ops->getLDataManager(), loop_time);

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            mesh_mapping->updateBoundaryLocation(loop_time);
            for (auto& mesh_partitioner : mesh_mapping->getMeshPartitioners())
            {
                mesh_partitioner->setPatchHierarchy(patch_hierarchy);
                mesh_partitioner->reinitElementMappings(2);
            }

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && uses_visit && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                data_writer->writePlotData(iteration_num, loop_time);
                lower_exodus_io->write_timestep("lower.ex2",
                                                *mesh_mapping->getMeshPartitioner(0)->getEquationSystems(),
                                                iteration_num / viz_dump_interval + 1,
                                                loop_time);
                upper_exodus_io->write_timestep("upper.ex2",
                                                *mesh_mapping->getMeshPartitioner(1)->getEquationSystems(),
                                                iteration_num / viz_dump_interval + 1,
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
            }
        }

    } // cleanup dynamically allocated objects prior to shutdown
} // main

void
interpolate_to_centroid(const int dst_idx,
                        const int src_idx,
                        const int scr_idx,
                        const int ls_idx,
                        const int vol_idx,
                        Pointer<PatchHierarchy<NDIM>> hierarchy,
                        const double time)
{
    // Now interpolate from cell centers to cell centroids
    {
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comps(1);
        ghost_cell_comps[0] =
            ITC(scr_idx, src_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR", true, nullptr);
        HierarchyGhostCellInterpolation hier_ghost_cell;
        hier_ghost_cell.initializeOperatorState(ghost_cell_comps, hierarchy);
        hier_ghost_cell.fillData(time);
    }

    RBFReconstructCache reconstruct_cache(8);
    reconstruct_cache.setLSData(ls_idx, vol_idx);
    reconstruct_cache.setUseCentroids(false);
    reconstruct_cache.setPatchHierarchy(hierarchy);
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();

            Pointer<CellData<NDIM, double>> Q_new_data = patch->getPatchData(dst_idx);
            Pointer<CellData<NDIM, double>> Q_scr_data = patch->getPatchData(scr_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*vol_data)(idx) < 1.0 && (*vol_data)(idx) > 0.0)
                {
                    // Reconstruct from cell center to cell centroid
                    VectorNd x_loc = find_cell_centroid(idx, *ls_data);
                    for (int d = 0; d < NDIM; ++d)
                        x_loc[d] = xlow[d] + dx[d] * (x_loc(d) - static_cast<double>(idx_low(d)));
                    (*Q_new_data)(idx) = reconstruct_cache.reconstructOnIndex(x_loc, idx, *Q_scr_data, patch);
                }
                else
                {
                    (*Q_new_data)(idx) = (*Q_scr_data)(idx);
                }
            }
        }
    }
}
