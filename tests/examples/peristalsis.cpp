#include <ADS/CutCellMeshMapping.h>
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
#include "peristalsis/IBBoundaryMeshMapping.cpp"
#include "peristalsis/QFcn.cpp"

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
        Pointer<SLAdvIntegrator> adv_diff_integrator =
            new SLAdvIntegrator("AdvDiffIntegrator",
                                app_initializer->getComponentDatabase("AdvDiffIntegrator"),
                                true /*register_for_restart*/);
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
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        adv_diff_integrator->registerTransportedQuantity(Q_var, true);
        adv_diff_integrator->setAdvectionVelocity(Q_var, ins_integrator->getAdvectionVelocityVariable());
        Pointer<QFcn> Q_init_fcn = new QFcn("LSInit", input_db->getDatabase("LSInit"));
        adv_diff_integrator->setInitialConditions(Q_var, Q_init_fcn);

        ib_initializer->setStructureNamesOnLevel(finest_ln, struct_list);
        ib_initializer->registerInitStructureFunction(generate_structure);
        ib_initializer->registerInitTargetPtFunction(generate_tethers);
        ib_ops->registerLInitStrategy(ib_initializer);
        Pointer<IBStandardForceGen> ib_spring_forces = new IBStandardForceGen();
        ib_ops->registerIBLagrangianForceFunction(ib_spring_forces);

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
            lower_mesh.reserve_nodes(num_vertices + 1);
            lower_mesh.reserve_elem(num_vertices);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                lower_mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0), node_num);
            }

            // Add extra node on the LEFT boundary
            VectorNd x = lower_channel(0.0, 0.0);
            lower_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

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
        }
        {
            // Upper mesh
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(0, finest_ln, num_vertices, vertex_posn, nullptr);
            upper_mesh.reserve_nodes(num_vertices + 1);
            upper_mesh.reserve_elem(num_vertices);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                upper_mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0), node_num);
            }

            // Add extra node on the LEFT boundary
            VectorNd x = upper_channel(0.0, 0.0);
            upper_mesh.add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

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
        adv_diff_integrator->registerGeneralBoundaryMeshMapping(mesh_mapping);

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_idx = var_db->registerVariableAndContext(ls_var, var_db->getContext("LS"), IntVector<NDIM>(2));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, var_db->getContext("VOL"), IntVector<NDIM>(2));

        // Group all scratch indices together
        ComponentSelector ls_idxs;
        ls_idxs.setFlag(ls_idx);
        ls_idxs.setFlag(vol_idx);

        // setup the LSAdvDiffIntegrator if necessary.
        adv_diff_integrator->registerLevelSetVariable(ls_var);
        adv_diff_integrator->restrictToLevelSet(Q_var, ls_var);

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellMeshMapping("CutCellMapping", app_initializer->getComponentDatabase("CutCellMapping"));
        Pointer<LSFromMesh> vol_fcn =
            new LSFromMesh("LSFromMesh", patch_hierarchy, mesh_mapping->getSystemManagers(), cut_cell_mapping, true);
        vol_fcn->registerBdryFcn(ls_bdry_fcn);
        vol_fcn->registerNormalReverseDomainId(0, 1);
        adv_diff_integrator->registerLevelSetVolFunction(ls_var, vol_fcn);

        // Setup systems.
        const std::string Q_exact_str = "Q_EXACT";
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            FESystemManager& fe_sys_manager = mesh_mapping->getSystemManager(part);
            EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
            auto& Q_exact_sys = eq_sys->add_system<ExplicitSystem>(Q_exact_str);
            Q_exact_sys.add_variable(Q_exact_str);
            Q_exact_sys.assemble_before_solve = false;
            Q_exact_sys.assemble();
        }

        mesh_mapping->initializeFEData();

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        std::string interp_type = input_db->getString("INTERP_TYPE");
        pout << "Using interp type " << interp_type << "\n";
        if (interp_type.compare("RBF") == 0)
        {
            auto adv_op_reconstruct =
                std::make_shared<RBFStructureReconstructions>("RBFReconstruct", input_db->getDatabase("AdvOps"));
            adv_op_reconstruct->setCutCellMapping(get_system_managers(adv_diff_integrator->getFEHierarchyMappings()),
                                                  cut_cell_mapping);
            adv_op_reconstruct->setQSystemName(Q_exact_str);
            adv_diff_integrator->registerAdvectionReconstruction(Q_var, adv_op_reconstruct);
        }
        else if (interp_type.compare("LAGRANGE") == 0)
        {
            auto adv_op_reconstruct =
                std::make_shared<LagrangeStructureReconstructions>("RBFReconstruct", input_db->getDatabase("AdvOps"));
            adv_op_reconstruct->setCutCellMapping(get_system_managers(adv_diff_integrator->getFEHierarchyMappings()),
                                                  cut_cell_mapping);
            adv_op_reconstruct->setInsideQSystemName(Q_exact_str);
            adv_op_reconstruct->setReconstructionOutside(false);
            adv_diff_integrator->registerAdvectionReconstruction(Q_var, adv_op_reconstruct);
        }

        auto limit_fcn = [](double /*current_time*/,
                            double /*new_time*/,
                            bool /*skip_synchronize_new_state_data*/,
                            int /*num_cycles*/,
                            void* ctx) -> void
        {
            pout << "  Limiting Q\n";
            auto integrator_variable_pair = static_cast<
                std::pair<Pointer<AdvDiffSemiImplicitHierarchyIntegrator>, Pointer<CellVariable<NDIM, double>>>*>(ctx);
            Pointer<AdvDiffSemiImplicitHierarchyIntegrator>& integrator = integrator_variable_pair->first;
            Pointer<CellVariable<NDIM, double>> Q_var = integrator_variable_pair->second;
            Pointer<PatchHierarchy<NDIM>> hierarchy = integrator->getPatchHierarchy();
            for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
                for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_var, integrator->getNewContext());
                    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx = ci();
                        (*Q_data)(idx) = std::max((*Q_data)(idx), 0.0);
                    }
                }
            }
        };
        auto integrator_variable_pair = std::make_pair(adv_diff_integrator, Q_var);
        if (input_db->getBool("LIMIT_FCN"))
            adv_diff_integrator->registerPostprocessIntegrateHierarchyCallback(
                limit_fcn, static_cast<void*>(&integrator_variable_pair));

        // Set the exact solution
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            FESystemManager& fe_sys_manager = mesh_mapping->getSystemManager(part);
            EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
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
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();

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

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";
        }

        // Check that values are non-negative
        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_var, adv_diff_integrator->getCurrentContext());
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(Q_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if ((*Q_data)(idx) < 0.0) pout << "Found negative value on idx " << idx << "\n";
                }
            }
        }

    } // cleanup dynamically allocated objects prior to shutdown
} // main
