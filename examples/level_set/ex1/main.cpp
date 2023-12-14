#include <ADS/CutCellVolumeMeshMapping.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/InternalBdryFill.h>
#include <ADS/LSFromMesh.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/ReinitializeLevelSet.h>
#include <ADS/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <libmesh/edge_edge2.h>
#include <libmesh/exodusII.h>
#include <libmesh/exodusII_io.h>
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

double alpha = 0.0;
double g = 0.0;
double upper_perim = 0.0;
double lower_perim = 0.0;
double MFAC = 0.0;
double dx = 0.0;
double L = 0.0;
double ls_time = 0.0;

VectorNd
upper_channel(const double s, const double t)
{
    VectorNd x;
    x(0) = s;
    x(1) = alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (s + t)));
    return x;
}

VectorNd
lower_channel(const double s, const double t)
{
    VectorNd x;
    x(0) = s;
    x(1) = -alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (s + t)));
    return x;
}
void
generate_channel(const int strct_num, int& num_vertices, std::vector<IBTK::Point>& vertex_posn)
{
    if (strct_num == 0)
    {
        // Generating upper level of channel.
        // Determine lag grid spacing.
        double ds = MFAC * upper_perim * dx;
        num_vertices = std::ceil(L / ds);
        vertex_posn.resize(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            VectorNd x = upper_channel(ds * (static_cast<double>(i) + 0.15), 0.0);
            vertex_posn[i] = x;
        }
    }
    else if (strct_num == 1)
    {
        // Generating lower level of channel.
        double ds = MFAC * lower_perim * dx;
        num_vertices = std::ceil(L / ds);
        vertex_posn.resize(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            VectorNd x = lower_channel(ds * (static_cast<double>(i) + 0.15), 0.0);
            vertex_posn[i] = x;
        }
    }
    return;
}

double
Q_fcn_peristalsis(double, const VectorNd& X, double time)
{
    VectorNd cent;
    cent(0) = 0.35;
    cent(1) = 0.2;
    double R = 0.25;
    double r = (X - cent).norm();
    return (X[1] <= (alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (X[0] + time)))) && r <= R) ?
               std::max(0.8 * std::cos(M_PI * r / (2.0 * R)) + 0.025, 0.025) :
               0.025;
}

void
ls_bdry_fcn(const VectorNd& X, double& ls_val)
{
    double yup = upper_channel(X[0], ls_time)[1];
    double ylow = lower_channel(X[0], ls_time)[1];
    if (X[1] <= yup && X[1] >= ylow)
        ls_val = -1.0;
    else
        ls_val = 1.0;
}

VectorNd cent;
double R = 0.0;

VectorNd
cylinder_pt(const double s)
{
    VectorNd x;
    x(0) = cent(0) + R * std::cos(2.0 * M_PI * s);
    x(1) = cent(1) + R * std::sin(2.0 * M_PI * s);
    return x;
}

void
generate_cylinder(int& num_vertices, std::vector<IBTK::Point>& vertex_posn)
{
    const double circum = 2.0 * M_PI * R;
    double ds = MFAC * circum * dx;
    num_vertices = std::ceil(L / ds);
    vertex_posn.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        VectorNd x = cylinder_pt(ds * (static_cast<double>(i) + 0.5));
        vertex_posn[i] = x;
    }
}

double
Q_fcn_cylinder(double, VectorNd X, double time)
{
    // Shift X to center.
    X -= cent;
    // Convert to polar coordinates
    double r = X.norm();
    double th = std::atan2(X[1], X[0]);
    static double eps = 1.0e-12;
    if (r >= (R - eps) /*&& th >= -eps && th <= (M_PI-eps)*/ && r <= (2.0 * R - eps))
        return std::max(0.8 * std::cos(M_PI * r / (2.0 * R) - M_PI * 0.5) + 0.025, 0.025);
    else
        return 0.025;
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

        bool use_channel = input_db->getBool("USE_CHANNEL");

        if (use_channel)
        {
            alpha = input_db->getDouble("ALPHA");
            g = input_db->getDouble("GAMMA");
            upper_perim = 1.02423522856; // Length of curve, assumes L = 1
            lower_perim = upper_perim;
        }
        else
        {
            R = input_db->getDouble("R");
            input_db->getDoubleArray("CENT", cent.data(), NDIM);
        }

        MFAC = input_db->getDouble("MFAC");
        L = input_db->getDouble("L");
        dx = input_db->getDouble("DXFINEST");

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();

        // Roundabout way to generate meshes that satisfy poor libmesh and IBAMR design...
        std::vector<libMesh::Mesh*> actual_meshes;
        if (use_channel)
        {
            actual_meshes.push_back(new libMesh::Mesh(init.getLibMeshInit().comm(), NDIM - 1));
            actual_meshes.push_back(new libMesh::Mesh(init.getLibMeshInit().comm(), NDIM - 1));

            {
                // Lower mesh
                int num_vertices = 0;
                std::vector<IBTK::Point> vertex_posn;
                generate_channel(1, num_vertices, vertex_posn);
                actual_meshes[0]->reserve_nodes(num_vertices + 2);
                actual_meshes[0]->reserve_elem(num_vertices + 1);
                for (int node_num = 0; node_num < num_vertices; ++node_num)
                {
                    actual_meshes[0]->add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0),
                                                node_num);
                }

                // Add extra node on the LEFT boundary
                VectorNd x = lower_channel(0.0, 0.0);
                actual_meshes[0]->add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

                // Add extra node on the RIGHT boundary
                x = lower_channel(1.0, 0.0);
                actual_meshes[0]->add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices + 1);

                // Generate elements
                for (int i = 0; i < num_vertices - 1; ++i)
                {
                    Elem* elem = actual_meshes[0]->add_elem(new libMesh::Edge2());
                    elem->set_node(0) = actual_meshes[0]->node_ptr(i);
                    elem->set_node(1) = actual_meshes[0]->node_ptr(i + 1);
                }

                // Last element is from node num_vertices to node 0
                Elem* elem = actual_meshes[0]->add_elem(new libMesh::Edge2());
                elem->set_node(0) = actual_meshes[0]->node_ptr(num_vertices);
                elem->set_node(1) = actual_meshes[0]->node_ptr(0);
                // Last element is from node num_vertices to node 0
                elem = actual_meshes[0]->add_elem(new libMesh::Edge2());
                elem->set_node(0) = actual_meshes[0]->node_ptr(num_vertices - 1);
                elem->set_node(1) = actual_meshes[0]->node_ptr(num_vertices + 1);
            }
            {
                // Upper mesh
                int num_vertices = 0;
                std::vector<IBTK::Point> vertex_posn;
                generate_channel(0, num_vertices, vertex_posn);
                actual_meshes[1]->reserve_nodes(num_vertices + 2);
                actual_meshes[1]->reserve_elem(num_vertices + 1);
                for (int node_num = 0; node_num < num_vertices; ++node_num)
                {
                    actual_meshes[1]->add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0),
                                                node_num);
                }

                // Add extra node on the LEFT boundary
                VectorNd x = upper_channel(0.0, 0.0);
                actual_meshes[1]->add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices);

                // Add extra node on the RIGHT boundary
                x = upper_channel(1.0, 0.0);
                actual_meshes[1]->add_point(libMesh::Point(x[0], x[1], 0.0), num_vertices + 1);

                // Generate elements
                for (int i = 0; i < num_vertices - 1; ++i)
                {
                    Elem* elem = actual_meshes[1]->add_elem(new libMesh::Edge2());
                    elem->set_node(0) = actual_meshes[1]->node_ptr(i);
                    elem->set_node(1) = actual_meshes[1]->node_ptr(i + 1);
                }

                // Last element is from node num_vertices to node 0
                Elem* elem = actual_meshes[1]->add_elem(new libMesh::Edge2());
                elem->set_node(0) = actual_meshes[1]->node_ptr(num_vertices);
                elem->set_node(1) = actual_meshes[1]->node_ptr(0);
                // Last element is from node num_vertices to node 0
                elem = actual_meshes[1]->add_elem(new libMesh::Edge2());
                elem->set_node(0) = actual_meshes[1]->node_ptr(num_vertices - 1);
                elem->set_node(1) = actual_meshes[1]->node_ptr(num_vertices + 1);
            }

            actual_meshes[0]->prepare_for_use();
            actual_meshes[1]->prepare_for_use();
        }
        else
        {
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_cylinder(num_vertices, vertex_posn);
            actual_meshes.push_back(new libMesh::Mesh(init.getLibMeshInit().comm(), NDIM - 1));
            actual_meshes[0]->reserve_nodes(num_vertices);
            actual_meshes[0]->reserve_elem(num_vertices);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                actual_meshes[0]->add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0),
                                            node_num);
            }

            // Generate elements
            for (int i = 0; i < num_vertices - 1; ++i)
            {
                Elem* elem = actual_meshes[0]->add_elem(new libMesh::Edge2());
                elem->set_node(0) = actual_meshes[0]->node_ptr(i);
                elem->set_node(1) = actual_meshes[0]->node_ptr(i + 1);
            }

            // Last element is from node num_vertices to node 0
            Elem* elem = actual_meshes[0]->add_elem(new libMesh::Edge2());
            elem->set_node(0) = actual_meshes[0]->node_ptr(num_vertices - 1);
            elem->set_node(1) = actual_meshes[0]->node_ptr(0);

            actual_meshes[0]->prepare_for_use();
        }

        std::vector<libMesh::MeshBase*> meshes;
        for (auto& mesh : actual_meshes) meshes.push_back(mesh);

        // Generate mesh mappings and level set information.
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "BoundaryMeshMapping", input_db->getDatabase("MeshMapping"), meshes);
        mesh_mapping->initializeEquationSystems();

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellVolumeMeshMapping("CutCellMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         mesh_mapping->getMeshPartitioners());
        Pointer<LSFromMesh> vol_fcn = new LSFromMesh("LSFromMesh", patch_hierarchy, cut_cell_mapping, true);
        if (use_channel)
        {
            vol_fcn->registerBdryFcn(ls_bdry_fcn);
            vol_fcn->registerNormalReverseDomainId(0, 1);
        }
        else
        {
            vol_fcn->registerNormalReverseDomainId(0, 0);
        }

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<NodeVariable<NDIM, double>> phi_var = new NodeVariable<NDIM, double>("SDF");
        Pointer<NodeVariable<NDIM, int>> valid_var = new NodeVariable<NDIM, int>("VALID");
        Pointer<SideVariable<NDIM, double>> u_var = new SideVariable<NDIM, double>("Normal");
        Pointer<CellVariable<NDIM, double>> u_draw_var = new CellVariable<NDIM, double>("U", NDIM);
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CellVariable<NDIM, double>> Q_old_var = new CellVariable<NDIM, double>("Q_old");
        Pointer<CellVariable<NDIM, double>> ls_cent_var = new CellVariable<NDIM, double>("LS_CENT");
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("LS");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(2));
        const int ls_cent_idx = var_db->registerVariableAndContext(ls_cent_var, ctx);
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(2));
        int phi_idx = var_db->registerVariableAndContext(phi_var, ctx, IntVector<NDIM>(2));
        const int valid_idx = var_db->registerVariableAndContext(valid_var, ctx);
        const int u_idx = var_db->registerVariableAndContext(u_var, ctx, 1);
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw_var, ctx);
        const int Q_idx = var_db->registerVariableAndContext(Q_var, ctx, 1);
        const int Q_old_idx = var_db->registerVariableAndContext(Q_old_var, ctx, 1);
        visit_data_writer->registerPlotQuantity("LS", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("VOL", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("PHI", "SCALAR", phi_idx);
        visit_data_writer->registerPlotQuantity("VALID", "SCALAR", valid_idx);
        visit_data_writer->registerPlotQuantity("NORMAL", "VECTOR", u_draw_idx);
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_idx);
        visit_data_writer->registerPlotQuantity("LS_CENT", "SCALAR", ls_cent_idx);

        // Group all scratch indices together
        ComponentSelector ls_idxs;
        ls_idxs.setFlag(ls_idx);
        ls_idxs.setFlag(ls_cent_idx);
        ls_idxs.setFlag(vol_idx);
        ls_idxs.setFlag(phi_idx);
        ls_idxs.setFlag(valid_idx);
        ls_idxs.setFlag(u_idx);
        ls_idxs.setFlag(u_draw_idx);
        ls_idxs.setFlag(Q_idx);
        ls_idxs.setFlag(Q_old_idx);

        mesh_mapping->initializeFEData();

        // Initialize hierarchy configuration and data on all patches.
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

        // Allocate patch data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idxs, 0.0);
        }

        // Create mesh visualization.
        std::vector<std::shared_ptr<ExodusII_IO>> exodus_io;
        std::vector<std::string> exodus_io_strs;
        if (use_channel)
        {
            exodus_io_strs = { "lower.ex2", "upper.ex2" };
            exodus_io.assign({ std::make_shared<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(0)),
                               std::make_shared<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(1)) });
        }
        else
        {
            exodus_io_strs = { "cylinder.ex2" };
            exodus_io.assign({ std::make_shared<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(0)) });
        }

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        double t = 0.0;
        double T = 1.0;
        double dt = 0.05;
        int iter_num = 0;
        while (t < T)
        {
            // Update position of structures
            if (use_channel)
            {
                ls_time = t;
                for (int i = 0; i < meshes.size(); ++i)
                {
                    std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = mesh_mapping->getMeshPartitioner(i);
                    EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();

                    System& X_bdry_sys = eq_sys->get_system("COORDINATES_SYSTEM");
                    System& dX_bdry_sys = eq_sys->get_system("DISPLACEMENT_SYSTEM");
                    NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();
                    NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

                    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();

                    auto node_it = eq_sys->get_mesh().local_nodes_begin();
                    const auto node_end = eq_sys->get_mesh().local_nodes_end();
                    for (; node_it != node_end; ++node_it)
                    {
                        Node* node = *node_it;
                        VectorNd xpt = i == 0 ? lower_channel((*node)(0), t) : upper_channel((*node)(0), t);
                        std::vector<dof_id_type> X_bdry_dof_indices;
                        for (int d = 0; d < NDIM; ++d)
                        {
                            X_bdry_dof_map.dof_indices(node, X_bdry_dof_indices, d);
                            X_bdry_vec->set(X_bdry_dof_indices[0], xpt(d));
                            dX_bdry_vec->set(X_bdry_dof_indices[0], xpt(d) - (*node)(d));
                        }
                    }

                    X_bdry_vec->close();
                    dX_bdry_vec->close();
                    X_bdry_sys.update();
                    dX_bdry_sys.update();
                }
            }
            std::for_each(mesh_mapping->getMeshPartitioners().begin(),
                          mesh_mapping->getMeshPartitioners().end(),
                          [&patch_hierarchy](const std::shared_ptr<FEMeshPartitioner>& fe_mesh) -> void
                          {
                              fe_mesh->setPatchHierarchy(patch_hierarchy);
                              fe_mesh->reinitElementMappings();
                          });

            cut_cell_mapping->initializeObjectState(patch_hierarchy);
            cut_cell_mapping->generateCutCellMappings();

            vol_fcn->updateVolumeAreaSideLS(vol_idx,
                                            vol_var,
                                            IBTK::invalid_index,
                                            nullptr,
                                            IBTK::invalid_index,
                                            nullptr,
                                            ls_idx,
                                            ls_var,
                                            0.0,
                                            false);

            // Set initial conditions for Q
            if (use_channel)
            {
                PointwiseFunction<PointwiseFunctions::ScalarFcn> Q_init("QInit", Q_fcn_peristalsis);
                Q_init.setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, t);
            }
            else
            {
                PointwiseFunction<PointwiseFunctions::ScalarFcn> Q_init("QInit", Q_fcn_cylinder);
                Q_init.setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, t);
            }

            // Now generate the signed distance function. First interpolate to cell centers
            Pointer<HierarchyMathOps> hier_math_ops = new HierarchyMathOps("HierMathOps", patch_hierarchy);
            using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<ITC> ghost_comp{ ITC(ls_idx, "LINEAR_REFINE", false, "NONE") };
            Pointer<HierarchyGhostCellInterpolation> ls_ghost_fill = new HierarchyGhostCellInterpolation();
            ls_ghost_fill->initializeOperatorState(ghost_comp, patch_hierarchy);

            ReinitializeLevelSet ls_method("LS", input_db->getDatabase("ReintializeLevelSet"));
            {
                const int coarsest_ln = 0;
                const int finest_ln = patch_hierarchy->getFinestLevelNumber();

                for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
                {
                    Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
                    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                    {
                        Pointer<Patch<NDIM>> patch = level->getPatch(p());

                        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                        Pointer<NodeData<NDIM, int>> valid_data = patch->getPatchData(valid_idx);

                        Box<NDIM> ghost_node_box = NodeGeometry<NDIM>::toNodeBox(valid_data->getGhostBox());

                        for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
                        {
                            const NodeIndex<NDIM>& idx = ni();

                            if (std::abs((*ls_data)(idx)) < 1.0)
                            {
                                (*valid_data)(idx) = 1;
                            }
                            else
                            {
                                (*valid_data)(idx) = 2;
                            }
                        }

                        // Now only compute level set in nearby indices of structure
                        for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
                        {
                            const NodeIndex<NDIM>& idx = ni();

                            if ((*valid_data)(idx) == 1)
                            {
                                // We have a correctly specified index. Set 5 closest indices to be invalid but
                                // changeable values
                                Box<NDIM> region(idx, idx);
                                region.grow(10);
                                for (NodeIterator<NDIM> ni2(region); ni2; ni2++)
                                {
                                    const NodeIndex<NDIM>& idx2 = ni2();
                                    if (ghost_node_box.contains(idx2) && (*valid_data)(idx2) != 1)
                                        (*valid_data)(idx2) = 0;
                                }
                            }
                        }
                    }
                }
            }

            HierarchyNodeDataOpsReal<NDIM, double> hier_nc_data_ops(patch_hierarchy);
            hier_nc_data_ops.copyData(phi_idx, ls_idx);
            ls_method.computeSignedDistanceFunction(phi_idx, *phi_var, patch_hierarchy, t, valid_idx);

            InternalBdryFill advect_in_normal("InternalFill", input_db->getDatabase("InternalFill"));
            advect_in_normal.advectInNormal(Q_idx, Q_var, phi_idx, phi_var, patch_hierarchy, t);

            visit_data_writer->writePlotData(patch_hierarchy, iter_num, t);
            for (size_t i = 0; i < exodus_io.size(); ++i)
            {
                exodus_io[i]->write_timestep(
                    exodus_io_strs[i], *mesh_mapping->getMeshPartitioner(i)->getEquationSystems(), iter_num + 1, t);
            }

            t += dt;
            ++iter_num;
        }

        // Deallocate patch data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(ls_idxs);
        }

        for (auto& mesh : actual_meshes) delete mesh;

    } // cleanup dynamically allocated objects prior to shutdown
} // main
