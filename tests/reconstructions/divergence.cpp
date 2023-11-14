#include <ibamr/config.h>

#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/LSFromMesh.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/RBFDivergenceReconstructions.h>
#include <ADS/app_namespaces.h>

#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"

#include <libmesh/edge_edge2.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh_generation.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

int finest_ln;
std::array<int, NDIM> N;
double MFAC = 0.0;
double dx = 0.0;
double L = 0.0;
double ds;
double R = 0.0;
VectorNd cent;

VectorNd
cylinder_pt(const double s, const double t)
{
    VectorNd x;
    x(0) = cent(0) + R * std::cos(2.0 * M_PI * s);
    x(1) = cent(1) + R * std::sin(2.0 * M_PI * s);
    return x;
}

void
generate_structure(int& num_vertices, std::vector<IBTK::Point>& vertex_posn)
{
    // Generating upper level of channel.
    // Determine lag grid spacing.
    double circum = 2.0 * M_PI * R;
    ds = MFAC * circum * dx;
    num_vertices = std::ceil(L / ds);
    vertex_posn.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        VectorNd x = cylinder_pt(ds * (static_cast<double>(i) + 0.5), 0.0);
        vertex_posn[i] = x;
    }
    return;
}

double
u_fcn(const double /*u*/, const VectorNd& x, const double time, const int axis)
{
    return std::pow(std::sin(0.5 * M_PI * x[0]) * std::sin(0.5 * M_PI * x[1]), 4.0);
}

VectorNd
u_draw_fcn(const VectorNd& /*u*/, const VectorNd& x, const double time)
{
    VectorNd u;
    for (int d = 0; d < NDIM; ++d) u[d] = std::pow(std::sin(0.5 * M_PI * x[0]) * std::sin(0.5 * M_PI * x[1]), 4.0);
    return u;
}

double
div_fcn(const double /*div*/, const VectorNd& x, const double time)
{
    return 2.0 * M_PI *
               (std::cos(0.5 * M_PI * x[1]) * std::sin(0.5 * M_PI * x[0]) *
                std::pow(std::sin(0.5 * M_PI * x[0]) * std::sin(0.5 * M_PI * x[1]), 3.0)) +
           2.0 * M_PI *
               (std::cos(0.5 * M_PI * x[0]) * std::sin(0.5 * M_PI * x[1]) *
                std::pow(std::sin(0.5 * M_PI * x[0]) * std::sin(0.5 * M_PI * x[1]), 3.0));
}

void
compute_divergence(std::shared_ptr<GeneralBoundaryMeshMapping>& mesh_mapping,
                   const std::string& div_sys_name,
                   const std::string& err_sys_name,
                   const int u_idx,
                   const int ls_idx,
                   Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = mesh_mapping->getMeshPartitioner();
    EquationSystems* eq_sys = mesh_mapping->getMeshPartitioner()->getEquationSystems();

    // Pull out relevant FE data
    System& X_bdry_sys = eq_sys->get_system(mesh_partitioner->COORDINATES_SYSTEM_NAME);
    const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();
    NumericVector<double>* X_bdry_vec = X_bdry_sys.current_local_solution.get();

    System& divu_sys = eq_sys->get_system(div_sys_name);
    const DofMap& divu_dof_map = divu_sys.get_dof_map();
    NumericVector<double>* divu_vec = divu_sys.solution.get();

    System& err_sys = eq_sys->get_system(err_sys_name);
    const DofMap& err_dof_map = err_sys.get_dof_map();
    NumericVector<double>* err_vec = err_sys.solution.get();

    // Loop through all local nodes
    auto it = eq_sys->get_mesh().local_nodes_begin();
    auto it_end = eq_sys->get_mesh().local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* const node = *it;

        // Get the current location of the node
        VectorNd xpt;
        std::vector<dof_id_type> X_dofs;
        X_bdry_dof_map.dof_indices(node, X_dofs);
        for (int d = 0; d < NDIM; ++d) xpt[d] = (*X_bdry_vec)(X_dofs[d]);

        // Compute the divergence. We first need the index of the point. Assume this point is on the specified level
        Pointer<GridGeometry<NDIM>> grid_geom = hierarchy->getGridGeometry();
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(hierarchy->getFinestLevelNumber());
        const CellIndex<NDIM>& idx = IndexUtilities::getCellIndex(xpt, grid_geom, level->getRatio());
        Pointer<Patch<NDIM>> patch;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            patch = level->getPatch(p());
            // Is this point located on this patch?
            if (patch->getBox().contains(idx)) continue;
        }

        // We have the patch and current index. Let's compute divergence. We are reconstruction from negative side.
        const static double ls = -1.0;
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        Pointer<SideData<NDIM, double>> u_data = patch->getPatchData(u_idx);
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const hier::Index<NDIM>& idx_low = patch->getBox().lower();
        // Convert xpt to index space
        VectorNd x_idx;
        for (int d = 0; d < NDIM; ++d) x_idx[d] = (xpt[d] - xlow[d]) / dx[d] + idx_low(d);
        double div = ADS::Reconstruct::divergence(
            x_idx, idx, ls, *u_data, *ls_data, ADS::Reconstruct::RBFPolyOrder::QUADRATIC, 12, dx, false);

        std::vector<dof_id_type> divu_dofs;
        divu_dof_map.dof_indices(node, divu_dofs);
        divu_vec->set(divu_dofs[0], div);

        std::vector<dof_id_type> err_dofs;
        err_dof_map.dof_indices(node, err_dofs);
        err_vec->set(err_dofs[0], std::abs(div - div_fcn(0.0, xpt, 0.0)));
    }

    divu_vec->close();
    divu_sys.update();
    err_vec->close();
    err_sys.update();
}

void
computeSurfaceErrors(std::shared_ptr<FEMeshPartitioner> fe_data_manager, const std::string& err_name)
{
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
    const MeshBase& mesh = eq_sys->get_mesh();
    ExplicitSystem& err_sys = eq_sys->get_system<ExplicitSystem>(err_name);
    NumericVector<double>* err_vec = err_sys.solution.get();

    const DofMap& err_dof_map = err_sys.get_dof_map();
    const FEType& err_fe_type = err_dof_map.variable_type(0);

    std::unique_ptr<FEBase> fe = FEBase::build(mesh.mesh_dimension(), err_fe_type);
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, mesh.mesh_dimension(), THIRD);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<std::vector<double>>& phi = fe->get_phi();
    const std::vector<double>& JxW = fe->get_JxW();

    auto it = mesh.local_nodes_begin();
    const auto it_end = mesh.local_nodes_end();
    double l1_norm = 0.0, l2_norm = 0.0, max_norm = 0.0;
    for (; it != it_end; ++it)
    {
        Node* n = *it;
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = (*n)(d);
        const int dof_index = n->dof_number(err_sys.number(), 0, 0);
        max_norm = std::max(max_norm, (*err_vec)(dof_index));
    }

    auto it_e = mesh.local_elements_begin();
    const auto& it_e_end = mesh.local_elements_end();
    for (; it_e != it_e_end; ++it_e)
    {
        Elem* el = *it_e;
        fe->reinit(el);
        boost::multi_array<double, 1> err_node;
        std::vector<dof_id_type> err_dof_indices;
        err_dof_map.dof_indices(el, err_dof_indices);
        IBTK::get_values_for_interpolation(err_node, *err_vec, err_dof_indices);
        for (unsigned int qp = 0; qp < JxW.size(); ++qp)
        {
            for (unsigned int n = 0; n < phi.size(); ++n)
            {
                l1_norm += err_node[n] * phi[n][qp] * JxW[qp];
                l2_norm += std::pow(err_node[n] * phi[n][qp], 2.0) * JxW[qp];
            }
        }
    }
    l2_norm = std::sqrt(l2_norm);

    pout << "Error at surface\n";
    pout << " L1-norm:  " << l1_norm << "\n";
    pout << " L2-norm:  " << l2_norm << "\n";
    pout << " max-norm: " << max_norm << "\n";

    err_vec->close();
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

        N[0] = N[1] = input_db->getInteger("NFINEST");
        finest_ln = input_db->getInteger("MAX_LEVELS") - 1;
        R = input_db->getDouble("R");
        L = input_db->getDouble("L");
        MFAC = input_db->getDouble("MFAC");
        dx = L / N[0];

        // Generate the mesh
        libMesh::Mesh mesh(ibtk_init.getLibMeshInit().comm(), NDIM - 1);
        {
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(num_vertices, vertex_posn);
            mesh.reserve_nodes(num_vertices);
            mesh.reserve_elem(num_vertices);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0), node_num);
            }

            // Generate elements
            for (int i = 0; i < num_vertices - 1; ++i)
            {
                Elem* elem = mesh.add_elem(new libMesh::Edge2());
                elem->set_node(0) = mesh.node_ptr(i);
                elem->set_node(1) = mesh.node_ptr(i + 1);
            }

            // Last element is from node num_vertices to node 0
            Elem* elem = mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = mesh.node_ptr(num_vertices - 1);
            elem->set_node(1) = mesh.node_ptr(0);
        }
        mesh.prepare_for_use();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", nullptr, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<SideVariable<NDIM, double>> u_var = new SideVariable<NDIM, double>("u");
        Pointer<CellVariable<NDIM, double>> u_draw_var = new CellVariable<NDIM, double>("u_draw", NDIM);
        Pointer<CellVariable<NDIM, double>> div_var = new CellVariable<NDIM, double>("div");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(4));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(4));
        const int u_idx = var_db->registerVariableAndContext(u_var, ctx, IntVector<NDIM>(0));
        const int u_draw_idx = var_db->registerVariableAndContext(u_draw_var, ctx);
        const int div_idx = var_db->registerVariableAndContext(div_var, ctx);

        ComponentSelector comps;
        comps.setFlag(ls_idx);
        comps.setFlag(vol_idx);
        comps.setFlag(u_idx);
        comps.setFlag(u_draw_idx);
        comps.setFlag(div_idx);

        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &mesh);
        mesh_mapping->initializeEquationSystems();

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
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("u", "VECTOR", u_draw_idx);
        visit_data_writer->registerPlotQuantity("div", "SCALAR", div_idx);
        std::unique_ptr<ExodusII_IO> mesh_output(new ExodusII_IO(*mesh_mapping->getBoundaryMesh()));
#endif

        // Allocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(comps);
        }

        // Set up boundary data
        const std::string div_sys_name = "div";
        const std::string err_sys_name = "err";
        EquationSystems* eq_sys = mesh_mapping->getMeshPartitioner()->getEquationSystems();
        auto& div_sys = eq_sys->add_system<ExplicitSystem>(div_sys_name);
        div_sys.add_variable(div_sys_name);
        div_sys.assemble_before_solve = false;
        div_sys.assemble();

        auto& err_sys = eq_sys->add_system<ExplicitSystem>(err_sys_name);
        err_sys.add_variable(err_sys_name);
        err_sys.assemble_before_solve = false;
        err_sys.assemble();

        mesh_mapping->initializeFEData();

        mesh_mapping->getMeshPartitioner()->setPatchHierarchy(patch_hierarchy);
        mesh_mapping->getMeshPartitioner()->reinitElementMappings();

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellVolumeMeshMapping("CutCellMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         mesh_mapping->getMeshPartitioners());
        LSFromMesh ls_vol_fcn("ls", patch_hierarchy, cut_cell_mapping, true);
        ls_vol_fcn.registerNormalReverseDomainId(0, 0);
        ls_vol_fcn.updateVolumeAreaSideLS(
            vol_idx, vol_var, IBTK::invalid_index, nullptr, IBTK::invalid_index, nullptr, ls_idx, ls_var, 0.0);

        PointwiseFunction<PointwiseFunctions::StaggeredFcn> u_pt_fcn("UFcn", u_fcn);
        PointwiseFunction<PointwiseFunctions::VectorFcn> u_draw_pt_fcn("UFcn", u_draw_fcn);
        PointwiseFunction<PointwiseFunctions::ScalarFcn> div_pt_fcn("divFcn", div_fcn);

        u_pt_fcn.setDataOnPatchHierarchy(u_idx, u_var, patch_hierarchy, 0.0);
        u_draw_pt_fcn.setDataOnPatchHierarchy(u_draw_idx, u_draw_var, patch_hierarchy, 0.0);
        div_pt_fcn.setDataOnPatchHierarchy(div_idx, div_var, patch_hierarchy, 0.0);

        compute_divergence(mesh_mapping, div_sys_name, err_sys_name, u_idx, ls_idx, patch_hierarchy);

        // Compute error
        computeSurfaceErrors(mesh_mapping->getMeshPartitioner(), err_sys_name);

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
        mesh_output->write_timestep("mesh.ex", *mesh_mapping->getMeshPartitioner()->getEquationSystems(), 1, 0.0);
#endif

        // Deallocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(comps);
        }

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
