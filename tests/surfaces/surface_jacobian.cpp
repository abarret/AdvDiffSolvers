#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/app_namespaces.h>
#include <ADS/surface_utilities.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <libmesh/edge_edge2.h>
#include <libmesh/exact_solution.h>
#include <libmesh/exodusII.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh_base.h>
#include <libmesh/mesh_tools.h>

#include <petscsys.h>

#include <array>
#include <memory>

double alpha = 0.0;
double g = 0.0;
double perim = 0.0;
double MFAC = 0.0;
double dx = 0.0;
double L = 0.0;
double t = 0.0;

VectorNd
channel(const double s, const double t)
{
    VectorNd x;
    x(0) = s;
    x(1) = alpha / (2.0 * M_PI) * (1.0 + g * std::sin(2.0 * M_PI * (s - t)));
    return x;
}

void
generate_structure(int& num_vertices, std::vector<IBTK::Point>& vertex_posn)
{
    // Generating upper level of channel.
    // Determine lag grid spacing.
    double ds = MFAC * perim * dx;
    num_vertices = std::ceil(L / ds);
    vertex_posn.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        VectorNd x = channel(ds * (static_cast<double>(i) + 0.15), 0.0);
        vertex_posn[i] = x;
    }
    return;
}

Vector3d
area_weighted_normal_vec(const double s, const double t)
{
    Vector3d T, n;
    T(0) = 1.0;
    T(1) = alpha * g * std::cos(2.0 * M_PI * (s - t));
    T(2) = 0.0;
    Vector3d ez = Vector3d::UnitZ();
    return T.cross(ez);
}

double
jacobian(const double s, const double t)
{
    Vector3d n = area_weighted_normal_vec(s, t);
    Vector3d N = area_weighted_normal_vec(s, 0);
    return N.norm() / n.norm();
}

double
jacobian(const libMesh::Point& p, const Parameters&, const std::string&, const std::string&)
{
    return jacobian(p(0), t);
}

class MeshMapping : public ADS::GeneralBoundaryMeshMapping
{
public:
    MeshMapping(std::string object_name, Pointer<Database> input_db, MeshBase* mesh)
        : GeneralBoundaryMeshMapping(std::move(object_name), input_db, mesh)
    {
        // intentionally blank.
    }

    void updateBoundaryLocation(double time, unsigned int part, bool end_of_timestep = false) override
    {
        // Loop through each element and set position
        // Set the X system to the element location.
        System& X_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_coords_sys_name);
        System& dX_bdry_sys = d_bdry_eq_sys_vec[part]->get_system(d_disp_sys_name);
        NumericVector<double>* X_bdry_vec = X_bdry_sys.solution.get();
        NumericVector<double>* dX_bdry_vec = dX_bdry_sys.solution.get();

        const DofMap& X_bdry_dof_map = X_bdry_sys.get_dof_map();

        int num_vertices = 0;
        std::vector<IBTK::Point> vertex_posn;
        generate_structure(num_vertices, vertex_posn);

        auto node_it = d_bdry_meshes[part]->local_nodes_begin();
        const auto node_end = d_bdry_meshes[part]->local_nodes_end();
        for (; node_it != node_end; ++node_it)
        {
            Node* node = *node_it;
            unsigned int node_id = node->id();
            std::vector<dof_id_type> X_bdry_dof_indices;
            for (int d = 0; d < NDIM; ++d)
            {
                X_bdry_dof_map.dof_indices(node, X_bdry_dof_indices, d);
                X_bdry_vec->set(X_bdry_dof_indices[0], channel(vertex_posn[node_id](0), time)[d]);
                dX_bdry_vec->set(X_bdry_dof_indices[0], channel(vertex_posn[node_id](0), time)[d] - (*node)(d));
            }
        }

        X_bdry_vec->close();
        dX_bdry_vec->close();
        X_bdry_sys.update();
        dX_bdry_sys.update();
    }
};

void
compute_error(FEMeshPartitioner& mesh_mapping, const std::string& J_str, const std::string& J_exact_str, double time);

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

        // Configure the IB solver.
        std::vector<std::string> struct_list = { "upper", "lower" };
        double N = input_db->getInteger("N");
        alpha = input_db->getDouble("ALPHA");
        g = input_db->getDouble("GAMMA");
        L = input_db->getDouble("L");
        MFAC = input_db->getDouble("MFAC");
        dx = L / N;
        perim = alpha * (L * M_PI + g * std::sin(L * M_PI) * std::sin(L * M_PI)) / (2.0 * M_PI * M_PI);

        // Generate finite element structure.
        libMesh::Mesh mesh(init.getLibMeshInit().comm(), NDIM - 1);
        {
            // Lower mesh
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(num_vertices, vertex_posn);
            mesh.reserve_nodes(num_vertices);
            mesh.reserve_elem(num_vertices - 1);
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
        }

        mesh.prepare_for_use();

        // Generate mesh mappings and level set information.
        auto mesh_mapping =
            std::make_shared<MeshMapping>("BoundaryMeshMapping", input_db->getDatabase("MeshMapping"), &mesh);
        mesh_mapping->initializeEquationSystems();

        // Setup systems.
        const std::string J_exact_str = "J_EXACT";
        const std::string J_str = "Jacobian";
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            const std::shared_ptr<FEMeshPartitioner>& mesh_partitioner = mesh_mapping->getMeshPartitioner(part);
            EquationSystems* eq_sys = mesh_partitioner->getEquationSystems();
            auto& J_exact_sys = eq_sys->add_system<ExplicitSystem>(J_exact_str);
            J_exact_sys.add_variable(J_exact_str);
            J_exact_sys.assemble_before_solve = false;
            J_exact_sys.assemble();

            auto& J_sys = eq_sys->add_system<ExplicitSystem>(J_str);
            J_sys.add_variable(J_str);
            J_sys.assemble_before_solve = false;
            J_sys.assemble();
        }

        mesh_mapping->initializeFEData();

        // Create mesh visualization.
        auto exodus_io = std::make_unique<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(0));

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Main time step loop.
        double time_end = 1.0;
        double dt = 0.1;
        int iteration_num = 0;
        while (!IBTK::rel_equal_eps(t, time_end))
        {
            pout << "Updating boundary at time " << t << "\n";
            mesh_mapping->updateBoundaryLocation(t, 0);
            pout << "Computing Jacobian at time " << t << "\n";
            update_jacobian(J_str, *mesh_mapping->getMeshPartitioner(0));
            pout << "Computing error in Jacobian at time " << t << "\n";
            compute_error(*mesh_mapping->getMeshPartitioner(0), J_str, J_exact_str, t);

            exodus_io->write_timestep(
                "mesh.ex2", *mesh_mapping->getMeshPartitioner(0)->getEquationSystems(), iteration_num + 1, t);
            t += dt;
            ++iteration_num;
        }

    } // cleanup dynamically allocated objects prior to shutdown
} // main

void
compute_error(FEMeshPartitioner& mesh_mapping, const std::string& J_str, const std::string& J_exact_str, double time)
{
    int num_vertices = 0;
    std::vector<IBTK::Point> vertex_posn;
    generate_structure(num_vertices, vertex_posn);
    EquationSystems* eq_sys = mesh_mapping.getEquationSystems();
    const MeshBase& mesh = eq_sys->get_mesh();

    auto& J_exact_sys = eq_sys->get_system<ExplicitSystem>(J_exact_str);
    NumericVector<double>* J_exact_vec = J_exact_sys.solution.get();

    auto it = mesh.local_nodes_begin();
    const auto it_end = mesh.local_nodes_end();
    for (; it != it_end; ++it)
    {
        const Node* node = *it;
        unsigned int node_id = node->id();
        J_exact_vec->set(node_id, jacobian((*node)(0), time));
    }

    J_exact_vec->close();
    J_exact_sys.update();

    // Now compute the error
    ExactSolution errs(*eq_sys);
    errs.attach_exact_value(jacobian);
    errs.compute_error(J_str, J_str);
    pout << " L1-norm:  " << errs.l1_error(J_str, J_str) << "\n";
    pout << " L2-norm:  " << errs.l2_error(J_str, J_str) << "\n";
    pout << " max-norm: " << errs.l_inf_error(J_str, J_str) << "\n";
}
