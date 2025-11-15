#include <ibamr/config.h>

#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/IndexElemMapping.h>
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

#include <libmesh/boundary_mesh.h>
#include <libmesh/communicator.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>

enum class InterfaceType
{
    DISK,
    PERISTALSIS,
    CHANNEL,
    SPHERE,
    ELEMENT,
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
    case InterfaceType::ELEMENT:
        return "ELEMENT";
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
    if (strcasecmp(str.c_str(), "ELEMENT") == 0) return InterfaceType::ELEMENT;
    return InterfaceType::UNKNOWN;
}

double R;
double r;
std::string elem_type;
VectorNd cent;
double L;
double ds;
double dx;
double y_low, y_up;
double theta;

std::unique_ptr<Mesh>
build_cylinder(IBTKInit& init)
{
    Mesh solid_mesh(init.getLibMeshInit().comm(), NDIM);
    MeshTools::Generation::build_sphere(solid_mesh, R, r, Utility::string_to_enum<ElemType>(elem_type));
    for (MeshBase::element_iterator it = solid_mesh.elements_begin(); it != solid_mesh.elements_end(); ++it)
    {
        Elem* const elem = *it;
        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            const bool at_mesh_bdry = !elem->neighbor_ptr(side);
            if (!at_mesh_bdry) continue;
            for (unsigned int k = 0; k < elem->n_nodes(); ++k)
            {
                if (!elem->is_node_on_side(k, side)) continue;
                Node& n = *elem->node_ptr(k);
                n = R * n.unit();
            }
        }
    }

    MeshTools::Modification::translate(solid_mesh, cent(0), cent(1));

    solid_mesh.prepare_for_use();
    BoundaryMesh bdry_mesh(solid_mesh.comm(), solid_mesh.mesh_dimension() - 1);
    solid_mesh.boundary_info->sync(bdry_mesh);
    bdry_mesh.set_spatial_dimension(NDIM);
    bdry_mesh.prepare_for_use();
    return std::make_unique<Mesh>(bdry_mesh);
}

void
build_channel(IBTKInit& init, std::vector<std::unique_ptr<Mesh>>& meshes)
{
    Mesh lower_mesh(init.getLibMeshInit().comm(), NDIM), upper_mesh(init.getLibMeshInit().comm(), NDIM);
    MeshTools::Generation::build_line(lower_mesh,
                                      static_cast<int>(ceil(L / ds)),
                                      0.0 + 0.0 * dx,
                                      L - 0.0 * dx,
                                      Utility::string_to_enum<ElemType>(elem_type));
    MeshTools::Generation::build_line(upper_mesh,
                                      static_cast<int>(ceil(L / ds)),
                                      0.0 + 0.0 * dx,
                                      L - 0.0 * dx,
                                      Utility::string_to_enum<ElemType>(elem_type));

    for (MeshBase::node_iterator it = lower_mesh.nodes_begin(); it != lower_mesh.nodes_end(); ++it)
    {
        Node* n = *it;
        libMesh::Point& X = *n;
        X(1) = y_low + std::tan(theta) * X(0);
    }

    for (MeshBase::node_iterator it = upper_mesh.nodes_begin(); it != upper_mesh.nodes_end(); ++it)
    {
        Node* n = *it;
        libMesh::Point& X = *n;
        X(1) = y_up + std::tan(theta) * X(0);
    }

    lower_mesh.prepare_for_use();
    upper_mesh.prepare_for_use();

    meshes.push_back(std::make_unique<Mesh>(lower_mesh));
    meshes.push_back(std::make_unique<Mesh>(upper_mesh));
}

std::unique_ptr<Mesh>
build_sphere(IBTKInit& init)
{
    Mesh solid_mesh(init.getLibMeshInit().comm(), NDIM);
    MeshTools::Generation::build_sphere(solid_mesh, R, r, Utility::string_to_enum<ElemType>(elem_type));
    for (MeshBase::element_iterator it = solid_mesh.elements_begin(); it != solid_mesh.elements_end(); ++it)
    {
        Elem* const elem = *it;
        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            const bool at_mesh_bdry = !elem->neighbor_ptr(side);
            if (!at_mesh_bdry) continue;
            for (unsigned int k = 0; k < elem->n_nodes(); ++k)
            {
                if (!elem->is_node_on_side(k, side)) continue;
                Node& n = *elem->node_ptr(k);
                n = R * n.unit();
            }
        }
    }

    MeshTools::Modification::translate(solid_mesh, cent(0), cent(1), cent(2));

    solid_mesh.prepare_for_use();
    BoundaryMesh bdry_mesh(solid_mesh.comm(), solid_mesh.mesh_dimension() - 1);
    solid_mesh.boundary_info->sync(bdry_mesh);
    bdry_mesh.set_spatial_dimension(NDIM);
    bdry_mesh.prepare_for_use();
    return std::make_unique<Mesh>(bdry_mesh);
}

std::unique_ptr<Mesh>
build_element(IBTKInit& init)
{
    auto mesh = std::make_unique<libMesh::Mesh>(init.getLibMeshInit().comm(), NDIM - 1);
    mesh->reserve_nodes(3);
    mesh->reserve_elem(1);
    unsigned int node_id = 0;
    mesh->add_point(libMesh::Point(-1.2, 1.6, 0.3), node_id++, 0);
    mesh->add_point(libMesh::Point(0.45, -1.1, -0.4), node_id++, 0);
    mesh->add_point(libMesh::Point(1.7, 1.1, -1.2), node_id++, 0);
    Elem* elem = mesh->add_elem(Elem::build_with_id(libMesh::TRI3, 0));
    elem->set_node(0) = mesh->node_ptr(0);
    elem->set_node(1) = mesh->node_ptr(1);
    elem->set_node(2) = mesh->node_ptr(2);
    mesh->prepare_for_use(false);
    return mesh;
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

        dx = input_db->getDouble("DX");
        ds = input_db->getDouble("MFAC") * dx;
        elem_type = input_db->getString("ELEM_TYPE");

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

        Pointer<CellVariable<NDIM, int>> pt_type_var = new CellVariable<NDIM, int>("pt_type");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int pt_type_idx = var_db->registerVariableAndContext(pt_type_var, ctx, IntVector<NDIM>(1));
        std::set<int> idx_set{ pt_type_idx };
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

        // Allocate data
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        allocate_patch_data(idxs, patch_hierarchy, 0.0, coarsest_ln, finest_ln);

        std::vector<std::unique_ptr<Mesh>> meshes;
        std::vector<MeshBase*> mesh_ptrs;
        // Fill level set data.
        if (interface == InterfaceType::DISK)
        {
            R = input_db->getDouble("R");
            r = std::log2(0.25 * 2.0 * M_PI * R / ds);
            meshes.push_back(std::move(build_cylinder(ibtk_init)));
        }
        else if (interface == InterfaceType::CHANNEL)
        {
            L = input_db->getDouble("L");
            theta = input_db->getDouble("THETA");
            y_low = input_db->getDouble("Y_LOW");
            y_up = input_db->getDouble("Y_UP");
            build_channel(ibtk_init, meshes);
        }
        else if (interface == InterfaceType::SPHERE)
        {
            R = input_db->getDouble("R");
            r = std::log2(0.25 * 2.0 * M_PI * R / ds);
            meshes.push_back(std::move(build_sphere(ibtk_init)));
        }
        else if (interface == InterfaceType::ELEMENT)
        {
            meshes.push_back(std::move(build_element(ibtk_init)));
        }

        for (const auto& mesh : meshes) mesh_ptrs.push_back(mesh.get());
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "mesh_mapping", input_db->getDatabase("MeshMapping"), mesh_ptrs);
        mesh_mapping->initializeEquationSystems();
        mesh_mapping->initializeFEData();

        std::vector<std::unique_ptr<FEToHierarchyMapping>> fe_hier_mappings;
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            static const IntVector<NDIM> gcw(1);
            fe_hier_mappings.push_back(
                std::make_unique<FEToHierarchyMapping>("FEToHierarchyMapping_" + std::to_string(part),
                                                       &mesh_mapping->getSystemManager(part),
                                                       nullptr,
                                                       patch_hierarchy->getNumberOfLevels(),
                                                       gcw));
            fe_hier_mappings[part]->setPatchHierarchy(patch_hierarchy);
            fe_hier_mappings[part]->reinitElementMappings(gcw);
        }
        Pointer<IndexElemMapping> idx_elem_mapping =
            new IndexElemMapping("idx_elem_mapping", input_db->getDatabase("IndexElemMapping"));

// Uncomment to draw data.
#define DRAW_DATA 1
#ifdef DRAW_DATA
        std::vector<std::unique_ptr<ExodusII_IO>> struct_writers;
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
            struct_writers.push_back(std::make_unique<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(part)));
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("Pt_type", "SCALAR", pt_type_idx);
#endif

        if (interface == InterfaceType::CHANNEL)
        {
            std::vector<int> reverse_norms = { 0, 1 };
            sharp_interface::classify_points_struct(pt_type_idx,
                                                    patch_hierarchy,
                                                    unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings),
                                                    idx_elem_mapping,
                                                    reverse_norms,
                                                    false);
        }
        else
        {
            sharp_interface::classify_points_struct(
                pt_type_idx, patch_hierarchy, unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings), idx_elem_mapping, false);
        }

        // Now find the image points
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            std::vector<std::vector<sharp_interface::ImagePointData>> ip_data_vec_vec =
                sharp_interface::find_image_points(
                    pt_type_idx, patch_hierarchy, ln, unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings));
            pout << "Writing image point data:\n";
            for (const auto& ip_data_vec : ip_data_vec_vec)
            {
                for (const auto& ip_data : ip_data_vec)
                {
                    pout << "Ghost point index:       " << ip_data.d_gp_idx << "\n";
                    pout << "Boundary Point location: " << ip_data.d_bp_location.transpose() << "\n";
                    pout << "Normal:                  " << ip_data.d_normal.transpose() << "\n";
                    pout << "Image point index:       " << ip_data.d_ip_idx << "\n";
                    pout << "Image point location:    " << ip_data.d_ip_location.transpose() << "\n";
                    pout << "Part:                    " << ip_data.d_part << "\n";
                }
            }
        }
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
            struct_writers[part]->write_timestep("exodus" + std::to_string(part) + ".ex",
                                                 *mesh_mapping->getSystemManager(part).getEquationSystems(),
                                                 1,
                                                 0.0);
#endif

        // Deallocate data
        deallocate_patch_data(idxs, patch_hierarchy, coarsest_ln, finest_ln);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
