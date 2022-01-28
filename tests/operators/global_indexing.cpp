#include <ibamr/config.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/GhostPoints.h>
#include <ADS/GlobalIndexing.h>
#include <ADS/app_namespaces.h>

#include "libmesh/mesh_modification.h"
#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

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
    LibMeshInit& init = ibtk_init.getLibMeshInit();

    // Suppress a warning
    SAMRAI::tbox::Logger::getInstance()->setWarning(false);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");

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

        // Set up the finite element mesh
        // Note we use this to create "augmented" dofs.
        const double R = input_db->getDouble("RADIUS");
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;
        const std::string elem_type = input_db->getString("ELEM_TYPE");
        Mesh mesh(init.comm(), NDIM);
        const double num_circum_segments = 2.0 * M_PI * R / ds;
        const int r = std::log2(0.25 * num_circum_segments);
        MeshTools::Generation::build_sphere(mesh, R, r, Utility::string_to_enum<ElemType>(elem_type));
        // Ensure the nodes on the surface are on the analytic boundary.
        MeshBase::element_iterator el_end = mesh.elements_end();
        for (MeshBase::element_iterator el = mesh.elements_begin(); el != el_end; ++el)
        {
            Elem* const elem = *el;
            for (unsigned int side = 0; side < elem->n_sides(); ++side)
            {
                const bool at_mesh_bdry = !elem->neighbor_ptr(side);
                if (!at_mesh_bdry) continue;
                for (unsigned int k = 0; k < elem->n_nodes(); ++k)
                {
                    if (!elem->is_node_on_side(k, side)) continue;
                    Node& n = elem->node_ref(k);
                    n = R * n.unit();
                }
            }
        }
        MeshTools::Modification::translate(mesh, input_db->getDouble("XCOM"), input_db->getDouble("YCOM"));
        mesh.prepare_for_use();
        // Extract boundary mesh
        BoundaryMesh bdry_mesh(mesh.comm(), mesh.mesh_dimension() - 1);
        BoundaryInfo bdry_info = mesh.get_boundary_info();
        bdry_info.sync(bdry_mesh);
        bdry_mesh.prepare_for_use();

        // Setup mesh mapping
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &bdry_mesh);
        const std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner = mesh_mapping->getMeshPartitioner(0);
        EquationSystems* eq_sys = fe_mesh_partitioner->getEquationSystems();
        std::string sys_name = "sys";
        auto& sys = eq_sys->add_system<ExplicitSystem>(sys_name);
        sys.add_variable(sys_name, FEType());

        auto ghost_pts = std::make_shared<GhostPoints>(
            "GhostPoints", app_initializer->getComponentDatabase("GhostPoints"), patch_hierarchy, fe_mesh_partitioner);
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

        ghost_pts->findNormals();
        ghost_pts->setupGhostNodes();
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();

        GlobalIndexing indexing("GlobalIndexing", patch_hierarchy, fe_mesh_partitioner, sys_name, ghost_pts, 1);
        indexing.setupDOFs();

        int eul_idx = indexing.getEulerianMap();
        const std::map<int, int>& lag_map = indexing.getLagrangianMap();
        const std::map<int, int>& ghost_map = indexing.getGhostMap();

        // Now print everything out
        {
            plog << "Printing Eulerian data\n";
            for (int ln = 0; ln <= finest_ln; ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
                for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    Pointer<CellData<NDIM, int>> eul_data = patch->getPatchData(eul_idx);
                    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx = ci();
                        plog << "idx: " << idx << " has dof: " << (*eul_data)(idx) << "\n";
                    }
                }
            }
            plog << "Printing Lagrangian data\n";
            for (const auto& lag_petsc_pair : lag_map)
            {
                plog << "lag dof: " << lag_petsc_pair.first << " has dof: " << lag_petsc_pair.second << "\n";
            }
            plog << "Printing ghost data\n";
            for (const auto& ghost_petsc_pair : ghost_map)
            {
                plog << "ghost dof: " << ghost_petsc_pair.first << " has dof: " << ghost_petsc_pair.second << "\n";
            }
        }

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
fillLSEntireDomain(int ls_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> ls_data = ls_idx == -1 ? nullptr : patch->getPatchData(ls_idx);

            if (ls_data) ls_data->fillAll(-1.0);
        }
    }
}
