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
#include <ibamr/app_namespaces.h>

#include "ibtk/CartGridFunctionSet.h"
#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "InsideLSFcn.h"

#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

#include "../../include/CCAD/LSCutCellLaplaceOperator.h"
#include "../../include/CCAD/LSFromMesh.h"
#include "../../include/CCAD/QInitial.h"
#include "../../include/CCAD/SBBoundaryConditions.h"
#include "../../include/CCAD/SBIntegrator.h"
#include "../../include/CCAD/SemiLagrangianAdvIntegrator.h"

using namespace LS;
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
        const string reaction_exodus_filename = app_initializer->getExodusIIFilename("reaction");

        // Get various standard options set in the input file.
        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const std::string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        // Create a simple FE mesh.
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;

        string IB_delta_function = input_db->getString("IB_DELTA_FUNCTION");
        string elem_type = input_db->getString("ELEM_TYPE");
        const int second_order_mesh = (input_db->getString("elem_order") == "SECOND");
        string bdry_elem_type = second_order_mesh ? "EDGE3" : "EDGE2";

        Mesh solid_mesh(init.comm(), NDIM);
        const double R = 1.0;
        const int r = log2(0.25 * 2.0 * M_PI * R / ds);
        MeshTools::Generation::build_sphere(solid_mesh, R, r, Utility::string_to_enum<ElemType>(bdry_elem_type));
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

        solid_mesh.prepare_for_use();
        BoundaryMesh reaction_mesh(solid_mesh.comm(), solid_mesh.mesh_dimension() - 1);
        solid_mesh.boundary_info->sync(reaction_mesh);
        reaction_mesh.set_spatial_dimension(NDIM);
        reaction_mesh.prepare_for_use();

        static const int REACTION_MESH_ID = 0;
        vector<MeshBase*> meshes(1);
        meshes[REACTION_MESH_ID] = &reaction_mesh;

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

        ib_method_ops->initializeFEEquationSystems();
        ib_method_ops->initializeFEData();
        // Create Eulerian boundary condition specification objects.
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, static_cast<RobinBcCoefStrategy<NDIM>*>(NULL));

        Pointer<CellVariable<NDIM, double>> ls_c_var = new CellVariable<NDIM, double>("ls_c_var");
        Pointer<NodeVariable<NDIM, double>> ls_n_var = new NodeVariable<NDIM, double>("ls_n_var");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<CellVariable<NDIM, double>> area_var = new CellVariable<NDIM, double>("AREA");
        Pointer<SideVariable<NDIM, double>> side_var = new SideVariable<NDIM, double>("SIDE");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_c_idx = var_db->registerVariableAndContext(ls_c_var, ctx, IntVector<NDIM>(2));
        const int ls_n_idx = var_db->registerVariableAndContext(ls_n_var, ctx, IntVector<NDIM>(2));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(2));
        const int area_idx = var_db->registerVariableAndContext(area_var, ctx, IntVector<NDIM>(2));
        const int side_idx = var_db->registerVariableAndContext(side_var, ctx, IntVector<NDIM>(2));

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

        for (unsigned int part = 0; part < meshes.size(); ++part)
        {
            ib_method_ops->getFEDataManager(part)->setPatchHierarchy(patch_hierarchy);
            ib_method_ops->getFEDataManager(part)->reinitElementMappings();
        }
        libMesh::UniquePtr<ExodusII_IO> reaction_exodus_io(new ExodusII_IO(*meshes[REACTION_MESH_ID]));
        FEDataManager* fe_data_manager = ib_method_ops->getFEDataManager(REACTION_MESH_ID);

        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls_c", "SCALAR", ls_c_idx);
        visit_data_writer->registerPlotQuantity("ls_n", "SCALAR", ls_n_idx);
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("area", "SCALAR", area_idx);

        // Allocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_c_idx);
            level->allocatePatchData(ls_n_idx);
            level->allocatePatchData(vol_idx);
            level->allocatePatchData(area_idx);
            level->allocatePatchData(side_idx);
        }

        Pointer<LSFindCellVolume> vol_fcn = new LSFindCellVolume("VolFcn", patch_hierarchy);
        Pointer<InsideLSFcn> ls_fcn = new InsideLSFcn("LSFcn",
                                                      app_initializer->getComponentDatabase("InsideLSFcn"),
                                                      &reaction_mesh,
                                                      &reaction_mesh,
                                                      fe_data_manager);

        ls_fcn->setDataOnPatchHierarchy(ls_c_idx, ls_c_var, patch_hierarchy, 0.0);
        ls_fcn->setDataOnPatchHierarchy(ls_n_idx, ls_n_var, patch_hierarchy, 0.0);

        vol_fcn->updateVolumeAndArea(vol_idx, vol_var, area_idx, area_var, ls_n_idx, ls_n_var, true);

        pout << "Writing control results.\n";
        reaction_exodus_io->write_timestep(reaction_exodus_filename, *fe_data_manager->getEquationSystems(), 1, 0.0);
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);

        // Now fill in ls data from mesh
        Pointer<LSFromMesh> ls_from_mesh =
            new LSFromMesh("LSFromMesh", patch_hierarchy, &reaction_mesh, fe_data_manager);
        ls_from_mesh->updateVolumeAreaLength(
            vol_idx, vol_var, area_idx, area_var, side_idx, side_var, ls_n_idx, ls_n_var);

        pout << "Writing new results.\n";
        reaction_exodus_io->write_timestep(reaction_exodus_filename, *fe_data_manager->getEquationSystems(), 2, 0.0);
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
