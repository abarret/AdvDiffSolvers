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

#include "LS/LSCutCellLaplaceOperator.h"
#include "LS/LSFromMesh.h"
#include "LS/QInitial.h"
#include "LS/SBBoundaryConditions.h"
#include "LS/SBIntegrator.h"
#include "LS/SemiLagrangianAdvIntegrator.h"

#include "ForcingFcn.h"
#include "QFcn.h"

#include <libmesh/boundary_mesh.h>
#include <libmesh/communicator.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

using namespace LS;

void postprocess_data(Pointer<PatchHierarchy<NDIM>> hierarchy,
                      Pointer<SemiLagrangianAdvIntegrator> integrator,
                      Pointer<CellVariable<NDIM, double>> Q_in_var,
                      int iteration_num,
                      double loop_time,
                      const std::string& dirname);

void update_mesh(MeshBase* mesh, FEDataManager* ib_data_manager, double time);

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
        VectorNd cent;
        input_db->getDoubleArray("CENTER", cent.data(), NDIM);

        string IB_delta_function = input_db->getString("IB_DELTA_FUNCTION");
        string elem_type = input_db->getString("ELEM_TYPE");
        const int second_order_mesh = (input_db->getString("elem_order") == "SECOND");
        string bdry_elem_type = second_order_mesh ? "EDGE3" : "EDGE2";

        Mesh solid_mesh(init.comm(), NDIM);
        const double R = 1.0;
        const int r = log2(0.25 * 2.0 * M_PI * R / ds);
        MeshTools::Generation::build_sphere(solid_mesh, R, r, Utility::string_to_enum<ElemType>(elem_type));
        // Turn into ellipse
        for (MeshBase::element_iterator it = solid_mesh.elements_begin(); it != solid_mesh.elements_end(); ++it)
        {
            Elem* const elem = *it;
            for (unsigned int n = 0; n < elem->n_nodes(); ++n)
            {
                Node* node = elem->node_ptr(n);
                libMesh::Point& pt = *node;
                double th = std::atan2(pt(1), pt(0));
                pt(0) = 1.25 * std::cos(th) - 0.75 * std::sin(th);
                pt(1) = 1.25 * std::cos(th) + 0.75 * std::sin(th);
            }
        }
        MeshTools::Modification::translate(solid_mesh, cent(0), cent(1));

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
        auto sb_data_manager =
            std::make_shared<SBSurfaceFluidCouplingManager>("SBDataManager",
                                                            app_initializer->getComponentDatabase("SBDataManager"),
                                                            ib_method_ops->getFEDataManager(REACTION_MESH_ID),
                                                            static_cast<Mesh*>(meshes[REACTION_MESH_ID]));

        // Set up visualization plot file writer.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        libMesh::UniquePtr<ExodusII_IO> reaction_exodus_io(uses_exodus ? new ExodusII_IO(*meshes[REACTION_MESH_ID]) :
                                                                         NULL);

        ib_method_ops->initializeFEData();

        sb_data_manager->initializeFEEquationSystems();
        sb_data_manager->setLSData(-1, -1, patch_hierarchy);

        adv_diff_integrator->setFEDataManagerNeedsInitialization(ib_method_ops->getFEDataManager(REACTION_MESH_ID));
        // Initialize hierarchy configuration and data on all patches.
        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

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

        sb_data_manager->updateJacobian();
        // Write out initial visualization data.
        EquationSystems* reaction_eq_sys = ib_method_ops->getFEDataManager(REACTION_MESH_ID)->getEquationSystems();
        if (dump_viz_data && uses_visit)
        {
            if (uses_exodus)
            {
                reaction_exodus_io->write_timestep(reaction_exodus_filename, *reaction_eq_sys, 1, 0.0);
            }
        }

        double time = 0.0;
        double final_time = 4.0;
        unsigned int num_ts = 100;
        double dt = final_time / static_cast<double>(num_ts);
        for (unsigned int i = 0; i < num_ts; ++i)
        {
            time = dt * static_cast<double>(i + 1);

            // Move structure to test Jacobian
            update_mesh(meshes[0], ib_method_ops->getFEDataManager(REACTION_MESH_ID), time);
            sb_data_manager->updateJacobian();

            if (dump_viz_data && uses_visit)
            {
                if (uses_exodus)
                {
                    reaction_exodus_io->write_timestep(reaction_exodus_filename, *reaction_eq_sys, i + 2, time);
                }
            }
        }
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
update_mesh(MeshBase* mesh, FEDataManager* fe_data_manager, double time)
{
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
    auto& X_sys = eq_sys->get_system<System>(fe_data_manager->COORDINATES_SYSTEM_NAME);
    auto& dX_sys = eq_sys->get_system<System>(IBFEMethod::COORD_MAPPING_SYSTEM_NAME);
    const DofMap& X_dof_map = X_sys.get_dof_map();
    const DofMap& dX_dof_map = dX_sys.get_dof_map();
    FEType X_fe_type = X_dof_map.variable_type(0);
    FEType dX_fe_type = dX_dof_map.variable_type(0);
    NumericVector<double>* X_vec = X_sys.solution.get();
    auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
    TBOX_ASSERT(X_petsc_vec != nullptr);
    double* X_local_soln = X_petsc_vec->get_array();

    NumericVector<double>* dX_vec = dX_sys.solution.get();
    auto dX_petsc_vec = dynamic_cast<PetscVector<double>*>(dX_vec);
    TBOX_ASSERT(dX_petsc_vec != nullptr);
    double* dX_local_soln = dX_petsc_vec->get_array();

    std::vector<dof_id_type> X_dof_indices, dX_dof_indices;

    auto it_iter = mesh->active_nodes_begin();
    const auto it_end = mesh->active_nodes_end();
    for (; it_iter != it_end; it_iter++)
    {
        const Node* const node = *it_iter;
        const libMesh::Point& pt = *node;
        VectorNd X;
        double th = std::atan2(pt(1), pt(0));
        X[0] = std::exp(sin(2 * M_PI * time) / (2.0 * M_PI)) *
               (pt(0) * std::cos(0.5 * M_PI * time) - pt(1) * std::sin(0.5 * M_PI * time));
        X[1] = std::exp(sin(2 * M_PI * time) / (2.0 * M_PI)) *
               (pt(1) * std::cos(0.5 * M_PI * time) + pt(0) * std::sin(0.5 * M_PI * time));
        for (int d = 0; d < NDIM; ++d)
        {
            X_dof_map.dof_indices(node, X_dof_indices, d);
            dX_dof_map.dof_indices(node, dX_dof_indices, d);
            X_local_soln[X_dof_indices[0]] = X(d);
            dX_local_soln[dX_dof_indices[0]] = X(d) - pt(d);
        }
    }
    X_petsc_vec->restore_array();
    dX_petsc_vec->restore_array();
}
