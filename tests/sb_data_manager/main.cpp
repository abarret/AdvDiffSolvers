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

#include <libmesh/boundary_mesh.h>
#include <libmesh/communicator.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

#include "../../include/ADS/LSCutCellLaplaceOperator.h"
#include "../../include/ADS/LSFromMesh.h"
#include "../../include/ADS/QInitial.h"
#include "../../include/ADS/SBBoundaryConditions.h"
#include "../../include/ADS/SBIntegrator.h"
#include "../../include/ADS/SemiLagrangianAdvIntegrator.h"

using namespace LS;

void update_mesh(MeshBase* mesh, FEDataManager* ib_data_manager, double time);

void setInitialCondition(const std::string& sf_name,
                         const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                         MeshBase* mesh);
void updateExactAndError(const std::string& sf_name,
                         const std::string& err_name,
                         const std::string& exact_name,
                         const std::string& J_exa_name,
                         const std::string& J_err_name,
                         const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                         MeshBase* mesh,
                         const double time);

static double lambda = 0.0;

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

        const std::string sf_name = "Surface";
        sb_data_manager->registerSurfaceConcentration(sf_name);
        auto sb_integrator = std::make_shared<SBIntegrator>("SBIntegrator",
                                                            app_initializer->getComponentDatabase("SBIntegrator"),
                                                            sb_data_manager,
                                                            static_cast<Mesh*>(meshes[REACTION_MESH_ID]));
        sb_integrator->setLSData(-1, -1, patch_hierarchy);

        lambda = input_db->getDouble("LAMBDA");
        auto rhs_fcn = [](double Q,
                          const std::vector<double>& fl_vals,
                          const std::vector<double>& sf_vals,
                          const double time,
                          void* ctx) -> double { return lambda * Q; };
        sb_data_manager->registerSurfaceReactionFunction(sf_name, rhs_fcn);

        EquationSystems* reaction_eq_sys = ib_method_ops->getFEDataManager(REACTION_MESH_ID)->getEquationSystems();
        const std::string err_name = "Error", exa_name = "Exact";
        auto& err_sys = reaction_eq_sys->add_system<ExplicitSystem>(err_name);
        err_sys.add_variable(err_name, FEType());
        auto& exa_sys = reaction_eq_sys->add_system<ExplicitSystem>(exa_name);
        exa_sys.add_variable(exa_name, FEType());

        const std::string J_err_name = "J_Error", J_exa_name = "J_Exact";
        auto& J_err_sys = reaction_eq_sys->add_system<ExplicitSystem>(J_err_name);
        J_err_sys.add_variable(J_err_name, FEType());
        auto& J_exa_sys = reaction_eq_sys->add_system<ExplicitSystem>(J_exa_name);
        J_exa_sys.add_variable(J_exa_name, FEType());

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

        // Set the initial condition
        setInitialCondition(sf_name, sb_data_manager, meshes[REACTION_MESH_ID]);

        sb_data_manager->updateJacobian();
        // Write out initial visualization data.
        updateExactAndError(
            sf_name, err_name, exa_name, J_exa_name, J_err_name, sb_data_manager, meshes[REACTION_MESH_ID], 0.0);
        if (dump_viz_data && uses_visit)
        {
            if (uses_exodus)
            {
                reaction_exodus_io->write_timestep(reaction_exodus_filename, *reaction_eq_sys, 1, 0.0);
            }
        }

        double final_time = 4.0;
        double current_time = 0.0, new_time;
        double dt = input_db->getDouble("DT_MAX");
        unsigned int i = 0;
        while (current_time < final_time)
        {
            dt = std::min(dt, final_time - current_time);
            new_time = current_time + dt;

            // Move structure to test Jacobian
            pout << "Timestepping at time: " << current_time << "\n";
            pout << "dt is: " << dt << "\n";
            update_mesh(meshes[0], ib_method_ops->getFEDataManager(REACTION_MESH_ID), current_time);
            sb_integrator->beginTimestepping(current_time, new_time);
            sb_integrator->integrateHierarchy(nullptr, current_time, new_time);
            sb_integrator->endTimestepping(current_time, new_time);
            current_time = new_time;
            pout << "Finished stepping at time: " << current_time << "\n";
            pout << "Updating mesh location at time: " << current_time << "\n";
            update_mesh(meshes[0], ib_method_ops->getFEDataManager(REACTION_MESH_ID), current_time);
            sb_data_manager->updateJacobian();
            pout << "Computing error at time: " << current_time << "\n\n\n";
            updateExactAndError(sf_name,
                                err_name,
                                exa_name,
                                J_exa_name,
                                J_err_name,
                                sb_data_manager,
                                meshes[REACTION_MESH_ID],
                                current_time);

            if (dump_viz_data && uses_visit)
            {
                if (uses_exodus)
                {
                    reaction_exodus_io->write_timestep(reaction_exodus_filename, *reaction_eq_sys, i + 2, current_time);
                }
            }
            ++i;
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

void
setInitialCondition(const std::string& sf_name,
                    const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                    MeshBase* mesh)
{
    EquationSystems* eq_sys = sb_data_manager->getFEDataManager()->getEquationSystems();
    auto& sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
    NumericVector<double>* soln = sys.solution.get();
    const DofMap& sys_dof_map = sys.get_dof_map();
    MeshBase::const_node_iterator node_it = mesh->active_nodes_begin();
    MeshBase::const_node_iterator node_end = mesh->active_nodes_end();
    for (; node_it != node_end; node_it++)
    {
        const Node* node = *node_it;
        std::vector<dof_id_type> dof_indices;
        sys_dof_map.dof_indices(node, dof_indices);
        for (const auto dof_index : dof_indices)
        {
            soln->set(dof_index, 1.0);
        }
    }
    soln->close();
}

void
updateExactAndError(const std::string& sf_name,
                    const std::string& err_name,
                    const std::string& exact_name,
                    const std::string& J_exa_name,
                    const std::string& J_err_name,
                    const std::shared_ptr<SBSurfaceFluidCouplingManager>& sb_data_manager,
                    MeshBase* mesh,
                    const double time)
{
    auto exact_fcn = [](double t, double l) -> double {
        return std::exp(l * t - std::sin(2.0 * M_PI * t) / (2.0 * M_PI));
    };

    auto J_fcn = [](double t) -> double { return std::exp(-std::sin(2.0 * M_PI * t) / (2.0 * M_PI)); };
    EquationSystems* eq_sys = sb_data_manager->getFEDataManager()->getEquationSystems();
    auto& sf_sys = eq_sys->get_system<TransientExplicitSystem>(sf_name);
    auto& err_sys = eq_sys->get_system<ExplicitSystem>(err_name);
    auto& exa_sys = eq_sys->get_system<ExplicitSystem>(exact_name);
    auto& J_sys = eq_sys->get_system<ExplicitSystem>(sb_data_manager->getJacobianName());
    auto& J_exa_sys = eq_sys->get_system<ExplicitSystem>(J_exa_name);
    auto& J_err_sys = eq_sys->get_system<ExplicitSystem>(J_err_name);

    const DofMap& sf_dof_map = sf_sys.get_dof_map();

    NumericVector<double>* sf_vec = sf_sys.solution.get();
    NumericVector<double>* err_vec = err_sys.solution.get();
    NumericVector<double>* exa_vec = exa_sys.solution.get();
    NumericVector<double>* J_vec = J_sys.solution.get();
    NumericVector<double>* J_err_vec = J_err_sys.solution.get();
    NumericVector<double>* J_exa_vec = J_exa_sys.solution.get();

    MeshBase::const_node_iterator node_it = mesh->active_nodes_begin();
    MeshBase::const_node_iterator node_end = mesh->active_nodes_end();
    for (; node_it != node_end; node_it++)
    {
        const Node* node = *node_it;
        std::vector<dof_id_type> dof_indices;
        sf_dof_map.dof_indices(node, dof_indices);
        for (const auto dof_index : dof_indices)
        {
            exa_vec->set(dof_index, exact_fcn(time, lambda));
            err_vec->set(dof_index, std::abs((*exa_vec)(dof_index) - (*sf_vec)(dof_index) * (*J_vec)(dof_index)));
            J_exa_vec->set(dof_index, J_fcn(time));
            J_err_vec->set(dof_index, std::abs(J_fcn(time) - (*J_vec)(dof_index)));
        }
    }
    J_vec->close();
    sf_vec->close();
    exa_vec->close();
    err_vec->close();
    J_exa_vec->close();
    J_err_vec->close();
}
