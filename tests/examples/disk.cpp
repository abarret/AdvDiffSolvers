#include "ibamr/config.h"

#include "ADS/CutCellVolumeMeshMapping.h"
#include "ADS/LSCutCellLaplaceOperator.h"
#include "ADS/LSFromMesh.h"
#include "ADS/SBAdvDiffIntegrator.h"
#include "ADS/SBBoundaryConditions.h"
#include "ADS/SBIntegrator.h"

#include <ibamr/FESurfaceDistanceEvaluator.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/IBFESurfaceMethod.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/RelaxationLSMethod.h>

#include "ibtk/CartGridFunctionSet.h"
#include "ibtk/PETScKrylovPoissonSolver.h"
#include "ibtk/muParserCartGridFunction.h"
#include "ibtk/muParserRobinBcCoefs.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "tbox/Pointer.h"
#include <ADS/app_namespaces.h>

#include <libmesh/boundary_mesh.h>
#include <libmesh/communicator.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>
#include <utility>

// Local includes
#include "disk/ForcingFcn.h"
#include "disk/QFcn.h"

#include "disk/ForcingFcn.cpp"
#include "disk/QFcn.cpp"

void computeFluidErrors(Pointer<CellVariable<NDIM, double>> Q_var,
                        const int Q_idx,
                        const int Q_error_idx,
                        const int Q_exact_idx,
                        const int vol_idx,
                        const int ls_idx,
                        Pointer<PatchHierarchy<NDIM>> hierarchy,
                        Pointer<QFcn> qfcn,
                        const double time);

void computeSurfaceErrors(const MeshBase& mesh,
                          std::shared_ptr<FEMeshPartitioner> fe_data_manager,
                          const std::string& sys_name,
                          const std::string& err_name,
                          double time);

static double k_on, k_off, sf_max, D_coef;

double
exact_surface(const VectorNd& /*x*/, double t)
{
    double denom = -k_on * (1.0 + t * (1.0 - t)) - k_off;
    return (-2.0 * D_coef * (t - 1.0) * t - sf_max * k_on * (1.0 + t * (1.0 - t))) / denom;
}
double
a_fcn(double Q_bdry, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double time, void* ctx)
{
    return k_on * (sf_max - sf_vals[0]) * Q_bdry;
}

double
g_fcn(double Q_bdry, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double time, void* ctx)
{
    return k_off * sf_vals[0];
}

double
sf_ode(double q, const std::vector<double>& fl_vals, const std::vector<double>& sf_vals, double t, void* ctx)
{
    double ode_val = k_on * (sf_max - q) * fl_vals[0] - k_off * q;
    double denom = k_off + k_on - k_on * (t * (t - 1.0));
    double force = 2.0 * D_coef * t - 2.0 * D_coef * t * t -
                   ((2.0 * t - 1.0) * (2.0 * D_coef * (k_off + k_on) + sf_max * k_off * k_on)) / (denom * denom);
    return ode_val + force;
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
    const LibMeshInit& init = ibtk_init.getLibMeshInit();
    PetscOptionsSetValue(nullptr, "-poisson_solve_ksp_rtol", "1.0e-12");

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");

        // Create a simple FE mesh.
        // Create a simple FE mesh.
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;
        sf_max = input_db->getDouble("SF_MAX");
        k_on = input_db->getDouble("K_ON");
        k_off = input_db->getDouble("K_OFF");
        D_coef = input_db->getDouble("D_COEF");
        VectorNd cent;
        input_db->getDoubleArray("CENTER", cent.data(), NDIM);

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
        Pointer<SBAdvDiffIntegrator> adv_diff_integrator = new SBAdvDiffIntegrator(
            "SBAdvDiffIntegrator", app_initializer->getComponentDatabase("SBAdvDiffIntegrator"), nullptr, false);

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

        // Create Eulerian boundary condition specification objects.
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, static_cast<RobinBcCoefStrategy<NDIM>*>(NULL));
        const bool periodic_domain = grid_geometry->getPeriodicShift().min() > 0;

        // Setup velocity
        Pointer<FaceVariable<NDIM, double>> u_var = new FaceVariable<NDIM, double>("U");
        adv_diff_integrator->registerAdvectionVelocity(u_var);

        Pointer<CartGridFunction> u_fcn =
            new muParserCartGridFunction("UFcn", app_initializer->getComponentDatabase("UFcn"), grid_geometry);
        adv_diff_integrator->setAdvectionVelocityFunction(u_var, u_fcn);

        // Setup boundary mesh mapping
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &reaction_mesh);
        adv_diff_integrator->registerGeneralBoundaryMeshMapping(mesh_mapping);

        // Setup the level set function
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS_In");
        adv_diff_integrator->registerLevelSetVariable(ls_var);

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellVolumeMeshMapping("CutCellMapping",
                                         app_initializer->getComponentDatabase("CutCellMapping"),
                                         mesh_mapping->getMeshPartitioners());
        Pointer<LSFromMesh> vol_fcn = new LSFromMesh("LSFromMesh", patch_hierarchy, cut_cell_mapping, true);
        adv_diff_integrator->registerLevelSetVolFunction(ls_var, vol_fcn);

        // Setup advected quantity
        Pointer<CellVariable<NDIM, double>> Q_in_var = new CellVariable<NDIM, double>("Q_in");
        Pointer<CartGridFunction> Q_in_init = new QFcn("QInit", app_initializer->getComponentDatabase("QInitial"));
        std::vector<RobinBcCoefStrategy<NDIM>*> Q_in_bcs(1);
        if (!periodic_domain)
        {
            const std::string Q_bcs_name = "Q_bcs";
            Q_in_bcs[0] =
                new muParserRobinBcCoefs(Q_bcs_name, app_initializer->getComponentDatabase(Q_bcs_name), grid_geometry);
        }

        adv_diff_integrator->registerTransportedQuantity(Q_in_var);
        adv_diff_integrator->setAdvectionVelocity(Q_in_var, u_var);
        adv_diff_integrator->setInitialConditions(Q_in_var, Q_in_init);
        adv_diff_integrator->setPhysicalBcCoef(Q_in_var, Q_in_bcs[0]);
        adv_diff_integrator->setDiffusionCoefficient(Q_in_var, input_db->getDouble("D_COEF"));
        adv_diff_integrator->restrictToLevelSet(Q_in_var, ls_var);

        auto sb_data_manager =
            std::make_shared<SBSurfaceFluidCouplingManager>("SBDataManager",
                                                            app_initializer->getComponentDatabase("SBDataManager"),
                                                            mesh_mapping->getMeshPartitioners());
        sb_data_manager->registerFluidConcentration(Q_in_var);
        std::string sf_name = "SurfaceConcentration";
        sb_data_manager->registerSurfaceConcentration(sf_name);
        sb_data_manager->registerFluidSurfaceDependence(sf_name, Q_in_var);
        sb_data_manager->registerSurfaceReactionFunction(sf_name, sf_ode, nullptr);
        sb_data_manager->registerFluidBoundaryCondition(Q_in_var, a_fcn, g_fcn, nullptr);
        sb_data_manager->registerInitialConditions(
            sf_name, [](const VectorNd& x, const Node* n) -> double { return exact_surface(x, 0.0); });
        sb_data_manager->initializeFEData();

        Pointer<SBIntegrator> sb_integrator = new SBIntegrator("SBIntegrator", sb_data_manager);
        adv_diff_integrator->registerSBIntegrator(sb_integrator, ls_var);
        adv_diff_integrator->registerLevelSetSBDataManager(ls_var, sb_data_manager);

        // Forcing term
        Pointer<CellVariable<NDIM, double>> F_var = new CellVariable<NDIM, double>("F");
        Pointer<ForcingFcn> forcing_fcn =
            new ForcingFcn("ForcingFcn", app_initializer->getComponentDatabase("ForcingFcn"));
        adv_diff_integrator->registerSourceTerm(F_var);
        adv_diff_integrator->setSourceTerm(Q_in_var, F_var);
        adv_diff_integrator->setSourceTermFunction(F_var, forcing_fcn);

        // Set up diffusion operators
        Pointer<LSCutCellLaplaceOperator> rhs_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInRHSOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        Pointer<LSCutCellLaplaceOperator> sol_in_oper = new LSCutCellLaplaceOperator(
            "LSCutCellInOperator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        // Create boundary operators
        Pointer<SBBoundaryConditions> bdry_conditions = new SBBoundaryConditions(
            "SBBoundaryConditions", sb_data_manager->getFLName(Q_in_var), sb_data_manager, cut_cell_mapping);
        bdry_conditions->setFluidContext(adv_diff_integrator->getCurrentContext());
        rhs_in_oper->setBoundaryConditionOperator(bdry_conditions);
        sol_in_oper->setBoundaryConditionOperator(bdry_conditions);

        adv_diff_integrator->setHelmholtzRHSOperator(Q_in_var, rhs_in_oper);
        Pointer<PETScKrylovPoissonSolver> Q_in_helmholtz_solver = new PETScKrylovPoissonSolver(
            "PoissonSolver", app_initializer->getComponentDatabase("PoissonSolver"), "poisson_solve_");
        Q_in_helmholtz_solver->setOperator(sol_in_oper);
        adv_diff_integrator->setHelmholtzSolver(Q_in_var, Q_in_helmholtz_solver);

        const std::string err_sys_name = "ERROR";
        ExplicitSystem& sys =
            sb_data_manager->getFEMeshPartitioner()->getEquationSystems()->add_system<ExplicitSystem>(err_sys_name);
        sys.add_variable("Error");

        mesh_mapping->initializeEquationSystems();
        sb_data_manager->fillInitialConditions();
        // Initialize hierarchy configuration and data on all patches.
        adv_diff_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Exact and error terms
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int Q_exact_idx = var_db->registerVariableAndContext(Q_in_var, var_db->getContext("Exact"));
        const int Q_error_idx = var_db->registerVariableAndContext(Q_in_var, var_db->getContext("Error"));
        // Allocate exact and error data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(Q_exact_idx);
            level->allocatePatchData(Q_error_idx);
        }

        const int Q_idx = var_db->mapVariableAndContextToIndex(Q_in_var, adv_diff_integrator->getCurrentContext());

        // Close the restart manager.
        RestartManager::getManager()->closeRestartFile();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        const int vol_idx = var_db->mapVariableAndContextToIndex(adv_diff_integrator->getVolumeVariable(ls_var),
                                                                 adv_diff_integrator->getCurrentContext());
        const int ls_idx = var_db->mapVariableAndContextToIndex(ls_var, adv_diff_integrator->getCurrentContext());
        forcing_fcn->setLSIndex(ls_idx, vol_idx);

        double dt = adv_diff_integrator->getMaximumTimeStepSize();

        // Write out initial visualization data.
        int iteration_num = adv_diff_integrator->getIntegratorStep();
        double loop_time = adv_diff_integrator->getIntegratorTime();

        // Main time step loop.
        double loop_time_end = adv_diff_integrator->getEndTime();
        while (!MathUtilities<double>::equalEps(loop_time, loop_time_end) && adv_diff_integrator->stepsRemaining())
        {
            iteration_num = adv_diff_integrator->getIntegratorStep();
            loop_time = adv_diff_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = adv_diff_integrator->getMaximumTimeStepSize();
            adv_diff_integrator->advanceHierarchy(dt);
            loop_time += dt;

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            // At specified intervals, write visualization and restart files,
            // and print out timer data.
            iteration_num += 1;
            pout << "\nWriting visualization files...\n\n";
            computeFluidErrors(
                Q_in_var, Q_idx, Q_error_idx, Q_exact_idx, vol_idx, ls_idx, patch_hierarchy, Q_in_init, loop_time);
            computeSurfaceErrors(reaction_mesh,
                                 sb_data_manager->getFEMeshPartitioner(),
                                 sb_data_manager->getSFNames()[0],
                                 err_sys_name,
                                 loop_time);
        }

        if (!periodic_domain) delete Q_in_bcs[0];
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
computeSurfaceErrors(const MeshBase& mesh,
                     std::shared_ptr<FEMeshPartitioner> fe_data_manager,
                     const std::string& sys_name,
                     const std::string& err_name,
                     double time)
{
    EquationSystems* eq_sys = fe_data_manager->getEquationSystems();
    TransientExplicitSystem& q_system = eq_sys->get_system<TransientExplicitSystem>(sys_name);
    ExplicitSystem& err_sys = eq_sys->get_system<ExplicitSystem>(err_name);
    NumericVector<double>* q_vec = q_system.solution.get();
    NumericVector<double>* err_vec = err_sys.solution.get();

    const DofMap& q_dof_map = q_system.get_dof_map();
    const FEType& q_fe_type = q_dof_map.variable_type(0);

    std::unique_ptr<FEBase> fe = FEBase::build(mesh.mesh_dimension(), q_fe_type);
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
        const int dof_index = n->dof_number(q_system.number(), 0, 0);
        err_vec->set(dof_index, std::abs(exact_surface(x, time) - (*q_vec)(dof_index)));
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
        q_dof_map.dof_indices(el, err_dof_indices);
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

    pout << "Error at surface at time: " << time << "\n";
    pout << " L1-norm:  " << l1_norm << "\n";
    pout << " L2-norm:  " << l2_norm << "\n";
    pout << " max-norm: " << max_norm << "\n";

    err_vec->close();
    q_vec->close();
}

void
computeFluidErrors(Pointer<CellVariable<NDIM, double>> Q_var,
                   const int Q_idx,
                   const int Q_error_idx,
                   const int Q_exact_idx,
                   const int vol_idx,
                   const int ls_idx,
                   Pointer<PatchHierarchy<NDIM>> hierarchy,
                   Pointer<QFcn> qfcn,
                   double time)
{
    qfcn->setLSIndex(ls_idx, vol_idx);
    qfcn->setDataOnPatchHierarchy(Q_exact_idx, Q_var, hierarchy, time, false);
    HierarchyMathOps hier_math_ops("HierarchyMathOps", hierarchy);
    hier_math_ops.setPatchHierarchy(hierarchy);
    hier_math_ops.resetLevels(0, hierarchy->getFinestLevelNumber());
    const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
    HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(hierarchy, 0, hierarchy->getFinestLevelNumber());
    hier_cc_data_ops.subtract(Q_error_idx, Q_exact_idx, Q_idx);
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_cc_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*wgt_data)(idx) = (*vol_data)(idx) > 0.0 ? (*wgt_data)(idx) : 0.0;
            }
        }
    }
    pout << "Error in fluid at time: " << time << "\n";
    pout << "  L1-norm:   " << std::setprecision(10) << hier_cc_data_ops.L1Norm(Q_error_idx, wgt_cc_idx) << "\n";
    pout << "  L2-norm:   " << std::setprecision(10) << hier_cc_data_ops.L2Norm(Q_error_idx, wgt_cc_idx) << "\n";
    pout << "  max-norm:  " << std::setprecision(10) << hier_cc_data_ops.maxNorm(Q_error_idx, wgt_cc_idx) << "\n";
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_cc_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*wgt_data)(idx) = (*vol_data)(idx) < 1.0 ? 0.0 : (*wgt_data)(idx);
            }
        }
    }
    pout << "Error in fluid without cut cells at time: " << time << "\n";
    pout << "  L1-norm:   " << std::setprecision(10) << hier_cc_data_ops.L1Norm(Q_error_idx, wgt_cc_idx) << "\n";
    pout << "  L2-norm:   " << std::setprecision(10) << hier_cc_data_ops.L2Norm(Q_error_idx, wgt_cc_idx) << "\n";
    pout << "  max-norm:  " << std::setprecision(10) << hier_cc_data_ops.maxNorm(Q_error_idx, wgt_cc_idx) << "\n";
}
