// Config files
#include <ibtk/config.h>

#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/PETScAugmentedKrylovLinearSolver.h>
#include <ADS/RBFFDPoissonSolver.h>
#include <ADS/app_namespaces.h>
#include <ADS/solver_utilities.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>

#include "libmesh/boundary_info.h"
#include "libmesh/boundary_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/exact_solution.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/explicit_system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/node.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/point.h"
#include "libmesh/utility.h"

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

static double c, d, L;
static int ierr;

double
exact(const VectorNd& p)
{
#if (NDIM == 2)
    return (c - 32.0 * d * M_PI * M_PI / (L * L)) * std::sin(4.0 * M_PI * p(0) / L) * std::sin(4.0 * M_PI * p(1) / L);
#else
    return (c - 3.0 * d * M_PI * M_PI) * std::sin(M_PI * p(0)) * std::sin(M_PI * p(1)) * std::sin(M_PI * p(2));
#endif
}

double
X(const VectorNd& p)
{
#if (NDIM == 2)
    return std::sin(4.0 * M_PI * p(0) / L) * std::sin(4.0 * M_PI * p(1) / L);
#else
    return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1)) * std::sin(M_PI * p(2));
#endif
}

void
fillSoln(EquationSystems* eq_sys, std::string sys_name, std::function<double(const IBTK::VectorNd& x)> fcn)
{
    auto& sys = eq_sys->get_system<ExplicitSystem>(sys_name);
    const MeshBase& mesh = eq_sys->get_mesh();
    const DofMap& dof_map = sys.get_dof_map();
    NumericVector<double>* vec = sys.solution.get();

    auto iter = mesh.local_nodes_begin();
    const auto iter_end = mesh.local_nodes_end();
    for (; iter != iter_end; ++iter)
    {
        const Node* const node = *iter;
        std::vector<dof_id_type> idx_vec;
        dof_map.dof_indices(node, idx_vec);
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = (*node)(d);
        vec->set(idx_vec[0], fcn(x));
    }
    vec->close();
    sys.update();
}

void
fillGhostCells(Vec& x, const std::shared_ptr<GhostPoints>& ghosts, const std::shared_ptr<GlobalIndexing>& indexing)
{
    const std::map<int, int>& ghost_idx_map = indexing->getGhostMap();
    const std::vector<GhostPoint>& eul_ghosts = ghosts->getEulerianGhostNodes();
    for (const auto& ghost : eul_ghosts)
    {
        ierr = VecSetValue(x, ghost_idx_map.at(ghost.getId()), X(ghost.getX()), INSERT_VALUES);
        IBTK_CHKERRQ(ierr);
    }
    const std::vector<GhostPoint>& lag_ghosts = ghosts->getLagrangianGhostNodes();
    for (const auto& ghost : lag_ghosts)
    {
        ierr = VecSetValue(x, ghost_idx_map.at(ghost.getId()), X(ghost.getX()), INSERT_VALUES);
        IBTK_CHKERRQ(ierr);
    }
}

/*******************************************************************************
 * For each run, the input filename must be given on the command line.  In all *
 * cases, the command line is:                                                 *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "cc_poisson.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        const std::string viz_dirname = app_initializer->getVizDumpDirectory();
        const std::string bdry_dirname = viz_dirname + "/bdry.ex2";
        const std::string vol_dirname = viz_dirname + "/vol.ex2";

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", NULL, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Create variables and register them with the variable database.
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("context");

        Pointer<CellVariable<NDIM, double>> exact_var = new CellVariable<NDIM, double>("exact");
        Pointer<CellVariable<NDIM, double>> x_var = new CellVariable<NDIM, double>("x");
        Pointer<CellVariable<NDIM, double>> b_var = new CellVariable<NDIM, double>("b");
        Pointer<CellVariable<NDIM, double>> error_var = new CellVariable<NDIM, double>("error");
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls");

        const int exact_idx = var_db->registerVariableAndContext(exact_var, ctx, IntVector<NDIM>(3));
        const int x_idx = var_db->registerVariableAndContext(x_var, ctx, IntVector<NDIM>(3));
        const int b_idx = var_db->registerVariableAndContext(b_var, ctx, IntVector<NDIM>(3));
        const int error_idx = var_db->registerVariableAndContext(error_var, ctx, IntVector<NDIM>(3));
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(3));

        // Initialize the AMR patch hierarchy.
        gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
        int tag_buffer = 1;
        int level_number = 0;
        bool done = false;
        while (!done && (gridding_algorithm->levelCanBeRefined(level_number)))
        {
            gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, tag_buffer);
            done = !patch_hierarchy->finerLevelExists(level_number);
            ++level_number;
        }

        // Allocate data on each level of the patch hierarchy.
        pout << "Allocating data\n";
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(exact_idx, 0.0);
            level->allocatePatchData(x_idx, 0.0);
            level->allocatePatchData(b_idx, 0.0);
            level->allocatePatchData(error_idx, 0.0);
            level->allocatePatchData(ls_idx, 0.0);
        }

        // Setup vector objects.
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> exact_eul_vec(
            "exact", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> x_eul_vec("x", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> b_eul_vec("b", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());

        exact_eul_vec.addComponent(exact_var, exact_idx, wgt_idx);
        x_eul_vec.addComponent(x_var, x_idx, wgt_idx);
        b_eul_vec.addComponent(b_var, b_idx, wgt_idx);

        // Setup exact solutions.
        pout << "Setting up solution data\n";
        muParserCartGridFunction exact_fcn(
            "Laplacian", app_initializer->getComponentDatabase("Laplacian"), grid_geometry);
        muParserCartGridFunction x_fcn("Q", app_initializer->getComponentDatabase("Q"), grid_geometry);
        muParserCartGridFunction ls_fcn("ls", app_initializer->getComponentDatabase("ls"), grid_geometry);

        exact_fcn.setDataOnPatchHierarchy(exact_idx, exact_var, patch_hierarchy, 0.0);
        x_fcn.setDataOnPatchHierarchy(x_idx, x_var, patch_hierarchy, 0.0);
        ls_fcn.setDataOnPatchHierarchy(ls_idx, ls_var, patch_hierarchy, 0.0);

        // Set up the finite element mesh
        // Note we use this to create "augmented" dofs.
        pout << "Creating finite element mesh\n";
        const double R = input_db->getDouble("RADIUS");
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;
        L = input_db->getDouble("L");
        c = input_db->getDouble("C");
        d = input_db->getDouble("D");
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
        pout << "Extracting boundary mesh\n";
        BoundaryMesh bdry_mesh(mesh.comm(), mesh.mesh_dimension() - 1);
        BoundaryInfo& bdry_info = mesh.get_boundary_info();
        bdry_info.sync(bdry_mesh);
        bdry_mesh.prepare_for_use();

// Uncomment to output visualization.
#define DRAW_OUTPUT
#ifdef DRAW_OUTPUT
        // Set up visualization
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity(exact_var->getName(), "SCALAR", exact_idx);
        visit_data_writer->registerPlotQuantity(x_var->getName(), "SCALAR", x_idx);
        visit_data_writer->registerPlotQuantity(b_var->getName(), "SCALAR", b_idx);
        visit_data_writer->registerPlotQuantity(error_var->getName(), "SCALAR", error_idx);
        visit_data_writer->registerPlotQuantity(ls_var->getName(), "SCALAR", ls_idx);
        auto bdry_io = libmesh_make_unique<ExodusII_IO>(bdry_mesh);
#endif

        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &bdry_mesh);
        std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner = mesh_mapping->getMeshPartitioner();
        EquationSystems* bdry_eq_sys = fe_mesh_partitioner->getEquationSystems();

        auto& exact_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("exact");
        exact_bdry_sys.add_variable("exact", FIRST);
        auto& x_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("x");
        x_bdry_sys.add_variable("x", FIRST);
        auto& b_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("b");
        b_bdry_sys.add_variable("b", FIRST);

        pout << "Creating solver\n";
        RBFFDPoissonSolver solver("PoissonSolver",
                                  app_initializer->getComponentDatabase("PoissonSolver"),
                                  patch_hierarchy,
                                  fe_mesh_partitioner,
                                  "x",
                                  "b",
                                  "solver_",
                                  PETSC_COMM_WORLD);
        input_db->printClassData(plog);
        mesh_mapping->initializeEquationSystems();
        fe_mesh_partitioner->setPatchHierarchy(patch_hierarchy);
        fe_mesh_partitioner->reinitElementMappings(3);
        pout << "Filling initial condition\n";
        fillSoln(bdry_eq_sys, "x", X);
        fillSoln(bdry_eq_sys, "exact", exact);

        solver.setLSIdx(ls_idx);
        pout << "Initializing solver state\n";
        solver.initializeSolverState(exact_eul_vec, exact_eul_vec);
        // Pull out vector and matrix
        Vec& x_vec = solver.getX();
        Mat& mat = solver.getMat();

        // Copy data to x_vec
        pout << "Copying data to petsc representation\n";
        const std::shared_ptr<GlobalIndexing>& global_indexing = solver.getGlobalIndexing();
        const int eul_map = global_indexing->getEulerianMap();
        const std::map<int, int>& lag_map = global_indexing->getLagrangianMap();
        const std::vector<int> dofs_per_proc = global_indexing->getDofsPerProc();
        copyDataToPetsc(x_vec, x_eul_vec, patch_hierarchy, x_bdry_sys, eul_map, lag_map, dofs_per_proc);
        fillGhostCells(x_vec, solver.getGhostPoints(), global_indexing);

        // Now we can apply the matrix
        pout << "Applying matrix\n";
        Vec b_vec;
        ierr = VecDuplicate(x_vec, &b_vec);
        IBTK_CHKERRQ(ierr);
        ierr = MatMult(mat, x_vec, b_vec);
        IBTK_CHKERRQ(ierr);
        // Now copy data back to original specifications
        pout << "Copying data from petsc representation\n";
        copyDataFromPetsc(b_vec, b_eul_vec, patch_hierarchy, b_bdry_sys, *solver.getBulkConditionMap());

        pout << "Checking errors\n";
        // Compute errors
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(
            patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_idx);
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    // EXCLUDE CUT CELLS
                    const double ls = node_to_cell(idx, *ls_data);
                    if (ls > -app_initializer->getComponentDatabase("PoissonSolver")->getDouble("eps"))
                        (*wgt_data)(idx) = 0.0;
                }
            }
        }
        hier_cc_data_ops.subtract(error_idx, b_idx, exact_idx);
        pout << "Norms of error: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(error_idx, wgt_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(error_idx, wgt_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(error_idx, wgt_idx) << "\n";

        ExactSolution error_estimator(*bdry_eq_sys);
        error_estimator.attach_exact_value(
            [](const libMesh::Point& p, const Parameters&, const std::string&, const std::string&) -> double {
                IBTK::VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = p(d);
                return exact(x);
            });
        error_estimator.compute_error("b", "b");
        double Q_error[3];
        Q_error[0] = error_estimator.l1_error("b", "b");
        Q_error[1] = error_estimator.l2_error("b", "b");
        Q_error[2] = error_estimator.l_inf_error("b", "b");
        pout << "Structure errors:\n"
             << "  L1-norm:  " << Q_error[0] << "\n"
             << "  L2-norm:  " << Q_error[1] << "\n"
             << "  max-norm: " << Q_error[2] << "\n";
        pout << std::setprecision(10) << Q_error[0] << " " << Q_error[1] << " " << Q_error[2] << "\n";
#ifdef DRAW_OUTPUT
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
        bdry_io->write_timestep(bdry_dirname, *bdry_eq_sys, 1, 0.0);
#endif

        // Now deallocate
        solver.deallocateSolverState();
        ierr = VecDestroy(&b_vec);
        IBTK_CHKERRQ(ierr);
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(exact_idx);
            level->deallocatePatchData(x_idx);
            level->deallocatePatchData(b_idx);
            level->deallocatePatchData(error_idx);
            level->deallocatePatchData(ls_idx);
        }
    } // cleanup dynamically allocated objects prior to shutdown
} // main
