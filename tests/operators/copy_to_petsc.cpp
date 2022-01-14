// Config files
#include <ibtk/config.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>

#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/PETScAugmentedKrylovLinearSolver.h>
#include <ADS/RBFPoissonSolver.h>
#include <ADS/app_namespaces.h>
#include <ADS/solver_utilities.h>

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

double
exact(const VectorNd& p)
{
#if (NDIM == 2)
    return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1));
#else
    return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1)) * std::sin(M_PI * p(2));
#endif
}

void
fillExact(EquationSystems* eq_sys, std::string sys_name)
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
        vec->set(idx_vec[0], exact(x));
    }
    vec->close();
    sys.update();
}

void fillEulQ(Pointer<PatchHierarchy<NDIM>> patch_hierarchy, int q_idx, int ei = -1);
void fillLagQ(Vec& vec, int ei = -1);

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
        Pointer<CellVariable<NDIM, double>> copied_var = new CellVariable<NDIM, double>("copied");
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls");

        const int exact_idx = var_db->registerVariableAndContext(exact_var, ctx, IntVector<NDIM>(3));
        const int copied_idx = var_db->registerVariableAndContext(copied_var, ctx, IntVector<NDIM>(3));
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
            level->allocatePatchData(copied_idx, 0.0);
            level->allocatePatchData(ls_idx, 0.0);
        }

        // Setup vector objects.
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> copied_eul_vec(
            "copied", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> exact_eul_vec(
            "exact", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());

        exact_eul_vec.addComponent(exact_var, exact_idx, wgt_idx);
        copied_eul_vec.addComponent(copied_var, copied_idx, wgt_idx);

        // Setup exact solutions.
        pout << "Setting up solution data\n";
        muParserCartGridFunction exact_fcn("Q", app_initializer->getComponentDatabase("Q"), grid_geometry);
        muParserCartGridFunction ls_fcn("ls", app_initializer->getComponentDatabase("ls"), grid_geometry);

        exact_fcn.setDataOnPatchHierarchy(exact_idx, exact_var, patch_hierarchy, 0.0);
        ls_fcn.setDataOnPatchHierarchy(ls_idx, ls_var, patch_hierarchy, 0.0);

        // Set up the finite element mesh
        // Note we use this to create "augmented" dofs.
        pout << "Creating finite element mesh\n";
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
        pout << "Extracting boundary mesh\n";
        BoundaryMesh bdry_mesh(mesh.comm(), mesh.mesh_dimension() - 1);
        BoundaryInfo bdry_info = mesh.get_boundary_info();
        bdry_info.sync(bdry_mesh);
        bdry_mesh.prepare_for_use();

        // Set up system for drawing
        auto bdry_io = libmesh_make_unique<ExodusII_IO>(bdry_mesh);

        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &bdry_mesh);
        std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner = mesh_mapping->getMeshPartitioner();
        EquationSystems* bdry_eq_sys = fe_mesh_partitioner->getEquationSystems();

        auto& exact_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("exact");
        exact_bdry_sys.add_variable("exact", FIRST);
        auto& copied_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("copied");
        copied_bdry_sys.add_variable("copied", FIRST);
        mesh_mapping->initializeEquationSystems();

        pout << "Filling initial condition\n";
        fillExact(bdry_eq_sys, "exact");

        pout << "Creating solver\n";
        Pointer<RBFPoissonSolver> solver = new RBFPoissonSolver("PoissonSolver",
                                                                app_initializer->getComponentDatabase("PoissonSolver"),
                                                                patch_hierarchy,
                                                                fe_mesh_partitioner,
                                                                "exact",
                                                                "exact",
                                                                "solver_",
                                                                PETSC_COMM_WORLD);
        solver->setLSIdx(ls_idx);
        solver->initializeSolverState(exact_eul_vec, exact_eul_vec);
        // Pull out vector
        Vec& x_vec = solver->getX();
        const int eul_map = solver->getEulerianMap();
        const std::map<int, int>& lag_map = solver->getLagrangianMap();
        const std::vector<int> dofs_per_proc = solver->getDofsPerProc();
        // Now copy data to x_vec
        copyDataToPetsc(x_vec, exact_eul_vec, patch_hierarchy, exact_bdry_sys, eul_map, lag_map, dofs_per_proc);
        // Now copy data back
        copyDataFromPetsc(x_vec, copied_eul_vec, patch_hierarchy, copied_bdry_sys, eul_map, lag_map, dofs_per_proc);

        pout << "Checking copied data\n";
        // Only check finest level
        Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber());
        int eul_wrong = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> exact_data = patch->getPatchData(exact_idx),
                                            copied_data = patch->getPatchData(copied_idx);
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if ((*exact_data)(idx) != (*copied_data)(idx))
                {
                    pout << "On patch index: " << idx << "\n";
                    pout << "Exact data:  " << (*exact_data)(idx) << "\n";
                    pout << "Copied data: " << (*copied_data)(idx) << "\n";
                    eul_wrong++;
                }
            }
        }

        NumericVector<double>* copied_lag_vec = copied_bdry_sys.solution.get();
        NumericVector<double>* exact_lag_vec = exact_bdry_sys.solution.get();
        DofMap& dof_map = copied_bdry_sys.get_dof_map();
        auto node_it = bdry_mesh.local_nodes_begin();
        const auto node_end = bdry_mesh.local_nodes_end();
        int lag_wrong = 0;
        for (; node_it != node_end; ++node_it)
        {
            const Node* node = *node_it;
            std::vector<dof_id_type> dofs;
            dof_map.dof_indices(node, dofs);
            for (const auto& dof : dofs)
            {
                if ((*copied_lag_vec)(dof) != (*exact_lag_vec)(dof))
                {
                    pout << "On Node: " << *node << "\n";
                    pout << "dof:     " << dof << "\n";
                    pout << "Exact data:  " << (*exact_lag_vec)(dof) << "\n";
                    pout << "Copied data: " << (*copied_lag_vec)(dof) << "\n";
                    lag_wrong++;
                }
            }
        }
        solver->deallocateSolverState();

        eul_wrong = IBTK_MPI::sumReduction(eul_wrong);
        lag_wrong = IBTK_MPI::sumReduction(lag_wrong);
        pout << "There are " << eul_wrong << " Eulerian indices wrong\n";
        pout << "There are " << lag_wrong << " Lagrangian indices wrong\n";
        pout << "There are " << eul_wrong + lag_wrong << " total indices wrong\n";

    } // cleanup dynamically allocated objects prior to shutdown
} // main
