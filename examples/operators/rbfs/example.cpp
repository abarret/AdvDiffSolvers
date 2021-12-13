// Config files
#include <ibtk/config.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/CCLaplaceOperator.h>
#include <ibtk/CCPoissonSolverManager.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>

#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/RBFLaplaceOperator.h>
#include <ADS/app_namespaces.h>

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

static double c = 0.0;
static double d = 0.0;

double
initial(const VectorNd& p)
{
#if (NDIM == 2)
    return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1));
#else
    return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1)) * std::sin(M_PI * p(2));
#endif
}

double
laplacian(const VectorNd& p)
{
#if (NDIM == 2)
    return (c + d * 2.0 * M_PI * M_PI) * sin(M_PI * p(0)) * std::sin(M_PI * p(1));
#else
    return (c + d * 3.0 * M_PI * M_PI) * sin(M_PI * p(0)) * std::sin(M_PI * p(1)) * std::sin(M_PI * p(2));
#endif
}

double
initialSol(const libMesh::Point& p,
           const Parameters& /*Parameters*/,
           const std::string& /*sys_name*/,
           const std::string& /*unknown_name*/)
{
    VectorNd x;
    for (int d = 0; d < NDIM; ++d) x[d] = p(d);
    return initial(x);
}

double
laplacianSol(const libMesh::Point& p,
             const Parameters& /*Parameters*/,
             const std::string& /*sys_name*/,
             const std::string& /*unknown_name*/)
{
    VectorNd x;
    for (int d = 0; d < NDIM; ++d) x[d] = p(d);
    return laplacian(x);
}

void
fillLaplacian(EquationSystems* eq_sys, std::string sys_name)
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
        vec->set(idx_vec[0], laplacian(x));
    }
    vec->close();
    sys.update();
}

void
fillInitial(EquationSystems* eq_sys, std::string sys_name)
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
        vec->set(idx_vec[0], initial(x));
    }
    vec->close();
    sys.update();
}

void printMatrix(Pointer<PatchHierarchy<NDIM>> patch_hierarchy, const int r_idx, Vec& vec);
void fillEulQ(Pointer<PatchHierarchy<NDIM>> patch_hierarchy, int q_idx, int ei = -1);
void fillLagQ(Vec& vec, int ei = -1);

static std::ofstream mat_file;

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
        input_db->printClassData(plog);

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

        Pointer<CellVariable<NDIM, double>> q_var = new CellVariable<NDIM, double>("q");
        Pointer<CellVariable<NDIM, double>> b_var = new CellVariable<NDIM, double>("b");
        Pointer<CellVariable<NDIM, double>> e_var = new CellVariable<NDIM, double>("e");
        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls");

        const int q_idx = var_db->registerVariableAndContext(q_var, ctx, IntVector<NDIM>(3));
        const int b_idx = var_db->registerVariableAndContext(b_var, ctx, IntVector<NDIM>(3));
        const int e_idx = var_db->registerVariableAndContext(e_var, ctx, IntVector<NDIM>(3));
        const int masked_e_idx =
            var_db->registerVariableAndContext(e_var, var_db->getContext("mask"), IntVector<NDIM>(0));
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(3));

        // Register variables for plotting.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        TBOX_ASSERT(visit_data_writer);

        visit_data_writer->registerPlotQuantity(q_var->getName(), "SCALAR", q_idx);
        visit_data_writer->registerPlotQuantity(b_var->getName(), "SCALAR", b_idx);
        visit_data_writer->registerPlotQuantity(e_var->getName(), "SCALAR", e_idx);
        visit_data_writer->registerPlotQuantity("masked e", "SCALAR", masked_e_idx);
        visit_data_writer->registerPlotQuantity(ls_var->getName(), "SCALAR", ls_idx);

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
            level->allocatePatchData(q_idx, 0.0);
            level->allocatePatchData(b_idx, 0.0);
            level->allocatePatchData(e_idx, 0.0);
            level->allocatePatchData(masked_e_idx, 0.0);
            level->allocatePatchData(ls_idx, 0.0);
        }

        // Setup vector objects.
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        const int wgt_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> q_vec("q", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> b_vec("b", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> e_vec("e", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());

        q_vec.addComponent(q_var, q_idx, wgt_idx);
        b_vec.addComponent(b_var, b_idx, wgt_idx);
        e_vec.addComponent(e_var, e_idx, wgt_idx);

        // Setup exact solutions.
        pout << "Setting up solution data\n";
        muParserCartGridFunction b_fcn("b", app_initializer->getComponentDatabase("B"), grid_geometry);
        muParserCartGridFunction q_fcn("Q", app_initializer->getComponentDatabase("Q"), grid_geometry);
        muParserCartGridFunction ls_fcn("ls", app_initializer->getComponentDatabase("ls"), grid_geometry);
        muParserCartGridFunction exact_fcn("exact", app_initializer->getComponentDatabase("Exact"), grid_geometry);

        c = input_db->getDouble("C");
        d = input_db->getDouble("D");

        //        b_fcn.setDataOnPatchHierarchy(b_idx, b_var, patch_hierarchy, 0.0);
        ls_fcn.setDataOnPatchHierarchy(ls_idx, ls_var, patch_hierarchy, 0.0);
        exact_fcn.setDataOnPatchHierarchy(e_idx, e_var, patch_hierarchy, 0.0);
        q_fcn.setDataOnPatchHierarchy(q_idx, q_var, patch_hierarchy, 0.0);

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

        // Set up dummy systems for drawing
        auto vol_io = libmesh_make_unique<ExodusII_IO>(mesh);
        auto bdry_io = libmesh_make_unique<ExodusII_IO>(bdry_mesh);

        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", app_initializer->getComponentDatabase("MeshMapping"), &bdry_mesh);
        std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner = mesh_mapping->getMeshPartitioner();
        EquationSystems* bdry_eq_sys = fe_mesh_partitioner->getEquationSystems();
        EquationSystems vol_eq_sys(mesh);

        auto& b_vol_sys = vol_eq_sys.add_system<ExplicitSystem>("b");
        auto& b_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("b");
        b_vol_sys.add_variable("b", FIRST);
        b_bdry_sys.add_variable("b", FIRST);
        auto& q_vol_sys = vol_eq_sys.add_system<ExplicitSystem>("q");
        auto& q_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("q");
        q_vol_sys.add_variable("q", FIRST);
        q_bdry_sys.add_variable("q", FIRST);
        auto& ex_vol_sys = vol_eq_sys.add_system<ExplicitSystem>("exact");
        auto& ex_bdry_sys = bdry_eq_sys->add_system<ExplicitSystem>("exact");
        ex_vol_sys.add_variable("exact", FIRST);
        ex_bdry_sys.add_variable("exact", FIRST);
        vol_eq_sys.init();
        mesh_mapping->initializeEquationSystems();

        pout << "Filling initial condition\n";
        fillLaplacian(bdry_eq_sys, "exact");
        fillInitial(bdry_eq_sys, "q");

        // Compute errors
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Box<NDIM>& box = patch->getBox();
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_idx);
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double ls = ADS::node_to_cell(idx, *ls_data);
                    (*wgt_data)(idx) =
                        ls < -app_initializer->getComponentDatabase("LaplaceOperator")->getDouble("eps") ? 1.0 : 0.0;
                }
            }
        }

        // Set up the augmented Vec
        // Use libMesh's PetscVector. Create a vector with n_nodes elements.
        // Note we will have to be careful about parallel implementation!
        plog << "Creating PETSc Vector\n";
        auto b_petsc_vec = static_cast<PetscVector<double>*>(b_bdry_sys.solution.get());
        auto q_petsc_vec = static_cast<PetscVector<double>*>(q_bdry_sys.solution.get());
        // Clone the underlying vector for augmented operator
        Vec q_cloned;
        int ierr = VecDuplicate(q_petsc_vec->vec(), &q_cloned);
        IBTK_CHKERRQ(ierr);
        ierr = VecCopy(q_petsc_vec->vec(), q_cloned);
        IBTK_CHKERRQ(ierr);
        const DofMap& dof_map = b_bdry_sys.get_dof_map();

        pout << "Creating operator\n";
        // Set up CartLaplaceOperator
        Pointer<RBFLaplaceOperator> lap_op =
            new RBFLaplaceOperator("LapOp",
                                   fe_mesh_partitioner,
                                   "q",
                                   patch_hierarchy,
                                   app_initializer->getComponentDatabase("LaplaceOperator"));
        lap_op->setLS(ls_idx);

        pout << "Initializing operator:\n";
        lap_op->setAugmentedVec(q_cloned);
        lap_op->initializeOperatorState(q_vec, b_vec);
        pout << "Applying operator\n";
        lap_op->apply(q_vec, b_vec);
        // Set up solver
        Vec b_out_vec = lap_op->getAugmentedVec();

        pout << "Copying solution data back to vector\n";
        // Copy data to Q vector
        double* b_out_arr;
        ierr = VecGetArray(b_out_vec, &b_out_arr);
        IBTK_CHKERRQ(ierr);
        auto iter = bdry_mesh.local_nodes_begin();
        const auto iter_end = bdry_mesh.local_nodes_end();
        for (; iter != iter_end; ++iter)
        {
            const Node* const node = *iter;
            std::vector<dof_id_type> dofs;
            dof_map.dof_indices(node, dofs);
            b_petsc_vec->set(dofs[0], b_out_arr[dofs[0]]);
        }
        b_petsc_vec->close();
        b_bdry_sys.update();
        VecRestoreArray(b_out_vec, &b_out_arr);

        // Compute errors
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                const double* const dx = pgeom->getDx();
                const Box<NDIM>& box = patch->getBox();
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_idx);
                for (CellIterator<NDIM> ci(box); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double ls = ADS::node_to_cell(idx, *ls_data);
                    (*wgt_data)(idx) =
                        ls < -app_initializer->getComponentDatabase("LaplaceOperator")->getDouble("eps") ?
#if (NDIM == 2)
                            dx[0] *
                                dx[1]
#endif
#if (NDIM == 3)
                                dx[0] *
                                dx[1] * dx[2]
#endif
                            :
                            0.0;
                }
            }
        }

        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(
            patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        hier_cc_data_ops.subtract(e_idx, e_idx, b_idx);
        pout << "Error in q:\n"
             << "  L1-norm:  " << std::setprecision(10) << hier_cc_data_ops.L1Norm(e_idx, wgt_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(e_idx, wgt_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(e_idx, wgt_idx) << "\n";
        pout << std::setprecision(10) << hier_cc_data_ops.L1Norm(e_idx, wgt_idx) << " "
             << hier_cc_data_ops.L2Norm(e_idx, wgt_idx) << " " << hier_cc_data_ops.maxNorm(e_idx, wgt_idx) << "\n";

        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber());
            double eps = app_initializer->getComponentDatabase("LaplaceOperator")->getDouble("eps");
            // Mask the error
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
                Pointer<CellData<NDIM, double>> mask_data = patch->getPatchData(masked_e_idx);
                Pointer<CellData<NDIM, double>> e_data = patch->getPatchData(e_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double ls = ADS::node_to_cell(idx, *ls_data);
                    (*mask_data)(idx) = (*e_data)(idx) * (ls < -eps ? 1.0 : 0.0);
                }
            }
        }

        ExactSolution error_estimator(*bdry_eq_sys);
        error_estimator.attach_exact_value(laplacianSol);
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
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
        vol_io->write_timestep(vol_dirname, vol_eq_sys, 1, 0.0);
        bdry_io->write_timestep(bdry_dirname, *bdry_eq_sys, 1, 0.0);

        // If necessary, write matrix to file
#define WRITE_MATRIX 0
#if WRITE_MATRIX
        {
            Vec vec = nullptr, vec_out = nullptr;
            int ierr = VecDuplicate(b_cloned, &vec);
            IBTK_CHKERRQ(ierr);
            int eul_dof = 0, lag_dof = 0;
            for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
                for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    const Box<NDIM>& box = patch->getBox();
                    for (CellIterator<NDIM> ci(box); ci; ci++)
                    {
                        ++eul_dof;
                    }
                }
            }
            ierr = VecGetSize(vec, &lag_dof);
            pout << "Eul dofs: " << eul_dof << "\n";
            pout << "Lag dofs: " << lag_dof << "\n";
            mat_file.open("matrix_vals");
            // Loop through dofs and apply laplacian
            for (int dof = 0; dof < eul_dof; ++dof)
            {
                pout << "Printing eul dof: " << dof << "\n";
                lap_op->setAugmentedVec(vec);
                fillLagQ(vec, -1);
                fillEulQ(patch_hierarchy, q_idx, dof);
                b_vec.setToScalar(0.0);
                lap_op->initializeOperatorState(q_vec, b_vec);
                lap_op->apply(q_vec, b_vec);
                vec_out = lap_op->getAugmentedVec();
                lap_op->deallocateOperatorState();
                printMatrix(patch_hierarchy, b_idx, vec_out);
            }
            // Now lag_dofs
            for (int dof = 0; dof < lag_dof; ++dof)
            {
                pout << "Printing lag dof: " << dof << "\n";
                fillLagQ(vec, dof);
                fillEulQ(patch_hierarchy, q_idx, -1);
                lap_op->setAugmentedVec(vec);
                b_vec.setToScalar(0.0);
                lap_op->initializeOperatorState(q_vec, b_vec);
                lap_op->apply(q_vec, b_vec);
                vec_out = lap_op->getAugmentedVec();
                lap_op->deallocateOperatorState();
                printMatrix(patch_hierarchy, b_idx, vec_out);
            }
            ierr = VecDestroy(&vec);
        }
#endif

    } // cleanup dynamically allocated objects prior to shutdown
} // main

void
printMatrix(Pointer<PatchHierarchy<NDIM>> patch_hierarchy, const int r_idx, Vec& vec)
{
    // First Eul dofs
    for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CellData<NDIM, double>> r_data = patch->getPatchData(r_idx);
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = *ci;
                double val = (*r_data)(idx);
                mat_file << std::to_string(val) << " ";
            }
        }
    }
    // Now loop through lag dofs
    double* vals;
    int size;
    int ierr = VecGetArray(vec, &vals);
    IBTK_CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec, &size);
    for (int i = 0; i < size; ++i)
    {
        mat_file << std::to_string(vals[i]) << " ";
    }
    ierr = VecRestoreArray(vec, &vals);
    mat_file << "\n";
    return;
}

void
fillEulQ(Pointer<PatchHierarchy<NDIM>> patch_hierarchy, const int q_idx, const int ei)
{
    int i = 0;
    for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            Pointer<CellData<NDIM, double>> q_data = patch->getPatchData(q_idx);
            q_data->fillAll(0.0);
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                if (i == ei) (*q_data)(idx) = 1.0;
                i++;
            }
        }
    }
    return;
}

void
fillLagQ(Vec& vec, const int ei)
{
    int ierr = VecZeroEntries(vec);
    IBTK_CHKERRQ(ierr);
    if (ei > 0)
    {
        ierr = VecSetValue(vec, ei, 1.0, INSERT_VALUES);
        IBTK_CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(vec);
    IBTK_CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vec);
    IBTK_CHKERRQ(ierr);
    return;
}
