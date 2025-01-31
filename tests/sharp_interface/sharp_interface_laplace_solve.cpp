#include <ibamr/config.h>

#include <ADS/CCSharpInterfaceFACPreconditionerStrategy.h>
#include <ADS/CCSharpInterfaceLaplaceOperator.h>
#include <ADS/CCSharpInterfaceScheduledJacobiSolver.h>
#include <ADS/CutCellMeshMapping.h>
#include <ADS/FullFACPreconditioner.h>
#include <ADS/GeneralBoundaryMeshMapping.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/SharpInterfaceGhostFill.h>
#include <ADS/ads_utilities.h>
#include <ADS/app_namespaces.h>
#include <ADS/sharp_interface_utilities.h>

#include "ibtk/CartGridFunctionSet.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/CCLaplaceOperator.h>
#include <ibtk/CCPoissonSolverManager.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/PETScKrylovLinearSolver.h>
#include <ibtk/PETScKrylovPoissonSolver.h>
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
    return InterfaceType::UNKNOWN;
}

enum class FunctionType
{
    POLYNOMIAL,
    TRIG,
    UNKNOWN
};

std::string
enum_to_string(FunctionType e)
{
    switch (e)
    {
    case FunctionType::POLYNOMIAL:
        return "POLYNOMIAL";
        break;
    case FunctionType::TRIG:
        return "TRIG";
        break;
    default:
        return "UNKNOWN";
        break;
    }
}
namespace ADS
{
template <>
FunctionType
string_to_enum<FunctionType>(const std::string& str)
{
    if (strcasecmp(str.c_str(), "POLYNOMIAL") == 0) return FunctionType::POLYNOMIAL;
    if (strcasecmp(str.c_str(), "TRIG") == 0) return FunctionType::TRIG;
    return FunctionType::UNKNOWN;
}
} // namespace ADS
FunctionType fcn_type;
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

double
x_fcn(double, const VectorNd& x, double)
{
    switch (fcn_type)
    {
    case FunctionType::POLYNOMIAL:
        return 1.0 + x[0] + x[1] + x[1] * x[1] + x[0] * x[0];
    case FunctionType::TRIG:
        return std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
    default:
        TBOX_ERROR("UNKNOWN FCN TYPE\n");
        return 0.0;
    }
}

double
bdry_fcn(const VectorNd& x)
{
    return x_fcn(0.0, x, 0.0);
}

double
lap_fcn(double, const VectorNd& x, double)
{
    switch (fcn_type)
    {
    case FunctionType::POLYNOMIAL:
        return 4.0;
    case FunctionType::TRIG:
        return -8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
    default:
        TBOX_ERROR("UNKNOWN FCN TYPE\n");
        return 0.0;
    }
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
    PetscOptionsSetValue(nullptr, "-solver_ksp_rtol", "1.0e-12");

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
        Pointer<CellVariable<NDIM, double>> X_var = new CellVariable<NDIM, double>("X");
        Pointer<CellVariable<NDIM, double>> err_var = new CellVariable<NDIM, double>("Error");
        Pointer<CellVariable<NDIM, double>> Y_var = new CellVariable<NDIM, double>("Y");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int pt_type_idx = var_db->registerVariableAndContext(pt_type_var, ctx, IntVector<NDIM>(1));
        const int X_idx = var_db->registerVariableAndContext(X_var, ctx, IntVector<NDIM>(1));
        const int err_idx = var_db->registerVariableAndContext(err_var, ctx, IntVector<NDIM>(1));
        const int Y_idx = var_db->registerVariableAndContext(Y_var, ctx, IntVector<NDIM>(1));
        std::set<int> idx_set{ pt_type_idx, X_idx, err_idx, Y_idx };
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
        fcn_type = ADS::string_to_enum<FunctionType>(input_db->getString("FUNCTION_TYPE"));

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

        for (const auto& mesh : meshes) mesh_ptrs.push_back(mesh.get());
        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "mesh_mapping", input_db->getDatabase("MeshMapping"), mesh_ptrs);
        mesh_mapping->initializeEquationSystems();
        mesh_mapping->initializeFEData();
        std::vector<std::unique_ptr<FEToHierarchyMapping>> fe_hier_mappings;
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
        {
            fe_hier_mappings.push_back(
                std::make_unique<FEToHierarchyMapping>("FEToHierarchyMapping_" + std::to_string(part),
                                                       &mesh_mapping->getSystemManager(part),
                                                       nullptr,
                                                       patch_hierarchy->getNumberOfLevels(),
                                                       1 /*ghost_width*/));
            fe_hier_mappings[part]->setPatchHierarchy(patch_hierarchy);
            fe_hier_mappings[part]->reinitElementMappings(1);
        }
        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellMeshMapping("cut_cell_mapping", input_db->getDatabase("CutCellMapping"));

        // Uncomment to draw data.
#define DRAW_DATA 1
#ifdef DRAW_DATA
        std::vector<std::unique_ptr<ExodusII_IO>> struct_writers;
        for (int part = 0; part < mesh_mapping->getNumParts(); ++part)
            struct_writers.push_back(std::make_unique<ExodusII_IO>(*mesh_mapping->getBoundaryMesh(part)));
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("Pt_type", "SCALAR", pt_type_idx);
        visit_data_writer->registerPlotQuantity("X", "SCALAR", X_idx);
        visit_data_writer->registerPlotQuantity("Y", "SCALAR", Y_idx);
        visit_data_writer->registerPlotQuantity("err", "SCALAR", err_idx);
#endif

        PointwiseFunction<PointwiseFunctions::ScalarFcn> X_fcn("X_fcn", x_fcn);
        X_fcn.setDataOnPatchHierarchy(err_idx, err_var, patch_hierarchy, 0.0);
        PointwiseFunction<PointwiseFunctions::ScalarFcn> Y_fcn("Y_fcn", lap_fcn);
        Y_fcn.setDataOnPatchHierarchy(Y_idx, Y_var, patch_hierarchy, 0.0);

        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
        hier_math_ops.resetLevels(coarsest_ln, finest_ln);
        const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();

        SAMRAIVectorReal<NDIM, double> x_vec("X_vec", patch_hierarchy, coarsest_ln, finest_ln),
            y_vec("Y_vec", patch_hierarchy, coarsest_ln, finest_ln);
        x_vec.addComponent(X_var, X_idx, wgt_cc_idx);
        y_vec.addComponent(Y_var, Y_idx, wgt_cc_idx);

        sharp_interface::SharpInterfaceGhostFill ghost_fill(
            "GhostFill", unique_ptr_vec_to_raw_ptr_vec(fe_hier_mappings), cut_cell_mapping);
        Pointer<sharp_interface::CCSharpInterfaceLaplaceOperator> laplace_op =
            new sharp_interface::CCSharpInterfaceLaplaceOperator("LaplaceOp", nullptr, ghost_fill, bdry_fcn, false);

        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            const std::vector<std::vector<sharp_interface::ImagePointData>>& img_pt_vec_vec =
                ghost_fill.getImagePointData(ln);

            int local_patch_num = 0;
            for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(ghost_fill.getIndexPatchIndex());
                Pointer<CellData<NDIM, double>> X_data = patch->getPatchData(X_idx);
                X_data->fillAll(0.0);

                Pointer<CellData<NDIM, double>> Y_data = patch->getPatchData(Y_idx);
                const std::vector<sharp_interface::ImagePointData>& img_data_vec = img_pt_vec_vec[local_patch_num];
                for (const auto& img_data : img_data_vec)
                {
                    (*Y_data)(img_data.d_gp_idx) = 2.0 * bdry_fcn(img_data.d_bp_location);
                }

                Pointer<CellData<NDIM, double>> wgt_cc_data = patch->getPatchData(wgt_cc_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if ((*i_data)(idx) == sharp_interface::INVALID) (*wgt_cc_data)(idx) = 0.0;
                }
            }
        }

        PETScKrylovLinearSolver solver("solver", nullptr, "solver_");
        solver.setOperator(laplace_op);
        solver.setHomogeneousBc(false);
        solver.setSolutionTime(0.0);
        PoissonSpecifications poisson_spec("poisson_spec");
        poisson_spec.setCConstant(0.0);
        poisson_spec.setDConstant(1.0);
        laplace_op->setPoissonSpecifications(poisson_spec);

        std::string precond_type = input_db->getString("PRECOND_TYPE");
        if (strcasecmp(precond_type.c_str(), "FAC") == 0)
        {
            Pointer<sharp_interface::CCSharpInterfaceFACPreconditionerStrategy> fac_strategy =
                new sharp_interface::CCSharpInterfaceFACPreconditionerStrategy(
                    "FACStrategy", mesh_mapping->getSystemManagers(), input_db->getDatabase("PreconditionerOp"));
            Pointer<FullFACPreconditioner> fac_precond = new FullFACPreconditioner(
                "FACPreconditioner", fac_strategy, input_db->getDatabase("FACPreconditioner"), "");

            solver.setPreconditioner(fac_precond);
        }
        else if (strcasecmp(precond_type.c_str(), "SRJ") == 0)
        {
            Pointer<sharp_interface::CCSharpInterfaceScheduledJacobiSolver> srj_solver =
                new sharp_interface::CCSharpInterfaceScheduledJacobiSolver(
                    "SharpInterfaceSolver", app_initializer->getComponentDatabase("SRJ_Solver"), ghost_fill, bdry_fcn);
            solver.setPreconditioner(srj_solver);
        }
        else
        {
            // intentionally blank
        }

        solver.initializeSolverState(x_vec, y_vec);

        solver.solveSystem(x_vec, y_vec);

        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);

            int local_patch_num = 0;
            for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(ghost_fill.getIndexPatchIndex());
                Pointer<CellData<NDIM, double>> X_data = patch->getPatchData(X_idx);
                Pointer<CellData<NDIM, double>> err_data = patch->getPatchData(err_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    if ((*i_data)(idx) == sharp_interface::INVALID)
                    {
                        (*X_data)(idx) = 0.0;
                        (*err_data)(idx) = 0.0;
                    }
                }
            }
        }

        // Compute error
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
        hier_cc_data_ops.subtract(err_idx, err_idx, X_idx);
        pout << "Error in X:\n"
             << "  L1-norm:  " << std::setprecision(10) << hier_cc_data_ops.L1Norm(err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << std::setprecision(10) << hier_cc_data_ops.L2Norm(err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << std::setprecision(10) << hier_cc_data_ops.maxNorm(err_idx, wgt_cc_idx) << "\n";

        HierarchyCellDataOpsReal<NDIM, int> hier_cc_int_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
        hier_cc_int_data_ops.copyData(pt_type_idx, ghost_fill.getIndexPatchIndex());

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
