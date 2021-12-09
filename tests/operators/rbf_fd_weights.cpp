#include <ibamr/config.h>

#include "ADS/CutCellVolumeMeshMapping.h"
#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/LSCartGridFunction.h"
#include "ADS/LSCutCellLaplaceOperator.h"
#include "ADS/LSFromLevelSet.h"
#include "ADS/RBFReconstructions.h"
#include "ADS/ZSplineReconstructions.h"
#include "ADS/ls_utilities.h"

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

#include "PoissonSpecifications.h"
#include "Variable.h"
#include "tbox/Pointer.h"
#include <ADS/PolynomialBasis.h>
#include <ADS/RBFFDWeightsCache.h>
#include <ADS/app_namespaces.h>

#include "libmesh/mesh_modification.h"
#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

class InsideLSFcn : public IBTK::CartGridFunction
{
public:
    InsideLSFcn(string object_name, const double R) : CartGridFunction(std::move(object_name)), d_R(R)
    {
#if !defined(NDEBUG)
        TBOX_ASSERT(!d_object_name.empty());
#endif
        return;
    } // InsideLSFcn

    bool isTimeDependent() const override
    {
        return true;
    }

    void setDataOnPatch(const int data_idx,
                        Pointer<hier::Variable<NDIM>> var,
                        Pointer<Patch<NDIM>> patch,
                        const double data_time,
                        const bool initial_time,
                        Pointer<PatchLevel<NDIM>> /*level*/) override
    {
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const hier::Index<NDIM>& idx_low = patch->getBox().lower();

        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(data_idx);
        for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            VectorNd X;
            for (int d = 0; d < NDIM; ++d) X[d] = xlow[d] + dx[d] * static_cast<double>(idx(d) - idx_low(d));

            (*ls_data)(idx) = d_R - X.norm();
        }
    } // setDataOnPatch

private:
    double d_R = std::numeric_limits<double>::quiet_NaN();
};

enum LS_TYPE
{
    LEVEL_SET,
    ENTIRE_DOMAIN,
    DEFAULT = -1
};

std::string
enum_to_string(LS_TYPE type)
{
    switch (type)
    {
    case LEVEL_SET:
        return "LEVEL_SET";
        break;
    case ENTIRE_DOMAIN:
        return "ENTIRE_DOMAIN";
        break;
    case DEFAULT:
    default:
        TBOX_ERROR("UNKNOWN ENUM\n");
        break;
    }
    return "UNKNOWN_TYPE";
}

LS_TYPE
string_to_enum(const std::string& type)
{
    if (strcasecmp(type.c_str(), "LEVEL_SET") == 0) return LEVEL_SET;
    if (strcasecmp(type.c_str(), "ENTIRE_DOMAIN") == 0) return ENTIRE_DOMAIN;
    return LS_TYPE::DEFAULT;
}

void fillLSEntireDomain(int ls_idx, Pointer<PatchHierarchy<NDIM>>);

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
        mesh_mapping->initializeEquationSystems();

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(4));

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

        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
//#define DRAW_DATA
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idx);
        }

        Pointer<InsideLSFcn> ls_fcn = new InsideLSFcn("LSFcn", input_db->getDouble("R"));

        LS_TYPE ls_type = string_to_enum(input_db->getString("LS_TYPE"));

        switch (ls_type)
        {
        case LS_TYPE::LEVEL_SET:
            ls_fcn->setDataOnPatchHierarchy(ls_idx, ls_var, patch_hierarchy, 0.0);
            break;
        case LS_TYPE::ENTIRE_DOMAIN:
            fillLSEntireDomain(ls_idx, patch_hierarchy);
            break;
        case LS_TYPE::DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE");
        }

        // Set up reconstruction
        Pointer<RBFFDWeightsCache> weights_op =
            new RBFFDWeightsCache("RBFWeights",
                                  mesh_mapping->getMeshPartitioner(),
                                  patch_hierarchy,
                                  app_initializer->getComponentDatabase("RBFWeights"));
        auto poly_fcn = [](const std::vector<VectorNd>& pt_vec, int poly_degree) -> MatrixXd {
            return PolynomialBasis::laplacianMonomials(pt_vec, poly_degree);
        };
        bool check_background_grid = input_db->getBool("CHECK_BACKGROUND_GRID_ONLY");
        weights_op->registerPolyFcn(poly_fcn);
        weights_op->setLS(ls_idx);
        weights_op->findRBFFDWeights();
        {
            // Output finite difference weights
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber());
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const std::vector<UPoint>& pts = weights_op->getRBFFDBasePoints(patch);
                for (const auto& pt : pts)
                {
                    if (pt.isNode()) continue;
                    plog << "On point: " << pt << "\n";
                    const std::vector<double>& weights = weights_op->getRBFFDWeights(patch, pt);
                    const std::vector<UPoint>& other_pts = weights_op->getRBFFDPoints(patch, pt);
                    bool print_these = true;
                    if (check_background_grid)
                    {
                        for (const auto other_pt : other_pts)
                        {
                            if (other_pt.isNode()) print_these = false;
                        }
                    }
                    for (size_t i = 0; print_these && i < weights.size(); ++i)
                    {
                        plog << "pt[" << i << "]: " << other_pts[i] << "\n";
                        plog << "With weight: " << weights[i] << "\n";
                    }
                    plog << "\n";
                }
            }
        }
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
#endif
        // Deallocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(ls_idx);
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
