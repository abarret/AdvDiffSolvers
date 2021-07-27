#include <ibamr/config.h>

#include "CCAD/CutCellVolumeMeshMapping.h"
#include "CCAD/LSCartGridFunction.h"
#include "CCAD/LSCutCellLaplaceOperator.h"
#include "CCAD/LSFromLevelSet.h"
#include "CCAD/LSFromMesh.h"

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
#include <CCAD/app_namespaces.h>

#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/transient_system.h>

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

            (*ls_data)(idx) = X.norm() - d_R;
        }
    } // setDataOnPatch

private:
    double d_R = std::numeric_limits<double>::quiet_NaN();
};

enum LS_TYPE
{
    LEVEL_SET,
    MESH,
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
    case MESH:
        return "MESH";
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
    if (strcasecmp(type.c_str(), "MESH") == 0) return MESH;
    if (strcasecmp(type.c_str(), "ENTIRE_DOMAIN") == 0) return ENTIRE_DOMAIN;
    return LS_TYPE::DEFAULT;
}

void fillLSEntireDomain(int ls_idx, int vol_idx, int area_idx, int side_idx, Pointer<PatchHierarchy<NDIM>>);

class QFcn : public LSCartGridFunction
{
public:
    QFcn(std::string object_name, LS_TYPE type, const VectorNd& cent)
        : LSCartGridFunction(std::move(object_name)), d_type(type), d_cent(cent)
    {
    }

    void registerFcn(std::function<double(const VectorNd&, const VectorNd&)> fcn)
    {
        d_fcn = fcn;
    }

    bool isTimeDependent() const override
    {
        return true;
    }

    void setDataOnPatch(int data_idx,
                        Pointer<hier::Variable<NDIM>> /*var*/,
                        Pointer<Patch<NDIM>> patch,
                        const double data_time,
                        const bool initial_time,
                        Pointer<PatchLevel<NDIM>> level) override
    {
        Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
        Q_data->fillAll(0.0);
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

        const hier::Index<NDIM>& idx_low = patch->getBox().lower();
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            if ((*vol_data)(idx) > 0.0)
            {
                VectorNd x = find_cell_centroid(idx, *ls_data);
                for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (x[d] - idx_low[d]);
                (*Q_data)(idx) = d_fcn(x, d_cent);
            }
        }
    }

private:
    LS_TYPE d_type;
    VectorNd d_cent;
    std::function<double(const VectorNd&, const VectorNd&)> d_fcn;
};

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

        // Create a simple FE mesh.
        const double dx = input_db->getDouble("DX");
        const double ds = input_db->getDouble("MFAC") * dx;

        string elem_type = input_db->getString("ELEM_TYPE");
        const int second_order_mesh = (input_db->getString("ELEM_ORDER") == "SECOND");
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
        BoundaryMesh bdry_mesh(solid_mesh.comm(), solid_mesh.mesh_dimension() - 1);
        solid_mesh.boundary_info->sync(bdry_mesh);
        bdry_mesh.set_spatial_dimension(NDIM);
        bdry_mesh.prepare_for_use();

        static const int REACTION_MESH_ID = 0;
        vector<MeshBase*> meshes(1);
        meshes[REACTION_MESH_ID] = &bdry_mesh;

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

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<CellVariable<NDIM, double>> area_var = new CellVariable<NDIM, double>("AREA");
        Pointer<SideVariable<NDIM, double>> side_var = new SideVariable<NDIM, double>("SIDE");

        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(2));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(2));
        const int area_idx = var_db->registerVariableAndContext(area_var, ctx, IntVector<NDIM>(2));
        const int side_idx = var_db->registerVariableAndContext(side_var, ctx, IntVector<NDIM>(2));
        const int Q_idx = var_db->registerVariableAndContext(Q_var, ctx, IntVector<NDIM>(2));
        const int Q_der_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("DER"), IntVector<NDIM>(1));
        const int Q_der_err_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("ERR"));

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
        FEDataManager* fe_data_manager = ib_method_ops->getFEDataManager(REACTION_MESH_ID);

        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
//#define DRAW_DATA
#ifdef DRAW_DATA
        libMesh::UniquePtr<ExodusII_IO> reaction_exodus_io(new ExodusII_IO(*meshes[REACTION_MESH_ID]));
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("area", "SCALAR", area_idx);
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_idx);
        visit_data_writer->registerPlotQuantity("Q_der", "SCALAR", Q_der_idx);
        visit_data_writer->registerPlotQuantity("Q_err", "SCALAR", Q_der_err_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idx);
            level->allocatePatchData(vol_idx);
            level->allocatePatchData(area_idx);
            level->allocatePatchData(side_idx);
            level->allocatePatchData(Q_idx);
            level->allocatePatchData(Q_der_idx);
            level->allocatePatchData(Q_der_err_idx);
        }

        Pointer<CutCellMeshMapping> cut_cell_mesh_mapping = new CutCellVolumeMeshMapping(
            "CutCellMeshMapping", app_initializer->getComponentDatabase("CutCellMeshMapping"), fe_data_manager);
        Pointer<LSFromMesh> mesh_vol_fcn = new LSFromMesh("MeshVolFcn", patch_hierarchy, cut_cell_mesh_mapping);
        Pointer<LSFromLevelSet> ls_vol_fcn = new LSFromLevelSet("LSVolFcn", patch_hierarchy);
        Pointer<InsideLSFcn> ls_fcn = new InsideLSFcn("LSFcn", R);

        LS_TYPE ls_type = string_to_enum(input_db->getString("LS_TYPE"));

        switch (ls_type)
        {
        case LS_TYPE::LEVEL_SET:
            ls_vol_fcn->registerLSFcn(ls_fcn);
            ls_vol_fcn->updateVolumeAreaSideLS(
                vol_idx, vol_var, area_idx, area_var, side_idx, side_var, ls_idx, ls_var, 0.0);
            break;
        case LS_TYPE::MESH:
            cut_cell_mesh_mapping->initializeObjectState(patch_hierarchy);
            cut_cell_mesh_mapping->generateCutCellMappings();
            mesh_vol_fcn->updateVolumeAreaSideLS(
                vol_idx, vol_var, area_idx, area_var, side_idx, side_var, ls_idx, ls_var, 0.0);
            break;
        case LS_TYPE::ENTIRE_DOMAIN:
            fillLSEntireDomain(ls_idx, vol_idx, area_idx, side_idx, patch_hierarchy);
            break;
        case LS_TYPE::DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE");
        }

        // Fill in exact values
        VectorNd cent;
        input_db->getDoubleArray("CENTER", cent.data(), NDIM);
        Pointer<QFcn> qFcn = new QFcn("QFcn", ls_type, cent);
        Pointer<QFcn> derQFcn = new QFcn("QFcn", ls_type, cent);
        std::function<double(const VectorNd&, const VectorNd&)> Q_fcn;
        std::function<double(const VectorNd&, const VectorNd&)> QDer_fcn;
        const double C = input_db->getDouble("C");
        const double D = input_db->getDouble("D");
        switch (ls_type)
        {
        case LEVEL_SET:
        case MESH:
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double {
                return std::exp(-10.0 * (X - cent).squaredNorm());
            };
            QDer_fcn = [C, D](const VectorNd& X, const VectorNd& cent) -> double {
                return std::exp(-10.0 * (X - cent).squaredNorm()) *
                       (C + 40.0 * D *
                                (-1.0 + 10.0 * (X[0] - cent[0]) * (X[0] - cent[0]) +
                                 10.0 * (X[1] - cent[1]) * (X[1] - cent[1])));
            };
            break;
        case ENTIRE_DOMAIN:
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double {
                return std::pow(std::sin(M_PI * X[0]) * std::sin(M_PI * X[1]), 4.0);
            };
            QDer_fcn = [C, D](const VectorNd& X, const VectorNd& cent) -> double {
                return 3.0 * D * M_PI * M_PI * std::pow(std::sin(2.0 * M_PI * X[0]), 2.0) *
                           std::pow(std::sin(M_PI * X[1]), 4.0) +
                       std::pow(std::sin(M_PI * X[0]), 4.0) *
                           ((C - 8.0 * D * M_PI * M_PI) * std::pow(std::sin(M_PI * X[1]), 4.0) +
                            3.0 * D * M_PI * M_PI * std::pow(std::sin(2.0 * M_PI * X[1]), 2.0));
            };
            break;
        case DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE\n");
            break;
        }
        qFcn->setLSIndex(ls_idx, vol_idx);
        qFcn->registerFcn(Q_fcn);
        derQFcn->setLSIndex(ls_idx, vol_idx);
        derQFcn->registerFcn(QDer_fcn);

        qFcn->setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);
        derQFcn->setDataOnPatchHierarchy(Q_der_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);
        derQFcn->setDataOnPatchHierarchy(Q_der_err_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);

        Pointer<LSCutCellLaplaceOperator> oper =
            new LSCutCellLaplaceOperator("Operator", app_initializer->getComponentDatabase("LSCutCellOperator"), false);
        SAMRAIVectorReal<NDIM, double> in("IN", patch_hierarchy, 0, finest_ln),
            out("OUT", patch_hierarchy, 0, finest_ln);
        in.addComponent(Q_var, Q_idx);
        out.addComponent(Q_var, Q_der_idx);
        oper->setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var, side_idx, side_var);
        PoissonSpecifications poisson_spec("poisson_specs");
        poisson_spec.setCConstant(C);
        poisson_spec.setDConstant(D);
        oper->setPoissonSpecifications(poisson_spec);
        oper->initializeOperatorState(in, out);

        oper->apply(in, out);

        // Compute errors
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy, 0, finest_ln);
        Pointer<HierarchyMathOps> hier_math_ops = new HierarchyMathOps("HierarchyMathOps", patch_hierarchy);
        const int wgt_cc_idx = hier_math_ops->getCellWeightPatchDescriptorIndex();
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> wgt_data = patch->getPatchData(wgt_cc_idx);
                Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    // EXCLUDE CUT CELLS
                    (*wgt_data)(idx) *= (*vol_data)(idx) < 1.0 ? 0.0 : 1.0;
                }
            }
        }
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif
        pout << "Norms of solution: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_der_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_der_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_der_idx, wgt_cc_idx) << "\n";

        pout << "Norms of exact: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_der_err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_der_err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_der_err_idx, wgt_cc_idx) << "\n";

        hier_cc_data_ops.subtract(Q_der_err_idx, Q_der_err_idx, Q_der_idx);
        pout << "Norms of error: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_der_err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_der_err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_der_err_idx, wgt_cc_idx) << "\n";
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 1, 0.0);
#endif
        // Deallocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(ls_idx);
            level->deallocatePatchData(vol_idx);
            level->deallocatePatchData(area_idx);
            level->deallocatePatchData(side_idx);
            level->deallocatePatchData(Q_idx);
            level->deallocatePatchData(Q_der_idx);
            level->deallocatePatchData(Q_der_err_idx);
        }
    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main

void
fillLSEntireDomain(int ls_idx, int vol_idx, int area_idx, int side_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(vol_idx);
            Pointer<CellData<NDIM, double>> area_data = patch->getPatchData(area_idx);
            Pointer<SideData<NDIM, double>> side_data = patch->getPatchData(side_idx);

            ls_data->fillAll(-1.0);
            vol_data->fillAll(1.0);
            area_data->fillAll(0.0);
            side_data->fillAll(1.0);
        }
    }
}
