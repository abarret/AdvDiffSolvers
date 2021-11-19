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

// Local includes
class InsideLSFcn : IBTK::CartGridFunction
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
        const int Q_lap_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("LAP"));
        const int Q_exa_lap_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("EXA"));

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

        LSCutCellLaplaceOperator cut_cell_oper("CutCellOper", app_initializer->getComponentDatabase("CutCellOper"), 1);
        cut_cell_oper.setLSIndices(ls_idx, ls_var, vol_idx, vol_var, area_idx, area_var, side_idx, side_var);
        SAMRAIVectorReal<NDIM, double> in("IN", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        SAMRAIVectorReal<NDIM, double> out("OUT", patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber());
        in.addComponent(Q_var, Q_idx);
        out.addComponent(Q_var, Q_lap_idx);
        cut_cell_oper.initializeOperatorState(in, out);

#if (0)
        libMesh::UniquePtr<ExodusII_IO> reaction_exodus_io(new ExodusII_IO(*meshes[REACTION_MESH_ID]));
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("area", "SCALAR", area_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idx);
            level->allocatePatchData(vol_idx);
            level->allocatePatchData(area_idx);
            level->allocatePatchData(side_idx);
            level->allocatePatchData(Q_idx);
            level->allocatePatchData(Q_lap_idx);
            level->allocatePatchData(Q_exa_lap_idx);
        }

        Pointer<CutCellMeshMapping> cut_cell_mesh_mapping = new CutCellVolumeMeshMapping(
            "CutCellMeshMapping", app_initializer->getComponentDatabase("CutCellMeshMapping"), fe_data_manager);
        cut_cell_mesh_mapping->initializeObjectState(patch_hierarchy);
        cut_cell_mesh_mapping->generateCutCellMappings();

        Pointer<LSFromMesh> mesh_vol_fcn = new LSFromMesh("MeshVolFcn", patch_hierarchy, cut_cell_mesh_mapping);
        Pointer<LSFromLevelSet> ls_vol_fcn = new LSFromLevelSet("LSVolFcn", patch_hierarchy);
        Pointer<InsideLSFcn> ls_fcn = new InsideLSFcn("LSFcn", R);

        Pointer<LSCartGridFunction> QFcn =
            new muParserCartGridFunction("Q_fcn", app_initializer->getComponentDatabase("QFcn"), grid_geometry);
        Pointer<LSCartGridFunction> lap_Q_fcn =
            new ("Q_lap", app_initializer->getComponentDatabase("QLapFcn"), grid_geometry);

        plog << "Computing errors with prescribed level set.\n";
        Q_fcn->setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, 0.0);
        ls_vol_fcn->registerLSFcn(ls_fcn);
        ls_vol_fcn->updateVolumeAreaSideLS(
            vol_idx, vol_var, area_idx, area_var, side_idx, side_var, ls_idx, ls_var, 0.0);
        cut_cell_oper.apply(in, out);
        HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy);
        HierarchySideDataOpsReal<NDIM, double> hier_sc_data_ops(patch_hierarchy);

        mesh_vol_fcn->updateVolumeAreaSideLS(
            vol_idx, vol_var, area_idx, area_var, side_idx, side_var, ls_idx, ls_var, 0.0);

        // Deallocate data
        for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(ls_idx);
            level->deallocatePatchData(vol_idx);
            level->deallocatePatchData(area_idx);
            level->deallocatePatchData(side_idx);
            level->deallocatePatchData(Q_idx);
            level->deallocatePatchData(Q_lap_idx);
            level->deallocatePatchData(Q_exa_lap_idx);
        }

    } // cleanup dynamically allocated objects prior to shutdown
    return 0;
} // main
