#include <ibamr/config.h>

#include "ADS/CutCellMeshMapping.h"
#include "ADS/LSCartGridFunction.h"
#include "ADS/ls_utilities.h"
#include <ADS/BoundaryReconstructCache.h>
#include <ADS/LSFromMesh.h>
#include <ADS/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include "GriddingAlgorithm.h"
#include "Variable.h"
#include "tbox/Pointer.h"

#include <libmesh/edge_edge2.h>
#include <libmesh/exact_solution.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

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
        Q_data->fillAll(std::numeric_limits<double>::quiet_NaN());

        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);

        const hier::Index<NDIM>& idx_low = patch->getBox().lower();
        Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            if (node_to_cell(idx, *ls_data) < 0.0)
            {
                VectorNd x = find_cell_centroid(idx, *ls_data);
                for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (x[d] - static_cast<double>(idx_low(d)));
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

    // Suppress a warning
    SAMRAI::tbox::Logger::getInstance()->setWarning(false);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "adv_diff.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");
        std::string ex_filename = app_initializer->getVizDumpDirectory() + "/bdry.ex2";

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

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("ls_var");
        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(4));
        const int Q_idx = var_db->registerVariableAndContext(Q_var, ctx, IntVector<NDIM>(4));

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
// #define DRAW_DATA
#ifdef DRAW_DATA
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        visit_data_writer->registerPlotQuantity("ls", "SCALAR", ls_idx);
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idx);
            level->allocatePatchData(Q_idx);
        }

        // Create a mesh
        Mesh solid_mesh(ibtk_init.getLibMeshInit().comm(), NDIM);
        double dx = input_db->getDouble("DX");
        double MFAC = input_db->getDouble("MFAC");
        double ds = MFAC * dx;
        double R = input_db->getDouble("R");
        int r = std::log2(0.25 * 2.0 * M_PI * R / ds);
        MeshTools::Generation::build_sphere(solid_mesh, R, r, Utility::string_to_enum<ElemType>("EDGE2"));
        // Make sure each node is on the physical geometry
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

        auto mesh_mapping = std::make_shared<GeneralBoundaryMeshMapping>(
            "MeshMapping", input_db->getDatabase("MeshMapping"), &bdry_mesh);
        mesh_mapping->initializeEquationSystems();

        // Set up exact, approximate, and error systems
        FESystemManager& fe_sys_manager = mesh_mapping->getSystemManager(0);
        EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
        std::string Q_sys_name = "Q", Q_exa_sys_name = "Q_exa";
        ExplicitSystem& Q_sys = eq_sys->add_system<ExplicitSystem>(Q_sys_name);
        Q_sys.add_variable(Q_sys_name, FEType());
        ExplicitSystem& Q_exa_sys = eq_sys->add_system<ExplicitSystem>(Q_exa_sys_name);
        Q_exa_sys.add_variable(Q_exa_sys_name, FEType());

        mesh_mapping->initializeFEData();

#ifdef DRAW_DATA
        ExodusII_IO exodus_io(eq_sys->get_mesh());
#endif

        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellMeshMapping("CutCellMapping", app_initializer->getComponentDatabase("CutCellMapping"));
        Pointer<LSFromMesh> ls_fcn =
            new LSFromMesh("LSFromMesh", patch_hierarchy, mesh_mapping->getSystemManagers(), cut_cell_mapping);

        LS_TYPE ls_type = string_to_enum(input_db->getString("LS_TYPE"));

        switch (ls_type)
        {
        case LS_TYPE::LEVEL_SET:
            ls_fcn->updateVolumeAreaSideLS(-1, nullptr, -1, nullptr, -1, nullptr, ls_idx, ls_var, 0.0, false);
            break;
        case LS_TYPE::ENTIRE_DOMAIN:
            fillLSEntireDomain(ls_idx, -1, -1, -1, patch_hierarchy);
            break;
        case LS_TYPE::DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE");
        }

        // Fill in exact values
        VectorNd cent;
        input_db->getDoubleArray("CENTER", cent.data(), NDIM);
        Pointer<QFcn> qFcn = new QFcn("QFcn", ls_type, cent);
        std::function<double(const VectorNd&, const VectorNd&)> Q_fcn;

        switch (ls_type)
        {
        case LEVEL_SET:
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double
            { return std::exp(-1.0 * (X - cent).squaredNorm()); };
            break;
        case ENTIRE_DOMAIN:
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double
            { return std::pow(std::sin(M_PI * X[0]) * std::sin(M_PI * X[1]), 4.0); };
            break;
        case DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE\n");
            break;
        }
        qFcn->setLSIndex(ls_idx, -1);
        qFcn->registerFcn(Q_fcn);

        qFcn->setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);

        // Set up reconstruction
        BoundaryReconstructCache reconstruct(ls_idx, patch_hierarchy, mesh_mapping);
        reconstruct.cacheData();

        // Now reconstruct the values on each node
        reconstruct.reconstruct(Q_sys_name, Q_idx);

        // Compute errors
        {
            const std::unique_ptr<BoundaryMesh>& mesh = mesh_mapping->getBoundaryMesh(0);
            NumericVector<double>* Q_vec = Q_sys.current_local_solution.get();
            NumericVector<double>* Q_exa_vec = Q_exa_sys.solution.get();
            const DofMap& Q_dof_map = Q_sys.get_dof_map();

            // First fill in exact solution.
            double max_norm = 0.0;
            auto it = mesh->local_nodes_begin();
            auto it_end = mesh->local_nodes_end();
            for (; it != it_end; ++it)
            {
                const Node* const node = *it;
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = (*node)(d);
                std::vector<dof_id_type> dof;
                Q_dof_map.dof_indices(node, dof);
                Q_exa_vec->set(dof[0], Q_fcn(x, cent));
                max_norm = std::max(max_norm, std::abs((*Q_exa_vec)(dof[0]) - (*Q_vec)(dof[0])));
            }

            // Now compute errors. We need quadrature rules for this.
            std::unique_ptr<FEBase> fe = FEBase::build(mesh->mesh_dimension(), Q_dof_map.variable_type(0));
            std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, mesh->mesh_dimension(), THIRD);
            fe->attach_quadrature_rule(qrule.get());
            const std::vector<std::vector<double>>& phi = fe->get_phi();
            const std::vector<double>& JxW = fe->get_JxW();
            double l1_norm = 0.0, l2_norm = 0.0;
            auto it_e = mesh->local_elements_begin();
            auto it_e_end = mesh->local_elements_end();
            for (; it_e != it_e_end; ++it_e)
            {
                Elem* el = *it_e;
                fe->reinit(el);
                boost::multi_array<double, 1> Q_node, Q_exa_node;
                std::vector<dof_id_type> dof_indices;
                Q_dof_map.dof_indices(el, dof_indices);
                IBTK::get_values_for_interpolation(Q_node, *Q_vec, dof_indices);
                IBTK::get_values_for_interpolation(Q_exa_node, *Q_exa_vec, dof_indices);
                for (unsigned int qp = 0; qp < JxW.size(); ++qp)
                {
                    for (unsigned int n = 0; n < phi.size(); ++n)
                    {
                        double err = std::abs(Q_node[n] - Q_exa_node[n]);
                        l1_norm += err * phi[n][qp] * JxW[qp];
                        l2_norm += std::pow(err * phi[n][qp], 2.0) * JxW[qp];
                    }
                }
            }
            l2_norm = std::sqrt(l2_norm);

            pout << "Error norms:\n";
            pout << " L1-norm:  " << l1_norm << "\n";
            pout << " L2-norm:  " << l2_norm << "\n";
            pout << " max-norm: " << max_norm << "\n";

            Q_exa_vec->close();
            Q_vec->close();
            Q_sys.update();
            Q_exa_sys.update();
        }

#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
        exodus_io.write_timestep(ex_filename, *eq_sys, 1, 0.0);
#endif

        // Deallocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->deallocatePatchData(ls_idx);
            level->deallocatePatchData(Q_idx);
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
            Pointer<NodeData<NDIM, double>> ls_data = ls_idx == -1 ? nullptr : patch->getPatchData(ls_idx);
            Pointer<CellData<NDIM, double>> vol_data = vol_idx == -1 ? nullptr : patch->getPatchData(vol_idx);
            Pointer<CellData<NDIM, double>> area_data = area_idx == -1 ? nullptr : patch->getPatchData(area_idx);
            Pointer<SideData<NDIM, double>> side_data = side_idx == -1 ? nullptr : patch->getPatchData(side_idx);

            if (ls_data) ls_data->fillAll(-1.0);
            if (vol_data) vol_data->fillAll(1.0);
            if (area_data) area_data->fillAll(0.0);
            if (side_data) side_data->fillAll(1.0);
        }
    }
}
