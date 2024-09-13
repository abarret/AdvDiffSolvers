#include <ibamr/config.h>

#include "ADS/LSCartGridFunction.h"
#include "ADS/LSCutCellLaplaceOperator.h"
#include "ADS/LSFromLevelSet.h"
#include "ADS/RBFReconstructions.h"
#include "ADS/ZSplineReconstructions.h"
#include "ADS/ls_utilities.h"
#include <ADS/app_namespaces.h>

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
    QFcn(std::string object_name, LS_TYPE type, const VectorNd& cent, int path_idx = IBTK::invalid_index)
        : LSCartGridFunction(std::move(object_name)), d_type(type), d_cent(cent), d_path_idx(path_idx)
    {
    }

    void registerFcn(std::function<double(const VectorNd&, const VectorNd&)> fcn)
    {
        d_fcn = fcn;
    }

    void setPathIndex(int path_idx)
    {
        d_path_idx = path_idx;
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
        Pointer<CellData<NDIM, double>> path_data = d_path_idx == -1 ? nullptr : patch->getPatchData(d_path_idx);
        if (Q_data->getDepth() == NDIM)
        {
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                // Set path data to be lower box corner
                for (int d = 0; d < NDIM; ++d) (*Q_data)(idx, d) = static_cast<double>(idx(d) - idx_low(d));
            }
        }
        else if (path_data)
        {
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                VectorNd x;
                for (int d = 0; d < NDIM; ++d)
                    x[d] = xlow[d] + dx[d] * ((*path_data)(idx, d) - static_cast<double>(idx_low(d)));
                (*Q_data)(idx) = d_fcn(x, d_cent);
            }
        }
        else
        {
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
                    for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (x[d] - static_cast<double>(idx_low(d)));
                    (*Q_data)(idx) = d_fcn(x, d_cent);
                }
            }
        }
    }

private:
    LS_TYPE d_type;
    VectorNd d_cent;
    std::function<double(const VectorNd&, const VectorNd&)> d_fcn;
    int d_path_idx;
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
        Pointer<CellVariable<NDIM, double>> vol_var = new CellVariable<NDIM, double>("VOL");
        Pointer<CellVariable<NDIM, double>> area_var = new CellVariable<NDIM, double>("AREA");

        Pointer<CellVariable<NDIM, double>> Q_var = new CellVariable<NDIM, double>("Q");
        Pointer<CellVariable<NDIM, double>> path_var = new CellVariable<NDIM, double>("Path", NDIM);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<VariableContext> ctx = var_db->getContext("CTX");
        const int ls_idx = var_db->registerVariableAndContext(ls_var, ctx, IntVector<NDIM>(4));
        const int vol_idx = var_db->registerVariableAndContext(vol_var, ctx, IntVector<NDIM>(4));
        const int area_idx = var_db->registerVariableAndContext(area_var, ctx, IntVector<NDIM>(4));
        const int Q_idx = var_db->registerVariableAndContext(Q_var, ctx, IntVector<NDIM>(4));
        const int Q_app_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("APP"), IntVector<NDIM>(1));
        const int Q_err_idx = var_db->registerVariableAndContext(Q_var, var_db->getContext("ERR"));
        const int path_idx = var_db->registerVariableAndContext(path_var, ctx);

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
        visit_data_writer->registerPlotQuantity("vol", "SCALAR", vol_idx);
        visit_data_writer->registerPlotQuantity("area", "SCALAR", area_idx);
        visit_data_writer->registerPlotQuantity("Q", "SCALAR", Q_idx);
        visit_data_writer->registerPlotQuantity("Q_der", "SCALAR", Q_app_idx);
        visit_data_writer->registerPlotQuantity("Q_err", "SCALAR", Q_err_idx);
        visit_data_writer->registerPlotQuantity("Path", "VECTOR", path_idx);
#endif

        // Allocate data
        for (int ln = 0; ln <= finest_ln; ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(ls_idx);
            level->allocatePatchData(vol_idx);
            level->allocatePatchData(area_idx);
            level->allocatePatchData(Q_idx);
            level->allocatePatchData(Q_app_idx);
            level->allocatePatchData(Q_err_idx);
            level->allocatePatchData(path_idx);
        }

        Pointer<LSFromLevelSet> ls_vol_fcn = new LSFromLevelSet("LSVolFcn", patch_hierarchy);
        Pointer<InsideLSFcn> ls_fcn = new InsideLSFcn("LSFcn", input_db->getDouble("R"));

        LS_TYPE ls_type = string_to_enum(input_db->getString("LS_TYPE"));

        switch (ls_type)
        {
        case LS_TYPE::LEVEL_SET:
            ls_vol_fcn->registerLSFcn(ls_fcn);
            ls_vol_fcn->updateVolumeAreaSideLS(vol_idx, vol_var, area_idx, area_var, -1, nullptr, ls_idx, ls_var, 0.0);
            break;
        case LS_TYPE::ENTIRE_DOMAIN:
            fillLSEntireDomain(ls_idx, vol_idx, area_idx, -1, patch_hierarchy);
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
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double {
                return std::exp(-1.0 * (X - cent).squaredNorm());
            };
            break;
        case ENTIRE_DOMAIN:
            Q_fcn = [](const VectorNd& X, const VectorNd& cent) -> double {
                return std::pow(std::sin(M_PI * X[0]) * std::sin(M_PI * X[1]), 4.0);
            };
            break;
        case DEFAULT:
        default:
            TBOX_ERROR("UNKNOWN TYPE\n");
            break;
        }
        qFcn->setLSIndex(ls_idx, vol_idx);
        qFcn->registerFcn(Q_fcn);

        qFcn->setDataOnPatchHierarchy(path_idx, path_var, patch_hierarchy, 0.0, false, 0, finest_ln);
        qFcn->setDataOnPatchHierarchy(Q_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);
        qFcn->setPathIndex(path_idx);
        qFcn->setDataOnPatchHierarchy(Q_app_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);
        qFcn->setDataOnPatchHierarchy(Q_err_idx, Q_var, patch_hierarchy, 0.0, false, 0, finest_ln);

        // Set up reconstruction
        Pointer<AdvectiveReconstructionOperator> op = nullptr;
        auto reconstruct_type = ADS::string_to_enum<AdvReconstructType>(input_db->getString("RECONSTRUCT_TYPE"));
        switch (reconstruct_type)
        {
        case AdvReconstructType::ZSPLINES:
            op = new ZSplineReconstructions("ZSPline", 2);
            break;
        case AdvReconstructType::RBF:
            op = new RBFReconstructions("RBF", Reconstruct::RBFPolyOrder::LINEAR, 5);
            break;
        default:
            TBOX_ERROR("UNKNOWN RECONSTRUCTION\n");
        }
        op->setLSData(ls_idx, vol_idx, ls_idx, vol_idx);
        op->allocateOperatorState(patch_hierarchy, 0.0, 0.0);
        op->applyReconstruction(Q_idx, Q_app_idx, path_idx);

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
                    (*wgt_data)(idx) *= (*vol_data)(idx) > 0.0 ? 1.0 : 0.0;
                }
            }
        }
#ifdef DRAW_DATA
        visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
#endif
        pout << "Norms of solution: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_app_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_app_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_app_idx, wgt_cc_idx) << "\n";

        pout << "Norms of exact: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_err_idx, wgt_cc_idx) << "\n";

        hier_cc_data_ops.subtract(Q_err_idx, Q_err_idx, Q_app_idx);
        pout << "Norms of error: \n"
             << "  L1-norm:  " << hier_cc_data_ops.L1Norm(Q_err_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << hier_cc_data_ops.L2Norm(Q_err_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << hier_cc_data_ops.maxNorm(Q_err_idx, wgt_cc_idx) << "\n";
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
            level->deallocatePatchData(Q_idx);
            level->deallocatePatchData(Q_app_idx);
            level->deallocatePatchData(Q_err_idx);
            level->deallocatePatchData(path_idx);
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
