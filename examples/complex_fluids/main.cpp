#include <ADS/CutCellMeshMapping.h>
#include <ADS/ExtrapolatedAdvDiffHierarchyIntegrator.h>
#include <ADS/LSFromMesh.h>
#include <ADS/PointwiseFunction.h>
#include <ADS/app_namespaces.h>

#include <ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h>
#include <ibamr/CFINSForcing.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBMethod.h>
#include <ibamr/IBRedundantInitializer.h>
#include <ibamr/IBStandardForceGen.h>
#include <ibamr/IBTargetPointForceSpec.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/StaggeredStokesSolverManager.h>
#include <ibamr/StokesSpecifications.h>
#include <ibamr/app_namespaces.h>

#include "ibtk/CartCellDoubleQuadraticRefine.h"
#include "ibtk/CartSideDoubleRT0Refine.h"
#include "ibtk/PETScKrylovLinearSolver.h"
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/LData.h>
#include <ibtk/LDataManager.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/LinearOperator.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <libmesh/edge_edge2.h>
#include <libmesh/exodusII.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/mesh_base.h>
#include <libmesh/mesh_tools.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>

#include <memory>

// Local includes
#include "IBBoundaryMeshMapping.h"
#include "VECFStrategy.h"

int finest_ln;
double K = 0.0;
double ETA = 0.0;
double MFAC = 0.0;
double dx = 0.0;
int num_nodes;
double ds;
double R = 0.0;
VectorNd cent;
double omega = 0.0;
bool use_asymmetric = true;
double t_start = 0.0;
double thalf = 0.5;
double tau = 0.25;
bool start_on_wall;

using VarIntPair = std::pair<Pointer<CellVariable<NDIM, double>>, Pointer<HierarchyIntegrator>>;
class Projector
{
public:
    Projector(Pointer<CFINSForcing> cf_forcing,
              Pointer<CellVariable<NDIM, double>> zb_var,
              Pointer<HierarchyIntegrator> zb_integrator,
              std::vector<VarIntPair> proj_vec,
              const ADS::Parameters& params)
        : d_cf_forcing(cf_forcing),
          d_zb_var(zb_var),
          d_zb_integrator(zb_integrator),
          d_proj_vec(proj_vec),
          d_params(params)
    {
    }

    void project()
    {
        projectStress();
        projectAdvectedQuantities();
    }
    void projectStress()
    {
        if (!d_cf_forcing) return;
        Pointer<CellVariable<NDIM, double>> cf_var = d_cf_forcing->getVariable();
        Pointer<HierarchyIntegrator> integrator = d_cf_forcing->getAdvDiffHierarchyIntegrator();
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int stress_new_idx = var_db->mapVariableAndContextToIndex(cf_var, integrator->getNewContext());
        const int zb_new_idx = var_db->mapVariableAndContextToIndex(d_zb_var, d_zb_integrator->getNewContext());
        const double C8 = d_params.C8;

        Pointer<PatchHierarchy<NDIM>> hierarchy = integrator->getPatchHierarchy();
        for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<CellData<NDIM, double>> stress_data = patch->getPatchData(stress_new_idx);
                Pointer<CellData<NDIM, double>> zb_data = patch->getPatchData(zb_new_idx);
                int num_projections = 0;
                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double zb = (*zb_data)(idx);
                    MatrixNd sig;
                    sig(0, 0) = (*stress_data)(idx, 0) + C8 * zb;
                    sig(1, 1) = (*stress_data)(idx, 1) + C8 * zb;
                    sig(0, 1) = sig(1, 0) = (*stress_data)(idx, 2);
                    Eigen::SelfAdjointEigenSolver<MatrixNd> eigs;
                    eigs.computeDirect(sig);
                    if (eigs.eigenvalues().minCoeff() < 0.0)
                    {
                        ++num_projections;
                        MatrixNd eig_vals(MatrixNd::Zero());
                        for (int d = 0; d < NDIM; ++d)
                        {
                            eig_vals(d, d) = std::max(eigs.eigenvalues()(d), 0.0);
                        }
                        MatrixNd eig_vecs = eigs.eigenvectors();
                        sig = eig_vecs * eig_vals * eig_vecs.transpose();
                        (*stress_data)(idx, 0) = sig(0, 0) - C8 * zb;
                        (*stress_data)(idx, 1) = sig(1, 1) - C8 * zb;
                        (*stress_data)(idx, 2) = sig(0, 1);
                    }

                    /*!
                     * This seems to be important to make sure that weird things don't happen downstream of the clot.
                     */
                    if (zb < 1.0e-5)
                    {
                        for (int d = 0; d < NDIM * (NDIM + 1) / 2; ++d) (*stress_data)(idx, d) = 0.0;
                    }
                }
                pout << "On level " << ln << " and patch num " << patch->getPatchNumber() << ", there were "
                     << num_projections << " projections\n";
            }
        }
    }

    void projectAdvectedQuantities()
    {
        // Replace all negatives with zero in the projected variable list.
        for (const auto& var_int_pair : d_proj_vec)
        {
            Pointer<CellVariable<NDIM, double>> var = var_int_pair.first;
            Pointer<HierarchyIntegrator> integrator = var_int_pair.second;
            auto var_db = VariableDatabase<NDIM>::getDatabase();
            const int new_idx = var_db->mapVariableAndContextToIndex(var, integrator->getNewContext());
            // Make sure this value is non-negative
            Pointer<PatchHierarchy<NDIM>> hierarchy = integrator->getPatchHierarchy();
            for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
                for (PatchLevel<NDIM>::Iterator p(level); p; p++)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    Pointer<CellData<NDIM, double>> data = patch->getPatchData(new_idx);
                    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                    {
                        const CellIndex<NDIM>& idx = ci();
                        (*data)(idx) = std::max((*data)(idx), 0.0);
                    }
                }
            }
        }
    }

private:
    Pointer<CFINSForcing> d_cf_forcing;
    Pointer<CellVariable<NDIM, double>> d_zb_var;
    Pointer<HierarchyIntegrator> d_zb_integrator;
    std::vector<VarIntPair> d_proj_vec;
    const ADS::Parameters& d_params;
};

VectorNd
cylinder_pt(const double s, double t)
{
    VectorNd x;
    x(0) = cent(0) + R * std::cos(2.0 * M_PI * s);
    x(1) = cent(1) + R * std::sin(2.0 * M_PI * s);
    t -= t_start;
    if (t > 0.0)
        x(0) += R * (std::tanh((t - thalf) / (tau)) + std::tanh(thalf / tau)) / (1.0 + std::tanh(thalf / tau)) *
                std::sin(2.0 * M_PI * omega * t);
    return x;
}

// Assumes x is given in reference configuration
VectorNd
cylinder_pt(VectorNd x, double t)
{
    t -= t_start;
    if (t > 0.0)
        x(0) += R * (std::tanh((t - thalf) / (tau)) + std::tanh(thalf / tau)) / (1.0 + std::tanh(thalf / tau)) *
                std::sin(2.0 * M_PI * omega * t);
    return x;
}

VectorNd
cylinder_vel(VectorNd x, double t)
{
    VectorNd u = VectorNd::Zero();
    t -= t_start;
    if (t > 0.0)
        u(0) = (1.0 / (std::cosh((t - thalf) / tau) * std::cosh((t - thalf) / tau)) * std::sin(2.0 * M_PI * t) +
                2 * M_PI * tau * std::cos(2 * M_PI * t) * (std::tanh((t - thalf) / tau) + std::tanh(thalf / tau))) /
               (10.0 * tau * (1.0 + std::tanh(thalf / tau)));
    return u;
}

VectorNd
cylinder_vel_0(double t)
{
    return cylinder_vel(VectorNd::Zero(), t);
}

void
generate_structure(const unsigned int& strct_num,
                   const int& ln,
                   int& num_vertices,
                   std::vector<IBTK::Point>& vertex_posn,
                   void* /*ctx*/)
{
    if (ln != finest_ln)
    {
        num_vertices = 0;
        vertex_posn.resize(num_vertices);
        return;
    }
    double circum = 2.0 * M_PI * R;
    num_vertices = std::floor(circum / ds);
    vertex_posn.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        VectorNd x = cylinder_pt((static_cast<double>(i) + 0.5) / static_cast<double>(num_vertices), 0.0);
        vertex_posn[i] = x;
    }
    num_nodes = num_vertices;
    return;
}

void
generate_tethers(const unsigned int& strct_num,
                 const int& ln,
                 std::multimap<int, IBRedundantInitializer::TargetSpec>& tg_pt_spec,
                 void* /*ctx*/)
{
    if (ln != finest_ln) return;
    for (int k = 0; k < num_nodes; ++k)
    {
        IBRedundantInitializer::TargetSpec e;
        e.stiffness = K / ds;
        e.damping = ETA / ds;
        tg_pt_spec.insert(std::make_pair(k, e));
    }
}

void
move_tethers(LDataManager* data_manager, const double time)
{
    // Update both local and ghost nodes.
    Pointer<LMesh> mesh = data_manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();
    const std::vector<LNode*>& ghost_nodes = mesh->getGhostNodes();
    nodes.insert(nodes.end(), ghost_nodes.begin(), ghost_nodes.end());

    // Also need reference information.
    Pointer<LData> X_ref_data = data_manager->getLData(data_manager->INIT_POSN_DATA_NAME, finest_ln);
    double* X_ref_vals = X_ref_data->getVecArray()->data();

    for (const auto node : nodes)
    {
        const auto& force_spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!force_spec) continue;
        const int lag_idx = node->getLagrangianIndex();
        const int petsc_idx = node->getLocalPETScIndex();
        IBTK::Point& X_target = force_spec->getTargetPointPosition();
        // Detect with side of channel we are on.
        Eigen::Map<VectorNd> X_ref(&X_ref_vals[petsc_idx * NDIM]);
        X_target = cylinder_pt(X_ref, time);
    }

    X_ref_data->restoreArrays();
}

void
print_max_dist_from_target(LDataManager* data_manager, const double time)
{
    double max_disp = 0.0;
    Pointer<LMesh> mesh = data_manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();

    Pointer<LData> X_data = data_manager->getLData(data_manager->POSN_DATA_NAME, finest_ln);
    double* X_vals = X_data->getVecArray()->data();
    Pointer<LData> X_ref_data = data_manager->getLData(data_manager->INIT_POSN_DATA_NAME, finest_ln);
    double* X_ref_vals = X_ref_data->getVecArray()->data();

    for (const auto& node : nodes)
    {
        const int petsc_idx = node->getLocalPETScIndex();
        Eigen::Map<VectorNd> x(&X_vals[petsc_idx * NDIM]), x_ref(&X_ref_vals[petsc_idx * NDIM]);

        VectorNd x_target = cylinder_pt(x_ref, time);
        max_disp = std::max(max_disp, (x - x_target).norm());
    }

    pout << "Max distance from target point: " << max_disp << "\n";
    pout << "Fraction of grid spacing:       " << max_disp / dx << "\n";
    if (max_disp / dx > 0.25)
    {
        // Error out if the distance from target point is more than 10% of a grid cell.
        //        TBOX_ERROR("Too far away from target point!");
    }
}

void
set_exact_location(LDataManager* data_manager, const double time)
{
    Pointer<LMesh> mesh = data_manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();

    Pointer<LData> X_data = data_manager->getLData(data_manager->POSN_DATA_NAME, finest_ln);
    double* X_vals = X_data->getVecArray()->data();
    Pointer<LData> X_ref_data = data_manager->getLData(data_manager->INIT_POSN_DATA_NAME, finest_ln);
    double* X_ref_vals = X_ref_data->getVecArray()->data();

    const std::pair<int, int>& lag_idxs = data_manager->getLagrangianStructureIndexRange(1, finest_ln);

    for (const auto& node : nodes)
    {
        const int petsc_idx = node->getLocalPETScIndex();
        const int lag_idx = node->getLagrangianIndex();
        if (lag_idx >= lag_idxs.first && lag_idx < lag_idxs.second)
        {
            Eigen::Map<VectorNd> x(&X_vals[petsc_idx * NDIM]), x_ref(&X_ref_vals[petsc_idx * NDIM]);
            x = cylinder_pt(x_ref, time);
        }
    }

    X_data->restoreArrays();
    X_ref_data->restoreArrays();
}

double
ls_bdry_fcn(const VectorNd&, const double)
{
    return -10.0;
}

namespace ADS
{
class BondSource : public CartGridFunction
{
public:
    BondSource(std::string object_name,
               Pointer<CellVariable<NDIM, double>> zb_var,
               Pointer<CellVariable<NDIM, double>> phib_var,
               Pointer<CellVariable<NDIM, double>> stress_var,
               Pointer<HierarchyIntegrator> zb_integrator,
               Pointer<HierarchyIntegrator> phib_integrator,
               Pointer<HierarchyIntegrator> stress_integrator,
               const Parameters& params)
        : CartGridFunction(std::move(object_name)),
          d_zb_integrator(zb_integrator),
          d_phib_integrator(phib_integrator),
          d_stress_integrator(stress_integrator),
          d_zb_var(zb_var),
          d_phib_var(phib_var),
          d_stress_var(stress_var),
          d_params(params)
    {
        // intentionally blank
    } // BondSource

    bool isTimeDependent() const override
    {
        return true;
    }

    void setDataOnPatch(const int data_idx,
                        Pointer<hier::Variable<NDIM>> /*var*/,
                        Pointer<Patch<NDIM>> patch,
                        const double data_time,
                        const bool initial_time,
                        Pointer<PatchLevel<NDIM>> /*patch_level*/) override
    {
        Pointer<CellData<NDIM, double>> ret_data = patch->getPatchData(data_idx);
        ret_data->fillAll(0.0);
        if (initial_time) return;

        Pointer<CellData<NDIM, double>> zb_data = patch->getPatchData(d_zb_var, d_zb_integrator->getCurrentContext());
        Pointer<CellData<NDIM, double>> phib_data =
            patch->getPatchData(d_phib_var, d_phib_integrator->getCurrentContext());
        Pointer<CellData<NDIM, double>> stress_data =
            patch->getPatchData(d_stress_var, d_stress_integrator->getCurrentContext());

        const double gamma = d_params.gamma;
        const double R0 = d_params.R0;
        const double C3 = d_params.C3;
        const double lambda = d_params.lambda;
        const double zb_crit_val = d_params.zb_crit_val;
        const double Kbb = d_params.Kbb;

        for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
        {
            const CellIndex<NDIM>& idx = ci();
            const double zb = (*zb_data)(idx);
            const double phib = (*phib_data)(idx);

            // Bond breaking.
            double yavg = R0;
            if (zb > zb_crit_val)
                yavg = std::sqrt(gamma * ((*stress_data)(idx, 0) + (*stress_data)(idx, 1)) / zb + R0 * R0);
            const double beta = C3 * (yavg > R0 ? std::exp(lambda * (yavg - R0)) : 1.0);

            // Bond formation
            double alpha = Kbb * (phib - 2.0 * zb) * (phib - 2.0 * zb);

            (*ret_data)(idx) = alpha - beta * zb;
        }
    }

private:
    Pointer<HierarchyIntegrator> d_zb_integrator, d_phib_integrator, d_stress_integrator;
    Pointer<CellVariable<NDIM, double>> d_zb_var, d_phib_var, d_stress_var;

    const Parameters& d_params;
}; // setDataOnPatch
} // namespace ADS
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

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "stokes.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<INSStaggeredHierarchyIntegrator> ins_integrator = new INSStaggeredHierarchyIntegrator(
            "FluidSolver", app_initializer->getComponentDatabase("INSHierarchyIntegrator"), false);
        Pointer<IBMethod> ib_ops = new IBMethod("IBMethod", app_initializer->getComponentDatabase("IBMethod"));
        Pointer<IBExplicitHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_ops,
                                              ins_integrator);
        Pointer<ExtrapolatedAdvDiffHierarchyIntegrator> adv_diff_EX_integrator =
            new ExtrapolatedAdvDiffHierarchyIntegrator(
                "ExtrapolatedIntegrator", app_initializer->getComponentDatabase("AdvDiffIntegrator"), true);
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
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
        ins_integrator->registerAdvDiffHierarchyIntegrator(adv_diff_EX_integrator);

        // Configure the IB solver.
        Pointer<IBRedundantInitializer> ib_initializer = new IBRedundantInitializer(
            "IBInitializer", app_initializer->getComponentDatabase("IBRedundantInitializer"));
        std::vector<std::string> struct_list = { "cylinder" };
        finest_ln = input_db->getInteger("MAX_LEVELS") - 1;
        omega = input_db->getDouble("OMEGA");
        K = input_db->getDouble("K");
        ETA = input_db->getDouble("ETA");
        R = input_db->getDouble("R");
        input_db->getDoubleArray("CENTER", cent.data(), NDIM);
        MFAC = input_db->getDouble("MFAC");
        use_asymmetric = input_db->getBool("USE_ASYMMETRIC");
        dx = input_db->getDouble("DX");
        t_start = input_db->getDouble("T_START");
        double circum = 2.0 * M_PI * R;
        ds = MFAC * dx;

        ib_initializer->setStructureNamesOnLevel(finest_ln, struct_list);
        ib_initializer->registerInitStructureFunction(generate_structure);
        ib_initializer->registerInitTargetPtFunction(generate_tethers);
        ib_ops->registerLInitStrategy(ib_initializer);
        Pointer<IBStandardForceGen> ib_target_forces = new IBStandardForceGen();
        ib_ops->registerIBLagrangianForceFunction(ib_target_forces);

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        Pointer<LSiloDataWriter> data_writer = new LSiloDataWriter(
            "DataWriter", input_db->getDatabase("Main")->getString("viz_dump_dirname") + "/lag_data", false);
        time_integrator->registerVisItDataWriter(visit_data_writer);
        ib_initializer->registerLSiloDataWriter(data_writer);
        ib_ops->registerLSiloDataWriter(data_writer);
        adv_diff_EX_integrator->registerVisItDataWriter(visit_data_writer);

        // Clot parameters
        ADS::Parameters params(input_db->getDatabase("Parameters"));

        /*
         * Boundary Condition Objects
         */
        // Velocity and pressure
        std::vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, nullptr);
        for (int d = 0; d < NDIM; ++d)
        {
            std::string u_bc_coef_name = "u_bcs_" + std::to_string(d);
            u_bc_coefs[d] =
                new muParserRobinBcCoefs(u_bc_coef_name, input_db->getDatabase(u_bc_coef_name), grid_geometry);
        }
        ins_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);

        // Default conditions
        auto default_bc_coef =
            std::make_unique<muParserRobinBcCoefs>("default_bc", input_db->getDatabase("default_bc"), grid_geometry);
        const std::vector<RobinBcCoefStrategy<NDIM>*> default_bc_coef_vec{ default_bc_coef.get() };

        /*
         * Initial conditions
         */
        const double R_material = input_db->getDouble("RM");
        const double R_decay = input_db->getDouble("R_DECAY");
        auto ciz = [&R_material, &R_decay](const IBTK::VectorNd& x) -> double
        {
            double r = (x - cent).norm();
            if (r < R_material)
                return std::tanh((R_material - r) / R_decay);
            else
                return 0.0;
        };

        const double phib_init_val = input_db->getDouble("PHIB_INIT");
        const double zb_init_val = input_db->getDouble("ZB_INIT");

        // Bound platelets
        Pointer<CartGridFunction> phib_init_fcn = new ADS::PointwiseFunction<PointwiseFunctions::ScalarFcn>(
            "Phib_init",
            [&ciz, &phib_init_val](double, const IBTK::VectorNd& x, double) -> double
            { return phib_init_val * ciz(x); });

        // Bond density
        Pointer<CartGridFunction> zb_init_fcn = new ADS::PointwiseFunction<PointwiseFunctions::ScalarFcn>(
            "zb_init",
            [&ciz, &zb_init_val](double, const IBTK::VectorNd& x, double) -> double { return zb_init_val * ciz(x); });

        // PhiB
        Pointer<CellVariable<NDIM, double>> phib_var = new CellVariable<NDIM, double>("PhiB");
        adv_diff_EX_integrator->registerTransportedQuantity(phib_var);
        adv_diff_EX_integrator->setPhysicalBcCoefs(phib_var, default_bc_coef_vec);
        adv_diff_EX_integrator->setAdvectionVelocity(phib_var, ins_integrator->getAdvectionVelocityVariable());
        adv_diff_EX_integrator->setInitialConditions(phib_var, phib_init_fcn);

        Pointer<CellVariable<NDIM, double>> zb_var = new CellVariable<NDIM, double>("Zb");
        adv_diff_EX_integrator->registerTransportedQuantity(zb_var);
        adv_diff_EX_integrator->setPhysicalBcCoefs(zb_var, default_bc_coef_vec);
        adv_diff_EX_integrator->setAdvectionVelocity(zb_var, ins_integrator->getAdvectionVelocityVariable());
        adv_diff_EX_integrator->setInitialConditions(zb_var, zb_init_fcn);

        // Stress
        Pointer<CFStrategy> cf_strategy =
            new VECFStrategy("CFStrategy", ins_integrator, zb_var, adv_diff_EX_integrator, params);
        Pointer<INSHierarchyIntegrator> cf_ins_integrator = ins_integrator;
        Pointer<CFINSForcing> cf_forcing = new CFINSForcing("CFForcing",
                                                            input_db->getDatabase("CFForcing"),
                                                            cf_ins_integrator,
                                                            grid_geometry,
                                                            adv_diff_EX_integrator,
                                                            visit_data_writer);
        Pointer<CellVariable<NDIM, double>> stress_var = cf_forcing->getVariable();
        cf_forcing->registerCFStrategy(cf_strategy);
        ins_integrator->registerBodyForceFunction(cf_forcing);

        Pointer<CellVariable<NDIM, double>> zb_src_var = new CellVariable<NDIM, double>("zb_src");
        Pointer<CartGridFunction> zb_src_fcn = new BondSource("BondSource",
                                                              zb_var,
                                                              phib_var,
                                                              stress_var,
                                                              adv_diff_EX_integrator,
                                                              adv_diff_EX_integrator,
                                                              adv_diff_EX_integrator,
                                                              params);
        adv_diff_EX_integrator->registerSourceTerm(zb_src_var);
        adv_diff_EX_integrator->setSourceTermFunction(zb_src_var, zb_src_fcn);
        adv_diff_EX_integrator->setSourceTerm(zb_var, zb_src_var);

        // Setup velocity and pressure initial conditions.
        Pointer<CartGridFunction> u_fcn =
            new muParserCartGridFunction("u", app_initializer->getComponentDatabase("u_init"), grid_geometry);
        Pointer<CartGridFunction> p_fcn =
            new muParserCartGridFunction("p", app_initializer->getComponentDatabase("p"), grid_geometry);
        ins_integrator->registerVelocityInitialConditions(u_fcn);
        ins_integrator->registerPressureInitialConditions(p_fcn);

        std::vector<VarIntPair> proj_vec{ std::make_pair(zb_var, adv_diff_EX_integrator),
                                          std::make_pair(phib_var, adv_diff_EX_integrator) };
        Projector projector(cf_forcing, zb_var, adv_diff_EX_integrator, proj_vec, params);
        auto reset_fcn = [](double current_time, double new_time, int cycle_num, void* ctx) -> void
        {
            auto projection = static_cast<Projector*>(ctx);
            projection->project();
        };
        adv_diff_EX_integrator->registerIntegrateHierarchyCallback(reset_fcn, static_cast<void*>(&projector));

        /***********************************
         * Level set for immersed boundary *
         ***********************************/
        // Generate finite element structure.
        libMesh::Mesh cylinder_mesh(ibtk_init.getLibMeshInit().comm(), NDIM - 1);

        // Note we can not copy the mesh from the LDataManager because we need to setup mesh points before the
        // LDataManager is populated. Instead, we construct the mesh using the same functions used to populate the
        // LDataManager.
        {
            int num_vertices = 0;
            std::vector<IBTK::Point> vertex_posn;
            generate_structure(0, finest_ln, num_vertices, vertex_posn, nullptr);
            cylinder_mesh.reserve_nodes(num_vertices);
            cylinder_mesh.reserve_elem(num_vertices);
            for (int node_num = 0; node_num < num_vertices; ++node_num)
            {
                cylinder_mesh.add_point(libMesh::Point(vertex_posn[node_num][0], vertex_posn[node_num][1], 0.0),
                                        node_num);
            }

            // Generate elements
            for (int i = 0; i < num_vertices - 1; ++i)
            {
                Elem* elem = cylinder_mesh.add_elem(new libMesh::Edge2());
                elem->set_node(0) = cylinder_mesh.node_ptr(i);
                elem->set_node(1) = cylinder_mesh.node_ptr(i + 1);
            }

            // Last element is from node num_vertices to node 0
            Elem* elem = cylinder_mesh.add_elem(new libMesh::Edge2());
            elem->set_node(0) = cylinder_mesh.node_ptr(num_vertices - 1);
            elem->set_node(1) = cylinder_mesh.node_ptr(0);
        }

        cylinder_mesh.prepare_for_use();

        // Generate mesh mappings and level set information
        std::vector<MeshBase*> meshes = { &cylinder_mesh };
        std::vector<int> part_nums = { 1, 0 };
        LDataManager* ib_manager = ib_ops->getLDataManager();
        auto mesh_mapping = std::make_shared<IBBoundaryMeshMapping>("BoundaryMeshMapping",
                                                                    input_db->getDatabase("MeshMapping"),
                                                                    meshes,
                                                                    ib_manager,
                                                                    finest_ln,
                                                                    part_nums,
                                                                    patch_hierarchy);
        mesh_mapping->initializeEquationSystems();
        Pointer<CutCellMeshMapping> cut_cell_mapping =
            new CutCellMeshMapping("CutCellMapping", app_initializer->getComponentDatabase("CutCellMapping"));
        Pointer<LSFromMesh> vol_fcn =
            new LSFromMesh("LSFromMesh", patch_hierarchy, mesh_mapping->getSystemManagers(), cut_cell_mapping, true);
        vol_fcn->registerBdryFcn(ls_bdry_fcn);
        vol_fcn->registerNormalReverseDomainId(0, 0);

        Pointer<NodeVariable<NDIM, double>> ls_var = new NodeVariable<NDIM, double>("LS");
        adv_diff_EX_integrator->registerLevelSetVariable(ls_var, vol_fcn);

        adv_diff_EX_integrator->setMeshMapping(mesh_mapping);

        /****************************************
         * Restrict variables to the level set. *
         ****************************************/
        adv_diff_EX_integrator->restrictToLevelSet(phib_var, ls_var);
        adv_diff_EX_integrator->restrictToLevelSet(zb_var, ls_var);
        adv_diff_EX_integrator->restrictToLevelSet(stress_var, ls_var);

        // Initialize all data
        mesh_mapping->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Get some time stepping information.
        unsigned int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        double time_end = time_integrator->getEndTime();
        double dt = 0.0;

        input_db->printClassData(plog);
        app_initializer.setNull();
        ib_ops->freeLInitStrategy();
        ib_initializer.setNull();

        // Visualization files info.
        double viz_dump_time_interval = input_db->getDouble("VIZ_DUMP_TIME_INTERVAL");
        double next_viz_dump_time = 0.0;
        // At specified intervals, write visualization files
        if (IBTK::abs_equal_eps(loop_time, next_viz_dump_time, 0.1 * dt) || loop_time >= next_viz_dump_time)
        {
            pout << "\nWriting visualization files...\n\n";
            time_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            data_writer->writePlotData(iteration_num, loop_time);
            next_viz_dump_time += viz_dump_time_interval;
        }
        // Main time step loop
        while (!IBTK::rel_equal_eps(loop_time, time_end) && time_integrator->stepsRemaining())
        {
            print_max_dist_from_target(ib_ops->getLDataManager(), loop_time);
            move_tethers(ib_ops->getLDataManager(), loop_time);
            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            mesh_mapping->updateBoundaryLocation(loop_time);
            // for (auto& mesh_partitioner : mesh_mapping->getMeshPartitioners())
            // {
            //     mesh_partitioner->setPatchHierarchy(patch_hierarchy);
            //     mesh_partitioner->reinitElementMappings(2);
            // }

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            iteration_num += 1;
            // At specified intervals, write visualization files
            if (IBTK::abs_equal_eps(loop_time, next_viz_dump_time, 0.1 * dt) || loop_time >= next_viz_dump_time)
            {
                pout << "\nWriting visualization files...\n\n";
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                data_writer->writePlotData(iteration_num, loop_time);
                next_viz_dump_time += viz_dump_time_interval;
            }

            if (dump_timer_data && (iteration_num % timer_dump_interval == 0))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
        }
    } // cleanup dynamically allocated objects prior to shutdown
} // main
