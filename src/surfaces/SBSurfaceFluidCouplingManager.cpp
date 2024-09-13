#include "ADS/SBSurfaceFluidCouplingManager.h"
#include "ADS/app_namespaces.h"
#include "ADS/surface_utilities.h"

#include "ibtk/IndexUtilities.h"
#include "ibtk/ibtk_utilities.h"

#include "libmesh/enum_preconditioner_type.h"
#include "libmesh/enum_solver_type.h"
#include "libmesh/equation_systems.h"
#include "libmesh/explicit_system.h"
#include "libmesh/petsc_linear_solver.h"
#include "libmesh/petsc_matrix.h"
#include "libmesh/transient_system.h"

namespace
{
static Timer* t_interpolateToBoundary = nullptr;
}

namespace ADS
{
SBSurfaceFluidCouplingManager::SBSurfaceFluidCouplingManager(std::string object_name,
                                                             const Pointer<Database>& input_db,
                                                             const std::vector<FESystemManager*>& fe_sys_managers)
    : d_object_name(std::move(object_name)),
      d_fe_system_managers(fe_sys_managers),
      d_J_sys_name("Jacobian"),
      d_scr_var(new CellVariable<NDIM, double>(d_object_name + "::SCR"))
{
    commonConstructor(input_db);
}

SBSurfaceFluidCouplingManager::SBSurfaceFluidCouplingManager(std::string object_name,
                                                             const Pointer<Database>& input_db,
                                                             FESystemManager* fe_sys_manager)
    : d_object_name(std::move(object_name)),
      d_fe_system_managers({ fe_sys_manager }),
      d_J_sys_name("Jacobian"),
      d_scr_var(new CellVariable<NDIM, double>(d_object_name + "::SCR"))
{
    commonConstructor(input_db);
}

void
SBSurfaceFluidCouplingManager::commonConstructor(Pointer<Database> input_db)
{
    d_rbf_reconstruct = new RBFReconstructCache(input_db->getInteger("stencil_width"));

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_scr_idx = var_db->registerVariableAndContext(
        d_scr_var,
        var_db->getContext(d_object_name + "::SCR"),
        std::floor(std::sqrt(static_cast<double>(d_rbf_reconstruct->getStencilWidth()))));

    unsigned int num_parts = getNumParts();
    pout << "Resizing vectors to size: " << num_parts << "\n";
    d_sf_names_vec.resize(num_parts);
    d_fl_names_vec.resize(num_parts);
    d_fl_vars_vec.resize(num_parts);
    d_sf_fl_map_vec.resize(num_parts);
    d_sf_sf_map_vec.resize(num_parts);
    d_fl_fl_map_vec.resize(num_parts);
    d_fl_sf_map_vec.resize(num_parts);
    d_sf_reaction_fcn_ctx_map_vec.resize(num_parts);
    d_fl_a_g_fcn_map_vec.resize(num_parts);
    d_sf_init_fcn_map_vec.resize(num_parts);

    IBTK_DO_ONCE(t_interpolateToBoundary =
                     TimerManager::getManager()->getTimer("ADS::SBDataManager::interpolateToBoundary()"););
    return;
}

SBSurfaceFluidCouplingManager::~SBSurfaceFluidCouplingManager()
{
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (level->checkAllocated(d_scr_idx)) level->deallocatePatchData(d_scr_idx);
    }
}

void
SBSurfaceFluidCouplingManager::registerFluidConcentration(Pointer<CellVariable<NDIM, double>> fl_var, unsigned int part)
{
    if (d_fe_eqs_initialized)
        TBOX_ERROR(d_object_name + ": can't register a fluid variable after equation systems have been initialized.");
    pout << d_fl_vars_vec.size();
    TBOX_ASSERT(fl_var);
    if (std::find(d_fl_vars_vec[part].begin(), d_fl_vars_vec[part].end(), fl_var) == d_fl_vars_vec[part].end())
    {
        d_fl_vars_vec[part].push_back(fl_var);
        d_fl_names_vec[part].push_back(fl_var->getName());
    }
}

void
SBSurfaceFluidCouplingManager::registerFluidConcentration(
    const std::vector<Pointer<CellVariable<NDIM, double>>>& fl_vars,
    unsigned int part)
{
    for (const auto& fl_var : fl_vars) registerFluidConcentration(fl_var, part);
}

void
SBSurfaceFluidCouplingManager::registerSurfaceConcentration(std::string surface_name, unsigned int part)
{
    if (d_fe_eqs_initialized)
        TBOX_ERROR(d_object_name + ": can't register a surface variable after equation systems have been initialized.");
    if (std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), surface_name) == d_sf_names_vec[part].end())
        d_sf_names_vec[part].push_back(std::move(surface_name));
}

void
SBSurfaceFluidCouplingManager::registerSurfaceConcentration(const std::vector<std::string>& surface_names,
                                                            unsigned int part)
{
    for (const auto& surface_name : surface_names) registerSurfaceConcentration(surface_name);
}

void
SBSurfaceFluidCouplingManager::registerFluidSurfaceDependence(const std::string& sf_name,
                                                              Pointer<CellVariable<NDIM, double>> fl_var,
                                                              unsigned int part)
{
    TBOX_ASSERT(std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), sf_name) !=
                d_sf_names_vec[part].end());
    const auto& fl_it = std::find(d_fl_vars_vec[part].begin(), d_fl_vars_vec[part].end(), fl_var);
    TBOX_ASSERT(fl_it != d_fl_vars_vec[part].end());
    const unsigned int l = std::distance(d_fl_vars_vec[part].begin(), fl_it);
    const std::string& fl_name = d_fl_names_vec[part][l];
    std::vector<std::string>& fl_vars_vec = d_sf_fl_map_vec[part][sf_name];
    if (std::find(fl_vars_vec.begin(), fl_vars_vec.end(), fl_name) == fl_vars_vec.end()) fl_vars_vec.push_back(fl_name);
    std::vector<std::string>& sf_vars_vec = d_fl_sf_map_vec[part][fl_name];
    if (std::find(sf_vars_vec.begin(), sf_vars_vec.end(), sf_name) == sf_vars_vec.end()) sf_vars_vec.push_back(sf_name);
}

void
SBSurfaceFluidCouplingManager::registerSurfaceSurfaceDependence(const std::string& part1_name,
                                                                const std::string& part2_name,
                                                                unsigned int part)
{
    TBOX_ASSERT(std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), part1_name) !=
                d_sf_names_vec[part].end());
    TBOX_ASSERT(std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), part2_name) !=
                d_sf_names_vec[part].end());
    std::vector<std::string>& sf_names_vec = d_sf_sf_map_vec[part][part1_name];
    if (std::find(sf_names_vec.begin(), sf_names_vec.end(), part2_name) != sf_names_vec.end())
        sf_names_vec.push_back(part2_name);
}

void
SBSurfaceFluidCouplingManager::registerSurfaceReactionFunction(const std::string& surface_name,
                                                               ReactionFcn fcn,
                                                               void* ctx /* = nullptr */,
                                                               unsigned int part)
{
    TBOX_ASSERT(std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), surface_name) !=
                d_sf_names_vec[part].end());
    d_sf_reaction_fcn_ctx_map_vec[part][surface_name] = std::make_pair(fcn, ctx);
}

void
SBSurfaceFluidCouplingManager::registerFluidBoundaryCondition(const Pointer<CellVariable<NDIM, double>>& fl_var,
                                                              ReactionFcn a_fcn,
                                                              ReactionFcn g_fcn,
                                                              void* ctx,
                                                              unsigned int part)
{
    TBOX_ASSERT(std::find(d_fl_vars_vec[part].begin(), d_fl_vars_vec[part].end(), fl_var) != d_fl_vars_vec[part].end());
    const std::string& fl_name = fl_var->getName();
    TBOX_ASSERT(std::find(d_fl_names_vec[part].begin(), d_fl_names_vec[part].end(), fl_name) !=
                d_fl_names_vec[part].end());
    d_fl_a_g_fcn_map_vec[part][fl_name] = std::make_tuple(a_fcn, g_fcn, ctx);
}

void
SBSurfaceFluidCouplingManager::initializeFEData()
{
    plog << d_object_name << ": Initializing FE data.\n";
    const bool from_restart = RestartManager::getManager()->isFromRestart();
    for (unsigned int part = 0; part < getNumParts(); ++part)
    {
        EquationSystems* eq_sys = d_fe_system_managers[part]->getEquationSystems();

        if (from_restart)
        {
            // Everything is already read in by VolumeBoundaryManager...
        }
        else
        {
            for (const auto& sf_name : d_sf_names_vec[part])
            {
                auto& surface_sys = eq_sys->add_system<TransientExplicitSystem>(sf_name);
                surface_sys.add_variable(sf_name, FEType());
            }

            for (const auto& fl_name : d_fl_names_vec[part])
            {
                auto& fluid_sys = eq_sys->add_system<ExplicitSystem>(fl_name);
                fluid_sys.add_variable(fl_name, FEType());
            }

            auto& J_sys = eq_sys->add_system<ExplicitSystem>(d_J_sys_name);
            J_sys.add_variable(d_J_sys_name, FEType());
        }
    }
}

void
SBSurfaceFluidCouplingManager::setLSData(const int ls_idx, const int vol_idx, Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_ls_idx = ls_idx;
    d_vol_idx = vol_idx;
    d_hierarchy = hierarchy;
    d_rbf_reconstruct->setLSData(ls_idx, vol_idx);
    d_rbf_reconstruct->setPatchHierarchy(hierarchy);
}

const std::string&
SBSurfaceFluidCouplingManager::interpolateToBoundary(Pointer<CellVariable<NDIM, double>> fl_var,
                                                     const int fl_idx,
                                                     const double time,
                                                     unsigned int part,
                                                     FEToHierarchyMapping* fe_hierarchy_mapping)
{
    ADS_TIMER_START(t_interpolateToBoundary);
    std::unique_ptr<FEToHierarchyMapping> fe_hierarchy_mapping_unique_ptr;
    if (!fe_hierarchy_mapping)
    {
        fe_hierarchy_mapping_unique_ptr =
            std::make_unique<FEToHierarchyMapping>(d_object_name + "::FEToHierarchyMapping",
                                                   d_fe_system_managers[part],
                                                   nullptr,
                                                   d_hierarchy->getNumberOfLevels(),
                                                   2 /*ghost width*/);
        fe_hierarchy_mapping_unique_ptr->setPatchHierarchy(d_hierarchy);
        fe_hierarchy_mapping_unique_ptr->reinitElementMappings(2 /*ghost_width*/);
        fe_hierarchy_mapping = fe_hierarchy_mapping_unique_ptr.get();
    }
    else
    {
        TBOX_ASSERT(&fe_hierarchy_mapping->getFESystemManager() == d_fe_system_managers[part]);
    }
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_scr_idx)) level->allocatePatchData(d_scr_idx);
    }
    const auto& fl_it = std::find(d_fl_vars_vec[part].begin(), d_fl_vars_vec[part].end(), fl_var);
    TBOX_ASSERT(fl_it != d_fl_vars_vec[part].end());
    TBOX_ASSERT(fl_idx != IBTK::invalid_index);
    const int l = std::distance(d_fl_vars_vec[part].begin(), fl_it);
    const std::string& fl_name = d_fl_names_vec[part][l];
    // First ensure we've filled ghost cells
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp(1);
    ghost_cell_comp[0] =
        ITC(d_scr_idx, fl_idx, "CONSERVATIVE_LINEAR_REFINE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr);
    HierarchyGhostCellInterpolation hier_ghost_cell;
    hier_ghost_cell.initializeOperatorState(ghost_cell_comp, d_hierarchy);
    hier_ghost_cell.fillData(time);

    const FESystemManager& fe_sys_manager = fe_hierarchy_mapping->getFESystemManager();
    EquationSystems* eq_sys = fe_sys_manager.getEquationSystems();
    System& fl_system = eq_sys->get_system(fl_name);
    const unsigned int n_vars = fl_system.n_vars();
    const DofMap& fl_dof_map = fl_system.get_dof_map();
    System& X_system = eq_sys->get_system(fe_sys_manager.getCoordsSystemName());
    const DofMap& X_dof_map = X_system.get_dof_map();

    NumericVector<double>* X_vec = X_system.current_local_solution.get();
    NumericVector<double>* fl_vec = fl_system.solution.get();

    auto X_petsc_vec = dynamic_cast<PetscVector<double>*>(X_vec);
    TBOX_ASSERT(X_petsc_vec != nullptr);
    const double* const X_local_soln = X_petsc_vec->get_array_read();

    fl_vec->zero();

    // Loop over patches and interpolate solution to the boundary
    // Assume we are only doing this on the finest level
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(d_hierarchy->getFinestLevelNumber());
    const std::vector<std::vector<Node*>>& active_patch_node_map = fe_hierarchy_mapping->getActivePatchNodeMap();
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const int local_patch_num = patch->getPatchNumber();

        const std::vector<Node*>& patch_nodes = active_patch_node_map[local_patch_num];
        const size_t num_active_patch_nodes = patch_nodes.size();
        if (!num_active_patch_nodes) continue;

        const Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const patch_x_low = pgeom->getXLower();
        const double* const patch_x_up = pgeom->getXUpper();
        std::array<bool, NDIM> touches_upper_regular_bdry;
        for (int d = 0; d < NDIM; ++d) touches_upper_regular_bdry[d] = pgeom->getTouchesRegularBoundary(d, 1);

        // Store the value of X at the nodes that are inside the current patch
        std::vector<dof_id_type> fl_node_idxs;
        std::vector<double> fl_node, X_node;
        fl_node_idxs.reserve(n_vars * num_active_patch_nodes);
        fl_node.reserve(n_vars * num_active_patch_nodes);
        X_node.reserve(NDIM * num_active_patch_nodes);
        std::vector<dof_id_type> fl_idxs, X_idxs;
        IBTK::Point X;
        for (unsigned int k = 0; k < num_active_patch_nodes; ++k)
        {
            const Node* const n = patch_nodes[k];
            bool inside_patch = true;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                IBTK::get_nodal_dof_indices(X_dof_map, n, d, X_idxs);
                X[d] = X_local_soln[X_petsc_vec->map_global_to_local_index(X_idxs[0])];
                inside_patch = inside_patch && (X[d] >= patch_x_low[d]) &&
                               ((X[d] < patch_x_up[d]) || (touches_upper_regular_bdry[d] && X[d] <= patch_x_up[d]));
            }
            if (inside_patch)
            {
                fl_node.resize(fl_node.size() + n_vars, 0.0);
                X_node.insert(X_node.end(), &X[0], &X[0] + NDIM);
                for (unsigned int i = 0; i < n_vars; ++i)
                {
                    IBTK::get_nodal_dof_indices(fl_dof_map, n, i, fl_idxs);
                    fl_node_idxs.insert(fl_node_idxs.end(), fl_idxs.begin(), fl_idxs.end());
                }
            }
        }

        TBOX_ASSERT(fl_node.size() <= n_vars * num_active_patch_nodes);
        TBOX_ASSERT(X_node.size() <= NDIM * num_active_patch_nodes);
        TBOX_ASSERT(fl_node_idxs.size() <= n_vars * num_active_patch_nodes);

        if (fl_node.empty()) continue;

        // Now we can interpolate from the fluid to the structure.
        Pointer<CellData<NDIM, double>> fl_data = patch->getPatchData(d_scr_idx);
        Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
        Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);
        for (size_t i = 0; i < fl_node.size(); ++i)
        {
            // Use a MLS linear approximation to evaluate data on structure
            const CellIndex<NDIM> cell_idx =
                IBTK::IndexUtilities::getCellIndex(&X_node[NDIM * i], pgeom, patch->getBox());
            const CellIndex<NDIM>& idx_low = patch->getBox().lower();
            VectorNd x_loc = {
                X_node[NDIM * i],
                X_node[NDIM * i + 1]
#if (NDIM == 3)
                ,
                X_node[NDIM * i + 2]
#endif
            };
            fl_node[i] = d_rbf_reconstruct->reconstructOnIndex(x_loc, cell_idx, *fl_data, patch);
        }
        fl_vec->add_vector(fl_node, fl_node_idxs);
    }
    X_petsc_vec->restore_array();
    fl_vec->close();
    ADS_TIMER_STOP(t_interpolateToBoundary);
    return fl_name;
}

const std::string&
SBSurfaceFluidCouplingManager::updateJacobian(unsigned int part)
{
    update_jacobian(d_J_sys_name, *d_fe_system_managers[part]);
    return d_J_sys_name;
}

void
SBSurfaceFluidCouplingManager::fillInitialConditions()
{
    for (unsigned int part = 0; part < getNumParts(); ++part)
    {
        for (const auto& sf_fcn_pair : d_sf_init_fcn_map_vec[part])
        {
            const std::string& sf_name = sf_fcn_pair.first;
            EquationSystems* eq_sys = d_fe_system_managers[part]->getEquationSystems();
            const MeshBase& mesh = eq_sys->get_mesh();
            System& sf_system = eq_sys->get_system(sf_name);
            const DofMap& sf_dof_map = sf_system.get_dof_map();
            System& X_system = eq_sys->get_system(d_fe_system_managers[part]->getCoordsSystemName());
            const DofMap& X_dof_map = X_system.get_dof_map();

            NumericVector<double>* X_vec = X_system.current_local_solution.get();
            NumericVector<double>* sf_vec = sf_system.solution.get();

            // Loop over nodes
            auto iter = mesh.local_nodes_begin();
            const auto iter_end = mesh.local_nodes_end();
            for (; iter != iter_end; ++iter)
            {
                const Node* const node = *iter;
                std::vector<dof_id_type> X_dof, sf_dof;
                X_dof_map.dof_indices(node, X_dof);
                sf_dof_map.dof_indices(node, sf_dof);
                VectorNd x;
                for (int d = 0; d < NDIM; ++d) x[d] = (*X_vec)(X_dof[d]);
                sf_vec->set(sf_dof[0], sf_fcn_pair.second(x, node));
            }
            sf_vec->close();
            sf_system.update();
        }
    }
}
} // namespace ADS
