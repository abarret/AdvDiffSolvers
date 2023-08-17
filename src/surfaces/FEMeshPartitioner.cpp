/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/FEMeshPartitioner.h"
#include <ADS/libmesh_utilities.h>

#include "ibtk/FECache.h"
#include "ibtk/FEMappingCache.h"
#include "ibtk/FEProjector.h"
#include "ibtk/IBTK_CHKERRQ.h"
#include "ibtk/IBTK_MPI.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/LEInteractor.h"
#include "ibtk/ibtk_utilities.h"
#include "ibtk/libmesh_utilities.h"

#include "BasePatchHierarchy.h"
#include "Box.h"
#include "CartesianCellDoubleWeightedAverage.h"
#include "CartesianGridGeometry.h"
#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellIterator.h"
#include "CellVariable.h"
#include "CoarsenAlgorithm.h"
#include "CoarsenOperator.h"
#include "HierarchyCellDataOpsReal.h"
#include "HierarchyDataOpsManager.h"
#include "HierarchyDataOpsReal.h"
#include "Index.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchData.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "RefineSchedule.h"
#include "SideData.h"
#include "SideGeometry.h"
#include "SideIndex.h"
#include "SideIterator.h"
#include "SideVariable.h"
#include "Variable.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "tbox/Database.h"
#include "tbox/PIO.h"
#include "tbox/Pointer.h"
#include "tbox/RestartManager.h"
#include "tbox/ShutdownRegistry.h"
#include "tbox/Timer.h"
#include "tbox/TimerManager.h"
#include "tbox/Utilities.h"

#include "libmesh/boundary_info.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_parallel_type.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/id_types.h"
#include "libmesh/libmesh_config.h"
#include "libmesh/libmesh_version.h"
#include "libmesh/linear_solver.h"
#include "libmesh/mesh_base.h"
#include "libmesh/node.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/petsc_linear_solver.h"
#include "libmesh/petsc_matrix.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/quadrature_grid.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/tensor_value.h"
#include "libmesh/type_vector.h"
#include "libmesh/variant_filter_iterator.h"

#include "petscvec.h"

IBTK_DISABLE_EXTRA_WARNINGS
#include "boost/multi_array.hpp"
IBTK_ENABLE_EXTRA_WARNINGS

#include "ADS/app_namespaces.h" // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Timers.
static Timer* t_reinit_element_mappings;
static Timer* t_build_ghosted_solution_vector;
static Timer* t_build_ghosted_vector;
static Timer* t_update_workload_estimates;
static Timer* t_initialize_level_data;
static Timer* t_reset_hierarchy_configuration;
static Timer* t_apply_gradient_detector;

// Local helper functions.
struct ElemComp
{
    inline bool operator()(const Elem* const x, const Elem* const y) const
    {
        return x->id() < y->id();
    } // operator()
};

template <class ContainerOfContainers>
inline void
collect_unique_elems(std::vector<Elem*>& elems, const ContainerOfContainers& elem_patch_map)
{
    std::set<Elem*, ElemComp> elem_set;
    for (auto it = elem_patch_map.begin(); it != elem_patch_map.end(); ++it)
    {
        elem_set.insert(it->begin(), it->end());
    }
    elems.assign(elem_set.begin(), elem_set.end());
    return;
} // collect_unique_elems

std::set<libMesh::subdomain_id_type>
collect_subdomain_ids(const libMesh::MeshBase& mesh)
{
    std::set<libMesh::subdomain_id_type> subdomain_ids;
    // Get all subdomain ids present, not just local elements.
    const auto el_begin = mesh.elements_begin();
    const auto el_end = mesh.elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        subdomain_ids.insert((*el_it)->subdomain_id());
    }
    return subdomain_ids;
}

#if LIBMESH_VERSION_LESS_THAN(1, 6, 0)
// libMesh's box intersection code is slow and not in a header (i.e., cannot be
// inlined). This is problematic for us since we presently call this function
// for every element on every patch: i.e., for N patches on the current
// processor and K *total* (i.e., on all processors) elements we call this
// function K*N times, which takes up a lot of time in regrids. For example:
// switching to this function lowers the time required to get to the end of the
// first time step in the TAVR model by 20%.
inline bool
bbox_intersects(const libMeshWrappers::BoundingBox& a, const libMeshWrappers::BoundingBox& b)
{
    const libMesh::Point& a_lower = a.first;
    const libMesh::Point& a_upper = a.second;

    const libMesh::Point& b_lower = b.first;
    const libMesh::Point& b_upper = b.second;

    // Since boxes are tensor products of line intervals it suffices to check
    // that the line segments for each coordinate axis overlap.
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        // Line segments can intersect in two ways:
        // 1. They can overlap.
        // 2. One can be inside the other.
        //
        // In the first case we want to see if either end point of the second
        // line segment lies within the first. In the second case we can simply
        // check that one end point of the first line segment lies in the second
        // line segment. Note that we don't need, in the second case, to do two
        // checks since that case is already covered by the first.
        if (!((a_lower(d) <= b_lower(d) && b_lower(d) <= a_upper(d)) ||
              (a_lower(d) <= b_upper(d) && b_upper(d) <= a_upper(d))) &&
            !((b_lower(d) <= a_lower(d) && a_lower(d) <= b_upper(d))))
        {
            return false;
        }
    }

    return true;
}
#endif

Pointer<Database>
setup_fe_projector_db(const Pointer<Database>& input_db)
{
    Pointer<Database> db;
    if (input_db->keyExists("FEProjector"))
    {
        db = input_db->getDatabase("FEProjector");
    }
    else
    {
        db = new InputDatabase("FEProjector");
    }
    return db;
}
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////
FEMeshPartitioner::FEMeshPartitioner(std::string object_name,
                                     const Pointer<Database>& input_db,
                                     const int max_levels,
                                     IntVector<NDIM> ghost_width,
                                     std::shared_ptr<FEData> fe_data,
                                     std::string coords_sys_name)
    : COORDINATES_SYSTEM_NAME(std::move(coords_sys_name)),
      d_fe_data(fe_data),
      d_fe_projector(new FEProjector(d_fe_data, setup_fe_projector_db(input_db))),
      d_level_lookup(max_levels - 1,
                     collect_subdomain_ids(d_fe_data->getEquationSystems()->get_mesh()),
                     input_db ? (input_db->keyExists("subdomain_ids_on_levels") ?
                                     input_db->getDatabase("subdomain_ids_on_levels") :
                                     nullptr) :
                                nullptr),
      d_object_name(std::move(object_name)),
      d_max_level_number(max_levels - 1),
      d_ghost_width(std::move(ghost_width))
{
    TBOX_ASSERT(!d_object_name.empty());
    d_node_patch_check = string_to_enum<NodeOutsidePatchCheckType>("NODE_OUTSIDE_WARN");

    // Setup Timers.
    IBTK_DO_ONCE(t_reinit_element_mappings =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::reinitElementMappings()");
                 t_build_ghosted_solution_vector =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::buildGhostedSolutionVector()");
                 t_build_ghosted_vector =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::buildGhostedVector()");
                 t_update_workload_estimates =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::updateWorkloadEstimates()");
                 t_initialize_level_data =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::initializeLevelData()");
                 t_reset_hierarchy_configuration =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::resetHierarchyConfiguration()");
                 t_apply_gradient_detector =
                     TimerManager::getManager()->getTimer("IBTK::FEMeshPartitioner::applyGradientDetector()");)
    return;
} // FEMeshPartitioner

EquationSystems*
FEMeshPartitioner::getEquationSystems() const
{
    return d_fe_data->getEquationSystems();
} // getEquationSystems

FEMeshPartitioner::SystemDofMapCache*
FEMeshPartitioner::getDofMapCache(const std::string& system_name)
{
    return d_fe_data->getDofMapCache(system_name);
} // getDofMapCache

FEMeshPartitioner::SystemDofMapCache*
FEMeshPartitioner::getDofMapCache(unsigned int system_num)
{
    return d_fe_data->getDofMapCache(system_num);
} // getDofMapCache

void
FEMeshPartitioner::setPatchHierarchy(Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    // Reset the hierarchy.
    TBOX_ASSERT(hierarchy);
    d_hierarchy = hierarchy;
    return;
} // setPatchHierarchy

Pointer<PatchHierarchy<NDIM>>
FEMeshPartitioner::getPatchHierarchy() const
{
    return d_hierarchy;
} // getPatchHierarchy

int
FEMeshPartitioner::getCoarsestPatchLevelNumber() const
{
    for (int ln = 0; ln <= d_max_level_number; ++ln)
    {
        if (d_level_lookup.levelHasElements(ln)) return ln;
    }
    TBOX_ERROR("There must be a coarsest patch level number.");
    return IBTK::invalid_level_number;
}

int
FEMeshPartitioner::getFinestPatchLevelNumber() const
{
    for (int ln = d_max_level_number; 0 <= ln; --ln)
    {
        if (d_level_lookup.levelHasElements(ln)) return ln;
    }
    TBOX_ERROR("There must be a finest patch level number.");
    return IBTK::invalid_level_number;
}

const IntVector<NDIM>&
FEMeshPartitioner::getGhostCellWidth() const
{
    return d_ghost_width;
} // getGhostCellWidth

const std::vector<std::vector<Elem*>>&
FEMeshPartitioner::getActivePatchElementMap(int ln) const
{
    if (ln == IBTK::invalid_level_number)
        return d_active_patch_elem_map.back();
    else
        return d_active_patch_elem_map[ln];
} // getActivePatchElementMap

const std::vector<std::vector<Node*>>&
FEMeshPartitioner::getActivePatchNodeMap(int ln) const
{
    if (ln == IBTK::invalid_level_number)
        return d_active_patch_node_map.back();
    else
        return d_active_patch_node_map[ln];
} // getActivePatchNodeMap

void
FEMeshPartitioner::reinitElementMappings(const IntVector<NDIM>& ghost_width)
{
    d_ghost_width = ghost_width;
    IBTK_TIMER_START(t_reinit_element_mappings);

    // We reinitialize mappings after repartitioning, so clear the cache since
    // its content is no longer relevant:
    d_fe_data->clearPatchHierarchyDependentData();

    // Delete cached hierarchy-dependent data.
    d_active_patch_elem_map.clear();
    d_active_patch_elem_map.resize(d_max_level_number + 1);
    d_active_patch_node_map.clear();
    d_active_patch_node_map.resize(d_max_level_number + 1);
    d_active_patch_ghost_dofs.clear();
    d_active_nodes.clear();
    d_system_ghost_vec.clear();
    d_system_ib_ghost_vec.clear();

    // Reset the mappings between grid patches and active mesh
    // elements.
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        collectActivePatchElements(d_active_patch_elem_map[ln],
                                   d_fe_data,
                                   COORDINATES_SYSTEM_NAME,
                                   d_hierarchy->getPatchLevel(ln),
                                   d_ghost_width);
        collectActivePatchNodes(d_active_patch_node_map[ln], d_active_patch_elem_map[ln], d_fe_data);
    }

    for (const std::vector<std::vector<Node*>>& level_nodes : d_active_patch_node_map)
        for (const std::vector<Node*>& patch_nodes : level_nodes)
            d_active_nodes.insert(patch_nodes.begin(), patch_nodes.end());

    // If we are not regridding in the usual way (i.e., if
    // IBHierarchyIntegrator::d_regrid_cfl_interval > 1) then it is possible
    // that an element has traveled outside of it's assigned patch level. If
    // this happens then IBFE won't work - we cannot correctly interpolate
    // velocity at that point. Hence try to detect it by checking that all nodes
    // are on the interior of some patch (or outside the domain) at the moment.
    {
        const Pointer<CartesianGridGeometry<NDIM>> hier_geom = d_hierarchy->getGridGeometry();
        // TODO - we only support single box geometries right now
        TBOX_ASSERT(hier_geom);
        const double* const hier_x_lower = hier_geom->getXLower();
        const double* const hier_x_upper = hier_geom->getXUpper();
        const int rank = IBTK_MPI::getRank();
        const int n_procs = IBTK_MPI::getNodes();
        const MeshBase& mesh = getEquationSystems()->get_mesh();
        auto X_petsc_vec = static_cast<PetscVector<double>*>(getCoordsVector());
        X_petsc_vec->close();
        const double* const X_local_soln = X_petsc_vec->get_array_read();
        const DofMap& X_dof_map = d_fe_data->getEquationSystems()->get_system(COORDINATES_SYSTEM_NAME).get_dof_map();

        std::vector<int> node_ranks(mesh.parallel_n_nodes());
        std::vector<dof_id_type> X_idxs;
        for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            int local_patch_num = 0;
            for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
            {
                const Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
                const double* const patch_x_lower = patch_geom->getXLower();
                const double* const patch_x_upper = patch_geom->getXUpper();

                for (const Node* n : d_active_patch_node_map[ln][local_patch_num])
                {
                    IBTK::Point X;
                    bool inside_patch = true;
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        IBTK::get_nodal_dof_indices(X_dof_map, n, d, X_idxs);
                        X[d] = X_local_soln[X_petsc_vec->map_global_to_local_index(X_idxs[0])];
                        // Due to how SAMRAI computes patch boundaries, even if the
                        // patch's domain is [0, 1]^2 the patches on the boundary
                        // may not actually end at 1.0. Hence allow a small
                        // tolerance here to account for the case where two patches
                        // are adjacent to each-other but their patch boundaries
                        // don't quite line up:
                        const double x_lower = patch_x_lower[d] - std::max(1.0, std::abs(patch_x_lower[d])) * 1e-14;
                        const double x_upper = patch_x_upper[d] + std::max(1.0, std::abs(patch_x_upper[d])) * 1e-14;
                        inside_patch = inside_patch && (x_lower <= X[d] && X[d] <= x_upper);
                    }
                    if (inside_patch)
                    {
                        node_ranks[n->id()] = rank + 1;
                    }
                    else
                    {
                        // Points are allowed to be outside the domain - they simply
                        // are no longer used for IB calculations.
                        bool inside_hier = true;
                        for (unsigned int d = 0; d < NDIM; ++d)
                        {
                            // Like above - permit points very close to the boundary
                            // to pass the check and treat them as being outside
                            const double x_lower = hier_x_lower[d] + std::max(1.0, std::abs(hier_x_lower[d])) * 1e-14;
                            const double x_upper = hier_x_upper[d] - std::max(1.0, std::abs(hier_x_upper[d])) * 1e-14;
                            inside_hier = inside_hier && (x_lower <= X[d] && X[d] <= x_upper);
                        }
                        if (!inside_hier)
                        {
                            node_ranks[n->id()] = n_procs + 1;
                        }
                    }
                }
            }
        }
        X_petsc_vec->restore_array();

        // send everything to rank 0 instead of doing an all-to-all:
        IBTK_MPI::allToOneSumReduction(node_ranks.data(), node_ranks.size());
        if (rank == 0)
        {
            const std::string message =
                "At least one node in the current mesh is inside the fluid domain and not associated with any "
                "patch. This class currently assumes that all elements are on the finest level and will not "
                "work correctly if this assumption does not hold. This usually happens when you use multiple "
                "patch levels and set the regrid CFL interval to a value larger than one. To change this check "
                "set node_outside_patch_check to a different value in the input database: see the documentation "
                "of FEMeshPartitioner for more information.";
            for (const int node_rank : node_ranks)
            {
                if (node_rank == 0)
                {
                    switch (d_node_patch_check)
                    {
                    case NODE_OUTSIDE_PERMIT:
                        break;
                    case NODE_OUTSIDE_WARN:
                        TBOX_WARNING(message);
                        break;
                    case NODE_OUTSIDE_ERROR:
                        TBOX_ERROR(message);
                        break;
                    default:
                        // we shouldn't get here
                        TBOX_ERROR("unrecognized value for d_node_patch_check");
                        break;
                    }
                    // no need to check more nodes if we already found one outside
                    break;
                }
            }
        }
    }

    IBTK_TIMER_STOP(t_reinit_element_mappings);
    return;
} // reinitElementMappings

NumericVector<double>*
FEMeshPartitioner::getSolutionVector(const std::string& system_name) const
{
    return d_fe_data->getEquationSystems()->get_system(system_name).solution.get();
} // getSolutionVector

NumericVector<double>*
FEMeshPartitioner::buildGhostedSolutionVector(const std::string& system_name, const bool localize_data)
{
    IBTK_TIMER_START(t_build_ghosted_solution_vector);

    reinitializeIBGhostedDOFs(system_name);
    NumericVector<double>* sol_vec = getSolutionVector(system_name);
    TBOX_ASSERT(sol_vec);
    if (!d_system_ghost_vec.count(system_name))
    {
        if (d_enable_logging)
        {
            plog << "FEMeshPartitioner::buildGhostedSolutionVector(): building ghosted solution vector for system: "
                 << system_name << "\n";
        }
        TBOX_ASSERT(d_active_patch_ghost_dofs.count(system_name));
        std::unique_ptr<NumericVector<double>> sol_ghost_vec = NumericVector<double>::build(sol_vec->comm());
        sol_ghost_vec->init(
            sol_vec->size(), sol_vec->local_size(), d_active_patch_ghost_dofs[system_name], true, GHOSTED);
        d_system_ghost_vec[system_name] = std::move(sol_ghost_vec);
    }
    NumericVector<double>* sol_ghost_vec = d_system_ghost_vec[system_name].get();
    if (localize_data) copy_and_synch(*sol_vec, *sol_ghost_vec, /*close_v_in*/ false);

    IBTK_TIMER_STOP(t_build_ghosted_solution_vector);
    return sol_ghost_vec;
} // buildGhostedSolutionVector

std::unique_ptr<PetscVector<double>>
FEMeshPartitioner::buildIBGhostedVector(const std::string& system_name)
{
    IBTK_TIMER_START(t_build_ghosted_vector);

    reinitializeIBGhostedDOFs(system_name);
    TBOX_ASSERT(d_system_ib_ghost_vec.find(system_name) != d_system_ib_ghost_vec.end());
    const std::unique_ptr<PetscVector<double>>& exemplar_ib_vector = d_system_ib_ghost_vec.at(system_name);
    TBOX_ASSERT(exemplar_ib_vector);
    // Sanity check:
    const System& system = d_fe_data->getEquationSystems()->get_system(system_name);
    if (system.solution->local_size() != exemplar_ib_vector->local_size())
    {
        TBOX_ERROR(
            "The locally stored example IB vector does not have the same local size as the relevant system's solution "
            "vector. This usually occurs when the finite element data has been modified (e.g., the mesh has been "
            "refined) outside of one of IBAMR's classes (like FEDataManager or IBFEMethod).");
    }

    std::unique_ptr<NumericVector<double>> clone = exemplar_ib_vector->zero_clone();
    auto ptr = dynamic_cast<PetscVector<double>*>(clone.release());
    TBOX_ASSERT(ptr);

    IBTK_TIMER_STOP(t_build_ghosted_vector);
    return std::unique_ptr<PetscVector<double>>(ptr);
}

NumericVector<double>*
FEMeshPartitioner::getCoordsVector() const
{
    return getSolutionVector(COORDINATES_SYSTEM_NAME);
} // getCoordsVector

NumericVector<double>*
FEMeshPartitioner::buildGhostedCoordsVector(const bool localize_data)
{
    return buildGhostedSolutionVector(COORDINATES_SYSTEM_NAME, localize_data);
} // buildGhostedCoordsVector

NumericVector<double>*
FEMeshPartitioner::buildDiagonalL2MassMatrix(const std::string& system_name)
{
    return d_fe_projector->buildDiagonalL2MassMatrix(system_name);
} // buildDiagonalL2MassMatrix

PetscVector<double>*
FEMeshPartitioner::buildIBGhostedDiagonalL2MassMatrix(const std::string& system_name)
{
    if (!d_L2_proj_matrix_diag_ghost.count(system_name))
    {
        std::unique_ptr<PetscVector<double>> M_vec = buildIBGhostedVector(system_name);
        *M_vec = *d_fe_projector->buildDiagonalL2MassMatrix(system_name);
        M_vec->close();
        d_L2_proj_matrix_diag_ghost[system_name] = std::move(M_vec);
    }
    return d_L2_proj_matrix_diag_ghost[system_name].get();
}

bool
FEMeshPartitioner::computeL2Projection(NumericVector<double>& U_vec,
                                       NumericVector<double>& F_vec,
                                       const std::string& system_name,
                                       const bool consistent_mass_matrix,
                                       const bool close_U,
                                       const bool close_F,
                                       const double tol,
                                       const unsigned int max_its)
{
    return d_fe_projector->computeL2Projection(*static_cast<PetscVector<double>*>(&U_vec),
                                               *static_cast<PetscVector<double>*>(&F_vec),
                                               system_name,
                                               consistent_mass_matrix,
                                               close_U,
                                               close_F,
                                               tol,
                                               max_its);
} // computeL2Projection

std::shared_ptr<FEData>&
FEMeshPartitioner::getFEData()
{
    return d_fe_data;
} // getFEData

const std::shared_ptr<FEData>&
FEMeshPartitioner::getFEData() const
{
    return d_fe_data;
} // getFEData
/////////////////////////////// PROTECTED ////////////////////////////////////

void
FEMeshPartitioner::setLoggingEnabled(bool enable_logging)
{
    d_enable_logging = enable_logging;
    return;
} // setLoggingEnabled

bool
FEMeshPartitioner::getLoggingEnabled() const
{
    return d_enable_logging;
} // getLoggingEnabled

/////////////////////////////// PRIVATE //////////////////////////////////////

int
FEMeshPartitioner::getPatchLevel(const Elem* elem) const
{
    return d_level_lookup[elem->subdomain_id()];
}

void
FEMeshPartitioner::reinitializeIBGhostedDOFs(const std::string& system_name)
{
    // Reset the sets of dofs corresponding to IB ghost data. This is usually
    // a superset of the standard (i.e., all dofs on unowned cells adjacent to
    // locally owned cells) ghost data.
    if (!d_active_patch_ghost_dofs.count(system_name))
    {
        const System& system = d_fe_data->getEquationSystems()->get_system(system_name);
        std::vector<libMesh::dof_id_type> ib_ghost_dofs;
        collectGhostDOFIndices(ib_ghost_dofs, d_active_nodes, system_name, d_fe_data);

        // Match the expected vector sizes by using the solution for non-ghost
        // sizes:
        const NumericVector<double>& solution = *system.solution;
        std::unique_ptr<PetscVector<double>> exemplar_ib_vector(new PetscVector<double>(
            system.comm(), solution.size(), solution.local_size(), ib_ghost_dofs, libMesh::GHOSTED));
        d_active_patch_ghost_dofs[system_name] = std::move(ib_ghost_dofs);
        d_system_ib_ghost_vec[system_name] = std::move(exemplar_ib_vector);
    }
}
/////////////////////////////// NAMESPACE ////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
