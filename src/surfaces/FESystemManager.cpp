/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/FESystemManager.h"
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
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>

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

FESystemManager::FESystemManager(std::string object_name,
                                 const Pointer<Database>& input_db,
                                 std::shared_ptr<FEData> fe_data,
                                 std::string coords_sys_name)
    : COORDINATES_SYSTEM_NAME(std::move(coords_sys_name)),
      d_fe_data(fe_data),
      d_fe_projector(new FEProjector(d_fe_data, setup_fe_projector_db(input_db))),
      d_object_name(std::move(object_name))
{
    TBOX_ASSERT(!d_object_name.empty());

    // Setup Timers.
    IBTK_DO_ONCE(
        t_reinit_element_mappings =
            TimerManager::getManager()->getTimer("ADS::FESystemManager::reinitElementMappings()");
        t_build_ghosted_solution_vector =
            TimerManager::getManager()->getTimer("ADS::FESystemManager::buildGhostedSolutionVector()");
        t_build_ghosted_vector = TimerManager::getManager()->getTimer("ADS::FESystemManager::buildGhostedVector()");
        t_update_workload_estimates =
            TimerManager::getManager()->getTimer("ADS::FESystemManager::updateWorkloadEstimates()");
        t_initialize_level_data = TimerManager::getManager()->getTimer("ADS::FESystemManager::initializeLevelData()");
        t_reset_hierarchy_configuration =
            TimerManager::getManager()->getTimer("ADS::FESystemManager::resetHierarchyConfiguration()");
        t_apply_gradient_detector =
            TimerManager::getManager()->getTimer("ADS::FESystemManager::applyGradientDetector()");)
    return;
} // FESystemManager

const std::string&
FESystemManager::getCoordsSystemName() const
{
    return COORDINATES_SYSTEM_NAME;
}

EquationSystems*
FESystemManager::getEquationSystems() const
{
    return d_fe_data->getEquationSystems();
} // getEquationSystems

FESystemManager::SystemDofMapCache*
FESystemManager::getDofMapCache(const std::string& system_name)
{
    return d_fe_data->getDofMapCache(system_name);
} // getDofMapCache

FESystemManager::SystemDofMapCache*
FESystemManager::getDofMapCache(unsigned int system_num)
{
    return d_fe_data->getDofMapCache(system_num);
} // getDofMapCache

NumericVector<double>*
FESystemManager::getSolutionVector(const std::string& system_name) const
{
    return d_fe_data->getEquationSystems()->get_system(system_name).solution.get();
} // getSolutionVector

NumericVector<double>*
FESystemManager::getCoordsVector() const
{
    return getSolutionVector(COORDINATES_SYSTEM_NAME);
} // getCoordsVector

NumericVector<double>*
FESystemManager::buildDiagonalL2MassMatrix(const std::string& system_name)
{
    return d_fe_projector->buildDiagonalL2MassMatrix(system_name);
} // buildDiagonalL2MassMatrix

bool
FESystemManager::computeL2Projection(NumericVector<double>& U_vec,
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

void
FESystemManager::clearPatchHierarchyDependentData()
{
    d_fe_data->clearPatchHierarchyDependentData();
}

std::shared_ptr<FEData>&
FESystemManager::getFEData()
{
    return d_fe_data;
} // getFEData

const std::shared_ptr<FEData>&
FESystemManager::getFEData() const
{
    return d_fe_data;
} // getFEData

void
FESystemManager::setLoggingEnabled(bool enable_logging)
{
    d_enable_logging = enable_logging;
    return;
} // setLoggingEnabled

bool
FESystemManager::getLoggingEnabled() const
{
    return d_enable_logging;
} // getLoggingEnabled

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
