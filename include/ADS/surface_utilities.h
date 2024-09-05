#ifndef included_ADS_surface_utilities
#define included_ADS_surface_utilities

#include <ADS/FEToHierarchyMapping.h>

#include <libmesh/equation_systems.h>

namespace ADS
{
/*!
 * \brief Update the Jacobian of the deformation map.
 *
 * The Jacobian is stored in the system given by J_sys_name and must be a valid ExplicitSystem stored in the
 * EquationSystems object manager by the fe_partitioner.
 *
 * Uses a fifth order Gaussian quadrature rule regardless of the finite element type.
 *
 * We assume the physical coordinates of the mesh elements are the reference configuration for the mesh.
 */
void update_jacobian(const std::string& J_sys_name, FESystemManager& fe_partitioner);

} // namespace ADS
#endif
