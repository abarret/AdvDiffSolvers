#ifndef included_ADS_utility_functions
#define included_ADS_utility_functions
#include "ADS/ls_utilities.h"

#include "ibtk/ibtk_utilities.h"

#include "NodeData.h"
#include "Variable.h"
#include "tbox/MathUtilities.h"

#include "libmesh/elem.h"
#include "libmesh/vector_value.h"

#include "boost/multi_array.hpp"

#include <Eigen/Dense>

namespace ADS
{
#define ADS_TIMER_START(timer) timer->start();

#define ADS_TIMER_STOP(timer) timer->stop();

using Simplex = std::array<std::pair<IBTK::VectorNd, double>, NDIM + 1>;

static double s_eps = 1.0e-12;

SAMRAI::pdat::NodeIndex<NDIM> get_node_index_from_corner(const SAMRAI::hier::Index<NDIM>& idx, int corner);

double length_fraction(double dx, double phi_l, double phi_u);

double area_fraction(double reg_area, double phi_ll, double phi_lu, double phi_uu, double phi_ul);

IBTK::VectorNd midpoint_value(const IBTK::VectorNd& pt0, double phi0, const IBTK::VectorNd& pt1, double phi1);

#if (NDIM == 2)
IBTK::VectorNd find_cell_centroid(const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);
#endif
#if (NDIM == 3)
// Slow, accurate computation of cell centroid
IBTK::VectorNd find_cell_centroid_slow(const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                       const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);

// Fast, but possibly wrong computation of cell centroid
IBTK::VectorNd find_cell_centroid(const SAMRAI::pdat::CellIndex<NDIM>& idx,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);
#endif

double node_to_cell(const SAMRAI::pdat::CellIndex<NDIM>& idx, const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);

double node_to_side(const SAMRAI::pdat::SideIndex<NDIM>& idx, const SAMRAI::pdat::NodeData<NDIM, double>& ls_data);

void copy_face_to_side(const int u_s_idx,
                       const int u_f_idx,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

bool find_intersection(libMesh::Point& p,
                       libMesh::Elem* elem,
                       const libMesh::Point& r,
                       const libMesh::VectorValue<double>& q);

std::string get_libmesh_restart_file_name(const std::string& restart_dump_dirname,
                                          const std::string& base_filename,
                                          unsigned int time_step_number,
                                          unsigned int part,
                                          const std::string& extension);

/// Functions for calculatin cell areas and volumes from level sets.
/*!
 * Find the volume and area of cell idx.
 *
 * Inputs:
 * xlow: Lower left point of cell
 * dx: pointer to NDIM length array of grid spacing
 * phi: Level set data in the form of nodal patch data
 * idx: index on which to find the volume
 *
 * Returns:
 * pair consisting of the volume (first) and the area (second)
 */
std::pair<double, double> find_volume_and_area(const IBTK::VectorNd& xlow,
                                               const double* const dx,
                                               SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeData<NDIM, double>> phi,
                                               const SAMRAI::pdat::CellIndex<NDIM>& idx);

/*!
 * Find the volume of a vector of simplices (tri in 2D, tet in 3D)
 */
double find_volume(const std::vector<Simplex>& simplices);

/*!
 * Find the surface area of a vector of simplices (tri in 2D, tet in 3D)
 */
double find_area(const std::vector<Simplex>& simplices);

/*!
 * Use a flood filling algorithm to compute the correct sign for the node centered level set on the provided patch
 * level. We assume that the sign of any values that are exactly equal to eps might need adjusting.
 *
 * This function requires that sgn_idx have at least one layer of ghost cells.
 *
 * Note: This function remains untested for levels that are not simply connected.
 */
void flood_fill_for_LS(int sgn_idx,
                       SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> sgn_var,
                       double eps,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level);

/*!
 * Use a flood filling algorithm to compute the correct sign for the cell centered level set on the provided patch
 * level. We assume that the sign of any values that are exactly equal to eps might need adjusting.
 *
 * This function requires that sgn_idx have at least one layer of ghost cells.
 *
 * Note: This function remains untested for levels that are not simply connected.
 */
void flood_fill_for_LS(int sgn_idx,
                       SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> sgn_var,
                       double eps,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level);
} // namespace ADS

#include <ADS/private/ls_functions_inc.h>
#endif
