/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_RBFFDWeightsCache
#define included_ADS_RBFFDWeightsCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ADS/FEMeshPartitioner.h"
#include <ADS/FDPoint.h>
#include <ADS/FDWeightsCache.h>

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/LaplaceOperator.h"
#include "ibtk/ibtk_utilities.h"

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/dof_map.h"
#include "libmesh/node.h"

#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class RBFFDWeightsCache is an implementation of FDWeightsCache that computes RBF-FD weights at all points
 * within a specified distance of the boundary described by the mesh partitioner. Three functions must be registered.
 * The first is the RBF evaluated at a point r. The other two are the operator applied to the RBF evaluated at the point
 * and a function that takes input a vector of inputs and returns the Vandermonde matrix of the linear operator applied
 * to the monomials.
 *
 * Note: This class only supports LINEAR operators.
 */
class RBFFDWeightsCache : public FDWeightsCache
{
public:
    /*!
     * \brief Constructor for class LaplaceOperator initializes the operator
     * coefficients and boundary conditions to default values.
     */
    RBFFDWeightsCache(std::string object_name,
                      std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                      SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                      SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    virtual ~RBFFDWeightsCache();

    /*!
     * \brief Set the level set
     */
    inline void setLS(int ls_idx)
    {
        d_ls_idx = ls_idx;
    }

    void
    registerPolyFcn(std::function<IBTK::MatrixXd(const std::vector<FDPoint>&, int, double, const FDPoint&)> poly_fcn,
                    std::function<double(double)> rbf_fcn,
                    std::function<double(double)> Lrbf_fcn)
    {
        d_poly_fcn = poly_fcn;
        d_rbf_fcn = rbf_fcn;
        d_Lrbf_fcn = Lrbf_fcn;
    }

    virtual void clearCache() override;
    void sortLagDOFsToCells();
    void findRBFFDWeights();

    void setNumGhostCells(int num_ghost_cells)
    {
        d_num_ghost_cells = num_ghost_cells;
        if (d_num_ghost_cells < d_stencil_size)
            TBOX_WARNING(
                "Number of ghost cells is less than stencil size. This could force one-sided stencils near patch "
                "boundaries.\n");
    }

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    RBFFDWeightsCache() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    RBFFDWeightsCache(const RBFFDWeightsCache& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    RBFFDWeightsCache& operator=(const RBFFDWeightsCache& that) = delete;

    std::string d_object_name;

    int d_num_ghost_cells = 3;
    double d_eps = std::numeric_limits<double>::quiet_NaN();

    int d_ls_idx = IBTK::invalid_index;
    double d_dist_to_bdry = std::numeric_limits<double>::quiet_NaN();
    int d_poly_degree = 3;
    int d_stencil_size = 2 * NDIM + 1;

    // Hierarchy configuration.
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    // Lag structure info
    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;
    std::map<SAMRAI::hier::Patch<NDIM>*, std::vector<libMesh::Node*>> d_idx_node_vec;

    std::function<IBTK::MatrixXd(const std::vector<FDPoint>&, int, double, const FDPoint&)> d_poly_fcn;
    std::function<double(double)> d_rbf_fcn, d_Lrbf_fcn;

    bool d_weights_found = false;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_RBFFDWeightsCache
