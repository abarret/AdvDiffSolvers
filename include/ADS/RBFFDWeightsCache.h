/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_RBFFDWeightsCache
#define included_ADS_RBFFDWeightsCache

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ADS/FEMeshPartitioner.h"

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
struct UPoint
{
public:
    UPoint(const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>& patch, const SAMRAI::pdat::CellIndex<NDIM>& idx)
        : d_idx(idx)
    {
        SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
        const double* const dx = pgeom->getDx();
        const double* const xlow = pgeom->getXLower();
        const SAMRAI::hier::Index<NDIM>& idx_low = patch->getBox().lower();
        for (unsigned int d = 0; d < NDIM; ++d)
            d_pt[d] = xlow[d] + dx[d] * (static_cast<double>(d_idx(d) - idx_low(d)) + 0.5);
    };

    UPoint(const IBTK::VectorNd& pt, libMesh::Node* node)
        : d_pt(pt),
          d_node(node){
              // intentionally blank
          };

    UPoint() : d_empty(true)
    {
        // intentionally blank
    }

    UPoint(const std::vector<double>& pt) : d_empty(true)
    {
        for (unsigned int d = 0; d < NDIM; ++d) d_pt[d] = pt[d];
    }

    double dist(const IBTK::VectorNd& x) const
    {
        return (d_pt - x).norm();
    }

    double dist(const UPoint& pt) const
    {
        return (d_pt - pt.getVec()).norm();
    }

    double operator()(const size_t i) const
    {
        return d_pt(i);
    }

    double operator[](const size_t i) const
    {
        return d_pt[i];
    }

    friend std::ostream& operator<<(std::ostream& out, const UPoint& pt)
    {
        out << "   location: " << pt.d_pt.transpose() << "\n";
        if (pt.isNode())
            out << "   node id: " << pt.d_node->id();
        else if (!pt.isEmpty())
            out << "   idx:     " << pt.d_idx;
        else
            out << "   pt is neither node nor index";
        return out;
    }

    friend bool operator==(const UPoint& lhs, const UPoint& rhs)
    {
        if (lhs.isNode() && rhs.isNode())
        {
            return lhs.d_node == rhs.d_node;
        }
        else if (lhs.isIdx() && rhs.isIdx())
        {
            return lhs.d_idx == rhs.d_idx;
        }
        else
        {
            return false;
        }
    }

    bool isEmpty() const
    {
        return d_empty;
    }

    bool isNode() const
    {
        return d_node != nullptr;
    }

    bool isIdx() const
    {
        return !isNode() && !isEmpty();
    }

    const libMesh::Node* const getNode() const
    {
        if (!isNode()) TBOX_ERROR("Not at a node\n");
        return d_node;
    }

    const SAMRAI::pdat::CellIndex<NDIM>& getIndex() const
    {
        if (isNode()) TBOX_ERROR("At at node\n");
        if (d_empty) TBOX_ERROR("Not a point\n");
        return d_idx;
    }

    const IBTK::VectorNd& getVec() const
    {
        return d_pt;
    }

private:
    IBTK::VectorNd d_pt;
    libMesh::Node* d_node = nullptr;
    SAMRAI::pdat::CellIndex<NDIM> d_idx;
    bool d_empty = false;
};

/*!
 * \brief Class RBFFDWeightsCache is a class that caches finite difference weights for general operators. This class
 * requires a fe_mesh_partitioner and a level set function to determine which cells need finite difference weights.
 * Three functions must be registered. The first is the RBF evaluated at a point r. The other two are the operator
 * applied to the RBF evaluated at the point and a function that takes input a vector of inputs and returns the
 * Vandermonde matrix of the linear operator applied to the monomials.*/
class RBFFDWeightsCache
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
    ~RBFFDWeightsCache();

    /*!
     * \brief Set the level set
     */
    inline void setLS(int ls_idx)
    {
        d_ls_idx = ls_idx;
    }

    const std::vector<std::vector<double>>& getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);
    const std::vector<std::vector<UPoint>>& getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);
    const std::vector<UPoint>& getRBFFDBasePoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    const std::vector<double>& getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                               const UPoint& pt);
    const std::vector<UPoint>& getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const UPoint& pt);
    bool isRBFFDBasePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> ptach, const UPoint& pt);

    void registerPolyFcn(std::function<IBTK::MatrixXd(const std::vector<IBTK::VectorNd>&, int)> poly_fcn,
                         std::function<double(double)> rbf_fcn,
                         std::function<double(double)> Lrbf_fcn)
    {
        d_poly_fcn = poly_fcn;
        d_rbf_fcn = rbf_fcn;
        d_Lrbf_fcn = Lrbf_fcn;
    }

    void clearCache();
    void sortLagDOFsToCells();
    void findRBFFDWeights();

    /*!
     * Debugging functions
     */
    void printPtMap(std::ostream& os);

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

    static unsigned int s_num_ghost_cells;
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
    std::map<SAMRAI::hier::Patch<NDIM>*, std::vector<libMesh::Node*>> d_idx_node_vec, d_idx_node_ghost_vec;

    // Weight and point information
    using PtVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::vector<UPoint>>;
    using PtPairVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::vector<std::vector<UPoint>>>;
    using WeightVecMap = std::map<SAMRAI::hier::Patch<NDIM>*, std::vector<std::vector<double>>>;
    PtVecMap d_base_pt_vec;
    PtPairVecMap d_pair_pt_vec;
    WeightVecMap d_pt_weight_vec;

    std::function<IBTK::MatrixXd(const std::vector<IBTK::VectorNd>&, int)> d_poly_fcn;
    std::function<double(double)> d_rbf_fcn, d_Lrbf_fcn;

    bool d_weights_found = false;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_RBFFDWeightsCache
