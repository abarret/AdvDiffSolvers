/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_RBFLaplaceOperator
#define included_ADS_RBFLaplaceOperator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/config.h>

#include "ADS/PETScAugmentedLinearOperator.h"
#include "ADS/RBFFDWeightsCache.h"

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

#include <memory>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class RBFLaplaceOperator is a concrete LaplaceOperator which implements
 * a globally second-order accurate cell-centered finite difference
 * discretization of a scalar elliptic operator of the form \f$ L = C I + \nabla
 * \cdot D \nabla\f$.
 */
class RBFLaplaceOperator : public PETScAugmentedLinearOperator
{
public:
    /*!
     * \brief Constructor for class LaplaceOperator initializes the operator
     * coefficients and boundary conditions to default values.
     */
    RBFLaplaceOperator(const std::string& object_name,
                       std::shared_ptr<FEMeshPartitioner> fe_mesh_partitioner,
                       const std::string& sys_name,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~RBFLaplaceOperator();

    /*!
     * \brief Set the level set
     */
    inline void setLS(int ls_idx)
    {
        d_ls_idx = ls_idx;
    }

    void setupBeforeApply(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                          SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& y);
    void fillBdryConds();

    /*!
     * \name Linear operator functionality.
     */
    //\{

    /*!
     * \brief Compute y=Ax.
     *
     * Before calling this function, the form of the vectors x and y should be
     * set properly by the user on all patch interiors on the range of levels
     * covered by the operator.  All data in these vectors should be allocated.
     * Thus, the user is responsible for managing the storage for the vectors.
     *
     * Conditions on arguments:
     * - vectors must have same hierarchy
     * - vectors must have same variables (except that x \em must
     * have enough ghost cells for computation of Ax).
     *
     * \note In general, the vectors x and y \em cannot be the same.
     *
     * Upon return from this function, the y vector will contain the result of
     * the application of A to x.
     *
     * initializeOperatorState must be called prior to any calls to
     * applyOperator.
     *
     * \see initializeOperatorState
     *
     * \param x input
     * \param y output: y=Ax
     */
    void apply(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
               SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& y) override;

    /*!
     * \brief Compute hierarchy-dependent data required for computing y=Ax (and
     * y=A'x).
     *
     * \param in input vector
     * \param out output vector
     *
     * \see KrylovLinearSolver::initializeSolverState
     */
    void initializeOperatorState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& in,
                                 const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& out) override;

    /*!
     * \brief Remove all hierarchy-dependent data computed by
     * initializeOperatorState().
     *
     * Remove all hierarchy-dependent data set by initializeOperatorState().  It
     * is safe to call deallocateOperatorState() even if the state is already
     * deallocated.
     *
     * \see initializeOperatorState
     * \see KrylovLinearSolver::deallocateSolverState
     */
    void deallocateOperatorState() override;

    //\}

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    RBFLaplaceOperator() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    RBFLaplaceOperator(const RBFLaplaceOperator& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    RBFLaplaceOperator& operator=(const RBFLaplaceOperator& that) = delete;

    void applyToLagDOFs(int x_idx, int y_idx);

    double getSolVal(const FDCachedPoint& pt, const SAMRAI::pdat::CellData<NDIM, double>& Q_data, Vec& vec) const;
    void setSolVal(double q, const FDCachedPoint& pt, SAMRAI::pdat::CellData<NDIM, double>& Q_data, Vec& vec) const;

    // Operator parameters.
    int d_ncomp = 0;

    // Cached communications operators.
    SAMRAI::tbox::Pointer<SAMRAI::xfer::VariableFillPattern<NDIM>> d_fill_pattern;
    std::vector<IBTK::HierarchyGhostCellInterpolation::InterpolationTransactionComponent> d_transaction_comps;
    SAMRAI::tbox::Pointer<IBTK::HierarchyGhostCellInterpolation> d_hier_bdry_fill, d_no_fill;

    // Scratch data.
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_x, d_b;

    int d_ls_idx = IBTK::invalid_index;
    double d_dist_to_bdry = std::numeric_limits<double>::quiet_NaN();

    // Hierarchy configuration.
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    // Lag structure info
    std::shared_ptr<FEMeshPartitioner> d_fe_mesh_partitioner;
    std::string d_sys_name = "";

    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_bc_coefs;

    double d_C = std::numeric_limits<double>::quiet_NaN();
    double d_D = std::numeric_limits<double>::quiet_NaN();

    std::unique_ptr<RBFFDWeightsCache> d_fd_weights;

    std::function<double(double)> d_rbf, d_lap_rbf;
    std::function<IBTK::MatrixXd(std::vector<IBTK::VectorNd>, int, double, const IBTK::VectorNd&)> d_polys;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_ADS_RBFLaplaceOperator
