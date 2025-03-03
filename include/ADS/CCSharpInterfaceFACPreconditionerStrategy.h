#ifndef included_ADS_CCSharpInterfaceFACPreconditionerStrategy
#define included_ADS_CCSharpInterfaceFACPreconditionerStrategy

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ADS/SharpInterfaceGhostFill.h>

#include "ibtk/CoarseFineBoundaryRefinePatchStrategy.h"
#include "ibtk/FACPreconditionerStrategy.h"
#include "ibtk/RobinPhysBdryPatchStrategy.h"
#include "ibtk/ibtk_utilities.h"

#include "CoarsenAlgorithm.h"
#include "CoarsenOperator.h"
#include "IntVector.h"
#include "PatchHierarchy.h"
#include "PoissonSpecifications.h"
#include "RefineAlgorithm.h"
#include "RefineOperator.h"
#include "RefinePatchStrategy.h"
#include "SAMRAIVectorReal.h"
#include "VariableContext.h"
#include "VariableFillPattern.h"
#include "tbox/Pointer.h"

#include <memory>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{

namespace sharp_interface
{
/*!
 * \brief Class CCSharpInterfaceFACPreconditionerStrategy is an abstract
 * FACPreconditionerStrategy implementing many of the operations required by
 * smoothers for the Poisson equation and related problems.
 *
 * Sample parameters for initialization from database (and their default
 * values): \verbatim

 smoother_type = "DEFAULT"                    // see setSmootherType()
 prolongation_method = "DEFAULT"              // see setProlongationMethod()
 restriction_method = "DEFAULT"               // see setRestrictionMethod()
 coarse_solver_type = "DEFAULT"               // see setCoarseSolverType()
 coarse_solver_rel_residual_tol = 1.0e-5      // see setCoarseSolverRelativeTolerance()
 coarse_solver_abs_residual_tol = 1.0e-50     // see setCoarseSolverAbsoluteTolerance()
 coarse_solver_max_iterations = 10            // see setCoarseSolverMaxIterations()
 \endverbatim
*/
class CCSharpInterfaceFACPreconditionerStrategy : public IBTK::FACPreconditionerStrategy
{
public:
    /*!
     * \brief Constructor.
     */
    CCSharpInterfaceFACPreconditionerStrategy(std::string object_name,
                                              std::vector<FESystemManager*> fe_sys_managers,
                                              SAMRAI::tbox::Database* input_db);

    /*!
     * \brief Destructor.
     */
    ~CCSharpInterfaceFACPreconditionerStrategy();

    /*!
     * \brief Set the SAMRAI::solv::RobinBcCoefStrategy object used to specify
     * physical boundary conditions.
     *
     * \note \a bc_coef may be NULL.  In this case, default boundary conditions
     * (as supplied to the class constructor) are employed.
     *
     * \param bc_coef  Pointer to an object that can set the Robin boundary condition
     *coefficients
     */
    virtual void setPhysicalBcCoef(SAMRAI::solv::RobinBcCoefStrategy<NDIM>* bc_coef);

    /*!
     * \brief Set the SAMRAI::solv::RobinBcCoefStrategy objects used to specify
     * physical boundary conditions.
     *
     * \note Any of the elements of \a bc_coefs may be NULL.  In this case,
     * default boundary conditions (as supplied to the class constructor) are
     * employed for that data depth.
     *
     * \param bc_coefs  Vector of pointers to objects that can set the Robin boundary condition
     *coefficients
     */
    virtual void setPhysicalBcCoefs(const std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*>& bc_coefs);

    /*!
     * \name Partial implementation of FACPreconditionerStrategy interface.
     */
    //\{

    /*!
     * \brief Zero the supplied vector.
     */
    void setToZero(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& vec, int level_num) override;

    /*!
     * \brief Restrict the residual quantity to the specified level from the
     * next finer level.
     *
     * \param src source residual
     * \param dst destination residual
     * \param dst_ln destination level number
     */
    void restrictResidual(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& src,
                          SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& dst,
                          int dst_ln) override;

    /*!
     * \brief Prolong the error quantity to the specified level from the next
     * coarser level.
     *
     * \param src source error vector
     * \param dst destination error vector
     * \param dst_ln destination level number of data transfer
     */
    void prolongError(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& src,
                      SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& dst,
                      int dst_ln) override;

    /*!
     * \brief Prolong the error quantity to the specified level from the next
     * coarser level and apply the correction to the fine-level error.
     *
     * \param src source error vector
     * \param dst destination error vector
     * \param dst_ln destination level number of data transfer
     */
    void prolongErrorAndCorrect(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& src,
                                SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& dst,
                                int dst_ln) override;

    /*!
     * \brief Compute hierarchy-dependent data.
     *
     * Note that although the vector arguments given to other methods in this
     * class may not necessarily be the same as those given to this method,
     * there will be similarities, including:
     *
     * - hierarchy configuration (hierarchy pointer and level range)
     * - number, type and alignment of vector component data
     * - ghost cell width of data in the solution (or solution-like) vector
     *
     * \param solution solution vector u
     * \param rhs right hand side vector f
     */
    void initializeOperatorState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& solution,
                                 const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& rhs) override;

    /*!
     * \brief Remove all hierarchy-dependent data.
     *
     * Remove all hierarchy-dependent data set by initializeOperatorState().
     *
     * \see initializeOperatorState
     */
    void deallocateOperatorState() override;

    void smoothError(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& error,
                     const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& residual,
                     int level_num,
                     int num_sweeps,
                     bool performing_pre_sweeps,
                     bool performing_post_sweeps) override;

    void computeResidual(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& residual,
                         const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& solution,
                         const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& rhs,
                         int coarsest_level_num,
                         int finest_level_num) override;

    bool solveCoarsestLevel(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& error,
                            const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& residual,
                            int coarsest_ln) override;
    //\}

    inline void setBdryFcn(std::function<double(const IBTK::VectorNd& x)> bdry_fcn)
    {
        d_bdry_fcn = bdry_fcn;
    }

private:
    /*!
     * \name Methods for executing, caching, and resetting communication
     * schedules.
     */
    //\{

    /*!
     * \brief Execute a refinement schedule for prolonging data.
     */
    void xeqScheduleProlongation(int dst_idx, int src_idx, int dst_ln);

    /*!
     * \brief Execute schedule for restricting solution or residual to the
     * specified level.
     */
    void xeqScheduleRestriction(int dst_idx, int src_idx, int dst_ln);

    /*!
     * \brief Execute schedule for filling ghosts on the specified level.
     */
    void xeqScheduleGhostFillNoCoarse(int dst_idx, int dst_ln);

    /*!
     * \brief Execute schedule for synchronizing data on the specified level.
     */
    void xeqScheduleDataSynch(int dst_idx, int dst_ln);

    //\}

    /*
     * Problem specification.
     */
    std::unique_ptr<SAMRAI::solv::RobinBcCoefStrategy<NDIM>> d_default_bc_coef;
    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_bc_coefs;

    /*
     * Ghost cell width.
     */
    SAMRAI::hier::IntVector<NDIM> d_gcw;

    /*!
     * \name Hierarchy-dependent objects.
     */
    //\{

    /*
     * Solution and rhs vectors.
     */
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_solution, d_rhs;

    /*
     * Reference patch hierarchy and range of levels involved in the solve.
     *
     * This variable is non-null between the initializeOperatorState() and
     * deallocateOperatorState() calls.  It is not truly needed, because the
     * hierarchy is obtainable through variables in most function argument
     * lists.  We use it to enforce working on one hierarchy at a time.
     */
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_coarsest_ln = IBTK::invalid_level_number, d_finest_ln = IBTK::invalid_level_number;

    /*
     * Patch overlap data.
     */
    std::vector<std::vector<SAMRAI::hier::BoxList<NDIM>>> d_patch_bc_box_overlap;

    /*
     * HierarchyDataOpsReal objects restricted to a single level of the patch
     * hierarchy.
     */
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::math::HierarchyDataOpsReal<NDIM, double>>> d_level_data_ops;

    //\}

    /*!
     * \name Internal context and scratch data.
     */
    //\{

    /*
     * Variable context for internally maintained hierarchy data.
     */
    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> d_context;

    /*
     * Patch descriptor variable and index for scratch data.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_scratch_var;
    int d_scratch_idx = IBTK::invalid_index;

    //\}

    /*!
     * \name Various refine and coarsen objects.
     */
    //\{

    /*
     * Physical boundary operators.
     */
    SAMRAI::tbox::Pointer<IBTK::RobinPhysBdryPatchStrategy> d_bc_op;

    /*
     * Coarse-fine interface interpolation objects.
     */
    SAMRAI::tbox::Pointer<IBTK::CoarseFineBoundaryRefinePatchStrategy> d_cf_bdry_op;

    /*
     * The names of the refinement operators used to prolong the coarse grid
     * correction.
     */
    std::string d_prolongation_method = "LINEAR_REFINE";

    /*
     * The names of the coarsening operators used to restrict the fine grid
     * error or residual.
     */
    std::string d_restriction_method = "CONSERVATIVE_COARSEN";

    /*
     * Variable fill pattern object.
     */
    SAMRAI::tbox::Pointer<SAMRAI::xfer::VariableFillPattern<NDIM>> d_synch_fill_pattern;

    //\}

    /*!
     * \name Various refine and coarsen objects.
     */
    //\{

    /*
     * Error prolongation (refinement) operator.
     */
    SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineOperator<NDIM>> d_prolongation_refine_operator;
    SAMRAI::tbox::Pointer<SAMRAI::xfer::RefinePatchStrategy<NDIM>> d_prolongation_refine_patch_strategy;
    SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineAlgorithm<NDIM>> d_prolongation_refine_algorithm;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineSchedule<NDIM>>> d_prolongation_refine_schedules;

    /*
     * Residual restriction (coarsening) operator.
     */
    SAMRAI::tbox::Pointer<SAMRAI::xfer::CoarsenOperator<NDIM>> d_restriction_coarsen_operator;
    SAMRAI::tbox::Pointer<SAMRAI::xfer::CoarsenAlgorithm<NDIM>> d_restriction_coarsen_algorithm;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::xfer::CoarsenSchedule<NDIM>>> d_restriction_coarsen_schedules;

    /*
     * Refine operator for cell data from same level.
     */
    SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineAlgorithm<NDIM>> d_ghostfill_nocoarse_refine_algorithm;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineSchedule<NDIM>>> d_ghostfill_nocoarse_refine_schedules;

    /*
     * Operator for data synchronization on same level.
     */
    SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineAlgorithm<NDIM>> d_synch_refine_algorithm;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::xfer::RefineSchedule<NDIM>>> d_synch_refine_schedules;

    //\}

    // Operators for sharp interface ghost filling
    // Note we need to create the CutCellMapping object on the dense hierarchy before we can create this.
    std::unique_ptr<SharpInterfaceGhostFill> d_si_ghost_fill;
    std::vector<FESystemManager*> d_fe_sys_managers;
    std::vector<std::unique_ptr<FEToHierarchyMapping>> d_fe_hierarchy_mapping;
    SAMRAI::tbox::Pointer<IndexElemMapping> d_idx_elem_mapping;

    // Boundary conditions for operator.
    // TODO: This needs to be more general
    std::function<double(const IBTK::VectorNd&)> d_bdry_fcn;

    // Relaxation parameter
    double d_w = 1.0;

    // Number of sweeps on coarsest level
    int d_num_sweeps_on_coarsest_level = 20;
};
} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_ADS_CCSharpInterfaceFACPreconditionerStrategy
