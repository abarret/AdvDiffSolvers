/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_AdvDiffImplicitIntegrator
#define included_ADS_AdvDiffImplicitIntegrator

/////////////////////////////// INCLUDES /////////////////////////////////////
#include "ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h"
#include "ibamr/ibamr_enums.h"

#include "ibtk/CartGridFunction.h"

#include "tbox/Database.h"
#include "tbox/Pointer.h"

#include <map>
#include <set>
#include <string>

/////////////////////////////// CLASS DEFINITION /////////////////////////////
namespace ADS
{

/*!
 * Class AdvDiffImplicitIntegratorStrategy is an abstract strategy class for computing the solution of the ODE z' =
 * f(z) using an implicit time stepper.
 *
 * The class has two abstract functions: computeFunction() which computes f(z) and computeJacobian which computes the
 * derivative of f(z). Note that these function evaluations are for the RHS of the ode.
 */
class AdvDiffImplicitIntegratorStrategy
{
public:
    AdvDiffImplicitIntegratorStrategy() = default;

    virtual ~AdvDiffImplicitIntegratorStrategy() = default;

    /*!
     * Evaluate the jacobian of f(z) in which dz/dt = f(z).
     *
     * Values in U consist of the values of the variables in the order they were registered as implicit variables.
     *
     * J has the correct size already listed. Only values need to be inserted.
     */
    virtual void computeJacobian(IBTK::MatrixXd& J, const IBTK::VectorXd& U, double time, const IBTK::VectorXd& Q) = 0;

    /*!
     * Evaluate f(z) in which dz/dt = f(z).
     *
     * Values in U consist of the values of the variables in the order they were registered as implicit variables.
     */
    virtual void computeFunction(IBTK::VectorXd& F, const IBTK::VectorXd& U, double time, const IBTK::VectorXd& Q) = 0;

private:
};
/*!
 * \brief Class AdvDiffImplicitIntegrator is an implementation of AdvDiffSemiImplicitHierarchyIntegrator that can treat
 * some source terms implicitly. It uses a splitting scheme to treat the implicit source terms separately from the
 * diffusion components. It uses backward Euler as the time stepper and Newton's method to solve the resulting non
 * linear system. All variables that will be treated implicitly should be labeled as such with the setImplicitVariable()
 * function.
 */
class AdvDiffImplicitIntegrator : public IBAMR::AdvDiffSemiImplicitHierarchyIntegrator
{
public:
    /*!
     * The constructor for class AdvDiffImplicitIntegrator sets
     * some default values, reads in configuration information from input and
     * restart databases, and registers the integrator object with the restart
     * manager when requested.
     */
    AdvDiffImplicitIntegrator(const std::string& object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              bool register_for_restart = true);

    /*!
     * The destructor for class AdvDiffImplicitIntegrator
     * unregisters the integrator object with the restart manager when the
     * object is so registered.
     */
    ~AdvDiffImplicitIntegrator() = default;

    /*!
     * Labels the provided variable as an implicit variable.
     */
    void setImplicitVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var);

    /*!
     * Registers additional variables that are not implicit variables, but which implicit variables depend on. These
     * values are also passed to the implicit strategy.
     *
     * The Q_var must be tracked by the advection diffusion solver, otherwise an unrecoverable exception will occur.
     */
    void setImplicitDependentVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var);

    /*!
     * Registers additional variables that are not implicit variable, but which implicit variables depend on. These
     * values are passed to the implicit strategy.
     *
     * This function adds a variable that is not tracked by the advection diffusion solver. A CartGridFunction must be
     * supplied to evaluate the function on the hierarchy.
     */
    void setImplicitDependentVariable(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                      std::shared_ptr<IBTK::CartGridFunction> Q_fcn);

    /*!
     * Register an implicit strategy object to compute the function and jacobian for the Newton iteration.
     *
     * Variables are passed to the strategy object in the order in which they were listed as implicit.
     */
    void registerImplicitStrategy(std::shared_ptr<AdvDiffImplicitIntegratorStrategy> implicit_strategy);

    /*!
     * Initialize the variables, basic communications algorithms, solvers, and
     * other data structures used by this time integrator object.
     *
     * This method is called automatically by initializePatchHierarchy() prior
     * to the construction of the patch hierarchy.  It is also possible for
     * users to make an explicit call to initializeHierarchyIntegrator() prior
     * to calling initializePatchHierarchy().
     */
    void
    initializeHierarchyIntegrator(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                  SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm<NDIM>> gridding_alg) override;

protected:
    /*!
     * Synchronously advance each level in the hierarchy over the given time
     * increment.
     */
    void integrateHierarchySpecialized(double current_time, double new_time, int cycle_num = 0) override;

    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>> d_Q_implicit_vars;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>> d_Q_implicit_dependent_vars;
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>, std::shared_ptr<IBTK::CartGridFunction>>
        d_Q_implicit_dependent_var_fcn_map;
    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> d_implicit_ctx;
    SAMRAI::hier::ComponentSelector d_implicit_comps;

    std::shared_ptr<AdvDiffImplicitIntegratorStrategy> d_implicit_strategy;
    int d_max_iterations = 100;
    double d_tol_for_newton = 1.0e-7;
    IBAMR::TimeSteppingType d_implicit_ts_type = IBAMR::TimeSteppingType::BACKWARD_EULER;

private:
    void doImplicitUpdate(double current_time, double new_time);
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_ADS_AdvDiffImplicitIntegrator
