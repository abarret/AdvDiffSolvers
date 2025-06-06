#ifndef included_ADS_ExtrapolatedConvectiveOperator
#define included_ADS_ExtrapolatedConvectiveOperator

#include <ibamr/config.h>

#include <ADS/InternalBdryFill.h>

#include "ibamr/CellConvectiveOperator.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class ExtrapolatedConvectiveOperator is a concrete ConvectiveOperator that will extrapolate the solution
 * across the interface before applying the advective operator.
 *
 * The class advects quantities in the normal direction according to the level set. It treats both sides independently
 * of each other. A level set must be registered via the setLSData() function.
 *
 * \see AdvDiffSemiImplicitHierarchyIntegrator
 */
class ExtrapolatedConvectiveOperator : public IBAMR::ConvectiveOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    ExtrapolatedConvectiveOperator(std::string object_name,
                                   SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                   SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                                   IBAMR::ConvectiveDifferencingType difference_form,
                                   std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> bc_coefs);

    /*!
     * \brief Default destructor.
     */
    virtual ~ExtrapolatedConvectiveOperator() = default;

    static SAMRAI::tbox::Pointer<IBAMR::ConvectiveOperator>
    allocate_operator(const std::string& object_name,
                      SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                      SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                      IBAMR::ConvectiveDifferencingType difference_form,
                      const std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*>& bc_coefs)
    {
        return new ExtrapolatedConvectiveOperator(object_name, Q_var, input_db, difference_form, bc_coefs);
    }

    void initializeOperatorState(const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
                                 const SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& y) override;

    void deallocateOperatorState() override;

    void setLSData(int ls_idx, SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var);

    /*!
     * \brief Compute N = u * grad Q
     */
    void apply(SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& x,
               SAMRAI::solv::SAMRAIVectorReal<NDIM, double>& y) override;

    void applyConvectiveOperator(int Q_idx, int N_idx) override;

private:
    /*!
     * \brief Deleted copy constructor.
     */
    ExtrapolatedConvectiveOperator(const ExtrapolatedConvectiveOperator& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    ExtrapolatedConvectiveOperator& operator=(const ExtrapolatedConvectiveOperator& that) = delete;

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    SAMRAI::tbox::Pointer<IBAMR::ConvectiveOperator> d_convec_op;
    std::string d_convec_op_type;
    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_bc_coefs;

    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_Q_pos_vec, d_Q_neg_vec;
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_N_pos_vec, d_N_neg_vec;

    int d_max_gcw_fill = std::numeric_limits<int>::max();

    std::unique_ptr<InternalBdryFill> d_internal_fill;

    int d_phi_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_phi_var;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_IBAMR_ExtrapolatedConvectiveOperator
