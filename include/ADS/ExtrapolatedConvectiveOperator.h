#ifndef included_ADS_ExtrapolatedConvectiveOperator
#define included_ADS_ExtrapolatedConvectiveOperator

#include <ibamr/config.h>

#include <ADS/InternalBdryFill.h>

#include "ibamr/CellConvectiveOperator.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class ExtrapolatedConvectiveOperator is a concrete ConvectiveOperator
 * that implements a upwind convective differencing operator based on the
 * piecewise parabolic method (PPM).
 *
 * Class ExtrapolatedConvectiveOperator computes the convective derivative of a
 * cell-centered velocity field using the xsPPM7 method of Rider, Greenough, and
 * Kamm.
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
                                   std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> bc_coefs,
                                   int max_gcw_fill);

    /*!
     * \brief Default destructor.
     */
    virtual ~ExtrapolatedConvectiveOperator() = default;

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

    SAMRAI::tbox::Pointer<IBAMR::ConvectiveOperator> d_convec_op;
    std::string d_convec_op_type;
    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_bc_coefs;

    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_Q_pos_vec, d_Q_neg_vec;
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_N_pos_vec, d_N_neg_vec;

    int d_max_gcw_fill = 1;

    std::unique_ptr<InternalBdryFill> d_internal_fill;

    int d_phi_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_phi_var;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_IBAMR_ExtrapolatedConvectiveOperator
