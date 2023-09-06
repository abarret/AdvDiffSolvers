#ifndef included_ADS_PointwiseFunction
#define included_ADS_PointwiseFunction

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
namespace ADS
{
template <typename F>
class PointwiseFunction : public IBTK::CartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    PointwiseFunction(std::string object_name, F f);

    /*!
     * \brief Destructor.
     */
    virtual ~PointwiseFunction() = default;

    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    PointwiseFunction() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    PointwiseFunction(const PointwiseFunction& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    PointwiseFunction& operator=(const PointwiseFunction& that) = delete;

    /*!
     * \brief Does this function depend on time?
     */
    bool isTimeDependent() const override
    {
        return true;
    }

    /*!
     * \brief Evaluate the function on the grid hierarchy.
     */
    void setDataOnPatch(int data_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                        double data_time,
                        bool initial_time = false,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level =
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>>(NULL)) override;

private:
    F d_f;
};

} // namespace ADS

#endif // #ifndef included_LS_PointwiseFunction
