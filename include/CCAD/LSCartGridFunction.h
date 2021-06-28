#ifndef included_CCAD_LSCartGridFunction
#define included_CCAD_LSCartGridFunction

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
namespace CCAD
{
class LSCartGridFunction : public IBTK::CartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    LSCartGridFunction(const std::string& object_name);

    /*!
     * \brief Destructor.
     */
    virtual ~LSCartGridFunction() = default;

    virtual inline void setLSIndex(const int ls_idx, const int vol_idx)
    {
        d_ls_idx = ls_idx;
        d_vol_idx = vol_idx;
    }

    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    LSCartGridFunction() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    LSCartGridFunction(const LSCartGridFunction& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    LSCartGridFunction& operator=(const LSCartGridFunction& that) = delete;

protected:
    // Level set info
    int d_ls_idx = IBTK::invalid_index;
    int d_vol_idx = IBTK::invalid_index;
};

} // namespace CCAD

#endif //#ifndef included_LS_LSCartGridFunction
