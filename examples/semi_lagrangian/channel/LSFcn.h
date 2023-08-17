#ifndef included_ADS_LSFcn
#define included_ADS_LSFcn

#include <ibtk/CartGridFunction.h>

namespace ADS
{

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
class LSFcn : public IBTK::CartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    LSFcn(std::string object_name);

    /*!
     * \brief Destructor.
     */
    ~LSFcn() = default;

    /*!
     * Indicates whether the concrete CartGridFunction object is time dependent.
     */
    bool isTimeDependent() const override
    {
        return true;
    }

    /*!
     * Set the data on the patch interior to the exact answer.
     */
    void setDataOnPatch(int data_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                        double data_time,
                        bool initial_time = false,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level =
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>>(nullptr)) override;
};
} // namespace ADS
#endif // #ifndef included_LSFcn
