#ifndef included_ADS_QFcn
#define included_ADS_QFcn

#include "ADS/IntegrateFunction.h"
#include "ADS/LSCartGridFunction.h"

#include <ibamr/AdvDiffHierarchyIntegrator.h>

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

#include <CartesianGridGeometry.h>

namespace ADS
{

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
class QFcn : public ADS::LSCartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    QFcn(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~QFcn() = default;

    /*!
     * Indicates whether the concrete CartGridFunction object is time dependent.
     */
    bool isTimeDependent() const
    {
        return true;
    }

    /*!
     * Set the data on the patch interior to the exact answer. Note that we are setting the cell average here.
     */
    void setDataOnPatch(int data_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                        double data_time,
                        bool initial_time = false,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level =
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>>(nullptr)) override;

protected:
private:
    /*!
     * Read input values, indicated above, from given database.
     */
    void getFromInput(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db);

    IBTK::VectorNd d_com;
    enum class FcnType
    {
        SINE,
        TRIANGLE,
        DISK,
        UNKNOWN
    } d_fcn_type = FcnType::UNKNOWN;
};
} // namespace ADS
#endif // #ifndef included_ADS_QFcn
