#ifndef included_QFcn
#define included_QFcn

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/IntegrateFunction.h"
#include "CCAD/LSCartGridFunction.h"

#include <ibamr/AdvDiffHierarchyIntegrator.h>

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

#include <CartesianGridGeometry.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
class QFcn : public CCAD::LSCartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    QFcn(const std::string& object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~QFcn();

    /*!
     * Indicates whether the concrete CartGridFunction object is time dependent.
     */
    bool isTimeDependent() const
    {
        return true;
    }

    /*!
     * Set the data on the patch hierarchy.
     */
    void setDataOnPatchHierarchy(int idx,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                 double data_time,
                                 bool initial_time = false,
                                 int coarsest_ln = -1,
                                 int finest_ln = -1) override;

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

    /*!
     * Update cell average to be total amount.
     */
    void updateAverageToTotal(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> var,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> Q_ctx,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                              int vol_idx);

    inline void setLSIndex(const int ls_idx, const int vol_idx)
    {
        d_ls_idx = ls_idx;
        d_vol_idx = vol_idx;
    }

protected:
private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    QFcn();

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    QFcn(const QFcn& from);

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    QFcn& operator=(const QFcn& that);

    /*!
     * Read input values, indicated above, from given database.
     */
    void getFromInput(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db);

    // Level set info
    int d_ls_idx = IBTK::invalid_index;
    int d_vol_idx = IBTK::invalid_index;

    double d_D = 0.0;
    double d_k_off = 0.0;
    double d_k_on = 0.0;
    double d_sf_max = 0.0;

    IBTK::VectorNd d_cent = { 0.0, 0.0 };
};

#endif //#ifndef included_LS_QFcn
