#ifndef included_CCAD_QInitial
#define included_CCAD_QInitial

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/IntegrateFunction.h"
#include "CCAD/LSCartGridFunction.h"

#include <ibamr/AdvDiffHierarchyIntegrator.h>
#include <ibamr/app_namespaces.h>

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

#include <CartesianGridGeometry.h>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
namespace CCAD
{
class QInitial : public LSCartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    QInitial(const string& object_name, Pointer<GridGeometry<NDIM>> grid_geom, Pointer<Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~QInitial();

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

protected:
private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    QInitial();

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    QInitial(const QInitial& from);

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    QInitial& operator=(const QInitial& that);

    /*!
     * Read input values, indicated above, from given database.
     */
    void getFromInput(Pointer<Database> db);

    bool d_solve_for_average = false;

    /*
     * The grid geometry.
     */
    Pointer<CartesianGridGeometry<NDIM>> d_grid_geom;

    /*
     * The initialization type.
     */
    string d_init_type = "ANNULUS";

    double d_D = 0.01;
    /*
     * Annulus information
     */
    double d_R1 = 0.25;
#if (NDIM == 2)
    VectorNd d_center = { 1.509, 1.521 };
#endif
#if (NDIM == 3)
    VectorNd d_center = { 1.509, 1.521, 1.514 };
#endif
    std::array<double, NDIM> d_vel;

    /*
     * Rotational information
     */
    double d_v = std::numeric_limits<double>::quiet_NaN();

    double d_val = std::numeric_limits<double>::quiet_NaN();
};

} // namespace CCAD

#endif //#ifndef included_LS_QInitial
