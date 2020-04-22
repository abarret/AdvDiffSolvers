// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2018 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#ifndef included_QInitial
#define included_QInitial

/////////////////////////////// INCLUDES /////////////////////////////////////

// IBTK INCLUDES
#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

// SAMRAI INCLUDES
#include <CartesianGridGeometry.h>

// C++ namespace delcarations
#include <ibamr/AdvDiffHierarchyIntegrator.h>
#include <ibamr/app_namespaces.h>

#include "IntegrateFunction.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advected scalar Q.
 */
class QInitial : public CartGridFunction
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
    double d_R1 = 0.25, d_R2 = 1.25;
    VectorNd d_center = { 1.509, 1.521 };
    std::vector<double> d_vel;

    /*
     * Channel information
     */
    double d_theta = M_PI / 12.0, d_y_p = 1.25, d_y_n = 0.5;
    VectorNd d_channel_center = { 1.5, 1.5 };

    /*
     * Rotational information
     */
    double d_v = std::numeric_limits<double>::quiet_NaN();

    // Level set info
    int d_ls_idx = IBTK::invalid_index;
    int d_vol_idx = IBTK::invalid_index;
};

/////////////////////////////// INLINE ///////////////////////////////////////

//#include "QInitial.I"

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_QInitial