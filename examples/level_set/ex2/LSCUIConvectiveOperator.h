// ---------------------------------------------------------------------
//
// Copyright (c) 2018 - 2023 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_LSCUIConvectiveOperator
#define included_ADS_LSCUIConvectiveOperator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ibamr/CellConvectiveOperator.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class LSCUIConvectiveOperator is a concrete ConvectiveOperator
 * that implements a upwind convective differencing operator based on the
 * cubic upwind interpolation (CUI).
 *
 * Class LSCUIConvectiveOperator computes the convective derivative of a
 * cell-centered field using the CUI method described by Waterson and Deconinck,
 * and Patel and Natarajan.
 *
 *
 * References
 * Waterson, NP. and Deconinck, H., <A HREF="https://www.sciencedirect.com/science/article/pii/S002199910700040X">
 * Design principles for bounded higher-order convection schemes – a unified approach</A>
 *
 * Patel, JK. and Natarajan, G., <A HREF="https://www.sciencedirect.com/science/article/pii/S0045793014004009">
 * A generic framework for design of interface capturing schemes for multi-fluid flows</A>
 *
 * \see AdvDiffSemiImplicitHierarchyIntegrator
 */
class LSCUIConvectiveOperator : public IBAMR::CellConvectiveOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    LSCUIConvectiveOperator(std::string object_name,
                            SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                            SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                            IBAMR::ConvectiveDifferencingType difference_form,
                            std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> bc_coefs);

    /*!
     * \brief Default destructor.
     */
    ~LSCUIConvectiveOperator();

    /*!
     * \brief Static function to construct an LSCUIConvectiveOperator.
     */
    static SAMRAI::tbox::Pointer<ConvectiveOperator>
    allocate_operator(const std::string& object_name,
                      SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                      SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                      IBAMR::ConvectiveDifferencingType difference_form,
                      const std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*>& bc_coefs)
    {
        return new LSCUIConvectiveOperator(object_name, Q_var, input_db, difference_form, bc_coefs);
    } // allocate_operator

    void setLSData(SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> ls_var,
                   int ls_idx,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    /*!
     * \brief Interpolate a cell-centered field Q to a face-centered field q on a single grid patch.
     */
    void interpolateToFaceOnPatch(SAMRAI::pdat::FaceData<NDIM, double>& q_interp_data,
                                  const SAMRAI::pdat::CellData<NDIM, double>& Q_cell_data,
                                  const SAMRAI::pdat::FaceData<NDIM, double>& u_data,
                                  const SAMRAI::hier::Patch<NDIM>& patch) override;

    void applyConvectiveOperator(int Q_idx, int N_idx) override;

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    LSCUIConvectiveOperator() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    LSCUIConvectiveOperator(const LSCUIConvectiveOperator& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    LSCUIConvectiveOperator& operator=(const LSCUIConvectiveOperator& that) = delete;

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_var;
    int d_Q_scr_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_ls_var;
    int d_ls_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_phi_var;
    int d_phi_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, int>> d_valid_var;
    int d_valid_idx = IBTK::invalid_index;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_ADS_LSCUIConvectiveOperator
