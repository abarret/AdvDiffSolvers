// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2020 by the IBAMR developers
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

#ifndef included_LS_AdvectiveReconstructionOperator
#define included_LS_AdvectiveReconstructionOperator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include <ibtk/ibtk_utilities.h>

#include <PatchHierarchy.h>

#include <string>

namespace SAMRAI
{
namespace solv
{
template <int DIM, class TYPE>
class SAMRAIVectorReal;
} // namespace solv
} // namespace SAMRAI

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace LS
{
/*!
 * \brief Class AdvectiveReconstructionOperator is an abstract class for reconstructing solutions after a
 * semi-Lagrangian advection step.
 */
class AdvectiveReconstructionOperator : public SAMRAI::tbox::DescribedClass
{
public:
    /*!
     * \brief Class constructor.
     */
    AdvectiveReconstructionOperator(std::string object_name);

    /*!
     * \brief Destructor.
     */
    ~AdvectiveReconstructionOperator();

    /*!
     * \brief Deletec Operators
     */
    //\{
    AdvectiveReconstructionOperator() = delete;
    AdvectiveReconstructionOperator(const AdvectiveReconstructionOperator& from) = delete;
    AdvectiveReconstructionOperator& operator=(const AdvectiveReconstructionOperator& that) = delete;
    //\}

    /*!
     * \brief set level set data
     */
    void setCurrentLSData(int ls_idx, int vol_idx);
    void setNewLSData(int ls_idx, int vol_idx);
    void setLSData(int cur_ls_idx, int cur_vol_idx, int new_ls_idx, int new_vol_idx);

    /*!
     * \brief Compute N = u * grad Q.
     */
    virtual void applyReconstruction(int Q_idx, int N_idx, int path_idx) = 0;

    /*!
     * \brief Allocate any operator storage required to apply the reconstruction.
     *
     * \note If this function is called when the operator is allocated, deallocateOperatorState() will be called prior
     * to allocation.
     */
    virtual void allocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                       double current_time,
                                       double new_time);

    virtual void deallocateOperatorState();

protected:
    std::string d_object_name;
    /*!
     * \brief Level set, cell volume, and path data
     */
    //\{
    int d_cur_ls_idx = IBTK::invalid_index, d_cur_vol_idx = IBTK::invalid_index;
    int d_new_ls_idx = IBTK::invalid_index, d_new_vol_idx = IBTK::invalid_index;
    //\}

    bool d_is_allocated = false;

    double d_current_time = std::numeric_limits<double>::quiet_NaN(),
           d_new_time = std::numeric_limits<double>::quiet_NaN();
};
} // namespace LS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_LS_AdvectiveReconstructionOperator
