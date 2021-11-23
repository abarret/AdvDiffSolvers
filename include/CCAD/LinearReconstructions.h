/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_CCAD_LinearReconstructions
#define included_CCAD_LinearReconstructions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "CCAD/AdvectiveReconstructionOperator.h"
#include "CCAD/ls_utilities.h"
#include "CCAD/reconstructions.h"

#include "CellVariable.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace CCAD
{
/*!
 * \brief Class LinearReconstructions is a abstract class for an implementation of
 * a convective differencing operator.
 */
class LinearReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    LinearReconstructions(std::string object_name);

    /*!
     * \brief Destructor.
     */
    ~LinearReconstructions();

    /*!
     * \brief Deletec Operators
     */
    //\{
    LinearReconstructions() = delete;
    LinearReconstructions(const LinearReconstructions& from) = delete;
    LinearReconstructions& operator=(const LinearReconstructions& that) = delete;
    //\}

    /*!
     * \brief Initialize operator.
     */
    void allocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                               double current_time,
                               double new_time) override;

    /*!
     * \brief Deinitialize operator
     */
    void deallocateOperatorState() override;

    /*!
     * \brief Compute N = u * grad Q.
     */
    void applyReconstruction(int Q_idx, int N_idx, int path_idx) override;

private:
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    Reconstruct::RBFPolyOrder d_rbf_order = Reconstruct::RBFPolyOrder::LINEAR;
    unsigned int d_rbf_stencil_size = 5;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;
};
} // namespace CCAD

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_CCAD_LinearReconstructions
