/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_LagrangeReconstructions
#define included_ADS_LagrangeReconstructions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ADS/AdvectiveReconstructionOperator.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "CellVariable.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class LagrangeReconstructions is a abstract class for an implementation of
 * a convective differencing operator.
 */
class LagrangeReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    LagrangeReconstructions(std::string object_name);

    /*!
     * \brief Destructor.
     */
    ~LagrangeReconstructions();

    /*!
     * \brief Deletec Operators
     */
    //\{
    LagrangeReconstructions() = delete;
    LagrangeReconstructions(const LagrangeReconstructions& from) = delete;
    LagrangeReconstructions& operator=(const LagrangeReconstructions& that) = delete;
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
    unsigned int d_rbf_stencil_size = 8;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_ADS_LagrangeReconstructions
