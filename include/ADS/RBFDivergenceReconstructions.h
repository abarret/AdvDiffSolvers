#ifndef included_ADS_RBFDivergenceReconstructions
#define included_ADS_RBFDivergenceReconstructions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ADS/AdvectiveReconstructionOperator.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "CellVariable.h"
#include "SideVariable.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class RBFDivergenceReconstructions is a abstract class for an implementation of
 * a convective differencing operator.
 */
class RBFDivergenceReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    RBFDivergenceReconstructions(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~RBFDivergenceReconstructions();

    /*!
     * \brief Deleted Operators/Constructors
     */
    //\{
    RBFDivergenceReconstructions() = delete;
    RBFDivergenceReconstructions(const RBFDivergenceReconstructions& from) = delete;
    RBFDivergenceReconstructions& operator=(const RBFDivergenceReconstructions& that) = delete;
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
    /*!
     * \brief Compute the reconstruction using only the level set to determine sides. This method assumes ghost cells
     * are present and filled for Q_idx.
     */
    void applyReconstructionLS(int Q_idx, int N_idx, int path_idx);

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    Reconstruct::RBFPolyOrder d_rbf_order = Reconstruct::RBFPolyOrder::QUADRATIC;
    unsigned int d_rbf_stencil_size = 12;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_u_scr_var;
    int d_u_scr_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_div_var;
    int d_div_idx = IBTK::invalid_index;

    // Weight caching
    bool d_weights_cached = false;
    std::vector<double> d_wgts;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_RBFDivergenceReconstructions
