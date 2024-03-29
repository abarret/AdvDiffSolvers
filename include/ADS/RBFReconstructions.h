#ifndef included_ADS_RBFReconstructions
#define included_ADS_RBFReconstructions

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
 * \brief Class RBFReconstructions is a abstract class for an implementation of
 * a convective differencing operator.
 */
class RBFReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    RBFReconstructions(std::string object_name,
                       Reconstruct::RBFPolyOrder rbf_poly_order,
                       int stencil_size,
                       bool use_cut_cells = true);

    /*!
     * \brief Destructor.
     */
    ~RBFReconstructions();

    /*!
     * \brief Deletec Operators
     */
    //\{
    RBFReconstructions() = delete;
    RBFReconstructions(const RBFReconstructions& from) = delete;
    RBFReconstructions& operator=(const RBFReconstructions& that) = delete;
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
     * \brief Compute the reconstruction using cut cells and cell centroids. This method assumes ghost cells are present
     * and filled for Q_idx.
     */
    void applyReconstructionCutCell(int Q_idx, int N_idx, int path_idx);

    /*!
     * \brief Compute the reconstruction using only the level set to determine sides. This method assumes ghost cells
     * are present and filled for Q_idx.
     */
    void applyReconstructionLS(int Q_idx, int N_idx, int path_idx);

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    Reconstruct::RBFPolyOrder d_rbf_order = Reconstruct::RBFPolyOrder::LINEAR;
    unsigned int d_rbf_stencil_size = 5;
    bool d_use_cut_cells = true;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_LS_RBFReconstructions
