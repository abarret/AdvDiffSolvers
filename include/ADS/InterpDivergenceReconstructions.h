#ifndef included_ADS_InterpDivergenceReconstructions
#define included_ADS_InterpDivergenceReconstructions

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
 * \brief Class InterpDivergenceReconstructions is a concrete implementation of an AdvectiveReconstructionOperator that
 * computes the divergence of the flow field along path locations.
 *
 * This class first uses centered differences to compute a cell centered divergence from a side centered velocity field.
 * Then, we use quadratic interpolants in the bulk and polyharmonic splines near the boundary to interpolate the
 * resulting divergence.
 *
 * This operator does not use divergences that were computed by differences across a boundary.
 */
class InterpDivergenceReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    InterpDivergenceReconstructions(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~InterpDivergenceReconstructions();

    /*!
     * \brief Initialize operator. This allocates scratch data for the interpolant.
     */
    void allocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                               double current_time,
                               double new_time) override;

    /*!
     * \brief Deinitialize operator
     */
    void deallocateOperatorState() override;

    /*!
     * \brief Compute N = div(Q_idx) at locations stored in path_idx. Q_idx must be a side centered quantity.
     *
     * This first computes a centered difference approximation to the divergence, then interpolates the resulting cell
     * centered divergence to the path locations.
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
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_InterpDivergenceReconstructions
