#ifndef included_ADS_PPMReconstructions
#define included_ADS_PPMReconstructions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ADS/AdvectiveReconstructionOperator.h"
#include "ADS/CutCellVolumeMeshMapping.h"
#include "ADS/ls_utilities.h"
#include "ADS/reconstructions.h"

#include "CellVariable.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class PPMReconstructions is a concrete implementation of AdvectiveReconstructionOperator that performs WENO
 * interpolations to points.
 */
class PPMReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    PPMReconstructions(std::string object_name);

    /*!
     * \brief Destructor.
     */
    ~PPMReconstructions();

    /*!
     * \brief Deleted Operators/Constructors
     */
    //\{
    PPMReconstructions() = delete;
    PPMReconstructions(const PPMReconstructions& from) = delete;
    PPMReconstructions& operator=(const PPMReconstructions& that) = delete;
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

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;
};

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_PPMReconstructions
