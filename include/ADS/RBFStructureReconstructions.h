#ifndef included_ADS_RBFStructureReconstructions
#define included_ADS_RBFStructureReconstructions

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
 * \brief Class RBFStructureReconstructions is a abstract class for an implementation of
 * a convective differencing operator.
 */
class RBFStructureReconstructions : public AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    RBFStructureReconstructions(std::string object_name, Reconstruct::RBFPolyOrder rbf_poly_order, int stencil_size);

    /*!
     * \brief Destructor.
     */
    ~RBFStructureReconstructions();

    /*!
     * \brief Deleted Operators/Constructors
     */
    //\{
    RBFStructureReconstructions() = delete;
    RBFStructureReconstructions(const RBFStructureReconstructions& from) = delete;
    RBFStructureReconstructions& operator=(const RBFStructureReconstructions& that) = delete;
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

    /*!
     * \brief Provide information on the location of the mesh
     */
    void setCutCellMapping(SAMRAI::tbox::Pointer<CutCellVolumeMeshMapping> mesh_partitioner);

    /*!
     * \brief Provide the structural system name for the exact solution.
     */
    void setQSystemName(std::string Q_sys_name);

private:
    /*!
     * \brief Compute the reconstruction using only the level set to determine sides. This method assumes ghost cells
     * are present and filled for Q_idx.
     */
    void applyReconstructionLS(int Q_idx, int N_idx, int path_idx);

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    Reconstruct::RBFPolyOrder d_rbf_order = Reconstruct::RBFPolyOrder::LINEAR;
    unsigned int d_rbf_stencil_size = 5;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;

    // Structural information
    SAMRAI::tbox::Pointer<CutCellVolumeMeshMapping> d_cut_cell_mapping;

    // Structural value information
    std::string d_Q_sys_name;
};
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_LS_RBFStructureReconstructions
