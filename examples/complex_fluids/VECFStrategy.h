/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_ADS_VECFStrategy
#define included_ADS_VECFStrategy

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ibamr/CFStrategy.h"
#include "ibamr/ibamr_enums.h"
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellVariable.h"
#include "Index.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchGeometry.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "Variable.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"
#include "tbox/Utilities.h"

#include <string>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace ADS
{
/*!
 * \brief Class VECFStrategy is a concrete CFStrategy
 */
class VECFStrategy : public IBAMR::CFStrategy
{
public:
    /*!
     * \brief This constructor reads in the parameters for the model from the input database.
     *
     * If `use_sigma = false`, determines bond length assuming the stress tensor corresponds to tau.
     */
    VECFStrategy(const std::string& object_name,
                 SAMRAI::tbox::Pointer<IBAMR::INSStaggeredHierarchyIntegrator> ins_integrator,
                 SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> zb_var,
                 SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> zb_integrator,
                 double C8,
                 double beta);

    /*!
     * \brief Empty destructor.
     */
    ~VECFStrategy() = default;

    /*!
     * Computes the RHS of the stress evolution equation. C_idx contains the current stress data that should be used to
     * evaluate the relaxation function.
     *
     * R_idx should contain the return data. Note that evolve_type here must be STANDARD.
     */
    void computeRelaxation(int R_idx,
                           SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                           int C_idx,
                           SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> C_var,
                           IBAMR::TensorEvolutionType evolve_type,
                           SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                           double data_time) override;

    /*!
     * Given the evolved quantity stored in stress_idx, compute the stress that shows up in the momentum equation.
     *
     * Note that at least one layer of ghost cells is gaurenteed to exist in stress_idx and they must be overwritten in
     * this routine.
     */
    void computeStress(int stress_idx,
                       SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> stress_var,
                       SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                       double data_time) override;

private:
    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_zb_var;
    SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> d_zb_integrator;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_EE_var;
    int d_EE_idx = IBTK::invalid_index;

    SAMRAI::tbox::Pointer<IBAMR::INSStaggeredHierarchyIntegrator> d_ins_integrator;

    double d_C8 = std::numeric_limits<double>::quiet_NaN();
    double d_beta = std::numeric_limits<double>::quiet_NaN();
};
} // namespace ADS
#endif
