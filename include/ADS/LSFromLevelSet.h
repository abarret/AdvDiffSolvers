#ifndef included_ADS_LSFromLevelSet
#define included_ADS_LSFromLevelSet

#include "ADS/LSFindCellVolume.h"

#include "ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h"
#include "ibamr/ConvectiveOperator.h"
#include "ibamr/ibamr_utilities.h"

#include "Box.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellVariable.h"
#include "tbox/Pointer.h"

#include <array>
#include <vector>

namespace ADS
{
class LSFromLevelSet : public LSFindCellVolume
{
public:
    LSFromLevelSet(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    ~LSFromLevelSet() = default;

    void registerLSFcn(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> ls_fcn);

    void updateVolumeAreaSideLS(int vol_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> vol_var,
                                int area_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> area_var,
                                int side_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> side_var,
                                int phi_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                                double data_time,
                                bool extended_box = false) override;

private:
    SAMRAI::tbox::Pointer<IBTK::CartGridFunction> d_ls_fcn;
};
} // namespace ADS

#endif
