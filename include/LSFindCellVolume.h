#ifndef included_LSFindCellVolume
#define included_LSFindCellVolume

#include "ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h"
#include "ibamr/ConvectiveOperator.h"
#include "ibamr/IBFEMethod.h"
#include "ibamr/ibamr_utilities.h"

#include "ibtk/CartCellRobinPhysBdryOp.h"
#include "ibtk/CartGridFunction.h"
#include "ibtk/FEDataManager.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/app_namespaces.h"
#include "ibtk/libmesh_utilities.h"

#include "Box.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellVariable.h"
#include "HierarchyDataOpsManager.h"
#include "HierarchyDataOpsReal.h"
#include "Patch.h"
#include "PatchLevel.h"
#include "Variable.h"
#include "VariableDatabase.h"
#include "tbox/Pointer.h"

#include "libmesh/dof_map.h"
#include "libmesh/equation_systems.h"
#include "libmesh/explicit_system.h"
#include "libmesh/fe_base.h"
#include "libmesh/mesh.h"

namespace IBAMR
{
class LSFindCellVolume
{
public:
    LSFindCellVolume(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    ~LSFindCellVolume() = default;

    void updateVolumeAndArea(int vol_idx,
                             SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> vol_var,
                             int area_idx,
                             SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> area_var,
                             int phi_idx,
                             SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> phi_var,
                             bool extended_box = false);

private:
    // Volume calculations
    using Simplex = std::array<std::pair<VectorNd, double>, NDIM + 1>;
    using LDSimplex = std::array<std::pair<VectorNd, double>, NDIM>;
    using PolytopePt = std::tuple<VectorNd, int, int>;
    void findVolumeAndArea(const double* const xlow,
                           const double* const dx,
                           const SAMRAI::hier::Index<NDIM>& patch_lower,
                           SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeData<NDIM, double>> phi,
                           const SAMRAI::pdat::CellIndex<NDIM>& idx,
                           double& volume,
                           double& area);
    double findVolume(const std::vector<Simplex>& simplices);
    double findArea(const std::vector<Simplex>& simplices);

    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
};
} // namespace IBAMR

#endif
