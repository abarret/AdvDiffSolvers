#ifndef included_ADS_LSFromMesh
#define included_ADS_LSFromMesh

#include "ibtk/config.h"

#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
#include "ADS/LSFindCellVolume.h"

#include "ibamr/AdvDiffSemiImplicitHierarchyIntegrator.h"
#include "ibamr/ConvectiveOperator.h"
#include "ibamr/ibamr_utilities.h"

#include "ibtk/CartCellRobinPhysBdryOp.h"
#include "ibtk/CartGridFunction.h"
#include "ibtk/FEDataManager.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/app_namespaces.h"

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

#include "libmesh/elem.h"
#include "libmesh/mesh.h"
#include "libmesh/vector_value.h"

#include <functional>

IBTK_DISABLE_EXTRA_WARNINGS
#include <boost/multi_array.hpp>
IBTK_ENABLE_EXTRA_WARNINGS

namespace ADS
{
class LSFromMesh : public LSFindCellVolume
{
public:
    LSFromMesh(std::string object_name,
               SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
               const SAMRAI::tbox::Pointer<CutCellMeshMapping>& cut_cell_mesh_mapping,
               bool use_inside = true);

    virtual ~LSFromMesh() = default;

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

    inline void registerNormalReverseDomainId(unsigned int bdry_id, unsigned int part = 0)
    {
        d_norm_reverse_domain_ids[part].insert(bdry_id);
    }

    inline void registerNormalReverseDomainId(const std::vector<unsigned int>& bdry_ids, unsigned int part = 0)
    {
        for (const auto& bdry_id : bdry_ids) registerNormalReverseDomainId(bdry_id, part);
    }

    inline void registerNormalReverseElemId(unsigned int bdry_id, unsigned int part = 0)
    {
        d_norm_reverse_elem_ids[part].insert(bdry_id);
    }

    inline void registerNormalReverseElemId(const std::vector<unsigned int>& bdry_ids, unsigned int part = 0)
    {
        for (const auto& bdry_id : bdry_ids) registerNormalReverseElemId(bdry_id, part);
    }

    inline void registerReverseNormal(const int part = 0)
    {
        d_reverse_normal[part] = 1;
    }

    using BdryFcn = std::function<void(const IBTK::VectorNd&, double&)>;

    inline void registerBdryFcn(BdryFcn fcn)
    {
        d_bdry_fcn = fcn;
    }

private:
    void commonConstructor();

    void updateLSAwayFromInterface(int phi_idx);
    // This does a flood filling algorithm for d_sgn_idx.
    // We assume that any value less than eps on the given level is correctly set.
    // NOTE: eps must be positive.
    void floodFillForLS(int ln, double eps);

    bool d_use_inside = true;

    SAMRAI::tbox::Pointer<CutCellMeshMapping> d_cut_cell_mesh_mapping;

    std::vector<std::set<unsigned int>> d_norm_reverse_domain_ids, d_norm_reverse_elem_ids;
    std::vector<int> d_reverse_normal;

    BdryFcn d_bdry_fcn;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_sgn_var;
    int d_sgn_idx = IBTK::invalid_index;
};
} // namespace ADS

#endif
