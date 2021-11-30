#ifndef included_LSPipeFlow
#define included_LSPipeFlow

#include "ibtk/config.h"

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

IBTK_DISABLE_EXTRA_WARNINGS
#include <boost/multi_array.hpp>
IBTK_ENABLE_EXTRA_WARNINGS

class LSPipeFlow : public ADS::LSFindCellVolume
{
public:
    LSPipeFlow(std::string object_name,
               SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
               libMesh::MeshBase* lower_mesh,
               libMesh::MeshBase* upper_mesh,
               IBTK::FEDataManager* lower_manager,
               IBTK::FEDataManager* upper_manager,
               SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    virtual ~LSPipeFlow() = default;

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
    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    using Simplex = std::array<std::pair<IBTK::VectorNd, double>, NDIM + 1>;
    using LDSimplex = std::array<std::pair<IBTK::VectorNd, double>, NDIM>;
    using PolytopePt = std::tuple<IBTK::VectorNd, int, int>;
    void findVolume(const double* const xlow,
                    const double* const dx,
                    const SAMRAI::hier::Index<NDIM>& patch_lower,
                    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeData<NDIM, double>> phi_data,
                    const SAMRAI::pdat::CellIndex<NDIM>& idx,
                    double& volume);
    double findVolume(const std::vector<Simplex>& simplices);

    libMesh::MeshBase* d_lower_mesh;
    libMesh::MeshBase* d_upper_mesh;
    IBTK::FEDataManager* d_lower_manager;
    IBTK::FEDataManager* d_upper_manager;
    bool d_use_inside = true;

    static double s_large_val;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_sgn_var;
    int d_sgn_idx = IBTK::invalid_index;

    double d_theta = 0.0;
    double d_L = 0.0;
    double d_y_up = 0.0;
    double d_y_low = 0.0;
};
#endif
