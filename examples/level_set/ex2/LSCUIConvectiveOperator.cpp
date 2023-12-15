/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include <ADS/InternalBdryFill.h>
#include <ADS/ReinitializeLevelSet.h>
#include <ADS/ads_utilities.h>
#include <ADS/ls_functions.h>

// Local includes
#include "LSCUIConvectiveOperator.h"

// FORTRAN ROUTINES
#if (NDIM == 2)
#define CUI_EXTRAPOLATE_FC IBAMR_FC_FUNC_(cui_extrapolate2d, CUI_EXTRAPOLATE2D)
#endif

#if (NDIM == 3)
#define CUI_EXTRAPOLATE_FC IBAMR_FC_FUNC_(cui_extrapolate3d, CUI_EXTRAPOLATE3D)
#endif

extern "C"
{
    void CUI_EXTRAPOLATE_FC(
#if (NDIM == 2)
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const double*,
        double*,
        const int&,
        const int&,
        const int&,
        const int&,
        const double*,
        const double*,
        double*,
        double*
#endif
#if (NDIM == 3)
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const double*,
        double*,
        double*,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const double*,
        const double*,
        const double*,
        double*,
        double*,
        double*
#endif
    );
}

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// NOTE: The number of ghost cells required by the advection scheme
// These values were chosen to work with CUI (the cubic interpolation
// upwind method of Waterson and Deconinck).
static const int Q_MIN_GCW = 2;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

LSCUIConvectiveOperator::LSCUIConvectiveOperator(std::string object_name,
                                                 Pointer<CellVariable<NDIM, double>> Q_var,
                                                 Pointer<Database> input_db,
                                                 const ConvectiveDifferencingType difference_form,
                                                 std::vector<RobinBcCoefStrategy<NDIM>*> bc_coefs)
    : CellConvectiveOperator(std::move(object_name), Q_var, Q_MIN_GCW, input_db, difference_form, std::move(bc_coefs)),
      d_Q_var(Q_var),
      d_phi_var(new NodeVariable<NDIM, double>(d_object_name + "::phi")),
      d_valid_var(new NodeVariable<NDIM, int>(d_object_name + "::valid"))
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    std::string var_name = d_object_name + "::phi";
    if (var_db->checkVariableExists(var_name)) d_phi_var = var_db->getVariable(var_name);
    d_phi_idx = var_db->registerVariableAndContext(d_phi_var, var_db->getContext(d_object_name + "::CTX"), 1);

    var_name = d_object_name + "::valid";
    if (var_db->checkVariableExists(var_name)) d_valid_var = var_db->getVariable(var_name);
    d_valid_idx = var_db->registerVariableAndContext(d_valid_var, var_db->getContext(d_object_name + "::CTX"), 1);

    d_Q_scr_idx = var_db->registerVariableAndContext(d_Q_var, var_db->getContext(d_object_name + "::CTX"), Q_MIN_GCW);

    return;
} // LSCUIConvectiveOperator

LSCUIConvectiveOperator::~LSCUIConvectiveOperator()
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    var_db->removePatchDataIndex(d_phi_idx);
}

void
LSCUIConvectiveOperator::setLSData(Pointer<NodeVariable<NDIM, double>> ls_var,
                                   const int ls_idx,
                                   Pointer<PatchHierarchy<NDIM>> hierarchy)
{
    d_ls_var = ls_var;
    d_ls_idx = ls_idx;
    d_hierarchy = hierarchy;
}

void
LSCUIConvectiveOperator::interpolateToFaceOnPatch(FaceData<NDIM, double>& q_interp_data,
                                                  const CellData<NDIM, double>& Q_cell_data,
                                                  const FaceData<NDIM, double>& u_data,
                                                  const Patch<NDIM>& patch)
{
    const auto& patch_box = patch.getBox();
    const auto& patch_lower = patch_box.lower();
    const auto& patch_upper = patch_box.upper();

    const IntVector<NDIM>& Q_cell_data_gcw = Q_cell_data.getGhostCellWidth();
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_cell_data_gcw.min() == Q_cell_data_gcw.max());
#endif
    const IntVector<NDIM>& u_data_gcw = u_data.getGhostCellWidth();
#if !defined(NDEBUG)
    TBOX_ASSERT(u_data_gcw.min() == u_data_gcw.max());
#endif
    const IntVector<NDIM>& q_interp_data_gcw = q_interp_data.getGhostCellWidth();
#if !defined(NDEBUG)
    TBOX_ASSERT(q_interp_data_gcw.min() == q_interp_data_gcw.max());
#endif
    const CellData<NDIM, double>& Q0_data = Q_cell_data;
    CellData<NDIM, double> Q1_data(patch_box, 1, Q_cell_data_gcw);
#if (NDIM == 3)
    CellData<NDIM, double> Q2_data(patch_box, 1, Q_cell_data_gcw);
#endif

    // Interpolate from cell centers to cell faces.
    for (int d = 0; d < Q_cell_data.getDepth(); ++d)
    {
        CUI_EXTRAPOLATE_FC(
#if (NDIM == 2)
            patch_lower(0),
            patch_upper(0),
            patch_lower(1),
            patch_upper(1),
            Q_cell_data_gcw(0),
            Q_cell_data_gcw(1),
            Q0_data.getPointer(d),
            Q1_data.getPointer(),
            u_data_gcw(0),
            u_data_gcw(1),
            q_interp_data_gcw(0),
            q_interp_data_gcw(1),
            u_data.getPointer(0),
            u_data.getPointer(1),
            q_interp_data.getPointer(0, d),
            q_interp_data.getPointer(1, d)
#endif
#if (NDIM == 3)
                patch_lower(0),
            patch_upper(0),
            patch_lower(1),
            patch_upper(1),
            patch_lower(2),
            patch_upper(2),
            Q_cell_data_gcw(0),
            Q_cell_data_gcw(1),
            Q_cell_data_gcw(2),
            Q0_data.getPointer(d),
            Q1_data.getPointer(),
            Q2_data.getPointer(),
            u_data_gcw(0),
            u_data_gcw(1),
            u_data_gcw(2),
            q_interp_data_gcw(0),
            q_interp_data_gcw(1),
            q_interp_data_gcw(2),
            u_data.getPointer(0),
            u_data.getPointer(1),
            u_data.getPointer(2),
            q_interp_data.getPointer(0, d),
            q_interp_data.getPointer(1, d),
            q_interp_data.getPointer(2, d)
#endif
        );
    }
    return;
} // interpolateToFaceOnPatch

void
LSCUIConvectiveOperator::applyConvectiveOperator(const int Q_idx, const int N_idx)
{
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    // Allocate patch data
    allocate_patch_data({ d_Q_scr_idx, d_valid_idx, d_phi_idx }, d_hierarchy, d_solution_time, coarsest_ln, finest_ln);

    // Prepare valid data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(d_phi_idx);
            Pointer<NodeData<NDIM, int>> valid_data = patch->getPatchData(d_valid_idx);

            Box<NDIM> ghost_node_box = NodeGeometry<NDIM>::toNodeBox(valid_data->getGhostBox());

            phi_data->copy(*ls_data);

            for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();

                if (std::abs((*ls_data)(idx)) < 1.0)
                    (*valid_data)(idx) = 1;
                else
                    (*valid_data)(idx) = 2;
            }

            // Now only compute level set in nearby indices of structure
            for (NodeIterator<NDIM> ni(patch->getBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();

                if ((*valid_data)(idx) == 1)
                {
                    Box<NDIM> region(idx, idx);
                    region.grow(10);
                    for (NodeIterator<NDIM> ni2(region); ni2; ni2++)
                    {
                        const NodeIndex<NDIM>& idx2 = ni2();
                        if (ghost_node_box.contains(idx2) && (*valid_data)(idx2) != 1) (*valid_data)(idx2) = 0;
                    }
                }
            }
        }
    }

    // Generate signed distance function from level set.
    ReinitializeLevelSet ls_method("LS", nullptr);
    ls_method.computeSignedDistanceFunction(d_phi_idx, *d_phi_var, d_hierarchy, d_solution_time, d_valid_idx);

    // Now fill internal boundaries.
    HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(d_hierarchy);
    hier_cc_data_ops.copyData(d_Q_scr_idx, Q_idx);
    InternalBdryFill advect_in_normal("InternalFill", nullptr);
    advect_in_normal.advectInNormal(Q_idx, d_Q_var, d_phi_idx, d_phi_var, d_hierarchy, d_solution_time);

    CellConvectiveOperator::applyConvectiveOperator(Q_idx, N_idx);

    hier_cc_data_ops.copyData(Q_idx, d_Q_scr_idx);

    deallocate_patch_data({ d_Q_scr_idx, d_valid_idx, d_phi_idx }, d_hierarchy, coarsest_ln, finest_ln);
}

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
