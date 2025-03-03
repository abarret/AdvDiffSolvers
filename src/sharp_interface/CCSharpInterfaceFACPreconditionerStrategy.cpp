#include "ADS/CCSharpInterfaceFACPreconditionerStrategy.h"
#include "ADS/ads_utilities.h"
#include "ADS/app_namespaces.h"
#include "ADS/sharp_interface_utilities.h"

#include "ibtk/CartCellDoubleQuadraticCFInterpolation.h"
#include "ibtk/CoarseFineBoundaryRefinePatchStrategy.h"
#include "ibtk/ExtendedRobinBcCoefStrategy.h"
#include "ibtk/FACPreconditionerStrategy.h"
#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/RefinePatchStrategySet.h"
#include "ibtk/RobinPhysBdryPatchStrategy.h"
#include "ibtk/ibtk_utilities.h"

#include "Box.h"
#include "CartesianGridGeometry.h"
#include "CoarsenAlgorithm.h"
#include "CoarsenOperator.h"
#include "CoarsenSchedule.h"
#include "HierarchyDataOpsManager.h"
#include "HierarchyDataOpsReal.h"
#include "LocationIndexRobinBcCoefs.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "PoissonSpecifications.h"
#include "RefineAlgorithm.h"
#include "RefineOperator.h"
#include "RefinePatchStrategy.h"
#include "RefineSchedule.h"
#include "RobinBcCoefStrategy.h"
#include "SAMRAIVectorReal.h"
#include "Variable.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "VariableFillPattern.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"
#include "tbox/Timer.h"
#include "tbox/TimerManager.h"
#include "tbox/Utilities.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{

namespace sharp_interface
{
/////////////////////////////// STATIC ///////////////////////////////////////

namespace
{
// Timers.
static Timer* t_restrict_residual;
static Timer* t_prolong_error;
static Timer* t_prolong_error_and_correct;
static Timer* t_initialize_operator_state;
static Timer* t_deallocate_operator_state;
static Timer* t_smooth_error;
static Timer* t_compute_residual;
static Timer* t_solve_coarsest_level;
} // namespace

/////////////////////////////// PUBLIC ///////////////////////////////////////

CCSharpInterfaceFACPreconditionerStrategy::CCSharpInterfaceFACPreconditionerStrategy(
    std::string object_name,
    std::vector<FESystemManager*> fe_sys_managers,
    Database* input_db)
    : FACPreconditionerStrategy(std::move(object_name)),
      d_default_bc_coef(
          new LocationIndexRobinBcCoefs<NDIM>(d_object_name + "::default_bc_coef", Pointer<Database>(nullptr))),
      d_bc_coefs(1, d_default_bc_coef.get()),
      d_gcw(1),
      d_fe_sys_managers(std::move(fe_sys_managers)),
      d_idx_elem_mapping(new IndexElemMapping(d_object_name + "::IndexElemMapping", true)),
      d_scratch_var(new CellVariable<NDIM, double>(d_object_name + "::ScratchVar"))
{
    // Setup a default boundary condition object that specifies homogeneous
    // Dirichlet boundary conditions.
    auto p_default_bc_coef = dynamic_cast<LocationIndexRobinBcCoefs<NDIM>*>(d_default_bc_coef.get());
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        p_default_bc_coef->setBoundaryValue(2 * d, 0.0);
        p_default_bc_coef->setBoundaryValue(2 * d + 1, 0.0);
    }

    // Setup scratch variables.
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
    d_context = var_db->getContext(d_object_name + "::CONTEXT");
    const IntVector<NDIM> ghosts = d_gcw;
    if (var_db->checkVariableExists(d_scratch_var->getName()))
    {
        d_scratch_var = var_db->getVariable(d_scratch_var->getName());
        d_scratch_idx = var_db->mapVariableAndContextToIndex(d_scratch_var, d_context);
        var_db->removePatchDataIndex(d_scratch_idx);
    }
    d_scratch_idx = var_db->registerVariableAndContext(d_scratch_var, d_context, ghosts);

    // Grab values from the optional database
    if (input_db)
    {
        d_w = input_db->getDoubleWithDefault("relaxation_parameter", d_w);
        d_num_sweeps_on_coarsest_level =
            input_db->getDoubleWithDefault("sweeps_on_coarsest", d_num_sweeps_on_coarsest_level);
    }

    // Setup Timers.
    IBTK_DO_ONCE(
        t_restrict_residual =
            TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceFACPreconditionerStrategy::restrictResidual()");
        t_prolong_error =
            TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceFACPreconditionerStrategy::prolongError()");
        t_prolong_error_and_correct = TimerManager::getManager()->getTimer(
            "ADS::CCSharpInterfaceFACPreconditionerStrategy::prolongErrorAndCorrect()");
        t_initialize_operator_state = TimerManager::getManager()->getTimer(
            "ADS::CCSharpInterfaceFACPreconditionerStrategy::initializeOperatorState()");
        t_deallocate_operator_state = TimerManager::getManager()->getTimer(
            "ADS::CCSharpInterfaceFACPreconditionerStrategy::deallocateOperatorState()");
        t_smooth_error =
            TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceFACPreconditionerStrategy::smoothError()");
        t_compute_residual =
            TimerManager::getManager()->getTimer("ADS::CCSharpInterfaceFACPreconditionerStrategy::computeResidual()");
        t_solve_coarsest_level = TimerManager::getManager()->getTimer(
            "ADS::CCSharpInterfaceFACPreconditionerStrategy::solveCoarsestLevel()"););
    return;
} // CCSharpInterfaceFACPreconditionerStrategy

CCSharpInterfaceFACPreconditionerStrategy::~CCSharpInterfaceFACPreconditionerStrategy()
{
    if (d_is_initialized)
    {
        TBOX_ERROR(d_object_name << "::~CCSharpInterfaceFACPreconditionerStrategy()\n"
                                 << "  subclass must call deallocateOperatorState in subclass destructor" << std::endl);
    }
    return;
} // ~CCSharpInterfaceFACPreconditionerStrategy

void
CCSharpInterfaceFACPreconditionerStrategy::setPhysicalBcCoef(RobinBcCoefStrategy<NDIM>* const bc_coef)
{
    setPhysicalBcCoefs(std::vector<RobinBcCoefStrategy<NDIM>*>(1, bc_coef));
    return;
} // setPhysicalBcCoef

void
CCSharpInterfaceFACPreconditionerStrategy::setPhysicalBcCoefs(const std::vector<RobinBcCoefStrategy<NDIM>*>& bc_coefs)
{
    d_bc_coefs.resize(bc_coefs.size());
    for (unsigned int l = 0; l < bc_coefs.size(); ++l)
    {
        if (bc_coefs[l])
        {
            d_bc_coefs[l] = bc_coefs[l];
        }
        else
        {
            d_bc_coefs[l] = d_default_bc_coef.get();
        }
    }
    return;
} // setPhysicalBcCoefs

void
CCSharpInterfaceFACPreconditionerStrategy::setToZero(SAMRAIVectorReal<NDIM, double>& vec, int level_num)
{
    plog << "Setting to zero on level " << level_num << "\n";
    const int data_idx = vec.getComponentDescriptorIndex(0);
    d_level_data_ops[level_num]->setToScalar(data_idx, 0.0, /*interior_only*/ false);
    return;
} // setToZero

void
CCSharpInterfaceFACPreconditionerStrategy::restrictResidual(const SAMRAIVectorReal<NDIM, double>& src,
                                                            SAMRAIVectorReal<NDIM, double>& dst,
                                                            int dst_ln)
{
    plog << "Restricting residual to level " << dst_ln << "\n";
    IBTK_TIMER_START(t_restrict_residual);

    const int src_idx = src.getComponentDescriptorIndex(0);
    const int dst_idx = dst.getComponentDescriptorIndex(0);

    if (src_idx != dst_idx)
    {
        d_level_data_ops[dst_ln]->copyData(dst_idx, src_idx, /*interior_only*/ false);
    }
    xeqScheduleRestriction(dst_idx, src_idx, dst_ln);

    IBTK_TIMER_STOP(t_restrict_residual);
    return;
} // restrictResidual

void
CCSharpInterfaceFACPreconditionerStrategy::prolongError(const SAMRAIVectorReal<NDIM, double>& src,
                                                        SAMRAIVectorReal<NDIM, double>& dst,
                                                        int dst_ln)
{
    plog << "Prolonging error to level " << dst_ln << "\n";
    IBTK_TIMER_START(t_prolong_error);

    const int dst_idx = dst.getComponentDescriptorIndex(0);
    const int src_idx = src.getComponentDescriptorIndex(0);

    // Prolong the correction from the coarse src level data into the fine level
    // dst data.
    xeqScheduleProlongation(dst_idx, src_idx, dst_ln);

    IBTK_TIMER_STOP(t_prolong_error);
    return;
} // prolongError

void
CCSharpInterfaceFACPreconditionerStrategy::prolongErrorAndCorrect(const SAMRAIVectorReal<NDIM, double>& src,
                                                                  SAMRAIVectorReal<NDIM, double>& dst,
                                                                  int dst_ln)
{
    plog << "Prolonging and correcting to level " << dst_ln << "\n";
    IBTK_TIMER_START(t_prolong_error_and_correct);

    const int dst_idx = dst.getComponentDescriptorIndex(0);
    const int src_idx = src.getComponentDescriptorIndex(0);

    // Prolong the correction from the coarse level src data into the fine level
    // scratch data and then correct the fine level dst data.
    if (src_idx != dst_idx)
    {
        d_level_data_ops[dst_ln - 1]->add(dst_idx, dst_idx, src_idx, /*interior_only*/ false);
    }
    Pointer<PatchLevel<NDIM>> crs_level = d_hierarchy->getPatchLevel(dst_ln - 1);
    for (PatchLevel<NDIM>::Iterator p(crs_level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = crs_level->getPatch(p());
        Pointer<CellData<NDIM, double>> src_data = patch->getPatchData(src_idx);
        src_data->fillAll(dst_ln - 1);
    }
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(dst_ln);
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<CellData<NDIM, double>> scr_data = patch->getPatchData(d_scratch_idx);
        Pointer<CellData<NDIM, double>> src_data = patch->getPatchData(src_idx);
        src_data->fillAll(dst_ln);
    }
    xeqScheduleProlongation(d_scratch_idx, src_idx, dst_ln);
    d_level_data_ops[dst_ln]->add(dst_idx, dst_idx, d_scratch_idx, /*interior_only*/ false);

    IBTK_TIMER_STOP(t_prolong_error_and_correct);
    return;
} // prolongErrorAndCorrect

void
CCSharpInterfaceFACPreconditionerStrategy::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& solution,
                                                                   const SAMRAIVectorReal<NDIM, double>& rhs)
{
    IBTK_TIMER_START(t_initialize_operator_state);

    // Deallocate the solver state if the solver is already initialized.
    if (d_is_initialized) deallocateOperatorState();

    // Setup solution and rhs vectors.
    d_solution = solution.cloneVector(solution.getName());
    d_rhs = rhs.cloneVector(rhs.getName());

    Pointer<hier::Variable<NDIM>> sol_var = d_solution->getComponentVariable(0);
    const int sol_idx = d_solution->getComponentDescriptorIndex(0);
    const int rhs_idx = d_rhs->getComponentDescriptorIndex(0);

    // Reset the hierarchy configuration.
    d_hierarchy = solution.getPatchHierarchy();
    d_coarsest_ln = solution.getCoarsestLevelNumber();
    d_finest_ln = solution.getFinestLevelNumber();

    d_bc_op = new CartCellRobinPhysBdryOp(d_scratch_idx, d_bc_coefs, false);
    d_cf_bdry_op = new CartCellDoubleQuadraticCFInterpolation();
    d_cf_bdry_op->setConsistentInterpolationScheme(false);
    d_cf_bdry_op->setPatchHierarchy(d_hierarchy);

    // Set up sharp interface ghost filling
    int num_parts = d_fe_sys_managers.size();
    d_fe_hierarchy_mapping.reserve(num_parts);
    for (int part = 0; part < num_parts; ++part)
    {
        d_fe_hierarchy_mapping.push_back(
            std::make_unique<FEToHierarchyMapping>(d_object_name + "::FEToHierarchyMapping_" + std::to_string(part),
                                                   d_fe_sys_managers[part],
                                                   nullptr,
                                                   d_hierarchy->getNumberOfLevels(),
                                                   d_gcw));
        d_fe_hierarchy_mapping[part]->setPatchHierarchy(d_hierarchy);
        d_fe_hierarchy_mapping[part]->reinitElementMappings(d_gcw);
    }

    d_si_ghost_fill = std::make_unique<SharpInterfaceGhostFill>(d_object_name + "::SIGhostFill",
                                                                unique_ptr_vec_to_raw_ptr_vec(d_fe_hierarchy_mapping),
                                                                d_idx_elem_mapping,
                                                                d_coarsest_ln,
                                                                d_finest_ln);

    // Setup level operators.
    d_level_data_ops.resize(d_finest_ln + 1);
    HierarchyDataOpsManager<NDIM>* hier_data_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    for (int ln = d_coarsest_ln; ln <= d_finest_ln; ++ln)
    {
        d_level_data_ops[ln] = hier_data_ops_manager->getOperationsDouble(sol_var,
                                                                          d_hierarchy,
                                                                          /*get_unique*/ true);
        d_level_data_ops[ln]->resetLevels(ln, ln);
    }

    // Allocate scratch data.
    for (int ln = d_coarsest_ln; ln <= d_finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(d_scratch_idx)) level->allocatePatchData(d_scratch_idx);
    }

    // Get overlap information for setting patch boundary conditions.
    d_patch_bc_box_overlap.resize(d_finest_ln + 1);
    for (int ln = d_coarsest_ln; ln <= d_finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const int num_local_patches = level->getProcessorMapping().getLocalIndices().getSize();
        d_patch_bc_box_overlap[ln].resize(num_local_patches);
        int patch_counter = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Box<NDIM>& ghost_box = Box<NDIM>::grow(patch_box, 1);
            d_patch_bc_box_overlap[ln][patch_counter] = BoxList<NDIM>(ghost_box);
            d_patch_bc_box_overlap[ln][patch_counter].removeIntersections(patch_box);
        }
    }

    // Get the transfer operators.
    Pointer<CartesianGridGeometry<NDIM>> geometry = d_hierarchy->getGridGeometry();
    d_prolongation_refine_operator = geometry->lookupRefineOperator(sol_var, d_prolongation_method);
    d_restriction_coarsen_operator = geometry->lookupCoarsenOperator(sol_var, d_restriction_method);
    d_cf_bdry_op->setConsistentInterpolationScheme(false);
    d_cf_bdry_op->setPatchDataIndex(d_scratch_idx);
    d_cf_bdry_op->setPatchHierarchy(d_hierarchy);

    // Make space for saving communication schedules.  There is no need to
    // delete the old schedules first because we have deallocated the solver
    // state above.
    std::vector<RefinePatchStrategy<NDIM>*> prolongation_refine_patch_strategies;
    prolongation_refine_patch_strategies.push_back(d_cf_bdry_op);
    prolongation_refine_patch_strategies.push_back(d_bc_op);
    d_prolongation_refine_patch_strategy = new RefinePatchStrategySet(
        prolongation_refine_patch_strategies.begin(), prolongation_refine_patch_strategies.end(), false);

    d_prolongation_refine_schedules.resize(d_finest_ln + 1);
    d_restriction_coarsen_schedules.resize(d_finest_ln);
    d_ghostfill_nocoarse_refine_schedules.resize(d_finest_ln + 1);
    d_synch_refine_schedules.resize(d_finest_ln + 1);

    d_prolongation_refine_algorithm = new RefineAlgorithm<NDIM>();
    d_restriction_coarsen_algorithm = new CoarsenAlgorithm<NDIM>();
    d_ghostfill_nocoarse_refine_algorithm = new RefineAlgorithm<NDIM>();
    d_synch_refine_algorithm = new RefineAlgorithm<NDIM>();

    d_prolongation_refine_algorithm->registerRefine(
        d_scratch_idx, sol_idx, d_scratch_idx, d_prolongation_refine_operator, d_synch_fill_pattern);
    d_restriction_coarsen_algorithm->registerCoarsen(d_scratch_idx, rhs_idx, d_restriction_coarsen_operator);
    d_ghostfill_nocoarse_refine_algorithm->registerRefine(
        sol_idx, sol_idx, sol_idx, Pointer<RefineOperator<NDIM>>(), d_synch_fill_pattern);
    d_synch_refine_algorithm->registerRefine(
        sol_idx, sol_idx, sol_idx, Pointer<RefineOperator<NDIM>>(), d_synch_fill_pattern);

    // TODO: Here we take a pessimistic approach and are recreating refine schedule for
    // (coarsest_reset_ln - 1) level as well.
    for (int dst_ln = d_coarsest_ln + 1; dst_ln <= d_finest_ln; ++dst_ln)
    {
        d_prolongation_refine_schedules[dst_ln] =
            d_prolongation_refine_algorithm->createSchedule(d_hierarchy->getPatchLevel(dst_ln),
                                                            Pointer<PatchLevel<NDIM>>(),
                                                            dst_ln - 1,
                                                            d_hierarchy,
                                                            d_prolongation_refine_patch_strategy.getPointer());
    }

    for (int dst_ln = d_coarsest_ln; dst_ln < d_finest_ln; ++dst_ln)
    {
        d_restriction_coarsen_schedules[dst_ln] = d_restriction_coarsen_algorithm->createSchedule(
            d_hierarchy->getPatchLevel(dst_ln), d_hierarchy->getPatchLevel(dst_ln + 1));
    }

    for (int ln = d_coarsest_ln; ln <= d_finest_ln; ++ln)
    {
        d_ghostfill_nocoarse_refine_schedules[ln] =
            d_ghostfill_nocoarse_refine_algorithm->createSchedule(d_hierarchy->getPatchLevel(ln), d_bc_op.getPointer());
        d_synch_refine_schedules[ln] = d_synch_refine_algorithm->createSchedule(d_hierarchy->getPatchLevel(ln));
    }

    // Indicate that the operator is initialized.
    d_is_initialized = true;

    IBTK_TIMER_STOP(t_initialize_operator_state);
    return;
} // initializeOperatorState

void
CCSharpInterfaceFACPreconditionerStrategy::deallocateOperatorState()
{
    if (!d_is_initialized) return;

    IBTK_TIMER_START(t_deallocate_operator_state);

    // Deallocate scratch data.
    deallocate_patch_data(d_scratch_idx, d_hierarchy, d_coarsest_ln, d_finest_ln);

    // Delete the solution and rhs vectors.
    d_solution->freeVectorComponents();
    d_solution.setNull();

    d_rhs->freeVectorComponents();
    d_rhs.setNull();

    d_hierarchy.setNull();
    d_coarsest_ln = -1;
    d_finest_ln = -1;

    d_prolongation_refine_operator.setNull();
    d_prolongation_refine_patch_strategy.setNull();
    d_prolongation_refine_algorithm.setNull();
    d_prolongation_refine_schedules.resize(0);

    d_restriction_coarsen_operator.setNull();
    d_restriction_coarsen_algorithm.setNull();
    d_restriction_coarsen_schedules.resize(0);

    d_ghostfill_nocoarse_refine_algorithm.setNull();
    d_ghostfill_nocoarse_refine_schedules.resize(0);

    d_synch_refine_algorithm.setNull();
    d_synch_refine_schedules.resize(0);

    // Indicate that the operator is not initialized.
    d_is_initialized = false;

    IBTK_TIMER_STOP(t_deallocate_operator_state);
    return;
} // deallocateOperatorState

void
CCSharpInterfaceFACPreconditionerStrategy::smoothError(SAMRAIVectorReal<NDIM, double>& error,
                                                       const SAMRAIVectorReal<NDIM, double>& residual,
                                                       const int level_num,
                                                       int num_sweeps,
                                                       const bool performing_pre_sweeps,
                                                       const bool performing_post_sweeps)
{
    plog << "Smoothing error on level " << level_num << " with " << num_sweeps << " sweeps.\n";
    if (num_sweeps == 0) return;
    ADS_TIMER_START(t_smooth_error);

    num_sweeps *= 2;

    // Grab the vector components.
    const int u_idx = error.getComponentDescriptorIndex(0);
    const int f_idx = residual.getComponentDescriptorIndex(0);

#ifndef NDEBUG
    TBOX_ASSERT(d_hierarchy.getPointer() == error.getPatchHierarchy().getPointer());
#endif

    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(level_num);

    // Cache coarse-fine interface ghost cell values in the "scratch" data.
    // Note we cache these values so that we don't have to communicate coarse to fine after the first sweep.
    // TODO: Should check if this is necessary. This also assumes that coarse
    if (level_num > 0 && num_sweeps > 1)
    {
        int patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CellData<NDIM, double>> u_scr_data = patch->getPatchData(d_scratch_idx);

            u_scr_data->getArrayData().copy(
                u_data->getArrayData(), d_patch_bc_box_overlap[level_num][patch_num], IntVector<NDIM>(0));
        }
    }

    // Loop over sweeps
    for (int sweep = 0; sweep < num_sweeps; ++sweep)
    {
        if (level_num > 0)
        {
            if (sweep > 0)
            {
                // Copy the cached coarse-fine ghost cell values
                int patch_num = 0;
                for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());

                    Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(u_idx);
                    Pointer<CellData<NDIM, double>> u_scr_data = patch->getPatchData(d_scratch_idx);

                    u_data->getArrayData().copy(
                        u_scr_data->getArrayData(), d_patch_bc_box_overlap[level_num][patch_num], IntVector<NDIM>(0));
                }
            }

            // Fill ghost cells. We only use values on our current level to fill in ghost cells.
            xeqScheduleGhostFillNoCoarse(u_idx, level_num);

            // Compute the normal extension of the solution at coarse-fine interfaces.
            d_cf_bdry_op->setPatchDataIndex(u_idx);
            const IntVector<NDIM>& ratio = level->getRatioToCoarserLevel();
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                d_cf_bdry_op->computeNormalExtension(*patch, ratio, d_gcw);
            }
        }
        else if (sweep > 0)
        {
            xeqScheduleGhostFillNoCoarse(u_idx, level_num);
        }

        const std::vector<ImagePointWeightsMap>& img_wgts_vec = d_si_ghost_fill->getImagePointWeights(level_num);
        int local_patch_num = 0;

        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            double denom = 0.0;
            for (int d = 0; d < NDIM; ++d) denom -= 2.0 / (dx[d] * dx[d]);

            Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CellData<NDIM, double>> f_data = patch->getPatchData(f_idx);

            Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(d_si_ghost_fill->getIndexPatchIndex());
            const ImagePointWeightsMap& img_wgts = img_wgts_vec[local_patch_num];

            CellIterator<NDIM> ci(patch->getBox());
            if (sweep % 2 == 1) ci++;
            for (; ci; ci++)
            {
                ci++;
                if (!ci) break;
                const CellIndex<NDIM>& idx = ci();
                const int idx_val = (*i_data)(idx);
                double new_val = 0.0;
                if (idx_val == FLUID)
                {
                    double val = (*f_data)(idx);
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> one(0);
                        one(d) = 1;
                        val -= ((*u_data)(idx + one) + (*u_data)(idx - one)) / (dx[d] * dx[d]);
                    }
                    new_val = val / denom;
                }
                else if (idx_val == GHOST)
                {
                    const ImagePointWeights& wgts = img_wgts.at(std::make_pair(idx, patch));
                    double val = (*f_data)(idx);
                    double gp_wgt = 1.0;
                    for (int i = 0; i < wgts.s_num_pts; ++i)
                    {
                        if (wgts.d_idxs[i] == idx)
                        {
                            gp_wgt += wgts.d_weights[i];
                        }
                        else
                        {
                            val -= wgts.d_weights[i] * (*u_data)(wgts.d_idxs[i]);
                        }
                    }
                    new_val = val / gp_wgt;
                }
                else if (idx_val == INVALID)
                {
                    new_val = 0.0;
                }
                (*u_data)(idx) = d_w * new_val + (1.0 - d_w) * (*u_data)(idx);
            }
        }
    }

    ADS_TIMER_STOP(t_smooth_error);
}

void
CCSharpInterfaceFACPreconditionerStrategy::computeResidual(SAMRAIVectorReal<NDIM, double>& residual,
                                                           const SAMRAIVectorReal<NDIM, double>& solution,
                                                           const SAMRAIVectorReal<NDIM, double>& rhs,
                                                           int coarsest_level_num,
                                                           int finest_level_num)
{
    plog << "Computing residual from level " << coarsest_level_num << " to " << finest_level_num << "\n";
    ADS_TIMER_START(t_compute_residual);
    const int u_idx = solution.getComponentDescriptorIndex(0);
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_comps = { ITC(
        u_idx, "LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", true, d_bc_coefs, d_synch_fill_pattern) };
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(ghost_comps, d_hierarchy, coarsest_level_num, finest_level_num);
    hier_ghost_fill.setHomogeneousBc(d_homogeneous_bc);
    hier_ghost_fill.fillData(d_solution_time);

    for (int ln = coarsest_level_num; ln <= finest_level_num; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);

        const std::vector<ImagePointWeightsMap>& img_wgts_vec = d_si_ghost_fill->getImagePointWeights(ln);
        const std::vector<std::vector<ImagePointData>>& img_data_vec = d_si_ghost_fill->getImagePointData(ln);
        std::function<double(const IBTK::VectorNd& x)> bdry_fcn = d_bdry_fcn;
        if (d_homogeneous_bc) bdry_fcn = [](const VectorNd&) -> double { return 0.0; };
        int local_patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++local_patch_num)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(u_idx);
            Pointer<CellData<NDIM, double>> b_data = patch->getPatchData(residual.getComponentDescriptorIndex(0));
            Pointer<CellData<NDIM, double>> rhs_data = patch->getPatchData(rhs.getComponentDescriptorIndex(0));
            Pointer<CellData<NDIM, int>> i_data = patch->getPatchData(d_si_ghost_fill->getIndexPatchIndex());

            apply_laplacian_on_patch(patch, img_wgts_vec[local_patch_num], *u_data, *b_data, *i_data);

            // Now difference rhs and residual.
            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                (*b_data)(idx) = (*rhs_data)(idx) - (*b_data)(idx);
            }
        }
    }
    ADS_TIMER_STOP(t_compute_residual);
}

bool
CCSharpInterfaceFACPreconditionerStrategy::solveCoarsestLevel(SAMRAIVectorReal<NDIM, double>& error,
                                                              const SAMRAIVectorReal<NDIM, double>& residual,
                                                              int coarsest_ln)
{
    plog << "Solving on coarsest level " << coarsest_ln << "\n";
    ADS_TIMER_START(t_solve_coarsest_level);

    smoothError(error, residual, coarsest_ln, d_num_sweeps_on_coarsest_level, false, false);

    ADS_TIMER_STOP(t_solve_coarsest_level);
    return true;
} // solveCoarsestLevel

/////////////////////////////// PROTECTED ////////////////////////////////////

void
CCSharpInterfaceFACPreconditionerStrategy::xeqScheduleProlongation(const int dst_idx,
                                                                   const int src_idx,
                                                                   const int dst_ln)
{
    plog << "Prolonging " << src_idx << " to " << dst_idx << " to level " << dst_ln << "\n";
    d_cf_bdry_op->setPatchDataIndex(dst_idx);
    d_bc_op->setPatchDataIndex(dst_idx);
    d_bc_op->setPhysicalBcCoefs(d_bc_coefs);
    d_bc_op->setHomogeneousBc(d_homogeneous_bc);
    for (const auto& bc_coef : d_bc_coefs)
    {
        auto extended_bc_coef = dynamic_cast<ExtendedRobinBcCoefStrategy*>(bc_coef);
        if (extended_bc_coef)
        {
            extended_bc_coef->setTargetPatchDataIndex(dst_idx);
            extended_bc_coef->setHomogeneousBc(d_homogeneous_bc);
        }
    }
    RefineAlgorithm<NDIM> refiner;
    refiner.registerRefine(dst_idx, src_idx, dst_idx, d_prolongation_refine_operator, d_synch_fill_pattern);
    refiner.resetSchedule(d_prolongation_refine_schedules[dst_ln]);
    d_prolongation_refine_schedules[dst_ln]->fillData(d_solution_time);
    d_prolongation_refine_algorithm->resetSchedule(d_prolongation_refine_schedules[dst_ln]);
    for (const auto& bc_coef : d_bc_coefs)
    {
        auto extended_bc_coef = dynamic_cast<ExtendedRobinBcCoefStrategy*>(bc_coef);
        if (extended_bc_coef) extended_bc_coef->clearTargetPatchDataIndex();
    }
    return;
} // xeqScheduleProlongation

void
CCSharpInterfaceFACPreconditionerStrategy::xeqScheduleRestriction(const int dst_idx,
                                                                  const int src_idx,
                                                                  const int dst_ln)
{
    CoarsenAlgorithm<NDIM> coarsener;
    coarsener.registerCoarsen(dst_idx, src_idx, d_restriction_coarsen_operator);
    coarsener.resetSchedule(d_restriction_coarsen_schedules[dst_ln]);
    d_restriction_coarsen_schedules[dst_ln]->coarsenData();
    d_restriction_coarsen_algorithm->resetSchedule(d_restriction_coarsen_schedules[dst_ln]);
    return;
} // xeqScheduleRestriction

void
CCSharpInterfaceFACPreconditionerStrategy::xeqScheduleGhostFillNoCoarse(const int dst_idx, const int dst_ln)
{
    d_bc_op->setPatchDataIndex(dst_idx);
    d_bc_op->setPhysicalBcCoefs(d_bc_coefs);
    d_bc_op->setHomogeneousBc(d_homogeneous_bc);
    for (const auto& bc_coef : d_bc_coefs)
    {
        auto extended_bc_coef = dynamic_cast<ExtendedRobinBcCoefStrategy*>(bc_coef);
        if (extended_bc_coef)
        {
            extended_bc_coef->setTargetPatchDataIndex(dst_idx);
            extended_bc_coef->setHomogeneousBc(d_homogeneous_bc);
        }
    }
    RefineAlgorithm<NDIM> refiner;
    refiner.registerRefine(dst_idx, dst_idx, dst_idx, Pointer<RefineOperator<NDIM>>(), d_synch_fill_pattern);
    refiner.resetSchedule(d_ghostfill_nocoarse_refine_schedules[dst_ln]);
    d_ghostfill_nocoarse_refine_schedules[dst_ln]->fillData(d_solution_time);
    d_ghostfill_nocoarse_refine_algorithm->resetSchedule(d_ghostfill_nocoarse_refine_schedules[dst_ln]);
    for (const auto& bc_coef : d_bc_coefs)
    {
        auto extended_bc_coef = dynamic_cast<ExtendedRobinBcCoefStrategy*>(bc_coef);
        if (extended_bc_coef) extended_bc_coef->clearTargetPatchDataIndex();
    }
    return;
} // xeqScheduleGhostFillNoCoarse

void
CCSharpInterfaceFACPreconditionerStrategy::xeqScheduleDataSynch(const int dst_idx, const int dst_ln)
{
    RefineAlgorithm<NDIM> refiner;
    refiner.registerRefine(dst_idx, dst_idx, dst_idx, Pointer<RefineOperator<NDIM>>(), d_synch_fill_pattern);
    refiner.resetSchedule(d_synch_refine_schedules[dst_ln]);
    d_synch_refine_schedules[dst_ln]->fillData(d_solution_time);
    d_synch_refine_algorithm->resetSchedule(d_synch_refine_schedules[dst_ln]);
    return;
} // xeqScheduleDataSynch

} // namespace sharp_interface
} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
