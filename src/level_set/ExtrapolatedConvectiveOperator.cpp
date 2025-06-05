#include "ADS/ExtrapolatedConvectiveOperator.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include <ADS/ads_utilities.h>

#include <ibamr/AdvDiffConvectiveOperatorManager.h>

namespace ADS
{

ExtrapolatedConvectiveOperator::ExtrapolatedConvectiveOperator(std::string object_name,
                                                               Pointer<CellVariable<NDIM, double>> Q_var,
                                                               Pointer<Database> input_db,
                                                               const ConvectiveDifferencingType difference_form,
                                                               std::vector<RobinBcCoefStrategy<NDIM>*> bc_coefs,
                                                               int max_gcw_fill)
    : ConvectiveOperator(std::move(object_name), difference_form),
      d_bc_coefs(std::move(bc_coefs)),
      d_max_gcw_fill(max_gcw_fill)
{
    std::string convec_op_type = input_db->getStringWithDefault("convec_op_type", "CUI");
    d_convec_op = AdvDiffConvectiveOperatorManager::getManager()->allocateOperator(
        convec_op_type, d_object_name + "::ConvecOp", Q_var, input_db, d_difference_form, d_bc_coefs);
    return;
} // ExtrapolatedConvectiveOperator

void
ExtrapolatedConvectiveOperator::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& x,
                                                        const SAMRAIVectorReal<NDIM, double>& y)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = x.getPatchHierarchy();
#ifndef NDEBUG
    TBOX_ASSERT(hierarchy == y.getPatchHierarchy());
    TBOX_ASSERT(d_phi_var);
    TBOX_ASSERT(d_phi_idx != IBTK::invalid_index);
#endif

    d_Q_pos_vec = x.cloneVector(d_object_name + "::POS_Q");
    d_Q_neg_vec = x.cloneVector(d_object_name + "::NEG_Q");

    d_N_pos_vec = y.cloneVector(d_object_name + "::POS_N");
    d_N_neg_vec = y.cloneVector(d_object_name + "::NEG_N");

    // Allocate patch data
    d_Q_pos_vec->allocateVectorData(d_solution_time);
    d_Q_neg_vec->allocateVectorData(d_solution_time);
    d_N_pos_vec->allocateVectorData(d_solution_time);
    d_N_neg_vec->allocateVectorData(d_solution_time);

    // Create the internal fill object
    // First create the database for internal fill
    Pointer<Database> db = new MemoryDatabase("InternalFill");
    db->putInteger("max_gcw", d_max_gcw_fill);
    d_internal_fill = std::make_unique<InternalBdryFill>(d_object_name + "::InternalFill", db);

    d_convec_op->setTimeInterval(d_current_time, d_new_time);
    d_convec_op->setSolutionTime(d_solution_time);
    d_convec_op->setAdvectionVelocity(d_u_idx);
    d_convec_op->initializeOperatorState(x, y);
}

void
ExtrapolatedConvectiveOperator::setLSData(const int ls_idx, Pointer<NodeVariable<NDIM, double>> ls_var)
{
    d_phi_idx = ls_idx;
    d_phi_var = ls_var;
}

void
ExtrapolatedConvectiveOperator::deallocateOperatorState()
{
    d_Q_pos_vec->deallocateVectorData();
    d_Q_pos_vec->freeVectorComponents();
    d_Q_pos_vec = nullptr;

    d_Q_neg_vec->deallocateVectorData();
    d_Q_neg_vec->freeVectorComponents();
    d_Q_neg_vec = nullptr;

    d_N_pos_vec->deallocateVectorData();
    d_N_pos_vec->freeVectorComponents();
    d_N_pos_vec = nullptr;

    d_N_neg_vec->deallocateVectorData();
    d_N_neg_vec->freeVectorComponents();
    d_N_neg_vec = nullptr;

    d_internal_fill.reset(nullptr);

    d_convec_op->deallocateOperatorState();
}

void
ExtrapolatedConvectiveOperator::apply(SAMRAIVectorReal<NDIM, double>& x, SAMRAIVectorReal<NDIM, double>& y)
{
    Pointer<PatchHierarchy<NDIM>> hierarchy = x.getPatchHierarchy();
    d_Q_pos_vec->copyVector(Pointer<SAMRAIVectorReal<NDIM, double>>(&x, false));
    d_Q_neg_vec->copyVector(Pointer<SAMRAIVectorReal<NDIM, double>>(&x, false));
    // First extrapolate across the boundary
    std::vector<InternalBdryFill::Parameters> Q_params;
    for (int comp = 0; comp < x.getNumberOfComponents(); ++comp)
    {
        const int Q_pos_idx = d_Q_pos_vec->getComponentDescriptorIndex(comp);
        Pointer<CellVariable<NDIM, double>> Q_pos_var = d_Q_pos_vec->getComponentVariable(comp);

        const int Q_neg_idx = d_Q_neg_vec->getComponentDescriptorIndex(comp);
        Pointer<CellVariable<NDIM, double>> Q_neg_var = d_Q_neg_vec->getComponentVariable(comp);

        Q_params.emplace_back(Q_pos_idx, Q_pos_var, false);
        Q_params.emplace_back(Q_neg_idx, Q_neg_var, true);
    }

    d_internal_fill->advectInNormal(Q_params, d_phi_idx, d_phi_var, hierarchy, d_solution_time);

    // Now apply the convective operator
    d_convec_op->apply(*d_Q_pos_vec, *d_N_pos_vec);
    d_convec_op->apply(*d_Q_neg_vec, *d_N_neg_vec);

    // Now recreate the vector
    const int coarsest_ln = 0;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(d_phi_idx);
            for (int comp = 0; comp < x.getNumberOfComponents(); ++comp)
            {
                Pointer<CellData<NDIM, double>> N_pos_data =
                    patch->getPatchData(d_N_pos_vec->getComponentDescriptorIndex(comp));
                Pointer<CellData<NDIM, double>> N_neg_data =
                    patch->getPatchData(d_N_neg_vec->getComponentDescriptorIndex(comp));
                Pointer<CellData<NDIM, double>> N_data = patch->getPatchData(y.getComponentDescriptorIndex(comp));

                for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
                {
                    const CellIndex<NDIM>& idx = ci();
                    const double ls = ADS::node_to_cell(idx, *phi_data);
                    if (ls > 0.0)
                        (*N_data)(idx) = (*N_pos_data)(idx);
                    else
                        (*N_data)(idx) = (*N_neg_data)(idx);
                }
            }
        }
    }
}

void
ExtrapolatedConvectiveOperator::applyConvectiveOperator(int Q_idx, int N_idx)
{
}

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
