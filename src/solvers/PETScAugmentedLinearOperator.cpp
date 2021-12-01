/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/PETScAugmentedLinearOperator.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep

#include "ibtk/IBTK_CHKERRQ.h"

#include "Box.h"
#include "SAMRAIVectorReal.h"
#include "tbox/Pointer.h"

#include <string>
#include <utility>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

PETScAugmentedLinearOperator::PETScAugmentedLinearOperator(std::string object_name, bool homogeneous_bc)
    : LinearOperator(std::move(object_name), homogeneous_bc)
{
    // intentionally blank
    return;
} // PETScAugmentedLinearOperator()

PETScAugmentedLinearOperator::~PETScAugmentedLinearOperator()
{
    deallocateOperatorState();
    return;
} // ~PETScAugmentedLinearOperator()

void
PETScAugmentedLinearOperator::setAugmentedVec(const Vec& vec)
{
    d_aug_x_vec = vec;
}

const Vec&
PETScAugmentedLinearOperator::getAugmentedVec() const
{
    return d_aug_y_vec;
}

void
PETScAugmentedLinearOperator::setAugmentedRhsForBcs(Vec& aug_y)
{
    d_aug_rhs_y_vec = aug_y;
}

void
PETScAugmentedLinearOperator::modifyRhsForBcs(SAMRAIVectorReal<NDIM, double>& y)
{
    if (d_homogeneous_bc) return;

    // Set y := y - A*0, i.e., shift the right-hand-side vector to account for
    // inhomogeneous boundary conditions.
    // Prepare copies for Cartesian data
    Pointer<SAMRAIVectorReal<NDIM, double>> x = y.cloneVector("");
    Pointer<SAMRAIVectorReal<NDIM, double>> b = y.cloneVector("");
    x->allocateVectorData();
    b->allocateVectorData();
    x->setToScalar(0.0);
    // Prepare copies for augmented data.
    Vec temp_x_copy;

    int ierr = VecDuplicate(d_aug_x_vec, &temp_x_copy);
    IBTK_CHKERRQ(ierr);
    ierr = VecCopy(d_aug_x_vec, temp_x_copy);
    IBTK_CHKERRQ(ierr);
    ierr = VecZeroEntries(d_aug_x_vec);
    IBTK_CHKERRQ(ierr);
    // Apply the operator. Note that apply() also works on d_aug_x_vec and d_aug_y_vec.
    apply(*x, *b);
    // Now subtract y from the result.
    y.subtract(Pointer<SAMRAIVectorReal<NDIM, double>>(&y, false), b);
    // And subtract d_aug_y_vec from temp_y_copy.
    ierr = VecAXPY(d_aug_rhs_y_vec, -1.0, d_aug_y_vec);
    IBTK_CHKERRQ(ierr);
    // Now reset and free the vectors
    x->freeVectorComponents();
    b->freeVectorComponents();
    ierr = VecCopy(temp_x_copy, d_aug_x_vec);
    IBTK_CHKERRQ(ierr);
    ierr = VecDestroy(&temp_x_copy);
    IBTK_CHKERRQ(ierr);
    return;
} // modifyRhsForBcs

void
PETScAugmentedLinearOperator::applyAdd(SAMRAIVectorReal<NDIM, double>& x,
                                       SAMRAIVectorReal<NDIM, double>& y,
                                       SAMRAIVectorReal<NDIM, double>& z)
{
    TBOX_ERROR(d_object_name + "::applyAdd() is not currently implemented.\n");
    return;
}

//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
