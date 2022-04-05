// ---------------------------------------------------------------------
//
// Copyright (c) 2021 - 2021 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/FDWeightsCache.h"
#include "ADS/KDTree.h"
#include "ADS/PolynomialBasis.h"
#include "ADS/app_namespaces.h" // IWYU pragma: keep
#include "ADS/ls_functions.h"
#include "ADS/reconstructions.h"

#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/IBTK_CHKERRQ.h"
#include "ibtk/IndexUtilities.h"
#include "ibtk/ibtk_utilities.h"

#include "CellVariable.h"
#include "MultiblockDataTranslator.h"
#include "PatchHierarchy.h"
#include "PoissonSpecifications.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Timer.h"

#include <Eigen/Dense>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace ADS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

FDWeightsCache::FDWeightsCache(std::string object_name) : d_object_name(std::move(object_name))
{
    // intentionally blank
    return;
}

FDWeightsCache::~FDWeightsCache()
{
    clearCache();
    return;
}

void
FDWeightsCache::clearCache()
{
    d_base_pt_set.clear();
    d_pair_pt_map.clear();
    d_pt_weight_map.clear();
}

void
FDWeightsCache::clearPoint(Pointer<Patch<NDIM>> patch, const FDPoint& pt)
{
    std::set<FDPoint>& base_pt_set = d_base_pt_set[patch.getPointer()];
    auto it = base_pt_set.find(pt);
    if (it == base_pt_set.end()) return;
    base_pt_set.erase(it);
    d_pair_pt_map[patch.getPointer()].erase(pt);
    d_pt_weight_map[patch.getPointer()].erase(pt);
}

void
FDWeightsCache::cachePoint(Pointer<Patch<NDIM>> patch,
                           const FDPoint& pt,
                           const std::vector<FDPoint>& fd_pts,
                           const std::vector<double>& fd_weights)
{
    // Cache the point. Replace it if it already exists
    d_base_pt_set[patch.getPointer()].insert(pt);
    d_pair_pt_map[patch.getPointer()][pt] = fd_pts;
    d_pt_weight_map[patch.getPointer()][pt] = fd_weights;
}

const std::map<FDPoint, std::vector<double>>&
FDWeightsCache::getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) const
{
    return d_pt_weight_map.at(patch.getPointer());
}

const std::map<FDPoint, std::vector<FDPoint>>&
FDWeightsCache::getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) const
{
    return d_pair_pt_map.at(patch.getPointer());
}

const std::set<FDPoint>&
FDWeightsCache::getRBFFDBasePoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) const
{
    return d_base_pt_set.at(patch.getPointer());
}

const std::vector<double>&
FDWeightsCache::getRBFFDWeights(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDPoint& pt) const
{
#if !defined(NDEBUG)
    if (!isBasePoint(patch, pt)) TBOX_ERROR("pt " << pt << " is not a base point on this patch");
#endif
    return d_pt_weight_map.at(patch.getPointer()).at(pt);
}

const std::vector<FDPoint>&
FDWeightsCache::getRBFFDPoints(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDPoint& pt) const
{
#if !defined(NDEBUG)
    if (!isBasePoint(patch, pt)) TBOX_ERROR("pt " << pt << " is not a base point on this patch");
#endif
    return d_pair_pt_map.at(patch.getPointer()).at(pt);
}

bool
FDWeightsCache::isBasePoint(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch, const FDPoint& pt) const
{
    return d_base_pt_set.at(patch.getPointer()).find(pt) != d_base_pt_set.at(patch.getPointer()).end();
}

void
FDWeightsCache::printPtMap(std::ostream& os, Pointer<PatchHierarchy<NDIM>> hierarchy) const
{
    const int ln = hierarchy->getFinestLevelNumber();
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
    unsigned int patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        os << "On patch number: " << patch_num << "\n";
        os << "There are " << d_base_pt_set.at(patch.getPointer()).size() << " key-value pairs present\n";
        for (const auto& pt : d_base_pt_set.at(patch.getPointer()))
        {
            os << "   On point " << pt << "\n";
            const std::vector<FDPoint>& fd_pts = d_pair_pt_map.at(patch.getPointer()).at(pt);
            const std::vector<double>& wgts = d_pt_weight_map.at(patch.getPointer()).at(pt);
            for (size_t i = 0; i < fd_pts.size(); ++i)
            {
                os << "     FD point: " << fd_pts[i] << "\n";
                os << "     weight:   " << wgts[i] << "\n";
            }
        }
        os << "\n";
    }
}
//////////////////////////////////////////////////////////////////////////////

} // namespace ADS

//////////////////////////////////////////////////////////////////////////////
