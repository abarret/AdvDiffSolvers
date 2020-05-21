#include "ibtk/DebuggingUtilities.h"

#include "LSFindCellVolume.h"

#include "utility_functions.h"

namespace IBAMR
{
const double LSFindCellVolume::s_eps = 1.0e-12;

LSFindCellVolume::LSFindCellVolume(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy)
    : d_object_name(std::move(object_name)), d_hierarchy(hierarchy)
{
    // intentionally blank
    return;
} // Constructor

void
LSFindCellVolume::updateVolumeAndArea(int vol_idx,
                                      Pointer<CellVariable<NDIM, double>> /*vol_var*/,
                                      int area_idx,
                                      Pointer<CellVariable<NDIM, double>> /*area_var*/,
                                      int phi_idx,
                                      Pointer<NodeVariable<NDIM, double>> phi_var,
                                      bool extended_box)
{
    int coarsest_ln = 0, finest_ln = d_hierarchy->getFinestLevelNumber();

    Pointer<NodeVariable<NDIM, double>> phi_n_var = phi_var;
    TBOX_ASSERT(phi_n_var);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());

            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(phi_idx);
            Pointer<CellData<NDIM, double>> area_data;
            if (area_idx != IBTK::invalid_index) area_data = patch->getPatchData(area_idx);
            Pointer<CellData<NDIM, double>> vol_data;
            if (vol_idx != IBTK::invalid_index) vol_data = patch->getPatchData(vol_idx);

            const Box<NDIM>& box = extended_box ? phi_data->getGhostBox() : patch->getBox();
            const hier::Index<NDIM>& patch_lower = box.lower();
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();
            const double* const xlow = pgeom->getXLower();
            const double cell_volume = dx[0] * dx[1];
            for (CellIterator<NDIM> ci(box); ci; ci++)
            {
                const CellIndex<NDIM>& idx = ci();
                double volume, area;
                findVolumeAndArea(xlow, dx, patch_lower, phi_data, idx, volume, area);
                if (area_idx != IBTK::invalid_index) (*area_data)(idx) = area;
                if (vol_idx != IBTK::invalid_index) (*vol_data)(idx) = volume / cell_volume;
            }
        }
    }
}

void
LSFindCellVolume::findVolumeAndArea(const double* const xlow,
                                    const double* const dx,
                                    const hier::Index<NDIM>& patch_lower,
                                    Pointer<NodeData<NDIM, double>> phi_data,
                                    const CellIndex<NDIM>& idx,
                                    double& volume,
                                    double& area)
{
    // Create the initial simplices.
    std::vector<Simplex> simplices;
    // Create a vector of pairs of points and phi values
    VectorNd X;
    double phi;
    int num_p = 0, num_n = 0;
#if (NDIM == 2)
    boost::multi_array<std::pair<VectorNd, double>, NDIM> indices(boost::extents[2][2]);
#endif
#if (NDIM == 3)
    boost::multi_array<std::pair<VectorNd, double>, NDIM> indices(boost::extents[2][2][2]);
#endif
#if (NDIM == 2)
    for (int x = 0; x <= 1; ++x)
    {
        X(0) = xlow[0] + dx[0] * (idx(0) - patch_lower(0) + x);
        for (int y = 0; y <= 1; ++y)
        {
            X(1) = xlow[1] + dx[1] * (idx(1) - patch_lower(1) + y);
            NodeIndex<NDIM> n_idx(idx, IntVector<NDIM>(x, y));
            phi = (*phi_data)(n_idx);
            if (std::abs(phi) < s_eps) phi = phi < 0.0 ? -s_eps : s_eps;
            indices[x][y] = std::make_pair(X, phi);
            if (phi > 0)
            {
                // Found a positive phi
                num_p++;
            }
            else
            {
                // Found a negative phi
                num_n++;
            }
        }
    }
#endif
#if (NDIM == 3)
    for (int x = 0; x <= 1; ++x)
    {
        X(0) = xlow[0] + dx[0] * (idx(0) - patch_lower(0) + x);
        for (int y = 0; y <= 1; ++y)
        {
            X(1) = xlow[1] + dx[1] * (idx(1) - patch_lower(1) + y);
            for (int z = 0; z <= 1; ++z)
            {
                X(2) = xlow[2] + dx[2] * (idx(2) - patch_lower(2) + z);
                NodeIndex<NDIM> n_idx(idx, IntVector<NDIM>(x, y, z));
                phi = (*phi_data)(n_idx);
                indices[x][y][z] = std::make_pair(X, phi);
                if (phi > 0)
                {
                    num_p++;
                }
                else
                {
                    num_n++;
                }
            }
        }
    }
#endif
#if (NDIM == 2)
    // Divide grid cell in half to form two simplices.
    simplices.push_back({ indices[0][0], indices[1][0], indices[1][1] });
    simplices.push_back({ indices[0][0], indices[0][1], indices[1][1] });
#endif
#if (NDIM == 3)
    // Divide grid cell to form simplices.
    simplices.push_back({ indices[0][0][0], indices[1][0][0], indices[0][1][0], indices[0][0][1] });
    simplices.push_back({ indices[1][1][0], indices[1][0][0], indices[0][1][0], indices[1][1][1] });
    simplices.push_back({ indices[1][0][1], indices[1][0][0], indices[1][1][1], indices[0][0][1] });
    simplices.push_back({ indices[0][1][1], indices[1][1][1], indices[0][1][0], indices[0][0][1] });
    simplices.push_back({ indices[1][1][1], indices[1][0][0], indices[0][1][0], indices[0][0][1] });
#endif
    if (num_n == NDIM * NDIM)
    {
        // Grid cell is completely contained within physical boundary.
        volume = dx[0] * dx[1];
        area = 0.0;
    }
    else if (num_p == NDIM * NDIM)
    {
        // Grid cell is completely outside of physical boundary.
        volume = 0.0;
        area = 0.0;
    }
    else
    {
        volume = findVolume(simplices);
        area = findArea(simplices);
    }
}

double
LSFindCellVolume::findVolume(const std::vector<Simplex>& simplices)
{
    // Loop over simplices
    std::vector<std::array<VectorNd, NDIM + 1>> final_simplices;
    for (const auto& simplex : simplices)
    {
        std::vector<int> n_phi, p_phi;
        for (size_t k = 0; k < simplex.size(); ++k)
        {
            const std::pair<VectorNd, double>& pt_pair = simplex[k];
            double phi = pt_pair.second;
            if (phi < 0)
            {
                n_phi.push_back(k);
            }
            else
            {
                p_phi.push_back(k);
            }
        }
        // Determine new simplices
#if (NDIM == 2)
        VectorNd pt0, pt1, pt2;
        double phi0, phi1, phi2;
        if (n_phi.size() == 1)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[p_phi[0]].first;
            pt2 = simplex[p_phi[1]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[p_phi[0]].second;
            phi2 = simplex[p_phi[1]].second;
            // Simplex is between P0, P01, P02
            VectorNd P01 = midpoint_value(pt0, phi0, pt1, phi1);
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            final_simplices.push_back({ pt0, P01, P02 });
        }
        else if (n_phi.size() == 2)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[p_phi[0]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[n_phi[1]].second;
            phi2 = simplex[p_phi[0]].second;
            // Simplex is between P0, P1, P02
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            final_simplices.push_back({ pt0, pt1, P02 });
            // and P1, P12, P02
            VectorNd P12 = midpoint_value(pt1, phi1, pt2, phi2);
            final_simplices.push_back({ pt1, P12, P02 });
        }
        else if (n_phi.size() == 3)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[n_phi[2]].first;
            final_simplices.push_back({ pt0, pt1, pt2 });
        }
        else if (n_phi.size() == 0)
        {
            continue;
        }
        else
        {
            TBOX_ERROR("This statement should not be reached!");
        }
#endif
#if (NDIM == 3)
        VectorNd pt0, pt1, pt2, pt3;
        double phi0, phi1, phi2, phi3;
        if (n_phi.size() == 1)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[p_phi[0]].first;
            pt2 = simplex[p_phi[1]].first;
            pt3 = simplex[p_phi[2]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[p_phi[0]].second;
            phi2 = simplex[p_phi[1]].second;
            phi3 = simplex[p_phi[2]].second;
            // Simplex is between P0, P01, P02, P03
            VectorNd P01 = midpoint_value(pt0, phi0, pt1, phi1);
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            VectorNd P03 = midpoint_value(pt0, phi0, pt3, phi3);
            final_simplices.push_back({ P0, P01, P02, P03 });
        }
        else if (n_phi.size() == 2)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[p_phi[0]].first;
            pt3 = simplex[p_phi[1]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[n_phi[1]].second;
            phi2 = simplex[p_phi[0]].second;
            phi3 = simplex[p_phi[1]].second;
            // Simplices are between P0, P1, P02, P13
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            VectorNd P13 = midpoint_value(pt1, phi1, pt3, phi3);
            final_simplices.push_back({ pt0, pt1, P02, P13 });
            // and P12, P1, P02, P13
            VectorNd P12 = midpoint_value(pt1, phi1, pt2, phi2);
            final_simplices.push_back({ P12, pt1, P02, P13 });
            // and P0, P03, P02, P13
            VectorNd P03 = midpoint_value(pt0, phi0, pt3, phi3);
            final_simplices.push_back({ pt0, P03, P02, P13 });
        }
        else if (n_phi.size() == 4)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[p_phi[0]].first;
            pt3 = simplex[p_phi[1]].first;
            final_simplices.push_back({ pt0, pt1, pt2, pt3 });
        }
        else if (n_phi.size() == 0)
        {
            continue;
        }
        else
        {
            TBOX_ERROR("This statement should not be reached!");
        }
#endif
    }
    // Loop over simplices and compute volume
    double volume = 0.0;
    for (const auto& simplex : final_simplices)
    {
        VectorNd pt1 = simplex[0], pt2 = simplex[1], pt3 = simplex[2];
        double a = (pt1 - pt2).norm(), b = (pt2 - pt3).norm(), c = (pt1 - pt3).norm();
        double p = 0.5 * (a + b + c);
        volume += std::sqrt(p * (p - a) * (p - b) * (p - c));
        //         // Volume is given by 1/4 determinant of matrix
        //         Eigen::MatrixXd A(NDIM,NDIM);
        //         for (int d = 0; d < NDIM; ++d)
        //         {
        //             A.col(d) = simplex[d+1] - simplex[0];
        //         }
        //         volume += 1.0/static_cast<double>(NDIM)*std::abs(A.determinant());
    }
    return volume;
}

double
LSFindCellVolume::findArea(const std::vector<Simplex>& simplices)
{
    // We need a lower dimensional simplex here. We're computing areas
    std::vector<std::array<VectorNd, NDIM>> final_simplices;
    for (const auto& simplex : simplices)
    {
        // Loop through simplices.
        // Determine boundary points on simplex to form polytope.
        std::vector<int> n_phi, p_phi;
        for (unsigned long k = 0; k < simplex.size(); ++k)
        {
            const std::pair<VectorNd, double>& pt_pair = simplex[k];
            double phi = pt_pair.second;
            if (phi < 0)
            {
                n_phi.push_back(k);
            }
            else
            {
                p_phi.push_back(k);
            }
        }
// Determine new simplices
#if (NDIM == 2)
        VectorNd pt0, pt1, pt2;
        double phi0, phi1, phi2;
        if (n_phi.size() == 1)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[p_phi[0]].first;
            pt2 = simplex[p_phi[1]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[p_phi[0]].second;
            phi2 = simplex[p_phi[1]].second;
        }
        else if (n_phi.size() == 2)
        {
            // Points are between n_phi an
            pt0 = simplex[p_phi[0]].first;
            pt1 = simplex[n_phi[0]].first;
            pt2 = simplex[n_phi[1]].first;
            phi0 = simplex[p_phi[0]].second;
            phi1 = simplex[n_phi[0]].second;
            phi2 = simplex[n_phi[1]].second;
        }
        else if (n_phi.size() == 0 || n_phi.size() == 3)
        {
            // If phi is all same sign then curve doesn't pass through simplex.
            continue;
        }
        else
        {
            TBOX_ERROR("This statement should not be reached!");
        }
        // Simplex is between P01 and P02
        VectorNd P01 = midpoint_value(pt0, phi0, pt1, phi1);
        VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
        final_simplices.push_back({ P01, P02 });
#endif
#if (NDIM == 3)
        VectorNd pt0, pt1, pt2, pt3;
        double phi0, phi1, phi2, phi3;
        if (n_phi.size() == 1 || n_phi.size() == 3)
        {
            if (n_phi.size() == 1)
            {
                pt0 = simplex[n_phi[0]].first;
                pt1 = simplex[p_phi[0]].first;
                pt2 = simplex[p_phi[1]].first;
                pt3 = simplex[p_phi[2]].first;
                phi0 = simplex[n_phi[0]].second;
                phi1 = simplex[p_phi[0]].second;
                phi2 = simplex[p_phi[1]].second;
                phi3 = simplex[p_phi[2]].second;
            }
            else
            {
                pt0 = simplex[p_phi[0]].first;
                pt1 = simplex[n_phi[0]].first;
                pt2 = simplex[n_phi[1]].first;
                pt3 = simplex[n_phi[2]].first;
                phi0 = simplex[p_phi[0]].second;
                phi1 = simplex[n_phi[0]].second;
                phi2 = simplex[n_phi[1]].second;
                phi3 = simplex[n_phi[2]].second;
            }
            // Simplex is between P01, P02, P03
            VectorNd P01 = midpoint_value(pt0, phi0, pt1, phi1);
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            VectorNd P03 = midpoint_value(pt0, phi0, pt3, phi3);
            final_simplices.push_back({ P01, P02, P03 });
        }
        else if (n_phi.size() == 2)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[p_phi[0]].first;
            pt3 = simplex[p_phi[1]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[n_phi[1]].second;
            phi2 = simplex[p_phi[0]].second;
            phi3 = simplex[p_phi[1]].second;
            // Simplices are between P02, P03, P13
            VectorNd P02 = midpoint_value(pt0, phi0, pt2, phi2);
            VectorNd P03 = midpoint_value(pt0, phi0, pt3, phi3);
            VectorNd P13 = midpoint_value(pt1, phi1, pt3, phi3);
            final_simplices.push_back({ P02, P03, P13 });
            // and P02, P12, P13
            VectorNd P12 = midpoint_value(pt1, phi1, pt2, phi2);
            final_simplices.push_back({ P02, P12, P13 });
        }
        else if (n_phi.size() == 0 || n_phi.size() == 4)
        {
            continue;
        }
        else
        {
            TBOX_ERROR("This statement should not be reached!");
        }
#endif
    }
    // Loop through final_simplices and calculate area
    double area = 0.0;
    for (const auto& simplex : final_simplices)
    {
#if (NDIM == 2)
        // A simplex is two points here, so the area is distance between them
        VectorNd pt1 = simplex[0], pt2 = simplex[1];
        area += (pt1 - pt2).norm();
#endif
#if (NDIM == 3)
        // A simplex is a triangle, area is sqrt(p*(p-a)*(p-b)*(p-c)) where p is half of perimeter
        VectorNd pt1 = simplex[0], pt2 = simplex[1], pt3 = simplex[2];
        double a = (pt1 - pt2).norm(), b = (pt2 - pt3).norm(), c = (pt1 - pt3).norm();
        double p = 0.5 * (a + b + c);
        area += std::sqrt(p * (p - a) * (p - b) * (p - c));
#endif
    }
    return area;
}

} // namespace IBAMR
