/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ADS/app_namespaces.h"
#include "ADS/ls_functions.h"

#include <ibtk/IBTK_MPI.h>

#include <RefineAlgorithm.h>
#include <RefineOperator.h>
#include <RefineSchedule.h>

#include <queue>

namespace ADS
{
double
area_fraction(const double reg_area, const double phi_ll, const double phi_lu, const double phi_uu, const double phi_ul)
{
    // Find list of vertices
    std::vector<IBTK::Vector2d> vertices;
    // Start at bottom left
    if (phi_ll < 0.0) vertices.push_back({ 0.0, 0.0 });
    // Go clockwise towards top
    if (phi_ll * phi_lu < 0.0) vertices.push_back({ 0.0, -phi_ll / (phi_lu - phi_ll) });
    if (phi_lu < 0.0) vertices.push_back({ 0.0, 1.0 });
    if (phi_lu * phi_uu < 0.0) vertices.push_back({ -phi_lu / (phi_uu - phi_lu), 1.0 });
    if (phi_uu < 0.0) vertices.push_back({ 1.0, 1.0 });
    if (phi_uu * phi_ul < 0.0) vertices.push_back({ 1.0, 1.0 - phi_uu / (phi_ul - phi_uu) });
    if (phi_ul < 0.0) vertices.push_back({ 1.0, 0.0 });
    if (phi_ul * phi_ll < 0.0) vertices.push_back({ 1.0 - phi_ul / (phi_ll - phi_ul), 0.0 });

    // We have vertices, now use shoelace formula to find area
    double A = 0.0;
    for (unsigned int i = 0; i < vertices.size(); ++i)
    {
        const IBTK::Vector2d& vertex = vertices[i];
        const IBTK::Vector2d& vertex_n = vertices[(i + 1) % vertices.size()];
        A += vertex(0) * vertex_n(1) - vertex_n(0) * vertex(1);
    }
    return 0.5 * std::abs(A) * reg_area;
}

double
length_fraction(const double dx, const double phi_l, const double phi_u)
{
    double L = 0.0;
    if (phi_l < 0.0 && phi_u > 0.0)
    {
        L = phi_l / (phi_l - phi_u);
    }
    else if (phi_l > 0.0 && phi_u < 0.0)
    {
        L = phi_u / (phi_u - phi_l);
    }
    else if (phi_l < 0.0 && phi_u < 0.0)
    {
        L = 1.0;
    }
    return L * dx;
}

std::pair<double, double>
findVolumeAndArea(const VectorNd& xloc,
                  const double* const dx,
                  Pointer<NodeData<NDIM, double>> phi_data,
                  const CellIndex<NDIM>& idx)
{
    double volume = 0.0, area = 0.0;
    // Create the initial simplices.
    std::vector<Simplex> simplices;
    // Create a vector of pairs of points and phi values
    VectorNd Xloc;
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
        Xloc(0) = xloc(0) + static_cast<double>(x) * dx[0];
        for (int y = 0; y <= 1; ++y)
        {
            Xloc(1) = xloc(1) + static_cast<double>(y) * dx[1];
            NodeIndex<NDIM> n_idx(idx, IntVector<NDIM>(x, y));
            phi = (*phi_data)(n_idx);
            if (std::abs(phi) < s_eps) phi = phi < 0.0 ? -s_eps : s_eps;
            indices[x][y] = std::make_pair(Xloc, phi);
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
        Xloc(0) = xloc(0) + static_cast<double>(x) * dx[0];
        for (int y = 0; y <= 1; ++y)
        {
            Xloc(1) = xloc(1) + static_cast<double>(y) * dx[1];
            for (int z = 0; z <= 1; ++z)
            {
                Xloc(2) = xloc(2) + static_cast<double>(z) * dx[2];
                NodeIndex<NDIM> n_idx(idx, IntVector<NDIM>(x, y, z));
                phi = (*phi_data)(n_idx);
                indices[x][y][z] = std::make_pair(Xloc, phi);
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
    return std::make_pair(volume, area);
}

double
findVolume(const std::vector<Simplex>& simplices)
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
            final_simplices.push_back({ pt0, P01, P02, P03 });
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
        else if (n_phi.size() == 3)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[n_phi[2]].first;
            pt3 = simplex[p_phi[0]].first;
            phi0 = simplex[n_phi[0]].second;
            phi1 = simplex[n_phi[1]].second;
            phi2 = simplex[n_phi[2]].second;
            phi3 = simplex[p_phi[0]].second;
            // Simplex is between P0, P1, P2, P13
            VectorNd P13 = midpoint_value(pt1, phi1, pt3, phi3);
            final_simplices.push_back({ pt0, pt1, pt2, P13 });
            // and P0, P03, P2, P13
            VectorNd P03 = midpoint_value(pt0, phi0, pt3, phi3);
            final_simplices.push_back({ pt0, P03, pt2, P13 });
            // and P23, P03, P2, P13
            VectorNd P23 = midpoint_value(pt2, phi2, pt3, phi3);
            final_simplices.push_back({ P23, P03, pt2, P13 });
        }
        else if (n_phi.size() == 4)
        {
            pt0 = simplex[n_phi[0]].first;
            pt1 = simplex[n_phi[1]].first;
            pt2 = simplex[n_phi[2]].first;
            pt3 = simplex[n_phi[3]].first;
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
#if (NDIM == 2)
        VectorNd pt1 = simplex[0], pt2 = simplex[1], pt3 = simplex[2];
        double a = (pt1 - pt2).norm(), b = (pt2 - pt3).norm(), c = (pt1 - pt3).norm();
        double p = 0.5 * (a + b + c);
        volume += std::sqrt(p * (p - a) * (p - b) * (p - c));
#endif
#if (NDIM == 3)
        // Volume is given by 1/NDIM! * determinant of matrix
        Eigen::MatrixXd A(NDIM, NDIM);
        for (int d = 0; d < NDIM; ++d)
        {
            A.col(d) = simplex[d + 1] - simplex[0];
        }
        volume += 1.0 / 6.0 * std::abs(A.determinant());
#endif
    }
    return volume;
}

double
findArea(const std::vector<Simplex>& simplices)
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

void
flood_fill_for_LS(const int sgn_idx,
                  Pointer<NodeVariable<NDIM, double>> /*sgn_var*/,
                  const double eps,
                  Pointer<PatchLevel<NDIM>> level)
{
    TBOX_ASSERT(eps > 0.0);
    RefineAlgorithm<NDIM> ghost_fill_alg;
    ghost_fill_alg.registerRefine(sgn_idx, sgn_idx, sgn_idx, nullptr);
    Pointer<RefineSchedule<NDIM>> ghost_fill_sched = ghost_fill_alg.createSchedule(level);
    // Do a flood fill algorithm
    std::vector<int> patch_filled_vec(level->getNumberOfPatches());
    unsigned int patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& box = patch->getBox();
        Box<NDIM> n_box(box);
        n_box.growUpper(1);
        Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(sgn_idx);
        NodeData<NDIM, int> idx_touched(box, 1, phi_data->getGhostCellWidth());
        idx_touched.fillAll(0);
        std::queue<NodeIndex<NDIM>> idx_queue;
        bool found_pt = false;
        for (NodeIterator<NDIM> ni(box); ni; ni++)
        {
            const NodeIndex<NDIM>& idx = ni();
            if ((*phi_data)(idx) <= 0.0)
            {
                idx_queue.push(idx);
                found_pt = true;
            }
        }
        patch_filled_vec[patch_num] = found_pt ? 1 : 0;
        // We have our starting point. Now, loop through queue
        while (idx_queue.size() > 0)
        {
            const NodeIndex<NDIM>& idx = idx_queue.front();
            // If this point is uninitialized, it is interior
            if (idx_touched(idx) == 0 && ((*phi_data)(idx) == eps || (*phi_data)(idx) <= 0.0))
            {
                // Insert the point into touched list
                idx_touched(idx) = 1;
                if ((*phi_data)(idx) == eps) (*phi_data)(idx) = -eps;
                // Add neighboring points if they haven't been touched yet
                NodeIndex<NDIM> idx_s = idx + IntVector<NDIM>(0, -1);
                NodeIndex<NDIM> idx_n = idx + IntVector<NDIM>(0, 1);
                NodeIndex<NDIM> idx_e = idx + IntVector<NDIM>(1, 0);
                NodeIndex<NDIM> idx_w = idx + IntVector<NDIM>(-1, 0);
                if (n_box.contains(idx_s) && ((*phi_data)(idx_s) == eps || (*phi_data)(idx_s) < 0.0) &&
                    idx_touched(idx_s) == 0)
                    idx_queue.push(idx_s);
                if (n_box.contains(idx_n) && ((*phi_data)(idx_n) == eps || (*phi_data)(idx_n) < 0.0) &&
                    idx_touched(idx_n) == 0)
                    idx_queue.push(idx_n);
                if (n_box.contains(idx_e) && ((*phi_data)(idx_e) == eps || (*phi_data)(idx_e) < 0.0) &&
                    idx_touched(idx_e) == 0)
                    idx_queue.push(idx_e);
                if (n_box.contains(idx_w) && ((*phi_data)(idx_w) == eps || (*phi_data)(idx_w) < 0.0) &&
                    idx_touched(idx_w) == 0)
                    idx_queue.push(idx_w);
            }
            idx_queue.pop();
        }
    }

    // At this point if there's any box that hasn't been filled, then it's either entirely inside or outside.
    // We'll fill ghost cells, then check the ghost boxes. If there is any negative in the ghost box, then the entire
    // patch is inside
    int num_negative_found = 1;
    while (num_negative_found > 0)
    {
        num_negative_found = 0;
        ghost_fill_sched->fillData(0.0);
        patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            bool found_negative = false;
            if (patch_filled_vec[patch_num] == 1) continue;
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            // Loop through ghost cells
            Pointer<NodeData<NDIM, double>> phi_data = patch->getPatchData(sgn_idx);
            for (NodeIterator<NDIM> ni(phi_data->getGhostBox()); ni; ni++)
            {
                const NodeIndex<NDIM>& idx = ni();
                if (patch->getBox().contains(idx)) continue;
                if ((*phi_data)(idx) == -eps)
                {
                    found_negative = true;
                    break;
                }
            }
            if (found_negative)
            {
                phi_data->fillAll(-eps, phi_data->getGhostBox());
                num_negative_found++;
                patch_filled_vec[patch_num] = 1;
            }
        }
        num_negative_found = IBTK_MPI::sumReduction(num_negative_found);
    }
}

void
flood_fill_for_LS(const int sgn_idx,
                  Pointer<CellVariable<NDIM, double>> /*sgn_var*/,
                  const double eps,
                  Pointer<PatchLevel<NDIM>> level)
{
    TBOX_ASSERT(eps > 0.0);
    RefineAlgorithm<NDIM> ghost_fill_alg;
    ghost_fill_alg.registerRefine(sgn_idx, sgn_idx, sgn_idx, nullptr);
    Pointer<RefineSchedule<NDIM>> ghost_fill_sched = ghost_fill_alg.createSchedule(level);
    // Do a flood fill algorithm
    std::vector<int> patch_filled_vec(level->getNumberOfPatches());
    unsigned int patch_num = 0;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& box = patch->getBox();
        Pointer<CellData<NDIM, double>> phi_data = patch->getPatchData(sgn_idx);
        CellData<NDIM, int> idx_touched(box, 1, phi_data->getGhostCellWidth());
        idx_touched.fillAll(0);
        std::queue<CellIndex<NDIM>> idx_queue;
        bool found_pt = false;
        for (CellIterator<NDIM> ni(box); ni; ni++)
        {
            const CellIndex<NDIM>& idx = ni();
            if ((*phi_data)(idx) <= 0.0)
            {
                idx_queue.push(idx);
                found_pt = true;
            }
        }
        patch_filled_vec[patch_num] = found_pt ? 1 : 0;
        // We have our starting point. Now, loop through queue
        while (idx_queue.size() > 0)
        {
            const CellIndex<NDIM>& idx = idx_queue.front();
            // If this point is uninitialized, it is interior
            if (idx_touched(idx) == 0 && ((*phi_data)(idx) == eps || (*phi_data)(idx) <= 0.0))
            {
                // Insert the point into touched list
                idx_touched(idx) = 1;
                if ((*phi_data)(idx) == eps) (*phi_data)(idx) = -eps;
                // Add neighboring points if they haven't been touched yet
                CellIndex<NDIM> idx_s = idx + IntVector<NDIM>(0, -1);
                CellIndex<NDIM> idx_n = idx + IntVector<NDIM>(0, 1);
                CellIndex<NDIM> idx_e = idx + IntVector<NDIM>(1, 0);
                CellIndex<NDIM> idx_w = idx + IntVector<NDIM>(-1, 0);
                if (box.contains(idx_s) && ((*phi_data)(idx_s) == eps || (*phi_data)(idx_s) < 0.0) &&
                    idx_touched(idx_s) == 0)
                    idx_queue.push(idx_s);
                if (box.contains(idx_n) && ((*phi_data)(idx_n) == eps || (*phi_data)(idx_n) < 0.0) &&
                    idx_touched(idx_n) == 0)
                    idx_queue.push(idx_n);
                if (box.contains(idx_e) && ((*phi_data)(idx_e) == eps || (*phi_data)(idx_e) < 0.0) &&
                    idx_touched(idx_e) == 0)
                    idx_queue.push(idx_e);
                if (box.contains(idx_w) && ((*phi_data)(idx_w) == eps || (*phi_data)(idx_w) < 0.0) &&
                    idx_touched(idx_w) == 0)
                    idx_queue.push(idx_w);
            }
            idx_queue.pop();
        }
    }

    // At this point if there's any box that hasn't been filled, then it's either entirely inside or outside.
    // We'll fill ghost cells, then check the ghost boxes. If there is any negative in the ghost box, then the entire
    // patch is inside
    int num_negative_found = 1;
    while (num_negative_found > 0)
    {
        num_negative_found = 0;
        ghost_fill_sched->fillData(0.0);
        patch_num = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_num)
        {
            bool found_negative = false;
            if (patch_filled_vec[patch_num] == 1) continue;
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            // Loop through ghost cells
            Pointer<CellData<NDIM, double>> phi_data = patch->getPatchData(sgn_idx);
            for (CellIterator<NDIM> ni(phi_data->getGhostBox()); ni; ni++)
            {
                const CellIndex<NDIM>& idx = ni();
                if (patch->getBox().contains(idx)) continue;
                if ((*phi_data)(idx) == -eps)
                {
                    found_negative = true;
                    break;
                }
            }
            if (found_negative)
            {
                phi_data->fillAll(-eps, phi_data->getGhostBox());
                num_negative_found++;
                patch_filled_vec[patch_num] = 1;
            }
        }
        num_negative_found = IBTK_MPI::sumReduction(num_negative_found);
    }
}
} // namespace ADS
