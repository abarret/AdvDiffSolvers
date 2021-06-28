/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/ls_functions.h"

#include <ibamr/app_namespaces.h>

namespace CCAD
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
} // namespace CCAD
