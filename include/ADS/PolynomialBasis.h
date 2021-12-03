#ifndef included_ADS_PolynomialBasis
#define included_ADS_PolynomialBasis

#include <ibtk/ibtk_utilities.h>

#include <vector>
namespace ADS
{
namespace PolynomialBasis
{
template <class Point>
IBTK::MatrixXd formMonomials(const std::vector<Point>& pts, int deg);

template <class Point>
IBTK::MatrixXd laplacianMonomials(std::vector<Point>& pts, int deg);

template <class Point>
IBTK::MatrixXd dPdxMonomials(std::vector<Point>& pts, int deg);

template <class Point>
IBTK::MatrixXd dPdyMonomials(std::vector<Point>& pts, int deg);

#if (NDIM == 3)
template <class Point>
IBTK::MatrixXd dPdzMonomials(std::vector<Point>& pts, int deg);
#endif

int getNumberOfPolynomials(int deg);

double
pow(const double q, const int i)
{
    if (i == 0) return 1.0;
    if (q == 0)
        return 0.0;
    else
        return std::pow(q, i);
}
} // namespace PolynomialBasis
} // namespace ADS

#include <ADS/private/PolynomialBasis_inc.h>
#endif
