#ifndef included_ADS_PolynomialBasis_inc
#define included_ADS_PolynomialBasis_inc

#include <ibtk/ibtk_utilities.h>

#include <ADS/PolynomialBasis.h>

#include <vector>
namespace ADS
{
namespace PolynomialBasis
{
template <class Point>
IBTK::MatrixXd
formMonomials(const std::vector<Point>& pts, int deg)
{
    // Number of polynomials
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        const Point& pt = pts[row];
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) = ADS::PolynomialBasis::pow(pt(0), i) * ADS::PolynomialBasis::pow(pt(1), j);
                }
            }
        }
#endif
#if (NDIM == 3)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) = ADS::PolynomialBasis::pow(pt(0), j) * ADS::PolynomialBasis::pow(pt(1), k) *
                                      ADS::PolynomialBasis::pow(pt(2), i - k - j);
                }
            }
        }
#endif
    }
    return mat;
}

template <class Point>
IBTK::MatrixXd
laplacianMonomials(std::vector<Point>& pts, int deg)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        const Point& pt = pts[row];
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) =
                        i * (i - 1) * ADS::PolynomialBasis::pow(pt(0), i - 2) * ADS::PolynomialBasis::pow(pt(1), j) +
                        j * (j - 1) * ADS::PolynomialBasis::pow(pt(0), i) * ADS::PolynomialBasis::pow(pt(1), j - 2);
                }
            }
        }
#endif
#if (NDIM == 3)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) =
                        j * (j - 1) * ADS::PolynomialBasis::pow(pt(0), j - 2) * ADS::PolynomialBasis::pow(pt(1), k) *
                            ADS::PolynomialBasis::pow(pt(2), i - k - j) +
                        k * (k - 1) * ADS::PolynomialBasis::pow(pt(0), j) * ADS::PolynomialBasis::pow(pt(1), k - 2) *
                            ADS::PolynomialBasis::pow(pt(2), i - k - j) +
                        (i - k - j) * (i - k - j - 1) * ADS::PolynomialBasis::pow(pt(0), j) *
                            ADS::PolynomialBasis::pow(pt(1), k) * ADS::PolynomialBasis::pow(pt(2), i - k - j - 2);
                }
            }
        }
#endif
    }
    return mat;
}

template <class Point>
IBTK::MatrixXd
dPdxMonomials(std::vector<Point>& pts, int deg)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        const Point& pt = pts[row];
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) = i * ADS::PolynomialBasis::pow(pt(0), i - 1) * ADS::PolynomialBasis::pow(pt(1), j);
                }
            }
        }
#endif
#if (NDIM == 3)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) = j * ADS::PolynomialBasis::pow(pt(0), j - 1) *
                                      ADS::PolynomialBasis::pow(pt(1), k) * ADS::PolynomialBasis::pow(pt(2), i - k - j);
                }
            }
        }
#endif
    }
    return mat;
}

template <class Point>
IBTK::MatrixXd
dPdyMonomials(std::vector<Point>& pts, int deg)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        const Point& pt = pts[row];
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) = j * ADS::PolynomialBasis::pow(pt(0), i) * ADS::PolynomialBasis::pow(pt(1), j - 1);
                }
            }
        }
#endif
#if (NDIM == 3)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) = k * ADS::PolynomialBasis::pow(pt(0), j) *
                                      ADS::PolynomialBasis::pow(pt(1), k - 1) *
                                      ADS::PolynomialBasis::pow(pt(2), i - k - j);
                }
            }
        }
#endif
    }
    return mat;
}

#if (NDIM == 3)
template <class Point>
IBTK::MatrixXd
dPdzMonomials(std::vector<Point>& pts, int deg)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        const Point& pt = pts[row];
        int col = 0;
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) = (i - k - j) * ADS::PolynomialBasis::pow(pt(0), j) *
                                      ADS::PolynomialBasis::pow(pt(1), k) *
                                      ADS::PolynomialBasis::pow(pt(2), i - j - k - 1);
                }
            }
        }
    }
    return mat;
}
#endif

int
getNumberOfPolynomials(int deg)
{
#if (NDIM == 2)
    return (deg + 1) * (deg + 2) / 2;
#endif
#if (NDIM == 3)
    return (deg + 1) * (deg + 2) * (deg + 3) / 6;
#endif
}
} // namespace PolynomialBasis
} // namespace ADS
#endif
