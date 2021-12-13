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
    Point shft;
    for (unsigned int d = 0; d < NDIM; ++d) shft(d) = 0.0;
    return formMonomials(pts, deg, 1.0, shft);
}

template <class Point>
IBTK::MatrixXd
formMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft)
{
    // Number of polynomials
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    // Shift and scale monomials
    for (size_t row = 0; row < pts.size(); ++row)
    {
        Point pt = pts[row];
        pt = (pt - shft) / ds;
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
laplacianMonomials(const std::vector<Point>& pts, int deg)
{
    Point shft;
    for (unsigned int d = 0; d < NDIM; ++d) shft(d) = 0.0;
    return laplacianMonomials(pts, deg, 1.0, shft);
}

template <class Point>
IBTK::MatrixXd
laplacianMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        Point pt = pts[row];
        pt = (pt - shft) / ds;
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) = i / ds * (i - 1) / ds * ADS::PolynomialBasis::pow(pt(0), i - 2) *
                                          ADS::PolynomialBasis::pow(pt(1), j) +
                                      j / ds * (j - 1) / ds * ADS::PolynomialBasis::pow(pt(0), i) *
                                          ADS::PolynomialBasis::pow(pt(1), j - 2);
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
                        j / ds * (j - 1) / ds * ADS::PolynomialBasis::pow(pt(0), j - 2) *
                            ADS::PolynomialBasis::pow(pt(1), k) * ADS::PolynomialBasis::pow(pt(2), i - k - j) +
                        k / ds * (k - 1) / ds * ADS::PolynomialBasis::pow(pt(0), j) *
                            ADS::PolynomialBasis::pow(pt(1), k - 2) * ADS::PolynomialBasis::pow(pt(2), i - k - j) +
                        (i - k - j) / ds * (i - k - j - 1) / ds * ADS::PolynomialBasis::pow(pt(0), j) *
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
dPdxMonomials(const std::vector<Point>& pts, int deg)
{
    Point shft;
    for (unsigned int d = 0; d < NDIM; ++d) shft(d) = 0.0;
    return dPdxMonomials(pts, deg, 1.0, shft);
}

template <class Point>
IBTK::MatrixXd
dPdxMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        Point pt = pts[row];
        pt = (pt - shft) / ds;
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) =
                        i / ds * ADS::PolynomialBasis::pow(pt(0), i - 1) * ADS::PolynomialBasis::pow(pt(1), j);
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
                    mat(row, col++) = j / ds * ADS::PolynomialBasis::pow(pt(0), j - 1) *
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
dPdyMonomials(const std::vector<Point>& pts, int deg)
{
    Point shft;
    for (unsigned int d = 0; d < NDIM; ++d) shft(d) = 0.0;
    return dPdyMonomials(pts, deg, 1.0, shft);
}

template <class Point>
IBTK::MatrixXd
dPdyMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        Point pt = pts[row];
        pt = (pt - shft) / ds;
        int col = 0;
#if (NDIM == 2)
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = 0; j <= deg; ++j)
            {
                if ((i + j) <= deg)
                {
                    mat(row, col++) =
                        j / ds * ADS::PolynomialBasis::pow(pt(0), i) * ADS::PolynomialBasis::pow(pt(1), j - 1);
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
                    mat(row, col++) = k / ds * ADS::PolynomialBasis::pow(pt(0), j) *
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
dPdzMonomials(const std::vector<Point>& pts, int deg)
{
    Point shft;
    for (unsigned int d = 0; d < NDIM; ++d) shft(d) = 0.0;
    return dPdzMonomials(pts, deg, 1.0, shft);
}

template <class Point>
IBTK::MatrixXd
dPdzMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft)
{
    size_t num_pts = pts.size();
    int num_poly = getNumberOfPolynomials(deg);
    IBTK::MatrixXd mat = IBTK::MatrixXd::Zero(num_pts, num_poly);
    for (size_t row = 0; row < pts.size(); ++row)
    {
        Point pt = pts[row];
        pt = (pt - shft) / ds;
        int col = 0;
        for (int i = 0; i <= deg; ++i)
        {
            for (int j = i; j >= 0; --j)
            {
                for (int k = i - j; k >= 0; --k)
                {
                    mat(row, col++) = (i - k - j) / ds * ADS::PolynomialBasis::pow(pt(0), j) *
                                      ADS::PolynomialBasis::pow(pt(1), k) *
                                      ADS::PolynomialBasis::pow(pt(2), i - j - k - 1);
                }
            }
        }
    }
    return mat;
}
#endif

inline int
getNumberOfPolynomials(int deg)
{
#if (NDIM == 2)
    return (deg + 1) * (deg + 2) / 2;
#endif
#if (NDIM == 3)
    return (deg + 1) * (deg + 2) * (deg + 3) / 6;
#endif
}

inline double
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
#endif
