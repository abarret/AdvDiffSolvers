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
IBTK::MatrixXd formMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft);

template <class Point>
IBTK::MatrixXd laplacianMonomials(const std::vector<Point>& pts, int deg);
template <class Point>
IBTK::MatrixXd laplacianMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft);

template <class Point>
IBTK::MatrixXd dPdxMonomials(const std::vector<Point>& pts, int deg);
template <class Point>
IBTK::MatrixXd dPdxMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft);

template <class Point>
IBTK::MatrixXd dPdyMonomials(const std::vector<Point>& pts, int deg);
template <class Point>
IBTK::MatrixXd dPdyMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft);

#if (NDIM == 3)
template <class Point>
IBTK::MatrixXd dPdzMonomials(const std::vector<Point>& pts, int deg);
template <class Point>
IBTK::MatrixXd dPdzMonomials(const std::vector<Point>& pts, int deg, double ds, const Point& shft);
#endif

int getNumberOfPolynomials(int deg);

double pow(const double q, const int i);

enum class RHSType
{
    INTERPOLATORY,
    LAPLACIAN,
    HELMHOLTZ,
    USER_DEFINED,
    UNKNOWN = -1
};

/*!
 * \brief Routine for converting strings to enums.
 */
template <typename T>
inline T
string_to_enum(const std::string& /*val*/)
{
    TBOX_ERROR("UNSUPPORTED ENUM TYPE\n");
    return -1;
}

/*!
 * \brief Routine for converting enums to strings.
 */
template <typename T>
inline std::string enum_to_string(T /*val*/)
{
    TBOX_ERROR("UNSUPPORTED ENUM TYPE\n");
    return "UNKNOWN";
}
template <>
inline RHSType
string_to_enum<RHSType>(const std::string& val)
{
    if (strcasecmp(val.c_str(), "INTERPOLATORY") == 0) return RHSType::INTERPOLATORY;
    if (strcasecmp(val.c_str(), "LAPLACIAN") == 0) return RHSType::LAPLACIAN;
    if (strcasecmp(val.c_str(), "HELMHOLTZ") == 0) return RHSType::HELMHOLTZ;
    if (strcasecmp(val.c_str(), "USER_DEFINED") == 0) return RHSType::USER_DEFINED;
    return RHSType::UNKNOWN;
}

template <>
inline std::string
enum_to_string<RHSType>(RHSType val)
{
    if (val == RHSType::INTERPOLATORY) return "INTERPOLATORY";
    if (val == RHSType::LAPLACIAN) return "LAPLACIAN";
    if (val == RHSType::HELMHOLTZ) return "HELMHOLTZ";
    if (val == RHSType::USER_DEFINED) return "USER_DEFINED";
    return "UNKNOWN";
}
} // namespace PolynomialBasis
} // namespace ADS

#include <ADS/private/PolynomialBasis_inc.h>
#endif
