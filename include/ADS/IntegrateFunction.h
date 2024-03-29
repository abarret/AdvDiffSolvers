#ifndef included_ADS_IntegrateFunction
#define included_ADS_IntegrateFunction

#include "ibtk/ibtk_enums.h"
#include "ibtk/ibtk_utilities.h"

#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellIndex.h"
#include "NodeData.h"
#include "PatchHierarchy.h"

namespace ADS
{
class IntegrateFunction
{
public:
    IntegrateFunction& operator=(const IntegrateFunction& that) = delete;
    IntegrateFunction(const IntegrateFunction& from) = delete;
    static IntegrateFunction* getIntegrator();

    static void freeIntegrators();

    using fcn_type = std::function<double(IBTK::VectorNd, double)>; // double (*)(IBTK::VectorXd x, double t);

    void integrateFcnOnPatchHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                      int ls_idx,
                                      int Q_idx,
                                      fcn_type fcn,
                                      double t);

    void integrateFcnOnPatchLevel(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level,
                                  int ls_idx,
                                  int Q_idx,
                                  fcn_type fcn,
                                  double t);

    void integrateFcnOnPatch(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                             int ls_idx,
                             int Q_idx,
                             fcn_type fcn,
                             double t,
                             SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level = nullptr);

private:
    IntegrateFunction() = default;
    ~IntegrateFunction() = default;

    using Simplex = std::array<std::pair<IBTK::VectorNd, double>, NDIM + 1>;
    using LDSimplex = std::array<std::pair<IBTK::VectorNd, double>, NDIM>;
    using PolytopePt = std::tuple<IBTK::VectorNd, int, int>;

    double integrateOverIndex(const double* const dx,
                              const IBTK::VectorNd& XLow,
                              const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                              const SAMRAI::pdat::CellIndex<NDIM>& idx,
                              fcn_type fcn,
                              double t);

    double integrate(const std::vector<Simplex>& simplices, fcn_type fcn, double t);

    double integrateOverSimplex(const std::array<IBTK::VectorNd, NDIM + 1>& X_pts, fcn_type fcn, double t);

#if (NDIM == 2)
    inline IBTK::VectorNd referenceToPhysical(const IBTK::VectorNd& xi, const std::array<IBTK::VectorNd, 3>& X_pts)
    {
        IBTK::VectorNd X;
        X = X_pts[0] * (1.0 - xi(0) - xi(1)) + X_pts[1] * xi(0) + X_pts[2] * xi(1);
        return X;
    }
#endif
#if (NDIM == 3)
    inline IBTK::VectorNd referenceToPhysical(const IBTK::VectorNd& xi, const std::array<IBTK::VectorNd, 4>& X_pts)
    {
        IBTK::VectorNd X;
        IBTK::MatrixNd M;
        M.block(0, 0, 3, 1) = X_pts[1] - X_pts[0];
        M.block(0, 1, 3, 1) = X_pts[2] - X_pts[0];
        M.block(0, 2, 3, 1) = X_pts[3] - X_pts[0];
        return M * xi + X_pts[0];
    }
#endif

    static IntegrateFunction* s_integrate_function;
    static unsigned char s_shutdown_priority;

    /*
     * Gaussian integral points
     */
#if (NDIM == 2)
    static const std::array<double, 9> s_weights;
    static const std::array<IBTK::VectorNd, 9> s_quad_pts;
#endif
#if (NDIM == 3)
    static const std::array<double, 27> s_weights;
    static const std::array<IBTK::VectorNd, 27> s_quad_pts;
#endif

    static const double s_eps;
}; // class IntegrateFunction

} // namespace ADS
#endif
