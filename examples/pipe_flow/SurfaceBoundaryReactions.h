#ifndef included_SurfaceBoundaryReactions
#define included_SurfaceBoundaryReactions

#include "CCAD/LSCutCellBoundaryConditions.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/equation_systems.h"
#include <libmesh/mesh.h>

class SurfaceBoundaryReactions : public CCAD::LSCutCellBoundaryConditions
{
public:
    SurfaceBoundaryReactions(const std::string& object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             libMesh::Mesh* mesh,
                             IBTK::FEDataManager* fe_manager);

    ~SurfaceBoundaryReactions() = default;

    /*!
     * \brief Deleted default constructor.
     */
    SurfaceBoundaryReactions() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    SurfaceBoundaryReactions(const SurfaceBoundaryReactions& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    SurfaceBoundaryReactions& operator=(const SurfaceBoundaryReactions& that) = delete;

    void applyBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> Q_var,
                                int Q_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> R_var,
                                int R_idx,
                                SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                double time) override;

    void updateSurfaceConcentration(int fl_idx,
                                    double current_time,
                                    double new_time,
                                    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

private:
    void spreadToFluid(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);
    void interpolateToBoundary(int Q_idx,
                               const std::string& Q_sys_name,
                               SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                               double current_time);

    /*!
     * Find an intersection between the element elem and the side defined by the point r and the search direction and
     * magnitude q.
     */
    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    double reconstructMLS(const IBTK::VectorNd& x,
                          const SAMRAI::pdat::CellIndex<NDIM>& idx,
                          const SAMRAI::pdat::NodeData<NDIM, double>& ls_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& vol_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    double reconstructRBF(const IBTK::VectorNd& x,
                          const SAMRAI::pdat::CellIndex<NDIM>& idx,
                          const SAMRAI::pdat::NodeData<NDIM, double>& ls_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& vol_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    double d_D_coef = std::numeric_limits<double>::quiet_NaN();
    double d_k_on = std::numeric_limits<double>::quiet_NaN(), d_k_off = std::numeric_limits<double>::quiet_NaN();
    double d_cb_max = std::numeric_limits<double>::quiet_NaN();

    libMesh::Mesh* d_mesh = nullptr;
    IBTK::FEDataManager* d_fe_data_manager = nullptr;

    bool d_perturb_nodes = false;

    bool d_use_rbfs = true;

    static std::string s_fluid_sys_name, s_surface_sys_name;
};
#endif
