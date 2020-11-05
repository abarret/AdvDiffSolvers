#ifndef included_SBIntegrator
#define included_SBIntegrator

#include "ibtk/FEDataManager.h"

#include "LS/RBFReconstructCache.h"

#include "CellData.h"
#include "NodeData.h"

#include "libmesh/equation_systems.h"
#include <libmesh/mesh.h>

namespace LS
{
class SBIntegrator
{
public:
    using ReactionFcn =
        std::function<double(double, const std::vector<double>&, const std::vector<double>&, double, void*)>;

    SBIntegrator(std::string object_name,
                 SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                 libMesh::Mesh* mesh,
                 IBTK::FEDataManager* fe_data_manager);

    /*!
     * \brief Deleted default constructor.
     */
    SBIntegrator() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    SBIntegrator(const SBIntegrator& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    SBIntegrator& operator=(const SBIntegrator& that) = delete;

    void registerSurfaceConcentration(std::string surface_name);
    void registerSurfaceConcentration(const std::vector<std::string>& surface_names);

    void registerFluidConcentration(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var);
    void registerFluidConcentration(
        const std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& fl_vars);

    void registerFluidSurfaceDependence(const std::string& surface_name,
                                        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var);
    void registerSurfaceSurfaceDependence(const std::string& part1_name, const std::string& part2_name);

    void registerSurfaceReactionFunction(const std::string& surface_name, ReactionFcn fcn, void* ctx = nullptr);

    void initializeFEEquationSystems();

    IBTK::FEDataManager* getFEDataManager();
    const std::vector<std::string>& getSFNames();
    const std::vector<std::string>& getFLNames();

    void setLSData(int ls_idx, int vol_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    void
    integrateHierarchy(SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx, double current_time, double new_time);

    void beginTimestepping(double current_time, double new_time);
    void endTimestepping(double current_time, double new_time);

    void interpolateToBoundary(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var,
                               SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                               double time);

private:
    /*!
     * Find an intersection between the element elem and the side defined by the point r and the search direction and
     * magnitude q.
     */
    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    double reconstructRBF(const IBTK::VectorNd& x,
                          const SAMRAI::pdat::CellIndex<NDIM>& idx,
                          const SAMRAI::pdat::NodeData<NDIM, double>& ls_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& vol_vals,
                          const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    void cacheRBFData();

    std::string d_object_name;

    libMesh::Mesh* d_mesh;
    IBTK::FEDataManager* d_fe_data_manager;

    std::vector<std::string> d_sf_names;
    std::vector<std::string> d_fl_names;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>> d_fl_vars;
    std::map<std::string, std::vector<std::string>> d_sf_fl_map;
    std::map<std::string, std::vector<std::string>> d_sf_sf_map;
    std::map<std::string, ReactionFcn> d_sf_reaction_fcn_map;
    std::map<std::string, void*> d_sf_ctx_map;

    RBFReconstructCache d_rbf_reconstruct;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_stencil_size = -1;
    bool d_update_weights = false;

    bool d_fe_eqs_initialized = false;

    bool d_perturb_nodes = false;

    int d_vol_idx = IBTK::invalid_index, d_ls_idx = IBTK::invalid_index;
};

} // namespace LS
#endif
