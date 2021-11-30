#ifndef included_ADS_SBSurfaceFluidCouplingManager
#define included_ADS_SBSurfaceFluidCouplingManager

#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
#include "ADS/RBFReconstructCache.h"

#include "ibtk/FEDataManager.h"

#include "CellData.h"
#include "NodeData.h"

#include "libmesh/equation_systems.h"
#include <libmesh/boundary_mesh.h>
#include <libmesh/mesh.h>

namespace ADS
{
class SBSurfaceFluidCouplingManager : public SAMRAI::tbox::DescribedClass
{
public:
    /*!
     * \brief Constructor.
     */
    SBSurfaceFluidCouplingManager(std::string name,
                                  const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db,
                                  const std::vector<std::shared_ptr<FEMeshPartitioner>>& fe_mesh_partitioners);

    SBSurfaceFluidCouplingManager(std::string name,
                                  const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db,
                                  const std::shared_ptr<FEMeshPartitioner>& fe_mesh_partitioner);

    /*!
     * \brief Deconstructor. Cleans up any allocated data objects.
     */
    ~SBSurfaceFluidCouplingManager();

    /*!
     * \brief Deleted default constructor.
     */
    SBSurfaceFluidCouplingManager() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    SBSurfaceFluidCouplingManager(const SBSurfaceFluidCouplingManager& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    SBSurfaceFluidCouplingManager& operator=(const SBSurfaceFluidCouplingManager& that) = delete;

    /*!
     * \brief Register ReconstructCache
     */
    void registerReconstructCache(SAMRAI::tbox::Pointer<ReconstructCache> reconstruct_cache)
    {
        d_rbf_reconstruct = reconstruct_cache;
    }

    /*!
     * \brief Register surface concentrations that will be tracked by the data manager.
     * @{
     */
    void registerSurfaceConcentration(std::string surface_name, unsigned int part = 0);
    void registerSurfaceConcentration(const std::vector<std::string>& surface_names, unsigned int part = 0);
    /*!
     * @}
     */

    /*!
     * \brief Register fluid phase concentrations. The data manager will keep a reference to the variable and provide an
     * interface to calculate concentrations along the surface.
     *
     * \see interpolateToBoundary
     * @{
     */
    void registerFluidConcentration(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var,
                                    unsigned int part = 0);
    void registerFluidConcentration(
        const std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>& fl_vars,
        unsigned int part = 0);
    /*!
     * @}
     */

    /*!
     * \brief Register dependence between the given surface concentration and fluid variable.
     */
    void registerFluidSurfaceDependence(const std::string& surface_name,
                                        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var,
                                        unsigned int part = 0);

    /*!
     * \brief Register dependence between two surface concentrations.
     */
    void registerSurfaceSurfaceDependence(const std::string& part1_name,
                                          const std::string& part2_name,
                                          unsigned int part = 0);

    /*!
     * \brief Register a reaction function and context for a given surface quantity.
     */
    void registerSurfaceReactionFunction(const std::string& surface_name,
                                         ReactionFcn fcn,
                                         void* ctx = nullptr,
                                         unsigned int part = 0);

    /*!
     * \brief Register boundary conditions for a given fluid quantity.
     */
    void registerFluidBoundaryCondition(const SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>& fl_var,
                                        ReactionFcn a_fcn,
                                        ReactionFcn g_fcn,
                                        void* ctx = nullptr,
                                        unsigned int part = 0);

    /*!
     * \brief Create the systems and insert them into the appropriate EquationSystems object. Note that this must be
     * done prior to any EquationSystems::init() call.
     */
    void initializeFEData();

    /*!
     * \brief Set the level set and volume data used by this manager to integrate fluid concentrations to the boundary.
     */
    void setLSData(int ls_idx, int vol_idx, SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    /*!
     * \brief Interpolate the data stored in the SAMRAI variable and context pair to the surface.
     * @{
     */
    const std::string& interpolateToBoundary(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var,
                                             const int idx,
                                             double time,
                                             unsigned int part);
    inline const std::string&
    interpolateToBoundary(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> fl_var,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                          double time,
                          unsigned int part)
    {
        auto var_db = SAMRAI::hier::VariableDatabase<NDIM>::getDatabase();
        const int idx = var_db->mapVariableAndContextToIndex(fl_var, ctx);
        return interpolateToBoundary(fl_var, idx, time, part);
    }
    inline const std::string& interpolateToBoundary(const std::string& fl_name,
                                                    SAMRAI::tbox::Pointer<SAMRAI::hier::VariableContext> ctx,
                                                    double time,
                                                    unsigned int part)
    {
        auto fl_it = std::find(d_fl_names_vec[part].begin(), d_fl_names_vec[part].end(), fl_name);
        TBOX_ASSERT(fl_it != d_fl_names_vec[part].end());
        return interpolateToBoundary(
            d_fl_vars_vec[part][std::distance(d_fl_names_vec[part].begin(), fl_it)], ctx, time, part);
    }
    inline const std::string&
    interpolateToBoundary(const std::string& fl_name, const int idx, double time, unsigned int part = 0)
    {
        auto fl_it = std::find(d_fl_names_vec[part].begin(), d_fl_names_vec[part].end(), fl_name);
        TBOX_ASSERT(fl_it != d_fl_names_vec[part].end());
        return interpolateToBoundary(
            d_fl_vars_vec[part][std::distance(d_fl_names_vec[part].begin(), fl_it)], idx, time, part);
    }
    /*!
     * @}
     */

    /*!
     * \brief Update the Jacobian of the mapping. Returns the string of the system name.
     * @{
     */
    const std::string& updateJacobian(unsigned int part);
    const std::string& updateJacobian()
    {
        for (unsigned int part = 0; part < d_fe_mesh_partitioners.size(); ++part) updateJacobian(part);
        return d_J_sys_name;
    }
    /*!
     * @}
     */

    /*!
     * \brief Return the string containing the Jacobian of the mapping.
     */
    inline const std::string& getJacobianName()
    {
        return d_J_sys_name;
    }

    /*!
     * \brief Returns the FEDataManager object used by this manager.
     */
    inline std::shared_ptr<FEMeshPartitioner>& getFEMeshPartitioner(unsigned int part = 0)
    {
        return d_fe_mesh_partitioners[part];
    }

    /*!
     * \brief Return the surface concentration variable names stored by this manager.
     */
    inline const std::vector<std::string>& getSFNames(unsigned int part = 0)
    {
        return d_sf_names_vec[part];
    }

    /*!
     * \brief Return the fluid concentration variable names stored by this manager.
     */
    inline const std::vector<std::string>& getFLNames(unsigned int part = 0)
    {
        return d_fl_names_vec[part];
    }

    /*!
     * \brief Return the SAMRAI variable represented by the fluid variable name.
     */
    inline const SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>&
    getFLVariable(const std::string& fl_name, unsigned int part = 0)
    {
        auto fl_it = std::find(d_fl_names_vec[part].begin(), d_fl_names_vec[part].end(), fl_name);
        TBOX_ASSERT(fl_it != d_fl_names_vec[part].end());
        return d_fl_vars_vec[part][std::distance(d_fl_names_vec[part].begin(), fl_it)];
    }

    /*!
     * \brief Return the SAMRAI variable represented by the fluid variable name.
     */
    inline const std::string& getFLName(const SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>& fl_var,
                                        unsigned int part = 0)
    {
        auto fl_it = std::find(d_fl_vars_vec[part].begin(), d_fl_vars_vec[part].end(), fl_var);
        TBOX_ASSERT(fl_it != d_fl_vars_vec[part].end());
        return d_fl_names_vec[part][std::distance(d_fl_vars_vec[part].begin(), fl_it)];
    }

    /*!
     * \brief Get the pair of coupling lists for the given surface quantity. The first vector contains the surface
     * system names and the second vector contains the fluid system names.
     */
    inline void getSFCouplingLists(const std::string& sf_name,
                                   std::vector<std::string>& sf_names,
                                   std::vector<std::string>& fl_names,
                                   unsigned int part = 0)
    {
        sf_names = d_sf_sf_map_vec[part][sf_name];
        fl_names = d_sf_fl_map_vec[part][sf_name];
    }

    /*!
     * \brief Get the pair of coupling lists for the given fluid quantity. The first vector contains the surface system
     * names and the second vector contains the fluid system names.
     */
    inline void getFLCouplingLists(const std::string& fl_name,
                                   std::vector<std::string>& sf_names,
                                   std::vector<std::string>& fl_names,
                                   unsigned int part = 0)
    {
        sf_names = d_fl_sf_map_vec[part][fl_name];
        fl_names = d_fl_fl_map_vec[part][fl_name];
    }

    /*!
     * \brief Return the reaction function and function context pair for the given surface quantity.
     */
    inline const ReactionFcnCtx& getSFReactionFcnCtxPair(const std::string& sf_name, unsigned int part = 0)
    {
        return d_sf_reaction_fcn_ctx_map_vec[part][sf_name];
    }

    /*!
     * \brief Return the boundary condition functions for the given fluid concentration. Note that the first element is
     * the a function and the second element is the g function.
     *
     * @{
     */
    inline const BdryConds&
    getFLBdryConditionFcns(const SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>& fl_var,
                           unsigned int part = 0)
    {
        return getFLBdryConditionFcns(fl_var->getName(), part);
    }
    inline const BdryConds& getFLBdryConditionFcns(const std::string& fl_name, unsigned int part = 0)
    {
        return d_fl_a_g_fcn_map_vec[part][fl_name];
    }
    /*!
     * @}
     */

    libMesh::BoundaryMesh* getMesh(unsigned int part = 0)
    {
        return static_cast<libMesh::BoundaryMesh*>(&d_fe_mesh_partitioners[part]->getEquationSystems()->get_mesh());
    }

    unsigned int getNumParts()
    {
        return d_fe_mesh_partitioners.size();
    }

    using InitialConditionFcn = std::function<double(const IBTK::VectorNd& X, const libMesh::Node* const node)>;

    inline void registerInitialConditions(const std::string& sf_name, InitialConditionFcn fcn, unsigned int part = 0)
    {
        TBOX_ASSERT(std::find(d_sf_names_vec[part].begin(), d_sf_names_vec[part].end(), sf_name) !=
                    d_sf_names_vec[part].end());
        d_sf_init_fcn_map_vec[part][sf_name] = fcn;
    }

    void fillInitialConditions();

protected:
    std::string d_object_name;
    std::vector<std::shared_ptr<FEMeshPartitioner>> d_fe_mesh_partitioners;

    std::vector<std::vector<std::string>> d_sf_names_vec;
    std::vector<std::vector<std::string>> d_fl_names_vec;
    std::vector<std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>>>> d_fl_vars_vec;

    std::vector<std::map<std::string, std::vector<std::string>>> d_sf_fl_map_vec;
    std::vector<std::map<std::string, std::vector<std::string>>> d_sf_sf_map_vec;
    std::vector<std::map<std::string, std::vector<std::string>>> d_fl_fl_map_vec;
    std::vector<std::map<std::string, std::vector<std::string>>> d_fl_sf_map_vec;
    std::vector<std::map<std::string, ReactionFcnCtx>> d_sf_reaction_fcn_ctx_map_vec;
    std::vector<std::map<std::string, BdryConds>> d_fl_a_g_fcn_map_vec;
    std::vector<std::map<std::string, InitialConditionFcn>> d_sf_init_fcn_map_vec;

    std::string d_J_sys_name;

    SAMRAI::tbox::Pointer<ReconstructCache> d_rbf_reconstruct;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    int d_stencil_size = -1;
    bool d_update_weights = false;

    bool d_fe_eqs_initialized = false;

    int d_vol_idx = IBTK::invalid_index, d_ls_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_scr_var;
    int d_scr_idx = IBTK::invalid_index;

private:
    void commonConstructor(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);
};

} // namespace ADS
#endif
