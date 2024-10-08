#ifndef included_ADS_GeneralBoundaryMeshMapping
#define included_ADS_GeneralBoundaryMeshMapping
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FESystemManager.h"
#include "ADS/ads_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * GeneralBoundaryMeshMapping is a class that generalizes the notion of a boundary mesh. It maintains an EquationSystems
 * object on the mesh, handles restarts of Lagrangian data, and maintains a FESystemManager for the object.
 * Implementations for this object should define how the object moves. A default implementation of no motion is
 * included.
 */
class GeneralBoundaryMeshMapping
{
public:
    /*!
     * \brief Constructor that takes in a vector of meshes.
     */
    GeneralBoundaryMeshMapping(std::string object_name,
                               SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                               const std::vector<libMesh::MeshBase*>& base_meshes,
                               const std::string& restart_read_dirname = "",
                               unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor that takes in a mesh.
     */
    GeneralBoundaryMeshMapping(std::string object_name,
                               SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                               libMesh::MeshBase* base_mesh,
                               const std::string& restart_read_dirname = "",
                               unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor that leaves object in undefined state.
     */
    GeneralBoundaryMeshMapping(std::string object_name);

    /*!
     * \brief Default constructor.
     */
    GeneralBoundaryMeshMapping() = delete;

    /*!
     * \brief Default deconstructor.
     */
    virtual ~GeneralBoundaryMeshMapping() = default;

    /*!
     * \brief Deleted copy constructor.
     */
    GeneralBoundaryMeshMapping(const GeneralBoundaryMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    GeneralBoundaryMeshMapping& operator=(const GeneralBoundaryMeshMapping& that) = delete;

    /*!
     * \name Get the FESystemManager.
     */
    //\{

    virtual inline FESystemManager& getSystemManager(unsigned int part = 0)
    {
        return *d_bdry_fe_sys_managers[part];
    }

    virtual inline std::vector<FESystemManager*> getSystemManagers()
    {
        return unique_ptr_vec_to_raw_ptr_vec(d_bdry_fe_sys_managers);
    }

    virtual inline std::vector<FESystemManager*> getSystemManagers(std::set<unsigned int> mesh_nums)
    {
        std::vector<FESystemManager*> ptrs;
        for (const auto& mesh_num : mesh_nums) ptrs.push_back(d_bdry_fe_sys_managers[mesh_num].get());
        return ptrs;
    }

    //\}

    /*!
     * Update the location of the boundary mesh. An optional argument is provided if the location of the structure is
     * needed at the end of the timestep. By default, this function loops over parts and calls the part specific
     * function.
     */
    virtual void updateBoundaryLocation(double time, bool end_of_timestep = false);
    /*!
     * Update the location of the boundary mesh for a specific part. An optional argument is provided if the location of
     * the structure is needed at the end of the timestep. By default this function does nothing.
     */
    virtual void updateBoundaryLocation(double time, unsigned int part, bool end_of_timestep = false);

    /*!
     * Set the initial location of the boundary mesh for all parts. By default, this loops through all parts and calls
     * initializeBoundaryLocation(time, part).
     *
     * This is useful in cases where updateBoundaryLocation() requires information that is not initialized until after
     * initializeFEData() is called.
     */
    virtual void initializeBoundaryLocation(double time);

    /*!
     * Set the initial location of the boundary mesh for the specified part. By default, this calls
     * updateBoundaryLocation().
     *
     * This is useful in cases where updateBoundaryLocation() requires information that is not initialized until after
     * initializeFEData() is called.
     */
    virtual void initializeBoundaryLocation(double time, unsigned int part);

    /*!
     * \brief Create the EquationSystems object for the boundary mesh. If the boundary mesh is a nullptr, this routine
     * calls buildBoundaryMesh.
     */
    virtual void initializeEquationSystems();

    /*!
     * \brief Initialize the equations systems data. Note all systems should be registered with the EquationSystems
     * prior to this call. This function also initialized the location of the boundary mesh.
     */
    virtual void initializeFEData();

    /*!
     * \brief Write data to a restart file.
     */
    virtual void writeFEDataToRestartFile(const std::string& restart_dump_dirname, unsigned int time_step_number);

    /*!
     * \brief Construct the boundary mesh. By default this copies whatever was given in the constructor.
     */
    virtual void buildBoundaryMesh();

    inline const std::unique_ptr<libMesh::BoundaryMesh>& getBoundaryMesh(unsigned int part = 0) const
    {
        return d_bdry_meshes[part];
    }

    inline const std::vector<std::unique_ptr<libMesh::BoundaryMesh>>& getBoundaryMeshes() const
    {
        return d_bdry_meshes;
    }

    inline int getNumParts() const
    {
        return d_bdry_meshes.size();
    }

protected:
    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> d_input_db;

    std::vector<std::unique_ptr<FESystemManager>> d_bdry_fe_sys_managers;
    // TODO: Figure out ownership requirements for d_bdry_meshes. Meshes could be given to us, or we could create them.
    // For now, we just use a raw pointer, along with a flag to determine if we need to clean it up.
    std::vector<libMesh::MeshBase*> d_base_meshes;
    std::vector<std::unique_ptr<libMesh::BoundaryMesh>> d_bdry_meshes;
    std::vector<std::unique_ptr<libMesh::EquationSystems>> d_bdry_eq_sys_vec;
    std::string d_coords_sys_name = "COORDINATES_SYSTEM";
    std::string d_disp_sys_name = "DISPLACEMENT_SYSTEM";

    // Restart data
    std::string d_libmesh_restart_file_extension = "xdr";
    std::string d_restart_read_dirname = "";
    unsigned int d_restart_restore_num = 0;

private:
};

} // namespace ADS
#endif
