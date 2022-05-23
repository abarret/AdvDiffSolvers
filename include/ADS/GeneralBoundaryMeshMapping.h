#ifndef included_ADS_GeneralBoundaryMeshMapping
#define included_ADS_GeneralBoundaryMeshMapping
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * GeneralBoundaryMeshMapping is a class that generalizes the notion of a boundary mesh. It maintains an EquationSystems
 * object on the mesh, handles restarts of Lagrangian data, and maintains a FEMeshPartitioner for the object.
 * Implementations for this object should define how the object moves. A default implementation of no motion is
 * included.
 */
class GeneralBoundaryMeshMapping
{
public:
    /*!
     * \brief Constructor that takes in a vector of boundary meshes. Note that GeneralBoundaryMeshMapping assumes
     * ownership of the meshes.
     */
    GeneralBoundaryMeshMapping(std::string object_name,
                               SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                               const std::vector<libMesh::MeshBase*>& bdry_meshes,
                               const std::string& restart_read_dirname = "",
                               unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor that takes in a boundary mesh. Note that GeneralBoundaryMeshMapping assumes ownership of the
     * mesh.
     */
    GeneralBoundaryMeshMapping(std::string object_name,
                               SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                               libMesh::MeshBase* bdry_mesh,
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
    virtual ~GeneralBoundaryMeshMapping();

    /*!
     * \brief Deleted copy constructor.
     */
    GeneralBoundaryMeshMapping(const GeneralBoundaryMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    GeneralBoundaryMeshMapping& operator=(const GeneralBoundaryMeshMapping& that) = delete;

    /*!
     * \name Get the FEMeshPartitioners.
     */
    //\{

    virtual inline std::shared_ptr<FEMeshPartitioner>& getMeshPartitioner(unsigned int part = 0)
    {
        return d_bdry_mesh_partitioners[part];
    }

    virtual inline const std::vector<std::shared_ptr<FEMeshPartitioner>>& getMeshPartitioners()
    {
        return d_bdry_mesh_partitioners;
    }

    virtual inline std::vector<std::shared_ptr<FEMeshPartitioner>> getMeshPartitioners(std::set<unsigned int> mesh_nums)
    {
        std::vector<std::shared_ptr<FEMeshPartitioner>> mesh_partitioners;
        for (const auto& mesh_num : mesh_nums) mesh_partitioners.push_back(d_bdry_mesh_partitioners[mesh_num]);
        return mesh_partitioners;
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
     * \brief Initialize the equations systems. Note all systems should be registered with the Equation systems prior to
     * this call. This function also initialized the location of the boundary mesh.
     */
    virtual void initializeEquationSystems();

    virtual void initializeFEData();

    /*!
     * \brief Write data to a restart file.
     */
    virtual void writeFEDataToRestartFile(const std::string& restart_dump_dirname, unsigned int time_step_number);

protected:
    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> d_input_db;

    std::vector<std::shared_ptr<IBTK::FEData>> d_fe_data;
    std::vector<std::shared_ptr<FEMeshPartitioner>> d_bdry_mesh_partitioners;
    // TODO: Figure out ownership requirements for d_bdry_meshes. Meshes could be given to us, or we could create them.
    // For now, we just use a raw pointer, along with a flag to determine if we need to clean it up.
    std::vector<libMesh::MeshBase*> d_bdry_meshes;
    std::vector<int> d_own_bdry_mesh;
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
