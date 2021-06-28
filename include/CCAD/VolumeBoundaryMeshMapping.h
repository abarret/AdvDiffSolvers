#ifndef included_CCAD_VolumeBoundaryMeshMapping
#define included_CCAD_VolumeBoundaryMeshMapping
#include "CCAD/CutCellMeshMapping.h"
#include "CCAD/FEMeshPartitioner.h"
#include "CCAD/ls_functions.h"
#include "CCAD/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace CCAD
{
/*!
 * VolumeBoundaryMeshMapping maintains a description of the Lagrangian mesh from the point of view of the background
 * mesh. We maintain a mapping from each cut cell index to a vector of element and element parent pairs.
 */
class VolumeBoundaryMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    VolumeBoundaryMeshMapping(std::string object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                              libMesh::MeshBase* vol_mesh,
                              IBTK::FEDataManager* vol_fe_data_manager,
                              const std::vector<std::set<libMesh::boundary_id_type>>& bdry_ids_vec,
                              bool register_for_restart = false,
                              const std::string& restart_read_dirname = "",
                              unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor.
     */
    VolumeBoundaryMeshMapping(std::string object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                              const std::vector<libMesh::MeshBase*>& vol_meshes,
                              const std::vector<IBTK::FEDataManager*>& vol_fe_data_managers,
                              const std::vector<std::set<libMesh::boundary_id_type>>& bdry_ids_vec,
                              const std::vector<unsigned int>& part_vec,
                              bool register_for_restart = false,
                              const std::string& restart_read_dirname = "",
                              unsigned int restart_restore_number = 0);

    /*!
     * \brief Default deconstructor.
     */
    ~VolumeBoundaryMeshMapping() = default;

    /*!
     * \brief Deleted default constructor.
     */
    VolumeBoundaryMeshMapping() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    VolumeBoundaryMeshMapping(const VolumeBoundaryMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    VolumeBoundaryMeshMapping& operator=(const VolumeBoundaryMeshMapping& that) = delete;

    inline std::shared_ptr<FEMeshPartitioner>& getMeshPartitioner(unsigned int part)
    {
        return d_bdry_mesh_partitioners[part];
    }

    inline const std::vector<std::shared_ptr<FEMeshPartitioner>>& getMeshPartitioners()
    {
        return d_bdry_mesh_partitioners;
    }

    inline std::vector<std::shared_ptr<FEMeshPartitioner>> getMeshPartitioners(std::set<unsigned int> mesh_nums)
    {
        std::vector<std::shared_ptr<FEMeshPartitioner>> mesh_partitioners;
        for (const auto& mesh_num : mesh_nums) mesh_partitioners.push_back(d_bdry_mesh_partitioners[mesh_num]);
        return mesh_partitioners;
    }

    void matchBoundaryToVolume(unsigned int part, std::string sys_name = "");

    /*!
     * \brief Initialize the equations systems. Note all systems should be registered with the Equation systems prior to
     * this call. This function also initialized the location of the boundary mesh, so the volume mesh should be
     * properly set up prior to calling this.
     */
    void initializeEquationSystems();

    void matchBoundaryToVolume(std::string sys_name = "");

    void writeFEDataToRestartFile(const std::string& restart_dump_dirname, unsigned int time_step_number);

private:
    void commonConstructor(const std::vector<std::set<libMesh::boundary_id_type>>& bdry_ids,
                           const std::vector<unsigned int>& parts,
                           SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    std::string d_object_name;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    std::vector<libMesh::MeshBase*> d_vol_meshes;
    std::vector<IBTK::FEDataManager*> d_vol_fe_data_managers;

    std::vector<std::shared_ptr<IBTK::FEData>> d_fe_data;
    std::vector<std::shared_ptr<FEMeshPartitioner>> d_bdry_mesh_partitioners;
    std::vector<std::unique_ptr<libMesh::BoundaryMesh>> d_bdry_meshes;
    std::vector<std::unique_ptr<libMesh::EquationSystems>> d_bdry_eq_sys_vec;
    std::string d_coords_sys_name = "COORDINATES_SYSTEM";
    std::string d_disp_sys_name = "DISPLACEMENT_SYSTEM";
    std::vector<unsigned int> d_vol_id_vec;

    // Restart data
    bool d_register_for_restart = false;
    std::string d_libmesh_restart_read_dir, d_libmesh_restart_file_extension = "xdr";
    unsigned int d_libmesh_restart_restore_number;
};

} // namespace CCAD
#endif
