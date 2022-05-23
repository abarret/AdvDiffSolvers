#ifndef included_ADS_VolumeBoundaryMeshMapping
#define included_ADS_VolumeBoundaryMeshMapping
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * VolumeBoundaryMeshMapping is a concrete implementation of GeneralBoundaryMeshMapping. It is used to efficiently match
 * an extracted boundary mesh with that of the corresponding solid mesh. It creates and uses a FEMeshPartitioner to
 * maintain a mapping between background Eulerian patches to boundary elements.
 */
class VolumeBoundaryMeshMapping : public GeneralBoundaryMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    VolumeBoundaryMeshMapping(std::string object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              libMesh::MeshBase* vol_mesh,
                              IBTK::FEDataManager* vol_fe_data_manager,
                              std::vector<std::set<libMesh::boundary_id_type>> bdry_ids_vec,
                              const std::string& restart_read_dirname = "",
                              unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor.
     */
    VolumeBoundaryMeshMapping(std::string object_name,
                              SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                              const std::vector<libMesh::MeshBase*>& vol_meshes,
                              const std::vector<IBTK::FEDataManager*>& vol_fe_data_managers,
                              std::vector<std::set<libMesh::boundary_id_type>> bdry_ids_vec,
                              std::vector<unsigned int> part_vec,
                              const std::string& restart_read_dirname = "",
                              unsigned int restart_restore_number = 0);

    /*!
     * \brief Default deconstructor.
     */
    virtual ~VolumeBoundaryMeshMapping() = default;

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

    virtual void updateBoundaryLocation(double time, unsigned int part, bool end_of_timestep = false) override;

    void initializeEquationSystems() override;

private:
    std::vector<libMesh::MeshBase*> d_vol_meshes;
    std::vector<IBTK::FEDataManager*> d_vol_fe_data_managers;
    std::vector<std::set<libMesh::boundary_id_type>> d_bdry_ids;
    std::vector<unsigned int> d_parts;

    std::vector<unsigned int> d_vol_id_vec;
};

} // namespace ADS
#endif
