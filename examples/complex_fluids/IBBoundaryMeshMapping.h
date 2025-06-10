#ifndef included_clotting_IBBoundaryMeshMapping
#define included_clotting_IBBoundaryMeshMapping
#include "ADS/GeneralBoundaryMeshMapping.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/LDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

/*!
 * IBBoundaryMeshMapping is a concrete implementation of GeneralBoundaryMeshMapping. It is used to efficiently match
 * an extracted boundary mesh with that of the corresponding solid mesh. It creates and uses a FEMeshPartitioner to
 * maintain a mapping between background Eulerian patches to boundary elements.
 */
class IBBoundaryMeshMapping : public ADS::GeneralBoundaryMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    IBBoundaryMeshMapping(std::string object_name,
                          SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                          libMesh::MeshBase* mesh,
                          IBTK::LDataManager* data_manager,
                          int level_num,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          const std::string& restart_read_dirname = "",
                          unsigned int restart_restore_number = 0);

    /*!
     * \brief Constructor.
     */
    IBBoundaryMeshMapping(std::string object_name,
                          SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                          const std::vector<libMesh::MeshBase*>& meshes,
                          IBTK::LDataManager* data_manager,
                          int level_num,
                          std::vector<int> struct_nums,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          const std::string& restart_read_dirname = "",
                          unsigned int restart_restore_number = 0);

    /*!
     * \brief Default deconstructor.
     */
    virtual ~IBBoundaryMeshMapping() = default;

    using ADS::GeneralBoundaryMeshMapping::updateBoundaryLocation;
    virtual void updateBoundaryLocation(double time, unsigned int part, bool end_of_timestep = false) override;

    using ADS::GeneralBoundaryMeshMapping::initializeBoundaryLocation;
    virtual void initializeBoundaryLocation(double time, unsigned int part) override;

private:
    IBTK::LDataManager* d_base_data_manager;
    std::vector<int> d_part_nums;
    int d_level_num = IBTK::invalid_level_number;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
};

#endif
