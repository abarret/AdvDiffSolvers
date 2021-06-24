#ifndef included_CutCellVolumeMeshMapping
#define included_CutCellVolumeMeshMapping
#include "ibtk/FEDataManager.h"

#include "LS/CutCellMeshMapping.h"
#include "LS/FEMeshPartitioner.h"
#include "LS/ls_functions.h"
#include "LS/ls_utilities.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace LS
{
/*!
 * CutCellVolumeMeshMapping maintains a description of the Lagrangian mesh from the point of view of the background
 * mesh. We maintain a mapping from each cut cell index to a vector of element and element parent pairs.
 */
class CutCellVolumeMeshMapping : public CutCellMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    CutCellVolumeMeshMapping(std::string object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             const std::shared_ptr<FEMeshPartitioner>& fe_mesh_partitioner);

    /*!
     * \brief Constructor.
     */
    CutCellVolumeMeshMapping(std::string object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             const std::vector<std::shared_ptr<FEMeshPartitioner>>& fe_mesh_paritioners);

    /*!
     * \brief Default deconstructor.
     */
    virtual ~CutCellVolumeMeshMapping();

    /*!
     * \brief Deleted default constructor.
     */
    CutCellVolumeMeshMapping() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    CutCellVolumeMeshMapping(const CutCellVolumeMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    CutCellVolumeMeshMapping& operator=(const CutCellVolumeMeshMapping& that) = delete;

    void generateCutCellMappings() override;

    inline void registerMappingFcn(MappingFcn fcn, unsigned int part = 0)
    {
        d_mapping_fcns[part] = fcn;
    }

    inline std::shared_ptr<FEMeshPartitioner>& getMeshPartitioner(unsigned int part = 0)
    {
        return d_bdry_mesh_partitioners[part];
    }

private:
    void commonConstructor(const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db);

    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    std::vector<std::shared_ptr<FEMeshPartitioner>> d_bdry_mesh_partitioners;

    std::vector<MappingFcn> d_mapping_fcns;

    std::vector<std::vector<libMesh::Elem*>> d_active_patch_elem_map;
};

} // namespace LS
#endif
