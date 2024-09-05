#ifndef included_ADS_CutCellVolumeMeshMapping
#define included_ADS_CutCellVolumeMeshMapping
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEToHierarchyMapping.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
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
                             FEToHierarchyMapping* fe_hierarchy_mappings);

    /*!
     * \brief Constructor.
     */
    CutCellVolumeMeshMapping(std::string object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings);

    /*!
     * \brief Constructor.
     */
    CutCellVolumeMeshMapping(std::string object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             IBTK::FEDataManager* fe_data_manager);

    /*!
     * \brief Constructor.
     */
    CutCellVolumeMeshMapping(std::string object_name,
                             SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                             const std::vector<IBTK::FEDataManager*>& fe_data_managers);

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

    inline const FEToHierarchyMapping* getMeshMapping(unsigned int part = 0)
    {
        return d_fe_hierarchy_mappings.at(part);
    }

    inline const std::vector<FEToHierarchyMapping*>& getMeshMappings() const
    {
        return d_fe_hierarchy_mappings;
    }

private:
    void commonConstructor(const SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& input_db);

    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    std::vector<FEToHierarchyMapping*> d_fe_hierarchy_mappings;
    std::vector<IBTK::FEDataManager*> d_fe_data_managers;
    std::vector<MappingFcn> d_mapping_fcns;
};

} // namespace ADS
#endif
