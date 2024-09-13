#ifndef included_ADS_CutCellMeshMapping
#define included_ADS_CutCellMeshMapping
#include "ADS/FEToHierarchyMapping.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * CutCellMeshMapping maintains a description of the Lagrangian mesh from the point of view of the background
 * mesh. We maintain a mapping from each cut cell index to a vector of element and element parent pairs.
 */
class CutCellMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    CutCellMeshMapping(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Constructor.
     */
    CutCellMeshMapping(std::string object_name, bool perturb_nodes);

    /*!
     * \brief Default deconstructor.
     */
    ~CutCellMeshMapping() = default;

    /*!
     * Generate and cache the mappings between indices and partial elements. The mesh in each FEDataManager or
     * FEToHierarchyMapping should correspond to a surface mesh. Note that these objects should be set and
     * reinitElementMappings() should already be called.
     */
    void generateCutCellMappings(const std::vector<IBTK::FEDataManager*>& fe_data_managers);

    void generateCutCellMappings(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings);

    /*!
     * Clear the cached cut cells. This is called automatically in each call to generateCutCellMappings()
     */
    void clearCache();

    /*!
     * Return the vector of indices and cut cell maps for the given patch level number. Note the index in the vector is
     * the local patch number.
     */
    inline const std::vector<std::map<IndexList, std::vector<CutCellElems>>>& getIdxCutCellElemsMap(const int ln)
    {
        return d_idx_cut_cell_elems_map_vec[ln];
    }

private:
    /*!
     * Set up the data structures and reserve space.
     */
    void initializeObject(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    void generateCutCellMappings(libMesh::System& X_sys,
                                 IBTK::FEDataManager::SystemDofMapCache* X_dof_map_cache,
                                 const std::vector<std::vector<libMesh::Elem*>>& active_patch_elem_map,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                 int level_num,
                                 int part);

    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    std::string d_object_name;
    bool d_perturb_nodes = false;
    bool d_elems_cached = false;

    std::vector<std::vector<std::map<IndexList, std::vector<CutCellElems>>>> d_idx_cut_cell_elems_map_vec;
};

} // namespace ADS
#endif
