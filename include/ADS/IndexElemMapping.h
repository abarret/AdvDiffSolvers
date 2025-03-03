#ifndef included_ADS_IndexElemMapping
#define included_ADS_IndexElemMapping
#include "ADS/FEToHierarchyMapping.h"
#include "ADS/ls_functions.h"
#include "ADS/ls_utilities.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/mesh.h"

namespace ADS
{
/*!
 * IndexElemMapping maintains a description of the Lagrangian mesh from the point of view of the background
 * mesh. We maintain a mapping from each cut cell index to a vector of element and element parent pairs.
 */
class IndexElemMapping
{
public:
	struct ElemData
	{
	public:
		ElemData(std::vector<libMesh::Point> elem_pts, libMesh::Elem* elem, unsigned int part)
		: elem_pts(std::move(elem_pts)), elem(elem), part(part)
	{}
		std::vector<libMesh::Point> elem_pts;
		libMesh::Elem* elem;
		unsigned int part;
	};
    /*!
     * \brief Constructor.
     */
    IndexElemMapping(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Constructor.
     */
    IndexElemMapping(std::string object_name, bool perturb_nodes);

    /*!
     * \brief Default deconstructor.
     */
    ~IndexElemMapping() = default;

    /*!
     * Generate and cache the mappings between indices and partial elements on all levels of the patch hierarchy. Note
     * that all managers are assumed to have the same number of levels in the patch hierarchy. The mesh in each
     * FEDataManager or FEToHierarchyMapping should correspond to a surface mesh. Note that these objects should be set
     * and reinitElementMappings() should already be called.
     */
    ///{
    void generateCellElemMappingOnHierarchy(const std::vector<IBTK::FEDataManager*>& fe_data_managers);
    void generateCellElemMappingOnHierarchy(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings);
    ///}

    /*!
     * Generate and cache the mappings between indices and partial elements on a specified level. The mesh in each
     * FEDataManager or FEToHierarchyMapping should correspond to a surface mesh. Note that these objects should be set
     * and reinitElementMappings() should already be called.
     *
     * If level_num is an invalid level number, then the finest level number in the patch hierarchy is used.
     */
    ///{
    void generateCellElemMapping(const std::vector<IBTK::FEDataManager*>& fe_data_managers,
                                 int level_num = IBTK::invalid_level_number);

    void generateCellElemMapping(const std::vector<FEToHierarchyMapping*>& fe_hierarchy_mappings,
                                 int level_num = IBTK::invalid_level_number);
    ///}

    /*!
     * Clear the cached cut cells. This is called automatically in each call to generateCutCellMappings()
     */
    void clearCache(int ln = IBTK::invalid_level_number);

    /*!
     * Return the vector of indices and cut cell maps for the given patch level number. Note the index in the vector is
     * the local patch number.
     */
    inline const std::vector<std::vector<std::pair<SAMRAI::pdat::CellIndex<NDIM>, std::vector<ElemData>>>>& getIdxElemMap(const int ln)
    {
        return d_idx_cell_elem_vec_vec[ln];
    }

private:
    /*!
     * Set up the data structures and reserve space on a specified level. If the level number is invalid, sets up data
     * structures on the finest level.
     */
    void initializeObjectOnLevel(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy, int ln);

    void generateCellElemMapping(libMesh::System& X_sys,
                                 IBTK::FEDataManager::SystemDofMapCache* X_dof_map_cache,
                                 const std::vector<std::vector<libMesh::Elem*>>& active_patch_elem_map,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                 int level_num,
                                 int part);

    std::string d_object_name;
    bool d_perturb_nodes = false;
    std::vector<bool> d_elems_cached_per_level;

    std::vector<std::vector<std::vector<std::pair<SAMRAI::pdat::CellIndex<NDIM>, std::vector<ElemData>>>>> d_idx_cell_elem_vec_vec;
};

} // namespace ADS
#endif
