#ifndef included_CutCellMeshMapping
#define included_CutCellMeshMapping
#include "ibtk/FEDataManager.h"

#include "LS/ls_functions.h"
#include "LS/ls_utilities.h"

#include "libmesh/mesh.h"

namespace LS
{
/*!
 * CutCellMeshMapping maintains a description of the Lagrangian mesh from the point of view of the background mesh. We
 * maintain a mapping from each cut cell index to a vector of element and element parent pairs.
 */
class CutCellMeshMapping
{
public:
    /*!
     * \brief Constructor.
     */
    CutCellMeshMapping(std::string object_name,
                       SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                       libMesh::Mesh* mesh,
                       IBTK::FEDataManager* fe_data_manager);

    /*!
     * \brief Default deconstructor.
     */
    ~CutCellMeshMapping();

    /*!
     * \brief Deleted default constructor.
     */
    CutCellMeshMapping() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    CutCellMeshMapping(const CutCellMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    CutCellMeshMapping& operator=(const CutCellMeshMapping& that) = delete;

    void initializeObjectState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    void deinitializeObjectState();

    void generateCutCellMappings();

    inline const std::vector<std::map<LS::IndexList, std::vector<CutCellElems>>>& getIdxCutCellElemsMap(const int ln)
    {
        return d_idx_cut_cell_elems_map_vec[ln];
    }

    using MappingFcn = std::function<void(libMesh::Node*, libMesh::Elem*, libMesh::Point& x_cur)>;

    inline void registerMappingFcn(MappingFcn fcn)
    {
        d_mapping_fcn = fcn;
    }

private:
    bool findIntersection(libMesh::Point& p, libMesh::Elem* elem, libMesh::Point r, libMesh::VectorValue<double> q);

    std::string d_object_name;
    bool d_is_initialized = false;
    bool d_perturb_nodes = false;

    libMesh::Mesh* d_mesh = nullptr;
    IBTK::FEDataManager* d_fe_data_manager = nullptr;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    std::vector<std::vector<std::map<LS::IndexList, std::vector<CutCellElems>>>> d_idx_cut_cell_elems_map_vec;

    MappingFcn d_mapping_fcn = nullptr;
};

} // namespace LS
#endif
