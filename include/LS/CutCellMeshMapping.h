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
class CutCellMeshMapping : public SAMRAI::tbox::DescribedClass
{
public:
    /*!
     * \brief Constructor.
     */
    CutCellMeshMapping(std::string object_name, SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Default deconstructor.
     */
    virtual ~CutCellMeshMapping();

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

    virtual void initializeObjectState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy);

    virtual void deinitializeObjectState();

    virtual void generateCutCellMappings() = 0;

    inline const std::vector<std::map<LS::IndexList, std::vector<CutCellElems>>>& getIdxCutCellElemsMap(const int ln)
    {
        return d_idx_cut_cell_elems_map_vec[ln];
    }

protected:
    std::string d_object_name;
    bool d_is_initialized = false;
    bool d_perturb_nodes = false;

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    std::vector<std::vector<std::map<LS::IndexList, std::vector<CutCellElems>>>> d_idx_cut_cell_elems_map_vec;
};

} // namespace LS
#endif
