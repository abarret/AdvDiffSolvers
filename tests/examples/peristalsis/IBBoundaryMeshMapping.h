#ifndef included_moveLS_IBBoundaryMeshMapping
#define included_moveLS_IBBoundaryMeshMapping
#include "ADS/CutCellMeshMapping.h"
#include "ADS/FEMeshPartitioner.h"
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
                          const std::string& restart_read_dirname = "",
                          unsigned int restart_restore_number = 0);

    /*!
     * \brief Default deconstructor.
     */
    virtual ~IBBoundaryMeshMapping() = default;

    /*!
     * \brief Deleted default constructor.
     */
    IBBoundaryMeshMapping() = delete;

    /*!
     * \brief Deleted copy constructor.
     */
    IBBoundaryMeshMapping(const IBBoundaryMeshMapping& from) = delete;

    /*!
     * \brief Deleted assignment operator.
     */
    IBBoundaryMeshMapping& operator=(const IBBoundaryMeshMapping& that) = delete;

    using ADS::GeneralBoundaryMeshMapping::updateBoundaryLocation;
    virtual void updateBoundaryLocation(double time, unsigned int part, bool end_of_timestep = false) override;

    using ADS::GeneralBoundaryMeshMapping::initializeBoundaryLocation;
    virtual void initializeBoundaryLocation(double time, unsigned int part) override;

private:
    inline double upper_channel(const double s, const double t)
    {
        return d_alpha / (2.0 * M_PI) * (1.0 + d_gamma * std::sin(2.0 * M_PI * (s - t)));
    }

    inline double lower_channel(const double s, const double t)
    {
        return -d_alpha / (2.0 * M_PI) * (1.0 + d_gamma * std::sin(2.0 * M_PI * (s - t)));
    }
    IBTK::LDataManager* d_base_data_manager;
    std::vector<int> d_part_nums;
    int d_level_num = IBTK::invalid_level_number;

    double d_alpha = std::numeric_limits<double>::quiet_NaN();
    double d_gamma = std::numeric_limits<double>::quiet_NaN();
};

#endif
