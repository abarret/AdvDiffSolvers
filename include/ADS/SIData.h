#ifndef included_ADS_SIData
#define included_ADS_SIData

#include <tbox/AbstractStream.h>
#include <tbox/Database.h>

#include <CellIndex.h>
#include <Index.h>
#include <IntVector.h>

#include <array>

namespace ADS
{
namespace sharp_interface
{

struct IPWeights
{
    static constexpr int s_num_pts = 4;
    IPWeights(std::array<double, s_num_pts> weights, std::array<SAMRAI::pdat::CellIndex<NDIM>, s_num_pts> idxs)
        : d_weights(std::move(weights)), d_idxs(std::move(idxs))
    {
    }
    std::array<double, s_num_pts> d_weights;
    std::array<SAMRAI::pdat::CellIndex<NDIM>, s_num_pts> d_idxs;
};

class SIData
{
public:
    SIData() = default;

    SIData& operator=(const SIData& other);

    ~SIData() = default;

    void copySourceItem(const SAMRAI::hier::Index<NDIM>& index,
                        const SAMRAI::hier::IntVector<NDIM>& src_offset,
                        const SIData& src_item);

    size_t getDataStreamSize() const;

    void packStream(SAMRAI::tbox::AbstractStream& stream);

    void unpackStream(SAMRAI::tbox::AbstractStream& stream, const SAMRAI::hier::IntVector<NDIM>& offset);

    void getFromDatabase(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& database);

    void putToDatabase(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database>& database);

private:
    double d_val = std::numeric_limits<double>::quiet_NaN();
    IPWeights d_ip_weights;
};
} // namespace sharp_interface
} // namespace ADS
#endif
