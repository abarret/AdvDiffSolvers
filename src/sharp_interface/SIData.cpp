#include <ADS/SIData.h>
#include <ADS/app_namespaces.h>

namespace ADS
{
namespace sharp_interface
{

SIData&
SIData::operator=(const SIData& other)
{
    if (this == &other) return *this;
    this->d_ip_weights = other.d_ip_weights;
    this->d_val = other.d_val;
    return *this;
}

void
SIData::copySourceItem(const hier::Index<NDIM>& /*idx*/, const IntVector<NDIM>& /*src_offset*/, const SIData& src_item)
{
    this->d_ip_weights = src_item.d_ip_weights;
    this->d_val = src_item.d_val;
}

size_t
SIData::getDataStreamSize() const
{
    // d_val
    size_t size = AbstractStream::sizeofDouble();
    // Each double
    size += AbstractStream::sizeofDouble(IPWeights::s_num_pts);
    // Each cell index
    size += IPWeights::s_num_pts * AbstractStream::sizeofInt() * NDIM;
    return size;
}

void
SIData::packStream(AbstractStream& stream)
{
    stream.pack(&d_val);
    stream.pack(d_ip_weights.d_weights.data(), IPWeights::s_num_pts);
    for (int i = 0; i < IPWeights::s_num_pts; ++i) stream.pack(&d_ip_weights.d_idxs[i](0), NDIM);
}

void
SIData::unpackStream(AbstractStream& stream, const IntVector<NDIM>& /*offset*/)
{
    stream.unpack(&d_val);
    stream.unpack(d_ip_weights.d_weights.data(), IPWeights::s_num_pts);
    for (int i = 0; i < IPWeights::s_num_pts; ++i) stream.unpack(&d_ip_weights.d_idxs[i](0), NDIM);
}

void
SIData::getFromDatabase(Pointer<Database>& db)
{
    d_val = db->getDouble("d_val");
    db->getDoubleArray("weights", d_ip_weights.d_weights.data(), IPWeights::s_num_pts);
    for (int i = 0; i < IPWeights::s_num_pts; ++i)
        db->getIntegerArray("idx_" + std::to_string(i), &d_ip_weights.d_idxs[i](0), NDIM);
}

void
SIData::putToDatabase(Pointer<Database>& db)
{
    db->putDouble("d_val", d_val);
    db->putDoubleArray("weights", d_ip_weights.d_weights.data(), IPWeights::s_num_pts);
    for (int i = 0; i < IPWeights::s_num_pts; ++i)
        db->putIntegerArray("idx_" + std::to_string(i), &d_ip_weights.d_idxs[i](0), NDIM);
}
} // namespace sharp_interface
} // namespace ADS
