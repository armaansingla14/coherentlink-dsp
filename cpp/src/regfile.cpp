#include "clink/regfile.hpp"

#include "clink/crc16.hpp"

namespace clink {

Request encode_request(std::uint8_t op, std::uint16_t addr, std::uint32_t data) {
    Request f{};
    f[0] = op;
    f[1] = static_cast<std::uint8_t>(addr >> 8);
    f[2] = static_cast<std::uint8_t>(addr);
    for (int i = 0; i < 4; ++i) f[3 + i] = static_cast<std::uint8_t>(data >> (24 - 8 * i));
    const std::uint16_t crc = crc16(f.data(), 7);
    f[7] = static_cast<std::uint8_t>(crc >> 8);
    f[8] = static_cast<std::uint8_t>(crc);
    return f;
}

Response encode_response(Status st, std::uint32_t data) {
    Response f{};
    f[0] = static_cast<std::uint8_t>(st);
    for (int i = 0; i < 4; ++i) f[1 + i] = static_cast<std::uint8_t>(data >> (24 - 8 * i));
    const std::uint16_t crc = crc16(f.data(), 5);
    f[5] = static_cast<std::uint8_t>(crc >> 8);
    f[6] = static_cast<std::uint8_t>(crc);
    return f;
}

void RegisterFile::reset() {
    values_.resize(defs_.size());
    for (std::size_t i = 0; i < defs_.size(); ++i) values_[i] = defs_[i].reset;
    last_strobe_ = 0;
}

const RegisterDef* RegisterFile::find(std::uint16_t addr) const {
    for (const auto& d : defs_)
        if (d.addr == addr) return &d;
    return nullptr;
}

Status RegisterFile::read(std::uint16_t addr, std::uint32_t& value) const {
    const RegisterDef* d = find(addr);
    if (!d) return Status::BadAddr;
    value = values_[static_cast<std::size_t>(d - defs_.data())] & ~d->wo_mask;
    return Status::Ok;
}

Status RegisterFile::write(std::uint16_t addr, std::uint32_t value) {
    const RegisterDef* d = find(addr);
    if (!d) return Status::BadAddr;
    if ((d->rw_mask | d->w1c_mask | d->wo_mask) == 0) return Status::AccessViolation;
    std::uint32_t& cur = values_[static_cast<std::size_t>(d - defs_.data())];
    cur = (cur & ~d->rw_mask) | (value & d->rw_mask);
    cur &= ~(value & d->w1c_mask);
    last_strobe_ = value & d->wo_mask;
    return Status::Ok;
}

Response RegisterFile::handle(const Request& f) {
    const std::uint16_t crc = static_cast<std::uint16_t>((f[7] << 8) | f[8]);
    if (crc16(f.data(), 7) != crc) return encode_response(Status::BadCrc, 0);
    const std::uint16_t addr = static_cast<std::uint16_t>((f[1] << 8) | f[2]);
    std::uint32_t data = 0;
    for (int i = 0; i < 4; ++i) data = (data << 8) | f[3 + i];
    if (!find(addr)) return encode_response(Status::BadAddr, 0);
    if (f[0] == kOpRead) {
        std::uint32_t v = 0;
        const Status st = read(addr, v);
        return encode_response(st, v);
    }
    if (f[0] == kOpWrite) return encode_response(write(addr, data), 0);
    return encode_response(Status::BadOp, 0);
}

void RegisterFile::hw_set(std::uint16_t addr, std::uint32_t value) {
    if (const RegisterDef* d = find(addr)) values_[static_cast<std::size_t>(d - defs_.data())] = value;
}

std::optional<std::uint32_t> RegisterFile::hw_get(std::uint16_t addr) const {
    if (const RegisterDef* d = find(addr)) return values_[static_cast<std::size_t>(d - defs_.data())];
    return std::nullopt;
}

}  // namespace clink
