#include "clink/crc16.hpp"

#include <array>

namespace clink {

namespace {
constexpr std::array<std::uint16_t, 256> make_table() {
    std::array<std::uint16_t, 256> t{};
    for (unsigned b = 0; b < 256; ++b) {
        std::uint16_t crc = static_cast<std::uint16_t>(b << 8);
        for (int i = 0; i < 8; ++i)
            crc = static_cast<std::uint16_t>((crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1));
        t[b] = crc;
    }
    return t;
}
constexpr auto kTable = make_table();
}  // namespace

std::uint16_t crc16(const std::uint8_t* data, std::size_t len, std::uint16_t init) {
    std::uint16_t crc = init;
    for (std::size_t i = 0; i < len; ++i)
        crc = static_cast<std::uint16_t>((crc << 8) ^ kTable[((crc >> 8) ^ data[i]) & 0xFF]);
    return crc;
}

}  // namespace clink
