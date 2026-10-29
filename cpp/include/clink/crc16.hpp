#pragma once
#include <cstddef>
#include <cstdint>

namespace clink {

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
std::uint16_t crc16(const std::uint8_t* data, std::size_t len, std::uint16_t init = 0xFFFF);

}  // namespace clink
