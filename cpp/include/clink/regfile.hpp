#pragma once
// Register file and bus slave implementing the framed register protocol
// (see python/coherentlink/asic/bus.py for the frame layout).

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace clink {

struct RegisterDef {
    const char* name;
    std::uint16_t addr;
    std::uint32_t reset;
    std::uint32_t rw_mask;
    std::uint32_t w1c_mask;
    std::uint32_t wo_mask;
};

enum class Status : std::uint8_t { Ok = 0, BadCrc = 1, BadAddr = 2, AccessViolation = 3, BadOp = 4, Busy = 5 };

constexpr std::uint8_t kOpRead = 0x52;
constexpr std::uint8_t kOpWrite = 0x57;
constexpr std::size_t kRequestLen = 9;
constexpr std::size_t kResponseLen = 7;

using Request = std::array<std::uint8_t, kRequestLen>;
using Response = std::array<std::uint8_t, kResponseLen>;

Request encode_request(std::uint8_t op, std::uint16_t addr, std::uint32_t data);
Response encode_response(Status st, std::uint32_t data);

class RegisterFile {
public:
    template <std::size_t N>
    explicit RegisterFile(const std::array<RegisterDef, N>& defs) : defs_(defs.begin(), defs.end()) {
        reset();
    }

    void reset();
    Status read(std::uint16_t addr, std::uint32_t& value) const;
    Status write(std::uint16_t addr, std::uint32_t value);
    Response handle(const Request& frame);

    // Hardware-side access (bypasses access policy), e.g. to set status bits.
    void hw_set(std::uint16_t addr, std::uint32_t value);
    std::optional<std::uint32_t> hw_get(std::uint16_t addr) const;

    // Strobe bits (WO) seen on the last write, for the surrounding logic.
    std::uint32_t last_strobe() const { return last_strobe_; }

private:
    const RegisterDef* find(std::uint16_t addr) const;
    std::vector<RegisterDef> defs_;
    std::vector<std::uint32_t> values_;
    std::uint32_t last_strobe_ = 0;
};

}  // namespace clink
