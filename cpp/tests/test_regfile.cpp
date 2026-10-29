#include "clink/crc16.hpp"
#include "clink/regfile.hpp"
#include "clink/regmap_table.hpp"
#include "test_framework.hpp"

using namespace clink;

namespace {
Status status_of(const Response& r) { return static_cast<Status>(r[0]); }
std::uint32_t data_of(const Response& r) {
    return (std::uint32_t{r[1]} << 24) | (std::uint32_t{r[2]} << 16) | (std::uint32_t{r[3]} << 8) | r[4];
}
bool crc_ok(const Response& r) { return crc16(r.data(), 5) == ((r[5] << 8) | r[6]); }
}  // namespace

TEST(reset_values_and_chip_id) {
    RegisterFile rf(kDspRegisters);
    auto r = rf.handle(encode_request(kOpRead, 0x000, 0));
    CHECK(status_of(r) == Status::Ok);
    CHECK(data_of(r) == 0xC0DE0516u);
    CHECK(crc_ok(r));
}

TEST(read_only_and_bad_address) {
    RegisterFile rf(kDspRegisters);
    CHECK(status_of(rf.handle(encode_request(kOpWrite, 0x000, 1))) == Status::AccessViolation);
    CHECK(status_of(rf.handle(encode_request(kOpRead, 0x3FC, 0))) == Status::BadAddr);
    CHECK(status_of(rf.handle(encode_request(0x11, 0x000, 0))) == Status::BadOp);
}

TEST(bad_crc_rejected_and_state_unchanged) {
    RegisterFile rf(kDspRegisters);
    auto req = encode_request(kOpWrite, 0x00C, 1234);
    req[5] ^= 0x01;
    CHECK(status_of(rf.handle(req)) == Status::BadCrc);
    CHECK(*rf.hw_get(0x00C) == 17000u);
}

TEST(rw_w1c_and_wo_semantics) {
    RegisterFile rf(kDspRegisters);
    // CTRL: write DSP_EN (bit1) together with SOFT_RESET strobe (bit0)
    CHECK(status_of(rf.handle(encode_request(kOpWrite, 0x008, 0x3))) == Status::Ok);
    CHECK(rf.last_strobe() == 0x1);
    auto r = rf.handle(encode_request(kOpRead, 0x008, 0));
    CHECK((data_of(r) & 0x1) == 0);  // WO bit reads as zero
    CHECK((data_of(r) & 0x2) == 0x2);
    // IRQ_STATUS: hardware sets bits, host clears with write-one
    rf.hw_set(0x018, 0xB);
    rf.handle(encode_request(kOpWrite, 0x018, 0x1));
    CHECK(*rf.hw_get(0x018) == 0xA);
    // bits outside defined fields are not stored
    rf.handle(encode_request(kOpWrite, 0x00C, 0xFFFFFFFFu));
    CHECK(*rf.hw_get(0x00C) == 0x00FFFFFFu);
}
