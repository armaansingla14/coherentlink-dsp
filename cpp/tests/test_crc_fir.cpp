#include <cmath>
#include <cstring>
#include <vector>

#include "clink/crc16.hpp"
#include "clink/fir_q15.hpp"
#include "test_framework.hpp"

using namespace clink;

TEST(crc16_check_value) {
    const char* s = "123456789";
    CHECK(crc16(reinterpret_cast<const std::uint8_t*>(s), std::strlen(s)) == 0x29B1);
    CHECK(crc16(nullptr, 0) == 0xFFFF);
}

TEST(crc16_detects_single_bit_flips) {
    std::uint8_t frame[7] = {0x57, 0x00, 0x08, 0xDE, 0xAD, 0xBE, 0xEF};
    const std::uint16_t ref = crc16(frame, 7);
    for (int byte = 0; byte < 7; ++byte)
        for (int bit = 0; bit < 8; ++bit) {
            frame[byte] ^= static_cast<std::uint8_t>(1 << bit);
            CHECK(crc16(frame, 7) != ref);
            frame[byte] ^= static_cast<std::uint8_t>(1 << bit);
        }
}

TEST(q15_conversion_and_saturation) {
    CHECK(to_q15(0.5) == 16384);
    CHECK(to_q15(-1.0) == -32768);
    CHECK(to_q15(1.0) == 32767);   // saturates
    CHECK(to_q15(-2.0) == -32768);
    CHECK_NEAR(from_q15(to_q15(0.123)), 0.123, 1.0 / 32768);
}

TEST(fir_impulse_response_equals_taps) {
    std::vector<std::int16_t> taps{1000, -2000, 3000, 16384};
    FirQ15 fir(taps);
    std::vector<std::int16_t> x(8, 0);
    x[0] = 32767;  // ~1.0
    auto y = fir.filter(x);
    for (std::size_t k = 0; k < taps.size(); ++k) CHECK(std::abs(y[k] - taps[k]) <= 1);
    CHECK(y[5] == 0);
}

TEST(fir_matches_double_precision_within_quantisation) {
    std::vector<double> h{0.1, 0.25, 0.3, 0.25, 0.1};
    std::vector<std::int16_t> hq;
    for (double v : h) hq.push_back(to_q15(v));
    FirQ15 fir(hq);
    std::vector<std::int16_t> x;
    std::vector<double> xd;
    for (int i = 0; i < 200; ++i) {
        const double v = 0.4 * std::sin(0.1 * i) + 0.2 * std::cos(0.37 * i);
        xd.push_back(v);
        x.push_back(to_q15(v));
    }
    auto y = fir.filter(x);
    double max_err = 0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        double ref = 0;
        for (std::size_t k = 0; k < h.size() && k <= n; ++k) ref += h[k] * xd[n - k];
        max_err = std::max(max_err, std::fabs(from_q15(y[n]) - ref));
    }
    CHECK(max_err < 4.0 / 32768);
    CHECK(fir.saturations() == 0);
}

TEST(fir_saturates_on_overflow) {
    FirQ15 fir({32767, 32767, 32767});
    std::vector<std::int16_t> x(5, 30000);
    auto y = fir.filter(x);
    CHECK(y[2] == 32767);
    CHECK(fir.saturations() > 0);
    std::vector<std::int16_t> xn(5, -32768);
    y = fir.filter(xn);
    CHECK(y[3] == -32767 || y[3] == -32768);
}
