#include "clink/fir_q15.hpp"

#include <cmath>
#include <stdexcept>

namespace clink {

std::int16_t saturate16(std::int64_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return static_cast<std::int16_t>(v);
}

std::int16_t to_q15(double v) { return saturate16(static_cast<std::int64_t>(std::floor(v * 32768.0 + 0.5))); }

double from_q15(std::int16_t v) { return static_cast<double>(v) / 32768.0; }

FirQ15::FirQ15(std::vector<std::int16_t> taps) : taps_(std::move(taps)), delay_(taps_.size(), 0) {
    if (taps_.empty()) throw std::invalid_argument("FIR needs at least one tap");
}

void FirQ15::reset() {
    std::fill(delay_.begin(), delay_.end(), 0);
    pos_ = 0;
    saturations_ = 0;
}

std::int16_t FirQ15::process(std::int16_t x) {
    const std::size_t n = taps_.size();
    delay_[pos_] = x;
    std::int64_t acc = 0;
    std::size_t idx = pos_;
    for (std::size_t k = 0; k < n; ++k) {
        acc += static_cast<std::int32_t>(taps_[k]) * static_cast<std::int32_t>(delay_[idx]);
        idx = (idx == 0) ? n - 1 : idx - 1;
    }
    pos_ = (pos_ + 1) % n;
    // Arithmetic shift with round-half-up (floor((acc + 2^14) / 2^15)).
    const std::int64_t rounded = (acc + (1 << 14)) >> 15;
    const std::int16_t y = saturate16(rounded);
    if (y != rounded) ++saturations_;
    return y;
}

std::vector<std::int16_t> FirQ15::filter(const std::vector<std::int16_t>& x) {
    reset();
    std::vector<std::int16_t> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = process(x[i]);
    return y;
}

}  // namespace clink
