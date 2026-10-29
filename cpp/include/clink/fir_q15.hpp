#pragma once
// Fixed-point FIR filter in Q1.15 arithmetic, as it would be implemented in
// an ASIC datapath: 16-bit samples and coefficients, wide accumulator,
// round-half-up on the final shift and saturation to int16.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clink {

std::int16_t to_q15(double v);           // round to nearest, saturate
double from_q15(std::int16_t v);
std::int16_t saturate16(std::int64_t v);

class FirQ15 {
public:
    explicit FirQ15(std::vector<std::int16_t> taps);
    // Streaming: one input sample -> one output sample.
    std::int16_t process(std::int16_t x);
    // Block convolution with zero initial state; output has the same length.
    std::vector<std::int16_t> filter(const std::vector<std::int16_t>& x);
    void reset();
    std::size_t saturations() const { return saturations_; }
    const std::vector<std::int16_t>& taps() const { return taps_; }

private:
    std::vector<std::int16_t> taps_;
    std::vector<std::int16_t> delay_;
    std::size_t pos_ = 0;
    std::size_t saturations_ = 0;
};

}  // namespace clink
