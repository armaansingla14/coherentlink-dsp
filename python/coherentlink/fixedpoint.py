"""Bit-exact Python model of the C++ Q1.15 FIR (cpp/src/fir_q15.cpp) and a
helper to run the RRC matched filter in fixed point."""

from __future__ import annotations

import numpy as np


def to_q15(x: np.ndarray, bits: int = 16) -> np.ndarray:
    """Quantise to Q1.15 with round-to-nearest and saturation. ``bits < 16``
    emulates a narrower datapath by rounding to multiples of 2**(16 - bits)."""
    step = 1 << (16 - bits)
    q = np.floor(np.asarray(x) * 32768.0 / step + 0.5) * step
    return np.clip(q, -32768, 32767).astype(np.int16)


def from_q15(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / 32768.0


def fir_q15(taps: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
    """Causal FIR with int64 accumulation, round-half-up shift by 15 and
    saturation. Returns (output, number of saturated samples)."""
    acc = np.convolve(x.astype(np.int64), taps.astype(np.int64))[: x.size]  # exact integer MACs
    y = (acc + (1 << 14)) >> 15
    sat = int(np.count_nonzero((y > 32767) | (y < -32768)))
    return np.clip(y, -32768, 32767).astype(np.int16), sat


def matched_filter_q15(x: np.ndarray, taps: np.ndarray, sps: int, input_rms: float = 0.25,
                       bits: int = 16):
    """Fixed-point matched filter on I and Q.

    The input is scaled so its RMS is ``input_rms`` of full scale, the taps
    are normalised to a peak of 0.5 (one bit of headroom), and the output is
    rescaled to match :func:`pulse.matched_filter`."""
    scale_in = input_rms / np.sqrt(np.mean(np.abs(x) ** 2))
    h = taps[::-1].conj().real
    scale_h = 0.5 / np.max(np.abs(h))
    hq = to_q15(h * scale_h, bits)
    yi, si = fir_q15(hq, to_q15(x.real * scale_in, bits))
    yq, sq = fir_q15(hq, to_q15(x.imag * scale_in, bits))
    y = (from_q15(yi) + 1j * from_q15(yq)) / (scale_in * scale_h * np.sqrt(sps))
    d = (taps.size - 1) // 2
    out = np.zeros_like(y)
    out[: y.size - d] = y[d:]
    return out, si + sq
