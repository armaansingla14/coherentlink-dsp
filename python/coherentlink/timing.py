"""Symbol timing recovery.

* ``oerder_meyr`` - non-data-aided feed-forward estimator using the symbol-rate
  spectral line of |x|^2. The 2-sps input is interpolated to 4 sps in the
  frequency domain first; the estimate is applied with a fractional delay.
  Vectorised, used for the long BER runs.
* ``GardnerLoop`` - classic feedback loop with a Gardner timing-error detector,
  PI loop filter and cubic Farrow interpolator. Tracks sampling-clock
  frequency offsets; it is a per-symbol Python loop, so it is used on shorter
  sequences.
"""

from __future__ import annotations

import numpy as np

from . import fftcore
from .channel import fractional_delay


def _resample_fft(x: np.ndarray, factor: int) -> np.ndarray:
    n = x.size
    X = np.fft.fft(x)
    Y = np.zeros(n * factor, dtype=complex)
    h = n // 2
    Y[:h] = X[:h]
    Y[-h:] = X[-h:]
    return np.fft.ifft(Y) * factor


def oerder_meyr_offset(x: np.ndarray, sps: int, block: int | None = None) -> np.ndarray:
    """Timing offset estimate(s) in symbol periods, in [-0.5, 0.5).

    With ``block`` set, one estimate per block of ``block`` symbols is
    returned (unwrapped across blocks); otherwise a single estimate.
    """
    if sps != 2:
        raise ValueError("implemented for 2 samples per symbol")
    x4 = _resample_fft(x, 2)  # 4 sps
    p = np.abs(x4) ** 2
    rot = np.exp(-2j * np.pi * np.arange(p.size) / 4)
    terms = p * rot
    if block is None:
        return np.array([-np.angle(terms.sum()) / (2 * np.pi)])
    nb = p.size // (4 * block)
    acc = terms[: nb * 4 * block].reshape(nb, -1).sum(axis=1)
    return -np.unwrap(np.angle(acc)) / (2 * np.pi)


def oerder_meyr(x: np.ndarray, sps: int) -> tuple[np.ndarray, float]:
    """Estimate and remove a static timing offset. Returns (aligned, tau)."""
    tau = float(oerder_meyr_offset(x, sps)[0])
    return fractional_delay(x, -tau * sps), tau


def farrow_cubic(x: np.ndarray, i: int, mu: float) -> complex:
    """Cubic Lagrange interpolation between x[i] and x[i+1] at fraction mu."""
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    c0 = x0
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + c0


class GardnerLoop:
    """Gardner TED + PI loop filter + cubic interpolator at 2 samples/symbol."""

    def __init__(self, bn_ts: float = 0.01, zeta: float = 0.707, ted_gain: float = 2.0):
        theta = bn_ts / (zeta + 1 / (4 * zeta))
        d = 1 + 2 * zeta * theta + theta ** 2
        self.kp = 4 * zeta * theta / d / ted_gain
        self.ki = 4 * theta ** 2 / d / ted_gain

    def run(self, x: np.ndarray, sps: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Returns (symbols, timing estimate per symbol in samples)."""
        if sps != 2:
            raise ValueError("implemented for 2 samples per symbol")
        out, taus = [], []
        tau = 0.0       # fractional sample offset of the strobe
        integ = 0.0
        prev = 0j
        k = 2.0          # position of the next symbol strobe (in samples)
        while k + 3 < x.size:
            base = int(np.floor(k))
            mu = k - base
            sym = farrow_cubic(x, base, mu)
            mid_pos = k - 1.0
            mb = int(np.floor(mid_pos))
            mid = farrow_cubic(x, mb, mid_pos - mb)
            if out:
                e = np.real(np.conj(mid) * (prev - sym))
                integ += self.ki * e
                tau += self.kp * e + integ
            out.append(sym)
            taus.append(tau)
            prev = sym
            k = 2.0 * (len(out) + 1) + tau
        return np.asarray(out), np.asarray(taus)
