"""Carrier recovery: frequency-offset estimation and phase recovery.

* ``estimate_freq_offset`` - 4th-power spectral peak search (works for QPSK
  and square 16-QAM, both of which have a non-zero E[s^4]).
* ``viterbi_viterbi`` - 4th-power feed-forward phase estimator for QPSK.
* ``blind_phase_search`` - Pfau et al. BPS for arbitrary QAM.
* ``resolve_ambiguity`` - removes the pi/2 ambiguity using known preamble
  symbols.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .modulation import Constellation


def estimate_freq_offset(x: np.ndarray, symbol_rate: float, nfft: int | None = None) -> float:
    """Frequency offset in Hz from the peak of the spectrum of x^4."""
    y = x ** 4
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(max(4 * y.size, 1024))))
    spec = np.abs(np.fft.fft(y, nfft))
    k = int(np.argmax(spec))
    # parabolic refinement on the log-magnitude
    a, b, c = np.log(spec[[(k - 1) % nfft, k, (k + 1) % nfft]] + 1e-30)
    delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    f = (k + delta) / nfft
    if f >= 0.5:
        f -= 1.0
    return f * symbol_rate / 4


def remove_freq_offset(x: np.ndarray, f_off: float, symbol_rate: float) -> np.ndarray:
    n = np.arange(x.size)
    return x * np.exp(-2j * np.pi * f_off / symbol_rate * n)


def unwrap_quarter(phi: np.ndarray) -> np.ndarray:
    """Unwrap a phase estimate that is only defined modulo pi/2."""
    return np.unwrap(4 * phi) / 4


def viterbi_viterbi(x: np.ndarray, window: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """V&V 4th-power phase estimation with a sliding average of 2*window+1."""
    s4 = x ** 4
    avg = uniform_filter1d(s4.real, 2 * window + 1, mode="nearest") + \
        1j * uniform_filter1d(s4.imag, 2 * window + 1, mode="nearest")
    phi = unwrap_quarter(np.angle(-avg) / 4)  # QPSK^4 = -1
    return x * np.exp(-1j * phi), phi


def blind_phase_search(x: np.ndarray, const: Constellation, n_test: int = 32,
                       window: int = 32, chunk: int = 1 << 15) -> tuple[np.ndarray, np.ndarray]:
    """Blind phase search over [-pi/4, pi/4) with sliding-window error sums."""
    phases = (np.arange(n_test) / n_test - 0.5) * (np.pi / 2)
    rot = np.exp(1j * phases)
    best = np.empty(x.size)
    for start in range(0, x.size, chunk):
        lo = max(0, start - window)
        hi = min(x.size, start + chunk + window)
        seg = x[lo:hi]
        cand = seg[:, None] * rot[None, :]
        dec = const.points[const.decide(cand.reshape(-1))].reshape(cand.shape)
        dist = np.abs(cand - dec) ** 2
        dist = uniform_filter1d(dist, 2 * window + 1, axis=0, mode="nearest")
        b = phases[np.argmin(dist, axis=1)]
        best[start:min(start + chunk, x.size)] = -b[start - lo: start - lo + min(chunk, x.size - start)]
    phi = unwrap_quarter(best)
    return x * np.exp(-1j * phi), phi


def resolve_ambiguity(x: np.ndarray, reference: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
    """Rotate by k*pi/2 so that x[start:start+len(reference)] best matches the
    known reference symbols."""
    seg = x[start: start + reference.size]
    scores = [np.real(np.vdot(reference, seg * 1j ** k)) for k in range(4)]
    k = int(np.argmax(scores))
    return x * 1j ** k, k


def frame_sync(x: np.ndarray, preamble: np.ndarray, start: int, max_lag: int = 64) -> tuple[np.ndarray, int]:
    """Locate a known preamble expected at index ``start`` within +-max_lag
    symbols using a magnitude correlation (insensitive to carrier phase), and
    realign the stream. Returns (aligned symbols, lag); lag > 0 means late.

    A carrier frequency offset makes the CD equaliser see a shifted spectrum,
    which turns into a bulk delay of several symbols, so this step is needed
    after CDC + FOE.
    """
    if start < max_lag or start + preamble.size + max_lag > x.size:
        raise ValueError("preamble search window outside the frame")
    n = preamble.size
    scores = [np.abs(np.vdot(preamble, x[start + lag: start + lag + n]))
              for lag in range(-max_lag, max_lag + 1)]
    lag = int(np.argmax(scores)) - max_lag
    return np.roll(x, -lag), lag
