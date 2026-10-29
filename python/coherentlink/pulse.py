"""Root-raised-cosine pulse shaping and matched filtering."""

from __future__ import annotations

import numpy as np
from scipy import signal


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Unit-energy RRC impulse response, ``span`` symbols long (odd length)."""
    if not 0 < beta <= 1:
        raise ValueError("roll-off must be in (0, 1]")
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                         + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    out = np.zeros(symbols.size * sps, dtype=complex)
    out[::sps] = symbols
    return out


def shape(symbols: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    """Upsample and filter; output is aligned so that sample k*sps is symbol k."""
    x = upsample(symbols, sps)
    y = signal.fftconvolve(x, taps)
    d = (taps.size - 1) // 2
    return y[d:d + x.size] * np.sqrt(sps)


def matched_filter(x: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    y = signal.fftconvolve(x, taps[::-1].conj())
    d = (taps.size - 1) // 2
    return y[d:d + x.size] / np.sqrt(sps)
