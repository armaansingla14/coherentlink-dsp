"""Chromatic-dispersion compensation.

Two equivalent implementations:

* ``cd_fir_taps`` + time-domain convolution (Savory 2008 truncated impulse
  response), convenient for short blocks and for the fixed-point C++ FIR.
* ``OverlapSaveCDC`` - frequency-domain equaliser using overlap-save with the
  radix-2 FFT from :mod:`fftcore`, which is how CDC is done at line rate.
"""

from __future__ import annotations

import numpy as np

from . import fftcore
from .channel import C_LIGHT, LinkParams, cd_transfer


def cd_memory_samples(p: LinkParams, bandwidth: float | None = None) -> int:
    """Delay spread of chromatic dispersion in samples over ``bandwidth``
    (default: the full sampled band fs, which is what the all-pass
    frequency-domain equaliser covers)."""
    bw = p.fs if bandwidth is None else bandwidth
    spread = abs(p.beta2) * p.length_km * 1e3 * 2 * np.pi * bw
    return int(np.ceil(spread * p.fs))


def cd_fir_taps(p: LinkParams) -> np.ndarray:
    """Time-domain CD compensating FIR (Savory, Opt. Express 2008).

    Savory's expression uses the opposite sign convention for the carrier
    phase to :func:`channel.cd_transfer`, hence the final conjugate. The
    truncated response is only accurate when the tap count is large (long
    links); for short links the frequency-domain equaliser is preferred."""
    d = p.dispersion_ps_nm_km * 1e-12 / 1e-6  # s/m^2
    z = p.length_km * 1e3
    lam = p.wavelength
    ts = 1 / p.fs
    half = int(np.floor(abs(d) * lam ** 2 * z / (2 * C_LIGHT * ts ** 2)))
    k = np.arange(-half, half + 1)
    a = np.sqrt(1j * C_LIGHT * ts ** 2 / (d * lam ** 2 * z))
    return np.conj(a * np.exp(-1j * np.pi * C_LIGHT * ts ** 2 / (d * lam ** 2 * z) * k ** 2))


class OverlapSaveCDC:
    """Frequency-domain CD equaliser with overlap-save block processing."""

    def __init__(self, p: LinkParams, nfft: int | None = None):
        # 25 % margin over the delay spread keeps the circular-convolution
        # wrap-around of the truncated equaliser response out of the kept part
        self.overlap = int(np.ceil(1.25 * cd_memory_samples(p)))
        self.overlap += self.overlap % 2
        if nfft is None:
            nfft = 1 << int(np.ceil(np.log2(4 * max(self.overlap, 64))))
        if nfft <= self.overlap:
            raise ValueError("FFT size must exceed the CD memory")
        self.nfft = nfft
        self.step = nfft - self.overlap
        self.h_inv = np.conj(cd_transfer(nfft, p.fs, p.beta2, p.length_km * 1e3))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        ov, step, nfft = self.overlap, self.step, self.nfft
        n_blocks = int(np.ceil(n / step))
        half = ov // 2
        padded = np.zeros(n_blocks * step + ov, dtype=complex)
        padded[half: half + n] = x
        idx = np.arange(n_blocks)[:, None] * step + np.arange(nfft)[None, :]
        blocks = padded[np.minimum(idx, padded.size - 1)]
        blocks[idx >= padded.size] = 0
        y = fftcore.ifft(fftcore.fft(blocks) * self.h_inv)
        out = y[:, half: half + step].reshape(-1)
        return out[:n]
