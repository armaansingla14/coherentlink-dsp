"""Single-polarisation coherent optical channel model.

Impairments, applied in this order on the oversampled waveform:
chromatic dispersion (all-pass quadratic phase), static fractional timing
offset, laser frequency offset, laser phase noise (Wiener process with the
combined Tx + LO linewidth) and complex AWGN set from the OSNR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fftcore

C_LIGHT = 299_792_458.0


@dataclass
class LinkParams:
    symbol_rate: float = 32e9
    sps: int = 2
    wavelength: float = 1550e-9
    dispersion_ps_nm_km: float = 17.0
    length_km: float = 1000.0
    linewidth_hz: float = 100e3        # combined Tx + LO
    freq_offset_hz: float = 1.0e9
    timing_offset: float = 0.37        # fraction of a symbol
    osnr_db: float = 15.0
    rrc_beta: float = 0.1

    @property
    def fs(self) -> float:
        return self.symbol_rate * self.sps

    @property
    def beta2(self) -> float:
        """Group-velocity dispersion [s^2/m]."""
        d = self.dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s/m^2
        return -d * self.wavelength ** 2 / (2 * np.pi * C_LIGHT)

    @property
    def accumulated_dispersion_ps_nm(self) -> float:
        return self.dispersion_ps_nm_km * self.length_km


def cd_transfer(n: int, fs: float, beta2: float, length_m: float) -> np.ndarray:
    """Frequency response of chromatic dispersion on an n-point FFT grid."""
    w = 2 * np.pi * fftcore.fftfreq(n, 1 / fs)
    return np.exp(-1j * beta2 / 2 * w ** 2 * length_m)


def apply_cd(x: np.ndarray, p: LinkParams) -> np.ndarray:
    n = 1 << (x.size - 1).bit_length()
    xp = np.zeros(n, dtype=complex)
    xp[: x.size] = x
    y = fftcore.ifft(fftcore.fft(xp) * cd_transfer(n, p.fs, p.beta2, p.length_km * 1e3))
    return y[: x.size]


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a band-limited signal by a (fractional) number of samples."""
    n = 1 << (x.size - 1).bit_length()
    xp = np.zeros(n, dtype=complex)
    xp[: x.size] = x
    f = fftcore.fftfreq(n)
    return fftcore.ifft(fftcore.fft(xp) * np.exp(-2j * np.pi * f * delay_samples))[: x.size]


def phase_noise(n: int, linewidth: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    sigma = np.sqrt(2 * np.pi * linewidth / fs)
    return rng.uniform(-np.pi, np.pi) + np.cumsum(rng.normal(0, sigma, n))


def add_awgn(x: np.ndarray, snr_db: float, sps: int, rng: np.random.Generator) -> np.ndarray:
    """AWGN for a waveform with unit average sample power at ``sps`` samples
    per symbol, such that Es/N0 after the matched filter equals ``snr_db``."""
    var = sps / 10 ** (snr_db / 10)
    noise = rng.normal(0, np.sqrt(var / 2), (x.size, 2)) @ np.array([1, 1j])
    return x + noise


def transmit(x: np.ndarray, p: LinkParams, snr_db: float, rng: np.random.Generator,
             cd: bool = True) -> np.ndarray:
    y = apply_cd(x, p) if cd else x
    y = fractional_delay(y, p.timing_offset * p.sps)
    t = np.arange(y.size) / p.fs
    phi = 2 * np.pi * p.freq_offset_hz * t + phase_noise(y.size, p.linewidth_hz, p.fs, rng)
    y = y * np.exp(1j * phi)
    return add_awgn(y, snr_db, p.sps, rng)
