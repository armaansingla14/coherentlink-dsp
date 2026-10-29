"""Radix-2 FFT used by the frequency-domain DSP blocks.

The transform is vectorised over leading axes so that overlap-save blocks are
processed in one call. It is validated against ``numpy.fft`` in the tests.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _bitrev(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


@lru_cache(maxsize=64)
def _twiddles(size: int, inverse: bool) -> np.ndarray:
    sign = 1.0 if inverse else -1.0
    return np.exp(sign * 2j * np.pi * np.arange(size // 2) / size)


def fft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """FFT along the last axis (length must be a power of two).

    ``inverse=True`` computes the unnormalised inverse; use :func:`ifft`.
    """
    x = np.asarray(x, dtype=complex)
    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("FFT length must be a power of two")
    lead = x.shape[:-1]
    a = x[..., _bitrev(n)].reshape(-1, n)
    size = 2
    while size <= n:
        half = size // 2
        a = a.reshape(a.shape[0], n // size, size)
        top = a[..., :half]
        bot = a[..., half:] * _twiddles(size, inverse)
        a = np.concatenate([top + bot, top - bot], axis=-1)
        size *= 2
    return a.reshape(*lead, n)


def ifft(x: np.ndarray) -> np.ndarray:
    return fft(x, inverse=True) / np.asarray(x).shape[-1]


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    k = np.arange(n)
    k[k >= (n + 1) // 2] -= n
    return k / (n * d)
