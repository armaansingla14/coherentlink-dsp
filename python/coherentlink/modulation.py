"""Gray-mapped QPSK and 16-QAM constellations, hard-decision demapping and
closed-form AWGN error rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc


def _gray(n: int) -> np.ndarray:
    v = np.arange(n)
    return v ^ (v >> 1)


@dataclass(frozen=True)
class Constellation:
    name: str
    points: np.ndarray        # complex, unit average energy, indexed by bit label
    bits_per_symbol: int

    @property
    def order(self) -> int:
        return self.points.size

    def map(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.size % self.bits_per_symbol:
            raise ValueError("bit count must be a multiple of bits_per_symbol")
        b = bits.reshape(-1, self.bits_per_symbol)
        weights = 1 << np.arange(self.bits_per_symbol - 1, -1, -1)
        return self.points[b @ weights]

    def decide(self, symbols: np.ndarray) -> np.ndarray:
        """Nearest-point decision, returns label indices."""
        return _nearest(self, np.asarray(symbols))

    def demap(self, symbols: np.ndarray) -> np.ndarray:
        idx = self.decide(symbols)
        shifts = np.arange(self.bits_per_symbol - 1, -1, -1)
        return ((idx[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _pam_levels(m: int) -> np.ndarray:
    """Gray-labelled M-PAM levels: level of label g is at position gray^-1(g)."""
    g = _gray(m)
    lev = np.empty(m)
    lev[g] = 2 * np.arange(m) - (m - 1)
    return lev


def _nearest(c: Constellation, s: np.ndarray) -> np.ndarray:
    # Square QAM: independent slicing per quadrature axis.
    m = int(round(np.sqrt(c.order)))
    half = c.bits_per_symbol // 2
    scale = c._scale  # type: ignore[attr-defined]
    lev_to_label = c._lev_to_label  # type: ignore[attr-defined]

    def slice_axis(v):
        k = np.clip(np.round((v / scale + (m - 1)) / 2), 0, m - 1).astype(np.int64)
        return lev_to_label[k]

    return (slice_axis(s.real) << half) | slice_axis(s.imag)


def square_qam(m_total: int, name: str) -> Constellation:
    m = int(round(np.sqrt(m_total)))
    if m * m != m_total:
        raise ValueError("only square QAM supported")
    bps = int(np.log2(m_total))
    lev = _pam_levels(m)
    labels = np.arange(m_total)
    i_lab, q_lab = labels >> (bps // 2), labels & (m - 1)
    pts = lev[i_lab] + 1j * lev[q_lab]
    scale = 1.0 / np.sqrt(np.mean(np.abs(pts) ** 2))
    c = Constellation(name=name, points=pts * scale, bits_per_symbol=bps)
    # position k (0..m-1, ascending amplitude) -> Gray label
    lev_to_label = np.empty(m, dtype=np.int64)
    lev_to_label[((lev + (m - 1)) // 2).astype(int)] = np.arange(m)
    object.__setattr__(c, "_scale", scale)
    object.__setattr__(c, "_lev_to_label", lev_to_label)
    return c


QPSK = square_qam(4, "QPSK")
QAM16 = square_qam(16, "16QAM")
FORMATS = {"qpsk": QPSK, "16qam": QAM16}


def theoretical_ber(fmt: Constellation, snr_db: np.ndarray | float) -> np.ndarray:
    """Gray-coded BER on AWGN for symbol SNR Es/N0 (dB)."""
    snr = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    q = lambda x: 0.5 * erfc(x / np.sqrt(2))  # noqa: E731
    if fmt.order == 4:
        return q(np.sqrt(snr))
    if fmt.order == 16:
        # nearest-neighbour approximation, accurate above ~5 dB
        return 0.75 * q(np.sqrt(snr / 5.0)) + 0.5 * q(3 * np.sqrt(snr / 5.0))
    raise ValueError("unsupported format")


def osnr_to_snr_db(osnr_db, symbol_rate: float, ref_bw: float = 12.5e9, pols: int = 1):
    """Convert OSNR in a 0.1 nm (12.5 GHz) reference bandwidth to Es/N0.

    OSNR = p * Rs / (2 * Bref) * SNR  (Essiambre et al.), p = polarisations.
    """
    return np.asarray(osnr_db, dtype=float) + 10 * np.log10(2 * ref_bw / (pols * symbol_rate))


def snr_to_osnr_db(snr_db, symbol_rate: float, ref_bw: float = 12.5e9, pols: int = 1):
    return np.asarray(snr_db, dtype=float) - 10 * np.log10(2 * ref_bw / (pols * symbol_rate))
