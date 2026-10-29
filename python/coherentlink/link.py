"""Frame-level transmitter / channel / receiver simulation and BER counting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import carrier, cdcomp, channel, fec, pulse, timing
from .modulation import FORMATS, Constellation, osnr_to_snr_db


@dataclass
class RxConfig:
    cdc: bool = True
    timing: bool = True
    foe: bool = True
    cpr: bool = True
    sync: bool = True
    cpr_window: int = 32
    bps_test_phases: int = 32
    preamble: int = 256          # known symbols for frame sync and pi/2 ambiguity
    guard: int = 512             # symbols excluded from BER at each frame edge;
                                 # the preamble sits in the last part of the guard


@dataclass
class FrameResult:
    pre_errors: int              # raw (pre-FEC) bit errors over the payload
    pre_bits: int
    post_errors: int             # information-bit errors after decoding
    post_bits: int
    est_freq_offset: float = 0.0
    est_timing: float = 0.0
    evm_rms: float = 0.0


def make_frame_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, n_bits, dtype=np.uint8)


def receive(rx: np.ndarray, p: channel.LinkParams, const: Constellation, cfg: RxConfig,
            preamble_syms: np.ndarray, taps: np.ndarray):
    pre_start = cfg.guard - cfg.preamble
    """Run the receiver DSP chain on one oversampled frame. Returns symbols
    (1 sps, unit power, phase-corrected) and diagnostics."""
    x = rx
    if cfg.cdc:
        x = cdcomp.OverlapSaveCDC(p)(x)
    x = pulse.matched_filter(x, taps, p.sps)
    tau = 0.0
    if cfg.timing:
        x, tau = timing.oerder_meyr(x, p.sps)
    s = x[:: p.sps]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    f_est = 0.0
    if cfg.foe:
        f_est = carrier.estimate_freq_offset(s, p.symbol_rate)
        s = carrier.remove_freq_offset(s, f_est, p.symbol_rate)
    if cfg.sync:
        s, _ = carrier.frame_sync(s, preamble_syms, pre_start)
    if cfg.cpr:
        if const.order == 4:
            s, _ = carrier.viterbi_viterbi(s, cfg.cpr_window)
        else:
            s, _ = carrier.blind_phase_search(s, const, cfg.bps_test_phases, cfg.cpr_window)
        s, _ = carrier.resolve_ambiguity(s, preamble_syms, pre_start)
    return s, f_est, tau


def simulate_frame(p: channel.LinkParams, fmt: str, n_symbols: int, cfg: RxConfig,
                   rng: np.random.Generator, code=None, interleaver_seed: int = 7,
                   rx_params: channel.LinkParams | None = None) -> FrameResult:
    """Simulate one frame. With ``code`` the payload is FEC-encoded and the
    post-FEC counts refer to information bits.

    ``rx_params`` lets the receiver assume different link parameters from the
    real channel (e.g. a mis-configured dispersion value for the CDC)."""
    const = FORMATS[fmt]
    bps = const.bits_per_symbol
    taps = pulse.rrc_taps(p.rrc_beta, p.sps, 64)
    n_total_bits = n_symbols * bps
    g = cfg.guard
    payload_bits = (n_symbols - 2 * g) * bps

    tx_bits = make_frame_bits(n_total_bits, rng)
    info = None
    if code is not None:
        n_cw = payload_bits // code.n
        info = make_frame_bits(n_cw * code.k, rng)
        coded = fec.interleave(code.encode(info), interleaver_seed)
        tx_bits[g * bps: g * bps + coded.size] = coded

    syms = const.map(tx_bits)
    wave = pulse.shape(syms, taps, p.sps)
    snr_db = float(osnr_to_snr_db(p.osnr_db, p.symbol_rate))
    rx = channel.transmit(wave, p, snr_db, rng, cd=True)

    s, f_est, tau = receive(rx, rx_params or p, const, cfg, syms[g - cfg.preamble: g], taps)
    rx_bits = const.demap(s)
    sl = slice(g * bps, g * bps + payload_bits)
    pre = int(np.count_nonzero(rx_bits[sl] != tx_bits[sl]))
    evm = float(np.sqrt(np.mean(np.abs(s[g:-g] - syms[g:-g]) ** 2)))
    if code is None:
        return FrameResult(pre, payload_bits, pre, payload_bits, f_est, tau, evm)
    hard = fec.deinterleave(rx_bits[g * bps: g * bps + n_cw * code.n], interleaver_seed)
    dec = code.decode(hard)
    dec = dec[0] if isinstance(dec, tuple) else dec
    post = int(np.count_nonzero(dec != info))
    return FrameResult(pre, payload_bits, post, info.size, f_est, tau, evm)


def run_ber(p: channel.LinkParams, fmt: str, n_frames: int, n_symbols: int, cfg: RxConfig,
            seed: int = 0, code=None) -> dict:
    """Simulate ``n_frames`` independent frames and accumulate bit errors.

    Returns pre-FEC BER (``ber``) and, if a code is given, post-FEC BER on the
    information bits (``post_ber``)."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(4, dtype=np.int64)
    fo, evm = [], []
    for _ in range(n_frames):
        r = simulate_frame(p, fmt, n_symbols, cfg, rng, code=code)
        tot += [r.pre_errors, r.pre_bits, r.post_errors, r.post_bits]
        fo.append(r.est_freq_offset)
        evm.append(r.evm_rms)
    return dict(errors=int(tot[0]), bits=int(tot[1]), ber=tot[0] / tot[1],
                post_errors=int(tot[2]), post_bits=int(tot[3]), post_ber=tot[2] / tot[3],
                symbols=n_frames * (n_symbols - 2 * cfg.guard),
                foe_mean_hz=float(np.mean(fo)), evm_rms=float(np.mean(evm)))


def with_osnr(p: channel.LinkParams, osnr_db: float) -> channel.LinkParams:
    return replace(p, osnr_db=osnr_db)
