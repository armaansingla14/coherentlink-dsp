from dataclasses import replace

import numpy as np
import pytest

from coherentlink import carrier, cdcomp, channel, link, modulation as M, pulse, timing


@pytest.fixture(scope="module")
def tx():
    rng = np.random.default_rng(10)
    s = M.QPSK.map(rng.integers(0, 2, 2 * 8192).astype(np.uint8))
    taps = pulse.rrc_taps(0.1, 2, 64)
    return s, taps, pulse.shape(s, taps, 2)


def test_overlap_save_cdc_inverts_cd(tx):
    _, _, w = tx
    p = channel.LinkParams()
    y = cdcomp.OverlapSaveCDC(p)(channel.apply_cd(w, p))
    assert np.max(np.abs(y[2000:-2000] - w[2000:-2000])) < 0.02


def test_cd_fir_taps_compensate(tx):
    _, _, w = tx
    for km, tol in ((80.0, 0.2), (1000.0, 0.06)):
        p = channel.LinkParams(length_km=km)
        h = cdcomp.cd_fir_taps(p)
        y = np.convolve(channel.apply_cd(w, p), h, mode="same")
        err = np.abs(y - w)[2000:-2000]
        assert np.sqrt(np.mean(err ** 2)) < tol


def test_oerder_meyr_estimates_static_offset(tx):
    s, taps, w = tx
    for tau in (-0.3, 0.12, 0.41):
        y = pulse.matched_filter(channel.fractional_delay(w, tau * 2), taps, 2)
        est = timing.oerder_meyr_offset(y, 2)[0]
        assert est == pytest.approx(tau, abs=0.02)


def test_gardner_loop_converges_and_tracks_clock_offset(tx):
    from scipy.signal import resample
    s, taps, w = tx
    rng = np.random.default_rng(1)
    y = pulse.matched_filter(channel.fractional_delay(w, 0.6), taps, 2)
    out, taus = timing.GardnerLoop(0.01).run(channel.add_awgn(y, 25, 1, rng))
    assert np.mean(taus[-500:]) == pytest.approx(0.6, abs=0.05)
    k = np.arange(3000, 4000)
    assert np.mean(np.abs(out[k] - s[k + 1])) < 0.15
    ws = resample(w, int(w.size * (1 + 2e-4)))   # 200 ppm clock offset
    out, taus = timing.GardnerLoop(0.02).run(pulse.matched_filter(ws, taps, 2))
    slope = np.polyfit(np.arange(2000, 6000), taus[2000:6000], 1)[0]
    assert slope == pytest.approx(2 * 2e-4, rel=0.15)


def test_freq_offset_estimation():
    rng = np.random.default_rng(5)
    s = M.QAM16.map(rng.integers(0, 2, 4 * 20000).astype(np.uint8))
    rs, fo = 32e9, -1.7e9
    r = channel.add_awgn(s * np.exp(2j * np.pi * fo / rs * np.arange(s.size)), 20, 1, rng)
    assert carrier.estimate_freq_offset(r, rs) == pytest.approx(fo, abs=2e6)


@pytest.mark.parametrize("fmt", ["qpsk", "16qam"])
def test_phase_recovery_tracks_phase_noise(fmt):
    const = M.FORMATS[fmt]
    rng = np.random.default_rng(6)
    s = const.map(rng.integers(0, 2, const.bits_per_symbol * 20000).astype(np.uint8))
    ph = channel.phase_noise(s.size, 200e3, 32e9, rng)
    r = channel.add_awgn(s * np.exp(1j * ph), 20, 1, rng)
    if fmt == "qpsk":
        y, _ = carrier.viterbi_viterbi(r, 32)
    else:
        y, _ = carrier.blind_phase_search(r, const, 32, 32)
    y, _ = carrier.resolve_ambiguity(y, s[:256])
    ser = np.mean(const.decide(y) != const.decide(s))
    assert ser < 1e-3


def test_frame_sync_finds_lag():
    rng = np.random.default_rng(7)
    s = M.QPSK.map(rng.integers(0, 2, 2 * 4000).astype(np.uint8))
    r = np.roll(s, 5) * np.exp(1j * 0.7)
    y, lag = carrier.frame_sync(r, s[256:512], 256)
    assert lag == 5
    np.testing.assert_allclose(y[300:3000], s[300:3000] * np.exp(1j * 0.7))


@pytest.mark.parametrize("fmt,osnr", [("qpsk", 13.0), ("16qam", 20.0)])
def test_full_link_close_to_theory(fmt, osnr):
    p = replace(channel.LinkParams(), osnr_db=osnr)
    r = link.run_ber(p, fmt, 1, 1 << 15, link.RxConfig(), seed=3)
    snr = float(M.osnr_to_snr_db(osnr, p.symbol_rate))
    theory = float(M.theoretical_ber(M.FORMATS[fmt], snr))
    assert r["ber"] < 10 * theory + 1e-4
    assert r["foe_mean_hz"] == pytest.approx(p.freq_offset_hz, abs=5e6)


def test_link_fails_without_cdc():
    p = replace(channel.LinkParams(), osnr_db=20.0)
    r = link.run_ber(p, "qpsk", 1, 1 << 14, link.RxConfig(cdc=False), seed=3)
    assert r["ber"] > 0.2
