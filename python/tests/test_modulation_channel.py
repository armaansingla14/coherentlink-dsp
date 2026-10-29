import numpy as np
import pytest

from coherentlink import channel, fftcore, modulation as M, pulse


@pytest.mark.parametrize("fmt", [M.QPSK, M.QAM16])
def test_map_demap_roundtrip_and_unit_energy(fmt):
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, fmt.bits_per_symbol * 1000).astype(np.uint8)
    s = fmt.map(bits)
    np.testing.assert_array_equal(fmt.demap(s), bits)
    assert np.mean(np.abs(fmt.points) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("fmt", [M.QPSK, M.QAM16])
def test_gray_labelling_neighbours_differ_by_one_bit(fmt):
    pts = fmt.points
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = np.min(d[d > 1e-9])
    for a in range(fmt.order):
        for b in range(fmt.order):
            if abs(d[a, b] - dmin) < 1e-9:
                assert bin(a ^ b).count("1") == 1


def test_awgn_ber_matches_theory_qpsk():
    rng = np.random.default_rng(1)
    snr_db = 7.0
    bits = rng.integers(0, 2, 400_000).astype(np.uint8)
    s = M.QPSK.map(bits)
    r = channel.add_awgn(s, snr_db, 1, rng)
    ber = np.mean(M.QPSK.demap(r) != bits)
    assert ber == pytest.approx(float(M.theoretical_ber(M.QPSK, snr_db)), rel=0.08)


def test_osnr_snr_conversion_roundtrip():
    snr = M.osnr_to_snr_db(15.0, 32e9)
    assert float(snr) == pytest.approx(15.0 + 10 * np.log10(25 / 32))
    assert float(M.snr_to_osnr_db(snr, 32e9)) == pytest.approx(15.0)


@pytest.mark.parametrize("n", [1, 2, 16, 256, 4096])
def test_fft_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
    np.testing.assert_allclose(fftcore.fft(x), np.fft.fft(x, axis=-1), atol=1e-8)
    np.testing.assert_allclose(fftcore.ifft(fftcore.fft(x)), x, atol=1e-10)


def test_fft_rejects_bad_length():
    with pytest.raises(ValueError):
        fftcore.fft(np.zeros(10))


def test_rrc_is_nyquist_after_matched_filter():
    rng = np.random.default_rng(2)
    s = M.QAM16.map(rng.integers(0, 2, 4 * 2000).astype(np.uint8))
    taps = pulse.rrc_taps(0.1, 2, 64)
    w = pulse.shape(s, taps, 2)
    assert np.mean(np.abs(w) ** 2) == pytest.approx(1.0, rel=0.02)
    y = pulse.matched_filter(w, taps, 2)[::2]
    assert np.max(np.abs(y[100:-100] - s[100:-100])) < 0.01


def test_cd_is_all_pass_and_invertible():
    p = channel.LinkParams()
    rng = np.random.default_rng(3)
    x = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    y = channel.apply_cd(x, p)
    assert np.sum(np.abs(y) ** 2) == pytest.approx(np.sum(np.abs(x) ** 2), rel=1e-9)
    assert np.max(np.abs(y - x)) > 1.0  # 17 000 ps/nm visibly distorts


def test_fractional_delay_shifts_by_integer():
    x = np.zeros(64, complex)
    x[10] = 1
    y = channel.fractional_delay(x, 3.0)
    assert np.argmax(np.abs(y)) == 13


def test_phase_noise_variance():
    rng = np.random.default_rng(4)
    lw, fs = 1e6, 64e9
    ph = np.array([np.diff(channel.phase_noise(1000, lw, fs, rng)) for _ in range(20)]).ravel()
    assert np.var(ph) == pytest.approx(2 * np.pi * lw / fs, rel=0.05)
