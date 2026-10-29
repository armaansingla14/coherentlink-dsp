import numpy as np
import pytest

from coherentlink import fec


def test_hamming_corrects_any_single_error():
    h = fec.Hamming74()
    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, 4 * 100).astype(np.uint8)
    code = h.encode(info)
    assert np.all(code.reshape(-1, 7) @ h.H.T % 2 == 0)
    for pos in range(7):
        c = code.reshape(-1, 7).copy()
        c[:, pos] ^= 1
        np.testing.assert_array_equal(h.decode(c.reshape(-1)), info)


def test_hamming_double_error_is_not_corrected():
    h = fec.Hamming74()
    info = np.array([1, 0, 1, 1], dtype=np.uint8)
    c = h.encode(info)
    c[[0, 1]] ^= 1
    assert not np.array_equal(h.decode(c), info)


@pytest.fixture(scope="module")
def ldpc():
    return fec.LdpcCode(n=504, seed=3)


def test_ldpc_structure(ldpc):
    col_w = ldpc.H.sum(axis=0)
    row_w = ldpc.H.sum(axis=1)
    assert np.all(col_w == 3) and np.all(row_w == 6)
    assert 0.49 < ldpc.rate < 0.52


def test_ldpc_codewords_satisfy_parity(ldpc):
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, ldpc.k * 20).astype(np.uint8)
    cw = ldpc.encode(info)
    assert not ldpc.syndrome(cw).any()
    dec, ok = ldpc.decode(cw)
    assert ok.all()
    np.testing.assert_array_equal(dec, info)


def test_ldpc_bit_flipping_corrects_sparse_errors(ldpc):
    rng = np.random.default_rng(2)
    n_cw = 200
    info = rng.integers(0, 2, ldpc.k * n_cw).astype(np.uint8)
    cw = ldpc.encode(info)
    err = (rng.random(cw.size) < 0.005).astype(np.uint8)
    dec, ok = ldpc.decode(cw ^ err)
    pre = err.mean()
    post = np.mean(dec != info)
    assert post < pre / 10
    assert ok.mean() > 0.9


def test_interleaver_roundtrip():
    x = np.arange(1000) % 2
    y = fec.interleave(x.astype(np.uint8), seed=4)
    np.testing.assert_array_equal(fec.deinterleave(y, seed=4), x)
