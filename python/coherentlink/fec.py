"""Hard-decision forward error correction.

* ``Hamming74`` - systematic Hamming(7,4) with syndrome decoding (corrects one
  error per codeword).
* ``LdpcCode`` - regular (dv=3, dc=6) Gallager LDPC code, systematic encoding
  through GF(2) Gaussian elimination, decoded with the hard-decision
  Gallager-B bit-flipping algorithm, vectorised over many codewords.
* ``interleave``/``deinterleave`` - random bit interleaver so that error
  bursts (e.g. from a carrier-phase cycle slip) are spread over codewords.
"""

from __future__ import annotations

import numpy as np


class Hamming74:
    n, k = 7, 4
    # G = [I4 | P], H = [P^T | I3]
    P = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)

    def __init__(self):
        self.G = np.concatenate([np.eye(4, dtype=np.uint8), self.P], axis=1)
        self.H = np.concatenate([self.P.T, np.eye(3, dtype=np.uint8)], axis=1)
        # syndrome (as int) -> error position
        self._syn_to_pos = np.full(8, -1)
        for pos in range(7):
            s = self.H[:, pos]
            self._syn_to_pos[s[0] * 4 + s[1] * 2 + s[2]] = pos

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1, 4)
        return (b @ self.G % 2).astype(np.uint8).reshape(-1)

    def decode(self, code: np.ndarray) -> np.ndarray:
        c = np.asarray(code, dtype=np.uint8).reshape(-1, 7).copy()
        s = c @ self.H.T % 2
        si = s[:, 0] * 4 + s[:, 1] * 2 + s[:, 2]
        pos = self._syn_to_pos[si]
        rows = np.flatnonzero(pos >= 0)
        c[rows, pos[rows]] ^= 1
        return c[:, :4].reshape(-1)


def _gf2_rref(H: np.ndarray):
    """Row-reduce H over GF(2). Returns (reduced H, pivot columns)."""
    H = H.copy() % 2
    rows, cols = H.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.flatnonzero(H[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            H[[r, p]] = H[[p, r]]
        others = np.flatnonzero(H[:, c])
        others = others[others != r]
        H[others] ^= H[r]
        pivots.append(c)
        r += 1
    return H[:r], pivots


class LdpcCode:
    """Regular Gallager (dv, dc) LDPC code with hard-decision decoding."""

    def __init__(self, n: int = 1008, dv: int = 3, dc: int = 6, seed: int = 1):
        if n % dc:
            raise ValueError("n must be a multiple of dc")
        rng = np.random.default_rng(seed)
        m = n * dv // dc
        sub = m // dv
        base = np.zeros((sub, n), dtype=np.uint8)
        for i in range(sub):
            base[i, i * dc:(i + 1) * dc] = 1
        blocks = [base] + [base[:, rng.permutation(n)] for _ in range(dv - 1)]
        H = np.concatenate(blocks, axis=0)
        Hr, piv = _gf2_rref(H)
        # Column order: information columns first, pivot (parity) columns last.
        info = [c for c in range(n) if c not in set(piv)]
        self.perm = np.array(info + piv)
        self.H = H[:, self.perm]           # same code, permuted columns
        self.n = n
        self.k = len(info)
        Hr = Hr[:, self.perm]
        # In permuted order Hr = [A | I]; parity = A @ info (mod 2)
        self._A = Hr[:, : self.k]
        self.rows_of_col = [np.flatnonzero(self.H[:, j]) for j in range(n)]
        self._check_idx = np.array([np.flatnonzero(r) for r in self.H])  # (m, dc)
        self.dv = dv

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        u = np.asarray(bits, dtype=np.uint8).reshape(-1, self.k)
        parity = (u.astype(np.int64) @ self._A.T.astype(np.int64)) % 2
        return np.concatenate([u, parity.astype(np.uint8)], axis=1).reshape(-1)

    def syndrome(self, code: np.ndarray) -> np.ndarray:
        c = np.asarray(code, dtype=np.uint8).reshape(-1, self.n)
        return c[:, self._check_idx].sum(axis=2) % 2

    def decode(self, code: np.ndarray, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Gallager-B style bit flipping. Returns (info bits, converged flags)."""
        c = np.asarray(code, dtype=np.uint8).reshape(-1, self.n).copy()
        Hint = self.H.astype(np.int32)
        for _ in range(max_iter):
            s = c[:, self._check_idx].sum(axis=2) % 2            # (B, m)
            ok = ~s.any(axis=1)
            if ok.all():
                break
            unsat = s.astype(np.int32) @ Hint                     # (B, n)
            # flip bits involved in the most unsatisfied checks (majority rule)
            thresh = np.maximum(unsat.max(axis=1, keepdims=True), (self.dv + 1) // 2)
            flip = (unsat >= thresh) & (unsat > 0) & ~ok[:, None]
            c ^= flip.astype(np.uint8)
        s = c[:, self._check_idx].sum(axis=2) % 2
        return c[:, : self.k].reshape(-1), ~s.any(axis=1)


def interleave(bits: np.ndarray, seed: int = 7) -> np.ndarray:
    return bits[np.random.default_rng(seed).permutation(bits.size)]


def deinterleave(bits: np.ndarray, seed: int = 7) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(bits.size)
    out = np.empty_like(bits)
    out[perm] = bits
    return out
