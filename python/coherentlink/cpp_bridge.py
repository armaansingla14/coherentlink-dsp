"""Wrappers around cpp/build/clink_tool for cross-language checks."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[2] / "cpp" / "build" / "clink_tool"


def available() -> bool:
    return TOOL.exists()


def crc16(data: bytes) -> int:
    out = subprocess.run([str(TOOL), "crc", data.hex()], check=True, capture_output=True, text=True)
    return int(out.stdout.strip(), 16)


def fir_q15(taps: np.ndarray, x: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        t, i, o = (Path(td) / n for n in ("taps.txt", "in.txt", "out.txt"))
        np.savetxt(t, taps.astype(int), fmt="%d")
        np.savetxt(i, x.astype(int), fmt="%d")
        subprocess.run([str(TOOL), "fir", str(t), str(i), str(o)], check=True, capture_output=True)
        return np.loadtxt(o, dtype=np.int64).astype(np.int16)


def bus(frames: list[bytes]) -> list[bytes]:
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "req.txt", Path(td) / "resp.txt"
        i.write_text("\n".join(f.hex() for f in frames) + "\n")
        subprocess.run([str(TOOL), "bus", str(i), str(o)], check=True)
        return [bytes.fromhex(line) for line in o.read_text().split()]
