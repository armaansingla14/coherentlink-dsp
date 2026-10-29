"""Cross-checks between the Python models and the C++ library.

Skipped when cpp/build/clink_tool has not been built (scripts/build_cpp.sh).
"""

from dataclasses import replace

import numpy as np
import pytest

from coherentlink import cpp_bridge, fixedpoint, pulse
from coherentlink.asic import bus, device, regmap
from coherentlink.channel import LinkParams

pytestmark = pytest.mark.skipif(not cpp_bridge.available(), reason="C++ tool not built")


def test_crc_matches():
    rng = np.random.default_rng(0)
    for n in (1, 7, 9, 64):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        assert cpp_bridge.crc16(data) == bus.crc16(data)


def test_fir_q15_bit_exact():
    rng = np.random.default_rng(1)
    taps = fixedpoint.to_q15(pulse.rrc_taps(0.1, 2, 16) * 0.9)
    x = fixedpoint.to_q15(np.clip(rng.normal(0, 0.3, 3000), -1, 0.999))
    py, _ = fixedpoint.fir_q15(taps, x)
    np.testing.assert_array_equal(cpp_bridge.fir_q15(taps, x), py)


def test_fir_q15_saturation_bit_exact():
    taps = np.full(8, 30000, dtype=np.int16)
    x = np.array([32767, -32768, 32767, 32767, -1, 0, 32767] * 20, dtype=np.int16)
    py, sat = fixedpoint.fir_q15(taps, x)
    assert sat > 0
    np.testing.assert_array_equal(cpp_bridge.fir_q15(taps, x), py)


def test_bus_slave_matches_python_device():
    """Random reads/writes (and corrupted frames) produce identical responses
    from the C++ register file and the Python device model. Side-effect bits
    (soft reset, DSP enable, command GO) are masked since only the Python
    model implements the behaviour behind them."""
    rng = np.random.default_rng(2)
    addrs = [r.addr for r in regmap.DSP_REGMAP] + [0x0FC, 0x200]
    avoid = {0x008: 0x3, 0x020: 1 << 31}
    frames = []
    for _ in range(400):
        addr = int(rng.choice(addrs))
        op = bus.OP_READ if rng.random() < 0.5 else bus.OP_WRITE
        data = int(rng.integers(0, 2 ** 32)) & ~avoid.get(addr, 0)
        f = bus.encode_request(op, addr, data)
        if rng.random() < 0.05:
            f = bytes([f[0] ^ 0x04]) + f[1:]
        frames.append(f)
    dev = device.DspAsicModel(device.Environment(replace(LinkParams())))
    py = [dev.handle_frame(f) for f in frames]
    assert cpp_bridge.bus(frames) == py
