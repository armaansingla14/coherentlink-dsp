"""Behavioural model of the modem DSP ASIC behind the register bus.

The model enforces the register access policies, executes commands
asynchronously (a command completes after a configurable number of bus
transactions, emulating time passing while firmware polls), and runs the
actual Python DSP chain for RUN_FRAME so that telemetry reflects real
receiver behaviour on the configured channel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .. import fec, link
from ..channel import LinkParams
from . import bus
from .regmap import (CMD_CLEAR_COUNTERS, CMD_NOP, CMD_RUN_FRAME, CMD_SELF_TEST, DSP_REGMAP,
                     ERR_BAD_OPCODE, ERR_NO_SIGNAL, ERR_NOT_READY, ERR_SELF_TEST, Access,
                     RegisterMap)


@dataclass
class Environment:
    """Optical channel seen by the device (set by the test harness)."""
    link: LinkParams
    signal_present: bool = True


class DspAsicModel:
    def __init__(self, env: Environment, regmap: RegisterMap = DSP_REGMAP, frame_symbols: int = 1 << 14,
                 cmd_latency: int = 3, ready_latency: int = 5, seed: int = 0):
        self.map = regmap
        self.env = env
        self.frame_symbols = frame_symbols
        self.cmd_latency = cmd_latency
        self.ready_latency = ready_latency
        self.rng = np.random.default_rng(seed)
        self._ldpc = None
        self.reset()

    # ---- register storage -------------------------------------------------
    def reset(self):
        self.regs = {r.addr: r.reset for r in self.map}
        self._pending = None          # (opcode, remaining ticks)
        self._ready_countdown = None
        self.counters = dict(pre=0, post=0, bits=0, info_bits=0, frames=0)

    def _get(self, reg: str, fld: str) -> int:
        r = self.map[reg]
        return r.field(fld).get(self.regs[r.addr])

    def _set(self, reg: str, fld: str, value: int):
        r = self.map[reg]
        self.regs[r.addr] = r.field(fld).put(self.regs[r.addr], value)

    def _raw_set(self, reg: str, value: int):
        self.regs[self.map[reg].addr] = value & 0xFFFFFFFF

    # ---- bus side ------------------------------------------------------------
    def handle_frame(self, frame: bytes) -> bytes:
        self._tick()
        req = bus.decode_request(frame)
        if req is None:
            return bus.encode_response(bus.Status.BAD_CRC)
        op, addr, data = req
        reg = self.map.at(addr)
        if reg is None:
            return bus.encode_response(bus.Status.BAD_ADDR)
        if op == bus.OP_READ:
            return bus.encode_response(bus.Status.OK, self.read(addr))
        if op == bus.OP_WRITE:
            return bus.encode_response(self.write(addr, data))
        return bus.encode_response(bus.Status.BAD_OP)

    def read(self, addr: int) -> int:
        reg = self.map.at(addr)
        return self.regs[addr] & ~reg.mask_of(Access.WO)

    def write(self, addr: int, data: int) -> int:
        reg = self.map.at(addr)
        if not reg.writable:
            return bus.Status.ACCESS_VIOLATION
        if reg.name == "CMD" and self._pending is not None:
            return bus.Status.BUSY
        wo = data & reg.mask_of(Access.WO)
        if reg.name == "CTRL" and wo & reg.field("SOFT_RESET").mask:
            self.reset()               # reset wins over any other bits written
            return bus.Status.OK
        cur = self.regs[addr]
        rw = reg.mask_of(Access.RW)
        w1c = reg.mask_of(Access.W1C)
        new = (cur & ~rw) | (data & rw)       # RO bits keep their value
        new &= ~(data & w1c)                  # write-one-to-clear
        self.regs[addr] = new
        if wo:
            self._on_strobe(reg.name, wo)
        if reg.name == "CTRL":
            self._on_ctrl(cur, new)
        return bus.Status.OK

    # ---- behaviour -------------------------------------------------------------
    def _on_ctrl(self, old: int, new: int):
        f = self.map["CTRL"].field("DSP_EN")
        if f.get(new) and not f.get(old):
            self._ready_countdown = self.ready_latency
        if not f.get(new):
            self._set("STATUS", "DSP_READY", 0)
            self._ready_countdown = None

    def _on_strobe(self, reg: str, bits: int):
        if reg == "CMD" and bits & self.map["CMD"].field("GO").mask:
            self._pending = [self._get("CMD", "OPCODE"), self.cmd_latency]
            self._set("CMD_STATUS", "DONE", 0)
            self._set("CMD_STATUS", "ERROR", 0)
            self._set("CMD_STATUS", "ERR_CODE", 0)
            self._set("STATUS", "BUSY", 1)

    def _tick(self):
        if self._ready_countdown is not None:
            self._ready_countdown -= 1
            if self._ready_countdown <= 0:
                self._set("STATUS", "DSP_READY", 1)
                self._ready_countdown = None
        if self._pending is not None:
            self._pending[1] -= 1
            if self._pending[1] <= 0:
                opcode = self._pending[0]
                self._pending = None
                err = self._execute(opcode)
                self._set("STATUS", "BUSY", 0)
                self._set("CMD_STATUS", "DONE", 1)
                if err:
                    self._set("CMD_STATUS", "ERROR", 1)
                    self._set("CMD_STATUS", "ERR_CODE", err)
                    self._set("IRQ_STATUS", "CMD_ERR", 1)

    def _execute(self, opcode: int) -> int:
        if opcode == CMD_NOP:
            return 0
        if opcode == CMD_CLEAR_COUNTERS:
            self.counters = dict(pre=0, post=0, bits=0, info_bits=0, frames=0)
            self._publish()
            return 0
        if opcode == CMD_SELF_TEST:
            p = replace(self.env.link, osnr_db=40.0, length_km=1.0, freq_offset_hz=0.0, linewidth_hz=0.0)
            r = link.simulate_frame(p, "qpsk", 1 << 12, link.RxConfig(), self.rng)
            return 0 if r.pre_errors == 0 else ERR_SELF_TEST
        if opcode == CMD_RUN_FRAME:
            if not (self._get("CTRL", "DSP_EN") and self._get("STATUS", "DSP_READY")):
                return ERR_NOT_READY
            if not self.env.signal_present:
                self._set("IRQ_STATUS", "LOS", 1)
                self._set("STATUS", "FOE_LOCK", 0)
                self._set("STATUS", "FRAME_SYNC", 0)
                return ERR_NO_SIGNAL
            self._run_frame()
            return 0
        return ERR_BAD_OPCODE

    def _rx_params(self) -> LinkParams:
        """Receiver-side parameters derived from register configuration."""
        p = self.env.link
        ps_nm = self._get("CD_COMP", "PS_NM")
        return replace(p, length_km=max(ps_nm / p.dispersion_ps_nm_km, 1e-3))

    def _run_frame(self):
        fmt = "16qam" if self._get("CTRL", "MOD_FMT") == 1 else "qpsk"
        cfg = link.RxConfig(cdc=bool(self._get("CTRL", "CDC_EN")), cpr=bool(self._get("CTRL", "CPR_EN")),
                            timing=bool(self._get("CTRL", "TR_EN")),
                            cpr_window=max(1, self._get("CPR_CFG", "WINDOW")),
                            bps_test_phases=max(4, self._get("CPR_CFG", "BPS_PHASES")))
        code = None
        if self._get("CTRL", "FEC_EN"):
            if self._ldpc is None:
                self._ldpc = fec.LdpcCode()
            code = self._ldpc
        r = link.simulate_frame(self.env.link, fmt, self.frame_symbols, cfg, self.rng, code=code,
                                rx_params=self._rx_params())
        c = self.counters
        c["pre"] += r.pre_errors
        c["bits"] += r.pre_bits
        c["post"] += r.post_errors
        c["info_bits"] += r.post_bits
        c["frames"] += 1
        evm = max(r.evm_rms, 1e-6)
        self._raw_set("TM_FOE", int(round(r.est_freq_offset / 1e3)) & 0xFFFFFFFF)
        self._set("TM_EVM", "PERMILLE", min(int(round(evm * 1000)), 0xFFFF))
        snr = max(0.0, -20 * np.log10(evm))
        self._set("TM_SNR", "CENTI_DB", min(int(round(snr * 100)), 0xFFFF))
        locked = 1 if evm < 0.6 else 0
        self._set("STATUS", "FOE_LOCK", locked)
        self._set("STATUS", "FRAME_SYNC", locked)
        self._publish()
        self._set("IRQ_STATUS", "FRAME_DONE", 1)
        if code is not None and r.post_errors:
            self._set("IRQ_STATUS", "FEC_FAIL", 1)

    def _publish(self):
        c = self.counters
        sat = lambda v: min(v, 0xFFFFFFFF)  # noqa: E731
        self._raw_set("TM_PRE_FEC_ERR", sat(c["pre"]))
        self._raw_set("TM_POST_FEC_ERR", sat(c["post"]))
        self._raw_set("TM_BITS_LO", c["bits"] & 0xFFFFFFFF)
        self._raw_set("TM_BITS_HI", (c["bits"] >> 32) & 0xFFFFFFFF)
        self._raw_set("TM_INFO_BITS_LO", c["info_bits"] & 0xFFFFFFFF)
        self._raw_set("TM_INFO_BITS_HI", (c["info_bits"] >> 32) & 0xFFFFFFFF)
        self._raw_set("TM_FRAMES", sat(c["frames"]))
