"""Host-side register driver and modem controller.

``RegisterDriver`` turns register/field accesses into CRC-protected bus
frames, retries on transport errors and maps device status codes to typed
exceptions. ``ModemController`` builds bring-up, command and telemetry
flows on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import bus
from .regmap import (CHIP_ID_VALUE, CMD_CLEAR_COUNTERS, CMD_RUN_FRAME, CMD_SELF_TEST, DSP_REGMAP, Access,
                     RegisterMap)
from .sequencer import Expect, PollUntil, Sequence, Write


class AsicError(Exception):
    """Base class for interface errors."""


class BusError(AsicError):
    """Transport failed after all retries."""


class AddressError(AsicError):
    pass


class AccessViolation(AsicError):
    pass


class CommandError(AsicError):
    def __init__(self, opcode: int, code: int):
        super().__init__(f"command {opcode:#04x} failed with error code {code}")
        self.opcode = opcode
        self.code = code


class CommandTimeout(AsicError):
    pass


class RegisterDriver:
    def __init__(self, transport, regmap: RegisterMap = DSP_REGMAP, max_retries: int = 4):
        self.t = transport
        self.map = regmap
        self.max_retries = max_retries
        self.stats = {"reads": 0, "writes": 0, "retries": 0}

    def _addr(self, reg) -> int:
        if isinstance(reg, int):
            return reg
        return self.map[reg].addr

    def _transact(self, op: int, addr: int, data: int = 0, busy_ok: bool = False) -> int:
        frame = bus.encode_request(op, addr, data)
        last = "no response"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats["retries"] += 1
            resp = bus.decode_response(self.t.transfer(frame))
            if resp is None:
                last = "lost or corrupted response"
                continue
            status, value = resp
            if status == bus.Status.OK:
                return value
            if status == bus.Status.BAD_CRC:
                last = "request corrupted"
                continue
            if status == bus.Status.BUSY:
                if busy_ok and attempt:
                    # an earlier attempt was applied but its response was lost
                    return 0
                last = "device busy"
                continue
            if status == bus.Status.BAD_ADDR:
                raise AddressError(f"no register at {addr:#05x}")
            if status == bus.Status.ACCESS_VIOLATION:
                raise AccessViolation(f"register at {addr:#05x} is read-only")
            raise BusError(f"unexpected status {status:#x}")
        raise BusError(f"{'read' if op == bus.OP_READ else 'write'} {addr:#05x} failed: {last}")

    def read(self, reg) -> int:
        self.stats["reads"] += 1
        return self._transact(bus.OP_READ, self._addr(reg))

    def write(self, reg, value: int, busy_ok: bool = False) -> None:
        self.stats["writes"] += 1
        self._transact(bus.OP_WRITE, self._addr(reg), value, busy_ok)

    def read_field(self, reg: str, field: str) -> int:
        return self.map[reg].field(field).get(self.read(reg))

    def write_fields(self, reg: str, **fields: int) -> None:
        """Read-modify-write of one or more fields.

        W1C and WO bits are masked out of the read-back value so that the
        write does not accidentally clear pending interrupts or re-trigger
        strobes; only the fields named here are written with their value.
        """
        r = self.map[reg]
        value = self.read(reg) & ~r.mask_of(Access.W1C, Access.WO, Access.RO)
        for name, v in fields.items():
            value = r.field(name).put(value, v)
        self.write(reg, value)


@dataclass
class ModemConfig:
    modulation: str = "qpsk"      # "qpsk" | "16qam"
    cd_ps_nm: int = 17000
    cdc: bool = True
    cpr: bool = True
    timing: bool = True
    fec: bool = False
    cpr_window: int = 32
    bps_phases: int = 32
    irq_mask: int = 0xF


@dataclass
class Telemetry:
    frames: int
    freq_offset_hz: float
    pre_fec_errors: int
    pre_fec_bits: int
    post_fec_errors: int
    info_bits: int
    evm_rms: float
    snr_db: float
    foe_lock: bool
    frame_sync: bool

    @property
    def pre_fec_ber(self) -> float:
        return self.pre_fec_errors / self.pre_fec_bits if self.pre_fec_bits else float("nan")

    @property
    def post_fec_ber(self) -> float:
        return self.post_fec_errors / self.info_bits if self.info_bits else float("nan")


def _signed32(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


class ModemController:
    def __init__(self, driver: RegisterDriver, poll_limit: int = 50):
        self.drv = driver
        self.poll_limit = poll_limit
        self.trace: list[str] = []

    def bring_up_sequence(self, cfg: ModemConfig) -> Sequence:
        fmt = {"qpsk": 0, "16qam": 1}[cfg.modulation]
        return Sequence("bring-up", [
            Expect("CHIP_ID", "ID", CHIP_ID_VALUE),
            Write("CTRL", SOFT_RESET=1),
            Write("CD_COMP", PS_NM=cfg.cd_ps_nm),
            Write("CPR_CFG", WINDOW=cfg.cpr_window, BPS_PHASES=cfg.bps_phases),
            Write("CTRL", MOD_FMT=fmt, CDC_EN=int(cfg.cdc), CPR_EN=int(cfg.cpr), TR_EN=int(cfg.timing),
                  FEC_EN=int(cfg.fec), DSP_EN=1),
            PollUntil("STATUS", "DSP_READY", 1, limit=self.poll_limit),
            Write("IRQ_STATUS", raw=0xF),
            Write("IRQ_MASK", MASK=cfg.irq_mask),
        ])

    def bring_up(self, cfg: ModemConfig) -> None:
        self.bring_up_sequence(cfg).run(self.drv, self.trace)

    def command(self, opcode: int) -> None:
        r = self.drv.map["CMD"]
        word = r.field("GO").put(r.field("OPCODE").put(0, opcode), 1)
        self.drv.write("CMD", word, busy_ok=True)
        for _ in range(self.poll_limit):
            st = self.drv.read("CMD_STATUS")
            f = self.drv.map["CMD_STATUS"]
            if f.field("DONE").get(st):
                if f.field("ERROR").get(st):
                    self.drv.write("IRQ_STATUS", self.drv.map["IRQ_STATUS"].field("CMD_ERR").mask)
                    raise CommandError(opcode, f.field("ERR_CODE").get(st))
                return
        raise CommandTimeout(f"command {opcode:#04x} not done after {self.poll_limit} polls")

    def run_frames(self, n: int) -> Telemetry:
        for _ in range(n):
            self.command(CMD_RUN_FRAME)
        irq = self.drv.read("IRQ_STATUS")
        if irq:
            self.drv.write("IRQ_STATUS", irq)  # acknowledge (W1C)
        return self.telemetry()

    def clear_counters(self) -> None:
        self.command(CMD_CLEAR_COUNTERS)

    def self_test(self) -> None:
        self.command(CMD_SELF_TEST)

    def telemetry(self) -> Telemetry:
        d = self.drv
        status = d.read("STATUS")
        st = d.map["STATUS"]
        bits = d.read("TM_BITS_LO") | (d.read("TM_BITS_HI") << 32)
        info = d.read("TM_INFO_BITS_LO") | (d.read("TM_INFO_BITS_HI") << 32)
        return Telemetry(
            frames=d.read("TM_FRAMES"),
            freq_offset_hz=_signed32(d.read("TM_FOE")) * 1e3,
            pre_fec_errors=d.read("TM_PRE_FEC_ERR"),
            pre_fec_bits=bits,
            post_fec_errors=d.read("TM_POST_FEC_ERR"),
            info_bits=info,
            evm_rms=d.read("TM_EVM") / 1000,
            snr_db=d.read("TM_SNR") / 100,
            foe_lock=bool(st.field("FOE_LOCK").get(status)),
            frame_sync=bool(st.field("FRAME_SYNC").get(status)),
        )
