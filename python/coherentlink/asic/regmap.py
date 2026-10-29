"""Register map of the modem DSP ASIC model.

Registers are 32 bits wide and word aligned. Each field carries an access
policy:

    RW   read/write
    RO   read-only (writes are rejected with ACCESS_VIOLATION)
    W1C  write-one-to-clear (interrupt status)
    WO   write-only, self-clearing (reads return 0), e.g. soft reset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Access(str, Enum):
    RW = "RW"
    RO = "RO"
    W1C = "W1C"
    WO = "WO"


@dataclass(frozen=True)
class Field:
    name: str
    lsb: int
    width: int
    access: Access
    reset: int = 0
    desc: str = ""

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.lsb

    def get(self, word: int) -> int:
        return (word & self.mask) >> self.lsb

    def put(self, word: int, value: int) -> int:
        if value < 0 or value >= (1 << self.width):
            raise ValueError(f"{self.name}: value {value} does not fit in {self.width} bits")
        return (word & ~self.mask) | (value << self.lsb)


@dataclass(frozen=True)
class Register:
    name: str
    addr: int
    fields: tuple[Field, ...]
    desc: str = ""

    def __post_init__(self):
        used = 0
        for f in self.fields:
            if f.lsb + f.width > 32:
                raise ValueError(f"{self.name}.{f.name} exceeds 32 bits")
            if used & f.mask:
                raise ValueError(f"{self.name}.{f.name} overlaps another field")
            used |= f.mask

    @property
    def reset(self) -> int:
        w = 0
        for f in self.fields:
            w = f.put(w, f.reset)
        return w

    def mask_of(self, *access: Access) -> int:
        m = 0
        for f in self.fields:
            if f.access in access:
                m |= f.mask
        return m

    @property
    def writable(self) -> bool:
        return any(f.access != Access.RO for f in self.fields)

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name} has no field {name}")


@dataclass
class RegisterMap:
    registers: tuple[Register, ...]
    _by_name: dict = field(init=False, repr=False)
    _by_addr: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._by_name = {}
        self._by_addr = {}
        for r in self.registers:
            if r.addr % 4:
                raise ValueError(f"{r.name}: address not word aligned")
            if r.addr in self._by_addr or r.name in self._by_name:
                raise ValueError(f"duplicate register {r.name} @ {r.addr:#x}")
            self._by_addr[r.addr] = r
            self._by_name[r.name] = r

    def __getitem__(self, name: str) -> Register:
        return self._by_name[name]

    def at(self, addr: int) -> Register | None:
        return self._by_addr.get(addr)

    def __iter__(self):
        return iter(self.registers)


def _f(name, lsb, width, access, reset=0, desc=""):
    return Field(name, lsb, width, Access(access), reset, desc)


CHIP_ID_VALUE = 0xC0DE0516

# Command opcodes for CMD.OPCODE
CMD_NOP = 0x00
CMD_RUN_FRAME = 0x01
CMD_CLEAR_COUNTERS = 0x02
CMD_SELF_TEST = 0x03

# Command error codes reported in CMD_STATUS.ERR_CODE
ERR_NONE = 0
ERR_BAD_OPCODE = 1
ERR_NOT_READY = 2
ERR_NO_SIGNAL = 3
ERR_SELF_TEST = 4

DSP_REGMAP = RegisterMap((
    Register("CHIP_ID", 0x000, (_f("ID", 0, 32, "RO", CHIP_ID_VALUE),), "Device identifier"),
    Register("VERSION", 0x004, (_f("MINOR", 0, 8, "RO", 3), _f("MAJOR", 8, 8, "RO", 1))),
    Register("CTRL", 0x008, (
        _f("SOFT_RESET", 0, 1, "WO", desc="write 1 to reset all registers"),
        _f("DSP_EN", 1, 1, "RW"),
        _f("CDC_EN", 2, 1, "RW", 1),
        _f("CPR_EN", 3, 1, "RW", 1),
        _f("FEC_EN", 4, 1, "RW", 0),
        _f("TR_EN", 5, 1, "RW", 1, "timing recovery"),
        _f("MOD_FMT", 8, 2, "RW", 0, "0 = QPSK, 1 = 16-QAM"),
    )),
    Register("CD_COMP", 0x00C, (_f("PS_NM", 0, 24, "RW", 17000, "accumulated dispersion [ps/nm]"),)),
    Register("CPR_CFG", 0x010, (_f("WINDOW", 0, 8, "RW", 32), _f("BPS_PHASES", 8, 8, "RW", 32))),
    Register("STATUS", 0x014, (
        _f("DSP_READY", 0, 1, "RO"), _f("FOE_LOCK", 1, 1, "RO"), _f("FRAME_SYNC", 2, 1, "RO"),
        _f("BUSY", 3, 1, "RO"),
    )),
    Register("IRQ_STATUS", 0x018, (
        _f("FRAME_DONE", 0, 1, "W1C"), _f("LOS", 1, 1, "W1C", desc="loss of signal"),
        _f("FEC_FAIL", 2, 1, "W1C"), _f("CMD_ERR", 3, 1, "W1C"),
    )),
    Register("IRQ_MASK", 0x01C, (_f("MASK", 0, 4, "RW", 0xF),)),
    Register("CMD", 0x020, (_f("OPCODE", 0, 8, "RW"), _f("GO", 31, 1, "WO"))),
    Register("CMD_STATUS", 0x024, (_f("DONE", 0, 1, "RO"), _f("ERROR", 1, 1, "RO"), _f("ERR_CODE", 8, 8, "RO"))),
    Register("TM_FOE", 0x030, (_f("KHZ", 0, 32, "RO", desc="signed frequency offset estimate [kHz]"),)),
    Register("TM_PRE_FEC_ERR", 0x034, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_POST_FEC_ERR", 0x038, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_BITS_LO", 0x03C, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_BITS_HI", 0x040, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_EVM", 0x044, (_f("PERMILLE", 0, 16, "RO", desc="RMS EVM in 0.1 %"),)),
    Register("TM_FRAMES", 0x048, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_SNR", 0x04C, (_f("CENTI_DB", 0, 16, "RO", desc="SNR estimate from EVM in 0.01 dB"),)),
    Register("TM_INFO_BITS_LO", 0x050, (_f("COUNT", 0, 32, "RO"),)),
    Register("TM_INFO_BITS_HI", 0x054, (_f("COUNT", 0, 32, "RO"),)),
))
