"""Framed register-bus protocol with CRC-16 protection.

Request (9 bytes):   op | addr[15:8] | addr[7:0] | data[31:0] (big endian) | crc16
Response (7 bytes):  status | data[31:0] | crc16

The CRC is CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over all preceding
bytes. ``FaultyLink`` sits between driver and device and can corrupt or drop
frames to exercise the driver's error handling.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

OP_READ = 0x52   # 'R'
OP_WRITE = 0x57  # 'W'


class Status(IntEnum):
    OK = 0x00
    BAD_CRC = 0x01
    BAD_ADDR = 0x02
    ACCESS_VIOLATION = 0x03
    BAD_OP = 0x04
    BUSY = 0x05


def _crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


_TABLE = _crc_table()


def crc16(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


def _with_crc(body: bytes) -> bytes:
    return body + crc16(body).to_bytes(2, "big")


def encode_request(op: int, addr: int, data: int = 0) -> bytes:
    if not 0 <= addr <= 0xFFFF:
        raise ValueError("address out of range")
    return _with_crc(bytes([op]) + addr.to_bytes(2, "big") + (data & 0xFFFFFFFF).to_bytes(4, "big"))


def decode_request(frame: bytes):
    """Returns (op, addr, data) or None if the CRC/length is wrong."""
    if len(frame) != 9 or crc16(frame[:7]) != int.from_bytes(frame[7:], "big"):
        return None
    return frame[0], int.from_bytes(frame[1:3], "big"), int.from_bytes(frame[3:7], "big")


def encode_response(status: int, data: int = 0) -> bytes:
    return _with_crc(bytes([status]) + (data & 0xFFFFFFFF).to_bytes(4, "big"))


def decode_response(frame: bytes):
    """Returns (status, data) or None if the CRC/length is wrong."""
    if frame is None or len(frame) != 7 or crc16(frame[:5]) != int.from_bytes(frame[5:], "big"):
        return None
    return frame[0], int.from_bytes(frame[1:5], "big")


class FaultyLink:
    """Transport with optional bit-error and frame-drop injection."""

    def __init__(self, device, bit_error_rate: float = 0.0, drop_rate: float = 0.0, seed: int = 0):
        self.device = device
        self.ber = bit_error_rate
        self.drop = drop_rate
        self.rng = np.random.default_rng(seed)
        self.stats = {"frames": 0, "corrupted": 0, "dropped": 0}

    def _corrupt(self, frame: bytes) -> bytes:
        if self.ber <= 0:
            return frame
        bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
        flips = self.rng.random(bits.size) < self.ber
        if flips.any():
            self.stats["corrupted"] += 1
            bits ^= flips.astype(np.uint8)
        return np.packbits(bits).tobytes()

    def transfer(self, request: bytes) -> bytes | None:
        self.stats["frames"] += 1
        if self.rng.random() < self.drop:
            self.stats["dropped"] += 1
            return None
        response = self.device.handle_frame(self._corrupt(request))
        return self._corrupt(response)
