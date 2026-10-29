"""Declarative register command sequences.

A ``Sequence`` is a list of steps (field writes, expected reads, polls) that
a driver executes in order. Each executed step is appended to a trace, and a
failure raises ``SequenceError`` naming the step, which keeps bring-up flows
readable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SequenceError(Exception):
    def __init__(self, seq: str, index: int, step, reason: str):
        super().__init__(f"{seq}: step {index} ({step}) failed: {reason}")
        self.index = index
        self.step = step
        self.reason = reason


@dataclass
class Write:
    reg: str
    raw: int | None = None
    fields: dict = field(default_factory=dict)

    def __init__(self, reg: str, raw: int | None = None, **fields: int):
        self.reg, self.raw, self.fields = reg, raw, fields

    def run(self, drv):
        if self.raw is not None:
            drv.write(self.reg, self.raw)
        else:
            drv.write_fields(self.reg, **self.fields)

    def __str__(self):
        what = f"{self.raw:#x}" if self.raw is not None else ", ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"write {self.reg} {what}"


@dataclass
class Expect:
    reg: str
    field_name: str
    value: int

    def run(self, drv):
        got = drv.read_field(self.reg, self.field_name)
        if got != self.value:
            return f"read {got:#x}, expected {self.value:#x}"
        return None

    def __str__(self):
        return f"expect {self.reg}.{self.field_name} == {self.value:#x}"


@dataclass
class PollUntil:
    reg: str
    field_name: str
    value: int
    limit: int = 50

    def run(self, drv):
        for _ in range(self.limit):
            if drv.read_field(self.reg, self.field_name) == self.value:
                return None
        return f"timeout after {self.limit} polls"

    def __str__(self):
        return f"poll {self.reg}.{self.field_name} == {self.value}"


@dataclass
class Sequence:
    name: str
    steps: list

    def run(self, drv, trace: list | None = None):
        for i, step in enumerate(self.steps):
            try:
                reason = step.run(drv)
            except Exception as exc:  # driver errors carry the context we need
                raise SequenceError(self.name, i, step, f"{type(exc).__name__}: {exc}") from exc
            if trace is not None:
                trace.append(f"{self.name}[{i}] {step}" + (f" -> {reason}" if reason else ""))
            if reason:
                raise SequenceError(self.name, i, step, reason)
