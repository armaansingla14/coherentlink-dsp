from dataclasses import replace

import pytest

from coherentlink.asic import bus, device, driver, regmap
from coherentlink.asic.regmap import Access, Field, Register, RegisterMap
from coherentlink.asic.sequencer import Expect, PollUntil, Sequence, SequenceError, Write
from coherentlink.channel import LinkParams


def make(osnr=14.0, ber=0.0, drop=0.0, seed=0, **dev_kw):
    env = device.Environment(replace(LinkParams(), osnr_db=osnr))
    dev = device.DspAsicModel(env, frame_symbols=1 << 13, seed=seed, **dev_kw)
    link = bus.FaultyLink(dev, bit_error_rate=ber, drop_rate=drop, seed=seed)
    drv = driver.RegisterDriver(link)
    return env, dev, drv, driver.ModemController(drv)


# ---- register map ---------------------------------------------------------------

def test_crc16_ccitt_false_check_value():
    assert bus.crc16(b"123456789") == 0x29B1


def test_frame_roundtrip_and_crc_detection():
    f = bus.encode_request(bus.OP_WRITE, 0x08, 0xDEADBEEF)
    assert bus.decode_request(f) == (bus.OP_WRITE, 0x08, 0xDEADBEEF)
    bad = bytes([f[0] ^ 0x10]) + f[1:]
    assert bus.decode_request(bad) is None
    assert bus.decode_response(None) is None


def test_regmap_rejects_overlaps_and_misalignment():
    with pytest.raises(ValueError):
        Register("X", 0, (Field("A", 0, 4, Access.RW), Field("B", 3, 2, Access.RW)))
    with pytest.raises(ValueError):
        RegisterMap((Register("X", 2, ()),))
    with pytest.raises(ValueError):
        RegisterMap((Register("X", 0, ()), Register("Y", 0, ())))


def test_field_put_get_and_range_check():
    f = regmap.DSP_REGMAP["CTRL"].field("MOD_FMT")
    w = f.put(0, 1)
    assert f.get(w) == 1 and w == 0x100
    with pytest.raises(ValueError):
        f.put(0, 4)


def test_reset_values_visible_over_bus():
    _, _, drv, _ = make()
    assert drv.read("CHIP_ID") == regmap.CHIP_ID_VALUE
    assert drv.read_field("CD_COMP", "PS_NM") == 17000
    assert drv.read_field("VERSION", "MAJOR") == 1


# ---- access policies --------------------------------------------------------------

def test_read_only_register_rejects_writes():
    _, _, drv, _ = make()
    with pytest.raises(driver.AccessViolation):
        drv.write("CHIP_ID", 0)
    assert drv.read("CHIP_ID") == regmap.CHIP_ID_VALUE


def test_unknown_address_raises():
    _, _, drv, _ = make()
    with pytest.raises(driver.AddressError):
        drv.read(0x3FC)


def test_fully_read_only_status_register_is_protected():
    _, dev, drv, _ = make()
    dev._set("CMD_STATUS", "DONE", 1)
    with pytest.raises(driver.AccessViolation):
        drv.write("CMD_STATUS", 0)
    assert drv.read_field("CMD_STATUS", "DONE") == 1


def test_w1c_clears_only_written_bits():
    _, dev, drv, _ = make()
    dev._raw_set("IRQ_STATUS", 0b1011)
    drv.write("IRQ_STATUS", 0b0001)
    assert drv.read("IRQ_STATUS") == 0b1010


def test_rmw_does_not_clear_pending_interrupts_or_strobe():
    _, dev, drv, _ = make()
    dev._raw_set("IRQ_STATUS", 0b0101)
    drv.write_fields("CTRL", FEC_EN=1)          # unrelated RMW
    assert drv.read("IRQ_STATUS") == 0b0101
    assert drv.read_field("CTRL", "SOFT_RESET") == 0  # WO reads back as 0


def test_soft_reset_restores_defaults():
    _, _, drv, _ = make()
    drv.write_fields("CD_COMP", PS_NM=1234)
    drv.write_fields("CTRL", SOFT_RESET=1)
    assert drv.read_field("CD_COMP", "PS_NM") == 17000


# ---- transport errors ---------------------------------------------------------------

def test_driver_recovers_from_corruption_and_drops():
    _, _, drv, _ = make(ber=2e-3, drop=0.05, seed=3)
    drv.max_retries = 8   # ~30 % of transactions see an error at this rate
    for i in range(200):
        drv.write_fields("CD_COMP", PS_NM=i)
        assert drv.read_field("CD_COMP", "PS_NM") == i
    assert drv.stats["retries"] > 0
    assert drv.t.stats["corrupted"] > 0 and drv.t.stats["dropped"] > 0


def test_driver_gives_up_on_dead_link():
    _, _, drv, _ = make(drop=1.0)
    with pytest.raises(driver.BusError):
        drv.read("CHIP_ID")


# ---- sequencing and commands -----------------------------------------------------------

def test_bring_up_sequence_and_trace():
    _, _, drv, m = make()
    m.bring_up(driver.ModemConfig(modulation="16qam", fec=True))
    assert drv.read_field("STATUS", "DSP_READY") == 1
    assert drv.read_field("CTRL", "MOD_FMT") == 1
    assert drv.read_field("CTRL", "FEC_EN") == 1
    assert len(m.trace) == 8 and m.trace[0].startswith("bring-up[0] expect CHIP_ID")


def test_sequence_reports_failing_step():
    _, _, drv, _ = make()
    seq = Sequence("bad", [Write("CD_COMP", PS_NM=5), Expect("CD_COMP", "PS_NM", 6)])
    with pytest.raises(SequenceError) as e:
        seq.run(drv)
    assert e.value.index == 1


def test_poll_timeout_when_dsp_never_ready():
    _, _, drv, _ = make(ready_latency=10_000)
    drv.write_fields("CTRL", DSP_EN=1)
    with pytest.raises(SequenceError, match="timeout"):
        Sequence("wait", [PollUntil("STATUS", "DSP_READY", 1, limit=20)]).run(drv)


def test_run_frame_before_enable_is_rejected():
    _, _, drv, m = make()
    with pytest.raises(driver.CommandError) as e:
        m.command(regmap.CMD_RUN_FRAME)
    assert e.value.code == regmap.ERR_NOT_READY
    assert drv.read_field("IRQ_STATUS", "CMD_ERR") == 0  # acknowledged by the controller


def test_bad_opcode_and_busy():
    _, dev, drv, m = make(cmd_latency=5)
    with pytest.raises(driver.CommandError) as e:
        m.command(0x7E)
    assert e.value.code == regmap.ERR_BAD_OPCODE
    drv.write("CMD", (1 << 31) | regmap.CMD_NOP)
    frame = bus.encode_request(bus.OP_WRITE, regmap.DSP_REGMAP["CMD"].addr, (1 << 31))
    assert bus.decode_response(dev.handle_frame(frame))[0] == bus.Status.BUSY


def test_command_timeout():
    _, _, _, m = make(cmd_latency=1000)
    m.poll_limit = 10
    with pytest.raises(driver.CommandTimeout):
        m.command(regmap.CMD_NOP)


def test_loss_of_signal_reported():
    env, _, drv, m = make()
    m.bring_up(driver.ModemConfig())
    env.signal_present = False
    with pytest.raises(driver.CommandError) as e:
        m.command(regmap.CMD_RUN_FRAME)
    assert e.value.code == regmap.ERR_NO_SIGNAL
    assert drv.read_field("IRQ_STATUS", "LOS") == 1


def test_telemetry_from_running_frames():
    _, _, drv, m = make(osnr=13.0, ber=1e-3, seed=4)
    m.bring_up(driver.ModemConfig())
    m.self_test()
    tm = m.run_frames(2)
    assert tm.frames == 2 and tm.foe_lock and tm.frame_sync
    assert abs(tm.freq_offset_hz - 1e9) < 5e6
    assert tm.pre_fec_bits > 0 and tm.pre_fec_ber < 1e-2
    assert 8 < tm.snr_db < 16
    assert drv.read("IRQ_STATUS") == 0  # FRAME_DONE acknowledged
    m.clear_counters()
    assert m.telemetry().frames == 0


def test_misconfigured_dispersion_is_visible_in_telemetry():
    _, _, _, m = make(osnr=16.0)
    m.bring_up(driver.ModemConfig(cd_ps_nm=2000))  # real link has 17000 ps/nm
    tm = m.run_frames(1)
    assert tm.pre_fec_ber > 0.1
    assert not tm.frame_sync


def test_fec_enabled_counts_info_bits():
    _, _, _, m = make(osnr=10.0)
    m.bring_up(driver.ModemConfig(fec=True))
    tm = m.run_frames(1)
    assert 0 < tm.info_bits < tm.pre_fec_bits
    assert tm.post_fec_ber <= tm.pre_fec_ber
