#!/usr/bin/env python
"""BER sweeps, DSP ablations, fixed-point and register-bus experiments.

Writes results/*.csv and results/summary.md. Everything is simulated.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from coherentlink import channel, fec, fixedpoint, link, modulation as M, pulse  # noqa: E402
from coherentlink.asic import bus, device, driver  # noqa: E402

RESULTS = ROOT / "results"
OSNR_GRID = {"qpsk": [8, 9, 10, 11, 12, 13], "16qam": [14, 15, 16, 17, 18, 19, 20]}
ABLATION_OSNR = {"qpsk": 12.0, "16qam": 19.0}


def write_csv(path: Path, rows: list[dict]):
    keys = list(rows[0])
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


def fmt_ber(errors: int, bits: int) -> str:
    if errors == 0:
        return f"0 (< {1 / bits:.1e})"
    return f"{errors / bits:.2e}"


def osnr_at_ber(osnr, ber, target):
    """Log-linear interpolation of the OSNR where BER crosses ``target``.

    Returns (value, text). If the next measured point has zero errors the
    crossing can only be bounded, and the text reads "<= x".
    """
    osnr, ber = np.asarray(osnr, float), np.asarray(ber, float)
    if ber[0] < target:
        return float("nan"), f"< {osnr[0]:g} (below grid)"
    for i in range(len(osnr) - 1):
        if ber[i] >= target > ber[i + 1]:
            if ber[i + 1] == 0:
                return float("nan"), f"<= {osnr[i + 1]:g}"
            a, b = np.log10(ber[i]), np.log10(ber[i + 1])
            v = osnr[i] + (np.log10(target) - a) / (b - a) * (osnr[i + 1] - osnr[i])
            return v, f"{v:.2f}"
    return float("nan"), f"> {osnr[-1]:g} (above grid)"


def ber_sweep(base, frames, nsym, seed, md):
    codes = {"hamming74": fec.Hamming74(), "ldpc_1008": fec.LdpcCode()}
    rows = []
    for fmt, grid in OSNR_GRID.items():
        for osnr in grid:
            p = replace(base, osnr_db=float(osnr))
            snr = float(M.osnr_to_snr_db(osnr, p.symbol_rate))
            theory = float(M.theoretical_ber(M.FORMATS[fmt], snr))
            for cname, code in codes.items():
                t0 = time.time()
                r = link.run_ber(p, fmt, frames, nsym, link.RxConfig(), seed=seed + osnr, code=code)
                rows.append(dict(format=fmt, osnr_db=float(osnr), snr_db=snr, code=cname,
                                 code_rate=code.rate, symbols=r["symbols"],
                                 pre_fec_errors=r["errors"], pre_fec_bits=r["bits"], pre_fec_ber=r["ber"],
                                 post_fec_errors=r["post_errors"], post_fec_bits=r["post_bits"],
                                 post_fec_ber=r["post_ber"], theory_ber=theory, evm_rms=r["evm_rms"],
                                 seconds=time.time() - t0))
                print(f"{fmt} OSNR {osnr} {cname}: pre {r['ber']:.2e} post {r['post_ber']:.2e} "
                      f"({r['symbols']} symbols, {time.time() - t0:.1f}s)", flush=True)
    write_csv(RESULTS / "ber_vs_osnr.csv", rows)

    md += ["## BER vs OSNR (full DSP chain)", "",
           "Pre-FEC BER pools the payload of both coded runs at each point; post-FEC BER is on "
           "information bits. Theory = Gray-coded AWGN BER at the same Es/N0.", ""]
    for fmt in OSNR_GRID:
        md += [f"### {fmt.upper()}", "",
               "| OSNR (dB) | Es/N0 (dB) | symbols | theory | pre-FEC BER | post Hamming(7,4) | post LDPC (3,6) |",
               "|---|---|---|---|---|---|---|"]
        pre_curve = []
        for osnr in OSNR_GRID[fmt]:
            rs = [r for r in rows if r["format"] == fmt and r["osnr_db"] == osnr]
            ham = next(r for r in rs if r["code"] == "hamming74")
            ldpc = next(r for r in rs if r["code"] == "ldpc_1008")
            pe = sum(r["pre_fec_errors"] for r in rs)
            pb = sum(r["pre_fec_bits"] for r in rs)
            pre_curve.append(pe / pb)
            md.append(f"| {osnr} | {ham['snr_db']:.2f} | {sum(r['symbols'] for r in rs)} | {ham['theory_ber']:.2e} | "
                      f"{fmt_ber(pe, pb)} | {fmt_ber(ham['post_fec_errors'], ham['post_fec_bits'])} | "
                      f"{fmt_ber(ldpc['post_fec_errors'], ldpc['post_fec_bits'])} |")
        grid = OSNR_GRID[fmt]
        th = [float(M.theoretical_ber(M.FORMATS[fmt], float(M.osnr_to_snr_db(o, base.symbol_rate)))) for o in grid]
        o_sim = osnr_at_ber(grid, pre_curve, 1e-3)[0]
        o_th = osnr_at_ber(grid, th, 1e-3)[0]
        md += ["", f"Required OSNR for pre-FEC BER 1e-3: simulated {o_sim:.2f} dB, AWGN theory {o_th:.2f} dB "
               f"(implementation penalty {o_sim - o_th:.2f} dB).", ""]
        for cname in ("hamming74", "ldpc_1008"):
            post = [next(r for r in rows if r["format"] == fmt and r["osnr_db"] == o and r["code"] == cname)
                    for o in grid]
            vals = [r["post_fec_ber"] for r in post]
            _, t_post = osnr_at_ber(grid, vals, 1e-4)
            _, t_pre = osnr_at_ber(grid, pre_curve, 1e-4)
            rate = post[0]["code_rate"]
            md.append(f"- {cname} (rate {rate:.3f}, i.e. {10 * np.log10(1 / rate):.2f} dB more energy per "
                      f"information bit at the same symbol rate): OSNR for BER 1e-4 uncoded {t_pre} dB, "
                      f"after decoding {t_post} dB")
        md.append("")
    return rows


def ablations(base, frames, nsym, seed, md):
    configs = {
        "full chain": link.RxConfig(),
        "no CD compensation": link.RxConfig(cdc=False),
        "no timing recovery": link.RxConfig(timing=False),
        "no frequency-offset estimation": link.RxConfig(foe=False),
        "no frame sync": link.RxConfig(sync=False),
        "no carrier phase recovery": link.RxConfig(cpr=False),
    }
    rows = []
    for fmt, osnr in ABLATION_OSNR.items():
        p = replace(base, osnr_db=osnr)
        for name, cfg in configs.items():
            r = link.run_ber(p, fmt, frames, nsym, cfg, seed=seed)
            rows.append(dict(format=fmt, osnr_db=osnr, config=name, symbols=r["symbols"],
                             errors=r["errors"], bits=r["bits"], ber=r["ber"], evm_rms=r["evm_rms"]))
            print(f"ablation {fmt} {name}: {r['ber']:.3e}", flush=True)
    write_csv(RESULTS / "ablation.csv", rows)
    md += ["## DSP stage ablation (pre-FEC BER)", "",
           "| configuration | QPSK @ 12 dB OSNR | 16-QAM @ 19 dB OSNR |", "|---|---|---|"]
    for name in configs:
        q = next(r for r in rows if r["format"] == "qpsk" and r["config"] == name)
        m = next(r for r in rows if r["format"] == "16qam" and r["config"] == name)
        md.append(f"| {name} | {fmt_ber(q['errors'], q['bits'])} | {fmt_ber(m['errors'], m['bits'])} |")
    md += ["", f"Each cell: {rows[0]['symbols']} symbols.", ""]


def fixed_point(seed, md):
    rng = np.random.default_rng(seed)
    taps = pulse.rrc_taps(0.1, 2, 64)
    rows = []
    for snr_db in (15.0, 25.0):
        s = M.QAM16.map(rng.integers(0, 2, 4 * 50_000).astype(np.uint8))
        w = channel.add_awgn(pulse.shape(s, taps, 2), snr_db, 2, rng)
        k = slice(200, -200)
        ref = pulse.matched_filter(w, taps, 2)[::2]
        mer = lambda y: 10 * np.log10(1 / np.mean(np.abs(y[k] - s[k]) ** 2))  # noqa: E731
        rows.append(dict(input_snr_db=snr_db, datapath="float64", mer_db=mer(ref), saturations=0))
        for bits in (16, 12, 10, 8):
            y, sat = fixedpoint.matched_filter_q15(w, taps, 2, bits=bits)
            rows.append(dict(input_snr_db=snr_db, datapath=f"{bits}-bit", mer_db=mer(y[::2]), saturations=sat))
    write_csv(RESULTS / "fixed_point.csv", rows)
    md += ["## Fixed-point RRC matched filter (16-QAM, 129 taps, 50k symbols)", "",
           "MER of the matched-filter output; inputs at 25% RMS of full scale, taps scaled to 0.5 peak. "
           "The 16-bit case is the bit-exact model of the C++ `FirQ15`.", "",
           "| input SNR | float64 | 16-bit | 12-bit | 10-bit | 8-bit |", "|---|---|---|---|---|---|"]
    for snr_db in (15.0, 25.0):
        rs = [r for r in rows if r["input_snr_db"] == snr_db]
        md.append(f"| {snr_db:.0f} dB | " + " | ".join(f"{r['mer_db']:.2f} dB" for r in rs) + " |")
    md.append("")


def bus_faults(seed, md):
    rows = []
    for ber, drop in ((0.0, 0.0), (1e-4, 0.0), (1e-3, 0.01), (3e-3, 0.05)):
        env = device.Environment(channel.LinkParams())
        dev = device.DspAsicModel(env, seed=seed)
        link_ = bus.FaultyLink(dev, bit_error_rate=ber, drop_rate=drop, seed=seed)
        drv = driver.RegisterDriver(link_, max_retries=4)
        ok = fail = mismatch = 0
        rng = np.random.default_rng(seed)
        n = 2000
        for i in range(n):
            v = int(rng.integers(0, 1 << 24))
            try:
                drv.write("CD_COMP", v)
                if drv.read("CD_COMP") != v:
                    mismatch += 1
                ok += 1
            except driver.BusError:
                fail += 1
        tx = drv.stats["reads"] + drv.stats["writes"]
        rows.append(dict(bit_error_rate=ber, drop_rate=drop, write_read_pairs=n, completed=ok, bus_errors=fail,
                         data_mismatches=mismatch, retries=drv.stats["retries"],
                         retries_per_transaction=drv.stats["retries"] / tx,
                         frames_corrupted=link_.stats["corrupted"], frames_dropped=link_.stats["dropped"]))
    write_csv(RESULTS / "bus_faults.csv", rows)
    md += ["## Register bus under injected faults", "",
           "2000 write/read-back pairs per row, driver max 4 retries per transaction.", "",
           "| bit error rate | frame drop rate | completed | BusError raised | silent data mismatches | "
           "retries / transaction |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['bit_error_rate']:g} | {r['drop_rate']:g} | {r['completed']} | {r['bus_errors']} | "
                  f"{r['data_mismatches']} | {r['retries_per_transaction']:.3f} |")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8, help="frames per (OSNR, code) point")
    ap.add_argument("--frame-symbols", type=int, default=1 << 16)
    ap.add_argument("--ablation-frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    base = channel.LinkParams()
    t0 = time.time()
    md = ["# Experiment summary", "",
          f"Simulated single-polarisation link: {base.symbol_rate / 1e9:.0f} GBd, RRC roll-off {base.rrc_beta}, "
          f"{base.sps} samples/symbol, {base.length_km:.0f} km SSMF ({base.accumulated_dispersion_ps_nm:.0f} ps/nm), "
          f"combined laser linewidth {base.linewidth_hz / 1e3:.0f} kHz, frequency offset "
          f"{base.freq_offset_hz / 1e9:.1f} GHz, static timing offset {base.timing_offset} symbol. "
          f"Frames of {args.frame_symbols} symbols, {args.frames} frames per (OSNR, code) point, "
          "the first and last 512 symbols of each frame are excluded from counting.", ""]
    ber_sweep(base, args.frames, args.frame_symbols, args.seed, md)
    ablations(base, args.ablation_frames, args.frame_symbols, args.seed, md)
    fixed_point(args.seed, md)
    bus_faults(args.seed, md)
    md.append(f"Total runtime: {time.time() - t0:.0f} s.")
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
