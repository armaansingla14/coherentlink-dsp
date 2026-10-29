# coherentlink-dsp

A simulation of the receiver DSP in a single-polarisation coherent optical
modem (QPSK and 16-QAM), together with a model of the ASIC control interface
that firmware would use to configure the DSP and read its telemetry.

- **Python / NumPy**: transmitter, fibre channel, receiver DSP chain, FEC,
  BER measurement, and the ASIC register-interface layer (register map, CRC
  framed bus, behavioural device model, driver, command sequencer).
- **C++17** (`cpp/`): Q1.15 fixed-point FIR, CRC-16 and a register-file bus
  slave generated from the same register map. Checked bit-exact against the
  Python models.

All results are from simulation. No optical hardware or ASIC was involved.

## Features

**Signal chain**

| block | implementation |
|---|---|
| Mapping | Gray-coded QPSK and 16-QAM, hard-decision slicing, closed-form AWGN BER |
| Pulse shaping | Root-raised-cosine (roll-off 0.1, 64-symbol span) at 2 samples/symbol, matched filter |
| FFT | Radix-2 iterative FFT vectorised over blocks (`fftcore.py`), tested against `numpy.fft` |
| Channel | Chromatic dispersion (quadratic all-pass phase), static fractional timing offset, laser frequency offset, Wiener phase noise, AWGN set from OSNR (0.1 nm reference bandwidth) |
| CD compensation | Overlap-save frequency-domain equaliser; time-domain FIR (Savory) also provided |
| Timing recovery | Oerder-Meyr feed-forward estimator (used in the sweeps); Gardner TED loop with PI filter and cubic Farrow interpolator (tracks clock-frequency offset) |
| Frequency offset | 4th-power spectral peak with parabolic interpolation |
| Frame sync | Preamble magnitude correlation; recovers the bulk delay that a frequency offset causes through the CD equaliser |
| Carrier phase | Viterbi-Viterbi (QPSK), blind phase search with 32 test phases (16-QAM), pi/2 ambiguity resolved with the preamble |
| FEC | Hamming(7,4) syndrome decoding; regular (3,6) LDPC, n = 1008, systematic encoding by GF(2) elimination, hard-decision bit-flipping decoder; random bit interleaver |

**ASIC interface layer** (`python/coherentlink/asic/`)

- Register map with typed fields and access policies (RW, RO, W1C, WO),
  reset values, and overlap/alignment checks.
- Framed bus protocol: 9-byte request, 7-byte response, CRC-16/CCITT-FALSE.
  A `FaultyLink` transport injects bit errors and dropped frames.
- Behavioural device model: enforces access policies, soft reset, a DSP-ready
  delay, asynchronous commands (`RUN_FRAME`, `CLEAR_COUNTERS`, `SELF_TEST`)
  with busy handling and error codes, and interrupt status bits (frame done,
  loss of signal, FEC failure, command error). `RUN_FRAME` runs the real DSP
  chain using the register configuration (format, CDC/CPR/timing/FEC enables,
  configured dispersion) and publishes telemetry: frequency-offset estimate,
  pre/post-FEC error and bit counters (64-bit), EVM, SNR estimate, lock flags.
- Driver: retries on CRC errors and lost responses, maps device status to
  typed exceptions, and read-modify-write that masks W1C/WO bits so pending
  interrupts are not cleared by accident.
- Declarative command sequences (write / expect / poll-until) with a trace;
  the bring-up flow is defined as data and fails with the offending step.

**C++ library** (`cpp/`)

- `FirQ15`: 16-bit samples and coefficients, 64-bit accumulator, round-half-up
  and saturation with a saturation counter.
- `crc16`: table-driven CRC-16/CCITT-FALSE.
- `RegisterFile`: bus slave for the same frame format and access rules. Its
  register table (`regmap_table.hpp`) is generated from the Python map by
  `scripts/gen_regmap_header.py`.
- `clink_tool`: CLI used by the Python parity tests.

## Layout

```
python/coherentlink/
  modulation.py fftcore.py pulse.py channel.py
  cdcomp.py timing.py carrier.py fec.py link.py
  fixedpoint.py cpp_bridge.py
  asic/ regmap.py bus.py device.py driver.py sequencer.py
python/tests/      pytest suite, including C++ parity tests
cpp/               include/clink, src, tests (ctest), tools/clink_tool.cpp
scripts/
  build_cpp.sh           regenerate register table, build and run ctest in Docker
  gen_regmap_header.py
  run_experiments.py     BER sweeps, ablations, fixed-point and bus experiments
results/           CSV files and summary.md from the last run
```

## How to run

```bash
./scripts/build_cpp.sh                 # C++ build + ctest (devtools Docker image)
python -m pytest -q                    # Python tests (parity tests need the C++ build)
python scripts/run_experiments.py      # about 7.5 minutes on a desktop CPU
```

Python dependencies: `numpy`, `scipy`.

## Results

Produced by `python scripts/run_experiments.py` (defaults: seed 0, 8 frames
of 65,536 symbols per OSNR point and code). Full tables are in
`results/summary.md` and the CSV files next to it.

Simulated link: 32 GBd, 1000 km of standard fibre (17,000 ps/nm), 100 kHz
combined laser linewidth, 1 GHz frequency offset, 0.37-symbol timing offset.
Each OSNR point counts 1,032,192 symbols for the pre-FEC BER (516,096 per
code for post-FEC BER).

**Pre-FEC BER with the full DSP chain vs AWGN theory**

| format | OSNR | theory | simulated |
|---|---|---|---|
| QPSK | 10 dB | 2.59e-03 | 3.13e-03 |
| QPSK | 12 dB | 2.17e-04 | 2.96e-04 |
| 16-QAM | 16 dB | 4.74e-03 | 6.24e-03 |
| 16-QAM | 19 dB | 1.60e-04 | 3.65e-04 |

Required OSNR for BER 1e-3 (interpolated): QPSK 11.07 dB vs 10.86 dB theory
(0.21 dB implementation penalty); 16-QAM 18.09 dB vs 17.59 dB (0.50 dB). The
16-QAM penalty grows at high OSNR (residual phase noise after blind phase
search and the truncated CD equaliser response).

**Post-FEC BER** (information bits, same OSNR, same symbol rate)

| format | OSNR | pre-FEC | Hamming(7,4) | LDPC (3,6) bit-flip |
|---|---|---|---|---|
| QPSK | 9 dB | 7.60e-03 | 5.29e-04 | 9.65e-06 |
| QPSK | 10 dB | 3.13e-03 | 8.99e-05 | 0 errors in 518,144 info bits |
| 16-QAM | 15 dB | 1.22e-02 | 1.25e-03 | 4.25e-05 |
| 16-QAM | 16 dB | 6.24e-03 | 3.86e-04 | 4.82e-06 |

OSNR needed for BER 1e-4: QPSK 12.70 dB uncoded, 9.94 dB with Hamming(7,4),
8.08 dB with the LDPC code; 16-QAM 19.92 / 16.70 / 14.57 dB. These
comparisons are at equal symbol rate, so the coded links carry less payload
(rates 0.571 and 0.502, i.e. 2.43 dB and 2.99 dB more energy per information
bit); they are not net coding gains.

**DSP stage ablation** (258,048 symbols per cell)

| configuration | QPSK @ 12 dB | 16-QAM @ 19 dB |
|---|---|---|
| full chain | 3.08e-04 | 3.51e-04 |
| no CD compensation | 5.00e-01 | 5.00e-01 |
| no timing recovery | 2.72e-01 | 4.27e-01 |
| no frequency-offset estimation | 5.00e-01 | 5.00e-01 |
| no frame sync | 5.00e-01 | 5.00e-01 |
| no carrier phase recovery | 6.16e-01 | 4.47e-01 |

Every stage is required at this impairment level; removing any one of them
breaks the link.

**Fixed-point matched filter.** MER at the output of the 129-tap RRC
matched filter (16-QAM, 50k symbols): 16-bit and 12-bit datapaths match
float64 to 0.01 dB at 15 and 25 dB input SNR; 10-bit loses 0.04 dB at 25 dB;
8-bit loses 0.11 dB at 15 dB and 0.57 dB at 25 dB.

**Register bus under injected faults** (2000 write/read-back pairs, up to
4 retries per transaction)

| bit error rate | frame drop rate | completed | BusError raised | silent data mismatches |
|---|---|---|---|---|
| 0 | 0 | 2000 | 0 | 0 |
| 1e-4 | 0 | 2000 | 0 | 0 |
| 1e-3 | 0.01 | 1999 | 1 | 0 |
| 3e-3 | 0.05 | 1979 | 21 | 0 |

The CRC and retry logic never returned wrong data; at the highest error rate
about 1% of transactions ran out of retries and were reported as `BusError`.

**Tests**: 60 pytest tests (DSP blocks, FEC, register map and access policies,
driver error handling, sequencing, telemetry, C++/Python parity) and 10 C++
tests run through ctest.

## Limitations

- Single polarisation; no polarisation demultiplexing (no CMA/MIMO
  equaliser), no PMD, no fibre nonlinearity, and ideal ADC/DAC apart from the
  fixed-point matched-filter experiment.
- The BER sweeps use the feed-forward timing estimator for a static timing
  offset. The Gardner loop handles clock-frequency offset but is tested on
  shorter sequences only, because it is a per-symbol Python loop.
- Frame sync and the pi/2 phase ambiguity use a known 256-symbol preamble per
  frame; cycle slips inside a frame are not corrected.
- FEC decoding is hard-decision only. Soft-decision LDPC decoding (as used in
  real coherent modems) would give substantially more gain.
- The ASIC is a behavioural Python model with a C++ bus slave; there is no
  RTL. No MATLAB code is included.
- C++ code was built and tested with g++ 13 inside a Docker container.

## License

MIT, see `LICENSE`.
