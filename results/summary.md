# Experiment summary

Simulated single-polarisation link: 32 GBd, RRC roll-off 0.1, 2 samples/symbol, 1000 km SSMF (17000 ps/nm), combined laser linewidth 100 kHz, frequency offset 1.0 GHz, static timing offset 0.37 symbol. Frames of 65536 symbols, 8 frames per (OSNR, code) point, the first and last 512 symbols of each frame are excluded from counting.

## BER vs OSNR (full DSP chain)

Pre-FEC BER pools the payload of both coded runs at each point; post-FEC BER is on information bits. Theory = Gray-coded AWGN BER at the same Es/N0.

### QPSK

| OSNR (dB) | Es/N0 (dB) | symbols | theory | pre-FEC BER | post Hamming(7,4) | post LDPC (3,6) |
|---|---|---|---|---|---|---|
| 8 | 6.93 | 1032192 | 1.32e-02 | 1.51e-02 | 2.06e-03 | 1.24e-04 |
| 9 | 7.93 | 1032192 | 6.37e-03 | 7.60e-03 | 5.29e-04 | 9.65e-06 |
| 10 | 8.93 | 1032192 | 2.59e-03 | 3.13e-03 | 8.99e-05 | 0 (< 1.9e-06) |
| 11 | 9.93 | 1032192 | 8.56e-04 | 1.10e-03 | 3.39e-06 | 0 (< 1.9e-06) |
| 12 | 10.93 | 1032192 | 2.17e-04 | 2.96e-04 | 0 (< 1.7e-06) | 0 (< 1.9e-06) |
| 13 | 11.93 | 1032192 | 3.94e-05 | 6.30e-05 | 0 (< 1.7e-06) | 0 (< 1.9e-06) |

Required OSNR for pre-FEC BER 1e-3: simulated 11.07 dB, AWGN theory 10.86 dB (implementation penalty 0.21 dB).

- hamming74 (rate 0.571, i.e. 2.43 dB more energy per information bit at the same symbol rate): OSNR for BER 1e-4 uncoded 12.70 dB, after decoding 9.94 dB
- ldpc_1008 (rate 0.502, i.e. 2.99 dB more energy per information bit at the same symbol rate): OSNR for BER 1e-4 uncoded 12.70 dB, after decoding 8.08 dB

### 16QAM

| OSNR (dB) | Es/N0 (dB) | symbols | theory | pre-FEC BER | post Hamming(7,4) | post LDPC (3,6) |
|---|---|---|---|---|---|---|
| 14 | 12.93 | 1032192 | 1.78e-02 | 2.11e-02 | 3.95e-03 | 3.17e-04 |
| 15 | 13.93 | 1032192 | 9.83e-03 | 1.22e-02 | 1.25e-03 | 4.25e-05 |
| 16 | 14.93 | 1032192 | 4.74e-03 | 6.24e-03 | 3.86e-04 | 4.82e-06 |
| 17 | 15.93 | 1032192 | 1.93e-03 | 2.79e-03 | 5.59e-05 | 0 (< 9.6e-07) |
| 18 | 16.93 | 1032192 | 6.34e-04 | 1.11e-03 | 1.61e-05 | 0 (< 9.6e-07) |
| 19 | 17.93 | 1032192 | 1.60e-04 | 3.65e-04 | 2.54e-06 | 0 (< 9.6e-07) |
| 20 | 18.93 | 1032192 | 2.90e-05 | 8.99e-05 | 0 (< 8.5e-07) | 0 (< 9.6e-07) |

Required OSNR for pre-FEC BER 1e-3: simulated 18.09 dB, AWGN theory 17.59 dB (implementation penalty 0.50 dB).

- hamming74 (rate 0.571, i.e. 2.43 dB more energy per information bit at the same symbol rate): OSNR for BER 1e-4 uncoded 19.92 dB, after decoding 16.70 dB
- ldpc_1008 (rate 0.502, i.e. 2.99 dB more energy per information bit at the same symbol rate): OSNR for BER 1e-4 uncoded 19.92 dB, after decoding 14.57 dB

## DSP stage ablation (pre-FEC BER)

| configuration | QPSK @ 12 dB OSNR | 16-QAM @ 19 dB OSNR |
|---|---|---|
| full chain | 3.08e-04 | 3.51e-04 |
| no CD compensation | 5.00e-01 | 5.00e-01 |
| no timing recovery | 2.72e-01 | 4.27e-01 |
| no frequency-offset estimation | 5.00e-01 | 5.00e-01 |
| no frame sync | 5.00e-01 | 5.00e-01 |
| no carrier phase recovery | 6.16e-01 | 4.47e-01 |

Each cell: 258048 symbols.

## Fixed-point RRC matched filter (16-QAM, 129 taps, 50k symbols)

MER of the matched-filter output; inputs at 25% RMS of full scale, taps scaled to 0.5 peak. The 16-bit case is the bit-exact model of the C++ `FirQ15`.

| input SNR | float64 | 16-bit | 12-bit | 10-bit | 8-bit |
|---|---|---|---|---|---|
| 15 dB | 14.99 dB | 14.99 dB | 14.99 dB | 14.99 dB | 14.88 dB |
| 25 dB | 24.99 dB | 24.99 dB | 24.99 dB | 24.95 dB | 24.42 dB |

## Register bus under injected faults

2000 write/read-back pairs per row, driver max 4 retries per transaction.

| bit error rate | frame drop rate | completed | BusError raised | silent data mismatches | retries / transaction |
|---|---|---|---|---|---|
| 0 | 0 | 2000 | 0 | 0 | 0.000 |
| 0.0001 | 0 | 2000 | 0 | 0 | 0.013 |
| 0.001 | 0.01 | 1999 | 1 | 0 | 0.145 |
| 0.003 | 0.05 | 1979 | 21 | 0 | 0.556 |

Total runtime: 442 s.
