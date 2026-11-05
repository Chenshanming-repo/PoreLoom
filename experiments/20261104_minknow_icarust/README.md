# 20261104_minknow_icarust: MinKNOW adapter against Icarust

`smoke.py` checks the deployment path of `python -m poreloom.minknow` (`src/poreloom/minknow/`)
without a sequencer. It starts Icarust as in the closed-loop matrix
(`experiments/20261011_simloop/closed_loop.py`: R10 4 kHz, 512 channels, TEST-role references)
and runs the adapter as a separate process with the connection steps used on a MinKNOW host:

1. manager service on the configured port, TLS trusted through `MINKNOW_TRUSTED_CA`
2. `flow_cell_positions` -> the position's secure port -> `Connection`
3. running acquisition, flow cell channel count, sample rate (checked against the bundle), read
   classifications, per-channel calibration
4. one `get_live_reads` stream; `unblock` for reject, `stop_further_data` for accept

`analyze_live.py` then joins the adapter's logs with the simulator audit log (truth).

```bash
python experiments/20261104_minknow_icarust/smoke.py --out OUT --bundle RUN/poreloom_sh_cp.pt \
    --icarust-root $PORELOOM_RESULTS/poreloom_simloop_20261011/icarust --ld-library-path LD \
    --device cuda:0 --seed 20331101 --duration 300 --target-permille 10
```

## Result (4 November 2026)

One 300 s run (seed 20331101, 1% target, the sim g1 beta = 1 bundle of the paper's replacement
PoreLoom runs, shared RTX 2080 Ti) next to the three 1200 s Icarust runs of the same bundle in the
paper:

| | adapter, 300 s | paper runs, 1200 s (seeds 20331101-03) |
|---|---|---|
| reads in window | 68,753 (691 target) | 276,961 / 281,377 / 281,984 |
| host effective reject | 0.969 | 0.969 / 0.969 / 0.969 |
| target command reject | 0.046 (32 / 691) | 0.054 / 0.050 / 0.061 |
| target fraction of retained bases | 0.261 | 0.247 / 0.255 / 0.248 |
| command latency p50 / p95, s | 0.150 / 0.221 | 0.151 / 0.220, 0.102 / 0.126, 0.102 / 0.127 |

The adapter connected through the manager (position `SIM`, reported MinKNOW 6.1.4), accepted
first-chunk classes `strand` and `adapter`, sent 75,010 actions, and received 140
`FAILED_READ_FINISHED` acknowledgements (the read had already ended), with no packet gaps or
overlaps. This checks the software path only; it says nothing about behaviour on a sequencer.
