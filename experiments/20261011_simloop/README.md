# 20261011_simloop: Icarust closed-loop support

Support files of the Icarust closed-loop runs. The paper's closed-loop drivers
(`experiments/20261014_final/icarust_live.py` and `icarust_run.py`,
`experiments/20261018_icarust_w6000/`) load them from this directory by default;
`PORELOOM_SIMLOOP` points them elsewhere.

| File | Role |
|---|---|
| `closed_loop.py` | Icarust launch (R10 4 kHz profile with the TEST-role references, TLS and port config, tmpfs output) and the live client: one `get_live_reads` stream for all channels, per-read ADC buffers, a separate batched inference thread, `unblock` (0.1 s) for reject and `stop_further_data` for accept |
| `analyze_live.py` | joins the client's `decisions.jsonl` and `acks.jsonl` with the simulator audit log (truth) and writes `summary.json` and `read_outcomes.jsonl` |
| `common.py` | shared paths and the strategy registry imported by both |
| `icarust/writer_batch.patch` | the writer change applied to the Icarust build used for every run: flush batch 4000 -> 1000 reads (64 -> 16 MB) |

The files are unchanged from the runs except that machine paths are placeholders
(`/path/to/poreloom_work`, `/path/to/conda/envs/poreloom/lib`).

- The strategies registered directly in `closed_loop.py` (PoreLoom-A/B, GRU-threshold, two-window
  and GANBase variants) belong to the October exploratory study. They load models through code
  that is not part of this repository (`signal_explore`, `twowin.py`) and cannot be run from here.
  The paper's strategies are defined in `experiments/20261014_final/icarust_live.py`, which
  replaces `build_strategy`.
- `analyze_live.py` takes truth from `e7_engineering_audit.jsonl`, which is written by the
  instrumented Icarust build pinned by `ICARUST_SHA` in `closed_loop.py`.
- For a real sequencer, use the packaged adapter `python -m poreloom.minknow`
  (`src/poreloom/minknow/`) instead of this client.
