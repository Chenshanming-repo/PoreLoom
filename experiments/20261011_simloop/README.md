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
| `icarust.patch` | the simulator of every run: upstream Icarust commit `ef05586` (v0.0.7a) plus the truth audit log, seeded read schedule, identity-checked actions, bounded writer, and writer flush every 1000 reads; the header of the file lists the changes |

The Python files are unchanged from the runs except that machine paths are placeholders
(`/path/to/poreloom_work`, `/path/to/conda/envs/poreloom/lib`).

## Building the simulator

```bash
mkdir icarust_writerfix && cd icarust_writerfix
git clone https://github.com/LooseLab/Icarust src_tree && cd src_tree
git checkout ef055864183ca2f149b748d29c34fe4ec1e74cac
git apply /path/to/PoreLoom/experiments/20261011_simloop/icarust.patch
cargo build --release --locked --target-dir ../target   # needs protoc, HDF5 and OpenSSL (Icarust README)
```

`closed_loop.py` looks for the simulator under `$SE/results/poreloom_simloop_20261011/icarust/`
(`SE` in `common.py`):

- `icarust_writerfix/src_tree/` (the patched source, also the simulator's working directory) and
  `icarust_writerfix/target/release/icarust`
- `tls/`: `ca.crt`, and `server.crt` and `server.key` for `localhost` / `127.0.0.1` signed by it
- `vbz/`: `libvbz_hdf_plugin_x86_64.so` from the source's `vbz_plugin/`
- `references/`: `host.fa`, `target.fa` and their `manifest.json`

`ICARUST_SHA` in `closed_loop.py` pins the binary of the paper runs. A rebuild from the same source
has a different hash because build paths are compiled into the binary, so set `ICARUST_SHA` to the
hash of your build.

Checked on 4 November 2026: the patch applies to that commit with `git apply` and with `patch -p1`,
and the result is identical, file for file, to the simulator source of the paper runs (Icarust's own
test certificates in `static/tls_certs/` stay in place; nothing uses them). The rebuild passes
Icarust's 33 tests (`cargo test --release --locked`) and ran the adapter smoke test in
[`../20261104_minknow_icarust/`](../20261104_minknow_icarust/README.md).

- The strategies registered directly in `closed_loop.py` (PoreLoom-A/B, GRU-threshold, two-window
  and GANBase variants) belong to the October exploratory study. They load models through code
  that is not part of this repository (`signal_explore`, `twowin.py`) and cannot be run from here.
  The paper's strategies are defined in `experiments/20261014_final/icarust_live.py`, which
  replaces `build_strategy`.
- `analyze_live.py` takes truth from `e7_engineering_audit.jsonl`, which only the patched
  simulator writes; upstream Icarust cannot be used with it.
- For a real sequencer, use the packaged adapter `python -m poreloom.minknow`
  (`src/poreloom/minknow/`) instead of this client.
