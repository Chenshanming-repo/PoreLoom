# PoreLoom

Streaming adaptive-sampling control on raw nanopore signal: causal ADC windows,
ReadCurrent encoder + GRU, cost-aware controllers, callback-based streaming adapter
and paired evaluation.

## Layout

```
src/poreloom/   core, models, data, training, evaluation, streaming, baselines, utils
scripts/        generic CLI entrypoints (prepare / train / evaluate / verify / report)
configs/        shared configs; paths.example.yaml
tests/          unit/ mirrors src/poreloom; fixtures/
env/            environment specs
third_party/    upstream baseline sources (not included; ReadCurrent, CAFES, NanoDeep, SquiggleNet)
```

## Environments

- Training / evaluation: `env/readcurrent.yml` (Python 3.9, PyTorch 1.12.1)
- Raw BLOW5 preparation: `env/raw-requirements.txt` (Python 3.10, pyslow5)

## Data and results

Data and results live outside the repository:

```bash
cp .env.example .env   # set PORELOOM_DATA / PORELOOM_RESULTS
```

## PoreLoom method (main)

`src/poreloom/method/` is the complete method, PoreLoom-SH-ACC C-p, trained from random initialization on one
dataset with no weights from other data:

```bash
python -m poreloom.method all --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

`--fs` defaults to 4000 Hz. Set the raw decision endpoint with `--max-samples N` (default 7500), or set its duration
with `--max-decision-seconds T`; the endpoint is `floor(T * fs)` samples, including the skipped prefix. For example,
`--fs 5000 --max-decision-seconds 2.1` is equivalent to `--fs 5000 --max-samples 10500`. Training, length features,
head count and inference use the saved configuration. Prefix caches must cover the requested endpoint.

Stages: window encoder from random init (stage A) -> streaming encoder (S1) -> C-p action head (beta by maximum
DEV-B ACC) -> TEST replay. `--replicate r` for independent training seeds. Dataset layout: `poreloom.method.data`.

## Baselines

`poreloom.baselines` holds the four comparison tools on native windows: NanoDeep
(ADC raw[1000:4000]), ReadCurrent and CAFES (ADC raw[1500:4500]) and SquiggleNet
(pA raw[1500:4500]). It has model builders, per-tool normalization, training
recipes and host-margin scoring. Upstream code is read from `third_party/`, which is not
included in this repository.

```bash
# Score with the archived native fits (default checkpoint and development source)
python scripts/score_baseline.py --tool readcurrent --role C_screen --output scores/readcurrent_C.npz

# Refit: NanoDeep / SquiggleNet read the native export directly
python scripts/train_baseline.py --tool nanodeep --output runs/nanodeep/seed20260929
# ReadCurrent / CAFES need tile datasets first
python scripts/prepare_baseline_tiles.py --output runs/tiles
python scripts/train_baseline.py --tool readcurrent --tiles runs/tiles --output runs/readcurrent/seed20260929
```

`--smoke` runs a tiny fit. ReadCurrent/CAFES training uses the patched upstream
trainer, which needs torch >= 1.13 (the archived fits used torch 2.6).

## Tests

```bash
python -m pytest -q
```

## Provenance

Code imported from `signal_explore` (tag `import-from-signal_explore`). Results produced
before the import are verified in `signal_explore` against their source snapshots.
