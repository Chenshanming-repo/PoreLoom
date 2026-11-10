# PoreLoom

Streaming raw-signal classification with adaptive stopping for selective nanopore sequencing.

PoreLoom processes raw signal as chunks arrive and chooses Accept, Reject, or Continue for each read.
It can decide early when the available evidence is sufficient and continue observing ambiguous reads
up to a maximum budget.

<p align="center">
  <img src="docs/poreloom_overview.png" width="900" alt="PoreLoom evidence accumulation and read-specific stopping">
</p>

**Overview.** (a) Causally normalized chunks enter a streaming encoder. Accept retains a read,
Reject requests ejection, and Continue waits for more signal. (b) Cached features are reused;
boundary features are recomputed and pooled to represent the prefix. (c) A length-specific head
maps the current feature vector to normalized host pore time and target-rejection risk for each
action. The final head excludes Continue.

## Installation

```bash
conda env create -f env/poreloom.yml      # Python 3.9, PyTorch 1.12.1
conda activate poreloom
pip install -e .                        # add [raw] for pyslow5, [dev] for pytest, [minknow] for minknow_api
```

Raw BLOW5 preparation uses a separate environment (`env/raw-requirements.txt`, Python 3.10,
pyslow5). Configure data and result roots outside the repository:

```bash
cp .env.example .env   # set PORELOOM_DATA and PORELOOM_RESULTS
```

Alternatively, copy `configs/paths.example.yaml` to `configs/paths.yaml`.

## Example

[example/](example/README.md) is a quick test of an installation: it runs the bundled PoreLoom
model and the three 3000-sample baselines on 1,000 simulated test reads in about a minute on a CPU,
reports the metrics, checks every read against its saved reference, and draws example reads.

```bash
pip install -e . matplotlib
python example/run_example.py --device cpu
```

## Running PoreLoom

Train and evaluate the complete method on one dataset from random initialization:

```bash
python -m poreloom.method all --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

The stages are window pretraining (`stageA`), streaming-prefix training (`stream`), FACP fitting
at `beta = 1.0` (`head`), and test replay (`test`). Each stage can replace `all` in the command.
Model and action-head parameters are fitted on training reads; validation reads select checkpoints
and control early stopping. The main protocol fixes beta at 1.
Use a new output directory when retraining. Refit the action heads when changing `T_norm`.

| Option | Meaning |
|---|---|
| `--fs` | Acquisition rate in Hz (default 4000); replay times are samples / fs. |
| `--max-samples N` | Raw decision endpoint including the skipped prefix (default 7500, a 6000-sample usable budget). |
| `--max-decision-seconds T` | Endpoint as `floor(T * fs)` raw samples. For example, `--fs 5000 --max-decision-seconds 2.1` equals `--fs 5000 --max-samples 10500`. |
| `--replicate r` | Training repeat; offsets the training seeds. |
| `--t-norm T` | Host pore-time normalization in seconds (default 15). |

The encoder's length-feature scales remain 7500 samples and 4000 Hz across acquisition rates
and budgets. `--fs` sets the acquisition rate used for replay timing.

The prefix-cache layout uses `train/`, `dev/`, and `test/`, documented in
`poreloom.method.data`. Prepare caches covering the requested endpoint. Later stages and inference
use the saved configuration. Budget experiments can use prepared caches directly; rebuilding them from raw data requires the original
preprocessing readers, configured with `PORELOOM_DATA_PREP`.

Evaluate the five fixed symmetric-threshold controls on an existing trained encoder:

```bash
python -m poreloom.method.thresholds --data <dataset root> --run <trained run dir> --out <new control dir> --fs <Hz> --device cuda:0
```

This evaluates all five deltas from Table 3 and records `forced_decision_fraction_all_reads`
for the denominator used in Table S2. Every threshold is fixed before evaluation.

## Running on a MinKNOW device (template)

`src/poreloom/minknow/` connects a trained bundle to one MinKNOW sequencing position through the
read-until API (`minknow_api`). It finds the position through the manager service, reads each
channel's calibration, streams raw signal with `get_live_reads`, and sends `unblock` for reject
and `stop_further_data` for accept. The decision computation is the one used in the closed-loop
runs; a unit test checks it against `experiments/20261014_final/icarust_live.py`.

**Validation: tested against the Icarust simulator.**

```bash
pip install -e .[minknow]
# start a sequencing run in MinKNOW with its own adaptive sampling off, then
python -m poreloom.minknow --bundle RUN/poreloom_sh_cp.pt --out LOGS --position MN12345 \
    --device cuda:0 --duration 600 --dry-run
```

`--dry-run` computes and logs every decision without sending it; remove it to act on reads.
Set `MINKNOW_TRUSTED_CA` to MinKNOW's `conf/rpc-certs/ca.crt` if the certificate is not found
automatically, and pass `--token` when MinKNOW requires a developer API token. The device sample
rate must equal the bundle's `fs` (checked at start). `LOGS` receives `manifest.json`,
`decisions.jsonl`, `acks.jsonl`, and `reads.jsonl`. See `python -m poreloom.minknow --help` for
channel ranges, unblock duration, and read-classification filters, and
[experiments/20261104_minknow_icarust/](experiments/20261104_minknow_icarust/README.md) for the
Icarust smoke test.

## Baselines

The main comparison uses each baseline's own normalization and native binary decision
(probability threshold 0.5):

```bash
python -m poreloom.method.baselines --tool readcurrent --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

`--tool` accepts `readcurrent`, `nanodeep`, or `squigglenet`. Use `--recipe upstream` for
NanoDeep to reproduce its paper training configuration.

For baseline runs, install the upstream code under `third_party/`:

| Directory | Upstream | Manuscript commit | License |
|---|---|---|---|
| `third_party/ReadCurrent` | [ReadCurrent](https://github.com/Ming-Ni-Group/ReadCurrent) | `6db8921` | MIT |
| `third_party/NanoDeep` | [NanoDeep](https://github.com/lysovosyl/NanoDeep) | `1abba04` | MIT |
| `third_party/SquiggleNet` | [SquiggleNet](https://github.com/welch-lab/SquiggleNet) | `fb7b206` | MIT |

The PoreLoom backbone in `src/poreloom/method/backbone.py` retains the ReadCurrent architecture
and MIT notice within this repository.

## Method

- **Causal normalization.** After skipping the first 1500 raw samples, the signal is normalized in
  fixed blocks of 480 samples. Each block uses the preceding usable signal's median and MAD,
  combined with training-set priors early in a read. Normalized values beyond 3.5 in absolute
  value are replaced by the last inlier, or zero if none precedes them. Normalized values remain fixed as signal arrives, with the same output for any incoming chunk boundaries.
- **Streaming encoding and feature reuse.** A ReadCurrent-derived backbone is pretrained on
  windows from random initialization, then trained on streaming prefixes. Finalized feature
  frames are cached as a running sum and count; features near the growing right boundary are
  recomputed. Pooling and a length correction yield the target-minus-host logit margin.
- **Factorized action-cost policy (FACP).** A head selected by observed length predicts normalized
  host pore time `U` and target-rejection risk `R` for each legal action. `U` is the host's pore time
  under the action divided by `T_norm` = 15 s, a fixed reference for every dataset and experiment:
  a representative 6000-bp fragment at 400 bases/s. The main policy minimizes
  `U + R` (`beta = 1`), with heads fitted backward on training reads and early-stopped on validation
  reads. The six inputs are the current, previous, and minimum logit margins, two length features,
  and the number of decision steps. Each head interval is evaluated at most once per read.

The default usable budget is 6000 samples, after the 1500 skipped samples. The final head is
applied at the first observation reaching the 7500-raw-sample endpoint (1.875 s at 4 kHz,
1.5 s at 5 kHz), provided the read is still available. Decisions start at 432 usable samples,
and ordinary head intervals span 768 usable samples. The additional beta experiment is confined
to Supplementary Note 6 and Figure S4; all main analyses use beta 1.

## Results

Offline evaluation replays all test reads with observations every 0.4 s. Reads without a completed
command are retained, including reads too short to fill a baseline window. Accuracy and rejection
rates count completed Reject commands; host pore time is shortened only when the subsequent
0.1-s ejection also completes before natural read termination. Closed-loop accuracy and rejection
rates instead count effective ejections.

Values are equally weighted means ± sample SD (`n=3`, `ddof=1`). Units are three simulated groups,
three R9 training repeats on one shared split, or three read-disjoint R10 mixed or sputum groups.
Budget experiments use three training repeats on one fixed group and test set; timing and closed-loop experiments use three timed repeats and
three simulator seeds, respectively. Paired differences and ratios are calculated within units.

| Dataset | Method | Accuracy | Hosts rejected | Targets rejected | Pore time per host read |
|---|---|---|---|---|---|
| Simulated | PoreLoom | 95.40 ± 0.14% | 97.12 ± 0.29% | 6.31 ± 0.44% | 1.609 ± 0.041 s |
| | Best baseline per metric | 92.41 ± 0.54% (RC-6000) | 92.77 ± 1.67% (RC-6000) | 5.18 ± 1.31% (ND-6000) | 3.154 ± 0.321 s (RC-6000) |
| R9 | PoreLoom | 91.67 ± 0.21% | 88.46 ± 0.94% | 5.11 ± 0.53% | 2.090 ± 0.101 s |
| | Best baseline per metric | 84.59 ± 4.85% (ND-3000) | 79.09 ± 14.45% (ND-3000) | 6.14 ± 1.91% (RC-6000) | 2.474 ± 0.200 s (ND-6000) |
| R10 mixed | PoreLoom | 95.68 ± 0.17% | 94.97 ± 0.11% | 3.60 ± 0.31% | 1.437 ± 0.020 s |
| | Best baseline per metric | 89.80 ± 2.71% (ND-6000) | 90.77 ± 2.39% (ND-3000) | 4.91 ± 1.73% (RC-6000) | 1.970 ± 0.346 s (ND-3000) |
| R10 sputum | PoreLoom | 92.09 ± 0.50% | 91.62 ± 0.36% | 7.44 ± 0.68% | 1.526 ± 0.025 s |
| | Best baseline per metric | 90.41 ± 1.01% (RC-6000) | 93.18 ± 1.46% (RC-6000) | 12.37 ± 0.56% (RC-6000) | 1.790 ± 0.172 s (RC-3000) |

Baselines are ReadCurrent (RC), NanoDeep (ND), and SquiggleNet, each trained with 3000- and
6000-sample windows and evaluated at the native probability threshold of 0.5. Each baseline row
above gives the best of these six models **separately for each metric**, as in manuscript Table 1.
PoreLoom's mean host pore time is 15–49% below the lowest baseline mean in each dataset. Its mean
host rejection is 4.20–9.37 percentage points above the highest baseline mean on simulated, R9, and
R10 mixed data; on sputum data, 6000-sample ReadCurrent rejects 1.56 percentage points more hosts,
while PoreLoom uses 26% less host pore time than that model.

- **Early decisions (Figure 2).** Among completed decisions, 59.14%, 41.09%, 82.53%, and 54.34%
  occur within 3000 usable samples on simulated, R9, R10 mixed, and sputum data, respectively;
  3.9–13.3% reach the 6000-sample budget. Figure 2a uses the frozen encoder's native 0.5 threshold
  without FACP.
- **Streaming encoding (Figure S1).** Compared with sliding-window encoding, streaming-prefix
  encoding improves mean accuracy by 1.28–2.73 percentage points, reduces target rejection by
  0.52–2.48 percentage points, and reduces host pore time in every dataset. Each encoder has
  a separately fitted FACP.
- **Feature reuse (Note 4, Figure S2).** On an RTX 2080 Ti, updating 512 reads at different prefix
  lengths after each additional 0.4 s of signal takes 132.9 ± 16.5 ms with reuse versus
  571.4 ± 1.8 ms with full recomputation. The paired speed ratio is 4.35 ± 0.58. These timings
  exclude FACP, queueing, and ejection. The maximum logit difference across tested configurations
  is 4.34e-6.
- **Stopping rules (Table 3, Table S2).** Fixed symmetric thresholds on the same frozen encoder
  reject if target probability `p < delta`, accept if `p >= 1 - delta`, and otherwise continue;
  at the 6000-sample budget they decide at `p = 0.5`. The current comparison uses
  `delta = 0.01, 0.05, 0.1, 0.2, 0.3`. At 0.01, mean host pore time is 0.17–0.42 s longer than
  FACP (11–26%); target rejection is 0.45–1.06 percentage points lower on simulated, R9, and R10
  mixed data and 2.46 percentage points higher on sputum data. About 23–51% of all test reads are
  decided at the budget, versus 4–13% under FACP. At 0.05, mean host pore time remains 0.26–0.30 s
  longer than FACP on simulated and R10 mixed data and is similar on R9 (0.02 s shorter). On sputum, deltas of
  0.05–0.3 shorten host pore time by 0.13–0.29 s relative to FACP and raise target rejection by
  3.35–8.33 percentage points.
- **FACP on frozen baselines (Table S3).** Adding FACP to the 3000-sample baselines raises mean
  accuracy by 1.42–4.46 percentage points in all 12 dataset–classifier combinations. Target
  rejection decreases in 11 combinations and host pore time decreases in 8. The augmented
  baselines still have lower mean host rejection, higher target rejection, and longer host pore
  time than PoreLoom in every dataset.
- **Larger budgets (Figure 3, Table S4).** On R10 mixed group 1, extending the usable budget
  from 6000 to 18000 samples raises mean accuracy from 95.52% to 96.16% and reduces host pore
  time from 1.438 to 1.289 s. On sputum group 1, extending to 12000 samples raises accuracy
  from 92.65% to 94.21% and reduces host pore time from 1.495 to 1.406 s. The fraction
  of all test hosts decided at the budget falls from 2.27 ± 0.13% to 0.44 ± 0.03% and from
  12.37 ± 1.19% to 4.04 ± 0.42%, respectively.
- **Closed-loop simulation (Table 2, Table S5).** Icarust runs use 512 channels for 1200 s.
  At target-molecule fractions of 1%, 10%, and 30%, PoreLoom's target-base enrichment over
  matched nonselective sequencing is 8.19 ± 0.44, 5.02 ± 0.06, and 2.65 ± 0.01. These means
  exceed the best baseline, 3000-sample NanoDeep, by approximately 33%, 20%, and 9%.
- **Taxon holdouts (Note 5, Tables S6–S7).** PoreLoom has the highest host rejection in
  11 of 12 sputum group–holdout comparisons and the lowest rejection of all test bacterial reads
  in all 12. Retention of reads from excluded taxa is 39.07–77.17%; its rejection of those reads
  is lower than ReadCurrent's in all 12 comparisons, SquiggleNet's in 11, and NanoDeep's in 8.

## Data

All sequencing data are public. Models are fitted only on their own dataset's training set;
validation is used for checkpoint selection and early stopping, and test reads are reserved
for evaluation. Main training and test sets are class balanced. The validation set is selected
from held-out reads by deterministic read-ID hashing.

| Family | Source | Hz | Training / Validation / Test per group | Experimental units |
|---|---|---|---|---|
| Simulated | GRCh38.p14 and eight bacterial references, Squigulator `dna-r10-min` | 4000 | 40,000 / 10,000 / 30,000 | 3 molecule and waveform groups |
| R9 | SRR25423905 (PRJNA994188), HEK293T and seven bacteria, R9.4.1 | 4000 | 40,000 / 10,000 / 20,000 | 3 training repeats on one split |
| R10 mixed | HG002 (SRR23215365, Zenodo 7695412) and Zymo D6322 (SRR23437037), R10.4.1 | 4000 | 40,000 / 10,000 / 20,000 | 3 read-disjoint groups |
| R10 sputum | ONT `pathogen_surveillance_2025.09`, barcodes 01–14, R10.4.1 | 5000 | 79,922–79,994 / 20,000 / 58,868–60,002 | 3 groups from one run |

Table S1 lists exact counts for each group, with R9 listed once because its three training repeats
share one split. R10 mixed is a computational mixture of two separate acquisitions. Sputum uses
genomic-locus grouping to keep 10-kb alignment-start bins separate across groups and partitions. Supplementary Note 1 gives accessions, labels, and split details.

## Reproducing the paper

The current publication inputs are frozen in `paper/data/revision_20261109_tnorm15/`.
They use the fixed `T_norm = 15 s` reference. Rebuild the current main and supplementary
tables and figures directly from these saved results:

```bash
python -m pip install -r paper/requirements.txt
python paper/scripts/build_revision.py
python paper/scripts/verify_publication.py
```

[paper/README.md](paper/README.md) maps every current figure and table to its source data and
explains precision, experimental units, and denominators. Figure 2a's fixed-0.5 means and SDs use
the supplied precision of 0.01 percentage points.

Current training and evaluation use `python -m poreloom.method all`, and the fixed-threshold
comparison uses `python -m poreloom.method.thresholds`. For larger budgets, set the raw endpoint
with `--max-samples` to the usable budget plus 1500. The frozen publication snapshot contains
the 9 November beta=1 results, extended fixed-threshold comparison, budget and taxon-holdout
experiments, beta sensitivity results, and exported closed-loop summaries.

See [experiments/README.md](experiments/README.md) for experimental workflows and SLURM templates.

The closed-loop comparison has 72 runs: eight strategies (nonselective sequencing, PoreLoom,
and three baselines at two windows), three target fractions, and three simulator seeds. The
PoreLoom means and SDs are provided in
`paper/data/revision_20261109_tnorm15/offline/paper_numbers.json`.
Repeating the matrix requires Icarust built from upstream commit `ef05586` (v0.0.7a) with
`icarust.patch`, trained checkpoints, and the simulation support files `closed_loop.py` and
`analyze_live.py`. All three files are in
[experiments/20261011_simloop/](experiments/20261011_simloop/README.md) (the default
`PORELOOM_SIMLOOP`), whose README gives the build steps; configure model paths and compute settings. Raw signals, preprocessing
readers, read-ID split membership, checkpoints, and the HPC rerun environment provide the inputs
needed to repeat the experiments.

## Repository layout

```
src/poreloom/method/      data, backbone, FACP, pipeline, and main-comparison baselines
src/poreloom/baselines/   baseline models and preprocessing
src/poreloom/minknow/     MinKNOW read-until adapter (template; tested on Icarust only)
src/poreloom/             streaming, replay, controller, and evaluation modules
example/                  1,000 simulated test reads and trained models for a quick test
experiments/              experimental workflows and cluster job templates
paper/data/revision_20261109_tnorm15/  current frozen publication results (T_norm = 15 s)
paper/                    current tables, figures, rebuild scripts, and provenance
configs/, env/            path templates and environment specifications
tests/                    unit tests
```

## Tests

```bash
python -m pytest -q
```

## License

PoreLoom is released under the MIT License (see `LICENSE`). `src/poreloom/method/backbone.py`
contains ReadCurrent code (MIT License, Copyright (c) 2024 Kechen Fan).

## Citation

Citation information will be added after publication.
