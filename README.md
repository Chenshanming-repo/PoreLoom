# PoreLoom

Streaming raw-signal classification with adaptive stopping for selective nanopore sequencing.

This repository contains the implementation and experiments reported in the Bioinformatics manuscript and its
Supplementary Notes (30 October 2026 version).

Fixed-window classifiers wait for the same amount of signal on every read. PoreLoom instead reads raw signal as chunks
arrive and, at each callback, chooses Accept, Reject, or Continue for each read. Reads whose origin is already clear are
decided early, while uncertain reads keep observing up to a maximum budget.

<p align="center">
  <img src="docs/poreloom_overview.png" width="900" alt="PoreLoom overview">
</p>

**Overview.** (a) Chunks pass through causal normalization and a streaming encoder. Each block (white) is scaled with
statistics of earlier blocks (teal). Accept sequences the read to its natural end, Reject requests ejection, and
Continue waits for the next chunk. (b) A ReadCurrent-derived backbone reuses stable features as the prefix grows and
recomputes features near its right boundary. Pooling and a length correction produce the target-minus-host logit.
(c) The current, previous, and minimum logits, observation length, and number of decision steps inform a
length-specific head. The chosen action minimizes the predicted normalized host time plus the weighted target-rejection
risk. The final head at the budget endpoint excludes Continue.

## Method

- **Causal normalization.** After the first 1500 samples are skipped, the signal is scaled in blocks of 480 samples.
  Each block uses the median and MAD of the preceding signal, averaged with training-set values early in a read when
  few samples are available. Values more than 3.5 standard deviations from the median are replaced by the most recent
  inlier. Normalized values never change once computed and do not depend on how the signal is split into chunks.
- **Streaming encoder with feature reuse.** The backbone follows the ReadCurrent architecture, is initialized by window
  pretraining, and is then trained on streaming prefixes of many lengths. Finalized feature frames are kept as a running
  sum and count, and only the frames affected by the right boundary are recomputed.
- **Factorized action-cost policy (FACP).** For each action, a head selected by the observed length predicts the
  normalized host pore time and the target-rejection risk. The policy chooses the action with the lowest
  `host time + beta * risk`. Heads are fitted backward from the final decision, and `beta` is selected on DEV-B.

By default the usable budget is 6000 samples, so the forced decision falls at 7500 raw samples (1.875 s at 4 kHz).

## Results

All offline numbers come from replay of recorded TEST reads with new signal every 0.4 s, averaged over three units
per dataset. Every method is evaluated on all TEST reads, including reads too short to fill a fixed window, which are
retained without a decision. Host and target rejection count completed Reject commands; host pore time is the mean
time a host read occupies the pore, including 0.1 s for effective ejection.

Repeated results use equal-weight means ± sample SD (n=3, ddof=1). The units are three simulated groups, three R9
training seeds on one fixed split, or three disjoint R10 read groups. They are not independent biological samples.
Budget experiments use three training repeats on the same fixed group and TEST set; timing and closed-loop
experiments use three timed repeats and three simulation seeds, respectively. Paired changes and speed ratios are
calculated within units before summarization.

| Dataset | Method | Accuracy | Hosts rejected | Targets rejected | Pore time per host read |
|---|---|---|---|---|---|
| Simulated | PoreLoom | 95.71 ± 0.08% | 95.73 ± 0.32% | 4.32 ± 0.28% | 1.960 ± 0.080 s |
| | Best baseline per metric | 92.60 ± 0.34% (ND-6000) | 92.61 ± 0.69% (RC-6000) | 7.34 ± 1.44% (ND-6000) | 3.163 ± 0.383 s (ND-6000) |
| R9 | PoreLoom | 91.84 ± 0.20% | 90.58 ± 0.80% | 6.91 ± 0.51% | 1.843 ± 0.079 s |
| | Best baseline per metric | 86.19 ± 2.00% (ND-3000) | 85.80 ± 0.95% (ND-3000) | 5.68 ± 2.47% (ND-6000) | 2.477 ± 0.095 s (ND-3000) |
| R10 mixed | PoreLoom | 95.65 ± 0.18% | 94.90 ± 0.70% | 3.60 ± 0.80% | 1.439 ± 0.103 s |
| | Best baseline per metric | 91.71 ± 0.30% (ND-6000) | 87.20 ± 2.19% (ND-3000) | 2.23 ± 0.27% (ND-6000) | 2.664 ± 0.143 s (ND-6000) |
| R10 sputum | PoreLoom | 92.68 ± 0.28% | 93.90 ± 0.81% | 8.53 ± 0.73% | 1.397 ± 0.087 s |
| | Best baseline per metric | 90.31 ± 1.12% (RC-6000) | 91.44 ± 2.78% (RC-6000) | 10.81 ± 0.89% (RC-6000) | 1.862 ± 0.199 s (RC-3000) |

Baselines are ReadCurrent (RC), NanoDeep (ND), and SquiggleNet trained with 3000- or 6000-sample windows. The second
row of each dataset lists, for each metric, the best of the six baseline models. PoreLoom's host rejection is 2.46–7.70
percentage points above the highest baseline rate, and its host pore time is 25–46% below the lowest baseline value.
The complete mean ± sample-SD comparison is Table 1 of the paper.

- **Early decisions.** Among completed decisions, 50.15%, 36.68%, 73.93%, and 56.22% occur within 3000 usable samples
  on simulated, R9, R10 mixed, and sputum data, while 5.8–14.2% reach the 6000-sample budget.
- **Streaming encoding.** Compared with a sliding-window encoder from the same initialization, the streaming encoder
  improves mean accuracy by 1.87–2.63 percentage points, lowers target rejection by 0.76–3.19 percentage points, and
  reduces host pore time in every dataset.
- **Feature reuse.** On an RTX 2080 Ti, updating 512 reads at different prefix lengths after each 0.4 s of signal takes
  132.9 ± 16.5 ms with feature reuse versus 571.4 ± 1.8 ms with full recomputation. The mean paired speed ratio is
  4.35 ± 0.58 across three timed repeats. These timings exclude the stopping policy, queueing, and ejection.
  Reused and recomputed logits differ by at most 4.35e-6 (Supplementary Note 5 and Fig. S2).
- **Stopping rules.** On the same frozen encoder, fixed-length stopping needs 30–48% more host pore time than FACP.
  Constant and length-calibrated thresholds reach host rejection within 1.1 percentage points of FACP. FACP has the
  shortest host pore time on simulated, R9, and sputum data, and a constant threshold is 0.03 s faster on R10 mixed data.
- **FACP on existing classifiers.** Adding FACP to frozen 3000-sample baselines lowers target rejection by 1.00–5.02
  percentage points and raises accuracy by 0.38–5.65 percentage points in all 12 baseline/dataset means. The augmented
  baselines still reject fewer hosts, reject more targets, and use more host pore time than PoreLoom in every dataset.
- **Larger budgets.** Retraining with an 18000-sample budget on R10 mixed group 1 raises mean accuracy from 95.58% to
  96.25% while host pore time falls from 1.610 to 1.369 s. A 12000-sample budget on sputum group 1 raises accuracy from
  93.03% to 94.57%, with host pore time rising from 1.333 to 1.399 s. The fraction of all TEST hosts decided at the
  budget endpoint falls from 4.46 ± 0.16% to 0.87 ± 0.04% for R10 mixed and from 8.56 ± 0.15% to 3.55 ± 0.37% for
  sputum (Table S5). Each SD describes three training repeats on the same fixed data group and TEST set.
- **Closed-loop simulation.** In Icarust runs with 512 channels for 1200 s, PoreLoom's retained target-base yield is
  7.84, 4.88, and 2.62 times that of nonselective sequencing at 1%, 10%, and 30% targets. The best baseline,
  3000-sample NanoDeep, reaches 6.15, 4.19, and 2.43, so PoreLoom's enrichment is 27.5%, 16.4%, and 7.9% higher.

## Installation

```bash
conda env create -f env/poreloom.yml      # Python 3.9, PyTorch 1.12.1
conda activate poreloom
pip install -e .                          # add [raw] for BLOW5 input (pyslow5), [dev] for pytest
```

Raw BLOW5 preparation uses a separate environment (`env/raw-requirements.txt`, Python 3.10, pyslow5). Data and result
roots are set outside the repository:

```bash
cp .env.example .env   # set PORELOOM_DATA and PORELOOM_RESULTS
```

or copy `configs/paths.example.yaml` to `configs/paths.yaml`.

## Running PoreLoom

`poreloom.method` trains and evaluates the complete method on one dataset from random initialization, with no weights
from other data:

```bash
python -m poreloom.method all --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

The stages are window pretraining (`stageA`), streaming-prefix training (`stream`), FACP fitting with operating-point
selection on DEV-B (`head`), and TEST replay (`test`). Each stage can be run on its own in place of `all`.

| Option | Meaning |
|---|---|
| `--fs` | Sampling rate in Hz (default 4000). Replay times are samples / fs. |
| `--max-samples N` | Raw decision endpoint, including the skipped prefix (default 7500, i.e. a 6000-sample budget). |
| `--max-decision-seconds T` | Endpoint given as a duration: `floor(T * fs)` samples. `--fs 5000 --max-decision-seconds 2.1` equals `--fs 5000 --max-samples 10500`. |
| `--replicate r` | Training repeat; all stage seeds are offset by r. |

The dataset layout (TRAIN, DEV and TEST prefix caches) is documented in `poreloom.method.data`. Prefix caches must
cover the requested endpoint. Later stages and inference read the saved configuration. The budget experiment can use
prepared caches directly; rebuilding them from raw data requires the original preprocessing readers, configured with
`PORELOOM_DATA_PREP`.

## Baselines

The main comparison uses ReadCurrent, NanoDeep, and SquiggleNet as fixed-window classifiers, each with its own
normalization and a DEV-B threshold:

```bash
python -m poreloom.method.baselines --tool readcurrent --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

`--tool` is one of `readcurrent`, `nanodeep`, `squigglenet`. Use `--recipe upstream` for NanoDeep to reproduce the
training configuration used in the paper. The 6000-sample comparison is run by
`experiments/20261016_window6000/win6000.py`.

These baselines load upstream code from `third_party/`, which is not distributed with this repository. Clone the
upstream repositories there before running them:

| Directory | Upstream | License |
|---|---|---|
| `third_party/ReadCurrent` | https://github.com/Ming-Ni-Group/ReadCurrent (commit 6db8921) | MIT |
| `third_party/NanoDeep` | https://github.com/lysovosyl/NanoDeep | MIT |
| `third_party/SquiggleNet` | https://github.com/welch-lab/SquiggleNet | MIT |

The PoreLoom backbone in `src/poreloom/method/backbone.py` is a copy of the ReadCurrent convolutional architecture and
keeps its MIT license notice, so PoreLoom itself does not need `third_party/`.

## Data

All sequencing data are public. Main test sets are class balanced, and every model is trained only on the TRAIN
partition of its own dataset. DEV is split by read-ID hash into DEV-A (early stopping and head fitting) and DEV-B
(operating-point selection).

| Family | Source | Hz | TRAIN / DEV / TEST per unit | Units |
|---|---|---|---|---|
| Simulated | GRCh38.p14 host and eight bacterial references, Squigulator `dna-r10-min` | 4000 | 40k / 20k / 30k | 3 molecule and waveform batches |
| R9 | SRR25423905 (PRJNA994188), HEK293T and seven bacteria, R9.4.1 | 4000 | 40k / 20k / 20k | 3 training seeds on one split |
| R10 mixed | HG002 (SRR23215365, Zenodo 7695412) and Zymo D6322 (SRR23437037), R10.4.1 | 4000 | 40k / 20k / 20k | 3 read-disjoint subsets |
| R10 sputum | ONT `pathogen_surveillance_2025.09`, barcodes 01–14, R10.4.1 | 5000 | ~80k / ~40k / ~60k | 3 groups from one run |

The R10 mixed data are a computational mixture of two separate acquisitions, not one sequencing run. Labeling,
deduplication, and partition details are given in Supplementary Note 1 of the paper.

## Reproducing the paper

`experiments/` contains the scripts that produced the reported runs, organized by date. They were run on a SLURM
cluster. The `hpc/` job files use placeholders (`/path/to/poreloom_work`, `<account>`, `<gpu partition>`) for the
working directory and cluster settings.

| Paper result | Current manuscript location | Experimental scripts |
|---|---|---|
| Main comparison, 3000- and 6000-sample baselines | Table 1 | `experiments/20261014_final/hpc/`, `summarize.py`; `experiments/20261016_window6000/win6000.py` |
| Closed-loop simulation | Table 2; Note 5, Table S7 | `experiments/20261014_final/icarust_run.py`; `experiments/20261018_icarust_w6000/icarust_w6000.py` |
| Stopping-rule controls | Table 3; Note 2 | `experiments/20261015_ablation/ablation.py` |
| Fixed-prefix accuracy and decision positions | Fig. 2; Note 2 | `experiments/20261015_ablation/fixed_window.py`; `experiments/20261014_final/signal_budget.py` |
| Streaming versus sliding-window encoding | Table S6, Fig. S2a,b | `experiments/20261015_ablation/ablation.py` |
| Target-rejection penalty sensitivity | Note 2, Fig. S1 | `experiments/20261015_ablation/ablation.py`; `paper/scripts/plot_beta.py` |
| Fixed 0.5 versus DEV-B-selected thresholds | Table S3 | `paper/scripts/threshold_comparison.py` |
| Matched target rejection | Note 7, Fig. S4 | `paper/scripts/analyze_matched_rejection.py` |
| Feature-reuse timing | Note 5, Fig. S2c | `experiments/20261016_stream_timing/timing.py` |
| FACP on frozen 3000-sample baselines | Table S2 | `experiments/20261015_window_tools/native_cp.py` |
| Observation budgets and endpoint decisions | Fig. 3; Note 4, Table S5 | `experiments/20261016_budget_up/budget_up.py` |
| Read-length eligibility | Note 3, Table S4 | `paper/scripts/length_eligibility.py` |
| Leave-taxon-out evaluation | Note 6, Table S8, Fig. S3 | `experiments/20261014_final/lso.py` |
| Partition counts | Note 1, Table S1 | `paper/figure_data/data_partitions.csv` |

The closed-loop comparison combines 45 runs of nonselective sequencing, PoreLoom, and 3000-sample baselines with
27 runs of the 6000-sample baselines. Running these scripts requires Icarust 0.6.0, the trained checkpoints, and
the original simulation support files `closed_loop.py` and `analyze_live.py`, which are not bundled here. Set
`PORELOOM_SIMLOOP` to their directory and configure the model paths and compute settings before running the matrix.

`paper/figure_data/` contains the saved unit-level inputs for the current manuscript. Rebuild its three main tables,
eight supplementary tables, numeric macros, and seven figures without raw sequencing data or a GPU:

```bash
cd paper
python -m pip install -r requirements.txt
python scripts/prepare_summaries.py
python scripts/plot_figures.py
python scripts/verify_publication.py
```

The experimental figures use means with sample-SD bands or error bars. The overview figure is schematic.
[paper/README.md](paper/README.md) maps every current table and figure to its files and explains the experimental units.
`paper/figure_data/budget.csv` includes `host_forced_fraction`: the fraction of all TEST host reads with a completed
Accept or Reject decision at the budget-forced observation. Its 24 rows cover eight budget conditions and three
training repeats on each fixed data group. `budget_summary.csv` provides their means and sample SDs.

Reconstructing the unit-level CSVs from recorded experimental outcomes requires a separate results archive containing
`results/` and `provenance/`. This archive and the raw sequencing signals are not bundled in this code repository.
When that archive is available locally, set its path before running:

```bash
PORELOOM_RESULTS_ARCHIVE=/path/to/archive python scripts/prepare_results.py
python scripts/verify_publication.py
```

The saved partition-count CSV describes the original fixed splits; it does not contain the read-ID membership lists.
The code-only repository supports regeneration of the publication outputs from saved results. Repeating experiments
also requires the external preprocessing readers, checkpoints, simulation support files, and data described above.

## Repository layout

```
src/poreloom/method/      PoreLoom: data, backbone, streaming encoder, FACP heads, pipeline, main-comparison baselines
src/poreloom/baselines/   baseline models and preprocessing (NanoDeep, ReadCurrent, SquiggleNet)
src/poreloom/             streaming, replay, controller, and evaluation modules used by the experiments
experiments/              scripts and job files of the reported experiments
paper/                    figure data, tables, and scripts that regenerate the manuscript figures
configs/, env/            path template and environment specs
tests/                    unit tests
```

## Tests

```bash
python -m pytest -q
```

## License

PoreLoom is released under the MIT License (see `LICENSE`). `src/poreloom/method/backbone.py` contains code from
ReadCurrent (MIT License, Copyright (c) 2024 Kechen Fan).

## Citation

Citation information will be added after publication.
