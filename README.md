# PoreLoom

Streaming adaptive decisions for selective nanopore sequencing from raw signals.

PoreLoom reads raw nanopore signal as chunks arrive and, at each callback, chooses to Accept, Reject, or Continue for
every read. All model parameters are fixed during inference. The method has three parts:

- Causal normalization and a streaming convolutional encoder. Each 480-sample block is normalized using only the
  preceding signal, so earlier blocks never change. The backbone follows the ReadCurrent architecture and is trained on
  streaming prefixes after window pretraining.
- An encoder cache. Finalized feature frames are kept as a running sum and count, and only the frames that depend on the
  current right boundary are recomputed.
- The factorized action-cost policy (FACP). For each action, a length-bucketed head predicts host pore occupancy and
  target-rejection risk from a short history of the target-minus-host logit. The policy picks the action with the
  lowest estimated cost. Heads are fitted backward from the forced decision, and the penalty is selected on DEV-B.

By default the first 1500 samples are skipped and the usable budget is 6000 samples, so the forced decision falls at
7500 samples. Across simulated, R9, R10 computational-mixture, and R10 sputum data, PoreLoom reaches mean host rejection
of 90.58–95.73% and has shorter mean host pore occupancy than ReadCurrent, NanoDeep, and SquiggleNet at 3000- and
6000-sample windows. Among completed decisions, 36.68–73.93% occur within 3000 usable samples. The target
false-rejection rate is higher than that of some alternatives, for example the 6000-sample NanoDeep model on R9 and R10
mixed data, and it increases for held-out taxa.

## Installation

```bash
conda env create -f env/readcurrent.yml   # Python 3.9, PyTorch 1.12.1
conda activate readcurrent
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
| `--select` | Operating point on DEV-B: `max_acc` (default) or `frr:<limit>`. |

The dataset layout (TRAIN, DEV and TEST prefix caches) is documented in `poreloom.method.data`. Prefix caches must
cover the requested endpoint. Later stages and inference read the saved configuration.

## Baselines

The main comparison uses ReadCurrent, NanoDeep, and SquiggleNet as fixed-window classifiers, each with its own
normalization and a DEV-B threshold:

```bash
python -m poreloom.method.baselines --tool readcurrent --data <dataset root> --out <run dir> --fs <Hz> --device cuda:0
```

`--tool` is one of `readcurrent`, `nanodeep`, `squigglenet`. For NanoDeep, `--recipe upstream` uses the authors'
training settings instead of the shared recipe.

`poreloom.baselines` and `scripts/` contain an earlier native-window setup (NanoDeep on raw[1000:4000]; ReadCurrent,
CAFES, and SquiggleNet on raw[1500:4500]) with model builders, per-tool normalization, training and host-margin
scoring:

```bash
python scripts/train_baseline.py --tool nanodeep --output runs/nanodeep/seed20260929
python scripts/prepare_baseline_tiles.py --output runs/tiles          # ReadCurrent and CAFES need tiles first
python scripts/train_baseline.py --tool readcurrent --tiles runs/tiles --output runs/readcurrent/seed20260929
python scripts/score_baseline.py --tool readcurrent --role C_screen --output scores/readcurrent_C.npz
```

`--smoke` runs a tiny fit. The ReadCurrent and CAFES trainer needs torch >= 1.13.

These baselines load upstream code from `third_party/`, which is not distributed with this repository. Clone the
upstream repositories there before running them:

| Directory | Upstream | License |
|---|---|---|
| `third_party/ReadCurrent` | https://github.com/Ming-Ni-Group/ReadCurrent (commit 6db8921) | MIT |
| `third_party/NanoDeep` | https://github.com/lysovosyl/NanoDeep | MIT |
| `third_party/SquiggleNet` | https://github.com/welch-lab/SquiggleNet | MIT |
| `third_party/CAFES` | https://github.com/Chenshanming-repo/CAFES | MIT |

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
deduplication, and partition details are given in Supplementary Section S1 of the paper.

## Reproducing the paper

`experiments/` contains the scripts that produced the reported runs, organized by date. They were run on a SLURM
cluster. The `hpc/` job files use placeholders (`/path/to/poreloom_work`, `<account>`, `<gpu partition>`) for the
working directory and cluster settings.

| Paper result | Scripts |
|---|---|
| Main comparison with 3000- and 6000-sample baselines (Fig. 2a–c) | `experiments/20261014_final/hpc/`, `summarize.py`; `experiments/20261016_window6000/win6000.py` |
| Icarust target yield (Fig. 2d) | `experiments/20261014_final/icarust_run.py`, `icarust_live.py` |
| Fixed-prefix accuracy (Fig. 2e) | `experiments/20261015_ablation/fixed_window.py` |
| Decision positions (Fig. 2f) | `experiments/20261014_final/signal_budget.py`; `paper/scripts/signal_position_data.py` |
| Streaming versus window encoder, stopping-rule controls (Figs. 3a,b and 4a–c) | `experiments/20261015_ablation/ablation.py` |
| Cached versus full-prefix update time (Figs. 3c,d) | `experiments/20261016_stream_timing/timing.py` |
| FACP on frozen 3000-sample baselines (Figs. 4d–f) | `experiments/20261015_window_tools/native_cp.py` |
| Larger decision budgets (Fig. 5) | `experiments/20261016_budget_up/budget_up.py` |
| Leave-taxon-out (Fig. 6) | `experiments/20261014_final/lso.py` |
| Online/offline decision agreement (Discussion) | `experiments/20261014_final/icarust_check.py` |
| Callback cadence (Supplementary) | `experiments/20261015_ablation/icarust_chunk.py` |
| Cross-dataset transfer (Supplementary) | `experiments/20261015_cross_domain/` |

`paper/` regenerates the figures, tables, and numbers of the manuscript. `paper/figure_data/` holds the unrounded CSV
input of every plotted comparison, so the figures can be redrawn without the raw results:

```bash
cd paper
python -m pip install -r requirements.txt    # Python >= 3.10; no PyTorch or GPU needed
python scripts/plot_figures.py               # writes figures/
python scripts/draw_overview.py              # Fig. 1 schematic
```

The raw experimental results (read-level decisions and outcomes, selections, and logs for all reported runs) and the
read-level partition tables are archived on Zenodo: DOI to be added. To rebuild `figure_data/`, `tables/`, and
`numbers.tex` from them, extract the `results/` directory of the archive into `paper/results/` and run:

```bash
python scripts/prepare_results.py
```

## Repository layout

```
src/poreloom/method/      PoreLoom: data, backbone, streaming encoder, FACP heads, pipeline, main-comparison baselines
src/poreloom/baselines/   native-window baselines (NanoDeep, ReadCurrent, CAFES, SquiggleNet)
src/poreloom/             streaming, replay, controller, and evaluation modules used by the experiments
scripts/                  native-baseline CLI entry points
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
