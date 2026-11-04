# Publication outputs

This directory reproduces the figures and tables of the Bioinformatics manuscript and
Supplementary Notes revised on **4 November 2026**. The active inputs are in
`data/revision_20261104/`, with source paths, SHA-256 digests, and author corrections recorded
in that directory's `provenance.json`. The build reads the local snapshot.

From the repository root:

```bash
python -m pip install -r paper/requirements.txt
python paper/scripts/build_revision.py
python paper/scripts/verify_publication.py
```

The build regenerates three main tables, seven supplementary tables, and seven figures from
saved results with the Python dependencies above. Tables are LaTeX fragments; figures are PDF,
SVG, and PNG. The publication manifest is
`provenance/publication_manifest.json`, and the verification report is
`review/publication_verification.json`. The manuscript source package provides full article
and supplementary-document compilation.

## Tables and source data

All input paths below are relative to `data/revision_20261104/`.

| Manuscript | LaTeX fragment in `tables/` | Frozen source |
|---|---|---|
| Table 1: offline comparison | `main_overall.tex` | `table1_new.csv`, `table1_units.csv` |
| Table 2: closed-loop enrichment | `closed_loop.tex` | `icarust_aggregate.json` |
| Table 3: FACP and fixed symmetric thresholds | `supp_stopping.tex` | `table3_fixed_ext_summary.csv`, `table3_fixed_ext_units.csv` |
| Table S1: training, validation, and test counts | `supp_partitions_summary.tex` | `tableS1_counts.csv` |
| Table S2: forced-decision percentages | `supp_forced_decision.tex` | `table3_fixed_ext_decisions.csv`, `table3_fixed_ext_units.csv` |
| Table S3: frozen 3000-sample baselines with FACP | `supp_baseline_facp.tex` | `tableS2_summary.csv`, `tableS2_units.csv` |
| Table S4: observation budgets | `supp_budget_summary.tex` | `budget_summary.csv`, `budget_units.csv` |
| Table S5: effective closed-loop outcomes | `supp_simulation_summary.tex` | `icarust_aggregate.json` |
| Table S6: held-out-taxon and host rejection | `supp_lso_summary.tex` | `lso_rows.csv`, current `new`/`base` rows |
| Table S7: rejection of all test bacterial reads | `supp_all_bacterial_rejection.tex` | `lso_rows.csv`, current `new`/`base` rows |

Input filenames preserve the export names; their embedded table numbers can predate the
current manuscript numbering. In particular, `tableS2_*` supplies current Table S3, while
`tableS4_*` supplies the encoder comparison in current Figure S1.

The main policy has beta fixed at 1, and fixed-window baselines use native probability 0.5.
Table 3 uses symmetric thresholds `delta = 0.01, 0.05, 0.1, 0.2, 0.3`: reject at
`p < delta`, accept at `p >= 1 - delta`, otherwise continue; at the budget, reject at
`p < 0.5`. The extended exports also preserve delta 0.4 and tuned-reference rows for provenance;
the table builder displays the five thresholds listed above. Table S2 reports the budget-forced
decision fraction
separately from the four performance metrics in Table 3.

Table S1 displays exact read counts. Its author-corrected validation counts are 10,000 for
Simulated, R9, and R10 mixed, and 20,000 for sputum. R9 is listed once as Group 1 because its
three training repeats share one split. The correction is recorded in `data/revision_20261104/provenance.json`.

## Figures

| Manuscript | Output stem in `figures/` | Frozen source or drawing script |
|---|---|---|
| Figure 1: overview | `fig01_overview` | `scripts/draw_overview.py` |
| Figure 2: encoder accuracy and FACP stopping positions | `fig02_overall` | `fig2a_p05_summary.csv`, `fig2b_cdf_summary.csv`, `fig2b_cdf_units.csv` |
| Figure 3: observation budgets | `fig05_budget` | `budget_units.csv` |
| Figure S1: sliding-window versus streaming-prefix encoding | `figS_encoding` | `tableS4_units.csv` |
| Figure S2: feature-reuse timing | `fig03_encoder_cache` | `timing_raw.csv` |
| Figure S3: taxon-holdout proportions | `figS_taxon_holdouts` | `figS3_composition.csv` |
| Figure S4: beta sensitivity | `figS_beta` | `figS_beta_units.csv`, `figS_beta_summary.csv` |

Figure 2a evaluates the frozen encoder with a fixed 0.5 probability threshold. Its source,
`fig2a_p05_summary.csv`, contains 128 mean/SD pairs (four datasets × 32 lengths) from section 3.4
of the author's revision instructions, cross-checked against `paper_numbers.json`. The supplied
precision is 0.01 percentage points.

Figure 2 uses shape-preserving cubic interpolation through the supplied means and SDs, without
point markers. The discrete CDF jump at the 6000-sample budget is retained. Figure 2b includes only completed Accept or Reject decisions;
reads without a completed decision are excluded from its denominator.

Figure S4 is an additional experiment in Note 6. It independently fits policies at 41 beta values
from 0.03 to 300 and at beta 1, using the same training/validation protocol as the main policy.
The dashed line marks beta 1, the value used in the main analyses.

## Units, statistics, and denominators

Repeated results use equally weighted means ± sample SD (`ddof=1`, three units), without pooling
reads across units. Units are newly simulated groups, R9 training repeats on one shared split,
or read-disjoint R10 mixed and sputum groups. Paired changes, enrichment ratios, and timing speed
ratios are computed within units before summarization. Figure bands and error bars show ±1 SD.

Offline accuracy and rejection rates count Reject commands completed before natural termination.
A read without a completed command is retained. Host pore time accounts for effective ejection,
which additionally requires the 0.1-s ejection to finish before the read ends. Closed-loop
accuracy and rejection rates use effective ejections. Tables 1 and 3 include all test reads,
including short ones.

The following fractions have different denominators:

| Output | Numerator | Denominator |
|---|---|---|
| Figure 2b; `fig2b_decisions_summary.csv` columns `within_3000_*`, `at_budget_*` | Completed decisions within 3000 samples or at the budget | All completed decisions |
| Table S2; `table3_fixed_ext_decisions.csv` columns `at_budget_*` | Completed Accept or Reject decisions at the 6000-sample budget | All test reads |
| Table S4; `budget_summary.csv` columns `host_forced_*` | Host reads with a completed Accept or Reject decision at the budget | All test host reads |

Reads without a completed decision remain in the denominators of Tables S2 and S4. Stored
fractions are converted to percentages for tables.

The budget experiment uses one fixed R10 mixed group and one fixed sputum group, with three
training repeats on the same test set in each group. Every budget refits the encoder and policy.
Sputum repeat 0 at 6000 samples is the main sputum group-1 model; R10 mixed 6000-sample models
are trained separately for this experiment. Timing uses three paired timed repeats. Closed-loop
results use three seed-matched ratios to nonselective sequencing for each target fraction.
The current aggregate selects the nine replacement beta=1 PoreLoom runs together with the native
baseline runs.

## Historical assets and experimental reruns

`legacy/20261030/` preserves the earlier scripts, figure data, and provenance, including
matched-rejection, tuned-threshold, common-long-read, and selected-beta comparisons.
The current manuscript assets are generated by `scripts/build_revision.py` from the November snapshot.

Experimental reruns use raw signals, original read-ID split membership, preprocessing readers,
trained checkpoints, simulation support files, and the HPC environment described in the root
[README](../README.md). Dated experiment protocols and current training entrypoints are listed
in [experiments/README.md](../experiments/README.md).
