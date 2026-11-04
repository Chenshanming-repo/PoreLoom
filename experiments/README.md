# Experiment entrypoints

The current manuscript is the 4 November 2026 revision. Its default protocol
fits FACP on training reads, early-stops on validation reads (the parity-0
half of the historical DEV caches), and fixes beta to 1. Baselines use native
binary argmax, equivalent to probability 0.5. DEV-B is not used by this protocol.

For a new training run, use a fresh output directory:

```bash
python -m poreloom.method all --data DATA --out NEW_RUN --fs 4000 --device cuda:0
python -m poreloom.method.baselines --tool readcurrent --data DATA --out NEW_BASELINE --fs 4000 --device cuda:0
python -m poreloom.method.thresholds --data DATA --run NEW_RUN --out NEW_THRESHOLDS --fs 4000 --device cuda:0
```

Use `--fs 5000` for sputum. For NanoDeep add `--recipe upstream`. The main
baseline command uses 3000-sample windows. The threshold command replays the
five Table 3 deltas (0.01, 0.05, 0.1, 0.2, 0.3), forces probability 0.5 at the
budget endpoint, and reports forced decisions over all test reads for Table S2.

For larger budgets, use `poreloom.method all --max-samples N`, where N is the
usable budget plus 1500 skipped samples. Prefix caches must cover N. The
manuscript pipeline keeps length-feature scales at 7500 samples and 4000 Hz;
`--fs` still determines actual observation and pore times. Old checkpoints
preserve the feature scales with which they were trained.

The dated directories preserve the October experiment protocols, run layouts,
DEV-B operating-point selection, and earlier stopping controls. Current figure
and table builds use the frozen November snapshot described in
[paper/README.md](../paper/README.md).
