#!/usr/bin/env python3
"""6000-sample baseline strategies for the reported closed-loop comparison.

The archived extension uses the same simulator client, ADC calibration and
NativePolicy as the 3000-sample matrix. Each retrained baseline first classifies
raw[1500:7500] and applies its locked DEV-B threshold. NanoDeep uses the authors'
training recipe. Model locations are configured in models.json.
"""
from pathlib import Path
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('final_icarust_live', REPO/'experiments/20261014_final/icarust_live.py')
il = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = il; _spec.loader.exec_module(il)
cl = il.cl

W6 = json.loads((HERE/'models.json').read_text())
WINDOW, START = 6000, 1500
NAMES = [f'{t}_w6000' for t in il.TOOLS]


def patch_window():
    """Same patch as 20261016_window6000/win6000.py run (one strategy per process, so the W3000 ones are untouched)."""
    from poreloom.baselines import preprocessing as pp
    pp.WINDOW = WINDOW
    for c in pp.CONTRACTS.values(): c.update(raw_start=START, raw_end=START + WINDOW)


class W6000Policy(il.NativePolicy):
    def __init__(self, tool, device, calibration):
        patch_window()
        il.MODELS[tool] = W6[tool]
        window = json.loads((Path(W6[tool])/'window.json').read_text())
        if window != dict(window=WINDOW, raw_start=START, raw_end=START + WINDOW): raise ValueError(f'{tool}: not a W6000 run')
        super().__init__(tool, device, calibration)
        self.name = f'{tool}_w6000'
        self.pins.update(window=window, w6000_wrapper=cl.pin(__file__))


_build = il.build


def build(name, device, calibration):
    return W6000Policy(name[:-len('_w6000')], device, calibration) if name in NAMES else _build(name, device, calibration)


il.build = build
cl.build_strategy = build
cl.STRATEGIES = il.NAMES + NAMES

if __name__ == '__main__':
    cl.main()
