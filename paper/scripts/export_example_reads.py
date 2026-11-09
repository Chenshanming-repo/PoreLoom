#!/usr/bin/env python3
"""Export the manuscript's retained read examples from the frozen PDF artwork.

This artifact export needs no raw sequencing signals or model checkpoint.
The 15-second replay of the selected decisions is recorded beside the inputs.
"""
from pathlib import Path
import shutil

import fitz

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'data/revision_20261109_tnorm15/reference_figures/figS_example_reads.pdf'
    destination = ROOT / 'figures/figS_example_reads.pdf'
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, destination)
    with fitz.open(source) as document:
        assert len(document) == 1
        page = document[0]
        page.get_pixmap(dpi=220).save(destination.with_suffix('.png'))
        destination.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))


if __name__ == '__main__':
    main()
