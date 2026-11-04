"""Locate completed experiment inputs independently of publication outputs.

Set PORELOOM_RESULTS_ARCHIVE to an archive root containing results/ and the
optional provenance/ metadata. Plotting saved CSVs does not need this archive.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(os.environ.get('PORELOOM_RESULTS_ARCHIVE', str(ROOT))).expanduser().resolve()
