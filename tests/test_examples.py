"""The README's two-minute demo keeps printing what the README says it prints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SODIUM = ROOT / "examples" / "sodium"


def _rescore(*extra: str) -> str:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rescore.py"), str(SODIUM), *extra],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def _eta2(block: str) -> float:
    line = next(l for l in block.splitlines() if "band_interpolation_eta2" in l)
    return float(line.split("band_interpolation_eta2")[1].split()[0])


def test_the_sodium_demo_refuses_the_frontier_and_certifies_the_second_repair():
    out = _rescore()
    blocks = {b.split()[0]: b for b in out.split("\n\n") if b.startswith("wan")}
    assert "REFUSED" in blocks["wan"].splitlines()[0]
    assert "REFUSED" in blocks["wan_r1"].splitlines()[0]
    assert "CERTIFIED" in blocks["wan_r2"].splitlines()[0]
    assert abs(_eta2(blocks["wan"]) - 41.41) < 0.05
    assert abs(_eta2(blocks["wan_r1"]) - 38.69) < 0.05
    assert abs(_eta2(blocks["wan_r2"]) - 2.967) < 0.005


def test_the_bands_converter_writes_fractional_kpoints_the_criteria_read():
    sys.path.insert(0, str(ROOT / "scripts"))
    from qe_bands_to_npz import fractional_kpoints

    from autowan.qe_outputs import read_parent

    parent = read_parent(SODIUM / "parent" / "data-file-schema.xml")
    frac = fractional_kpoints(parent)
    # the nscf parent is the 11x11x11 construction mesh: every k lands on it
    assert frac.shape == (11**3, 3)
    assert np.abs(frac * 11 - np.round(frac * 11)).max() < 1e-6
