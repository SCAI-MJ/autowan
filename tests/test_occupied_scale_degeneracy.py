"""The occupied-bandwidth scale survives a symmetry-enforced degeneracy at the Fermi level.

Graphene's Dirac pair at K is degenerate by symmetry and comes out of the parent split
only numerically (8.8e-6 eV on the QE 15x15x1 mesh). With the Fermi level a hair above
the Dirac point, the pi* band is "occupied" at K and K' alone, so before 2026-10-09 the
block-above-the-largest-gap rule took that one degenerate level as the occupied block:
a scale of 8.7e-15 eV, a 1 % tolerance of zero, and occupied_manifold_is_represented and
model_bands_correspond_to_parent_bands failing on every state
(2026-10-08). The fixture is the real parent's lowest eight bands.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan.gate import (
    DEGENERACY_RESOLUTION_EV,
    SCALES,
    NonEvaluable,
    exclusion_tolerance_ev,
    occupied_block_above_largest_gap,
)

FIXTURE = Path(__file__).parent / "fixtures" / "graphene_dirac_parent_bands.npz"


def _graphene():
    data = np.load(FIXTURE)
    return data["eigenvalues_ev"], float(data["fermi_energy_ev"])


def _construction(e, ef):
    parent = SimpleNamespace(eigenvalues_ev=np.asarray(e), fermi_energy_ev=ef)
    return SimpleNamespace(parent=parent, construction_id="synthetic")


def _old_rule(e, ef):
    """The rule as it stood to 2026-10-08, kept here to show what it gave."""
    e = np.asarray(e)
    occupied = e[e <= ef]
    lower = e.min(axis=1)
    upper = np.minimum(e.max(axis=1), ef)
    intervals = sorted(zip(lower[lower <= ef], upper[lower <= ef]))
    blocks = []
    for lo, hi in intervals:
        if blocks and lo <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], hi))
        else:
            blocks.append((lo, hi))
    bottom = float(occupied.min())
    if len(blocks) > 1:
        above = max(
            range(1, len(blocks)), key=lambda i: (blocks[i][0] - blocks[i - 1][1], i)
        )
        bottom = float(blocks[above][0])
    return bottom, float(occupied.max())


def test_the_real_graphene_parent_is_the_defect_the_old_rule_had():
    e, ef = _graphene()
    # pi* (band 5) is below the Fermi level only at K and K'
    assert int((e[0, :, 4] <= ef).sum()) == 2
    # and lies a numerical hair above pi (band 4) there
    split = float(e[0, :, 4].min() - e[0, :, 3].max())
    assert 0 < split < 1e-4
    bottom, top = _old_rule(e, ef)
    assert top - bottom < 1e-12


def test_graphene_gets_the_whole_occupied_manifold_as_its_scale():
    e, ef = _graphene()
    bottom, top = occupied_block_above_largest_gap(e, ef)
    occupied = e[e <= ef]
    assert bottom == pytest.approx(float(occupied.min()))
    assert top == pytest.approx(float(occupied.max()))
    scale = SCALES["occupied_bandwidth_ev"](_construction(e, ef))
    assert 15.0 < scale < 25.0  # sigma + pi valence width, about 19.5 eV
    assert exclusion_tolerance_ev(e, ef) == pytest.approx(0.01 * scale)


def _bands(*ranges, nk=20):
    """(1, nk, nb) eigenvalues: each band runs linearly over its (lo, hi) across k."""
    cols = [np.linspace(lo, hi, nk) for lo, hi in ranges]
    return np.stack(cols, axis=-1)[None]


def test_a_real_semicore_gap_is_still_chosen():
    # semicore at -30..-29, valence -6..-1, E_F 0 with a band crossing it
    e = _bands((-30.0, -29.0), (-6.0, -1.0), (-2.0, 3.0))
    bottom, top = occupied_block_above_largest_gap(e, 0.0)
    assert bottom == pytest.approx(-6.0)
    assert top <= 0.0


def test_a_split_below_the_resolution_is_no_gap():
    # valence -6..-1; a second band starting 1e-5 eV above the first's top
    e = _bands((-6.0, -1.0), (-1.0 + 1e-5, 2.0))
    bottom, _ = occupied_block_above_largest_gap(e, 0.0)
    assert bottom == pytest.approx(-6.0)


def test_a_split_above_the_resolution_is_still_a_gap():
    e = _bands((-6.0, -4.0), (-4.0 + 10 * DEGENERACY_RESOLUTION_EV, 2.0))
    bottom, _ = occupied_block_above_largest_gap(e, 0.0)
    assert bottom == pytest.approx(-4.0 + 10 * DEGENERACY_RESOLUTION_EV)


def test_a_degenerate_sliver_above_a_real_gap_is_skipped():
    # an insulator whose Fermi level lands exactly on the conduction-band minimum:
    # valence -8..-3 (a semicore-free manifold), CB touches E_F at one k only
    nk = 20
    vb = np.linspace(-8.0, -3.0, nk)
    cb = np.full(nk, 2.0)
    cb[0] = 0.0
    e = np.stack([vb, cb], axis=-1)[None]
    bottom, top = occupied_block_above_largest_gap(e, 0.0)
    assert bottom == pytest.approx(-8.0)
    assert top == pytest.approx(0.0)


def test_an_occupied_manifold_with_no_spread_supplies_no_scale():
    e = np.full((1, 4, 2), -1.0)
    e[..., 1] = 5.0
    with pytest.raises(NonEvaluable):
        SCALES["occupied_bandwidth_ev"](_construction(e, 0.0))
    with pytest.raises(NonEvaluable):
        exclusion_tolerance_ev(e, 0.0)
