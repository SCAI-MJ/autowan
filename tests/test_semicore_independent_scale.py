"""The occupied-bandwidth scale is measured above the largest gap below the Fermi level.

Every tolerance a check derives from ``occupied_bandwidth_ev`` used to be set by the depth of the
deepest occupied state -- that is, by how much semicore the pseudopotential or PAO basis carries
(54 eV for bcc Fe with ``Fe_PBE19S``, 73 eV for CrI3), not by the material. The scale is now the
spread of the occupied block above the largest gap below E_F, for both certification targets.

The gap is found on the band-edge envelope: each band's energy range over every k-point, of both
spin channels, clipped at E_F and merged. A band's own coarse sampling therefore cannot split it,
and a gap in one spin channel that the other channel fills is not a gap. Where the parent has more
than one gap below E_F, the *largest* sets the boundary -- which is not always the gap nearest the
valence, so a shallower semicore shell above the largest gap stays inside the scale (the Sr_sv
case below). That is the rule as specified; its consequence is reported, not tuned away.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan import gate as G
from autowan.core_checks import CORE_CHECKS
from autowan.gate import SCALES, CertificationGate, Construction

_SPEC = importlib.util.spec_from_file_location(
    "analytic_constructions",
    Path(__file__).parent / "fixtures" / "analytic" / "analytic_constructions.py",
)
ac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ac)


def _check(check_id):
    return next(c for c in CORE_CHECKS if c.check_id == check_id)


def _scale(parent):
    return SCALES["occupied_bandwidth_ev"](ac.construction(parent, ac.D_ONLY_MODEL))


def _cos(k=ac.K):
    return np.cos(2 * np.pi * k)


# ---------------------------------------------------------------------------
# the rule on analytic parents
# ---------------------------------------------------------------------------


def test_a_separated_semicore_manifold_no_longer_changes_the_scale():
    without = _scale(ac.s_d_parent())
    with_semicore = _scale(ac.semicore_s_d_parent())
    assert without == pytest.approx(ac.VALENCE_BANDWIDTH_EV)
    # the semicore level 44 eV below the valence used to make this 49.99 eV
    assert with_semicore == pytest.approx(without)


def test_with_no_gap_below_the_fermi_level_the_whole_occupied_spread_is_kept():
    parent = ac.s_d_parent()
    occupied = parent.eigenvalues_ev[parent.eigenvalues_ev <= ac.FERMI_EV]
    assert _scale(parent) == pytest.approx(float(np.ptp(occupied)))


def test_of_two_semicore_gaps_the_larger_sets_the_boundary_fe_pbe19s_like():
    # 3s at -90 eV, 3p over -53 .. -52.5 eV, then the valence from -6 eV: gaps of 37 and 46.5 eV.
    parent = ac.make_parent(
        [[ac.flat(-90.0), -52.75 + 0.25 * _cos(), ac.s_band(), ac.d_band()]]
    )
    assert _scale(parent) == pytest.approx(ac.VALENCE_BANDWIDTH_EV)
    bottom, top = G.occupied_block_above_largest_gap(
        parent.eigenvalues_ev, parent.fermi_energy_ev
    )
    assert bottom == pytest.approx(ac.VALENCE_BOTTOM_EV)
    assert top == pytest.approx(ac.VALENCE_TOP_EV)


def test_of_two_semicore_gaps_the_larger_sets_the_boundary_even_below_a_shallower_shell():
    # Sr_sv-like: 4s at -38 eV, 4p over -17 .. -16 eV, valence from -6 eV. The 4s -> 4p gap
    # (21 eV) is larger than the 4p -> valence gap (10 eV), so the 4p shell stays in the scale.
    parent = ac.make_parent(
        [[ac.flat(-38.0), -16.5 - 0.5 * _cos(), ac.s_band(), ac.d_band()]]
    )
    bottom, top = G.occupied_block_above_largest_gap(
        parent.eigenvalues_ev, parent.fermi_energy_ev
    )
    assert bottom == pytest.approx(-17.0)
    assert top == pytest.approx(ac.VALENCE_TOP_EV)
    assert _scale(parent) == pytest.approx(ac.VALENCE_TOP_EV + 17.0)


def test_a_gap_in_one_spin_channel_that_the_other_fills_is_not_a_gap():
    up_low = -10.0 + 0.5 * (1 - _cos())  # -10 .. -9 eV
    down_low = -9.5 + 3.5 * (1 - _cos())  # -9.5 .. -2.5 eV, across the spin-up gap
    both = ac.make_parent([[up_low, ac.d_band()], [down_low, ac.d_band()]])
    occupied = both.eigenvalues_ev[both.eigenvalues_ev <= ac.FERMI_EV]
    assert _scale(both) == pytest.approx(float(np.ptp(occupied)))
    # read on the spin-up channel alone, -9 .. -3 eV is a 6 eV gap and the block above it is 3 eV
    up_only = ac.make_parent([[up_low, ac.d_band()]])
    assert _scale(up_only) == pytest.approx(ac.VALENCE_TOP_EV + 3.0)


def test_a_coarsely_sampled_band_is_not_split_by_its_own_sampling():
    band = -10.5 + 5.0 * (1 - _cos())  # -10.5 .. -0.5 eV on eight points
    assert (
        np.diff(np.sort(band)).max() > 3.0
    )  # holes wider than 3 eV between its samples
    parent = ac.make_parent([[band, ac.flat(5.0)]])
    assert not parent.is_metallic()
    assert _scale(parent) == pytest.approx(10.0)


def test_the_band_distance_bar_of_a_semicore_bearing_insulator_moves_with_the_scale():
    valence = -6.0 + 2.5 * (1 - _cos())  # -6 .. -1 eV
    conduction = 1.0 + 1.5 * (1 - _cos())  # 1 .. 4 eV
    parent = ac.make_parent([[ac.flat(-50.0), valence, conduction]])
    assert not parent.is_metallic()
    c = ac.construction(
        parent, ac.hr_diagonal([(-50.0, 0.0), (-3.5, -1.25), (2.5, -0.75)])
    )
    # one percent of the 5 eV valence block, where the semicore level made it 0.49 eV
    assert _check("band_distance_from_parent").threshold.value(c) == pytest.approx(0.05)


def test_the_default_coverage_tolerance_tightens_and_a_default_verdict_moves():
    """A full-valence model that places its d band 0.1 eV too high.

    Under a semicore-set scale (49.99 eV, a 0.50 eV tolerance) every occupied state had a partner
    and the construction passed coverage. Against the valence block (5.99 eV, 60 meV) the seven
    occupied d states are unrepresented: 7 of 23 occupied states. This is the direction in which
    the scale moves a default-target verdict.
    """
    parent = ac.semicore_s_d_parent()
    model = ac.hr_diagonal([(ac.SEMICORE_EV, 0.0), (-4.0, -1.0), (-1.25 + 0.1, -0.875)])
    report = CertificationGate(CORE_CHECKS).run(ac.construction(parent, model))
    v = {x.check_id: x for x in report.verdicts}["occupied_manifold_is_represented"]
    assert v.evaluable and v.applies
    assert v.metric_value == pytest.approx(7 / 23)
    assert v.passed is False
    assert "60 meV" in v.note
