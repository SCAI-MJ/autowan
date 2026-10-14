"""An honoured semicore exclusion is a gate-wide fact: every check that reads parent bands
counts only the bands the deck asked its model to represent.

The 2026-10-01 gate patch taught one check, ``occupied_manifold_is_represented``, about a
deck's ``exclude_bands``. The others still counted the excluded semicore shells, and a
construction refused by those miscounts was published as a residual -- sodium by the
frontier check (an excluded band below E_F makes the parent "bracket" the Fermi level at
every k-point), TiO2 by band distance (every excluded state is paired with a model band
tens of eV away). A half-converted gate is the defect this file guards against
(``partial-gate-patch-left-as-artefact``), so the first test is structural: the set of
checks that read parent bands is enumerated, and each of them reads them through the one
represented-band helper and never straight off ``parent.eigenvalues_ev``.

Fixed by the plan and asserted here: the honour rule (an isolated block below every
retained band, more than the gate's band-matching tolerance -- 1 % of the full parent's
occupied bandwidth -- apart) and every threshold. A deck that excludes nothing gets
today's result byte for byte (the golden in ``test_gate_excluded_bands.py`` stays green,
and the represented helper returns the very array the checks read before).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from autowan import core_checks
from autowan.core_checks import (
    CORE_CHECKS,
    _band_distance_from_parent,
    _frontier_manifold_straddles_fermi_level,
    _model_bands_correspond_to_parent_bands,
    _occupied_manifold_is_represented,
)
from autowan.gate import CertificationGate, Construction, parent_bands_for_check
from autowan.inputs import read_construction
from autowan.qe_outputs import ParentCalculation

#: A cubic cell whose lattice vectors equal alat, so fractional and Cartesian k agree.
A = 4.0
NK = 8
KFRAC = np.array([[i / NK, 0.0, 0.0] for i in range(NK)])

#: Every core check that reads the parent's eigenvalues, enumerated by reading the gate
#: module (2026-10-03). The threshold scale ``occupied_bandwidth_ev`` also reads them; it
#: is a threshold, fixed by the plan on the full parent, and is not a check.
PARENT_BAND_CHECKS = (
    "frontier_manifold_straddles_fermi_level",
    "occupied_manifold_is_represented",
    "model_bands_correspond_to_parent_bands",
    "band_distance_from_parent",
)

EVALUATORS = {
    "frontier_manifold_straddles_fermi_level": _frontier_manifold_straddles_fermi_level,
    "occupied_manifold_is_represented": _occupied_manifold_is_represented,
    "model_bands_correspond_to_parent_bands": _model_bands_correspond_to_parent_bands,
    "band_distance_from_parent": _band_distance_from_parent,
}

HONOURED = "excluded by the deck and not counted"
NOT_HONOURED = "not honoured"


# ---------------------------------------------------------------------------
# fixtures: a QE-shaped parent and a diagonal nearest-neighbour model
# ---------------------------------------------------------------------------


def _band(onsite: float, hop: float) -> np.ndarray:
    """``onsite + 2 hop cos(2 pi k)`` on the fixture mesh -- one nearest-neighbour band."""
    return onsite + 2.0 * hop * np.cos(2 * np.pi * KFRAC[:, 0])


def _parent(bands: list[np.ndarray], fermi: float = 0.0) -> ParentCalculation:
    eig = np.stack(bands, axis=1)[None, :, :]  # (spin, k, band)
    return ParentCalculation(
        eigenvalues_ev=eig,
        occupations=(eig <= fermi).astype(float),
        kpoints_cart=KFRAC.copy(),
        fermi_energy_ev=fermi,
        lattice_ang=np.eye(3) * A,
        alat_ang=A,
        species=("X",),
        positions_frac=np.zeros((1, 3)),
        n_symmetries=48,
        n_atomic_wfc=None,
        converged=None,
        calculation="nscf",
    )


class _Run:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _model(levels) -> _Run:
    """A diagonal nearest-neighbour model, with a zero second shell so it decays."""
    n = len(levels)
    rvec = np.array([[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]])
    h = np.zeros((5, n, n), dtype=complex)
    for i, (onsite, hop) in enumerate(levels):
        h[2, i, i] = onsite
        h[1, i, i] = h[3, i, i] = hop
    return _Run(
        num_wann=n,
        num_bands=n,
        centres_ang=np.zeros((n, 3)),
        spreads_ang2=np.ones(n),
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=_Run(converged=True, n_iterations=10, final_delta=1e-11),
        hr_rvectors=rvec,
        hr_degeneracies=np.ones(5),
        hr_matrix=h,
    )


def _win(path: Path, num_wann: int, extra: str = "") -> Path:
    """An aiida-wannier90 deck as the writer emits it: sorted keys, ``mp_grid = n, n, n``."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "aiida.win").write_text(
        "auto_projections = .true.\n"
        "conv_window = 3\n"
        "dis_froz_max =   2.0000000000d+00\n"
        "dis_froz_proj = .true.\n"
        "dis_proj_max =   9.5000000000d-01\n"
        "dis_proj_min =   1.0000000000d-02\n"
        f"{extra}"
        "fermi_energy =   0.0000000000d+00\n"
        f"mp_grid = {NK}, 1, 1\n"
        f"num_bands = {num_wann}\n"
        f"num_wann = {num_wann}\n"
        "\nbegin atoms_cart\nang\nX 0.0 0.0 0.0\nend atoms_cart\n"
    )
    return path


def _construction(parent, model, run_dir: Path, material_class: str) -> Construction:
    return Construction(
        "fixture",
        parent=parent,
        wannier=model,
        material_class=material_class,
        inputs=read_construction(run_dir, "aiida").gate_inputs(0),
    )


SEMICORE = -30.0

#: Sodium-like: one nearly-free-electron band that is empty over part of the zone
#: (0.5 - 2 cos 2 pi k is below E_F = 0 at 3 of the 8 k-points) and a conduction band.
NA_LIKE_LEVELS = ((0.5, -1.0), (5.0, 0.5))


def _na_like(tmp_path: Path, excluded: bool) -> Construction:
    parent = _parent([np.full(NK, SEMICORE)] + [_band(o, t) for o, t in NA_LIKE_LEVELS])
    extra = "exclude_bands = 1\n" if excluded else ""
    return _construction(
        parent, _model(NA_LIKE_LEVELS), _win(tmp_path, 2, extra), "metal"
    )


#: TiO2-like: two valence bands, a gap, a conduction band, under one semicore band.
TIO2_LIKE_LEVELS = ((-5.0, -0.5), (-2.0, 0.5), (3.0, 0.5))


def _tio2_like(tmp_path: Path, excluded: bool, model_levels=TIO2_LIKE_LEVELS):
    parent = _parent(
        [np.full(NK, SEMICORE)] + [_band(o, t) for o, t in TIO2_LIKE_LEVELS]
    )
    extra = "exclude_bands = 1\n" if excluded else ""
    return _construction(
        parent,
        _model(model_levels),
        _win(tmp_path, len(model_levels), extra),
        "gapped",
    )


def _sro_like(tmp_path: Path, excluded: bool) -> Construction:
    """The SrO pattern: the excluded pair overlaps the lowest retained band (not isolated).

    Band 2 reaches -15.64 eV and band 3 starts at -16.0 eV: they overlap by 0.36 eV, as on
    the stored SrO parent, so the exclusion is not honoured at any tolerance. (The earlier
    0.2 eV gap was refused only by a tolerance of 1 % of the full ~33 eV occupied spread;
    the semicore-independent scale of the VASP gate integration gives ~0.16 eV, which
    would honour it.)
    """
    deep, low, high = np.full(NK, -34.0), _band(-15.84, 0.1), _band(-15.6, 0.2)
    parent = _parent([deep, low, high, *[_band(o, t) for o, t in TIO2_LIKE_LEVELS]])
    levels = ((-15.6, 0.2), *TIO2_LIKE_LEVELS)
    extra = "exclude_bands = 1-2\n" if excluded else ""
    return _construction(parent, _model(levels), _win(tmp_path, 4, extra), "gapped")


# ---------------------------------------------------------------------------
# the enumeration: every parent-band check, and only through the helper
# ---------------------------------------------------------------------------


def test_the_parent_band_checks_are_enumerated_in_the_gate():
    assert tuple(core_checks.PARENT_BAND_CHECKS) == PARENT_BAND_CHECKS
    core_ids = {c.check_id for c in CORE_CHECKS}
    assert set(PARENT_BAND_CHECKS) <= core_ids


def test_no_core_check_reads_parent_eigenvalues_except_through_the_represented_helper():
    # A check that reads c.parent.eigenvalues_ev directly is a check the exclusion never
    # reaches -- the partial gate patch of 2026-10-01. Every core check's evaluator is
    # read; the parent-band ones must go through parent_bands_for_check, the rest must
    # not touch the parent's eigenvalues at all.
    for check in CORE_CHECKS:
        source = inspect.getsource(check.evaluate)
        assert "eigenvalues_ev" not in source, check.check_id
        reads_helper = "parent_bands_for_check" in source
        assert reads_helper == (check.check_id in PARENT_BAND_CHECKS), check.check_id


def test_the_helper_returns_the_very_array_the_checks_read_when_nothing_is_excluded(
    tmp_path,
):
    c = _tio2_like(tmp_path, excluded=False)
    bands, note = parent_bands_for_check(c, 0)
    assert bands is c.parent.eigenvalues_ev[0] or np.shares_memory(
        bands, c.parent.eigenvalues_ev
    )
    assert np.array_equal(bands, c.parent.eigenvalues_ev[0])
    assert note == ""


# ---------------------------------------------------------------------------
# sodium-like: the frontier check
# ---------------------------------------------------------------------------


def test_frontier_counts_the_excluded_semicore_without_an_exclusion(tmp_path):
    outcome = _frontier_manifold_straddles_fermi_level(_na_like(tmp_path, False))
    assert outcome.passed is False
    assert outcome.metric_value == pytest.approx(3 / 8)


def test_frontier_reads_only_the_represented_bands_when_the_exclusion_is_honoured(
    tmp_path,
):
    outcome = _frontier_manifold_straddles_fermi_level(_na_like(tmp_path, True))
    assert outcome.passed is True
    assert outcome.metric_value == 1.0
    assert HONOURED in outcome.note
    assert not outcome.note.startswith(";")


def test_the_sodium_like_construction_certifies_only_with_the_exclusion(tmp_path):
    without = CertificationGate(CORE_CHECKS).run(_na_like(tmp_path / "a", False))
    with_ = CertificationGate(CORE_CHECKS).run(_na_like(tmp_path / "b", True))
    # Without the exclusion both parent-band checks that see the semicore refuse it.
    assert set(without.failed_check_ids) == {
        "frontier_manifold_straddles_fermi_level",
        "occupied_manifold_is_represented",
    }
    assert with_.passed, with_.failed_check_ids + with_.non_evaluable_check_ids


# ---------------------------------------------------------------------------
# TiO2-like: band distance
# ---------------------------------------------------------------------------


def test_band_distance_pairs_the_semicore_with_the_model_without_an_exclusion(
    tmp_path,
):
    c = _tio2_like(tmp_path, False)
    report = CertificationGate(CORE_CHECKS).run(c)
    row = report.measurements()["band_distance_from_parent"]
    assert row["value"] > row["threshold"]
    assert "band_distance_from_parent" in report.failed_check_ids


def test_band_distance_reads_only_the_represented_bands_when_honoured(tmp_path):
    c = _tio2_like(tmp_path, True)
    outcome = _band_distance_from_parent(c)
    assert outcome.metric_value < 1e-9
    assert HONOURED in outcome.note
    # The window is the check's own definition applied to the represented bands: lowest
    # represented occupied state (-6 eV) to E_F + 10 % of their occupied bandwidth (5 eV).
    assert "-6.00..0.50 eV" in outcome.note


def test_the_tio2_like_construction_certifies_with_the_exclusion(tmp_path):
    report = CertificationGate(CORE_CHECKS).run(_tio2_like(tmp_path, True))
    assert report.passed, report.failed_check_ids + report.non_evaluable_check_ids


@pytest.mark.parametrize("make", [_na_like, _tio2_like, _sro_like])
def test_every_threshold_is_unchanged_by_the_exclusion(tmp_path, make):
    without = CertificationGate(CORE_CHECKS).run(make(tmp_path / "a", False))
    with_ = CertificationGate(CORE_CHECKS).run(make(tmp_path / "b", True))
    for check in CORE_CHECKS:
        assert (
            without.measurements()[check.check_id]["threshold"]
            == with_.measurements()[check.check_id]["threshold"]
        ), check.check_id


# ---------------------------------------------------------------------------
# the model-band correspondence check: its window starts at the represented manifold
# ---------------------------------------------------------------------------


def test_model_band_window_starts_at_the_bottom_of_the_represented_bands(tmp_path):
    # Without an exclusion the reproduction window's lower edge is the parent's lowest
    # band (the semicore, -30 eV); with it honoured, the lowest represented band (-6 eV).
    without = _model_bands_correspond_to_parent_bands(_tio2_like(tmp_path / "a", False))
    with_ = _model_bands_correspond_to_parent_bands(_tio2_like(tmp_path / "b", True))
    assert without.record["reproduction_window_ev"][0] == SEMICORE
    assert with_.record["reproduction_window_ev"][0] == pytest.approx(-6.0)
    assert HONOURED in with_.note


def test_a_model_band_on_an_excluded_semicore_level_is_not_a_partner_of_anything(
    tmp_path,
):
    # A model that carries a band on the excluded semicore level: without the exclusion
    # the semicore is a parent state and the band is matched; with it honoured the band
    # lies below the represented manifold, outside the window the check judges.
    levels = ((SEMICORE, 0.0), *TIO2_LIKE_LEVELS)
    without = _model_bands_correspond_to_parent_bands(
        _tio2_like(tmp_path / "a", False, levels)
    )
    with_ = _model_bands_correspond_to_parent_bands(
        _tio2_like(tmp_path / "b", True, levels)
    )
    assert without.metric_value == 0.0
    assert "0 of " in with_.note
    n_without = int(without.note.split(" of ")[1].split()[0])
    n_with = int(with_.note.split(" of ")[1].split()[0])
    assert n_without - n_with == NK  # the eight semicore-level model states drop out


# ---------------------------------------------------------------------------
# one honour decision per construction, shared by every parent-band check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, material_class, expected",
    [
        (_na_like, "metal", HONOURED),
        (_tio2_like, "gapped", HONOURED),
        (_sro_like, "gapped", NOT_HONOURED),
    ],
)
def test_every_parent_band_check_takes_the_same_honour_decision(
    tmp_path, make, material_class, expected
):
    c = make(tmp_path, True)
    assert c.material_class == material_class
    for check_id, evaluate in EVALUATORS.items():
        if material_class not in next(
            k.applies_to for k in CORE_CHECKS if k.check_id == check_id
        ):
            continue
        assert expected in evaluate(c).note, check_id


# ---------------------------------------------------------------------------
# the SrO pattern: a non-isolated exclusion leaves every check on all bands
# ---------------------------------------------------------------------------


def test_a_non_isolated_exclusion_leaves_every_parent_band_check_on_all_bands(
    tmp_path,
):
    declared = _sro_like(tmp_path / "a", True)
    literal = _sro_like(tmp_path / "b", False)
    for check_id, evaluate in EVALUATORS.items():
        got, want = evaluate(declared), evaluate(literal)
        assert got.metric_value == want.metric_value, check_id
        assert got.passed == want.passed, check_id
        assert got.note.startswith(want.note), check_id
        assert NOT_HONOURED in got.note, check_id
    with_ = CertificationGate(CORE_CHECKS).run(declared)
    without = CertificationGate(CORE_CHECKS).run(literal)
    assert with_.failed_check_ids == without.failed_check_ids
    assert "band_distance_from_parent" in with_.failed_check_ids


# ---------------------------------------------------------------------------
# no exclusion: byte-identical records, every check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_na_like, _tio2_like, _sro_like])
def test_without_an_exclusion_no_parent_band_check_carries_an_exclusion_note(
    tmp_path, make
):
    c = make(tmp_path, False)
    for check_id, evaluate in EVALUATORS.items():
        note = evaluate(c).note
        assert "exclude" not in note and "excluded" not in note, check_id


def test_an_exclusion_of_unoccupied_bands_only_changes_nothing(tmp_path):
    # Excluding a band with no occupied state cannot change what any parent-band check
    # sees: the represented set is the full set and the note is empty.
    parent = _parent([_band(o, t) for o, t in TIO2_LIKE_LEVELS] + [_band(8.0, 0.5)])
    plain = _construction(
        parent, _model(TIO2_LIKE_LEVELS), _win(tmp_path / "a", 3), "gapped"
    )
    declared = _construction(
        parent,
        _model(TIO2_LIKE_LEVELS),
        _win(tmp_path / "b", 3, "exclude_bands = 4\n"),
        "gapped",
    )
    for check_id, evaluate in EVALUATORS.items():
        got, want = evaluate(declared), evaluate(plain)
        assert (got.passed, got.metric_value, got.note, got.record) == (
            want.passed,
            want.metric_value,
            want.note,
            want.record,
        ), check_id
