"""The gate is told which bands a deck excluded, and never lets an exclusion redefine the target.

The 2025 frontier construction excludes the pseudopotential's semicore shells: pw2wannier90
drops their projectors and Wannier90 drops the lowest bands (``exclude_bands``). A model
built that way is not asked to represent those bands, and a gate that counts them against
it refuses every such construction by convention rather than by construction. So the
gate reads ``exclude_bands`` from the deck and ``occupied_manifold_is_represented`` counts
only the bands the model is asked to represent.

The other half is the adversarial control. A deck can exclude any band it likes, and if the
gate honoured every declaration a construction could drop a valence band it failed to
reproduce and be certified for it. The gate honours an exclusion only when the excluded
bands are an isolated block below everything the model represents -- the highest excluded
eigenvalue lies below the lowest retained one by more than the check's own matching
tolerance. An exclusion that reaches into the represented manifold is not honoured, and the
check counts every occupied state exactly as it did before.

A deck that excludes nothing -- or excludes only bands with no occupied state -- receives a
verdict byte-identical to the one ``autowan/dev`` gave before this change (the golden file
was written by that code, ``tests/fixtures/gate_regression/README.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from autowan.core_checks import (
    CORE_CHECKS,
    PRE_PROFILE_CORE_CHECKS,
    _occupied_manifold_is_represented,
)
from autowan.gate import CertificationGate, Construction, NonEvaluable
from autowan.inputs import InputError, read_construction
from autowan.qe_outputs import ParentCalculation

GOLDEN = (
    Path(__file__).parent / "fixtures" / "gate_regression" / "no_exclusion_reports.json"
)
SYNTHETIC_VASP = Path(__file__).parent / "fixtures" / "vasp" / "synthetic"

#: A cubic cell whose lattice vectors equal alat, so fractional and Cartesian k agree.
A = 4.0
NK = 8
KFRAC = np.array([[i / NK, 0.0, 0.0] for i in range(NK)])


# ---------------------------------------------------------------------------
# a QE-shaped construction: semicore + two valence bands + one conduction band
# ---------------------------------------------------------------------------


def _band(onsite: float, hop: float) -> np.ndarray:
    """``onsite + 2 hop cos(2 pi k)`` on the fixture mesh -- one nearest-neighbour band."""
    return onsite + 2.0 * hop * np.cos(2 * np.pi * KFRAC[:, 0])


#: The bands the model reproduces (onsite, nearest-neighbour hopping), lowest first.
VALENCE_AND_CONDUCTION = ((-5.0, -0.5), (-2.0, 0.5), (3.0, 0.5))


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


def _model(levels=VALENCE_AND_CONDUCTION) -> _Run:
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


def _win(tmp_path: Path, num_wann: int, extra: str = "") -> Path:
    """An aiida-wannier90 deck as the writer emits it: sorted keys, ``mp_grid = n, n, n``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "aiida.win").write_text(
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
    return tmp_path


def _construction(parent, model, run_dir: Path, material_class="metal") -> Construction:
    inputs = read_construction(run_dir, "aiida").gate_inputs(0)
    return Construction(
        "fixture",
        parent=parent,
        wannier=model,
        material_class=material_class,
        inputs=inputs,
    )


def _semicore_parent() -> ParentCalculation:
    """A flat semicore band 25 eV below the valence, then the three model bands."""
    return _parent(
        [np.full(NK, -30.0)] + [_band(o, t) for o, t in VALENCE_AND_CONDUCTION]
    )


# ---------------------------------------------------------------------------
# the deck tells the gate which bands it excluded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("exclude_bands = 1-4", (1, 2, 3, 4)),  # aiida-wannier90's grouped string
        ("exclude_bands = 1-3,5", (1, 2, 3, 5)),
        ("exclude_bands = 1:4", (1, 2, 3, 4)),  # Wannier90 also reads ':' ranges
        ("exclude_bands : 1-2; 7", (1, 2, 7)),
        ("exclude_bands = 3", (3,)),
        ("EXCLUDE_BANDS = 1 2 3 ! a comment", (1, 2, 3)),
    ],
)
def test_the_gate_inputs_carry_the_excluded_bands(tmp_path, spelling, expected):
    run = _win(tmp_path, 3, extra=f"{spelling}\n")
    assert read_construction(run, "aiida").gate_inputs(0)["exclude_bands"] == expected


def test_a_deck_that_excludes_nothing_carries_no_exclusion(tmp_path):
    run = _win(tmp_path, 3)
    assert "exclude_bands" not in read_construction(run, "aiida").gate_inputs(0)


def test_an_unreadable_exclusion_is_an_input_error_not_a_silent_empty_list(tmp_path):
    run = _win(tmp_path, 3, extra="exclude_bands = 1-x\n")
    with pytest.raises(InputError):
        read_construction(run, "aiida").gate_inputs(0)


# ---------------------------------------------------------------------------
# semicore exclusion flips the occupied-manifold check (the QE fixture)
# ---------------------------------------------------------------------------


def test_semicore_counted_against_a_model_never_asked_for_it_fails_the_check(tmp_path):
    # The deck as autowan/dev saw it: nothing excluded, so the semicore band is an
    # occupied state with no model band within 1 % of the occupied bandwidth.
    c = _construction(_semicore_parent(), _model(), _win(tmp_path, 3))
    report = CertificationGate(CORE_CHECKS).run(c)
    assert "occupied_manifold_is_represented" in report.failed_check_ids
    assert report.measurements()["occupied_manifold_is_represented"][
        "value"
    ] == pytest.approx(1 / 3)


def test_with_the_semicore_excluded_the_check_counts_only_the_represented_bands(
    tmp_path,
):
    c = _construction(
        _semicore_parent(), _model(), _win(tmp_path, 3, extra="exclude_bands = 1\n")
    )
    outcome = _occupied_manifold_is_represented(c)
    assert outcome.metric_value == 0.0
    assert "0 of 16" in outcome.note  # two represented occupied bands x 8 k-points
    assert "excluded" in outcome.note
    report = CertificationGate(CORE_CHECKS).run(c)
    assert report.passed, report.failed_check_ids + report.non_evaluable_check_ids


def test_the_tolerance_and_threshold_are_unchanged_by_the_exclusion(tmp_path):
    # Thresholds are fixed: 1 % of the parent's full occupied bandwidth, semicore
    # included, before and after. Only the set of states counted changes.
    without = CertificationGate(CORE_CHECKS).run(
        _construction(_semicore_parent(), _model(), _win(tmp_path / "a", 3))
    )
    with_ = CertificationGate(CORE_CHECKS).run(
        _construction(
            _semicore_parent(),
            _model(),
            _win(tmp_path / "b", 3, extra="exclude_bands = 1\n"),
        )
    )
    for check in (
        "occupied_manifold_is_represented",
        "model_bands_correspond_to_parent_bands",
    ):
        assert (
            without.measurements()[check]["threshold"]
            == with_.measurements()[check]["threshold"]
        )


# ---------------------------------------------------------------------------
# the adversarial control: excluding a band the model must represent never certifies
# ---------------------------------------------------------------------------


def _connected_valence_parent() -> ParentCalculation:
    """Two valence bands that touch at the zone boundary, and a conduction band.

    Band 1 runs -6 .. -4 eV and band 2 -4 .. -2 eV: they meet at k = 1/2, so neither can
    be removed without cutting the occupied manifold the model is asked to represent.
    """
    return _parent([_band(-5.0, -0.5), _band(-3.0, 0.5), _band(3.0, 0.5)])


def test_excluding_a_valence_band_inside_the_target_window_does_not_certify(tmp_path):
    # The model reproduces only bands 2 and 3 and the deck declares band 1 excluded. If
    # the gate honoured the declaration it would count bands 2 and 3 and certify.
    model = _model(((-3.0, 0.5), (3.0, 0.5)))
    c = _construction(
        _connected_valence_parent(),
        model,
        _win(tmp_path, 2, extra="exclude_bands = 1\n"),
    )
    outcome = _occupied_manifold_is_represented(c)
    assert "not honoured" in outcome.note
    report = CertificationGate(CORE_CHECKS).run(c)
    assert not report.passed
    assert "occupied_manifold_is_represented" in report.failed_check_ids


def test_excluding_a_band_above_one_the_model_represents_does_not_certify(tmp_path):
    # Band 2 of the semicore parent is a valence band with band 1 (semicore) retained
    # below it: an exclusion in the middle of the occupied manifold is never isolated
    # below the represented bands.
    model = _model(((-30.0, 0.0), (-2.0, 0.5), (3.0, 0.5)))
    c = _construction(
        _semicore_parent(), model, _win(tmp_path, 3, extra="exclude_bands = 2\n")
    )
    report = CertificationGate(CORE_CHECKS).run(c)
    assert not report.passed
    assert "occupied_manifold_is_represented" in report.failed_check_ids


def test_a_builder_exclusion_cutting_a_band_complex_is_not_honoured(tmp_path):
    # The SrO pattern measured on the stored parent: the lowest-N rule cuts a complex
    # whose top excluded band overlaps the lowest retained one (by 0.36 eV there). The
    # fixture overlaps by the same 0.36 eV: its earlier 0.20 eV gap relied on a matching
    # tolerance taken from the full occupied spread, which the semicore-independent scale
    # (the VASP gate integration) narrows to 1 % of the block above the -34 eV shell.
    deep, low, high = np.full(NK, -34.0), _band(-15.84, 0.1), _band(-15.6, 0.2)
    parent = _parent(
        [deep, low, high, *[_band(o, t) for o, t in VALENCE_AND_CONDUCTION]]
    )
    model = _model(((-15.6, 0.2), *VALENCE_AND_CONDUCTION))
    c = _construction(parent, model, _win(tmp_path, 4, extra="exclude_bands = 1-2\n"))
    outcome = _occupied_manifold_is_represented(c)
    assert "not honoured" in outcome.note
    assert outcome.metric_value > 0.05


def test_excluding_every_occupied_band_leaves_nothing_to_certify(tmp_path):
    # An isolated block, honoured -- but then no occupied state is left to represent,
    # which is non-evaluable, and non-evaluable is not a pass.
    parent = _parent([_band(-5.0, -0.2), _band(3.0, 0.5)])
    c = _construction(
        parent, _model(((3.0, 0.5),)), _win(tmp_path, 1, extra="exclude_bands = 1\n")
    )
    with pytest.raises(NonEvaluable):
        _occupied_manifold_is_represented(c)
    assert not CertificationGate(CORE_CHECKS).run(c).passed


# ---------------------------------------------------------------------------
# regression: no exclusion -> byte-identical to autowan/dev
# ---------------------------------------------------------------------------


def _report_record(c: Construction) -> dict:
    # The golden pins the gate before profiles; PRE_PROFILE_CORE_CHECKS is that gate (the
    # default with centres_sit_on_atomic_sites back in the drift guard's place), so an
    # unchanged golden shows the profile change moved nothing else.
    report = CertificationGate(PRE_PROFILE_CORE_CHECKS).run(c)
    return {
        "inputs": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in c.inputs.items()
        },
        "passed": report.passed,
        "failed": list(report.failed_check_ids),
        "non_evaluable": list(report.non_evaluable_check_ids),
        "coverage": report.coverage,
        "measurements": report.measurements(),
    }


def _regression_cases(tmp_path: Path) -> dict[str, Construction]:
    """Constructions whose decks exclude nothing, or nothing with an occupied state."""
    from autowan.backends import get_backend

    semicore = _semicore_parent()
    vasp_parent = get_backend("vasp").read_parent(SYNTHETIC_VASP / "vasprun.xml")
    vasp_model = _Run(
        num_wann=2,
        num_bands=vasp_parent.n_bands,
        centres_ang=np.zeros((2, 3)),
        spreads_ang2=np.array([1.0, 1.1]),
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=_Run(converged=True, n_iterations=10, final_delta=1e-11),
        hr_rvectors=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        hr_degeneracies=np.array([1, 1, 1]),
        hr_matrix=np.array(
            [
                [[-6.0, 0.0], [0.0, 5.0]],
                [[0.01, 0.0], [0.0, 0.01]],
                [[0.01, 0.0], [0.0, 0.01]],
            ],
            dtype=complex,
        ),
    )
    vasp_dir = tmp_path / "vasp"
    vasp_dir.mkdir(parents=True)
    (vasp_dir / "aiida.win").write_text(
        "num_wann = 2\nmp_grid = 2 1 1\nconv_window = 3\n"
    )
    return {
        "semicore_counted_metal": _construction(
            semicore, _model(), _win(tmp_path / "a", 3)
        ),
        "semicore_counted_gapped": _construction(
            semicore, _model(), _win(tmp_path / "b", 3), material_class="gapped"
        ),
        "no_semicore_metal": _construction(
            _parent([_band(o, t) for o, t in VALENCE_AND_CONDUCTION]),
            _model(),
            _win(tmp_path / "c", 3),
        ),
        "connected_valence_missing_band": _construction(
            _connected_valence_parent(),
            _model(((-3.0, 0.5), (3.0, 0.5))),
            _win(tmp_path / "d", 2),
        ),
        # An exclusion that touches no occupied state cannot change this check.
        "unoccupied_band_excluded": _construction(
            _parent(
                [_band(o, t) for o, t in VALENCE_AND_CONDUCTION] + [_band(8.0, 0.5)]
            ),
            _model(),
            _win(tmp_path / "e", 3, extra="exclude_bands = 4\n"),
        ),
        "vasp_synthetic_parent": _construction(
            vasp_parent, vasp_model, vasp_dir, material_class="gapped"
        ),
    }


TARGET_WINDOW_RECORD_KEYS = (
    "target_window_source",
    "target_window_ev",
    "reproduction_window_ev",
    "deck_frozen_window_ev",
)


def _dump(records: dict) -> str:
    return json.dumps(records, indent=1, sort_keys=True, default=float) + "\n"


def test_a_deck_that_excludes_no_occupied_band_gets_the_autowan_dev_verdict(tmp_path):
    records = {}
    for name, c in _regression_cases(tmp_path).items():
        record = _report_record(c)
        record["inputs"].pop("exclude_bands", None)  # new key, absent on autowan/dev
        # the deck's projectors, read for d_only_downfolding (v6): a new input key
        record["inputs"].pop("projector_groups", None)
        # Four record keys the target-window gate change (autowan/feat/gate-reproduction-
        # window-from-target) adds to this one check, additively; every verdict, value,
        # threshold and other key is unchanged. Absent when the golden was written.
        band = record["measurements"].get("model_bands_correspond_to_parent_bands", {})
        for key in TARGET_WINDOW_RECORD_KEYS:
            band.pop(key, None)
        records[name] = record
    assert _dump(records) == GOLDEN.read_text()


if __name__ == "__main__":  # pragma: no cover - writes the golden file
    import tempfile

    if sys.argv[1:] != ["--write-golden"]:
        raise SystemExit("usage: test_gate_excluded_bands.py --write-golden")
    with tempfile.TemporaryDirectory() as d:
        cases = _regression_cases(Path(d))
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(_dump({n: _report_record(c) for n, c in cases.items()}))
    print(f"wrote {GOLDEN}")
