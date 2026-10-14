"""A declared magnetic-subspace certification target, beside the default whole-manifold one.

The default target holds a model to every occupied state of its parent, which is autowan's
stated design position and is not changed here. A d-only model of the magnetic manifold, built for
an exchange calculation, fails it by construction whenever the parent carries semicore or ligand
states. ``magnetic_subspace`` is an *added* target, chosen when the construction is declared:
under it, ``occupied_manifold_is_represented`` counts only the parent states inside the declared
target window whose magnetic-d weight clears a fixed floor, and measures a partner against a
tolerance scaled by that manifold's own bandwidth. Nothing else about the gate changes with the
target, and an undeclared construction keeps the default.

The magnetic-d weight of each parent state is supplied at declaration, computed by the backend
that can read it (for OpenMX, the projectability of the state onto the magnetic sites' d orbitals);
the core gate never imports a backend.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autowan
from autowan import gate as G
from autowan.backends import gate_for
from autowan.core_checks import CORE_CHECKS
from autowan.gate import (
    SCALES,
    CertificationGate,
    Construction,
    GateDefinitionError,
    NonEvaluable,
    ScaleAwareThreshold,
)

_SPEC = importlib.util.spec_from_file_location(
    "analytic_constructions",
    Path(__file__).parent / "fixtures" / "analytic" / "analytic_constructions.py",
)
ac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ac)

COVERAGE = "occupied_manifold_is_represented"
WINDOW = (
    -4.0,
    2.0,
)  # relative to E_F: every d state (-3 .. +0.5 eV) and part of the s band


def _target(weight=None, window=WINDOW):
    return G.magnetic_subspace_target(
        window_ev=window,
        magnetic_d_weight=ac.MAGNETIC_D_WEIGHT if weight is None else weight,
        weight_source="analytic: 0.9 on the d band, 0.1 on the s band, 0 on the semicore level",
    )


def _verdict(construction, check_id=COVERAGE):
    report = CertificationGate(CORE_CHECKS).run(construction)
    return report, {v.check_id: v for v in report.verdicts}[check_id]


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------


def test_the_two_targets_are_named_and_an_undeclared_construction_keeps_the_default():
    assert G.CERTIFICATION_TARGETS == ("whole_occupied_manifold", "magnetic_subspace")
    c = ac.construction(ac.semicore_s_d_parent(), ac.D_ONLY_MODEL)
    assert c.target is G.WHOLE_OCCUPIED_MANIFOLD
    assert c.target.name == "whole_occupied_manifold"
    report, _ = _verdict(c)
    assert report.target == "whole_occupied_manifold"


def test_a_magnetic_subspace_declaration_is_refused_unless_it_is_complete():
    good = dict(
        window_ev=WINDOW, magnetic_d_weight=ac.MAGNETIC_D_WEIGHT, weight_source="stated"
    )
    assert G.magnetic_subspace_target(**good).name == "magnetic_subspace"
    with pytest.raises(GateDefinitionError):
        G.CertificationTarget("somewhere_else")
    with pytest.raises(GateDefinitionError):
        G.CertificationTarget("magnetic_subspace")  # no window, no weights
    # A whole-manifold target may declare a reproduction window (autowan/dev, the
    # target-window gate change), but never magnetic weights.
    assert G.CertificationTarget(
        "whole_occupied_manifold", window_ev=(-1.0, 1.0)
    ).window_ev == (-1.0, 1.0)
    with pytest.raises(GateDefinitionError):
        G.CertificationTarget(
            "whole_occupied_manifold", magnetic_d_weight=ac.MAGNETIC_D_WEIGHT
        )
    with pytest.raises(GateDefinitionError):
        G.magnetic_subspace_target(**{**good, "window_ev": (2.0, -4.0)})
    with pytest.raises(GateDefinitionError):
        G.magnetic_subspace_target(**{**good, "window_ev": (float("nan"), 1.0)})
    with pytest.raises(GateDefinitionError):
        G.magnetic_subspace_target(
            **{**good, "magnetic_d_weight": -ac.MAGNETIC_D_WEIGHT}
        )
    with pytest.raises(GateDefinitionError):
        G.magnetic_subspace_target(**{**good, "weight_source": "  "})


def test_the_weight_floor_and_the_tolerance_are_scale_aware_and_calibrated():
    from autowan.core_checks import MAGNETIC_SUBSPACE_TOLERANCE

    floor = G.MAGNETIC_D_WEIGHT_FLOOR
    assert isinstance(floor, ScaleAwareThreshold)
    assert (floor.fraction, floor.scale) == (0.5, "occupied_state_fraction")
    assert len(floor.calibration.strip()) > 80
    assert isinstance(MAGNETIC_SUBSPACE_TOLERANCE, ScaleAwareThreshold)
    assert (
        MAGNETIC_SUBSPACE_TOLERANCE.fraction,
        MAGNETIC_SUBSPACE_TOLERANCE.scale,
    ) == (
        0.01,
        "magnetic_manifold_bandwidth_ev",
    )
    assert len(MAGNETIC_SUBSPACE_TOLERANCE.calibration.strip()) > 80
    assert "magnetic_manifold_bandwidth_ev" in SCALES


def test_the_magnetic_manifold_bandwidth_is_the_spread_of_the_counted_states():
    parent = ac.semicore_s_d_parent()
    c = ac.construction(parent, ac.D_ONLY_MODEL, target=_target())
    mask = G.magnetic_subspace_mask(c, 0)
    assert mask.shape == parent.eigenvalues_ev[0].shape
    assert (
        mask[:, 2].all() and not mask[:, :2].any()
    )  # every d state; no s, no semicore
    assert SCALES["magnetic_manifold_bandwidth_ev"](c) == pytest.approx(
        ac.D_MANIFOLD_BANDWIDTH_EV
    )
    undeclared = ac.construction(parent, ac.D_ONLY_MODEL)
    with pytest.raises(NonEvaluable):
        SCALES["magnetic_manifold_bandwidth_ev"](undeclared)


# ---------------------------------------------------------------------------
# a d-only model of a semicore-bearing parent, under both targets
# ---------------------------------------------------------------------------


def test_a_d_only_model_is_refused_under_the_default_target():
    c = ac.construction(ac.semicore_s_d_parent(), ac.D_ONLY_MODEL)
    _, v = _verdict(c)
    # 8 semicore + 8 s + 7 occupied d states; the model reproduces only the 7 d states
    assert v.metric_value == pytest.approx(16 / 23)
    assert v.passed is False


def test_the_same_d_only_model_passes_coverage_under_the_magnetic_subspace_target():
    c = ac.construction(ac.semicore_s_d_parent(), ac.D_ONLY_MODEL, target=_target())
    report, v = _verdict(c)
    assert report.target == "magnetic_subspace"
    assert v.applies and v.evaluable
    assert v.metric_value == pytest.approx(0.0)
    assert v.passed is True
    assert "magnetic" in v.note


@pytest.mark.parametrize(
    "model, expected",
    [
        (ac.D_ONLY_MODEL_SHIFTED, 1.0),  # 0.3 eV off, against a 35 meV tolerance
        (ac.D_ONLY_MODEL_FLAT, 6 / 8),  # meets the d band at k = 1/4 and 3/4 only
    ],
)
def test_a_defective_d_only_model_is_still_refused_under_the_magnetic_subspace_target(
    model, expected
):
    c = ac.construction(ac.semicore_s_d_parent(), model, target=_target())
    _, v = _verdict(c)
    assert v.evaluable
    assert v.metric_value == pytest.approx(expected)
    assert v.passed is False


def test_the_declared_window_not_the_fermi_level_bounds_the_magnetic_manifold():
    """A model exact at the seven occupied d states and 1 eV off at the one empty one.

    Counting only occupied states, it would pass; the declared window runs to +2 eV, so the
    unoccupied d state at +0.5 eV belongs to the manifold and is missed: 1 of 8 > 5 %.
    """
    values = ac.d_band().copy()
    values[4] += 1.0  # k = 1/2, the only unoccupied d state
    c = ac.construction(
        ac.semicore_s_d_parent(), ac.hr_from_band(values), target=_target()
    )
    _, v = _verdict(c)
    assert v.metric_value == pytest.approx(1 / 8)
    assert v.passed is False
    # a declarer who wants only the occupied part says so with the window
    c_occ = ac.construction(
        ac.semicore_s_d_parent(),
        ac.hr_from_band(values),
        target=_target(window=(-4.0, 0.0)),
    )
    _, v_occ = _verdict(c_occ)
    assert v_occ.metric_value == pytest.approx(0.0)
    assert v_occ.passed is True


def test_no_counted_state_is_non_evaluable_never_a_pass():
    weight = np.full_like(ac.MAGNETIC_D_WEIGHT, 0.1)  # nothing is majority-d
    c = ac.construction(
        ac.semicore_s_d_parent(), ac.D_ONLY_MODEL, target=_target(weight)
    )
    report, v = _verdict(c)
    assert v.applies and not v.evaluable
    assert COVERAGE in report.non_evaluable_check_ids
    assert not report.passed


def test_weights_that_do_not_match_the_parents_grid_are_non_evaluable():
    weight = np.tile(
        np.array([0.0, 0.9]), (1, ac.N_K, 1)
    )  # two bands, the parent has three
    c = ac.construction(
        ac.semicore_s_d_parent(), ac.D_ONLY_MODEL, target=_target(weight)
    )
    _, v = _verdict(c)
    assert v.applies and not v.evaluable
    assert "weight" in v.note


def test_the_target_changes_no_verdict_but_the_coverage_one():
    parent = ac.semicore_s_d_parent()
    default = CertificationGate(CORE_CHECKS).run(
        ac.construction(parent, ac.D_ONLY_MODEL)
    )
    magnetic = CertificationGate(CORE_CHECKS).run(
        ac.construction(parent, ac.D_ONLY_MODEL, target=_target())
    )
    left, right = default.measurements(), magnetic.measurements()
    assert set(left) == set(right)
    changed = {cid for cid in left if left[cid] != right[cid]}
    assert changed == {COVERAGE}
    # and the coverage check is one object under both targets: same threshold, same identity
    assert [c.check_id for c in CORE_CHECKS].count(COVERAGE) == 1


def test_an_explicit_default_declaration_is_the_undeclared_construction():
    parent = ac.semicore_s_d_parent()
    left = CertificationGate(CORE_CHECKS).run(ac.construction(parent, ac.D_ONLY_MODEL))
    right = CertificationGate(CORE_CHECKS).run(
        ac.construction(parent, ac.D_ONLY_MODEL, target=G.WHOLE_OCCUPIED_MANIFOLD)
    )
    assert left.measurements() == right.measurements()


# ---------------------------------------------------------------------------
# one gate: the target set and the new check are the same objects under every backend
# ---------------------------------------------------------------------------


def test_the_target_set_and_the_core_set_are_one_object_under_every_backend():
    gates = [gate_for(name) for name in ("qe", "vasp")]
    new_check = next(
        c
        for c in CORE_CHECKS
        if c.check_id == "metal_frozen_window_contains_fermi_level"
    )
    for gate in gates:
        assert gate.targets is G.CERTIFICATION_TARGETS
        core = gate.checks[: len(CORE_CHECKS)]
        assert all(a is b for a, b in zip(core, CORE_CHECKS))
        assert any(c is new_check for c in gate.checks)
    assert gates[0].checks == gates[1].checks == CORE_CHECKS


# ---------------------------------------------------------------------------
# the OpenMX-specific part stays inside backends/openmx
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.append(base)
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def test_the_core_gate_and_the_qe_and_vasp_backends_import_nothing_from_openmx():
    src = Path(autowan.__file__).parent
    paths = [
        src / "gate.py",
        src / "core_checks.py",
        src / "backends" / "qe.py",
        *sorted((src / "backends" / "vasp").glob("*.py")),
    ]
    for path in paths:
        offending = [m for m in _imported_modules(path) if "openmx" in m.lower()]
        assert offending == [], f"{path.relative_to(src)} imports {offending}"
    # gate.py and core_checks.py take nothing from any backend at all
    for name in ("gate.py", "core_checks.py"):
        assert not [m for m in _imported_modules(src / name) if "backends" in m]
