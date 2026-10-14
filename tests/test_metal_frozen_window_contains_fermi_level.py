"""A metal's frozen window must contain the Fermi level.

The frozen window is the energy range a disentangled construction is obliged to reproduce
exactly. On a metal the states that matter most sit at E_F, and a frozen window placed entirely
below it (or above it) releases them to the disentanglement. In the 2026-09-13 gate-transfer run
the construction ``defective_fe_bcc_mlwf_frozen_window_below_ef`` -- bcc Fe with its frozen
window cut to -4 .. -2 eV -- flipped no check, because every window-reading check judges inside
the declared frozen window and none asks where that window sits.

This check is discrete: containment carries no constant. It applies to metals only (an
insulator's frozen window routinely stops at the valence maximum), under both certification
targets, and it is a core check, so it is the same object under every backend. A construction
that declares no energy frozen window makes no claim this check can contradict and passes it,
as ``subspace_selection_converged`` passes a construction that needed no disentanglement.

The frozen window is read from the construction's ``frozen_window_ev`` (absolute eV; the OpenMX
MLWF reader sets it from ``Wannier.Inner.Window``), else from the Wannier90 inputs
``dis_froz_max`` and, when given, ``dis_froz_min`` (absolute eV; Wannier90's default lower bound
is the outer window's, which lies below E_F).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from autowan.backends import gate_for
from autowan.core_checks import CORE_CHECKS
from autowan.gate import Construction, NonEvaluable

_SPEC = importlib.util.spec_from_file_location(
    "analytic_constructions",
    Path(__file__).parent / "fixtures" / "analytic" / "analytic_constructions.py",
)
ac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ac)

CHECK_ID = "metal_frozen_window_contains_fermi_level"


def _check():
    return next(c for c in CORE_CHECKS if c.check_id == CHECK_ID)


def _construction(frozen="absent", inputs=None, ef=0.0, material_class="metal"):
    parent = SimpleNamespace(fermi_energy_ev=ef)
    wannier = (
        SimpleNamespace()
        if frozen == "absent"
        else SimpleNamespace(frozen_window_ev=frozen)
    )
    return Construction(
        "frozen",
        parent=parent,
        wannier=wannier,
        material_class=material_class,
        inputs=inputs or {},
    )


# ---------------------------------------------------------------------------
# the check as declared
# ---------------------------------------------------------------------------


def test_it_is_a_discrete_core_check_for_metals_appended_after_the_original_ten():
    check = _check()
    assert check.verdict_type == "discrete"
    assert check.threshold is None
    assert check.applies_to == ("metal",)
    assert check.origin == "core"
    assert check.reference == "dft_bands"
    assert CORE_CHECKS[-1] is check
    # eleven until the two convergence checks left the default (2026-10-09)
    assert len(CORE_CHECKS) == 9
    for name in ("qe", "vasp"):
        assert gate_for(name).checks[len(CORE_CHECKS) - 1] is check


# ---------------------------------------------------------------------------
# where the window sits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frozen, passed, distance",
    [
        ((-4.0, -2.0), False, 2.0),  # entirely below E_F: the defect
        ((0.5, 2.0), False, 0.5),  # entirely above E_F
        ((-1.0, 1.0), True, 0.0),
        ((-2.0, 0.0), True, 0.0),  # touching counts as containing
    ],
)
def test_the_frozen_window_must_contain_the_fermi_level(frozen, passed, distance):
    outcome = _check().evaluate(_construction(frozen))
    assert outcome.passed is passed
    assert outcome.metric_value == pytest.approx(distance)
    assert outcome.metric_units == "eV"


def test_the_window_is_judged_against_this_parents_own_fermi_level():
    assert _check().evaluate(_construction((8.0, 12.0), ef=10.0)).passed is True
    assert _check().evaluate(_construction((6.0, 8.0), ef=10.0)).passed is False


@pytest.mark.parametrize("frozen", ["absent", None])
def test_a_construction_that_declares_no_frozen_window_passes_and_says_why(frozen):
    outcome = _check().evaluate(_construction(frozen))
    assert outcome.passed is True
    assert "no frozen window" in outcome.note.lower()


@pytest.mark.parametrize(
    "inputs, passed, distance",
    [
        ({"dis_froz_max": "12.0"}, True, 0.0),
        ({"dis_froz_max": "9.0"}, False, 1.0),
        (
            {"dis_froz_max": "1.2d1"},
            True,
            0.0,
        ),  # Fortran exponent, as a .win may write it
        ({"dis_froz_max": 11.5}, True, 0.0),
        ({"dis_froz_min": "10.5", "dis_froz_max": "12.0"}, False, 0.5),
    ],
)
def test_the_wannier90_frozen_window_is_read_from_the_inputs(inputs, passed, distance):
    outcome = _check().evaluate(_construction(inputs=inputs, ef=10.0))
    assert outcome.passed is passed
    assert outcome.metric_value == pytest.approx(distance)


def test_an_unreadable_frozen_bound_is_non_evaluable_not_a_pass():
    with pytest.raises(NonEvaluable):
        _check().evaluate(_construction(inputs={"dis_froz_max": "abc"}, ef=10.0))


def test_a_declared_frozen_window_on_the_run_takes_precedence_over_the_inputs():
    c = _construction((-4.0, -2.0), inputs={"dis_froz_max": "5.0"})
    assert _check().evaluate(c).passed is False


# ---------------------------------------------------------------------------
# through the gate
# ---------------------------------------------------------------------------


def test_the_gate_refuses_a_metal_whose_frozen_window_lies_below_the_fermi_level():
    parent = ac.semicore_s_d_parent()
    assert parent.is_metallic()
    below = ac.construction(parent, ac.D_ONLY_MODEL, frozen_window_ev=(-4.0, -2.0))
    report = gate_for("qe").run(below)
    v = {x.check_id: x for x in report.verdicts}[CHECK_ID]
    assert v.applies and v.evaluable and v.passed is False
    assert CHECK_ID in report.failed_check_ids
    straddling = ac.construction(parent, ac.D_ONLY_MODEL, frozen_window_ev=(-1.0, 1.0))
    v = {x.check_id: x for x in gate_for("qe").run(straddling).verdicts}[CHECK_ID]
    assert v.passed is True


def test_the_check_is_withheld_from_a_gapped_manifold():
    valence = ac.flat(-3.0)
    parent = ac.make_parent([[valence, ac.flat(2.0)]])
    assert not parent.is_metallic()
    c = ac.construction(parent, ac.hr_cosine(-3.0, 0.0), frozen_window_ev=(-4.0, -3.5))
    v = {x.check_id: x for x in gate_for("vasp").run(c).verdicts}[CHECK_ID]
    assert not v.applies
    assert "gapped" in v.note
