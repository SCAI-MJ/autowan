"""The gate's rules are the test: what it refuses to admit, and what it refuses to certify.

Three properties carry the gate's authority, and each is here because losing it has a
name. A check may not reference another localized representation, or a comparison
between representations acquires one of them as its own reference. A scalar check may
not carry a bare constant, or a threshold fitted on one material silently decides
another. And a check that could not be evaluated may not read as a check that passed,
or a construction is certified on the evidence it failed to produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from autowan.gate import (
    CertificationGate,
    Check,
    CheckOutcome,
    Construction,
    GateDefinitionError,
    NonEvaluable,
    ScaleAwareThreshold,
)


class _Parent:
    """Enough of a parent calculation for the gate's scales."""

    def __init__(self, fermi=0.0, lo=-5.0, hi=5.0):
        self.eigenvalues_ev = np.linspace(lo, hi, 40).reshape(1, 4, 10)
        self.fermi_energy_ev = fermi
        self.lattice_ang = np.eye(3) * 3.0
        self.alat_ang = 3.0
        self.positions_frac = np.zeros((1, 3))
        self.kpoints_cart = np.zeros((4, 3))


class _Wannier:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _construction(**kw):
    return Construction(
        construction_id="test",
        parent=_Parent(),
        wannier=_Wannier(),
        material_class=kw.pop("material_class", "gapped"),
        inputs=kw.pop("inputs", {}),
    )


def _always(passed, value=1.0):
    return lambda c: CheckOutcome(passed, value, "unit")


def test_a_check_may_not_reference_another_localized_representation():
    with pytest.raises(GateDefinitionError, match="never another"):
        Check("bad", "q", "discrete", "other_representation", _always(True))


def test_a_scalar_check_may_not_carry_a_bare_constant():
    with pytest.raises(GateDefinitionError, match="ScaleAwareThreshold"):
        Check("bad", "q", "scalar_vs_scale", "self", _always(None))


def test_a_threshold_must_record_where_its_fraction_was_calibrated():
    with pytest.raises(GateDefinitionError, match="calibration"):
        ScaleAwareThreshold(
            fraction=0.01, scale="occupied_bandwidth_ev", calibration=""
        )


def test_an_unknown_scale_is_refused():
    with pytest.raises(GateDefinitionError, match="unknown scale"):
        ScaleAwareThreshold(
            fraction=0.01, scale="whatever_feels_right", calibration="x"
        )


def test_a_contributed_check_replaces_a_core_one_with_a_warning_and_a_new_hash():
    # Since 2026-10-07, there is no "may add, never replace" refusal: a contributed
    # check with a profile check's identifier replaces it, loudly, and the recorded hash
    # of the check list changes so the replacement is never invisible in a verdict.
    from autowan.gate import GateProfileWarning

    core = Check("shared_id", "q", "discrete", "self", _always(True))
    shadow = Check(
        "shared_id", "q", "discrete", "self", _always(False), origin="a_contributor"
    )
    with pytest.warns(GateProfileWarning, match="replaces"):
        gate = CertificationGate([core], [shadow])
    assert [c.origin for c in gate.checks] == ["a_contributor"]
    report = gate.run(_construction())
    assert not report.passed
    assert report.profile_hash != CertificationGate([core]).profile_hash
    assert all(v.profile_hash == report.profile_hash for v in report.verdicts)


def test_a_contributed_check_may_not_claim_to_be_core():
    core = Check("core_check", "q", "discrete", "self", _always(True))
    liar = Check("added", "q", "discrete", "self", _always(True), origin="core")
    with pytest.raises(GateDefinitionError, match="must name their contributor"):
        CertificationGate([core], [liar])


def test_a_contributed_check_can_only_make_the_gate_stricter():
    core = Check("core_check", "q", "discrete", "self", _always(True))
    strict = Check(
        "added", "q", "discrete", "self", _always(False), origin="a_contributor"
    )
    construction = _construction()

    assert CertificationGate([core]).run(construction).passed
    assert not CertificationGate([core], [strict]).run(construction).passed


def test_a_check_that_could_not_be_evaluated_is_not_a_check_that_passed():
    def missing(_c):
        raise NonEvaluable("the run recorded no real-space Hamiltonian")

    gate = CertificationGate(
        [
            Check("present", "q", "discrete", "self", _always(True)),
            Check("absent", "q", "discrete", "self", missing),
        ]
    )
    report = gate.run(_construction())

    assert report.failed_check_ids == ()
    assert report.non_evaluable_check_ids == ("absent",)
    assert not report.passed
    assert report.coverage == 0.5


def test_a_check_that_does_not_apply_is_recorded_rather_than_waived():
    gate = CertificationGate(
        [
            Check(
                "gapped_only",
                "q",
                "discrete",
                "dft_bands",
                _always(False),
                applies_to=("gapped",),
            ),
            Check("both", "q", "discrete", "self", _always(True)),
        ]
    )
    report = gate.run(_construction(material_class="metal"))

    assert "gapped_only" not in report.evaluated_check_ids
    assert "gapped_only" not in report.failed_check_ids
    assert report.passed
    verdict = next(v for v in report.verdicts if v.check_id == "gapped_only")
    assert verdict.applies is False
    assert "metal" in verdict.note


def test_the_scalar_verdict_is_the_gates_to_make_not_the_checks():
    """A scalar check reports a number; the gate compares it. The check has no vote."""
    scalar = Check(
        "scalar",
        "q",
        "scalar_vs_scale",
        "self",
        lambda c: CheckOutcome(None, 0.5, "eV"),
        threshold=ScaleAwareThreshold(
            fraction=0.5, scale="occupied_bandwidth_ev", calibration="test"
        ),
    )
    report = CertificationGate([scalar]).run(_construction())
    verdict = report.verdicts[0]
    assert verdict.threshold_value is not None
    assert verdict.passed is (0.5 <= verdict.threshold_value)


def test_a_construction_with_nothing_checkable_is_not_certified():
    def missing(_c):
        raise NonEvaluable("nothing here")

    report = CertificationGate([Check("only", "q", "discrete", "self", missing)]).run(
        _construction()
    )
    assert report.evaluated_check_ids == ()
    assert not report.passed
