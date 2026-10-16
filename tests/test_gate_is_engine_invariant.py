"""No threshold changes between engines.

The claim the VASP backend exists to make testable: the certification gate is one
object, and switching the engine underneath it changes nothing about what a
construction must satisfy -- not a check, not a fraction, not the scale a fraction is
taken of, not the record of where it was calibrated, not which material classes it
applies to. Execution and parsing change; the gate does not. The tests here compare the
gate *definition* built from each backend, then score one construction through both
and compare the reports, then check that the verdict rows a results notebook loads
carry the same scale column whichever engine produced the parent.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from autowan.backends import gate_for, get_backend
from autowan.core_checks import (
    CENTRES_SIT_ON_ATOMIC_SITES,
    CONVERGENCE_CHECKS,
    CORE_CHECKS,
)
from autowan.gate import (
    CertificationGate,
    Check,
    CheckOutcome,
    Construction,
    GateReport,
    ScaleAwareThreshold,
)
from autowan.qe_outputs import ParentCalculation
from autowan.verdicts import (
    PRE_PROFILE_VERDICT_COLUMNS,
    VERDICT_COLUMNS,
    verdict_rows,
)

SYNTHETIC = Path(__file__).parent / "fixtures" / "vasp" / "synthetic"
SMOKE = Path(__file__).parent.parent / "data" / "results" / "vasp_backend_smoke"


def _definition(gate: CertificationGate) -> tuple:
    """Everything a check declares, and nothing about how it is evaluated."""
    return tuple(
        (
            c.check_id,
            c.quantity,
            c.verdict_type,
            c.reference,
            tuple(c.applies_to),
            c.origin,
            (
                None
                if c.threshold is None
                else (c.threshold.fraction, c.threshold.scale, c.threshold.calibration)
            ),
        )
        for c in gate.checks
    )


@pytest.fixture
def gates():
    return gate_for(get_backend("qe")), gate_for(get_backend("vasp"))


# ---------------------------------------------------------------------------
# the definition object
# ---------------------------------------------------------------------------


def test_the_gate_definition_object_is_identical_under_both_backends(gates):
    qe, vasp = gates
    assert qe.checks == vasp.checks


def test_neither_backend_adds_to_or_removes_from_the_core_set(gates):
    qe, vasp = gates
    assert qe.checks == CORE_CHECKS
    assert vasp.checks == CORE_CHECKS
    # the ten original checks plus metal_frozen_window_contains_fermi_level, less the two
    # convergence checks that left the default on 2026-10-09
    assert len(vasp.checks) == 9
    assert get_backend("qe").contributed_checks() == ()
    assert get_backend("vasp").contributed_checks() == ()


def test_every_fraction_scale_calibration_and_applicability_compares_equal(gates):
    qe, vasp = gates
    assert _definition(qe) == _definition(vasp)
    scalar = [c for c in vasp.checks if c.verdict_type == "scalar_vs_scale"]
    assert len(scalar) == 5
    for check in scalar:
        assert isinstance(check.threshold, ScaleAwareThreshold)
        assert check.threshold.calibration.strip()
    withheld = [c.check_id for c in vasp.checks if c.applies_to != ("metal", "gapped")]
    assert withheld == [
        "band_distance_from_parent",
        "metal_frozen_window_contains_fermi_level",
    ]


def test_the_definition_is_the_same_object_not_a_copy_that_happens_to_agree(gates):
    qe, vasp = gates
    for a, b in zip(qe.checks, vasp.checks):
        assert a is b
    # the certification targets a construction may declare are one object too
    from autowan.gate import CERTIFICATION_TARGETS

    assert qe.targets is vasp.targets is CERTIFICATION_TARGETS


# ---------------------------------------------------------------------------
# one construction, two gates, one report
# ---------------------------------------------------------------------------


class _Wannier:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _construction(parent: ParentCalculation) -> Construction:
    """A construction whose measurable pieces are synthetic but well-formed."""
    num_wann = 2
    wannier = _Wannier(
        num_wann=num_wann,
        num_bands=parent.n_bands,
        centres_ang=np.zeros((num_wann, 3)),
        spreads_ang2=np.array([1.0, 1.1]),
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=_Wannier(converged=True, n_iterations=10, final_delta=1e-11),
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
    return Construction(
        "probe",
        parent=parent,
        wannier=wannier,
        material_class="metal" if parent.is_metallic() else "gapped",
        inputs={
            "num_wann": num_wann,
            "mp_grid": (2, 1, 1),
            "conv_window": 3,
            "spin_index": 0,
        },
    )


def test_the_same_construction_receives_the_same_report_from_both_gates(gates):
    qe, vasp = gates
    parent = get_backend("vasp").read_parent(SYNTHETIC / "vasprun.xml")
    construction = _construction(parent)
    left, right = qe.run(construction), vasp.run(construction)
    assert isinstance(left, GateReport) and isinstance(right, GateReport)
    assert left.measurements() == right.measurements()
    assert left.evaluated_check_ids == right.evaluated_check_ids
    assert left.failed_check_ids == right.failed_check_ids
    assert left.non_evaluable_check_ids == right.non_evaluable_check_ids
    assert left.passed == right.passed


def test_thresholds_resolve_from_the_parent_whichever_engine_produced_it(gates):
    """A fraction of the occupied bandwidth is the same number for the same bands."""
    _, vasp = gates
    parent = get_backend("vasp").read_parent(SYNTHETIC / "vasprun.xml")
    twin = ParentCalculation(
        **{f.name: getattr(parent, f.name) for f in fields(ParentCalculation)}
    )
    for check in vasp.checks:
        if check.threshold is None:
            continue
        try:
            a = check.threshold.value(_construction(parent))
            b = check.threshold.value(_construction(twin))
        except Exception as exc:  # NonEvaluable is a legitimate answer for both
            with pytest.raises(type(exc)):
                check.threshold.value(_construction(twin))
            continue
        assert a == pytest.approx(b)
    bandwidth = [c for c in vasp.checks if c.check_id == "band_distance_from_parent"][0]
    # 1 % of the occupied bandwidth measured above the largest gap below E_F (5.2 eV). The
    # synthetic parent's occupied bands span -6 .. -4 eV and 3 .. 5 eV, a 7 eV gap apart, so the
    # scale is the 2 eV block above that gap (3 .. 5 eV), not the 11 eV spread -6 .. 5 eV.
    assert bandwidth.threshold.value(_construction(parent)) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# contributed checks land identically, in the same place
# ---------------------------------------------------------------------------


def test_a_contributed_check_lands_in_both_gates_in_the_same_place():
    extra = Check(
        "community_extra",
        "a community quantity",
        "discrete",
        "self",
        lambda c: CheckOutcome(True, 0.0, "unit"),
        origin="community",
    )
    qe = gate_for(get_backend("qe"), extra=(extra,))
    vasp = gate_for(get_backend("vasp"), extra=(extra,))
    assert qe.checks == vasp.checks
    assert qe.checks[-1] is extra


# ---------------------------------------------------------------------------
# the verdict rows a results notebook loads
# ---------------------------------------------------------------------------


def test_verdict_rows_carry_the_contract_columns_and_the_same_scale_under_both_gates(
    gates,
):
    qe, vasp = gates
    parent = get_backend("vasp").read_parent(SYNTHETIC / "vasprun.xml")
    report = vasp.run(_construction(parent))
    left, right = verdict_rows(report, qe), verdict_rows(report, vasp)
    assert left == right
    assert VERDICT_COLUMNS == (
        "check_id",
        "status",
        "value",
        "threshold",
        "scale",
        "applies",
        "profile",
        "profile_hash",
    )
    assert {row["profile"] for row in left} == {"default"}
    assert len({row["profile_hash"] for row in left}) == 1
    assert len(left) == len(CORE_CHECKS)
    for row in left:
        assert tuple(row) == VERDICT_COLUMNS
        assert row["status"] in ("passed", "failed", "non_evaluable", "withheld")
        assert isinstance(row["applies"], bool)
    scale_of = {
        c.check_id: (c.threshold.scale if c.threshold else None) for c in CORE_CHECKS
    }
    assert {row["check_id"]: row["scale"] for row in left} == scale_of
    withheld = [row for row in left if not row["applies"]]
    assert all(row["status"] == "withheld" for row in withheld)


@pytest.mark.parametrize("material", ["Si", "SrO"])
def test_the_smoke_verdicts_on_disk_have_the_contract_schema_and_the_core_scales(
    material,
):
    path = SMOKE / material / "gate_verdicts.json"
    if not path.is_file():
        pytest.skip(f"no smoke verdicts at {path}: the fixture run has not been done")
    rows = json.loads(path.read_text())
    assert isinstance(rows, list) and rows
    for row in rows:
        # written before gate profiles added the profile and its hash to every row
        assert tuple(row) in (VERDICT_COLUMNS, PRE_PROFILE_VERDICT_COLUMNS), row
    scale_of = {
        c.check_id: (c.threshold.scale if c.threshold else None)
        for c in (*CORE_CHECKS, *CONVERGENCE_CHECKS, CENTRES_SIT_ON_ATOMIC_SITES)
    }
    on_disk = {row["check_id"]: row["scale"] for row in rows}
    # the smoke verdicts on disk were written by the ten-check gate, before
    # metal_frozen_window_contains_fermi_level joined the core set and before gate profiles
    # moved centres_sit_on_atomic_sites out of the default for the drift guard
    # (2026-10-07); every row they carry must still name its registered scale, and the
    # only default checks they can lack are those two
    assert set(scale_of) - set(on_disk) <= {
        "metal_frozen_window_contains_fermi_level",
        "centres_stay_near_home_cell",
    }
    assert on_disk == {cid: scale_of[cid] for cid in on_disk}
    provenance = json.loads((SMOKE / material / "provenance.json").read_text())
    assert provenance["backend"] == "vasp"
    assert provenance["material"] == material
