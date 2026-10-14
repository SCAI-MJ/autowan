"""A stop & report: escalate carries a structured report on what binds (2026-10-13).

The only structured parent-calculation outcome used to be escalate + num_bands_needed;
every other limit (k-mesh, cutoff, smearing, pseudopotential, spin, structure, a criterion
this parent cannot reach) could only leave as a plain escalation, recorded as not
certified, so a model had no reason to stop on a correct diagnosis. Now escalate may carry
``report``: ``binding_limit``, ``requested_changes`` (nbnd, mp_grid, ecutwfc, ecutrho,
degauss, smearing, nspin, starting_magnetization, pseudo, other), ``evidence`` and
``expected_effect``. The run's status is "report returned" and the record carries the
report. A bare num_bands_needed is the band-pool report, kept behind the loop's
validate_deck confirmation. The gate is untouched.
"""

from __future__ import annotations

import pytest
from test_repair_band_count_refusal import _loop, _ScriptedModel
from test_terminal_reason_and_band_pool_prescription import _escalate

from autowan.actions import (
    BINDING_LIMITS,
    ActionRejected,
    Stage,
    action_schema,
    validate_action,
)
from autowan.exits import exit_row
from autowan.orchestrator import Run
from autowan.prompts import REPAIR_TASK
from autowan.providers import strict_schema
from autowan.workflow import CALCULATION_FAILED

KMESH = {
    "binding_limit": "parent_kmesh",
    "requested_changes": {"mp_grid": [13, 13, 13], "other": "same everything else"},
    "evidence": [
        "truncated_hopping_eta2 = 52.9 meV at r_c = L_BvK/4 = 9.80 Ang",
        "truncation_diagnostics: eta_2 at L/3 = 21 meV, at L/2 = 12 meV",
    ],
    "expected_effect": "a 13x13x13 mesh puts r_c at 14.2 Ang, past the decay",
}


def _validate(payload, **kw):
    return validate_action(
        payload,
        stage=Stage.REPAIR,
        evaluated_check_ids=(CALCULATION_FAILED,),
        unresolved_check_ids=(CALCULATION_FAILED,),
        **kw,
    )


def test_the_schema_offers_the_report_on_escalate_and_survives_strict_mode():
    schema = action_schema(
        Stage.REPAIR, (CALCULATION_FAILED,), unresolved_check_ids=(CALCULATION_FAILED,)
    )
    report = schema["properties"]["report"]
    assert report["properties"]["binding_limit"]["enum"] == list(BINDING_LIMITS)
    assert set(report["properties"]["requested_changes"]["properties"]) == {
        "nbnd", "mp_grid", "ecutwfc", "ecutrho", "degauss", "smearing", "nspin",
        "starting_magnetization", "pseudo", "other",
    }  # fmt: skip
    assert "report" not in schema["required"]  # optional: backward compatible
    strict = strict_schema(schema)
    assert strict["properties"]["report"]["anyOf"][1] == {"type": "null"}


def test_a_kmesh_report_validates_with_its_fields():
    action = _validate(_escalate("r_c is set by the mesh", report=KMESH))
    assert action.kind == "escalate"
    assert action.report["binding_limit"] == "parent_kmesh"
    assert action.report["requested_changes"] == {
        "mp_grid": [13, 13, 13],
        "other": "same everything else",
    }
    assert len(action.report["evidence"]) == 2
    assert action.num_bands_needed is None


def test_strict_mode_nulls_in_requested_changes_are_dropped():
    nulls = {**KMESH, "requested_changes": {"mp_grid": [12, 12, 12], "nbnd": None,
                                            "ecutwfc": None}}  # fmt: skip
    action = _validate(_escalate(report=nulls))
    assert action.report["requested_changes"] == {"mp_grid": [12, 12, 12]}


@pytest.mark.parametrize(
    "report, message",
    [
        ({**KMESH, "binding_limit": "luck"}, "binding_limit must be one of"),
        ({**KMESH, "evidence": []}, "report.evidence"),
        ({**KMESH, "requested_changes": {"kpoints": 9}}, "not one of"),
        ({**KMESH, "requested_changes": {"mp_grid": [9, 9]}}, "mp_grid"),
        ({**KMESH, "requested_changes": {"nspin": 3}}, "nspin"),
        ({**KMESH, "expected_effect": " "}, "expected_effect"),
    ],
)
def test_a_malformed_report_is_refused_with_the_reason(report, message):
    with pytest.raises(ActionRejected, match=message):
        _validate(_escalate(report=report))


def test_a_report_on_another_action_is_ignored_not_refused():
    payload = {
        "action": "modify_parameters",
        "diagnosis": "d",
        "evidence": [CALCULATION_FAILED],
        "targets": [CALCULATION_FAILED],
        "expected_effect": "e",
        "confidence": 0.5,
        "changes": [{"parameter": "num_iter", "new": 5000, "reason": "r"}],
        "report": KMESH,
    }
    action = _validate(payload)
    assert action.report is None
    assert any("report given with modify_parameters" in o for o in action.observations)


def test_a_bare_num_bands_needed_is_the_band_pool_report():
    action = _validate(_escalate(num_bands_needed=60), excluded_below=4)
    assert action.num_bands_needed == 60
    assert action.report["binding_limit"] == "parent_band_pool"
    assert action.report["requested_changes"] == {"nbnd": 64}
    assert action.report["synthesized"] is True


def test_a_band_pool_report_with_nbnd_sets_num_bands_needed():
    report = {**KMESH, "binding_limit": "parent_band_pool",
              "requested_changes": {"nbnd": 64}}  # fmt: skip
    action = _validate(_escalate(report=report), excluded_below=4)
    assert action.num_bands_needed == 60
    with pytest.raises(ActionRejected, match="needs requested_changes.nbnd"):
        _validate(_escalate(report={**report, "requested_changes": {}}))
    with pytest.raises(ActionRejected, match="must be parent_band_pool"):
        _validate(_escalate(num_bands_needed=60, report=KMESH))


def test_a_report_is_returned_recorded_and_restored(tmp_path):
    model = _ScriptedModel([_escalate("r_c is set by the mesh", report=KMESH)])
    result, runner = _loop(tmp_path, model)
    run = result.run
    assert result.status == "report returned"
    assert result.prescription is None or "mesh" not in result.prescription
    assert run.report["binding_limit"] == "parent_kmesh"
    assert run.report["requested_changes"]["mp_grid"] == [13, 13, 13]
    assert run.report["diagnosis"] == "r_c is set by the mesh"
    assert run.report["confirmed_by_validate_deck"] is None
    assert run.terminal_reason.startswith("report returned: the model stopped")
    assert "mp_grid=[13, 13, 13]" in run.terminal_reason
    record = run.to_dict()
    assert record["report"] == run.report
    assert exit_row(record)["exit"] == "report returned"
    assert Run.from_dict(record).report == run.report
    # the action record keeps the report too
    assert run.trailing_proposal["action"]["report"]["binding_limit"] == "parent_kmesh"


def test_a_plain_escalation_without_a_report_reads_as_before(tmp_path):
    model = _ScriptedModel([_escalate("the spread will not converge")])
    result, _ = _loop(tmp_path, model)
    assert result.status == "best effort, not certified"
    assert result.run.report is None
    assert "report" not in result.run.to_dict()


def test_an_old_prescription_record_still_reads_as_a_prescription():
    old = {"passed": False, "attempts": [], "prescription": "needed 64 bands"}
    assert exit_row(old)["exit"] == "DFT prescription returned"


def test_the_prompt_says_a_well_founded_report_is_a_result():
    text = " ".join(REPAIR_TASK.split())
    assert "A well-founded stop & report is a legitimate result, not a failure" in text
    assert "counts as a wrong report" in text
    assert "keep repairing when they can" in text
