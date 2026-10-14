"""Every exit records a terminal reason; a confirmed band-pool escalation is a prescription.

Two campaign v1/v2 records drove this (campaign v3):

* **V3Si sparse_tb rep1 (v1).** Attempt 3's localisation hit the 6 h step limit
  (``wan_r2:localise exceeded 21600 s and was stopped``). A ``CalculationError`` ends
  the run, and that path wrote no ``terminal_reason``: the record read ``None`` after
  three attempts.
* **Na sparse_tb rep1 (v2).** The parent stored 16 bands; the deck excludes the four
  semicore bands, so the construction can draw on 12. The model probed ``num_bands =
  16`` with ``validate_deck`` (refused), then escalated naming the band pool. The run
  ended ``not certified``. The design's exit for a repair the parent cannot supply is
  ``DFT prescription returned``, naming the band count; and the band-ceiling
  arithmetic compared a request with the parent's *total* bands, ignoring the
  exclusion, so even a final answer asking for 16 would not have been recorded.

The loop harness is ``test_repair_band_count_refusal``'s (SrO-like parent, 52 bands,
``exclude_bands = 1-4``, ``num_bands = 48``; nothing launched).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from test_repair_band_count_refusal import (
    CALCULATION_FAILED,
    SEED,
    _change,
    _loop,
    _OfflineEngine,
    _ScriptedModel,
    _sro_deck,
    _sro_parent,
)

from autowan import workflow
from autowan.actions import ActionRejected, Stage, action_schema, validate_action
from autowan.execution import CalculationError
from autowan.exits import exit_row
from autowan.orchestrator import RunConfig
from autowan.providers import ModelResponse

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))



def _escalate(diagnosis="the knobs cannot reach this", **extra):
    return {
        "action": "escalate",
        "diagnosis": diagnosis,
        "evidence": [CALCULATION_FAILED],
        "targets": [CALCULATION_FAILED],
        "expected_effect": "a human looks at it",
        "confidence": 0.7,
        **extra,
    }


class _ToolCallingModel(_ScriptedModel):
    """Answers from a list, each with the read-only tool calls made on the way."""

    def __init__(self, answers):
        super().__init__([a for a, _ in answers])
        self.calls = [c for _, c in answers]

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        self.payloads.append(payload_text)
        content = self.answers.pop(0)
        return ModelResponse(
            content=content,
            model="scripted",
            raw_text="",
            tool_calls=list(self.calls.pop(0)),
        )


def _refused_probe(num_bands):
    return {
        "name": "validate_deck",
        "arguments": {"changes": [{"parameter": "num_bands", "new": num_bands}]},
        "result": '{"ok":false,"method":"smv","problems":["num_bands = %d, but the '
        'parent stored 52 bands and the deck excludes 4"]}' % num_bands,
        "counted": True,
    }


# ---------------------------------------------------------------------------
# fix 3: a terminal reason on every exit
# ---------------------------------------------------------------------------


class _TimesOutOnRepair(_OfflineEngine):
    """Attempt 0 runs; the first repair's localisation hits the step limit (V3Si)."""

    def _run(self, step, command, **kw):
        if step == f"{SEED}_r1:localise":
            raise CalculationError(
                f"{SEED}_r1:localise exceeded 21600 s and was stopped"
            )
        return super()._run(step, command, **kw)


def test_a_repair_that_times_out_ends_with_a_reason(tmp_path):
    """The V3Si sparse_tb v1 path: ended on a step limit, reason was None."""
    model = _ScriptedModel([_change("num_iter", 4000, 6000)])
    result, runner = _loop(tmp_path, model, runner=_TimesOutOnRepair(tmp_path))
    run = result.run
    assert [a.status for a in run.attempts] == ["calculation_failed"] * 2
    reason = run.terminal_reason
    assert reason and "attempt 1" in reason and "21600 s" in reason
    assert "without a verdict" in reason
    assert exit_row(run.to_dict())["terminal_reason"] == reason
    assert result.status == "best effort, not certified"


def test_a_frontier_attempt_that_cannot_launch_ends_with_a_reason(tmp_path):
    from autowan.execution import WannierRunner

    _sro_deck(tmp_path)
    runner = WannierRunner(
        run_dir=tmp_path,
        wannier90=str(tmp_path / "missing-wannier90.x"),
        pw2wannier90=str(tmp_path / "missing-pw2wannier90.x"),
        mpi_launcher=(),
    )
    result, _ = _loop(tmp_path, _ScriptedModel([]), runner=runner)
    assert "attempt 0" in result.run.terminal_reason
    assert "could not be launched" in result.run.terminal_reason


@pytest.mark.parametrize(
    "answers, expected",
    [
        ([], "stopped by the provider"),
        ([_escalate()], "the model escalated"),
        ([{"action": "nonsense"}] * 10, "did not validate"),
        (
            [_change("num_iter", 4000 + 1000 * i, 5000 + 1000 * i) for i in range(10)],
            "",  # budget or the repeated-raise refusal: anything, as long as it is said
        ),
    ],
    ids=["provider-stop", "escalation", "never-validates", "budget"],
)
def test_every_exit_records_a_reason(tmp_path, answers, expected):
    result, _ = _loop(
        tmp_path, _ScriptedModel(answers), run_config=RunConfig(total_attempts=4)
    )
    reason = result.run.terminal_reason
    assert isinstance(reason, str) and reason.strip()
    assert expected in reason
    assert exit_row(result.run.to_dict())["terminal_reason"] == reason


def test_no_model_records_a_reason(tmp_path):
    _sro_deck(tmp_path)
    result = workflow.wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "unused.xml",
        seed=SEED,
        runner=_OfflineEngine(tmp_path),
        provider_factory=None,
        config=workflow.WorkflowConfig(draw_figures=False),
        backend=__import__("test_repair_band_count_refusal")._StoredParentQe(
            _sro_parent()
        ),
    )
    assert "no model was supplied" in result.run.terminal_reason


def test_an_exception_in_the_loop_is_recorded_before_it_propagates(tmp_path):
    class _Broken(_OfflineEngine):
        def _run(self, step, command, **kw):
            if step.endswith("_r1:localise"):
                raise RuntimeError("disk vanished")
            return super()._run(step, command, **kw)

    _sro_deck(tmp_path, smv=True)
    log = tmp_path / "autowan_log.json"
    with pytest.raises(RuntimeError, match="disk vanished"):
        workflow.wannierize(
            run_dir=tmp_path,
            parent_xml=tmp_path / "unused.xml",
            seed=SEED,
            runner=_Broken(tmp_path),
            provider_factory=_ScriptedModel([_change("num_iter", 4000, 6000)]),
            config=workflow.WorkflowConfig(draw_figures=False),
            backend=__import__("test_repair_band_count_refusal")._StoredParentQe(
                _sro_parent()
            ),
            log_path=log,
        )
    record = json.loads(log.read_text())
    assert "RuntimeError" in record["terminal_reason"]
    assert "disk vanished" in record["terminal_reason"]


def test_a_path_that_forgets_its_reason_is_caught_by_the_backstop(
    tmp_path, monkeypatch
):
    """A future exit that sets nothing still cannot leave the record silent."""
    monkeypatch.setattr(workflow, "_run_loop", lambda run, *a, **k: None)
    result, _ = _loop(tmp_path, _ScriptedModel([]))
    assert "no recorded reason" in result.run.terminal_reason


# ---------------------------------------------------------------------------
# fix 2: the band ceiling counts the bottom exclusion
# ---------------------------------------------------------------------------


def _bands_action(n):
    return {
        "action": "modify_parameters",
        "diagnosis": "more bands to draw from",
        "evidence": ["x"],
        "expected_effect": "smoother subspace",
        "confidence": 0.5,
        "changes": [{"parameter": "num_bands", "new": n, "reason": "more room"}],
    }


def test_the_band_ceiling_is_the_parent_less_the_bottom_exclusion():
    common = dict(stage=Stage.REPAIR, evaluated_check_ids=("x",), parent_band_count=16)
    # Na: 16 stored, 1-4 excluded, so 12 are available
    with pytest.raises(ActionRejected, match="at most 12 are available"):
        validate_action(_bands_action(16), excluded_below=4, **common)
    with pytest.raises(ActionRejected, match="num_bands_needed"):
        validate_action(_bands_action(13), excluded_below=4, **common)
    assert validate_action(_bands_action(12), excluded_below=4, **common)
    # without an exclusion the ceiling is the parent, as before
    assert validate_action(_bands_action(16), **common)


def test_a_rejected_band_request_records_the_parent_bands_it_implies(tmp_path):
    _sro_deck(tmp_path, smv=True)
    from autowan.inputs import read_construction

    inputs = read_construction(tmp_path, SEED)
    run = workflow.Run(SEED)
    run.start_attempt({"x": 1})
    workflow._record_parent_limit_requests(
        run, _sro_parent(), {"changes": [{"parameter": "num_bands", "new": 50}]}, inputs
    )
    assert run._parent_limit_requests == [
        {
            "attempt": 0,
            "parameter": "num_bands",
            "requested": 54,
            "parent_bands": 52,
            "excluded_below": 4,
        }
    ]
    # 48 fits (52 - 4): nothing recorded
    run._parent_limit_requests = []
    workflow._record_parent_limit_requests(
        run, _sro_parent(), {"changes": [{"parameter": "num_bands", "new": 48}]}, inputs
    )
    assert run._parent_limit_requests == []


def test_num_bands_needed_belongs_to_escalate_only():
    schema = action_schema(Stage.REPAIR, ("x",))
    assert schema["properties"]["num_bands_needed"]["type"] == "integer"
    common = dict(stage=Stage.REPAIR, evaluated_check_ids=("x",))
    payload = {k: v for k, v in _escalate().items() if k != "targets"}
    payload["evidence"] = ["x"]
    action = validate_action({**payload, "num_bands_needed": 20}, **common)
    assert action.kind == "escalate" and action.num_bands_needed == 20
    with pytest.raises(ActionRejected, match="positive integer"):
        validate_action({**payload, "num_bands_needed": 0}, **common)
    # on another action (a strict schema makes every field present) it claims nothing
    edit = validate_action({**_bands_action(10), "num_bands_needed": 20}, **common)
    assert edit.num_bands_needed is None
    assert any("belongs to escalate" in o for o in edit.observations)


def test_strict_dialect_redundancies_are_tolerated_and_a_real_switch_is_not():
    """A strict schema makes every field present: restating the current family, or a
    reason on an edit, is accepted (the reason joins the observations); naming another
    family on an edit is still a switch and still refused."""
    common = dict(stage=Stage.REPAIR, evaluated_check_ids=("x",), method="smv")
    edit = validate_action(
        {**_bands_action(10), "method": "smv", "reason": "more room"}, **common
    )
    assert edit.kind == "modify_parameters" and edit.method is None
    assert "reason given with modify_parameters: more room" in edit.observations
    with pytest.raises(ActionRejected, match="switch_method's"):
        validate_action({**_bands_action(10), "method": "scdm"}, **common)


# ---------------------------------------------------------------------------
# fix 2: a confirmed band-pool escalation ends as a DFT prescription
# ---------------------------------------------------------------------------


def test_a_band_pool_escalation_confirmed_by_validate_deck_is_a_prescription(
    tmp_path,
):
    model = _ScriptedModel(
        [_escalate("the band pool binds: 48 bands are not enough", num_bands_needed=60)]
    )
    result, _ = _loop(tmp_path, model)
    run = result.run
    # 2026-10-13: a confirmed band-pool escalation is a stop & report (binding limit
    # parent_band_pool); the prescription text is still produced beside it
    assert result.status == "report returned"
    assert run.report["binding_limit"] == "parent_band_pool"
    assert run.report["requested_changes"] == {"nbnd": 64}
    assert run.report["confirmed_by_validate_deck"] is True
    assert run.report["synthesized"] is True
    row = exit_row({**run.to_dict(), "prescription": result.prescription})
    assert row["exit"] == "report returned"
    # the band count is named: 60 kept + 4 excluded = 64 parent bands, against 52
    assert run.terminal_reason.startswith("report returned")
    assert "64 bands" in run.terminal_reason and "52" in run.terminal_reason
    assert "num_bands = 60" in run.terminal_reason
    assert "needed 64 bands; the parent carries 52" in result.prescription
    pool = run.trailing_proposal["band_pool"]
    assert pool["confirmed"] and pool["ceiling"] == 48
    assert pool["validate_deck"]["ok"] is False
    assert run.trailing_proposal["action"]["num_bands_needed"] == 60


def test_a_band_pool_claim_the_parent_can_meet_is_not_a_prescription(tmp_path):
    model = _ScriptedModel(
        [_escalate("the band pool binds", num_bands_needed=40)]  # 40 <= 48
    )
    result, _ = _loop(tmp_path, model)
    assert result.status == "best effort, not certified"
    assert "does not confirm it" in result.run.terminal_reason
    assert result.run.trailing_proposal["band_pool"]["confirmed"] is False
    # the claimed report is kept, unconfirmed, and not returned
    assert result.run.report is None
    assert result.run.trailing_proposal["report_unconfirmed"]["requested_changes"] == {
        "nbnd": 44
    }


def test_a_refused_validate_deck_probe_plus_a_band_escalation_is_a_prescription(
    tmp_path,
):
    """The Na v2 shape: no num_bands_needed, but the model probed past the ceiling."""
    model = _ToolCallingModel(
        [
            (
                _escalate(
                    "The band pool is at its maximum: only 3 bands are spare for 9 "
                    "functions"
                ),
                [_refused_probe(52)],
            )
        ]
    )
    result, _ = _loop(tmp_path, model)
    assert result.status == "report returned"
    assert result.run.report["requested_changes"] == {"nbnd": 56}
    pool = result.run.trailing_proposal["band_pool"]
    assert pool["num_bands_needed"] == 52 and pool["parent_bands_needed"] == 56
    assert "refused validate_deck probe" in pool["source"]


def test_an_escalation_that_does_not_speak_of_bands_is_left_alone(tmp_path):
    model = _ToolCallingModel(
        [(_escalate("the spread will not converge"), [_refused_probe(52)])]
    )
    result, _ = _loop(tmp_path, model)
    assert result.status == "best effort, not certified"
    assert result.run.terminal_reason.startswith("the model escalated:")
    assert "band_pool" not in result.run.trailing_proposal


# ---------------------------------------------------------------------------
# replay: the stored Na sparse_tb v2 escalation (cluster2 only, read-only)
# ---------------------------------------------------------------------------

NA_V2 = _stored("AUTOWAN_TEST_NA_V2")
stored = pytest.mark.skipif(
    not (NA_V2 / "wan_r3.win").is_file(), reason="the v2 Na run tree lives on cluster2"
)


@stored
def test_replay_the_na_v2_escalation_is_now_a_confirmed_prescription(tmp_path):
    from types import SimpleNamespace

    from autowan.actions import Action
    from autowan.backends import get_backend
    from autowan.methods import DeckContext

    for name in ("wan.win", "wan.pw2wan", "wan_r3.win", "wan_r3.pw2wan"):
        shutil.copy(NA_V2 / name, tmp_path / name)
    (tmp_path / "external_projectors").symlink_to(NA_V2 / "external_projectors")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "aiida.save").symlink_to(
        (NA_V2 / "out" / "aiida.save").resolve()
    )
    qe = get_backend("qe")
    parent_xml = tmp_path / "out" / "aiida.save" / "data-file-schema.xml"
    parent = qe.read_parent(parent_xml)
    assert parent.n_bands == 16
    record = json.loads((NA_V2 / "autowan_log.json").read_text())
    assert record["status"] == "best effort, not certified"
    assert record["prescription"] is None
    diagnosis = record["terminal_reason"].removeprefix("the model escalated: ")
    meta = record["trailing_proposal"]
    inputs = qe.read_construction(tmp_path, "wan_r3")
    state = SimpleNamespace(
        current_inputs=inputs,
        current_seed="wan_r3",
        last_run_bands=workflow._band_tuple(inputs.win),
    )
    ctx = DeckContext(
        "qe", tmp_path, parent, parent_xml, qe.read_construction(tmp_path, "wan")
    )
    run = workflow.Run.from_dict(record)
    action = Action(
        kind="escalate",
        diagnosis=diagnosis,
        evidence=("truncated_hopping_eta2",),
        expected_effect="",
        confidence=0.5,
    )
    pool = workflow._band_pool_escalation(
        action, meta, run, state, parent, qe, ctx, tmp_path
    )
    assert pool["confirmed"], pool
    assert pool["num_bands_needed"] == 16 and pool["excluded_below"] == 4
    assert pool["ceiling"] == 12 and pool["parent_bands_needed"] == 20
    assert pool["validate_deck"]["ok"] is False
