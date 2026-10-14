"""A provider stop is resumable from the same attempt, never silently restarted; the
exits table carries method, method_switched, tools_called and stage."""

from __future__ import annotations

import json

import pytest
import repair_fixtures as fx

from autowan.exits import (
    ATTEMPT_COLUMNS,
    EXIT_COLUMNS,
    attempt_rows,
    exit_row,
    write_rows,
)
from autowan.orchestrator import Run, RunConfig
from autowan.providers import ProviderLowBalance, ProviderUnreachable
from autowan.workflow import (
    NothingToResume,
    RunStoppedByProvider,
    WorkflowConfig,
    wannierize,
)


def _go(tmp_path, model, runner, resume=False, total=5):
    return wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "x.xml",
        seed=fx.SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=total), draw_figures=False
        ),
        backend=fx.StoredParentQe(fx.parent()),
        log_path=tmp_path / "run.json",
        resume=resume,
    )


@pytest.mark.parametrize(
    "stop", [ProviderLowBalance("balance USD 3"), ProviderUnreachable("network down")]
)
def test_a_provider_stop_is_recorded_and_resumed_from_the_same_attempt(tmp_path, stop):
    fx.write_deck(tmp_path)
    runner = fx.RefusingEngine(tmp_path)
    first = fx.ToolScript(
        [
            ([], fx.answer(changes=[("num_iter", 5000)])),
            ([("frozen_capacity", {"e_max": 1.0})], stop),
        ]
    )
    stopped = _go(tmp_path, first, runner)
    assert stopped.status == "stopped by the provider (resumable)"
    assert len(stopped.run.attempts) == 2
    (record,) = stopped.run.provider_stops
    assert record["after_attempt"] == 1 and record["kind"] == stop.kind
    assert record["resumed_at"] is None
    log = json.loads((tmp_path / "run.json").read_text())
    assert log["stopped_by_provider"] and log["status"].startswith(
        "stopped by the provider"
    )
    assert log["trailing_proposal"]["tools_called"][0]["name"] == "frozen_capacity"
    builds_before = list(runner.builds)

    # never silently restarted: the same call without resume refuses
    with pytest.raises(RunStoppedByProvider, match="resume=True"):
        _go(tmp_path, fx.ToolScript([]), runner)
    assert runner.builds == builds_before

    second = fx.ToolScript(
        [([], fx.answer(changes=[("num_iter", 6000)])), ([], fx.answer("escalate"))]
    )
    resumed = _go(tmp_path, second, runner, resume=True)
    attempts = resumed.run.attempts
    assert [a.index for a in attempts] == [0, 1, 2]
    assert attempts[2].parameters["num_iter"] == "6000"
    # the attempts already made were not built again
    assert runner.builds[: len(builds_before)] == builds_before
    assert [b[0] for b in runner.builds[len(builds_before) :]] == [attempts[2].seed]
    assert resumed.run.provider_stops[0]["resumed_at"] is not None
    assert not resumed.run.stopped_by_provider
    assert "escalated" in resumed.run.terminal_reason
    # the request after the resume still carries the record of the earlier attempts
    assert "num_iter" in second.requests[0]["payload_text"]


def test_resume_without_a_stop_is_refused(tmp_path):
    fx.write_deck(tmp_path)
    runner = fx.RefusingEngine(tmp_path)
    _go(tmp_path, fx.ToolScript([([], fx.answer("escalate"))]), runner)
    with pytest.raises(NothingToResume):
        _go(tmp_path, fx.ToolScript([]), runner, resume=True)


def _record():
    run = Run("SrO")
    run.backend = "qe"
    run.model = "anthropic/claude-opus-5.5"
    run.model_settings = {"reasoning": {"effort": "high"}, "max_tokens": 32000}
    run.gate_profile = {"name": "band_interpolation", "hash": "abc"}
    run.finish_attempt(
        run.start_attempt({"a": 1}, seed="SrO", method="pdwf"), calculation_error="x"
    )
    run.advance_after_failure()
    from autowan.actions import Stage, validate_action

    action = validate_action(
        {
            "action": "switch_method",
            "method": "smv",
            "reason": "why",
            "diagnosis": "d",
            "evidence": ["c"],
            "expected_effect": "e",
            "confidence": 0.5,
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=("c",),
        method="pdwf",
    )
    attempt = run.start_attempt(
        {"a": 2},
        action=action,
        seed="SrO_r1",
        method="smv",
        proposal={
            "tools_called": [{"name": "validate_deck"}, {"name": "reference_pages"}],
            "requests": 3,
            "cost_usd": 0.02,
        },
    )
    from autowan.orchestrator import GateOutcome

    run.finish_attempt(attempt, outcome=GateOutcome(True, ("c",), ()))
    return run.to_dict()


def test_the_exits_row_carries_method_switch_tools_and_stage(tmp_path):
    row = exit_row(_record())
    assert set(EXIT_COLUMNS) == set(row)
    assert row["exit"] == "certified after repair"
    assert row["stage"] == "repair" and row["method"] == "smv"
    assert row["method_switched"] is True and row["switches"] == "pdwf->smv@1"
    assert (
        row["tools_called"] == "validate_deck;reference_pages"
        and row["tool_calls"] == 2
    )
    assert row["requests"] == 3 and row["usd"] == pytest.approx(0.02)
    assert (row["reasoning"], row["max_tokens"]) == ("high", 32000)
    assert row["gate_profile"] == "band_interpolation"
    rows = attempt_rows(_record())
    assert [r["method"] for r in rows] == ["pdwf", "smv"]
    assert rows[1]["switch_reason"] == "why"
    path = write_rows(tmp_path / "exits.csv", [row], EXIT_COLUMNS)
    assert path.read_text().splitlines()[0].split(",") == list(EXIT_COLUMNS)
    assert set(ATTEMPT_COLUMNS) == set(rows[0])
