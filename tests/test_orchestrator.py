"""The loop must never turn an exhausted budget into an acceptance.

These cover the two ways a retry loop misreports itself: by counting one parameter set
tried repeatedly as several distinct attempts, and by folding a provider outage into
the physics record so an infrastructure failure reads as a material that resists repair.
"""

from __future__ import annotations

import json

import pytest

from autowan.actions import Stage
from autowan.orchestrator import (
    BudgetExhausted,
    GateOutcome,
    Run,
    RunConfig,
)

CHECKS = ("mlwf_spread_functional_converged", "parent_scf_converged")
FAIL = GateOutcome(
    passed=False,
    evaluated_check_ids=CHECKS,
    failed_check_ids=("mlwf_spread_functional_converged",),
)
PASS = GateOutcome(passed=True, evaluated_check_ids=CHECKS, failed_check_ids=())


def test_a_passing_baseline_never_consults_a_model():
    run = Run("Si")
    assert run.available_actions() == ()
    attempt = run.start_attempt({"num_iter": 1000})
    run.finish_attempt(attempt, outcome=PASS)
    assert run.passed
    assert run.stage is Stage.TERMINAL
    assert run.terminal_reason == "both gates passed"


def test_one_repair_stage_with_every_action_and_one_cap_of_eight():
    """AUTOMATIC -> REPAIR -> TERMINAL: no generic/reference split."""
    run = Run("bcc_Fe")
    assert run.config.total_attempts == 8
    run.finish_attempt(run.start_attempt({"num_iter": 1000}), outcome=FAIL)
    assert run.advance_after_failure() is Stage.REPAIR
    assert set(run.available_actions()) == {
        "modify_parameters",
        "switch_method",
        "escalate",
    }
    for value in range(2, 9):
        run.finish_attempt(run.start_attempt({"num_iter": 1000 * value}), outcome=FAIL)
        stage = run.advance_after_failure()
    assert len(run.attempts) == 8
    assert stage is Stage.TERMINAL
    assert "8" in run.terminal_reason
    with pytest.raises(BudgetExhausted):
        run.start_attempt({"num_iter": 99})


def test_repeating_a_parameter_set_is_refused_rather_than_spending_the_budget():
    run = Run("bcc_Fe")
    run.finish_attempt(run.start_attempt({"num_iter": 1000}), outcome=FAIL)
    run.advance_after_failure()
    with pytest.raises(BudgetExhausted, match="already attempted"):
        run.start_attempt({"num_iter": 1000})


def test_a_provider_outage_is_not_recorded_as_a_physics_failure():
    run = Run("bcc_Fe")
    attempt = run.start_attempt({"num_iter": 1000})
    run.finish_attempt(attempt, model_error="provider returned HTTP 529")
    assert attempt.model_error is not None
    assert attempt.calculation_error is None
    assert attempt.outcome is None
    assert not run.passed


def test_an_exhausted_budget_returns_a_best_effort_that_is_explicitly_not_passing():
    run = Run("bcc_Fe", RunConfig(total_attempts=2))
    partial = GateOutcome(
        passed=False,
        evaluated_check_ids=CHECKS,
        failed_check_ids=(),  # cleared every check but the gate said no
    )
    run.finish_attempt(run.start_attempt({"num_iter": 1000}), outcome=FAIL)
    run.advance_after_failure()
    run.finish_attempt(run.start_attempt({"num_iter": 5000}), outcome=partial)
    run.advance_after_failure()

    assert not run.passed
    best = run.best_effort()
    assert best is not None
    assert best.outcome.passed is False


def test_the_log_keeps_the_argument_beside_the_change(tmp_path):
    from autowan.actions import validate_action

    action = validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "still descending at the limit",
            "evidence": ["mlwf_spread_functional_converged"],
            "expected_effect": "reaches the convergence window",
            "confidence": 0.7,
            "changes": [
                {
                    "parameter": "num_iter",
                    "old": 1000,
                    "new": 20000,
                    "reason": "delta 1e-2",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=CHECKS,
    )
    run = Run("bcc_Fe")
    run.finish_attempt(run.start_attempt({"num_iter": 1000}), outcome=FAIL)
    run.advance_after_failure()
    run.finish_attempt(
        run.start_attempt({"num_iter": 20000}, action=action), outcome=FAIL
    )

    import json

    log = json.loads(run.write_log(tmp_path / "run.json").read_text())
    recorded = log["attempts"][1]["action"]
    assert recorded["diagnosis"]
    assert recorded["evidence"] == ["mlwf_spread_functional_converged"]
    assert recorded["changes"][0]["reason"] == "delta 1e-2"


def test_the_record_is_on_disk_before_the_run_ends(tmp_path):
    # A run that is killed -- a wall-clock limit, a cancelled job -- must still leave
    # what it had learned. Written only on the way out, the record is missing for
    # exactly the runs that are hardest to repeat.
    run = Run("m", RunConfig(total_attempts=4))
    run.log_path = tmp_path / "run.json"
    attempt = run.start_attempt({"num_wann": 4})
    run.finish_attempt(attempt, calculation_error="the deck was refused")

    written = json.loads((tmp_path / "run.json").read_text())
    assert len(written["attempts"]) == 1
    assert written["attempts"][0]["calculation_error"] == "the deck was refused"
    assert not list(tmp_path.glob("*.partial"))


def test_a_run_can_be_asked_whether_a_proposal_repeats_an_earlier_attempt(tmp_path):
    # Asked before the action is accepted, a repeat is one answer to reject; asked
    # after, it ends a run with most of its budget unspent.
    run = Run("m", RunConfig(total_attempts=6))
    run.finish_attempt(run.start_attempt({"num_wann": 4, "num_iter": 100}))
    assert run.would_repeat({"num_iter": 100, "num_wann": 4})
    assert not run.would_repeat({"num_wann": 4, "num_iter": 200})


def test_a_number_and_its_string_are_the_same_parameter_set():
    # Values read back from a deck are strings; values computed from a proposal are
    # numbers. Compared by repr they were different sets, so a repeat was never
    # recognised while there was still time to ask for something else.
    run = Run("m", RunConfig(total_attempts=6))
    run.finish_attempt(run.start_attempt({"num_wann": "18", "dis_froz_max": "15.1172"}))
    assert run.would_repeat({"num_wann": 18, "dis_froz_max": 15.1172})
    assert not run.would_repeat({"num_wann": 18, "dis_froz_max": 15.5})


def test_a_budget_already_raised_and_still_short_cannot_be_raised_again():
    # Three proposals in a row on the same unresolved check -- 12000, 24000, 48000 --
    # each a new value, so no repeat guard saw any of them.
    run = Run("m", RunConfig(total_attempts=8))
    raised = {
        "action": "modify_parameters",
        "changes": [
            {"parameter": "num_iter", "old": 4000, "new": 12000, "reason": "more steps"}
        ],
    }
    attempt = run.start_attempt({"num_iter": 12000})
    attempt.action = raised
    run.finish_attempt(attempt)
    assert run.already_raised("num_iter", 24000) == 0
    assert run.already_raised("num_iter", 8000) is None, "not a further raise"
    assert run.already_raised("trial_step", 0.1) is None, "a different lever"


def test_a_window_and_the_same_window_written_longer_are_one_parameter_set():
    # The protocol writes an energy to four decimals; the same window arriving as an
    # offset is converted at full precision. Compared verbatim they were different
    # sets, and a repair that reproduced an earlier construction exactly was allowed
    # through -- then caught where it ends the run instead of being asked again.
    run = Run("m", RunConfig(total_attempts=6))
    run.finish_attempt(run.start_attempt({"dis_froz_max": "15.1172", "num_wann": 18}))
    assert run.would_repeat({"dis_froz_max": 15.117163310089875, "num_wann": "18"})
    assert not run.would_repeat({"dis_froz_max": 15.25, "num_wann": 18})


def test_a_repaired_parameter_set_is_not_mistaken_for_one_already_tried():
    # The guard compares canonicalised sets, so a repair that moves any knob by more
    # than the comparison tolerance is a new deck. It must stay that way: a repaired
    # set read as a repeat ends a run with most of its budget unspent.
    run = Run("bcc_Fe")
    run.finish_attempt(
        run.start_attempt({"num_iter": "1000", "dis_froz_max": "7.8057"}), outcome=FAIL
    )
    assert not run.would_repeat({"num_iter": 4000, "dis_froz_max": 7.8057})
    assert not run.would_repeat(
        {"num_iter": 1000, "dis_froz_max": 7.8057, "projectors": "Ba:s;p;d"}
    )
    # and the same set, however it is spelled, still is one
    assert run.would_repeat({"num_iter": 1000, "dis_froz_max": 7.805700000001})
