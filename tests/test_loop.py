"""The loop's two silent failure modes, and the record it is supposed to leave.

A repair that reaches the projections invalidates the overlaps. Reusing them anyway
builds a construction that does not reflect the change while the gate scores it as
though it did — no error, no warning, a certificate for something that was never
computed. That is the exact class of failure this whole design exists to prevent, so
it gets a test rather than a comment.

The second is quieter: an action whose diagnosis and rationale never reach the run log
leaves a record of parameters with no argument attached, which is neither reproducible
nor useful as a contribution to a shared corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autowan.execution import StepResult, WannierRunner


class _RecordingRunner(WannierRunner):
    """Records what each build was told it could reuse, and runs nothing."""

    def __init__(self, run_dir):
        super().__init__(run_dir=run_dir)
        self.builds: list[tuple[str, str | None]] = []

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


def test_a_projection_change_forces_the_overlaps_to_be_recomputed(tmp_path):
    """scdm_mu belongs to the overlap input, so its overlaps may not be inherited."""
    runner = _RecordingRunner(tmp_path)
    assert runner.invalidates_overlaps({"scdm_mu": 20.0})
    assert not runner.invalidates_overlaps({"num_iter": 4000})


def test_reuse_is_declined_for_an_overlap_touching_change():
    """The decision the loop makes, isolated from the calculation it would drive."""
    overlap_source = "base"
    for changed, expected in (
        ({"num_iter": 4000}, "base"),  # localisation only: inherit
        ({"conv_window": 5}, "base"),  # localisation only: inherit
        ({"scdm_mu": 20.0}, None),  # reaches the projections: recompute
        ({"num_iter": 10, "scdm_sigma": 3.0}, None),
    ):
        result = (
            None
            if WannierRunner.invalidates_overlaps(changed)
            else (overlap_source or "current")
        )
        assert result == expected, changed


def test_an_action_is_recorded_on_the_attempt_it_produced(tmp_path):
    from autowan.actions import Stage, validate_action
    from autowan.orchestrator import GateOutcome, Run

    checks = ("spread_minimisation_converged",)
    action = validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "monotone descent stopped at the iteration limit",
            "evidence": list(checks),
            "targets": list(checks),
            "expected_effect": "the minimisation reaches its convergence window",
            "confidence": 0.8,
            "changes": [
                {
                    "parameter": "num_iter",
                    "old": 400,
                    "new": 4000,
                    "reason": "still descending at the limit",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=checks,
        unresolved_check_ids=checks,
    )

    run = Run("deck")
    run.finish_attempt(
        run.start_attempt({"num_iter": 400}),
        outcome=GateOutcome(
            passed=False, evaluated_check_ids=checks, failed_check_ids=checks
        ),
    )
    run.advance_after_failure()
    # the action describes the deck the *next* attempt reads
    run.finish_attempt(
        run.start_attempt({"num_iter": 4000}, action=action),
        outcome=GateOutcome(
            passed=True, evaluated_check_ids=checks, failed_check_ids=()
        ),
    )

    log = json.loads(run.write_log(tmp_path / "run.json").read_text())
    assert log["attempts"][0]["action"] is None
    recorded = log["attempts"][1]["action"]
    assert recorded["diagnosis"]
    assert recorded["targets"] == list(checks) if "targets" in recorded else True
    assert recorded["changes"][0]["reason"] == "still descending at the limit"


def test_a_withheld_review_adds_a_check_rather_than_revising_a_measurement():
    from autowan.gate import GateReport, Verdict
    from autowan.review_gate import ReviewOutcome
    from autowan.workflow import _with_review_fault

    passing = Verdict(
        "a_real_check",
        evaluable=True,
        passed=True,
        metric_value=1.0,
        metric_units="eV",
        threshold_value=2.0,
        applies=True,
        note="",
    )
    report = GateReport("deck", "metal", (passing,))
    assert report.passed

    review = ReviewOutcome(
        accepted=False,
        observations=(),
        fault="the manifold detaches over half the zone",
        model="m",
    )
    amended = _with_review_fault(report, review)

    assert not amended.passed
    assert "visual_review" in amended.failed_check_ids
    # the measurement the physics gate took is untouched
    assert amended.verdicts[0] is passing
    assert "detaches" in amended.verdicts[-1].note


def test_a_refused_deck_becomes_something_a_repair_can_target():
    """Wannier90 rejecting a deck is a diagnosis, not the end of the run.

    Some repairs are only correct in combination: changing the projector count without
    moving the frozen window leaves more bands inside it than there are functions to
    reproduce them, and Wannier90 says exactly that and stops. Reported as a plain
    error the run ends there; reported as a failing check the next proposal can fix
    the pair.
    """
    from autowan.workflow import CALCULATION_FAILED, _calculation_failure_report

    message = (
        "ERROR AT K-POINT 3 THERE ARE 7 BANDS INSIDE THE INNER WINDOW "
        "AND ONLY 6 TARGET BANDS"
    )
    report = _calculation_failure_report("deck_r1", "metal", message)

    assert not report.passed
    assert report.failed_check_ids == (CALCULATION_FAILED,)
    # the repair loop needs it evaluated, or there is nothing to aim at
    assert report.evaluated_check_ids == (CALCULATION_FAILED,)
    assert message in report.verdicts[0].note


def test_the_repair_is_shown_only_its_own_construction_figures():
    """An earlier attempt's picture beside this attempt's numbers invites a confident
    wrong reading of which construction is described."""
    from pathlib import Path

    figures = [
        Path("deck_bands.png"),
        Path("deck_spreads.png"),
        Path("deck_r1_bands.png"),
        Path("deck_r1_spreads.png"),
    ]
    own = [p for p in figures if p.name.startswith("deck_r1_")]
    assert own == [Path("deck_r1_bands.png"), Path("deck_r1_spreads.png")]


def test_wannier90_refusing_a_deck_is_recovered_with_its_reason(tmp_path):
    """It exits zero having declined, writes no final state, and says why in a .werr.

    A caller reading only the exit status learns nothing; reading only the exception
    learns "the run did not finish", which no repair can act on. The line Wannier90
    actually printed is what tells the model the outer window it proposed holds fewer
    states than there are functions to fill.
    """
    from autowan.workflow import _refusal_reason

    (tmp_path / "deck.werr").write_text(
        " Wannier90: Execution started on 28Aug2026 at 19:13:46 \n"
        " Exiting.......\n"
        " dis_windows: Energy window contains fewer states than number of target WFs\n"
    )
    reason = _refusal_reason(tmp_path / "deck.wout")
    assert reason is not None
    assert "fewer states than number of target WFs" in reason
    assert "Execution started" not in reason


def test_no_refusal_line_anywhere_returns_nothing_rather_than_a_guess(tmp_path):
    from autowan.workflow import _refusal_reason

    assert _refusal_reason(tmp_path / "absent.wout") is None


def test_the_refusal_is_read_from_the_exit_line_not_the_hint_paragraph(tmp_path):
    # Wannier90 prints a generic hint containing the word "error" further down the
    # output; a keyword scan reported that instead of the cause.
    (tmp_path / "deck.werr").write_text(
        " Wannier90: Execution started on 28Aug2026 at 23:31:44\n"
        " Exiting.......\n"
        " w90_wannier90_readwrite_read: mismatch in deck.eig\n"
        " (rank: 0)\n"
    )
    (tmp_path / "deck.wout").write_text(
        " A common cause of this error is using the wrong pseudopotential\n"
    )
    from autowan.workflow import _refusal_reason

    assert _refusal_reason(tmp_path / "deck.wout") == (
        "w90_wannier90_readwrite_read: mismatch in deck.eig"
    )
