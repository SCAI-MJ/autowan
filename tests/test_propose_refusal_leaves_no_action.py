"""A proposal the loop refused may not be the one it acts on.

``_propose`` hands every refusal back to the model and asks again, up to
``VALIDATION_ATTEMPTS`` times. Two of its refusals -- a knob already raised without
moving the gate (``already_raised``), and a parameter set an earlier attempt already
tried (``would_repeat``) -- raise after the proposal has been bound to ``action``. If
the model keeps making the same refused proposal until the retries run out, the loop
falls through with that refused action still bound and the run goes on to build it:
the guard said no, and the deck was written anyway.

The repair is not in any one refusal branch. Whatever the reason a proposal was
refused, exhausting the retries on refusals must end the attempt with no action, and
say that the answer did not validate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autowan import workflow
from autowan.actions import Stage, validate_action
from autowan.orchestrator import GateOutcome, Run

CHECK = "spread_minimisation_converged"


def _raise_num_iter(old, new):
    """A real, validated modify_parameters action raising the iteration budget."""
    return validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "the spread is still falling at the iteration limit",
            "evidence": [CHECK],
            "targets": [CHECK],
            "expected_effect": "the minimisation reaches its convergence window",
            "confidence": 0.6,
            "changes": [
                {
                    "parameter": "num_iter",
                    "old": old,
                    "new": new,
                    "reason": "still descending at the limit",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=(CHECK,),
        unresolved_check_ids=(CHECK,),
    )


def _run_with_one_raise_that_moved_nothing():
    """Attempt 0 at num_iter 400, attempt 1 raised it to 4000, the check still failed."""
    failed = GateOutcome(
        passed=False, evaluated_check_ids=(CHECK,), failed_check_ids=(CHECK,)
    )
    run = Run("deck")
    run.finish_attempt(run.start_attempt({"num_iter": 400}), outcome=failed)
    run.advance_after_failure()
    run.finish_attempt(
        run.start_attempt({"num_iter": 4000}, action=_raise_num_iter(400, 4000)),
        outcome=failed,
    )
    return run


class _SameAnswerEveryTime:
    """A provider whose answer never changes; counts how often it was asked."""

    def __init__(self):
        self.asked = 0

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        from autowan.providers import ModelResponse

        self.asked += 1
        return ModelResponse(content={}, model="stub", raw_text="")


@pytest.fixture
def isolated_propose(monkeypatch):
    """_propose with its payload, schema and validation collaborators stubbed out.

    What is left is the refusal loop itself: the provider, the backend's support
    check, and the run's two repeat guards.
    """

    def _no_wannier_output(path):
        raise workflow.WannierOutputError(f"no output at {path}")

    monkeypatch.setattr(workflow, "read_run", _no_wannier_output)
    monkeypatch.setattr(workflow, "signature_of", lambda parent, wannier: {})
    monkeypatch.setattr(
        workflow,
        "RepairPayload",
        lambda **kw: SimpleNamespace(
            render=lambda stage: "payload",
            attempt_history=list(kw.get("attempt_history", [])),
            material_signature={},
            references=None,
        ),
    )
    monkeypatch.setattr(workflow, "action_schema", lambda *a, **k: {})
    monkeypatch.setattr(workflow, "_species_counts", lambda parent: {"Si": 2})
    monkeypatch.setattr(workflow, "_band_count", lambda inputs: 16)
    monkeypatch.setattr(workflow, "_kept_occupied", lambda parent, inputs: 4)
    from autowan.families import Method

    monkeypatch.setattr(workflow, "method_of", lambda inputs: Method.SMV)

    parent = SimpleNamespace(
        species=("Si", "Si"),
        n_bands=16,
        fermi_energy_ev=6.0,
        is_metallic=lambda: False,
        occupied_bands=lambda: 4,
    )
    report = SimpleNamespace(
        failed_check_ids=(CHECK,),
        non_evaluable_check_ids=(),
        evaluated_check_ids=(CHECK,),
        measurements=lambda: {},
    )

    def call(run, provider, tmp_path):
        return workflow._propose(
            lambda model: provider,
            SimpleNamespace(model="stub", tools=False),
            run,
            parent,
            tmp_path,
            "deck_r1",
            SimpleNamespace(values={"num_iter": 4000}),
            report,
            (),
            backend="qe",
        )

    return call


def test_a_proposal_refused_as_already_raised_every_time_ends_with_no_action(
    isolated_propose, monkeypatch, tmp_path
):
    run = _run_with_one_raise_that_moved_nothing()
    proposal = _raise_num_iter(4000, 8000)
    assert run.already_raised("num_iter", 8000) == 1, "the guard must refuse it"
    monkeypatch.setattr(workflow, "validate_action", lambda *a, **k: proposal)
    provider = _SameAnswerEveryTime()

    result = isolated_propose(run, provider, tmp_path)

    assert provider.asked == workflow.VALIDATION_ATTEMPTS
    assert result is None, (
        "the refused proposal survived the refusal loop and would be built: "
        f"{result!r}"
    )
    assert run.terminal_reason is not None
    assert "did not validate" in run.terminal_reason
    assert "already raised" in run.terminal_reason


def test_a_proposal_refused_as_a_repeat_every_time_ends_with_no_action(
    isolated_propose, monkeypatch, tmp_path
):
    run = _run_with_one_raise_that_moved_nothing()
    # A change the raise guard does not object to, whose resulting deck is attempt 0's.
    proposal = _raise_num_iter(4000, 400)
    assert run.already_raised("num_iter", 400) is None
    monkeypatch.setattr(workflow, "validate_action", lambda *a, **k: proposal)
    monkeypatch.setattr(workflow, "_prospective", lambda *a, **k: {"num_iter": 400})
    assert run.would_repeat({"num_iter": 400}), "the guard must refuse it"
    provider = _SameAnswerEveryTime()

    result = isolated_propose(run, provider, tmp_path)

    assert provider.asked == workflow.VALIDATION_ATTEMPTS
    assert result is None, (
        "the refused proposal survived the refusal loop and would be built: "
        f"{result!r}"
    )
    assert run.terminal_reason is not None
    assert "did not validate" in run.terminal_reason
    assert "parameter set of an attempt already made" in run.terminal_reason


def test_a_proposal_accepted_after_a_refusal_is_still_returned(
    isolated_propose, monkeypatch, tmp_path
):
    """The repair must not turn every refusal into an ended run."""
    run = _run_with_one_raise_that_moved_nothing()
    answers = iter([_raise_num_iter(4000, 8000), _raise_num_iter(4000, 8000)])
    accepted = validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "the iteration budget is not the lever; the window is",
            "evidence": [CHECK],
            "targets": [CHECK],
            "expected_effect": "the minimisation reaches its convergence window",
            "confidence": 0.5,
            "changes": [
                {"parameter": "conv_window", "old": 3, "new": 5, "reason": "noisy"}
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=(CHECK,),
        unresolved_check_ids=(CHECK,),
    )
    monkeypatch.setattr(
        workflow, "validate_action", lambda *a, **k: next(answers, accepted)
    )
    monkeypatch.setattr(
        workflow, "_prospective", lambda *a, **k: {"num_iter": 4000, "conv_window": 5}
    )
    provider = _SameAnswerEveryTime()

    result = isolated_propose(run, provider, tmp_path)

    assert provider.asked == 3
    assert result is not None
    action, next_seed, meta = result
    assert action is accepted
    assert meta["requests"] == 3
    assert next_seed == "deck_r2"
    assert run.terminal_reason is None
