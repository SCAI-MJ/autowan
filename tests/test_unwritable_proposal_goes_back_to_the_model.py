"""A proposal whose deck cannot be written goes back to the model, not to a rebuild.

MnO, dmft_subspace_window, runs_v8 (code 092d896f7), both spin channels. After attempt
1 the model switched to SMV and, in the same action, named projectors and a
``semicore_exclusion``. The deck writer refused that combination on the SMV starting
deck (a PAO deck): "name the projectors in one action and move semicore_exclusion in
the next". Two things then went wrong:

- the repeat probe in ``_propose`` builds the proposed deck and caught the same
  refusal, but swallowed it ("an unwritable proposal is refused elsewhere"), so the
  answer validated;
- the loop recorded the unwritable attempt and ``continue``-d to the *top* of the loop,
  which rebuilds the current deck -- attempt 1's, unchanged because nothing was
  written -- and the repeat guard ended the run: "this parameter set was already
  attempted (attempt 1)". The model never saw the writer's message, and 3 of 8 attempts
  were used. Every unwritable proposal in runs_v8/runs_v9 (MnO up/down, SrO, GaAs, Ba,
  Na, graphene) ended its run this way.

Now an unwritable proposal is refused inside the validation loop, with the writer's
reason, and the model answers again (up to ``VALIDATION_ATTEMPTS``) without spending an
attempt; and a deck that still fails to be written after validation is a refused
attempt followed by a new proposal, never a rebuild of the last deck.
"""

from __future__ import annotations

from test_repair_band_count_refusal import SEED, _change, _loop, _ScriptedModel

import autowan.workflow as workflow
from autowan.inputs import InputError
from autowan.workflow import VALIDATION_ATTEMPTS

#: The writer's refusal in the MnO runs, verbatim.
WRITER_REFUSAL = (
    "naming projectors turns this deck's pseudo-atomic orbitals into analytic ones, "
    "and which semicore shells the named set already covers is then not known; name "
    "the projectors in one action and move semicore_exclusion in the next"
)

#: The scripted answer the writer refuses (any edit would do; this one is recognisable).
UNWRITABLE = ("num_iter", 4000, 7777)


def _refuse(monkeypatch, *, probe=True, real=True):
    """Make ``derive_deck`` refuse the UNWRITABLE edit, in the probe and/or for real."""
    original = workflow.derive_deck

    def derive_deck(**kwargs):
        is_probe = str(kwargs.get("new_seed", "")).endswith("_probe")
        if kwargs["changes"].get(UNWRITABLE[0]) == UNWRITABLE[2] and (
            (probe and is_probe) or (real and not is_probe)
        ):
            raise InputError(WRITER_REFUSAL)
        return original(**kwargs)

    monkeypatch.setattr(workflow, "derive_deck", derive_deck)


def test_an_unwritable_proposal_is_a_validation_refusal_the_model_sees(
    tmp_path, monkeypatch
):
    _refuse(monkeypatch)
    model = _ScriptedModel([_change(*UNWRITABLE), _change("num_iter", 4000, 6000)])
    result, runner = _loop(tmp_path, model)
    attempts = result.run.attempts

    # no attempt was spent on it: attempt 1 is the second answer's deck, and it ran
    assert all("unwritable" not in a.parameters for a in attempts)
    assert int(float(attempts[1].parameters["num_iter"])) == 6000
    assert runner.localised == [SEED, f"{SEED}_r1"]
    # the writer's reason went back to the model with the retry
    assert WRITER_REFUSAL not in model.payloads[0]
    assert WRITER_REFUSAL in model.payloads[1]
    assert "already attempted" not in (result.run.terminal_reason or "")


def test_unwritable_answers_end_at_the_validation_limit_with_the_reason(
    tmp_path, monkeypatch
):
    _refuse(monkeypatch)
    model = _ScriptedModel([_change(*UNWRITABLE)] * (VALIDATION_ATTEMPTS + 2))
    result, _ = _loop(tmp_path, model)

    assert len(result.run.attempts) == 1, "no attempt is spent on an unwritable answer"
    assert len(model.payloads) == VALIDATION_ATTEMPTS
    reason = result.run.terminal_reason or ""
    assert f"did not validate in {VALIDATION_ATTEMPTS} tries" in reason
    assert WRITER_REFUSAL in reason
    assert "already attempted" not in reason


def test_a_deck_unwritable_after_validation_is_refused_and_a_new_proposal_follows(
    tmp_path, monkeypatch
):
    """The fallback: the probe passed it, the real write refused it."""
    _refuse(monkeypatch, probe=False, real=True)
    model = _ScriptedModel([_change(*UNWRITABLE), _change("num_iter", 4000, 6000)])
    result, runner = _loop(tmp_path, model)
    attempts = result.run.attempts

    assert "already attempted" not in (result.run.terminal_reason or "")
    assert len(attempts) >= 3, "the run ended on the unwritable attempt"
    refused = attempts[1]
    assert refused.refused and refused.parameters.get("unwritable") == f"{SEED}_r1"
    assert WRITER_REFUSAL in refused.calculation_error
    # the next answer was asked for (carrying the reason) and built; the last deck was
    # not rebuilt in between
    assert WRITER_REFUSAL in model.payloads[1]
    assert int(float(attempts[2].parameters["num_iter"])) == 6000
    assert runner.overlaps_built == [SEED, f"{SEED}_r2"]
