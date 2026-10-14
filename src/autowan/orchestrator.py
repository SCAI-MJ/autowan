"""The deterministic state machine, and the record it leaves behind.

Everything a language model does in this workflow happens inside this loop, and the
loop is ordinary code: it owns the stage, the budget, which tools are reachable,
whether a proposed action is applied, and what gets written down. The model proposes;
it never decides, never executes, and never advances a stage.

The ladder is AUTOMATIC (the engine's frontier, no model) -> REPAIR -> TERMINAL, with
one cap of eight attempts in all. Each attempt records the method family its deck was
built by, whether a ``switch_method`` produced it, and the read-only tools the model
called while proposing it.

A run the provider stopped (no credit, a low balance, an unreachable network) is not a
finished run. The stop is recorded with the attempt it interrupted, and the run resumes
from that attempt (:meth:`Run.from_dict`, ``wannierize(resume=True)``); it is never
restarted silently from the beginning.

Two failure modes get their own handling because conflating them corrupts the record.
A provider that times out is not a construction that failed physics, and a repeated
proposal is not progress — a loop that re-applies the same parameters until its budget
runs out will report "exhausted after five attempts" for what was really one attempt
tried five times.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .actions import STAGE_ACTIONS, Action, Stage


class BudgetExhausted(RuntimeError):
    """The run may not take another attempt."""


@dataclass
class GateOutcome:
    """What the deterministic gate, and then the review gate, made of one attempt."""

    passed: bool
    evaluated_check_ids: tuple[str, ...]
    failed_check_ids: tuple[str, ...]
    measurements: dict[str, Any] = field(default_factory=dict)
    source: str = "physics_gate"
    """``physics_gate`` or ``review_gate`` — a rejection from either enters the same repair loop."""


@dataclass
class Attempt:
    """One construction, its parameters, and how it was judged."""

    index: int
    stage: str
    parameters: dict[str, Any]
    outcome: GateOutcome | None = None
    action: dict[str, Any] | None = None
    """The action that produced these parameters; absent for the automatic baseline."""
    calculation_error: str | None = None
    requests: list[str] = field(default_factory=list)
    """Requests in this attempt that were not repairs, with the answer each was given.

    A model that keeps asking to look again, or to search again, ends the attempt on a
    bound; without this the reason had to be inferred from the terminal message."""
    model_error: str | None = None
    """Kept apart from :attr:`calculation_error`: a provider failure says nothing
    about the physics, and folding the two together would let an outage read as a
    material that resists repair."""
    started_at: str = ""
    finished_at: str = ""
    invalid_reason: str | None = None
    refused: bool = False
    """A deck the stored parent, the deck check or the engine refused, distinct from
    infrastructure failure. The message in calculation_error goes back to the model
    with the next proposal.
    """
    seed: str | None = None
    """The deck this attempt built."""
    method: str | None = None
    """The method family that deck is built by (smv, pdwf, scdm)."""
    method_switched: bool = False
    switch: dict[str, Any] | None = None
    """``{"from", "to", "reason"}`` when a switch_method action produced this deck."""
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    """Read-only tool calls made while proposing *this* attempt's deck."""
    model_requests: int = 0
    model_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    gate_profile: str | None = None
    """The gate profile this attempt's verdict was made under, and its check-list hash."""
    gate_profile_hash: str | None = None
    schema_enforcement: dict[str, Any] | None = None
    """Whether the answer that produced this deck was schema-enforced at decode time
    (``{"requested", "mechanism", "enforced"}``); ``None`` for the automatic attempt."""
    stalled: dict[str, Any] | None = None
    """Set when a step of this construction stopped making progress and was stopped
    (:class:`~.execution.StepStalled`): the step, the idle time, the limit and the rule
    that set it. Such an attempt is also :attr:`refused` -- it is spent, and its
    message goes back to the model -- but it reads as ``stalled``, not as a deck an
    engine or the deck check refused. On attempt 0 it records that the frontier's own
    construction stalled under this runner: no gate verdict on it exists."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    """Each engine step that completed: its wall time and the progress it showed."""
    overlaps_from: str | None = None
    """The construction whose overlaps this one reused, if it reused any."""
    frontier_point: int | None = None
    """Set on the attempt a frontier handoff starts from (:class:`FrontierHandoff`): the
    frontier's own run at this ladder point, read from its outputs and scored here, never
    built by this run and never one of this run's own (model-proposed) attempts."""

    @property
    def status(self) -> str:
        """An invalid construction is not a measurement for the gate to judge."""
        if self.invalid_reason is not None:
            return "invalid"
        if self.stalled is not None:
            return "stalled"
        if self.refused:
            return "deck_refused"
        if self.calculation_error is not None:
            return "calculation_failed"
        if self.outcome is not None:
            return "passed" if self.outcome.passed else "gate_refused"
        if self.model_error is not None:
            return "model_error"
        return "unfinished"


@dataclass
class RunConfig:
    """Budget and policy for one run. Recorded verbatim in the log."""

    total_attempts: int = 8
    """Attempts in all, the automatic one included (so up to seven repairs)."""
    stop_on_repeated_parameters: bool = True
    tool_calls_per_attempt: int = 8
    """Read-only tool calls the model may make while proposing one attempt."""


class Run:
    """One material through the ladder, with its full record."""

    def __init__(self, material: str, config: RunConfig | None = None) -> None:
        self.material = material
        self.config = config or RunConfig()
        self.stage = Stage.AUTOMATIC
        self.attempts: list[Attempt] = []
        self._seen_states: set[tuple] = set()
        self._stage_attempts: dict[Stage, int] = {s: 0 for s in Stage}
        self.terminal_reason: str | None = None
        #: Every reference search this run made, with what came back. A retrieval that
        #: shaped a repair is part of the repair's provenance, so it is recorded
        #: whether or not the search found anything.
        self.searches: list[dict[str, Any]] = []
        #: Where to keep the record up to date, if the caller wants it kept. Set, it
        #: is rewritten after every attempt rather than once at the end: a run can
        #: stop without reaching the end -- a wall-clock limit, a cancelled job, an
        #: exception nobody caught -- and a record written only on the way out is
        #: missing for exactly the runs hardest to repeat. One perovskite run reached
        #: its third repair over ninety minutes and left nothing behind at all.
        self.log_path: Path | None = None
        #: The engine, the model and its settings, and the gate profile, as run.
        self.backend: str | None = None
        self.model: str | None = None
        self.model_settings: dict[str, Any] = {}
        self.gate_profile: dict[str, Any] | None = None
        #: Every provider stop, with the attempt it interrupted and when it resumed.
        self.provider_stops: list[dict[str, Any]] = []
        #: Every stop from outside the run -- a scheduler wall-time limit, a
        #: cancellation -- with the attempt it interrupted and when it resumed. Not a
        #: result: the interrupted attempt is taken off the record and rebuilt on resume.
        self.infrastructure_stops: list[dict[str, Any]] = []
        #: What a proposal that produced no attempt cost (a stop, an escalation, an
        #: answer that never validated): its tool calls, requests and dollars.
        self.trailing_proposal: dict[str, Any] | None = None
        #: What the provider was asked to enforce on every answer (requested,
        #: mechanism), set when the first proposal is made; per answer it is on the
        #: attempt. ``None`` when no model was consulted.
        self.schema_enforcement: dict[str, Any] | None = None
        #: A frontier handoff (:meth:`start_from_frontier`): the frontier's own runs
        #: before this run took over, as recorded, and their parameter sets, which the
        #: repeat guard treats like this run's own. ``None`` for an ordinary run.
        self.frontier_handoff: dict[str, Any] | None = None
        self._frontier_states: dict[tuple, int] = {}
        #: A stop & report the model returned (escalate with ``report``): the binding
        #: limit, the requested parent changes, the evidence, the expected effect, and
        #: for the band pool the loop's validate_deck confirmation. ``None`` otherwise.
        self.report: dict[str, Any] | None = None

    # -- stage ------------------------------------------------------------

    def available_actions(self) -> tuple[str, ...]:
        return STAGE_ACTIONS[self.stage]

    def advance_after_failure(self) -> Stage:
        """After a refused attempt: into repair, or to the end once the budget is spent.

        Escalation is driven by the loop, not by the model asking for more authority.
        """
        if self.stage is Stage.AUTOMATIC:
            self.stage = Stage.REPAIR
        if (
            self.stage is Stage.REPAIR
            and len(self.attempts) >= self.config.total_attempts
        ):
            self.stage = Stage.TERMINAL
            self.terminal_reason = (
                f"attempt budget of {self.config.total_attempts} spent without a pass"
            )
        return self.stage

    # -- attempts ---------------------------------------------------------

    def _budget_for(self, stage: Stage) -> int:
        return {
            Stage.AUTOMATIC: 1,
            Stage.REPAIR: max(0, self.config.total_attempts - 1),
            Stage.TERMINAL: 0,
        }[stage]

    def check_can_attempt(self, parameters: dict[str, Any]) -> None:
        """Refuse an attempt that would exceed a budget or repeat a state."""
        if self.stage is Stage.TERMINAL:
            raise BudgetExhausted(self.terminal_reason or "run has ended")
        if len(self.attempts) >= self.config.total_attempts:
            raise BudgetExhausted(
                f"total budget of {self.config.total_attempts} attempts is spent"
            )
        if self._stage_attempts[self.stage] >= self._budget_for(self.stage):
            raise BudgetExhausted(
                f"stage {self.stage.value} has spent its {self._budget_for(self.stage)} attempts"
            )
        state = _state_key(parameters)
        if self.config.stop_on_repeated_parameters and state in self._frontier_states:
            raise BudgetExhausted(
                f"this parameter set is the frontier's own run "
                f"{self._frontier_states[state]} before the handoff; re-running it "
                f"would consume the budget without producing new information. The "
                f"set: {dict(sorted(parameters.items()))}"
            )
        if self.config.stop_on_repeated_parameters and state in self._seen_states:
            # Name the attempt it matches and how. The check upstream is meant to
            # catch this while there is still time to ask for something else, and
            # when it does not, the difference between the two sets is the only
            # thing that says why -- twice it was a formatting difference nobody
            # could see from the message alone.
            earlier = next(
                (a.index for a in self.attempts if _state_key(a.parameters) == state),
                None,
            )
            raise BudgetExhausted(
                f"this parameter set was already attempted (attempt {earlier}); "
                f"re-running it would consume the budget without producing new "
                f"information. The set: {dict(sorted(parameters.items()))}"
            )

    def already_raised(self, parameter: str, new: Any) -> int | None:
        """The attempt that last raised this knob, if raising it did not clear anything.

        The policy says a budget already raised and still short is evidence against the
        budget, and the model raised one three times in a row on the same unresolved
        check -- 12000, 24000, 48000 iterations, each proposal a new value so no repeat
        guard saw it. What the run knows is not that the number is wrong but that this
        lever has been pulled and the gate did not move, and that is a fact it can be
        told.
        """
        try:
            proposed = float(new)
        except (TypeError, ValueError):
            return None
        for attempt in self.attempts:
            action = attempt.action
            if action is None:
                continue
            for change in action.get("changes", ()):
                if change.get("parameter") != parameter:
                    continue
                try:
                    before, after = float(change["old"]), float(change["new"])
                except (TypeError, ValueError, KeyError):
                    continue
                if after > before and proposed > after:
                    return attempt.index
        return None

    def would_repeat(self, parameters: Mapping[str, Any]) -> bool:
        """Whether this parameter set has already been attempted in this run.

        Asked before an action is accepted rather than after: a proposal that
        reproduces an earlier deck is one answer to reject, not a reason to end a run
        with most of its budget unspent.
        """
        key = _state_key(dict(parameters))
        return key in self._seen_states or key in self._frontier_states

    def frontier_run_of(self, parameters: Mapping[str, Any]) -> int | None:
        """The frontier run (before a handoff) with this parameter set, if any."""
        return self._frontier_states.get(_state_key(dict(parameters)))

    # -- frontier handoff ---------------------------------------------------

    def start_from_frontier(self, handoff: Mapping[str, Any]) -> None:
        """Record the frontier's runs before this run took over (a frontier handoff).

        ``handoff`` is :meth:`FrontierHandoff.to_record`: the ladder point the frontier
        stopped at, its runs (``history``, each with the full parameter set its deck
        had), and its next proposal. The runs are the frontier's, not this run's: they
        are not attempts and spend none of this run's budget, but their parameter sets
        are refused like this run's own (the outcome is already known).
        """
        self.frontier_handoff = dict(handoff)
        self._frontier_states = {}
        for entry in handoff.get("history") or ():
            params = entry.get("parameters")
            if params:
                self._frontier_states.setdefault(
                    _state_key(dict(params)), int(entry["run"])
                )

    def accounting(self) -> dict[str, Any] | None:
        """For a frontier handoff: the frontier's Wannier90 runs, this run's own
        attempts, and the Wannier90 run (counted over both) the certificate came at."""
        if self.frontier_handoff is None:
            return None
        frontier = len(self.frontier_handoff.get("history") or ())
        own = [a for a in self.attempts if a.frontier_point is None]
        ran = [a for a in own if _ran_wannier90(a)]
        passing = next(
            (a for a in own if a.outcome is not None and a.outcome.passed), None
        )
        return {
            "frontier_attempts": frontier,
            "handoff_point": self.frontier_handoff.get("point"),
            "autowan_attempts": len(own),
            "autowan_attempt_budget": max(0, self.config.total_attempts - 1),
            "autowan_wannier90_runs": len(ran),
            "w90_runs_total": frontier + len(ran),
            "certified_at_autowan_attempt": passing.index if passing else None,
            "certified_at_w90_run": (
                frontier + sum(1 for a in ran if a.index <= passing.index)
                if passing
                else None
            ),
        }

    def start_attempt(
        self,
        parameters: dict[str, Any],
        action: Action | None = None,
        *,
        seed: str | None = None,
        method: str | None = None,
        proposal: Mapping[str, Any] | None = None,
    ) -> Attempt:
        """Open the next attempt for the deck ``seed``.

        ``proposal`` carries what proposing it cost: the tool calls, requests, tokens
        and dollars -- kept with the attempt they produced.
        """
        self.check_can_attempt(parameters)
        self._seen_states.add(_state_key(parameters))
        self._stage_attempts[self.stage] += 1
        switched = action is not None and action.kind == "switch_method"
        proposal = dict(proposal or {})
        attempt = Attempt(
            index=len(self.attempts),
            stage=self.stage.value,
            parameters=dict(parameters),
            action=_action_record(action) if action else None,
            started_at=_now(),
            seed=seed,
            method=method,
            method_switched=switched,
            switch=(
                {
                    "from": action.method_from,
                    "to": action.method,
                    "reason": action.reason,
                }
                if switched
                else None
            ),
            tools_called=list(proposal.get("tools_called", [])),
            model_requests=int(proposal.get("requests", 0)),
            model_usage=dict(proposal.get("usage", {})),
            cost_usd=proposal.get("cost_usd"),
            schema_enforcement=proposal.get("schema_enforcement"),
        )
        self.attempts.append(attempt)
        return attempt

    def record_provider_stop(self, error: Exception) -> None:
        """The provider stopped the run while the next attempt was being proposed."""
        self.provider_stops.append(
            {
                "after_attempt": len(self.attempts) - 1,
                "kind": getattr(error, "kind", type(error).__name__),
                "error": str(error),
                "stopped_at": _now(),
                "resumed_at": None,
            }
        )
        self.terminal_reason = (
            f"stopped by the provider after attempt {len(self.attempts) - 1} "
            f"({getattr(error, 'kind', 'error')}): {error}. Resumable from that attempt"
        )
        self._checkpoint()

    @property
    def stopped_by_provider(self) -> bool:
        return (
            bool(self.provider_stops) and self.provider_stops[-1]["resumed_at"] is None
        )

    def record_infrastructure_stop(self, reason: str) -> dict[str, Any]:
        """The run was stopped from outside. Not a result, and resumable.

        An attempt still in progress has no verdict and never will: it is taken off
        the record (its budget and parameter set released) and kept with the stop, so a
        resume rebuilds the same deck without asking the model for it again.
        """
        interrupted = None
        if self.attempts and not self.attempts[-1].finished_at:
            attempt = self.attempts.pop()
            stage = _stage(attempt.stage)
            self._stage_attempts[stage] = max(0, self._stage_attempts[stage] - 1)
            self._seen_states.discard(_state_key(attempt.parameters))
            interrupted = {
                **asdict(attempt),
                "outcome": None,
            }
        stop = {
            "after_attempt": len(self.attempts) - 1,
            "kind": "infrastructure",
            "error": reason,
            "stopped_at": _now(),
            "resumed_at": None,
            "interrupted_attempt": interrupted,
        }
        self.infrastructure_stops.append(stop)
        self.terminal_reason = (
            f"stopped from outside the run after attempt {len(self.attempts) - 1}"
            + (
                f", during attempt {interrupted['index']} ({interrupted.get('seed')}), "
                f"which is not counted and is rebuilt on resume"
                if interrupted
                else ""
            )
            + f": {reason}. An infrastructure stop, not a result; resumable"
        )
        self._checkpoint()
        return stop

    @property
    def stopped_by_infrastructure(self) -> bool:
        return (
            bool(self.infrastructure_stops)
            and self.infrastructure_stops[-1]["resumed_at"] is None
        )

    def record_request(self, note: str) -> None:
        """Note a request in this attempt that was not a repair, and its answer.

        :attr:`Attempt.requests` is the field that says why an attempt produced no
        repair, and nothing wrote to it: across two eleven-attempt runs it was empty
        in every attempt of both, so why each ended had to be inferred from a terminal
        message describing only the last thing that happened.
        """
        if self.attempts:
            self.attempts[-1].requests.append(note)

    def finish_attempt(
        self,
        attempt: Attempt,
        *,
        outcome: GateOutcome | None = None,
        calculation_error: str | None = None,
        model_error: str | None = None,
        invalid_reason: str | None = None,
        refused: bool = False,
    ) -> None:
        attempt.outcome = outcome
        attempt.calculation_error = calculation_error
        attempt.model_error = model_error
        attempt.invalid_reason = invalid_reason
        attempt.refused = refused
        attempt.finished_at = _now()
        self._checkpoint()
        if outcome is not None and outcome.passed:
            self.stage = Stage.TERMINAL
            self.terminal_reason = "both gates passed"

    # -- result -----------------------------------------------------------

    @property
    def passed(self) -> bool:
        return any(a.outcome is not None and a.outcome.passed for a in self.attempts)

    def best_effort(self) -> Attempt | None:
        """The attempt that cleared the most checks, for a run that never passed.

        Returned explicitly labelled as non-passing. An exhausted budget must never
        read as an acceptance, which is the one way an automated gate can do more
        harm than no gate at all.
        """
        scored = [a for a in self.attempts if a.outcome is not None]
        if not scored:
            return None
        return max(
            scored,
            key=lambda a: len(a.outcome.evaluated_check_ids)
            - len(a.outcome.failed_check_ids),
        )

    @property
    def cost_usd(self) -> float | None:
        costs = [a.cost_usd for a in self.attempts if a.cost_usd is not None]
        costs += [s.get("cost_usd") for s in self.provider_stops if s.get("cost_usd")]
        if self.trailing_proposal and self.trailing_proposal.get("cost_usd"):
            costs.append(self.trailing_proposal["cost_usd"])
        return round(sum(costs), 6) if costs else None

    def schema_enforcement_summary(self) -> dict[str, Any] | None:
        """The run's enforcement setting, with how many answers it actually held for."""
        if self.schema_enforcement is None:
            return None
        answers = [a.schema_enforcement for a in self.attempts if a.schema_enforcement]
        if self.trailing_proposal and self.trailing_proposal.get("schema_enforcement"):
            answers.append(self.trailing_proposal["schema_enforcement"])
        return {
            **{
                k: v
                for k, v in self.schema_enforcement.items()
                if k not in ("answers_enforced", "answers_unenforced")
            },
            "answers_enforced": sum(1 for a in answers if a.get("enforced")),
            "answers_unenforced": sum(1 for a in answers if not a.get("enforced")),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "backend": self.backend,
            "model": self.model,
            "model_settings": self.model_settings,
            "gate_profile": self.gate_profile,
            "config": asdict(self.config),
            "stage": self.stage.value,
            "passed": self.passed,
            "terminal_reason": self.terminal_reason,
            "stopped_by_provider": self.stopped_by_provider,
            "provider_stops": list(self.provider_stops),
            "stopped_by_infrastructure": self.stopped_by_infrastructure,
            "infrastructure_stops": list(self.infrastructure_stops),
            "trailing_proposal": self.trailing_proposal,
            "cost_usd": self.cost_usd,
            "schema_enforcement": self.schema_enforcement_summary(),
            "searches": list(self.searches),
            **({"report": self.report} if self.report is not None else {}),
            **(
                {
                    "frontier_handoff": self.frontier_handoff,
                    "accounting": self.accounting(),
                }
                if self.frontier_handoff is not None
                else {}
            ),
            "attempts": [
                {
                    **asdict(a),
                    "status": a.status,
                    "outcome": asdict(a.outcome) if a.outcome else None,
                }
                for a in self.attempts
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Run":
        """Rebuild a run from its record, to resume it from the attempt it stopped at."""
        config = RunConfig(
            **{
                k: v
                for k, v in (record.get("config") or {}).items()
                if k in RunConfig.__dataclass_fields__
            }
        )
        run = cls(str(record["material"]), config)
        run.backend = record.get("backend")
        run.model = record.get("model")
        run.model_settings = dict(record.get("model_settings") or {})
        run.gate_profile = record.get("gate_profile")
        run.schema_enforcement = record.get("schema_enforcement")
        run.terminal_reason = record.get("terminal_reason")
        run.searches = list(record.get("searches") or [])
        if record.get("frontier_handoff"):
            run.start_from_frontier(record["frontier_handoff"])
        run.report = record.get("report")
        run.provider_stops = [dict(s) for s in record.get("provider_stops") or []]
        run.infrastructure_stops = [
            dict(s) for s in record.get("infrastructure_stops") or []
        ]
        trailing = record.get("trailing_proposal")
        if trailing and run.provider_stops:
            # Spent before the stop; kept with the stop so a resume does not lose it.
            run.provider_stops[-1].setdefault("cost_usd", trailing.get("cost_usd"))
            run.provider_stops[-1].setdefault("proposal", trailing)
        fields = Attempt.__dataclass_fields__
        for raw in record.get("attempts") or []:
            data = {k: v for k, v in raw.items() if k in fields and k != "outcome"}
            outcome = raw.get("outcome")
            attempt = Attempt(
                **data,
                outcome=(
                    GateOutcome(
                        passed=bool(outcome["passed"]),
                        evaluated_check_ids=tuple(
                            outcome.get("evaluated_check_ids") or ()
                        ),
                        failed_check_ids=tuple(outcome.get("failed_check_ids") or ()),
                        measurements=dict(outcome.get("measurements") or {}),
                        source=outcome.get("source", "physics_gate"),
                    )
                    if outcome
                    else None
                ),
            )
            run.attempts.append(attempt)
            run._seen_states.add(_state_key(attempt.parameters))
            stage = _stage(attempt.stage)
            run._stage_attempts[stage] = run._stage_attempts.get(stage, 0) + 1
        run.stage = _stage(record.get("stage", Stage.REPAIR.value))
        return run

    def write_log(self, path: str | Path | None = None) -> Path:
        path = Path(path if path is not None else self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and moved into place, so a run killed mid-write
        # leaves the previous record rather than half of this one.
        scratch = path.with_suffix(path.suffix + ".partial")
        scratch.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        scratch.replace(path)
        return path

    def _checkpoint(self) -> None:
        if self.log_path is not None:
            try:
                self.write_log()
            except OSError:
                pass  # a record that cannot be written must not end the run


def _ran_wannier90(attempt: Attempt) -> bool:
    """Whether this attempt's Wannier90 localisation was launched (a stalled one too)."""
    if attempt.stalled and str(attempt.stalled.get("step", "")).endswith(":localise"):
        return True
    return any(
        str(step.get("step", "")).endswith(":localise") for step in attempt.steps or ()
    )


def _stage(value: str) -> Stage:
    """A recorded stage; the retired generic/reference stages read as repair."""
    try:
        return Stage(value)
    except ValueError:
        return Stage.REPAIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_key(parameters: dict[str, Any]) -> tuple:
    """A hashable, order-independent summary of a parameter set.

    Values are canonicalised before comparison because they arrive by two routes: read
    back from a deck they are strings, and computed from a proposal they are numbers.
    Comparing their reprs made "18" and 18 different parameter sets, so no repeat was
    ever recognised early -- it was caught at the point where the attempt begins, which
    ends the run instead of asking the model for something else.
    """
    return tuple(sorted((k, _canonical(v)) for k, v in parameters.items()))


#: Significant figures a numeric parameter is compared to. An energy window written by
#: a protocol to four decimals and the same window arriving as an offset converted at
#: full precision are the same construction, and comparing their full spellings made
#: them different parameter sets -- 15.1172 against 15.117163310089875, a difference of
#: a tenth of a millielectronvolt, which no calculation here can tell apart.
COMPARISON_FIGURES = 6


def _canonical(value: Any) -> str:
    """One spelling for a value however it reached us, to a physical tolerance."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, str) and _FORTRAN_REAL.match(value):
        # a deck's Fortran spelling (9.5000000000d-01) is the number 0.95
        value = value.strip().lower().replace("d", "e")
    try:
        return f"{float(value):.{COMPARISON_FIGURES}g}"
    except (TypeError, ValueError):
        return str(value)


_FORTRAN_REAL = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)[dD][-+]?\d+\s*$")


def _action_record(action: Action) -> dict[str, Any]:
    """Everything about an action that belongs in the log.

    The rationale is kept beside the change rather than discarded once the change is
    applied: a parameter set with no argument attached is not reproducible science,
    and it is also useless as a contribution to a shared corpus of what worked.
    """
    return {
        "action": action.kind,
        "diagnosis": action.diagnosis,
        "evidence": list(action.evidence),
        "observations": list(action.observations),
        "expected_effect": action.expected_effect,
        "confidence": action.confidence,
        "targets": list(action.targets),
        "method": action.method,
        "method_from": action.method_from,
        "reason": action.reason,
        "num_bands_needed": action.num_bands_needed,
        "report": action.report,
        "changes": [
            {"parameter": c.parameter, "old": c.old, "new": c.new, "reason": c.reason}
            for c in action.changes
        ],
    }
