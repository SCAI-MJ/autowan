"""The callable: one material in, a certified construction or a named failure out.

This is where the ladder runs. The engine's frontier method is attempted with no model
involved; if the gate accepts it, that result is returned untouched and nothing is
spent. If the gate refuses, one repair stage follows: the model may call read-only
tools (deck validation, method defaults, frozen capacity, centres, band character,
reference pages, the frontier's next candidates, the run corpus) and then proposes one
action -- an edit within the deck's method family, a recorded switch to another family,
or an escalation. Every repaired deck is checked before anything runs on it; a deck the
check refuses is a refused attempt with the reason handed back. One cap of eight
attempts bounds the whole run.

An exhausted budget returns the best construction with an explicit non-passing status.
That is the one outcome an automated gate must never blur, because a false pass is
worse than no gate at all. A run the provider stopped is neither: it is recorded as
stopped and resumes from the same attempt (``resume=True``), never silently restarted.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import recipes
from .actions import (
    ActionRejected,
    Stage,
    action_schema,
    implied_wannier_count,
    validate_action,
)
from .backends import Backend, get_backend
from .deckcheck import check_deck
from .execution import (
    CalculationError,
    ConstructionRunner,
    DeckRefused,
    InfrastructureStop,
    StepResult,
    StepStalled,
    ensure_convergence_test,
)
from .gate import (
    WHOLE_OCCUPIED_MANIFOLD,
    CertificationTarget,
    Construction,
    GateReport,
)
from .gate_profiles import build_gate, get_profile
from .inputs import (
    ConstructionInputs,
    InputError,
    _excluded_from_text,
    method_of,
    read_excluded_bands,
    read_win,
)
from .methods import (
    DeckContext,
    StartingDeckUnavailable,
    derive_deck,
    probe_directory,
)
from .orchestrator import BudgetExhausted, GateOutcome, Run, RunConfig, _action_record
from .parent_adequacy import (
    AttemptEvidence,
    ParentAdequacyReport,
    TerminalDiagnosis,
    diagnose_terminal,
)
from .payload import Artifact, RepairPayload, signature_of, trajectory_of
from .plots import draw_band_comparison, draw_spread_profile
from .prompts import REPAIR_TASK, system_prompt
from .providers import DEFAULT_MODEL, Image, ProviderError, settings_for
from .qe_outputs import ParentCalculation
from .review_gate import ReviewGate
from .tools import RepairTools
from .w90_outputs import WannierOutputError, projectability_not_applied, read_run


class RunStoppedByProvider(RuntimeError):
    """The record at ``log_path`` is a run the provider stopped; resume it, do not restart."""


class RunStoppedByInfrastructure(RunStoppedByProvider):
    """The record at ``log_path`` is a run stopped from outside; resume it, do not restart.

    A subclass, so a caller that already refuses to restart a provider-stopped run
    refuses this one too."""


class NothingToResume(RuntimeError):
    """``resume=True`` was asked of a record with no unresolved provider or
    infrastructure stop."""


@dataclass
class WannierisationResult:
    """What the workflow produced, and how it got there."""

    certified: bool
    seed: str | None
    """The construction that was accepted, or the best one attempted."""

    report: GateReport | None
    run: Run
    parent: ParentCalculation
    figures: tuple[Path, ...] = ()
    parent_adequacy: ParentAdequacyReport | None = None
    diagnosis: TerminalDiagnosis | None = None
    prescription: str | None = None

    @property
    def status(self) -> str:
        if self.certified:
            return "certified"
        # A provider stop is not a result: the run is unfinished and resumable.
        if self.run.stopped_by_provider:
            return "stopped by the provider (resumable)"
        # Invalid outranks a prescription. An invalid attempt means the engine did not
        # do what the deck asked, so the run is not a measurement and must not headline
        # a material-level conclusion; a prescription is only a hypothesis about the
        # parent. ``prescription``, ``diagnosis`` and the log keep it either way.
        if any(a.invalid_reason is not None for a in self.run.attempts):
            return "invalid"
        # A stop & report is a result in its own right (checked by redoing the parent
        # as requested), ahead of the terminal diagnosis's prescription.
        if getattr(self.run, "report", None) is not None:
            return "report returned"
        if self.prescription is not None:
            return "DFT prescription returned"
        return "best effort, not certified"


#: What the model is shown of each frontier run before a handoff; everything else on an
#: entry (its raw deck values, its full parameter set) is for the record and the
#: repeat guard only.
HANDOFF_MODEL_FIELDS = (
    "run",
    "ladder_point",
    "deck",
    "certified",
    "failing_checks",
    "measurements",
    "calculation_error",
)


@dataclass(frozen=True)
class FrontierHandoff:
    """The frontier ran first and did not certify; this run takes over from its last state.

    Frontier handoff (since 2026-10-13): the frontier gets a budget of Wannier90
    runs -- its default deck (run 0) and its own retry ladder up to ``point`` -- each
    judged under this run's criterion. When none certifies, autowan starts from the
    frontier's last ladder state: the deck in the run directory under ``seed`` *is* the
    frontier's deck at ``point``, with that run's Wannier90 outputs beside it (read and
    scored here as attempt 0, never rebuilt), and the model is shown the frontier's runs
    (``history``: per run the deck values that matter and the gate's outcome under this
    criterion). This is the frontier's own executed work. The tool
    ``frontier_next_proposal`` then applies the frontier's retry rule to the run's latest
    run deck, one step at a time (second user revision, 2026-10-13); ``next_proposal``
    is only what it gives at the handoff (ladder point ``point + 1``), kept on the
    record. The frontier's runs are not this run's attempts and spend none of its
    budget.
    """

    point: int
    history: tuple[Mapping[str, Any], ...]
    next_proposal: Mapping[str, Any] | None = None

    def model_view(self) -> dict[str, Any]:
        """What the payload carries: the runs, each reduced to its model-facing fields."""
        return {
            "point": self.point,
            "runs": [
                {k: entry.get(k) for k in HANDOFF_MODEL_FIELDS if k in entry}
                for entry in self.history
            ],
        }

    def to_record(self, base_parameters: Mapping[str, Any]) -> dict[str, Any]:
        """The record the run keeps: every entry with the full parameter set its deck had
        (``base_parameters``, the handed-over deck's, with the entry's own ``deck_values``
        over the keys both carry; the ladder's decks differ only in those)."""
        history = []
        for entry in self.history:
            entry = dict(entry)
            values = entry.get("deck_values") or {}
            entry["parameters"] = {
                **dict(base_parameters),
                **{k: v for k, v in values.items() if k in base_parameters},
            }
            history.append(entry)
        return {
            "point": self.point,
            "history": history,
            "next_proposal": (
                dict(self.next_proposal) if self.next_proposal is not None else None
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrontierHandoff":
        return cls(
            point=int(record["point"]),
            history=tuple(dict(e) for e in record.get("history") or ()),
            next_proposal=record.get("next_proposal"),
        )


@dataclass
class WorkflowConfig:
    """Everything the caller may set without touching the loop."""

    run_config: RunConfig = field(default_factory=RunConfig)
    model: str = DEFAULT_MODEL
    spin_index: int = 0
    draw_figures: bool = True
    review: bool = False
    """Whether to run the multimodal review after the deterministic gate accepts."""
    recipe_dir: str | Path | None = recipes.DEFAULT_CORPUS
    """The corpus ``search_references`` searches; the shipped one unless set."""
    target: CertificationTarget = WHOLE_OCCUPIED_MANIFOLD
    """The target the gate judges each construction against.

    Its ``window_ev``, relative to the parent's Fermi level, is the reproduction
    window. The default declares none, so the deck's own window is used as before.
    """
    gate_profile: Any = "default"
    """The gate profile: a name in :data:`~.gate_profiles.PROFILES` or a
    :class:`~.gate_profiles.GateProfile`. It is what "certified" means for this run."""
    frontier_rescan: str | Path | None = None
    """The frontier's stored retry ladder (CSV or JSON). The ``frontier_candidates`` tool
    reads only its thresholds and order from it, never any stored outcome."""
    rescan_key: str | None = None
    """The construction's key in that grid; the run's material (seed) by default."""
    frontier_dataset_same_composition: bool = True
    """Whether the ``frontier_dataset`` tool may return the published frontier's records of
    this run's own composition (their thresholds after the published rescan, eta2 and
    success). False holds back every record whose reduced composition is the parent's,
    for every query (elements only, any formula, family or limit), like
    ``search_references`` holds back its own composition."""
    tools: bool = True
    """Offer the read-only tools through function calling."""
    recipe_same_material_holdback: bool = False
    """Hold back ``search_references`` entries from this run's own material. Off by
    default (public use may draw on an earlier repair of the same material); a study
    that measures transfer, such as the poster campaign, turns it on."""
    frontier_handoff: FrontierHandoff | None = None
    """Start from the frontier's last ladder state instead of running its deck (see
    :class:`FrontierHandoff`). The deck under ``seed`` must be that state's, with its
    Wannier90 outputs; attempt 0 is then the frontier's run, scored from them."""
    reference_path: Any = None
    """The DFT bands on a high-symmetry path the acceptance criteria are scored on: a
    :class:`~.criteria.ReferenceBandPath`, a QE ``dft_bands_path.npz``, or a VASP
    band-path NSCF directory. Without it a criterion is non-evaluable, so a criterion
    profile cannot certify."""


def _gate_construction(
    config, seed, *, parent, wannier, material_class, inputs
) -> Construction:
    """Carry the caller's target into the construction the gate judges."""
    return Construction(
        seed,
        parent=parent,
        wannier=wannier,
        material_class=material_class,
        inputs=inputs,
        target=config.target,
    )


def _band_tuple(win: Path) -> tuple[Any, Any]:
    """The band count and exclusion that determine which stored bands a deck takes.

    Compared, never trusted: a spelling this cannot read is kept as written, so a deck
    whose band assignment is unreadable differs from the last one that ran and meets
    the deck check, which refuses it with the reason.
    """
    num_bands = read_win(win).get("num_bands")
    try:
        count: Any = int(str(num_bands).strip()) if num_bands is not None else None
    except ValueError:
        count = str(num_bands)
    try:
        excluded: Any = read_excluded_bands(win)
    except InputError as exc:
        excluded = f"unreadable: {exc}"
    return count, excluded


def _parameters(inputs: ConstructionInputs) -> dict[str, Any]:
    """What an attempt records as its parameter set: the deck's values, its method, and
    the overlap input's settings (``pw2wan.<key>``).

    The overlap input is part of the construction: SCDM's mu, sigma and entanglement
    function, the PAO subset (``atom_proj_exclude``) and the projector source live there,
    not in the ``.win``. Recorded without them, two SCDM decks that differed only in
    ``scdm_sigma`` had the same parameter set and every such proposal was refused as a
    repeat (runs_v11 graphene C1, GPT, 2026-10-13).
    """
    return {
        **dict(inputs.values),
        "method": method_of(inputs).value,
        **_overlap_input_values(getattr(inputs, "pw2wan", None)),
    }


#: Overlap-input keys that name files or the run, not the construction.
_OVERLAP_INPUT_IGNORED = ("seedname", "outdir", "prefix")


def _overlap_input_values(path) -> dict[str, Any]:
    """``key = value`` assignments of a pw2wannier90 namelist, as ``pw2wan.<key>``."""
    if path is None or not Path(path).is_file():
        return {}
    out: dict[str, Any] = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.split("!")[0].strip().rstrip(",")
        if "=" not in stripped or stripped.startswith(("&", "/")):
            continue
        key, _, value = stripped.partition("=")
        key = key.strip().lower()
        if not key or key in _OVERLAP_INPUT_IGNORED:
            continue
        out[f"pw2wan.{key}"] = value.strip().strip("'\"").lower()
    return out


def _gate_profile_record(config) -> dict:
    profile = get_profile(getattr(config, "gate_profile", "default"))
    return {
        "name": profile.name,
        "hash": profile.hash,
        "checks": list(profile.check_ids),
        "target_manifold": profile.target_manifold,
    }


def _sub_manifold(config) -> bool:
    """Whether the run's gate profile targets a sub-manifold (``downfolded_d``)."""
    return get_profile(getattr(config, "gate_profile", "default")).sub_manifold


def _target_text(config, species=None, parent=None) -> str:
    """The active profile's statement of what passes: the repair prompt's target.

    ``species`` lets a profile with a per-material target (``dmft_subspace``) name the
    structure's own, and ``parent`` with the run's reference path say whether that
    target manifold is isolated or entangled (which of its states are compared); every
    other profile's text does not depend on them.
    """
    profile = get_profile(getattr(config, "gate_profile", "default"))
    reference = None
    if species is not None and profile.sub_manifold:
        try:
            reference = _reference_path(config)
        except Exception:  # noqa: BLE001 -- the text then states the rule only
            reference = None
    return profile.target_text(species, reference_path=reference, parent=parent)


def _reference_path(config):
    """The reference band path the criteria need, read once per run."""
    source = config.reference_path
    if source is None or not isinstance(source, (str, Path)):
        return source
    from .criteria import read_qe_reference_band_path, read_vasp_reference_band_path

    source = Path(source)
    if source.is_dir():
        return read_vasp_reference_band_path(source)
    return read_qe_reference_band_path(source)


def _construction_reader(config, backend, parent, run_dir) -> Callable[[str], Any]:
    """seed -> the Construction the gate judged for it, with the run's reference path.

    What the criterion tools read: the same inputs :class:`_Scorer` hands the gate, read
    back from the run directory, so a tool sees exactly the construction the verdict was
    made on. The reference path is read once, on the first call.
    """
    run_dir = Path(run_dir)
    material_class = "metal" if parent.is_metallic() else "gapped"
    cache: dict[str, Any] = {}

    def read(seed: str) -> Construction:
        if "reference" not in cache:
            cache["reference"] = _reference_path(config)
        inputs = backend.read_construction(run_dir, seed)
        c = _gate_construction(
            config,
            seed,
            parent=parent,
            wannier=read_run(run_dir / f"{seed}.wout"),
            material_class=material_class,
            inputs={
                **inputs.gate_inputs(config.spin_index),
                "route": backend.route(inputs),
            },
        )
        c.reference_path = cache["reference"]
        return c

    return read


@dataclass
class _Loop:
    """The state one pass of the loop reads and writes."""

    current_seed: str
    current_inputs: ConstructionInputs
    report: GateReport | None = None
    pending_action: Any = None
    pending_meta: dict = field(default_factory=dict)
    overlap_source: str | None = None
    last_run_bands: Any = None
    best: tuple[GateReport, str] | None = None
    best_wannier: Any = None
    figures: list[Path] = field(default_factory=list)
    last_refused: bool = False


def wannierize(
    *,
    run_dir: str | Path,
    parent_xml: str | Path,
    seed: str,
    runner: ConstructionRunner,
    provider_factory: Callable[[str], Any] | None = None,
    config: WorkflowConfig | None = None,
    extra_checks: Sequence = (),
    log_path: str | Path | None = None,
    backend: str | Backend = "qe",
    declared_moments: Mapping[str, float] | None = None,
    resume: bool = False,
) -> WannierisationResult:
    """Build a certified Wannier construction, repairing it if the gate refuses.

    ``seed`` names an already-prepared deck: the frontier method's own output, which
    the workflow never rewrites. Every repair is written to a fresh seedname derived
    from it, so the record of what was tried survives the run.

    ``resume=True`` continues a run the provider stopped, from the attempt it stopped
    at, using the record at ``log_path``. Without it, a record that shows an unresolved
    provider stop is refused rather than overwritten. A run stopped from outside (an
    :class:`~.execution.InfrastructureStop`: a scheduler wall-time limit, a
    cancellation) is resumed the same way, and the attempt it interrupted is rebuilt.
    """
    run_dir = Path(run_dir)
    config = config or WorkflowConfig()
    backend = get_backend(backend)
    parent = backend.read_parent(parent_xml)
    material_class = "metal" if parent.is_metallic() else "gapped"
    profile = get_profile(getattr(config, "gate_profile", "default"))
    reference = _reference_path(config)
    gate = build_gate(backend, profile, extra_checks, reference_path=reference)
    log_path = Path(log_path) if log_path else None

    if resume:
        if log_path is None or not log_path.is_file():
            raise NothingToResume(
                "resume=True needs the record of the stopped run (log_path)"
            )
        run = Run.from_dict(json.loads(log_path.read_text()))
        if not (run.stopped_by_provider or run.stopped_by_infrastructure):
            raise NothingToResume(
                f"{log_path} shows no unresolved provider or infrastructure stop; "
                f"there is nothing to resume"
            )
        frontier_seed = (
            run.attempts[0].seed if run.attempts and run.attempts[0].seed else seed
        )
    else:
        if log_path is not None and log_path.is_file():
            try:
                earlier = Run.from_dict(json.loads(log_path.read_text()))
            except (ValueError, KeyError, TypeError):
                earlier = None
            if earlier is not None and earlier.stopped_by_provider:
                raise RunStoppedByProvider(
                    f"{log_path} records a run the provider stopped after attempt "
                    f"{earlier.provider_stops[-1]['after_attempt']}; resume it with "
                    f"resume=True, or move the record away to start over"
                )
            if earlier is not None and earlier.stopped_by_infrastructure:
                raise RunStoppedByInfrastructure(
                    f"{log_path} records a run stopped from outside after attempt "
                    f"{earlier.infrastructure_stops[-1]['after_attempt']}; resume it "
                    f"with resume=True, or move the record away to start over"
                )
        run = Run(seed, config.run_config)
        frontier_seed = seed
    run.log_path = log_path
    run.backend = backend.name
    run.model = config.model if provider_factory is not None else None
    run.model_settings = (
        settings_for(config.model).to_dict() if provider_factory else {}
    )
    run.gate_profile = _gate_profile_record(config)

    base_inputs = backend.read_construction(run_dir, frontier_seed)
    preflight, preflight_error = None, None
    try:
        preflight = backend.assess_parent(
            parent_xml, base_inputs, declared_moments=declared_moments
        )
    except Exception as exc:
        # An unavailable assessment neither certifies nor blocks the parent.
        preflight_error = str(exc)
    if not resume and preflight is not None and preflight.blocking:
        diagnosis = diagnose_terminal(preflight, [])
        run.stage = Stage.TERMINAL
        run.terminal_reason = "parent–deck contract violation blocks certification"
        result = WannierisationResult(
            certified=False,
            seed=None,
            report=None,
            run=run,
            parent=parent,
            parent_adequacy=preflight,
            diagnosis=diagnosis,
            prescription=diagnosis.message(),
        )
        if log_path:
            _write_result_log(result, log_path, preflight_error)
        return result

    ctx = DeckContext(
        code=backend.name,
        run_dir=run_dir,
        parent=parent,
        parent_source=Path(parent_xml),
        frontier=base_inputs,
        spin_index=config.spin_index,
    )
    scorer = _Scorer(config, gate, backend, parent, material_class, run_dir)

    if resume:
        _prime_reference(runner, run)
        if run.stopped_by_infrastructure:
            state, skip_build = _resume_after_infrastructure_stop(
                run, backend, run_dir, scorer, seed, base_inputs
            )
        else:
            state = _restore(run, backend, run_dir, scorer)
            stop = run.provider_stops[-1]
            stop["resumed_at"] = _now()
            run.terminal_reason = None
            run.stage = Stage.REPAIR
            skip_build = True
    else:
        state = _Loop(seed, base_inputs, last_run_bands=_band_tuple(base_inputs.win))
        skip_build = False

    handoff = getattr(config, "frontier_handoff", None)
    ended = False
    if handoff is not None and not resume:
        # The frontier's last run is attempt 0: scored from its outputs, never rebuilt,
        # and the loop goes straight to the first proposal.
        ended = _handoff_attempt(
            run, state, backend, parent, run_dir, config, scorer, handoff
        )
        skip_build = True

    try:
        if not ended:
            _run_loop(
                run,
                state,
                runner,
                backend,
                parent,
                run_dir,
                config,
                scorer,
                provider_factory,
                ctx,
                skip_build,
                material_class,
            )
    except InfrastructureStop as exc:
        # Stopped from outside: not a result. The record says so, sets aside the
        # attempt in progress, and the run is resumable.
        run.record_infrastructure_stop(str(exc))
        raise
    except BaseException as exc:
        # An exception nobody caught still ends the run: the record says so before
        # the exception propagates, so no record is left without a reason.
        run.terminal_reason = (
            f"the workflow raised {type(exc).__name__} and the run ended: {exc}"
        )
        run._checkpoint()
        raise
    if run.terminal_reason is None:
        # Every path out of the loop records its own reason; this is the backstop, so a
        # path added later without one still cannot leave the record silent.
        last = run.attempts[-1] if run.attempts else None
        run.terminal_reason = (
            f"the run ended after attempt {last.index} ({last.status}) with no "
            f"recorded reason"
            if last is not None
            else "the run ended before any attempt with no recorded reason"
        )

    result = WannierisationResult(
        certified=run.passed,
        seed=state.best[1] if state.best else None,
        report=state.best[0] if state.best else None,
        run=run,
        parent=parent,
        figures=tuple(state.figures),
        parent_adequacy=preflight,
    )
    if (
        not result.certified
        and not run.stopped_by_provider
        and backend.adequacy is not None
    ):
        requests = getattr(run, "_parent_limit_requests", [])
        evidence = [
            AttemptEvidence.from_record(
                {
                    **record,
                    "parent_limit_requests": [
                        r for r in requests if r["attempt"] == record["index"]
                    ],
                }
            )
            for record in run.to_dict()["attempts"]
        ]
        result.diagnosis = diagnose_terminal(
            (
                preflight
                if preflight is not None
                else ParentAdequacyReport((), engine=backend.name)
            ),
            evidence,
            prescribe=backend.adequacy.prescribe,
            hamiltonian=state.best_wannier,
        )
        if result.diagnosis.returns_prescription:
            result.prescription = result.diagnosis.message()
    if (
        result.status == "report returned"
        and (run.report or {}).get("binding_limit") == "parent_band_pool"
    ):
        nbnd = (run.report.get("requested_changes") or {}).get("nbnd")
        if nbnd is not None and f"{nbnd} bands" not in str(run.terminal_reason):
            run.terminal_reason = (
                f"{run.terminal_reason} [the repair needs a parent with {nbnd} bands; "
                f"this one carries {parent.n_bands}]"
            )
    if result.status == "DFT prescription returned":
        requests = getattr(run, "_parent_limit_requests", [])
        needed = max((int(r["requested"]) for r in requests), default=None)
        if not str(run.terminal_reason or "").startswith("DFT prescription returned"):
            run.terminal_reason = (
                "DFT prescription returned"
                + (
                    f": the repair needs a parent with {needed} bands; this one "
                    f"carries {parent.n_bands}"
                    if needed is not None
                    else ""
                )
                + f". The loop ended because {run.terminal_reason}"
            )
    if log_path:
        _write_result_log(result, log_path, preflight_error)
    return result


def _run_loop(
    run,
    state,
    runner,
    backend,
    parent,
    run_dir,
    config,
    scorer,
    provider_factory,
    ctx,
    skip_build,
    material_class,
) -> None:
    """Build, judge and repair until the run ends. Every exit records its reason."""
    while True:
        if not skip_build:
            ended = _attempt(
                run,
                state,
                runner,
                backend,
                parent,
                run_dir,
                config,
                scorer,
                provider_factory,
                ctx=ctx,
            )
            if ended:
                break
        skip_build = False
        run.advance_after_failure()
        if run.stage is Stage.TERMINAL:
            break
        if provider_factory is None:
            run.terminal_reason = (
                "no model was supplied, so no repair could be proposed"
            )
            break
        proposal = _propose(
            provider_factory,
            config,
            run,
            parent,
            run_dir,
            state.current_seed,
            state.current_inputs,
            state.report,
            state.figures,
            backend=backend,
            ctx=ctx,
            last_run_bands=state.last_run_bands,
        )
        if isinstance(proposal, ProviderError):
            run.record_provider_stop(proposal)
            break
        if proposal is None:
            break
        action, next_seed, meta = proposal
        if action.kind == "escalate":
            run.trailing_proposal = {**meta, "action": _action_record(action)}
            pool = _band_pool_escalation(
                action, meta, run, state, parent, backend, ctx, run_dir
            )
            if pool is not None:
                run.trailing_proposal["band_pool"] = pool
            report = action.report
            if report is None and pool is not None and pool["confirmed"]:
                # a band-pool claim found from a refused validate_deck probe
                report = {
                    "binding_limit": "parent_band_pool",
                    "requested_changes": {"nbnd": pool["parent_bands_needed"]},
                    "evidence": [],
                    "expected_effect": action.expected_effect,
                    "synthesized": True,
                }
            if (
                report is not None
                and report["binding_limit"] == "parent_band_pool"
                and not (pool is not None and pool["confirmed"])
            ):
                # the loop's own validate_deck does not confirm the band-pool claim: it
                # is not returned as a report (the parent can meet it)
                run.trailing_proposal["report_unconfirmed"] = report
                report = None
            if report is not None:
                run.report = {
                    **report,
                    "attempt": len(run.attempts) - 1,
                    "diagnosis": action.diagnosis,
                    "checks_cited": list(action.evidence),
                    "confidence": action.confidence,
                    "confirmed_by_validate_deck": (
                        bool(pool["confirmed"])
                        if pool is not None
                        and report["binding_limit"] == "parent_band_pool"
                        else None
                    ),
                }
            if pool is not None and pool["confirmed"]:
                _record_parent_limit(
                    run,
                    parameter="num_bands",
                    requested=pool["parent_bands_needed"],
                    parent_bands=pool["parent_bands"],
                    num_bands_needed=pool["num_bands_needed"],
                    excluded_below=pool["excluded_below"],
                    source=f"escalation ({pool['source']}), confirmed by validate_deck",
                )
                run.terminal_reason = (
                    f"the model escalated on the parent band pool, and validate_deck "
                    f"confirms it: the repair needs num_bands = "
                    f"{pool['num_bands_needed']}, i.e. {pool['parent_bands_needed']} "
                    f"parent bands with the lowest {pool['excluded_below']} excluded, "
                    f"and the parent carries {pool['parent_bands']} (at most "
                    f"{pool['ceiling']} available). The model's diagnosis: "
                    f"{action.diagnosis}"
                )
            elif pool is not None:
                run.terminal_reason = (
                    f"the model escalated: {action.diagnosis} [It names the parent band "
                    f"pool, but validate_deck does not confirm it: num_bands = "
                    f"{pool['num_bands_needed']} against a ceiling of {pool['ceiling']}"
                    f"{'; ' + pool['error'] if pool.get('error') else ''}]"
                )
            else:
                run.terminal_reason = f"the model escalated: {action.diagnosis}"
            if run.report is not None:
                changes = run.report.get("requested_changes") or {}
                run.terminal_reason = (
                    f"report returned: the model stopped and reported the binding limit "
                    f"{run.report['binding_limit']}"
                    + (
                        f", requesting {', '.join(f'{k}={v}' for k, v in changes.items())}"
                        if changes
                        else ""
                    )
                    + f". {run.terminal_reason}"
                )
            break
        changed = {c.parameter: c.new for c in action.changes}
        try:
            derived = derive_deck(
                backend=backend,
                source=state.current_inputs,
                ctx=ctx,
                run_dir=run_dir,
                new_seed=next_seed,
                changes=changed,
                method=action.method,
                fermi_energy_ev=parent.fermi_energy_ev,
                species_counts=_species_counts(parent),
            )
        except (InputError, StartingDeckUnavailable) as exc:
            # A deck that cannot be written is a refused attempt: the reason goes back
            # with the next proposal, and the budget is what bounds a model that keeps
            # proposing one. (The repeat probe in ``_propose`` refuses most of these
            # before an attempt is spent; this is what it could not foresee.) Nothing
            # was written, so the current deck is still the last one built: the loop
            # goes straight to the next proposal. Rebuilding it here repeated that
            # attempt's parameters and the repeat guard ended the run, the reason
            # unseen (MnO dmft_subspace_window, runs_v8).
            state.pending_action, state.pending_meta = action, meta
            if _refuse_unwritable(
                run, state, action, next_seed, str(exc), material_class
            ):
                break
            skip_build = True
            continue
        # A repaired deck is ours, so it gets the convergence test Wannier90 runs
        # none of by default; the frontier deck (attempt 0) never does.
        if ensure_convergence_test(derived.win):
            derived = backend.read_construction(run_dir, next_seed)
        state.pending_action, state.pending_meta = action, meta
        if action.kind == "switch_method" or runner.invalidates_overlaps(changed):
            state.overlap_source = None
        else:
            state.overlap_source = state.overlap_source or (
                None if state.last_refused else state.current_seed
            )
        state.current_seed = next_seed
        state.current_inputs = derived


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _refuse_unwritable(run, state, action, next_seed, message, material_class) -> bool:
    """Spend an attempt on a deck that could not be written. Returns whether the run ends."""
    parameters = {
        "unwritable": next_seed,
        "method": action.method or (method_of(state.current_inputs).value),
        "changes": repr([(c.parameter, c.new) for c in action.changes]),
    }
    try:
        attempt = run.start_attempt(
            parameters,
            action=action,
            seed=next_seed,
            method=parameters["method"],
            proposal=state.pending_meta,
        )
    except BudgetExhausted as exc:
        run.terminal_reason = str(exc)
        return True
    reason = f"the proposed deck could not be written: {message}"
    run.finish_attempt(attempt, calculation_error=reason, refused=True)
    state.report = _calculation_failure_report(next_seed, material_class, reason)
    state.last_refused = True
    state.pending_action, state.pending_meta = None, {}
    return False


@dataclass
class _Scorer:
    config: Any
    gate: Any
    backend: Any
    parent: Any
    material_class: str
    run_dir: Path

    def score(self, seed: str, inputs: ConstructionInputs, wannier) -> GateReport:
        return self.gate.run(
            _gate_construction(
                self.config,
                seed,
                parent=self.parent,
                wannier=wannier,
                material_class=self.material_class,
                inputs={
                    **inputs.gate_inputs(self.config.spin_index),
                    "route": self.backend.route(inputs),
                },
            )
        )


def _attempt(
    run,
    state,
    runner,
    backend,
    parent,
    run_dir,
    config,
    scorer,
    provider_factory,
    *,
    ctx: DeckContext | None = None,
) -> bool:
    """Build and judge the current deck. Returns whether the run ends here."""
    material_class = scorer.material_class
    inputs = state.current_inputs
    try:
        attempt = run.start_attempt(
            _parameters(inputs),
            action=state.pending_action,
            seed=state.current_seed,
            method=method_of(inputs).value,
            proposal=state.pending_meta,
        )
    except BudgetExhausted as exc:
        # The budget ending is an outcome, not a crash: the whole run record survives.
        run.terminal_reason = str(exc)
        return True
    repaired = state.pending_action is not None
    state.pending_action, state.pending_meta = None, {}
    state.last_refused = False

    current_bands = _band_tuple(inputs.win)
    if repaired:
        # Every repaired deck is checked before pw2wannier90 or VASP spends anything on
        # it. The band arithmetic is checked only when the band assignment moved: a
        # baseline that ran is never second-guessed by the parent's reader.
        check = check_deck(
            inputs,
            parent=parent,
            code=backend.name,
            run_dir=run_dir,
            band_arithmetic=current_bands != state.last_run_bands,
            ctx=ctx,
            spin_index=config.spin_index,
            whole_occupied_target=not _sub_manifold(config),
        )
        if not check.ok:
            message = f"the deck check refused it before anything ran: {check.reason()}"
            run.finish_attempt(attempt, calculation_error=message, refused=True)
            state.report = _calculation_failure_report(
                state.current_seed, material_class, message
            )
            state.last_refused = True
            return False
    attempt.overlaps_from = state.overlap_source
    try:
        steps = runner.build(
            state.current_seed, reuse_overlaps_from=state.overlap_source
        )
        state.last_run_bands = current_bands
    except StepStalled as exc:
        # A step that stopped making progress (execution.StepStalled; the rule is in
        # progress.py). On a proposed deck it is a refused attempt: the budget is
        # spent, the reason goes back to the model, and the run goes on. On attempt 0
        # it is the frontier's own construction that stalled: recorded as such
        # (status "stalled", no gate verdict), and the repair loop starts from it.
        completed = list(getattr(runner, "last_steps", []) or [])
        attempt.steps = [s.to_record() for s in completed]
        attempt.stalled = {**exc.record, "frontier": not repaired}
        if attempt.index == 0:
            _remember_reference(runner, state.current_seed, completed)
        message = str(exc)
        if not repaired:
            message = (
                f"the frontier construction (attempt 0, run as published) stalled: "
                f"{message}"
            )
        run.finish_attempt(attempt, calculation_error=message, refused=True)
        state.report = _calculation_failure_report(
            state.current_seed, material_class, message
        )
        # A localisation that stalled leaves complete overlaps behind, which a next
        # deck that does not change them may reuse; any earlier stall leaves none.
        state.last_refused = exc.kind != "localise"
        if exc.kind == "localise":
            state.last_run_bands = current_bands
        return False
    except DeckRefused as exc:
        # A deck nobody can run is information for the next proposal, not the end of
        # the run. The attempt is spent, and its message goes back.
        run.finish_attempt(attempt, calculation_error=str(exc), refused=True)
        state.report = _calculation_failure_report(
            state.current_seed, material_class, str(exc)
        )
        state.last_refused = True
        return False
    except CalculationError as exc:
        # An executable that cannot be launched, overlaps that are missing, a fixed
        # limit a caller set: not about the deck, and the run ends -- with that said
        # in the record. (The V3Si sparse_tb v1 run ended here, on the 6 h limit the
        # campaign then passed; there is no fixed limit by default any more, and a step
        # that stops progressing is a StepStalled above.)
        attempt.steps = [s.to_record() for s in getattr(runner, "last_steps", []) or []]
        run.finish_attempt(attempt, calculation_error=str(exc))
        run.terminal_reason = (
            f"the calculation of attempt {attempt.index} ({state.current_seed}) did not "
            f"complete, and the run ends without a verdict on that deck: {exc}"
        )
        return True
    attempt.steps = [s.to_record() for s in steps]
    if attempt.index == 0:
        _remember_reference(runner, state.current_seed, steps)
    failed = [s for s in steps if not s.ok]
    if failed:
        message = f"{failed[0].step} exited {failed[0].returncode}: {failed[0].stderr_tail[-600:]}"
        run.finish_attempt(attempt, calculation_error=message)
        # A deck the code refuses to run is a diagnosis: some repairs are only correct
        # in combination, and the refusal is what lets the next proposal fix the pair.
        state.report = _calculation_failure_report(
            state.current_seed, material_class, message
        )
        return False
    try:
        wannier = read_run(run_dir / f"{state.current_seed}.wout")
    except WannierOutputError as exc:
        # Wannier90 can exit zero having refused the deck: the same event as a non-zero
        # exit, handed back as a diagnosis.
        message = backend.refusal_reason(run_dir, state.current_seed) or str(exc)
        run.finish_attempt(attempt, calculation_error=message)
        state.report = _calculation_failure_report(
            state.current_seed, material_class, message
        )
        return False
    reason = projectability_not_applied(inputs.win.read_text(), wannier)
    if reason:
        run.finish_attempt(attempt, invalid_reason=reason)
        run.terminal_reason = f"invalid construction {state.current_seed}: {reason}"
        return True
    report = scorer.score(state.current_seed, inputs, wannier)
    attempt.gate_profile = report.profile
    attempt.gate_profile_hash = report.profile_hash
    run.finish_attempt(
        attempt,
        outcome=GateOutcome(
            passed=report.passed,
            evaluated_check_ids=report.evaluated_check_ids,
            failed_check_ids=report.failed_check_ids,
            measurements=report.measurements(),
        ),
    )
    if state.best is None or _score(report) > _score(state.best[0]):
        state.best = (report, state.current_seed)
        state.best_wannier = wannier
    if config.draw_figures:
        state.figures.extend(
            _draw(
                parent, wannier, run_dir, state.current_seed, inputs, config.spin_index
            )
        )
    state.report = report
    if report.passed:
        if not config.review:
            return True
        review = _review(
            provider_factory,
            config,
            parent,
            wannier,
            run_dir,
            state.current_seed,
            state.figures,
        )
        if review is None or review.accepted:
            return True
        # A withheld acceptance re-enters the repair loop with the named fault as the
        # thing to address; the physics gate's own verdict stands.
        state.report = _with_review_fault(report, review)
    return False


def _handoff_attempt(
    run, state, backend, parent, run_dir, config, scorer, handoff
) -> bool:
    """Attempt 0 of a frontier handoff: the frontier's last run, read and scored.

    Nothing is built: the deck and its Wannier90 outputs are the frontier's own, already
    in the run directory. Returns whether the run ends here (a pass, which a handoff is
    never given, or an invalid construction).
    """
    inputs = state.current_inputs
    seed = state.current_seed
    parameters = _parameters(inputs)
    attempt = run.start_attempt(parameters, seed=seed, method=method_of(inputs).value)
    attempt.frontier_point = int(handoff.point)
    # registered after attempt 0 opened: its own parameter set is the last frontier run
    run.start_from_frontier(handoff.to_record(parameters))
    state.last_refused = False
    material_class = scorer.material_class
    wout = Path(run_dir) / f"{seed}.wout"
    refusal = (
        backend.refusal_reason(run_dir, seed)
        if wout.with_suffix(".werr").is_file()
        else None
    )
    wannier = None
    if refusal is None:
        try:
            wannier = read_run(wout)
        except WannierOutputError as exc:
            refusal = backend.refusal_reason(run_dir, seed) or str(exc)
    if wannier is None:
        message = (
            f"the frontier's run at ladder point {handoff.point} (its last run before "
            f"the handoff) produced no construction: Wannier90 stopped: {refusal}"
        )
        run.finish_attempt(attempt, calculation_error=message)
        state.report = _calculation_failure_report(seed, material_class, message)
        return False
    reason = projectability_not_applied(inputs.win.read_text(), wannier)
    if reason:
        run.finish_attempt(attempt, invalid_reason=reason)
        run.terminal_reason = f"invalid construction {seed}: {reason}"
        return True
    report = scorer.score(seed, inputs, wannier)
    attempt.gate_profile = report.profile
    attempt.gate_profile_hash = report.profile_hash
    run.finish_attempt(
        attempt,
        outcome=GateOutcome(
            passed=report.passed,
            evaluated_check_ids=report.evaluated_check_ids,
            failed_check_ids=report.failed_check_ids,
            measurements=report.measurements(),
        ),
    )
    state.best = (report, seed)
    state.best_wannier = wannier
    if config.draw_figures:
        state.figures.extend(
            _draw(parent, wannier, run_dir, seed, inputs, config.spin_index)
        )
    state.report = report
    if report.passed:
        run.terminal_reason = (
            f"the frontier's run at ladder point {handoff.point} passes the gate here; "
            f"there is nothing to hand over"
        )
        return True
    return False


def _remember_reference(runner, seed: str, steps) -> None:
    """Attempt 0's step times bound later steps' silent phases (progress.py)."""
    remember = getattr(runner, "remember_reference", None)
    if remember is not None:
        remember(seed, steps)


def _prime_reference(runner, run: Run) -> None:
    """On resume, the attempt-0 step times come from the record, not from this process."""
    if not run.attempts or not run.attempts[0].steps:
        return
    first = run.attempts[0]
    _remember_reference(
        runner,
        first.seed or run.material,
        [
            StepResult(
                str(s.get("step")), int(s.get("returncode", 1)), float(s["seconds"])
            )
            for s in first.steps
            if s.get("seconds") is not None
        ],
    )


def _resume_after_infrastructure_stop(run, backend, run_dir, scorer, seed, base_inputs):
    """The loop's state after a stop from outside, and whether to build first.

    An attempt the stop interrupted is rebuilt as it was proposed -- same deck, same
    action, same proposal cost -- rather than proposed again; with none interrupted,
    the loop goes on from the last attempt that finished, as after a provider stop.
    """
    stop = run.infrastructure_stops[-1]
    stop["resumed_at"] = _now()
    run.terminal_reason = None
    if run.attempts:
        state = _restore(run, backend, run_dir, scorer)
    else:
        state = _Loop(seed, base_inputs, last_run_bands=_band_tuple(base_inputs.win))
    interrupted = stop.get("interrupted_attempt")
    if not interrupted:
        run.stage = Stage.REPAIR if run.attempts else Stage.AUTOMATIC
        return state, bool(run.attempts)
    rebuilt = interrupted.get("seed") or seed
    state.current_seed = rebuilt
    state.current_inputs = backend.read_construction(run_dir, rebuilt)
    record = interrupted.get("action")
    state.pending_action = _action_from_record(record) if record else None
    state.pending_meta = {
        "tools_called": list(interrupted.get("tools_called") or []),
        "requests": int(interrupted.get("model_requests") or 0),
        "usage": dict(interrupted.get("model_usage") or {}),
        "cost_usd": interrupted.get("cost_usd"),
        "schema_enforcement": interrupted.get("schema_enforcement"),
    }
    state.overlap_source = interrupted.get("overlaps_from")
    run.stage = (
        Stage.AUTOMATIC if interrupted.get("stage") == "automatic" else Stage.REPAIR
    )
    return state, False


def _action_from_record(record: Mapping[str, Any]):
    """The action an attempt's record carries (``orchestrator._action_record``)."""
    from .actions import Action, ParameterChange

    return Action(
        kind=record["action"],
        diagnosis=record.get("diagnosis") or "",
        evidence=tuple(record.get("evidence") or ()),
        expected_effect=record.get("expected_effect") or "",
        confidence=float(record.get("confidence") or 0.0),
        targets=tuple(record.get("targets") or ()),
        changes=tuple(
            ParameterChange(
                c["parameter"], c.get("old"), c.get("new"), c.get("reason") or ""
            )
            for c in record.get("changes") or ()
        ),
        observations=tuple(record.get("observations") or ()),
        method=record.get("method"),
        method_from=record.get("method_from"),
        reason=record.get("reason"),
        num_bands_needed=record.get("num_bands_needed"),
        report=record.get("report"),
    )


def _restore(run: Run, backend, run_dir: Path, scorer: "_Scorer") -> _Loop:
    """The loop's state at the attempt a provider stop interrupted."""
    last = run.attempts[-1]
    seed = last.seed or run.material
    inputs = backend.read_construction(run_dir, seed)
    state = _Loop(seed, inputs)
    if last.outcome is not None:
        wannier = read_run(run_dir / f"{seed}.wout")
        state.report = scorer.score(seed, inputs, wannier)
    else:
        state.report = _calculation_failure_report(
            seed,
            scorer.material_class,
            last.calculation_error or last.invalid_reason or "not run",
        )
        # a stalled localisation left complete overlaps behind (see _attempt)
        state.last_refused = last.refused and not (
            last.stalled and str(last.stalled.get("step", "")).endswith(":localise")
        )
    scored = [a for a in run.attempts if a.outcome is not None and a.seed]
    if scored:
        best = max(
            scored,
            key=lambda a: len(a.outcome.evaluated_check_ids)
            - len(a.outcome.failed_check_ids)
            - sum(
                1
                for m in a.outcome.measurements.values()
                if isinstance(m, Mapping)
                and m.get("applies", True)
                and not m.get("evaluable", True)
            ),
        )
        best_inputs = backend.read_construction(run_dir, best.seed)
        best_wannier = read_run(run_dir / f"{best.seed}.wout")
        state.best = (scorer.score(best.seed, best_inputs, best_wannier), best.seed)
        state.best_wannier = best_wannier
    ran = [
        a
        for a in run.attempts
        if a.seed and not a.refused and a.calculation_error is None
    ]
    if ran:
        state.last_run_bands = _band_tuple(run_dir / f"{ran[-1].seed}.win")
    return state


def _write_result_log(result, path, preflight_error):
    path = result.run.write_log(path)
    record = json.loads(path.read_text())
    record.update(
        parent_adequacy=(
            result.parent_adequacy.to_dict()
            if result.parent_adequacy is not None
            else None
        ),
        terminal_diagnosis=(
            result.diagnosis.to_dict() if result.diagnosis is not None else None
        ),
        prescription=result.prescription,
        parent_limit_requests=getattr(result.run, "_parent_limit_requests", []),
        status=result.status,
    )
    if preflight_error is not None:
        record["preflight_error"] = preflight_error
    scratch = path.with_suffix(path.suffix + ".partial")
    scratch.write_text(json.dumps(record, indent=2, default=str) + "\n")
    scratch.replace(path)


# ---------------------------------------------------------------------------


#: The check identifier a refused deck is reported under. It is not a physical check
#: and never appears in a passing report; it exists so the repair loop has something to
#: target when the calculation did not get far enough to be measured.
CALCULATION_FAILED = recipes.CALCULATION_DID_NOT_RUN


def _refusal_reason(wout: Path) -> str | None:
    """The line Wannier90 prints when it declines a deck, if it printed one.

    Its refusals go to the output file and to a ``.werr`` beside it, not to stderr. The
    refusal follows a line of dots -- ``Exiting.......`` and then the message -- and that
    shape is read first; a keyword scan for "error" picked up the generic hint paragraph
    instead of the cause.
    """
    for path in (wout.with_suffix(".werr"), wout):
        if not path.is_file():
            continue
        lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines()]
        for i, line in enumerate(lines):
            if line.startswith("Exiting"):
                after = [
                    ln for ln in lines[i + 1 :] if ln and not ln.startswith("(rank")
                ]
                if after:
                    return " | ".join(after[:3])
        named = [
            ln
            for ln in lines
            if ln
            and not ln.startswith("Wannier90:")
            and ("Error" in ln or "error" in ln or ln.startswith("dis_windows"))
        ]
        if named:
            return " | ".join(named[-3:])
    return None


def _calculation_failure_report(
    seed: str, material_class: str, message: str
) -> GateReport:
    """Turn a refused calculation into something a repair can be aimed at."""
    from .gate import Verdict

    return GateReport(
        seed,
        material_class,
        (
            Verdict(
                CALCULATION_FAILED,
                evaluable=True,
                passed=False,
                metric_value=None,
                metric_units="",
                threshold_value=None,
                applies=True,
                note=message,
            ),
        ),
    )


def _band_count(inputs) -> int | None:
    """How many bands the deck declares, which bounds any projector set proposed."""
    raw = inputs.values.get("num_bands")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _kept_occupied(parent, inputs) -> int | None:
    """Bands occupied everywhere that the deck keeps (excluded semicore not counted)."""
    if parent.is_metallic():
        return None
    occupied = parent.occupied_bands()
    excluded = set(_excluded_from_text(inputs.win.read_text()))
    return occupied - sum(1 for b in excluded if b <= occupied)


def _species_counts(parent) -> dict[str, int]:
    """How many atoms of each species the cell holds -- what a projector count needs."""
    counts: dict[str, int] = {}
    for symbol in parent.species:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _score(report: GateReport) -> int:
    """How close a construction came, for picking the best effort."""
    return (
        len(report.evaluated_check_ids)
        - len(report.failed_check_ids)
        - len(report.non_evaluable_check_ids)
    )


def _draw(
    parent,
    wannier,
    run_dir: Path,
    seed: str,
    inputs: ConstructionInputs,
    spin_index: int,
) -> list[Path]:
    figures_dir = run_dir / "figures"
    try:
        bands = draw_band_comparison(
            parent,
            wannier,
            figures_dir / f"{seed}_bands.png",
            spin_index=spin_index,
            dis_win_max=_as_float(inputs.values.get("dis_win_max")),
            dis_froz_max=_as_float(inputs.values.get("dis_froz_max")),
            title=seed,
        )
        spreads = draw_spread_profile(wannier, figures_dir / f"{seed}_spreads.png")
    except Exception:
        # A figure is a convenience for a reviewer; failing to draw one must not
        # abandon a construction the measurements already judged.
        return []
    return [bands.path, spreads.path]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).lower().replace("d", "e"))
    except ValueError:
        return None


#: Added to the repair task in a frontier handoff (:class:`FrontierHandoff`).
HANDOFF_TASK = """
This run took over from the frontier method. The frontier ran first: its default deck
(run 0) and its own retry ladder, points 1 to {point}, each a Wannier90 construction
judged under this run's criterion, and none certified. They are listed in the payload
as the frontier's runs, with their outcomes; they are not your attempts and spend none
of your budget. Your starting deck is its last one (ladder point {point}). The tool
frontier_next_proposal says what the frontier's own retry rule would run next from your
latest run deck (untested; changes only dis_proj_min/max), and the rest of its untried
ladder from there; you may use it or not, and calling it costs none of your tool budget.
"""

VALIDATION_ATTEMPTS = 4
"""How many times a refused answer is handed back before the run gives up."""


def _attempt_history(run, unresolved) -> list[dict]:
    return [
        {
            "attempt": (
                a.index
                if a.frontier_point is None
                else f"{a.index} (not yours: the frontier's own run at its ladder point "
                f"{a.frontier_point}, its last run before the handoff; the deck you "
                f"start from)"
            ),
            "method": a.method,
            "switched": a.switch,
            "changed": {
                c["parameter"]: c["new"] for c in (a.action or {}).get("changes", [])
            },
            "measured": {
                check: round(m["value"], 6)
                for check, m in (
                    (a.outcome.measurements or {}) if a.outcome else {}
                ).items()
                if check in unresolved and m.get("value") is not None
            },
            "still_failing": list(a.outcome.failed_check_ids if a.outcome else ()),
            "calculation_error": a.calculation_error,
        }
        for a in run.attempts
    ]


def _propose(
    provider_factory,
    config,
    run,
    parent,
    run_dir,
    seed,
    inputs,
    report,
    figures,
    backend: str | Backend = "qe",
    *,
    ctx: DeckContext | None = None,
    last_run_bands: Any = None,
):
    """Ask the model for one action, with its tools, and validate it.

    Returns ``(action, next_seed, meta)``; ``None`` when the run should end on the
    model's side (no valid answer); or the :class:`ProviderError` that stopped it.
    ``meta`` is what proposing it cost: tool calls, requests, tokens, dollars.

    The figures for this construction go with the request: a scalar says how far the
    model sits from the parent on average; it cannot say *where*.
    """
    backend = get_backend(backend)
    if report is None:
        return None
    unresolved = report.failed_check_ids + report.non_evaluable_check_ids
    if not unresolved:
        return None
    evaluated = report.evaluated_check_ids + report.non_evaluable_check_ids
    run._unresolved = tuple(unresolved)
    method = method_of(inputs)
    if ctx is None:
        ctx = DeckContext(
            code=backend.name,
            run_dir=Path(run_dir),
            parent=parent,
            parent_source=Path(run_dir) / "parent",
            frontier=inputs,
        )

    try:
        wannier = read_run(Path(run_dir) / f"{seed}.wout")
    except WannierOutputError:
        wannier = None

    # Only this construction's own figures: an earlier attempt's picture beside this
    # attempt's numbers is a way to be confidently wrong about which one is described.
    own_figures = [p for p in figures if p.name.startswith(f"{seed}_")]
    artifacts = tuple(
        Artifact(
            name=p.name,
            path=p,
            media_type="image/png",
            caption=(
                "interpolated bands, and the deviation from the parent resolved by energy"
                if "bands" in p.name
                else "final spread of each Wannier function, sorted"
            ),
        )
        for p in own_figures
    )
    payload = RepairPayload(
        material_signature=signature_of(parent, wannier),
        gate_report=report.measurements(),
        current_parameters={
            **dict(inputs.values),
            "semicore_exclusion": _current_semicore(inputs, ctx),
        },
        convergence_trajectory=trajectory_of(wannier) if wannier else [],
        artifacts=artifacts,
        fermi_energy_ev=parent.fermi_energy_ev,
        # The measured value of every unresolved check goes into the history, so the
        # model can see which of its own moves did anything.
        attempt_history=_attempt_history(run, unresolved),
        method=method.value,
        code=backend.name,
        frontier_history=(
            config.frontier_handoff.model_view()
            if getattr(config, "frontier_handoff", None) is not None
            else None
        ),
    )
    species = tuple(sorted(set(parent.species)))
    schema = action_schema(
        Stage.REPAIR,
        evaluated,
        unresolved_check_ids=unresolved,
        allowed_species=species,
        code=backend.name,
        method=method,
    )
    tools = None
    specs = ()
    if config.tools:
        tools = RepairTools(
            ctx=ctx,
            backend=backend,
            run=run,
            inputs=inputs,
            seed=seed,
            recipe_dir=Path(config.recipe_dir) if config.recipe_dir else None,
            frontier_rescan_source=(
                Path(config.frontier_rescan) if config.frontier_rescan else None
            ),
            rescan_key=config.rescan_key,
            frontier_handoff=getattr(config, "frontier_handoff", None),
            recipe_same_material_holdback=getattr(
                config, "recipe_same_material_holdback", False
            ),
            frontier_dataset_same_composition=config.frontier_dataset_same_composition,
            last_run_bands=last_run_bands,
            band_tuple=_band_tuple,
            species=species,
            construction_of=_construction_reader(config, backend, parent, run_dir),
        )
        specs = tools.specs()
    handoff = getattr(config, "frontier_handoff", None)
    system = system_prompt(
        REPAIR_TASK + (HANDOFF_TASK.format(point=handoff.point) if handoff else ""),
        _target_text(config, list(parent.species), parent),
    )
    if handoff is not None:
        system = system.replace("frontier_candidates", "frontier_next_proposal")

    meta: dict[str, Any] = {
        "tools_called": [],
        "requests": 0,
        "usage": {},
        "cost_usd": None,
    }
    cap = run.config.tool_calls_per_attempt
    action = None
    for _ in range(VALIDATION_ATTEMPTS):
        used = sum(1 for c in meta["tools_called"] if c.get("counted", True))
        try:
            provider = provider_factory(config.model)
            if run.schema_enforcement is None and hasattr(
                provider, "schema_enforcement"
            ):
                run.schema_enforcement = provider.schema_enforcement()
            if specs:
                response = provider.propose(
                    system=system,
                    payload_text=payload.render(run.stage),
                    schema=schema,
                    images=[Image(a.path) for a in artifacts],
                    tools=specs,
                    run_tool=tools,
                    tool_call_cap=max(0, cap - used),
                )
            else:
                response = provider.propose(
                    system=system,
                    payload_text=payload.render(run.stage),
                    schema=schema,
                    images=[Image(a.path) for a in artifacts],
                )
        except ProviderError as exc:
            # Tool calls and dollars spent before the failure belong on the record.
            meta["tools_called"] += list(getattr(exc, "tool_calls", []) or [])
            meta["requests"] += int(getattr(exc, "requests", 0) or 0)
            if getattr(exc, "cost_usd", None) is not None:
                meta["cost_usd"] = round((meta["cost_usd"] or 0.0) + exc.cost_usd, 8)
            if getattr(exc, "stop", True):
                run.trailing_proposal = meta
                if run.attempts:
                    run.attempts[-1].model_error = str(exc)
                return exc
            run.attempts[-1].model_error = f"the answer was unusable: {exc}"
            payload.attempt_history = list(payload.attempt_history) + [
                {"attempt": "the answer just given", "rejected_because": str(exc)}
            ]
            continue
        _merge_meta(meta, response)
        payload.tool_calls = list(meta["tools_called"])
        _record_parent_limit_requests(run, parent, response.content, inputs)
        try:
            action = validate_action(
                response.content,
                stage=run.stage,
                evaluated_check_ids=evaluated,
                allowed_species=species,
                unresolved_check_ids=unresolved,
                species_counts=_species_counts(parent),
                band_count=_band_count(inputs),
                parent_band_count=parent.n_bands,
                occupied_bands=_kept_occupied(parent, inputs),
                method=method,
                code=backend.name,
                spinors=bool(getattr(parent, "spinor", False)),
                excluded_below=_excluded_below(inputs),
                sub_manifold_target=_sub_manifold(config),
            )
            if action.kind in ("modify_parameters", "switch_method"):
                for c in action.changes:
                    if not backend.knowledge.supports(c.parameter):
                        reason = backend.knowledge.unsupported_reasons.get(
                            c.parameter,
                            f"{backend.name} has no deck rule for {c.parameter}",
                        )
                        raise ActionRejected(f"{c.parameter}: {reason}")
                    where = run.already_raised(c.parameter, c.new)
                    if where is not None and action.kind == "modify_parameters":
                        raise ActionRejected(
                            f"{c.parameter} was already raised in attempt {where} and "
                            f"the gate did not move; raising it further is the same "
                            f"lever. Something else has to change"
                        )
                prospective = _prospective(
                    action, inputs, parent, run, backend=backend, ctx=ctx
                )
                frontier_run = run.frontier_run_of(prospective)
                if frontier_run is not None:
                    raise ActionRejected(
                        f"that is the parameter set of the frontier's own run "
                        f"{frontier_run} before the handoff; its outcome is in the "
                        f"frontier's runs above. Change something else, or change "
                        f"this by more"
                    )
                if run.would_repeat(prospective):
                    raise ActionRejected(
                        "that is the parameter set of an attempt already made in this "
                        "run; it would produce the same result. Change something else, "
                        "or change this by more"
                    )
            break
        except ActionRejected as exc:
            action = None
            run.attempts[-1].model_error = f"the answer did not validate: {exc}"
            payload.attempt_history = list(payload.attempt_history) + [
                {"attempt": "the answer just given", "rejected_because": str(exc)}
            ]
    if action is None:
        run.trailing_proposal = meta
        run.terminal_reason = (
            f"the model's answer did not validate in {VALIDATION_ATTEMPTS} tries; "
            f"the last refusal was: {run.attempts[-1].model_error}"
        )
        return None
    next_seed = f"{run.material}_r{len(run.attempts)}"
    return action, next_seed, meta


def _current_semicore(inputs, ctx) -> Any:
    """The deck's semicore exclusion as the knob names it, or why it has no value."""
    from .semicore import SemicoreUnavailable, catalogue

    # Shown to the model, never relied on: anything that cannot be read is said so.
    try:
        pw2wan = getattr(inputs, "pw2wan", None)
        pw = pw2wan.read_text() if pw2wan is not None and pw2wan.is_file() else None
        cat = catalogue(inputs.win.read_text(), pw, inputs.win.parent, ctx)
    except (SemicoreUnavailable, InputError) as exc:
        return f"not available: {exc}"
    except (AttributeError, OSError, ValueError) as exc:
        return f"not available: the deck could not be read ({exc})"
    if not cat.known:
        return (
            f"bottom {cat.bottom} bands excluded; which shells cannot be told from the "
            "deck (name every species)"
        )
    return cat.current_value()


def _merge_meta(meta: dict, response) -> None:
    meta["tools_called"] = list(meta["tools_called"]) + list(
        getattr(response, "tool_calls", []) or []
    )
    meta["requests"] += int(getattr(response, "requests", 1) or 1)
    for key, value in (getattr(response, "usage", {}) or {}).items():
        if isinstance(value, (int, float)):
            meta["usage"][key] = meta["usage"].get(key, 0) + value
    seconds = getattr(response, "seconds", None)
    if seconds is not None:
        # Wall time waiting on the model: the local arm's only cost besides tokens.
        meta["usage"]["seconds"] = round(
            meta["usage"].get("seconds", 0.0) + float(seconds), 3
        )
    cost = getattr(response, "cost_usd", None)
    if cost is not None:
        meta["cost_usd"] = round((meta["cost_usd"] or 0.0) + float(cost), 8)
    enforcement = getattr(response, "schema_enforcement", None)
    if enforcement:
        # The answer that is finally taken is the one this records.
        meta["schema_enforcement"] = dict(enforcement)


def _excluded_below(inputs) -> int:
    """How many of the lowest parent bands the deck excludes (its bottom block)."""
    from .semicore import bottom_and_top

    win = getattr(inputs, "win", None)
    if win is None:
        return 0
    try:
        return bottom_and_top(_excluded_from_text(win.read_text()))[0]
    except (InputError, OSError):
        return 0


def _record_parent_limit(run, **record) -> None:
    """One piece of evidence that the parent's band pool binds the repair."""
    record = {"attempt": len(run.attempts) - 1, **record}
    if not hasattr(run, "_parent_limit_requests"):
        run._parent_limit_requests = []
    if record not in run._parent_limit_requests:
        run._parent_limit_requests.append(record)


def _band_pool_escalation(action, meta, run, state, parent, backend, ctx, run_dir):
    """Whether an escalation names the parent band pool, and whether validate_deck agrees.

    The claim is the action's ``num_bands_needed``; failing that, a ``validate_deck``
    call the model made while proposing this escalation that asked for more bands than
    the parent supplies and was refused, when the escalation's own text speaks of bands.
    The claim is then put to ``validate_deck`` by the loop itself: it is confirmed only
    when the tool refuses that ``num_bands`` *and* the count exceeds the parent's
    bands less the bottom exclusion. Returns ``None`` when no band-pool claim is made.
    """
    inputs = state.current_inputs
    below = _excluded_below(inputs)
    ceiling = int(parent.n_bands) - below
    needed, source = action.num_bands_needed, "num_bands_needed"
    if needed is None:
        text = " ".join([action.diagnosis, *action.observations]).lower()
        if "band" not in text:
            return None
        probed = []
        for call in meta.get("tools_called") or []:
            if call.get("name") != "validate_deck":
                continue
            result = str(call.get("result") or "").replace(" ", "")
            if not result.startswith('{"ok":false'):
                continue
            for change in (call.get("arguments") or {}).get("changes") or []:
                if not isinstance(change, Mapping):
                    continue
                if change.get("parameter") != "num_bands":
                    continue
                try:
                    value = int(change.get("new"))
                except (TypeError, ValueError):
                    continue
                if value > ceiling:
                    probed.append(value)
        if not probed:
            return None
        needed, source = max(probed), "a refused validate_deck probe in its proposal"
    pool = {
        "num_bands_needed": int(needed),
        "excluded_below": below,
        "parent_bands": int(parent.n_bands),
        "ceiling": ceiling,
        "parent_bands_needed": int(needed) + below,
        "source": source,
        "confirmed": False,
    }
    try:
        tools = RepairTools(
            ctx=ctx,
            backend=backend,
            run=run,
            inputs=inputs,
            seed=state.current_seed,
            last_run_bands=state.last_run_bands,
            band_tuple=_band_tuple,
            species=tuple(sorted(set(parent.species))),
        )
        verdict = tools.tool_validate_deck(
            changes=[{"parameter": "num_bands", "new": int(needed)}]
        )
    except Exception as exc:  # a confirmation that cannot run confirms nothing
        pool["error"] = f"validate_deck could not run: {exc}"
        return pool
    pool["validate_deck"] = {
        "ok": bool(verdict.get("ok")),
        "problems": list(verdict.get("problems") or []),
    }
    pool["confirmed"] = (not verdict.get("ok")) and int(needed) > ceiling
    return pool


def _record_parent_limit_requests(run, parent, content, inputs=None):
    """Retain band-ceiling evidence even when the proposed action is rejected.

    The ceiling is the parent's bands less the deck's bottom (semicore) exclusion, and
    what is recorded is the parent band count the request implies (``requested`` =
    the bands asked for plus the excluded ones), which is what a prescription sets.
    """
    if not isinstance(content, Mapping):
        return
    changes = content.get("changes")
    if not isinstance(changes, (list, tuple)):
        return
    moves_bottom = any(
        isinstance(c, Mapping) and c.get("parameter") == "semicore_exclusion"
        for c in changes
    )
    below = 0 if (inputs is None or moves_bottom) else _excluded_below(inputs)
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        parameter = change.get("parameter")
        try:
            if parameter == "num_bands":
                value = change.get("new")
                if isinstance(value, bool):
                    continue
                requested = int(value)
                if float(value) != requested:
                    continue
            elif parameter == "projectors":
                requested = implied_wannier_count(
                    change.get("new"),
                    _species_counts(parent),
                    getattr(parent, "spinor", False),
                )
            else:
                continue
        except (TypeError, ValueError, KeyError, OverflowError):
            continue
        if requested > parent.n_bands - below:
            record = dict(
                parameter=parameter,
                requested=requested + below,
                parent_bands=parent.n_bands,
            )
            if below:
                record["excluded_below"] = below
            _record_parent_limit(run, **record)


def _prospective(
    action,
    inputs,
    parent,
    run=None,
    backend: str | Backend = "qe",
    ctx: DeckContext | None = None,
) -> dict:
    """The parameter set this action would produce, obtained by producing it.

    Mirroring the deck writer's derivations by hand is how this guard kept missing; the
    deck is written into a scratch directory and read back, so the comparison is against
    what the run will actually record.
    """
    backend = get_backend(backend)
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        run_dir = inputs.win.parent
        probe_directory(run_dir, inputs, directory)
        source = backend.read_construction(directory, inputs.seed)
        try:
            if getattr(action, "method", None) is not None:
                if ctx is None:
                    return {}
                derived = derive_deck(
                    backend=backend,
                    source=source,
                    ctx=ctx,
                    run_dir=directory,
                    new_seed=f"{inputs.seed}_probe",
                    changes={c.parameter: c.new for c in action.changes},
                    method=action.method,
                    fermi_energy_ev=parent.fermi_energy_ev,
                    species_counts=_species_counts(parent),
                )
            elif ctx is not None:
                # derive_deck resolves a semicore shell list against the run's parent and
                # first deck, which a bare writer cannot.
                derived = derive_deck(
                    backend=backend,
                    source=source,
                    ctx=ctx,
                    run_dir=directory,
                    new_seed=f"{inputs.seed}_probe",
                    changes={c.parameter: c.new for c in action.changes},
                    method=None,
                    fermi_energy_ev=parent.fermi_energy_ev,
                    species_counts=_species_counts(parent),
                )
            else:
                derived = backend.write_derived(
                    source,
                    directory,
                    f"{inputs.seed}_probe",
                    {c.parameter: c.new for c in action.changes},
                    fermi_energy_ev=parent.fermi_energy_ev,
                    species_counts=_species_counts(parent),
                )
        except (InputError, StartingDeckUnavailable) as exc:
            # The writer refuses this proposal: that is a validation refusal, and its
            # reason goes back to the model with the retry, before any attempt is
            # spent on a deck that cannot exist.
            raise ActionRejected(
                f"the proposed deck could not be written: {exc}"
            ) from exc
        except Exception as exc:
            # Anything else is the probe's own trouble, not the proposal's: here it
            # simply is not a repeat of anything. Recorded, because a guard that
            # silently stops guarding is worse than one that is absent.
            if run is not None and run.attempts:
                run.attempts[-1].requests.append(
                    f"the repeat check could not build the proposed deck: {exc}"
                )
            return {}
        return _parameters(derived)


def _with_review_fault(report: GateReport, review) -> GateReport:
    """Record the review's fault as an unresolved check the next repair must target."""
    from .gate import Verdict

    fault = Verdict(
        check_id="visual_review",
        evaluable=True,
        passed=False,
        metric_value=None,
        metric_units="",
        threshold_value=None,
        applies=True,
        note=review.fault or "the review withheld acceptance",
    )
    return GateReport(
        report.construction_id, report.material_class, report.verdicts + (fault,)
    )


def _review(provider_factory, config, parent, wannier, run_dir, seed, figures):
    """The review's outcome, or ``None`` when there was no review to be had."""
    if provider_factory is None or not figures:
        return None
    try:
        return ReviewGate(provider_factory(config.model)).review(
            figures=[Image(p) for p in figures[-2:]],
            captions=[p.name for p in figures[-2:]],
            signature=signature_of(parent, wannier),
        )
    except (ProviderError, ValueError):
        return None
