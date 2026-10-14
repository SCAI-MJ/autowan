"""The read-only tools a repair model may call before it proposes.

Each tool is deterministic, reads what is already on disk, and changes nothing: calling
one costs no attempt. The model reaches them through the provider's function calling,
up to a per-attempt cap, and every call is logged with the attempt it served, so whether
a repair was found from general principles, from a deck check, or from a reference page
stays separable after the fact.

First wave (autowan item 4 decisions 12 and 14):

``validate_deck``        would this proposal run? (method/knob consistency, Wannier
                         count, band arithmetic, the gate's honour rule for the
                         bottom (semicore) exclusion, frozen capacity, p <= 1 for
                         PDWF), with the semicore shells and every cut they imply
``method_defaults``      the starting deck of a method family on this engine
``frozen_capacity``      how many states a frozen window up to e_max would hold
``centres_and_spreads``  each function's centre, spread, nearest atom and distance
``band_character``       species / angular-momentum composition of the states in an
                         energy window, and every parent band in it (excluded ones
                         too): its energy span, the gap above it, its character
``reference_pages``      short paraphrased pages on the method and its knobs
``frontier_candidates``  the frontier's own next proposals from the current state: its
                         PDWF retry ladder's projectability thresholds, in its order,
                         minus those already run here -- untested, never scored
``frontier_next_proposal`` in a frontier handoff, in place of frontier_candidates: the
                         frontier's retry rule applied to the latest deck that ran here
                         -- that deck with only its projectability thresholds moved to
                         the ladder's next pair; one untested proposal, advancing only
                         after a new deck has run
``frontier_dataset``     the published PDWF frontier runs on similar materials (Qiao
                         2023; Jiang 2025), grouped by pseudopotential family
``search_references``    autowan's own corpus of repairs (own composition held back when
                         the run asks for it: ``recipe_same_material_holdback``)

Second wave (the poster campaign's failed runs, 2026-10-09):

``truncation_diagnostics`` for the sparse tight-binding criterion: r_c and L_BvK, the
                         hopping norm binned by the real hopping distance
                         |R + tau_j - tau_i|, the share of |H|^2 beyond r_c, the decay
                         length, the truncated eta_2 beside the untruncated one, where
                         the truncated model's error sits (energy window, k-point,
                         band), and the functions that carry the beyond-r_c weight
``band_error_by_energy`` eta at E_F + nu for nu = 0..5 eV, with the worst k-points and
                         bands in each window, for the construction's own model
``structure``            the lattice and atoms as stored and wrapped into [0, 1), atoms
                         stored outside the cell flagged, nearest neighbours per site,
                         and the vacuum / periodic-direction reading criterion 2 uses

The criterion numbers these report are the gate's own: the tools run the registered
check, or call the module functions the check calls, and never re-derive a definition.

Nothing a tool returns names the material: the signature the payload already carries
(species and counts) is the most any tool reveals.
"""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import recipes, semicore
from .actions import (
    BY_NAME,
    ActionRejected,
    _parameter_value_schema,
    _validate_value,
    cross_family_reason,
)
from .band_distance_eta import ETA2_SUCCESS_MEV
from .deckcheck import check_deck, frozen_counts, kept_band_energies
from .families import (
    DESCRIPTIONS,
    UNAVAILABLE_REASONS,
    Method,
    as_method,
    available,
    knobs_for,
)
from .gate import NonEvaluable
from .inputs import (
    InputError,
    _excluded_from_text,
    _parse_mp_grid,
    assignments,
    deck_atom_species,
    method_of,
    pao_excluded,
    pao_layout,
    to_relative,
)
from .methods import (
    DeckContext,
    StartingDeckUnavailable,
    derive_deck,
    probe_directory,
    starting_deck,
)
from .providers import ToolSpec

REFERENCE_DIR = Path(__file__).parent / "references"
#: Longest tool result handed back, in characters.
RESULT_LIMIT = 8000
#: Tool calls one attempt may make before it must answer.
DEFAULT_TOOL_CALL_CAP = 8


def reference_topics() -> dict[str, str]:
    """Topic -> the first heading of its page."""
    out = {}
    for path in sorted(REFERENCE_DIR.glob("*.md")):
        if path.stem.lower() == "readme":
            continue
        first = path.read_text().splitlines()[0].lstrip("# ").strip()
        out[path.stem] = first
    return out


def _dump(value: Any) -> str:
    text = json.dumps(value, separators=(",", ":"), default=_default)
    if len(text) > RESULT_LIMIT:
        text = text[:RESULT_LIMIT] + "\n... (truncated)"
    return text


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _change_entries(schema_species) -> dict:
    # ``new`` is typed as the union of the vocabulary's value shapes -- the same union
    # the final answer uses -- so a strict or guided schema can express it.
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["parameter", "new"],
            "properties": {
                "parameter": {"type": "string", "enum": [p for p in BY_NAME]},
                "new": {
                    "anyOf": [
                        _parameter_value_schema(BY_NAME[n], tuple(schema_species))
                        for n in BY_NAME
                    ]
                },
            },
        },
    }


@dataclass
class RepairTools:
    """The tools for one construction, bound to its run, deck and parent."""

    ctx: DeckContext
    backend: Any
    run: Any
    inputs: Any
    seed: str
    recipe_dir: Path | None = None
    frontier_rescan_source: Path | None = None
    rescan_key: str | None = None
    recipe_same_material_holdback: bool = False
    """Hold back ``search_references`` entries distilled from this run's own material
    (its composition, or its seed for entries without species). Off by default: public
    use may draw on an earlier repair of the same material, as ``frontier_dataset``
    already may (decision 14); the poster drivers turn it on (2026-10-13, round 2)."""
    frontier_handoff: Any = None
    """A :class:`~.workflow.FrontierHandoff`. The menu then carries
    ``frontier_next_proposal`` (the frontier's retry rule applied to this run's latest
    run deck) in place of ``frontier_candidates``."""
    frontier_dataset_source: Path | None = None
    frontier_dataset_same_composition: bool = True
    last_run_bands: Any = None
    band_tuple: Callable | None = None
    species: tuple[str, ...] = ()
    construction_of: Callable[[str], Any] | None = None
    """seed -> the gate's :class:`~.gate.Construction` for that seed, carrying the run's
    reference band path. The criterion tools read the last construction through it."""
    calls: list[dict] = field(default_factory=list)

    # -- the menu ----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        methods = [m.value for m in available(self.ctx.code)]
        energy = {
            "type": "number",
            "description": "eV relative to the Fermi level, positive above",
        }
        return [
            ToolSpec(
                "validate_deck",
                "Check a proposed deck without running it: whether its knobs belong to "
                "its method family, the Wannier count against the projectors or orbitals, "
                "the band arithmetic against the parent, whether the gate would honour "
                "its bottom (semicore) exclusion (the excluded block must be isolated "
                "below every kept band), the frozen states against num_wann at every "
                "k-point and spin, and projectability <= 1 for PDWF. It also lists the "
                "semicore shells semicore_exclusion can name, every bottom cut they "
                "imply with its gap and the gate's verdict, and the lowest bands. "
                "Give the same changes you would propose, and the method if switching.",
                {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": methods},
                        "changes": _change_entries(self.species),
                    },
                },
            ),
            ToolSpec(
                "method_defaults",
                "The starting deck of a method family on this engine for this "
                "construction: what switch_method would build before your edits, its "
                "knobs, and the protocol it follows.",
                {
                    "type": "object",
                    "required": ["method"],
                    "properties": {"method": {"type": "string", "enum": methods}},
                },
            ),
            ToolSpec(
                "frozen_capacity",
                "How many kept states a frozen window from e_min (default: the lowest "
                "kept band) up to e_max holds at the worst k-point and spin, against "
                "num_wann, and the highest e_max that still fits.",
                {
                    "type": "object",
                    "required": ["e_max"],
                    "properties": {"e_max": energy, "e_min": energy},
                },
            ),
            ToolSpec(
                "centres_and_spreads",
                "Each Wannier function of the last construction that ran: its centre, "
                "spread, the nearest atom (species and site index) and the distance to it.",
                {"type": "object", "properties": {}},
            ),
            ToolSpec(
                "band_character",
                "The species and angular-momentum composition of the parent states in an "
                "energy window, and where the composition was read from; then every "
                "parent band in the window, the deck's excluded ones included: its "
                "energy span, the gap to the next band, whether the deck excludes it, "
                "and its character where the parent supplies one.",
                {
                    "type": "object",
                    "required": ["e_min", "e_max"],
                    "properties": {"e_min": energy, "e_max": energy},
                },
            ),
            ToolSpec(
                "reference_pages",
                "A short page on one topic: what the knobs of a method do, the published "
                "protocols and their sources. Topics: "
                + "; ".join(f"{k} ({v})" for k, v in reference_topics().items()),
                {
                    "type": "object",
                    "required": ["topic"],
                    "properties": {
                        "topic": {"type": "string", "enum": list(reference_topics())}
                    },
                },
            ),
            (
                ToolSpec(
                    "frontier_next_proposal",
                    "What the frontier's own retry rule would run next from your latest "
                    "run deck (untested; changes only dis_proj_min/max): the most recent "
                    "deck that actually ran in this run (at the start, the frontier's "
                    "ladder point it stopped at), with only its projectability "
                    "thresholds replaced by the ladder's next pair -- every other value "
                    "of that deck, your changes included, kept -- and, after it, the "
                    "rest of the ladder's untried pairs from that deck in the order the "
                    "rule would take them. Proposals only: none is run or scored, and "
                    "nothing is known about how any would do. It advances only after a "
                    "new deck has run; asked again before that, it answers the same. "
                    "Using it is up to you; calling it does not count against your "
                    "tool budget. Says so when the ladder is exhausted or the deck has "
                    "no thresholds.",
                    {"type": "object", "properties": {}},
                    uncapped=True,
                )
                if self.frontier_handoff is not None
                else ToolSpec(
                    "frontier_candidates",
                    "What the frontier method would try next from the current state: "
                    "the projectability thresholds (dis_proj_min, dis_proj_max) of its "
                    "own automatic retry ladder for its PDWF deck, in the order its "
                    "retry rule takes them, without the ones this run has already run. "
                    "These are untested proposals: no candidate has been run or scored, "
                    "and nothing is known about how any of them would do. Says so when "
                    "the frontier proposes none.",
                    {"type": "object", "properties": {}},
                )
            ),
            ToolSpec(
                "frontier_dataset",
                "The published PDWF frontier's own Wannierisations of similar "
                "materials (Qiao 2023, Jiang 2025; up to 12 rows): formula, space "
                "group, method arm, spin, pseudopotential family, num_wann, "
                "exclude_bands, the projectability thresholds actually used, "
                "dis_froz_max, k-mesh, eta2 and exit status. Ranked by element overlap "
                "(this run's species when none are given), grouped by pseudopotential "
                "family. "
                + (
                    "Same-composition records are included."
                    if self.frontier_dataset_same_composition
                    else "Records of this run's own composition are held back from "
                    "every query."
                ),
                {
                    "type": "object",
                    "properties": {
                        "elements": {"type": "array", "items": {"type": "string"}},
                        "formula": {"type": "string"},
                        "pseudo_family": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                    },
                },
            ),
            ToolSpec(
                "search_references",
                "Search autowan's corpus of earlier repairs that resolved a check still "
                "open here, ranked by how alike the calculations are (never by name). "
                + (
                    "Entries from this material's own composition are held back."
                    if self.recipe_same_material_holdback
                    else "Entries from this material's own composition are included."
                ),
                {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string", "maxLength": 300}},
                },
            ),
            ToolSpec(
                "truncation_diagnostics",
                "For the sparse tight-binding criterion (eta_2 of the model with every "
                "hopping longer than r_c = L_BvK/4 removed), on the last construction "
                "that ran: r_c and L_BvK and how L_BvK was derived (periodic "
                "directions); the hopping norm binned by the real hopping distance "
                "|R + tau_j - tau_i|; the share of |H|^2 beyond r_c; the fitted decay "
                "length; the truncated eta_2 beside the untruncated one; where the "
                "truncated model's error sits (eta at E_F + 0, 1, 2 eV, worst k-points "
                "and bands); and the centre and spread of the functions that carry the "
                "hopping weight beyond r_c. profile: a gate profile to read the "
                "criterion under (default: this run's).",
                {
                    "type": "object",
                    "properties": {"profile": {"type": "string"}},
                },
            ),
            ToolSpec(
                "band_error_by_energy",
                "For the last construction that ran, against the DFT reference path: "
                "eta (and eta_max) at E_F + nu for nu = 0, 1, 2, 3, 4, 5 eV, and in each "
                "window the k-points and bands where the error is largest. Shows "
                "whether a failing eta comes from near E_F or from the top of the "
                "window.",
                {"type": "object", "properties": {}},
            ),
            ToolSpec(
                "structure",
                "The crystal structure (no name): lattice vectors, lengths and angles; "
                "each atom's species and position as stored and wrapped into [0, 1), "
                "with atoms stored outside the cell flagged; each site's nearest "
                "neighbours and distances; and the vacuum gap and periodic directions "
                "per lattice vector, read the way the sparse criterion reads them.",
                {"type": "object", "properties": {}},
            ),
        ]

    # -- dispatch ----------------------------------------------------------

    def __call__(self, name: str, arguments: dict) -> str:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"there is no tool named {name!r}"
        # Under schema enforcement every optional argument is sent, as null when the
        # model leaves it out; null means "not given".
        given = {k: v for k, v in (arguments or {}).items() if v is not None}
        try:
            result = handler(**given)
        except TypeError as exc:
            result = {"error": f"bad arguments for {name}: {exc}"}
        except (InputError, ValueError, OSError, NonEvaluable) as exc:
            result = {"error": str(exc)}
        text = result if isinstance(result, str) else _dump(result)
        self.calls.append({"name": name, "arguments": arguments})
        return text

    # -- helpers -----------------------------------------------------------

    @property
    def fermi(self) -> float:
        return float(self.ctx.parent.fermi_energy_ev)

    def _abs(self, value):
        return None if value is None else float(value) + self.fermi

    def _rel(self, value):
        return None if value is None else round(float(value) - self.fermi, 4)

    # -- tools -------------------------------------------------------------

    def tool_validate_deck(self, changes=None, method=None):
        current = method_of(self.inputs)
        target = as_method(method) if method else current
        problems: list[str] = []
        if target not in available(self.ctx.code):
            return {
                "ok": False,
                "problems": [
                    f"{target.value.upper()} is not available on {self.ctx.code}: "
                    + UNAVAILABLE_REASONS.get((self.ctx.code, target), "no route")
                ],
            }
        parsed: dict[str, Any] = {}
        for entry in changes or []:
            if not isinstance(entry, dict):
                problems.append("each change must be an object")
                continue
            name = entry.get("parameter")
            if name not in BY_NAME:
                problems.append(f"{name!r} is not an editable parameter")
                continue
            if name not in knobs_for(target):
                problems.append(cross_family_reason(name, target))
                continue
            try:
                parsed[name] = _validate_value(
                    BY_NAME[name], entry.get("new"), self.species
                )
            except ActionRejected as exc:
                problems.append(str(exc))
        if problems:
            return {"ok": False, "method": target.value, "problems": problems}
        if target is current and not parsed:
            deck_inputs = self.inputs
            with tempfile.TemporaryDirectory() as scratch:
                return self._check(deck_inputs, Path(scratch), target, current)
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            probe_directory(self.ctx.run_dir, self.inputs, scratch)
            source = self.backend.read_construction(scratch, self.inputs.seed)
            probe_ctx = DeckContext(
                code=self.ctx.code,
                run_dir=self.ctx.run_dir,
                parent=self.ctx.parent,
                parent_source=self.ctx.parent_source,
                frontier=self.ctx.frontier,
                spin_index=self.ctx.spin_index,
                _cache=self.ctx._cache,
            )
            try:
                derived = derive_deck(
                    backend=self.backend,
                    source=source,
                    ctx=probe_ctx,
                    run_dir=scratch,
                    new_seed=f"{self.inputs.seed}_probe",
                    changes=parsed,
                    method=target if target is not current else None,
                    fermi_energy_ev=self.fermi,
                    species_counts=_counts(self.ctx.parent.species),
                )
            except (InputError, StartingDeckUnavailable) as exc:
                return {"ok": False, "method": target.value, "problems": [str(exc)]}
            return self._check(derived, scratch, target, current)

    def _check(self, deck, scratch, target, current):
        changed = (
            self.band_tuple(deck.win) != self.last_run_bands
            if self.band_tuple is not None
            else True
        )
        result = check_deck(
            deck,
            parent=self.ctx.parent,
            code=self.ctx.code,
            run_dir=deck.win.parent,
            band_arithmetic=changed,
            ctx=self.ctx,
            spin_index=self.ctx.spin_index,
            whole_occupied_target=(
                (getattr(self.run, "gate_profile", None) or {}).get(
                    "target_manifold", "whole_occupied_manifold"
                )
                == "whole_occupied_manifold"
            ),
        )
        values = {
            k: to_relative(k, v, self.fermi)
            for k, v in deck.values.items()
            if k not in ("projection_source",)
        }
        values["exclude_bands"] = assignments(deck.win.read_text()).get("exclude_bands")
        profile = getattr(self.run, "gate_profile", None) or {}
        return {
            **result.to_dict(),
            "switch": (
                None if target is current else f"{current.value} -> {target.value}"
            ),
            "deck": values,
            # What the deck must eventually satisfy: the run's gate profile, and which of
            # its checks the last construction left failing or unevaluated.
            "gate_profile": profile.get("name"),
            "gate_profile_checks": profile.get("checks"),
            "checks_unresolved_now": list(getattr(self.run, "_unresolved", ())),
            "semicore": semicore.describe(
                deck.win.read_text(),
                (
                    deck.pw2wan.read_text()
                    if deck.pw2wan is not None and deck.pw2wan.is_file()
                    else None
                ),
                deck.win.parent,
                self.ctx,
                self.ctx.spin_index,
            ),
        }

    def tool_method_defaults(self, method):
        target = as_method(method)
        out = {
            "method": target.value,
            "available_on_this_engine": [m.value for m in available(self.ctx.code)],
            "what_it_is": DESCRIPTIONS[target],
            "knobs": list(knobs_for(target)),
            "current_method": method_of(self.inputs).value,
        }
        try:
            start = starting_deck(target, self.ctx)
            out["starting_deck"] = start.summary()
        except StartingDeckUnavailable as exc:
            out["starting_deck"] = f"not available: {exc}"
            return out
        try:
            cat = semicore.catalogue(
                start.win_text, start.pw2wan_text, self.ctx.run_dir, self.ctx
            )
            out["semicore_exclusion"] = {
                "excluded_now": cat.current_value(),
                "bottom_exclusion_bands": cat.bottom,
                "shells": [s.to_dict() for s in cat.shells],
            }
        except (semicore.SemicoreUnavailable, InputError) as exc:
            out["semicore_exclusion"] = f"not available: {exc}"
        return out

    def _deck_bands(self):
        text = self.inputs.win.read_text()
        values = assignments(text)
        try:
            num_bands = int(float(values.get("num_bands")))
        except (TypeError, ValueError):
            num_bands = None
        try:
            num_wann = int(float(values.get("num_wann")))
        except (TypeError, ValueError):
            num_wann = None
        excluded = _excluded_from_text(text)
        return kept_band_energies(self.ctx.parent, excluded, num_bands), num_wann

    def tool_frozen_capacity(self, e_max, e_min=None):
        bands, num_wann = self._deck_bands()
        hi, lo = self._abs(e_max), self._abs(e_min)
        counts = frozen_counts(bands, lo, hi)
        worst = int(counts.max()) if counts.size else 0
        spin, k = np.unravel_index(int(counts.argmax()), counts.shape)
        out = {
            "window_ev_relative": [e_min, e_max],
            "num_wann": num_wann,
            "frozen_states_max": worst,
            "at_kpoint": int(k) + 1,
            "fits": num_wann is not None and worst <= num_wann,
        }
        if counts.shape[0] > 1:
            out["at_spin"] = int(spin) + 1
            out["per_spin_max"] = [int(c.max()) for c in counts]
        if num_wann is not None and bands.shape[-1] > num_wann:
            pool = bands if lo is None else np.where(bands >= lo, bands, np.inf)
            nth = np.sort(pool, axis=-1)[..., num_wann]
            out["highest_e_max_that_fits"] = round(
                float(nth.min()) - self.fermi - 1e-3, 4
            )
        return out

    def _last_run(self):
        from .w90_outputs import WannierOutputError, read_run

        for attempt in reversed(self.run.attempts):
            seed = getattr(attempt, "seed", None)
            if attempt.outcome is None or not seed:
                continue
            try:
                return seed, read_run(self.ctx.run_dir / f"{seed}.wout")
            except WannierOutputError:
                continue
        return None, None

    def tool_centres_and_spreads(self):
        seed, wannier = self._last_run()
        if wannier is None or wannier.centres_ang is None:
            return "no construction has run far enough to report centres"
        rows = [
            self._function_row(wannier, index)
            for index in range(len(wannier.centres_ang))
        ]
        return {"construction_attempt": _attempt_of(self.run, seed), "functions": rows}

    def _function_row(self, wannier, index: int) -> dict:
        """Function ``index`` (0-based): centre, spread, nearest atom and distance."""
        parent = self.ctx.parent
        centre = np.asarray(wannier.centres_ang[index], dtype=float)
        atom, distance = nearest_site(parent.lattice_ang, parent.positions_frac, centre)
        spreads = getattr(wannier, "spreads_ang2", None)
        return {
            "wf": index + 1,
            "centre_ang": [round(float(x), 4) for x in centre],
            "spread_ang2": (
                None if spreads is None else round(float(spreads[index]), 4)
            ),
            "nearest_atom": f"{parent.species[atom]}#{atom + 1}",
            "distance_ang": round(float(distance), 4),
        }

    # -- the criterion tools (second wave) ----------------------------------

    def _last_construction(self):
        """``(attempt index, Construction)`` for the last attempt that ran and was judged."""
        if self.construction_of is None or self.run is None:
            return None, None
        from .w90_outputs import WannierOutputError

        for attempt in reversed(self.run.attempts):
            seed = getattr(attempt, "seed", None)
            if attempt.outcome is None or not seed:
                continue
            try:
                return attempt.index, self.construction_of(seed)
            except (WannierOutputError, InputError, OSError):
                continue
        return None, None

    def tool_band_error_by_energy(self):
        from .criteria import _criterion_fermi, _exclude, _path_bands

        index, c = self._last_construction()
        if c is None:
            return NO_CONSTRUCTION
        try:
            ref, dft, wan = _path_bands(c)
            ef, ef_source = _criterion_fermi(c, ref)
        except NonEvaluable as exc:
            return {"error": f"not evaluable: {exc}"}
        return {
            "construction_attempt": index,
            "model": "the construction's own H(R), Wannier90's interpolation, untruncated",
            "fermi_energy_ev": round(ef, 4),
            "fermi_energy_source": ef_source,
            "reference": "DFT bands on the separate high-symmetry path the criteria use",
            "success_mev": ETA2_SUCCESS_MEV,
            **energy_resolved_error(
                dft, wan, ef, _exclude(c), ref.kpoints_frac, ENERGY_WINDOWS_EV
            ),
        }

    def tool_truncation_diagnostics(self, profile=None):
        from .criteria import TRUNCATED_HOPPING_ETA2, _criterion_fermi, _exclude, _model
        from .gate import CertificationGate, GateDefinitionError
        from .gate_profiles import get_profile

        index, c = self._last_construction()
        if c is None:
            return NO_CONSTRUCTION
        active = (getattr(self.run, "gate_profile", None) or {}).get("name")
        name = profile or active or "default"
        try:
            judged = TRUNCATED_HOPPING_ETA2.check_id in get_profile(name).check_ids
        except GateDefinitionError as exc:  # an unknown name: say so, do not guess
            return {"error": f"gate profile {name!r}: {exc}"}
        verdict = (
            CertificationGate([TRUNCATED_HOPPING_ETA2], profile_name=name)
            .run(c)
            .verdicts[0]
        )
        record = dict(verdict.record or {})
        out: dict[str, Any] = {
            "construction_attempt": index,
            "profile": name,
            "judged_by_profile": judged,
            "criterion": TRUNCATED_HOPPING_ETA2.check_id,
            "evaluable": verdict.evaluable,
            "passed": verdict.passed,
            "eta2_truncated_mev": _round(verdict.metric_value),
            "eta2_untruncated_mev": _round(record.get("eta2_untruncated_mev")),
            "success_mev": ETA2_SUCCESS_MEV,
            "note": verdict.note,
        }
        if not verdict.evaluable:
            return out
        out["range"] = {
            k: _round(record[k]) if isinstance(record.get(k), float) else record.get(k)
            for k in (
                "r_c_ang",
                "l_bvk_ang",
                "l_bvk_definition",
                "periodic_dims",
                "periodic_axes",
                "vacuum_axes",
                "vacuum_gap_ang",
                "nearest_neighbour_ang",
            )
            if k in record
        }
        out["mp_grid"] = list(c.inputs.get("mp_grid") or ())
        out["decay"] = {
            "tau_ang": _round(record.get("tau_ang")),
            "tau_free_intercept_ang": _round(record.get("tau_free_intercept_ang")),
            "fit": "||H(R)||_F = ||H(0) - E_F||_F exp(-|R| / tau), the gate's fit",
        }
        out["eta2_at_other_radii_mev"] = {
            f"r = {_round(record.get(f'r_{n}_ang'))} Ang ({label})": _round(
                record.get(f"eta_trunc_{n}_mev")
            )
            for n, label in (("L3", "L_BvK/3"), ("L2", "L_BvK/2"))
            if f"r_{n}_ang" in record
        }
        r_c = float(record["r_c_ang"])
        model = _model(c)
        distances = hopping_distances_ang(model)
        out["hopping_norm_by_distance"] = hopping_norm_bins(model, distances, r_c)
        weights = beyond_radius_weights(model, distances, r_c)
        out["beyond_r_c"] = {
            "share_of_total_h2": _round(weights["share"], 6),
            "largest_element_ev": _round(weights["largest_ev"], 6),
            "largest_element_distance_ang": _round(weights["largest_distance_ang"]),
            "norm_definition": weights["definition"],
        }
        ref = c.reference_path
        ef, _ = _criterion_fermi(c, ref)
        dft = ref.channel(int(c.inputs.get("spin_index", 0)))
        truncated = model.bands(ref.kpoints_frac, r_c)
        out["truncated_error"] = {
            "model": f"hoppings longer than r_c = {r_c:.3f} Ang removed; length: "
            + str(record.get("truncation_length") or "as the criterion measures it"),
            **energy_resolved_error(
                dft, truncated, ef, _exclude(c), ref.kpoints_frac, (0.0, 1.0, 2.0)
            ),
        }
        out["functions_carrying_beyond_r_c_weight"] = self._dominant_functions(
            c.wannier, weights["per_function"]
        )
        return out

    #: Functions listed by truncation_diagnostics, at most.
    DOMINANT_FUNCTIONS = 6

    def _dominant_functions(self, wannier, per_function) -> list[dict]:
        share = np.asarray(per_function, dtype=float)
        if share.sum() <= 0:
            return []
        share = share / share.sum()
        order = np.argsort(-share, kind="stable")
        rows, covered = [], 0.0
        has_centres = getattr(wannier, "centres_ang", None) is not None
        for i in order[: self.DOMINANT_FUNCTIONS]:
            row = (
                self._function_row(wannier, int(i))
                if has_centres
                else {"wf": int(i) + 1}
            )
            row["share_of_beyond_r_c_h2"] = round(float(share[i]), 4)
            rows.append(row)
            covered += float(share[i])
            if covered >= 0.9:
                break
        return rows

    def tool_structure(self):
        parent = self.ctx.parent
        grid = None
        win = getattr(self.inputs, "win", None)
        if win is not None and win.is_file():
            raw = assignments(win.read_text()).get("mp_grid")
            if raw:
                grid = [int(g) for g in _parse_mp_grid(raw)]
        return describe_structure(
            parent.lattice_ang, parent.species, parent.positions_frac, grid
        )

    def tool_band_character(self, e_min, e_max):
        lo, hi = self._abs(e_min), self._abs(e_max)
        if hi < lo:
            lo, hi = hi, lo
        if self.ctx.code == "vasp":
            found = self._character_procar(lo, hi)
        else:
            found = self._character_projwfc(lo, hi)
            if found is None:
                found = self._character_amn(lo, hi)
        if isinstance(found, str):
            found = {"composition": found}
        found["bands"] = self._bands_in_window(lo, hi)
        return found

    #: Most parent bands band_character lists one by one.
    BAND_ROWS = 24

    def _bands_in_window(self, lo, hi):
        """Every parent band whose span meets the window, excluded ones included.

        The deck's own projections (the QE fallback above) never see an excluded band,
        so the bottom of the spectrum -- where a semicore cut is placed -- is read from
        the parent's eigenvalues here, with the character the parent supplies.
        """
        parent = self.ctx.parent
        spin = (
            self.ctx.spin_index
            if self.ctx.spin_index < parent.eigenvalues_ev.shape[0]
            else 0
        )
        bands = np.asarray(parent.eigenvalues_ev[spin])
        hits = [
            b + 1
            for b in range(bands.shape[1])
            if bands[:, b].max() >= lo and bands[:, b].min() <= hi
        ]
        if not hits:
            return []
        hits = hits[: self.BAND_ROWS]
        characters, _ = semicore.parent_band_characters(self.ctx, hits, spin)
        win = getattr(self.inputs, "win", None)
        excluded = (
            _excluded_from_text(win.read_text())
            if win is not None and win.is_file()
            else ()
        )
        rows = semicore.band_rows(parent, hits[0], hits[-1], excluded, spin, characters)
        return [r for r in rows if r["band"] in set(hits)]

    def _compose(
        self, weights_by_label: dict[str, float], states: int, source: str, lo, hi
    ):
        total = sum(weights_by_label.values())
        if states == 0 or total <= 0:
            return {
                "window_ev_relative": [self._rel(lo), self._rel(hi)],
                "states": states,
                "source": source,
                "composition": {},
            }
        comp = {
            k: round(v / total, 3)
            for k, v in sorted(weights_by_label.items(), key=lambda kv: -kv[1])
            if v / total >= 0.005
        }
        return {
            "window_ev_relative": [self._rel(lo), self._rel(hi)],
            "states": states,
            "source": source,
            "composition": comp,
        }

    def _character_procar(self, lo, hi):
        from .backends.vasp.outputs import VaspOutputError, read_procar

        procar = self.ctx.parent_source.with_name("PROCAR")
        try:
            proj = read_procar(procar)
        except VaspOutputError as exc:
            return f"no band character available: {exc}"
        species = list(self.ctx.parent.species)
        letters = [
            "d" if o.startswith(("x2-y2", "dx2")) else o[0].lower()
            for o in proj.orbitals
        ]
        mask = (proj.energies_ev >= lo) & (proj.energies_ev <= hi)
        weights = proj.weights[mask]  # state, ion, orbital
        out: dict[str, float] = {}
        for ion, symbol in enumerate(species):
            for column, letter in enumerate(letters):
                if letter not in "spdf":
                    continue
                key = f"{symbol} {letter}"
                out[key] = out.get(key, 0.0) + float(weights[:, ion, column].sum())
        return self._compose(
            out,
            int(mask.sum()),
            f"PROCAR of the parent (LORBIT = 11), {procar.name}",
            lo,
            hi,
        )

    def _character_projwfc(self, lo, hi):
        from .projectability import ProjectabilityError

        directory = self.ctx.parent_source.parent
        xml = next(
            (
                p
                for p in [
                    directory / "atomic_proj.xml",
                    *directory.glob("*.save/atomic_proj.xml"),
                ]
                if p.is_file()
            ),
            None,
        )
        stdout = next(
            (
                p
                for p in [*directory.glob("*projwfc*.out"), *directory.glob("projwfc*")]
                if p.is_file()
            ),
            None,
        )
        if xml is None or stdout is None:
            return None
        try:
            labels = _projwfc_states(stdout.read_text(errors="replace"))
            energies, weights = _atomic_proj_states(xml)
        except (ProjectabilityError, ValueError, OSError):
            return None
        if not labels or weights.shape[-1] != len(labels):
            return None
        mask = (energies >= lo) & (energies <= hi)
        out: dict[str, float] = {}
        for column, label in enumerate(labels):
            out[label] = out.get(label, 0.0) + float(weights[mask, column].sum())
        return self._compose(
            out, int(mask.sum()), f"projwfc.x ({xml.name}, {stdout.name})", lo, hi
        )

    def _character_amn(self, lo, hi):
        from .overlaps import OverlapReadError, read_projections

        seed, _ = self._last_run()
        seed = seed or self.seed
        deck = self.backend.read_construction(self.ctx.run_dir, seed)
        try:
            labels = projector_labels(deck, self.ctx.run_dir)
            proj = read_projections(self.ctx.run_dir, seed)
        except (InputError, OverlapReadError) as exc:
            return f"no band character available: no projwfc.x output and {exc}"
        if proj.energies_ev is None or proj.amn.shape[2] != len(labels):
            return "no band character available: the overlaps do not match the deck's projectors"
        w = proj.weights()
        mask = (proj.energies_ev >= lo) & (proj.energies_ev <= hi)
        out: dict[str, float] = {}
        for column, label in enumerate(labels):
            out[label] = out.get(label, 0.0) + float(w[..., column][mask].sum())
        return self._compose(
            out,
            int(mask.sum()),
            f"the projections of attempt {_attempt_of(self.run, seed)} ({seed}.amn): the "
            f"weight each state carries on the deck's own projectors, not an independent "
            f"decomposition; states the projectors miss do not show",
            lo,
            hi,
        )

    def tool_reference_pages(self, topic):
        path = REFERENCE_DIR / f"{topic}.md"
        if topic not in reference_topics() or not path.is_file():
            return f"no page {topic!r}; topics: {', '.join(reference_topics())}"
        return path.read_text()

    def tool_frontier_candidates(self):
        """The frontier's next proposals: its PDWF retry ladder, minus what ran here.

        Only the thresholds and their order are read from the stored ladder; whatever
        else the file holds (statuses, scores, per-criterion verdicts) is never read
        into the answer (user decision 2026-10-12: the frontier may tell autowan only
        what its deck would propose next from the current state).
        """
        none = "not available: the frontier proposes no further candidates for this construction"
        if self.frontier_handoff is not None:
            # a handoff run offers frontier_next_proposal in its place
            return "there is no tool named 'frontier_candidates'"
        if self.frontier_rescan_source is None:
            return none
        rows = load_rescan(
            Path(self.frontier_rescan_source), self.rescan_key or self.run.material
        )
        tried = self._thresholds_run_here()
        candidates = []
        for row in rows:
            pair = (
                _threshold(row.get("dis_proj_min")),
                _threshold(row.get("dis_proj_max")),
            )
            if None in pair or pair in tried:
                continue
            candidates.append(
                {
                    "ladder_point": row.get("point"),
                    "dis_proj_min": pair[0],
                    "dis_proj_max": pair[1],
                }
            )
        if not candidates:
            return none
        return {
            "note": (
                "untested proposals: the frontier's retry ladder for its PDWF deck, in "
                "the order its retry rule would try them next; none has been run or "
                "scored. Each changes only dis_proj_min and dis_proj_max of a PDWF deck."
            ),
            "candidates": candidates,
        }

    def tool_frontier_next_proposal(self):
        """Handoff mode: the frontier's retry rule applied to the latest run deck.

        User decision 2026-10-13 (second revision): one step at a time. The input is the
        most recent deck that actually ran in this run (attempt 0 of a handoff is the
        frontier's last ladder point); the output is that deck with only dis_proj_min
        and dis_proj_max replaced by the ladder's next pair. The next pair follows the
        latest deck's own pair when that pair is on the ladder; when it is not, it is
        the first ladder pair (in ladder order) this run -- the frontier's runs before
        the handoff included -- has not used. A pair that would rebuild a deck already
        run (the frontier's included) is passed over, and the note says so. Nothing but
        thresholds and order is read from the ladder file; nothing scored reaches the
        model.
        """
        if self.frontier_handoff is None:
            return "there is no tool named 'frontier_next_proposal'"
        rows = (
            load_rescan(
                Path(self.frontier_rescan_source), self.rescan_key or self.run.material
            )
            if self.frontier_rescan_source is not None
            else []
        )
        ladder = []
        for row in rows:
            pair = (
                _threshold(row.get("dis_proj_min")),
                _threshold(row.get("dis_proj_max")),
            )
            if None not in pair and pair not in [p for _, p in ladder]:
                ladder.append((_point_number(row.get("point")), pair))
        latest = self._latest_run_attempt()
        if latest is None:
            return {"note": "no deck has run in this run yet", "proposal": None}
        params = dict(getattr(latest, "parameters", None) or {})
        current = (
            _threshold(params.get("dis_proj_min")),
            _threshold(params.get("dis_proj_max")),
        )
        label = _attempt_label(latest)
        if None in current:
            return {
                "note": (
                    f"the latest run deck ({label}, method "
                    f"{params.get('method')}) has no projectability thresholds; the "
                    f"frontier's retry rule changes only dis_proj_min and dis_proj_max "
                    f"of a PDWF deck, so it proposes nothing from it"
                ),
                "proposal": None,
            }
        position = next((i for i, (_, p) in enumerate(ladder) if p == current), None)
        if position is not None:
            order = ladder[position + 1 :]
            rule = (
                f"the ladder pair after the latest run deck's own (ladder point "
                f"{ladder[position][0]})"
            )
        else:
            used = self._threshold_pairs_used()
            order = [(n, p) for n, p in ladder if p not in used]
            rule = (
                "the latest run deck's thresholds are off the ladder: the first ladder "
                "pair, in ladder order, this run has not used"
            )
        skipped, untried = [], []
        for point, pair in order:
            proposed = {
                **params,
                "dis_proj_min": pair[0],
                "dis_proj_max": pair[1],
            }
            earlier = self._run_with(proposed)
            if earlier is not None:
                skipped.append(f"ladder point {point} (it would rebuild {earlier})")
                continue
            untried.append((point, pair))
        for point, pair in untried[:1]:
            return {
                "note": (
                    f"what the frontier's own retry rule would run next from your latest "
                    f"run deck ({label}): {rule}. Untested -- not run or scored. Every "
                    f"other value of that deck is kept; apply it as a change of "
                    f"dis_proj_min and dis_proj_max to that deck. "
                    f"untried_ladder_from_that_deck lists, in the rule's order, every "
                    f"pair it would try from that deck (the first is the next); "
                    f"proposals only, no outcomes."
                    + (f" Passed over: {'; '.join(skipped)}." if skipped else "")
                ),
                "proposal": {
                    "from_deck_of": label,
                    "ladder_point": point,
                    "changes": {
                        "dis_proj_min": {"from": current[0], "to": pair[0]},
                        "dis_proj_max": {"from": current[1], "to": pair[1]},
                    },
                    "kept_from_that_deck": {
                        k: params.get(k)
                        for k in ("method", "num_wann", "num_bands", "exclude_bands")
                        if params.get(k) is not None
                    },
                },
                "untried_ladder_from_that_deck": [
                    {
                        "ladder_point": n,
                        "dis_proj_min": q[0],
                        "dis_proj_max": q[1],
                        **({"next": True} if i == 0 else {}),
                    }
                    for i, (n, q) in enumerate(untried)
                ],
            }
        return {
            "note": (
                f"the frontier's ladder is exhausted from your latest run deck "
                f"({label}): {rule}, and there is none left"
                + (f" (passed over: {'; '.join(skipped)})" if skipped else "")
            ),
            "proposal": None,
        }

    def _latest_run_attempt(self):
        """The most recent attempt whose deck actually ran (a handoff's attempt 0 did)."""
        from .orchestrator import _ran_wannier90

        for attempt in reversed(getattr(self.run, "attempts", None) or ()):
            if getattr(attempt, "frontier_point", None) is not None:
                return attempt
            if _ran_wannier90(attempt):
                return attempt
        return None

    def _threshold_pairs_used(self) -> set:
        """Every threshold pair a deck that ran here used, the frontier's runs included."""
        from .orchestrator import _ran_wannier90

        used = set()
        for entry in getattr(self.frontier_handoff, "history", None) or ():
            values = entry.get("deck_values") or {}
            used.add(
                (
                    _threshold(values.get("dis_proj_min")),
                    _threshold(values.get("dis_proj_max")),
                )
            )
        for attempt in getattr(self.run, "attempts", None) or ():
            if getattr(attempt, "frontier_point", None) is None and not _ran_wannier90(
                attempt
            ):
                continue
            params = getattr(attempt, "parameters", None) or {}
            used.add(
                (
                    _threshold(params.get("dis_proj_min")),
                    _threshold(params.get("dis_proj_max")),
                )
            )
        used.discard((None, None))
        return used

    def _run_with(self, parameters) -> str | None:
        """Which deck already run (or the frontier's) has this parameter set, if any."""
        frontier_run_of = getattr(self.run, "frontier_run_of", None)
        if frontier_run_of is not None:
            run = frontier_run_of(parameters)
            if run is not None:
                return f"the frontier's run {run}"
        from .orchestrator import _state_key

        key = _state_key(dict(parameters))
        for attempt in getattr(self.run, "attempts", None) or ():
            if _state_key(dict(getattr(attempt, "parameters", None) or {})) == key:
                return _attempt_label(attempt)
        return None

    def _thresholds_run_here(self) -> set:
        """(dis_proj_min, dis_proj_max) of every deck this run has built, current included."""
        seen = set()
        for attempt in getattr(self.run, "attempts", None) or ():
            params = getattr(attempt, "parameters", None) or {}
            pair = (
                _threshold(params.get("dis_proj_min")),
                _threshold(params.get("dis_proj_max")),
            )
            if None not in pair:
                seen.add(pair)
        win = getattr(self.inputs, "win", None)
        if win is not None and Path(win).is_file():
            from .inputs import read_win_text_value

            text = Path(win).read_text()
            pair = (
                _threshold(read_win_text_value(text, "dis_proj_min")),
                _threshold(read_win_text_value(text, "dis_proj_max")),
            )
            if None not in pair:
                seen.add(pair)
        return seen

    def tool_frontier_dataset(
        self, elements=None, formula=None, pseudo_family=None, limit=None
    ):
        from .frontier_dataset import DEFAULT_TABLE, MAX_ROWS, lookup

        parent = self.ctx.parent
        counts = _counts(parent.species)
        own = "".join(f"{s}{n}" for s, n in sorted(counts.items()))
        if not elements and not formula:
            elements = sorted(counts)
            formula = own
        return lookup(
            # With same-composition records off, the PARENT's composition is held back
            # for every query form, not only the default one (a query by elements or
            # by another formula reached the run's own records before 2026-10-13).
            exclude_composition=None if self.frontier_dataset_same_composition else own,
            elements=elements or (),
            formula=formula,
            pseudo_family=pseudo_family,
            n_sites=len(parent.species),
            magnetic=getattr(parent, "n_spin", 1) > 1,
            limit=limit or MAX_ROWS,
            path=self.frontier_dataset_source or DEFAULT_TABLE,
            same_composition=self.frontier_dataset_same_composition,
            spinor=bool(getattr(parent, "spinor", False)),
        )

    def tool_search_references(self, query):
        directory = Path(self.recipe_dir) if self.recipe_dir else recipes.DEFAULT_CORPUS
        if not directory.is_dir():
            found = []
        else:
            from .payload import signature_of

            signature = signature_of(self.ctx.parent, None)
            unresolved = list(getattr(self.run, "_unresolved", ()))
            found = recipes.search(
                recipes.load(directory),
                signature=signature,
                unresolved=unresolved,
                query=query or "",
                exclude_material=(
                    self.run.material if self.recipe_same_material_holdback else None
                ),
                exclude_composition=(
                    signature.get("species")
                    if self.recipe_same_material_holdback
                    else None
                ),
                engine=self.ctx.code,
            )
        self.run.searches.append(
            {
                "attempt": len(self.run.attempts) - 1,
                "query": query,
                "retrieved": [r.recipe_id for r, _ in found],
                "scores": [score for _, score in found],
            }
        )
        return recipes.render(found)


NO_CONSTRUCTION = (
    "no construction has run far enough to be scored against the reference path"
)
#: The windows band_error_by_energy reports: eta at mu = E_F + nu, nu in eV.
ENERGY_WINDOWS_EV = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
#: Rows per window: the worst k-points and the worst bands.
WORST_ROWS = 3


def _round(value, digits: int = 4):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return round(value, digits) if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# where a band error sits
# ---------------------------------------------------------------------------


def energy_resolved_error(dft, wan, fermi_ev, exclude, kpoints_frac, nus) -> dict:
    """eta at mu = E_F + nu for each nu, and where the error sits in each window.

    Every number is :func:`~.band_distance_eta.band_distance_eta` itself, the gate's
    definition: on the whole path for the window's eta, on one k-point for a k-point's
    share, and on one index-matched band pair (after the deck's exclusions) for a band's.
    ``eta_max`` is the largest weighted |e_DFT - e_Wan|; the rows are ranked by it, so a
    large difference far above the window, which the occupation weight switches off,
    does not rank.
    """
    from .band_distance_eta import band_distance_eta

    dft = np.asarray(dft, dtype=float)
    wan = np.asarray(wan, dtype=float)
    keep = np.ones(dft.shape[1], dtype=bool)
    if exclude:
        keep[np.asarray(exclude) - 1] = False
    dft_index = np.flatnonzero(keep) + 1  # DFT numbering of the kept bands
    kept = dft[:, keep]
    n = min(kept.shape[1], wan.shape[1])
    windows = []
    for nu in nus:
        whole = band_distance_eta(dft, wan, fermi_ev, nu_ev=nu, exclude_bands=exclude)
        per_k = [
            band_distance_eta(
                dft[k : k + 1],
                wan[k : k + 1],
                fermi_ev,
                nu_ev=nu,
                exclude_bands=exclude,
            )
            for k in range(dft.shape[0])
        ]
        per_band = [
            band_distance_eta(kept[:, j : j + 1], wan[:, j : j + 1], fermi_ev, nu_ev=nu)
            for j in range(n)
        ]
        k_order = [
            k
            for k in np.argsort([-_finite_or_zero(r.eta_max_mev) for r in per_k])
            if _finite_or_zero(per_k[k].eta_max_mev) > 0
        ][:WORST_ROWS]
        b_order = [
            j
            for j in np.argsort([-_finite_or_zero(r.eta_max_mev) for r in per_band])
            if _finite_or_zero(per_band[j].eta_max_mev) > 0
        ][:WORST_ROWS]
        worst_k = []
        for k in k_order:
            at_k = [
                band_distance_eta(
                    kept[k : k + 1, j : j + 1],
                    wan[k : k + 1, j : j + 1],
                    fermi_ev,
                    nu_ev=nu,
                ).eta_max_mev
                for j in range(n)
            ]
            j = int(np.argmax([_finite_or_zero(v) for v in at_k]))
            worst_k.append(
                {
                    "k_index": int(k) + 1,
                    "k_frac": [round(float(x), 4) for x in kpoints_frac[k]],
                    "eta_mev": _round(per_k[k].eta_mev, 2),
                    "eta_max_mev": _round(per_k[k].eta_max_mev, 2),
                    "worst_band_dft": int(dft_index[j]),
                    "e_dft_minus_ef_ev": _round(kept[k, j] - fermi_ev, 3),
                    "e_wan_minus_dft_mev": _round((wan[k, j] - kept[k, j]) * 1e3, 1),
                }
            )
        worst_bands = []
        for j in b_order:
            diff = (wan[:, j] - kept[:, j]) * 1e3
            worst_bands.append(
                {
                    "band_dft": int(dft_index[j]),
                    "band_wannier": int(j) + 1,
                    "dft_span_minus_ef_ev": [
                        _round(kept[:, j].min() - fermi_ev, 3),
                        _round(kept[:, j].max() - fermi_ev, 3),
                    ],
                    "eta_mev": _round(per_band[j].eta_mev, 2),
                    "eta_max_mev": _round(per_band[j].eta_max_mev, 2),
                    "largest_abs_difference_mev": _round(np.abs(diff).max(), 1),
                    "at_k_index": int(np.argmax(np.abs(diff))) + 1,
                }
            )
        windows.append(
            {
                "nu_ev": float(nu),
                "eta_mev": _round(whole.eta_mev, 3),
                "eta_max_mev": _round(whole.eta_max_mev, 3),
                "worst_kpoints": worst_k,
                "worst_bands": worst_bands,
            }
        )
    return {
        "n_kpoints": int(dft.shape[0]),
        "n_bands_compared": int(n),
        "excluded_dft_bands": list(exclude),
        "windows": windows,
    }


def _finite_or_zero(value) -> float:
    return float(value) if value is not None and np.isfinite(value) else 0.0


# ---------------------------------------------------------------------------
# hoppings by real distance
# ---------------------------------------------------------------------------


def hopping_distances_ang(model) -> np.ndarray:
    """Every element's hopping length, as the sparse criterion measures it.

    ``RealSpaceModel.element_lengths_ang`` -- |R + tau_j - tau_i|, tau the Wannier
    centres -- from :func:`~.criteria.real_space_model`; where the model carries no
    centres, the criterion falls back to |R| for every element, and so does this.
    """
    lengths = getattr(model, "element_lengths_ang", None)
    if lengths is not None:
        return np.asarray(lengths, dtype=float)
    return np.broadcast_to(
        np.asarray(model.lengths_ang, dtype=float)[:, None, None],
        np.asarray(model.hoppings).shape,
    )


def _beyond(model, radius: float) -> np.ndarray:
    """(vector, i, j): the elements the criterion removes at ``radius``."""
    return ~np.asarray(model.kept_elements(radius), dtype=bool)


def _hopping_magnitudes(model, distances):
    """|H_ij(R)|^2 and the matching distances, the on-site energies left out.

    The diagonal of H(0) is each function's on-site energy, measured from the code's
    arbitrary zero; it is not a hopping, and including it would make every share below
    depend on where the code puts zero energy.
    """
    h2 = np.abs(np.asarray(model.hoppings)) ** 2
    d = np.asarray(distances, dtype=float)
    onsite = np.zeros_like(h2, dtype=bool)
    origin = np.flatnonzero(np.all(np.asarray(model.vectors) == 0, axis=1))
    for v in origin:
        np.fill_diagonal(onsite[v], True)
    return h2, d, ~onsite


NORM_DEFINITION = (
    "sum of |H_ij(R)|^2 over every element of the model as Wannier90 interpolates it "
    "(Wigner-Seitz degeneracies and minimal images applied), the on-site energies "
    "(diagonal of H(0)) left out; d = |R + tau_j - tau_i|"
)


def hopping_norm_bins(model, distances, r_c: float, per_r_c: int = 4) -> dict:
    """The hopping norm in shells of real hopping distance, r_c an edge.

    Shells are r_c / ``per_r_c`` wide. Each carries its Frobenius norm (the root of its
    summed |H|^2), its largest element, the number of non-zero elements and its share
    of the total |H|^2; the shells beyond r_c are marked.
    """
    h2, d, use = _hopping_magnitudes(model, distances)
    width = r_c / per_r_c
    total = float(h2[use].sum())
    # Shell b holds b*width < d <= (b+1)*width (d = 0 in the first), so d = r_c falls
    # inside r_c, as the criterion keeps it.
    shell = np.maximum(np.ceil(d / width - 1e-9).astype(int) - 1, 0)
    rows = []
    for b in np.unique(shell[use]):
        sel = use & (shell == b)
        mass = float(h2[sel].sum())
        nonzero = int(np.count_nonzero(h2[sel] > 0))
        if nonzero == 0:
            continue
        lo, hi = b * width, (b + 1) * width
        rows.append(
            {
                "d_ang": [round(float(lo), 3), round(float(hi), 3)],
                "norm_ev": round(float(np.sqrt(mass)), 6),
                "largest_ev": round(float(np.sqrt(h2[sel].max())), 6),
                "elements": nonzero,
                "share": round(mass / total, 6) if total > 0 else None,
                "beyond_r_c": bool(b >= per_r_c),
            }
        )
    return {
        "r_c_ang": round(float(r_c), 4),
        "shell_width_ang": round(float(width), 4),
        "definition": NORM_DEFINITION,
        "shells": rows,
    }


def beyond_radius_weights(model, distances, radius: float) -> dict:
    """The |H|^2 beyond ``radius``: its share, its largest element, and per function.

    A function's weight is half the |H|^2 of every element beyond the radius that it
    is an end of (its row and its column of H(R)), so the weights sum to the total.
    """
    h2, d, use = _hopping_magnitudes(model, distances)
    beyond = use & _beyond(model, radius)
    total = float(h2[use].sum())
    mass = np.where(beyond, h2, 0.0)
    per_function = 0.5 * (mass.sum(axis=(0, 2)) + mass.sum(axis=(0, 1)))
    if beyond.any():
        flat = int(np.argmax(np.where(beyond, h2, -1.0)))
        largest = float(np.sqrt(h2.reshape(-1)[flat]))
        largest_d = float(d.reshape(-1)[flat])
    else:
        largest, largest_d = 0.0, None
    return {
        "share": float(mass.sum()) / total if total > 0 else 0.0,
        "largest_ev": largest,
        "largest_distance_ang": largest_d,
        "per_function": per_function,
        "definition": NORM_DEFINITION,
    }


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

#: Atoms closer than this to a cell face, in fractional units, count as inside.
_FRAC_TOL = 1e-6


def _wrap(frac) -> np.ndarray:
    frac = np.asarray(frac, dtype=float)
    wrapped = frac - np.floor(frac)
    wrapped[wrapped >= 1.0 - _FRAC_TOL] = 0.0
    wrapped[np.abs(wrapped) < _FRAC_TOL] = 0.0
    return wrapped


def _image_shifts(lattice, radius: float) -> np.ndarray:
    """Every integer translation that can bring a wrapped point within ``radius``."""
    recip = np.linalg.inv(np.asarray(lattice, dtype=float)).T  # rows b_i, no 2 pi
    reach = [int(np.ceil(radius * np.linalg.norm(b))) + 1 for b in recip]
    axes = [np.arange(-n, n + 1) for n in reach]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def _covering_radius_bound(lattice) -> float:
    """Half the summed lattice-vector lengths: no point lies further from the lattice."""
    return 0.5 * float(np.linalg.norm(np.asarray(lattice, dtype=float), axis=1).sum())


def nearest_site(lattice, positions_frac, point_ang) -> tuple[int, float]:
    """The atom (0-based) nearest a Cartesian point, over all periodic images."""
    lattice = np.asarray(lattice, dtype=float)
    sites = _wrap(positions_frac)
    point = _wrap(np.linalg.solve(lattice.T, np.asarray(point_ang, dtype=float)))
    shifts = _image_shifts(lattice, _covering_radius_bound(lattice))
    best, best_d = 0, np.inf
    for atom, site in enumerate(sites):
        d = np.linalg.norm((site + shifts - point) @ lattice, axis=1).min()
        if d < best_d:
            best, best_d = atom, float(d)
    return best, best_d


def neighbour_shells(lattice, positions_frac, species, shells: int = 2, tol=0.05):
    """Per site, the first ``shells`` neighbour shells: distance and species counts."""
    lattice = np.asarray(lattice, dtype=float)
    sites = _wrap(positions_frac)
    lengths = np.linalg.norm(lattice, axis=1)
    shifts = _image_shifts(lattice, max(lengths.max(), _covering_radius_bound(lattice)))
    out = []
    for a, site in enumerate(sites):
        rows = []
        for b, other in enumerate(sites):
            d = np.linalg.norm((other + shifts - site) @ lattice, axis=1)
            for value in d:
                if value > 1e-6:
                    rows.append((float(value), b))
        rows.sort()
        found = []
        for value, b in rows:
            if found and value <= found[-1]["distance_ang"] + tol:
                label = species[b]
                found[-1]["neighbours"][label] = (
                    found[-1]["neighbours"].get(label, 0) + 1
                )
                continue
            if len(found) == shells:
                break
            found.append(
                {"distance_ang": round(value, 4), "neighbours": {species[b]: 1}}
            )
        out.append(
            {
                "site": f"{species[a]}#{a + 1}",
                "nearest_ang": found[0]["distance_ang"] if found else None,
                "shells": found,
            }
        )
    return out


def describe_structure(lattice, species, positions_frac, mp_grid=None) -> dict:
    """The structure facts a repair needs, anonymous: species only, never a name."""
    from types import SimpleNamespace

    from .criteria import VACUUM_GAP_NN_MULTIPLE, periodic_axes, vacuum_gaps_ang
    from .gate import Construction

    lattice = np.asarray(lattice, dtype=float)
    stored = np.asarray(positions_frac, dtype=float).reshape(-1, 3)
    wrapped = _wrap(stored)
    outside = np.any((stored < -_FRAC_TOL) | (stored >= 1.0 - _FRAC_TOL), axis=1)
    lengths = np.linalg.norm(lattice, axis=1)

    def angle(i, j):
        cos = lattice[i] @ lattice[j] / (lengths[i] * lengths[j])
        return round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 3)

    atoms = []
    for i, s in enumerate(species):
        row = {
            "site": f"{s}#{i + 1}",
            "frac_stored": [round(float(x), 6) for x in stored[i]],
            "frac_wrapped": [round(float(x), 6) for x in wrapped[i]],
            "cart_wrapped_ang": [round(float(x), 4) for x in wrapped[i] @ lattice],
            "stored_outside_cell": bool(outside[i]),
        }
        if outside[i]:
            row["lattice_translation"] = [
                int(x) for x in np.rint(stored[i] - wrapped[i]).astype(int)
            ]
        atoms.append(row)
    shells = neighbour_shells(lattice, stored, list(species))
    nn = min((s["nearest_ang"] for s in shells if s["nearest_ang"]), default=None)
    gaps = vacuum_gaps_ang(lattice, stored)
    counts: dict[str, int] = {}
    for s in species:
        counts[s] = counts.get(s, 0) + 1
    out: dict[str, Any] = {
        "lattice_ang": [[round(float(x), 6) for x in row] for row in lattice],
        "lengths_ang": [round(float(x), 4) for x in lengths],
        "angles_deg": {"alpha": angle(1, 2), "beta": angle(0, 2), "gamma": angle(0, 1)},
        "volume_ang3": round(float(abs(np.linalg.det(lattice))), 4),
        "species_counts": counts,
        "atoms_stored_outside_cell": int(outside.sum()),
        "atoms": atoms,
        "nearest_neighbour_ang": nn,
        "neighbours": shells,
    }
    periodicity: dict[str, Any] = {
        "rule": (
            "a lattice direction is non-periodic only when the k-mesh does not sample it "
            f"(mp_grid = 1) and the structure leaves a vacuum gap along it of at least "
            f"{VACUUM_GAP_NN_MULTIPLE:g} nearest-neighbour distances; the sparse "
            "criterion then takes L_BvK over the periodic directions only"
        ),
        "mp_grid": mp_grid,
        "axes": [],
    }
    axes = None
    if mp_grid is not None and len(mp_grid) == 3:
        parent = SimpleNamespace(lattice_ang=lattice, positions_frac=stored)
        probe = Construction(
            "structure",
            parent=parent,
            wannier=None,
            material_class="metal",
            inputs={"mp_grid": list(mp_grid)},
        )
        axes, _ = periodic_axes(probe)
        periodicity["periodic_dims"] = len(axes)
    for i in range(3):
        periodicity["axes"].append(
            {
                "axis": i,
                "length_ang": round(float(lengths[i]), 4),
                "widest_empty_layer_ang": round(gaps[i], 4),
                "empty_layer_over_nn": round(gaps[i] / nn, 3) if nn else None,
                "mp_grid": None if mp_grid is None else int(mp_grid[i]),
                "periodic": None if axes is None else i in axes,
            }
        )
    out["periodicity"] = periodicity
    return out


def _attempt_of(run, seed) -> int | None:
    for attempt in run.attempts:
        if getattr(attempt, "seed", None) == seed:
            return attempt.index
    return None


def _counts(species) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in species:
        out[s] = out.get(s, 0) + 1
    return out


def projector_labels(inputs, run_dir: Path) -> list[str]:
    """A ``species l`` label for every projector column of a deck's ``.amn``."""
    win_text = inputs.win.read_text()
    pw_text = inputs.pw2wan.read_text() if inputs.pw2wan is not None else None
    if pw_text is not None and assignments(pw_text).get(
        "atom_proj", ""
    ).lower().startswith((".t", "t")):
        layout = pao_layout(run_dir, win_text, pw_text)
        excluded = pao_excluded(pw_text)
        return [f"{s.species} {'spdf'[s.l]}" for s in layout if s.index not in excluded]
    from .deckcheck import _projection_block

    block = _projection_block(win_text)
    if not block:
        raise InputError("the deck's projectors are not atomic orbitals (SCDM)")
    atoms = deck_atom_species(win_text)
    labels = []
    sizes = {
        "s": (1, "s"),
        "p": (3, "p"),
        "d": (5, "d"),
        "f": (7, "f"),
        "sp": (2, "sp"),
        "sp2": (3, "sp2"),
        "sp3": (4, "sp3"),
    }
    for line in block:
        parts = line.replace(" ", "").split(":")
        site, orbitals = parts[0], parts[1].split(";")
        n_atoms = atoms.count(site)
        for _ in range(n_atoms):
            for orbital in orbitals:
                size, name = sizes.get(orbital.lower(), (1, orbital))
                labels += [f"{site} {name}"] * size
    return labels


def _projwfc_states(text: str) -> list[str]:
    """``species l`` per state from projwfc.x's ``state #`` table."""
    import re

    labels = []
    for match in re.finditer(
        r"state #\s*\d+:\s*atom\s+\d+\s*\(\s*(\w+)\s*\).*?\(l=\s*(\d)", text
    ):
        labels.append(f"{match.group(1)} {'spdf'[int(match.group(2))]}")
    return labels


def _atomic_proj_states(xml: Path):
    """Energies (eV) and per-orbital weights from ``atomic_proj.xml``, flattened."""
    import xml.etree.ElementTree as ET

    from .projectability import RY_TO_EV

    root = ET.parse(xml).getroot()
    energies, weights = [], []
    for eig_el, proj_el in zip(root.findall(".//E"), root.findall(".//PROJS")):
        eig = np.array((eig_el.text or "").split(), dtype=float) * RY_TO_EV
        columns = []
        for wfc in proj_el.findall("ATOMIC_WFC"):
            raw = np.array((wfc.text or "").split(), dtype=float)
            columns.append(raw[0::2] ** 2 + raw[1::2] ** 2)
        energies.append(eig)
        weights.append(np.array(columns).T)
    return np.concatenate(energies), np.concatenate(weights)


def _attempt_label(attempt) -> str:
    point = getattr(attempt, "frontier_point", None)
    if point is not None:
        return f"attempt {attempt.index}, the frontier's ladder point {point}"
    return f"attempt {attempt.index}"


def _point_number(value) -> int | None:
    """A ladder point as stored (``"10"``, ``10``, ``10.0``), as an integer."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _threshold(value) -> float | None:
    """A projectability threshold as a float rounded to 1e-6 (Fortran ``d`` exponents read)."""
    if value is None or value == "":
        return None
    try:
        return round(float(str(value).strip().lower().replace("d", "e")), 6)
    except ValueError:
        return None


def load_rescan(source: Path, key: str) -> list[dict]:
    """Rows of a stored rescan grid for one construction, in the package's order.

    A CSV with a ``material`` column, or a JSON object keyed by material whose values
    are row lists. Columns ending in ``_pass`` are read as per-criterion verdicts.
    """
    if not source.is_file():
        return []
    if source.suffix == ".json":
        data = json.loads(source.read_text())
        rows = data.get(key) or []
        if isinstance(rows, dict):
            rows = rows.get("grid") or []
        return [dict(r) for r in rows]
    with source.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("material") == key]
    if rows and "point" in rows[0]:
        rows.sort(key=lambda r: float(r["point"]))
    return rows
