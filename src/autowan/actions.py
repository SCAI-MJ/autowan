"""The action vocabulary a model may use, and what the repair stage permits.

The ladder is AUTOMATIC -> REPAIR -> TERMINAL. The automatic method (the published
frontier for the engine) runs with no model in the loop; if the gate refuses it, one
repair stage follows with every tool available from its first attempt and a single cap
on attempts. Read-only tools (deck validation, method defaults, band character,
references, ...) are called through the provider's function calling and cost no
attempt; what the model may *do* is one of the actions below, and the schema handed to
the provider lists only those.

What the vocabulary deliberately excludes is as load-bearing as what it contains.
There is no action that edits a gate threshold, none that touches the parent
calculation, and none that runs a command. A model cannot make a construction pass
by making the criterion easier, because no reachable knob does that.

The construction method is part of the vocabulary. A deck is built by one family --
SMV, PDWF or SCDM (:mod:`.families`) -- and ``modify_parameters`` may move only that
family's knobs and the shared ones. Changing family is ``switch_method``: an explicit
action that costs an attempt and is recorded with where it came from, where it went
and why. A knob from another family is refused with that reason, so no path changes
the method silently.

The line between reachable and unreachable is not "does this affect convergence" but
"does this move the bar". ``conv_window`` -- how many consecutive iterations must meet
the tolerance -- is reachable, because Wannier90's default of -1 runs no convergence
test at all and any permitted value turns one on; its floor of 2 keeps it from becoming
a single lucky step. ``conv_tol``, the tolerance itself, is not reachable, and a
minimisation that will not converge must be stabilised rather than forgiven.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from .families import (
    UNAVAILABLE_REASONS,
    Method,
    as_method,
    available,
    families_of,
    knob_unavailable,
    knobs_for,
)

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """Where a run is in the ladder."""

    AUTOMATIC = "automatic"
    """The engine's frontier method runs untouched. No model is consulted at all."""

    REPAIR = "repair"
    """The model diagnoses and acts, with every read-only tool available."""

    TERMINAL = "terminal"
    """The run has ended, by acceptance, exhaustion, or a stop."""


ActionKind = Literal["modify_parameters", "switch_method", "escalate"]

#: Which actions each stage permits.
STAGE_ACTIONS: dict[Stage, tuple[ActionKind, ...]] = {
    Stage.AUTOMATIC: (),
    Stage.REPAIR: ("modify_parameters", "switch_method", "escalate"),
    Stage.TERMINAL: (),
}


# ---------------------------------------------------------------------------
# the editable parameter vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """One editable knob, with the range that keeps it physical."""

    name: str
    stage_file: Literal["win", "pw2wan"]
    kind: Literal["float", "int", "projectors", "pao_subset", "semicore"]
    unit: str
    what_it_does: str
    minimum: float | None = None
    maximum: float | None = None
    rebuilds_overlaps: bool = False
    """Whether changing this forces the overlap step to run again.

    Usually implied by ``stage_file``: a knob in the overlap input plainly changes what
    that step produces. ``num_bands`` breaks the implication -- it lives in the .win,
    but Wannier90 writes it into the .nnkp and pw2wannier90 sizes the .amn/.mmn/.eig
    from that, so a construction that changes it and reuses the old files is stopped
    with "mismatch in <seed>.eig".
    """
    relative_to_fermi: bool = False
    """Whether this knob is stated as an offset from the Fermi level.

    Where true, the offset *is* the interface: the model proposes it, the record keeps
    it, and only the writer converts to the absolute number the input file wants. Both
    conventions in play at once is not a nuance, it is an unforced error — shown the
    absolute value of a window and told it reads naturally relative to the Fermi level,
    a model returned the offset, and a window meant for two electronvolts above the
    Fermi level landed ten below it.
    """


EDITABLE_PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "num_bands",
        "win",
        "int",
        "bands",
        "how much of the parent spectrum the disentanglement may choose from; it "
        "cannot exceed what the parent calculation produced, and it must leave room "
        "for every Wannier function asked for",
        minimum=1,
        rebuilds_overlaps=True,
    ),
    # The bottom (semicore) exclusion, named by shell. Wannier90 excludes by band index,
    # and the shells decide which orbitals leave or join the projector set; a shell list
    # fixes both together, where a bare band count would leave it to an energy-order rule
    # that fails exactly where two shells overlap. Every family's own consequence --
    # PDWF's and QE SMV's pseudo-atomic orbitals, analytic SMV's semicore projectors,
    # SCDM's count -- is written by the deck writer, never left to a second knob.
    Parameter(
        "semicore_exclusion",
        "win",
        "semicore",
        "shell labels",
        "which semicore shells of each species are kept out of the construction as "
        "the lowest parent bands (exclude_bands 1..N); an empty list brings every "
        "semicore shell of that species in, and a species not named keeps its current "
        "exclusion. N is the bands those shells hold (2l+1 per atom, one per electron "
        "for spinor bands); num_bands, num_wann and the orbital set (pseudo-atomic "
        "orbitals, semicore trial functions, or SCDM's count) follow, and the top "
        "exclusion is kept. validate_deck lists the shells and whether each cut is "
        "isolated",
        rebuilds_overlaps=True,
    ),
    Parameter(
        "dis_win_max",
        "win",
        "float",
        "eV",
        "SMV: upper edge of the outer window; states above it cannot enter the subspace",
        relative_to_fermi=True,
    ),
    Parameter(
        "dis_win_min",
        "win",
        "float",
        "eV",
        "SMV: lower edge of the outer window",
        relative_to_fermi=True,
    ),
    Parameter(
        "dis_froz_max",
        "win",
        "float",
        "eV",
        "SMV and PDWF: upper edge of the frozen energy window; states below it are "
        "reproduced exactly (under PDWF, together with those frozen by projectability)",
        relative_to_fermi=True,
    ),
    Parameter(
        "dis_froz_min",
        "win",
        "float",
        "eV",
        "SMV: lower edge of the frozen window",
        relative_to_fermi=True,
    ),
    Parameter(
        "dis_proj_min",
        "win",
        "float",
        "dimensionless",
        "PDWF: projectability below which a state is excluded outright",
        minimum=0.0,
        maximum=1.0,
    ),
    Parameter(
        "dis_proj_max",
        "win",
        "float",
        "dimensionless",
        "PDWF: projectability at or above which a state is frozen into the subspace",
        minimum=0.0,
        maximum=1.0,
    ),
    Parameter(
        "dis_num_iter",
        "win",
        "int",
        "iterations",
        "iteration budget for subspace selection",
        minimum=1,
        maximum=100000,
    ),
    Parameter(
        "num_iter",
        "win",
        "int",
        "iterations",
        "iteration budget for spread minimisation",
        minimum=1,
        maximum=100000,
    ),
    Parameter(
        "dis_mix_ratio",
        "win",
        "float",
        "dimensionless",
        "mixing of successive subspace iterations; below one damps an oscillation",
        minimum=0.0,
        maximum=1.0,
    ),
    Parameter(
        "conv_window",
        "win",
        "int",
        "iterations",
        "how many consecutive iterations must meet the tolerance before the "
        "minimisation is declared converged",
        minimum=2,
        maximum=100,
    ),
    Parameter(
        "trial_step",
        "win",
        "float",
        "dimensionless",
        "line-search step for the spread minimisation; a smaller step damps an "
        "oscillating descent",
        minimum=1e-3,
        maximum=10.0,
    ),
    # Naming the projectors explicitly is an SMV move. It replaces whatever produced the
    # overlaps -- the pseudopotential's own atomic orbitals read by the overlap step --
    # with analytic orbitals, so the projections must be recomputed from the ones now
    # named. It never touches a projectability window: a deck that has one is a PDWF
    # deck, whose orbitals are chosen with ``pao_subset`` instead.
    Parameter(
        "projectors",
        "pw2wan",
        "projectors",
        "orbital labels",
        "SMV: which analytic orbitals seed the Wannier functions, per species or at a "
        "given position (a bond centre, an interstice: {site: 'f=x,y,z' fractional or "
        "'c=x,y,z' Cartesian in Angstrom, orbitals}); naming them replaces the deck's "
        "projection source and forces the overlaps to be recomputed",
    ),
    # PDWF selects among the pseudo-atomic orbitals the external projector list
    # supplies (atom_proj_exclude in the overlap input). Projectability is measured
    # against exactly those orbitals, which is why the subset, not a hydrogenic set,
    # is the PDWF way to change what the functions can become.
    Parameter(
        "pao_subset",
        "pw2wan",
        "pao_subset",
        "orbital labels",
        "PDWF: which of the pseudopotential's atomic orbitals to keep, per species, "
        "named by their labels (e.g. 3d, 4s); a species not named keeps its current "
        "set. num_wann follows the count kept",
    ),
    Parameter(
        "scdm_mu",
        "pw2wan",
        "float",
        "eV",
        "SCDM: centre of the occupation weight f(e) = erfc((e - mu)/sigma)/2",
        relative_to_fermi=True,
    ),
    Parameter(
        "scdm_sigma",
        "pw2wan",
        "float",
        "eV",
        "SCDM: width of the occupation weight",
        minimum=1e-3,
        maximum=100.0,
    ),
)

BY_NAME: dict[str, Parameter] = {p.name: p for p in EDITABLE_PARAMETERS}

#: Orbital labels a projector entry may name: whole shells, hybrids, and the single
#: real components of a p or d shell in Wannier90's own spelling (one function each).
#: Anything outside this set is either a typo or a shell no pseudopotential in this
#: workflow carries.
ORBITAL_LABELS = (
    "s",
    "p",
    "d",
    "f",
    "sp",
    "sp2",
    "sp3",
    "px",
    "py",
    "pz",
    "dz2",
    "dxz",
    "dyz",
    "dx2-y2",
    "dxy",
)

#: A positioned projector's site: Wannier90's ``f=x,y,z`` (fractional) or ``c=x,y,z``
#: (Cartesian, Angstrom), three numbers.
_NUM = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
SITE_PATTERN = rf"^[cCfF]={_NUM},{_NUM},{_NUM}$"

#: What a stop & report names as the limit no knob of the construction can move
#: (escalate's ``report.binding_limit``). A well-founded report is a result: it is
#: checked by redoing the parent calculation as requested, and counts as wrong when the
#: requested change does not lead to certification.
BINDING_LIMITS = (
    "parent_band_pool",
    "parent_kmesh",
    "parent_numerics",
    "parent_pseudo_or_semicore",
    "parent_spin_or_magnetism",
    "structure",
    "criterion_unreachable",
    "other",
)

#: The structured parent-calculation changes a report may request (Quantum ESPRESSO
#: spelling; ``other`` is free text for anything else).
REQUESTED_CHANGE_SCHEMA: dict[str, Any] = {
    "nbnd": {"type": "integer", "minimum": 1},
    "mp_grid": {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 3,
        "maxItems": 3,
    },
    "ecutwfc": {"type": "number", "minimum": 0},
    "ecutrho": {"type": "number", "minimum": 0},
    "degauss": {"type": "number", "minimum": 0},
    "smearing": {"type": "string", "maxLength": 40},
    "nspin": {"type": "integer", "enum": [1, 2, 4]},
    "starting_magnetization": {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["species", "value"],
            "properties": {
                "species": {"type": "string"},
                "value": {"type": "number"},
            },
        },
    },
    "pseudo": {"type": "string", "maxLength": 300},
    "other": {"type": "string", "maxLength": 600},
}


def report_schema() -> dict[str, Any]:
    """escalate's ``report``: the stop & report a model returns when the evidence says
    no knob of the construction can reach the target on this parent."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["binding_limit", "evidence", "expected_effect"],
        "description": (
            "escalate only: a stop & report. Name what binds (binding_limit), what to "
            "change in the parent calculation (requested_changes), the numbers from "
            "the tools that show it (evidence) and what you expect once it is changed "
            "(expected_effect). It is checked by redoing the parent as requested; a "
            "report whose change does not lead to certification counts as wrong"
        ),
        "properties": {
            "binding_limit": {"type": "string", "enum": list(BINDING_LIMITS)},
            "requested_changes": {
                "type": "object",
                "additionalProperties": False,
                "properties": dict(REQUESTED_CHANGE_SCHEMA),
            },
            "evidence": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "maxLength": 400},
                "description": "the check ids and the numbers from tool outputs that "
                "support the claim (e.g. eta_2 at several truncation radii)",
            },
            "expected_effect": {"type": "string", "maxLength": 600},
        },
    }


class ActionRejected(ValueError):
    """A model response fell outside the vocabulary, the stage, or a declared range."""


# ---------------------------------------------------------------------------
# validated actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterChange:
    """One knob moved, with the value it held before."""

    parameter: str
    old: Any
    new: Any
    reason: str


@dataclass(frozen=True)
class Action:
    """A validated model action."""

    kind: ActionKind
    diagnosis: str
    evidence: tuple[str, ...]
    """Check identifiers the diagnosis rests on, each one the gate actually evaluated.

    A passing check is legitimate evidence -- that the subspace selection converged is
    what rules the subspace out as the cause -- so this is not restricted to failures.
    """

    expected_effect: str
    confidence: float
    targets: tuple[str, ...] = ()
    """Which unresolved checks this action is meant to clear.

    Separate from :attr:`evidence` because they answer different questions, and
    collapsing them lets a repair claim to address checks that were never in trouble.
    Every entry here must be a check the gate failed or could not evaluate.
    """
    changes: tuple[ParameterChange, ...] = ()
    observations: tuple[str, ...] = ()
    """Free-text readings of the numbers, which :attr:`evidence` has no room for.

    Evidence is confined to check identifiers so a diagnosis cannot rest on a
    measurement nobody took; a relation between measurements goes here. Nothing
    validates these strings. They exist to be read in the log."""
    method: str | None = None
    """``switch_method`` only: the family the next deck is built by."""
    method_from: str | None = None
    """``switch_method`` only: the family the run is leaving."""
    reason: str | None = None
    """``switch_method`` only: why the family is changing, in the model's words."""
    num_bands_needed: int | None = None
    """``escalate`` only: the ``num_bands`` (bands kept after exclusions) the repair
    needs, when the escalation names the parent's band pool as the binding limit. The
    loop confirms it with ``validate_deck`` and, if confirmed, returns a DFT
    prescription for that many bands rather than a not-certified verdict."""
    report: dict[str, Any] | None = None
    """``escalate`` only: the stop & report (:func:`report_schema`), validated --
    ``binding_limit``, ``requested_changes``, ``evidence``, ``expected_effect``. A bare
    ``num_bands_needed`` is the same report with ``binding_limit = parent_band_pool``
    and ``requested_changes.nbnd`` (``synthesized: true``)."""

    def changes_by_file(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {"win": {}, "pw2wan": {}}
        for c in self.changes:
            out[BY_NAME[c.parameter].stage_file][c.parameter] = c.new
        return out

    def parameter_state(self) -> tuple[tuple[str, str], ...]:
        """A hashable summary of what this action sets, for loop detection."""
        state = sorted((c.parameter, repr(c.new)) for c in self.changes)
        if self.method is not None:
            state.append(("method", self.method))
        return tuple(state)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def _parameter_value_schema(
    p: Parameter, allowed_species: Sequence[str] = ()
) -> dict[str, Any]:
    species = (
        {"type": "string", "enum": sorted(allowed_species)}
        if allowed_species
        else {"type": "string"}
    )
    if p.kind == "projectors":
        orbitals = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": list(ORBITAL_LABELS)},
        }
        return {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["species", "orbitals"],
                        "properties": {"species": species, "orbitals": orbitals},
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["site", "orbitals"],
                        "properties": {
                            "site": {
                                "type": "string",
                                "pattern": SITE_PATTERN,
                                "description": (
                                    "one position, not an atom: 'f=x,y,z' in "
                                    "fractional coordinates of the cell or 'c=x,y,z' "
                                    "Cartesian in Angstrom, e.g. a bond midpoint"
                                ),
                            },
                            "orbitals": orbitals,
                        },
                    },
                ]
            },
        }
    if p.kind == "pao_subset":
        return {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["species", "orbitals"],
                "properties": {
                    "species": species,
                    "orbitals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "pseudo-atomic orbital labels, e.g. 3d, 4s; "
                        "an empty list drops every orbital of this species",
                    },
                },
            },
        }
    if p.kind == "semicore":
        return {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["species", "shells"],
                "properties": {
                    "species": species,
                    "shells": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "the semicore shells of this species to exclude, by the "
                            "labels validate_deck lists (e.g. 4s, or a pseudo-atomic "
                            "orbital label such as s1); [] includes them all"
                        ),
                    },
                },
            },
        }
    entry: dict[str, Any] = {"type": "integer" if p.kind == "int" else "number"}
    if p.minimum is not None:
        entry["minimum"] = p.minimum
    if p.maximum is not None:
        entry["maximum"] = p.maximum
    return entry


def action_schema(
    stage: Stage,
    evaluated_check_ids: Sequence[str],
    unresolved_check_ids: Sequence[str] = (),
    allowed_species: Sequence[str] = (),
    *,
    code: str = "qe",
    method: str | Method | None = None,
) -> dict[str, Any]:
    """The JSON schema the final answer must conform to, for this stage.

    Four enums do the containment. ``action`` lists only what this stage permits.
    ``evidence`` is drawn from the checks the gate actually evaluated, so a diagnosis
    cannot cite a measurement that was never taken. ``targets`` is drawn from the
    unresolved ones alone, so a repair cannot claim to fix something that was never
    broken. ``parameter`` is the editable vocabulary of the families this engine
    offers, which contains no threshold; which of them the current family may move is
    checked on validation, with the reason handed back.
    """
    allowed = STAGE_ACTIONS[stage]
    if not allowed:
        raise ActionRejected(f"stage {stage.value} consults no model")
    families = available(code)
    names = sorted(
        {knob for family in families for knob in knobs_for(family)},
        key=list(BY_NAME).index,
    )
    change_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["parameter", "new", "reason"],
        "properties": {
            "parameter": {"type": "string", "enum": names},
            "old": {},
            "new": {
                "anyOf": [
                    _parameter_value_schema(BY_NAME[n], allowed_species) for n in names
                ]
            },
            "reason": {"type": "string", "maxLength": 400},
        },
    }

    props: dict[str, Any] = {
        "action": {"type": "string", "enum": list(allowed)},
        "diagnosis": {"type": "string", "maxLength": 600},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": list(evaluated_check_ids)},
        },
        "observations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "description": (
                "what the numbers say, in your own words: relations between "
                "measurements that no single check identifier names"
            ),
        },
        "expected_effect": {"type": "string", "maxLength": 400},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "changes": {"type": "array", "items": change_schema},
    }
    if "switch_method" in allowed:
        props["method"] = {
            "type": "string",
            "enum": [m.value for m in families],
            "description": "switch_method only: the family the next deck is built by",
        }
        props["reason"] = {
            "type": "string",
            "maxLength": 400,
            "description": "switch_method only: why this family, rather than an edit "
            "inside the current one",
        }
    if "escalate" in allowed:
        props["num_bands_needed"] = {
            "type": "integer",
            "minimum": 1,
            "description": (
                "escalate only, and only when the binding limit is the parent "
                "calculation's band pool: the num_bands (bands kept after the "
                "exclusions) the repair needs. Check it with validate_deck first; if "
                "the parent cannot supply it, the run returns a DFT prescription for "
                "a parent with that many bands"
            ),
        }
    if "escalate" in allowed:
        props["report"] = report_schema()
    if unresolved_check_ids:
        props["targets"] = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": list(unresolved_check_ids)},
            "description": "Which unresolved checks this action is meant to clear.",
        }

    required = ["action", "diagnosis", "evidence", "expected_effect", "confidence"]
    if unresolved_check_ids:
        required.append("targets")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


SHELL_SIZE = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7,
    "sp": 2,
    "sp2": 3,
    "sp3": 4,
    "px": 1,
    "py": 1,
    "pz": 1,
    "dz2": 1,
    "dxz": 1,
    "dyz": 1,
    "dx2-y2": 1,
    "dxy": 1,
}
"""What each shell contributes per atom.

Lives beside the vocabulary rather than beside the deck writer because the count a
projector set implies is a fact about the request, checked before the deck is written:
a set implying more functions than the calculation has bands is refused, not written.
"""


def implied_wannier_count(projectors, species_counts, spinors: bool = False) -> int:
    """How many Wannier functions a projector set implies for this cell.

    A species entry seeds every atom of that species; a positioned entry (``site``)
    seeds exactly one point of the cell.

    With ``spinors`` (a noncollinear or spin-orbit parent, ``spinors = .true.``) every
    orbital Wannier90 is given becomes two spinor functions, one per spin component,
    exactly as Wannier90 expands a projections line with no ``(u)``/``(d)`` qualifier.
    """
    total = 0
    for entry in projectors:
        per_site = sum(SHELL_SIZE.get(o, 1) for o in entry["orbitals"])
        # A positioned entry is one site; a species entry is every atom of it.
        sites = 1 if "site" in entry else species_counts.get(entry["species"], 1)
        total += per_site * sites
    return 2 * total if spinors else total


def _validate_species_entries(
    value: Any, allowed_species: Sequence[str], name: str, *, allow_empty: bool = False
) -> list[tuple[str, list]]:
    if not isinstance(value, list) or not value:
        raise ActionRejected(f"{name} must be a non-empty list")
    out = []
    seen = set()
    for entry in value:
        if not isinstance(entry, dict) or entry.keys() != {"species", "orbitals"}:
            raise ActionRejected(
                f"each {name} entry needs exactly 'species' and 'orbitals'"
            )
        species = entry["species"]
        if allowed_species and species not in allowed_species:
            raise ActionRejected(
                f"{name} names species {species!r}, which is not in this "
                f"structure ({', '.join(allowed_species)})"
            )
        if species in seen:
            raise ActionRejected(f"{name} names {species} twice; give it one entry")
        seen.add(species)
        orbitals = entry["orbitals"]
        if not isinstance(orbitals, list) or (not orbitals and not allow_empty):
            raise ActionRejected(f"{species}: orbitals must be a non-empty list")
        if len(set(map(str, orbitals))) != len(orbitals):
            raise ActionRejected(
                f"{species}: {sorted(map(str, orbitals))} names an orbital twice"
            )
        out.append((species, orbitals))
    return out


def _site_position(site: Any) -> str:
    """A positioned projector's site, normalised to Wannier90's ``f=``/``c=`` spelling."""
    if not isinstance(site, str):
        raise ActionRejected(f"projectors: site {site!r} must be a string")
    compact = site.replace(" ", "")
    match = re.fullmatch(SITE_PATTERN, compact, re.I)
    if match is None:
        raise ActionRejected(
            f"projectors: site {site!r} must read 'f=x,y,z' (fractional) or "
            "'c=x,y,z' (Cartesian, Angstrom)"
        )
    frame = compact[0].lower()
    try:
        coords = [float(x) for x in compact[2:].split(",")]
    except ValueError:
        raise ActionRejected(f"projectors: site {site!r} has a non-numeric coordinate")
    if not all(math.isfinite(x) for x in (c + 0.0 for c in coords)):
        raise ActionRejected(f"projectors: site {site!r} has a non-finite coordinate")
    return f"{frame}=" + ",".join(
        f"{x:.6f}".rstrip("0").rstrip(".") or "0" for x in (c + 0.0 for c in coords)
    )


def _validate_projectors(
    value: Any, allowed_species: Sequence[str]
) -> tuple[dict, ...]:
    """Species entries and positioned entries, each naming known orbital labels.

    A species entry ``{species, orbitals}`` seeds every atom of that species; a
    positioned entry ``{site, orbitals}`` seeds one point of the cell (a bond centre,
    an interstice), written as Wannier90 takes it: ``f=x,y,z`` or ``c=x,y,z``.
    """
    if not isinstance(value, list) or not value:
        raise ActionRejected("projectors must be a non-empty list")
    species_entries, site_entries = [], []
    for entry in value:
        if isinstance(entry, dict) and entry.keys() == {"site", "orbitals"}:
            site_entries.append(entry)
        elif isinstance(entry, dict) and entry.keys() == {"species", "orbitals"}:
            species_entries.append(entry)
        else:
            raise ActionRejected(
                "each projectors entry needs exactly 'species' and 'orbitals', or "
                "exactly 'site' and 'orbitals'"
            )
    out = []
    if species_entries:
        for species, orbitals in _validate_species_entries(
            species_entries, allowed_species, "projectors"
        ):
            out.append(
                {"species": species, "orbitals": _orbital_labels(species, orbitals)}
            )
    seen_sites = set()
    for entry in site_entries:
        site = _site_position(entry["site"])
        if site in seen_sites:
            raise ActionRejected(
                f"projectors names site {site} twice; give it one entry"
            )
        seen_sites.add(site)
        orbitals = entry["orbitals"]
        if not isinstance(orbitals, list) or not orbitals:
            raise ActionRejected(f"{site}: orbitals must be a non-empty list")
        if len(set(map(str, orbitals))) != len(orbitals):
            raise ActionRejected(
                f"{site}: {sorted(map(str, orbitals))} names an orbital twice"
            )
        out.append({"site": site, "orbitals": _orbital_labels(site, orbitals)})
    return tuple(out)


def _orbital_labels(where: str, orbitals: Sequence[Any]) -> tuple[str, ...]:
    for o in orbitals:
        if o not in ORBITAL_LABELS:
            raise ActionRejected(
                f"{where}: {o!r} is not an orbital label ({', '.join(ORBITAL_LABELS)})"
            )
    return tuple(orbitals)


def _validate_pao_subset(
    value: Any, allowed_species: Sequence[str]
) -> tuple[dict, ...]:
    out = []
    # An empty list drops the species' orbitals altogether: the PDWF way to build a
    # model on some species only (a d-only model on the transition-metal sites). The
    # deck writer refuses a subset that keeps no orbital at all.
    entries = _validate_species_entries(
        value, allowed_species, "pao_subset", allow_empty=True
    )
    for species, orbitals in entries:
        labels = []
        for o in orbitals:
            if not isinstance(o, str) or not o.strip():
                raise ActionRejected(f"{species}: {o!r} is not an orbital label")
            labels.append(o.strip())
        out.append({"species": species, "orbitals": tuple(labels)})
    return tuple(out)


def _validate_semicore(value: Any, allowed_species: Sequence[str]) -> tuple[dict, ...]:
    """Per species, the semicore shells to exclude; an empty list is allowed."""
    if not isinstance(value, list) or not value:
        raise ActionRejected(
            "semicore_exclusion must be a non-empty list of {species, shells}"
        )
    out, seen = [], set()
    for entry in value:
        if not isinstance(entry, dict) or entry.keys() != {"species", "shells"}:
            raise ActionRejected(
                "each semicore_exclusion entry needs exactly 'species' and 'shells'"
            )
        species = entry["species"]
        if allowed_species and species not in allowed_species:
            raise ActionRejected(
                f"semicore_exclusion names species {species!r}, which is not in this "
                f"structure ({', '.join(allowed_species)})"
            )
        if species in seen:
            raise ActionRejected(
                f"semicore_exclusion names {species} twice; give it one entry"
            )
        seen.add(species)
        shells = entry["shells"]
        if not isinstance(shells, list):
            raise ActionRejected(
                f"{species}: shells must be a list (empty includes all)"
            )
        labels = []
        for shell in shells:
            if not isinstance(shell, str) or not shell.strip():
                raise ActionRejected(f"{species}: {shell!r} is not a shell label")
            labels.append(shell.strip())
        if len(set(labels)) != len(labels):
            raise ActionRejected(f"{species}: {sorted(labels)} names a shell twice")
        out.append({"species": species, "shells": tuple(labels)})
    return tuple(out)


def _validate_value(p: Parameter, value: Any, allowed_species: Sequence[str]) -> Any:
    if p.kind == "semicore":
        return _validate_semicore(value, allowed_species)
    if p.kind == "projectors":
        return _validate_projectors(value, allowed_species)
    if p.kind == "pao_subset":
        return _validate_pao_subset(value, allowed_species)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionRejected(f"{p.name}: expected a number, got {value!r}")
    if p.kind == "int":
        if float(value) != int(value):
            raise ActionRejected(f"{p.name}: expected an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if p.minimum is not None and value < p.minimum:
        raise ActionRejected(
            f"{p.name}={value} is below the allowed minimum {p.minimum}"
        )
    if p.maximum is not None and value > p.maximum:
        raise ActionRejected(
            f"{p.name}={value} is above the allowed maximum {p.maximum}"
        )
    return value


def cross_family_reason(parameter: str, method: str | Method) -> str:
    """Why a knob is not this family's to move, said so the model can act on it."""
    method = as_method(method)
    owners = families_of(parameter)
    owned = ", ".join(knobs_for(method))
    if not owners:
        return f"{parameter} is not an editable parameter"
    names = " or ".join(o.value.upper() for o in owners)
    return (
        f"{parameter} is a {names} knob, and this construction is built "
        f"by {method.value.upper()}, whose knobs are: {owned}. Moving it would change "
        f"the method silently; to build by {names} use switch_method "
        f"(it costs an attempt and is recorded)"
    )


def validate_action(
    payload: Any,
    *,
    stage: Stage,
    evaluated_check_ids: Sequence[str],
    allowed_species: Sequence[str] = (),
    unresolved_check_ids: Sequence[str] = (),
    species_counts: Mapping[str, int] | None = None,
    band_count: int | None = None,
    parent_band_count: int | None = None,
    occupied_bands: int | None = None,
    method: str | Method = Method.SMV,
    code: str = "qe",
    spinors: bool = False,
    excluded_below: int = 0,
    sub_manifold_target: bool = False,
) -> Action:
    """Turn a raw provider response into an action, or refuse it.

    Schema conformance is requested of the provider and never assumed of it. A model
    that ignores the schema, a provider that does not enforce it, and a truncated
    response all arrive here as the same untrusted object.

    ``method`` is the family of the deck the action starts from, read from the deck;
    ``code`` the engine, which decides which families exist. ``spinors`` says the deck
    is a spinor one, so a named projector set implies twice its orbitals.
    ``excluded_below`` is the deck's bottom (semicore) exclusion: those parent bands
    are not available to ``num_bands``, so the ceiling is the parent's bands minus them.
    ``sub_manifold_target`` says the run's gate profile holds the model to a
    sub-manifold (``downfolded_d``), which lifts the floor that a projector set imply at
    least the kept occupied bands: that floor exists for the whole-manifold target only.
    """
    method = as_method(method)
    if not isinstance(payload, dict):
        raise ActionRejected(f"expected an object, got {type(payload).__name__}")

    required = {"action", "diagnosis", "evidence", "expected_effect", "confidence"}
    if unresolved_check_ids:
        required = required | {"targets"}
    missing = required - payload.keys()
    if missing:
        raise ActionRejected(f"missing required field(s): {', '.join(sorted(missing))}")

    unknown = payload.keys() - (
        required
        | {
            "changes",
            "targets",
            "observations",
            "method",
            "reason",
            "num_bands_needed",
            "report",
        }
    )
    if unknown:
        raise ActionRejected(f"unexpected field(s): {', '.join(sorted(unknown))}")

    kind = payload["action"]
    allowed = STAGE_ACTIONS[stage]
    if kind not in allowed:
        raise ActionRejected(
            f"action {kind!r} is not available in stage {stage.value} "
            f"(available: {', '.join(allowed) or 'none'})"
        )

    evidence_field = payload["evidence"]
    if not isinstance(evidence_field, list) or not evidence_field:
        raise ActionRejected("evidence must be a non-empty list of check identifiers")
    # An entry that is not a check identifier is a reading of the numbers rather than
    # a citation of one, so it is moved to `observations` instead of refused.
    evidence = [e for e in evidence_field if e in evaluated_check_ids]
    strays = [str(e) for e in evidence_field if e not in evaluated_check_ids]
    if not evidence:
        raise ActionRejected(
            "evidence cites no check the gate evaluated "
            f"({', '.join(evaluated_check_ids)}); readings of the numbers belong in "
            "'observations'"
        )
    observations = payload.get("observations") or []
    if not isinstance(observations, list):
        raise ActionRejected("observations must be a list of strings")
    observations = [str(o) for o in observations] + strays

    targets = payload.get("targets") or []
    if unresolved_check_ids:
        if not isinstance(targets, list) or not targets:
            raise ActionRejected(
                "targets must name at least one unresolved check this action addresses"
            )
        for target in targets:
            if target not in unresolved_check_ids:
                raise ActionRejected(
                    f"targets names {target!r}, which the gate did not leave unresolved. "
                    f"The unresolved checks are: {', '.join(unresolved_check_ids)}"
                )
    elif targets:
        raise ActionRejected("nothing is unresolved, so this action can target nothing")

    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ActionRejected(f"confidence must be a number, got {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ActionRejected(f"confidence={confidence} lies outside [0, 1]")

    for field_name in ("diagnosis", "expected_effect"):
        if not isinstance(payload[field_name], str) or not payload[field_name].strip():
            raise ActionRejected(f"{field_name} must be a non-empty string")

    # -- the family the action acts under --------------------------------
    target_method = None
    reason = None
    if kind == "switch_method":
        raw = payload.get("method")
        try:
            target_method = as_method(raw)
        except ValueError as exc:
            raise ActionRejected(f"switch_method: {exc}") from exc
        if target_method not in available(code):
            why = UNAVAILABLE_REASONS.get(
                (code, target_method), f"{code} offers no {target_method.value} route"
            )
            raise ActionRejected(
                f"{target_method.value.upper()} is not available on {code} "
                f"(available: {', '.join(m.value for m in available(code))}): {why}"
            )
        if target_method is method:
            raise ActionRejected(
                f"this construction is already built by {method.value.upper()}; an "
                f"edit inside the family is modify_parameters"
            )
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ActionRejected("switch_method needs a non-empty reason")
        reason = reason.strip()
    else:
        # Under schema enforcement a strict dialect makes every field present, so a
        # model may restate the current family or add a reason to an edit. Restating is
        # redundant and is accepted; naming another family is a switch and is refused;
        # a reason is a reading and joins the observations.
        given = payload.get("method")
        if given is not None:
            try:
                restated = as_method(given) is method
            except ValueError:
                restated = False
            if not restated:
                raise ActionRejected(
                    f"{kind} may not carry 'method' ({given!r}); naming a family is "
                    f"switch_method's"
                )
        stray_reason = payload.get("reason")
        if stray_reason is not None:
            if not isinstance(stray_reason, str):
                raise ActionRejected(f"{kind}: reason must be a string")
            if stray_reason.strip():
                observations.append(f"reason given with {kind}: {stray_reason.strip()}")
    acting_method = target_method or method

    raw_changes = payload.get("changes") or []
    if kind == "modify_parameters" and not raw_changes:
        raise ActionRejected("modify_parameters carries no change")
    if kind == "escalate" and raw_changes:
        raise ActionRejected("escalate may not carry parameter changes")
    needed = payload.get("num_bands_needed")
    if needed is not None and kind != "escalate":
        # Present only because a strict schema made it so; it claims nothing here.
        observations.append(
            f"num_bands_needed={needed!r} given with {kind}; ignored (it belongs to "
            f"escalate)"
        )
        needed = None
    if needed is not None:
        if (
            isinstance(needed, bool)
            or not isinstance(needed, (int, float))
            or float(needed) != int(needed)
            or int(needed) < 1
        ):
            raise ActionRejected(
                f"num_bands_needed must be a positive integer, got {needed!r}"
            )
        needed = int(needed)
    report = _validate_report(
        payload.get("report"), kind, needed, excluded_below, observations,
        str(payload["expected_effect"]),
    )  # fmt: skip
    if report is not None and report["binding_limit"] == "parent_band_pool":
        if needed is None:
            nbnd = (report.get("requested_changes") or {}).get("nbnd")
            if nbnd is None:
                raise ActionRejected(
                    "report: binding_limit parent_band_pool needs requested_changes."
                    "nbnd (the parent's band count) or num_bands_needed"
                )
            needed = int(nbnd) - max(0, int(excluded_below or 0))
            if needed < 1:
                raise ActionRejected(
                    f"report: nbnd={nbnd} leaves no band above the deck's bottom "
                    f"exclusion of {excluded_below}"
                )

    changes: list[ParameterChange] = []
    seen: set[str] = set()
    for c in raw_changes:
        if not isinstance(c, dict):
            raise ActionRejected("each change must be an object")
        name = c.get("parameter")
        p = BY_NAME.get(name)
        if p is None:
            raise ActionRejected(
                f"{name!r} is not an editable parameter; the vocabulary is "
                f"{', '.join(knobs_for(acting_method))}"
            )
        if name not in knobs_for(acting_method):
            raise ActionRejected(cross_family_reason(name, acting_method))
        unavailable = knob_unavailable(code, acting_method, name)
        if unavailable:
            raise ActionRejected(
                f"{name} has no route on {code} {acting_method.value.upper()}: "
                f"{unavailable}"
            )
        if name in seen:
            raise ActionRejected(f"{name} is assigned twice in one action")
        seen.add(name)
        if "new" not in c:
            raise ActionRejected(f"{name}: no new value given")
        reason_text = c.get("reason")
        if not isinstance(reason_text, str) or not reason_text.strip():
            raise ActionRejected(f"{name}: reason must be a non-empty string")
        changes.append(
            ParameterChange(
                parameter=name,
                old=c.get("old"),
                new=_validate_value(p, c["new"], allowed_species),
                reason=reason_text.strip(),
            )
        )

    # num_bands cannot reach past what the parent calculation produced -- there is no
    # more spectrum to take -- and a projector set cannot ask for more functions than
    # the bands it draws them from. The two interact: a repair may widen the projectors
    # *and* raise num_bands in one action, and then the ceiling is the raised value.
    proposed_bands = next(
        (int(c.new) for c in changes if c.parameter == "num_bands"), None
    )
    moves_bottom = any(c.parameter == "semicore_exclusion" for c in changes)
    below = 0 if moves_bottom else max(0, int(excluded_below or 0))
    if proposed_bands is not None and parent_band_count is not None:
        ceiling = parent_band_count - below
        if proposed_bands > ceiling:
            held = f"it produced {parent_band_count} bands" + (
                f" and the deck excludes the lowest {below}" if below else ""
            )
            raise ActionRejected(
                f"num_bands={proposed_bands} asks for more than the parent calculation "
                f"can supply: {held}, so at most {ceiling} are available; it is a "
                f"ceiling, not a "
                f"request. If the repair needs more, escalate with num_bands_needed "
                f"and the run returns a DFT prescription for that parent"
            )
    effective_bands = proposed_bands if proposed_bands is not None else band_count
    if effective_bands is not None and species_counts:
        for c in changes:
            if c.parameter != "projectors":
                continue
            implied = implied_wannier_count(c.new, species_counts, spinors)
            if implied > effective_bands:
                raise ActionRejected(
                    f"those projectors imply {implied} Wannier functions, but this "
                    f"construction carries {effective_bands} bands to draw them from; "
                    f"ask for fewer shells, or raise num_bands in the same action"
                )
            if (
                occupied_bands is not None
                and implied < occupied_bands
                and not sub_manifold_target
            ):
                raise ActionRejected(
                    f"those projectors imply only {implied} Wannier functions, and "
                    f"{occupied_bands} of the bands this deck keeps are occupied "
                    f"everywhere in the zone; the set has to cover them, because this "
                    f"run's gate profile holds the model to the whole occupied "
                    f"manifold (occupied_manifold_is_represented). Only a profile "
                    f"whose target is a sub-manifold (downfolded_d: the transition-"
                    f"metal d manifold) allows fewer functions than occupied bands"
                )

    action = Action(
        kind=kind,
        diagnosis=payload["diagnosis"].strip(),
        evidence=tuple(evidence),
        observations=tuple(observations),
        targets=tuple(targets),
        expected_effect=payload["expected_effect"].strip(),
        confidence=float(confidence),
        changes=tuple(changes),
        method=target_method.value if target_method else None,
        method_from=method.value if target_method else None,
        reason=reason,
        num_bands_needed=needed,
        report=report,
    )
    _check_internal_consistency(action)
    return action


def _validate_report(raw, kind, needed, excluded_below, observations, expected_effect):
    """escalate's report, validated; a bare num_bands_needed becomes one."""
    if raw is not None and kind != "escalate":
        # present only because a strict schema made it so; it claims nothing here
        observations.append(
            f"a report given with {kind} was ignored (it is escalate's)"
        )
        return None
    if kind != "escalate":
        return None
    if raw is None:
        if needed is None:
            return None
        return {
            "binding_limit": "parent_band_pool",
            "requested_changes": {
                "nbnd": int(needed) + max(0, int(excluded_below or 0))
            },
            "evidence": [],
            "expected_effect": expected_effect.strip(),
            "synthesized": True,
        }
    if not isinstance(raw, dict):
        raise ActionRejected("report must be an object")
    unknown = raw.keys() - {"binding_limit", "requested_changes", "evidence",
                            "expected_effect"}  # fmt: skip
    if unknown:
        raise ActionRejected(
            f"report: unexpected field(s): {', '.join(sorted(unknown))}"
        )
    limit = raw.get("binding_limit")
    if limit not in BINDING_LIMITS:
        raise ActionRejected(
            f"report.binding_limit must be one of {', '.join(BINDING_LIMITS)}; got "
            f"{limit!r}"
        )
    evidence = raw.get("evidence")
    if not isinstance(evidence, list) or not [e for e in evidence if str(e).strip()]:
        raise ActionRejected(
            "report.evidence must list the check ids and tool numbers the claim rests on"
        )
    effect = raw.get("expected_effect")
    if not isinstance(effect, str) or not effect.strip():
        raise ActionRejected("report.expected_effect must be a non-empty string")
    changes = raw.get("requested_changes") or {}
    if not isinstance(changes, dict):
        raise ActionRejected("report.requested_changes must be an object")
    clean: dict[str, Any] = {}
    for key, value in changes.items():
        if key not in REQUESTED_CHANGE_SCHEMA:
            raise ActionRejected(
                f"report.requested_changes: {key!r} is not one of "
                f"{', '.join(REQUESTED_CHANGE_SCHEMA)} (anything else goes in 'other')"
            )
        if value is None:
            continue
        clean[key] = _requested_value(key, value)
    if needed is not None and limit != "parent_band_pool":
        raise ActionRejected(
            "num_bands_needed names the parent band pool; with it, report.binding_limit "
            f"must be parent_band_pool, not {limit}"
        )
    if needed is not None and "nbnd" not in clean:
        clean["nbnd"] = int(needed) + max(0, int(excluded_below or 0))
    return {
        "binding_limit": limit,
        "requested_changes": clean,
        "evidence": [str(e).strip() for e in evidence if str(e).strip()],
        "expected_effect": effect.strip(),
    }


def _requested_value(key: str, value: Any) -> Any:
    spec = REQUESTED_CHANGE_SCHEMA[key]
    kind = spec["type"]
    bad = ActionRejected(f"report.requested_changes.{key}: {value!r} is not valid")
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad
        if float(value) != int(value) or int(value) < spec.get("minimum", -(10**9)):
            raise bad
        if "enum" in spec and int(value) not in spec["enum"]:
            raise bad
        return int(value)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad
        if float(value) < spec.get("minimum", float("-inf")):
            raise bad
        return float(value)
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise bad
        return value.strip()[: spec.get("maxLength", 600)]
    if key == "mp_grid":
        if not isinstance(value, list) or len(value) != 3:
            raise bad
        out = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) < 1:
                raise bad
            out.append(int(v))
        return out
    if key == "starting_magnetization":
        if not isinstance(value, list):
            raise bad
        out = []
        for entry in value:
            if not isinstance(entry, dict) or not isinstance(entry.get("species"), str):
                raise bad
            if isinstance(entry.get("value"), bool) or not isinstance(
                entry.get("value"), (int, float)
            ):
                raise bad
            out.append({"species": entry["species"], "value": float(entry["value"])})
        return out
    raise bad


def _check_internal_consistency(action: Action) -> None:
    """Refuse an action whose own values contradict each other.

    Each knob passes its own range and the set still describes nothing: a frozen
    window above the outer window selects an empty subspace, and a projectability
    floor above its ceiling excludes every state.
    """
    c = {ch.parameter: ch.new for ch in action.changes}
    pairs = (
        (
            "dis_froz_max",
            "dis_win_max",
            "the frozen window must sit inside the outer window",
        ),
        ("dis_win_min", "dis_win_max", "dis_win_min must lie below dis_win_max"),
        ("dis_froz_min", "dis_froz_max", "dis_froz_min must lie below dis_froz_max"),
        ("dis_proj_min", "dis_proj_max", "dis_proj_min must lie below dis_proj_max"),
    )
    for lo, hi, message in pairs:
        if lo in c and hi in c and c[lo] > c[hi]:
            raise ActionRejected(f"{lo}={c[lo]} against {hi}={c[hi]}: {message}")
