"""A corpus of repairs that worked, and retrieval over it by physics.

The repair stage's ``search_references`` tool searches it. What it searches is not a
manual: it is the record every run already produces, distilled to the step where a
check went from failing to passing, kept with the electronic-structure signature of the
calculation it happened on, the engine it ran on, and the regime it belongs to.

The package ships its corpus (:data:`DEFAULT_CORPUS`) and searches it by default. An
entry whose ``regime`` is ``obsolete`` -- distilled under a construction protocol the
workflow no longer starts from -- is kept for provenance and left out of retrieval
unless asked for.

Two decisions follow from that shape.

*Entries are distilled, not written.* Asking people to write recipes by hand collects
nothing; every run of this tool is a candidate entry, and the distillation reads the run
log rather than anybody's summary of it. A repair that resolved nothing is not an entry.

*Retrieval matches on the signature, not on the material's name.* A recipe for a
one-orbital simple metal should reach the next one-orbital simple metal whether or not
anyone knows they are related, and should not reach a wide-gap oxide that happens to
share an element. The material's name is kept in the provenance so a reader can trace
the entry; it carries no weight in the ranking, and the policy tells the model not to
reason from it.

A recipe is evidence about what has worked. It is data the model reads, never
instruction it follows: whatever the corpus says, the answer still has to pass the same
validation and the same gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The check identifier a run reports when the calculation itself never produced a
#: construction. Kept here as well as in the workflow because a recipe distilled from
#: such an attempt names it, and the corpus outlives the run that wrote it.
CALCULATION_DID_NOT_RUN = "calculation_did_not_run"

#: The corpus that ships with the package; the default ``recipe_dir``.
DEFAULT_CORPUS = Path(__file__).parent / "corpus"

#: The regime label that keeps an entry out of the default corpus.
OBSOLETE = "obsolete"


@dataclass(frozen=True)
class Recipe:
    """One repair that resolved something, with the calculation it happened on."""

    recipe_id: str
    signature: dict[str, Any]
    resolved: tuple[str, ...]
    """Checks that were failing or unevaluable before this change and passing after.

    The entry's reason for existing. A change that resolved nothing is not evidence
    about what works, however sensible its rationale sounded.
    """
    still_unresolved: tuple[str, ...]
    changes: tuple[dict[str, Any], ...]
    diagnosis: str
    provenance: dict[str, Any]
    measurements: dict[str, Any] = field(default_factory=dict)
    engine: str | None = None
    """The code the run used (``qe``, ``vasp``); ``None`` for entries that predate it."""
    regime: str = "current"
    """``obsolete`` keeps the entry out of the default corpus."""
    method: str | None = None
    """The method family the repaired construction was built by."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def distil_run(
    log: Mapping[str, Any],
    signature: Mapping[str, Any],
    engine: str | None = None,
) -> list[Recipe]:
    """Read a run record and keep the steps where a check stopped failing.

    The comparison is against the attempt the change was made *from*, which is not
    always the one before it in the list: an attempt whose calculation never ran has no
    verdicts to compare against, so the last attempt that produced any is used.
    """
    out: list[Recipe] = []
    material = str(log.get("material", "unknown"))
    engine = engine or log.get("backend")
    previous: set[str] | None = None

    for attempt in log.get("attempts", ()):
        outcome = attempt.get("outcome")
        action = attempt.get("action")
        if outcome and action and previous is not None:
            before = previous
            after = _unresolved(outcome)
            resolved = tuple(c for c in before if c not in after)
            if resolved:
                out.append(
                    Recipe(
                        recipe_id=_identifier(material, attempt.get("index"), resolved),
                        signature=dict(signature),
                        resolved=resolved,
                        still_unresolved=tuple(sorted(after)),
                        changes=tuple(action.get("changes", ())),
                        diagnosis=str(action.get("diagnosis", "")),
                        provenance={
                            "material": material,
                            "attempt": attempt.get("index"),
                            "recorded_at": attempt.get("finished_at"),
                            "certified": bool(outcome.get("passed")),
                        },
                        measurements=_named(outcome, resolved),
                        engine=engine,
                        method=attempt.get("method"),
                    )
                )
        if outcome:
            previous = _unresolved(outcome)
        elif attempt.get("calculation_error"):
            # A construction that would not run is unresolved in the strongest sense,
            # and the change that made it run is the most useful entry the corpus can
            # hold. Comparing only against attempts that produced verdicts discarded
            # exactly those: the alkali metal went from a deck Wannier90 refused to a
            # certified construction and contributed nothing.
            # Added to what was already known, not substituted for it. A failed
            # attempt produces no verdicts, so the last ones still stand; replacing
            # them loses the checks a later repair also cleared.
            previous = (previous or set()) | {CALCULATION_DID_NOT_RUN}
    return out


def _identifier(material: str, index: Any, resolved: Sequence[str]) -> str:
    """A name for the entry that does not repeat the material's.

    The model is told to reason from the signature and not from the material, and was
    then shown an identifier with the material in it: asked to repair a rocksalt oxide
    it cited "a very similar SrO3-containing system" and argued from the formula. The
    material stays in the provenance, where a person tracing the entry reads it.

    This removes the label, not the composition. An entry's species counts and the
    shells its projectors name are physics a projector recipe cannot be written
    without, and a reader can assemble a formula from them. What is gone is the free
    hint -- the model now has to go through what the species supply rather than
    recognise a string.
    """
    digest = hashlib.sha1(f"{material}-{index}".encode()).hexdigest()[:8]
    return f"{sorted(resolved)[0]}-{digest}"


def _unresolved(outcome: Mapping[str, Any]) -> set[str]:
    """Checks the gate did not settle: failed, or applicable and not evaluable."""
    failed = set(outcome.get("failed_check_ids", ()))
    measurements = outcome.get("measurements") or {}
    unevaluable = {
        check
        for check, m in measurements.items()
        if isinstance(m, Mapping)
        and m.get("applies", True)
        and not m.get("evaluable", True)
    }
    return failed | unevaluable


def _named(outcome: Mapping[str, Any], checks: Sequence[str]) -> dict[str, Any]:
    measurements = outcome.get("measurements") or {}
    return {c: measurements[c] for c in checks if c in measurements}


# ---------------------------------------------------------------------------
# the corpus on disk
# ---------------------------------------------------------------------------


def write(recipe: Recipe, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{recipe.recipe_id}.json"
    path.write_text(json.dumps(recipe.to_dict(), indent=2, default=str) + "\n")
    return path


def load(directory: str | Path, *, include_obsolete: bool = False) -> list[Recipe]:
    """Every recipe in a directory. A file that does not parse is skipped, not fatal.

    Entries marked ``regime: obsolete`` are left out unless ``include_obsolete``.

    The corpus is contributed, so one malformed entry must not take the stage down with
    it -- the failure mode of a shared collection is that somebody's file is broken,
    and the cost of that should be one missing recipe.
    """
    out: list[Recipe] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            d = json.loads(path.read_text())
            out.append(
                Recipe(
                    recipe_id=str(d["recipe_id"]),
                    signature=dict(d.get("signature") or {}),
                    resolved=tuple(d.get("resolved") or ()),
                    still_unresolved=tuple(d.get("still_unresolved") or ()),
                    changes=tuple(d.get("changes") or ()),
                    diagnosis=str(d.get("diagnosis", "")),
                    provenance=dict(d.get("provenance") or {}),
                    measurements=dict(d.get("measurements") or {}),
                    engine=d.get("engine"),
                    regime=str(d.get("regime") or "current"),
                    method=d.get("method"),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    if not include_obsolete:
        out = [r for r in out if r.regime != OBSOLETE]
    return out


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def search(
    corpus: Iterable[Recipe],
    *,
    signature: Mapping[str, Any],
    unresolved: Sequence[str],
    query: str = "",
    limit: int = 4,
    exclude_material: str | None = None,
    exclude_composition: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> list[tuple[Recipe, float]]:
    """The entries that resolved what is unresolved here, nearest signature first.

    Answering the right question is a filter and not a score: an entry that resolved
    nothing the caller is stuck on is not a weaker match, it is a match to a different
    problem. Among those that qualify, the ranking is by how alike the two calculations
    are -- whether the manifold is isolated, how wide the gap is, how many bands cross
    the level, how big the cell is -- with a small bonus for words the caller asked for.

    An entry distilled from this material is a memory of an earlier run on it, not a
    reference from elsewhere. Retrieval that returns it measures recall where the stage
    is supposed to measure transfer, so the caller says which material is its own and
    has those held back. The material is named by its composition
    (``exclude_composition``, matched against the entry's recorded species), because a
    seed is a spelling and not an identity: ``Ti_vasp`` and ``Ti_pdwf_rep2`` are runs on
    the titanium that wrote ``Ti_pdwf``'s entries. ``exclude_material`` (the seed) is
    kept for entries written before the species were recorded, which have nothing else.
    """
    wanted = set(unresolved)
    own = composition_of(exclude_composition)
    scored: list[tuple[Recipe, float]] = []
    for recipe in corpus:
        if _same_material(recipe, exclude_material, own):
            continue
        if wanted and not (set(recipe.resolved) & wanted):
            continue
        score = _similarity(signature, recipe.signature, query, recipe)
        if engine and recipe.engine == engine:
            # The same knob is spelled and behaves the same way on the same engine.
            score = round(score + 0.5, 4)
        scored.append((recipe, score))
    scored.sort(key=lambda pair: -pair[1])
    return scored[:limit]


def composition_of(species_counts: Mapping[str, int] | None) -> dict[str, int]:
    """The material's identity: its species in their smallest whole-number ratio.

    Two atoms of titanium and one are the same metal; a conventional silicon cell and
    a primitive one are the same semiconductor. The cell that was computed is a choice
    of the calculation, not of the material, and so is the seed the run was given.
    """
    # A corpus entry is data the loop reads, and load() already tolerates a malformed
    # one; retrieval must not be the place it finally raises. A species field that is
    # not a mapping, or a count that is not a positive whole number, records nothing.
    if not isinstance(species_counts, Mapping):
        return {}
    counts: dict[str, int] = {}
    for symbol, n in species_counts.items():
        try:
            count = int(n)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(symbol)] = count
    if not counts:
        return {}
    divisor = 0
    for n in counts.values():
        divisor = math.gcd(divisor, n)
    return {s: n // divisor for s, n in sorted(counts.items())}


def _same_material(
    recipe: Recipe, seed: str | None, composition: Mapping[str, int]
) -> bool:
    """Whether the entry was distilled from a run on the caller's own material.

    Decided on composition alone when both sides carry one: a supercell, a replicate
    and a run on another engine are the same material under any seed, and two seeds
    that happen to share a prefix are not the same material when their compositions
    differ. The seed rule is the fallback for the cases with nothing better -- an entry
    written before the species were recorded, or a caller that knows only the seed.
    Polymorphs share a composition and are held back together: the stage errs towards
    measuring less recall, not more.
    """
    recorded = composition_of(recipe.signature.get("species"))
    if recorded and composition:
        return recorded == composition
    if not seed:
        return False
    theirs = str(recipe.provenance.get("material", ""))
    return theirs == seed or theirs.startswith(f"{seed}_")


def _similarity(
    here: Mapping[str, Any], there: Mapping[str, Any], query: str, recipe: Recipe
) -> float:
    score = 0.0
    if here.get("manifold_is_isolated") == there.get("manifold_is_isolated"):
        score += 2.0
    score += _close(here.get("band_gap_ev"), there.get("band_gap_ev"), scale=2.0)
    score += _close(
        here.get("bands_crossing_fermi_level"),
        there.get("bands_crossing_fermi_level"),
        scale=4.0,
    )
    score += _close(here.get("atoms_in_cell"), there.get("atoms_in_cell"), scale=4.0)
    score += _close(
        here.get("occupied_bandwidth_ev"),
        there.get("occupied_bandwidth_ev"),
        scale=10.0,
    )
    # The number of atomic orbitals a species supplies is what the hard regime is about,
    # so a shared element counts -- but only through that, never through the name alone.
    if set(here.get("species") or {}) & set(there.get("species") or {}):
        score += 0.5
    if query:
        words = {w for w in query.lower().split() if len(w) > 3}
        text = (recipe.diagnosis + " " + " ".join(recipe.resolved)).lower()
        score += 0.25 * sum(1 for w in words if w in text)
    if recipe.provenance.get("certified"):
        score += 0.5
    return round(score, 4)


def _close(a: Any, b: Any, *, scale: float) -> float:
    """One point for identical, falling off over ``scale``, zero when either is absent."""
    if a is None or b is None:
        return 0.0
    try:
        return max(0.0, 1.0 - abs(float(a) - float(b)) / scale)
    except (TypeError, ValueError):
        return 0.0


def render(results: Sequence[tuple[Recipe, float]]) -> str:
    """The retrieved entries as the model sees them.

    Written as evidence with its provenance attached, not as advice: each entry says
    what it resolved, on what kind of calculation, and by changing what. The material's
    name appears once, in the provenance, so a reader can trace the entry.
    """
    if not results:
        return (
            "REFERENCE SEARCH\n"
            "  nothing in the corpus resolved any of the checks still open here."
        )
    lines = ["REFERENCE SEARCH", f"  {len(results)} entries, closest first"]
    for recipe, score in results:
        lines.append("")
        lines.append(
            f"  [{recipe.recipe_id}]  match {score:.2f}"
            + (f"  engine {recipe.engine}" if recipe.engine else "")
            + (f"  method {recipe.method}" if recipe.method else "")
        )
        lines.append(f"    resolved: {', '.join(recipe.resolved)}")
        if recipe.still_unresolved:
            lines.append(f"    left open: {', '.join(recipe.still_unresolved)}")
        for change in recipe.changes:
            lines.append(
                f"    changed {change.get('parameter')}: "
                f"{change.get('old')} -> {change.get('new')}"
            )
            if change.get("reason"):
                lines.append(f"      because: {change['reason']}")
        sig = recipe.signature
        lines.append(
            "    on a calculation with: "
            f"{'an isolated' if sig.get('manifold_is_isolated') else 'no isolated'} "
            f"manifold, gap {sig.get('band_gap_ev')} eV, "
            f"{sig.get('bands_crossing_fermi_level')} bands crossing the Fermi level, "
            f"{sig.get('atoms_in_cell')} atoms, species {sig.get('species')}"
        )
        # No material label: the entry is evidence about a kind of calculation, and
        # naming the one it came from invites arguing from the name. The composition
        # is still there, in the species counts, because a projector recipe is about
        # species -- this withholds the hint, not the physics.
        if recipe.provenance.get("certified"):
            lines.append("    this construction went on to be certified")
    return "\n".join(lines)
