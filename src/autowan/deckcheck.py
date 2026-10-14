"""Check a deck before anything runs on it: is it a consistent construction at all?

Every check here is arithmetic on the deck and the stored parent, so it costs nothing
and is deterministic. The same function answers the model's ``validate_deck`` tool and
gates every repaired deck before Wannier90 or the overlap step sees it: a deck it
refuses becomes a refused attempt, with the reason handed back, instead of an engine
error after the overlaps have been spent.

What is checked:

* the method family is one this engine offers, and the deck is internally that family
  (no projections block on a PDWF or SCDM deck, PAO projections on a QE PDWF deck);
* the Wannier count agrees with what the projectors imply (SMV) or the PAOs kept
  (PDWF), and does not exceed ``num_bands``; a spinor deck counts both spin components
  (2j+1 per j-resolved PAO), and its ``spinors`` flag must match the parent's;
* the band arithmetic: ``num_bands`` plus the excluded bands is what the parent stored;
* the bottom (semicore) exclusion is one the gate honours: an excluded block holding
  occupied states must lie, over the whole mesh, more than 1 % of the occupied bandwidth
  below every kept band (:func:`.gate.judge_exclusion`, the gate's own rule), or every
  excluded occupied state counts as missing and no repair of anything else can pass;
  and, where the deck's semicore shells are known, no pseudo-atomic orbital of an
  excluded shell is kept;
* at every k-point and spin, the frozen states fit in ``num_wann`` and the outer window
  holds at least ``num_wann`` states;
* for PDWF, projectability does not exceed one (when the overlaps are on hand).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .families import UNAVAILABLE_REASONS, Method, available
from .inputs import (
    ConstructionInputs,
    InputError,
    _excluded_from_text,
    assignments,
    deck_atom_species,
    deck_method,
    deck_spinors,
    pao_excluded,
    pao_layout,
)

#: Tolerance on projectability above one before a PDWF deck is refused.
PROJECTABILITY_TOLERANCE = 1e-2


@dataclass
class DeckCheck:
    method: str
    problems: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def reason(self) -> str:
        return "; ".join(self.problems)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "method": self.method,
            "problems": list(self.problems),
            "facts": self.facts,
            "notes": list(self.notes),
        }


def _number(value) -> float | None:
    try:
        return float(str(value).strip().lower().replace("d", "e"))
    except (TypeError, ValueError):
        return None


def _projection_block(text: str) -> list[str]:
    lines, inside = [], False
    for line in text.splitlines():
        cleaned = line.split("!")[0].split("#")[0].strip()
        lowered = cleaned.lower()
        if lowered.startswith("begin projections"):
            inside = True
            continue
        if lowered.startswith("end projections"):
            break
        if inside and cleaned:
            lines.append(cleaned)
    return lines


_SHELL = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7,
    "sp": 2,
    "sp2": 3,
    "sp3": 4,
    "sp3d": 5,
    "sp3d2": 6,
}


def _orbital_functions(token: str) -> int | None:
    """Functions one orbital token names on one atom, before any spinor doubling."""
    token = token.lower()
    if token.startswith("l="):
        parts = token.split(",")
        try:
            l = int(parts[0][2:])
        except ValueError:
            return None
        mr = [p for p in parts[1:] if p.startswith("mr=")]
        if mr:
            # ``l=2,mr=1,4`` names those m_r only (the values after mr= run on).
            listed = [mr[0][3:]] + [p for p in parts[parts.index(mr[0]) + 1 :]]
            return len([v for v in listed if v.strip()])
        return 2 * l + 1 if l >= 0 else -l + 1
    return _SHELL.get(token)


_SPIN_QUALIFIER = re.compile(r"\(([ud](?:,[ud])?)\)")
_AXIS = re.compile(r"\[[^\]]*\]")


def implied_by_block(
    lines: list[str], species_counts: dict[str, int], spinors: bool = False
) -> int | None:
    """Functions a projections block implies, or ``None`` if a line is not readable.

    With ``spinors`` every orbital is two spinor functions, unless its line qualifies the
    spin component (``Fe:d(u)``, ``Fe:d(d)``: one each; ``(u,d)``: both); a spin
    quantisation axis (``[0,0,1]``) changes no count.
    """
    total = 0
    for line in lines:
        compact = line.replace(" ", "")
        qualifier = _SPIN_QUALIFIER.search(compact)
        components = 2 if spinors else 1
        if qualifier is not None:
            if not spinors:
                return None  # Wannier90 refuses a spin qualifier without spinors
            components = len(qualifier.group(1).split(","))
        compact = _AXIS.sub("", _SPIN_QUALIFIER.sub("", compact))
        parts = compact.split(":")
        if len(parts) < 2:
            return None
        site, orbitals = parts[0], parts[1]
        if site.lower() in ("random",):
            return None
        per_atom = 0
        for orbital in orbitals.split(";"):
            count = _orbital_functions(orbital)
            if count is None:
                return None
            per_atom += count
        per_atom *= components
        if site in species_counts:
            total += per_atom * species_counts[site]
        elif site.lower().startswith(("c=", "f=")):
            total += per_atom
        else:
            return None
    return total


def kept_band_energies(parent, excluded, num_bands: int | None) -> np.ndarray:
    keep = [b for b in range(parent.n_bands) if (b + 1) not in set(excluded)]
    if num_bands is not None:
        keep = keep[:num_bands]
    return parent.eigenvalues_ev[..., keep]


def frozen_counts(
    bands: np.ndarray, froz_min: float | None, froz_max: float | None
) -> np.ndarray:
    """States inside the frozen window at each (spin, k)."""
    if froz_max is None:
        return np.zeros(bands.shape[:2], dtype=int)
    lower = bands >= (froz_min if froz_min is not None else -np.inf)
    return (lower & (bands <= froz_max)).sum(axis=-1)


def check_deck(
    inputs: ConstructionInputs,
    *,
    parent,
    code: str,
    run_dir: Path | None = None,
    band_arithmetic: bool = True,
    projectability: np.ndarray | None = None,
    ctx=None,
    spin_index: int = 0,
    whole_occupied_target: bool = True,
) -> DeckCheck:
    """Every deterministic objection to running ``inputs``; empty when there is none.

    ``ctx`` (the run's :class:`.methods.DeckContext`) adds the semicore-shell checks,
    which need the run's first deck and the parent's pseudopotentials; ``spin_index`` is
    the channel the construction is for, as the gate reads it. ``whole_occupied_target``
    is False under a sub-manifold gate profile (``downfolded_d``), which runs no
    whole-manifold check: an exclusion the gate would not honour is then stated, not
    refused.
    """
    win_text = inputs.win.read_text()
    pw_text = (
        inputs.pw2wan.read_text()
        if inputs.pw2wan is not None and inputs.pw2wan.is_file()
        else None
    )
    method = deck_method(win_text, pw_text)
    out = DeckCheck(method.value)
    values = assignments(win_text)
    block = _projection_block(win_text)

    # -- the family ------------------------------------------------------
    if method not in available(code):
        out.problems.append(
            f"{method.value.upper()} is not available on {code}: "
            + UNAVAILABLE_REASONS.get((code, method), "no route")
        )
    if method in (Method.PDWF, Method.SCDM) and block:
        out.problems.append(
            f"a {method.value.upper()} deck names analytic projections; those belong "
            f"to SMV, and Wannier90 would use them instead of the {method.value.upper()} "
            f"projections"
        )
    if method is Method.PDWF and code == "qe":
        if pw_text is None or not assignments(pw_text).get(
            "atom_proj", ""
        ).lower().startswith((".t", "t")):
            out.problems.append(
                "a PDWF deck projects on the pseudopotential's atomic orbitals "
                "(atom_proj = .true.), and this one does not"
            )
    if method is Method.PDWF:
        lo, hi = _number(values.get("dis_proj_min")), _number(
            values.get("dis_proj_max")
        )
        if lo is not None and hi is not None and lo > hi:
            out.problems.append(f"dis_proj_min = {lo} lies above dis_proj_max = {hi}")

    # -- counts ----------------------------------------------------------
    num_wann = _number(values.get("num_wann"))
    num_bands = _number(values.get("num_bands"))
    num_wann = int(num_wann) if num_wann is not None else None
    num_bands = int(num_bands) if num_bands is not None else None
    out.facts.update(num_wann=num_wann, num_bands=num_bands)
    species = deck_atom_species(win_text) or list(parent.species)
    counts: dict[str, int] = {}
    for symbol in species:
        counts[symbol] = counts.get(symbol, 0) + 1
    if num_wann is None:
        out.problems.append("the deck declares no num_wann")
    # -- spinors -----------------------------------------------------------
    spinors = deck_spinors(win_text)
    parent_spinor = bool(getattr(parent, "spinor", False))
    out.facts["spinors"] = spinors
    if spinors != parent_spinor:
        out.problems.append(
            "the deck declares spinors = .true. but the parent's bands are not spinors"
            if spinors
            else "the parent is a spinor (noncollinear / spin-orbit) calculation and "
            "the deck does not declare spinors = .true.; pw2wannier90 and VASP refuse "
            "it, and each named orbital would count once instead of once per spin"
        )
    if method is Method.SMV and block and num_wann is not None:
        implied = implied_by_block(block, counts, spinors)
        if implied is not None and implied != num_wann:
            out.problems.append(
                f"the projections imply {implied} Wannier functions but num_wann = "
                f"{num_wann}"
            )
        out.facts["projectors_imply"] = implied
    if (
        method is Method.PDWF
        and code == "qe"
        and pw_text is not None
        and num_wann is not None
    ):
        try:
            layout = pao_layout(run_dir or inputs.win.parent, win_text, pw_text)
            kept = len(layout) - len(pao_excluded(pw_text) & {s.index for s in layout})
            out.facts["paos_kept"] = kept
            if kept != num_wann:
                out.problems.append(
                    f"the deck keeps {kept} pseudo-atomic orbitals but num_wann = {num_wann}"
                )
        except InputError as exc:
            out.notes.append(f"PAO count not checked: {exc}")
    if num_wann is not None and num_bands is not None and num_wann > num_bands:
        out.problems.append(
            f"num_wann = {num_wann} exceeds num_bands = {num_bands}: there are not "
            f"enough bands to draw the functions from"
        )

    # -- band arithmetic --------------------------------------------------
    excluded = _excluded_from_text(win_text)
    out.facts["excluded_bands"] = len(excluded)
    raw = parent.n_bands
    if band_arithmetic and num_bands is not None:
        outside = [b for b in excluded if not 1 <= b <= raw]
        if outside:
            out.problems.append(
                f"exclude_bands lists {outside[:5]} outside the parent's 1..{raw}"
            )
        elif num_bands != raw - len(excluded):
            out.problems.append(
                f"num_bands = {num_bands}, but the parent stored {raw} bands and the "
                f"deck excludes {len(excluded)}, so the overlaps carry "
                f"{raw - len(excluded)}; num_bands must be {raw - len(excluded)}"
            )
    elif num_bands is not None and num_bands > raw - len(excluded):
        out.problems.append(
            f"num_bands = {num_bands} asks for more than the {raw - len(excluded)} "
            f"bands the parent stored outside the exclusion"
        )

    # -- the bottom (semicore) exclusion: the gate's honour rule -----------
    _check_exclusion(out, parent, excluded, spin_index, whole_occupied_target)
    if ctx is not None:
        _check_semicore_shells(
            out, win_text, pw_text, run_dir or inputs.win.parent, ctx
        )

    # -- windows ----------------------------------------------------------
    bands = kept_band_energies(parent, excluded, num_bands)
    froz_max = _number(values.get("dis_froz_max"))
    froz_min = _number(values.get("dis_froz_min"))
    frozen = frozen_counts(bands, froz_min, froz_max)
    if method is Method.PDWF and projectability is not None:
        threshold = _number(values.get("dis_proj_max"))
        if threshold is not None and projectability.shape[-1] == bands.shape[-1]:
            by_p = projectability >= threshold
            energy = (bands[0] <= (froz_max if froz_max is not None else -np.inf)) & (
                bands[0] >= (froz_min if froz_min is not None else -np.inf)
            )
            frozen = np.array([(by_p | energy).sum(axis=-1)])
            out.notes.append("frozen states include those frozen by projectability")
    if method is Method.PDWF and projectability is None:
        out.notes.append(
            "states frozen by projectability are not counted (no overlaps on hand); "
            "the energy-window count is a lower bound"
        )
    if num_wann is not None and froz_max is not None and frozen.size:
        worst = int(frozen.max())
        out.facts["frozen_states_max"] = worst
        if worst > num_wann:
            spin, k = np.unravel_index(int(frozen.argmax()), frozen.shape)
            out.problems.append(
                f"the frozen window holds {worst} states at k-point {k + 1}"
                + (f", spin {spin + 1}" if frozen.shape[0] > 1 else "")
                + f", more than num_wann = {num_wann}; Wannier90 refuses that deck"
            )
    win_min, win_max = _number(values.get("dis_win_min")), _number(
        values.get("dis_win_max")
    )
    if (
        num_wann is not None
        and (win_min is not None or win_max is not None)
        and bands.size
    ):
        lo = win_min if win_min is not None else -np.inf
        hi = win_max if win_max is not None else np.inf
        pool = ((bands >= lo) & (bands <= hi)).sum(axis=-1)
        out.facts["outer_window_states_min"] = int(pool.min())
        if int(pool.min()) < num_wann:
            out.problems.append(
                f"the outer window holds only {int(pool.min())} states at some k-point, "
                f"fewer than num_wann = {num_wann}"
            )
    if froz_max is not None and win_max is not None and froz_max > win_max:
        out.problems.append("the frozen window reaches above the outer window")

    # -- projectability ---------------------------------------------------
    if method is Method.PDWF and projectability is not None and projectability.size:
        top = float(projectability.max())
        out.facts["projectability_max"] = round(top, 4)
        if top > 1.0 + PROJECTABILITY_TOLERANCE:
            out.problems.append(
                f"projectability reaches {top:.3f} > 1: the projectors are not "
                f"orthonormal, so the PDWF thresholds would freeze and drop the wrong "
                f"states"
            )
    return out


def _check_exclusion(
    out: DeckCheck,
    parent,
    excluded,
    spin_index: int,
    whole_occupied_target: bool = True,
) -> None:
    """State the gate's verdict on the deck's exclusion, and refuse one it would not honour.

    A deck whose excluded block holds occupied states the gate will not set aside is
    scored against every band: the excluded occupied states have no model band near
    them, ``occupied_manifold_is_represented`` fails by construction, and no edit of any
    other knob changes that. Refusing it here costs no overlap step.
    """
    from .semicore import bottom_and_top, exclusion_verdict

    if not excluded:
        return
    channel = spin_index if spin_index < parent.eigenvalues_ev.shape[0] else 0
    verdict, tolerance = exclusion_verdict(parent, excluded, channel)
    bottom, top = bottom_and_top(excluded)
    out.facts["exclusion"] = {
        "bottom_bands": bottom,
        "top_bands": len(top),
        "gate_verdict": verdict.status,
        "gap_ev": None if verdict.gap_ev is None else round(verdict.gap_ev, 3),
        "tolerance_ev": None if tolerance is None else round(tolerance, 4),
    }
    if verdict.status == "not_honoured" and tolerance is None:
        # A parent with no occupied bandwidth to scale by (flat model bands) gives the
        # gate no tolerance: it honours nothing occupied, and says so in its own note.
        out.notes.append(verdict.note.lstrip("; "))
    elif verdict.status == "not_honoured" and not whole_occupied_target:
        # The refusal below exists because occupied_manifold_is_represented would count
        # the excluded occupied states as missing; a sub-manifold profile does not run it.
        out.notes.append(
            verdict.note.lstrip("; ")
            + " (not refused: this gate profile targets a sub-manifold and runs no "
            "whole-occupied-manifold check)"
        )
    elif verdict.status == "not_honoured":
        reason = verdict.note.lstrip("; ")
        out.problems.append(
            f"{reason}: the gate would score this construction against every band and "
            f"count each excluded occupied state as missing "
            f"(occupied_manifold_is_represented), so it cannot pass whatever else "
            f"changes. Move the cut to an isolated gap with semicore_exclusion "
            f"(validate_deck lists the shells and which cuts the gate honours)"
        )


def _check_semicore_shells(out: DeckCheck, win_text, pw_text, deck_dir, ctx) -> None:
    """Where the deck's semicore shells are known, its orbitals must agree with its cut."""
    from .semicore import SemicoreUnavailable, catalogue, read_record

    try:
        cat = catalogue(win_text, pw_text, Path(deck_dir), ctx)
    except (SemicoreUnavailable, InputError) as exc:
        out.notes.append(f"semicore shells not checked: {exc}")
        return
    out.facts["semicore_excluded"] = cat.current_value() if cat.known else None
    if not cat.known:
        if read_record(win_text) is not None:
            out.problems.append(
                "the deck's semicore record and its bottom exclusion disagree: "
                + "; ".join(cat.notes)
            )
        else:
            out.notes.extend(cat.notes)
        return
    excluded = [s for s in cat.shells if s.excluded]
    kept = [s for s in cat.shells if not s.excluded]
    above = [
        (x, y)
        for x in excluded
        for y in kept
        if x.atomic_energy_ev is not None
        and y.atomic_energy_ev is not None
        and x.atomic_energy_ev > y.atomic_energy_ev
    ]
    if above:
        x, y = max(above, key=lambda p: p[0].atomic_energy_ev - p[1].atomic_energy_ev)
        out.notes.append(
            f"the deck excludes {x.describe()} but keeps {y.describe()}, which lies "
            f"{x.atomic_energy_ev - y.atomic_energy_ev:.1f} eV deeper in the pseudo-atom; "
            "the excluded bands are the lowest by index, so the cut assumes they are "
            "the excluded shells' states (band_character shows their character where "
            "the parent supplies one)"
        )
    if cat.code == "vasp" and cat.bottom:
        _check_procar_character(out, cat, ctx)
    if cat.route == "pao" and pw_text is not None:
        dropped = pao_excluded(pw_text)
        kept = [
            s.describe()
            for s in cat.shells
            if s.excluded and not set(s.states) <= dropped
        ]
        if kept:
            out.problems.append(
                f"the deck keeps the pseudo-atomic orbitals of {', '.join(kept)}, whose "
                f"bands its bottom exclusion removes: functions with no bands to "
                f"describe. Bring the shell in with semicore_exclusion instead"
            )


def _check_procar_character(out: DeckCheck, cat, ctx) -> None:
    """Every excluded band must be dominated by an excluded shell's (species, l)."""
    from .semicore import procar_dominant

    procar = Path(ctx.parent_source).with_name("PROCAR")
    if not procar.is_file():
        out.notes.append(
            "excluded bands' character not checked: no PROCAR beside the parent"
        )
        return
    wanted = {(s.species, s.l) for s in cat.shells if s.excluded}
    dominant = procar_dominant(
        procar, ctx.parent.species, range(1, cat.bottom + 1), ctx.spin_index
    )
    wrong = [
        f"band {b} ({species} {'spdf'[l]}, {fraction:.2f})"
        for b, (species, l, fraction) in sorted(dominant.items())
        if (species, l) not in wanted
    ]
    if wrong:
        out.problems.append(
            "the excluded bands are not the excluded shells' states: "
            + "; ".join(wrong[:4])
            + " (PROCAR, k-averaged); exclude the shells those bands belong to"
        )
