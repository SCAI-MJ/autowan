"""The bottom (semicore) exclusion: which shells a construction leaves out, and what follows.

A deck's ``exclude_bands`` removes the lowest N parent bands by index before Wannier90
reads anything. Which *shells* those bands are decides which orbitals belong in the
projector set, so the knob that moves the cut (``semicore_exclusion``) names shells, per
species, and everything else follows from that list:

* N is the bands the excluded shells hold: 2l+1 per atom, or one band per electron for
  spinor bands (2j+1 for a j-resolved pseudo-atomic orbital);
* ``exclude_bands`` is 1..N plus the deck's top (headroom) block, unchanged, and
  ``num_bands`` is what is left;
* PDWF and QE SMV on pseudo-atomic orbitals (PAOs) drop or keep those shells' PAOs
  (``atom_proj_exclude``), and ``num_wann`` is the PAOs kept;
* analytic SMV adds one projector per semicore shell brought in (a second function of
  an angular momentum the species already carries is Wannier90's ``r=2``) and removes
  the ones it added when the shell goes back out;
* SCDM has no orbitals: ``num_wann`` moves by the bands brought in or taken out.

Why shells and not a band count: when two shells overlap in energy (an anion s band in
a cation's semicore p block) "the lowest N bands" names no orbital set, and an
energy-order rule picks the wrong one exactly where the knob is needed. The shell list
fixes the orbitals; the cut it implies is then judged by the gate's own honour rule
(:func:`.gate.judge_exclusion`): the excluded block must lie, over the whole mesh, more
than 1 % of the occupied bandwidth below every band kept. A cut through a connected
manifold is refused by the deck check before anything runs.

Shell catalogue per engine:

* Quantum ESPRESSO: the PAO list of the deck (or of the run's first deck, for an SCDM or
  analytic-SMV deck), each orbital named by its label (``s1``, ``p1``... or the label
  table's ``4s``) and, when the parent's pseudopotential files are beside it, by the
  pseudo-atom's own shell name (``4S``) with its occupation and energy. The candidates
  are the pseudo-atom's closed shells, plus whatever the deck already excludes.
* VASP (SMV): the semicore shells of our POTCARs (ZVAL minus the checked valence count,
  deepest shells first; :func:`.backends.vasp.inputs.semicore_shells`).

Which shells a deck excludes now is read from its ``! semicore_exclusion:`` record when
this knob wrote it, and otherwise inferred: the excluded closed-shell PAOs (QE) or all
or none of the POTCAR semicore shells (VASP), when their band count equals the bottom
block. When it cannot be told, every species has to be named.
"""

from __future__ import annotations

import functools
import itertools
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .families import Method, knob_unavailable
from .inputs import (
    InputError,
    _excluded_from_text,
    assignments,
    deck_atom_species,
    deck_method,
    deck_spinors,
    pao_excluded,
    pao_layout,
)

KNOB = "semicore_exclusion"
RECORD = "! semicore_exclusion:"
PROJECTION_RECORD = "! semicore_projections:"
RY_TO_EV = 13.605693122994
#: Wannier90 (and VASP's overlap step) build hydrogenic radial functions r = 1..3.
MAX_RADIAL = 3
#: Most catalogue shells whose subsets the cut table enumerates.
MAX_SHELLS_ENUMERATED = 12


class SemicoreUnavailable(InputError):
    """This deck has no semicore exclusion route, or no shell list to name one by."""


@dataclass(frozen=True)
class Shell:
    """One semicore (or closed) shell of one species, as the knob names it."""

    species: str
    label: str
    """What the knob takes: the PAO label (QE) or the POTCAR shell (VASP)."""
    name: str | None
    """The pseudo-atom's own shell name (``4s``), when known."""
    l: int
    j: float | None
    bands_per_atom: int
    atoms: int
    occupation: float | None
    """Electrons per atom in the pseudo-atom's reference configuration."""
    atomic_energy_ev: float | None
    states: tuple[int, ...]
    """pw2wannier90's PAO state indices over every atom (QE), for atom_proj_exclude."""
    excluded: bool

    @property
    def bands(self) -> int:
        return self.bands_per_atom * self.atoms

    @property
    def letter(self) -> str:
        return "spdf"[self.l]

    def aliases(self) -> set[str]:
        out = {self.label.lower()}
        if self.name:
            out.add(self.name.lower())
        return out

    def describe(self) -> str:
        bits = [f"{self.species} {self.label}"]
        if self.name and self.name.lower() != self.label.lower():
            bits.append(f"({self.name})")
        return " ".join(bits)

    def to_dict(self) -> dict:
        out = {
            "species": self.species,
            "shell": self.label,
            "l": self.l,
            "bands": self.bands,
            "bands_per_atom": self.bands_per_atom,
            "excluded_now": self.excluded,
        }
        if self.name:
            out["name"] = self.name
        if self.j is not None:
            out["j"] = self.j
        if self.occupation is not None:
            out["occupation"] = self.occupation
        if self.atomic_energy_ev is not None:
            out["pseudo_atom_energy_ev"] = round(self.atomic_energy_ev, 3)
        return out


@dataclass(frozen=True)
class Catalogue:
    """The semicore shells a deck can exclude, and its current band assignment."""

    code: str
    route: str
    """``pao`` (atom_proj_exclude), ``projections`` (analytic), or ``scdm``."""
    shells: tuple[Shell, ...]
    bottom: int
    """The deck's bottom exclusion: bands 1..bottom."""
    top: tuple[int, ...]
    """Excluded bands above the bottom block (headroom), unchanged by the knob."""
    total_bands: int
    known: bool
    """Whether the shells the bottom block removes were identified."""
    source: str
    recorded_projections: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def by_species(self) -> dict[str, list[Shell]]:
        out: dict[str, list[Shell]] = {}
        for shell in self.shells:
            out.setdefault(shell.species, []).append(shell)
        return out

    def current_value(self) -> list[dict] | None:
        if not self.known:
            return None
        return [
            {"species": s, "shells": [sh.label for sh in shells if sh.excluded]}
            for s, shells in self.by_species().items()
        ]

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "shells": [s.to_dict() for s in self.shells],
            "bottom_exclusion_bands": self.bottom,
            "current": (
                self.current_value()
                if self.known
                else "not identified: name every species"
            ),
            "shells_read_from": self.source,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# reading records and band assignments
# ---------------------------------------------------------------------------


def bottom_and_top(excluded: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """The contiguous block from band 1, and the rest."""
    ordered = sorted(set(excluded))
    bottom = 0
    while bottom < len(ordered) and ordered[bottom] == bottom + 1:
        bottom += 1
    return bottom, tuple(ordered[bottom:])


def read_record(win_text: str) -> dict[str, tuple[str, ...]] | None:
    """The ``! semicore_exclusion:`` record this knob writes, or ``None``."""
    for line in win_text.splitlines():
        if line.strip().startswith(RECORD):
            body = line.strip()[len(RECORD) :]
            out = {}
            for part in body.split(";"):
                species, _, shells = part.partition(":")
                species = species.strip()
                if not species:
                    continue
                labels = [x for x in re.split(r"[,\s]+", shells.strip()) if x]
                out[species] = tuple(x for x in labels if x.lower() != "none")
            return out
    return None


def read_projection_record(win_text: str) -> tuple[str, ...]:
    for line in win_text.splitlines():
        if line.strip().startswith(PROJECTION_RECORD):
            body = line.strip()[len(PROJECTION_RECORD) :]
            return tuple(x.strip() for x in body.split("|") if x.strip())
    return ()


def record_lines(
    excluded: Mapping[str, Sequence[str]], projections: Sequence[str] = ()
) -> list[str]:
    parts = [
        f"{species}: {', '.join(labels) if labels else 'none'}"
        for species, labels in excluded.items()
    ]
    lines = [f"{RECORD} " + "; ".join(parts)]
    if projections:
        lines.append(f"{PROJECTION_RECORD} " + " | ".join(projections))
    return lines


def with_record(text: str, lines: Sequence[str]) -> str:
    """``text`` with its semicore records replaced by ``lines`` (after the header line)."""
    kept = [
        line
        for line in text.splitlines()
        if not line.strip().startswith((RECORD, PROJECTION_RECORD))
    ]
    insert_at = 1 if kept and kept[0].startswith("! derived") else 0
    return "\n".join(kept[:insert_at] + list(lines) + kept[insert_at:]) + "\n"


# ---------------------------------------------------------------------------
# the pseudo-atom's shells (Quantum ESPRESSO)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpfOrbital:
    name: str
    l: int
    occupation: float
    energy_ev: float | None
    j: float | None = None


def _attrs(text: str) -> dict[str, str]:
    return {k.lower(): v for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', text)}


def _float(value) -> float | None:
    try:
        return float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return None


def read_upf_orbitals(path: Path) -> list[UpfOrbital]:
    """The pseudo-atomic orbitals a UPF file carries, in file order (v2 or v1)."""
    text = Path(path).read_text(errors="replace")
    out: list[UpfOrbital] = []
    relwfc = {
        int(a.get("index", 0)): _float(a.get("jchi"))
        for a in (_attrs(m) for m in re.findall(r"<PP_RELWFC\.\d+([^>]*)>", text))
    }
    for i, raw in enumerate(re.findall(r"<PP_CHI\.\d+([^>]*)>", text), start=1):
        a = _attrs(raw)
        l, occ = a.get("l"), _float(a.get("occupation"))
        if l is None or occ is None:
            continue
        energy = _float(a.get("pseudo_energy"))
        j = _float(a.get("jchi")) if "jchi" in a else relwfc.get(i)
        out.append(
            UpfOrbital(
                a.get("label", f"{'SPDF'[int(l)]}{i}"),
                int(l),
                occ,
                None if energy is None or energy == 0 else energy * RY_TO_EV,
                j,
            )
        )
    if out:
        return out
    # UPF v1: the header's "Wavefunctions nl l occ" table, energies from PP_INFO.
    energies = {}
    for m in re.finditer(
        r"(?m)^\s*(\d[SPDFspdf])\s+\d+\s+(\d)\s+([\d.]+)\s+\S+\s+\S+\s+(\S+)\s*$", text
    ):
        energies[m.group(1).upper()] = _float(m.group(4))
    header = text.split("<PP_HEADER>", 1)[-1].split("</PP_HEADER>", 1)[0]
    rows = header.split("Wavefunctions", 1)
    if len(rows) == 2:
        for line in rows[1].splitlines()[1:]:
            m = re.match(r"\s*(\d[SPDFspdf])\s+(\d)\s+([\d.]+)\s*$", line)
            if not m:
                if out:
                    break
                continue
            name = m.group(1).upper()
            e = energies.get(name)
            out.append(
                UpfOrbital(
                    name,
                    int(m.group(2)),
                    float(m.group(3)),
                    None if e is None else e * RY_TO_EV,
                )
            )
    return out


def parent_pseudopotentials(parent_source: Path) -> dict[str, Path]:
    """Species -> the UPF file beside a QE parent's ``data-file-schema.xml``."""
    source = Path(parent_source)
    if not source.is_file():
        return {}
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError:
        return {}
    out = {}
    for species in root.iter("species"):
        name = species.get("name")
        pseudo = species.findtext("pseudo_file")
        if name and pseudo and (source.parent / pseudo.strip()).is_file():
            out[name] = source.parent / pseudo.strip()
    return out


def _closed(occupation: float | None, l: int, j: float | None) -> bool | None:
    if occupation is None:
        return None
    full = (2 * j + 1) if j is not None else 2 * (2 * l + 1)
    return occupation >= full - 1e-6


# ---------------------------------------------------------------------------
# catalogues
# ---------------------------------------------------------------------------


def catalogue(
    win_text: str,
    pw_text: str | None,
    deck_dir: Path,
    ctx,
) -> Catalogue:
    """The shells this deck can exclude at the bottom, and what it excludes now."""
    code = ctx.code
    method = deck_method(win_text, pw_text)
    unavailable = knob_unavailable(code, method, KNOB)
    if unavailable:
        raise SemicoreUnavailable(unavailable)
    bottom, top = bottom_and_top(_excluded_from_text(win_text))
    total = int(ctx.parent.n_bands)
    if code == "vasp":
        return _vasp_catalogue(win_text, ctx, bottom, top, total)
    return _qe_catalogue(
        win_text, pw_text, Path(deck_dir), ctx, method, bottom, top, total
    )


def _is_pao_deck(pw_text: str | None) -> bool:
    return pw_text is not None and assignments(pw_text).get(
        "atom_proj", ""
    ).lower().startswith((".t", "t"))


def _qe_catalogue(win_text, pw_text, deck_dir, ctx, method, bottom, top, total):
    if _is_pao_deck(pw_text):
        route = "pao"
        layout_win, layout_pw, layout_dir = win_text, pw_text, deck_dir
    else:
        route = "scdm" if method is Method.SCDM else "projections"
        frontier = ctx.frontier
        if frontier.pw2wan is None or not _is_pao_deck(frontier.pw2wan.read_text()):
            raise SemicoreUnavailable(
                "this deck names no pseudo-atomic orbitals and neither does the run's "
                "first deck, so there is no shell list to name a semicore exclusion by"
            )
        layout_win = frontier.win.read_text()
        layout_pw = frontier.pw2wan.read_text()
        layout_dir = frontier.win.parent
    try:
        layout = pao_layout(layout_dir, layout_win, layout_pw)
    except InputError as exc:
        raise SemicoreUnavailable(
            f"the pseudo-atomic orbital list is unreadable: {exc}"
        )
    pseudos = parent_pseudopotentials(ctx.parent_source)
    upf = {}
    for species, path in pseudos.items():
        try:
            upf[species] = read_upf_orbitals(path)
        except OSError:
            continue

    atoms = deck_atom_species(layout_win)
    counts: dict[str, int] = {}
    for symbol in atoms:
        counts[symbol] = counts.get(symbol, 0) + 1
    # Orbitals per species, in file order, with every atom's states.
    orbitals: dict[str, list[dict]] = {}
    first_atom: dict[str, int] = {}
    for state in layout:
        first_atom.setdefault(state.species, state.atom)
        entries = orbitals.setdefault(state.species, [])
        entry = next((e for e in entries if e["label"] == state.label), None)
        if entry is None:
            entry = {
                "label": state.label,
                "l": state.l,
                "j": state.j,
                "states": [],
                "per_atom": 0,
            }
            entries.append(entry)
        entry["states"].append(state.index)
        if state.atom == first_atom[state.species]:
            entry["per_atom"] += 1
    # Name each orbital by the pseudo-atom's shell, matched by (l, j) and rank.
    notes = []
    for species, entries in orbitals.items():
        chi = upf.get(species)
        if not chi:
            continue
        groups: dict[tuple, list] = {}
        for e in entries:
            groups.setdefault((e["l"], e["j"]), []).append(e)
        upf_groups: dict[tuple, list] = {}
        for o in chi:
            upf_groups.setdefault(
                (o.l, o.j if any(e["j"] is not None for e in entries) else None), []
            ).append(o)
        for key, members in groups.items():
            found = upf_groups.get(key, [])
            if len(found) != len(members):
                notes.append(
                    f"{species}: the projector list and {pseudos[species].name} differ "
                    f"in their l={key[0]} orbitals; shells are named by label only"
                )
                continue
            for e, o in zip(members, found):
                e["upf"] = o

    pw_excluded = pao_excluded(pw_text) if route == "pao" else (pao_excluded(layout_pw))
    record = read_record(win_text)

    def closed(e) -> bool | None:
        o = e.get("upf")
        return None if o is None else _closed(o.occupation, o.l, o.j)

    # What the bottom block removes now.
    known, current = False, {}
    if record is not None:
        current = {s: set(v) for s, v in record.items()}
        known = True
    else:
        source_bottom = bottom
        if route != "pao":
            fb, _ = bottom_and_top(_excluded_from_text(layout_win))
            source_bottom = fb
        guess = {}
        for species, entries in orbitals.items():
            guess[species] = {
                e["label"]
                for e in entries
                if set(e["states"]) <= pw_excluded and closed(e) is not False
            }
        guessed_bands = sum(
            e["per_atom"] * counts.get(s, 0)
            for s, entries in orbitals.items()
            for e in entries
            if e["label"] in guess[s]
        )
        if guessed_bands == bottom and source_bottom == bottom:
            known, current = True, guess
        elif bottom == 0:
            known, current = True, {s: set() for s in orbitals}
        else:
            notes.append(
                f"the bottom {bottom} excluded bands do not match the excluded "
                f"closed-shell orbitals ({guessed_bands} bands); which shells they are "
                "cannot be told from this deck"
            )

    def below_valence(e, entries) -> bool:
        """A closed shell is a semicore candidate only below every open shell.

        A filled valence s shell above a partly filled d shell (a d8 s2 reference) is
        closed but not semicore; where the pseudo-atom's energies are unknown, every
        closed shell is listed and the cut table judges it.
        """
        own = e.get("upf")
        open_energies = [
            x["upf"].energy_ev
            for x in entries
            if x.get("upf") is not None
            and closed(x) is False
            and x["upf"].occupation > 0
            and x["upf"].energy_ev is not None
        ]
        if own is None or own.energy_ev is None or not open_energies:
            return True
        return own.energy_ev < min(open_energies)

    shells = []
    for species, entries in orbitals.items():
        for e in entries:
            is_current = e["label"] in current.get(species, set())
            if not ((closed(e) and below_valence(e, entries)) or is_current):
                continue
            o = e.get("upf")
            shells.append(
                Shell(
                    species=species,
                    label=e["label"],
                    name=(o.name.lower() if o is not None else None),
                    l=e["l"],
                    j=e["j"],
                    bands_per_atom=e["per_atom"],
                    atoms=counts.get(species, 0),
                    occupation=None if o is None else o.occupation,
                    atomic_energy_ev=None if o is None else o.energy_ev,
                    states=tuple(e["states"]),
                    excluded=is_current,
                )
            )
    # Deepest first where the pseudo-atoms say; otherwise the projector list's order.
    if all(s.atomic_energy_ev is not None for s in shells):
        shells.sort(key=lambda s: s.atomic_energy_ev)
    source = "pseudo-atomic orbital list" + (
        f" named by the parent's pseudopotentials ({', '.join(p.name for p in pseudos.values())})"
        if upf
        else " (no pseudopotential file beside the parent: closed shells unknown, "
        "only the excluded ones are listed)"
    )
    if known:
        # A record that disagrees with the band block is a deck nobody should trust.
        recorded_bands = sum(s.bands for s in shells if s.excluded)
        if recorded_bands != bottom:
            known = False
            notes.append(
                f"the deck's semicore record holds {recorded_bands} bands but the "
                f"bottom exclusion is {bottom}"
            )
    return Catalogue(
        code="qe",
        route=route,
        shells=tuple(shells),
        bottom=bottom,
        top=top,
        total_bands=total,
        known=known,
        source=source,
        recorded_projections=read_projection_record(win_text),
        notes=tuple(notes),
    )


def _vasp_catalogue(win_text, ctx, bottom, top, total):
    from collections import Counter

    from .backends.vasp.inputs import VALENCE_TABLE, _potcar_path, semicore_shells
    from .backends.vasp.outputs import VaspOutputError

    try:
        potcar = _potcar_path(Path(ctx.parent_source), None)
        zval, table = _potcar_tables(*_stat_key(potcar))
    except (InputError, VaspOutputError, OSError) as exc:
        raise SemicoreUnavailable(f"the POTCAR shells are unreadable: {exc}")
    spinor = bool(getattr(ctx.parent, "spinor", False))
    per_band = 1 if spinor else 2
    counts = Counter(ctx.parent.species)
    record = read_record(win_text)
    shells = []
    for symbol in dict.fromkeys(ctx.parent.species):
        if symbol not in VALENCE_TABLE or symbol not in zval or symbol not in table:
            raise SemicoreUnavailable(
                f"{symbol}: no checked valence row, so its semicore shells are not known"
            )
        semicore_e = zval[symbol] - VALENCE_TABLE[symbol][0]
        for label in semicore_shells(symbol, semicore_e, table[symbol]):
            l, occ, energy = table[symbol][label]
            shells.append(
                dict(
                    species=symbol,
                    label=label,
                    l=l,
                    occ=occ,
                    energy=energy,
                    per_atom=int(round(occ)) // per_band,
                    atoms=counts[symbol],
                )
            )
    all_bands = sum(s["per_atom"] * s["atoms"] for s in shells)
    notes = []
    if record is not None:
        current = {s: set(v) for s, v in record.items()}
        known = True
    elif bottom == all_bands:
        current = {s["species"]: set() for s in shells}
        for s in shells:
            current[s["species"]].add(s["label"])
        known = True
    elif bottom == 0:
        current, known = {}, True
    else:
        current, known = {}, False
        notes.append(
            f"the bottom {bottom} excluded bands are neither none nor all of the "
            f"POTCAR semicore shells ({all_bands} bands); which they are cannot be told"
        )
    out = tuple(
        Shell(
            species=s["species"],
            label=s["label"],
            name=s["label"],
            l=s["l"],
            j=None,
            bands_per_atom=s["per_atom"],
            atoms=s["atoms"],
            occupation=s["occ"],
            atomic_energy_ev=s["energy"],
            states=(),
            excluded=s["label"] in current.get(s["species"], set()),
        )
        for s in sorted(shells, key=lambda s: s["energy"])
    )
    if known and sum(s.bands for s in out if s.excluded) != bottom:
        known = False
        notes.append("the deck's semicore record disagrees with its bottom exclusion")
    return Catalogue(
        code="vasp",
        route="projections",
        shells=out,
        bottom=bottom,
        top=top,
        total_bands=total,
        known=known,
        source=f"semicore shells of {Path(potcar).name} (ZVAL minus the checked valence)",
        recorded_projections=read_projection_record(win_text),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# the plan an edit writes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemicorePlan:
    """A resolved ``semicore_exclusion`` edit: what the writer puts in the deck."""

    route: str
    excluded: dict[str, tuple[str, ...]]
    bottom: int
    exclude_bands: tuple[int, ...]
    num_bands: int
    bands_in: int
    """Bands brought into the pool (negative: taken out)."""
    pao_in: tuple[int, ...] = ()
    pao_out: tuple[int, ...] = ()
    pao_excluded_shells: tuple[int, ...] = ()
    """States of every shell the edit leaves excluded (for agreement with pao_subset)."""
    semicore_projectors: tuple[tuple[str, str], ...] = ()
    """(species, letter) of every semicore shell the deck now includes (analytic)."""
    newly_excluded: tuple[str, ...] = ()
    newly_included: tuple[str, ...] = ()
    recorded_projections: tuple[str, ...] = ()
    newly_excluded_unrecorded: tuple[str, ...] = ()
    """Analytic decks: shells going out whose projector this knob did not write."""

    def summary(self) -> str:
        moved = []
        if self.newly_included:
            moved.append("brings in " + ", ".join(self.newly_included))
        if self.newly_excluded:
            moved.append("excludes " + ", ".join(self.newly_excluded))
        return (
            f"semicore_exclusion {'; '.join(moved) or 'changes nothing'}: "
            f"exclude_bands bottom 1..{self.bottom} ({self.bottom} bands)"
        )


def resolve(cat: Catalogue, value: Sequence[Mapping[str, Any]]) -> SemicorePlan:
    """Turn the model's per-species shell lists into the deck edit they imply."""
    by_species = cat.by_species()
    requested: dict[str, set[str]] = {}
    for entry in value:
        species = entry["species"]
        labels = list(entry.get("shells") or ())
        shells = by_species.get(species, [])
        if labels and not shells:
            raise InputError(
                f"{species} has no semicore shell this deck can exclude "
                f"(catalogue: {_listing(cat)})"
            )
        chosen = set()
        for label in labels:
            match = [s for s in shells if label.lower() in s.aliases()]
            if not match:
                raise InputError(
                    f"{species} has no semicore shell {label!r}; its shells are "
                    + ", ".join(s.describe() for s in shells)
                )
            chosen.add(match[0].label)
        requested[species] = chosen
    if not cat.known:
        missing = [s for s in by_species if s not in requested]
        if missing:
            raise InputError(
                "which shells this deck's bottom exclusion removes cannot be told, so "
                f"every species with semicore shells must be named; missing: "
                f"{', '.join(missing)}"
            )
    new = {}
    for species, shells in by_species.items():
        if species in requested:
            new[species] = requested[species]
        else:
            new[species] = {s.label for s in shells if s.excluded}
    excluded_shells = [s for s in cat.shells if s.label in new[s.species]]
    bottom = sum(s.bands for s in excluded_shells)
    if cat.top and min(cat.top) <= bottom:
        raise InputError(
            f"excluding {bottom} bands at the bottom reaches the deck's top exclusion "
            f"(from band {min(cat.top)})"
        )
    num_bands = cat.total_bands - bottom - len(cat.top)
    if num_bands < 1:
        raise InputError(f"excluding {bottom} bands leaves no band to Wannierise")
    newly_in = [s for s in cat.shells if s.excluded and s.label not in new[s.species]]
    newly_out = [s for s in cat.shells if not s.excluded and s.label in new[s.species]]
    return SemicorePlan(
        route=cat.route,
        excluded={s: tuple(sorted(v)) for s, v in new.items()},
        bottom=bottom,
        exclude_bands=tuple(range(1, bottom + 1)) + tuple(cat.top),
        num_bands=num_bands,
        bands_in=cat.bottom - bottom,
        pao_in=tuple(i for s in newly_in for i in s.states),
        pao_out=tuple(i for s in newly_out for i in s.states),
        pao_excluded_shells=tuple(i for s in excluded_shells for i in s.states),
        semicore_projectors=tuple(
            (s.species, s.letter)
            for s in cat.shells
            if s.label not in new[s.species] and _needs_projector(cat, s)
        ),
        newly_excluded=tuple(s.describe() for s in newly_out),
        newly_included=tuple(s.describe() for s in newly_in),
        recorded_projections=cat.recorded_projections,
        newly_excluded_unrecorded=tuple(
            s.describe()
            for s in newly_out
            if cat.route == "projections" and not _recorded(cat, s)
        ),
    )


def _recorded(cat: Catalogue, shell: Shell) -> bool:
    """Whether this knob wrote a projector line of this shell's (species, l)."""
    for line in cat.recorded_projections:
        parts = line.replace(" ", "").split(":")
        if (
            len(parts) >= 2
            and parts[0] == shell.species
            and shell.letter in parts[1].split(";")
        ):
            return True
    return False


def _needs_projector(cat: Catalogue, shell: Shell) -> bool:
    """On an analytic deck, a semicore shell carries a projector while it is included.

    VASP's catalogue holds semicore shells only, and the SMV deck names valence channels
    only, so every included one needs its own. On QE the catalogue also lists closed
    shells the run never excluded (an anion's s shell); those keep whatever the named
    projector set gives them, and only a shell this knob brings in gets a line.
    """
    if cat.code == "vasp":
        return True
    return shell.excluded or _recorded(cat, shell)


def _listing(cat: Catalogue) -> str:
    return ", ".join(s.describe() for s in cat.shells) or "empty"


def semicore_projection_lines(
    valence_lines: Sequence[str], projectors: Sequence[tuple[str, str]]
) -> list[str]:
    """One line per included semicore shell; a repeated (species, l) takes the next r."""
    used: dict[tuple[str, str], int] = {}
    for line in valence_lines:
        parts = line.replace(" ", "").split(":")
        if len(parts) < 2:
            continue
        radial = next(
            (int(p[2:]) for p in parts[2:] if re.fullmatch(r"r=\d+", p.lower())), 1
        )
        for orbital in parts[1].split(";"):
            key = (parts[0], orbital.lower())
            used[key] = max(used.get(key, 0), radial)
    out = []
    for species, letter in projectors:
        r = used.get((species, letter), 0) + 1
        if r > MAX_RADIAL:
            raise InputError(
                f"{species} already carries {r - 1} {letter} functions; a further one "
                f"needs radial r={r}, and Wannier90 builds r = 1..{MAX_RADIAL} only"
            )
        used[(species, letter)] = r
        out.append(f"{species}:{letter}" + (f":r={r}" if r > 1 else ""))
    return out


# ---------------------------------------------------------------------------
# what validate_deck and band_character show
# ---------------------------------------------------------------------------


def gate_tolerance(parent) -> float | None:
    from .gate import NonEvaluable, exclusion_tolerance_ev

    try:
        return exclusion_tolerance_ev(parent.eigenvalues_ev, parent.fermi_energy_ev)
    except NonEvaluable:
        return None


def exclusion_verdict(parent, excluded: Sequence[int], spin_index: int = 0):
    """The gate's honour verdict on a deck's exclusion (one channel), as data.

    Returns ``(verdict, tolerance_ev)``. With no occupied-bandwidth scale the gate
    honours nothing that removes an occupied state, and neither does this.
    """
    from .gate import ExclusionVerdict, judge_exclusion

    bands = np.asarray(parent.eigenvalues_ev[spin_index])
    tolerance = gate_tolerance(parent)
    if tolerance is None:
        valid = [i - 1 for i in excluded if 1 <= i <= bands.shape[1]]
        if excluded and np.any(bands[:, valid] <= parent.fermi_energy_ev):
            return (
                ExclusionVerdict(
                    "not_honoured",
                    "; the occupied bandwidth that sets the gate's tolerance is not "
                    "defined for this parent, so no occupied exclusion is honoured",
                ),
                None,
            )
        return ExclusionVerdict("irrelevant" if excluded else "none"), None
    return (
        judge_exclusion(bands, excluded, parent.fermi_energy_ev, tolerance),
        tolerance,
    )


def cut_table(parent, cat: Catalogue, spin_index: int = 0) -> list[dict[str, Any]]:
    """Every bottom cut a set of catalogue shells implies, judged by the gate's rule."""
    sums: dict[int, list[dict]] = {}
    shells = list(cat.shells)[:MAX_SHELLS_ENUMERATED]
    energies = [s.atomic_energy_ev for s in shells]
    ordered = (
        sorted(shells, key=lambda s: s.atomic_energy_ev)
        if all(e is not None for e in energies)
        else None
    )
    for r in range(len(shells) + 1):
        for combo in itertools.combinations(shells, r):
            n = sum(s.bands for s in combo)
            entry: dict[str, Any] = {"shells": [s.describe() for s in combo]}
            if ordered is not None:
                # The deepest shells of the pseudo-atoms, taken in energy order. Advice,
                # not a rule: in a crystal two shells a few eV apart can overlap.
                entry["deepest_by_pseudo_atom_energy"] = set(combo) == set(ordered[:r])
            sums.setdefault(n, []).append(entry)
    for n in sums:
        sums[n].sort(key=lambda e: not e.get("deepest_by_pseudo_atom_energy", False))
    sums.setdefault(cat.bottom, sums.get(cat.bottom, []))
    rows = []
    bands = np.asarray(parent.eigenvalues_ev[spin_index])
    ef = parent.fermi_energy_ev
    for n in sorted(sums):
        if n >= bands.shape[1]:
            continue
        verdict, tol = exclusion_verdict(
            parent, tuple(range(1, n + 1)) + tuple(cat.top), spin_index
        )
        gap = float(bands[:, n].min() - bands[:, n - 1].max()) if n > 0 else None
        rows.append(
            {
                "exclude_lowest": n,
                "gap_above_ev": None if gap is None else round(gap, 3),
                "honoured_by_gate": (
                    verdict.status in ("none", "irrelevant", "honoured")
                    if n > 0
                    else True
                ),
                "current": n == cat.bottom,
                "shell_sets": sums[n][:6],
            }
        )
    return rows


def band_rows(
    parent,
    first: int,
    last: int,
    excluded: Sequence[int] = (),
    spin_index: int = 0,
    characters: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Per parent band (1-based, inclusive range): its energy span relative to E_F."""
    bands = np.asarray(parent.eigenvalues_ev[spin_index])
    ef = parent.fermi_energy_ev
    rows = []
    excluded = set(excluded)
    for b in range(max(1, first), min(last, bands.shape[1]) + 1):
        lo, hi = float(bands[:, b - 1].min()), float(bands[:, b - 1].max())
        row = {
            "band": b,
            "min_ev": round(lo - ef, 3),
            "max_ev": round(hi - ef, 3),
            "excluded_by_deck": b in excluded,
        }
        if b < bands.shape[1]:
            row["gap_to_next_ev"] = round(float(bands[:, b].min()) - hi, 3)
        if characters and b in characters:
            row["character"] = characters[b]
        rows.append(row)
    return rows


def _stat_key(path: Path) -> tuple[str, int, int]:
    """A file's identity for the caches below: path, size and modification time."""
    stat = os.stat(path)
    return str(path), stat.st_size, stat.st_mtime_ns


@functools.lru_cache(maxsize=8)
def _potcar_tables(path: str, size: int, mtime: int):
    from .backends.vasp.inputs import _species_shells
    from .backends.vasp.outputs import potcar_valences

    return potcar_valences(Path(path)), _species_shells(Path(path), None)


@functools.lru_cache(maxsize=4)
def _read_procar(path: str, size: int, mtime: int):
    """A parsed PROCAR, read once per file version: the deck check and validate_deck ask
    for it on every call, and a spin-orbit PROCAR runs to hundreds of megabytes."""
    from .backends.vasp.outputs import read_procar

    return read_procar(Path(path))


def _procar_totals(procar: Path, species, bands: Sequence[int], spin: int = 0):
    """Per band: k-summed weight on each (species, l), from a VASP PROCAR."""
    from .backends.vasp.outputs import VaspOutputError

    try:
        proj = _read_procar(*_stat_key(Path(procar)))
    except (VaspOutputError, OSError):
        return {}
    columns = [
        "d" if o.startswith(("x2-y2", "dx2")) else o[0].lower() for o in proj.orbitals
    ]
    weights = proj.weights.reshape(
        proj.n_spin, proj.n_kpoints, proj.n_bands, proj.n_ions, -1
    )
    species = list(species)
    out = {}
    for b in bands:
        if b > proj.n_bands:
            continue
        w = weights[min(spin, proj.n_spin - 1), :, b - 1].sum(axis=0)  # ion, orbital
        totals: dict[tuple[str, int], float] = {}
        for ion, symbol in enumerate(species[: proj.n_ions]):
            for column, letter in enumerate(columns):
                if letter in "spdf":
                    key = (symbol, "spdf".index(letter))
                    totals[key] = totals.get(key, 0.0) + float(w[ion, column])
        if sum(totals.values()) > 0:
            out[b] = totals
    return out


def procar_dominant(procar: Path, species, bands: Sequence[int], spin: int = 0):
    """Band -> (species, l, fraction) of its largest k-averaged weight."""
    out = {}
    for b, totals in _procar_totals(procar, species, bands, spin).items():
        whole = sum(totals.values())
        (symbol, l), value = max(totals.items(), key=lambda kv: kv[1])
        out[b] = (symbol, l, value / whole)
    return out


def procar_band_characters(procar: Path, species, bands: Sequence[int], spin: int = 0):
    """Band -> its two largest (species, l) weights, k-averaged, as text."""
    out = {}
    for b, totals in _procar_totals(procar, species, bands, spin).items():
        whole = sum(totals.values())
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:2]
        out[b] = ", ".join(f"{s} {'spdf'[l]} {v / whole:.2f}" for (s, l), v in top)
    return out


def parent_band_characters(
    ctx, bands: Sequence[int], spin: int = 0
) -> tuple[dict, str]:
    """Per-band character where the parent supplies it, and where it came from."""
    if ctx.code == "vasp":
        procar = Path(ctx.parent_source).with_name("PROCAR")
        if procar.is_file():
            found = procar_band_characters(procar, ctx.parent.species, bands, spin)
            if found:
                return found, f"PROCAR of the parent ({procar.name}), k-averaged"
        return {}, "no PROCAR beside the parent"
    return {}, (
        "not measured per band on Quantum ESPRESSO without projwfc.x output; the "
        "shells are named by count (catalogue) and the pseudo-atom's energies"
    )


def describe(
    win_text: str,
    pw_text: str | None,
    deck_dir: Path,
    ctx,
    spin_index: int = 0,
) -> dict[str, Any]:
    """Everything the model needs to choose a bottom cut, for validate_deck."""
    try:
        cat = catalogue(win_text, pw_text, deck_dir, ctx)
    except SemicoreUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    excluded = _excluded_from_text(win_text)
    reach = max([cat.bottom, sum(s.bands for s in cat.shells)]) + 2
    characters, source = parent_band_characters(ctx, range(1, reach + 1), spin_index)
    return {
        "available": True,
        **cat.to_dict(),
        "cuts": cut_table(ctx.parent, cat, spin_index),
        "lowest_bands": band_rows(
            ctx.parent, 1, reach, excluded, spin_index, characters
        ),
        "band_character_source": source,
        "rule": (
            "the gate sets the excluded block aside only if, over the whole mesh, its "
            "highest band lies more than 1 % of the occupied bandwidth below every kept "
            "band; otherwise every excluded occupied state counts as missing"
        ),
    }


__all__ = [
    "Catalogue",
    "KNOB",
    "SemicorePlan",
    "SemicoreUnavailable",
    "Shell",
    "catalogue",
    "cut_table",
    "describe",
    "exclusion_verdict",
    "resolve",
    "semicore_projection_lines",
]
