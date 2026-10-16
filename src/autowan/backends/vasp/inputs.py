"""Keep VASP's overlap setup separate from the standalone localisation deck.

Repairs belong in each construction's own .win. The library called by VASP sees a
filtered copy, so a localisation keyword it cannot parse never reaches that step.

The first deck takes the Quantum ESPRESSO campaign's band ratio (the parent-side
rule). The parent's headroom bands above it are excluded, not absent, so a repair
that raises num_bands uncovers them through exclude_bands without a DFT rerun.

The automatic first deck names every non-local POTCAR channel, with one Wannier
function per occupied radial channel: a channel whose pseudopotential was built at
two occupied atomic shells (Mn_sv's 3s semicore and 4s valence) gets one function per
shell, the second as Wannier90's ``r=2`` radial function; a channel with one occupied
shell or none keeps one function. Its frozen window
ends 2 eV above the highest occupied state (E_F for a metal): VASP's smeared Fermi
level can sit inside an insulator's gap. The window is capped so that no spin or
k-point has more frozen states than Wannier functions.
Explicit projector/POTCAR overrides may describe a smaller subspace; their capped
window can exclude occupied states that the automatic deck must retain.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...actions import BY_NAME
from ...families import Method, knobs_for
from ...inputs import (
    ConstructionInputs,
    InputError,
    _apply_semicore,
    _excluded_from_text,
    _rebalanced_exclusion,
    _rewrite_win,
    _semicore_plan,
    assignments,
    deck_method,
    deck_spinors,
    projection_source,
    read_win,
    require_projectability_flag,
    to_absolute,
    wannier_count,
)
from ...qe_outputs import ParentCalculation
from .knowledge import VASP_KNOWLEDGE
from .outputs import (
    VaspOutputError,
    _vectors,
    locproj_auto_channels,
    potcar_channels,
    potcar_valences,
    read_kmesh,
    read_parent,
    read_procar,
)

# Freeze 2 eV above the occupied manifold, before capping to the model's capacity.
FROZEN_WINDOW_ABOVE_OCCUPIED_EV = 2.0

#: The first deck's band count is the rule the headroom-less campaign parents were
#: computed with (the development campaign's
#: ``nscf_nbands``), so a parent with headroom gives the same unrepaired deck as one
#: without. Two bands of slack on the frozen-window count; a multiple of 8 bands.
FROZEN_COUNT_MARGIN = 2
BAND_MULTIPLE = 8

#: VASP's overlap step builds hydrogenic radial functions for Wannier90's radial
#: quantum numbers 1-3 only (vasp 6.4.1 ``mlwf.F``: ``proj_radial > 0 .and. < 4``).
MAX_RADIAL_FUNCTIONS = 3
# A Description reference energy (7 decimals) is the eigenvalue of an Atomic
# configuration shell (printed to 4 decimals) when they agree to the printed precision.
_SHELL_ENERGY_TOLERANCE_EV = 1e-3

#: The automatic first decks the 2026-09-29 benchmark compared: hydrogenic functions
#: on every POTCAR channel, VASP's own pseudopotential projections (LOCPROJ_AUTO), and
#: SCDM (LSCDM).
RECIPES = ("hydrogenic", "locproj_auto", "scdm")
#: The hydrogenic deck disentangled by its energy window alone (no projectability
#: window), the 2026-09-30 default.
ENERGY_WINDOW_RECIPE = "hydrogenic_energy_window"
#: The VASP frontier: SCDM as the Varrassi et al. (2025) GW workflow runs it.
#: The package default since 2026-10-07.
SCDM_FRONTIER_RECIPE = "scdm_frontier"
#: autowan's SMV starting deck on VASP: semicore bands excluded (counted from the
#: POTCARs), one hydrogenic function per valence channel, energy window only (2026-10-07).
SEMICORE_EXCLUDED_RECIPE = "semicore_excluded"
#: The VASP frontier since 2026-10-08: JARVIS-tools'
#: Wannierisation automation (Garrity & Choudhary, Sci. Data 8, 106 (2021)), its deck
#: written by JARVIS's own writer and wrapped by :func:`jarvis_frontier_deck`.
JARVIS_FRONTIER_RECIPE = "jarvis_frontier"
DECK_RECIPES = (
    *RECIPES,
    ENERGY_WINDOW_RECIPE,
    SCDM_FRONTIER_RECIPE,
    SEMICORE_EXCLUDED_RECIPE,
    JARVIS_FRONTIER_RECIPE,
)
#: What ``deck_from_parent`` builds when no recipe is named. It was ``hydrogenic``: a
#: projectability window on hydrogenic functions, whose projectability exceeds one on
#: VASP, so Wannier90 froze and dropped the wrong states (a PDWF deck VASP cannot run).
DEFAULT_RECIPE = SCDM_FRONTIER_RECIPE

#: INCAR tags each recipe's overlap step needs. Wannier90 never reads them, so the deck
#: records them on a comment line and a construction carries its own overlap recipe.
RECIPE_OVERLAP_TAGS = {
    "hydrogenic": {},
    "locproj_auto": {"LOCPROJ_AUTO": ".TRUE."},
    "scdm": {"LSCDM": ".TRUE."},
    ENERGY_WINDOW_RECIPE: {},
}
OVERLAP_TAG_COMMENT = "! overlap-step INCAR:"


def first_deck_band_count(num_wann: int, frozen_max: int, parent_bands: int) -> int:
    """Take the campaign's band ratio, limited to the bands the parent carries.

    ``frozen_max`` is the most parent states at or below the deck's ``dis_froz_max``
    at any spin and k-point. Bands above the count stay in the parent, excluded.
    """
    needed = max(2 * num_wann, num_wann + frozen_max + FROZEN_COUNT_MARGIN)
    rounded = ((needed + BAND_MULTIPLE - 1) // BAND_MULTIPLE) * BAND_MULTIPLE
    return min(parent_bands, rounded)


def overlap_step_incar_tags(text: str) -> dict[str, str]:
    """The INCAR tags a deck asks its overlap step for (its ``! overlap-step INCAR:``)."""
    tags = {}
    for line in text.splitlines():
        match = re.match(
            r"\s*!\s*overlap-step INCAR:\s*(\w+)\s*=\s*(.*?)\s*$", line, re.IGNORECASE
        )
        if match:
            tags[match.group(1).upper()] = match.group(2)
    return tags


def _frozen_window_limits(
    parent: ParentCalculation, num_wann: int
) -> tuple[float, float | None]:
    bands = parent.eigenvalues_ev
    if parent.is_metallic():
        anchor = parent.fermi_energy_ev
    else:
        occupied = bands[bands <= parent.fermi_energy_ev]
        if not occupied.size:
            raise InputError(
                f"no occupied states to anchor {num_wann} Wannier functions"
            )
        anchor = float(occupied.max())
    if num_wann < 1:
        raise InputError(
            f"{num_wann} Wannier functions cannot hold the occupied manifold"
        )
    cap = (
        float(np.sort(bands, axis=-1)[..., num_wann].min())
        if parent.n_bands > num_wann
        else None
    )
    return anchor, cap


def _capped_window_top(anchor: float, cap: float | None) -> float:
    top = anchor + FROZEN_WINDOW_ABOVE_OCCUPIED_EV
    if cap is not None and top >= cap:
        top = min(cap - 1e-3, float(np.nextafter(cap, -np.inf)))
    return top


def frozen_window_top(parent: ParentCalculation, num_wann: int) -> float:
    """Freeze above occupied states, without exceeding any spin/k-point's capacity."""
    anchor, cap = _frozen_window_limits(parent, num_wann)
    top = _capped_window_top(anchor, cap)
    if top <= anchor:
        raise InputError(
            f"{num_wann} Wannier functions cannot hold the occupied manifold"
        )
    return top


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_DESCRIPTION_ROW = re.compile(
    rf"\s*(\d+)\s+({_NUMBER})\s+(\d+)\s+{_NUMBER}(?:\s+\d+\s+{_NUMBER})?\s*"
)


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _atomic_configuration(
    lines: list[str], symbol: str, path: Path
) -> list[tuple[int, int, float, float]]:
    """The header's ``Atomic configuration`` rows: (n, l, eigenvalue in eV, occupation)."""
    start = next(
        (i for i, line in enumerate(lines) if "Atomic configuration" in line), None
    )
    if start is None:
        raise InputError(
            f"{path}: no Atomic configuration table for {symbol}; the occupied radial "
            "channels the deck's function count needs are read from it"
        )
    entries, shells = None, []
    for line in lines[start + 1 :]:
        cols = line.split()
        if len(cols) == 2 and cols[1] == "entries":
            entries = int(cols[0])
            continue
        if cols[:2] == ["n", "l"]:
            continue
        try:
            n, l = int(cols[0]), int(cols[1])
            energy, occupation = _float(cols[3]), _float(cols[4])
        except (ValueError, IndexError):
            break
        shells.append((n, l, energy, occupation))
        if entries is not None and len(shells) == entries:
            break
    if not shells or (entries is not None and len(shells) != entries):
        raise InputError(f"{path}: truncated Atomic configuration table for {symbol}")
    return shells


def _description_rows(
    lines: list[str], symbol: str, path: Path
) -> list[tuple[int, float, int]]:
    """The header's ``Description`` rows: (l, reference energy in eV, TYP)."""
    start = next((i for i, line in enumerate(lines) if "Description" in line), None)
    if start is None:
        raise InputError(f"{path}: missing pseudopotential Description table")
    rows = []
    for line in lines[start + 2 :]:
        match = _DESCRIPTION_ROW.fullmatch(line)
        if match is None:
            break
        rows.append((int(match.group(1)), _float(match.group(2)), int(match.group(3))))
    if not rows:
        raise InputError(f"{path}: empty Description table for {symbol}")
    return rows


def occupied_radial_channels(
    path: str | Path,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Per species and non-local channel, the occupied shells its partial waves sit at.

    A partial wave is an occupied radial channel when its ``Description`` reference
    energy is the eigenvalue of an occupied shell of the same l in the header's
    ``Atomic configuration`` table. Mn_sv: s -> (3s, 4s), p -> (3p,), d -> (3d,).
    A channel no occupied shell was built at (Cu p, W_sv_GW f) maps to ``()``. The
    channel set itself is ``potcar_channels``': the local potential is not a channel.
    Reads a POTCAR or the OUTCAR's echo of its header (both carry the two tables).
    """
    path = Path(path)
    try:
        channels = potcar_channels(path)
    except VaspOutputError as exc:
        raise InputError(str(exc)) from exc
    blocks = re.split(r"(?m)^\s*VRHFIN\s*=", path.read_text(errors="replace"))[1:]
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for block in blocks:
        symbol = block.partition(":")[0].strip()
        lines = block.splitlines()
        occupied = [
            (n, l, energy)
            for n, l, energy, occupation in _atomic_configuration(lines, symbol, path)
            if occupation > 0
        ]
        rows = _description_rows(lines, symbol, path)
        per_channel = {}
        for letter in channels[symbol]:
            angular = "spdf".index(letter)
            shells: list[str] = []
            for row_l, energy, typ in rows:
                if row_l != angular or typ == 7:
                    continue
                for n, l, level in occupied:
                    label = f"{n}{letter}"
                    if (
                        l == angular
                        and abs(level - energy) <= _SHELL_ENERGY_TOLERANCE_EV
                        and label not in shells
                    ):
                        shells.append(label)
            per_channel[letter] = tuple(shells)
        if symbol in result and result[symbol] != per_channel:
            raise InputError(f"{path}: conflicting occupied shells for {symbol}")
        result[symbol] = per_channel
    return result


def functions_per_atom(channels: Mapping[str, tuple[str, ...]]) -> int:
    """One function per (l, occupied radial channel); every channel keeps at least one."""
    return sum(
        (2 * "spdf".index(letter) + 1) * max(1, len(shells))
        for letter, shells in channels.items()
    )


def semicore_aware_projectors(
    channels: Mapping[str, Mapping[str, tuple[str, ...]]], species
) -> list[dict]:
    """The automatic deck's projector entries, one function per occupied radial channel.

    Each species' first entry names every channel once (radial function r=1, the
    all-channel deck's line); every further occupied shell of a channel adds an entry
    with the next radial quantum number (Mn_sv: ``Mn:s;p;d`` then ``Mn:s:r=2``).
    """
    projectors = []
    for symbol in species:
        if symbol not in channels:
            raise InputError(f"no pseudopotential channels for {symbol}")
        per_channel = channels[symbol]
        projectors.append({"species": symbol, "orbitals": tuple(per_channel)})
        for letter, shells in per_channel.items():
            if len(shells) > MAX_RADIAL_FUNCTIONS:
                raise InputError(
                    f"{symbol} {letter}: {len(shells)} occupied radial channels "
                    f"({', '.join(shells)}), but VASP's overlap step writes hydrogenic "
                    f"functions for r=1..{MAX_RADIAL_FUNCTIONS} only"
                )
            for radial in range(2, len(shells) + 1):
                projectors.append(
                    {"species": symbol, "orbitals": (letter,), "radial": radial}
                )
    return projectors


def projection_lines(projectors) -> list[str]:
    """Wannier90 projection lines; an entry's ``radial`` > 1 becomes ``:r=<n>``."""
    lines = []
    for entry in projectors:
        line = f"{entry['species']}:{';'.join(entry['orbitals'])}"
        radial = entry.get("radial", 1)
        if radial != 1:
            line += f":r={radial}"
        lines.append(line)
    return lines


def _multi_shell_note(channels: Mapping[str, Mapping[str, tuple[str, ...]]]) -> str:
    named = [
        f"{symbol} {letter} ({', '.join(shells)})"
        for symbol, per_channel in channels.items()
        for letter, shells in per_channel.items()
        if len(shells) > 1
    ]
    return (
        "! one function per occupied radial channel: " + "; ".join(named)
        if named
        else ""
    )


def _channel_source(
    path: Path, potcar: str | Path | None, reader=potcar_channels
) -> tuple[dict, str]:
    """Use explicit POTCAR or agree on the parent's available header tables."""
    try:
        if potcar is not None:
            return reader(potcar), str(potcar)
        sources = {}
        for candidate in (path.with_name("OUTCAR"), path.with_name("POTCAR")):
            if not candidate.is_file():
                continue
            text = candidate.read_text(errors="replace")
            blocks = re.split(r"(?m)^\s*VRHFIN\s*=", text)[1:]
            if any("Description" in block for block in blocks):
                sources[candidate] = reader(candidate)
    except VaspOutputError as exc:
        raise InputError(str(exc)) from exc
    if not sources:
        raise InputError(
            f"no pseudopotential table was found in {path.with_name('POTCAR')} "
            f"or {path.with_name('OUTCAR')}"
        )
    channels = next(iter(sources.values()))
    source = " and ".join(map(str, sources))
    if any(other != channels for other in sources.values()):
        raise InputError(f"pseudopotential channels disagree between {source}")
    return channels, source


def read_incar(text: str) -> dict[str, str]:
    tags = {}
    for line in text.splitlines():
        for assignment in re.split(r"[!#]", line, maxsplit=1)[0].split(";"):
            key, separator, value = assignment.partition("=")
            if separator and key.strip():
                tags[key.strip().upper()] = value.strip()
    return tags


def write_incar(tags: Mapping[str, str]) -> str:
    return "".join(f"{key.upper()} = {value}\n" for key, value in tags.items())


def read_construction(run_dir: str | Path, seed: str) -> ConstructionInputs:
    win = Path(run_dir) / f"{seed}.win"
    values = read_win(win)
    values["projection_source"] = projection_source(win, None)
    return ConstructionInputs(seed=seed, win=win, pw2wan=None, values=values)


def write_derived(
    source: ConstructionInputs,
    run_dir: str | Path,
    new_seed: str,
    changes: Mapping[str, Any],
    fermi_energy_ev: float | None = None,
    species_counts: Mapping[str, int] | None = None,
) -> ConstructionInputs:
    """Write a repaired VASP deck in the family of its source.

    Wannier90 knobs go into the .win. SCDM's mu and sigma are overlap-step INCAR tags
    (``CUTOFF_MU`` in absolute eV, ``CUTOFF_SIGMA``; ``CUTOFF_TYPE = erfc``), recorded
    on the deck's ``! overlap-step INCAR:`` lines that the runner applies. A frontier
    deck's ``NBANDS`` follows ``num_bands``, and its Wannier count follows VASP's own
    rule, ``min(NBANDS, n_pseudo)``.
    """
    from ...actions import cross_family_reason

    if not changes:
        raise InputError("no changes to apply")
    text = source.win.read_text()
    method = deck_method(text)
    # Validate before writing: a refused proposal must not leave half a deck.
    for name in changes:
        if name not in BY_NAME:
            raise InputError(f"{name!r} is not an editable parameter")
        if name in VASP_KNOWLEDGE.unsupported_parameters:
            raise InputError(
                f"VASP cannot write {name}: {VASP_KNOWLEDGE.unsupported_reasons[name]}"
            )
        if name not in knobs_for(method):
            raise InputError(cross_family_reason(name, method))
    changes = dict(changes)
    if fermi_energy_ev is not None:
        changes = {k: to_absolute(k, v, fermi_energy_ev) for k, v in changes.items()}
    tags: dict[str, Any] = {}
    if "scdm_mu" in changes:
        tags["CUTOFF_MU"] = changes.pop("scdm_mu")
    if "scdm_sigma" in changes:
        tags["CUTOFF_SIGMA"] = changes.pop("scdm_sigma")
    if tags and "CUTOFF_TYPE" not in overlap_step_incar_tags(text):
        tags["CUTOFF_TYPE"] = "erfc"
    remove: set[str] = set()
    projections = None
    plan = _semicore_plan(changes)
    if "projectors" in changes:
        named = changes.pop("projectors")
        projections = projection_lines(named)
        remove.add("auto_projections")
        if species_counts:
            changes["num_wann"] = wannier_count(
                named, species_counts, spinors=deck_spinors(text)
            )
    record = None
    if plan is not None:
        if is_frontier_overlap_step(text):
            from ...families import knob_unavailable

            raise InputError(
                "semicore_exclusion: "
                + (knob_unavailable("vasp", method, "semicore_exclusion") or "no route")
            )
        record = _apply_semicore(
            plan,
            win_text=text,
            pw2wan_text=None,
            run_dir=Path(run_dir),
            win_changes=changes,
            pw2wan_changes={},
            pw2wan_remove=set(),
            win_remove=remove,
            projections=projections,
            subset=None,
            species_counts=species_counts,
        )
        projections = record.pop("projections", projections)
    elif "num_bands" in changes:
        old = read_win(source.win).get("num_bands")
        try:
            old_bands = int(float(old)) if old is not None else None
        except ValueError:
            old_bands = None
        new_bands = int(changes["num_bands"])
        rebalanced = _rebalanced_exclusion(text, old_bands, new_bands)
        if rebalanced is not None:
            if rebalanced:
                changes["exclude_bands"] = rebalanced
            else:
                remove.add("exclude_bands")
        if is_frontier_overlap_step(text):
            bottom = 0
            excluded = sorted(_excluded_from_text(text))
            while bottom < len(excluded) and excluded[bottom] == bottom + 1:
                bottom += 1
            tags["NBANDS"] = str(new_bands + bottom)
            pseudo = recorded_pseudo_count(text)
            if pseudo is not None:
                changes["num_wann"] = min(new_bands, pseudo)
    target = Path(run_dir) / f"{new_seed}.win"
    _rewrite_win(
        source.win,
        target,
        changes,
        parent=source.seed,
        remove=remove,
        projections=projections,
    )
    if tags:
        target.write_text(set_overlap_tags(target.read_text(), tags))
    if record is not None:
        from ...semicore import with_record

        target.write_text(with_record(target.read_text(), record["lines"]))
    derived = read_construction(run_dir, new_seed)
    if deck_method(derived.win.read_text()) is not method:
        raise InputError(
            "the derived deck changed method family; that is switch_method's to do"
        )
    return derived


def overlap_step_win(text: str) -> str:
    allowed = {key.lower() for key in VASP_KNOWLEDGE.overlap_step_keys}
    kept, block = [], None
    for line in text.splitlines():
        cleaned = re.split(r"[!#]", line, maxsplit=1)[0].strip().lower()
        begin = re.match(r"begin\s*:?\s*(\w+)", cleaned)
        end = re.match(r"end\s*:?\s*(\w+)", cleaned)
        if begin:
            block = begin.group(1)
            if block in allowed:
                kept.append(line)
        elif block is not None:
            if block in allowed:
                kept.append(line)
            if end:
                block = None
        elif cleaned and re.split(r"[\s=:]", cleaned, maxsplit=1)[0] in allowed:
            kept.append(line)
    return "\n".join(kept) + "\n"


def deck_from_parent(
    vasprun_xml: str | Path,
    run_dir: str | Path,
    seed: str,
    projectors=None,
    potcar: str | Path | None = None,
    recipe: str = DEFAULT_RECIPE,
) -> ConstructionInputs:
    """Write a VASP first deck by ``recipe`` (default: the SCDM frontier).

    ``scdm_frontier`` and ``semicore_excluded`` are built by
    :func:`scdm_frontier_deck` and :func:`semicore_excluded_deck`. The remaining
    recipes are the hydrogenic family this function has always written: name every
    non-local POTCAR channel, unless projectors are supplied.

    One function per (species, l, occupied radial channel): a channel built at two
    occupied shells (Mn_sv s: 3s and 4s) is named twice, the second as ``r=2``; a
    channel with one occupied shell or none is named once (``occupied_radial_channels``).

    ``recipe`` picks the automatic first deck. ``hydrogenic`` (the default) is this
    deck. ``hydrogenic_energy_window`` is the same deck without the projectability
    window (``dis_proj_min``/``dis_proj_max``/``dis_froz_proj``): VASP's default first
    deck, since its raw projections exceed unit projectability. ``locproj_auto`` names the orbitals VASP's LOCPROJ_AUTO selects (the first
    bound partial wave of each l, read from the POTCAR). ``scdm`` keeps the same
    Wannier count but names no projectors and sets no projectability window: VASP's
    SCDM supplies the AMN. All three share the automatic frozen-window rule.

    The first deck takes the Quantum ESPRESSO campaign's band ratio (the parent-side
    rule). The parent's headroom bands above it are excluded, not absent, so a repair
    that raises num_bands uncovers them through exclude_bands without a DFT rerun.

    Freeze to the highest occupied state + 2 eV (E_F + 2 eV for metals), not
    VASP's smeared Fermi level inside an insulator's gap. Cap the window below
    the first band that would overfill num_wann at any spin or k-point.
    Explicit projector/POTCAR overrides can select a smaller subspace, in which
    case the cap may exclude occupied states rather than refusing that override.
    Enable projectability thresholds with dis_froz_proj, or Wannier90 ignores them.
    """
    if recipe == SCDM_FRONTIER_RECIPE:
        if projectors is not None:
            raise InputError("the SCDM frontier names no projectors")
        return scdm_frontier_deck(vasprun_xml, run_dir, seed, potcar=potcar)
    if recipe == SEMICORE_EXCLUDED_RECIPE:
        if projectors is not None:
            raise InputError("the semicore-excluded deck derives its own projectors")
        return semicore_excluded_deck(vasprun_xml, run_dir, seed, potcar=potcar)
    if recipe == JARVIS_FRONTIER_RECIPE:
        raise InputError(
            "the JARVIS frontier deck wraps the text JARVIS-tools' own writer produced "
            "(jarvis.io.wannier.inputs.Wannier90win); call jarvis_frontier_deck with it"
        )
    path = Path(vasprun_xml)
    parent, mesh, kpoints = _parent_mesh(path)
    if recipe not in DECK_RECIPES:
        raise InputError(
            f"unknown VASP first-deck recipe {recipe!r}; known: {', '.join(DECK_RECIPES)}"
        )
    explicit_channels = projectors is not None or potcar is not None
    source = "explicit projectors"
    shell_note = ""
    if recipe == "locproj_auto":
        if projectors is not None:
            raise InputError(
                "LOCPROJ_AUTO chooses its own projectors; a named set is another deck"
            )
        potcar_path = Path(potcar) if potcar is not None else path.with_name("POTCAR")
        try:
            selection = locproj_auto_channels(potcar_path)
        except VaspOutputError as exc:
            raise InputError(str(exc)) from exc
        projectors = []
        for symbol in dict.fromkeys(parent.species):
            if symbol not in selection:
                raise InputError(
                    f"{potcar_path}: no LOCPROJ_AUTO channels for {symbol}"
                )
            projectors.append({"species": symbol, "orbitals": selection[symbol]})
        source = f"LOCPROJ_AUTO (first bound partial wave per l) from {potcar_path}"
        # VASP's selection is an automatic deck, not an override: it is held to the
        # same window rule as the other recipes (it must hold the occupied states).
        explicit_channels = False
    elif projectors is None:
        channels, source = _channel_source(path, potcar, occupied_radial_channels)
        for symbol in dict.fromkeys(parent.species):
            if symbol not in channels:
                raise InputError(f"{source}: no pseudopotential channels for {symbol}")
        projectors = semicore_aware_projectors(
            channels, tuple(dict.fromkeys(parent.species))
        )
        shell_note = _multi_shell_note(
            {s: channels[s] for s in dict.fromkeys(parent.species)}
        )
    # vasp_ncl: every named orbital is two spinor functions (spinors = .true.).
    count = wannier_count(projectors, Counter(parent.species), spinors=parent.spinor)
    anchor, cap = _frozen_window_limits(parent, count)
    top = (
        _capped_window_top(anchor, cap)
        if explicit_channels
        else frozen_window_top(parent, count)
    )
    anchor_name = "Fermi level" if parent.is_metallic() else "highest occupied state"
    if cap is None:
        cap_note = "no cap: num_bands <= num_wann"
    else:
        binding = top < anchor + FROZEN_WINDOW_ABOVE_OCCUPIED_EV
        cap_note = f"cap {cap:.12g} eV {'binding' if binding else 'not binding'}"
    if top <= anchor:
        cap_note += "; explicit override excludes occupied states"
    window_comment = (
        f"! dis_froz_max: {anchor_name} {anchor:.12g} eV + "
        f"{FROZEN_WINDOW_ABOVE_OCCUPIED_EV:g} eV ({cap_note}; num_wann = {count})"
    )
    frozen_max = int((parent.eigenvalues_ev <= top).sum(axis=-1).max())
    num_bands = first_deck_band_count(count, frozen_max, parent.n_bands)
    scalars = {
        "num_bands": num_bands,
        "num_wann": count,
        "dis_froz_max": top,
        "dis_proj_min": "0.01",
        "dis_proj_max": "0.95",
        "dis_froz_proj": ".true.",
        "conv_window": 3,
        "conv_tol": "1.0e-7",
        "dis_conv_tol": "1.0e-7",
        "num_iter": 4000,
        "dis_num_iter": 4000,
        "num_cg_steps": 200,
        "write_hr": ".true.",
        "mp_grid": " ".join(map(str, mesh)),
    }
    if parent.spinor:
        scalars["spinors"] = ".true."
    if recipe in ("scdm", ENERGY_WINDOW_RECIPE):
        # SCDM's AMN comes from VASP's density-matrix columns, not from trial
        # functions, so there is no projectability to window. The energy-window deck
        # has trial functions, but VASP's are not orthonormal (p > 1), so it freezes
        # by energy alone.
        for key in ("dis_proj_min", "dis_proj_max", "dis_froz_proj"):
            scalars.pop(key)
    lines = []
    for key, value in scalars.items():
        if key == "dis_froz_max":
            lines.append(window_comment)
        lines.append(f"{key} = {value}")
        if key == "num_bands" and parent.n_bands > num_bands:
            lines.append(f"exclude_bands = {num_bands + 1}-{parent.n_bands}")
    tags = [
        f"{OVERLAP_TAG_COMMENT} {key} = {value}"
        for key, value in RECIPE_OVERLAP_TAGS[recipe].items()
    ]
    notes = [shell_note] if shell_note else []
    if recipe == "scdm":
        lines += ["", f"! Wannier count from the channels of: {source}", *notes, *tags]
    else:
        lines += [
            "",
            f"! projector channels: {source}",
            *notes,
            *tags,
            "begin projections",
            *projection_lines(projectors),
            "end projections",
        ]
    lines += ["", "begin unit_cell_cart", "Ang"]
    lines += [" ".join(f"{x:.12g}" for x in row) for row in parent.lattice_ang]
    lines += ["end unit_cell_cart", "", "begin atoms_frac"]
    lines += [
        f"{symbol} " + " ".join(f"{x:.12g}" for x in row)
        for symbol, row in zip(parent.species, parent.positions_frac)
    ]
    lines += ["end atoms_frac", "", "begin kpoints"]
    # Keep the XML's ordering; sorting a mesh detaches the overlaps from its bands.
    lines += [" ".join(f"{x:.12g}" for x in row) for row in kpoints]
    lines += ["end kpoints"]
    text = "\n".join(lines) + "\n"
    require_projectability_flag(text, f"{seed}.win")
    (Path(run_dir) / f"{seed}.win").write_text(text)
    return read_construction(run_dir, seed)


# ---------------------------------------------------------------------------
# shared pieces of the two 2026-10-07 decks
# ---------------------------------------------------------------------------


def _parent_mesh(path: Path):
    parent = read_parent(path)
    try:
        mesh = read_kmesh(path)
    except VaspOutputError as exc:
        raise InputError(f"{exc}; VASP deck requires a full mesh (ISYM = -1)") from exc
    root = ET.parse(path).getroot()
    kpoints = _vectors(root, "kpoints/varray[@name='kpointlist']")
    if len(mesh) != 3 or any(n < 1 for n in mesh) or len(kpoints) != int(np.prod(mesh)):
        raise InputError(
            "VASP2WANNIER90 needs the full k-point mesh: run the parent with ISYM = -1"
        )
    return parent, mesh, kpoints


def _structure_lines(parent, kpoints) -> list[str]:
    lines = ["", "begin unit_cell_cart", "Ang"]
    lines += [" ".join(f"{x:.12g}" for x in row) for row in parent.lattice_ang]
    lines += ["end unit_cell_cart", "", "begin atoms_frac"]
    lines += [
        f"{symbol} " + " ".join(f"{x:.12g}" for x in row)
        for symbol, row in zip(parent.species, parent.positions_frac)
    ]
    lines += ["end atoms_frac", "", "begin kpoints"]
    # Keep the XML's ordering; sorting a mesh detaches the overlaps from its bands.
    lines += [" ".join(f"{x:.12g}" for x in row) for row in kpoints]
    lines += ["end kpoints"]
    return lines


def _incar_parameters(path: Path) -> dict[str, str]:
    """The run's INCAR as vasprun.xml records it (``incar`` then ``parameters``)."""
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for item in root.findall("parameters//i") + root.findall("parameters//v"):
        if item.get("name"):
            out[item.get("name").upper()] = (item.text or "").strip()
    for item in root.findall("incar/i") + root.findall("incar/v"):
        if item.get("name"):
            out[item.get("name").upper()] = (item.text or "").strip()
    return out


def _potcar_path(path: Path, potcar: str | Path | None) -> Path:
    if potcar is not None:
        return Path(potcar)
    for candidate in (path.with_name("POTCAR"), path.with_name("OUTCAR")):
        if candidate.is_file():
            return candidate
    raise InputError(
        f"no POTCAR (or OUTCAR echo) beside {path}; the deck's counts are read from it"
    )


# ---------------------------------------------------------------------------
# the SCDM frontier (Varrassi et al. 2025)
# ---------------------------------------------------------------------------

#: The Wannier step's .win as the GW workflow writes it (``WANNIER90_WIN``): no windows,
#: no projections, no num_wann (VASP supplies it).
SCDM_FRONTIER_WIN = {
    "dis_num_iter": 500,
    "dis_conv_tol": "1.0e-8",
    "num_iter": 1000,
    "conv_tol": "1.0e-4",
}

#: INCAR tags of the Wannier step, overlaid on the parent's ENCUT, PREC, ISPIN/MAGMOM,
#: ISYM and k-mesh. ``LWRITE_MMN_AMN`` is ``.FALSE.`` in their code; it only writes the
#: overlap files, which autowan's gate and localisation read.
SCDM_FRONTIER_TAGS = {
    "ALGO": "None",
    "NELM": "1",
    "ISTART": "1",
    "LWANNIER90_RUN": ".TRUE.",
    "LSCDM": ".TRUE.",
    "ISMEAR": "0",
    "SIGMA": "0.02",
    "LORBIT": "11",
    "LWRITE_MMN_AMN": ".TRUE.",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

#: The .win keys the frontier overlap step hands VASP in ``WANNIER90_WIN``.
FRONTIER_WIN_KEYS = tuple(SCDM_FRONTIER_WIN)

_PSEUDO_COUNT_NOTE = (
    "! VASP's own Wannier count estimate (LPRJ_FROM_PSEUDO): n_pseudo ="
)


def vasp_default_nbands(
    nelect: float,
    nions: int,
    *,
    magmom_sum: float = 0.0,
    ispin: int = 1,
    noncollinear: bool = False,
    npar: int = 1,
) -> tuple[int, int]:
    """VASP 6.4.1's default NBANDS (``main.F`` 1288-1315): ``(raw, rounded to NPAR)``.

    ``max(NINT(NELECT+2)/2 + max(NIONS/2, 3), INT(0.6 NELECT)) + NMAG`` with
    ``NMAG = (INT(sum MAGMOM) + 1)/2`` for ``ISPIN = 2`` (integer division throughout),
    doubled for a noncollinear run, then rounded up to a multiple of NPAR.
    """
    nmag = 0
    if ispin > 1 or noncollinear:
        nmag = int(magmom_sum)
    nmag = (nmag + 1) // 2
    raw = (
        max(int(round(nelect + 2)) // 2 + max(nions // 2, 3), int(0.6 * nelect)) + nmag
    )
    if noncollinear:
        raw *= 2
    if npar < 1:
        raise InputError(f"NPAR must be at least 1, got {npar}")
    rounded = ((raw + npar - 1) // npar) * npar
    return raw, rounded


def pseudo_wannier_count(potcar: Path, species) -> int:
    """``LPRJ_FROM_PSEUDO``'s count: the first bound partial wave of each occupied l."""
    try:
        selection = locproj_auto_channels(potcar)
    except VaspOutputError as exc:
        raise InputError(str(exc)) from exc
    total = 0
    for symbol in species:
        if symbol not in selection:
            raise InputError(f"{potcar}: no bound partial waves read for {symbol}")
        total += sum(2 * "spdf".index(letter) + 1 for letter in selection[symbol])
    return total


def scdm_frontier_deck(
    vasprun_xml: str | Path,
    run_dir: str | Path,
    seed: str,
    *,
    potcar: str | Path | None = None,
    npar: int = 1,
    nbands: int | None = None,
) -> ConstructionInputs:
    """The VASP frontier: SCDM exactly as the Varrassi et al. (2025) Wannier step runs.

    ``ALGO = None``/``NELM = 1``/``ISTART = 1`` read the parent's WAVECAR without
    re-diagonalising; ``LWANNIER90_RUN`` and ``LSCDM`` with every ``CUTOFF_*`` left at
    VASP's defaults; ``ISMEAR = 0``, ``SIGMA = 0.02``, ``LORBIT = 11``. ``NUM_WANN`` is
    not set: VASP takes ``min(NBANDS, n_pseudo)`` (``mlwf.F`` 509-512), which the deck
    predicts so Wannier90 is told the same count, and the overlap step is checked
    against it. The .win carries only ``dis_num_iter 500, dis_conv_tol 1e-8,
    num_iter 1000, conv_tol 1e-4``: no windows, no projections.

    ``NBANDS`` is set explicitly to VASP's default formula, rounded to a multiple of
    ``npar`` (and ``NPAR`` is written so VASP does not round again); both numbers are
    recorded. ``npar`` must be the overlap step's MPI rank count: VASP's Wannier
    interface refuses NCORE != 1 ("The PEAD routines do not work for NCORE /= 1"), and
    the runner refuses a launcher whose rank count differs. ``nbands`` overrides it -- the parent's 4 num_wann is the sensitivity
    reading of the report.
    """
    path = Path(vasprun_xml)
    parent, mesh, kpoints = _parent_mesh(path)
    if parent.spinor:
        raise InputError(
            "the VASP frontier is SCDM as the Varrassi et al. (2025) GW workflow runs it, "
            "and that workflow publishes no spin-orbit (vasp_ncl) protocol; a SOC parent "
            "has no VASP frontier deck. The SMV starting deck (semicore_excluded) runs "
            "with spinors, and is not a frontier"
        )
    params = _incar_parameters(path)
    try:
        nelect = float(params["NELECT"])
    except (KeyError, ValueError) as exc:
        raise InputError(
            f"{path} records no NELECT; NBANDS is computed from it"
        ) from exc
    ispin = int(float(params.get("ISPIN", "1") or 1))
    magmom = [float(x) for x in params.get("MAGMOM", "").split() if x]
    noncollinear = params.get("LNONCOLLINEAR", "F").upper().startswith(("T", ".T"))
    raw, rounded = vasp_default_nbands(
        nelect,
        len(parent.species),
        magmom_sum=sum(magmom) if ispin > 1 else 0.0,
        ispin=ispin,
        noncollinear=noncollinear,
        npar=npar,
    )
    chosen = rounded if nbands is None else int(nbands)
    if chosen > parent.n_bands:
        raise InputError(
            f"NBANDS = {chosen} exceeds the {parent.n_bands} bands the parent WAVECAR "
            f"holds; the frontier reads that file without diagonalising"
        )
    potcar_path = _potcar_path(path, potcar)
    n_pseudo = pseudo_wannier_count(potcar_path, parent.species)
    num_wann = min(chosen, n_pseudo)
    tags = {**SCDM_FRONTIER_TAGS, "NBANDS": str(chosen), "NPAR": str(npar)}
    nbands_note = (
        f"! NBANDS = {chosen}: VASP default formula gives {raw}, rounded to a multiple "
        f"of NPAR = {npar} -> {rounded}"
        + ("" if nbands is None else f"; overridden to {chosen} (sensitivity reading)")
    )
    lines = [
        "! VASP SCDM frontier: the Wannier step of the Varrassi et al. (2025) GW workflow",
        "! (npj Comput. Mater.; arXiv:2410.15948)",
        nbands_note,
        f"{_PSEUDO_COUNT_NOTE} {n_pseudo} (read from {potcar_path.name}); "
        f"num_wann = min(NBANDS, n_pseudo) = {num_wann}; NUM_WANN is not set",
        "! overlap-step mode: frontier (reads the parent WAVECAR; no ICHARG, no NUM_WANN)",
        *[f"{OVERLAP_TAG_COMMENT} {key} = {value}" for key, value in tags.items()],
        f"num_wann = {num_wann}",
        f"num_bands = {chosen}",
    ]
    if parent.n_bands > chosen:
        # Not passed to VASP (its NBANDS is already this count): it tells the gate which
        # parent bands the construction represents.
        lines.append(f"exclude_bands = {chosen + 1}-{parent.n_bands}")
    lines += [f"{key} = {value}" for key, value in SCDM_FRONTIER_WIN.items()]
    lines += ["write_hr = .true.", f"mp_grid = {' '.join(map(str, mesh))}"]
    lines += _structure_lines(parent, kpoints)
    text = "\n".join(lines) + "\n"
    (Path(run_dir) / f"{seed}.win").write_text(text)
    return read_construction(run_dir, seed)


def is_frontier_overlap_step(text: str) -> bool:
    """The SCDM frontier's overlap step (``ALGO = None``; VASP writes its own .win)."""
    return overlap_step_incar_tags(text).get(
        "ALGO", ""
    ).strip().lower() == "none" and not is_jarvis_overlap_step(text)


def recorded_pseudo_count(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(_PSEUDO_COUNT_NOTE):
            match = re.match(r"\s*(\d+)", line[len(_PSEUDO_COUNT_NOTE) :])
            if match:
                return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# the JARVIS frontier (JARVIS-tools' Wannierisation; an SCDM deck)
# ---------------------------------------------------------------------------

#: The marker line that routes a deck's overlap step through the JARVIS mode.
JARVIS_MODE_LINE = (
    "! overlap-step mode: jarvis (reads the parent WAVECAR, ALGO = None; the .win is "
    "passed to VASP)"
)

#: The Wannier step's INCAR as JARVIS writes it (``jarvis/tasks/vasp/vasp.py`` 788-804,
#: usnistgov/jarvis 3b0c9d0), overlaid on the parent's ENCUT, ISPIN/MAGMOM, ISYM and
#: k-mesh. ``LSORBIT`` is added for a spinor parent; ``NBANDS`` is the parent's.
#: ``ISTART = 1`` is what VASP chooses with the WAVECAR present. The runner keeps
#: ``ICHARG = 11``: the WAVECAR here is a fixed-density NSCF's, not a self-consistent
#: run's, so the density is read rather than rebuilt from orbitals at SIGMA = 0.01.
#: ``NUM_WANN`` is set from the deck (VASP 6 otherwise estimates its own count).
JARVIS_FRONTIER_TAGS = {
    "PREC": "Accurate",
    "ALGO": "None",
    "ISMEAR": "0",
    "NSW": "0",
    "NELM": "1",
    "SIGMA": "0.01",
    "LWANNIER90": ".TRUE.",
    "LWRITE_MMN_AMN": ".TRUE.",
    "NEDOS": "5000",
    "LORBIT": "11",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "ISTART": "1",
}


def is_jarvis_overlap_step(text: str) -> bool:
    return any(line.strip() == JARVIS_MODE_LINE for line in text.splitlines())


def jarvis_frontier_deck(
    vasprun_xml: str | Path,
    run_dir: str | Path,
    seed: str,
    *,
    jarvis_win: str,
) -> ConstructionInputs:
    """The VASP frontier: the deck JARVIS-tools' writer produced, run on our parent.

    ``jarvis_win`` is the text of JARVIS's ``hr_wannier.win``
    (``Wannier90win.write_hr_win`` over ``write_win``): ``num_wann`` and
    ``exclude_bands = 1-N`` from its semicore table, ``dis_froz_max = E_F + 2``,
    ``dis_froz_min = -1000``, the projections, ``guiding_centres``, ``kmesh_tol``,
    ``dis_num_iter``, ``dis_mix_ratio``, ``num_iter``, ``num_print_cycles``, and
    ``num_bands`` = JARVIS NBANDS - N. Its lines are kept verbatim; the deck adds only
    what VASP's interface would append (``spinors``, ``mp_grid``, the cell, the atoms and
    the k-points) and, when the parent WAVECAR holds more bands than JARVIS's NBANDS J,
    excludes bands J+1..NBANDS(parent): the overlap step reads the WAVECAR as it is, and
    the overlaps carry exactly bands N+1..J, as a JARVIS run at NBANDS = J writes them.
    """
    from ...inputs import _format_ranges

    path = Path(vasprun_xml)
    parent, mesh, kpoints = _parent_mesh(path)
    declared = assignments(jarvis_win)
    for key in ("num_wann", "num_bands"):
        if key not in declared:
            raise InputError(f"the JARVIS deck declares no {key}")
    if not re.search(r"^\s*begin\s+projections", jarvis_win, re.I | re.M):
        raise InputError("the JARVIS deck names no projections")
    num_wann = int(float(declared["num_wann"]))
    num_bands = int(float(declared["num_bands"]))
    bottom = sorted(_excluded_from_text(jarvis_win))
    if bottom != list(range(1, len(bottom) + 1)):
        raise InputError(
            f"the JARVIS deck's exclude_bands {_format_ranges(bottom)} is not 1-N"
        )
    n_ex = len(bottom)
    jarvis_nbands = num_bands + n_ex
    if jarvis_nbands > parent.n_bands:
        raise InputError(
            f"JARVIS's NBANDS {jarvis_nbands} (num_bands {num_bands} + {n_ex} excluded) "
            f"exceeds the {parent.n_bands} bands the parent WAVECAR holds"
        )
    if num_wann > num_bands:
        raise InputError(f"num_wann {num_wann} exceeds num_bands {num_bands}")
    excluded = bottom + list(range(jarvis_nbands + 1, parent.n_bands + 1))
    tags = {**JARVIS_FRONTIER_TAGS, "NBANDS": str(parent.n_bands)}
    if parent.spinor:
        tags["LSORBIT"] = ".TRUE."
    lines = [
        "! VASP JARVIS frontier: JARVIS-tools' Wannierisation (Garrity & Choudhary, "
        "Sci. Data 8, 106 (2021)),",
        "! jarvis/tasks/vasp/vasp.py 766-831 and jarvis/io/wannier/inputs.py; the "
        "lines below the tags are JARVIS's writer output",
        JARVIS_MODE_LINE,
        (
            f"! JARVIS NBANDS = {jarvis_nbands}: read from a parent WAVECAR of "
            f"{parent.n_bands} bands, bands {jarvis_nbands + 1}-{parent.n_bands} excluded"
            if jarvis_nbands < parent.n_bands
            else f"! JARVIS NBANDS = {jarvis_nbands} = the parent WAVECAR's band count"
        ),
        *[f"{OVERLAP_TAG_COMMENT} {key} = {value}" for key, value in tags.items()],
    ]
    wrote_exclusion = False
    for line in jarvis_win.splitlines():
        if re.match(r"\s*exclude_bands\b", line, re.I):
            lines.append(f"! JARVIS writer: {line.strip()}")
            lines.append(f"exclude_bands = {_format_ranges(excluded)}")
            wrote_exclusion = True
        else:
            lines.append(line.rstrip())
    if excluded and not wrote_exclusion:
        lines.append(f"exclude_bands = {_format_ranges(excluded)}")
    if parent.spinor and "spinors" not in declared:
        lines.append("spinors = .true.")
    lines.append(f"mp_grid = {' '.join(map(str, mesh))}")
    # VASP's interface writes the mesh at full precision; vasprun.xml keeps 8 decimals,
    # and JARVIS's kmesh_tol = 1e-9 then finds no neighbour shells (0.11111111 is 1/9
    # to 1e-9). A Gamma-centred mesh point is a multiple of 1/n: snap to it, in order.
    grid = np.asarray(mesh, dtype=float)
    snapped = np.round(np.asarray(kpoints, dtype=float) * grid) / grid
    if np.max(np.abs(snapped - np.asarray(kpoints, dtype=float))) > 1e-6:
        raise InputError("the parent's k-points are not a Gamma-centred mesh")
    lines += _structure_lines(_with_poscar_cell(parent, path), snapped + 0.0)
    text = "\n".join(lines) + "\n"
    (Path(run_dir) / f"{seed}.win").write_text(text)
    return read_construction(run_dir, seed)


def _with_poscar_cell(parent, vasprun: Path):
    """The parent with the cell and positions VASP read (its POSCAR), at full precision.

    vasprun.xml prints the cell to 8 decimals. Under JARVIS's kmesh_tol = 1e-9 that
    splits the degenerate b-vector shells of a hexagonal cell (Ti: "found too many
    nearest neighbours"); VASP's own overlap step used the POSCAR's numbers.
    """
    from dataclasses import replace

    poscar = vasprun.with_name("POSCAR")
    if not poscar.is_file():
        return parent
    lines = poscar.read_text().splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale <= 0:
        return parent
    lattice = lattice * scale
    start = 7 if lines[7].strip()[:1].lower() in ("d", "c") else 8
    if lines[start].strip()[:1].lower() != "d":
        return parent
    n = len(parent.species)
    positions = np.array(
        [[float(x) for x in lines[start + 1 + i].split()[:3]] for i in range(n)]
    )
    # VASP wraps fractional positions into [0, 1): compare modulo a lattice vector and
    # keep the full-precision value on the vasprun.xml side of the wrap.
    shift = positions - np.asarray(parent.positions_frac, dtype=float)
    shift -= np.round(shift)
    if not (
        np.allclose(lattice, parent.lattice_ang, atol=1e-6)
        and np.allclose(shift, 0.0, atol=1e-6)
    ):
        raise InputError(f"{poscar} does not hold the cell of {vasprun.name}")
    positions = np.asarray(parent.positions_frac, dtype=float) + shift
    return replace(parent, lattice_ang=lattice, positions_frac=positions)


# ---------------------------------------------------------------------------
# the semicore-excluded SMV deck
# ---------------------------------------------------------------------------

#: Valence electrons and the valence channels kept, per element: the JARVIS-WTB table
#: (Garrity & Choudhary, Sci. Data 8, 106 (2021); ``jarvis/io/wannier/default_semicore.json``
#: at usnistgov/jarvis 3b0c9d0). Only the valence column is taken from it: the semicore
#: count is our POTCAR's ZVAL minus this, never the table's excluded-electron column,
#: which assumes other POTCARs (JARVIS: Fe_pv, Ni_pv, Mo_pv, Cu_pv, Mn_pv, Na_pv, Ti_pv,
#: V_pv, W_pv, Bi, ...). The rows cover every element of the 24-material test set. Our POTCAR, ZVAL and the semicore it leaves, per element:
#:
#: ===  ==============  ====  =======================  ========================================
#: El   our POTCAR      ZVAL  semicore (ZVAL - val)    checked
#: ===  ==============  ====  =======================  ========================================
#: As   As              5     0                        PROCAR band character (GaAs)
#: Ba   Ba_sv           10    8: 5s 5p                 PROCAR band character
#: C    C               4     0                        PROCAR band character
#: Cu   Cu              11    0                        PROCAR band character
#: Ga   Ga_d            13    10: 3d                   PROCAR band character
#: La   La              11    8: 5s 5p                 PROCAR band character
#: Mn   Mn_sv           15    8: 3s 3p                 PROCAR band character
#: Na   Na              1     0                        PROCAR band character
#: O    O               6     2: 2s                    PROCAR band character
#: Si   Si              4     0                        PROCAR band character
#: Sr   Sr_sv           10    8: 4s 4p                 PROCAR band character
#: Ti   Ti_sv           12    8: 3s 3p                 PROCAR band character
#: V    V_sv            13    8: 3s 3p                 PROCAR band character
#: W    W_sv_GW         14    8: 5s 5p                 PROCAR band character
#: Bi   Bi_d            15    10: 5d                   extension parents
#: Co   Co_sv           17    8: 3s 3p                 extension parents
#: Fe   Fe_sv           16    8: 3s 3p                 extension parents
#: Mo   Mo_sv           14    8: 4s 4p                 extension parents
#: Ni   Ni_sv_GW        18    8: 3s 3p                 extension parents
#: Pt   Pt_sv_GW        18    8: 5s 5p                 extension parents
#: S    S               6     2: 3s (as O 2s)          extension parents
#: Se   Se              6     0 (the table keeps 4s)   extension parents
#: Sn   Sn_d            14    10: 4d                   extension parents
#: ===  ==============  ====  =======================  ========================================
#:
#: An element with no row is refused, never guessed (:class:`NoValenceRow`).
VALENCE_TABLE: dict[str, tuple[int, str]] = {
    "As": (5, "sp"),
    "Ba": (2, "sd"),
    "Bi": (5, "sp"),
    "C": (4, "sp"),
    "Co": (9, "sd"),
    "Cu": (11, "sd"),
    "Fe": (8, "sd"),
    "Ga": (3, "sp"),
    "La": (3, "sdf"),
    "Mn": (7, "sd"),
    "Mo": (6, "sd"),
    "Na": (1, "sp"),
    "Ni": (10, "sd"),
    "O": (4, "p"),
    "Pt": (10, "sd"),
    "S": (4, "p"),
    "Se": (6, "sp"),
    "Si": (4, "sp"),
    "Sn": (4, "sp"),
    "Sr": (2, "sd"),
    "Ti": (4, "sd"),
    "V": (5, "sd"),
    "W": (6, "sd"),
}


class NoValenceRow(InputError):
    """An element the VASP SMV deck has no checked valence count for."""


@dataclass(frozen=True)
class SemicoreAssignment:
    """Which bands a deck excludes as semicore, and how that was decided."""

    excluded_bands: int
    per_species: dict
    """symbol -> (semicore electrons per atom, semicore shells, valence channels, source)."""
    verified_by: str


def _species_shells(potcar: Path, symbol_lines) -> dict:
    """Per species: occupied PP shells as {label: (l, occupation, eigenvalue in eV)}."""
    text = potcar.read_text(errors="replace")
    blocks = re.split(r"(?m)^\s*VRHFIN\s*=", text)[1:]
    radial = occupied_radial_channels(potcar)
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for block in blocks:
        symbol = block.partition(":")[0].strip()
        lines = block.splitlines()
        config = _atomic_configuration(lines, symbol, potcar)
        occupation = {
            f"{n}{'spdf'[l]}": (occ, e) for n, l, e, occ in config if l < 4 and occ > 0
        }
        shells = {}
        for letter, labels in radial.get(symbol, {}).items():
            for label in labels:
                occ, energy = occupation.get(label, (0.0, 0.0))
                shells[label] = ("spdf".index(letter), occ, energy)
        out[symbol] = shells
    return out


def semicore_shells(symbol: str, semicore_e: float, shells: Mapping) -> tuple[str, ...]:
    """The POTCAR shells that hold an element's ``semicore_e`` semicore electrons.

    The semicore electrons are the deepest ones: the occupied shells are taken from the
    bottom of the POTCAR's atomic eigenvalues until ZVAL - valence is reached, exactly.
    Not "every shell but the highest n of each valence letter": the GW POTCARs are
    built on d10 references with no outer s shell (Ni_sv_GW 3s 3p 3d, Pt_sv_GW 5s 5p
    5d), where that rule would call the 3s (5s) semicore shell valence.
    ``shells`` maps a label to ``(l, occupation, eigenvalue)``.
    """
    chosen: list[str] = []
    remaining = semicore_e
    for label, (_l, occ, _energy) in sorted(shells.items(), key=lambda kv: kv[1][2]):
        if remaining <= 1e-6 or occ <= 0:
            continue
        chosen.append(label)
        remaining -= occ
    if abs(remaining) > 1e-6:
        raise InputError(
            f"{symbol}: the deepest shells {chosen or 'none'} hold "
            f"{semicore_e - remaining:g} electrons, but ZVAL - valence = "
            f"{semicore_e:g}; no set of whole shells is the semicore"
        )
    return tuple(chosen)


def semicore_assignment(
    parent,
    potcar: Path,
    procar: Path | None = None,
) -> SemicoreAssignment:
    """Count the semicore bands from our POTCARs and check them against band character.

    Per atom, semicore electrons = ZVAL - valence electrons (VALENCE_TABLE; an element
    with no row is refused), half that many bands per spin channel -- or that many bands
    for a spinor (``vasp_ncl``) parent, whose bands hold one electron each. The bottom N
    bands are
    then required to be those states: separated from band N+1 at every k-point and
    spin, and -- when a PROCAR is present -- dominated, at every k-point, by a
    (species, l) the assignment calls semicore. Energy order alone never decides it:
    in SrO the O 2s band lies below Sr 4p.
    """
    zval = potcar_valences(potcar)
    shells = _species_shells(potcar, None)
    per_species = {}
    semicore_l: dict[str, set[int]] = {}
    for symbol in dict.fromkeys(parent.species):
        if symbol not in zval or symbol not in shells:
            raise InputError(f"{potcar}: no valence data for {symbol}")
        if symbol not in VALENCE_TABLE:
            raise NoValenceRow(
                f"{symbol}: no row in the VASP SMV valence table, so its semicore "
                f"count (ZVAL {zval[symbol]:g} minus the valence electrons) is unknown "
                f"and is not guessed. Add a VALENCE_TABLE row (valence electrons, "
                f"channels) for this POTCAR after checking it against band character "
                f"(PROCAR) on a parent; known: {', '.join(sorted(VALENCE_TABLE))}"
            )
        valence_e, letters = VALENCE_TABLE[symbol]
        source = "JARVIS-WTB valence column, semicore from our POTCAR ZVAL"
        semicore_e = zval[symbol] - valence_e
        if (
            semicore_e < 0
            or abs(semicore_e - round(semicore_e)) > 1e-6
            or round(semicore_e) % 2
        ):
            raise InputError(
                f"{symbol}: ZVAL {zval[symbol]:g} minus {valence_e} valence electrons "
                f"leaves {semicore_e:g}, not an even number of semicore electrons"
            )
        semicore = list(semicore_shells(symbol, semicore_e, shells[symbol]))
        semicore_l[symbol] = {"spdf".index(lab[-1]) for lab in semicore}
        per_species[symbol] = (int(round(semicore_e)), tuple(semicore), letters, source)
    counts = Counter(parent.species)
    # Electrons -> bands: two per band in a spin-degenerate or collinear channel, one
    # per spinor band (vasp_ncl). A collinear run excludes the same N in each channel.
    per_band = 1 if parent.spinor else 2
    n_excluded = sum(per_species[s][0] // per_band * counts[s] for s in per_species)
    bands = parent.eigenvalues_ev
    verified = "gap above band N at every k-point and spin"
    if n_excluded:
        if n_excluded >= bands.shape[-1]:
            raise InputError(f"{n_excluded} semicore bands leave none to Wannierise")
        top = bands[..., n_excluded - 1].max()
        bottom = bands[..., n_excluded].min()
        if not bottom > top:
            raise InputError(
                f"band {n_excluded} (last semicore) reaches {top:.3f} eV and band "
                f"{n_excluded + 1} starts at {bottom:.3f} eV: the semicore block is not "
                f"separated, so excluding it by count would cut a manifold"
            )
        if procar is not None and Path(procar).is_file():
            _check_semicore_character(parent, Path(procar), n_excluded, semicore_l)
            verified = "PROCAR channel character and the gap above band N"
    return SemicoreAssignment(n_excluded, per_species, verified)


def _check_semicore_character(
    parent, procar: Path, n_excluded: int, semicore_l
) -> None:
    try:
        proj = read_procar(procar)
    except VaspOutputError as exc:
        raise InputError(str(exc)) from exc
    if proj.n_ions != len(parent.species):
        raise InputError(f"{procar}: {proj.n_ions} ions against {len(parent.species)}")
    columns = [
        "d" if o.startswith(("x2-y2", "dx2")) else o[0].lower() for o in proj.orbitals
    ]
    l_of = np.array(["spdf".index(c) if c in "spdf" else -1 for c in columns])
    weights = proj.weights.reshape(
        proj.n_spin, proj.n_kpoints, proj.n_bands, proj.n_ions, -1
    )
    species = list(parent.species)
    for band in range(n_excluded):
        w = weights[:, :, band]  # spin, k, ion, orbital
        for spin in range(proj.n_spin):
            for k in range(proj.n_kpoints):
                best, best_weight = None, -1.0
                for symbol in dict.fromkeys(species):
                    ions = [i for i, s in enumerate(species) if s == symbol]
                    for l in range(4):
                        value = float(
                            w[spin, k][np.ix_(ions, np.where(l_of == l)[0])].sum()
                        )
                        if value > best_weight:
                            best, best_weight = (symbol, l), value
                if best is None or best[1] not in semicore_l.get(best[0], set()):
                    raise InputError(
                        f"band {band + 1} at k-point {k + 1} is dominated by "
                        f"{best[0]} {'spdf'[best[1]]} character, which the assignment "
                        f"does not call semicore; excluding the bottom {n_excluded} "
                        f"bands would remove a valence state"
                    )


def semicore_excluded_deck(
    vasprun_xml: str | Path,
    run_dir: str | Path,
    seed: str,
    *,
    potcar: str | Path | None = None,
    procar: str | Path | None = None,
) -> ConstructionInputs:
    """autowan's SMV deck on VASP: semicore excluded, one function per valence channel.

    The bottom N bands are excluded, N counted from our POTCARs and checked against
    band character (:func:`semicore_assignment`); one hydrogenic function per valence
    channel (no ``r=2`` second shell); frozen window 2 eV above the highest occupied
    state (E_F for a metal), capped so no k-point or spin holds more frozen states
    than functions; energy-window disentanglement only (no projectability window).
    """
    path = Path(vasprun_xml)
    parent, mesh, kpoints = _parent_mesh(path)
    potcar_path = _potcar_path(path, potcar)
    procar_path = Path(procar) if procar is not None else path.with_name("PROCAR")
    assignment = semicore_assignment(parent, potcar_path, procar_path)
    n_ex = assignment.excluded_bands
    projectors = [
        {"species": s, "orbitals": tuple(assignment.per_species[s][2])}
        for s in dict.fromkeys(parent.species)
    ]
    count = wannier_count(projectors, Counter(parent.species), spinors=parent.spinor)
    kept = parent.eigenvalues_ev[..., n_ex:]
    if parent.is_metallic():
        anchor = parent.fermi_energy_ev
    else:
        occupied = kept[kept <= parent.fermi_energy_ev]
        if not occupied.size:
            raise InputError("no occupied valence states above the semicore block")
        anchor = float(occupied.max())
    cap = (
        float(np.sort(kept, axis=-1)[..., count].min())
        if kept.shape[-1] > count
        else None
    )
    top = _capped_window_top(anchor, cap)
    if top <= anchor:
        raise InputError(
            f"{count} valence functions cannot hold the occupied valence manifold"
        )
    frozen_max = int((kept <= top).sum(axis=-1).max())
    num_bands = first_deck_band_count(count, frozen_max, parent.n_bands - n_ex)
    excluded = list(range(1, n_ex + 1)) + list(
        range(n_ex + num_bands + 1, parent.n_bands + 1)
    )
    from ...inputs import _format_ranges

    lines = [
        "! VASP SMV starting deck: semicore excluded",
        f"! semicore bands excluded: {n_ex} ({assignment.verified_by})",
        *[
            f"!   {s}: {e} semicore electrons per atom in {', '.join(sh) or 'none'}; "
            f"valence channels {ch}; {src}"
            for s, (e, sh, ch, src) in assignment.per_species.items()
        ],
        f"! dis_froz_max: {'Fermi level' if parent.is_metallic() else 'highest occupied state'} "
        f"{anchor:.12g} eV + {FROZEN_WINDOW_ABOVE_OCCUPIED_EV:g} eV "
        f"({'no cap' if cap is None else f'cap {cap:.12g} eV'}; num_wann = {count})",
        f"num_bands = {num_bands}",
        f"num_wann = {count}",
    ]
    if parent.spinor:
        lines.insert(
            -2,
            "! spinors: vasp_ncl parent; every valence orbital is two spinor functions "
            "and every semicore electron one excluded band",
        )
        lines.append("spinors = .true.")
    if excluded:
        lines.append(f"exclude_bands = {_format_ranges(excluded)}")
    lines += [
        f"dis_froz_max = {top}",
        "conv_window = 3",
        "conv_tol = 1.0e-7",
        "dis_conv_tol = 1.0e-7",
        "num_iter = 4000",
        "dis_num_iter = 4000",
        "num_cg_steps = 200",
        "write_hr = .true.",
        f"mp_grid = {' '.join(map(str, mesh))}",
        "",
        f"! projector channels: valence channels per {potcar_path.name}",
        "begin projections",
        *projection_lines(projectors),
        "end projections",
    ]
    lines += _structure_lines(parent, kpoints)
    text = "\n".join(lines) + "\n"
    require_projectability_flag(text, f"{seed}.win")
    (Path(run_dir) / f"{seed}.win").write_text(text)
    return read_construction(run_dir, seed)


# ---------------------------------------------------------------------------
# overlap-step tags a repair may move (SCDM's mu and sigma)
# ---------------------------------------------------------------------------


def set_overlap_tags(text: str, tags: Mapping[str, Any]) -> str:
    """Replace or add ``! overlap-step INCAR:`` lines."""
    wanted = {k.upper(): v for k, v in tags.items()}
    out = []
    for line in text.splitlines():
        match = re.match(r"\s*!\s*overlap-step INCAR:\s*(\w+)\s*=", line, re.IGNORECASE)
        if match and match.group(1).upper() in wanted:
            continue
        out.append(line)
    insert = [f"{OVERLAP_TAG_COMMENT} {k} = {v}" for k, v in wanted.items()]
    return "\n".join(insert + out) + "\n"
