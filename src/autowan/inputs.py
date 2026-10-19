"""Reading and rewriting the two input files a construction is defined by.

A repair is a set of parameter assignments; turning it into a calculation means
writing them into ``<seed>.win`` or ``<seed>.pw2wan`` without disturbing anything the
model did not touch. Which file a parameter belongs to is not decided here — it is
already declared in the editable vocabulary, so a knob cannot end up in one file in
the validator and another on disk.

Every attempt writes its own seedname. Overwriting an input in place would leave a run
whose log names parameters no surviving file contains, and the record of what was tried
is the point of running this way at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .actions import BY_NAME, implied_wannier_count
from .families import Method

#: Keys read back out of a ``.win``. Anything else is carried through untouched.
WIN_SCALARS = (
    "num_wann",
    "num_bands",
    "num_iter",
    "dis_num_iter",
    "conv_window",
    "conv_tol",
    "trial_step",
    "dis_win_max",
    "dis_win_min",
    "dis_froz_max",
    "dis_froz_min",
    "dis_proj_min",
    "dis_proj_max",
    "dis_mix_ratio",
)

#: How the deck says its projectors are supplied. Read and reported because the
#: parameters alone do not say it: two decks can name the same shells and take their
#: radial functions from different places, and a repair that changes the source looks
#: like a change of nothing unless the source is on the record.
PROJECTION_SOURCE_KEYS = {
    "atom_proj": "the pseudopotential's own atomic orbitals",
    "scdm_proj": "selected columns of the density matrix",
}


class InputError(RuntimeError):
    """An input file is missing, or a parameter cannot be written into one."""


class ProjectabilityWindowIgnored(InputError):
    """Wannier90 would parse the thresholds without applying their window."""


def _parse_mp_grid(value: Any) -> tuple[int, int, int]:
    """Convert an ``mp_grid`` spelling to its three positive dimensions."""
    tokens = re.split(r"[,\s]+", str(value).strip())
    if len(tokens) != 3 or any(
        re.fullmatch(r"[0-9]+", token) is None or int(token) <= 0 for token in tokens
    ):
        raise InputError(
            f"invalid mp_grid {value!r}: expected three strictly positive integers"
        )
    return int(tokens[0]), int(tokens[1]), int(tokens[2])


def _projectability_assignments(text: str) -> dict[str, str]:
    """Read only keyword assignments, so comments and block data cannot set flags."""
    values = {}
    in_block = False
    for line in text.splitlines():
        cleaned = _clean(line).lower()
        if re.match(r"begin(?:\s|:|=)", cleaned):
            in_block = True
            continue
        if re.match(r"end(?:\s|:|=)", cleaned):
            in_block = False
            continue
        if in_block:
            continue
        match = re.match(
            r"(dis_proj_min|dis_proj_max|dis_proj_auto|dis_froz_proj)"
            r"(?:\s*[=:]\s*|\s+)(.*)$",
            cleaned,
        )
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _logical_true(value: str) -> bool:
    return value.lower().removeprefix(".").startswith("t")


def requests_projectability(text: str) -> bool:
    """A flag alone requests a window when the engine supplies default thresholds."""
    values = _projectability_assignments(text)
    return (
        "dis_proj_min" in values
        or "dis_proj_max" in values
        or _logical_true(values.get("dis_proj_auto", ""))
        or _logical_true(values.get("dis_froz_proj", ""))
    )


def require_projectability_flag(text: str, deck: str = "") -> None:
    """Refuse the engine's silent path before a writer leaves any deck behind.

    Wannier90 reads ``dis_proj_min`` / ``dis_proj_max`` (and ``dis_proj_auto``) and
    applies them only when ``dis_froz_proj`` is true; the flag defaults to false and
    nothing is printed when it is missing. A deck in that state runs energy-window
    disentanglement while reading as a projectability construction, which is how the
    2026-08-28 campaign's twenty decks came to be reported as something they were not.
    """
    values = _projectability_assignments(text)
    if requests_projectability(text) and not _logical_true(
        values.get("dis_froz_proj", "")
    ):
        asked = ", ".join(f"{key} = {value}" for key, value in values.items())
        raise ProjectabilityWindowIgnored(
            f"{deck or 'deck'} asks for a projectability window ({asked}) without "
            "dis_froz_proj = .true.; Wannier90 would parse the thresholds and ignore "
            "them, running energy-window disentanglement instead"
        )


@dataclass(frozen=True)
class ConstructionInputs:
    """The two files that define one construction, and what they say."""

    seed: str
    win: Path
    pw2wan: Path | None
    values: dict[str, Any]

    def gate_inputs(self, spin_index: int = 0) -> dict[str, Any]:
        """The subset the gate's checks read, typed as they expect.

        ``exclude_bands`` is read from the deck itself, not from :attr:`values`, so the
        parameters a repair model is shown do not change; it is present only when the
        deck excludes something, which keeps a deck that excludes nothing exactly as the
        gate saw it before.
        """
        out: dict[str, Any] = {"spin_index": spin_index}
        for key in ("num_wann", "num_bands"):
            if key in self.values:
                out[key] = int(float(self.values[key]))
        if "mp_grid" in self.values:
            out["mp_grid"] = _parse_mp_grid(self.values["mp_grid"])
        excluded = read_excluded_bands(self.win) if self.win.is_file() else ()
        if excluded:
            out["exclude_bands"] = excluded
        for key in (
            "num_iter",
            "dis_num_iter",
            "conv_window",
            "conv_tol",
            "trial_step",
            "dis_win_max",
            "dis_froz_min",
            "dis_froz_max",
            "dis_proj_min",
            "dis_proj_max",
            "dis_mix_ratio",
        ):  # projection_source is for the model, not a check
            if key in self.values:
                out[key] = self.values[key]
        if self.win.is_file():
            pw2wan_text = (
                self.pw2wan.read_text()
                if self.pw2wan is not None and self.pw2wan.is_file()
                else None
            )
            out["projector_groups"] = projector_groups(
                self.win.read_text(), pw2wan_text, self.win.parent
            )
        return out


def _clean(line: str) -> str:
    return line.split("!")[0].split("#")[0].strip()


def read_excluded_bands(path: Path) -> tuple[int, ...]:
    """The 1-based band indices a deck's ``exclude_bands`` removes, ascending.

    Wannier90 drops these bands before it reads anything else, so the model is never
    asked to represent them; the gate needs the list to count only the bands the model
    was asked for. The spelling is Wannier90's own range vector: ranges written with
    ``-`` or ``:``, items separated by blanks, ``,`` or ``;`` -- aiida-wannier90 emits
    ``exclude_bands = 1-4``. An assignment that cannot be read is an error, not an empty
    list: an empty list would silently restore the bands the deck removed.
    """
    for line in path.read_text().splitlines():
        cleaned = _clean(line)
        match = re.match(r"exclude_bands(?:\s*[=:]\s*|\s+)(.*)$", cleaned, re.I)
        if not match:
            continue
        bands: set[int] = set()
        for item in re.split(r"[,;\s]+", match.group(1).strip()):
            if not item:
                continue
            bounds = re.fullmatch(r"([0-9]+)(?:[-:]([0-9]+))?", item)
            if bounds is None:
                raise InputError(f"unreadable exclude_bands item {item!r} in {path}")
            lo = int(bounds.group(1))
            hi = int(bounds.group(2) or lo)
            if lo <= 0 or hi < lo:
                raise InputError(f"invalid exclude_bands range {item!r} in {path}")
            bands.update(range(lo, hi + 1))
        if not bands:
            raise InputError(f"exclude_bands assigned no band in {path}")
        return tuple(sorted(bands))
    return ()


def read_win(path: Path) -> dict[str, Any]:
    """Scalar assignments and the k-mesh, as written."""
    if not path.is_file():
        raise InputError(f"no Wannier90 input at {path}")
    values: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        stripped = _clean(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("mp_grid"):
            values["mp_grid"] = stripped.split(None, 1)[1].lstrip("= ").strip()
        elif "=" in stripped:
            name, _, value = stripped.partition("=")
            if name.strip().lower() in WIN_SCALARS:
                values[name.strip().lower()] = value.strip()
    block = _projections_block(path)
    if block:
        values["projectors"] = block
    return values


def _projections_block(path: Path) -> str:
    """The deck's projections, as one normalised string.

    Recorded because the scalars do not distinguish two projector sets that imply the
    same count: Ti:s;p;d and Ti:d;sp3 are both eighteen functions on two atoms and
    every recorded value was identical, so a genuinely different construction was
    refused as a repeat of the one before it.
    """
    lines, inside = [], False
    for line in path.read_text().splitlines():
        lowered = _clean(line).lower()
        if lowered.startswith("begin projections"):
            inside = True
            continue
        if lowered.startswith("end projections"):
            break
        if inside and _clean(line):
            lines.append(_clean(line).replace(" ", ""))
    return ";".join(lines)


def projection_source(win: Path, pw2wan: Path | None) -> str:
    """Describe where this deck's projectors come from, in words.

    A deck that delegates the choice and a deck that names it can carry identical
    shells, identical windows and identical budgets, and produce very differently
    localized functions. Naming the source is what makes changing it a visible move.
    """
    if pw2wan is not None and pw2wan.is_file():
        text = pw2wan.read_text().lower()
        for key, description in PROJECTION_SOURCE_KEYS.items():
            for line in text.splitlines():
                stripped = line.split("!")[0].strip()
                if stripped.startswith(key) and "true" in stripped:
                    return description
    if win.is_file():
        body = win.read_text()
        block = body.lower().split("begin projections", 1)
        if len(block) == 2:
            named = block[1].split("end projections", 1)[0].strip()
            if named:
                return "analytic orbitals named in the input"
        for line in body.splitlines():
            if line.split("!")[0].strip().lower().startswith("auto_projections"):
                return "delegated to the overlap step, which did not say how"
    return "not recorded"


def read_construction(run_dir: Path, seed: str) -> ConstructionInputs:
    win = run_dir / f"{seed}.win"
    pw2wan = run_dir / f"{seed}.pw2wan"
    values = read_win(win)
    values["projection_source"] = projection_source(
        win, pw2wan if pw2wan.is_file() else None
    )
    return ConstructionInputs(
        seed=seed,
        win=win,
        pw2wan=pw2wan if pw2wan.is_file() else None,
        values=values,
    )


#: Knobs the model states as an offset from the Fermi level, converted on the way to
#: the file and back on the way out. One convention reaches the model; one reaches
#: Wannier90; neither ever sees the other's.
def to_absolute(name: str, value: Any, fermi_energy_ev: float) -> Any:
    parameter = BY_NAME.get(name)
    if parameter is None or not parameter.relative_to_fermi:
        return value
    return float(value) + float(fermi_energy_ev)


def to_relative(name: str, value: Any, fermi_energy_ev: float) -> Any:
    parameter = BY_NAME.get(name)
    if parameter is None or not parameter.relative_to_fermi or value is None:
        return value
    try:
        return round(
            float(str(value).lower().replace("d", "e")) - float(fermi_energy_ev), 4
        )
    except (TypeError, ValueError):
        return value


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, (list, tuple)):
        # projectors: [{"species": "Fe", "orbitals": ("s", "d")}, ...]; a positioned
        # entry {"site": "f=0.5,0.5,0", "orbitals": ("s",)} is written as its site
        parts = []
        for entry in value:
            orbitals = ";".join(entry["orbitals"])
            where = entry["site"] if "site" in entry else entry["species"]
            parts.append(f"{where}:{orbitals}")
        return "\n".join(parts)
    return str(value)


def _assignment_key(line: str) -> str | None:
    """The lower-case key a line assigns, without any ``(i)`` index, or ``None``."""
    cleaned = _clean(line)
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*\d+\s*\))?\s*[=:]", cleaned)
    if match is None:
        return None
    return match.group(1).lower()


def assignments(text: str) -> dict[str, str]:
    """Scalar ``key = value`` assignments outside blocks, keys lower-cased."""
    values: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        cleaned = _clean(line)
        lowered = cleaned.lower()
        if lowered.startswith("begin"):
            in_block = True
            continue
        if lowered.startswith("end") and in_block:
            in_block = False
            continue
        if in_block or not cleaned:
            continue
        match = re.match(r"([A-Za-z_][A-Za-z0-9_()]*)\s*[=:]\s*(.*)$", cleaned)
        if match:
            values[match.group(1).lower()] = match.group(2).strip().strip(",")
    return values


def logical(value: Any) -> bool:
    return _logical_true(str(value)) if value is not None else False


def deck_spinors(win_text: str) -> bool:
    """Whether a deck declares spinor Wannier functions (``spinors = .true.``).

    A noncollinear or spin-orbit parent has one channel of two-component spinor bands;
    the deck must say so, or pw2wannier90 refuses it ("Spinorbit without spinor=T") and
    Wannier90 counts each named orbital once instead of once per spin component.
    """
    return logical(assignments(win_text).get("spinors"))


# ---------------------------------------------------------------------------
# the method a deck is built by
# ---------------------------------------------------------------------------


def deck_method(win_text: str, pw2wan_text: str | None = None) -> Method:
    """Which family a deck is built by, read from the deck alone.

    SCDM when the overlap step is told to select density-matrix columns
    (``scdm_proj`` on Quantum ESPRESSO; ``LSCDM`` among a VASP deck's overlap-step
    tags); PDWF when the deck asks for a projectability window (which
    :func:`require_projectability_flag` already guarantees is applied); SMV otherwise.
    """
    pw = assignments(pw2wan_text or "")
    if logical(pw.get("scdm_proj")):
        return Method.SCDM
    for line in win_text.splitlines():
        match = re.match(
            r"\s*!\s*overlap-step INCAR:\s*LSCDM\s*=\s*(\S+)", line, re.IGNORECASE
        )
        if match and _logical_true(match.group(1)):
            return Method.SCDM
    if requests_projectability(win_text):
        return Method.PDWF
    return Method.SMV


def method_of(inputs: "ConstructionInputs") -> Method:
    pw = (
        inputs.pw2wan.read_text()
        if inputs.pw2wan is not None and inputs.pw2wan.is_file()
        else None
    )
    return deck_method(inputs.win.read_text(), pw)


# ---------------------------------------------------------------------------
# the pseudo-atomic orbitals a PDWF deck draws from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaoState:
    """One pseudo-atomic orbital state, as pw2wannier90 numbers them."""

    index: int
    """1-based, atom-major, then the species' orbital list, then m."""
    atom: int
    species: str
    label: str
    l: int
    m: int
    """1..2l+1, or 1..2j+1 (m_j from -j up) for a j-resolved spinor orbital."""
    j: float | None = None
    """Total angular momentum of a fully relativistic (j-resolved) spinor orbital."""
    spin: str | None = None
    """``up``/``down`` for a spinor deck whose orbitals carry no j (noncollinear
    without spin-orbit): pw2wannier90 then doubles each orbital, up block first."""


def deck_atom_species(win_text: str) -> list[str]:
    """The species of each atom, in the order the deck's atoms block lists them."""
    species, inside = [], False
    for line in win_text.splitlines():
        lowered = _clean(line).lower()
        if re.match(r"begin\s+atoms_(cart|frac)", lowered):
            inside = True
            continue
        if re.match(r"end\s+atoms_(cart|frac)", lowered):
            break
        if inside:
            tokens = _clean(line).split()
            if len(tokens) >= 4:
                species.append(tokens[0])
    return species


def pao_excluded(pw2wan_text: str) -> set[int]:
    """The PAO states a deck's ``atom_proj_exclude`` list removes."""
    out = set()
    for line in pw2wan_text.splitlines():
        match = re.match(
            r"\s*atom_proj_exclude\s*\(\s*\d+\s*\)\s*=\s*(\d+)", line, re.IGNORECASE
        )
        if match:
            out.add(int(match.group(1)))
    return out


def pao_layout(run_dir: Path, win_text: str, pw2wan_text: str) -> tuple[PaoState, ...]:
    """Every PAO state the overlap step can project on, labelled.

    Read from the external projector list the deck names (``atom_proj_ext`` with
    ``atom_proj_dir``): each ``<species>.dat`` gives the angular momentum of every
    orbital in file order, and the list's ``projectors.json`` (as aiida-wannier90-
    workflows ships it) gives their labels. Without a label table the orbitals are
    named by l and order (``s1``, ``s2``, ``p1``...). The numbering is pw2wannier90's
    own (atom, then orbital, then m), which is what ``atom_proj_exclude`` indexes.

    A spinor deck (``spinors = .true.``) is numbered as pw2wannier90 numbers spinor
    projectors (``read_atomproj``/``atomproj_wfc``, QE 7.5): with a fully relativistic
    set, whose ``.dat`` carries a third line of j values, every orbital supplies 2j+1
    states (m_j = -j..j); without j (noncollinear, no spin-orbit) every orbital is
    doubled, its 2l+1 spin-up states then its 2l+1 spin-down ones. The orbital's label
    covers all of its states, so a PAO subset keeps or drops an orbital whole.
    """
    pw = assignments(pw2wan_text)
    if not logical(pw.get("atom_proj")):
        raise InputError(
            "the deck does not project on pseudo-atomic orbitals (atom_proj)"
        )
    if not logical(pw.get("atom_proj_ext")) or not pw.get("atom_proj_dir"):
        raise InputError(
            "the PAO list is read from the external projector directory "
            "(atom_proj_ext with atom_proj_dir), and this deck names none"
        )
    directory = Path(run_dir) / pw["atom_proj_dir"].strip("'\"")
    if not directory.is_dir():
        raise InputError(f"the deck's projector directory {directory} is missing")
    labels_path = directory / "projectors.json"
    table: dict[str, list] = {}
    if labels_path.is_file():
        try:
            table = json.loads(labels_path.read_text())
        except ValueError as exc:
            raise InputError(f"{labels_path} is not readable JSON: {exc}") from exc
    spinors = deck_spinors(win_text)
    per_species: dict[str, list[tuple[str, int, float | None]]] = {}
    states: list[PaoState] = []
    index = 0
    for atom, species in enumerate(deck_atom_species(win_text), start=1):
        if species not in per_species:
            dat = directory / f"{species}.dat"
            if not dat.is_file():
                raise InputError(f"no projector file {dat} for {species}")
            lines = dat.read_text().splitlines()
            try:
                ls = [int(x) for x in lines[1].split()]
            except (IndexError, ValueError) as exc:
                raise InputError(f"{dat}: unreadable angular momentum line") from exc
            js = _j_line(lines, len(ls)) if spinors else None
            if js is not None and any(abs(j - l) != 0.5 for j, l in zip(js, ls)):
                raise InputError(f"{dat}: a j value is not l +- 1/2")
            entries = table.get(species) or []
            labels = [str(e.get("label")) for e in entries if isinstance(e, Mapping)]
            if len(labels) != len(ls) or any(
                int(e.get("l", -1)) != l for e, l in zip(entries, ls)
            ):
                counts: dict[int, int] = {}
                labels = []
                for l in ls:
                    counts[l] = counts.get(l, 0) + 1
                    labels.append(f"{'spdf'[l]}{counts[l]}")
            per_species[species] = [
                (label, l, None if js is None else js[i])
                for i, (label, l) in enumerate(zip(labels, ls))
            ]
        for label, l, j in per_species[species]:
            if j is not None:
                blocks = [(int(round(2 * j + 1)), None)]
            elif spinors:
                blocks = [(2 * l + 1, "up"), (2 * l + 1, "down")]
            else:
                blocks = [(2 * l + 1, None)]
            for size, spin in blocks:
                for m in range(1, size + 1):
                    index += 1
                    states.append(PaoState(index, atom, species, label, l, m, j, spin))
    if not states:
        raise InputError("the deck lists no atoms to project on")
    return tuple(states)


def _j_line(lines: Sequence[str], n_orbitals: int) -> list[float] | None:
    """The j value of each orbital, when the ``.dat`` is a fully relativistic set.

    pw2wannier90 reads this third line only for a spin-orbit run; a scalar-relativistic
    file goes straight to its radial table (2 + n columns). A line of exactly n
    half-integers is the j line.
    """
    try:
        values = [float(x) for x in lines[2].split()]
    except (IndexError, ValueError):
        return None
    if len(values) != n_orbitals or any(
        abs(2 * v - round(2 * v)) > 1e-6 or round(2 * v) % 2 != 1 for v in values
    ):
        return None
    return values


# ---------------------------------------------------------------------------
# what the functions are seeded on, per site (read for the gate)
# ---------------------------------------------------------------------------

#: Named orbitals of a projections line: (angular momentum, functions per site). A
#: hybrid has no single l and is recorded as ``None``.
_ORBITAL_TOKENS: dict[str, tuple[int | None, int]] = {
    "s": (0, 1),
    "p": (1, 3),
    "d": (2, 5),
    "f": (3, 7),
    "pz": (1, 1),
    "px": (1, 1),
    "py": (1, 1),
    "dz2": (2, 1),
    "dxz": (2, 1),
    "dyz": (2, 1),
    "dx2-y2": (2, 1),
    "dxy": (2, 1),
    "fz3": (3, 1),
    "fxz2": (3, 1),
    "fyz2": (3, 1),
    "fz(x2-y2)": (3, 1),
    "fxyz": (3, 1),
    "fx(x2-3y2)": (3, 1),
    "fy(3x2-y2)": (3, 1),
    "sp": (None, 2),
    "sp2": (None, 3),
    "sp3": (None, 4),
    "sp3d": (None, 5),
    "sp3d2": (None, 6),
}
_SPIN_COMPONENTS = re.compile(r"\(([ud](?:,[ud])?)\)")
_QUANTISATION_AXIS = re.compile(r"\[[^\]]*\]")


def _orbital_token(token: str) -> tuple[int | None, int] | None:
    """(l, functions per site) of one orbital token, or ``None`` if unreadable."""
    token = token.strip().lower()
    if token.startswith("l="):
        parts = token.split(",")
        try:
            l = int(parts[0][2:])
        except ValueError:
            return None
        mr = [i for i, part in enumerate(parts) if part.startswith("mr=")]
        if mr:
            listed = [parts[mr[0]][3:], *parts[mr[0] + 1 :]]
            count = len([v for v in listed if v.strip()])
        else:
            count = 2 * l + 1 if l >= 0 else -l + 1
        return (l if l >= 0 else None), count
    return _ORBITAL_TOKENS.get(token)


def _projection_line(line: str, spinors: bool) -> dict[str, Any] | None:
    """One projections line as a site group, or ``None`` if it cannot be read."""
    compact = line.replace(" ", "")
    components = 2 if spinors else 1
    qualifier = _SPIN_COMPONENTS.search(compact)
    if qualifier is not None:
        if not spinors:
            return None
        components = len(qualifier.group(1).split(","))
    compact = _QUANTISATION_AXIS.sub("", _SPIN_COMPONENTS.sub("", compact))
    parts = compact.split(":")
    if len(parts) < 2 or not parts[0] or parts[0].lower() == "random":
        return None
    site, orbitals = parts[0], parts[1]
    ls: list[int | None] = []
    count = 0
    for token in orbitals.split(";"):
        parsed = _orbital_token(token)
        if parsed is None:
            return None
        ls.append(parsed[0])
        count += parsed[1]
    group: dict[str, Any] = {
        "site": site,
        "species": None,
        "l": sorted({-1 if l is None else l for l in ls}),
        "functions_per_site": count * components,
        "orbitals": [t.strip() for t in orbitals.split(";")],
    }
    lowered = site.lower()
    if lowered.startswith(("c=", "f=")):
        try:
            group["position"] = [float(x) for x in site[2:].split(",")]
        except ValueError:
            return None
        if len(group["position"]) != 3:
            return None
        group["frame"] = "cartesian_ang" if lowered.startswith("c=") else "fractional"
    else:
        group["species"] = site
    return group


def projector_groups(
    win_text: str, pw2wan_text: str | None, run_dir: Path
) -> dict[str, Any]:
    """What a construction's functions are seeded on, site by site, read from its deck.

    ``route`` says where the projectors come from: ``projections`` (an analytic
    projections block, SMV), ``pao`` (the pseudopotential's atomic orbitals kept by
    ``atom_proj_exclude``: PDWF, and QE SMV started from the frontier's PAOs), ``scdm``
    (no projectors: the count alone defines the set) or ``unknown``. Each group is a set
    of identical sites -- a species label with the number of deck atoms carrying it, a
    single ``c=``/``f=`` position, or one atom's kept PAOs -- with the angular momenta it
    names (``-1`` for a hybrid) and the functions per site, spinor components included.
    A deck this cannot read is reported with ``error`` and no groups, never guessed at.
    """
    method = deck_method(win_text, pw2wan_text)
    spinors = deck_spinors(win_text)
    atoms = deck_atom_species(win_text)
    counts: dict[str, int] = {}
    for label in atoms:
        counts[label] = counts.get(label, 0) + 1
    out: dict[str, Any] = {
        "method": method.value,
        "route": "unknown",
        "spinors": spinors,
        "groups": [],
        "error": None,
    }
    if method is Method.SCDM:
        out["route"] = "scdm"
        return out
    block = _block_lines(win_text)
    if block:
        out["route"] = "projections"
        groups = []
        unit = "ang"
        for line in block:
            if line.strip().lower() in ("ang", "angstrom", "bohr"):
                unit = "bohr" if line.strip().lower() == "bohr" else "ang"
                continue
            group = _projection_line(line, spinors)
            if (
                group is not None
                and group.get("frame") == "cartesian_ang"
                and unit == "bohr"
            ):
                group["frame"] = "cartesian_bohr"
            if group is None:
                out["error"] = f"unreadable projections line {line!r}"
                return out
            if group["species"] is not None:
                if group["species"] not in counts:
                    out["error"] = (
                        f"projections line {line!r} names {group['species']}, which the "
                        "deck's atoms block does not carry"
                    )
                    return out
                group["n_sites"] = counts[group["species"]]
            else:
                group["n_sites"] = 1
            groups.append(group)
        out["groups"] = groups
        return out
    if pw2wan_text is not None and logical(assignments(pw2wan_text).get("atom_proj")):
        out["route"] = "pao"
        try:
            layout = pao_layout(run_dir, win_text, pw2wan_text)
        except InputError as exc:
            out["error"] = str(exc)
            return out
        excluded = pao_excluded(pw2wan_text)
        per_atom: dict[int, dict[str, Any]] = {}
        for state in layout:
            entry = per_atom.setdefault(
                state.atom,
                {
                    "site": state.species,
                    "species": state.species,
                    "atom": state.atom,
                    "l": set(),
                    "labels": [],
                    "all_labels": [],
                    "functions_per_site": 0,
                    "n_sites": 1,
                },
            )
            if state.label not in entry["all_labels"]:
                entry["all_labels"].append(state.label)
            if state.index in excluded:
                continue
            entry["l"].add(state.l)
            if state.label not in entry["labels"]:
                entry["labels"].append(state.label)
            entry["functions_per_site"] += 1
        out["groups"] = [
            {**entry, "l": sorted(entry["l"])}
            for _, entry in sorted(per_atom.items())
            if entry["functions_per_site"]
        ]
        return out
    out["error"] = "the deck names no projections block and no pseudo-atomic orbitals"
    return out


def pao_subset_of(
    layout: Sequence[PaoState], excluded: Iterable[int]
) -> dict[str, tuple[str, ...]]:
    """Per species, the orbital labels a deck keeps (first atom of each species)."""
    excluded = set(excluded)
    kept: dict[str, list[str]] = {}
    first_atom: dict[str, int] = {}
    for state in layout:
        first_atom.setdefault(state.species, state.atom)
        if state.atom != first_atom[state.species] or state.index in excluded:
            continue
        labels = kept.setdefault(state.species, [])
        if state.label not in labels:
            labels.append(state.label)
    for species in first_atom:
        kept.setdefault(species, [])
    return {s: tuple(v) for s, v in kept.items()}


def exclusion_for_subset(
    layout: Sequence[PaoState],
    current: Iterable[int],
    subset: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """The ``atom_proj_exclude`` list that keeps ``subset`` (named species only)."""
    named = {e["species"]: set(e["orbitals"]) for e in subset}
    labels = {}
    for state in layout:
        labels.setdefault(state.species, [])
        if state.label not in labels[state.species]:
            labels[state.species].append(state.label)
    for species, wanted in named.items():
        if species not in labels:
            raise InputError(
                f"pao_subset names {species}, which the deck does not carry"
            )
        unknown = sorted(wanted - set(labels[species]))
        if unknown:
            raise InputError(
                f"pao_subset names {', '.join(unknown)} for {species}, which its "
                f"projector list does not supply (it has: {', '.join(labels[species])})"
            )
    current = set(current)
    out = []
    for state in layout:
        if state.species in named:
            if state.label not in named[state.species]:
                out.append(state.index)
        elif state.index in current:
            out.append(state.index)
    return tuple(out)


# ---------------------------------------------------------------------------
# writing a derived construction
# ---------------------------------------------------------------------------


def _format_ranges(bands: Iterable[int]) -> str:
    bands = sorted(set(bands))
    parts, start = [], None
    for i, band in enumerate(bands):
        if start is None:
            start = band
        if i + 1 == len(bands) or bands[i + 1] != band + 1:
            parts.append(f"{start}-{band}" if band != start else f"{band}")
            start = None
    return ", ".join(parts)


def _rebalanced_exclusion(
    text: str, old_bands: int | None, new_bands: int
) -> str | None:
    """Move a top ``exclude_bands`` range so it follows a new ``num_bands``.

    The overlap step writes every band the parent produced that the deck does not
    exclude, and Wannier90 requires ``num_bands`` plus the excluded ones to add up to
    that. A deck that excludes a block at the bottom (semicore) and a block at the top
    (headroom) shifts only the top block when ``num_bands`` changes; one with no top
    block is left as it is, and a band count it cannot satisfy is refused by the deck
    check rather than repaired by an exclusion nobody asked for.

    Returns the new assignment value (``""`` to drop the line), or ``None`` to leave
    the deck's exclusion untouched.
    """
    if old_bands is None:
        return None
    excluded = sorted(_excluded_from_text(text))
    if not excluded:
        return None
    bottom = 0
    while bottom < len(excluded) and excluded[bottom] == bottom + 1:
        bottom += 1
    start = bottom + old_bands + 1
    top = excluded[bottom:]
    if not top or top != list(range(start, start + len(top))):
        return None
    last = top[-1]
    new_top = list(range(bottom + new_bands + 1, last + 1))
    return _format_ranges(list(range(1, bottom + 1)) + new_top)


def _excluded_from_text(text: str) -> tuple[int, ...]:
    for line in text.splitlines():
        cleaned = _clean(line)
        match = re.match(r"exclude_bands(?:\s*[=:]\s*|\s+)(.*)$", cleaned, re.I)
        if not match:
            continue
        bands: set[int] = set()
        for item in re.split(r"[,;\s]+", match.group(1).strip()):
            bounds = re.fullmatch(r"([0-9]+)(?:[-:]([0-9]+))?", item)
            if bounds is None:
                return ()
            lo, hi = int(bounds.group(1)), int(bounds.group(2) or bounds.group(1))
            bands.update(range(lo, hi + 1))
        return tuple(sorted(bands))
    return ()


def _family_violation(name: str, method: Method) -> str | None:
    from .actions import cross_family_reason
    from .families import knobs_for

    if name in knobs_for(method):
        return None
    return cross_family_reason(name, method)


def _semicore_plan(win_changes: dict) -> Any:
    """Pop a resolved ``semicore_exclusion`` edit from the .win changes, or ``None``.

    The knob is resolved against the run's parent and first deck
    (:func:`.semicore.resolve`, called by :func:`.methods.derive_deck`); a raw shell list
    reaching a writer has skipped that, and is refused rather than guessed at.
    """
    if "semicore_exclusion" not in win_changes:
        return None
    from .semicore import SemicorePlan

    plan = win_changes.pop("semicore_exclusion")
    if not isinstance(plan, SemicorePlan):
        raise InputError(
            "semicore_exclusion is resolved against the run's parent and first deck; "
            "write it through derive_deck, which knows them"
        )
    return plan


def _block_lines(text: str) -> list[str]:
    lines, inside = [], False
    for line in text.splitlines():
        cleaned = _clean(line)
        lowered = cleaned.lower()
        if lowered.startswith("begin projections"):
            inside = True
            continue
        if lowered.startswith("end projections"):
            break
        if inside and cleaned:
            lines.append(cleaned)
    return lines


def _apply_semicore(
    plan,
    *,
    win_text: str,
    pw2wan_text: str | None,
    run_dir: Path,
    win_changes: dict,
    pw2wan_changes: dict,
    pw2wan_remove: set,
    win_remove: set,
    projections,
    subset,
    species_counts,
) -> dict:
    """Write a resolved semicore edit into the change sets of either engine's writer.

    Sets the band assignment (bottom 1..N plus the unchanged top block; a ``num_bands``
    asked for in the same action then moves the top block as it always does), and the
    family's own consequence: the pseudo-atomic orbitals kept, the analytic semicore
    projectors, or SCDM's count. Returns the record lines and, for an analytic deck, the
    projection lines to write.
    """
    from .semicore import record_lines, semicore_projection_lines

    out: dict[str, Any] = {}
    bottom = list(range(1, plan.bottom + 1))
    top = [b for b in plan.exclude_bands if b > plan.bottom]
    num_bands = plan.num_bands
    if "num_bands" in win_changes:
        requested = int(win_changes["num_bands"])
        if top:
            top = list(range(plan.bottom + requested + 1, top[-1] + 1))
        num_bands = requested
    win_changes["num_bands"] = num_bands
    excluded = bottom + top
    if excluded:
        win_changes["exclude_bands"] = _format_ranges(excluded)
    else:
        win_changes.pop("exclude_bands", None)
        win_remove.add("exclude_bands")

    semicore_lines: list[str] = []
    if plan.route == "pao" and projections is not None:
        raise InputError(
            "naming projectors turns this deck's pseudo-atomic orbitals into analytic "
            "ones, and which semicore shells the named set already covers is then not "
            "known; name the projectors in one action and move semicore_exclusion in "
            "the next"
        )
    if plan.route == "pao":
        if pw2wan_text is None:
            raise InputError("a PAO deck's semicore orbitals live in its overlap input")
        layout = pao_layout(run_dir, win_text, pw2wan_text)
        base = set(pw2wan_changes.get("atom_proj_exclude", pao_excluded(pw2wan_text)))
        if subset:
            named = {e["species"] for e in subset}
            by_index = {s.index: s for s in layout}
            clash_in = sorted(
                {
                    f"{by_index[i].species} {by_index[i].label}"
                    for i in plan.pao_in
                    if i in base and by_index[i].species in named
                }
            )
            clash_out = sorted(
                {
                    f"{by_index[i].species} {by_index[i].label}"
                    for i in plan.pao_excluded_shells
                    if i not in base and by_index[i].species in named
                }
            )
            if clash_in or clash_out:
                raise InputError(
                    "pao_subset and semicore_exclusion disagree: "
                    + "; ".join(
                        (
                            [
                                f"the subset leaves out {', '.join(clash_in)}, which the "
                                "semicore edit brings in"
                            ]
                            if clash_in
                            else []
                        )
                        + (
                            [
                                f"the subset keeps {', '.join(clash_out)}, whose bands the "
                                "semicore edit excludes"
                            ]
                            if clash_out
                            else []
                        )
                    )
                )
        final = (base - set(plan.pao_in)) | set(plan.pao_out)
        pw2wan_remove.add("atom_proj_exclude")
        pw2wan_changes["atom_proj_exclude"] = sorted(final)
        win_changes["num_wann"] = len(layout) - len(final)
    elif plan.route == "projections":
        recorded = set(plan.recorded_projections)
        if projections is not None:
            valence = list(projections)
        else:
            valence = [
                line
                for line in _block_lines(win_text)
                if line.replace(" ", "") not in recorded
            ]
            unrecorded = list(plan.newly_excluded_unrecorded)
            if unrecorded:
                raise InputError(
                    f"the projectors of {', '.join(unrecorded)} are part of the "
                    "projector set the deck names, not lines this knob added, so which "
                    "line to remove is not known; name the projectors without them "
                    "(projectors) in the same action"
                )
        semicore_lines = semicore_projection_lines(valence, plan.semicore_projectors)
        out["projections"] = valence + semicore_lines
        from .deckcheck import implied_by_block

        counts = dict(species_counts or {})
        if not counts:
            for symbol in deck_atom_species(win_text):
                counts[symbol] = counts.get(symbol, 0) + 1
        implied = implied_by_block(out["projections"], counts, deck_spinors(win_text))
        if implied is None:
            raise InputError(
                "the projections block cannot be counted after the semicore edit"
            )
        win_changes["num_wann"] = implied
    elif plan.route == "scdm":
        current = win_changes.get("num_wann", read_win_text_value(win_text, "num_wann"))
        try:
            current = int(float(current))
        except (TypeError, ValueError) as exc:
            raise InputError("the SCDM deck declares no num_wann to move") from exc
        win_changes["num_wann"] = current + plan.bands_in
    else:
        raise InputError(f"unknown semicore route {plan.route!r}")
    out["lines"] = record_lines(plan.excluded, semicore_lines)
    return out


def read_win_text_value(text: str, key: str) -> str | None:
    return assignments(text).get(key)


def write_derived(
    source: ConstructionInputs,
    run_dir: Path,
    new_seed: str,
    changes: Mapping[str, Any],
    fermi_energy_ev: float | None = None,
    species_counts: Mapping[str, int] | None = None,
) -> ConstructionInputs:
    """Write a new construction from an old one plus a set of changes, in its family.

    Lines the changes do not name are copied verbatim, including blocks this module
    does not parse -- the k-point list, the cell, the atom positions. A rewriter that
    reconstructed those from what it understood would quietly drop whatever it did not.

    The deck's method family is read first and every change must be one of its knobs:
    no edit here can turn a PDWF deck into an energy-window one or the reverse. Naming
    projectors is SMV's; choosing among the pseudo-atomic orbitals is PDWF's
    (``pao_subset``, written as ``atom_proj_exclude``); mu and sigma are SCDM's.
    """
    if not changes:
        raise InputError("no changes to apply")
    for name in changes:
        if name not in BY_NAME:
            raise InputError(f"{name!r} is not an editable parameter")
    win_text = source.win.read_text()
    # A source whose projectability window Wannier90 would ignore is not a PDWF deck
    # nor an SMV one; nothing is derived from it.
    require_projectability_flag(win_text, source.win.name)
    pw2wan_text = source.pw2wan.read_text() if source.pw2wan is not None else None
    method = deck_method(win_text, pw2wan_text)
    for name in changes:
        violation = _family_violation(name, method)
        if violation:
            raise InputError(violation)

    if fermi_energy_ev is not None:
        changes = {k: to_absolute(k, v, fermi_energy_ev) for k, v in changes.items()}
    changes = dict(changes)

    win_changes = {k: v for k, v in changes.items() if BY_NAME[k].stage_file == "win"}
    pw2wan_changes: dict[str, Any] = {
        k: v for k, v in changes.items() if BY_NAME[k].stage_file == "pw2wan"
    }
    win_remove: set[str] = set()
    pw2wan_remove: set[str] = set()
    projections = None
    plan = _semicore_plan(win_changes)

    # Explicit projectors (SMV) reach both files: the .win stops delegating the choice
    # and names it, and the overlap input stops supplying orbitals of its own. Left
    # half-done, Wannier90 reads a projections block while pw2wannier90 fills the .amn
    # from somewhere else entirely. No projectability key is touched: an SMV deck has
    # none, and :func:`_family_violation` has refused this edit on any deck that does.
    if "projectors" in pw2wan_changes:
        named = pw2wan_changes.pop("projectors")
        projections = _format(named).splitlines()
        win_remove.add("auto_projections")
        pw2wan_remove |= {
            "atom_proj",
            "atom_proj_ext",
            "atom_proj_dir",
            "atom_proj_ortho",
            "atom_proj_exclude",
            "atom_proj_frozen",
            "scdm_proj",
            "scdm_entanglement",
            "scdm_mu",
            "scdm_sigma",
        }
        if species_counts:
            win_changes["num_wann"] = wannier_count(
                named, species_counts, spinors=deck_spinors(win_text)
            )

    # PDWF chooses among the pseudo-atomic orbitals; the projectability window and
    # dis_froz_proj stay exactly as they were.
    if "pao_subset" in pw2wan_changes:
        subset = pw2wan_changes.pop("pao_subset")
        if pw2wan_text is None:
            raise InputError(
                "pao_subset belongs to the overlap input, and the deck has none"
            )
        layout = pao_layout(run_dir, win_text, pw2wan_text)
        excluded = exclusion_for_subset(layout, pao_excluded(pw2wan_text), subset)
        if len(excluded) >= len(layout):
            raise InputError(
                "pao_subset keeps no pseudo-atomic orbital at all; keep at least one"
            )
        pw2wan_remove.add("atom_proj_exclude")
        pw2wan_changes["atom_proj_exclude"] = list(excluded)
        win_changes["num_wann"] = len(layout) - len(excluded)

    record = None
    if plan is not None:
        record = _apply_semicore(
            plan,
            win_text=win_text,
            pw2wan_text=pw2wan_text,
            run_dir=run_dir,
            win_changes=win_changes,
            pw2wan_changes=pw2wan_changes,
            pw2wan_remove=pw2wan_remove,
            win_remove=win_remove,
            projections=projections,
            subset=changes.get("pao_subset"),
            species_counts=species_counts,
        )
        projections = record.pop("projections", projections)
    elif "num_bands" in win_changes:
        old = read_win(source.win).get("num_bands")
        try:
            old_bands = int(float(old)) if old is not None else None
        except ValueError:
            old_bands = None
        rebalanced = _rebalanced_exclusion(
            win_text, old_bands, int(win_changes["num_bands"])
        )
        if rebalanced is not None:
            if rebalanced:
                win_changes["exclude_bands"] = rebalanced
            else:
                win_remove.add("exclude_bands")

    _rewrite_win(
        source.win,
        run_dir / f"{new_seed}.win",
        win_changes,
        parent=source.seed,
        remove=win_remove,
        projections=projections,
    )
    if record is not None:
        from .semicore import with_record

        target = run_dir / f"{new_seed}.win"
        target.write_text(with_record(target.read_text(), record["lines"]))

    if pw2wan_changes or pw2wan_remove:
        if source.pw2wan is None:
            raise InputError(
                f"{', '.join(pw2wan_changes)} belong to the overlap input, and "
                f"{source.seed} has none"
            )
        _rewrite_pw2wan(
            source.pw2wan,
            run_dir / f"{new_seed}.pw2wan",
            pw2wan_changes,
            new_seed,
            remove=pw2wan_remove,
        )
    elif source.pw2wan is not None:
        _rewrite_pw2wan(source.pw2wan, run_dir / f"{new_seed}.pw2wan", {}, new_seed)

    derived = read_construction(run_dir, new_seed)
    after = method_of(derived)
    if after is not method:
        # The writer's own guarantee, checked on what it wrote rather than assumed.
        raise InputError(
            f"the derived deck reads as {after.value.upper()} where its source was "
            f"{method.value.upper()}; a method changes only through switch_method"
        )
    return derived


def wannier_count(
    projectors, species_counts: Mapping[str, int], spinors: bool = False
) -> int:
    """How many Wannier functions a projector set implies.

    The count is not a free parameter beside the projectors -- it is determined by them,
    and Wannier90 refuses a deck whose declared count disagrees. A spinor deck counts
    each orbital twice (both spin components).
    """
    return implied_wannier_count(projectors, species_counts, spinors)


def rewrite_win_text(
    text: str,
    changes: Mapping[str, Any],
    parent: str = "",
    *,
    remove: Iterable[str] = (),
    projections: Sequence[str] | None | bool = None,
    note: str | None = None,
) -> str:
    """``text`` with ``changes`` set and ``remove`` dropped.

    ``projections``: ``None`` keeps the block, ``False`` drops it, a list of lines
    replaces it. Every value set is written in a header naming the deck it came from,
    so a chain of these reads as a lineage rather than an unattributed pile of edits.
    """
    changes = dict(changes)
    replaced = {k.lower() for k in changes} | {k.lower() for k in remove}
    drop_block = projections is not None
    kept, in_projections = [], False
    for line in text.splitlines():
        lowered = _clean(line).lower()
        if drop_block:
            if lowered.startswith("begin projections"):
                in_projections = True
                continue
            if lowered.startswith("end projections"):
                in_projections = False
                continue
            if in_projections:
                continue
        key = _assignment_key(line)
        if key is not None and key in replaced:
            continue
        kept.append(line)

    origin = f" from {parent}" if parent else ""
    named = sorted(set(changes) | ({"projectors"} if projections else set()))
    header = [f"! derived{origin}: {', '.join(named) or 'nothing changed'}"]
    if note:
        header += [f"! {line}" for line in note.splitlines()]
    for name, value in changes.items():
        header.append(f"{name} = {_format(value)}")
    if projections:
        header += ["", "begin projections", *projections, "end projections"]
    return "\n".join(header + [""] + kept) + "\n"


def _rewrite_win(
    source: Path,
    target: Path,
    changes: Mapping[str, Any],
    parent: str = "",
    *,
    remove: Iterable[str] = (),
    projections: Sequence[str] | None | bool = None,
    note: str | None = None,
) -> None:
    """Write ``target`` as ``source`` rewritten by :func:`rewrite_win_text`."""
    text = rewrite_win_text(
        source.read_text(),
        changes,
        parent,
        remove=remove,
        projections=projections,
        note=note,
    )
    require_projectability_flag(text, str(target))
    target.write_text(text)


def _pw2wan_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str) and not re.fullmatch(
        r"\.(true|false)\.|'.*'|\".*\"", value, re.I
    ):
        try:
            float(value)
            return value
        except ValueError:
            return f"'{value}'"
    return str(value)


def rewrite_pw2wan_text(
    text: str,
    changes: Mapping[str, Any],
    new_seed: str,
    *,
    remove: Iterable[str] = (),
) -> str:
    changes = dict(changes)
    replaced = {k.lower() for k in changes} | {k.lower() for k in remove} | {"seedname"}
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/"):
            continue
        key = _assignment_key(stripped)
        if key is not None and key in replaced:
            continue
        kept.append(line)
    body = [f"  seedname = '{new_seed}'"]
    for name, value in changes.items():
        if isinstance(value, (list, tuple)) and not (
            value and isinstance(value[0], Mapping)
        ):
            body += [f"  {name}({i}) = {v}" for i, v in enumerate(value, start=1)]
        else:
            body.append(f"  {name} = {_pw2wan_value(value)}")
    return "\n".join(kept + body + ["/"]) + "\n"


def _rewrite_pw2wan(
    source: Path,
    target: Path,
    changes: Mapping[str, Any],
    new_seed: str,
    *,
    remove: Iterable[str] = (),
) -> None:
    target.write_text(
        rewrite_pw2wan_text(source.read_text(), changes, new_seed, remove=remove)
    )
