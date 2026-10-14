"""Readers for what a Wannier90 run leaves behind.

The certification gate judges a construction against the calculation that produced
it, so it needs the construction's own record: where the Wannier functions ended up,
how wide they are, whether each minimisation stage actually converged or merely
stopped, and how the resulting real-space Hamiltonian decays. Wannier90 writes all
of that, but across three files in three formats — ``<seed>.wout`` as human-readable
prose, ``<seed>_hr.dat`` as a dense matrix listing, ``<seed>_wsvec.dat`` as a
per-element supercell table. These readers turn them into one object.

Nothing here judges anything. A reader that silently substituted a default for a
missing quantity would hand the gate a verdict rather than a measurement, so an
absent field stays ``None`` and the gate records the check as non-evaluable.

The one comparison made here, :func:`projectability_not_applied`, is not a judgment
of the construction either: it asks whether the output is evidence of the method the
deck requested. Wannier90 prints its ``Projectability  Windows`` box only when it
applies that window, so a deck that asked for one and an output without the box
describe two different calculations, and the record says so instead of scoring it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .inputs import _logical_true, _projectability_assignments, requests_projectability

BOHR_TO_ANG = 0.529177210903


class WannierOutputError(RuntimeError):
    """A Wannier90 output file is absent or does not parse."""


@dataclass(frozen=True)
class MinimisationStage:
    """One minimisation stage's convergence record.

    ``converged`` is Wannier90's own verdict, taken from the line it prints when the
    criterion is met — not re-derived from the trajectory, because the criterion
    involves a convergence *window* whose width lives in the input.
    """

    name: str
    n_iterations: int
    final_value: float | None
    final_delta: float | None
    converged: bool
    recent_deltas: tuple[float, ...] = ()
    """The last handful of steps, newest last.

    The final delta alone cannot tell a minimisation that is nearly there from one
    that is wandering: both report a small number. Titanium's spread changed by about
    1e-5 per step for eight thousand steps, alternating in sign, and no iteration
    budget would have reached the tolerance -- but a model shown only the last step
    reads it as almost converged and asks for more iterations."""

    trajectory: tuple[tuple[int, float, int], ...] = ()
    """Every printed step as ``(iteration, delta, significant figures)``, oldest first.

    For the spread minimisation the delta is the change of the total spread in Å² --
    the quantity Wannier90's own convergence test compares with ``conv_tol`` -- taken
    from the ``<-- DLTA`` line (7 significant figures) where Wannier90 printed one and
    from the ``<-- CONV`` row (3 significant figures) otherwise. Wannier90 prints a step
    only every ``num_print_cycles`` iterations (and the first and last), so the
    iterations here need not be consecutive. Only the spread minimisation records one."""

    @property
    def stopped_at_iteration_limit(self) -> bool:
        """Ran out of iterations rather than meeting the criterion."""
        return not self.converged and self.n_iterations > 0


@dataclass(frozen=True)
class ProjectabilityWindow:
    """The thresholds the engine applied, which need not be those in the deck."""

    lower: float
    """States with projectability below this were discarded (the applied dis_proj_min)."""
    upper: float
    """States at or above this were frozen (the applied dis_proj_max)."""
    automatic: bool
    """Whether Wannier90 chose the thresholds itself (dis_proj_auto, multi-Otsu)."""
    n_classes: int | None
    """The multi-Otsu class count it used; ``None`` for thresholds set in the deck."""


@dataclass
class WannierRun:
    """Everything the gate reads from one Wannier90 run."""

    seedname: str
    num_wann: int | None = None
    num_bands: int | None = None
    centres_ang: np.ndarray | None = None
    spreads_ang2: np.ndarray | None = None
    omega_i: float | None = None
    omega_d: float | None = None
    omega_od: float | None = None
    omega_total: float | None = None
    lattice_ang: np.ndarray | None = None
    disentanglement_used: bool = False
    frozen_window_specified: bool | None = None
    disentanglement: MinimisationStage | None = None
    wannierisation: MinimisationStage | None = None
    hr_rvectors: np.ndarray | None = None
    hr_degeneracies: np.ndarray | None = None
    hr_matrix: np.ndarray | None = None
    hr_ws_shifts: "WsShifts | None" = None
    """``<seed>_wsvec.dat``: the minimal-image shifts Wannier90 applies to each element
    of the real-space Hamiltonian when it interpolates (``use_ws_distance``)."""
    notes: list[str] = field(default_factory=list)
    projectability_window: ProjectabilityWindow | None = None
    wannierise_settings: dict[str, float] = field(default_factory=dict)
    """The spread-minimisation settings Wannier90 says it ran with, from the
    ``WANNIERISE`` box of the ``.wout`` header (``num_iter``, ``conv_tol`` in Å²,
    ``conv_window``, ``num_print_cycles``) and ``iprint`` from the ``MAIN`` box. These
    are the values the engine applied -- the deck's, or Wannier90's defaults for keys the
    deck left unset. A key Wannier90 did not print is absent."""

    @property
    def total_spread_ang2(self) -> float | None:
        if self.spreads_ang2 is None:
            return None
        return float(self.spreads_ang2.sum())


_CENTRE_RE = re.compile(
    r"WF centre and spread\s+(\d+)\s+\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s+([-\d.]+)"
)
_LATTICE_RE = re.compile(r"a_([123])\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def projectability_not_applied(deck_text: str, run: WannierRun) -> str | None:
    """Compare the requested window with engine evidence, not construction quality."""
    if not requests_projectability(deck_text) or run.projectability_window is not None:
        return None
    values = _projectability_assignments(deck_text)
    requested = ", ".join(f"{key} = {value}" for key, value in values.items())
    flag = (
        "set dis_froz_proj true"
        if _logical_true(values.get("dis_froz_proj", ""))
        else "did not set dis_froz_proj true"
    )
    return (
        f"{run.seedname}: the deck requested projectability ({requested}) and {flag}, "
        f"but {run.seedname}.wout has no 'Projectability Windows' block; the engine "
        "ran without the projectability window"
    )


def _read_projectability_window(lines: list[str]) -> ProjectabilityWindow | None:
    """Only the printed box is evidence; energy-window subroutine names are not."""
    inside = False
    lower = upper = None
    n_classes = None
    for line in lines:
        if re.fullmatch(r"\s*\|\s*Projectability\s+Windows\s*\|\s*", line):
            inside = True
            continue
        if not inside:
            continue
        if not line.strip().startswith("|"):
            break
        discarded = re.search(r"Discarded:\s*0\.0+\s+to\s+([\d.]+)", line)
        frozen = re.search(r"Frozen:\s*([\d.]+)\s+to\s+1\.0+", line)
        automatic = re.search(
            r"Thresholds determined automatically\s*\(multi-Otsu,\s*(\d+) classes\)",
            line,
        )
        if discarded:
            lower = float(discarded.group(1))
        if frozen:
            upper = float(frozen.group(1))
        if automatic:
            n_classes = int(automatic.group(1))
    if lower is None or upper is None:
        return None
    return ProjectabilityWindow(lower, upper, n_classes is not None, n_classes)


def _final_state_block(lines: list[str]) -> list[str]:
    """The last ``Final State`` block, which holds the converged Wannier functions.

    Wannier90 prints a centre/spread table at every iteration; only the block under
    ``Final State`` describes the construction the gate is asked to certify.
    """
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "Final State"]
    if not starts:
        raise WannierOutputError("no 'Final State' block: the run did not finish")
    out = []
    for ln in lines[starts[-1] + 1 :]:
        if "WF centre and spread" in ln or "Sum of centres" in ln:
            out.append(ln)
        elif out:
            break
    return out


#: The header rows of the ``.wout`` the spread-minimisation trajectory is judged with,
#: by box: (box title, row label, key). Wannier90 prints these in ``param_write``.
_HEADER_ROWS = (
    ("MAIN", "Output verbosity", "iprint"),
    ("WANNIERISE", "Total number of iterations", "num_iter"),
    ("WANNIERISE", "Convergence tolerence", "conv_tol"),
    ("WANNIERISE", "Convergence window", "conv_window"),
    ("WANNIERISE", "Iterations between writing output", "num_print_cycles"),
)
_BOX_TITLE_RE = re.compile(r"^\s*\*-+\s*([A-Z][A-Z ]*?)\s*-+\*\s*$")
_BOX_ROW_RE = re.compile(r"^\s*\|\s*(.+?)\s*(?:\(.*\))?\s*:\s*(\S+)\s*\|\s*$")
_DLTA_TOTAL_RE = re.compile(r"O_TOT=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s*<-- DLTA")


def _read_wannierise_settings(lines: list[str]) -> dict[str, float]:
    """The settings in the header's ``MAIN`` and ``WANNIERISE`` boxes, first copy only.

    The two boxes share row labels with ``DISENTANGLE`` ("Total number of iterations",
    "Convergence window"), so a row is read only inside the box that owns it.
    """
    wanted = {(box, label): key for box, label, key in _HEADER_ROWS}
    seen_boxes: set[str] = set()
    out: dict[str, float] = {}
    box: str | None = None
    for line in lines:
        title = _BOX_TITLE_RE.match(line)
        if title:
            name = title.group(1).strip()
            box = name if name not in seen_boxes else None
            seen_boxes.add(name)
            continue
        if box is None:
            continue
        if line.strip().startswith("*"):
            box = None
            continue
        row = _BOX_ROW_RE.match(line)
        if not row:
            continue
        for (owner, label), key in wanted.items():
            if owner == box and row.group(1).startswith(label):
                try:
                    out[key] = float(row.group(2).replace("D", "E").replace("d", "e"))
                except ValueError:
                    pass
    return out


def _read_spread_trajectory(
    lines: list[str], length_scale: float
) -> tuple[tuple[int, float, int], ...]:
    """Each printed spread-minimisation step as ``(iteration, delta in Å², figures)``.

    The ``<-- CONV`` row prints the step's change of the total spread in E12.3; the
    ``<-- DLTA`` line Wannier90 writes right after it carries the same difference in
    E15.7, and replaces it when present. Both are printed in the run's length unit
    squared and are converted back to Å², the unit ``conv_tol`` is compared in.
    """
    steps: list[list] = []
    for ln in lines:
        stripped = ln.rstrip()
        if stripped.endswith("<-- CONV"):
            parts = stripped.replace("<-- CONV", "").split()
            if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
                continue
            try:
                steps.append([int(parts[0]), float(parts[1]) * length_scale**2, 3])
            except ValueError:
                continue
        elif stripped.endswith("<-- DLTA") and steps and steps[-1][2] == 3:
            m = _DLTA_TOTAL_RE.search(stripped)
            if m:
                value = float(m.group(1).replace("D", "E").replace("d", "e"))
                steps[-1][1] = value * length_scale**2
                steps[-1][2] = 7
    return tuple((int(i), float(d), int(f)) for i, d, f in steps)


def _read_stage(
    lines: list[str],
    marker: str,
    name: str,
    value_col: int,
    delta_col: int,
    converged_phrase: str,
) -> MinimisationStage | None:
    """Read one minimisation stage from its marked convergence table."""
    rows = [ln for ln in lines if ln.rstrip().endswith(marker)]
    data = []
    for ln in rows:
        parts = ln.replace(marker, "").split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        try:
            data.append(
                (int(parts[0]), float(parts[value_col]), float(parts[delta_col]))
            )
        except (ValueError, IndexError):
            continue
    if not data:
        return None
    converged = any(converged_phrase in ln for ln in lines)
    last = data[-1]
    return MinimisationStage(
        name=name,
        n_iterations=last[0],
        final_value=last[1],
        final_delta=last[2],
        converged=converged,
        recent_deltas=tuple(row[2] for row in data[-8:]),
    )


def read_wout(path: str | Path) -> WannierRun:
    """Parse ``<seed>.wout``.

    Wannier90 reports lengths in whatever ``length_unit`` the input asked for, and
    the two conventions differ by a factor of ~3.5 in spread. The unit is read off
    the section headers rather than assumed, so a Bohr-unit run and an Angstrom-unit
    run reach the gate on the same scale.
    """
    path = Path(path)
    if not path.is_file():
        raise WannierOutputError(f"no Wannier90 output at {path}")
    lines = path.read_text(errors="replace").splitlines()
    run = WannierRun(seedname=path.stem)
    run.projectability_window = _read_projectability_window(lines)

    for ln in lines:
        if "Number of Wannier Functions" in ln:
            run.num_wann = int(ln.split(":")[1].split("|")[0])
        elif "Number of input Bloch states" in ln or "Number of Bloch states" in ln:
            run.num_bands = int(ln.split(":")[1].split("|")[0])
        elif "Using band disentanglement" in ln:
            run.disentanglement_used = "T" in ln.split(":")[1]
        elif "No frozen states were specified" in ln:
            run.frozen_window_specified = False
        elif "Number of states in the frozen window" in ln:
            run.frozen_window_specified = True

    # Lengths: the spread table's unit follows the 'Spreads (X)' header.
    spread_unit = "Bohr^2"
    for ln in lines:
        if "Spreads (" in ln and "Omega I" in ln:
            spread_unit = ln.split("Spreads (")[1].split(")")[0].strip()
            break
    length_scale = BOHR_TO_ANG if spread_unit.startswith("Bohr") else 1.0

    centres, spreads = [], []
    for ln in _final_state_block(lines):
        m = _CENTRE_RE.search(ln)
        if m:
            centres.append([float(m.group(i)) for i in (2, 3, 4)])
            spreads.append(float(m.group(5)))
    if centres:
        run.centres_ang = np.array(centres) * length_scale
        run.spreads_ang2 = np.array(spreads) * length_scale**2

    # The spread summary is boxed with a run of '=' characters, so the value must be
    # matched against the label rather than taken as "whatever follows the equals".
    for label, attr in (
        ("I", "omega_i"),
        ("D", "omega_d"),
        ("OD", "omega_od"),
        ("Total", "omega_total"),
    ):
        pat = re.compile(rf"Omega\s+{label}\s*=\s*([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)")
        for ln in lines:
            m = pat.search(ln)
            if m:
                setattr(
                    run, attr, float(m.group(1).replace("D", "E")) * length_scale**2
                )
                break

    lat_unit_ang = "Lattice Vectors (Ang)" in "\n".join(lines[:80])
    lat = {}
    for ln in lines:
        m = _LATTICE_RE.search(ln)
        if m:
            lat[int(m.group(1))] = [float(m.group(i)) for i in (2, 3, 4)]
        if len(lat) == 3:
            break
    if len(lat) == 3:
        scale = 1.0 if lat_unit_ang else BOHR_TO_ANG
        run.lattice_ang = np.array([lat[1], lat[2], lat[3]]) * scale

    run.disentanglement = _read_stage(
        lines,
        "<-- DIS",
        "disentanglement",
        value_col=2,
        delta_col=3,
        converged_phrase="Disentanglement convergence criteria satisfied",
    )
    run.wannierisation = _read_stage(
        lines,
        "<-- CONV",
        "wannierisation",
        value_col=3,
        delta_col=1,
        converged_phrase="Wannierisation convergence criteria satisfied",
    )
    run.wannierise_settings = _read_wannierise_settings(lines)
    if run.wannierisation is not None:
        run.wannierisation = replace(
            run.wannierisation,
            trajectory=_read_spread_trajectory(lines, length_scale),
        )
    return run


def read_hr(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``<seed>_hr.dat`` into ``(R_vectors, degeneracies, H)``.

    ``H`` has shape ``(n_R, num_wann, num_wann)`` in eV, and ``R_vectors`` are
    integer lattice translations. The degeneracy of each Wigner-Seitz point is
    returned unapplied: dividing by it is a convention the caller may or may not
    want, and a reader that applied it silently would make the two choices
    indistinguishable downstream.
    """
    path = Path(path)
    if not path.is_file():
        raise WannierOutputError(f"no real-space Hamiltonian at {path}")
    raw = path.read_text().splitlines()
    num_wann = int(raw[1])
    n_r = int(raw[2])

    deg: list[int] = []
    cursor = 3
    while len(deg) < n_r:
        deg.extend(int(x) for x in raw[cursor].split())
        cursor += 1
    if len(deg) != n_r:
        raise WannierOutputError(
            f"{path}: degeneracy list holds {len(deg)}, expected {n_r}"
        )

    body = np.array(
        [ln.split() for ln in raw[cursor : cursor + n_r * num_wann * num_wann]],
        dtype=float,
    )
    expected = n_r * num_wann * num_wann
    if body.shape[0] != expected:
        raise WannierOutputError(
            f"{path}: {body.shape[0]} matrix elements for {n_r} R-vectors x "
            f"{num_wann}^2 = {expected} expected"
        )

    rvec = body[:, :3].reshape(n_r, num_wann * num_wann, 3)[:, 0, :].astype(int)
    ham = (body[:, 5] + 1j * body[:, 6]).reshape(n_r, num_wann, num_wann)
    # hr.dat runs the column index fastest, so the reshape above lands transposed.
    return rvec, np.array(deg, dtype=int), np.transpose(ham, (0, 2, 1))


@dataclass(frozen=True)
class WsShifts:
    """Wannier90's per-element minimal-image shifts, aligned with ``read_hr``'s order.

    ``counts[r, i, j]`` is how many equidistant images element ``H_ij(R_r)`` is split
    over, and ``shifts`` lists their lattice-vector shifts ``T`` (integer, in the lattice
    basis) consecutively in ``(r, i, j)`` order. Wannier90's own interpolation is
    ``H_ij(k) = sum_R sum_T exp(2 pi i k.(R+T)) H_ij(R) / (deg(R) counts)``; without the
    shifts a Fourier sum over ``R`` alone departs from Wannier90's band path by up to
    18 meV (silicon, 11^3 mesh).
    """

    counts: np.ndarray
    shifts: np.ndarray
    use_ws_distance: bool | None = None


def read_wsvec(path: str | Path, rvectors: np.ndarray, num_wann: int) -> WsShifts:
    """Parse ``<seed>_wsvec.dat`` for the Hamiltonian whose R-vectors are ``rvectors``.

    Wannier90 writes one block per ``(R, i, j)`` in the same order as ``_hr.dat`` --
    ``R i j``, the image count ``n``, then ``n`` shifts -- so the file is read as one
    integer stream and only the counts need a pass in Python. The R-vector and indices
    of every block are checked against that order rather than trusted.
    """
    path = Path(path)
    text = path.read_text()
    header, _, body = text.partition("\n")
    flag = None
    if "use_ws_distance" in header:
        flag = ".true." in header.lower() or "=t" in header.lower().replace(" ", "")
    flat = np.fromstring(body, dtype=np.int64, sep=" ")
    rvectors = np.asarray(rvectors, dtype=np.int64)
    n_r = len(rvectors)
    n_blocks = n_r * num_wann * num_wann
    counts = np.empty(n_blocks, dtype=np.int64)
    starts = np.empty(n_blocks, dtype=np.int64)
    p = 0
    size = flat.size
    values = flat.tolist() if n_blocks < 200_000 else None
    for b in range(n_blocks):
        if p + 6 > size:
            raise WannierOutputError(f"{path}: ends after {b} of {n_blocks} blocks")
        n = values[p + 5] if values is not None else int(flat[p + 5])
        counts[b] = n
        starts[b] = p
        p += 6 + 3 * n
    if p != size:
        raise WannierOutputError(
            f"{path}: {size - p} integers left after {n_blocks} blocks"
        )
    heads = flat[starts[:, None] + np.arange(5)[None, :]]
    r_index = np.repeat(np.arange(n_r), num_wann * num_wann)
    i_index = np.tile(np.repeat(np.arange(1, num_wann + 1), num_wann), n_r)
    j_index = np.tile(np.arange(1, num_wann + 1), n_r * num_wann)
    expected = np.column_stack([rvectors[r_index], i_index, j_index])
    if not np.array_equal(heads, expected):
        raise WannierOutputError(
            f"{path}: blocks are not in the (R, i, j) order of the Hamiltonian"
        )
    if np.any(counts < 1):
        raise WannierOutputError(f"{path}: an element has no image")
    offsets = np.repeat(starts + 6, counts) + 3 * (
        np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    shifts = flat[offsets[:, None] + np.arange(3)[None, :]]
    return WsShifts(counts.reshape(n_r, num_wann, num_wann), shifts, flag)


def read_run(seed_path: str | Path) -> WannierRun:
    """Read ``<seed>.wout`` and, when present, the real-space Hamiltonian beside it."""
    seed_path = Path(seed_path)
    stem = seed_path.with_suffix("")
    run = read_wout(stem.with_suffix(".wout"))
    hr = stem.parent / f"{stem.name}_hr.dat"
    if hr.is_file():
        run.hr_rvectors, run.hr_degeneracies, run.hr_matrix = read_hr(hr)
        wsvec = stem.parent / f"{stem.name}_wsvec.dat"
        if wsvec.is_file():
            try:
                run.hr_ws_shifts = read_wsvec(
                    wsvec, run.hr_rvectors, run.hr_matrix.shape[1]
                )
            except (WannierOutputError, ValueError) as exc:
                run.notes.append(f"{wsvec.name} not used: {exc}")
    else:
        run.notes.append(
            f"no {hr.name}: the real-space checks (Wigner-Seitz bound, hopping "
            "decay) cannot be evaluated for this construction"
        )
    return run
