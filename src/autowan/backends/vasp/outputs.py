"""Translate VASP output to the one parent representation the gate consumes.

VASP already reports eV and Angstrom, but its fractional k-points must be expressed
in the gate's Cartesian convention. A wrong conversion would compare different
points in reciprocal space while leaving the band arrays superficially plausible.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...qe_outputs import ParentCalculation


class VaspOutputError(RuntimeError):
    """A VASP output is missing, malformed, or disagrees with its companion."""


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise VaspOutputError(f"no VASP output at {path}")
    return path.read_text(errors="replace")


def potcar_channels(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read every non-local projector channel from POTCAR or its OUTCAR echo.

    Check the first LMMAX count from each pseudopotential's own OUTCAR header,
    ignoring later counts from the rest of the file.

    The table's last row may be the local potential marked TYP 23 rather than 7
    (Na, Cu, V_sv, Ti_sv, Ga_d). It is dropped only when that makes the table agree
    with the pseudopotential's own projector inventory: the ``Non local Part`` blocks
    in a POTCAR, the ``non local Contribution for L=`` lines of an OUTCAR echo, or,
    for an echo that prints no such lines, its LMMAX.
    """
    path = Path(path)
    blocks = re.split(r"(?m)^\s*VRHFIN\s*=", _read_text(path))[1:]
    channels = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
    row = re.compile(
        rf"\s*(\d+)\s+{number}\s+(\d+)\s+{number}" rf"(?:\s+\d+\s+{number})?\s*"
    )
    for block in blocks:
        symbol, separator, _ = block.partition(":")
        symbol = symbol.strip()
        lines = block.splitlines()
        description = next(
            (i for i, line in enumerate(lines) if "Description" in line), None
        )
        if not separator or not symbol or description is None:
            raise VaspOutputError(f"{path}: missing pseudopotential Description table")
        if description + 1 >= len(lines) or not re.match(
            r"\s*l\s+E\s+TYP\s+RCUT", lines[description + 1]
        ):
            raise VaspOutputError(f"{path}: malformed Description header for {symbol}")
        nonlocal_rows = []
        nonlocal_indices = []
        row_count = 0
        last_typ = 7
        for line in lines[description + 2 :]:
            match = row.fullmatch(line)
            if match is None:
                break
            row_count += 1
            angular, typ = map(int, match.groups())
            last_typ = typ
            if typ != 7:
                if angular not in range(4):
                    raise VaspOutputError(
                        f"{path}: unsupported l={angular} for {symbol}"
                    )
                nonlocal_rows.append(angular)
                nonlocal_indices.append(row_count - 1)
        if not row_count:
            raise VaspOutputError(f"{path}: empty Description table for {symbol}")
        angulars = set(nonlocal_rows)
        parts = set()
        projectors = []
        for i, line in enumerate(lines):
            if "Non local Part" in line:
                try:
                    angular, nproj, rmax = lines[i + 1].split()
                    parts.add(int(angular))
                    projectors.extend([int(angular)] * int(nproj))
                    float(rmax.replace("D", "E").replace("d", "e"))
                except (ValueError, IndexError) as exc:
                    raise VaspOutputError(
                        f"{path}: malformed Non local Part for {symbol}"
                    ) from exc
        lmmax = re.search(r"\bLMMAX\s*=\s*(\d+)", block)
        # Echoes after the header's first count belong to the rest of the run.
        header = block[: lmmax.start()] if lmmax is not None else block
        projectors.extend(
            int(echo[1])
            for echo in re.finditer(
                r"non local Contribution for L=\s*(\d+)\s+read in", header
            )
        )
        if projectors:
            # Match rows in table order: only the final row may remain as the
            # local-potential reference, even when it is typed as non-local.
            remaining = list(zip(nonlocal_indices, nonlocal_rows))
            for angular in projectors:
                match = next((r for r in remaining if r[1] == angular), None)
                if match is None:
                    raise VaspOutputError(
                        f"{path}: {symbol} Non local Parts disagree with Description"
                    )
                remaining.remove(match)
            if remaining and not (
                len(remaining) == 1 and remaining[0][0] == row_count - 1
            ):
                raise VaspOutputError(
                    f"{path}: {symbol} Non local Parts disagree with Description"
                )
            angulars = set(projectors)
        elif last_typ != 7 and nonlocal_rows and lmmax is not None:
            # An echo that prints no projector lines: VASP's LMMAX is what identifies
            # a trailing local row, dropped only when that reproduces the count.
            count = int(lmmax.group(1))
            reduced = nonlocal_rows[:-1]
            if _lm_count(nonlocal_rows) != count and _lm_count(reduced) == count:
                nonlocal_rows = reduced
                angulars = set(nonlocal_rows)
        if lmmax is not None:
            if _lm_count(projectors or nonlocal_rows) != int(lmmax.group(1)):
                raise VaspOutputError(
                    f"{path}: {symbol} Description disagrees with LMMAX"
                )
        if not projectors and "Non local Part" in block and parts != angulars:
            raise VaspOutputError(
                f"{path}: {symbol} Non local Parts disagree with Description"
            )
        letters = tuple("spdf"[angular] for angular in sorted(angulars))
        if symbol in channels and channels[symbol] != letters:
            raise VaspOutputError(
                f"{path}: conflicting projector channels for {symbol}"
            )
        channels[symbol] = letters
    if not channels:
        raise VaspOutputError(f"{path}: no pseudopotential Description table found")
    return channels


def _lm_count(angulars) -> int:
    return sum(2 * angular + 1 for angular in angulars)


def locproj_auto_channels(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Name the orbitals VASP's ``LOCPROJ_AUTO`` will project on, before VASP runs.

    VASP 6.4.1 (``locproj.F``, ``LPRJ_FROM_PSEUDO``) walks each pseudopotential's
    projector channels in POTCAR order, skips a channel whose atomic occupancy is zero,
    and keeps the first remaining channel of each l (its PS partial wave). The
    occupancies are the ``uccopancies in atom`` block of the POTCAR body; an OUTCAR
    echoes only the header, so it cannot answer this.
    """
    path = Path(path)
    blocks = re.split(r"(?m)^\s*VRHFIN\s*=", _read_text(path))[1:]
    selection: dict[str, tuple[str, ...]] = {}
    for block in blocks:
        symbol = block.partition(":")[0].strip()
        lines = block.splitlines()
        channels: list[int] = []
        for i, line in enumerate(lines):
            if "Non local Part" in line:
                try:
                    angular, nproj = lines[i + 1].split()[:2]
                    channels += [int(angular)] * int(nproj)
                except (ValueError, IndexError) as exc:
                    raise VaspOutputError(
                        f"{path}: malformed Non local Part for {symbol}"
                    ) from exc
        start = next(
            (i for i, line in enumerate(lines) if "uccopancies in atom" in line), None
        )
        if not channels or start is None:
            raise VaspOutputError(
                f"{path}: no atomic occupancies for {symbol}; LOCPROJ_AUTO's "
                "selection is read from the POTCAR itself"
            )
        size = len(channels) ** 2
        values: list[float] = []
        for line in lines[start + 1 :]:
            if len(values) >= size:
                break
            try:
                values += [float(x.replace("D", "E")) for x in line.split()]
            except ValueError:
                break
        if len(values) < size:
            raise VaspOutputError(f"{path}: truncated atomic occupancies for {symbol}")
        picked: list[int] = []
        for index, angular in enumerate(channels):
            if values[index * len(channels) + index] == 0.0:
                continue  # not a bound state of the atom: LOCPROJ_AUTO skips it
            if angular not in picked:
                picked.append(angular)
        letters = tuple("spdf"[angular] for angular in picked)
        if symbol in selection and selection[symbol] != letters:
            raise VaspOutputError(f"{path}: conflicting occupancies for {symbol}")
        selection[symbol] = letters
    if not selection:
        raise VaspOutputError(f"{path}: no pseudopotential occupancies found")
    return selection


def potcar_valences(path: str | Path) -> dict[str, float]:
    """Read the first valence charge in each pseudopotential's header."""
    path = Path(path)
    valences = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
    for block in re.split(r"(?m)^\s*VRHFIN\s*=", _read_text(path))[1:]:
        symbol, separator, _ = block.partition(":")
        match = re.search(rf"\bZVAL\s*=\s*({number})", block)
        if separator and symbol.strip() and match:
            valences[symbol.strip()] = float(
                match[1].replace("D", "E").replace("d", "e")
            )
    return valences


def read_kmesh(vasprun_xml: str | Path) -> tuple[int, int, int]:
    """Read the mesh VASP ran, requiring an unreduced k-point list."""
    path = Path(vasprun_xml)
    try:
        root = ET.fromstring(_read_text(path))
        divisions = root.findtext("kpoints/generation/v[@name='divisions']")
        if divisions is None:
            raise VaspOutputError(f"{path}: no mesh divisions")
        mesh = tuple(int(value) for value in divisions.split())
        if len(mesh) != 3 or any(value < 1 for value in mesh):
            raise ValueError("expected three positive integers")
    except (ET.ParseError, ValueError) as exc:
        raise VaspOutputError(f"{path}: malformed mesh divisions: {exc}") from exc
    if len(root.findall("kpoints/varray[@name='kpointlist']/v")) != (
        mesh[0] * mesh[1] * mesh[2]
    ):
        raise VaspOutputError(f"{path}: full k-point mesh required; run with ISYM = -1")
    return mesh


def _vectors(root: ET.Element, path: str) -> np.ndarray:
    rows = root.findall(path + "/v")
    if not rows:
        raise VaspOutputError(f"VASP XML carries no {path}")
    return np.array([[float(x) for x in (r.text or "").split()] for r in rows])


def _setting(root: ET.Element, name: str, default: int) -> int:
    for path in (f"incar/i[@name='{name}']", f"parameters//i[@name='{name}']"):
        item = root.find(path)
        if item is not None and item.text:
            return int(item.text)
    return default


def _flag(root: ET.Element, name: str) -> bool:
    """A logical INCAR tag as vasprun.xml records it (``incar`` then ``parameters``)."""
    for path in (f"incar/i[@name='{name}']", f"parameters//i[@name='{name}']"):
        item = root.find(path)
        if item is not None and item.text:
            return item.text.strip().upper().lstrip(".").startswith("T")
    return False


def _real(root: ET.Element, name: str) -> float | None:
    for path in (f"parameters//i[@name='{name}']", f"incar/i[@name='{name}']"):
        item = root.find(path)
        if item is not None and item.text:
            try:
                return float(item.text)
            except ValueError:
                return None
    return None


def read_parent(vasprun_xml: str | Path) -> ParentCalculation:
    path = Path(vasprun_xml)
    if not path.is_file():
        raise VaspOutputError(f"no VASP XML at {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise VaspOutputError(f"cannot read VASP XML at {path}: {exc}") from exc
    calculations = root.findall("calculation/eigenvalues/array/set")
    if not calculations:
        raise VaspOutputError(f"{path} carries no eigenvalues")
    rows = [
        [
            [
                [float(x) for x in (r.text or "").split()[:2]]
                for r in kpoint.findall("r")
            ]
            for kpoint in spin.findall("set")
        ]
        for spin in calculations[-1].findall("set")
        if spin.get("comment", "").startswith("spin")
    ]
    try:
        bands = np.array(rows, dtype=float)
        if bands.ndim != 4 or bands.shape[-1] != 2 or bands.size == 0:
            raise ValueError("expected spin, k-point, band, energy/occupation axes")
    except ValueError as exc:
        raise VaspOutputError(f"{path} has malformed eigenvalues: {exc}") from exc

    outcar_path = path.with_name("OUTCAR")
    outcar = _read_text(outcar_path) if outcar_path.is_file() else ""
    fermi_items = root.findall("calculation/dos/i[@name='efermi']")
    if fermi_items and fermi_items[-1].text:
        fermi = float(fermi_items[-1].text)
    else:
        found = re.findall(r"E-fermi\s*:\s*([-+\d.Ee]+)", outcar)
        if not found:
            raise VaspOutputError(f"{path} records no Fermi level, nor does OUTCAR")
        fermi = float(found[-1])

    structure = root.find("structure[@name='finalpos']")
    if structure is None:
        structure = root.find("structure[@name='initialpos']")
    if structure is None:
        raise VaspOutputError(f"{path} records no finalpos or initialpos structure")
    lattice = _vectors(structure, "crystal/varray[@name='basis']")
    positions = _vectors(structure, "varray[@name='positions']")
    species = tuple(
        (row.findtext("c") or "").strip()
        for row in root.findall("atominfo/array[@name='atoms']/set/rc")
    )
    alat = float(np.linalg.norm(lattice[0]))
    kfrac = _vectors(root, "kpoints/varray[@name='kpointlist']")
    if len(kfrac) != bands.shape[1]:
        raise VaspOutputError(f"{path}: eigenvalues and k-point counts disagree")

    eigenval_path = path.with_name("EIGENVAL")
    if eigenval_path.is_file():
        other, _ = read_eigenval(eigenval_path)
        if other.shape != bands[..., 0].shape or not np.allclose(
            other,
            bands[..., 0],
            atol=1e-3,
            rtol=0,
        ):
            raise VaspOutputError(
                f"{eigenval_path}: EIGENVAL disagrees with XML eigenvalues"
            )

    converged = None
    if "EDIFF was not reached" in outcar:
        converged = False
    elif "aborting loop because EDIFF is reached" in outcar:
        converged = True
    symmetry = re.findall(r"Found\s+(\d+)\s+space group operations", outcar, re.I)
    nsym = (
        1 if _setting(root, "ISYM", 0) == -1 else int(symmetry[-1]) if symmetry else 0
    )
    # vasp_ncl (LNONCOLLINEAR, LSORBIT implies it): one set of two-component spinor
    # bands, one electron each. VASP writes ISPIN = 1 for it, so the eigenvalue array
    # must carry one channel.
    spinor = _flag(root, "LNONCOLLINEAR") or _flag(root, "LSORBIT")
    if spinor and bands.shape[0] != 1:
        raise VaspOutputError(
            f"{path}: a noncollinear run with {bands.shape[0]} eigenvalue channels"
        )
    return ParentCalculation(
        eigenvalues_ev=bands[..., 0],
        occupations=bands[..., 1],
        kpoints_cart=kfrac @ np.linalg.inv(lattice / alat).T,
        fermi_energy_ev=fermi,
        lattice_ang=lattice,
        alat_ang=alat,
        species=species,
        positions_frac=positions,
        n_symmetries=nsym,
        n_atomic_wfc=None,
        converged=converged,
        calculation="nscf" if _setting(root, "ICHARG", 2) >= 10 else "scf",
        spinor=spinor,
        n_electrons=_real(root, "NELECT"),
    )


def read_eigenval(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    lines = _read_text(path).splitlines()
    try:
        n_spin = int(lines[0].split()[-1])
        _, n_kpoints, n_bands = map(int, lines[5].split())
        body = iter(line.split() for line in lines[6:] if line.strip())
        energies, kpoints = [], []
        for _ in range(n_kpoints):
            kpoints.append([float(x) for x in next(body)[:3]])
            energies.append(
                [[float(x) for x in next(body)[1 : 1 + n_spin]] for _ in range(n_bands)]
            )
        values = np.array(energies).transpose(2, 0, 1)
        if values.shape != (n_spin, n_kpoints, n_bands):
            raise ValueError("incomplete band table")
        return values, np.array(kpoints)
    except (ValueError, IndexError, StopIteration) as exc:
        raise VaspOutputError(f"malformed EIGENVAL at {path}: {exc}") from exc


@dataclass
class VaspProjections:
    energies_ev: np.ndarray
    projectability: np.ndarray
    weights: np.ndarray
    orbitals: tuple[str, ...]
    n_kpoints: int
    n_bands: int
    n_ions: int
    n_spin: int
    kpoint_weights: np.ndarray
    noncollinear: bool = False
    """Whether each band carried the three magnetisation tables of a ``vasp_ncl`` run.

    They are read and kept apart (:attr:`magnetisation`): they are the spin density's
    projections, not charge, and must never be counted as further states."""
    magnetisation: np.ndarray | None = None
    """``(state, 3, ion, orbital)``: the m_x, m_y, m_z tables of a noncollinear PROCAR."""


def read_procar(path: str | Path) -> VaspProjections:
    """Read charge projection tables in file order: spin, k-point, then band.

    Each band carries one ion table (one row per ion, a ``tot`` row only when there is
    more than one ion). A noncollinear (``vasp_ncl``) file follows it with three more
    tables of the same shape and no header: the m_x, m_y and m_z projections of the
    magnetisation density. Those are kept apart, never read as more states: a spinor band
    is one state holding one electron, and its charge table already sums both spinor
    components.
    """
    path = Path(path)
    text = _read_text(path)
    header = re.search(
        r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)",
        text,
    )
    if header is None:
        raise VaspOutputError(f"{path}: PROCAR records no k-point, band and ion counts")
    nk, nb, ni = map(int, header.groups())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    energies, totals, weights, k_weights, magnet = [], [], [], [], []
    orbitals: tuple[str, ...] = ()
    n_tables = None

    def table(i: int) -> tuple[list[list[float]], float, int]:
        """One ion table from line ``i``: rows, its total, and the next line index."""
        rows = []
        for ion in range(ni):
            row = lines[i].split()
            if int(row[0]) != ion + 1 or len(row) != len(orbitals) + 2:
                raise ValueError("incomplete ion projection row")
            rows.append([float(x) for x in row[1:-1]])
            i += 1
        if ni > 1:
            total = lines[i].split()
            if total[0] != "tot":
                raise ValueError("missing total projection row")
            value = float(total[-1])
            i += 1
        else:
            # VASP writes no tot row for a single ion; the ion's own total is the sum.
            value = float(lines[i - 1].split()[-1])
        return rows, value, i

    try:
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("k-point"):
                k_weights.append(float(line.split("weight =", 1)[1]))
                i += 1
                continue
            match = re.match(r"band\s+\d+\s+# energy\s+([-+\d.Ee]+)", line)
            if not match:
                i += 1
                continue
            energies.append(float(match.group(1)))
            i += 1
            if not lines[i].startswith("ion"):
                raise ValueError("missing orbital header")
            columns = tuple(lines[i].split()[1:-1])
            if orbitals and columns != orbitals:
                raise ValueError("orbital columns changed within PROCAR")
            orbitals = columns
            rows, total, i = table(i + 1)
            weights.append(rows)
            totals.append(total)
            # Further header-less tables before the next band are the magnetisation.
            extra = []
            while i < len(lines) and re.match(r"1\s", lines[i]):
                block, _, i = table(i)
                extra.append(block)
            if n_tables is None:
                n_tables = 1 + len(extra)
                if n_tables not in (1, 4):
                    raise ValueError(
                        f"{n_tables} projection tables per band; expected 1 (collinear) "
                        "or 4 (noncollinear: charge, m_x, m_y, m_z)"
                    )
            elif 1 + len(extra) != n_tables:
                raise ValueError(
                    "the number of projection tables changed within PROCAR"
                )
            if extra:
                magnet.append(extra)
        n_spin, remainder = divmod(len(energies), nk * nb)
        if remainder or n_spin not in (1, 2) or len(k_weights) != n_spin * nk:
            raise ValueError("incomplete spin/k-point/band table")
        if n_tables == 4 and n_spin != 1:
            raise ValueError("a noncollinear PROCAR carries one spin block")
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        raise VaspOutputError(f"malformed PROCAR at {path}: {exc}") from exc
    return VaspProjections(
        energies_ev=np.array(energies),
        projectability=np.array(totals),
        weights=np.array(weights),
        orbitals=orbitals,
        n_kpoints=nk,
        n_bands=nb,
        n_ions=ni,
        n_spin=n_spin,
        kpoint_weights=np.array(k_weights[:nk]),
        noncollinear=n_tables == 4,
        magnetisation=np.array(magnet) if magnet else None,
    )


def projector_supply(
    proj: VaspProjections,
    species: Sequence[str],
    fermi_energy_ev: float,
    margin_ev: float = 2.0,
    floor: float = 0.1,
) -> dict[str, tuple[str, ...]]:
    if len(species) != proj.n_ions:
        raise VaspOutputError(
            f"PROCAR has {proj.n_ions} ions but species lists {len(species)}"
        )
    channels = [
        "d" if o.startswith(("x2-y2", "dx2")) else o[0].lower() for o in proj.orbitals
    ]
    near = proj.energies_ev <= fermi_energy_ev + margin_ev
    supply = {}
    for symbol in dict.fromkeys(species):
        ion_mask = np.array([s == symbol for s in species])
        selected = proj.weights[near][:, ion_mask, :].sum(axis=1)
        supply[symbol] = tuple(
            channel
            for channel in ("s", "p", "d", "f")
            if channel in channels
            and np.any(selected[:, np.array(channels) == channel].sum(axis=1) >= floor)
        )
    return supply
