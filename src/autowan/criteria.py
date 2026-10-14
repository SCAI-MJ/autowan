"""Acceptance criteria scored against DFT bands on a separate high-symmetry path.

Three optional gate checks, one per research use of a Wannier model (Item 4 decisions 2
and 10, 2026-10-06/07). Each compares the construction's own model with DFT bands on a
band path that is *not* the construction mesh, using the paper's band distance

    eta = sqrt( sum_nk w_nk (e_DFT - e_Wan)^2 / sum_nk w_nk ),
    w_nk = sqrt( f_FD(e_DFT; mu, sigma) f_FD(e_Wan; mu, sigma) ),

with sigma = 0.1 eV, bands matched by index after the deck's ``exclude_bands`` are
removed, and success at eta <= 20 meV -- exactly ``bands_distance_raw`` of
``aiida_wannier90_workflows.utils.bands.distance`` (:mod:`.band_distance_eta`).

* ``band_interpolation_eta2`` -- eta at mu = E_F + 2 eV (Qiao, Pizzi & Marzari 2023;
  arXiv:2303.07877, 2507.06840), with E_F the **scf** Fermi level the paper workflow hands
  Wannier90 as ``fermi_energy``. The nscf (mesh) Fermi level the gate's parent carries is
  recorded beside it, with eta at that level as a diagnostic.
* ``truncated_hopping_eta2`` -- the same eta for the model with every hopping longer than
  r_c = L_BvK / 4 removed and the Hamiltonian re-diagonalised. A hopping's length is
  |R + tau_j - tau_i|, the distance between the two Wannier centres it connects (since
  v5; v1-v4 cut on the lattice vector |R|, which in a cell with centres far apart -- NiO's
  elongated four-atom AFM-II cell, centres up to 10.9 A apart -- removed nearest-neighbour
  hoppings of 2.1 A). L/3, L/2 and the fitted decay length tau are diagnostics; the
  verdict is at L/4 only.
* ``conduction_band_eta`` -- for insulators only (no band crosses E_F on the mesh **and**
  the gap is at least sigma), the cumulative eta at mu = CBM + 2 eV. The window-only eta
  over [CBM, CBM + 2 eV] is a diagnostic. For a metal the check is withheld, not failed.

**The Wannier bands are the construction's own model**, Fourier-interpolated from its
H(R) onto the reference path's k-list with Wannier90's minimal-image convention
(``<seed>_wsvec.dat``) applied, so the bands compared are those Wannier90's own band plot
would draw (to 0.2 meV on the stored QE constructions) and the truncation acts on the
hopping vectors the model actually uses.

**L_BvK** is the length of the shortest non-zero lattice vector of the Born--von Karman
supercell, the superlattice spanned by ``mp_grid[i] * a_i``: equivalently, twice the
inscribed radius of its Wigner--Seitz cell. For a cubic N x N x N supercell it is the edge
N a. The ladder then reads naturally: L/2 is the largest sphere wholly inside the
Wigner--Seitz supercell (every hopping the mesh can represent in every direction), and
L/4 keeps half that range. At the campaign density (0.2 A^-1, including 2 pi) L is about
2 pi / 0.2 = 31 A, so r_c is about 8 A, the figure the discussion quoted. Taking L as the
inscribed radius instead would put r_c near 4 A, inside the second-neighbour shell of
most of the twenty materials, which no use of a sparse model intends.

**Slabs and wires (v3).** A direction the k-mesh does not sample (``mp_grid = 1``) and
along which the structure leaves a vacuum gap (:data:`VACUUM_GAP_NN_MULTIPLE`
nearest-neighbour distances or more) is not periodic, and L_BvK is taken over the
periodic directions only. Over all three, a monolayer's L_BvK is its vacuum-padded c
axis (19 A for MoS2), which put r_c at 4.8 A, inside the second-neighbour shell. The
record carries ``periodic_dims`` (3 for bulk, 2 for a slab) and the measured gap. A bulk
crystal is unaffected: without a vacuum gap every direction stays periodic.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .band_distance_eta import (
    ETA2_SUCCESS_MEV,
    ETA_NU_EV,
    ETA_SIGMA_EV,
    band_distance_eta,
)
from .gate import (
    Check,
    CheckOutcome,
    Construction,
    NonEvaluable,
    NotApplicable,
    ScaleAwareThreshold,
)

#: The verdict's truncation radius, as a fraction of L_BvK (decision 2, form iii).
TRUNCATION_FRACTION = 0.25
#: Diagnostic truncation radii, reported and never judged.
DIAGNOSTIC_TRUNCATION_FRACTIONS = (1.0 / 3.0, 0.5)
#: Search range for the shortest superlattice vector. Coefficients up to 3 cover the
#: Minkowski-reduced vector of any supercell of a reduced primitive cell with a mesh
#: anisotropy up to a factor three.
_SUPERLATTICE_SEARCH = 3
#: A lattice direction is a vacuum (non-periodic) direction when the k-mesh does not
#: sample it (``mp_grid = 1``) **and** the structure leaves an empty layer along it at
#: least this many nearest-neighbour distances thick. A van der Waals gap between bonded
#: layers stays below it (graphite 3.35 A = 2.4 x its 1.42 A bond; bulk MoS2's S-S
#: interlayer gap 3.0 A = 1.2 x its 2.41 A bond), while a slab's vacuum exceeds it by a
#: wide margin (monolayer MoS2 at c = 19.1 A: 16.0 A = 6.6 x; graphene: > 10 x).
VACUUM_GAP_NN_MULTIPLE = 3.0


# ---------------------------------------------------------------------------
# the reference band path
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ReferenceBandPath:
    """DFT bands on a high-symmetry path, and the Fermi levels that go with them.

    ``kpoints_frac`` are fractional coordinates in the reciprocal basis of the parent
    cell (the cell Wannier90's lattice vectors span). ``bands_ev`` is ``(n_spin, n_k,
    n_bands)``; a two-dimensional array is read as one spin channel.
    ``fermi_energy_ev`` is the scf Fermi level -- the one the paper's workflow passes
    to Wannier90 and to ``bands_distance`` -- and ``nscf_fermi_energy_ev`` the
    construction-mesh value, recorded beside it.
    """

    kpoints_frac: np.ndarray
    bands_ev: np.ndarray
    fermi_energy_ev: float | None
    nscf_fermi_energy_ev: float | None = None
    source: str = ""
    fermi_source: str = ""
    labels: tuple = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = np.asarray(self.kpoints_frac, dtype=float)
        bands = np.asarray(self.bands_ev, dtype=float)
        if bands.ndim == 2:
            bands = bands[None]
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("reference k-points must be (n_k, 3) fractional")
        if bands.ndim != 3 or bands.shape[1] != k.shape[0]:
            raise ValueError(
                f"reference bands {bands.shape} do not match {k.shape[0]} k-points"
            )
        object.__setattr__(self, "kpoints_frac", k)
        object.__setattr__(self, "bands_ev", bands)
        for name in ("fermi_energy_ev", "nscf_fermi_energy_ev"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    @property
    def n_spin(self) -> int:
        return int(self.bands_ev.shape[0])

    def channel(self, spin_index: int) -> np.ndarray:
        if not 0 <= spin_index < self.n_spin:
            raise NonEvaluable(
                f"the reference path has {self.n_spin} spin channel(s), not index "
                f"{spin_index}"
            )
        return self.bands_ev[spin_index]


_QE_FERMI = re.compile(r"the Fermi energy is\s+([-\d.]+)\s+ev", re.IGNORECASE)
_QE_HOMO_LUMO = re.compile(
    r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)"
)
_OUTCAR_FERMI = re.compile(r"E-fermi\s*:\s*([-\d.]+)")


def qe_output_fermi_energy_ev(path: str | Path) -> float:
    """The last Fermi level a pw.x output prints (smeared runs)."""
    text = Path(path).read_text(errors="replace")
    found = _QE_FERMI.findall(text)
    if not found:
        raise ValueError(f"{path}: no 'the Fermi energy is' line")
    return float(found[-1])


def outcar_fermi_energy_ev(path: str | Path) -> float:
    """The last ``E-fermi`` a VASP OUTCAR prints."""
    found = _OUTCAR_FERMI.findall(Path(path).read_text(errors="replace"))
    if not found:
        raise ValueError(f"{path}: no E-fermi line")
    return float(found[-1])


def read_qe_reference_band_path(
    npz_path: str | Path,
    *,
    fermi_energy_ev: float | None = None,
    scf_output: str | Path | None = None,
    nscf_output: str | Path | None = None,
) -> ReferenceBandPath:
    """The QE paper-settings baseline's ``dft_bands_path.npz`` (reports/2026-10-07/01).

    The archive holds ``bands`` -- ``(n_k, n_bands)``, or ``(2, n_k, n_bands)`` for a
    collinear magnet -- from the pw.x ``bands`` run on Wannier90's own path, and
    ``kpoints``, fractional. The scf Fermi level is ``fermi_energy_ev`` when given,
    otherwise read from ``scf_output`` (default ``scf.out`` beside the archive); the nscf
    level likewise from ``nscf.out``. A missing nscf level is recorded as missing.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as archive:
        bands = np.asarray(archive["bands"], dtype=float)
        kpoints = np.asarray(archive["kpoints"], dtype=float)
    fermi_source = "given"
    if fermi_energy_ev is None:
        scf_output = Path(scf_output or npz_path.with_name("scf.out"))
        fermi_energy_ev = qe_output_fermi_energy_ev(scf_output)
        fermi_source = f"scf output {scf_output}"
    nscf_output = Path(nscf_output or npz_path.with_name("nscf.out"))
    nscf = qe_output_fermi_energy_ev(nscf_output) if nscf_output.is_file() else None
    return ReferenceBandPath(
        kpoints,
        bands,
        fermi_energy_ev,
        nscf,
        source=f"QE pw.x bands {npz_path}",
        fermi_source=fermi_source,
    )


def read_eigenval(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """VASP ``EIGENVAL``: ``(kpoints_frac (n_k, 3), bands (n_spin, n_k, n_bands))``."""
    lines = Path(path).read_text().splitlines()
    n_spin = int(lines[0].split()[3])
    _, n_k, n_bands = (int(x) for x in lines[5].split()[:3])
    cursor = 6
    kpoints = np.empty((n_k, 3))
    bands = np.empty((n_spin, n_k, n_bands))
    for ik in range(n_k):
        while not lines[cursor].strip():
            cursor += 1
        kpoints[ik] = [float(x) for x in lines[cursor].split()[:3]]
        cursor += 1
        for ib in range(n_bands):
            fields = lines[cursor].split()
            for s in range(n_spin):
                bands[s, ik, ib] = float(fields[1 + s])
            cursor += 1
    return kpoints, bands


def read_vasp_reference_band_path(
    band_dir: str | Path,
    *,
    scf_outcar: str | Path | None = None,
    nscf_outcar: str | Path | None = None,
    fermi_energy_ev: float | None = None,
) -> ReferenceBandPath:
    """A band-path VASP NSCF directory (an ``EIGENVAL`` on the QE reference path).

    Bands and k-points come from ``EIGENVAL`` (an explicit ``Reciprocal`` list, so the
    coordinates are fractional in the parent's reciprocal basis and equal the QE path
    one for one); ``path.json``'s k-list, when present, must agree with it. The
    path's own E-fermi is not used: a non-self-consistent run on a line of k-points has
    no Fermi level worth the name. The scf level is read from ``scf_outcar``, by default
    the parent's ``scf/OUTCAR`` named in ``settings.json``; the nscf level from the
    parent's ``nscf/OUTCAR``.
    """
    band_dir = Path(band_dir)
    kpoints, bands = read_eigenval(band_dir / "EIGENVAL")
    labels: tuple = ()
    path_json = band_dir / "path.json"
    if path_json.is_file():
        path = json.loads(path_json.read_text())
        listed = np.asarray(path.get("kpoints_frac", []), dtype=float)
        if listed.shape != kpoints.shape or np.max(np.abs(listed - kpoints)) > 1e-6:
            raise ValueError(f"{band_dir}: EIGENVAL k-points differ from path.json")
        labels = tuple(
            tuple(x) if isinstance(x, list) else x for x in path.get("labels", ())
        )
    parent_dir = None
    settings = band_dir / "settings.json"
    if settings.is_file():
        parent_dir = json.loads(settings.read_text()).get("parent_dir")
    fermi_source = "given"
    if fermi_energy_ev is None:
        if scf_outcar is None:
            if parent_dir is None:
                raise ValueError(
                    f"{band_dir}: no scf OUTCAR given and no parent_dir in settings.json"
                )
            scf_outcar = Path(parent_dir) / "scf" / "OUTCAR"
        fermi_energy_ev = outcar_fermi_energy_ev(scf_outcar)
        fermi_source = f"scf OUTCAR {scf_outcar}"
    if nscf_outcar is None and parent_dir is not None:
        nscf_outcar = Path(parent_dir) / "nscf" / "OUTCAR"
    nscf = (
        outcar_fermi_energy_ev(nscf_outcar)
        if nscf_outcar is not None and Path(nscf_outcar).is_file()
        else None
    )
    return ReferenceBandPath(
        kpoints,
        bands,
        fermi_energy_ev,
        nscf,
        source=f"VASP band-path NSCF {band_dir}",
        fermi_source=fermi_source,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# the construction's own model, as Wannier90 interpolates it
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RealSpaceModel:
    """H(R) with the Wigner--Seitz degeneracies and minimal-image shifts applied.

    ``hoppings[v]`` is the block at lattice vector ``vectors[v]`` (integer, lattice
    basis), already divided by ``deg(R)`` and by the element's image count, so that
    ``H(k) = sum_v exp(2 pi i k.v) hoppings[v]`` is Wannier90's own interpolation.
    ``lengths_ang`` are the Cartesian lengths |R| of those vectors (the decay fit's
    abscissa). ``element_lengths_ang[v, i, j]`` is the length of the hopping itself,
    |R_v + tau_j - tau_i| with tau the Wannier centres: the distance from function i in
    the home cell to function j in cell R_v, which is what a truncation radius is
    compared with (since v5). Without centres it falls back to |R_v| for every element.
    """

    vectors: np.ndarray
    hoppings: np.ndarray
    lengths_ang: np.ndarray
    ws_distance_applied: bool
    element_lengths_ang: np.ndarray | None = None

    def kept_elements(self, max_length_ang: float | None) -> np.ndarray:
        """Boolean (vector, i, j): the hoppings no longer than ``max_length_ang``."""
        shape = self.hoppings.shape
        if max_length_ang is None:
            return np.ones(shape, dtype=bool)
        if self.element_lengths_ang is None:
            keep = self.lengths_ang <= max_length_ang * (1 + 1e-9)
            return np.broadcast_to(keep[:, None, None], shape)
        return self.element_lengths_ang <= max_length_ang * (1 + 1e-9)

    def bands(self, kpoints_frac, max_length_ang: float | None = None) -> np.ndarray:
        keep = self.kept_elements(max_length_ang)
        vectors = keep.any(axis=(1, 2))
        k = np.atleast_2d(np.asarray(kpoints_frac, dtype=float))
        phases = np.exp(2j * np.pi * (k @ self.vectors[vectors].T))
        blocks = np.where(keep, self.hoppings, 0.0)[vectors]
        h = np.einsum("kv,vij->kij", phases, blocks)
        h = 0.5 * (h + np.conjugate(np.transpose(h, (0, 2, 1))))
        return np.linalg.eigvalsh(h)


def real_space_model(wannier: Any, lattice_ang, centres_ang=None) -> RealSpaceModel:
    """Expand a construction's H(R) over Wannier90's minimal images.

    ``centres_ang`` (the construction's Wannier centres, Cartesian) gives every element
    its hopping length |R + tau_j - tau_i|, in the same frame as Wannier90's
    ``use_ws_distance`` images (``<seed>_wsvec.dat`` is computed from these centres), so
    the length is the one of the image the interpolation actually uses.
    """
    h = np.asarray(wannier.hr_matrix)
    rvec = np.asarray(wannier.hr_rvectors, dtype=np.int64)
    deg = np.asarray(wannier.hr_degeneracies, dtype=float)
    lattice = np.asarray(lattice_ang, dtype=float)
    ws = getattr(wannier, "hr_ws_shifts", None)
    n_r, nw, _ = h.shape
    if ws is None:
        vectors = rvec
        hoppings = h / deg[:, None, None]
    else:
        counts = np.asarray(ws.counts).reshape(-1)
        values = (h / deg[:, None, None]).reshape(-1) / counts
        r_index = np.repeat(np.arange(n_r), nw * nw)
        flat_index = np.repeat(np.arange(n_r * nw * nw), counts)
        images = rvec[r_index[flat_index]] + np.asarray(ws.shifts)
        vectors, inverse = np.unique(images, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        element = flat_index % (nw * nw)
        hoppings = np.zeros((len(vectors), nw * nw), dtype=complex)
        np.add.at(hoppings, (inverse, element), values[flat_index])
        hoppings = hoppings.reshape(len(vectors), nw, nw)
    cart = vectors @ lattice
    lengths = np.linalg.norm(cart, axis=1)
    element = None
    if centres_ang is not None:
        tau = np.asarray(centres_ang, dtype=float).reshape(-1, 3)
        if tau.shape[0] == nw:
            # d[v, i, j] = R_v + tau_j - tau_i
            d = cart[:, None, None, :] + tau[None, None, :, :] - tau[None, :, None, :]
            element = np.linalg.norm(d, axis=-1)
    return RealSpaceModel(vectors, hoppings, lengths, ws is not None, element)


_MODEL_CACHE: dict[int, tuple[Any, Any, RealSpaceModel]] = {}


def _model(c: Construction) -> RealSpaceModel:
    """The construction's real-space model, built once per Wannier run."""
    h = c.require("hr_matrix")
    c.require("hr_rvectors")
    c.require("hr_degeneracies")
    key = id(c.wannier)
    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached[0] is c.wannier and cached[1] is h:
        return cached[2]
    centres = getattr(c.wannier, "centres_ang", None)
    model = real_space_model(c.wannier, c.parent.lattice_ang, centres)
    if len(_MODEL_CACHE) >= 4:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _MODEL_CACHE[key] = (c.wannier, h, model)
    return model


def bvk_length_ang(lattice_ang, mp_grid, periodic_axes=None) -> float:
    """L_BvK: the shortest non-zero vector of the Born--von Karman superlattice.

    The superlattice is spanned by ``mp_grid[i] * a_i``; its shortest vector is twice
    the inscribed radius of the Wigner--Seitz supercell (see the module docstring).

    ``periodic_axes`` restricts the superlattice to the periodic directions of a slab or
    a wire (:func:`periodic_axes`): along a vacuum direction the mesh is one cell thick by
    construction, and the vacuum-padded cell length there says nothing about the range of
    hoppings the model represents. ``None`` (the default) is the bulk superlattice.
    """
    grid = np.asarray(mp_grid, dtype=float).reshape(-1)
    if grid.shape != (3,) or np.any(grid < 1):
        raise ValueError(f"mp_grid must be three positive integers, got {mp_grid!r}")
    superlattice = grid[:, None] * np.asarray(lattice_ang, dtype=float)
    axes = (0, 1, 2) if periodic_axes is None else tuple(sorted(periodic_axes))
    if not axes:
        raise ValueError("no periodic direction: L_BvK is undefined")
    superlattice = superlattice[list(axes)]
    span = np.arange(-_SUPERLATTICE_SEARCH, _SUPERLATTICE_SEARCH + 1)
    n = (
        np.array(np.meshgrid(*([span] * len(axes)), indexing="ij"))
        .reshape(len(axes), -1)
        .T
    )
    n = n[np.any(n != 0, axis=1)]
    return float(np.linalg.norm(n @ superlattice, axis=1).min())


def vacuum_gaps_ang(lattice_ang, positions_frac) -> list[float]:
    """The thickest empty layer along each lattice direction, in Angstrom.

    Along direction i the atoms sit on planes of constant fractional coordinate x_i; the
    largest periodic gap between consecutive planes, times the interplanar spacing
    1 / |b_i| (b the reciprocal basis without 2 pi), is the thickness of the empty layer
    measured perpendicular to the other two lattice vectors.
    """
    lattice = np.asarray(lattice_ang, dtype=float)
    frac = np.asarray(positions_frac, dtype=float).reshape(-1, 3)
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(lattice).T, axis=1)
    gaps = []
    for i in range(3):
        x = np.sort(np.mod(frac[:, i], 1.0))
        steps = np.diff(np.concatenate([x, [x[0] + 1.0]]))
        gaps.append(float(steps.max() * spacing[i]))
    return gaps


def periodic_axes(c: Construction) -> tuple[tuple[int, ...], dict[str, Any]]:
    """The lattice directions the construction is periodic along, with the evidence.

    A direction is non-periodic only when both hold: the k-mesh does not sample it
    (``mp_grid[i] = 1``) and the structure has a vacuum gap along it of at least
    ``VACUUM_GAP_NN_MULTIPLE`` nearest-neighbour distances. A bulk crystal sampled by
    one k-point along a direction (no gap) stays periodic there, so every bulk result
    is unchanged.
    """
    grid = [int(round(float(g))) for g in np.asarray(c.inputs["mp_grid"]).reshape(-1)]
    single = [i for i, g in enumerate(grid) if g == 1]
    record: dict[str, Any] = {}
    if not single:
        return (0, 1, 2), record
    frac = getattr(c.parent, "positions_frac", None)
    if frac is None or len(frac) == 0:
        record["periodicity_note"] = (
            "mp_grid is 1 along a lattice vector but the parent records no atomic "
            "positions, so no vacuum gap could be measured; treated as bulk"
        )
        return (0, 1, 2), record
    from .gate import _nearest_neighbour_distance_ang

    gaps = vacuum_gaps_ang(c.parent.lattice_ang, frac)
    nn = _nearest_neighbour_distance_ang(c)
    vacuum = tuple(i for i in single if gaps[i] >= VACUUM_GAP_NN_MULTIPLE * nn)
    record["vacuum_gap_ang"] = {str(i): gaps[i] for i in single}
    record["nearest_neighbour_ang"] = nn
    record["vacuum_axes"] = list(vacuum)
    return tuple(i for i in range(3) if i not in vacuum), record


def decay_lengths_ang(model: RealSpaceModel, fermi_energy_ev: float) -> dict:
    """Fitted decay length tau of ||H(R)||_F with |R|.

    ``tau_ang`` is the one-parameter fit the discussion specified,
    ||H(R)||_F = ||H(0)||_F exp(-|R|/tau), least squares in the logarithm over every
    non-zero lattice vector with a non-zero block. H(0) is taken with E_F subtracted from
    its diagonal: the on-site energies otherwise carry the code's arbitrary energy zero
    (QE's E_F sits near 10 eV, VASP's near 5 eV) and tau would measure that. A
    free-intercept fit over the same points, independent of H(0), is reported beside it.
    """
    norms = np.linalg.norm(model.hoppings, axis=(1, 2))
    origin = model.lengths_ang == 0
    if not origin.any():
        raise NonEvaluable("the real-space Hamiltonian has no R = 0 block")
    onsite = model.hoppings[origin][0] - fermi_energy_ev * np.eye(
        model.hoppings.shape[1]
    )
    f0 = float(np.linalg.norm(onsite))
    use = (~origin) & (norms > 0)
    out = {"tau_ang": None, "tau_free_intercept_ang": None, "n_vectors_fitted": 0}
    if use.sum() < 2 or f0 <= 0:
        return out
    x = model.lengths_ang[use]
    y = np.log(norms[use])
    slope = float(np.sum(x * (y - math.log(f0))) / np.sum(x * x))
    free = float(np.polyfit(x, y, 1)[0])
    out["tau_ang"] = -1.0 / slope if slope < 0 else math.inf
    out["tau_free_intercept_ang"] = -1.0 / free if free < 0 else math.inf
    out["n_vectors_fitted"] = int(use.sum())
    return out


# ---------------------------------------------------------------------------
# shared pieces of the three checks
# ---------------------------------------------------------------------------


def _reference(c: Construction) -> ReferenceBandPath:
    ref = getattr(c, "reference_path", None)
    if ref is None:
        raise NonEvaluable(
            f"{c.construction_id}: no reference band path was supplied; the acceptance "
            "criteria are scored against DFT bands on a separate high-symmetry path"
        )
    return ref


def _criterion_fermi(c: Construction, ref: ReferenceBandPath) -> tuple[float, str]:
    """The scf Fermi level: the reference's, else a deck ``fermi_energy`` in the inputs."""
    if ref.fermi_energy_ev is not None:
        return float(ref.fermi_energy_ev), ref.fermi_source or "reference path (scf)"
    deck = c.inputs.get("fermi_energy")
    if deck is not None:
        return float(str(deck).lower().replace("d", "e")), "deck fermi_energy"
    raise NonEvaluable(
        f"{c.construction_id}: no scf Fermi level -- neither the reference path nor "
        "the deck carries one"
    )


def _path_bands(c: Construction) -> tuple[ReferenceBandPath, np.ndarray, np.ndarray]:
    """The reference, its DFT bands for this spin, and the model's bands on its path."""
    ref = _reference(c)
    dft = ref.channel(int(c.inputs.get("spin_index", 0)))
    wan = _model(c).bands(ref.kpoints_frac)
    return ref, dft, wan


def _exclude(c: Construction) -> tuple[int, ...]:
    return tuple(int(i) for i in (c.inputs.get("exclude_bands") or ()))


def _eta(dft, wan, reference_ev, c: Construction, **kwargs):
    try:
        result = band_distance_eta(
            dft,
            wan,
            reference_ev,
            nu_ev=ETA_NU_EV,
            sigma_ev=ETA_SIGMA_EV,
            exclude_bands=_exclude(c),
            **kwargs,
        )
    except ValueError as exc:
        raise NonEvaluable(f"{c.construction_id}: {exc}") from exc
    return result


def _finite(value: float) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def insulator_classification(
    c: Construction, ref: ReferenceBandPath, fermi_energy_ev: float
) -> dict[str, Any]:
    """Decision 10's insulator rule, with the band edges and where E_F sits.

    An insulator has no band (of either spin) crossing E_F on the construction mesh
    **and** a gap of at least sigma = 0.1 eV. The edges are taken over the mesh and the
    path together -- the VBM is the higher of the two maxima below E_F, the CBM the lower
    of the two minima above it -- so a gap the path sees and the mesh misses (strontium:
    0.071 eV on the mesh, 0.020 eV on the path) is not overstated.
    """
    ef = float(fermi_energy_ev)
    mesh = np.asarray(c.parent.eigenvalues_ev, dtype=float)
    path = ref.bands_ev
    crossing_mesh = int(
        sum(np.count_nonzero((e.min(axis=0) < ef) & (e.max(axis=0) > ef)) for e in mesh)
    )
    crossing_path = int(
        sum(np.count_nonzero((e.min(axis=0) < ef) & (e.max(axis=0) > ef)) for e in path)
    )

    def edges(e):
        below, above = e[e <= ef], e[e > ef]
        return (
            float(below.max()) if below.size else None,
            float(above.min()) if above.size else None,
        )

    vbm_mesh, cbm_mesh = edges(mesh)
    vbm_path, cbm_path = edges(path)
    vbms = [v for v in (vbm_mesh, vbm_path) if v is not None]
    cbms = [v for v in (cbm_mesh, cbm_path) if v is not None]
    vbm = max(vbms) if vbms else None
    cbm = min(cbms) if cbms else None
    gap = None if vbm is None or cbm is None else cbm - vbm
    is_insulator = bool(crossing_mesh == 0 and gap is not None and gap >= ETA_SIGMA_EV)
    record: dict[str, Any] = {
        "is_insulator": is_insulator,
        "insulator_rule": "no band crosses E_F on the mesh and gap >= sigma = 0.1 eV",
        "crossing_bands_mesh": crossing_mesh,
        "crossing_bands_path": crossing_path,
        "vbm_ev": vbm,
        "cbm_ev": cbm,
        "gap_ev": gap,
        "gap_mesh_ev": (
            None if vbm_mesh is None or cbm_mesh is None else cbm_mesh - vbm_mesh
        ),
        "gap_path_ev": (
            None if vbm_path is None or cbm_path is None else cbm_path - vbm_path
        ),
        "ef_minus_vbm_ev": None if vbm is None else ef - vbm,
        "cbm_minus_ef_ev": None if cbm is None else cbm - ef,
    }
    if is_insulator:
        # How much of the conduction band an eta at E_F + 2 eV sees: 2 - (CBM - E_F).
        record["eta2_conduction_coverage_ev"] = ETA_NU_EV - (cbm - ef)
    record.update(occupation_count(c.parent, ef, crossing_mesh))
    return record


def occupation_count(parent, fermi_energy_ev: float, crossing: int) -> dict[str, Any]:
    """The electrons the filled bands hold, against the parent's own electron count.

    A gapped parent's bands are each full at every k-point or empty at every k-point, so
    the filled bands, times the electrons a band holds (two for a spin-degenerate band,
    one for a collinear spin channel's band or a spinor band), must give the parent's
    valence electron count exactly. This is the occupation logic of a spinor parent made
    explicit: a spin-orbit Pt or Bi2Se3 has one channel of bands holding one electron
    each, and reading them as two-electron bands would put the filled-band edge at half
    the occupied manifold. ``None`` entries say the count cannot be made (no electron
    count recorded, or a band crosses the level).
    """
    spinor = bool(getattr(parent, "spinor", False))
    e = np.asarray(parent.eigenvalues_ev, dtype=float)
    per_band = 1 if spinor or e.shape[0] == 2 else 2
    filled = int(np.count_nonzero(e.max(axis=1) <= fermi_energy_ev))
    n_electrons = getattr(parent, "n_electrons", None)
    consistent = None
    if n_electrons is not None and crossing == 0:
        consistent = bool(abs(filled * per_band - float(n_electrons)) < 1e-6)
    return {
        "spinor": spinor,
        "electrons_per_band": per_band,
        "filled_bands_mesh": filled,
        "filled_band_electrons": filled * per_band,
        "parent_electrons": None if n_electrons is None else float(n_electrons),
        "occupation_consistent": consistent,
    }


PASS_AT_20_MEV = ScaleAwareThreshold(
    ETA2_SUCCESS_MEV,
    "published_absolute_mev",
    calibration="20 meV: the success criterion on eta_2 of Qiao, Pizzi & Marzari 2023 "
    "(arXiv:2303.07877) and Jiang et al. 2025 (arXiv:2507.06840), with sigma = 0.1 eV. A "
    "published absolute criterion, adopted with its source so that a pass here is a pass "
    "in the paper's sense; applied unchanged to all three acceptance criteria "
    "(decision 2, 2026-10-06).",
)


# ---------------------------------------------------------------------------
# the three checks
# ---------------------------------------------------------------------------


def _band_interpolation_eta2(c: Construction) -> CheckOutcome:
    """eta_2 at the scf E_F + 2 eV on the reference path."""
    ref, dft, wan = _path_bands(c)
    ef, ef_source = _criterion_fermi(c, ref)
    result = _eta(dft, wan, ef, c)
    if not np.isfinite(result.eta_mev):
        raise NonEvaluable(f"{c.construction_id}: eta_2 is not finite on the path")
    nscf = getattr(c.parent, "fermi_energy_ev", None)
    at_nscf = _eta(dft, wan, nscf, c).eta_mev if nscf is not None else None
    record = {
        "eta2_max_mev": _finite(result.eta_max_mev),
        "mu_ev": ef + ETA_NU_EV,
        "fermi_energy_scf_ev": ef,
        "fermi_energy_source": ef_source,
        "fermi_energy_nscf_ev": None if nscf is None else float(nscf),
        "eta2_mev_at_nscf_fermi": _finite(at_nscf),
        "n_kpoints": result.n_kpoints,
        "n_bands_compared": result.n_bands_compared,
        "excluded_dft_bands": list(result.excluded_dft_bands),
        "ws_distance_applied": _model(c).ws_distance_applied,
        "reference": ref.source,
    }
    record.update(insulator_classification(c, ref, ef))
    return CheckOutcome(
        passed=None,
        metric_value=float(result.eta_mev),
        metric_units="meV",
        note=f"eta_2 = {result.eta_mev:.2f} meV at mu = E_F(scf) + 2 = "
        f"{ef + ETA_NU_EV:.3f} eV over {result.n_kpoints} path k-points and "
        f"{result.n_bands_compared} bands",
        record=record,
    )


def _truncated_hopping_eta2(c: Construction) -> CheckOutcome:
    """eta_2 of the model with every hopping longer than L_BvK / 4 removed."""
    ref = _reference(c)
    dft = ref.channel(int(c.inputs.get("spin_index", 0)))
    ef, ef_source = _criterion_fermi(c, ref)
    mesh = c.inputs.get("mp_grid")
    if mesh is None:
        raise NonEvaluable(f"{c.construction_id}: the k-mesh is not recorded")
    try:
        axes, periodicity = periodic_axes(c)
        length = bvk_length_ang(
            c.parent.lattice_ang, mesh, None if len(axes) == 3 else axes
        )
    except ValueError as exc:
        raise NonEvaluable(f"{c.construction_id}: {exc}") from exc
    model = _model(c)

    def at(fraction: float | None) -> tuple[float, int]:
        radius = None if fraction is None else fraction * length
        bands = model.bands(ref.kpoints_frac, radius)
        kept = int(np.count_nonzero(model.kept_elements(radius).any(axis=(1, 2))))
        return _eta(dft, bands, ef, c).eta_mev, kept

    eta, kept = at(TRUNCATION_FRACTION)
    if not np.isfinite(eta):
        raise NonEvaluable(f"{c.construction_id}: truncated eta_2 is not finite")
    full, total = at(None)
    record: dict[str, Any] = {
        "l_bvk_ang": length,
        "l_bvk_definition": (
            "shortest non-zero vector of the mp_grid superlattice"
            if len(axes) == 3
            else "shortest non-zero vector of the mp_grid superlattice over the "
            f"periodic lattice directions {list(axes)} only (vacuum along "
            f"{periodicity['vacuum_axes']})"
        ),
        "periodic_dims": len(axes),
        "periodic_axes": list(axes),
        **periodicity,
        "r_c_ang": TRUNCATION_FRACTION * length,
        "n_vectors_kept": kept,
        "n_vectors_total": total,
        "truncation_length": (
            "hopping length |R + tau_j - tau_i|, tau = the Wannier centres"
            if model.element_lengths_ang is not None
            else "lattice vector length |R| (no Wannier centres recorded)"
        ),
        "n_elements_kept": int(
            np.count_nonzero(model.kept_elements(TRUNCATION_FRACTION * length))
        ),
        "n_elements_total": int(model.hoppings.size),
        "eta2_untruncated_mev": _finite(full),
        "fermi_energy_scf_ev": ef,
        "fermi_energy_source": ef_source,
        "ws_distance_applied": model.ws_distance_applied,
        "reference": ref.source,
    }
    for fraction, name in zip(DIAGNOSTIC_TRUNCATION_FRACTIONS, ("L3", "L2")):
        value, n = at(fraction)
        record[f"r_{name}_ang"] = fraction * length
        record[f"eta_trunc_{name}_mev"] = _finite(value)
        record[f"n_vectors_kept_{name}"] = n
    record.update(decay_lengths_ang(model, ef))
    return CheckOutcome(
        passed=None,
        metric_value=float(eta),
        metric_units="meV",
        note=f"eta_2 = {eta:.2f} meV with hoppings longer than r_c = L_BvK/4 = "
        f"{TRUNCATION_FRACTION * length:.2f} Ang (|R + tau_j - tau_i|) removed ({kept} "
        f"of {total} lattice vectors keep a hopping; untruncated {full:.2f} meV)"
        + (
            ""
            if len(axes) == 3
            else f"; {len(axes)}D: L_BvK over the periodic directions {list(axes)} only"
        ),
        record=record,
    )


def _conduction_band_eta(c: Construction) -> CheckOutcome:
    """Cumulative eta at mu = CBM + 2 eV, for insulators."""
    ref = _reference(c)
    ef, ef_source = _criterion_fermi(c, ref)
    classification = insulator_classification(c, ref, ef)
    classification["fermi_energy_scf_ev"] = ef
    classification["fermi_energy_source"] = ef_source
    if (
        classification["is_insulator"]
        and classification["occupation_consistent"] is False
    ):
        # The gap the rule found is not where the electrons end: either E_F is not the
        # parent's, or the bands were read with the wrong electrons per band. Neither
        # licenses scoring the conduction band against that edge.
        raise NonEvaluable(
            f"{c.construction_id}: the filled bands below E_F hold "
            f"{classification['filled_band_electrons']} electrons at "
            f"{classification['electrons_per_band']} per band, but the parent has "
            f"{classification['parent_electrons']:g}; the CBM is not placed"
        )
    if not classification["is_insulator"]:
        gap = classification["gap_ev"]
        raise NotApplicable(
            "not an insulator by the rule (no band crossing E_F on the mesh and gap >= "
            f"0.1 eV): {classification['crossing_bands_mesh']} crossing band(s) on the "
            "mesh, gap " + ("none" if gap is None else f"{gap:.3f} eV"),
            record=classification,
        )
    _, dft, wan = _path_bands(c)
    cbm = classification["cbm_ev"]
    vbm = classification["vbm_ev"]
    cumulative = _eta(dft, wan, cbm, c)
    if not np.isfinite(cumulative.eta_mev):
        raise NonEvaluable(f"{c.construction_id}: conduction-band eta is not finite")
    midgap = 0.5 * (vbm + cbm)
    window = _eta(dft, wan, cbm, c, lower_cutoff_ev=midgap)
    record = dict(classification)
    record.update(
        {
            "mu_ev": cbm + ETA_NU_EV,
            "eta_max_mev": _finite(cumulative.eta_max_mev),
            "eta_window_only_mev": _finite(window.eta_mev),
            "window_only_definition": "same weights, states with DFT energy below "
            "mid-gap given zero weight: [CBM, CBM + 2 eV] only",
            "n_kpoints": cumulative.n_kpoints,
            "n_bands_compared": cumulative.n_bands_compared,
            "reference": ref.source,
        }
    )
    return CheckOutcome(
        passed=None,
        metric_value=float(cumulative.eta_mev),
        metric_units="meV",
        note=f"cumulative eta = {cumulative.eta_mev:.2f} meV at mu = CBM + 2 = "
        f"{cbm + ETA_NU_EV:.3f} eV; window-only "
        + (
            "n/a"
            if not np.isfinite(window.eta_mev)
            else f"{window.eta_mev:.2f} meV over [CBM, CBM + 2 eV]"
        ),
        record=record,
    )


BAND_INTERPOLATION_ETA2 = Check(
    "band_interpolation_eta2",
    "eta_2 of the model's bands on the reference path, at mu = E_F(scf) + 2 eV",
    "scalar_vs_scale",
    "dft_bands",
    _band_interpolation_eta2,
    threshold=PASS_AT_20_MEV,
    description="the Wannier-interpolated bands reproduce DFT bands on a separate "
    "high-symmetry path with eta_2 <= 20 meV (bands matched by index after excluded "
    "bands, Fermi-Dirac weights at the scf Fermi level + 2 eV, sigma 0.1 eV)",
)

TRUNCATED_HOPPING_ETA2 = Check(
    "truncated_hopping_eta2",
    "eta_2 on the reference path after removing every hopping longer than L_BvK / 4",
    "scalar_vs_scale",
    "dft_bands",
    _truncated_hopping_eta2,
    threshold=PASS_AT_20_MEV,
    description="after every hopping longer than a quarter of the Born-von Karman "
    "supercell size (L_BvK / 4) is removed -- a hopping's length being the distance "
    "between the two Wannier centres it connects -- the re-diagonalised model still "
    "reproduces the DFT path bands with eta_2 <= 20 meV",
)

CONDUCTION_BAND_ETA = Check(
    "conduction_band_eta",
    "cumulative eta on the reference path at mu = CBM + 2 eV (insulators only)",
    "scalar_vs_scale",
    "dft_bands",
    _conduction_band_eta,
    threshold=PASS_AT_20_MEV,
    description="for an insulator (no band crossing E_F on the mesh and a gap of at "
    "least 0.1 eV), the bands reproduce the DFT path bands up to the conduction-band "
    "minimum + 2 eV with cumulative eta <= 20 meV; withheld for a metal",
)

CRITERION_CHECKS: tuple[Check, ...] = (
    BAND_INTERPOLATION_ETA2,
    TRUNCATED_HOPPING_ETA2,
    CONDUCTION_BAND_ETA,
)
