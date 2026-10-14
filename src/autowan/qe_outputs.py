"""The parent calculation, read from Quantum ESPRESSO's XML.

Every check the certification gate applies resolves against the calculation the
Wannier functions came from, so the gate needs that calculation on its own terms:
the band energies it produced, the Fermi level it settled on, the cell and the
occupation of the states. Reading them from ``data-file-schema.xml`` rather than
from the human-readable output keeps the numbers at full precision and makes the
k-point ordering explicit, which the interpolation comparison depends on.

Quantum ESPRESSO writes this file in **Hartree**, unlike ``atomic_proj.xml``, which
is in Rydberg. The two live in the same directory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


class QeOutputError(RuntimeError):
    """The Quantum ESPRESSO XML is absent or does not carry what was asked of it."""


@dataclass
class ParentCalculation:
    """The self-consistent (or non-self-consistent) run a construction is judged against."""

    eigenvalues_ev: np.ndarray
    """``(n_spin, n_kpoints, n_bands)`` band energies."""

    occupations: np.ndarray
    """Same shape as :attr:`eigenvalues_ev`."""

    kpoints_cart: np.ndarray
    """``(n_kpoints, 3)`` in units of 2*pi/alat, as written."""

    fermi_energy_ev: float
    lattice_ang: np.ndarray
    alat_ang: float
    """The length unit Quantum ESPRESSO writes its k-points against.

    Needed to convert them to crystal coordinates: without it the Fourier sum over
    the real-space Hamiltonian is evaluated at the wrong points and returns bands
    that are smooth, plausible, and unrelated to the calculation.
    """

    species: tuple[str, ...]
    positions_frac: np.ndarray
    n_symmetries: int
    n_atomic_wfc: int | None
    converged: bool | None
    """Quantum ESPRESSO's own self-consistency flag.

    ``None`` when the file describes a non-self-consistent run, which records no
    such flag: the parent's convergence must then be read from the self-consistent
    calculation that produced the density, not inferred from silence here.
    """

    calculation: str
    """``scf``, ``nscf``, ``bands`` … as the file records it."""

    spinor: bool = False
    """Whether every band is a two-component spinor (QE ``noncolin``, VASP
    ``LNONCOLLINEAR``; spin-orbit or not).

    A spinor band holds one electron, not two, and there is one channel of them: a
    spin-orbit parent has ``n_spin == 1`` with twice the bands a scalar-relativistic one
    carries. Everything that turns electrons into bands, or functions into bands, reads
    this flag through :attr:`electrons_per_band`.
    """

    n_electrons: float | None = None
    """The valence electron count of the calculation (QE ``nelec``, VASP ``NELECT``)."""

    @property
    def n_spin(self) -> int:
        return self.eigenvalues_ev.shape[0]

    @property
    def electrons_per_band(self) -> int:
        """Electrons a fully occupied band holds in one channel: 1 for a spinor band or
        a collinear spin channel, 2 for a spin-degenerate band."""
        return 1 if self.spinor or self.n_spin == 2 else 2

    def filled_band_electrons(self) -> float:
        """Electrons in the bands lying entirely below the Fermi level, all channels.

        Exact for a gapped parent: every band is then either full at every k-point or
        empty at every k-point, so this must equal :attr:`n_electrons`. A mismatch means
        the Fermi level, the band count or the electrons-per-band reading is wrong.
        """
        below = self.eigenvalues_ev.max(axis=1) <= self.fermi_energy_ev
        return float(np.count_nonzero(below) * self.electrons_per_band)

    def occupation_consistent(self) -> bool | None:
        """Whether a gapped parent's filled bands hold exactly its electrons.

        ``None`` when it cannot be asked: no electron count is recorded, or a band
        crosses the Fermi level (a metal's bands are partly filled by construction).
        """
        if self.n_electrons is None or self.is_metallic():
            return None
        return abs(self.filled_band_electrons() - float(self.n_electrons)) < 1e-6

    @property
    def n_bands(self) -> int:
        return self.eigenvalues_ev.shape[2]

    def crossing_bands(self) -> int:
        """How many bands run below the Fermi level at one k and above it at another.

        This is the discrete metallicity determination the gate branches on, and it
        is a crossing test rather than a proximity test on purpose. Quantum ESPRESSO
        places the Fermi level of a gapped material at the valence-band maximum, so
        "is there a state near the Fermi level" answers *yes* for every insulator —
        the valence band touches it by construction. A band that is below the level
        somewhere and above it somewhere else is what a partially filled band is.
        """
        below = self.eigenvalues_ev.min(axis=1) < self.fermi_energy_ev
        above = self.eigenvalues_ev.max(axis=1) > self.fermi_energy_ev
        return int(np.count_nonzero(below & above))

    def is_metallic(self) -> bool:
        """Whether any band crosses the Fermi level.

        A material with an isolated occupied manifold admits a band-distance
        criterion; one whose bands cross the level does not, because its Wannier
        subspace is carved out of a continuum and the excluded weight is not an
        error of the construction.
        """
        return self.crossing_bands() > 0

    def occupied_bands(self) -> int:
        """How many bands lie entirely below the Fermi level at every k-point.

        The floor on a projector set for a gapped material: a construction with fewer
        functions than this cannot represent the occupied manifold, whatever else it
        gets right.
        """
        return int(
            np.count_nonzero(self.eigenvalues_ev.max(axis=1) <= self.fermi_energy_ev)
        )

    def band_gap_ev(self) -> float:
        """Gap across the Fermi level; ``0.0`` when a band crosses it."""
        below = self.eigenvalues_ev[self.eigenvalues_ev <= self.fermi_energy_ev]
        above = self.eigenvalues_ev[self.eigenvalues_ev > self.fermi_energy_ev]
        if below.size == 0 or above.size == 0:
            return 0.0
        return max(0.0, float(above.min() - below.max()))


def _text(root: ET.Element, path: str) -> str:
    el = root.find(path)
    if el is None or el.text is None:
        raise QeOutputError(f"missing <{path}> in the Quantum ESPRESSO XML")
    return el.text.strip()


def read_parent(xml_path: str | Path) -> ParentCalculation:
    """Read ``<prefix>.save/data-file-schema.xml``."""
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise QeOutputError(f"no Quantum ESPRESSO XML at {xml_path}")
    root = ET.parse(xml_path).getroot()

    bs = root.find(".//band_structure")
    if bs is None:
        raise QeOutputError(f"{xml_path} carries no <band_structure>")

    def flag(name: str) -> bool:
        el = bs.find(name)
        return el is not None and "true" in (el.text or "").lower()

    # noncolin (with or without spinorbit): one channel of two-component spinor bands,
    # each holding one electron. lsda: two channels of scalar bands. Never both.
    spinor = flag("noncolin")
    n_spin = 2 if flag("lsda") else 1
    if spinor and n_spin == 2:
        raise QeOutputError(
            f"{xml_path} declares both lsda and noncolin; a spinor calculation has one "
            "channel"
        )

    eig, occ, kpts = [], [], []
    for ks in bs.findall("ks_energies"):
        kp = ks.find("k_point")
        kpts.append([float(x) for x in (kp.text or "").split()])
        eig.append(np.fromstring(_text(ks, "eigenvalues"), sep=" ") * HARTREE_TO_EV)
        occ.append(np.fromstring(_text(ks, "occupations"), sep=" "))
    if not eig:
        raise QeOutputError(f"{xml_path} carries no <ks_energies>")

    # For a spin-polarised run the two channels are concatenated within one
    # ks_energies block, so the band axis is split rather than the k axis.
    stacked = np.array(eig)
    n_total = stacked.shape[1]
    if n_total % n_spin:
        raise QeOutputError(
            f"{xml_path}: {n_total} eigenvalues per k-point does not divide into "
            f"{n_spin} spin channels"
        )
    n_bands = n_total // n_spin
    eigenvalues = stacked.reshape(len(eig), n_spin, n_bands).transpose(1, 0, 2)
    occupations = np.array(occ).reshape(len(occ), n_spin, n_bands).transpose(1, 0, 2)

    fermi_el = bs.find("fermi_energy")
    if fermi_el is None or fermi_el.text is None:
        raise QeOutputError(
            f"{xml_path} records no Fermi level: the gate cannot place a frontier "
            "manifold without one"
        )
    fermi = float(fermi_el.text) * HARTREE_TO_EV

    struct = root.findall(".//atomic_structure")[-1]
    alat_bohr = struct.get("alat")
    if alat_bohr is None:
        raise QeOutputError(f"{xml_path}: <atomic_structure> records no alat")
    cell = struct.find("cell")
    if cell is None:
        raise QeOutputError(f"{xml_path} carries no <cell>")
    lattice = (
        np.array(
            [[float(x) for x in _text(cell, tag).split()] for tag in ("a1", "a2", "a3")]
        )
        * BOHR_TO_ANG
    )

    species, positions = [], []
    for atom in struct.findall(".//atom"):
        species.append(atom.get("name", "?"))
        positions.append([float(x) for x in (atom.text or "").split()])
    cart = np.array(positions) * BOHR_TO_ANG if positions else np.zeros((0, 3))
    frac = cart @ np.linalg.inv(lattice) if cart.size else cart

    nsym_el = root.find(".//symmetries/nsym")
    conv_el = root.find(".//convergence_achieved")
    calc_el = root.find(".//control_variables/calculation")
    calculation = (calc_el.text or "").strip() if calc_el is not None else ""
    if conv_el is None or conv_el.text is None:
        converged = None
    else:
        converged = "true" in conv_el.text.lower()
    n_wfc = struct.get("num_of_atomic_wfc")
    nelec_el = bs.find("nelec")
    n_electrons = (
        float(nelec_el.text) if nelec_el is not None and nelec_el.text else None
    )

    return ParentCalculation(
        eigenvalues_ev=eigenvalues,
        occupations=occupations,
        kpoints_cart=np.array(kpts),
        fermi_energy_ev=fermi,
        lattice_ang=lattice,
        alat_ang=float(alat_bohr) * BOHR_TO_ANG,
        species=tuple(species),
        positions_frac=frac,
        n_symmetries=int(nsym_el.text) if nsym_el is not None and nsym_el.text else 0,
        n_atomic_wfc=int(n_wfc) if n_wfc else None,
        converged=converged,
        calculation=calculation,
        spinor=spinor,
        n_electrons=n_electrons,
    )
