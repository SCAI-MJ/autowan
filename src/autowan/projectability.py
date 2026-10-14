"""Projectability of Bloch states onto pseudo-atomic orbitals, and the erfc fit
that turns it into SCDM parameters.

The projectability of band ``n`` at ``k`` is the weight the pseudo-atomic orbital
set carries in that state,

    p_{nk} = sum_I |<phi_I | psi_{nk}>|^2,

which is 1 for a state the atomic basis spans exactly and falls toward 0 for the
free-electron-like states that live above the atomic manifold. Read as a function
of energy it is a smooth step, and SCDM-k needs exactly that step: its ``erfc``
weight ``f(e) = erfc((e - mu) / sigma) / 2`` is a parameterisation of the same
curve. Fitting the measured projectability therefore *derives* ``scdm_mu`` and
``scdm_sigma`` from the material rather than transplanting them from another one.

Both quantities come from a `projwfc.x` run on the same non-self-consistent
calculation the Wannierisation consumes, so no separate calculation is needed and
the two arms of a comparison stay on one electronic structure.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

#: Offset, in units of the fitted width, between the projectability step and the
#: SCDM chemical potential. The step's midpoint is where half the atomic weight is
#: already gone; taking mu there discards states the projectors still describe, so
#: mu is placed on the high-projectability side of it.
DEFAULT_MU_OFFSET_SIGMA = 3.0


class ProjectabilityError(RuntimeError):
    """The projections could not be read, or the fit did not converge."""


@dataclass(frozen=True)
class ScdmParameters:
    """SCDM-k parameters derived from a measured projectability curve."""

    mu: float
    """``scdm_mu`` (eV) — the erfc midpoint shifted onto the high-projectability side."""

    sigma: float
    """``scdm_sigma`` (eV) — the fitted width of the step."""

    mu_fit: float
    """The raw fitted midpoint, before the offset. Reported so the shift is visible."""

    n_states: int
    """Number of (band, k) states the fit consumed."""

    residual: float
    """Root-mean-square residual of the fit, in projectability units."""

    def as_pw2wannier_keywords(self) -> dict[str, object]:
        return {
            "scdm_proj": True,
            "scdm_entanglement": "erfc",
            "scdm_mu": round(self.mu, 4),
            "scdm_sigma": round(self.sigma, 4),
        }


def _erfc_step(energy: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * erfc((energy - mu) / sigma)


#: ``projwfc.x`` writes eigenvalues and the Fermi level in ``atomic_proj.xml`` in
#: Rydberg. The unit is not recorded in the file, so it is cross-checked against the
#: parent calculation instead of trusted — see :func:`read_atomic_proj`.
RY_TO_EV = 13.605693122994


@dataclass(frozen=True)
class AtomicProjections:
    """What ``atomic_proj.xml`` carries, flattened over spin and k."""

    energies_ev: np.ndarray
    projectability: np.ndarray
    fermi_energy_ev: float
    n_bands: int
    n_kpoints: int
    n_spin: int
    n_atomic_wfc: int


def read_atomic_proj(
    xml_path: str | Path, *, fermi_energy_ev: float | None = None
) -> AtomicProjections:
    """Read ``atomic_proj.xml``, in Rydberg, and return it converted to eV.

    ``projwfc.x`` writes one ``<PROJS>`` block per k-point and spin channel, each
    holding one ``<ATOMIC_WFC>`` element per pseudo-atomic orbital with that
    orbital's complex overlap against every band. The projectability of a state is
    the sum of the squared moduli over orbitals, so the return is flat in
    ``(spin, k, band)`` — the fit does not care which k-point a state came from.

    The file records no unit. Passing ``fermi_energy_ev`` from the parent
    self-consistent run turns that into a checked assumption rather than a silent
    one: a Hartree/Rydberg mix-up moves every energy by a factor of two, which the
    projectability curve absorbs into a plausible-looking but wrong fit.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise ProjectabilityError(f"no atomic_proj.xml at {xml_path}")

    root = ET.parse(xml_path).getroot()
    header = root.find("HEADER")
    if header is None:
        raise ProjectabilityError(f"{xml_path} carries no <HEADER>")

    n_bands = int(header.get("NUMBER_OF_BANDS", 0))
    n_kpoints = int(header.get("NUMBER_OF_K-POINTS", 0))
    n_spin = int(header.get("NUMBER_OF_SPIN_COMPONENTS", 1))
    n_atomic_wfc = int(header.get("NUMBER_OF_ATOMIC_WFC", 0))
    e_fermi = float(header.get("FERMI_ENERGY", "nan")) * RY_TO_EV

    if fermi_energy_ev is not None and abs(e_fermi - fermi_energy_ev) > 0.05:
        raise ProjectabilityError(
            f"{xml_path}: Fermi level reads {e_fermi:.4f} eV under the Rydberg "
            f"convention but the parent calculation reports {fermi_energy_ev:.4f} eV. "
            "The file is not the one this calculation produced, or its energy unit "
            "is not Rydberg — either way the projectability curve would be scaled."
        )

    eig_blocks = root.findall(".//E")
    proj_blocks = root.findall(".//PROJS")
    if not proj_blocks:
        raise ProjectabilityError(f"{xml_path} carries no <PROJS> block")
    if len(eig_blocks) != len(proj_blocks):
        raise ProjectabilityError(
            f"{xml_path}: {len(eig_blocks)} eigenvalue blocks against "
            f"{len(proj_blocks)} projection blocks"
        )

    energies: list[np.ndarray] = []
    projectability: list[np.ndarray] = []

    for eig_el, proj_el in zip(eig_blocks, proj_blocks):
        eig = np.fromstring(eig_el.text or "", sep=" ") * RY_TO_EV
        weight = np.zeros_like(eig)
        for wfc in proj_el.findall("ATOMIC_WFC"):
            raw = np.fromstring(wfc.text or "", sep=" ")
            if raw.size != 2 * eig.size:
                raise ProjectabilityError(
                    f"{xml_path}: orbital block has {raw.size // 2} bands, "
                    f"eigenvalue block has {eig.size} — the projwfc run does not "
                    "match the nscf it is supposed to describe"
                )
            amp = raw[0::2] + 1j * raw[1::2]
            weight += np.abs(amp) ** 2
        energies.append(eig)
        projectability.append(weight)

    return AtomicProjections(
        energies_ev=np.concatenate(energies),
        projectability=np.concatenate(projectability),
        fermi_energy_ev=e_fermi,
        n_bands=n_bands,
        n_kpoints=n_kpoints,
        n_spin=n_spin,
        n_atomic_wfc=n_atomic_wfc,
    )


def fit_scdm_parameters(
    energies: np.ndarray,
    projectability: np.ndarray,
    *,
    mu_offset_sigma: float = DEFAULT_MU_OFFSET_SIGMA,
) -> ScdmParameters:
    """Fit ``erfc((e - mu)/sigma)/2`` to a measured projectability curve.

    The initial guess puts the midpoint at the energy where the running mean of the
    projectability first drops through one half, which is robust against the long
    low-projectability tail that free-electron states contribute.
    """
    energies = np.asarray(energies, dtype=float)
    projectability = np.clip(np.asarray(projectability, dtype=float), 0.0, 1.0)
    if energies.size != projectability.size:
        raise ProjectabilityError("energy and projectability arrays differ in length")
    if energies.size < 8:
        raise ProjectabilityError(
            f"only {energies.size} states available; the fit needs a populated curve"
        )

    order = np.argsort(energies)
    e_sorted = energies[order]
    p_sorted = projectability[order]

    window = max(3, e_sorted.size // 50)
    smooth = np.convolve(p_sorted, np.ones(window) / window, mode="same")
    below_half = np.nonzero(smooth < 0.5)[0]
    mu_guess = float(e_sorted[below_half[0]]) if below_half.size else float(np.median(e_sorted))
    sigma_guess = max(1.0, 0.05 * float(np.ptp(e_sorted)))

    try:
        (mu_fit, sigma_fit), _ = curve_fit(
            _erfc_step,
            e_sorted,
            p_sorted,
            p0=(mu_guess, sigma_guess),
            bounds=([e_sorted.min(), 1e-3], [e_sorted.max(), np.ptp(e_sorted)]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ProjectabilityError(f"erfc fit did not converge: {exc}") from exc

    residual = float(
        np.sqrt(np.mean((p_sorted - _erfc_step(e_sorted, mu_fit, sigma_fit)) ** 2))
    )
    return ScdmParameters(
        mu=float(mu_fit - mu_offset_sigma * sigma_fit),
        sigma=float(sigma_fit),
        mu_fit=float(mu_fit),
        n_states=int(e_sorted.size),
        residual=residual,
    )


def scdm_parameters_from_projwfc(
    atomic_proj_xml: str | Path,
    *,
    fermi_energy_ev: float | None = None,
    mu_offset_sigma: float = DEFAULT_MU_OFFSET_SIGMA,
) -> ScdmParameters:
    """Read a ``projwfc.x`` result and return the SCDM parameters it implies."""
    proj = read_atomic_proj(atomic_proj_xml, fermi_energy_ev=fermi_energy_ev)
    return fit_scdm_parameters(
        proj.energies_ev, proj.projectability, mu_offset_sigma=mu_offset_sigma
    )
