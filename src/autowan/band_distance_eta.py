"""The band-interpolation error eta_nu (arXiv:2507.06840, eqs. 6–7).

Following Vitale et al., eta is the weighted RMS energy difference, with weight
sqrt(f_FD(DFT) * f_FD(Wannier)) at chemical potential E_F + nu and width sigma.
The maximum error is max(weight * abs(difference)). The reference implementation
also masks both Fermi weights where the DFT energy is at or below an absolute
lower cutoff, by default -30 eV.

The k-set is Wannier90's band path versus a pw.x bands run on the same path:
these points mostly lie off the construction mesh. Bands match by index after
removing the deck's 1-based DFT exclusions, truncated to the common band count;
there is no nearest-energy matching or energy alignment. E_F is supplied by the
caller in the bands' energy reference; the re-score uses the parent calculation's
E_F. This reproduces bands_distance_raw in aiida_wannier90_workflows without an
AiiDA dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ETA_NU_EV = 2.0
ETA_SIGMA_EV = 0.1
ETA2_SUCCESS_MEV = 20.0
REFERENCE_LOWER_CUTOFF_EV = -30.0


@dataclass(frozen=True)
class BandDistance:
    """An eta measurement and the conventions needed to reproduce it."""

    eta_mev: float
    eta_max_mev: float
    nu_ev: float
    sigma_ev: float
    fermi_energy_ev: float
    lower_cutoff_ev: float | None
    n_kpoints: int
    n_bands_compared: int
    excluded_dft_bands: tuple[int, ...]
    lower_edge_ev: float | None = None
    """The soft lower window edge, when one was applied (``None``: the plain eta)."""

    @property
    def passes(self) -> bool | None:
        """Whether eta meets the published 20 meV criterion, if evaluable."""
        if not np.isfinite(self.eta_mev):
            return None
        return bool(self.eta_mev <= ETA2_SUCCESS_MEV)


def band_distance_eta(
    dft_bands_ev,
    wannier_bands_ev,
    fermi_energy_ev,
    *,
    nu_ev=ETA_NU_EV,
    sigma_ev=ETA_SIGMA_EV,
    exclude_bands=(),
    lower_cutoff_ev=REFERENCE_LOWER_CUTOFF_EV,
    lower_edge_ev=None,
) -> BandDistance:
    """Compare corresponding bands using the authors' Fermi-Dirac weights.

    Non-finite compared energies or zero total weight give an unevaluable result,
    including when no bands remain after exclusions and truncation.

    ``lower_edge_ev`` turns the occupation into a soft window: each band's factor
    becomes ``f(e; mu, sigma) * (1 - f(e; lower_edge, sigma))``, the states between
    the lower edge and mu, and the two factors are combined exactly as before (their
    geometric mean, the -30 eV mask applied). ``None`` (the default) is the published
    eta, unchanged.
    """
    dft = np.asarray(dft_bands_ev, dtype=float)
    wan = np.asarray(wannier_bands_ev, dtype=float)
    if dft.ndim != 2 or wan.ndim != 2:
        raise ValueError("band arrays must be two-dimensional (k-point, band)")
    if dft.shape[0] != wan.shape[0]:
        raise ValueError("DFT and Wannier k-point counts differ")
    excluded = tuple(int(i) for i in exclude_bands)
    if any(i < 1 or i > dft.shape[1] for i in excluded):
        raise ValueError("excluded DFT band indices must be 1-based and in range")
    keep = np.ones(dft.shape[1], dtype=bool)
    if excluded:
        keep[np.asarray(excluded) - 1] = False
    dft = dft[:, keep]
    n = min(dft.shape[1], wan.shape[1])
    dft, wan = dft[:, :n], wan[:, :n]
    eta = eta_max = float("nan")
    if np.isfinite(dft).all() and np.isfinite(wan).all():
        mu = fermi_energy_ev + nu_ev
        with np.errstate(over="ignore"):
            f_dft = 1.0 / (np.exp((dft - mu) / sigma_ev) + 1.0)
            f_wan = 1.0 / (np.exp((wan - mu) / sigma_ev) + 1.0)
            if lower_edge_ev is not None:
                # 1 - f(e; lower) = f(-(e - lower)): empty below the lower edge
                f_dft = f_dft / (np.exp(-(dft - lower_edge_ev) / sigma_ev) + 1.0)
                f_wan = f_wan / (np.exp(-(wan - lower_edge_ev) / sigma_ev) + 1.0)
        mask = (
            np.ones_like(dft, dtype=int)
            if lower_cutoff_ev is None
            else (dft > lower_cutoff_ev).astype(int)
        )
        weight = np.sqrt((f_dft * mask) * (f_wan * mask))
        total = weight.sum()
        if total > 0:
            difference = dft - wan
            eta = float(np.sqrt(np.sum(difference**2 * weight) / total) * 1000.0)
            eta_max = float(np.max(np.abs(difference) * weight) * 1000.0)
    return BandDistance(
        eta_mev=eta,
        eta_max_mev=eta_max,
        nu_ev=float(nu_ev),
        sigma_ev=float(sigma_ev),
        fermi_energy_ev=float(fermi_energy_ev),
        lower_cutoff_ev=None if lower_cutoff_ev is None else float(lower_cutoff_ev),
        n_kpoints=int(dft.shape[0]),
        n_bands_compared=int(n),
        excluded_dft_bands=excluded,
        lower_edge_ev=None if lower_edge_ev is None else float(lower_edge_ev),
    )


def eta2_record(
    dft_bands_ev,
    wannier_bands_ev,
    fermi_energy_ev,
    exclude_bands=(),
    *,
    lower_cutoff_ev=REFERENCE_LOWER_CUTOFF_EV,
) -> dict:
    """Return eta_2 and its conventions as a JSON-serialisable record."""
    result = band_distance_eta(
        dft_bands_ev,
        wannier_bands_ev,
        fermi_energy_ev,
        exclude_bands=exclude_bands,
        lower_cutoff_ev=lower_cutoff_ev,
    )
    return {
        "eta2_mev": result.eta_mev if np.isfinite(result.eta_mev) else None,
        "eta2_max_mev": result.eta_max_mev if np.isfinite(result.eta_max_mev) else None,
        "eta2_pass_20mev": result.passes,
        "nu_ev": result.nu_ev,
        "sigma_ev": result.sigma_ev,
        "fermi_energy_ev": result.fermi_energy_ev,
        "lower_cutoff_ev": result.lower_cutoff_ev,
        "n_kpoints": result.n_kpoints,
        "n_bands_compared": result.n_bands_compared,
        "excluded_dft_bands": list(result.excluded_dft_bands),
    }


def path_points_on_mesh(kpoints_frac, mp_grid, tol=1e-6) -> int:
    """Count points on a Gamma-centred mesh modulo reciprocal lattice vectors."""
    scaled = np.asarray(kpoints_frac, dtype=float) * np.asarray(mp_grid)
    return int(
        np.count_nonzero(np.all(np.abs(scaled - np.rint(scaled)) <= tol, axis=1))
    )


def read_wannier_path_bands(path) -> np.ndarray:
    """Read blank-line-separated Wannier90 band blocks as (nk, num_wann)."""
    blocks = []
    energies = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            energies.append(float(line.split()[1]))
        elif energies:
            blocks.append(energies)
            energies = []
    if energies:
        blocks.append(energies)
    if not blocks:
        raise ValueError("Wannier90 band path contains no bands")
    return np.asarray(blocks, dtype=float).T


def read_wannier_path_kpoints(path) -> np.ndarray:
    """Read the declared path count and fractional points from seedname_band.kpt."""
    with Path(path).open() as stream:
        count = int(stream.readline().strip())
        points = np.loadtxt(stream, ndmin=2)
    if points.shape != (count, 4):
        raise ValueError("Wannier90 path k-points do not match the declared count")
    return points[:, :3]


def read_dft_path_bands(npz_path, spin_index=0) -> tuple[np.ndarray, np.ndarray]:
    """Read a DFT path archive, selecting a spin when the bands have a spin axis."""
    with np.load(npz_path) as archive:
        bands = np.asarray(archive["bands"], dtype=float)
        kpoints = np.asarray(archive["kpoints"], dtype=float)
    if bands.ndim == 3:
        bands = bands[spin_index]
    if bands.ndim != 2 or kpoints.shape != (bands.shape[0], 3):
        raise ValueError("DFT path requires bands (nk, nb) and kpoints (nk, 3)")
    return bands, kpoints


def eta2_for_channel(
    channel_dir,
    dft_bands_npz,
    fermi_energy_ev,
    *,
    spin_index=0,
    exclude_bands=None,
    mp_grid=None,
    seedname="aiida",
    lower_cutoff_ev=REFERENCE_LOWER_CUTOFF_EV,
) -> dict:
    """Score a channel's Wannier90 band path against the same DFT path."""
    from .inputs import read_excluded_bands

    channel = Path(channel_dir)
    band_path = channel / f"{seedname}_band.dat"
    if not band_path.is_file():
        # No other path artefact is required to report a missing interpolation.
        deck_path = channel / f"{seedname}.win"
        if exclude_bands is None and deck_path.is_file():
            exclude_bands = read_excluded_bands(deck_path)
        record = eta2_record(
            np.empty((0, 0)),
            np.empty((0, 0)),
            fermi_energy_ev,
            lower_cutoff_ev=lower_cutoff_ev,
        )
        record["excluded_dft_bands"] = (
            [] if exclude_bands is None else [int(i) for i in exclude_bands]
        )
        record["error"] = "no Wannier90 band path"
        return record
    if exclude_bands is None:
        exclude_bands = read_excluded_bands(channel / f"{seedname}.win")
    wan = read_wannier_path_bands(band_path)
    wan_kpoints = read_wannier_path_kpoints(channel / f"{seedname}_band.kpt")
    dft, dft_kpoints = read_dft_path_bands(dft_bands_npz, spin_index)
    if wan_kpoints.shape != dft_kpoints.shape or wan.shape[0] != len(wan_kpoints):
        raise ValueError("DFT and Wannier path k-point counts differ")
    difference = float(np.max(np.abs(wan_kpoints - dft_kpoints)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise ValueError("DFT and Wannier path k-points differ by more than 1e-5")
    record = eta2_record(
        dft,
        wan,
        fermi_energy_ev,
        exclude_bands,
        lower_cutoff_ev=lower_cutoff_ev,
    )
    record["max_path_kpoint_difference"] = difference
    if mp_grid is not None:
        record["path_points_on_construction_mesh"] = path_points_on_mesh(
            wan_kpoints, mp_grid
        )
        record["path_points"] = int(len(wan_kpoints))
    return record
