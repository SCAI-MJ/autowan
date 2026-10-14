"""Band interpolation from the real-space Wannier Hamiltonian.

The central question a certification gate asks is whether the constructed model
reproduces the calculation it came from. Answering it needs the model's bands at the
same k-points the parent calculation used, and those come from a Fourier sum over the
real-space Hamiltonian Wannier90 writes,

    H(k) = sum_R exp(2 pi i k . R) H(R) / deg(R),

with ``k`` in fractional coordinates and ``deg(R)`` the Wigner-Seitz degeneracy that
Wannier90 records but does not apply. Dividing by it is what makes the sum a genuine
inverse of the discrete transform rather than an over-counting of the boundary shells.

The comparison is only meaningful where the construction was asked to be faithful.
Above the outer window the model has no obligation to any particular band, so a
distance measured over the full band set reports the disentanglement's freedom as if
it were the construction's error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InterpolationError(RuntimeError):
    """The Hamiltonian and the k-points cannot be combined as given."""


def cartesian_to_fractional_k(
    kpoints_cart_2pi_alat: np.ndarray, lattice_ang: np.ndarray, alat_ang: float
) -> np.ndarray:
    """Convert Quantum ESPRESSO's k-points to fractional (crystal) coordinates.

    Quantum ESPRESSO writes k in cartesian units of ``2*pi/alat``. With the real
    lattice ``A`` (rows are the vectors) expressed in the same ``alat``, the crystal
    coordinates are ``A @ k_cart`` — the reciprocal vectors and the real ones are
    dual, so the matrix that maps one way is the transpose-inverse of the other and
    the factor of ``2*pi`` cancels against the unit of ``k``.
    """
    a_in_alat = np.asarray(lattice_ang, dtype=float) / float(alat_ang)
    return np.asarray(kpoints_cart_2pi_alat, dtype=float) @ a_in_alat.T


def interpolate(
    hr_matrix: np.ndarray,
    rvectors: np.ndarray,
    degeneracies: np.ndarray,
    kpoints_frac: np.ndarray,
) -> np.ndarray:
    """Eigenvalues of the interpolated model at each k-point, ascending.

    Returns ``(n_kpoints, num_wann)`` in the units of ``hr_matrix`` (eV as Wannier90
    writes it). The Hamiltonian at each k is symmetrised before diagonalisation:
    it is Hermitian by construction, and forcing it removes the accumulation of
    round-off in the sum rather than hiding an asymmetry that would signal a
    mis-parsed file — that is checked separately, at read time.
    """
    hr_matrix = np.asarray(hr_matrix)
    rvectors = np.asarray(rvectors, dtype=float)
    degeneracies = np.asarray(degeneracies, dtype=float)
    kpoints_frac = np.atleast_2d(np.asarray(kpoints_frac, dtype=float))

    if hr_matrix.shape[0] != rvectors.shape[0] != degeneracies.shape[0]:
        raise InterpolationError(
            f"{hr_matrix.shape[0]} matrices, {rvectors.shape[0]} R-vectors and "
            f"{degeneracies.shape[0]} degeneracies do not describe one Hamiltonian"
        )
    if np.any(degeneracies <= 0):
        raise InterpolationError("a Wigner-Seitz degeneracy is not positive")

    weighted = hr_matrix / degeneracies[:, None, None]
    phases = np.exp(2j * np.pi * (kpoints_frac @ rvectors.T))  # (n_k, n_R)
    h_of_k = np.einsum("kr,rij->kij", phases, weighted)
    h_of_k = 0.5 * (h_of_k + np.conjugate(np.transpose(h_of_k, (0, 2, 1))))
    return np.linalg.eigvalsh(h_of_k)


@dataclass(frozen=True)
class BandComparison:
    """How far the interpolated bands sit from the parent calculation's."""

    rms_error_ev: float
    max_error_ev: float
    n_states_compared: int
    n_states_available: int
    window_ev: tuple[float, float]

    @property
    def coverage(self) -> float:
        """Fraction of the parent's states that fell inside the comparison window.

        Reported because a small error over two states is not the same measurement
        as a small error over two thousand, and a window that admits almost nothing
        produces the former while looking like the latter.
        """
        if self.n_states_available == 0:
            return 0.0
        return self.n_states_compared / self.n_states_available


def compare_to_parent(
    interpolated_ev: np.ndarray,
    parent_ev: np.ndarray,
    *,
    window_ev: tuple[float, float],
) -> BandComparison:
    """Distance between the two band sets inside an energy window.

    At each k-point the parent states inside the window are paired with their nearest
    interpolated partner. Nearest-pairing rather than index-pairing because the
    construction spans a subspace, not a contiguous band index range: when
    disentanglement drops a band, index ``n`` of the model and index ``n`` of the
    parent stop describing the same state and an index-wise difference reports a
    relabelling as an error.
    """
    interpolated_ev = np.atleast_2d(interpolated_ev)
    parent_ev = np.atleast_2d(parent_ev)
    if interpolated_ev.shape[0] != parent_ev.shape[0]:
        raise InterpolationError(
            f"{interpolated_ev.shape[0]} interpolated k-points against "
            f"{parent_ev.shape[0]} in the parent calculation"
        )

    lo, hi = window_ev
    squared: list[float] = []
    worst = 0.0
    for model_k, parent_k in zip(interpolated_ev, parent_ev):
        inside = parent_k[(parent_k >= lo) & (parent_k <= hi)]
        if inside.size == 0 or model_k.size == 0:
            continue
        distance = np.min(np.abs(inside[:, None] - model_k[None, :]), axis=1)
        squared.extend(distance**2)
        worst = max(worst, float(distance.max()))

    n_available = int(np.count_nonzero((parent_ev >= lo) & (parent_ev <= hi)))
    if not squared:
        return BandComparison(float("nan"), float("nan"), 0, n_available, window_ev)
    return BandComparison(
        rms_error_ev=float(np.sqrt(np.mean(squared))),
        max_error_ev=worst,
        n_states_compared=len(squared),
        n_states_available=n_available,
        window_ev=window_ev,
    )
