"""Read a construction's own overlap products: the projections and the band energies.

The ``.amn`` holds ``A_mn(k) = <psi_mk | g_n>`` for every kept band ``m`` and projector
``n``; the ``.eig`` the energies of the same bands. Together they answer two questions
the repair loop asks without running anything: how much of each Bloch state the
projectors describe (its projectability, ``p_mk = sum_n |A_mn(k)|^2``, which is at most
one only when the projectors are orthonormal), and which projector carries it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class OverlapReadError(RuntimeError):
    """The overlap products are absent or malformed."""


@dataclass(frozen=True)
class Projections:
    """``amn[k, m, n]`` with the band energies ``energies_ev[k, m]``."""

    amn: np.ndarray
    energies_ev: np.ndarray | None

    @property
    def projectability(self) -> np.ndarray:
        """``p[k, m] = sum_n |A_mn(k)|^2``."""
        return (np.abs(self.amn) ** 2).sum(axis=2)

    def weights(self) -> np.ndarray:
        """``w[k, m, n] = |A_mn(k)|^2``, the share of state (k, m) on projector n."""
        return np.abs(self.amn) ** 2


def read_amn(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise OverlapReadError(f"no projections at {path}")
    with path.open() as handle:
        handle.readline()
        try:
            nb, nk, nw = (int(x) for x in handle.readline().split()[:3])
        except ValueError as exc:
            raise OverlapReadError(f"{path}: unreadable header") from exc
        data = np.loadtxt(handle, ndmin=2)
    if data.shape[0] != nb * nk * nw or data.shape[1] < 5:
        raise OverlapReadError(
            f"{path}: {data.shape[0]} rows for {nb} bands x {nk} k-points x {nw} projectors"
        )
    amn = np.zeros((nk, nb, nw), dtype=complex)
    m = data[:, 0].astype(int) - 1
    n = data[:, 1].astype(int) - 1
    k = data[:, 2].astype(int) - 1
    amn[k, m, n] = data[:, 3] + 1j * data[:, 4]
    return amn


def read_eig(path: str | Path, n_bands: int, n_kpoints: int) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise OverlapReadError(f"no band energies at {path}")
    data = np.loadtxt(path, ndmin=2)
    if data.shape[0] != n_bands * n_kpoints:
        raise OverlapReadError(
            f"{path}: {data.shape[0]} rows for {n_bands} bands x {n_kpoints} k-points"
        )
    energies = np.zeros((n_kpoints, n_bands))
    energies[data[:, 1].astype(int) - 1, data[:, 0].astype(int) - 1] = data[:, 2]
    return energies


def read_projections(run_dir: str | Path, seed: str) -> Projections:
    run_dir = Path(run_dir)
    amn = read_amn(run_dir / f"{seed}.amn")
    nk, nb, _ = amn.shape
    try:
        energies = read_eig(run_dir / f"{seed}.eig", nb, nk)
    except OverlapReadError:
        energies = None
    return Projections(amn=amn, energies_ev=energies)
