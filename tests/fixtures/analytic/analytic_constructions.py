"""Closed-form parents and models for the gate-extension tests.

Every parent is a one-dimensional tight-binding toy on an eight-point mesh along x of a cubic
cell whose lattice vectors equal alat, so Cartesian k in 2pi/alat equals fractional k. Each band
is a written formula; nothing is fitted and nothing is read from a calculation, so every number a
test expects can be derived by hand. ``PROVENANCE.md`` beside this file states each formula.

The bands are chosen so that no two of them cross between mesh points: a crossing that falls
between two sampled k-points leaves an artificial sliver between the sorted bands, and the tests
here are about real gaps (semicore, spin) and within-band sampling, not about that artefact.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autowan.gate import Construction
from autowan.qe_outputs import ParentCalculation

LATTICE = np.eye(3) * 4.0
ALAT = 4.0
N_K = 8
K_FRAC = np.array([[j / N_K, 0.0, 0.0] for j in range(N_K)])
K = K_FRAC[:, 0]
FERMI_EV = 0.0
SEMICORE_EV = -50.0


def _cos(k):
    return np.cos(2.0 * np.pi * np.asarray(k, dtype=float))


def flat(value):
    return np.full(N_K, float(value))


def s_band(k=K):
    """A valence s-like band, -6 .. -2 eV."""
    return -6.0 + 2.0 * (1.0 - _cos(k))


def d_band(k=K):
    """The magnetic d-like band, -3 .. +0.5 eV. It crosses E_F = 0, so a parent holding it is a metal.

    It stays at least 2.5 eV above ``s_band`` at every k, so the two never cross while their
    energy ranges overlap (-3 .. -2 eV): the valence block of an s + d parent has no gap.
    """
    return -3.0 + 1.75 * (1.0 - _cos(k))


#: The top occupied state on the mesh: the d band at k = 3/8, -0.0126 eV.
VALENCE_TOP_EV = float(d_band()[d_band() <= FERMI_EV].max())
VALENCE_BOTTOM_EV = -6.0
#: Spread of the s + d valence block, 5.9874 eV.
VALENCE_BANDWIDTH_EV = VALENCE_TOP_EV - VALENCE_BOTTOM_EV
#: Spread of every d state on the mesh (-3 .. +0.5 eV), occupied or not.
D_MANIFOLD_BANDWIDTH_EV = 3.5


def make_parent(bands_per_spin) -> ParentCalculation:
    """``bands_per_spin``: over spins, the (N_K,) band arrays listed in ascending order at every k."""
    eig = np.array(
        [np.stack(list(bands), axis=1) for bands in bands_per_spin], dtype=float
    )
    if np.any(np.diff(eig, axis=2) < 0):
        raise ValueError("bands must be listed in ascending order at every k-point")
    return ParentCalculation(
        eigenvalues_ev=eig,
        occupations=(eig <= FERMI_EV).astype(float),
        kpoints_cart=K_FRAC.copy(),
        fermi_energy_ev=FERMI_EV,
        lattice_ang=LATTICE,
        alat_ang=ALAT,
        species=("Fe",),
        positions_frac=np.zeros((1, 3)),
        n_symmetries=1,
        n_atomic_wfc=None,
        converged=True,
        calculation="scf",
    )


def semicore_s_d_parent() -> ParentCalculation:
    """A flat semicore level at -50 eV under the s + d valence block (a 44 eV gap)."""
    return make_parent([[flat(SEMICORE_EV), s_band(), d_band()]])


def s_d_parent() -> ParentCalculation:
    """The same valence block with no semicore level."""
    return make_parent([[s_band(), d_band()]])


#: Magnetic-d weight of each state of ``semicore_s_d_parent``: 0 on the semicore level, 0.1 on
#: the s band, 0.9 on the d band. Shape (1 spin, N_K, 3 bands), aligned with its eigenvalues.
MAGNETIC_D_WEIGHT = np.tile(np.array([0.0, 0.1, 0.9]), (1, N_K, 1))

_R3 = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])


def hr_cosine(onsite, hop):
    """One band ``e(k) = onsite + 2 hop cos(2 pi k)`` as a Wannier90-style H(R)."""
    return (
        np.array([[[hop]], [[onsite]], [[hop]]], dtype=complex),
        _R3.copy(),
        np.ones(3),
    )


def hr_diagonal(blocks):
    """Uncoupled bands, one ``(onsite, hop)`` pair each, as a block-diagonal H(R)."""
    n = len(blocks)
    h = np.zeros((3, n, n), dtype=complex)
    for i, (onsite, hop) in enumerate(blocks):
        h[0, i, i] = hop
        h[1, i, i] = onsite
        h[2, i, i] = hop
    return h, _R3.copy(), np.ones(3)


def hr_from_band(values):
    """The one-band H(R) whose interpolated band equals ``values`` exactly on the mesh.

    A discrete Fourier transform over R = -3 .. 4 (degeneracy 1): on an eight-point mesh these
    eight vectors are a complete set, so the interpolation reproduces every sampled value.
    """
    values = np.asarray(values, dtype=float)
    rs = np.arange(-(N_K // 2) + 1, N_K // 2 + 1)
    kj = np.arange(N_K) / N_K
    h = np.array([[[np.mean(values * np.exp(-2j * np.pi * kj * r))]] for r in rs])
    return h, np.array([[r, 0, 0] for r in rs]), np.ones(len(rs))


#: A d-only model that reproduces ``d_band`` exactly: -1.25 - 1.75 cos(2 pi k).
D_ONLY_MODEL = hr_cosine(-1.25, -0.875)
#: The same model with its on-site energy 0.3 eV too high: it misses every d state.
D_ONLY_MODEL_SHIFTED = hr_cosine(-1.25 + 0.3, -0.875)
#: A d-only model with no dispersion: it meets the d band only where cos(2 pi k) = 0.
D_ONLY_MODEL_FLAT = hr_cosine(-1.25, 0.0)


def wannier(hr, **extra):
    h, rvec, deg = hr
    fields = dict(
        num_wann=h.shape[1],
        hr_matrix=h,
        hr_rvectors=rvec,
        hr_degeneracies=deg,
        centres_ang=None,
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def construction(parent, hr, *, material_class=None, target=None, inputs=None, **extra):
    """A construction of ``hr`` over ``parent``. ``target=None`` leaves the target undeclared."""
    w = wannier(hr, **extra)
    kwargs = {} if target is None else {"target": target}
    return Construction(
        "analytic",
        parent=parent,
        wannier=w,
        material_class=material_class
        or ("metal" if parent.is_metallic() else "gapped"),
        inputs={
            "spin_index": 0,
            "num_wann": w.num_wann,
            "mp_grid": (N_K, 1, 1),
            **(inputs or {}),
        },
        **kwargs,
    )


class TwoOrbitalParent:
    """A magnetic d orbital hybridised with one flat ligand p orbital (an OpenMX-like parent).

    ``H(k) = [[e_d(k), t], [t, e_p]]`` in an orthonormal two-orbital basis (d is orbital 0 on
    atom 0, p is orbital 1), with ``e_d(k) = -1 + 1.5 cos(2 pi k)``, ``e_p = -3`` eV and
    ``t = 0.4`` eV. The upper eigenstate is d-dominant at every k, the lower p-dominant, and
    their d projectabilities sum to one because the basis is orthonormal.
    """

    def __init__(self, t=0.4, e_p=-3.0):
        self.t, self.e_p = float(t), float(e_p)
        self.orbital_layout = ({"1d": (0, 1), "1p": (1, 2)},)
        self.kpoints_frac = K_FRAC.copy()
        self.kpoints_cart = K_FRAC.copy()
        self.lattice_ang = LATTICE
        self.alat_ang = ALAT
        self.fermi_energy_ev = FERMI_EV
        self.magnetic_site = 0
        self.species = ("Fe",)
        self.positions_frac = np.zeros((1, 3))
        self.eigenvalues_ev = np.array([[self._eig(k)[0] for k in K_FRAC]])

    def _h(self, kfrac):
        e_d = -1.0 + 1.5 * float(_cos(np.asarray(kfrac, dtype=float).ravel()[0]))
        return np.array([[e_d, self.t], [self.t, self.e_p]])

    def _eig(self, kfrac):
        return np.linalg.eigh(self._h(kfrac))

    def projectability_at(self, kfrac, orbital_indices, spin=0):
        e, v = self._eig(kfrac)
        idx = sorted(set(orbital_indices))
        return e, (np.abs(v[idx, :]) ** 2).sum(axis=0)

    def orbital_weights_at(self, kfrac, spin=0):
        e, v = self._eig(kfrac)
        return e, (np.abs(v) ** 2).T

    def d_projectability(self):
        """(N_K, 2) d projectability of the lower and upper state at each mesh k."""
        return np.array([self.projectability_at(k, [0])[1] for k in K_FRAC])

    def is_metallic(self):
        e = self.eigenvalues_ev
        return bool(np.any((e.min(axis=1) < FERMI_EV) & (e.max(axis=1) > FERMI_EV)))
