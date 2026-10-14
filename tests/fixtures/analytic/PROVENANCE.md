# Provenance of the analytic fixtures

`analytic_constructions.py` builds every parent and model the gate-extension tests use
(`test_magnetic_subspace_target.py`, `test_semicore_independent_scale.py`,
`test_metal_frozen_window_contains_fermi_level.py`, `test_openmx_projectability_on_kept_states.py`).
Nothing is fitted and nothing is read from a calculation: each band is a closed-form expression,
so a test's expected value can be derived by hand. Written 2026-09-15 (cluster2) for the
magnetic-subspace certification target work.

## Mesh and cell

- Cubic cell, lattice vectors `4 Å · I`, `alat = 4 Å`, so Cartesian k in `2π/alat` equals fractional k.
- Eight k-points `k_j = (j/8, 0, 0)`, `j = 0..7`. `E_F = 0 eV`.

## Bands (eV)

| name | formula | range on the mesh | role |
|---|---|---|---|
| semicore | flat `-50` | -50 | a separated semicore level (44 eV gap to the valence) |
| `s_band` | `-6 + 2 (1 - cos 2πk)` | -6 .. -2 | valence s-like band, magnetic-d weight 0.1 |
| `d_band` | `-3 + 1.75 (1 - cos 2πk)` | -3 .. +0.5 | magnetic d-like band, weight 0.9; crosses E_F (metal) |

`d_band - s_band >= 2.5 eV` at every k, so the two never cross while their energy ranges overlap
(-3 .. -2 eV): the s + d valence block has no gap. The top occupied state is the d band at
k = 3/8, -0.01256 eV, so the valence spread is 5.98744 eV; every d state together spans 3.5 eV.

Further parents built inline in the tests from the same helpers: an Fe_PBE19S-like parent
(flat -90 eV, a 3p-like band `-52.75 + 0.25 cos 2πk`, then s + d: gaps of 37 and 46.5 eV), an
Sr_sv-like parent (flat -38 eV, a 4p-like band `-16.5 - 0.5 cos 2πk`, then s + d: gaps of 21
and 10 eV), a two-spin parent whose spin-up gap (-9 .. -3 eV) is filled by a spin-down band
`-9.5 + 3.5 (1 - cos 2πk)`, and a single coarsely sampled band `-10.5 + 5 (1 - cos 2πk)`.

## Models (Wannier90-style H(R), R = -1, 0, 1 unless stated)

| name | H(R) | band |
|---|---|---|
| `D_ONLY_MODEL` | on-site -1.25, hop -0.875 | equals `d_band` exactly |
| `D_ONLY_MODEL_SHIFTED` | on-site -0.95, hop -0.875 | `d_band` + 0.3 eV: misses every d state |
| `D_ONLY_MODEL_FLAT` | on-site -1.25, hop 0 | meets `d_band` only where cos 2πk = 0 (2 of 8 k) |
| `hr_from_band(v)` | discrete Fourier transform over R = -3 .. 4 | reproduces the eight values `v` exactly on the mesh |
| `hr_diagonal(...)` | uncoupled bands | one (on-site, hop) pair per band |

## Two-orbital OpenMX-like parent (`TwoOrbitalParent`)

`H(k) = [[e_d(k), t], [t, e_p]]` in an orthonormal basis (d = orbital 0, ligand p = orbital 1, one
atom), `e_d(k) = -1 + 1.5 cos 2πk`, `e_p = -3`, `t = 0.4`. The upper state is d-dominant at every
k (d projectability 0.62 .. 0.99) and the lower one p-dominant; the two d projectabilities sum to
one. It exposes the OpenMX parent surface the backend checks read: `orbital_layout`,
`kpoints_frac`, `eigenvalues_ev`, `projectability_at(k, orbitals, spin)`, `orbital_weights_at`.
