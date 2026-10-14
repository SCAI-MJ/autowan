# Selected columns of the density matrix (SCDM)

SCDM builds localised starting functions without trial orbitals: it takes columns of a
(quasi-)density matrix built from the Bloch states, chosen by a QR factorisation with
column pivoting at the Gamma point, and orthonormalises them.

- For an isolated manifold the occupied states enter with weight 1. For entangled bands
  each state enters with a weight f(e): `erfc` gives f(e) = erfc((e - mu)/sigma)/2, so
  states well below mu count fully and states well above it do not; sigma sets how
  softly. Gaussian weights are also defined.
- Quantum ESPRESSO (pw2wannier90): `scdm_proj = .true.`, `scdm_entanglement` =
  isolated | erfc | gaussian, `scdm_mu`, `scdm_sigma` (eV); the .win uses
  `auto_projections = .true.` and no projections block.
- Vitale et al. 2020 (automated protocol): num_wann = the number of PAOs; mu and sigma
  from an erfc fit of the projectability of the states on the PAOs, with
  mu = mu_fit - 3 sigma_fit and sigma = sigma_fit; semicore excluded; the SCDM subspace
  is used without further energy-window disentanglement, followed by maximal
  localisation.
- Raising mu brings more higher-lying states into the columns; lowering it restricts
  the subspace toward the occupied states. A larger sigma blends the two smoothly.
- SCDM functions are not tied to atoms: bond-centred functions are common and expected.

Sources: Damle, Lin & Ying, J. Chem. Theory Comput. 11, 1463 (2015); Damle & Lin,
Multiscale Model. Simul. 16, 1392 (2018); Vitale, Pizzi, Marrazzo, Yates, Marzari &
Mostofi, npj Comput. Mater. 6, 66 (2020), arXiv:1909.00433.
