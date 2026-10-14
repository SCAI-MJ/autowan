# Spread minimisation (the gauge step)

After the subspace is fixed, Wannier90 rotates it at every k-point to minimise the total
spread Omega = Omega_I + Omega_D + Omega_OD. Only Omega_D + Omega_OD depend on this
gauge; Omega_I is set by the subspace (disentanglement minimises it).

- `num_iter` (default 100): the iteration budget. A descent still decreasing
  monotonically when the budget ends needs more iterations.
- `conv_tol` (default 1e-10 Ang^2) and `conv_window` (default -1, meaning no test):
  the run counts as converged only when the spread change stays below the tolerance
  for conv_window consecutive iterations. With conv_window unset (or 1) there is no test
  and Wannier90 runs all num_iter iterations. The default gate does not require
  convergence: it judges the construction (bands, hoppings, centres). A gate profile
  with the `convergence` checks does, and then applies the same test to the printed
  trajectory over a reference window of 3 ending at the last iteration when the deck
  runs none.
- `trial_step` (default 2.0) sets the line-search step; `num_cg_steps` (default 5) how
  many conjugate-gradient steps run before a reset. A step that keeps changing sign or a
  spread that rises after a minimum is oscillating: a smaller trial step damps it, and
  more iterations will not.
- A stall far above the tolerance with very unequal final spreads usually reflects the
  starting projections (a landscape with competing minima), not the optimiser.

Sources: Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997); Marzari, Mostofi, Yates,
Souza & Vanderbilt, Rev. Mod. Phys. 84, 1419 (2012); Wannier90 User Guide (v3.1),
"Wannierise" parameters.
