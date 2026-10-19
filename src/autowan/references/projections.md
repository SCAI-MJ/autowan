# Initial projections and the Wannier count

The initial projections g_n give the starting gauge (and, with disentanglement, the
starting subspace) through A_mn(k) = <psi_mk|g_n>.

- In the `.win`, a projections block names sites and angular parts, e.g. `X:s;p` or
  `X:l=2`; each l contributes 2l+1 functions per atom of that species, hybrids such as
  `sp3` contribute 4. The total must equal num_wann. A radial quantum number `r=2`
  selects a second, more extended hydrogenic radial function.
- A projection may sit on a point that is not an atom: `f=x,y,z:s` (fractional
  coordinates of the cell) or `c=x,y,z:s` (Cartesian, Angstrom unless the block says
  bohr). Such a line seeds that one point; bond-centred s-like functions for covalent
  bonds are the standard example. Single real components (`pz`, `px`, `py`, `dz2`,
  `dxz`, `dyz`, `dx2-y2`, `dxy`) give one function each.
- Hydrogenic (analytic) projections are the same functions for every pseudopotential.
  Pseudo-atomic orbitals (PAOs) taken from the pseudopotential follow the atom's own
  radial shape; with orthonormalised PAOs the projectability of every state is at most
  one, which projectability disentanglement relies on.
- Projections that mix very compact and very diffuse functions tend to give a spread
  functional with competing minima: some functions localise by delocalising others.
- Dropping a shell removes num_wann; the states it described then have to be absorbed
  by the remaining functions, which often moves centres off atoms.

Sources: Wannier90 User Guide (v3.1), chapter "Projections"; Marzari et al., Rev. Mod.
Phys. 84, 1419 (2012), section on initial projections.
