# Projectability-disentangled Wannier functions (PDWF)

PDWF chooses the subspace by how well each Bloch state is described by the
pseudopotential's own atomic orbitals (PAOs). For state (n,k) the projectability is
p_nk = sum_I |<phi_I|psi_nk>|^2 over the orthonormalised PAOs, between 0 and 1.

- `dis_proj_max`: states with p >= this value are frozen (reproduced exactly);
  `dis_proj_min`: states with p < this value are dropped from the pool. Both act only
  with `dis_froz_proj = .true.`; without it Wannier90 reads them and silently ignores
  them. The automated protocol starts at dis_proj_min = 0.01, dis_proj_max = 0.95.
- The protocol also keeps an energy frozen window: everything up to E_F + 2 eV for a
  metal, up to the conduction-band minimum + 2 eV for an insulator. A state is frozen
  if either criterion freezes it, and the frozen count must not exceed num_wann. The
  window's top (`dis_froz_max`) is a PDWF knob like the thresholds: the published
  method is projectability plus energy disentanglement (PD+ED).
- When the first attempt fails its criterion, the published workflow retries over a
  fixed ladder of threshold pairs and stops at the first construction that passes. The
  2025 archive ladder (24 points) steps dis_proj_min down in an outer loop -- 0.02,
  0.01, 0.005, 0.0025, 0.00125, 0 -- and, for each, dis_proj_max down in an inner loop
  -- 0.99, 0.95, 0.90, 0.85. (The 3.1.0 release shipped a shorter grid: dis_proj_max down
  to 0.85 at dis_proj_min 0.02 and 0.01.) The idea: start strict and relax one knob at a
  time. Lowering dis_proj_max freezes more of the well-described states, pinning the
  model to the DFT bands there; lowering dis_proj_min admits more weakly-projecting
  states into the pool the disentanglement may mix in, which is what a pool too small
  for num_wann needs (Wannier90: "Energy window contains fewer states than number of
  target WFs"); dis_proj_min = 0 drops nothing. Raising dis_proj_min does the opposite
  and shrinks the pool.
- The Wannier count is the number of PAOs kept. In Quantum ESPRESSO, pw2wannier90 uses
  `atom_proj = .true.` with an external projector list (`atom_proj_ext`,
  `atom_proj_dir`); `atom_proj_exclude` removes listed PAOs (semicore ones by default).
  Changing which PAOs are kept changes what the functions can represent and num_wann.
- Hydrogenic or other non-orthonormal projectors can give p > 1; the thresholds then
  stop meaning anything, so PDWF needs PAOs.

Sources: Qiao, Pizzi & Marzari, npj Comput. Mater. 9, 208 (2023), arXiv:2303.07877;
Qiao et al. (2025), arXiv:2507.06840 and its data archive (the retry ladder);
aiida-wannier90-workflows (projectability protocol and its retry loop); Wannier90 User
Guide (v3.1), `dis_froz_proj`.
