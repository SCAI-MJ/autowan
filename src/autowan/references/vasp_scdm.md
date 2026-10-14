# SCDM in VASP (the VASP frontier protocol)

VASP 6 computes SCDM projections itself in its Wannier90 interface.

- Tags: `LSCDM = .TRUE.`; the weight is `CUTOFF_TYPE` (erfc by default; also gaussian,
  fermi), `CUTOFF_MU` (an absolute energy in eV) and `CUTOFF_SIGMA` (eV).
- Defaults in VASP 6.4.1 (from the source, `mlwf.F`): sigma = 1 eV; mu = the Fermi level
  of a smeared filling of 0.8 num_wann bands; with no more bands than functions the
  weight is switched off (isolated case).
- `NUM_WANN` unset: VASP uses min(NBANDS, the count of the first bound partial wave of
  each occupied l in the POTCARs).
- The automated GW workflow of Varrassi et al. (2025) runs the Wannier step on the
  ground-state WAVECAR with `ALGO = None`, `NELM = 1`, `LWANNIER90_RUN`, `LSCDM`,
  `ISMEAR = 0`, `SIGMA = 0.02`, `LORBIT = 11`, no windows and no projections, and a
  .win with dis_num_iter 500, dis_conv_tol 1e-8, num_iter 1000, conv_tol 1e-4. With
  more bands than functions Wannier90 then disentangles from the SCDM projections over
  all bands without a frozen window.
- In autowan, mu and sigma are offered as `scdm_mu` (relative to E_F) and `scdm_sigma`;
  they are written as the overlap step's CUTOFF_MU / CUTOFF_SIGMA, so a change re-runs
  the overlap step.

Sources: Varrassi et al., npj Comput. Mater. (2025), arXiv:2410.15948, and the Wannier
step of the authors' public workchain (as summarised in autowan's protocol report,
2026-10-07); VASP 6.4.1 source (`mlwf.F`); VASP wiki, LSCDM / CUTOFF_* tags.
