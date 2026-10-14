# Semicore shells: including or excluding the bottom bands

A pseudopotential with semicore states puts filled shells (a cation's outer s and p
shells below the valence, a filled d shell) into the computed spectrum, as the lowest
bands. A construction either describes them with functions of their own or leaves them
out with `exclude_bands 1..N`; the knob `semicore_exclusion` chooses, per species and
per shell.

- **The cut is by index.** Wannier90 removes bands 1..N at every k-point, whatever their
  character. It is a clean cut only where band N lies below band N+1 everywhere in the
  zone; where they cross, the excluded block holds a different mixture of states at
  each k-point and the kept bands lose part of a connected manifold.
- **The gate's rule.** An exclusion holding occupied states is set aside only if the
  highest excluded eigenvalue lies more than 1 % of the occupied bandwidth below the
  lowest kept one, over the whole mesh. Otherwise every band is the target, and each
  excluded occupied state has no model band near it. `validate_deck` refuses such a
  deck before anything runs, and lists every cut the shells imply with its gap.
- **Character, not energy order.** Which shells sit lowest is decided by the bands'
  character. An anion s band can lie below, or inside, a cation's semicore p block; the
  pseudo-atom's eigenvalues are a guide, not a rule.
- **What follows from the shell list.** N is the bands the excluded shells hold: 2l+1
  per atom in each channel, or one band per electron for spinor bands (2j+1 for a
  j-resolved orbital). `exclude_bands` becomes 1..N plus any top (headroom) block,
  `num_bands` what is left. PDWF (and SMV on pseudo-atomic orbitals) keeps or drops the
  shells' own orbitals, so `num_wann` moves with them; analytic SMV adds one trial
  function per shell brought in (a second one of an angular momentum the species
  already carries uses radial function `r=2`); SCDM's `num_wann` moves by the bands
  brought in.
- **When to include.** A shell that overlaps a kept band has to be included (or the
  whole overlapping block excluded together, if that block is isolated from the
  valence). Including costs functions and puts its states inside the frozen region;
  their spreads are small and their bands flat, so they rarely limit a construction.
- **When to exclude.** A whole block well below the valence, separated by a gap, adds
  only functions and frozen states; the established workflows exclude it by default.

Sources: Wannier90 User Guide (v3.1), `exclude_bands`, `projections` (radial `r`);
Qiao, Pizzi & Marzari, npj Comput. Mater. 9, 208 (2023), semicore exclusion in the
PDWF protocol; Garrity & Choudhary, Sci. Data 8, 106 (2021), semicore counts per
POTCAR in an automated VASP workflow.
