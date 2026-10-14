# Excluding bands (semicore states)

`exclude_bands` removes listed bands before anything else: the overlap step does not
write them and Wannier90 never sees them. `num_bands` then counts only the bands that
remain, so num_bands + (number excluded) must equal the bands the parent stored.

- Deep semicore states (for example a cation's filled s and p shells, or an anion 2s
  far below the valence band) are usually excluded: they are flat, well separated, and
  carrying them only adds functions and frozen states. The established automatic
  workflows exclude semicore by default.
- Which bands are semicore is decided by their character (which atomic channel they
  come from), not by energy order alone: an anion s band can sit below a cation semicore
  p band.
- Excluding a block that is not separated by a gap from the next band cuts a connected
  manifold and leaves the model unable to reproduce it.
- A top block can be excluded too, which is how a deck takes fewer bands than the
  parent computed.
- The bottom cut is moved with `semicore_exclusion`, by shell; the page `semicore`
  says what follows from it and which cuts the gate honours.

Sources: Wannier90 User Guide (v3.1), `exclude_bands`; Qiao, Pizzi & Marzari, npj
Comput. Mater. 9, 208 (2023), on semicore exclusion in automated workflows.
