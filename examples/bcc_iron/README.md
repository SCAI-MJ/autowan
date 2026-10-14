# bcc iron — three automatic recipes, three different endings

> Recorded in August 2026 with an earlier autowan (energy-window recipes, the default
> gate). It is kept to read, not to re-score: the parent calculation is not shipped,
> and only one of the `.wout` files is. For a run you can re-score, see
> [`../sodium`](../sodium).

One material, one gate, one policy text, one action vocabulary. Three deterministic
recipes are run as the automatic stage, and each stops somewhere different.

Iron is classified a metal from its own bands — six of them cross the Fermi level — so
the gate withholds its band-distance check rather than widening it, and judges the
construction on the checks that carry no transplanted constant.

| automatic recipe | gate | generic repair, from general principles alone | ending |
|---|---|---|---|
| SCDM | refused | `conv_window = 5` | **certified** on the first repair |
| PDWF | refused | `conv_window = 5`, then `num_iter` 1000 -> 3000 | **still refused**; the model escalated and asked to search references |
| the Wannier90 manual's own iron recipe | refused | `num_iter` 400 -> 800 | **certified** on the first repair |

| construction | total spread (A^2) | widest/narrowest | disentangle | localise | converged |
|---|---|---|---|---|---|
| SCDM, automatic | 6.287 | 2.18 | 952 | 1000 | no |
| SCDM, repaired | 6.287 | 2.18 | 952 | 892 | **yes** |
| PDWF, automatic | 20.755 | 15.76 | 313 | 1000 | no |
| PDWF, two repairs | 19.323 | 15.62 | 313 | 3000 | no |
| expert recipe, automatic | 7.833 | 2.52 | 43 | 400 | no |
| expert recipe, repaired | 7.833 | 2.52 | 43 | 338 | **yes** |

Two things in that table are worth reading slowly.

**The smallest spread was refused.** SCDM's construction is the most localized of the
three and the gate turned it down, because localisation is not what the gate measures —
total spread is not among its eight checks. What was wrong was that nobody had checked:
Wannier90 runs no convergence test unless asked, so the run had neither converged nor
failed to. One parameter turned the test on and the same construction certified,
unchanged in every number. The gate did not improve it; it established what it was.

**Repair is not always possible, and the loop says so.** PDWF's minimisation genuinely
does not settle. Told to converge it did not, given three times the budget it did not,
and the model then declined to keep guessing and asked for reference search — the stage
that opens only after general principles have been spent. Its spreads separate into a
tight group of five and a diffuse group of four, a ratio of 15.8 against 2.2 for SCDM,
and the interpolated bands in `figures/fe_up_pdwf_r2_bands.png` are visibly rougher than
`figures/fe_up_scdm_r1_bands.png`. That is a projector problem, and more iterations of
anything will not fix it.

`autowan_scdm.json` and `autowan_pdwf.json` are the run records: every diagnosis, the
parameters each repair changed, and what it cost. The parent calculation's wavefunctions
are not here — 413 MB, and they belong to whoever reruns the density-functional step.
