# Wannier90 decks and outputs for the projectability-window rules

Real engine outputs, never hand-written. Each `.wout` is paired with the deck that
produced it, so a test can ask the question the rules are about: did the deck request
a projectability window, and does the engine's own output show that one was applied?

| fixture | deck asks for | `.wout` shows | source |
|---|---|---|---|
| `graphene_projectability_fixed.{win,wout}` | `dis_froz_proj = .true.`, `dis_proj_min = 0.01`, `dis_proj_max = 0.85` | the `Projectability  Windows` block (discarded below 0.01, frozen from 0.85) | Wannier90 test suite `test-suite/tests/testw90_graphene_pdwf/` (`graphene.win`, `benchmark.out.default.inp=graphene.win`) |
| `graphene_projectability_automatic.{win,wout}` | `dis_froz_proj = .true.`, `dis_proj_auto = .true.` | the block, with thresholds chosen by multi-Otsu (5 classes: 0.03906 / 0.94531) | Wannier90 test suite `test-suite/tests/testw90_graphene_dis_proj_auto/` |
| `si_campaign_projectability_ignored.{win,wout}` | `dis_proj_min = 0.01`, `dis_proj_max = 0.95`, **no** `dis_froz_proj` | energy windows only (`Inner: -5.88695 to 8.03810 eV`), no projectability block | the 2026-08-28 QE campaign, silicon (frozen 2026-08-29T15:35:20Z) |

The Wannier90 test-suite files are copied unaltered from the source checkout the
campaign's `wannier90.x` was built from (the Wannier90 development branch, commit
`cfa8441`, upstream `wannier-developers/wannier90`). They are Wannier90's, distributed
under its GNU GPL v2 licence, not under this package's MIT licence (see `NOTICE`). The block text the reader keys on
is printed by `dis_windows_proj` in that checkout's `src/disentangle.F90`, and only when
`dis_froz_proj` is true; `readwrite.F90` defaults the flag to `.false.`.

The silicon deck is copied unaltered. Its `.wout` is a trimmed excerpt of the 4,711-line
original: lines 1-262 (header, parameters, k-mesh, the disentanglement windows, the
first iterations), 396-427 (the end of the disentanglement and the start of the
localisation) and 4660-4711 (the last localisation step, `Final State`, the spread
summary). Lines were only removed, never edited, except that the repository's
`trailing-whitespace` pre-commit hook strips trailing blanks (Wannier90 pads some
headings with them; no value the reader takes is on such a line). The energy-window routine's
`In dis_proj_froz... done` line is kept on purpose: it names a subroutine, not the
projectability window, and a reader must not mistake one for the other.

sha256 of the sources:

```
f82419dcb0cd341f3ea3893be7315f3837cdf6c445121bab7ef459f7e24f769d  testw90_graphene_pdwf/graphene.win
af44160cd94a9b68c6cc9327db974e76f6ac92a0c69368efc14738274214c485  testw90_graphene_pdwf/benchmark.out.default.inp=graphene.win
6657715adc84c47aa4809d7531efb67939b3e8bd071ecf3225b7bc5878600b5b  testw90_graphene_dis_proj_auto/graphene.win
c189bc0ecc9c8e855cc9a938d92c9be1fd720fd337ba7de86cf043fbd36a579a  testw90_graphene_dis_proj_auto/benchmark.out.default.inp=graphene.win
522cff59b43c356191ebff119751b97ca567193ace7cb893aed19675ce7ec57a  campaign/Si/Si_pdwf.win
026cdc9d4292f6b48c97e67ca867690ab66f4a458458105ae0d57a8995d7c1c7  campaign/Si/Si_pdwf.wout (before trimming)
```
