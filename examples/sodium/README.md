# bcc sodium: the frontier refused, two repairs, certified

A real autowan run from the KPS 2026 poster campaign: QE 7.5, SSSP pseudopotential,
gate profile **`band_interpolation`** (C1), with Claude Opus 5.5 as the repair model.
Total cost: **USD 0.86** at API prices (two model calls).

Re-score it yourself (about 10 s, no DFT code, no key):

```bash
uv run python scripts/rescore.py examples/sodium
```

## What happened

| attempt | deck | what changed | η₂ (limit 20 meV) | gate |
|---|---|---|---|---|
| 0 | `wan` | nothing: the frontier's own PDWF deck, after its 10-point projectability ladder | 41.4 meV | REFUSED |
| 1 | `wan_r1` | `dis_froz_max` 2.08 → 4.5 eV above E_F | 38.7 meV | REFUSED |
| 2 | `wan_r2` | switched PDWF → SMV, projectors `Na:s;p;d` (4 → 9 functions) | **3.0 meV** | **CERTIFIED** |

Every structural check passed from the start. The only failure was band interpolation:
the four s+p functions reproduce DFT on the construction mesh, but not between the mesh
points.

### Attempt 1: the model's first diagnosis (USD 0.35)

Before proposing, the model called `band_error_by_energy`, `frozen_capacity`,
`band_character` and `validate_deck`. It concluded (abridged):

> The error builds up with energy (19.8 meV at ν = 0, 26.6 at ν = 1, 41.4 at ν = 2) [...]
> The frozen window stops at E_F + 2.075 eV. That edge cuts through the dispersive band
> 6 [...] The frontier's ladder only moved the projectability thresholds [...] That class
> of change is exhausted. The energy edge of the frozen window has not been touched [...]
> Raising dis_froz_max to E_F + 4.5 eV pins the s and p bands across the whole weighted
> region.

A reasonable idea, but the gate measured only a 2.7 meV improvement.

### Attempt 2: the method switch (USD 0.52)

This time the model called `truncation_diagnostics`, `search_references` (the corpus
had nothing for this case), `method_defaults` and `validate_deck`. It concluded
(abridged):

> The three p-like functions have spreads of 7.0 Å², and they carry all of the hopping
> weight beyond L_BvK/4 [...] So the windows and thresholds are spent. What limits the
> result is the basis: the Na 3p functions are diffuse and are being forced to also
> describe the d-hybridised states above E_F. The pseudopotential has no d orbitals, so
> this cannot be fixed inside PDWF.

It switched to SMV with analytic s, p and d projectors. It first tried a PDWF `3d`
subset, which `validate_deck` refused because the pseudopotential has no d orbital.
The gate certified the result at η₂ = 3.0 meV.

The complete record, with every tool call and its result, every request, the tokens and
the costs, is in [`run/autowan_log.json`](run/autowan_log.json).

## What is in this directory

| path | what |
|---|---|
| `example.json` | what `scripts/rescore.py` reads: paths, profile, seeds |
| `run/wan.*` | attempt 0: the frontier deck (`.win`, `.pw2wan`) and Wannier90's outputs |
| `run/wan_r1.*`, `run/wan_r2.*` | the two repairs, same files |
| `run/external_projectors/` | the frontier's external projectors (the `atom_proj_dir` of `wan.pw2wan`) |
| `run/frontier_ladder_points.csv` | the projectability thresholds the frontier tried before handing over |
| `run/autowan_log.json` | the full run record |
| `parent/data-file-schema.xml` | the nscf parent (eigenvalues, structure; no wavefunctions) |
| `reference/dft_bands_path.npz`, `reference/scf.out` | DFT bands on the separate path, and the scf output the Fermi level is read from |
| `dft_inputs/scf.in`, `dft_inputs/nscf.in` | to redo the DFT |

## Redoing the whole run

The parent wavefunctions are not shipped. To repeat the run from scratch:

1. Put the SSSP pseudopotential `na_pbe_v1.5.uspp.F.UPF` in `./pseudo/` (from the
   [SSSP library](https://www.materialscloud.org/discover/sssp)), then run `pw.x` on
   `dft_inputs/scf.in` and `dft_inputs/nscf.in`.
2. Copy `run/wan.win`, `run/wan.pw2wan` and `run/external_projectors/` next to the
   output (the decks expect `prefix = 'aiida'`, `outdir = './out/'`), then:

   ```bash
   uv run autowan --run-dir . --seed wan \
       --parent-xml out/aiida.save/data-file-schema.xml \
       --gate-profile band_interpolation \
       --reference-path /path/to/autowan/examples/sodium/reference/dft_bands_path.npz \
       --log run.json
   ```

A repeat will not reproduce the same model text, because the model is not
deterministic. It is the gate's numbers on a given deck that reproduce.
