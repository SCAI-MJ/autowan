# autowan

**Wannier functions that are checked by physics, with an AI that repairs the ones that fail.**

You give autowan a DFT calculation and the Wannier90 input that the standard automatic
method writes for it (PDWF for Quantum ESPRESSO, SCDM for VASP). We call that automatic
method **the frontier**. On VASP the SCDM deck is the one JARVIS-tools writes: the
poster's "JARVIS deck" and "SCDM" are the same thing. autowan runs the frontier's input
and scores the result with a **gate**: a fixed set of physical checks against your own
DFT bands. If the gate passes, you are done, and no AI was involved. If it fails, a
language model diagnoses why and proposes one change. The change is checked, run, and
scored by the same gate. After the frontier's attempt, the model gets up to seven
repairs (`--total-attempts 8`). The run ends in one of two ways:
- **certified**: a construction that passed every check. You get the seedname and the
  numbers.
- **stop & report**: the model's evidence that nothing on the Wannier side can fix this
  parent calculation, with the change to the DFT it asks for (more bands, a denser
  mesh, ...).

The model never decides whether a construction is good. The gate does, and the model
cannot change the gate.

```
  your DFT + frontier deck
            │
            ▼
   ┌─────────────────┐   pass   ┌────────────┐
   │  run & score    ├─────────►│ CERTIFIED  │
   │  (the gate)     │          └────────────┘
   └────────┬────────┘
            │ fail
            ▼
   ┌─────────────────┐  "the parent is the problem"  ┌────────────────┐
   │ model diagnoses │──────────────────────────────►│ STOP & REPORT  │
   │ + proposes one  │                               └────────────────┘
   │ validated change│
   └────────┬────────┘
            └──────► back to "run & score" (8 attempts in all)
```

**Words used below.**

| word | meaning |
|---|---|
| frontier | the published automatic Wannierisation method, run with no AI: PDWF on QE ([Qiao, Pizzi & Marzari 2023](https://doi.org/10.1038/s41524-023-01146-w)), SCDM on VASP (the deck JARVIS-tools writes: Garrity & Choudhary, Sci. Data **8**, 106 (2021); the poster calls it the "JARVIS deck") |
| deck | the Wannier90 input files of one construction: `<seed>.win` (and, for QE, `<seed>.pw2wan`) |
| parent | the DFT calculation the Wannier functions are built from (your `pw.x` nscf run) |
| gate | the checks that decide acceptance; a *gate profile* chooses which checks, for which use |
| η₂ | band distance between Wannier and DFT bands on a k-path the construction never saw, Fermi-Dirac weighted up to E_F + 2 eV ([Vitale et al. 2020](https://doi.org/10.1038/s41524-020-0312-y)); 20 meV is the limit used here |
| PDWF / SMV / SCDM | three ways to build the subspace: projectability disentanglement on the pseudopotential's orbitals / energy windows with trial projections (the classic recipe) / density-matrix columns |
| L_BvK | the size of the supercell the k-mesh represents (`mp_grid × lattice`); a model's hoppings cannot be trusted much beyond it |

> Came from the KPS 2026 poster? Start with the two-minute demo below. It needs no DFT
> code and no API key. The poster's words, as they appear here:
>
> - **C1 / C2 / C3** are the gate profiles `band_interpolation`, `sparse_tb` and
>   `dmft_subspace` ([table](#choosing-what-certified-means-gate-profiles)).
> - **frontier, n retries** (blue cell): before autowan takes over, the QE frontier gets a
>   budget of ten Wannier90 runs: its default deck, then its own retry ladder, which steps
>   the projectability thresholds `dis_proj_min`/`dis_proj_max`. *n* is the run that
>   passed. A red *n* passed only beyond that budget and is counted as a failure.
> - **autowan, k tries** (yellow cell): *k* model repairs, saved as `<seed>_r1` ... `<seed>_rk`.
> - **done by the frontier** (light-blue cell): the frontier passed on a retry, so autowan
>   was not needed.
> - **asks for new DFT** (flag): the run ended with a [stop & report](#when-it-stops-with-a-report).
> - **JARVIS deck**: the VASP frontier, an SCDM deck written by JARVIS-tools.
> - **C3 frontier**: an adaptation of the QE frontier deck to each material's fixed
>   target, not a published method ([C3 in detail](#c3-in-detail)).

---

## Try it in two minutes (no DFT, no API key)

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)
(or plain `pip`).

```bash
git clone https://github.com/SCAI-MJ/autowan
cd autowan
uv sync                                    # or: pip install -e .
uv run python scripts/rescore.py examples/sodium
```

(With pip, drop the `uv run` prefix everywhere: `python scripts/rescore.py examples/sodium`.)

The demo re-scores, on your machine, three Wannier constructions of bcc sodium from a
real autowan run. Sodium is a simple metal, and simple metals are hard for the automatic
method. Here is what it prints (shortened):

```
gate profile: band_interpolation
parent: E_F = 1.555 eV, 1 bands cross it -> metal

wan  (the frontier: PDWF): REFUSED
  pass  wannier_count_matches_request                  4 functions
  pass  centres_stay_near_home_cell                    0 Ang  (limit 3.659)
  ...
  FAIL  band_interpolation_eta2                        41.41 meV  (limit 20)

wan_r1  (repair 1: frozen window raised): REFUSED
  ...
  FAIL  band_interpolation_eta2                        38.69 meV  (limit 20)

wan_r2  (repair 2: switched to SMV, s+p+d projectors): CERTIFIED
  ...
  pass  band_interpolation_eta2                        2.967 meV  (limit 20)
```

How to read it:

1. **`wan` is the automatic method's own result.** Every structural check passes, but
   the interpolated bands miss the DFT bands by η₂ = 41 meV, and the limit is 20 meV.
   The gate refuses it.
2. **`wan_r1` is the model's first repair.** It raised the top of the frozen window
   from E_F + 2.1 to E_F + 4.5 eV. That is a reasonable idea, and it did not work
   (38.7 meV). The gate refuses it again.
3. **`wan_r2` is the model's second repair.** It concluded that four s+p functions
   cannot represent the d-hybridised states above E_F. It switched method and added d
   projectors (nine functions). Now η₂ is 3.0 meV, and the gate certifies it.

The whole run took two model calls and cost **USD 0.86** at API prices.
[`examples/sodium/README.md`](examples/sodium/README.md) shows the model's own
diagnoses, the tools it called, and the decks before and after each change.

Each line is one check. A `pass`/`FAIL` line gives the measured value and its limit;
`n/a` means the check does not apply to this material (band distance is not used for
metals) and does not count. The profiles are summarised in
[Gate profiles](#choosing-what-certified-means-gate-profiles);
`uv run autowan --list-gate-profiles` prints their full definitions.

---

## Run it on your own material

### What you need

| | needed for | notes |
|---|---|---|
| **Wannier90** with `dis_froz_proj` | every run | Wannier90 3.1.0 does **not** have it: build the [development branch](https://github.com/wannier-developers/wannier90) (developed against commit `cfa8441`), or a later release that includes it |
| **Quantum ESPRESSO** (`pw.x`, `pw2wannier90.x`) | the QE backend | developed against QE 7.5 |
| *or* **VASP** built with `LWANNIER90` | the VASP backend | POTCARs are never shipped; set `AUTOWAN_POTCAR_LIBRARY` |
| an **API key** (OpenRouter), *or* a local OpenAI-compatible server | the repair step only | scoring alone needs neither |

Point autowan at your executables if they are not on `PATH`:

```bash
export WANNIER90=/path/to/wannier90.x
export PW2WANNIER90=/path/to/pw2wannier90.x
```

The steps below are for Quantum ESPRESSO. The VASP backend (`--backend vasp`) follows the
same steps with VASP's files; `autowan --help` lists what it reads.

### Step 1: the parent calculation

In one directory (call it `calc/`), run `pw.x` scf and then nscf as usual for Wannier90:

- the nscf on the **full uniform k-mesh** that the deck's `mp_grid` names (`nosym` and
  `noinv`, every k-point listed);
- with **enough bands** (`nbnd`) to cover the energy window you want reproduced; the
  gate tells you if it is not enough.

`examples/sodium/dft_inputs/` has a working scf/nscf pair.

### Step 2: the frontier deck

Next to it, put the deck the frontier writes: `<seed>.win` and `<seed>.pw2wan`.
[aiida-wannier90-workflows](https://github.com/aiidateam/aiida-wannier90-workflows)
writes this deck for you. You can also write it by hand, starting from
[`examples/sodium/run/wan.win`](examples/sodium/run/wan.win) and
[`wan.pw2wan`](examples/sodium/run/wan.pw2wan). Check four things:

- `.win`: `auto_projections = .true.`, `dis_proj_min`/`dis_proj_max`, and
  **`dis_froz_proj = .true.`**. autowan refuses a deck with the thresholds but without
  that flag, because Wannier90 would silently ignore them.
- `.win`: `write_hr = .true.` and `write_xyz = .true.`. The gate reads the Hamiltonian
  and the centres.
- `.pw2wan`: `prefix` and `outdir` are those of your `pw.x` run, and `atom_proj = .true.`.
- `.pw2wan`, optional: `atom_proj_ext = .true.` with `atom_proj_dir` points to external
  projector files (the 2025 protocol, one `<species>.dat` per species, as in
  `examples/sodium/run/external_projectors/`). Without them, pw2wannier90 uses the
  pseudopotential's own orbitals.

For a collinear magnet, write one deck per spin channel and pass `--spin-index 0` (up)
or `--spin-index 1` (down).

### Step 3: the reference bands

`band_interpolation`, `sparse_tb` and `dmft_subspace` compare the Wannier model with DFT
bands on a k-path that is not the construction mesh. (`default` needs no reference;
skip this step to try it first.) autowan interpolates the model at whatever k-points the
reference holds, so any high-symmetry path will do (e.g. from
[SeeK-path](https://www.materialscloud.org/work/tools/seekpath), in crystal
coordinates).

1. Copy `calc/out/` to a separate directory, e.g. `bands/out/`. A `bands` run
   overwrites the save directory, and the nscf one must survive.
2. In `bands/`, run `pw.x` with `calculation = 'bands'`, the same `nbnd`, and
   `K_POINTS crystal` listing the path.
3. Convert it:

```bash
uv run python scripts/qe_bands_to_npz.py bands/out/<prefix>.save/data-file-schema.xml \
    --scf-out calc/scf.out --out calc/reference/dft_bands_path.npz
```

### Step 4: score the frontier, with no model

```bash
uv run autowan --run-dir calc --seed <seed> \
    --parent-xml calc/out/<prefix>.save/data-file-schema.xml \
    --gate-profile band_interpolation \
    --reference-path calc/reference/dft_bands_path.npz \
    --mpi "mpirun -np 8" \
    --no-repair --log score.json
```

This runs `pw2wannier90` and `wannier90` on the frontier deck and prints the gate's
verdict, check by check, as in the demo. No key, no cost.

### Step 5: let the model repair it

```bash
export OPENROUTER_API_KEY=sk-or-...        # your own key; autowan ships none
uv run autowan ...same arguments...  --log run.json   # without --no-repair
```

Every change goes to a new seedname (`<seed>_r1`, `<seed>_r2`, ...), so nothing you had
is overwritten. The terminal shows each diagnosis, change, verdict and cost as it
happens. `run.json` keeps all of it.

**If it ends certified**, the summary at the end names your construction
(`construction : <seed>_rN`; its files are `<seed>_rN.win`, `_hr.dat`, ...). **If it
ends with a stop & report**, see [below](#when-it-stops-with-a-report). If the provider
stops (no credit, network), the run is recorded as stopped; `--resume` continues from
the same attempt.

### Choosing what "certified" means: gate profiles

Pick the profile for what you will use the Wannier functions for:

| | `--gate-profile` | use it when you want | it certifies when |
|---|---|---|---|
| C1 | `band_interpolation` | bands, Fermi surfaces, transport from interpolation | η₂ ≤ 20 meV against the DFT path bands (Fermi-Dirac weights up to E_F + 2 eV) |
| C2 | `sparse_tb` | a short-range tight-binding model | η₂ ≤ 20 meV still holds after every hopping longer than L_BvK/4 is cut |
| C3 | `dmft_subspace` | the correlated subspace for DMFT (SrVO3, SrTiO3, NiO, MnO, Fe) | the literature's target orbitals, with their states reproduced to ≤ 20 meV RMS |

Every profile also runs the soundness checks ([below](#the-soundness-checks)). `default`,
which is what you get without `--gate-profile`, runs only those, with no η₂ criterion.
`downfolded_d` and `atom_centred_basis` are also available. Each verdict records
the profile's hash, so two results are comparable exactly when the hashes match.

#### The soundness checks

The criterion (C1, C2 or C3) is what a certificate is about. The soundness checks run
under every profile and only screen out constructions that are broken in a way the
criterion might not see. They fall into three groups:

**Run and input integrity.** Facts about the run, not about the choice of Wannier
functions. A standard Wannier90 run passes the first two by construction.

| check | passes when |
|---|---|
| `wannier_count_matches_request` | Wannier90 built as many functions as the deck's `num_wann` (it does not judge whether that number is physically right) |
| `hopping_shells_inside_representable_supercell` | every lattice vector R in `_hr.dat` lies inside the Wigner–Seitz supercell of `mp_grid` (a count of vectors, not a bound on the size of H(R)) |
| `metal_frozen_window_contains_fermi_level` | for a metal, a declared frozen window contains E_F (a property of the deck, checked after the run because the gate also scores decks autowan did not write) |

**Locality of the model.** Properties of the Wannier Hamiltonian itself.

| check | passes when |
|---|---|
| `centres_stay_near_home_cell` | every centre is within one nearest-neighbour distance of some atom, lattice images included |
| `hopping_decays_with_distance` | the largest hopping in the outermost 10 % of shells is ≤ 1 % of the largest on-site element |

**Agreement with the parent bands on the construction mesh.** These compare the model
with the DFT eigenvalues on the nscf mesh the construction used. C1 and C2 instead compare
on a separate band path at 20 meV. The tolerances here are coarse (1 % of the occupied
bandwidth, about 0.1 eV for most materials), so they catch a model that misses or invents
whole states, not one that is slightly off.

| check | passes when |
|---|---|
| `frontier_manifold_straddles_fermi_level` | at every mesh k-point where the parent has bands both below and above E_F, so does the model. For a gapped parent with E_F in the gap, the model must therefore contain some conduction states; a valence-only model fails it |
| `occupied_manifold_is_represented` | ≤ 5 % of the occupied parent states (all states below E_F, minus an isolated semicore block the deck excludes) lack a model band within 1 % of the occupied bandwidth |
| `model_bands_correspond_to_parent_bands` | ≤ 20 % of the model's bands inside the deck's frozen window (or up to E_F + 2 eV when the deck sets none) lie away from every parent band by more than the same tolerance. The bound is loose on purpose, because above the frozen window a disentangled model is free |
| `band_distance_from_parent` | gapped parents only: RMS model–parent distance from the bottom of the occupied bands to E_F + 10 % of the occupied bandwidth is ≤ 1 % of the occupied bandwidth |

A check that does not apply is *withheld*: it is recorded as not applicable and left out
of the verdict, never counted as passed or failed. `band_distance_from_parent` is withheld
for metals, because a metal has no isolated manifold to reproduce, and widening the
tolerance until it passed would make the number meaningless. C3 (and `downfolded_d`) aim
at a sub-manifold, so they drop the three checks that require the whole occupied manifold:
`frontier_manifold_straddles_fermi_level`, `occupied_manifold_is_represented` and
`band_distance_from_parent`.

For the poster campaign we checked whether any frontier construction lost its
certificate to a soundness check alone, with its C1, C2 or C3 value inside the limit.
None did. In one case the soundness checks changed the count: for MoS₂ C2, ladder point 3
had η = 19.4 meV after truncation but failed `hopping_decays_with_distance`, so the
frontier was certified at point 9 instead.

#### C3 in detail

C3 does not choose the correlated subspace. For each of its five materials the target
(which orbitals, hence `num_wann`) is fixed from the DMFT literature, and the question is
whether a model of that subspace can be built automatically:

| material | target, per spin | literature |
|---|---|---|
| SrVO3 | V t2g | Pavarini et al. 2004; Lechermann et al. 2006; Amadon et al. 2008 |
| SrTiO3 | Ti t2g (the empty conduction band) | Zhong, Toth & Held 2013; Souto-Casares, Spaldin & Ederer 2021 |
| NiO | Ni 3d + O 2p | Kuneš et al. 2007 |
| MnO | Mn 3d + O 2p | Kuneš et al. 2008 |
| Fe | Fe 4s + 3d | Şaşıoğlu et al. 2011; Leonov et al. 2011 |

Every other material is withheld. A construction passes when all three hold:

1. **Count and seeds.** `num_wann` equals the target's count, and the projectors name
   exactly the target orbitals (`V:dxy;dyz;dxz`, `Ni:d`, `O:p`, or the kept PAOs under
   PDWF). SCDM names no orbitals and is refused.
2. **Correlated functions on their sites.** Each centre is assigned to its nearest
   correlated atom (the transition metal). Each such atom must hold at least its
   correlated count (3 for t2g, 5 for d) within half a nearest-neighbour distance. The
   check counts centres; it does not test their orbital character. Ligand and s functions
   are held only by the drift guard.
3. **The target manifold, reproduced.** The target states are the DFT states on the band
   path whose Löwdin weight on the target orbitals is at least 0.5. The weight is the sum
   of |⟨φ|ψ_nk⟩|² over the target (atom, l, m) components, from `projwfc.x` run on the
   path's `pw.x` bands calculation. The basis φ is the pseudopotential's atomic
   wavefunctions (`PP_CHI`), Löwdin-orthogonalised. Each target state is paired with a model
   band, and the RMS of the differences must be ≤ 20 meV. An isolated manifold is compared
   over its whole width. An entangled one is compared up to the frontier's frozen-window
   top (E_F + 2 eV for a metal, CBM + 2 eV for a gapped parent). Whether the manifold is
   isolated is decided from the same bands and weights. The literature judges this
   agreement by eye; 20 meV is our choice, the same as C1 and C2.

**The C3 frontier is our adaptation, not a published method.** PDWF keeps a pseudo-atomic
orbital whole, so it cannot express a t2g-only target. Both variants start from the same
QE frontier deck as C1 (aiida-wannier90-workflows protocol, PDWF, semicore excluded):

| variant | materials | changes to the frontier deck |
|---|---|---|
| textbook SMV | SrVO3, SrTiO3 | PDWF keys and `dis_froz_max` removed; projections on the t2g orbitals; outer window on the target manifold; one run |
| PDWF-subspace | NiO, MnO, Fe | every non-target PAO excluded (NiO: `num_wann` 26 → 16); the energy frozen window removed, so states are frozen by projectability only; then the `dis_proj_min`/`dis_proj_max` retry ladder |

On the poster, SrVO3 and SrTiO3 pass on the first run, NiO passes at ladder point 3, MnO
passes in the up channel only, and Fe fails in both channels. A variant of PDWF-subspace
that keeps the energy frozen window has not been tried.

### When it stops with a report

Sometimes the problem is in the DFT, not in the Wannier construction: too few bands in
the parent, a k-mesh too coarse for the hoppings, a missing semicore shell. Then the model
ends the run with `escalate` and a structured report, saved under `"report"` in
`run.json`. It looks like this (an illustration, not a real run):

```json
"report": {
  "binding_limit": "parent_band_pool",
  "requested_changes": {"nbnd": 40},
  "evidence": ["frozen_capacity: at most 3 states per k below E_F+5 eV against 4 functions", "..."],
  "expected_effect": "with 40 bands the frozen window can hold the sp manifold and eta_2 should fall below 20 meV"
}
```

`binding_limit` is one of `parent_band_pool`, `parent_kmesh`, `parent_numerics`,
`parent_pseudo_or_semicore`, `parent_spin_or_magnetism`, `structure`,
`criterion_unreachable` or `other`.

**Treat a report as a claim to check, not a verdict.** Redo the DFT with the requested
change and run autowan again. If the change does not lead to a certified construction,
the report was wrong.

### Models and cost

- **OpenRouter** (the default): any model with tool calling, by its OpenRouter ID.
  `--model anthropic/claude-opus-5.5` is the default. Repairs in the poster campaign ran
  with Anthropic, OpenAI, Google, NVIDIA and Z.ai models.
- **Your own server** (`--provider local`, or `--base-url http://host:8000/v1`): any
  OpenAI-compatible endpoint with tool calling, e.g. vLLM. Set
  `AUTOWAN_LOCAL_BASE_URL` (and `AUTOWAN_LOCAL_API_KEY` if it needs one). The local
  preset's model is `RedHatAI/Qwen3.5-9B-quantized.w8a8`; `--model` picks another. Tokens are
  recorded and no cost is charged.

Every request records its tokens and cost. Costs are at **API list prices**, so they are
what a pay-per-token user would pay. In the sodium demo, each repair attempt with
Claude Opus 5.5 cost USD 0.35–0.52. `--min-balance` stops a run before your key runs
dry.

Only the step a change affects is re-run. Most repairs change only the localisation,
which takes seconds. The overlaps (`pw2wannier90`, minutes to tens of minutes) are
recomputed only when a change touches the projections.

### All environment variables

| variable | what it is | default |
|---|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter key; read and used, never written anywhere | (none; needed only to repair) |
| `AUTOWAN_LOCAL_BASE_URL` / `AUTOWAN_LOCAL_API_KEY` | your OpenAI-compatible server | (none) |
| `WANNIER90` / `PW2WANNIER90` / `VASP` | executables (also `--wannier90`, `--pw2wannier90`, `--vasp`) | found on `PATH` |
| `AUTOWAN_POTCAR_LIBRARY` | your VASP POTCAR library (`<label>/POTCAR`) | (none) |
| `AUTOWAN_TEST_*` | only for the test suite: stored run trees of your own | unset → those tests skip |

### From Python

```python
from autowan import WannierRunner, make_provider, wannierize

result = wannierize(
    run_dir="calc",
    parent_xml="calc/prefix.save/data-file-schema.xml",
    seed="my_deck",
    runner=WannierRunner(run_dir="calc"),
    provider_factory=lambda model: make_provider(model),
)
print(result.status, result.seed)
```

### Running the tests

```bash
uv sync --extra dev
uv run pytest -q          # ~1 min; tests that need engines or stored runs skip
```

---

## How the decision is protected

These rules are why a certificate means the same thing on every machine.

**Physics decides acceptance.** A deterministic gate scores every construction against
the calculation that produced it. A model never declares a construction acceptable and
is never asked to.

**A threshold is a fraction of a scale the material supplies.** No check carries a bare
constant. Each scalar threshold is a dimensionless fraction of a quantity read from
this material: its occupied bandwidth, its nearest-neighbour distance, or this
construction's own on-site hopping scale. Each threshold also records where the
fraction was calibrated. The same declaration resolves to 1.243 Å for iron and 1.169 Å
for silicon.

**A check that cannot be evaluated is not a check that passed.** Missing evidence costs
a construction its certificate. It is reported separately from a failure, because the
two need different repairs.

**A check that does not apply is withheld, not loosened.** A metal has no isolated
manifold, so band distance is not applied to metals, rather than being given a wider
threshold. Difficulty is never a reason for a gate to become easier.

**The model is consulted only after the frontier fails.** Before each proposal it may
call read-only tools: deck validation, method defaults, frozen-window capacity, centres
and spreads, band character, short reference pages, the frontier's own rescan, and the
corpus of earlier repairs. Tool calls cost no attempt and are capped per attempt. Every
call is logged with the attempt it served.

**The construction method never changes silently.** A deck belongs to one family:
SMV (energy windows with trial projections), PDWF (projectability on the
pseudopotential's orbitals) or SCDM (density-matrix columns). An edit may move only
that family's knobs. Changing family is the `switch_method` action, which costs an
attempt and is recorded with where it came from, where it went and why. Semicore
exclusion is a shared knob, `semicore_exclusion`, named by shell per species. The deck
check refuses a cut the gate would not honour.

**The model acts only through a closed vocabulary.** It returns a validated action
naming parameters from a fixed list, each within a declared range. No reachable knob
changes an acceptance threshold, touches the parent calculation, or runs a command.

## Extending it

**Your own checks.** `--add-check` / `--drop-check` adjust a profile, and `--check` adds
a check you wrote yourself (see `gates/README.md`). Acceptance is the conjunction over
the profile's checks. Dropping a default check prints a warning.

**Your own recipe directory.** The model's `search_references` tool searches
`--recipes <dir>`, by default the corpus shipped in the package. `--contribute-recipes
<dir>` distils a finished run's log into the steps where a check stopped failing, so you
can build a corpus from your own runs.

The shipped corpus has **60 entries** distilled from the certified repairs of the
KPS 2026 poster campaign (QE, PDWF frontier, SSSP pseudopotentials). It also keeps
**12 obsolete entries** from an earlier energy-window protocol, which are left out of
the default search. By default nothing is held back: in production the corpus is prior
knowledge, including entries from your own material. A benchmark that measures
transfer rather than recall can set these in `WorkflowConfig`:

- `recipe_same_material_holdback = True` leaves out recipes from the caller's material;
- `frontier_dataset_same_composition = False` hides the caller's composition from the
  frontier table.

`src/autowan/corpus/README.md` has the full contract.

## Contributing to a shared corpus or gate library: not open yet

The aim is that every run anyone does makes the next run better, and that people can
publish gates for their own use of the bands. How to do that is still being designed,
so there is no submission route yet: please do not open pull requests with recipe
entries or checks for now.

What we know so far, measured on the poster campaign's 1,274 repair steps over 22
materials:

- Retrieving past repairs by the material's electronic-structure signature (the
  current `search_references`) predicted whether a change would help no better than
  the average for that kind of change. The format will change.
- Earlier runs on the same material with the same DFT setup were strongly predictive.
  Whether that survives a different setup (another code, pseudopotential or k-mesh) is
  not yet measured.

Ideas, and runs you would be willing to share, are welcome as
[issues](https://github.com/SCAI-MJ/autowan/issues).

## Known limitations

- The multimodal review stage (`--review`) is lenient. It can only withhold an
  acceptance the measurements already granted, so it cannot do harm, but its detection
  rate has not been measured. Do not rely on it.
- The `dmft_subspace` (C3) targets cover five materials (SrVO3, SrTiO3, NiO, MnO, Fe).
  For any other material the criterion is withheld.
- The bundled examples are for reading and re-scoring. Re-running them needs the parent
  wavefunctions, which are not shipped. The inputs to redo the DFT are included.

## Bundled data and attribution

`src/autowan/data/frontier_wannier_lookup.csv.gz` is the table behind the read-only
`frontier_dataset` tool: one row per Wannierisation workflow of the published PDWF
frontier, extracted from Materials Cloud Archive records under **CC BY 4.0**. The table
is not covered by the MIT licence. If you use it, cite:

- J. Qiao, G. Pizzi, N. Marzari, *Projectability disentanglement for accurate and
  automated electronic-structure Hamiltonians*, npj Comput. Mater. **9**, 208 (2023);
  data doi:10.24435/materialscloud:x0-yf.
- Y. Jiang *et al.*, *Robust Wannierization including magnetization and spin-orbit
  coupling via projectability disentanglement*, npj Comput. Mater. **11**, 353 (2025);
  data doi:10.24435/materialscloud:ck-br.

`src/autowan/data/FRONTIER_DATASET.md` describes the extraction. `NOTICE` lists every
third-party file.

## Citation

autowan was presented as a poster at the 2026 Fall Meeting of the Korean Physical
Society (KPS 2026). Until a paper is out, please cite this repository:
https://github.com/SCAI-MJ/autowan

## Licence

MIT (see `LICENSE`). Third-party data keeps its own licence (see `NOTICE`).
