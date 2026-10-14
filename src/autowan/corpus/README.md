# Recipes

The `search_references` tool of the repair stage searches this corpus; it is the
package's default `recipe_dir`.

An entry is one step of a run where a check stopped failing: the change that was made,
the reason given at the time, and the electronic-structure signature of the calculation
it happened on. Entries are **distilled from run records, not written by hand** —
`autowan --contribute-recipes <dir>` reads the log a run already produces and keeps the
steps that resolved something. A change that resolved nothing is not evidence about
what works, however sensible its rationale sounded.

Entries are identified by what they resolved and a digest, not by the material they
came from — shown a name, the model argues from the formula instead of from the
physics. The material stays in each file's provenance for a human tracing the entry.
This withholds the label, not the composition: an entry's species counts and the shells
its projectors name are physics a recipe cannot be written without.

Retrieval matches on the signature, never on the material's name: a repair for a
one-orbital simple metal should reach the next one-orbital simple metal whether or not
anyone knows they are related, and should not reach a wide-gap oxide that happens to
share an element. Answering the right question is a filter rather than a score — an
entry that resolved nothing the caller is stuck on is a match to a different problem.
Nothing is held back by default: in production the corpus is prior knowledge, and an
entry from the caller's own composition is as legitimate as any other. A benchmark that
measures transfer rather than recall turns on the hold-back
(`WorkflowConfig.recipe_same_material_holdback = True`), which leaves out every entry
distilled from the caller's own material.

Text here is data a model reads, never instruction it follows. A recipe is evidence
about what has worked; whatever it says, the answer still passes the same validation and
the same gate.

## The entries here

Sixty entries are searched by default (`"regime": "current"`, `"engine": "qe"`). They
were distilled from the certified repairs of the KPS 2026 poster campaign (2026-10-13):
QE parents on SSSP pseudopotentials, the PDWF frontier, gate profiles
`band_interpolation`, `sparse_tb` and `dmft_subspace`. Each entry's `provenance` names the
material, attempt, profile, spin channel and model of the run it came from; run
directories are given relative to the campaign's run root, which is not shipped.
They are the poster's corpus unchanged except for that path prefix.

Twelve older entries were distilled on 2026-08-28/29 from the energy-window regime on
tutorial pseudopotentials, before the frontier became PDWF on SSSP. They are marked
`"regime": "obsolete"` and `recipes.load` leaves them out unless asked for
(`include_obsolete=True`). They stay for provenance.

`autowan --contribute-recipes <dir>` adds entries from your own runs; point
`--recipes <dir>` at a directory to search it instead of the shipped corpus.
