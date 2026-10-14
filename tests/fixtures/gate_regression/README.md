# Gate regression golden

`no_exclusion_reports.json` holds the certification gate's full report (verdict,
failed and non-evaluable checks, coverage, every measurement and note) for the
constructions built by `_regression_cases` in `tests/test_gate_excluded_bands.py`:
decks that exclude nothing, or exclude only bands with no occupied state.

It was written by the **unmodified `origin/autowan/dev` code at `bcc77a60d`**, before the
gate learned to read `exclude_bands`:

```bash
PYTHONPATH=src .venv/bin/python tests/test_gate_excluded_bands.py --write-golden
```

The test re-runs the same constructions through the current gate and requires the JSON to
be byte-identical (the new `exclude_bands` gate input is dropped before comparing, since
the old code never produced it). Regenerate it only from a gate whose behaviour on such
decks is meant to change, and say so in the commit.

**Regenerated 2026-10-07 on `autowan/feat/poster-campaign-integration`.** The merge of
`autowan/feat/vasp-gate-cascade-adequacy-integration` changes the gate's behaviour on these
decks on purpose: an eleventh core check (`metal_frozen_window_contains_fermi_level`) and the
semicore-independent occupied-bandwidth scale. The golden was rewritten by that branch's
`autowan/src` at `0f681ed9b` -- the same gate without the `exclude_bands` reading, which is
the role `bcc77a60d` played before -- running this file's `_regression_cases`. Against the
previous golden, the only differences are the eleventh check's rows and the tolerances and
thresholds the new scale sets.
