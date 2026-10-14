#!/usr/bin/env python3
"""Score finished Wannier constructions with the gate, without running anything.

Reads what a construction already left on disk (``<seed>.win``, ``.wout``, ``_hr.dat``,
``_wsvec.dat``, ``_centres.xyz``, ``_band.dat``) together with the parent calculation's
``data-file-schema.xml`` and, for a criterion profile, the DFT bands on the reference
path. No DFT code, no Wannier90 and no model are needed, so this is the quickest way to
see what the gate measures:

    python scripts/rescore.py examples/sodium

A directory holding ``example.json`` supplies its own defaults (run directory, parent,
reference path, profile, seeds); every flag below overrides them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autowan.backends import get_backend  # noqa: E402
from autowan.gate_profiles import build_gate  # noqa: E402
from autowan.w90_outputs import read_run  # noqa: E402
from autowan.workflow import WorkflowConfig, _reference_path, _Scorer  # noqa: E402


def _defaults(directory: Path) -> dict:
    manifest = directory / "example.json"
    if not manifest.is_file():
        return {"run_dir": str(directory)}
    spec = json.loads(manifest.read_text())
    for key in ("run_dir", "parent_xml", "reference_path"):
        if spec.get(key):
            spec[key] = str(directory / spec[key])
    return spec


def _line(verdict) -> str:
    if not verdict.applies:
        return f"  n/a   {verdict.check_id:<46} {verdict.note or ''}"
    if not verdict.evaluable:
        return f"  ????  {verdict.check_id:<46} {verdict.note or ''}"
    mark = "pass" if verdict.passed else "FAIL"
    detail = f"{verdict.metric_value:.4g} {verdict.metric_units}"
    if verdict.threshold_value is not None:
        detail += f"  (limit {verdict.threshold_value:.4g})"
    return f"  {mark}  {verdict.check_id:<46} {detail}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directory", type=Path, help="an example directory or a run directory")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--parent-xml", type=Path)
    ap.add_argument("--reference-path", type=Path)
    ap.add_argument("--gate-profile")
    ap.add_argument("--seeds", nargs="+")
    ap.add_argument("--backend", default=None, help="qe (default) or vasp")
    args = ap.parse_args()

    spec = _defaults(args.directory)
    run_dir = Path(args.run_dir or spec.get("run_dir") or args.directory)
    parent_xml = args.parent_xml or spec.get("parent_xml")
    reference_path = args.reference_path or spec.get("reference_path")
    profile = args.gate_profile or spec.get("gate_profile", "default")
    seeds = args.seeds or spec.get("seeds") or sorted(
        p.stem for p in run_dir.glob("*.wout")
    )
    labels = spec.get("labels", {})
    if parent_xml is None:
        ap.error("--parent-xml is required (or an example.json naming it)")

    backend = get_backend(args.backend or spec.get("backend", "qe"))
    parent = backend.read_parent(Path(parent_xml))
    material_class = "metal" if parent.is_metallic() else "gapped"
    reference = None
    if reference_path:
        reference = _reference_path(WorkflowConfig(reference_path=Path(reference_path)))
    config = WorkflowConfig(
        gate_profile=profile, reference_path=reference, draw_figures=False
    )
    gate = build_gate(backend, profile, (), reference_path=reference)
    scorer = _Scorer(config, gate, backend, parent, material_class, run_dir)

    print(f"gate profile: {profile}")
    print(
        f"parent: E_F = {parent.fermi_energy_ev:.3f} eV, "
        f"{parent.crossing_bands()} bands cross it -> {material_class}"
    )
    if reference is None:
        print("no reference band path: a criterion profile cannot certify without one")
    print()

    for seed in seeds:
        wout = run_dir / f"{seed}.wout"
        if not wout.is_file():
            print(f"{seed}: no {wout.name}, skipped\n")
            continue
        report = scorer.score(
            seed, backend.read_construction(run_dir, seed), read_run(wout)
        )
        verdict = "CERTIFIED" if report.passed else "REFUSED"
        label = f"  ({labels[seed]})" if seed in labels else ""
        print(f"{seed}{label}: {verdict}")
        for v in report.verdicts:
            print(_line(v))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
