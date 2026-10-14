#!/usr/bin/env python3
"""Score a directory of Wannier constructions against the certification gate.

Prints one row per construction and, beneath it, the verdict of every check — the
table a demonstration needs, because the claim is not that a repair lowered a number
but that it changed a gate's decision, and that is only legible when the decision and
the measurements behind it appear together.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autowan.core_checks import CORE_CHECKS  # noqa: E402
from autowan.gate import (  # noqa: E402
    CertificationGate,
    Construction,
)
from autowan.qe_outputs import read_parent  # noqa: E402
from autowan.w90_outputs import read_run  # noqa: E402


def _win_value(win: Path, key: str):
    if not win.is_file():
        return None
    for line in win.read_text().splitlines():
        stripped = line.split("#")[0].split("!")[0].strip()
        if not stripped or "=" not in stripped and not stripped.lower().startswith(key):
            continue
        name, _, value = stripped.partition("=")
        if name.strip().lower() == key:
            return value.strip()
        if key == "mp_grid" and stripped.lower().startswith("mp_grid"):
            return stripped.split(None, 1)[1]
    return None


def _inputs_from_win(win: Path, spin_index: int) -> dict:
    mesh = _win_value(win, "mp_grid")
    num_wann = _win_value(win, "num_wann")
    return {
        "num_wann": int(num_wann) if num_wann else None,
        "mp_grid": tuple(int(x) for x in mesh.split()) if mesh else None,
        "spin_index": spin_index,
        "num_iter": _win_value(win, "num_iter"),
        "conv_window": _win_value(win, "conv_window"),
        "trial_step": _win_value(win, "trial_step"),
        "dis_froz_max": _win_value(win, "dis_froz_max"),
        "dis_proj_min": _win_value(win, "dis_proj_min"),
        "scdm_mu": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--parent-xml", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="seednames to score, e.g. fe_up_pdwf fe_up_pdwfit")
    ap.add_argument("--spin-of", default="",
                    help="substring:index pairs mapping a seedname to a spin channel, "
                         "e.g. '_up_:0,_dn_:1'")
    ap.add_argument("--csv", type=Path, help="also write the verdict rows here")
    args = ap.parse_args()

    spin_map = []
    for pair in filter(None, args.spin_of.split(",")):
        needle, _, index = pair.partition(":")
        spin_map.append((needle, int(index)))

    parent = read_parent(args.parent_xml)
    material_class = "metal" if parent.is_metallic() else "gapped"
    gate = CertificationGate(CORE_CHECKS)
    print(f"parent: {args.parent_xml}")
    print(f"  E_F = {parent.fermi_energy_ev:.4f} eV   bands crossing it = "
          f"{parent.crossing_bands()}   -> {material_class}\n")

    rows = []
    for seed in args.seeds:
        wout = args.run_dir / f"{seed}.wout"
        if not wout.is_file():
            print(f"=== {seed}: not produced ===\n")
            continue
        spin = next((i for needle, i in spin_map if needle in seed), 0)
        inputs = _inputs_from_win(args.run_dir / f"{seed}.win", spin)
        report = gate.run(
            Construction(seed, parent=parent, wannier=read_run(wout),
                         material_class=material_class, inputs=inputs)
        )
        knobs = " ".join(
            f"{k}={v}" for k, v in inputs.items()
            if k in ("num_iter", "dis_froz_max", "conv_window", "trial_step") and v is not None
        )
        verdict = "CERTIFIED" if report.passed else "REFUSED"
        print(f"=== {seed}  [{verdict}]  coverage={report.coverage:.2f}  {knobs}")
        for v in report.verdicts:
            if not v.applies:
                mark, detail = "n/a ", v.note
            elif not v.evaluable:
                mark, detail = "????", v.note
            else:
                mark = "pass" if v.passed else "FAIL"
                detail = f"{v.metric_value:.4g} {v.metric_units}"
                if v.threshold_value is not None:
                    detail += f" vs {v.threshold_value:.4g}"
                if v.note:
                    detail += f"   ({v.note})"
            print(f"   {mark}  {v.check_id:<46} {detail}")
            rows.append({"seed": seed, "certified": report.passed, "check_id": v.check_id,
                         "applies": v.applies, "evaluable": v.evaluable, "passed": v.passed,
                         "metric_value": v.metric_value, "metric_units": v.metric_units,
                         "threshold_value": v.threshold_value, "note": v.note})
        if report.failed_check_ids:
            print(f"   -> failed: {', '.join(report.failed_check_ids)}")
        if report.non_evaluable_check_ids:
            print(f"   -> no verdict: {', '.join(report.non_evaluable_check_ids)}")
        print()

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}  ({len(rows)} verdict rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
