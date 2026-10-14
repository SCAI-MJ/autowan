#!/usr/bin/env python3
"""One table over every construction produced for a material, and the repairs applied.

Written for the reader who has not followed the run: the claim is not that a number
improved but that a gate changed its decision, so each row carries the decision, the
knobs in force, and the check that was unresolved when it was refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autowan.core_checks import CORE_CHECKS  # noqa: E402
from autowan.gate import CertificationGate, Construction  # noqa: E402
from autowan.qe_outputs import read_parent  # noqa: E402
from autowan.w90_outputs import WannierOutputError, read_run  # noqa: E402

KNOBS = ("num_iter", "conv_window", "trial_step", "dis_froz_max", "dis_win_max",
         "dis_proj_min", "dis_proj_max")


def read_win(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        stripped = line.split("!")[0].split("#")[0].strip()
        if stripped.lower().startswith("mp_grid"):
            out["mp_grid"] = stripped.split(None, 1)[1].lstrip("= ").strip()
        elif "=" in stripped:
            name, _, value = stripped.partition("=")
            out[name.strip().lower()] = value.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--parent-xml", required=True, type=Path)
    ap.add_argument("--spin-of", default="_up_:0,_dn_:1")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    spin_map = [(n, int(i)) for n, _, i in
                (p.partition(":") for p in filter(None, args.spin_of.split(",")))]
    parent = read_parent(args.parent_xml)
    material_class = "metal" if parent.is_metallic() else "gapped"
    gate = CertificationGate(CORE_CHECKS)

    print(f"parent  : {args.parent_xml}")
    print(f"          E_F = {parent.fermi_energy_ev:.4f} eV, "
          f"{parent.crossing_bands()} bands cross it -> {material_class}\n")

    header = f"{'construction':<20} {'decision':<11} {'cov':>5}  {'knobs':<44} unresolved"
    print(header); print("-" * len(header))
    rows = []
    for wout in sorted(args.run_dir.glob("*.wout")):
        seed = wout.stem
        win = read_win(args.run_dir / f"{seed}.win")
        if "num_wann" not in win:
            continue
        try:
            wannier = read_run(wout)
        except WannierOutputError:
            print(f"{seed:<20} {'no output':<11}"); continue
        spin = next((i for n, i in spin_map if n in seed), 0)
        inputs = {
            "num_wann": int(win["num_wann"]),
            "mp_grid": tuple(int(x) for x in win["mp_grid"].split()) if "mp_grid" in win else None,
            "spin_index": spin,
            **{k: win.get(k) for k in KNOBS},
        }
        report = gate.run(Construction(seed, parent=parent, wannier=wannier,
                                       material_class=material_class, inputs=inputs))
        knobs = " ".join(f"{k}={win[k]}" for k in KNOBS if k in win)
        unresolved = ", ".join(report.failed_check_ids + report.non_evaluable_check_ids) or "-"
        print(f"{seed:<20} {'CERTIFIED' if report.passed else 'refused':<11} "
              f"{report.coverage:>5.2f}  {knobs[:44]:<44} {unresolved}")
        rows.append({"seed": seed, "certified": report.passed,
                     "coverage": round(report.coverage, 3), "knobs": knobs,
                     "unresolved": unresolved,
                     "total_spread_ang2": wannier.total_spread_ang2})

    for log in sorted(args.run_dir.glob("stage1_run*.json")):
        record = json.loads(log.read_text())
        repair = record.get("repair") or {}
        print(f"\nrepair  : {record['material']}  via {repair.get('model')} "
              f"({repair.get('structured_mode')})")
        print(f"          applied {repair.get('applied')} -> "
              f"{'CERTIFIED' if repair.get('certified_after') else 'still refused'}"
              f"   cost ${(repair.get('usage') or {}).get('cost', 0):.4f}")

    if args.csv and rows:
        import csv as _csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
