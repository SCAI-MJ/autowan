#!/usr/bin/env python3
"""Score every material of a campaign, with the quantity that predicts the hard ones.

Alongside the gate's verdict this reports the ratio of projectors to bands inside the
window a projectability construction is asked to reproduce. The projectability
literature identifies that ratio as what separates the easy cases from the hard: the
projectors come from the pseudopotential and their number is fixed by chemistry, while
the bands to be disentangled are not, and where the two are comparable the
disentanglement has barely more information than it has unknowns.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from autowan.core_checks import CORE_CHECKS  # noqa: E402
from autowan.gate import CertificationGate, Construction  # noqa: E402
from autowan.inputs import read_construction  # noqa: E402
from autowan.qe_outputs import read_parent  # noqa: E402
from autowan.w90_outputs import WannierOutputError, read_run  # noqa: E402

WINDOW_EV = 2.0  # the window the protocol asks a construction to reproduce

#: The seed the weakened-baseline sweeps wrote. Other constructions of the same parents
#: (the frontier rebuild writes aiida-wannier90-workflows' ``aiida``) name theirs here.
DEFAULT_SEED = "{name}_pdwf"


def seed_for(name: str, template: str | None) -> str:
    """The Wannier90 seed of material ``name``: ``template`` with ``{name}`` filled in."""
    return (template or DEFAULT_SEED).format(name=name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--materials", nargs="+", required=True)
    ap.add_argument("--csv", type=Path)
    ap.add_argument(
        "--seed",
        default=None,
        help="Wannier90 seed per material, '{name}' filled in (default: '{name}_pdwf')",
    )
    args = ap.parse_args()

    gate = CertificationGate(CORE_CHECKS)
    rows = []
    header = (
        f"{'material':<9}{'class':<8}{'nwann':>6}{'bands<EF+2':>11}{'ratio':>7}  "
        f"{'verdict':<10}{'unrepr':>8}{'spurious':>9}  failed"
    )
    print(header)
    print("-" * len(header))

    for name in args.materials:
        d = args.root / name
        xml = d / f"{name}.save" / "data-file-schema.xml"
        seed = seed_for(name, args.seed)
        wout = d / f"{seed}.wout"
        if not xml.is_file():
            print(f"{name:<9} (no parent calculation)")
            continue
        parent = read_parent(xml)
        material_class = "metal" if parent.is_metallic() else "gapped"
        ef = parent.fermi_energy_ev
        in_window = float(
            np.mean([(bk <= ef + WINDOW_EV).sum() for bk in parent.eigenvalues_ev[0]])
        )

        inputs = read_construction(d, seed)
        try:
            wannier = read_run(wout)
        except WannierOutputError as exc:
            # A construction that would not run still has a ratio: the count it asked
            # for is in the deck, and that count against the spectrum is the whole
            # point of the column. Leaving the row blank hid the two materials the
            # ratio predicts hardest.
            declared = inputs.values.get("num_wann")
            nwann = int(float(declared)) if declared else 0
            ratio = nwann / in_window if in_window else float("nan")
            print(
                f"{name:<9}{material_class:<8}{nwann:>6}{in_window:>11.2f}"
                f"{ratio:>7.2f}  {'no output':<10}{'':>8}{'':>9}  {str(exc)[:40]}"
            )
            rows.append(
                {
                    "material": name,
                    "class": material_class,
                    "num_wann": nwann,
                    "bands_in_window": round(in_window, 2),
                    "ratio": round(ratio, 3),
                    "certified": False,
                    "failed_checks": str(exc),
                }
            )
            continue

        report = gate.run(
            Construction(
                name,
                parent=parent,
                wannier=wannier,
                material_class=material_class,
                inputs=inputs.gate_inputs(0),
            )
        )
        measured = report.measurements()
        unrepr = measured.get("occupied_manifold_is_represented", {}).get("value")
        spur = measured.get("model_bands_correspond_to_parent_bands", {}).get("value")
        nwann = wannier.num_wann or 0
        ratio = nwann / in_window if in_window else float("nan")
        verdict = "CERTIFIED" if report.passed else "refused"
        failed = (
            ", ".join(report.failed_check_ids + report.non_evaluable_check_ids) or "-"
        )
        print(
            f"{name:<9}{material_class:<8}{nwann:>6}{in_window:>11.2f}{ratio:>7.2f}  "
            f"{verdict:<10}"
            f"{(f'{unrepr:.1%}' if unrepr is not None else '-'):>8}"
            f"{(f'{spur:.1%}' if spur is not None else '-'):>9}  {failed[:46]}"
        )
        rows.append(
            {
                "material": name,
                "class": material_class,
                "num_wann": nwann,
                "bands_in_window": round(in_window, 2),
                "ratio": round(ratio, 3),
                "certified": report.passed,
                "unrepresented_occupied": None if unrepr is None else round(unrepr, 4),
                "spurious_model_bands": None if spur is None else round(spur, 4),
                "total_spread_ang2": wannier.total_spread_ang2,
                "failed_checks": failed,
            }
        )

    if args.csv and rows:
        fields = sorted({k for r in rows for k in r})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
