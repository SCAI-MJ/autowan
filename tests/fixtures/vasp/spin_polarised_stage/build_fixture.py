"""Rebuild this fixture from the real VASP 6.4.1 overlap-step stages it records.

Source (read-only scratch, 2026-09-29 parent-adequacy controls on VASP, report
reports/2026-09-29/10-autowan-parent-adequacy-prescription-vasp.md):

    <SCRATCH>/spin/clean/construction/mno.vasp2wannier90      ISPIN = 2 (MAGMOM 9 0)
    <SCRATCH>/spin/degraded/construction/mno.vasp2wannier90   ISPIN = 1

Both are the stage directory ``VaspWannierRunner.run_overlaps`` laid down and VASP
(vasp.6.4.1 with -DVASP2WANNIER90, LWANNIER90 = .TRUE.) then wrote into. What is kept:

* ``listing.tsv``: every name in the stage after VASP ran, its size, and who put it
  there (``parent-link`` = the runner's link to the parent; ``runner`` = INCAR and
  wannier90.win, written by the runner before VASP starts; ``vasp`` = written by VASP,
  including the redirected stdout/stderr);
* the overlap products' excerpts, byte-exact prefixes of the real files: every
  ``.eig`` line of the first k-point, the ``.amn``/``.mmn`` header and first lines;
* ``stage.json``: the stage's own E-fermi and magnetisation, read from its OUTCAR;
* ``parent/{INCAR,POSCAR,KPOINTS}``: the arm's parent inputs (the NSCF step the overlap
  step continues from), and ``mno.win``: the deck the runner staged as wannier90.win.

No POTCAR or OUTCAR text is copied (the OUTCAR echoes the licensed POTCAR header).

Usage: python build_fixture.py <SCRATCH>   (writes next to this file)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
STAGES = {
    "mno_ispin2": "spin/clean/construction/mno.vasp2wannier90",
    "mno_ispin1": "spin/degraded/construction/mno.vasp2wannier90",
}
PARENT_INPUTS = ("INCAR", "POSCAR", "KPOINTS")
RUNNER_WRITTEN = {"INCAR", "wannier90.win"}
PRODUCT = re.compile(r"^wannier90(\.[12]|\.up|\.dn)?\.(amn|mmn|eig)$")
AMN_MMN_LINES = 6  # comment line, the counts line, then the first entries


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(scratch: Path) -> None:
    provenance = {}
    for name, rel in STAGES.items():
        stage = scratch / rel
        out = HERE / name
        out.mkdir(exist_ok=True)
        rows = []
        for entry in sorted(stage.iterdir(), key=lambda p: p.name):
            if entry.is_symlink():
                kind, size = "parent-link", 0
            else:
                kind = "runner" if entry.name in RUNNER_WRITTEN else "vasp"
                size = entry.stat().st_size
            rows.append(f"{entry.name}\t{size}\t{kind}")
        (out / "listing.tsv").write_text("name\tsize\tkind\n" + "\n".join(rows) + "\n")
        products = {}
        for entry in sorted(stage.iterdir(), key=lambda p: p.name):
            if not PRODUCT.match(entry.name):
                continue
            lines = entry.read_text().splitlines(keepends=True)
            if entry.name.endswith(".eig"):
                kept = [ln for ln in lines if int(ln.split()[1]) == 1]
            else:
                kept = lines[:AMN_MMN_LINES]
            (out / entry.name).write_text("".join(kept))
            products[entry.name] = {
                "source_sha256": sha256(entry),
                "source_bytes": entry.stat().st_size,
                "kept_lines": len(kept),
            }
        outcar = (stage / "OUTCAR").read_text(errors="replace")
        fermi = re.findall(r"E-fermi\s*:\s*([-\d.]+)", outcar)
        ispin = re.search(r"ISPIN\s*=\s*(\d)", outcar)
        moment = re.findall(
            r"number of electron\s+[-\d.]+\s+magnetization\s+([-\d.]+)", outcar
        )
        stage_json = {
            "source": str(stage),
            "ispin": int(ispin.group(1)) if ispin else None,
            "fermi_energy_ev": float(fermi[-1]) if fermi else None,
            "magnetisation_bohr": float(moment[-1]) if moment else None,
            "products": products,
        }
        (out / "stage.json").write_text(json.dumps(stage_json, indent=2) + "\n")
        arm = stage.parent.parent
        (out / "parent").mkdir(exist_ok=True)
        for name_ in PARENT_INPUTS:
            (out / "parent" / name_).write_bytes(
                (arm / "parent" / "nscf" / name_).read_bytes()
            )
        (out / "mno.win").write_bytes((stage.parent / "mno.win").read_bytes())
        provenance[name] = stage_json
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
