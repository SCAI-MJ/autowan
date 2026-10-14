"""Score the Quantum ESPRESSO construction of a fixture material with the same gate object.

The VASP backend's claim is that the gate does not change with the engine. This script
puts the QE side of that comparison on disk: it reads the campaign's finished QE
construction (the ``.wout`` / ``_hr.dat`` Wannier90 left and the ``data-file-schema.xml``
of the parent), builds ``gate_for(get_backend("qe"))``, scores it, and writes the verdict
rows in the same schema the VASP smoke writes. No engine and no Wannier90 run: this is
post-processing of files already on disk.

The provenance carries the k-mesh the construction actually ran on -- the ``.win``'s
``mp_grid``, refused unless the ``.amn`` pw2wannier90 wrote holds exactly that many
k-points -- so the "same mesh" of a VASP-vs-QE comparison is a measurement on both sides,
and the reason this construction (of the several a campaign may leave) is the one scored.

    .venv/bin/python tests/fixtures/vasp/score_qe_reference.py \
        --run-dir $AUTOWAN_WORK/campaign/Si --seed Si_pdwf \
        --parent-xml $AUTOWAN_WORK/campaign/Si/Si.save/data-file-schema.xml \
        --material Si --out data/results/vasp_backend_smoke/Si/qe_reference/gate_verdicts.json \
        --choice "the campaign's automatic first deck (attempt 0)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autowan.backends import gate_for, get_backend
from autowan.gate import Construction
from autowan.verdicts import write_gate_verdicts
from autowan.w90_outputs import read_run


def _measured_mesh(run_dir: Path, seed: str, declared) -> dict:
    """The mesh the construction ran on: the .win's mp_grid, checked against the .amn."""
    if not declared:
        raise SystemExit(f"{seed}.win declares no mp_grid: the mesh cannot be stated")
    mesh = [int(x) for x in str(declared).split()]
    if len(mesh) != 3:
        raise SystemExit(f"{seed}.win mp_grid {declared!r} is not three integers")
    header = None
    amn = run_dir / f"{seed}.amn"
    if amn.is_file():
        with amn.open() as handle:
            handle.readline()
            header = [int(x) for x in handle.readline().split()[:3]]
        if header[1] != mesh[0] * mesh[1] * mesh[2]:
            raise SystemExit(
                f"{amn} holds {header[1]} k-points but {seed}.win declares mp_grid "
                f"{declared}: the construction did not run on the mesh it names"
            )
    return {
        "mp_grid": " ".join(map(str, mesh)),
        "mp_grid_source": (
            f"{seed}.win mp_grid, k-count cross-checked against the .amn header"
            if header
            else f"{seed}.win mp_grid (no .amn beside it to cross-check)"
        ),
        "amn_header_bands_kpoints_functions": header,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--parent-xml", type=Path, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument(
        "--choice",
        default="",
        help="why this construction is the one scored (recorded in provenance.json)",
    )
    parser.add_argument(
        "--role",
        default="reference",
        help="'reference' (the contract's comparison) or 'sensitivity' (a secondary row)",
    )
    args = parser.parse_args(argv)

    backend = get_backend("qe")
    parent = backend.read_parent(args.parent_xml)
    inputs = backend.read_construction(args.run_dir, args.seed)
    wannier = read_run(args.run_dir / f"{args.seed}.wout")
    mesh = _measured_mesh(args.run_dir, args.seed, inputs.values.get("mp_grid"))
    gate = gate_for(backend)
    construction = Construction(
        args.seed,
        parent=parent,
        wannier=wannier,
        material_class="metal" if parent.is_metallic() else "gapped",
        inputs={**inputs.gate_inputs(args.spin_index), "route": backend.route(inputs)},
    )
    report = gate.run(construction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    values = inputs.values
    path = write_gate_verdicts(
        args.out,
        report,
        gate,
        material=args.material,
        backend=backend.name,
        seed=args.seed,
        extra_provenance={
            "parent_xml": str(args.parent_xml),
            "run_dir": str(args.run_dir),
            **mesh,
            "role": args.role,
            "choice": args.choice,
            "num_wann": values.get("num_wann"),
            "num_bands": values.get("num_bands"),
            "dis_froz_max": values.get("dis_froz_max"),
            "projections": values.get("projectors") or values.get("projection_source"),
            "fermi_energy_ev": parent.fermi_energy_ev,
            "note": "post-processing of the campaign's finished QE construction; no engine run",
        },
    )
    print(
        f"{args.material} ({backend.name}, {args.seed}): passed={report.passed} -> {path}"
    )
    print(f"  mp_grid {mesh['mp_grid']} ({mesh['mp_grid_source']})")
    for verdict in report.verdicts:
        print(
            f"  {verdict.check_id:<46} applies={verdict.applies!s:<5} evaluable={verdict.evaluable!s:<5} "
            f"passed={verdict.passed!s:<5} value={verdict.metric_value} threshold={verdict.threshold_value}"
        )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
