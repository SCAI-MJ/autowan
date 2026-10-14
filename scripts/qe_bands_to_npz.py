#!/usr/bin/env python3
"""Turn a Quantum ESPRESSO ``bands`` run into the reference file the criteria read.

The criterion profiles (``band_interpolation``, ``sparse_tb``, ``dmft_subspace``) compare
the Wannier model with DFT bands on a high-symmetry path that is not the construction
mesh. Run ``pw.x`` with ``calculation = 'bands'`` on the path Wannier90 itself uses
(the k-points in ``<seed>_band.kpt`` after a run with ``bands_plot = .true.``, or the
``explicit_kpath`` block of the deck), then:

    python scripts/qe_bands_to_npz.py path/to/bands/prefix.save/data-file-schema.xml \
        --scf-out path/to/scf.out --out reference/dft_bands_path.npz

The archive holds ``bands`` (eV; ``(n_k, n_bands)``, or ``(2, n_k, n_bands)`` for a
collinear magnet) and ``kpoints`` (fractional, in the parent's reciprocal basis). The
scf output is copied beside it as ``scf.out``, which is where the Fermi level is read.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autowan.qe_outputs import read_parent  # noqa: E402


def fractional_kpoints(parent) -> np.ndarray:
    """QE writes k in Cartesian units of 2*pi/alat; k_frac_i = k_cart . a_i / alat."""
    return parent.kpoints_cart @ parent.lattice_ang.T / parent.alat_ang


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bands_xml", type=Path, help="data-file-schema.xml of the bands run")
    ap.add_argument("--scf-out", type=Path, required=True, help="the scf run's pw.x output")
    ap.add_argument("--out", type=Path, default=Path("reference/dft_bands_path.npz"))
    args = ap.parse_args()

    parent = read_parent(args.bands_xml)
    if parent.calculation != "bands":
        print(
            f"warning: {args.bands_xml} records calculation = '{parent.calculation}', "
            "not 'bands'",
            file=sys.stderr,
        )
    bands = parent.eigenvalues_ev
    bands = bands[0] if bands.shape[0] == 1 else bands
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, bands=bands, kpoints=fractional_kpoints(parent))
    shutil.copyfile(args.scf_out, args.out.with_name("scf.out"))
    print(
        f"wrote {args.out}: {bands.shape[-2]} k-points x {bands.shape[-1]} bands, "
        f"with {args.out.with_name('scf.out')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
