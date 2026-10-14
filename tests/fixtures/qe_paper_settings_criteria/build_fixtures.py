"""Write the compact regression fixtures from the stored QE paper-settings baseline.

Run once on cluster2 (where the run tree lives); the fixtures are committed so the
regression test runs anywhere::

    autowan/.venv/bin/python autowan/tests/fixtures/qe_paper_settings_criteria/build_fixtures.py

Each ``<material>.npz`` holds the construction's real-space Hamiltonian exactly as
Wannier90 wrote it (``aiida_hr.dat``: R, degeneracies, H) and its minimal-image shifts
(``aiida_wsvec.dat``), the pw.x bands on Wannier90's path (``dft_bands_path.npz``), the
Wannier90 band path itself (``aiida_band.dat``), the deck's exclusion, the scf and nscf
Fermi levels, and the eta table the run stored from ``aiida_wannier90_workflows``'
``bands_distance`` (mu = E_F ... E_F + 5 eV, columns mu, eta, max sqrt, max abs; eV).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from autowan.band_distance_eta import read_wannier_path_bands  # noqa: E402
from autowan.criteria import qe_output_fermi_energy_ev  # noqa: E402
from autowan.inputs import read_excluded_bands  # noqa: E402
from autowan.w90_outputs import read_run  # noqa: E402

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


BASELINE = _stored("AUTOWAN_TEST_BASELINE")
MATERIALS = ("Na", "Si", "GaAs")


def build(material: str) -> Path:
    raw = BASELINE / "results" / "raw" / material / "A_external_fixed"
    run = read_run(raw / "aiida.wout")
    with np.load(raw / "dft_bands_path.npz") as archive:
        dft, kpoints = archive["bands"], archive["kpoints"]
    prov = json.loads(
        (
            BASELINE / "results" / material / "A_external_fixed" / "provenance.json"
        ).read_text()
    )
    eta = prov["run"]["eta"]["nospin"]
    out = HERE / f"{material}.npz"
    np.savez_compressed(
        out,
        hr_matrix=run.hr_matrix,
        hr_rvectors=run.hr_rvectors,
        hr_degeneracies=run.hr_degeneracies,
        ws_counts=run.hr_ws_shifts.counts,
        ws_shifts=run.hr_ws_shifts.shifts.astype(np.int16),
        lattice_ang=run.lattice_ang,
        dft_bands=dft,
        kpoints=kpoints,
        w90_bands=read_wannier_path_bands(raw / "aiida_band.dat"),
        exclude_bands=np.asarray(read_excluded_bands(raw / "aiida.win"), dtype=int),
        fermi_scf_ev=qe_output_fermi_energy_ev(raw / "scf.out"),
        fermi_nscf_ev=qe_output_fermi_energy_ev(raw / "nscf.out"),
        stored_eta_table_ev=np.asarray(eta["table_mu_eta_maxsqrt_maxabs_eV"]),
        stored_fermi_ev=eta["fermi_energy_eV"],
    )
    return out


if __name__ == "__main__":
    for m in MATERIALS:
        print(build(m), build(m).stat().st_size)
