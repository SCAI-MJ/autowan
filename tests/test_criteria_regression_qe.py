"""The acceptance criteria reproduce ``aiida_wannier90_workflows``' band distance on stored QE runs.

Fixtures: three constructions of the QE paper-settings frontier baseline (Na: a metal with
semicore bands 1-4 excluded; Si: an insulator whose minimal-image shifts move the path
bands by up to 18 meV; GaAs: an insulator with bands 1-5 excluded and elements split over
up to six images), written by ``fixtures/qe_paper_settings_criteria/build_fixtures.py``.
Each carries the eta table that run stored from the authors' ``bands_distance`` (mu = E_F
... E_F + 5 eV).

Two things are pinned. First, :func:`band_distance_eta` on Wannier90's own band path is
``bands_distance_raw``, to rounding, at every stored mu: index matching after
``exclude_bands`` and sqrt(f_DFT f_Wan) weights. Second, the criterion check, which
interpolates the construction's H(R) onto the path itself, gives the stored eta_2 to
0.01 meV.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan.band_distance_eta import band_distance_eta
from autowan.criteria import (
    BAND_INTERPOLATION_ETA2,
    CONDUCTION_BAND_ETA,
    ReferenceBandPath,
    real_space_model,
)
from autowan.gate import CertificationGate, Construction
from autowan.w90_outputs import WsShifts

FIXTURES = Path(__file__).parent / "fixtures" / "qe_paper_settings_criteria"
MATERIALS = ("Na", "Si", "GaAs")


def _load(material):
    with np.load(FIXTURES / f"{material}.npz") as f:
        return {k: f[k] for k in f.files}


def _wannier(d, with_shifts=True):
    return SimpleNamespace(
        hr_matrix=d["hr_matrix"],
        hr_rvectors=d["hr_rvectors"],
        hr_degeneracies=d["hr_degeneracies"],
        hr_ws_shifts=(
            WsShifts(d["ws_counts"], d["ws_shifts"].astype(np.int64))
            if with_shifts
            else None
        ),
    )


def _construction(d):
    ef = float(d["fermi_scf_ev"])
    parent = SimpleNamespace(
        # the path stands in for the mesh: these fixtures carry no nscf eigenvalues
        eigenvalues_ev=d["dft_bands"][None],
        fermi_energy_ev=float(d["fermi_nscf_ev"]),
        lattice_ang=d["lattice_ang"],
    )
    inputs = {"spin_index": 0, "mp_grid": (11, 11, 11)}
    if d["exclude_bands"].size:
        inputs["exclude_bands"] = tuple(int(i) for i in d["exclude_bands"])
    return Construction(
        "fixture",
        parent=parent,
        wannier=_wannier(d),
        material_class="gapped",
        inputs=inputs,
        reference_path=ReferenceBandPath(d["kpoints"], d["dft_bands"], ef, None),
    )


@pytest.mark.parametrize("material", MATERIALS)
def test_eta_on_wannier90s_band_path_is_the_authors_bands_distance(material):
    d = _load(material)
    ef = float(d["stored_fermi_ev"])
    assert ef == pytest.approx(float(d["fermi_scf_ev"]), abs=1e-9)
    exclude = tuple(int(i) for i in d["exclude_bands"])
    for mu, eta, _max_sqrt, max_abs in d["stored_eta_table_ev"]:
        result = band_distance_eta(
            d["dft_bands"], d["w90_bands"], ef, nu_ev=mu - ef, exclude_bands=exclude
        )
        assert result.eta_mev == pytest.approx(eta * 1e3, rel=1e-9, abs=1e-9)
        assert result.eta_max_mev == pytest.approx(max_abs * 1e3, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("material", MATERIALS)
def test_the_hr_model_reproduces_wannier90s_band_path(material):
    d = _load(material)
    model = real_space_model(_wannier(d), d["lattice_ang"])
    assert model.ws_distance_applied
    difference = np.abs(model.bands(d["kpoints"]) - d["w90_bands"]).max()
    assert (
        difference < 0.5e-3
    )  # eV; Wannier90 prints its band path to 1e-6 eV after rounding H


def test_without_the_minimal_image_shifts_silicon_is_off_by_more_than_ten_mev():
    d = _load("Si")
    plain = real_space_model(_wannier(d, with_shifts=False), d["lattice_ang"])
    assert np.abs(plain.bands(d["kpoints"]) - d["w90_bands"]).max() > 10e-3


@pytest.mark.parametrize("material", MATERIALS)
def test_the_band_interpolation_criterion_gives_the_stored_eta2(material):
    d = _load(material)
    ef = float(d["fermi_scf_ev"])
    stored = d["stored_eta_table_ev"]
    eta2 = stored[np.argmin(np.abs(stored[:, 0] - (ef + 2.0))), 1] * 1e3
    report = CertificationGate([BAND_INTERPOLATION_ETA2], profile_name="t").run(
        _construction(d)
    )
    v = report.verdicts[0]
    assert v.metric_value == pytest.approx(eta2, abs=0.01)
    assert v.passed is bool(eta2 <= 20.0)
    assert v.record["ws_distance_applied"]


@pytest.mark.parametrize("material", ("Si", "GaAs"))
def test_the_conduction_criterion_is_bands_distance_raw_at_cbm_plus_two(material):
    d = _load(material)
    c = _construction(d)
    v = CertificationGate([CONDUCTION_BAND_ETA], profile_name="t").run(c).verdicts[0]
    assert v.applies and v.record["is_insulator"]
    cbm = v.record["cbm_ev"]
    exclude = tuple(int(i) for i in d["exclude_bands"])
    on_w90 = band_distance_eta(
        d["dft_bands"], d["w90_bands"], cbm, exclude_bands=exclude
    ).eta_mev
    assert v.metric_value == pytest.approx(on_w90, abs=0.01)
