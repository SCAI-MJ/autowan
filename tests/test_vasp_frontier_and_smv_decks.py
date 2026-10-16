"""The two VASP starting decks of 2026-10-07, and the package default that changed.

* SCDM frontier, exactly the Varrassi et al. (2025)
  Wannier step: ALGO = None, NELM = 1, ISTART = 1, LWANNIER90_RUN, LSCDM, ISMEAR = 0,
  SIGMA = 0.02, LORBIT = 11, NUM_WANN and CUTOFF_* unset, a .win of four keys and no
  windows; NBANDS explicit at VASP's default formula with the rounding recorded.
* SMV, semicore excluded: the bottom N bands, N from our POTCARs, checked against band
  character; one hydrogenic function per valence channel; energy window only.
* ``deck_from_parent`` no longer defaults to the hydrogenic projectability deck VASP
  cannot run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.inputs import (
    DEFAULT_RECIPE,
    SCDM_FRONTIER_RECIPE,
    deck_from_parent,
    overlap_step_incar_tags,
    read_incar,
    scdm_frontier_deck,
    semicore_assignment,
    semicore_excluded_deck,
    vasp_default_nbands,
)
from autowan.backends.vasp.outputs import read_parent
from autowan.families import Method
from autowan.inputs import InputError, assignments, deck_method, read_excluded_bands

SRO = Path(__file__).parent / "fixtures" / "vasp" / "SrO"


@pytest.fixture
def sro(tmp_path):
    parent = tmp_path / "parent"
    shutil.copytree(SRO, parent)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return parent, run_dir


def test_vasps_default_nbands_formula_and_its_npar_rounding():
    # SrO: NELECT 16, 2 ions: max(NINT(18)/2 + max(1, 3), INT(9.6)) = 12
    assert vasp_default_nbands(16, 2) == (12, 12)
    assert vasp_default_nbands(16, 2, npar=8) == (12, 16)
    # MnO-like collinear magnet: max(NINT(23)/2 + 3, INT(12.6)) = 14, NMAG = (5 + 1)/2 = 3
    assert vasp_default_nbands(21, 2, magmom_sum=5.0, ispin=2) == (17, 17)
    assert vasp_default_nbands(21, 2, magmom_sum=5.0, ispin=1)[0] == 14


def test_the_package_default_is_the_scdm_frontier():
    assert DEFAULT_RECIPE == SCDM_FRONTIER_RECIPE


def test_the_scdm_frontier_deck_is_the_protocol_and_nothing_else(sro, monkeypatch):
    parent, run_dir = sro
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    inputs = deck_from_parent(parent / "vasprun.xml", run_dir, "SrO")
    text = inputs.win.read_text()
    tags = overlap_step_incar_tags(text)
    assert tags == {
        "ALGO": "None",
        "NELM": "1",
        "ISTART": "1",
        "LWANNIER90_RUN": ".TRUE.",
        "LSCDM": ".TRUE.",
        "ISMEAR": "0",
        "SIGMA": "0.02",
        "LORBIT": "11",
        "LWRITE_MMN_AMN": ".TRUE.",
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
        "NBANDS": "12",
        "NPAR": "1",
    }
    assert not any(k.startswith("CUTOFF") or k == "NUM_WANN" for k in tags)
    values = assignments(text)
    assert values["num_wann"] == "12"  # min(NBANDS 12, n_pseudo 13)
    assert values["num_bands"] == "12"
    assert values["exclude_bands"] == "13-32"
    for key, value in (
        ("dis_num_iter", "500"),
        ("dis_conv_tol", "1.0e-8"),
        ("num_iter", "1000"),
        ("conv_tol", "1.0e-4"),
    ):
        assert values[key] == value
    assert not any(k.startswith(("dis_win", "dis_froz", "dis_proj")) for k in values)
    assert "begin projections" not in text
    assert (
        "VASP default formula gives 12, rounded to a multiple of NPAR = 1 -> 12" in text
    )
    assert deck_method(text) is Method.SCDM


def test_the_parents_band_count_is_the_sensitivity_reading(sro, monkeypatch):
    parent, run_dir = sro
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    inputs = scdm_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", nbands=32)
    values = assignments(inputs.win.read_text())
    assert values["num_bands"] == "32" and values["num_wann"] == "13"
    assert "exclude_bands" not in values
    with pytest.raises(InputError, match="exceeds the 32 bands"):
        scdm_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", nbands=40)


def test_the_frontier_overlap_step_reads_the_wavecar_and_lets_vasp_choose_num_wann(
    sro, monkeypatch
):
    parent, run_dir = sro
    for name in ("CHGCAR", "POTCAR", "WAVECAR"):
        (parent / name).write_text("stub\n")
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    scdm_frontier_deck(parent / "vasprun.xml", run_dir, "SrO")
    runner = VaspWannierRunner(run_dir=run_dir, parent_dir=parent, vasp="vasp_std")
    seen = {}

    def fake_run(step, command, *, cwd=None, **_):
        seen["incar"] = (cwd / "INCAR").read_text()
        seen["staged_win"] = (cwd / "wannier90.win").exists()
        from autowan.execution import StepResult

        return StepResult(step, 0, 0.0)

    monkeypatch.setattr(runner, "_run", fake_run)
    runner.run_overlaps("SrO")
    incar = seen["incar"]
    tags = read_incar(incar.split("WANNIER90_WIN")[0])
    assert tags["ALGO"] == "None" and tags["NELM"] == "1" and tags["ISTART"] == "1"
    assert tags["LSCDM"] == ".TRUE." and tags["NBANDS"] == "12"
    for absent in ("ICHARG", "NUM_WANN", "LWANNIER90"):
        assert absent not in tags, absent
    assert 'WANNIER90_WIN = "dis_num_iter = 500\ndis_conv_tol = 1.0e-8' in incar
    assert not seen["staged_win"], "VASP writes its own wannier90.win in frontier mode"


def test_the_frontier_refuses_to_run_without_the_parent_wavecar(sro, monkeypatch):
    from autowan.execution import CalculationError

    parent, run_dir = sro
    for name in ("CHGCAR", "POTCAR"):
        (parent / name).write_text("stub\n")
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    scdm_frontier_deck(parent / "vasprun.xml", run_dir, "SrO")
    runner = VaspWannierRunner(run_dir=run_dir, parent_dir=parent)
    with pytest.raises(CalculationError, match="WAVECAR"):
        runner.run_overlaps("SrO")


def test_the_semicore_count_comes_from_our_potcars_and_is_checked_on_character(sro):
    parent, _ = sro
    vasprun = read_parent(parent / "vasprun.xml")
    assignment = semicore_assignment(vasprun, parent / "OUTCAR", parent / "PROCAR")
    # Sr_sv: ZVAL 10 - 2 valence = 8 (4s 4p); O: 6 - 4 = 2 (2s, below Sr 4p)
    assert assignment.excluded_bands == 5
    assert assignment.per_species["Sr"][:3] == (8, ("4s", "4p"), "sd")
    assert assignment.per_species["O"][:3] == (2, ("2s",), "p")
    assert assignment.verified_by.startswith("PROCAR channel character")


def test_the_smv_deck_excludes_semicore_and_names_one_function_per_valence_channel(sro):
    parent, run_dir = sro
    inputs = semicore_excluded_deck(parent / "vasprun.xml", run_dir, "SrO")
    text = inputs.win.read_text()
    assert deck_method(text) is Method.SMV
    assert inputs.values["projectors"] == "Sr:s;d;O:p"
    assert inputs.values["num_wann"] == "9"  # Sr s+d (6) + O p (3)
    excluded = read_excluded_bands(inputs.win)
    assert excluded[:5] == (1, 2, 3, 4, 5)
    num_bands = int(inputs.values["num_bands"])
    assert num_bands + len(excluded) == 32
    assert ":r=2" not in text and "dis_proj" not in text and "dis_froz_proj" not in text
    assert "semicore bands excluded: 5 (PROCAR channel character" in text


def test_a_semicore_count_contradicted_by_band_character_is_refused(sro, monkeypatch):
    parent, run_dir = sro
    # pretend O keeps no valence electrons beyond 2p: 2s and the bottom 2p band would go
    monkeypatch.setitem(vasp_inputs.VALENCE_TABLE, "Sr", (0, "d"))
    with pytest.raises(InputError):
        semicore_excluded_deck(parent / "vasprun.xml", run_dir, "SrO")
