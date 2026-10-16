"""The ``jarvis_frontier`` recipe: JARVIS-tools' Wannierisation as the VASP frontier.

The VASP frontier since 2026-10-08 (an SCDM deck). The deck is the text JARVIS's own writer
produced (``Wannier90win.write_win`` + ``write_hr_win``, usnistgov/jarvis 3b0c9d0),
kept line for line; the recipe adds what VASP's interface would append (cell, atoms,
k-points, ``mp_grid``, ``spinors``) and, where the parent WAVECAR holds more bands than
JARVIS's NBANDS, the top exclusion that makes the overlaps carry exactly JARVIS's bands.
The overlap step is JARVIS's Wannier-step INCAR (``vasp.py`` 788-804): ``ALGO = None``,
``NELM = 1``, ``LWANNIER90``, ``LWRITE_MMN_AMN``, ``ISMEAR = 0``, ``SIGMA = 0.01``, with the
.win handed to VASP -- not the SCDM frontier's ``WANNIER90_WIN`` route.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.inputs import (
    DECK_RECIPES,
    JARVIS_FRONTIER_RECIPE,
    deck_from_parent,
    is_frontier_overlap_step,
    is_jarvis_overlap_step,
    jarvis_frontier_deck,
    overlap_step_incar_tags,
    read_incar,
    write_derived,
)
from autowan.execution import CalculationError, StepResult, check_band_count
from autowan.families import Method
from autowan.inputs import InputError, assignments, deck_method, read_excluded_bands

SRO = Path(__file__).parent / "fixtures" / "vasp" / "SrO"

#: JARVIS's hr_wannier.win for the SrO fixture, as the pinned writer (with the
#: soc=False exclude fix) emits it: Sr_sv [8, 2, s,d, 6] + O [2, 4, p, 3] -> 9
#: functions, 10 excluded electrons = 5 bands per channel; E_F 2.0179 eV; NBANDS 24.
JARVIS_SRO = (
    "write_hr=.true. \n"
    "num_bands = 19\n"
    "guiding_centres = TRUE \n"
    "kmesh_tol=1e-09\n"
    "num_wann =9\n"
    "exclude_bands     = 1-5\n"
    "dis_froz_max =4.0179\n"
    "dis_froz_min =-1000\n"
    "dis_num_iter =100\n"
    "dis_mix_ratio =0.5\n"
    "num_iter =100\n"
    "num_print_cycles =50\n"
    "begin projections\n"
    "Sr:s,d\n"
    "O:p\n"
    "end projections\n"
)


@pytest.fixture
def sro(tmp_path):
    parent = tmp_path / "parent"
    shutil.copytree(SRO, parent)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return parent, run_dir


def test_the_recipe_is_registered_and_deck_from_parent_points_to_the_wrapper(sro):
    parent, run_dir = sro
    assert JARVIS_FRONTIER_RECIPE == "jarvis_frontier"
    assert JARVIS_FRONTIER_RECIPE in DECK_RECIPES
    with pytest.raises(InputError, match="jarvis_frontier_deck"):
        deck_from_parent(
            parent / "vasprun.xml", run_dir, "SrO", recipe=JARVIS_FRONTIER_RECIPE
        )


def test_the_deck_keeps_every_jarvis_line_and_excludes_bands_above_jarvis_nbands(sro):
    parent, run_dir = sro
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
    )
    text = inputs.win.read_text()
    lines = text.splitlines()
    for line in JARVIS_SRO.splitlines():
        if line.startswith("exclude_bands"):
            assert f"! JARVIS writer: {line.strip()}" in lines
        else:
            assert line.rstrip() in lines, line
    values = assignments(text)
    # JARVIS NBANDS 24 = num_bands 19 + 5 excluded; the parent WAVECAR holds 32.
    assert read_excluded_bands(inputs.win) == (*range(1, 6), *range(25, 33))
    assert values["num_bands"] == "19" and values["num_wann"] == "9"
    assert values["dis_froz_min"] == "-1000" and values["dis_froz_max"] == "4.0179"
    assert values["mp_grid"] and "spinors" not in values
    check_band_count(inputs.win, 32)
    assert deck_method(text) is Method.SMV
    assert is_jarvis_overlap_step(text) and not is_frontier_overlap_step(text)


def test_the_overlap_step_tags_are_jarvis_wannier_step_incar(sro):
    parent, run_dir = sro
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
    )
    tags = overlap_step_incar_tags(inputs.win.read_text())
    assert tags == {
        "PREC": "Accurate",
        "ALGO": "None",
        "ISMEAR": "0",
        "NSW": "0",
        "NELM": "1",
        "SIGMA": "0.01",
        "LWANNIER90": ".TRUE.",
        "LWRITE_MMN_AMN": ".TRUE.",
        "NEDOS": "5000",
        "LORBIT": "11",
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
        "ISTART": "1",
        "NBANDS": "32",
    }


def test_jarvis_nbands_equal_to_the_parents_keeps_the_writer_exclusion_alone(sro):
    parent, run_dir = sro
    text = JARVIS_SRO.replace("num_bands = 19", "num_bands = 27")
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=text
    )
    assert read_excluded_bands(inputs.win) == tuple(range(1, 6))
    check_band_count(inputs.win, 32)


def test_a_writer_without_exclusion_still_gets_the_top_block(sro):
    parent, run_dir = sro
    text = JARVIS_SRO.replace("exclude_bands     = 1-5\n", "")
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=text
    )
    assert read_excluded_bands(inputs.win) == tuple(range(20, 33))
    check_band_count(inputs.win, 32)


def test_more_bands_than_the_parent_wavecar_holds_is_refused(sro):
    parent, run_dir = sro
    text = JARVIS_SRO.replace("num_bands = 19", "num_bands = 30")
    with pytest.raises(InputError, match="exceeds the 32 bands"):
        jarvis_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", jarvis_win=text)


def test_an_exclusion_that_is_not_the_bottom_block_is_refused(sro):
    parent, run_dir = sro
    text = JARVIS_SRO.replace("1-5", "2-6")
    with pytest.raises(InputError, match="not 1-N"):
        jarvis_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", jarvis_win=text)


def _stub_parent(parent: Path, wavecar: bool) -> None:
    for name in ("CHGCAR", "POTCAR") + (("WAVECAR",) if wavecar else ()):
        (parent / name).write_text("stub\n")


def test_the_runner_reads_the_wavecar_with_jarvis_incar_and_hands_vasp_the_win(
    sro, monkeypatch
):
    parent, run_dir = sro
    _stub_parent(parent, wavecar=True)
    jarvis_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO)
    runner = VaspWannierRunner(run_dir=run_dir, parent_dir=parent, vasp="vasp_std")
    seen = {}

    def fake_run(step, command, *, cwd=None, **_):
        seen["incar"] = (cwd / "INCAR").read_text()
        seen["win"] = (cwd / "wannier90.win").read_text()
        return StepResult(step, 0, 0.0)

    monkeypatch.setattr(runner, "_run", fake_run)
    runner.run_overlaps("SrO")
    tags = read_incar(seen["incar"])
    assert "WANNIER90_WIN" not in seen["incar"]
    assert tags["ALGO"] == "None" and tags["NELM"] == "1" and tags["ISTART"] == "1"
    assert tags["LWANNIER90"] == ".TRUE." and tags["LWRITE_MMN_AMN"] == ".TRUE."
    assert tags["ISMEAR"] == "0" and tags["SIGMA"] == "0.01"
    assert tags["NBANDS"] == "32" and tags["NUM_WANN"] == "9"
    assert tags["ICHARG"] == "11"
    win = assignments(seen["win"])
    assert win["num_wann"] == "9" and win["num_bands"] == "19"
    assert win["exclude_bands"].replace(" ", "") == "1-5,25-32"
    assert "begin projections" in seen["win"]


def test_the_runner_refuses_a_jarvis_deck_without_the_parent_wavecar(sro):
    parent, run_dir = sro
    _stub_parent(parent, wavecar=False)
    jarvis_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO)
    runner = VaspWannierRunner(run_dir=run_dir, parent_dir=parent)
    with pytest.raises(CalculationError, match="WAVECAR"):
        runner.run_overlaps("SrO")


def test_a_repair_of_the_jarvis_deck_keeps_its_overlap_step_and_moves_the_top_block(
    sro,
):
    parent, run_dir = sro
    source = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
    )
    derived = write_derived(source, run_dir, "SrO_r1", {"num_bands": 23})
    text = derived.win.read_text()
    assert is_jarvis_overlap_step(text)
    assert overlap_step_incar_tags(text)["NBANDS"] == "32"
    assert read_excluded_bands(derived.win) == (*range(1, 6), *range(29, 33))
    check_band_count(derived.win, 32)


def test_the_scdm_frontier_is_still_recognised(sro, monkeypatch):
    parent, run_dir = sro
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    inputs = deck_from_parent(parent / "vasprun.xml", run_dir, "SrO")
    text = inputs.win.read_text()
    assert is_frontier_overlap_step(text) and not is_jarvis_overlap_step(text)


def test_the_kpoints_are_written_at_full_precision_for_jarvis_kmesh_tol(sro):
    # kmesh_tol = 1e-9: vasprun.xml's 8-decimal k-points (1/9 -> 0.11111111) leave
    # Wannier90 without neighbour shells; the deck writes the exact mesh points.
    import numpy as np

    from autowan.inputs import read_win

    parent, run_dir = sro
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
    )
    text = inputs.win.read_text()
    block = text.split("begin kpoints")[1].split("end kpoints")[0].split("\n")
    kpts = np.array([[float(x) for x in ln.split()] for ln in block if ln.strip()])
    mesh = np.array([int(x) for x in read_win(inputs.win)["mp_grid"].split()])
    assert np.allclose(kpts * mesh, np.round(kpts * mesh), atol=1e-11)


def test_the_cell_is_the_poscars_at_full_precision(sro):
    # vasprun.xml prints 8 decimals; the deck writes the POSCAR's numbers (the cell VASP
    # used), so kmesh_tol = 1e-9 does not split a degenerate shell.
    import numpy as np

    parent, run_dir = sro
    poscar = (parent / "POSCAR").read_text().splitlines()
    vec = [float(x) for x in poscar[2].split()]
    poscar[2] = " ".join(f"{x + (2e-9 if x else 0.0):.15f}" for x in vec)
    (parent / "POSCAR").write_text("\n".join(poscar) + "\n")
    inputs = jarvis_frontier_deck(
        parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
    )
    cell = inputs.win.read_text().split("begin unit_cell_cart")[1].split("\n")[2]
    assert np.allclose([float(x) for x in cell.split()], [v + (2e-9 if v else 0) for v in vec],
                       atol=1e-12)  # fmt: skip
    poscar[2] = " ".join(f"{x + 0.01:.15f}" for x in vec)
    (parent / "POSCAR").write_text("\n".join(poscar) + "\n")
    with pytest.raises(InputError, match="does not hold the cell"):
        jarvis_frontier_deck(
            parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO
        )


def test_a_poscar_position_vasp_wrapped_into_the_cell_is_the_same_atom(sro):
    # graphene: POSCAR -0.000000666667 is 0.99999933 in vasprun.xml.
    parent, run_dir = sro
    poscar = (parent / "POSCAR").read_text().splitlines()
    first = poscar.index("Direct") + 1
    x = [float(v) for v in poscar[first].split()[:3]]
    poscar[first] = " ".join(f"{v - 1.0 if v > 0.5 else v:.15f}" for v in x)
    if poscar[first] == " ".join(f"{v:.15f}" for v in x):
        poscar[first] = " ".join(f"{v - 1.0:.15f}" for v in x)
    (parent / "POSCAR").write_text("\n".join(poscar) + "\n")
    jarvis_frontier_deck(parent / "vasprun.xml", run_dir, "SrO", jarvis_win=JARVIS_SRO)
