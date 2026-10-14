"""The VASP overlap step: what the runner stages, what it collects, what it refuses.

``vasp2wannier90`` is VASP itself, run again from the parent's charge density with
``LWANNIER90`` set, reading a ``wannier90.win`` whose seedname it does not let the
caller choose. So the engine-specific half of the runner is a staging problem: put the
parent's files and this deck's ``.win`` where VASP will find them, run it, and bring
the overlaps back under the seedname the rest of the loop uses. Everything the
localisation half does -- reuse rules, the deck-versus-overlaps check, Wannier90 itself
-- is the shared base class and is not re-tested here.

A refusal is read from what VASP wrote, never from its exit status alone: VASP prints
its refusals in a box on standard output and can exit zero having done nothing.

The mechanics run against a stub executable. One test runs the real binary and is
skipped by name unless it is pointed at one on a compute node.
"""

from __future__ import annotations

import os
import shutil
import stat
import textwrap
from pathlib import Path

import pytest

from autowan.backends import get_backend
from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.inputs import (
    deck_from_parent,
    overlap_step_win,
    read_incar,
)
from autowan.execution import CalculationError, ConstructionRunner
from autowan.inputs import InputError, read_construction

FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
SYNTHETIC = FIXTURES / "synthetic"

WIN = """\
num_wann  = 8
num_bands = 4
num_iter  = 400
conv_window = 3
dis_froz_max = 7.2
dis_froz_proj = .true.
dis_proj_min = 0.01
dis_proj_max = 0.95
write_hr = .true.

begin projections
Si:s;p
end projections

begin unit_cell_cart
Ang
-2.71534056  0.00000000  2.71534056
 0.00000000  2.71534056  2.71534056
-2.71534056  2.71534056  0.00000000
end unit_cell_cart

begin atoms_frac
Si  0.00 0.00 0.00
Si  0.25 0.25 0.25
end atoms_frac

mp_grid 2 1 1

begin kpoints
 0.0 0.0 0.0
 0.5 0.0 0.0
end kpoints
"""

INCAR = """\
SYSTEM = Si synthetic parent
ENCUT = 250
EDIFF = 1E-7
ISMEAR = 0 ; SIGMA = 0.05
NBANDS = 4
ISYM = -1
LORBIT = 11
LCHARG = .TRUE.
"""

#: A stand-in for vasp_std. Behaviour is chosen through STUB_MODE; every launch is
#: appended to STUB_LOG so a test can say how many times the engine ran.
STUB_VASP = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, sys
    from pathlib import Path
    mode = os.environ.get("STUB_MODE", "ok")
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(f"{os.getcwd()} {mode}\\n")
    incar = Path("INCAR").read_text().upper()
    assert "LWANNIER90" in incar, "the overlap step must set LWANNIER90"
    assert "ICHARG" in incar, "the overlap step must continue from the parent density"
    for required in ("POSCAR", "POTCAR", "KPOINTS", "CHGCAR", "wannier90.win"):
        assert Path(required).is_file(), f"{required} was not staged"
    if mode == "refuse_interface":
        print("|     VASP was compiled without the VASP2WANNIER90 interface.     |")
        print("|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |")
        Path("OUTCAR").write_text(" vasp.6.4.1 stub\\n")
        sys.exit(0)
    if mode == "library_refusal":
        Path("OUTCAR").write_text(" vasp.6.4.1 stub\\n")
        Path("wannier90.werr").write_text(
            " Wannier90: Execution started\\n Exiting.......\\n"
            " kmesh_get: something wrong, found too many nearest neighbours\\n"
        )
        sys.exit(1)
    Path("OUTCAR").write_text(
        " vasp.6.4.1 stub\\n"
        "------------------------ aborting loop because EDIFF is reached "
        "----------------------------------------\\n"
        " E-fermi :   5.2000\\n"
    )
    if mode == "no_products":
        sys.exit(0)
    prefix = "wannier90.up" if mode == "spin" else "wannier90"
    win = Path("wannier90.win").read_text()
    Path(f"{prefix}.amn").write_text("stub amn\\n   4    2    8\\n")
    Path(f"{prefix}.mmn").write_text("stub mmn\\n   4    2    6\\n")
    Path(f"{prefix}.eig").write_text("stub eig\\n")
    sys.exit(0)
    """
)

#: A stand-in for wannier90.x: exits zero and writes nothing.
STUB_W90 = "#!/bin/sh\nexit 0\n"


def _executable(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def parent(tmp_path):
    """A parent directory as a finished SCF leaves it, minus the heavy binaries."""
    parent = tmp_path / "parent"
    parent.mkdir()
    shutil.copy(SYNTHETIC / "vasprun.xml", parent / "vasprun.xml")
    shutil.copy(SYNTHETIC / "OUTCAR.converged", parent / "OUTCAR")
    shutil.copy(SYNTHETIC / "PROCAR", parent / "PROCAR")
    (parent / "INCAR").write_text(INCAR)
    (parent / "POSCAR").write_text(
        "Si\n1.0\n-2.7 0 2.7\n0 2.7 2.7\n-2.7 2.7 0\nSi\n2\nDirect\n0 0 0\n.25 .25 .25\n"
    )
    (parent / "POTCAR").write_text("  PAW_PBE Si 05Jan2001\n")
    (parent / "KPOINTS").write_text("Gamma\n0\nGamma\n2 1 1\n0 0 0\n")
    (parent / "CHGCAR").write_text("stub density\n")
    return parent


@pytest.fixture
def run(tmp_path, parent, monkeypatch):
    """A run directory holding the deck, and a runner wired to the stubs."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "si.win").write_text(WIN)
    log = tmp_path / "launches.log"
    log.write_text("")
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_MODE", "ok")
    runner = VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent,
        vasp=_executable(tmp_path / "vasp_stub", STUB_VASP),
        wannier90=_executable(tmp_path / "w90_stub", STUB_W90),
        mpi_launcher=(),
    )
    return run_dir, runner, log


def _launches(log: Path) -> list[str]:
    return [ln for ln in log.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------


def test_the_runner_is_a_construction_runner_and_the_parent_defaults_to_the_run_dir(
    tmp_path,
):
    runner = VaspWannierRunner(run_dir=tmp_path)
    assert isinstance(runner, ConstructionRunner)
    assert runner.parent_dir == tmp_path
    assert runner.vasp == "vasp_std"


def test_the_overlap_step_stages_the_parent_and_the_deck_where_vasp_looks(run):
    run_dir, runner, log = run
    results = runner.run_overlaps("si")
    assert results[-1].ok, results[-1].stderr_tail
    stage = runner.stage_dir("si")
    assert stage.is_dir() and stage.parent == run_dir
    for staged in ("INCAR", "POSCAR", "POTCAR", "KPOINTS", "CHGCAR", "wannier90.win"):
        assert (stage / staged).is_file(), staged
    launches = _launches(log)
    assert len(launches) == 1
    where, mode = launches[0].rsplit(" ", 1)
    assert Path(where).resolve() == stage.resolve() and mode == "ok"
    # the report of the run stays beside it
    assert (stage / "vasp.log").is_file()


def test_the_staged_incar_continues_from_the_parent_density_with_the_interface_on(run):
    run_dir, runner, _ = run
    runner.run_overlaps("si")
    tags = read_incar((runner.stage_dir("si") / "INCAR").read_text())
    assert tags["LWANNIER90"].upper() in (".TRUE.", "T", "TRUE")
    assert tags["ICHARG"] == "11"
    assert tags["NSW"] == "0"
    assert tags["LCHARG"].upper() in (".FALSE.", "F", "FALSE")
    # VASP takes the Wannier count from the INCAR, not from the .win: without it the
    # count is estimated from the POTCARs and the projection arrays are misread
    assert tags["NUM_WANN"] == "8"
    # what the parent set and the overlap step must not change
    assert tags["NBANDS"] == "4"
    assert tags["ISYM"] == "-1"
    assert tags["ENCUT"] == "250"
    assert tags["SIGMA"] == "0.05"


def test_a_deck_without_a_wannier_count_cannot_be_staged(run):
    run_dir, runner, log = run
    (run_dir / "si.win").write_text(WIN.replace("num_wann  = 8\n", ""))
    with pytest.raises(CalculationError, match="num_wann"):
        runner.run_overlaps("si")
    assert _launches(log) == []


def test_an_empty_wavecar_does_not_switch_on_istart(run):
    """LWAVE = .FALSE. leaves a zero-byte WAVECAR behind; reading it is a crash."""
    run_dir, runner, _ = run
    (runner.parent_dir / "WAVECAR").write_text("")
    runner.run_overlaps("si")
    tags = read_incar((runner.stage_dir("si") / "INCAR").read_text())
    assert tags["ISTART"] == "0"


def test_read_incar_handles_comments_and_several_tags_on_one_line():
    tags = read_incar(
        "ISMEAR = 0 ; SIGMA = 0.05 ! smearing\n# a comment\nLWANNIER90 = .TRUE.\n"
    )
    assert tags == {"ISMEAR": "0", "SIGMA": "0.05", "LWANNIER90": ".TRUE."}


def test_the_deck_vasp_reads_carries_only_what_the_overlap_setup_needs(run):
    """The library VASP calls is Wannier90 3.1, which refuses keywords it does not
    know; the projectability keys and the localisation budget belong to the standalone
    step and must not reach it. Which keys survive is data in the knowledge partition.
    """
    run_dir, runner, _ = run
    runner.run_overlaps("si")
    staged = (runner.stage_dir("si") / "wannier90.win").read_text()
    for kept in (
        "num_wann",
        "num_bands",
        "Si:s;p",
        "begin kpoints",
        "0.5 0.0 0.0",
        "mp_grid",
        "begin unit_cell_cart",
        "begin atoms_frac",
    ):
        assert kept in staged, kept
    for dropped in (
        "dis_froz_proj",
        "dis_proj_min",
        "dis_proj_max",
        "num_iter",
        "conv_window",
        "write_hr",
    ):
        assert dropped not in staged, dropped
    keys = get_backend("vasp").knowledge.overlap_step_keys
    assert "num_wann" in keys and "projections" in keys
    assert "dis_proj_min" not in keys


def test_a_parent_without_a_charge_density_is_refused_before_anything_is_launched(run):
    run_dir, runner, log = run
    (runner.parent_dir / "CHGCAR").unlink()
    with pytest.raises(CalculationError, match="CHGCAR"):
        runner.run_overlaps("si")
    assert _launches(log) == []


# ---------------------------------------------------------------------------
# collecting the products
# ---------------------------------------------------------------------------


def test_the_overlaps_come_back_under_the_seedname_the_loop_uses(run):
    run_dir, runner, _ = run
    runner.run_overlaps("si")
    for ext in ("amn", "mmn", "eig"):
        assert (run_dir / f"si.{ext}").is_file(), ext
    assert (run_dir / "si.amn").read_text().startswith("stub amn")
    assert runner.overlaps_present("si")
    assert runner.amn_header("si") == (4, 2, 8)


def test_a_spin_channel_selects_the_spin_resolved_products(run, monkeypatch):
    run_dir, runner, _ = run
    monkeypatch.setenv("STUB_MODE", "spin")
    runner.spin_channel = "up"
    results = runner.run_overlaps("si")
    assert results[-1].ok, results[-1].stderr_tail
    assert (run_dir / "si.amn").is_file()


def test_a_zero_exit_with_no_overlaps_is_a_failed_step_that_names_the_missing_file(
    run, monkeypatch
):
    run_dir, runner, _ = run
    monkeypatch.setenv("STUB_MODE", "no_products")
    results = runner.run_overlaps("si")
    assert not results[-1].ok
    assert "wannier90.amn" in results[-1].stderr_tail
    assert not (run_dir / "si.amn").exists()


def test_a_binary_without_the_interface_is_reported_in_its_own_words_with_the_meaning(
    run, monkeypatch
):
    run_dir, runner, _ = run
    monkeypatch.setenv("STUB_MODE", "refuse_interface")
    results = runner.run_overlaps("si")
    assert not results[-1].ok
    assert "VASP2WANNIER90" in results[-1].stderr_tail
    reason = get_backend("vasp").refusal_reason(run_dir, "si")
    assert reason is not None
    assert "VASP2WANNIER90" in reason
    # the knowledge partition's reading of the line travels with it
    assert get_backend("vasp").knowledge.explain(reason) is not None


def test_a_library_side_refusal_is_read_from_the_werr_vasp_leaves_in_the_stage(
    run, monkeypatch
):
    run_dir, runner, _ = run
    monkeypatch.setenv("STUB_MODE", "library_refusal")
    results = runner.run_overlaps("si")
    assert not results[-1].ok
    reason = get_backend("vasp").refusal_reason(run_dir, "si")
    assert reason is not None
    assert "kmesh_get" in reason
    assert "Execution started" not in reason


def test_a_wannier90_refusal_of_the_localisation_is_read_as_before(run):
    """Nothing engine-side went wrong; Wannier90 itself declined the deck."""
    run_dir, runner, _ = run
    (run_dir / "si.werr").write_text(
        " Wannier90: Execution started on 12Sep2026\n Exiting.......\n"
        " dis_windows: Energy window contains fewer states than number of target WFs\n"
    )
    reason = get_backend("vasp").refusal_reason(run_dir, "si")
    assert (
        reason
        == "dis_windows: Energy window contains fewer states than number of target WFs"
    )


def test_no_refusal_anywhere_returns_nothing_rather_than_a_guess(run):
    run_dir, _, _ = run
    assert get_backend("vasp").refusal_reason(run_dir, "si") is None


# ---------------------------------------------------------------------------
# the shared half, through the VASP runner
# ---------------------------------------------------------------------------


def test_a_build_that_reuses_overlaps_does_not_run_the_engine(run):
    run_dir, runner, log = run
    for ext in ("amn", "mmn", "eig"):
        (run_dir / f"base.{ext}").write_text(f"base {ext}\n   4    2    8\n")
    results = runner.build("si", reuse_overlaps_from="base")
    assert _launches(log) == []
    assert (run_dir / "si.amn").read_text().startswith("base amn")
    assert results[-1].step == "si:localise"


def test_a_build_from_scratch_runs_the_engine_once_then_wannier90(run):
    run_dir, runner, log = run
    results = runner.build("si")
    assert len(_launches(log)) == 1
    assert [r.step for r in results] == ["si:overlaps", "si:localise"]
    assert all(r.ok for r in results)


def test_a_repeated_overlap_step_starts_from_a_clean_stage(run):
    run_dir, runner, _ = run
    runner.run_overlaps("si")
    stale = runner.stage_dir("si") / "wannier90.amn"
    stale.write_text("stale\n")
    (runner.stage_dir("si") / "leftover.txt").write_text("from an earlier attempt\n")
    runner.run_overlaps("si")
    assert not (runner.stage_dir("si") / "leftover.txt").exists()
    assert (run_dir / "si.amn").read_text().startswith("stub amn")


# ---------------------------------------------------------------------------
# the deck from the parent
# ---------------------------------------------------------------------------


def test_a_deck_is_written_from_the_parent_in_vasps_own_kpoint_order(parent, tmp_path):
    run_dir = tmp_path / "deck"
    run_dir.mkdir()
    inputs = deck_from_parent(
        parent / "vasprun.xml",
        run_dir,
        "si",
        projectors=[{"species": "Si", "orbitals": ("s", "p")}],
        recipe="hydrogenic",
    )
    assert inputs.seed == "si" and inputs.pw2wan is None
    assert inputs.values["num_wann"] == "8"
    assert inputs.values["num_bands"] == "4"
    assert inputs.values["mp_grid"] == "2 1 1"
    assert inputs.values["projectors"] == "Si:s;p"
    assert float(inputs.values["conv_window"]) >= 1
    # highest occupied state (5.0 eV) + 2 eV, not VASP's smeared Fermi level (5.2 eV)
    assert float(inputs.values["dis_froz_max"]) == pytest.approx(5.0 + 2.0)
    text = (run_dir / "si.win").read_text()
    block = text.split("begin kpoints", 1)[1].split("end kpoints", 1)[0].split()
    assert [float(x) for x in block] == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    assert "write_hr" in text.lower()
    assert "2.71534056" in text  # the cell, in Angstrom
    assert read_construction(run_dir, "si") == inputs


#: A POTCAR header carrying the fields the channel reader reads, nothing licensed:
#: silicon's table as the PBE PAW set lists it -- two s and two p partial waves with
#: projectors, and a d row of TYP 7, the channel used as the local potential.
SI_POTCAR_TABLE = """\
  PAW_PBE Si 01Jan2000
   4.00000000000000000
 parameters from PSCTR are:
   VRHFIN =Si: s2p2
   LEXCH  = PE
   TITEL  = PAW_PBE Si 01Jan2000
   Atomic configuration
    3 entries
     n  l   j            E        occ.
     1  0  0.50     -1785.8800   2.0000
     3  0  0.50       -10.8127   2.0000
     3  1  0.50        -4.0817   2.0000
   Description
     l       E           TYP  RCUT    TYP  RCUT
     0    -10.8127223     23  1.900
     0     -0.3000000     23  1.900
     1     -4.0817457     23  1.900
     1      0.7000000     23  1.900
     2      0.0000000      7  1.900
   Error from kinetic energy argument (eV)
   NDATA  =      100
 Non local Part
           0           2   1.90000000000000
  0.100000000000E+01  0.200000000000E+01
 Non local Part
           1           2   1.90000000000000
  0.100000000000E+01  0.200000000000E+01
 End of Dataset
"""


def _win_assignments(text: str) -> dict[str, str]:
    """Every ``key = value`` line of a .win outside its blocks, comments dropped."""
    values, block = {}, False
    for line in text.splitlines():
        cleaned = line.split("!", 1)[0].split("#", 1)[0].strip()
        lowered = cleaned.lower()
        if lowered.startswith("begin"):
            block = True
        elif lowered.startswith("end"):
            block = False
        elif not block and "=" in cleaned:
            key, _, value = cleaned.partition("=")
            values[key.strip().lower()] = value.strip()
    return values


def test_without_named_projectors_the_deck_names_every_channel_its_potcar_carries(
    parent, tmp_path
):
    """The automatic deck reads the pseudopotential table, not the occupied supply.

    Every non-local channel is named -- here Si s and p, eight functions for two
    atoms -- and the TYP 7 d row, the local potential, is not. The PROCAR is removed:
    the channel set no longer depends on which channels the occupied states use.
    """
    (parent / "POTCAR").write_text(SI_POTCAR_TABLE)
    (parent / "PROCAR").unlink()
    run_dir = tmp_path / "deck"
    run_dir.mkdir()
    inputs = deck_from_parent(
        parent / "vasprun.xml", run_dir, "si", recipe="hydrogenic"
    )
    assert inputs.values["projectors"] == "Si:s;p"
    assert inputs.values["num_wann"] == "8"
    assert inputs.values["dis_proj_min"] == "0.01"
    assert inputs.values["dis_proj_max"] == "0.95"
    text = (run_dir / "si.win").read_text()
    assert f"! projector channels: {parent / 'POTCAR'}" in text


@pytest.mark.parametrize(
    "projectors",
    [[{"species": "Si", "orbitals": ("s", "p")}], None],
    ids=["named projectors", "every POTCAR channel"],
)
def test_a_deck_that_sets_projectability_thresholds_also_switches_them_on(
    parent, tmp_path, projectors
):
    """``dis_proj_min``/``dis_proj_max`` do nothing unless ``dis_froz_proj`` is on.

    Wannier90 parses the two thresholds and silently ignores them when
    ``dis_froz_proj`` is off, its default (w90dev ``readwrite.F90`` l. 854-856; only
    ``disentangle.F90`` l. 160's ``frozen_proj`` branch builds the projectability
    window): the deck would freeze by energy alone while reading as a projectability
    deck. So the deck that writes the thresholds writes the flag.
    The flag is a localisation setting; VASP's overlap library never sees it.
    """
    (parent / "POTCAR").write_text(SI_POTCAR_TABLE)
    run_dir = tmp_path / "deck"
    run_dir.mkdir()
    deck_from_parent(
        parent / "vasprun.xml",
        run_dir,
        "si",
        projectors=projectors,
        recipe="hydrogenic",
    )
    text = (run_dir / "si.win").read_text()
    values = _win_assignments(text)
    assert values["dis_proj_min"] == "0.01" and values["dis_proj_max"] == "0.95"
    assert values.get("dis_froz_proj") == ".true."
    assert "dis_froz_proj" not in overlap_step_win(text).lower()


def test_a_symmetry_reduced_parent_cannot_seed_a_deck(parent, tmp_path):
    text = (parent / "vasprun.xml").read_text()
    (parent / "vasprun.xml").write_text(
        text.replace(
            'name="divisions">       2        1        1',
            'name="divisions">       2        2        1',
        )
    )
    run_dir = tmp_path / "deck"
    run_dir.mkdir()
    with pytest.raises(InputError, match="ISYM"):
        deck_from_parent(
            parent / "vasprun.xml",
            run_dir,
            "si",
            projectors=[{"species": "Si", "orbitals": ("s", "p")}],
            recipe="hydrogenic",
        )


# ---------------------------------------------------------------------------
# the real binary, on a compute node only
# ---------------------------------------------------------------------------

REAL_VASP = os.environ.get("AUTOWAN_VASP")
REAL_PARENT = os.environ.get("AUTOWAN_VASP_PARENT_DIR")


@pytest.mark.skipif(
    not (REAL_VASP and REAL_PARENT and os.environ.get("AUTOWAN_VASP_SMOKE")),
    reason=(
        "the real overlap step needs a vasp_std built with -DVASP2WANNIER90 "
        "(AUTOWAN_VASP), a finished parent directory holding CHGCAR and POTCAR "
        "(AUTOWAN_VASP_PARENT_DIR) and AUTOWAN_VASP_SMOKE=1, on a compute node -- "
        "the login node runs no engine"
    ),
)
def test_the_real_binary_writes_overlaps_for_the_fixture_parent(tmp_path):
    parent = Path(REAL_PARENT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = deck_from_parent(
        parent / "vasprun.xml", run_dir, "smoke", recipe="hydrogenic"
    )
    runner = VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent,
        vasp=REAL_VASP,
        mpi_launcher=tuple(os.environ.get("AUTOWAN_MPI", "mpirun -np 4").split()),
        timeout_s=1800.0,
    )
    results = runner.run_overlaps("smoke")
    assert results[-1].ok, results[-1].stderr_tail
    header = runner.amn_header("smoke")
    assert header is not None
    bands, kpts, projections = header
    assert projections == int(inputs.values["num_wann"])
    assert bands == int(inputs.values["num_bands"])
    mesh = [int(x) for x in inputs.values["mp_grid"].split()]
    assert kpts == mesh[0] * mesh[1] * mesh[2]
