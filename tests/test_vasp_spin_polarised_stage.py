"""The VASP overlap step on a spin-polarised parent, replayed from a real stage.

VASP 6.4.1 names an ``ISPIN = 2`` overlap step's products by spin number (``mlwf.F``:
``SEED_NAME//'.'//str(ISP)``): ``wannier90.1.*`` is spin up, ``wannier90.2.*`` spin
down, and the ``.win``/``.wout`` stay unnumbered. An ``ISPIN = 1`` step writes the
unnumbered ``wannier90.{amn,mmn,eig}``. Each deck describes one spin channel, so the
runner has to be told which one to bring back; what it must never do is stop a
spin-polarised construction with a message that names only files VASP was never going
to write.

The fixtures under ``fixtures/vasp/spin_polarised_stage`` are two real stages of the
2026-09-29 parent-adequacy controls on VASP (MnO, the ``spin`` control: ``clean`` ran
``ISPIN = 2``, ``degraded`` ran ``ISPIN = 1``). ``listing.tsv`` is every name VASP left
in the stage; the product files are byte-exact prefixes of the real ones
(``build_fixture.py``, provenance and source hashes in ``stage.json``). The stub below
replays that stage: it writes exactly what VASP wrote there, nothing more.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.inputs import read_incar

STAGES = Path(__file__).parent / "fixtures" / "vasp" / "spin_polarised_stage"
ISPIN2 = STAGES / "mno_ispin2"
ISPIN1 = STAGES / "mno_ispin1"
PRODUCTS = ("amn", "mmn", "eig")

REPLAY = """\
#!{python}
import csv
import shutil
from pathlib import Path

stage = Path({stage!r})
with open(stage / "listing.tsv") as handle:
    for row in csv.DictReader(handle, delimiter="\\t"):
        # the runner links the parent and writes INCAR / wannier90.win itself, and
        # it redirects VASP's stdout and stderr into vasp.log / vasp.err
        if row["kind"] != "vasp" or row["name"] in ("vasp.log", "vasp.err"):
            continue
        kept = stage / row["name"]
        if kept.is_file():
            shutil.copyfile(kept, row["name"])
        else:
            Path(row["name"]).write_text(
                "replayed name only: " + row["size"] + " bytes in the real stage\\n"
            )
"""

# The parent-adequacy controls on VASP classify a stop by the first of these patterns
# its reason matches (experiments/parent_adequacy_vasp_controls/assess.py); the spin
# control's degraded arm is pre-registered to stop as "missing overlap products". A
# clearer message must not move that stop into another class.
EARLIER_STOP_CLASSES = re.compile(
    r"forrtl|BAD TERMINATION|SIGSEGV|segmentation fault|exceeded \d+ s|"
    r"could not be launched|illegal instruction|Traceback|"
    r"projectability\s*<\s*0\.0\s*or\s*>\s*1\.0|More states in the frozen window|"
    r"built over \d+ bands|mismatch in .*\.eig|num_bands|"
    r"Neighbour not found|k-?point|kmesh",
    re.I,
)


def _listing(stage: Path) -> dict[str, dict[str, str]]:
    with open(stage / "listing.tsv") as handle:
        return {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}


def _runner(tmp_path: Path, stage: Path, spin_channel: str | None):
    parent = tmp_path / "parent"
    parent.mkdir(parents=True)
    for name in ("INCAR", "POSCAR", "KPOINTS"):
        shutil.copy(stage / "parent" / name, parent / name)
    # the overlap step reads these; their content is VASP's business, not the runner's
    (parent / "POTCAR").write_text("stand-in: the licensed POTCAR is not committed\n")
    (parent / "CHGCAR").write_text("stand-in\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(stage / "mno.win", run_dir / "mno.win")
    stub = tmp_path / "vasp_replay"
    stub.write_text(REPLAY.format(python=sys.executable, stage=str(stage)))
    stub.chmod(0o755)
    runner = VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent,
        vasp=str(stub),
        wannier90="true",
        mpi_launcher=(),
        spin_channel=spin_channel,
    )
    return runner, run_dir


def _below(eig: Path, fermi_ev: float) -> int:
    """States of the first k-point below the stage's own Fermi level."""
    return sum(
        1
        for line in eig.read_text().splitlines()
        if int(line.split()[1]) == 1 and float(line.split()[2]) < fermi_ev
    )


# ---------------------------------------------------------------------------
# the fixture is the real layout
# ---------------------------------------------------------------------------


def test_the_real_spin_polarised_stage_numbers_its_products_and_nothing_else():
    names = set(_listing(ISPIN2))
    for ext in PRODUCTS:
        assert {f"wannier90.1.{ext}", f"wannier90.2.{ext}"} <= names
        assert f"wannier90.{ext}" not in names
        assert f"wannier90.up.{ext}" not in names and f"wannier90.dn.{ext}" not in names
    # the deck and the library's own log stay unnumbered
    assert {"wannier90.win", "wannier90.wout"} <= names
    assert json.loads((ISPIN2 / "stage.json").read_text())["ispin"] == 2


def test_the_real_unpolarised_stage_writes_unnumbered_products():
    names = set(_listing(ISPIN1))
    for ext in PRODUCTS:
        assert f"wannier90.{ext}" in names
        assert f"wannier90.1.{ext}" not in names
    assert json.loads((ISPIN1 / "stage.json").read_text())["ispin"] == 1


# ---------------------------------------------------------------------------
# a named channel reads its own numbered products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel, number", [("up", 1), ("dn", 2)])
def test_a_named_channel_reads_its_numbered_products_from_the_real_stage(
    tmp_path, channel, number
):
    runner, run_dir = _runner(tmp_path, ISPIN2, channel)
    [result] = runner.run_overlaps("mno")
    assert result.ok, result.stderr_tail
    for ext in PRODUCTS:
        assert (run_dir / f"mno.{ext}").read_bytes() == (
            ISPIN2 / f"wannier90.{number}.{ext}"
        ).read_bytes()


def test_the_overlap_step_keeps_the_parents_spin_polarisation(tmp_path):
    runner, _ = _runner(tmp_path, ISPIN2, "up")
    runner.run_overlaps("mno")
    staged = read_incar((runner.stage_dir("mno") / "INCAR").read_text())
    assert staged["ISPIN"] == "2"
    assert staged["MAGMOM"].split()[:2] == ["9.0", "0.0"]


def test_spin_one_is_the_majority_channel_of_the_ferromagnetic_parent(tmp_path):
    """Up is the channel the positive MAGMOM polarised: MnO's Mn d shell is filled in it
    and empty in the other, so the up products hold more occupied states. Measured at
    the first k-point against the stage's own Fermi level: 14 up, 8 down."""
    fermi = json.loads((ISPIN2 / "stage.json").read_text())["fermi_energy_ev"]
    read = {}
    for channel in ("up", "dn"):
        runner, run_dir = _runner(tmp_path / channel, ISPIN2, channel)
        [result] = runner.run_overlaps("mno")
        assert result.ok, result.stderr_tail
        read[channel] = _below(run_dir / "mno.eig", fermi)
    assert read == {"up": 14, "dn": 8}


def test_an_unpolarised_stage_with_no_channel_reads_the_unnumbered_products(tmp_path):
    runner, run_dir = _runner(tmp_path, ISPIN1, None)
    [result] = runner.run_overlaps("mno")
    assert result.ok, result.stderr_tail
    for ext in PRODUCTS:
        assert (run_dir / f"mno.{ext}").read_bytes() == (
            ISPIN1 / f"wannier90.{ext}"
        ).read_bytes()


# ---------------------------------------------------------------------------
# a mismatch between the channel asked for and the stage VASP wrote is named
# ---------------------------------------------------------------------------


def test_a_spin_polarised_stage_without_a_named_channel_says_which_files_it_found(
    tmp_path,
):
    """Not 'missing wannier90.amn' alone: VASP ran ISPIN = 2 and was never going to
    write that file. The refusal names the numbered products and how to choose one."""
    runner, run_dir = _runner(tmp_path, ISPIN2, None)
    [result] = runner.run_overlaps("mno")
    assert not result.ok
    message = result.stderr_tail
    assert "missing VASP overlap products" in message
    for number in (1, 2):
        assert f"wannier90.{number}.amn" in message
    assert "ISPIN = 2" in message
    assert "spin_channel" in message
    # nothing is collected: neither channel is the deck's by default
    for ext in PRODUCTS:
        assert not (run_dir / f"mno.{ext}").exists()


def test_an_unpolarised_stage_asked_for_a_channel_says_it_has_none(tmp_path):
    runner, run_dir = _runner(tmp_path, ISPIN1, "up")
    [result] = runner.run_overlaps("mno")
    assert not result.ok
    message = result.stderr_tail
    assert message.index("missing VASP overlap products") < message.index("ISPIN = 1")
    assert "wannier90.amn" in message
    assert EARLIER_STOP_CLASSES.search(message) is None, message
    for ext in PRODUCTS:
        assert not (run_dir / f"mno.{ext}").exists()


def test_the_spin_polarised_refusal_stays_in_its_own_stop_class(tmp_path):
    runner, _ = _runner(tmp_path, ISPIN2, None)
    [result] = runner.run_overlaps("mno")
    assert EARLIER_STOP_CLASSES.search(result.stderr_tail) is None, result.stderr_tail


# ---------------------------------------------------------------------------
# the command line: --spin-index reaches the runner as well as the gate
# ---------------------------------------------------------------------------


class _Launched(Exception):
    """Raised by the stand-in loop once it has seen what the CLI built."""


def _launch(monkeypatch, tmp_path, stage: Path, *extra: str) -> dict:
    from autowan import cli

    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        raise _Launched

    monkeypatch.setattr(cli, "wannierize", _record)
    parent = tmp_path / "parent"
    parent.mkdir(parents=True)
    for name in ("INCAR", "POSCAR", "KPOINTS"):
        shutil.copy(stage / "parent" / name, parent / name)
    with pytest.raises(_Launched):
        cli.main(
            [
                "--run-dir",
                str(tmp_path),
                "--parent-xml",
                str(parent / "vasprun.xml"),
                "--seed",
                "mno",
                "--backend",
                "vasp",
                "--no-repair",
                "--no-figures",
                *extra,
            ]
        )
    return seen


@pytest.mark.parametrize(
    "extra, index, channel",
    [((), 0, "up"), (("--spin-index", "0"), 0, "up"), (("--spin-index", "1"), 1, "dn")],
    ids=["default", "spin-0", "spin-1"],
)
def test_the_cli_spin_index_names_the_runners_channel_on_a_spin_polarised_parent(
    monkeypatch, tmp_path, extra, index, channel
):
    """One channel for the gate and the overlap step: 0 is up (``wannier90.1.*``),
    1 is down (``wannier90.2.*``). Before, only the gate was told the channel; the
    runner was not, and stopped on files an ``ISPIN = 2`` step never writes."""
    seen = _launch(monkeypatch, tmp_path, ISPIN2, *extra)
    assert seen["config"].spin_index == index
    assert seen["runner"].spin_channel == channel


def test_the_cli_names_no_channel_for_an_unpolarised_parent(monkeypatch, tmp_path):
    seen = _launch(monkeypatch, tmp_path, ISPIN1)
    assert seen["config"].spin_index == 0
    assert seen["runner"].spin_channel is None


def test_the_cli_passes_a_second_channel_on_an_unpolarised_parent_to_be_refused(
    monkeypatch, tmp_path
):
    """Asked for channel 1 of an ``ISPIN = 1`` parent, the CLI does not quietly read
    the only channel there is: the runner is told ``dn`` and says ``ISPIN = 1``."""
    seen = _launch(monkeypatch, tmp_path, ISPIN1, "--spin-index", "1")
    assert seen["runner"].spin_channel == "dn"
