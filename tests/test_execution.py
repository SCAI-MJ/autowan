"""What a repair invalidates, and what a rewriter must not lose.

Two properties keep the loop honest and cheap. A change is re-run only as far as it
reaches, and which step that is comes from the parameter vocabulary rather than a
second list that could drift from it. And a derived deck keeps every line the change
did not name — a rewriter that rebuilt only what it understood would silently drop the
k-point list, the cell, or the atom positions.
"""

from __future__ import annotations

import pytest

from autowan.execution import CalculationError, StepResult, WannierRunner
from autowan.inputs import (
    InputError,
    read_construction,
    write_derived,
)

WIN = """\
num_wann  = 9
num_bands = 14
num_iter  = 400
conv_window = 3
dis_froz_max = 30.0
write_hr = .true.

begin projections
Fe:s;p;d
end projections

mp_grid 8 8 8

begin kpoints
 0.0 0.0 0.0
 0.5 0.5 0.5
end kpoints
"""

PW2WAN = """\
&inputpp
  outdir = './'
  prefix = 'Fe'
  seedname = 'base'
  spin_component = 'up'
  atom_proj = .true.
/
"""


@pytest.fixture
def deck(tmp_path):
    (tmp_path / "base.win").write_text(WIN)
    (tmp_path / "base.pw2wan").write_text(PW2WAN)
    return tmp_path


def test_a_localisation_knob_does_not_invalidate_the_overlaps():
    assert not WannierRunner.invalidates_overlaps(["num_iter", "conv_window"])


def test_a_projection_knob_does_invalidate_them():
    assert WannierRunner.invalidates_overlaps(["scdm_mu"])
    assert WannierRunner.invalidates_overlaps(["num_iter", "scdm_sigma"])


def test_a_derived_deck_keeps_every_line_the_change_did_not_name(deck):
    source = read_construction(deck, "base")
    derived = write_derived(source, deck, "base_r1", {"num_iter": 4000})
    text = (deck / "base_r1.win").read_text()

    assert derived.values["num_iter"] == "4000"
    assert "num_iter  = 400\n" not in text
    for preserved in (
        "begin kpoints",
        "0.5 0.5 0.5",
        "mp_grid 8 8 8",
        "Fe:s;p;d",
        "write_hr = .true.",
    ):
        assert preserved in text


def test_changing_projectors_replaces_the_block_rather_than_appending(deck):
    source = read_construction(deck, "base")
    write_derived(
        source,
        deck,
        "base_r1",
        {"projectors": [{"species": "Fe", "orbitals": ("s", "d")}]},
    )
    text = (deck / "base_r1.win").read_text()
    assert text.count("begin projections") == 1
    assert "Fe:s;d" in text
    assert "Fe:s;p;d" not in text


def test_a_chain_of_projector_repairs_still_leaves_exactly_one_block(deck):
    # Each repair is written from the deck before it, so a rewriter that appended
    # instead of replacing would only show up after the second or third link -- and a
    # deck carrying two projections blocks is refused by Wannier90 with a message
    # about the .amn, which reads as a physics failure rather than a writing one.
    source = read_construction(deck, "base")
    for i, orbitals in enumerate((("s",), ("s", "p"), ("s", "s", "p", "d")), start=1):
        source = write_derived(
            source,
            deck,
            f"base_r{i}",
            {"projectors": [{"species": "Fe", "orbitals": orbitals}]},
        )
        text = (deck / f"base_r{i}.win").read_text()
        assert text.count("begin projections") == 1, f"link {i}"
        assert (
            len(
                [
                    ln
                    for ln in text.splitlines()
                    if ln.strip().lower().startswith("num_wann")
                ]
            )
            == 1
        ), f"link {i}"


def test_an_overlap_knob_reaches_the_overlap_input_and_renames_the_seed(deck):
    # scdm_mu is SCDM's knob, so the deck is an SCDM one.
    (deck / "base.pw2wan").write_text(
        PW2WAN.replace("atom_proj = .true.", "scdm_proj = .true.")
    )
    source = read_construction(deck, "base")
    write_derived(source, deck, "base_r1", {"scdm_mu": 20.5})
    text = (deck / "base_r1.pw2wan").read_text()
    assert "scdm_mu = 20.5" in text
    assert "seedname = 'base_r1'" in text
    assert "seedname = 'base'" not in text
    assert "spin_component = 'up'" in text


def test_a_parameter_outside_the_vocabulary_is_refused(deck):
    source = read_construction(deck, "base")
    with pytest.raises(InputError, match="not an editable parameter"):
        write_derived(source, deck, "base_r1", {"conv_tol": 1e-3})


def test_gate_inputs_are_typed_as_the_checks_expect(deck):
    values = read_construction(deck, "base").gate_inputs(spin_index=1)
    assert values["num_wann"] == 9
    assert values["mp_grid"] == (8, 8, 8)
    assert values["spin_index"] == 1


def test_reusing_overlaps_that_do_not_exist_is_refused(tmp_path):
    runner = WannierRunner(run_dir=tmp_path)
    with pytest.raises(CalculationError, match="no complete set of overlaps"):
        runner.reuse_overlaps("absent", "target")


def test_a_deck_without_a_convergence_test_is_given_one(tmp_path):
    """Wannier90 runs no convergence test unless asked, so a deck that does not ask
    produces constructions that neither converged nor failed to — and a gate reading
    the absent flag would report our omission as a verdict about the method."""
    from autowan.execution import DEFAULT_CONV_WINDOW, ensure_convergence_test

    win = tmp_path / "deck.win"
    win.write_text("num_wann = 9\nnum_iter = 400\n")
    assert ensure_convergence_test(win) is True
    assert f"conv_window = {DEFAULT_CONV_WINDOW}" in win.read_text()
    assert "num_wann = 9" in win.read_text()


def test_a_deck_that_already_states_one_is_left_alone(tmp_path):
    from autowan.execution import ensure_convergence_test

    win = tmp_path / "deck.win"
    win.write_text("conv_window = 7\nnum_wann = 9\n")
    assert ensure_convergence_test(win) is False
    assert win.read_text().count("conv_window") == 1
    assert "conv_window = 7" in win.read_text()


PDWF_WIN = """\
num_wann  = 9
num_bands = 14
auto_projections = .true.
dis_froz_proj = .true.
dis_proj_min = 0.01
dis_proj_max = 0.95
num_iter = 1000
write_hr = .true.

begin projections
end projections

mp_grid 8 8 8
"""

PDWF_PW2WAN = """\
&inputpp
  outdir = './'
  prefix = 'Fe'
  seedname = 'pdwf'
  spin_component = 'up'
  write_amn = .true.
  atom_proj = .true.
/
"""


def test_naming_projectors_on_a_pdwf_deck_is_refused(tmp_path):
    """Naming projectors on a projectability deck used to drop the window silently."""
    (tmp_path / "pdwf.win").write_text(PDWF_WIN)
    (tmp_path / "pdwf.pw2wan").write_text(PDWF_PW2WAN)
    source = read_construction(tmp_path, "pdwf")
    with pytest.raises(InputError, match="SMV knob"):
        write_derived(
            source,
            tmp_path,
            "pdwf_r1",
            {"projectors": [{"species": "Fe", "orbitals": ("s", "p", "d")}]},
        )
    assert not (tmp_path / "pdwf_r1.win").exists()


def test_naming_projectors_replaces_the_automatic_scheme_in_both_files(tmp_path):
    """SMV on the pseudopotential's orbitals -> SMV on named orbitals, in both files.

    Left half-applied it is worse than not applied: Wannier90 would read a projections
    block while pw2wannier90 filled the .amn from the pseudopotential's own atomic
    orbitals, and the construction would be built from projectors nobody chose.
    """
    smv = "\n".join(
        line
        for line in PDWF_WIN.splitlines()
        if not line.startswith(("dis_froz_proj", "dis_proj_"))
    )
    (tmp_path / "pdwf.win").write_text("dis_froz_max = 12.0\n" + smv + "\n")
    (tmp_path / "pdwf.pw2wan").write_text(PDWF_PW2WAN)
    source = read_construction(tmp_path, "pdwf")

    write_derived(
        source,
        tmp_path,
        "pdwf_r1",
        {"projectors": [{"species": "Fe", "orbitals": ("s", "p", "d")}]},
    )

    win = (tmp_path / "pdwf_r1.win").read_text()
    assert "Fe:s;p;d" in win
    for delegated in (
        "auto_projections",
        "dis_froz_proj",
        "dis_proj_min",
        "dis_proj_max",
    ):
        assert delegated not in win, f"{delegated} still delegates the choice"
    assert "num_wann  = 9" in win and "mp_grid 8 8 8" in win

    pw2wan = (tmp_path / "pdwf_r1.pw2wan").read_text()
    assert (
        "atom_proj" not in pw2wan
    ), "the overlap step still supplies its own projections"
    assert "seedname = 'pdwf_r1'" in pw2wan
    assert "write_amn = .true." in pw2wan


def test_naming_projectors_forces_the_overlaps_to_be_recomputed():
    assert WannierRunner.invalidates_overlaps({"projectors": []})


def test_a_fermi_relative_knob_is_proposed_as_an_offset_and_written_as_an_absolute(
    tmp_path,
):
    """Both conventions in play at once put a window ten volts from where it belonged.

    Shown the absolute edge of a frozen window and told it reads naturally relative to
    the Fermi level, a model returned the offset. The offset is now the only thing that
    crosses the interface; the writer does the arithmetic.
    """
    from autowan.inputs import to_absolute, to_relative

    (tmp_path / "deck.win").write_text("num_wann = 9\ndis_froz_max = 14.6407\n")
    source = read_construction(tmp_path, "deck")

    # the model proposes +2.4 eV above the Fermi level
    derived = write_derived(
        source, tmp_path, "deck_r1", {"dis_froz_max": 2.4}, fermi_energy_ev=12.6407
    )
    written = (tmp_path / "deck_r1.win").read_text()
    assert "dis_froz_max = 15.0407" in written, written

    # and reading it back for the model returns the offset, not the absolute
    assert to_relative("dis_froz_max", derived.values["dis_froz_max"], 12.6407) == 2.4
    # knobs that are not energies pass through untouched
    assert to_absolute("num_iter", 4000, 12.6407) == 4000
    assert to_relative("num_iter", 4000, 12.6407) == 4000


def test_naming_projectors_updates_the_count_they_imply(tmp_path):
    """The count is determined by the projector set, and Wannier90 refuses a deck where
    the two disagree — so a repair that widens the manifold and leaves the old count
    behind arrives as an error rather than as a change."""
    from autowan.inputs import wannier_count

    (tmp_path / "deck.win").write_text("num_wann = 1\nnum_bands = 2\n")
    (tmp_path / "deck.pw2wan").write_text(
        "&inputpp\n  prefix = 'Na'\n  seedname = 'deck'\n  atom_proj = .true.\n/\n"
    )
    source = read_construction(tmp_path, "deck")

    write_derived(
        source,
        tmp_path,
        "deck_r1",
        {"projectors": [{"species": "Na", "orbitals": ("s", "p")}]},
        species_counts={"Na": 1},
    )
    win = (tmp_path / "deck_r1.win").read_text()
    assert "num_wann = 4" in win, win  # one s plus three p
    assert "num_wann = 1" not in win
    assert "Na:s;p" in win

    # and the arithmetic itself, including several atoms of one species
    assert (
        wannier_count([{"species": "Fe", "orbitals": ("s", "p", "d")}], {"Fe": 1}) == 9
    )
    assert wannier_count([{"species": "O", "orbitals": ("s", "p")}], {"O": 3}) == 12


def test_raising_num_bands_shortens_the_band_exclusion(tmp_path):
    # The .eig holds every band the parent produced; a deck declaring fewer says which
    # ones it leaves out. Raising the declaration without shortening the exclusion is
    # refused by Wannier90 as a file mismatch.
    (tmp_path / "d.win").write_text(
        "num_wann = 4\nnum_bands = 8\nexclude_bands = 9-24\nnum_iter = 100\n"
    )
    (tmp_path / "d.pw2wan").write_text("&inputpp\n  seedname = 'd'\n/\n")
    source = read_construction(tmp_path, "d")
    write_derived(source, tmp_path, "d_r1", {"num_bands": 16})
    text = (tmp_path / "d_r1.win").read_text()
    assert "exclude_bands = 17-24" in text
    assert "exclude_bands = 9-24" not in text

    write_derived(source, tmp_path, "d_r2", {"num_bands": 24})
    assert "exclude_bands" not in (tmp_path / "d_r2.win").read_text()


def test_overlaps_left_by_an_earlier_run_are_not_adopted(tmp_path):
    # A second run of the same material writes the same seednames. Adopting whatever
    # files carry those names let one run inherit another's overlaps for a deck it had
    # since changed, and Wannier90 refused the result as though the physics were wrong.
    for ext in ("amn", "mmn", "eig", "nnkp"):
        (tmp_path / f"d_r1.{ext}").write_text("stale")
    (tmp_path / "d_r1.win").write_text("num_wann = 4\nconv_window = 3\n")
    calls = []
    runner = WannierRunner(run_dir=tmp_path, wannier90="w90", pw2wannier90="p2w")
    runner._run = lambda step, command: (
        calls.append(list(command)) or StepResult(step, 0, 0.0)
    )
    runner.build("d_r1")
    for ext in ("amn", "mmn", "eig", "nnkp"):
        assert not (tmp_path / f"d_r1.{ext}").is_file(), f"the stale .{ext} survived"
    assert any("-pp" in c for c in calls), "the overlaps must be rebuilt"


def test_the_prospective_set_matches_what_the_deck_writer_records(
    tmp_path, monkeypatch
):
    # The guard against re-running a parameter set compares against what the run
    # records. Mirroring the writer's derivations by hand kept missing keys nobody
    # thought of -- the implied num_wann, windows converted from offsets, the recorded
    # projection source -- so the comparison is made against a deck actually written.
    from types import SimpleNamespace

    from autowan.actions import ParameterChange
    from autowan.workflow import _prospective

    (tmp_path / "d.win").write_text(
        "num_wann = 9\nnum_bands = 20\nauto_projections = .true.\nconv_window = 3\n"
    )
    (tmp_path / "d.pw2wan").write_text(
        "&inputpp\n  seedname = 'd'\n  atom_proj = .true.\n/\n"
    )
    inputs = read_construction(tmp_path, "d")
    action = SimpleNamespace(
        changes=(
            ParameterChange(
                "projectors", None, [{"species": "Fe", "orbitals": ["s", "d"]}], "r"
            ),
        )
    )
    parent = SimpleNamespace(fermi_energy_ev=0.0, species=["Fe", "Fe"])

    prospective = _prospective(action, inputs, parent)
    real = write_derived(
        inputs,
        tmp_path,
        "d_r1",
        {"projectors": [{"species": "Fe", "orbitals": ["s", "d"]}]},
        fermi_energy_ev=0.0,
        species_counts={"Fe": 2},
    )
    assert prospective == {**real.values, "method": "smv"}


def test_two_projector_sets_of_the_same_size_are_different_parameter_sets(tmp_path):
    # Ti:s;p;d and Ti:d;sp3 are both eighteen functions on two atoms. Every recorded
    # scalar was identical, so the second was refused as a repeat of the first.
    for name, block in (("a", "Ti:s;p;d"), ("b", "Ti:d;sp3")):
        (tmp_path / f"{name}.win").write_text(
            f"num_wann = 18\nnum_bands = 40\n"
            f"begin projections\n{block}\nend projections\n"
        )
        (tmp_path / f"{name}.pw2wan").write_text("&inputpp\n  seedname = 'x'\n/\n")
    first = read_construction(tmp_path, "a").values
    second = read_construction(tmp_path, "b").values
    assert first != second
    assert first["projectors"] == "Ti:s;p;d"


def test_overlaps_left_by_an_earlier_run_are_not_adopted_under_a_reuse(tmp_path):
    # The sibling test above covers a build that recomputes. A build told to reuse
    # another construction's overlaps took a different path: it copied only when the
    # target seedname had no files, so a second run of the same material -- which
    # writes the same `_r1`, `_r2` seednames -- kept the first run's .amn for a deck
    # the repair had since rewritten. On sodium the inherited file held four
    # projections against a deck naming one, and Wannier90 refused it as though the
    # projector set were wrong. Whether to reuse is the caller's decision; *which*
    # files to reuse is not the disk's either.
    for ext in ("amn", "mmn", "eig", "nnkp"):
        (tmp_path / f"src.{ext}").write_text(f"from the named source {ext}")
        (tmp_path / f"d_r1.{ext}").write_text("stale, from an earlier run")
    (tmp_path / "d_r1.win").write_text("num_wann = 4\nconv_window = 3\n")
    runner = WannierRunner(run_dir=tmp_path, wannier90="w90", pw2wannier90="p2w")
    runner._run = lambda step, command: StepResult(step, 0, 0.0)
    runner.build("d_r1", reuse_overlaps_from="src")
    for ext in ("amn", "mmn", "eig", "nnkp"):
        assert (tmp_path / f"d_r1.{ext}").read_text() == (
            f"from the named source {ext}"
        ), f"the stale .{ext} was adopted instead of the named source"


def _amn(path, n_bands: int, n_kpts: int, n_proj: int) -> None:
    """The header Wannier90 reads back: a comment, then the three counts."""
    path.write_text(f" written by a test\n{n_bands:12d}{n_kpts:12d}{n_proj:12d}\n")


def test_a_deck_and_its_projections_disagreeing_is_named_rather_than_run(tmp_path):
    # Wannier90's own report of this is "<seed>.amn has not the right number of
    # projections", which says nothing about which number is wrong or where the file
    # came from, and the loop records it as an attempt that was never scored -- a
    # spent attempt that measured nothing. The two counts are on disk before the
    # localisation starts, so the disagreement can be named.
    (tmp_path / "d_r1.win").write_text("num_wann = 1\nconv_window = 3\n")
    _amn(tmp_path / "d_r1.amn", 9, 216, 4)
    for ext in ("mmn", "eig", "nnkp"):
        (tmp_path / f"d_r1.{ext}").write_text("present")
    runner = WannierRunner(run_dir=tmp_path, wannier90="w90", pw2wannier90="p2w")
    runner._run = lambda step, command: StepResult(step, 0, 0.0)
    with pytest.raises(CalculationError) as excinfo:
        runner.build("d_r1", reuse_overlaps_from="d_r1")
    message = str(excinfo.value)
    assert "4" in message and "1" in message, message
    assert "projection" in message


def test_a_deck_and_its_projections_agreeing_is_left_alone(tmp_path):
    (tmp_path / "d_r1.win").write_text("num_wann = 4\nconv_window = 3\n")
    _amn(tmp_path / "d_r1.amn", 9, 216, 4)
    for ext in ("mmn", "eig", "nnkp"):
        (tmp_path / f"d_r1.{ext}").write_text("present")
    runner = WannierRunner(run_dir=tmp_path, wannier90="w90", pw2wannier90="p2w")
    runner._run = lambda step, command: StepResult(step, 0, 0.0)
    runner.build("d_r1", reuse_overlaps_from="d_r1")
