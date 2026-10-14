"""A deck that asks for a projectability window must actually get one.

Wannier90 reads ``dis_proj_min`` / ``dis_proj_max`` and then ignores them unless
``dis_froz_proj = .true.``: the flag defaults to ``.false.`` (``readwrite.F90``) and the
projectability window is applied only when it is set (``disentangle.F90``). The engine
says nothing. All twenty decks of the 2026-08-28 QE campaign carried the thresholds and
not the flag, every one of their ``.wout`` files is an energy-window run, and the sweep
was reported as projectability disentanglement. The failure was silent, so two rules
close it, one on each side of the calculation:

1. **Writer side.** Every deck writer -- the campaign shell script, the QE derived-deck
   writer, the VASP first deck and the VASP derived-deck writer -- refuses a deck that
   carries ``dis_proj_min`` / ``dis_proj_max`` (or ``dis_proj_auto = .true.``) without
   ``dis_froz_proj = .true.``, before anything is written.
2. **Output side.** A construction whose deck requested a projectability window but
   whose ``.wout`` does not show Wannier90's ``Projectability  Windows`` block is
   recorded **invalid** -- its own state in the run record, distinct from a gate
   refusal. The certification gate is not run on it and none of its checks or
   thresholds change: an invalid construction is not a measurement of the method it
   was meant to be, so there is nothing for the gate to judge.

The run record (``orchestrator.Attempt`` / ``Run``) and the loop that writes it
(``workflow.wannierize``) are where "recorded invalid" lands; the ``.wout`` reader only
measures what Wannier90 printed. Fixtures are real engine output -- see
``fixtures/wannier90/README.md`` for provenance.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autowan.backends import get_backend
from autowan.backends.vasp.inputs import deck_from_parent, overlap_step_win
from autowan.execution import ConstructionRunner, StepResult
from autowan.inputs import (
    InputError,
    ProjectabilityWindowIgnored,
    read_construction,
    requests_projectability,
    require_projectability_flag,
    write_derived,
)
from autowan.orchestrator import GateOutcome, Run
from autowan.w90_outputs import projectability_not_applied, read_wout

HERE = Path(__file__).parent
W90 = HERE / "fixtures" / "wannier90"
SYNTHETIC = HERE / "fixtures" / "vasp" / "synthetic"
RUN_ONE = HERE.parent / "scripts" / "run_one.sh"

FIXED = "graphene_projectability_fixed"
AUTOMATIC = "graphene_projectability_automatic"
IGNORED = "si_campaign_projectability_ignored"


def _text(name: str, suffix: str) -> str:
    return (W90 / f"{name}.{suffix}").read_text()


def _without(text: str, *keys: str) -> str:
    """The deck with every assignment of these keys removed."""
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.strip().lower().startswith(keys)
    )


# ---------------------------------------------------------------------------
# rule 1, the guard itself: what counts as asking, and what counts as the flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deck",
    [
        "dis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        "dis_proj_min = 0.01\n",
        "dis_proj_max = 0.95\n",
        "dis_proj_auto = .true.\n",
        # set, but set false: Wannier90 reads it and still ignores the thresholds
        "dis_froz_proj = .false.\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        # a commented-out flag is no flag
        "! dis_froz_proj = .true.\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        # the flag only inside another line's trailing comment
        "dis_proj_min = 0.01 ! dis_froz_proj = .true.\ndis_proj_max = 0.95\n",
    ],
    ids=[
        "pair-without-flag",
        "min-alone",
        "max-alone",
        "automatic-without-flag",
        "flag-false",
        "flag-commented-out",
        "flag-in-trailing-comment",
    ],
)
def test_a_deck_asking_for_the_window_without_the_flag_is_refused(deck):
    assert requests_projectability(deck)
    with pytest.raises(ProjectabilityWindowIgnored, match="dis_froz_proj"):
        require_projectability_flag(deck)


def test_the_refusal_is_an_input_error_so_existing_callers_already_catch_it():
    assert issubclass(ProjectabilityWindowIgnored, InputError)


@pytest.mark.parametrize(
    "deck",
    [
        "dis_froz_proj = .true.\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        "DIS_FROZ_PROJ = .TRUE.\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        "dis_froz_proj = T\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n",
        "dis_froz_proj : true\ndis_proj_min : 0.01\ndis_proj_max : 0.95\n",
        # the automatic thresholds are exempt from the pair, not from the flag
        "dis_froz_proj = .true.\ndis_proj_auto = .true.\n",
        # w90dev defaults dis_proj_auto to true: the flag alone asks for the window
        "dis_froz_proj = .true.\n",
    ],
    ids=["dotted", "upper-case", "bare-T", "colon", "automatic", "flag-alone"],
)
def test_a_deck_carrying_the_flag_is_accepted(deck):
    assert requests_projectability(deck)
    require_projectability_flag(deck)


@pytest.mark.parametrize(
    "deck",
    [
        "num_wann = 8\ndis_froz_max = 8.0\n",
        "dis_proj_auto = .false.\ndis_froz_max = 8.0\n",
        # a longer keyword that merely starts with a projectability key
        "dis_proj_auto_num_classes = 5\n",
        "! dis_proj_min = 0.01\n",
    ],
    ids=["energy-window", "automatic-off", "longer-keyword", "commented"],
)
def test_a_deck_that_asks_for_no_window_is_left_alone(deck):
    assert not requests_projectability(deck)
    require_projectability_flag(deck)


def test_the_real_decks_are_classified_as_the_engine_treated_them():
    with pytest.raises(ProjectabilityWindowIgnored):
        require_projectability_flag(_text(IGNORED, "win"))
    require_projectability_flag(_text(FIXED, "win"))
    require_projectability_flag(_text(AUTOMATIC, "win"))


# ---------------------------------------------------------------------------
# rule 1 on the QE derived-deck writer
# ---------------------------------------------------------------------------


@pytest.fixture
def campaign_deck(tmp_path):
    """The silicon campaign deck exactly as it ran: thresholds, no flag."""
    shutil.copy(W90 / f"{IGNORED}.win", tmp_path / "Si_pdwf.win")
    (tmp_path / "Si_pdwf.pw2wan").write_text(
        "&inputpp\n  outdir = './'\n  prefix = 'Si'\n  seedname = 'Si_pdwf'\n"
        "  write_amn = .true.\n  write_mmn = .true.\n  write_eig = .true.\n"
        "  atom_proj = .true.\n/\n"
    )
    return tmp_path


def test_qe_refuses_to_derive_a_deck_whose_window_would_be_ignored(campaign_deck):
    source = read_construction(campaign_deck, "Si_pdwf")
    with pytest.raises(ProjectabilityWindowIgnored, match="dis_froz_proj"):
        write_derived(source, campaign_deck, "Si_pdwf_r1", {"num_iter": 8000})
    assert not (campaign_deck / "Si_pdwf_r1.win").exists(), "half a deck was left"
    assert not (campaign_deck / "Si_pdwf_r1.pw2wan").exists()


def test_qe_refuses_the_verbatim_copy_too(campaign_deck):
    """A change to the overlap input alone copies the .win unchanged -- still a deck."""
    source = read_construction(campaign_deck, "Si_pdwf")
    with pytest.raises(ProjectabilityWindowIgnored):
        write_derived(source, campaign_deck, "Si_pdwf_r1", {"scdm_mu": 7.5})
    assert not (campaign_deck / "Si_pdwf_r1.win").exists()
    assert not (campaign_deck / "Si_pdwf_r1.pw2wan").exists()


def test_qe_derives_from_a_deck_that_carries_the_flag_and_keeps_it(campaign_deck):
    deck = campaign_deck / "Si_pdwf.win"
    deck.write_text("dis_froz_proj = .true.\n" + deck.read_text())
    source = read_construction(campaign_deck, "Si_pdwf")
    write_derived(source, campaign_deck, "Si_pdwf_r1", {"num_iter": 8000})
    text = (campaign_deck / "Si_pdwf_r1.win").read_text()
    assert re.search(r"^dis_froz_proj\s*=\s*\.true\.", text, re.M)
    assert "dis_proj_min = 0.01" in text
    require_projectability_flag(text)


def test_a_repair_that_introduces_thresholds_on_an_smv_deck_is_refused(tmp_path):
    """A projectability threshold on an energy-window deck is another family's knob.

    It used to be refused only because the flag was missing; it is now refused before
    that, as a method change no edit may make (switch_method is the way)."""
    (tmp_path / "d.win").write_text(
        "num_wann = 8\nnum_bands = 16\ndis_froz_max = 8.0\n\n"
        "begin projections\nSi:sp3\nend projections\n"
    )
    source = read_construction(tmp_path, "d")
    with pytest.raises(InputError, match="PDWF knob"):
        write_derived(source, tmp_path, "d_r1", {"dis_proj_min": 0.02})
    assert not (tmp_path / "d_r1.win").exists()


def test_naming_projectors_on_a_pdwf_deck_is_refused_not_a_silent_method_change(
    campaign_deck,
):
    """Explicit projectors used to drop the window and its flag together, turning a PDWF
    construction into an energy-window one without saying so. That edit is now refused
    (item 4 decision 1): projectors are SMV's knob, and the deck stays PDWF."""
    deck = campaign_deck / "Si_pdwf.win"
    deck.write_text("dis_froz_proj = .true.\n" + deck.read_text())
    source = read_construction(campaign_deck, "Si_pdwf")
    with pytest.raises(InputError, match="SMV knob"):
        write_derived(
            source,
            campaign_deck,
            "Si_pdwf_r1",
            {"projectors": [{"species": "Si", "orbitals": ("s", "p")}]},
            species_counts={"Si": 2},
        )
    assert not (campaign_deck / "Si_pdwf_r1.win").exists()


# ---------------------------------------------------------------------------
# rule 1 on the VASP writers
# ---------------------------------------------------------------------------


def test_the_vasp_first_deck_carries_the_flag(tmp_path):
    shutil.copy(SYNTHETIC / "vasprun.xml", tmp_path / "vasprun.xml")
    run_dir = tmp_path / "deck"
    run_dir.mkdir()
    deck_from_parent(
        tmp_path / "vasprun.xml",
        run_dir,
        "si",
        projectors=[{"species": "Si", "orbitals": ("s", "p")}],
        recipe="hydrogenic",
    )
    text = (run_dir / "si.win").read_text()
    assert requests_projectability(text)
    assert re.search(r"^dis_froz_proj\s*=\s*\.true\.", text, re.M)
    require_projectability_flag(text)


def test_the_flag_never_reaches_the_vasp_overlap_step(tmp_path):
    """VASP's library is Wannier90 3.1, which refuses keywords it does not know."""
    shutil.copy(SYNTHETIC / "vasprun.xml", tmp_path / "vasprun.xml")
    deck_from_parent(
        tmp_path / "vasprun.xml",
        tmp_path,
        "si",
        projectors=[{"species": "Si", "orbitals": ("s", "p")}],
        recipe="hydrogenic",
    )
    staged = overlap_step_win((tmp_path / "si.win").read_text())
    assert "dis_froz_proj" not in staged
    assert "dis_proj" not in staged


VASP_CAMPAIGN_STYLE_WIN = """\
num_wann  = 8
num_bands = 16
num_iter  = 400
conv_window = 3
dis_froz_max = 7.0
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


def test_vasp_refuses_to_derive_a_deck_whose_window_would_be_ignored(tmp_path):
    (tmp_path / "si.win").write_text(VASP_CAMPAIGN_STYLE_WIN)
    backend = get_backend("vasp")
    source = backend.read_construction(tmp_path, "si")
    with pytest.raises(ProjectabilityWindowIgnored, match="dis_froz_proj"):
        backend.write_derived(source, tmp_path, "si_r1", {"num_iter": 4000})
    assert not (tmp_path / "si_r1.win").exists()


def test_vasp_derives_from_a_deck_that_carries_the_flag(tmp_path):
    (tmp_path / "si.win").write_text(
        "dis_froz_proj = .true.\n" + VASP_CAMPAIGN_STYLE_WIN
    )
    backend = get_backend("vasp")
    source = backend.read_construction(tmp_path, "si")
    backend.write_derived(source, tmp_path, "si_r1", {"num_iter": 4000})
    require_projectability_flag((tmp_path / "si_r1.win").read_text())


# ---------------------------------------------------------------------------
# rule 1 on the campaign shell script's deck writer
# ---------------------------------------------------------------------------

#: The flag as the shell script's writer must spell it: one Python string literal, one
#: deck line. Pinned so the refusal below can be exercised by taking exactly that line
#: away -- the only way to reach the guard in a writer that always sets the flag.
FLAG_LITERAL = '"dis_froz_proj = .true."'


def _campaign_deck_writer() -> str:
    """The embedded Python that writes ``<M>_pdwf.win`` in ``run_one.sh``."""
    text = RUN_ONE.read_text()
    match = re.search(
        r"^python3 - \"\$M\" \"\$NW\"[^\n]*<<'PYEOF'[^\n]*\n(.*?)^PYEOF$",
        text,
        re.S | re.M,
    )
    assert match, "run_one.sh no longer has the deck-writing heredoc"
    return match.group(1)


def _run_campaign_deck_writer(source: str, where: Path):
    (where / "Si.scf").write_text(
        " &control\n  prefix = 'Si'\n /\nATOMIC_SPECIES\n Si 28.086 Si.pz-vbc.UPF\n"
    )
    return subprocess.run(
        [sys.executable, "-", "Si", "8", "16", "6.0381", "2", "32"],
        input=source,
        cwd=where,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_campaign_script_writes_the_flag_into_its_deck(tmp_path):
    source = _campaign_deck_writer()
    assert source.count(FLAG_LITERAL) == 1
    result = _run_campaign_deck_writer(source, tmp_path)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "Si_pdwf.win").read_text()
    assert "dis_proj_min = 0.01" in text and "dis_proj_max = 0.95" in text
    assert re.search(r"^dis_froz_proj\s*=\s*\.true\.", text, re.M)
    require_projectability_flag(text)


def test_the_campaign_script_refuses_a_deck_without_the_flag(tmp_path):
    source = _campaign_deck_writer().replace(FLAG_LITERAL, '""')
    result = _run_campaign_deck_writer(source, tmp_path)
    assert result.returncode != 0
    assert "dis_froz_proj" in result.stderr + result.stdout
    assert not (tmp_path / "Si_pdwf.win").exists(), "the refused deck was written"


def test_a_refused_campaign_deck_stops_the_script():
    """Without this the script ran on: its heredocs' exit codes were never read."""
    invocation = next(
        line
        for line in RUN_ONE.read_text().splitlines()
        if line.startswith('python3 - "$M" "$NW"')
    )
    assert re.search(r"<<'PYEOF'\s*\|\|.*\bexit\b", invocation), invocation


def test_the_campaign_script_no_longer_calls_its_deck_the_aiida_build():
    text = RUN_ONE.read_text()
    assert "exactly as" not in text
    assert "PDWF as aiida-wannier90-workflows builds it" not in text


# ---------------------------------------------------------------------------
# rule 2, the reader: what Wannier90 printed
# ---------------------------------------------------------------------------


def test_the_reader_records_the_fixed_window_wannier90_applied():
    run = read_wout(W90 / f"{FIXED}.wout")
    window = run.projectability_window
    assert window is not None
    assert window.lower == pytest.approx(0.01)
    assert window.upper == pytest.approx(0.85)
    assert window.automatic is False


def test_the_reader_records_automatically_chosen_thresholds():
    run = read_wout(W90 / f"{AUTOMATIC}.wout")
    window = run.projectability_window
    assert window is not None
    assert window.automatic is True
    assert window.n_classes == 5
    assert window.lower == pytest.approx(0.03906)
    assert window.upper == pytest.approx(0.94531)


def test_an_energy_window_run_has_no_projectability_window():
    """``In dis_proj_froz... done`` names a subroutine of the energy-window route."""
    run = read_wout(W90 / f"{IGNORED}.wout")
    assert run.projectability_window is None
    # the reader still reads everything else it read before
    assert run.num_wann == 8 and run.centres_ang.shape == (8, 3)
    assert run.disentanglement.converged and run.wannierisation.converged


# ---------------------------------------------------------------------------
# rule 2, validity: the deck's request against the engine's output
# ---------------------------------------------------------------------------


def test_a_requested_window_absent_from_the_output_makes_the_construction_invalid():
    reason = projectability_not_applied(
        _text(IGNORED, "win"), read_wout(W90 / f"{IGNORED}.wout")
    )
    assert reason is not None
    assert "projectability" in reason.lower()
    assert "dis_froz_proj" in reason


@pytest.mark.parametrize("name", [FIXED, AUTOMATIC])
def test_a_requested_window_the_output_shows_is_valid(name):
    assert (
        projectability_not_applied(_text(name, "win"), read_wout(W90 / f"{name}.wout"))
        is None
    )


def test_a_deck_that_asked_for_no_window_is_not_invalidated_by_its_absence():
    energy_only = _without(
        _text(IGNORED, "win"), "dis_proj_min", "dis_proj_max", "auto_projections"
    )
    assert (
        projectability_not_applied(energy_only, read_wout(W90 / f"{IGNORED}.wout"))
        is None
    )


def test_the_flag_in_the_deck_does_not_substitute_for_the_block_in_the_output():
    """The pitfall this closes: the input keyword is a request, not evidence."""
    flagged = "dis_froz_proj = .true.\n" + _text(IGNORED, "win")
    assert (
        projectability_not_applied(flagged, read_wout(W90 / f"{IGNORED}.wout"))
        is not None
    )


# ---------------------------------------------------------------------------
# rule 2, the run record: invalid is its own state
# ---------------------------------------------------------------------------


def test_the_run_record_keeps_invalid_apart_from_a_gate_refusal():
    run = Run("Si")
    invalid = run.start_attempt({"num_iter": "4000"})
    run.finish_attempt(invalid, invalid_reason="no projectability window in Si.wout")
    assert invalid.status == "invalid"
    assert invalid.outcome is None and invalid.calculation_error is None
    assert not run.passed
    assert run.best_effort() is None

    logged = run.to_dict()["attempts"][0]
    assert logged["status"] == "invalid"
    assert logged["invalid_reason"] == "no projectability window in Si.wout"


@pytest.mark.parametrize(
    "finish, status",
    [
        (
            {
                "outcome": GateOutcome(
                    passed=False,
                    evaluated_check_ids=("spread",),
                    failed_check_ids=("spread",),
                )
            },
            "gate_refused",
        ),
        (
            {
                "outcome": GateOutcome(
                    passed=True, evaluated_check_ids=("spread",), failed_check_ids=()
                )
            },
            "passed",
        ),
        ({"calculation_error": "wannier90 exited 1"}, "calculation_failed"),
    ],
    ids=["gate-refused", "passed", "calculation-failed"],
)
def test_every_other_ending_has_its_own_status(finish, status):
    run = Run("Si")
    attempt = run.start_attempt({"num_iter": "4000"})
    run.finish_attempt(attempt, **finish)
    assert attempt.status == status
    assert run.to_dict()["attempts"][0]["status"] == status


class _LayingDownRunner(ConstructionRunner):
    """Runs nothing; lays a finished Wannier90 output down under the seed it is given."""

    def __init__(self, run_dir, wout):
        super().__init__(run_dir=run_dir)
        self.wout = wout
        self.builds = []

    def run_overlaps(self, seed):  # pragma: no cover - build is overridden
        raise AssertionError("this runner never computes overlaps")

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append(seed)
        shutil.copy(self.wout, self.run_dir / f"{seed}.wout")
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


def _within_the_synthetic_parent(deck: str) -> str:
    """The deck shrunk to what the 4-band synthetic parent computes.

    The parent-adequacy pre-flight blocks a deck that draws on more bands than the
    parent carries, before anything is built; these loop tests are about what happens
    after the build, so the deck asks for what this parent can supply.
    """
    return (
        deck.replace("num_wann  = 8", "num_wann  = 2")
        .replace("num_bands = 16", "num_bands = 4")
        .replace("Si:s;p", "Si:s")
    )


def _no_model(model):  # pragma: no cover - reaching it is the failure
    pytest.fail("a model was consulted about a construction that is not a measurement")


def test_the_loop_records_an_ignored_window_as_invalid_and_never_gates_it(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    (tmp_path / "si.win").write_text(
        _within_the_synthetic_parent(VASP_CAMPAIGN_STYLE_WIN)
    )
    runner = _LayingDownRunner(tmp_path, W90 / f"{IGNORED}.wout")
    log = tmp_path / "run.json"

    result = wannierize(
        run_dir=tmp_path,
        parent_xml=SYNTHETIC / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=_no_model,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
        log_path=log,
    )

    assert runner.builds == ["si"], "an invalid baseline must not seed a repair"
    [attempt] = result.run.attempts
    assert attempt.status == "invalid"
    assert attempt.outcome is None, "the gate was run on an invalid construction"
    assert attempt.calculation_error is None
    assert "dis_froz_proj" in attempt.invalid_reason
    assert result.report is None
    assert not result.certified
    assert result.status == "invalid"
    assert "invalid" in (result.run.terminal_reason or "")
    assert '"status": "invalid"' in log.read_text()


def test_the_loop_gates_a_construction_whose_window_was_applied(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    (tmp_path / "si.win").write_text(
        _within_the_synthetic_parent(
            "dis_froz_proj = .true.\n" + VASP_CAMPAIGN_STYLE_WIN
        )
    )
    runner = _LayingDownRunner(tmp_path, W90 / f"{FIXED}.wout")

    result = wannierize(
        run_dir=tmp_path,
        parent_xml=SYNTHETIC / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
    )

    [attempt] = result.run.attempts
    assert attempt.invalid_reason is None
    assert attempt.outcome is not None, "a valid construction reaches the gate"
    assert attempt.status in ("passed", "gate_refused")
    assert result.report is not None
    assert result.status != "invalid"


# ---------------------------------------------------------------------------
# rule 1 on the one-pass repair script's own deck writer
# ---------------------------------------------------------------------------
