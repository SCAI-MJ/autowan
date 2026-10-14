"""A deck the stored parent cannot satisfy is a refused attempt, not the end of a run.

Strontium oxide, replicate A, 2026-10-03. The baseline deck took 48 of the 52 bands the
non-self-consistent run produced, excluding the four semicore bands (``exclude_bands =
1-4``). The model named explicit Sr s+p and O s+p projectors and lowered ``num_bands``
to 26. Two things then went wrong, and neither was the model's request:

- the deck writer dropped ``exclude_bands = 1-4``. It reads the upper end of an
  exclusion range as the parent's band count, which is true only for an exclusion of
  the *top* bands; for a bottom exclusion it read "4", and deleted the line;
- pw2wannier90 wrote the overlaps over all 52 bands, the deck asked for 26, and the
  loop refused to run Wannier90 by raising an error that ended the run after 2 of its 8
  attempts. The run was recorded ``not certified`` for a deck nobody could have run.

The arithmetic is fixed by the parent: with ``N_raw`` bands stored and the exclusion
set ``E``, every member of ``E`` lies in ``1..N_raw`` and ``num_bands == N_raw - |E|``.
A deck that breaks it is information for the next proposal -- the loop records the
attempt as refused, spends the budget on it, and hands the message back. A provider
that stops answering, an executable that cannot be launched, or a step that runs out
of time is not a deck, and keeps the terminal reason it always had.

Runs recorded before this change keep the exits they were recorded with; nothing here
re-reads them.
"""

from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest

from autowan.backends.qe import QeBackend
from autowan.execution import (
    CalculationError,
    DeckRefused,
    StepResult,
    WannierRunner,
    check_band_count,
)
from autowan.inputs import (
    read_construction,
    read_excluded_bands,
    read_win,
    write_derived,
)
from autowan.orchestrator import RunConfig
from autowan.providers import ModelResponse, ProviderCreditsExhausted
from autowan.qe_outputs import ParentCalculation
from autowan.workflow import CALCULATION_FAILED, WorkflowConfig, wannierize

RAW_BANDS = 52
SEED = "SrO_A"

#: The baseline deck of the 2026-10-03 run, cut down to the keys the derivation and
#: the arithmetic read. The cell, k-points and path blocks are carried verbatim by the
#: writer and play no part here.
SRO_WIN = """\
auto_projections = .true.
conv_tol =   2.0000000000d-07
conv_window = 3
dis_conv_tol =   2.0000000000d-07
dis_froz_max =   1.3307467735d+01
dis_froz_proj = .true.
dis_num_iter = 4000
dis_proj_max =   9.5000000000d-01
dis_proj_min =   1.0000000000d-02
exclude_bands = 1-4
fermi_energy =   8.9973000000d+00
mp_grid = 2, 2, 2
num_bands = 48
num_iter = 4000
num_wann = 13
write_hr = .true.
begin atoms_cart
ang
Sr         0.0000000000       0.0000000000       0.0000000000
O        -2.5797389032       2.5797389032       2.5797389032
end atoms_cart
begin projections
end projections
"""

SRO_PW2WAN = """\
&inputpp
  outdir = './'
  prefix = 'aiida'
  seedname = 'SrO_A'
  write_amn = .true.
  write_mmn = .true.
  atom_proj = .true.
/
"""

#: The same deck built by SMV: the energy window alone (placed just above the occupied
#: manifold of the synthetic parent, so the frozen states fit the functions), PAO
#: projections. Naming projectors is an SMV knob; on the PDWF deck above it is refused
#: as a method change (autowan item 4 decision 1).
SRO_SMV_WIN = (
    "\n".join(
        line
        for line in SRO_WIN.replace(
            "dis_froz_max =   1.3307467735d+01", "dis_froz_max =   0.5"
        ).splitlines()
        if not line.startswith(("dis_froz_proj", "dis_proj_"))
    )
    + "\n"
)

PROJECTORS = [
    {"species": "Sr", "orbitals": ["s", "p"]},
    {"species": "O", "orbitals": ["s", "p"]},
]


def _sro_a_action(old_bands=48, new_bands=26, with_projectors=True):
    """The action the model returned on 2026-10-03, as the provider delivered it."""
    changes = []
    if with_projectors:
        changes.append(
            {
                "parameter": "projectors",
                "old": None,
                "new": PROJECTORS,
                "reason": "a balanced set of s and p functions on both atoms",
            }
        )
    if new_bands is not None:
        changes.append(
            {
                "parameter": "num_bands",
                "old": old_bands,
                "new": new_bands,
                "reason": "fewer bands for fewer functions",
            }
        )
    return {
        "action": "modify_parameters",
        "diagnosis": "the projector set mixes tight and diffuse functions",
        "evidence": [CALCULATION_FAILED],
        "targets": [CALCULATION_FAILED],
        "expected_effect": "the occupied manifold is represented",
        "confidence": 0.85,
        "changes": changes,
    }


def _change(parameter, old, new):
    return {
        "action": "modify_parameters",
        "diagnosis": "the last deck could not be run as written",
        "evidence": [CALCULATION_FAILED],
        "targets": [CALCULATION_FAILED],
        "expected_effect": "a deck the stored overlaps can satisfy",
        "confidence": 0.5,
        "changes": [
            {"parameter": parameter, "old": old, "new": new, "reason": "see message"}
        ],
    }


def _sro_parent(raw_bands=RAW_BANDS):
    """A gapped Sr+O parent with ``raw_bands`` bands, eight of them occupied."""
    n_k = 2
    occupied = np.linspace(-20.0, -1.0, 8)
    empty = np.linspace(1.0, 60.0, raw_bands - 8)
    bands = np.concatenate([occupied, empty])
    eigenvalues = np.tile(bands, (1, n_k, 1)).astype(float)
    occupations = (eigenvalues <= 0.0).astype(float)
    return ParentCalculation(
        eigenvalues_ev=eigenvalues,
        occupations=occupations,
        kpoints_cart=np.zeros((n_k, 3)),
        fermi_energy_ev=0.0,
        lattice_ang=np.eye(3) * 5.16,
        alat_ang=5.16,
        species=("Sr", "O"),
        positions_frac=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        n_symmetries=48,
        n_atomic_wfc=None,
        converged=None,
        calculation="nscf",
    )


class _StoredParentQe(QeBackend):
    """The QE backend with the parent read from memory instead of an XML file."""

    def __init__(self, parent):
        self._parent = parent

    def read_parent(self, source):
        return self._parent


class _OfflineEngine(WannierRunner):
    """pw2wannier90 and Wannier90 as the loop sees them, with nothing launched.

    The overlap step writes an ``.amn`` header the way pw2wannier90 does: over every
    stored band the deck does not exclude, with one projection per declared function.
    The localisation step returns a non-zero exit, so a deck that reaches it is
    recorded as an ordinary failed calculation -- which is how the tests tell a deck
    that ran from a deck that was refused.
    """

    def __init__(self, run_dir, raw_bands=RAW_BANDS, projections=None):
        super().__init__(run_dir=run_dir, wannier90="wannier90.x")
        self.raw_bands = raw_bands
        self.projections = dict(projections or {})
        self.overlaps_built: list[str] = []
        self.localised: list[str] = []

    def _run(self, step, command, **_):
        seed, _, kind = step.rpartition(":")
        if kind == "nnkp":
            return StepResult(step, 0, 0.0)
        if kind == "overlaps":
            win = self.run_dir / f"{seed}.win"
            excluded = read_excluded_bands(win)
            declared = int(read_win(win)["num_wann"])
            bands = self.raw_bands - len(excluded)
            nproj = self.projections.get(seed, declared)
            (self.run_dir / f"{seed}.amn").write_text(
                f" offline overlaps\n{bands:12d}{8:12d}{nproj:12d}\n"
            )
            for ext in ("mmn", "eig"):
                (self.run_dir / f"{seed}.{ext}").write_text("offline\n")
            self.overlaps_built.append(seed)
            return StepResult(step, 0, 0.0)
        if kind == "localise":
            self.localised.append(seed)
            return StepResult(step, 1, 0.0, stderr_tail="offline: wannier90 not run")
        raise AssertionError(f"unexpected step {step}")


class _ScriptedModel:
    """A provider that answers from a list, then stops as an exhausted account does."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.payloads: list[str] = []

    def __call__(self, model):
        return self

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        self.payloads.append(payload_text)
        if not self.answers:
            raise ProviderCreditsExhausted("scripted model has no further answers")
        return ModelResponse(content=self.answers.pop(0), model="scripted", raw_text="")


def _sro_deck(run_dir: Path, smv: bool = False) -> Path:
    (run_dir / f"{SEED}.win").write_text(SRO_SMV_WIN if smv else SRO_WIN)
    (run_dir / f"{SEED}.pw2wan").write_text(SRO_PW2WAN)
    return run_dir


def _loop(tmp_path, model, runner=None, run_config=None, parent=None, smv=True):
    _sro_deck(tmp_path, smv=smv)
    runner = runner or _OfflineEngine(tmp_path)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "unused.xml",
        seed=SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(run_config=run_config or RunConfig(), draw_figures=False),
        backend=_StoredParentQe(parent or _sro_parent()),
    )
    return result, runner


# ---------------------------------------------------------------------------
# question 1: what the derivation keeps
# ---------------------------------------------------------------------------


def test_the_sro_a_action_keeps_the_semicore_exclusion_it_never_named(tmp_path):
    """Regression: the bottom exclusion was read as a parent band count and deleted."""
    source = read_construction(_sro_deck(tmp_path, smv=True), SEED)
    write_derived(
        source,
        tmp_path,
        f"{SEED}_r1",
        {"projectors": PROJECTORS, "num_bands": 26},
        species_counts={"Sr": 1, "O": 1},
    )
    win = tmp_path / f"{SEED}_r1.win"
    assert read_excluded_bands(win) == (1, 2, 3, 4)
    values = read_win(win)
    assert int(values["num_bands"]) == 26
    assert int(values["num_wann"]) == 8  # s+p on two atoms


def test_a_band_count_change_alone_carries_a_bottom_exclusion_verbatim(tmp_path):
    source = read_construction(_sro_deck(tmp_path), SEED)
    write_derived(source, tmp_path, f"{SEED}_r1", {"num_bands": 40})
    text = (tmp_path / f"{SEED}_r1.win").read_text()
    assert read_excluded_bands(tmp_path / f"{SEED}_r1.win") == (1, 2, 3, 4)
    assert text.count("exclude_bands") == 1


def test_a_projector_change_alone_keeps_the_exclusion_and_the_band_count(tmp_path):
    source = read_construction(_sro_deck(tmp_path, smv=True), SEED)
    write_derived(
        source,
        tmp_path,
        f"{SEED}_r1",
        {"projectors": PROJECTORS},
        species_counts={"Sr": 1, "O": 1},
    )
    win = tmp_path / f"{SEED}_r1.win"
    assert read_excluded_bands(win) == (1, 2, 3, 4)
    assert int(read_win(win)["num_bands"]) == 48
    assert int(read_win(win)["num_wann"]) == 8


def test_naming_projectors_on_the_pdwf_deck_is_refused_not_switched(tmp_path):
    """Once pinned as policy, now refused: naming projectors used to drop the whole
    projectability scheme silently (``AUTOMATIC_PROJECTION_WIN``). Projectors are an
    SMV knob; the PDWF deck stays PDWF and no deck is written."""
    from autowan.inputs import InputError

    source = read_construction(_sro_deck(tmp_path), SEED)
    with pytest.raises(InputError, match="SMV knob"):
        write_derived(
            source,
            tmp_path,
            f"{SEED}_r1",
            {"projectors": PROJECTORS, "num_bands": 26},
            species_counts={"Sr": 1, "O": 1},
        )
    assert not (tmp_path / f"{SEED}_r1.win").exists()


def test_naming_projectors_on_the_smv_deck_replaces_the_pao_source_and_keeps_the_window(
    tmp_path,
):
    source = read_construction(_sro_deck(tmp_path, smv=True), SEED)
    write_derived(
        source,
        tmp_path,
        f"{SEED}_r1",
        {"projectors": PROJECTORS, "num_bands": 26},
        species_counts={"Sr": 1, "O": 1},
    )
    text = (tmp_path / f"{SEED}_r1.win").read_text()
    assert "auto_projections" not in text
    assert "dis_froz_max" in text
    pw2wan = (tmp_path / f"{SEED}_r1.pw2wan").read_text()
    assert "atom_proj" not in pw2wan


def test_a_change_that_names_no_projectors_keeps_the_projectability_window(tmp_path):
    source = read_construction(_sro_deck(tmp_path), SEED)
    write_derived(source, tmp_path, f"{SEED}_r1", {"num_iter": 8000})
    text = (tmp_path / f"{SEED}_r1.win").read_text()
    for kept in (
        "dis_froz_proj",
        "dis_proj_min",
        "dis_proj_max",
        "exclude_bands = 1-4",
    ):
        assert kept in text, kept


# ---------------------------------------------------------------------------
# question 2: the band-count arithmetic against the stored parent
# ---------------------------------------------------------------------------


def _win(tmp_path, body, name="deck.win"):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_the_sro_baseline_arithmetic_passes(tmp_path):
    """Valid control: 52 stored, 1-4 excluded, 48 represented."""
    check_band_count(
        _win(tmp_path, "num_wann = 13\nnum_bands = 48\nexclude_bands = 1-4\n"),
        RAW_BANDS,
    )


def test_a_recognised_top_exclusion_permits_a_reduction(tmp_path):
    check_band_count(
        _win(tmp_path, "num_wann = 8\nnum_bands = 26\nexclude_bands = 27-52\n"),
        RAW_BANDS,
    )


@pytest.mark.parametrize(
    "body",
    [
        "num_wann = 8\nnum_bands = 26\nexclude_bands = 1-4\n",  # the SrO A deck as fixed
        "num_wann = 8\nnum_bands = 26\n",  # the SrO A deck as derived on 2026-10-03
    ],
    ids=["with-exclusion", "exclusion-dropped"],
)
def test_the_sro_a_band_count_is_refused_with_the_arithmetic(tmp_path, body):
    with pytest.raises(DeckRefused) as excinfo:
        check_band_count(_win(tmp_path, body), RAW_BANDS)
    message = str(excinfo.value)
    assert "52" in message and "26" in message, message
    assert "num_bands" in message


def test_the_refusal_names_the_band_count_that_would_fit(tmp_path):
    with pytest.raises(DeckRefused) as excinfo:
        check_band_count(
            _win(tmp_path, "num_wann = 8\nnum_bands = 26\nexclude_bands = 1-4\n"),
            RAW_BANDS,
        )
    assert "48" in str(excinfo.value)


def test_an_exclusion_beyond_the_stored_bands_is_refused(tmp_path):
    with pytest.raises(DeckRefused, match="exclude_bands"):
        check_band_count(
            _win(tmp_path, "num_wann = 8\nnum_bands = 47\nexclude_bands = 1-4, 53\n"),
            RAW_BANDS,
        )


def test_a_deck_refusal_is_a_calculation_error_for_every_existing_caller():
    assert issubclass(DeckRefused, CalculationError)


# ---------------------------------------------------------------------------
# the engine's own refusal, after pw2wannier90
# ---------------------------------------------------------------------------


def _present_overlaps(run_dir, seed, bands, nproj):
    (run_dir / f"{seed}.amn").write_text(f" test\n{bands:12d}{8:12d}{nproj:12d}\n")
    for ext in ("mmn", "eig", "nnkp"):
        (run_dir / f"{seed}.{ext}").write_text("present\n")


def test_fresh_overlaps_that_disagree_with_their_deck_are_a_deck_refusal(tmp_path):
    (tmp_path / "d_r1.win").write_text(
        "num_wann = 8\nnum_bands = 26\nconv_window = 3\n"
    )
    (tmp_path / "d_r1.pw2wan").write_text("&inputpp\n  seedname = 'd_r1'\n/\n")
    runner = _OfflineEngine(tmp_path)  # writes overlaps over 52 bands, 8 projections
    with pytest.raises(DeckRefused) as excinfo:
        runner.build("d_r1")
    assert runner.overlaps_built == ["d_r1"], "the check follows pw2wannier90"
    assert runner.localised == [], "Wannier90 must not run on a disagreeing deck"
    assert "52" in str(excinfo.value) and "26" in str(excinfo.value)


def test_reused_overlaps_that_disagree_stay_an_infrastructure_error(tmp_path):
    """A mismatch on overlaps copied from another seed may be the copy, not the deck."""
    (tmp_path / "d_r2.win").write_text(
        "num_wann = 8\nnum_bands = 26\nconv_window = 3\n"
    )
    _present_overlaps(tmp_path, "d_r1", 52, 8)
    runner = _OfflineEngine(tmp_path)
    with pytest.raises(CalculationError) as excinfo:
        runner.build("d_r2", reuse_overlaps_from="d_r1")
    assert not isinstance(excinfo.value, DeckRefused)


def test_missing_overlaps_to_reuse_stay_an_infrastructure_error(tmp_path):
    (tmp_path / "d_r2.win").write_text(
        "num_wann = 8\nnum_bands = 48\nconv_window = 3\n"
    )
    runner = _OfflineEngine(tmp_path)
    with pytest.raises(
        CalculationError, match="no complete set of overlaps"
    ) as excinfo:
        runner.build("d_r2", reuse_overlaps_from="d_r1")
    assert not isinstance(excinfo.value, DeckRefused)


# ---------------------------------------------------------------------------
# the loop: refused, fed back, and the run goes on
# ---------------------------------------------------------------------------


def test_the_sro_a_action_is_a_refused_attempt_and_the_loop_goes_on(tmp_path):
    model = _ScriptedModel([_sro_a_action(), _change("num_bands", 26, 48)])
    result, runner = _loop(tmp_path, model)
    attempts = result.run.attempts

    refused = attempts[1]
    assert refused.status == "deck_refused"
    assert refused.action is not None, "the refused attempt keeps the action it ran"
    assert "52" in refused.calculation_error and "26" in refused.calculation_error
    # refused before pw2wannier90 was spent on it
    assert f"{SEED}_r1" not in runner.overlaps_built
    assert read_excluded_bands(tmp_path / f"{SEED}_r1.win") == (1, 2, 3, 4)

    # the run did not end there: the next proposal was asked for and executed
    assert len(attempts) >= 3
    corrected = attempts[2]
    assert corrected.status != "deck_refused"
    assert f"{SEED}_r2" in runner.overlaps_built
    assert f"{SEED}_r2" in runner.localised


def test_the_request_after_a_refusal_carries_the_refusal_message(tmp_path):
    model = _ScriptedModel([_sro_a_action(), _change("num_bands", 26, 48)])
    result, _ = _loop(tmp_path, model)
    message = result.run.attempts[1].calculation_error
    assert message
    assert len(model.payloads) >= 2
    assert message not in model.payloads[0]
    assert message in model.payloads[1]


def test_an_engine_refusal_after_pw2wannier90_is_a_refused_attempt(tmp_path):
    """Overlaps built for this deck that disagree with it are the deck's refusal.

    The projector change rebuilds the overlaps, the band count is untouched (so the
    arithmetic before pw2wannier90 has nothing to say), and the overlaps come back with
    a projection count the deck does not declare. That is refused after pw2wannier90,
    recorded as a refused attempt, and the next proposal is asked for. The refused
    deck's overlaps are not inherited by the next attempt.
    """
    model = _ScriptedModel(
        [
            _sro_a_action(new_bands=None),  # projectors only
            _change("num_iter", 4000, 6000),
        ]
    )
    engine = _OfflineEngine(tmp_path, projections={f"{SEED}_r1": 13})
    result, runner = _loop(tmp_path, model, runner=engine)
    attempts = result.run.attempts

    assert attempts[1].status == "deck_refused"
    assert f"{SEED}_r1" in runner.overlaps_built, "refused after pw2wannier90 ran"
    assert f"{SEED}_r1" not in runner.localised
    assert "projection" in attempts[1].calculation_error
    assert len(attempts) >= 3, "the engine's refusal ended the run"
    assert attempts[1].calculation_error in model.payloads[1]
    assert attempts[2].status != "deck_refused"
    assert (
        f"{SEED}_r2" in runner.overlaps_built
    ), "the refused deck's overlaps were reused"
    assert f"{SEED}_r2" in runner.localised


def test_a_descendant_that_keeps_the_refused_band_count_is_refused_again(tmp_path):
    """Refusal is judged against the last deck that ran, not the deck just refused."""
    model = _ScriptedModel(
        [
            _sro_a_action(),
            _change("num_iter", 4000, 6000),  # leaves 26 against 1-4 in place
            _change("num_bands", 26, 48),
        ]
    )
    result, runner = _loop(tmp_path, model)
    attempts = result.run.attempts
    assert [a.status for a in attempts[1:3]] == ["deck_refused", "deck_refused"]
    assert f"{SEED}_r2" not in runner.overlaps_built
    assert len(attempts) >= 4
    assert attempts[3].status != "deck_refused"
    assert f"{SEED}_r3" in runner.overlaps_built, "an unbuilt deck is no overlap source"


@pytest.mark.parametrize(
    "run_config, expected_attempts, reason",
    [
        # the single repair stage: one cap of eight attempts in all (decision 14)
        (RunConfig(total_attempts=8), 8, "attempt budget of 8 spent"),
        (RunConfig(total_attempts=5), 5, "attempt budget of 5 spent"),
    ],
    ids=["total-of-eight", "total-of-five"],
)
def test_repeated_refusals_spend_exactly_the_registered_budget(
    tmp_path, run_config, expected_attempts, reason
):
    answers = [_sro_a_action(new_bands=26)] + [
        _change("num_bands", 26 - i, 25 - i) for i in range(10)
    ]
    model = _ScriptedModel(answers)
    result, runner = _loop(tmp_path, model, run_config=run_config)
    attempts = result.run.attempts
    assert len(attempts) == expected_attempts
    assert all(a.status == "deck_refused" for a in attempts[1:])
    assert reason in (result.run.terminal_reason or "")
    assert not result.certified
    assert runner.overlaps_built == [SEED], "no refused deck reached pw2wannier90"


# ---------------------------------------------------------------------------
# controls: what is not a deck keeps its terminal reason
# ---------------------------------------------------------------------------


def test_a_provider_stop_keeps_its_terminal_reason(tmp_path):
    model = _ScriptedModel([])  # the first request finds the account empty
    result, _ = _loop(tmp_path, model, smv=False)
    attempts = result.run.attempts
    assert len(attempts) == 1
    assert attempts[0].model_error and "no further answers" in attempts[0].model_error
    assert "stopped by the provider" in (result.run.terminal_reason or "")
    assert result.run.stopped_by_provider
    assert all(a.status != "deck_refused" for a in attempts)


def test_an_executable_that_cannot_be_launched_ends_the_run_as_before(tmp_path):
    _sro_deck(tmp_path)
    model = _ScriptedModel([_change("num_bands", 48, 47)])
    runner = WannierRunner(
        run_dir=tmp_path,
        wannier90=str(tmp_path / "no-such-wannier90.x"),
        pw2wannier90=str(tmp_path / "no-such-pw2wannier90.x"),
        mpi_launcher=(),
    )
    result, _ = _loop(tmp_path, model, runner=runner)
    attempts = result.run.attempts
    assert len(attempts) == 1, "an infrastructure failure must still end the run"
    assert attempts[0].status == "calculation_failed"
    assert "could not be launched" in attempts[0].calculation_error
    assert model.payloads == [], "no model is consulted about a missing executable"


def test_a_step_that_runs_out_of_time_ends_the_run_as_before(tmp_path):
    _sro_deck(tmp_path)
    slow = tmp_path / "slow-wannier90.sh"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(slow.stat().st_mode | stat.S_IXUSR)
    model = _ScriptedModel([_change("num_bands", 48, 47)])
    runner = WannierRunner(
        run_dir=tmp_path,
        wannier90=str(slow),
        pw2wannier90=str(slow),
        mpi_launcher=(),
        timeout_s=0.3,
    )
    result, _ = _loop(tmp_path, model, runner=runner)
    attempts = result.run.attempts
    assert len(attempts) == 1
    assert attempts[0].status == "calculation_failed"
    assert "exceeded" in attempts[0].calculation_error
    assert model.payloads == []


def test_runs_recorded_before_the_change_are_not_relabelled():
    """The record vocabulary only grows: an attempt finished as before reads as before."""
    from autowan.orchestrator import Run

    run = Run("SrO_A")
    attempt = run.start_attempt({"num_bands": "48"})
    run.finish_attempt(
        attempt,
        calculation_error=(
            "SrO_A_r1.amn was built over 52 bands but SrO_A_r1.win asks for 26"
        ),
    )
    assert attempt.status == "calculation_failed"
    assert run.to_dict()["attempts"][0]["status"] == "calculation_failed"
