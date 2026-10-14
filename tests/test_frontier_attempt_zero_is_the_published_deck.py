"""Attempt 0 is the published frontier deck, byte for byte; only repaired decks are ours.

The workflow used to prepend ``conv_window = 3`` to every deck it ran, the frontier's
included, and the VASP frontier's overlap step wrote ``NSW = 0`` and ``IBRION = -1``
into its INCAR. Neither is in the protocol. A frontier that runs with our additions is a
modified frontier, and the comparison the campaign makes -- autowan against the
published method on the same material -- is then against something nobody published.

The keys the workflow injects (``execution.REPAIR_DECK_PREPARATION``) now reach repaired
decks only, and the VASP frontier's INCAR is the parent's overlaid with the deck's tags
and nothing else.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import repair_fixtures as fx

from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.execution import (
    FRONTIER_PARENT_TAGS_DROPPED,
    VaspWannierRunner,
)
from autowan.backends.vasp.inputs import (
    overlap_step_incar_tags,
    read_incar,
    scdm_frontier_deck,
)
from autowan.execution import (
    REPAIR_DECK_PREPARATION,
    StepResult,
    WannierRunner,
)
from autowan.orchestrator import RunConfig
from autowan.workflow import WorkflowConfig, wannierize

SRO_VASP = Path(__file__).parent / "fixtures" / "vasp" / "SrO"


class _RecordingQeEngine(WannierRunner):
    """Runs the real ``build`` and records the deck each step was handed; runs nothing."""

    def __init__(self, run_dir):
        super().__init__(run_dir=run_dir, wannier90="wannier90.x")
        self.decks: list[tuple[str, str]] = []

    def _record(self, seed):
        self.decks.append((seed, (self.run_dir / f"{seed}.win").read_text()))

    def run_overlaps(self, seed):
        self._record(seed)
        return [StepResult(f"{seed}:overlaps", 1, 0.0, stderr_tail="offline")]

    def reuse_overlaps(self, source_seed, target_seed):
        self._record(target_seed)
        return ["amn", "mmn", "eig"]

    def run_localisation(self, seed):
        return StepResult(f"{seed}:localise", 1, 0.0, stderr_tail="offline")


def _qe_frontier_without_convergence_test(run_dir: Path) -> bytes:
    fx.write_deck(run_dir)
    win = run_dir / f"{fx.SEED}.win"
    win.write_text(
        "".join(
            line
            for line in win.read_text().splitlines(keepends=True)
            if not line.startswith("conv_window")
        )
    )
    return win.read_bytes()


def test_the_injected_keys_are_listed():
    assert REPAIR_DECK_PREPARATION == ("conv_window",)
    assert set(FRONTIER_PARENT_TAGS_DROPPED) >= {"NSW", "IBRION", "ICHARG", "NUM_WANN"}


def test_qe_attempt_zero_runs_the_frontier_deck_exactly_and_a_repair_gets_the_test(
    tmp_path,
):
    published = _qe_frontier_without_convergence_test(tmp_path)
    runner = _RecordingQeEngine(tmp_path)
    model = fx.ToolScript([([], fx.answer(changes=[("num_iter", 5000)]))])
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "x.xml",
        seed=fx.SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=3), draw_figures=False
        ),
        backend=fx.StoredParentQe(fx.parent()),
    )
    seeds = [seed for seed, _ in runner.decks]
    assert seeds[0] == fx.SEED and len(seeds) == 2, seeds
    # attempt 0: exactly the deck the frontier wrote, before and after the run
    assert runner.decks[0][1].encode() == published
    assert (tmp_path / f"{fx.SEED}.win").read_bytes() == published
    assert "conv_window" not in result.run.attempts[0].parameters
    # attempt 1: ours, so it carries the convergence test
    repaired = runner.decks[1][1]
    assert "conv_window = 3" in repaired
    assert str(result.run.attempts[1].parameters.get("conv_window")) == "3"


def test_qe_frontier_deck_that_sets_its_own_window_is_left_alone(tmp_path):
    fx.write_deck(tmp_path)
    published = (tmp_path / f"{fx.SEED}.win").read_bytes()
    runner = _RecordingQeEngine(tmp_path)
    wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "x.xml",
        seed=fx.SEED,
        runner=runner,
        config=WorkflowConfig(draw_figures=False),
        backend=fx.StoredParentQe(fx.parent()),
    )
    assert runner.decks[0][1].encode() == published


@pytest.fixture
def vasp_frontier(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    shutil.copytree(SRO_VASP, parent)
    for name in ("CHGCAR", "POTCAR", "WAVECAR"):
        (parent / name).write_text("stub\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    scdm_frontier_deck(parent / "vasprun.xml", run_dir, "SrO")
    scdm_frontier_deck(parent / "vasprun.xml", reference, "SrO")
    return parent, run_dir, (reference / "SrO.win").read_bytes()


def test_vasp_attempt_zero_equals_the_scdm_frontier_builder_output(
    vasp_frontier, monkeypatch
):
    parent, run_dir, built = vasp_frontier
    runner = VaspWannierRunner(run_dir=run_dir, parent_dir=parent, vasp="vasp_std")
    seen = {}

    def fake_run(step, command, *, cwd=None, **_):
        seen["win"] = (run_dir / "SrO.win").read_bytes()
        seen["incar"] = (cwd / "INCAR").read_text()
        return StepResult(step, 1, 0.0, stderr_tail="offline")

    monkeypatch.setattr(runner, "_run", fake_run)
    result = wannierize(
        run_dir=run_dir,
        parent_xml=parent / "vasprun.xml",
        seed="SrO",
        runner=runner,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
    )
    assert len(result.run.attempts) == 1
    # the deck the overlap step ran is the builder's output, byte for byte
    assert seen["win"] == built
    assert (run_dir / "SrO.win").read_bytes() == built
    assert "conv_window" not in result.run.attempts[0].parameters
    # the INCAR is the parent's, overlaid with the deck's tags, and nothing else
    staged = read_incar(seen["incar"].split("WANNIER90_WIN")[0])
    expected = read_incar((parent / "INCAR").read_text())
    for key in FRONTIER_PARENT_TAGS_DROPPED:
        expected.pop(key, None)
    expected.update(overlap_step_incar_tags(built.decode()))
    assert staged == expected
    assert "NSW" not in staged and "IBRION" not in staged
    body = seen["incar"].split("WANNIER90_WIN", 1)[1]
    assert "conv_window" not in body


def test_the_frontier_overlap_step_refuses_a_rank_count_other_than_its_npar(
    vasp_frontier,
):
    """VASP's Wannier interface needs NCORE = 1, so NPAR is the rank count (smoke 20359)."""
    from autowan.backends.vasp.execution import launcher_ranks
    from autowan.execution import CalculationError

    parent, run_dir, _ = vasp_frontier
    assert launcher_ranks(("mpirun", "-np", "8")) == 8
    assert launcher_ranks(("srun",)) is None
    runner = VaspWannierRunner(
        run_dir=run_dir, parent_dir=parent, mpi_launcher=("mpirun", "-np", "8")
    )
    with pytest.raises(CalculationError, match="npar=8"):
        runner.run_overlaps("SrO")
