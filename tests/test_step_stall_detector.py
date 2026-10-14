"""No fixed wall-time limit; a step is stopped only when it stops making progress.

User decision 2026-10-08: some materials genuinely take long, so the package carries no
fixed step limit, and a limit is acceptable only with a dynamic, physically scaled
criterion. The V3Si sparse_tb v1 run is the case that drove it: its attempt-2
localisation (SMV, 62 functions, 224 bands, 7x7x7) was in its 2012th disentanglement
iteration, one every ~9 s, when the campaign's 6 h limit stopped it and ended the run.

Covered here:

* the rule itself (``progress.ProgressMonitor``) on a scripted clock: a stall is caught,
  a slow but progressing loop is not, the floor, the reference bound before any
  iteration, and no limit at all without one;
* the readers of Wannier90, pw2wannier90 and VASP output;
* real subprocesses: a stalled localisation and a stalled overlap step are stopped (with
  their process group), a slow but progressing one runs to the end, a silent start-up
  with no reference is left alone;
* the loop: a stalled proposed deck is a refused attempt (``stalled``) whose reason
  reaches the model, it costs one attempt, and the run continues (the V3Si path); a
  stalled frontier is recorded as the frontier's, with no gate verdict;
* infrastructure stops (SIGTERM from SLURM): recorded, not a result, resumable, and the
  interrupted attempt is rebuilt without asking the model again.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import threading
import time
from pathlib import Path

import pytest
from test_repair_band_count_refusal import (
    SEED,
    _change,
    _loop,
    _OfflineEngine,
    _ScriptedModel,
    _sro_deck,
    _sro_parent,
    _StoredParentQe,
)

from autowan import workflow
from autowan.execution import (
    ConstructionRunner,
    InfrastructureStop,
    StepResult,
    StepStalled,
    WannierRunner,
    stop_on_termination_signals,
)
from autowan.exits import exit_row
from autowan.orchestrator import RunConfig
from autowan.progress import (
    BOUNDARY,
    ITERATION,
    NEUTRAL,
    REFERENCE_FACTOR,
    STALL_FACTOR,
    STALL_FLOOR_S,
    ActivitySource,
    DeckSize,
    LineSource,
    ProgressMonitor,
    Stall,
    StallPolicy,
    pw2wannier90_line,
    size_scale,
    vasp_line,
    wannier90_line,
)

# ---------------------------------------------------------------------------
# the rule, on a scripted clock
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Script:
    """A source that returns whatever events the test queues for the next poll."""

    def __init__(self):
        self.queue: list[tuple[str, int]] = []

    def prime(self):
        pass

    def poll(self):
        out, self.queue = self.queue, []
        return out


def _monitor(policy=None, reference_s=None, scale=None):
    clock, source = _Clock(), _Script()
    monitor = ProgressMonitor(
        "wan_r2:localise",
        [source],
        policy or StallPolicy(),
        reference_s=reference_s,
        scale=scale,
        clock=clock,
    )
    return monitor, clock, source


def _iterate(monitor, clock, source, n, every_s):
    for _ in range(n):
        clock.t += every_s
        source.queue.append((ITERATION, 1))
        monitor.observe()
        assert monitor.stalled() is None


def test_the_constants_are_the_recorded_calibration():
    assert STALL_FACTOR == 20.0
    assert STALL_FLOOR_S == 300.0
    assert REFERENCE_FACTOR == 10.0
    assert StallPolicy().to_dict() == {
        "factor": 20.0,
        "floor_s": 300.0,
        "reference_factor": 10.0,
        "poll_s": 5.0,
    }


def test_a_loop_that_stops_is_caught_after_n_median_intervals():
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 50, every_s=36.0)  # V3Si v2 on a loaded node
    clock.t += 20 * 36.0 - 1
    monitor.observe()
    assert monitor.stalled() is None
    clock.t += 2
    stall = monitor.stalled()
    assert stall is not None and stall.rule == "loop"
    assert stall.limit_s == 720.0 and stall.median_interval_s == 36.0
    assert stall.iterations == 50
    message = stall.message()
    assert message.startswith("wan_r2:localise stalled: no progress for 721 s, 20x")
    assert "the median iteration time" in message


def test_a_slow_but_progressing_loop_is_never_stopped():
    """The V3Si v1 localisation: 9.2 s per iteration, 2012 iterations in 6 h."""
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 2012, every_s=9.2)
    # node load doubles, then triples, the per-iteration time; still progressing
    _iterate(monitor, clock, source, 200, every_s=18.4)
    _iterate(monitor, clock, source, 200, every_s=27.6)
    # a one-off pause well above the median, inside the limit
    clock.t += 15 * 9.2
    source.queue.append((ITERATION, 1))
    monitor.observe()
    assert monitor.stalled() is None
    assert monitor.iterations == 2413


def test_the_floor_protects_fast_loops_from_a_pause():
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 100, every_s=0.05)
    clock.t += 96.0  # the longest single interval in the calibration set
    assert monitor.stalled() is None
    clock.t += STALL_FLOOR_S - 96.0 + 1
    stall = monitor.stalled()
    assert stall is not None and stall.limit_s == STALL_FLOOR_S


def test_any_output_is_progress_even_without_an_iteration():
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 10, every_s=10.0)
    for _ in range(5):
        clock.t += 250.0
        source.queue.append(("activity", 0))
        monitor.observe()
        assert monitor.stalled() is None


def test_before_any_iteration_without_a_reference_there_is_no_limit():
    monitor, clock, source = _monitor()
    clock.t += 1e6  # reading a very large .mmn, say
    assert monitor.limit() == (None, "none")
    assert monitor.stalled() is None


def test_before_any_iteration_attempt_zeros_scaled_time_bounds_the_wait():
    monitor, clock, source = _monitor(reference_s=120.0 * 8.0, scale=8.0)
    assert monitor.limit() == (REFERENCE_FACTOR * 960.0, "reference")
    clock.t += 9600.0
    assert monitor.stalled() is None
    clock.t += 1.0
    stall = monitor.stalled()
    assert stall is not None and stall.rule == "reference"
    assert stall.reference_s == 120.0 and stall.scale == 8.0
    assert "outside an iteration loop" in stall.message()
    assert "attempt 0's 120 s" in stall.message()


def test_a_tiny_reference_still_gets_the_floor():
    monitor, clock, source = _monitor(reference_s=2.0, scale=1.0)
    assert monitor.limit() == (STALL_FLOOR_S, "reference")


def test_after_a_loop_ends_its_median_no_longer_applies():
    """Between the two loops, and after them, there is no iteration to time."""
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 30, every_s=1.0)
    clock.t += 1.0
    source.queue.append((BOUNDARY, 0))  # "Final Omega_I", "Time to disentangle"
    monitor.observe()
    clock.t += 10 * STALL_FLOOR_S  # writing a 3.8 GB result, with no reference
    assert monitor.stalled() is None


def test_a_second_loop_uses_its_own_median_once_it_has_one():
    monitor, clock, source = _monitor()
    _iterate(monitor, clock, source, 30, every_s=1.0)  # disentanglement
    source.queue.append((BOUNDARY, 0))
    monitor.observe()
    _iterate(monitor, clock, source, 30, every_s=60.0)  # wannierisation, slower
    assert monitor.median_interval() == 60.0
    clock.t += 1199.0
    assert monitor.stalled() is None


# ---------------------------------------------------------------------------
# reading the engines' output
# ---------------------------------------------------------------------------

WOUT = """\
 |  Iter     Omega_I(i-1)      Omega_I(i)      Delta (frac.)    Time   |<-- DIS
 +---------------------------------------------------------------------+<-- DIS
       1      38.95530383      38.75524634       5.162E-03      43.21    <-- DIS
       2      38.75524634      38.65510517       2.590E-03      52.41    <-- DIS

             <<<      Delta < 8.000E-07  over  3 iterations     >>>
        Final Omega_I    38.05399733 (Ang^2)
 +--------------------------------------------------------------------+<-- CONV
 | Iter  Delta Spread     RMS Gradient      Spread (Ang^2)      Time  |<-- CONV
 +--------------------------------------------------------------------+<-- CONV
      0     0.152E+02     0.0000000000       15.2146758095       0.01  <-- CONV
        O_D=      0.0000000 O_OD=      0.1300387 O_TOT=     15.2146758 <-- SPRD
 ------------------------------------------------------------------------------
 Cycle:      1
  WF centre and spread    1  ( -0.866253,  1.973841,  1.973841 )     1.97707588
  Sum of centres and spreads ( -0.000000,  0.000000,  0.000000 )    15.21467581
      1    -0.420E-01     0.6578474089       15.1726758095       0.02  <-- CONV
 Delta: O_D=  0.0000000E+00 O_OD= -0.4200000E-01 O_TOT= -0.4200000E-01 <-- DLTA
 Final State
"""


def test_the_wannier90_reader():
    kinds = [wannier90_line(line)[0] for line in WOUT.splitlines()]
    assert kinds == [
        NEUTRAL,
        NEUTRAL,
        ITERATION,
        ITERATION,
        NEUTRAL,
        BOUNDARY,  # <<< Delta < ... >>>
        BOUNDARY,  # Final Omega_I
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        ITERATION,
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        NEUTRAL,
        ITERATION,
        NEUTRAL,
        BOUNDARY,  # Final State
    ]


def test_the_pw2wannier90_and_vasp_readers():
    assert pw2wannier90_line("      11      12      13") == (ITERATION, 3)
    assert pw2wannier90_line("  MMN calculated")[0] == BOUNDARY
    assert pw2wannier90_line("  Number of local k points =      343")[0] == BOUNDARY
    assert vasp_line("DAV:   1    -0.4183E+03   -0.418E+03   -0.168E+04  7744") == (
        ITERATION,
        1,
    )
    assert vasp_line("DIA:   2    -0.400768722976E+02   -0.13642E-11")[0] == ITERATION
    assert vasp_line(" Computing MMN (overlap matrix elements)")[0] == BOUNDARY


def test_a_line_source_skips_an_earlier_run_and_follows_a_rewritten_file(tmp_path):
    wout = tmp_path / "wan.wout"
    wout.write_text(WOUT)  # left by an earlier run of the same seedname
    source = LineSource(wout, wannier90_line)
    source.prime()
    assert source.poll() == []
    wout.unlink()
    wout.write_text(WOUT.splitlines(keepends=True)[2])  # the new run starts over
    assert (ITERATION, 1) in source.poll()


def test_pw2wannier90_k_points_count_as_they_are_printed(tmp_path):
    out = tmp_path / "pw2wan.out"
    out.write_text("  Number of local k points =      343\n       1       2")
    source = LineSource(out, pw2wannier90_line, count_partial=True)
    events = source.poll()
    assert (BOUNDARY, 0) in events and (ITERATION, 2) in events
    with out.open("a") as handle:
        handle.write("       3\n       4")
    events = source.poll()
    assert [c for k, c in events if k == ITERATION] == [1, 1]


def test_an_activity_source_sees_a_growing_product(tmp_path):
    product = tmp_path / "wan.mmn"
    product.write_text("x")
    source = ActivitySource([tmp_path / "wan.*"])
    source.prime()
    assert source.poll() == []
    with product.open("a") as handle:
        handle.write("more")
    assert source.poll() == [("activity", 0)]


def test_the_size_scale_is_cubic_in_the_matrix_dimension_and_never_below_one():
    ref = DeckSize(num_bands=224, num_wann=44, num_kpts=343)
    assert size_scale(DeckSize(224, 62, 343), ref) == pytest.approx((62 / 44) ** 3)
    assert size_scale(DeckSize(112, 22, 343), ref) == 1.0
    assert size_scale(DeckSize(224, 44, 686), ref) == 2.0
    assert size_scale(DeckSize(None, None, None), ref) == 1.0
    assert DeckSize.from_values(
        {"num_bands": "224", "num_wann": "62", "mp_grid": "7, 7, 7"}
    ) == DeckSize(224, 62, 343)


# ---------------------------------------------------------------------------
# real subprocesses
# ---------------------------------------------------------------------------

FAST = StallPolicy(factor=5.0, floor_s=0.6, reference_factor=10.0, poll_s=0.05)

_DIS_LINE = (
    "{i:8d}      38.95530383      38.75524634       5.162E-03   {t:8.2f}    <-- DIS"
)


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _wannier90(tmp_path, lines, every_s, then="", pid_file=None):
    """A fake wannier90.x: writes ``lines`` DIS iterations to ``<seed>.wout``."""
    record_pid = f'echo $$ > "{pid_file}"\n' if pid_file else ""
    body = (
        record_pid
        + 'seed="$1"\n: > "$seed.wout"\n'
        + f"i=1\nwhile [ $i -le {lines} ]; do\n"
        + f"  printf '%8d      38.9      38.7       5.1E-03      1.00    <-- DIS\\n' $i"
        + ' >> "$seed.wout"\n'
        + f"  sleep {every_s}\n  i=$((i+1))\ndone\n"
        + then
    )
    return _script(tmp_path / "fake-wannier90.sh", body)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); its state says it is gone
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_the_package_has_no_fixed_step_limit(tmp_path):
    runner = ConstructionRunner(run_dir=tmp_path)
    assert runner.timeout_s is None
    assert runner.stall_policy == StallPolicy()


def test_a_fixed_limit_is_still_possible_and_warned_about(tmp_path):
    with pytest.warns(UserWarning, match="not scale-aware"):
        WannierRunner(run_dir=tmp_path, timeout_s=3600.0)


def test_a_stalled_localisation_is_stopped_with_its_process(tmp_path):
    pid_file = tmp_path / "pid"
    exe = _wannier90(tmp_path, 12, 0.05, then="sleep 600\n", pid_file=pid_file)
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    started = time.monotonic()
    with pytest.raises(StepStalled) as caught:
        runner.run_localisation("wan_r2")
    assert time.monotonic() - started < 30
    stall = caught.value.stall
    assert stall.rule == "loop" and stall.iterations == 12
    assert caught.value.kind == "localise"
    assert "wan_r2:localise stalled: no progress for" in str(caught.value)
    assert not _alive(int(pid_file.read_text()))


def test_a_slow_but_progressing_localisation_runs_to_the_end(tmp_path):
    """Longer than a fixed limit of 1.5 s would have allowed, and never idle for long."""
    exe = _wannier90(tmp_path, 10, 0.3)
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    result = runner.run_localisation("wan_r2")
    assert result.ok and result.seconds > 2.5
    assert result.progress["iterations"] == 10
    assert result.progress["median_interval_s"] > 0.2


def test_a_silent_start_up_without_a_reference_is_left_alone(tmp_path):
    """Before the first iteration and with no attempt-0 time: no limit, floor or not."""
    body = 'sleep 2\nprintf "%8d   1 2 3  1.0  <-- DIS\\n" 1 > "$1.wout"\n'
    exe = _script(tmp_path / "slow-start.sh", body)
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    assert runner.run_localisation("wan").ok


def test_a_silent_start_up_beyond_the_scaled_reference_is_stopped(tmp_path):
    (tmp_path / "wan.win").write_text(
        "num_bands = 20\nnum_wann = 10\nmp_grid = 2 2 2\n"
    )
    exe = _script(tmp_path / "hang.sh", "sleep 600\n")
    policy = StallPolicy(factor=5.0, floor_s=0.3, reference_factor=2.0, poll_s=0.05)
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=policy)
    runner.remember_reference("wan", [StepResult("wan:localise", 0, 0.2)])
    with pytest.raises(StepStalled) as caught:
        runner.run_localisation("wan")
    assert caught.value.stall.rule == "reference"
    assert caught.value.stall.limit_s == pytest.approx(0.4)


def test_a_stalled_overlap_step_is_stopped(tmp_path):
    """pw2wannier90 prints its k-points; a k-point loop that stops is caught."""
    body = 'echo "  Number of local k points =      343"\n' + "".join(
        f'printf "%8d" {k}\nsleep 0.05\n' for k in range(1, 21)
    )
    exe = _script(tmp_path / "pw2wan.sh", body + "sleep 600\n")
    runner = WannierRunner(
        run_dir=tmp_path, pw2wannier90=exe, mpi_launcher=(), stall_policy=FAST
    )
    runner.pw2wannier90_args = ()
    with pytest.raises(StepStalled) as caught:
        runner._run("wan:overlaps", [exe])
    assert caught.value.kind == "overlaps"
    assert caught.value.stall.iterations == 20


def test_the_v3si_like_step_now_completes(tmp_path):
    """The step the 6 h limit stopped: long, steady, progressing -- it now finishes.

    Scaled down: a fixed limit at a third of the step's length (the V3Si attempt-2
    localisation needed ~10 h of disentanglement against 6 h) stopped it before; the
    stall detector lets it run its iteration count out.
    """
    exe = _wannier90(tmp_path, 15, 0.2)
    with pytest.warns(UserWarning):
        fixed = WannierRunner(run_dir=tmp_path, wannier90=exe, timeout_s=1.0)
    with pytest.raises(Exception, match="exceeded 1 s"):
        fixed.run_localisation("wan_r2")
    detector = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    result = detector.run_localisation("wan_r2")
    assert result.ok and result.progress["iterations"] == 15


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def _stall(step, idle=900.0, median=30.0, iterations=400):
    return Stall(
        step=step,
        idle_s=idle,
        limit_s=600.0,
        rule="loop",
        iterations=iterations,
        median_interval_s=median,
        reference_s=None,
        scale=None,
        elapsed_s=12345.0,
        policy=StallPolicy().to_dict(),
    )


class _StallsOn(_OfflineEngine):
    def __init__(self, run_dir, stalled_step, **kw):
        super().__init__(run_dir, **kw)
        self.stalled_step = stalled_step

    def _run(self, step, command, **kw):
        if step == self.stalled_step:
            raise StepStalled(_stall(step))
        return super()._run(step, command, **kw)


def test_a_stalled_proposed_deck_is_a_refused_attempt_and_the_run_goes_on(tmp_path):
    """The V3Si path under the new rule: the stalled repair costs one attempt only."""
    model = _ScriptedModel(
        [_change("num_iter", 4000, 6000), _change("num_iter", 4000, 3000)]
    )
    result, runner = _loop(
        tmp_path, model, runner=_StallsOn(tmp_path, f"{SEED}_r1:localise")
    )
    attempts = result.run.attempts
    assert [a.status for a in attempts] == [
        "calculation_failed",
        "stalled",
        "calculation_failed",
    ]
    stalled = attempts[1]
    assert stalled.refused and stalled.stage == "repair"
    assert stalled.stalled["step"] == f"{SEED}_r1:localise"
    assert stalled.stalled["frontier"] is False
    assert stalled.calculation_error.startswith(
        f"{SEED}_r1:localise stalled: no progress for 900 s, 30x the median"
    )
    # the reason went back to the model with the next proposal
    assert "stalled: no progress for 900 s" in model.payloads[1]
    # the run did not end on it
    assert "did not complete" not in (result.run.terminal_reason or "")
    assert runner.localised == [SEED, f"{SEED}_r2"]
    row = exit_row(result.run.to_dict())
    assert row["stalled_attempts"] == f"1:{SEED}_r1:localise"


def test_a_stall_spends_exactly_one_attempt_of_the_budget(tmp_path):
    answers = [_change("num_iter", 4000, 4000 - 100 * i) for i in range(1, 10)]
    model = _ScriptedModel(answers)

    class _AlwaysStalls(_OfflineEngine):
        def _run(self, step, command, **kw):
            if step.endswith(":localise") and step != f"{SEED}:localise":
                raise StepStalled(_stall(step))
            return super()._run(step, command, **kw)

    result, _ = _loop(
        tmp_path,
        model,
        runner=_AlwaysStalls(tmp_path),
        run_config=RunConfig(total_attempts=4),
    )
    attempts = result.run.attempts
    assert len(attempts) == 4
    assert [a.status for a in attempts[1:]] == ["stalled"] * 3
    assert "attempt budget of 4" in result.run.terminal_reason


def test_a_localisation_stall_keeps_its_overlaps_for_the_next_deck(tmp_path):
    model = _ScriptedModel(
        [_change("num_iter", 4000, 6000), _change("num_iter", 4000, 3000)]
    )
    result, runner = _loop(
        tmp_path, model, runner=_StallsOn(tmp_path, f"{SEED}_r1:localise")
    )
    # attempt 1 reused attempt 0's overlaps; attempt 2 reuses the same ones again
    assert runner.overlaps_built == [SEED]
    assert [a.overlaps_from for a in result.run.attempts] == [None, SEED, SEED]


def test_a_stalled_frontier_is_recorded_as_the_frontiers_and_repair_begins(tmp_path):
    model = _ScriptedModel([_change("num_iter", 4000, 6000)])
    result, runner = _loop(
        tmp_path, model, runner=_StallsOn(tmp_path, f"{SEED}:localise")
    )
    first = result.run.attempts[0]
    assert first.index == 0 and first.stage == "automatic"
    assert first.status == "stalled" and first.refused
    assert (
        first.outcome is None
    ), "no gate verdict exists on a construction that stalled"
    assert first.stalled["frontier"] is True
    assert first.calculation_error.startswith(
        "the frontier construction (attempt 0, run as published) stalled: "
    )
    # the overlap step of attempt 0 completed, and is the reference for later ones
    assert [s["step"] for s in first.steps] == [f"{SEED}:nnkp", f"{SEED}:overlaps"]
    assert set(runner.reference_steps) == {"nnkp", "overlaps"}
    # the repair loop starts from it, and the model is told
    assert len(result.run.attempts) == 2
    assert "attempt 0, run as published) stalled" in model.payloads[0]


def test_attempt_zeros_step_times_become_the_reference(tmp_path):
    model = _ScriptedModel([])
    result, runner = _loop(tmp_path, model)
    first = result.run.attempts[0]
    assert [s["step"].rpartition(":")[2] for s in first.steps] == [
        "nnkp",
        "overlaps",
        "localise",
    ]
    # the offline localisation exits 1: a failed step is no reference
    assert set(runner.reference_steps) == {"nnkp", "overlaps"}
    assert runner.reference_steps["overlaps"]["size"] == {
        "num_bands": 48,
        "num_wann": 13,
        "num_kpts": 8,
    }


def test_a_reference_is_scaled_to_the_deck_it_bounds(tmp_path):
    _sro_deck(tmp_path)
    runner = _OfflineEngine(tmp_path)
    runner.remember_reference(SEED, [StepResult(f"{SEED}:localise", 0, 12.0)])
    bigger = (
        (tmp_path / f"{SEED}.win").read_text().replace("num_wann = 13", "num_wann = 26")
    )
    assert bigger != (tmp_path / f"{SEED}.win").read_text()
    (tmp_path / f"{SEED}_r1.win").write_text(bigger)
    seconds, scale = runner._reference_for(f"{SEED}_r1:localise")
    assert scale == pytest.approx(8.0) and seconds == pytest.approx(96.0)
    assert runner._reference_for(f"{SEED}_r1:overlaps") == (None, None)


def test_a_resumed_run_takes_the_reference_from_the_record(tmp_path):
    from autowan.orchestrator import Run

    _sro_deck(tmp_path)
    run = Run(SEED)
    attempt = run.start_attempt({"x": 1}, seed=SEED)
    attempt.steps = [{"step": f"{SEED}:localise", "returncode": 0, "seconds": 7.5}]
    runner = _OfflineEngine(tmp_path)
    workflow._prime_reference(runner, Run.from_dict(run.to_dict()))
    assert runner.reference_steps["localise"]["seconds"] == 7.5


# ---------------------------------------------------------------------------
# infrastructure stops
# ---------------------------------------------------------------------------


def _wannierize(tmp_path, model, runner, resume=False):
    return workflow.wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "unused.xml",
        seed=SEED,
        runner=runner,
        provider_factory=model,
        config=workflow.WorkflowConfig(draw_figures=False),
        backend=_StoredParentQe(_sro_parent()),
        log_path=tmp_path / "autowan_log.json",
        resume=resume,
    )


class _KilledOn(_OfflineEngine):
    def __init__(self, run_dir, step):
        super().__init__(run_dir)
        self.step = step

    def _run(self, step, command, **kw):
        if step == self.step:
            raise InfrastructureStop(
                "received SIGTERM: the job was stopped from outside"
            )
        return super()._run(step, command, **kw)


def test_an_infrastructure_stop_is_recorded_as_no_result_and_resumes(tmp_path):
    _sro_deck(tmp_path, smv=True)
    model = _ScriptedModel([_change("num_iter", 4000, 6000)])
    with pytest.raises(InfrastructureStop):
        _wannierize(tmp_path, model, _KilledOn(tmp_path, f"{SEED}_r1:localise"))
    record = json.loads((tmp_path / "autowan_log.json").read_text())
    assert record["stopped_by_infrastructure"] is True
    assert not record["stopped_by_provider"]
    assert [a["index"] for a in record["attempts"]] == [0], "the interrupted one is off"
    (stop,) = record["infrastructure_stops"]
    assert stop["interrupted_attempt"]["seed"] == f"{SEED}_r1"
    assert "not a result; resumable" in record["terminal_reason"]
    assert exit_row(record)["exit"] == "stopped by the infrastructure"
    assert len(model.payloads) == 1

    # not silently restarted
    with pytest.raises(workflow.RunStoppedByInfrastructure):
        _wannierize(tmp_path, _ScriptedModel([]), _OfflineEngine(tmp_path))

    # resumed: the interrupted deck is rebuilt, the model is not asked for it again
    again = _ScriptedModel([])
    runner = _OfflineEngine(tmp_path)
    result = _wannierize(tmp_path, again, runner, resume=True)
    attempts = result.run.attempts
    assert [a.seed for a in attempts[:2]] == [SEED, f"{SEED}_r1"]
    assert attempts[1].action["changes"][0]["new"] == 6000
    assert (
        attempts[1].model_requests
        == record["infrastructure_stops"][0]["interrupted_attempt"]["model_requests"]
    )
    assert f"{SEED}_r1" in runner.localised
    assert result.run.infrastructure_stops[-1]["resumed_at"] is not None
    assert not result.run.stopped_by_infrastructure
    assert "stopped by the provider" in result.run.terminal_reason  # no more answers


def test_an_infrastructure_stop_during_the_frontier_rebuilds_the_frontier(tmp_path):
    _sro_deck(tmp_path, smv=True)
    with pytest.raises(InfrastructureStop):
        _wannierize(
            tmp_path, _ScriptedModel([]), _KilledOn(tmp_path, f"{SEED}:localise")
        )
    record = json.loads((tmp_path / "autowan_log.json").read_text())
    assert record["attempts"] == []
    runner = _OfflineEngine(tmp_path)
    result = _wannierize(tmp_path, _ScriptedModel([]), runner, resume=True)
    first = result.run.attempts[0]
    assert first.seed == SEED and first.stage == "automatic" and first.action is None
    assert runner.localised[0] == SEED


def test_sigterm_becomes_an_infrastructure_stop_and_takes_the_step_with_it(tmp_path):
    pid_file = tmp_path / "pid"
    exe = _script(tmp_path / "long.sh", f'echo $$ > "{pid_file}"\nsleep 600\n')
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    previous = signal.getsignal(signal.SIGTERM)
    timer.start()
    try:
        with pytest.raises(InfrastructureStop, match="SIGTERM"):
            with stop_on_termination_signals():
                runner.run_localisation("wan")
    finally:
        timer.cancel()
    assert signal.getsignal(signal.SIGTERM) == previous
    deadline = time.monotonic() + 10
    while _alive(int(pid_file.read_text())) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.skipif(sys.platform != "linux", reason="process groups as on the cluster")
def test_the_step_runs_in_its_own_process_group(tmp_path):
    exe = _script(tmp_path / "pgid.sh", 'ps -o pgid= -p $$ > "$1.pgid"\n')
    runner = WannierRunner(run_dir=tmp_path, wannier90=exe, stall_policy=FAST)
    assert runner.run_localisation("wan").ok
    assert int((tmp_path / "wan.pgid").read_text()) != os.getpgid(0)
