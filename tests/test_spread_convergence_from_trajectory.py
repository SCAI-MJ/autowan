"""Spread-minimisation convergence read from the .wout trajectory (user, 2026-10-07).

Wannier90 tests the spread minimisation only when ``conv_window`` > 1. The published VASP
SCDM frontier sets none, so its run carries no verdict of its own and the check used to
be non-evaluable -- which made the frontier structurally uncertifiable. Such a run is now
judged with Wannier90's own criterion applied to the printed per-iteration spread
changes: |ΔΩ| <= conv_tol on each of the last ``REFERENCE_CONV_WINDOW`` iterations of
the run. These tests pin that rule on synthetic trajectories (converged, oscillating,
still descending, and the ways a trajectory can be unreadable) and on the stored Si VASP
SCDM frontier smoke.
"""

from __future__ import annotations

from pathlib import Path

import os
import pytest

from autowan.core_checks import (
    EVALUATED_FROM_TRAJECTORY,
    REFERENCE_CONV_WINDOW,
    WANNIER90_DEFAULT_CONV_TOL_ANG2,
    _spread_minimisation_converged,
    _subspace_selection_converged,
)
from autowan.gate import Construction, NonEvaluable
from autowan.w90_outputs import BOHR_TO_ANG, read_wout

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


FINAL_STATE = """ Final State
  WF centre and spread    1  (  0.000000,  0.000000,  0.000000 )     1.00000000
  Sum of centres and spreads (  0.000000,  0.000000,  0.000000 )     1.00000000

         Spreads ({unit}^2)       Omega I      =     0.500000000
        ================       Omega D      =     0.100000000
                               Omega OD     =     0.400000000
    Final Spread ({unit}^2)       Omega Total  =     1.000000000
"""


def _wout(
    path: Path,
    deltas: list[float],
    *,
    conv_tol: float = 1.0e-4,
    conv_window: int = -1,
    num_print_cycles: int = 1,
    iprint: int = 1,
    dlta: bool = True,
    unit: str = "Ang",
    converged_line: bool = False,
    disentangle_window: int | None = None,
) -> Path:
    """A minimal .wout with Wannier90's header boxes and spread trajectory.

    ``deltas[i]`` is the spread change of iteration i + 1, in the run's length unit
    squared; the run's num_iter is ``len(deltas)``. Rows are printed as Wannier90 prints
    them: every num_print_cycles iterations, plus the first and the last.
    """
    num_iter = len(deltas)
    lines = [
        " *---------------------------------- MAIN ------------------------------------*",
        " |  Number of Wannier Functions               :                 1             |",
        f" |  Output verbosity (1=low, 5=high)          :        {iprint:8d}             |",
        f" |  Length Unit                               :             {unit:>5s}             |",
        " *----------------------------------------------------------------------------*",
        " *------------------------------- WANNIERISE ---------------------------------*",
        f" |  Total number of iterations                :          {num_iter:8d}             |",
        f" |  Convergence tolerence                     :        {conv_tol:10.3E}             |",
        f" |  Convergence window                        :          {conv_window:8d}             |",
        f" |  Iterations between writing output         :          {num_print_cycles:8d}             |",
        " *----------------------------------------------------------------------------*",
    ]
    if disentangle_window is not None:
        lines += [
            " *------------------------------- DISENTANGLE --------------------------------*",
            " |  Using band disentanglement                :                 T             |",
            " |  Total number of iterations                :              9999             |",
            " |  Convergence tolerence                     :         1.000E-10             |",
            f" |  Convergence window                        :          {disentangle_window:8d}             |",
            " *----------------------------------------------------------------------------*",
        ]
    lines.append(
        " | Iter  Delta Spread     RMS Gradient      Spread (Ang^2)      Time  |<-- CONV"
    )
    omega = 10.0
    lines.append(
        f"      0  {omega:12.3E}     0.0000000000  {omega:18.10f}       0.00  <-- CONV"
    )
    if iprint >= 1:
        for it, d in enumerate(deltas, start=1):
            omega += d
            if not (it % num_print_cycles == 0 or it == 1 or it == num_iter):
                continue
            lines.append(f" Cycle: {it:6d}")
            lines.append(
                f" {it:6d}  {d:12.3E}  {0.1:15.10f}  {omega:18.10f}  {0.0:8.2f}  <-- CONV"
            )
            if dlta:
                lines.append(
                    f" Delta: O_D= {0.0:15.7E} O_OD= {d:15.7E} O_TOT= {d:15.7E} <-- DLTA"
                )
    if converged_line:
        lines.append(
            "             <<< Wannierisation convergence criteria satisfied >>>"
        )
    lines.append(FINAL_STATE.format(unit=unit))
    path.write_text("\n".join(lines) + "\n")
    return path


def _construction(wout: Path, **inputs) -> Construction:
    return Construction(
        "synthetic",
        parent=None,
        wannier=read_wout(wout),
        material_class="gapped",
        inputs=inputs,
    )


def _check(tmp_path, deltas, inputs=None, **kw):
    return _spread_minimisation_converged(
        _construction(_wout(tmp_path / "s.wout", deltas, **kw), **(inputs or {}))
    )


# ---------------------------------------------------------------------------
# the three trajectory shapes
# ---------------------------------------------------------------------------


def test_a_trajectory_that_settled_below_tolerance_passes_and_says_how(tmp_path):
    deltas = [-1.0 * 0.5**i for i in range(40)]  # geometric decay to ~1e-12
    out = _check(tmp_path, deltas, {"conv_tol": "1.0e-4"})
    assert out.passed is True
    assert out.note.startswith(EVALUATED_FROM_TRAJECTORY)
    rec = out.record
    assert rec["evidence"] == EVALUATED_FROM_TRAJECTORY
    assert rec["conv_window_used"] == REFERENCE_CONV_WINDOW == 3
    assert rec["conv_tol_ang2"] == pytest.approx(1e-4)
    assert rec["conv_tol_source"] == "deck"
    assert rec["window_iterations"] == [38, 39, 40]
    assert rec["delta_significant_figures"] == 7
    assert out.metric_value == 40


def test_an_oscillating_trajectory_fails(tmp_path):
    # Titanium-like: the step changes sign every iteration at a magnitude above the
    # tolerance and never shrinks. More iterations would not help.
    deltas = [3e-4 * (-1) ** i for i in range(200)]
    out = _check(tmp_path, deltas, {"conv_tol": "1.0e-4"})
    assert out.passed is False
    assert out.record["max_abs_delta_ang2"] == pytest.approx(3e-4)
    assert ">" in out.note


def test_a_trajectory_still_descending_at_the_end_fails(tmp_path):
    deltas = [-1e-2 / (i + 1) for i in range(50)]  # last step -2e-4
    out = _check(tmp_path, deltas, {"conv_tol": "1.0e-4"})
    assert out.passed is False


def test_the_window_must_hold_at_the_end_of_the_run_not_somewhere_before_it(tmp_path):
    # Three tiny steps in the middle, then a kick: Wannier90 with a window would have
    # stopped at iteration 12, but a run without one ran on, and its end state is what
    # the construction is.
    deltas = [-1e-2] * 10 + [1e-6, -1e-6, 1e-6] + [-1e-2] * 7 + [-1e-6, -5e-3]
    out = _check(tmp_path, deltas, {"conv_tol": "1.0e-4"})
    assert out.passed is False


def test_one_step_over_tolerance_inside_the_window_fails(tmp_path):
    deltas = [-1e-6] * 20 + [-2e-4, -1e-6, -1e-6]
    assert _check(tmp_path, deltas, {"conv_tol": "1.0e-4"}).passed is False
    deltas = [-1e-6] * 20 + [-2e-4, -1e-6, -1e-6, -1e-6]
    assert _check(tmp_path, deltas, {"conv_tol": "1.0e-4"}).passed is True


# ---------------------------------------------------------------------------
# which runs are read from the trajectory, and with what
# ---------------------------------------------------------------------------


def test_a_deck_with_its_own_window_keeps_wannier90s_verdict(tmp_path):
    # A converged trajectory but no "criteria satisfied" line: with conv_window = 3 the
    # engine's own verdict stands, and nothing is re-derived.
    deltas = [-1e-8] * 30
    wout = _wout(tmp_path / "s.wout", deltas, conv_window=3)
    out = _spread_minimisation_converged(
        _construction(wout, conv_window="3", conv_tol="1e-4")
    )
    assert out.passed is False
    assert out.record == {}
    wout = _wout(tmp_path / "t.wout", deltas, conv_window=3, converged_line=True)
    out = _spread_minimisation_converged(_construction(wout, conv_window="3"))
    assert out.passed is True and out.record == {}


def test_conv_window_one_runs_no_test_in_wannier90_and_is_read_from_the_trajectory(
    tmp_path,
):
    # wannierise.F90 calls its test only for conv_window > 1.
    out = _check(tmp_path, [-1e-8] * 10, {"conv_window": "1"}, conv_window=1)
    assert out.record["evidence"] == EVALUATED_FROM_TRAJECTORY
    assert out.record["deck_conv_window"] == 1


def test_the_engine_window_is_used_when_the_deck_record_has_none(tmp_path):
    # conv_noise_amp > 0 makes Wannier90 default the window to 5; the header says so.
    wout = _wout(tmp_path / "s.wout", [-1e-8] * 10, conv_window=5, converged_line=True)
    out = _spread_minimisation_converged(_construction(wout))
    assert out.passed is True and out.record == {}


def test_an_unset_conv_tol_takes_wannier90s_default(tmp_path):
    deltas = [-1e-6] * 10
    out = _check(tmp_path, deltas, conv_tol=1e-10)
    assert out.record["conv_tol_ang2"] == WANNIER90_DEFAULT_CONV_TOL_ANG2 == 1e-10
    assert out.record["conv_tol_source"] == "Wannier90 default"
    assert out.passed is False
    out = _check(tmp_path, [-1e-12] * 10)
    assert out.passed is True


def test_bohr_unit_deltas_are_compared_in_square_angstrom(tmp_path):
    # Printed in Bohr^2: 1.5e-4 Bohr^2 = 4.2e-5 Å^2, inside a 1e-4 Å^2 tolerance.
    out = _check(tmp_path, [-1.5e-4] * 10, {"conv_tol": "1e-4"}, unit="Bohr")
    assert out.passed is True
    assert out.record["max_abs_delta_ang2"] == pytest.approx(1.5e-4 * BOHR_TO_ANG**2)


def test_the_reference_window_is_the_one_repaired_decks_get():
    from autowan.execution import DEFAULT_CONV_WINDOW

    assert REFERENCE_CONV_WINDOW == DEFAULT_CONV_WINDOW


# ---------------------------------------------------------------------------
# a trajectory that cannot be judged is non-evaluable, with the reason
# ---------------------------------------------------------------------------


def test_a_sparsely_printed_trajectory_is_non_evaluable(tmp_path):
    with pytest.raises(NonEvaluable, match=r"num_print_cycles = 10"):
        _check(tmp_path, [-1e-8] * 100, num_print_cycles=10)


def test_printing_the_last_iteration_is_not_enough_without_its_neighbours(tmp_path):
    # Every second iteration: 100 and the final 101 are printed, 99 is not, so the
    # window 99-101 cannot be read.
    with pytest.raises(NonEvaluable, match="not all printed"):
        _check(tmp_path, [-1e-8] * 101, num_print_cycles=2)


def test_a_run_that_prints_no_iterations_is_non_evaluable(tmp_path):
    with pytest.raises(NonEvaluable, match=r"iprint = 0"):
        _check(tmp_path, [-1e-8] * 10, iprint=0)


def test_a_run_shorter_than_the_window_is_non_evaluable(tmp_path):
    with pytest.raises(NonEvaluable, match="num_iter = 2"):
        _check(tmp_path, [-1e-8] * 2)


def test_a_comparison_decided_inside_print_rounding_is_non_evaluable(tmp_path):
    # CONV rows only (3 significant figures): 1.0004e-4 prints as 0.100E-03, which
    # cannot be told from the tolerance.
    with pytest.raises(NonEvaluable, match="significant figures"):
        _check(tmp_path, [-1.0004e-4] * 10, {"conv_tol": "1e-4"}, dlta=False)
    # With the DLTA line (7 figures) the same steps are decided: above tolerance.
    assert _check(tmp_path, [-1.0004e-4] * 10, {"conv_tol": "1e-4"}).passed is False


def test_header_rows_are_read_from_the_box_that_owns_them(tmp_path):
    wout = _wout(
        tmp_path / "s.wout", [-1e-8] * 10, conv_window=-1, disentangle_window=3
    )
    settings = read_wout(wout).wannierise_settings
    assert settings["conv_window"] == -1
    assert settings["num_iter"] == 10
    assert settings["num_print_cycles"] == 1
    assert settings["iprint"] == 1


# ---------------------------------------------------------------------------
# disentanglement has no such gap
# ---------------------------------------------------------------------------


def test_the_subspace_check_does_not_depend_on_a_deck_window():
    # disentangle.F90 calls its test on every iteration (dis_conv_window defaults to 3,
    # and 0 or more is required), so the engine's verdict is always there to read.
    from types import SimpleNamespace

    from autowan.w90_outputs import MinimisationStage

    stage = MinimisationStage("disentanglement", 500, 1.0, 1e-11, True)
    wannier = SimpleNamespace(disentanglement_used=True, disentanglement=stage)
    c = Construction("c", parent=None, wannier=wannier, material_class="metal")
    assert _subspace_selection_converged(c).passed is True


# ---------------------------------------------------------------------------
# the stored Si VASP SCDM frontier smoke (not shipped; skipped when absent)
# ---------------------------------------------------------------------------

SMOKE = _stored("AUTOWAN_TEST_SMOKE")


@pytest.mark.skipif(not (SMOKE / "Si.wout").is_file(), reason="the stored smoke output is not shipped")
def test_the_si_vasp_scdm_frontier_converges_by_its_trajectory():
    run = read_wout(SMOKE / "Si.wout")
    assert run.wannierise_settings["conv_window"] == -1
    assert run.wannierisation.converged is False  # Wannier90 ran no test
    out = _spread_minimisation_converged(
        _construction(SMOKE / "Si.wout", num_iter="1000", conv_tol="1.0e-4")
    )
    assert out.passed is True
    assert out.record["window_iterations"] == [998, 999, 1000]
    assert out.record["max_abs_delta_ang2"] == pytest.approx(5.927886e-05)
    assert out.record["conv_tol_ang2"] == 1e-4


@pytest.mark.skipif(
    not (SMOKE / "Si_conv_window.wout").is_file(), reason="the stored smoke output is not shipped"
)
def test_the_si_diagnostic_with_a_window_keeps_the_engine_verdict():
    out = _spread_minimisation_converged(
        _construction(SMOKE / "Si_conv_window.wout", conv_window="3", conv_tol="1e-4")
    )
    assert out.passed is True and out.record == {}
