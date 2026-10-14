"""Tests for the individual physics checks.

The checks are where a gate is most likely to be wrong in a way that looks right: a
criterion that refuses a correct construction reads exactly like a construction that
deserved refusing, and the difference is only visible by asking what the parent
calculation itself does.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autowan.core_checks import _frontier_manifold_straddles_fermi_level
from autowan.gate import Construction, NonEvaluable

#: A cubic cell whose lattice vectors equal alat, so fractional and Cartesian k agree.
LATTICE = np.eye(3) * 4.0
ALAT = 4.0
KPOINTS = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]])

#: Two bands: a dispersing one that dips below the Fermi level away from the zone
#: centre, and a flat one well above it. The model therefore brackets the level at
#: every k-point except the centre -- chosen so no test turns on an exact touch, which
#: floating point would decide arbitrarily.
HR = np.array(
    [
        [[0.5, 0.0], [0.0, 0.0]],
        [[-0.5, 0.0], [0.0, 2.0]],
        [[0.5, 0.0], [0.0, 0.0]],
    ],
    dtype=complex,
)
RVEC = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
DEG = np.ones(3)


def _construction(parent_bands):
    parent = SimpleNamespace(
        kpoints_cart=KPOINTS,
        lattice_ang=LATTICE,
        alat_ang=ALAT,
        fermi_energy_ev=0.0,
        eigenvalues_ev=np.array([parent_bands]),
    )
    wannier = SimpleNamespace(hr_matrix=HR, hr_rvectors=RVEC, hr_degeneracies=DEG)
    return Construction("c", parent=parent, wannier=wannier, material_class="metal")


def test_the_model_is_judged_against_the_parents_own_frontier_not_against_unity():
    # The parent brackets the Fermi level at the two quarter points only -- a
    # nearly-free-electron metal whose band leaves the occupied region over part of the
    # zone. A model that reproduces exactly that must pass.
    parent_bands = np.array([[1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0]])
    outcome = _frontier_manifold_straddles_fermi_level(_construction(parent_bands))
    assert outcome.passed
    assert outcome.metric_value == pytest.approx(1.0)


def test_a_model_that_misses_a_frontier_point_the_parent_has_is_refused():
    # A parent that brackets at the zone centre as well. The model does not, and that
    # is a real miss rather than an artefact of the criterion.
    parent_bands = np.array([[-1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0]])
    outcome = _frontier_manifold_straddles_fermi_level(_construction(parent_bands))
    assert not outcome.passed
    assert "1 of the 4" in outcome.note


def test_a_parent_that_brackets_nowhere_leaves_the_check_without_a_verdict():
    # Not a pass and not a failure: there is no frontier to reproduce, and a check that
    # cannot be evaluated must say so rather than scoring the construction on nothing.
    parent_bands = np.array([[1.0], [2.0], [3.0], [4.0]])
    with pytest.raises(NonEvaluable):
        _frontier_manifold_straddles_fermi_level(_construction(parent_bands))


def test_the_recent_steps_survive_into_the_payload(tmp_path):
    # The final delta alone cannot tell a minimisation that is nearly there from one
    # that is wandering: both end on a small number.
    from autowan.payload import trajectory_of
    from autowan.w90_outputs import MinimisationStage

    stage = MinimisationStage(
        name="wannierisation",
        n_iterations=8000,
        final_value=28.57,
        final_delta=-1.6e-05,
        converged=False,
        recent_deltas=(-8.7e-06, -4.6e-05, -1.6e-05),
    )
    rows = trajectory_of(SimpleNamespace(disentanglement=None, wannierisation=stage))
    assert rows[0]["recent_deltas"] == [-8.7e-06, -4.6e-05, -1.6e-05]


def test_the_step_trend_is_stated_only_where_a_stage_did_not_converge():
    from autowan.actions import Stage
    from autowan.payload import RepairPayload

    rows = [
        {
            "stage": "disentanglement",
            "iterations": 1296.0,
            "final_value": 10.96,
            "final_delta": 9.8e-08,
            "converged": 1.0,
            "recent_deltas": [1.04e-7, 1.03e-7, 1.02e-7, 9.8e-8],
        },
        {
            "stage": "wannierisation",
            "iterations": 4000.0,
            "final_value": 28.68,
            "final_delta": -3.02e-06,
            "converged": 0.0,
            "recent_deltas": [-2.6e-6, -3.4e-6, -5.5e-6, -3.0e-6],
        },
    ]
    text = RepairPayload(
        material_signature={},
        gate_report={},
        current_parameters={},
        convergence_trajectory=rows,
    ).render(Stage.REPAIR)
    body = text[text.index("CONVERGENCE TRAJECTORY") :]
    assert body.count("these steps are") == 1
    assert "not shrinking" in body
