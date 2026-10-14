"""Criterion 2 on slabs: L_BvK over the periodic directions only (campaign v3).

Over all three lattice directions a monolayer's L_BvK is its vacuum-padded c axis, which
put r_c = L/4 at 4.8 Angstrom on MoS2 and graphene. A direction is a vacuum direction only
when the mesh does not sample it (mp_grid = 1) *and* the structure leaves a vacuum gap
along it; bulk results are unchanged.

The synthetic constructions reuse ``test_criteria``'s chain model along x (a = 3 A), with
the parent cell replaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_criteria import EF, _construction, _verdict

from autowan.criteria import (
    TRUNCATED_HOPPING_ETA2,
    VACUUM_GAP_NN_MULTIPLE,
    bvk_length_ang,
    periodic_axes,
    vacuum_gaps_ang,
)

#: Hoppings out to 2a = 6 A. r_c = 6 A keeps them all (eta = 0); r_c = 3 A drops R = 2.
CHAIN = {0: [[EF]], 1: [[1.0]], 2: [[0.2]]}


def _slab(c_axis: float, mp_grid=(8, 8, 1), positions=((0.0, 0.0, 0.5),)):
    c = _construction(CHAIN, mp_grid=mp_grid, positions=positions)
    c.parent.lattice_ang = np.diag([3.0, 3.0, c_axis])
    return c


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_vacuum_gap_is_measured_perpendicular_to_the_plane():
    # hexagonal in-plane cell, a monolayer 3.1 A thick at the middle of c = 19.1 A
    lat = np.array([[3.19, 0, 0], [-1.595, 2.7626, 0], [0, 0, 19.1]])
    frac = np.array(
        [
            [0, 0, 0.5],
            [1 / 3, 2 / 3, 0.5 - 1.55 / 19.1],
            [1 / 3, 2 / 3, 0.5 + 1.55 / 19.1],
        ]
    )
    gaps = vacuum_gaps_ang(lat, frac)
    assert gaps[2] == pytest.approx(19.1 - 3.1)
    # an oblique c axis: the gap is the perpendicular thickness, not the length along c
    tilted = lat.copy()
    tilted[2] = [1.0, 0.5, 19.1]
    assert vacuum_gaps_ang(tilted, frac)[2] == pytest.approx(19.1 - 3.1)


def test_bvk_length_over_the_periodic_directions_only():
    lat = np.diag([3.0, 3.0, 19.0])
    assert bvk_length_ang(lat, (12, 12, 1)) == pytest.approx(19.0)
    assert bvk_length_ang(lat, (12, 12, 1), (0, 1)) == pytest.approx(36.0)
    hexa = np.array([[3.19, 0, 0], [-1.595, 3.19 * math.sqrt(3) / 2, 0], [0, 0, 19.1]])
    assert bvk_length_ang(hexa, (12, 12, 1), (0, 1)) == pytest.approx(12 * 3.19)
    # a wire: one periodic direction
    assert bvk_length_ang(lat, (1, 1, 10), (2,)) == pytest.approx(190.0)
    with pytest.raises(ValueError):
        bvk_length_ang(lat, (1, 1, 1), ())


def test_a_slab_is_two_dimensional():
    c = _slab(12.0)
    axes, rec = periodic_axes(c)
    assert axes == (0, 1)
    assert rec["vacuum_axes"] == [2]
    assert rec["vacuum_gap_ang"]["2"] == pytest.approx(12.0)
    assert rec["nearest_neighbour_ang"] == pytest.approx(3.0)


def test_mp_grid_one_without_a_vacuum_gap_stays_periodic():
    # a bulk crystal sampled by one k-point along c: atoms every 3 A, no gap
    c = _slab(3.0)
    assert periodic_axes(c)[0] == (0, 1, 2)
    # a van der Waals-like gap below the vacuum multiple is not vacuum either
    c = _slab(3.0 * VACUUM_GAP_NN_MULTIPLE * 0.9)
    assert periodic_axes(c)[0] == (0, 1, 2)


def test_a_vacuum_gap_sampled_by_the_mesh_is_not_a_vacuum_direction():
    # a thick vacuum, but mp_grid samples it: the user asked for bulk-like periodicity
    c = _slab(12.0, mp_grid=(8, 8, 2))
    assert periodic_axes(c)[0] == (0, 1, 2)


def test_without_positions_a_single_k_direction_is_treated_as_bulk_and_said_so():
    c = _slab(12.0)
    c.parent.positions_frac = None
    axes, rec = periodic_axes(c)
    assert axes == (0, 1, 2) and "no atomic positions" in rec["periodicity_note"]


# ---------------------------------------------------------------------------
# the criterion
# ---------------------------------------------------------------------------


def test_slab_truncation_radius_uses_the_periodic_directions():
    # c = 12 A: over three directions L = 12, r_c = 3 A would drop R = 2 (6 A).
    v = _verdict(TRUNCATED_HOPPING_ETA2, _slab(12.0))
    rec = v.record
    assert rec["periodic_dims"] == 2 and rec["periodic_axes"] == [0, 1]
    assert rec["l_bvk_ang"] == pytest.approx(24.0)
    assert rec["r_c_ang"] == pytest.approx(6.0)
    assert "periodic lattice directions [0, 1]" in rec["l_bvk_definition"]
    assert v.passed and v.metric_value == pytest.approx(0.0, abs=1e-9)
    assert "2D" in v.note


def test_bulk_record_is_unchanged_apart_from_periodic_dims():
    v = _verdict(TRUNCATED_HOPPING_ETA2, _construction(CHAIN))
    rec = v.record
    assert rec["periodic_dims"] == 3 and rec["periodic_axes"] == [0, 1, 2]
    assert (
        rec["l_bvk_definition"]
        == "shortest non-zero vector of the mp_grid superlattice"
    )
    assert rec["l_bvk_ang"] == pytest.approx(24.0)
    assert not {"vacuum_axes", "vacuum_gap_ang", "periodicity_note"} & set(rec)
    assert "2D" not in v.note


def test_bulk_with_a_single_k_direction_keeps_the_old_length():
    # mp_grid (8, 8, 1) on the cubic 3 A cell: no gap, L = 3 A as before v3
    v = _verdict(TRUNCATED_HOPPING_ETA2, _construction(CHAIN, mp_grid=(8, 8, 1)))
    assert v.record["periodic_dims"] == 3
    assert v.record["l_bvk_ang"] == pytest.approx(3.0)
    assert v.record["vacuum_axes"] == []
