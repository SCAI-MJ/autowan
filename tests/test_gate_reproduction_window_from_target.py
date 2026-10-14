"""The band-reproduction check reads its window from the target, never from the deck.

The frontier-plus-rule arm tests whether a one-line parameter rule clears a residual,
and one of its rules lowers ``dis_froz_max``. A gate that took the reproduction window
from the deck's own ``dis_froz_max`` would grade a narrowed deck on its own narrowed
window, so "lower the window" would clear any material trivially. The target therefore
carries the window ν the research use needs, and the check reads it from there.

Three semantics are fixed, and each test below pins one of them:

* ν comes from the target. A deck at least as wide as ν is scored over ν, not over its
  own frozen window.
* A deck whose frozen window is narrower than ν is refused on this check with the
  distinct, contractual reason ``frozen_window_narrower_than_target``, naming both
  windows. The band-reproduction comparison over ν is still computed and recorded, so
  "narrowed and also fails to reproduce ν" stays distinguishable from "narrowed but
  reproduces ν".
* No target means today's behaviour, byte for byte, and the record says the window
  came from the deck (``target_window_source: deck``), so a scientific comparison can
  refuse an omitted target.

Reference frames, stated once because confusing them is the easy mistake.
``CertificationTarget.window_ev`` is ``(lo, hi)`` in eV *relative to the parent's Fermi
level*, as on ``autowan/feat/gate-magnetic-subspace-target`` (its
``magnetic_subspace_mask`` selects ``e >= ef + lo`` and ``e <= ef + hi``). A lower bound
of ``-inf`` means "from the bottom of the parent's bands", the lower edge the check has
always used. The deck's ``dis_froz_max`` and ``dis_froz_min`` are *absolute* eV, as
Wannier90 reads them. The synthetic parent below sits at E_F = 5 eV, so a target read
in the wrong frame fails these tests rather than passing by accident.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan.core_checks import CORE_CHECKS
from autowan.gate import CertificationGate, Construction, GateDefinitionError
from autowan.verdicts import VERDICT_COLUMNS, verdict_rows

CHECK = "model_bands_correspond_to_parent_bands"
REFUSAL = "frozen_window_narrower_than_target"
BASE_KEYS = ["value", "units", "threshold", "passed", "evaluable", "applies", "note"]

#: The parent's Fermi level, deliberately far from zero (see the module docstring).
EF = 5.0
LATTICE = np.eye(3) * 4.0
ALAT = 4.0
KPOINTS = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]])

#: Flat parent bands. The occupied ones (2.0 and 4.0 eV) span 2 eV, so the check's
#: matching tolerance is 1 % of that, 20 meV. The lowest parent band, 2.0 eV, is the
#: lower edge of the reproduction window whenever the target's lower bound is -inf.
PARENT_LEVELS = (2.0, 4.0, 5.5, 6.5, 9.0)

#: One dispersive occupied parent band joining the two flat occupied levels, one value
#: per k-point. The semicore-independent occupied-bandwidth scale (the VASP gate
#: integration) measures the occupied block above the largest gap in the band-edge
#: envelope; two flat levels alone are two blocks, and the block above the gap has no
#: spread, so no tolerance. Joined, the block is 2.0..4.0 eV and the tolerance stays
#: 20 meV. It adds parent states only; the check under test asks whether each model
#: band sits near some parent band, and every model band below already does.
CONNECTOR = (2.0, 3.0, 4.0, 3.0)

#: On-site-only models, so each model band is flat at the listed energy.
CLEAN = (2.0, 4.0, 5.5, 6.5)  # every band sits on a parent band
SPURIOUS_ABOVE_NU = (
    2.0,
    4.0,
    5.5,
    7.3,
    7.6,
)  # two bands between E_F+2 and E_F+3 match nothing
SPURIOUS_INSIDE_NU = (
    2.0,
    4.0,
    5.5,
    6.8,
)  # one band between E_F+1 and E_F+2 matches nothing


def _parent():
    bands = np.tile(np.array(PARENT_LEVELS), (len(KPOINTS), 1))
    bands = np.sort(np.column_stack([bands, CONNECTOR]), axis=1)
    return SimpleNamespace(
        kpoints_cart=KPOINTS,
        lattice_ang=LATTICE,
        alat_ang=ALAT,
        fermi_energy_ev=EF,
        eigenvalues_ev=np.array([bands]),
        positions_frac=np.zeros((1, 3)),
    )


def _wannier(levels):
    """Only the Hamiltonian is real; the checks that need anything else say so."""
    return SimpleNamespace(
        hr_matrix=np.diag(np.asarray(levels, dtype=float)).astype(complex)[None, :, :],
        hr_rvectors=np.zeros((1, 3), dtype=int),
        hr_degeneracies=np.ones(1),
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=None,
        num_wann=None,
        centres_ang=None,
    )


def _target(lo, hi):
    from autowan.gate import CertificationTarget

    return CertificationTarget("whole_occupied_manifold", (lo, hi))


def _construction(levels, *, target=None, **deck):
    extra = {} if target is None else {"target": target}
    return Construction(
        "probe",
        parent=_parent(),
        wannier=_wannier(levels),
        material_class="gapped",
        inputs={"spin_index": 0, **deck},
        **extra,
    )


def _score(construction):
    report = CertificationGate(CORE_CHECKS).run(construction)
    verdict = next(v for v in report.verdicts if v.check_id == CHECK)
    return report, verdict, report.measurements()[CHECK]


# ---------------------------------------------------------------------------
# a deck narrower than the target is refused, with both windows in the reason
# ---------------------------------------------------------------------------


def test_a_deck_narrower_than_the_target_is_refused_and_the_reason_names_both_windows():
    # Target nu = (-inf, +2.0) eV relative to E_F, i.e. up to 7.0 eV absolute. The deck
    # freezes up to E_F + 1 eV, written absolute as Wannier90 reads it: 6.0 eV, which is
    # +1.0 relative. Read in the wrong frame (2.0 eV absolute) the deck would look wider.
    report, verdict, record = _score(
        _construction(CLEAN, target=_target(-math.inf, 2.0), dis_froz_max=EF + 1.0)
    )
    assert verdict.evaluable and verdict.applies
    assert verdict.passed is False
    assert CHECK in report.failed_check_ids
    assert not report.passed
    assert verdict.refusal == REFUSAL
    assert record["refusal"] == REFUSAL
    assert REFUSAL in verdict.note
    assert "6.00" in verdict.note  # the deck's frozen window
    assert "7.00" in verdict.note  # the target's window
    assert record["target_window_source"] == "target"
    assert record["target_window_ev"] == [None, 2.0]
    assert record["deck_frozen_window_ev"] == [None, 6.0]
    assert record["reproduction_window_ev"] == [2.0, 7.0]


def test_the_target_window_is_read_relative_to_the_parent_fermi_level():
    # nu = (-inf, +0.5) relative is 5.5 eV absolute: a deck freezing to 6.0 eV covers it.
    _, verdict, record = _score(
        _construction(CLEAN, target=_target(-math.inf, 0.5), dis_froz_max=6.0)
    )
    assert verdict.refusal is None
    assert verdict.passed is True
    assert record["reproduction_window_ev"] == [2.0, 5.5]
    assert "2.00..5.50 eV" in verdict.note


def test_a_fortran_spelled_narrow_deck_is_refused_the_same_way():
    _, verdict, record = _score(
        _construction(CLEAN, target=_target(-math.inf, 2.0), dis_froz_max="6.0d0")
    )
    assert verdict.refusal == REFUSAL
    assert record["deck_frozen_window_ev"] == [None, 6.0]


def test_a_deck_whose_frozen_window_starts_above_a_finite_target_bound_is_refused():
    # nu = (-2.0, +2.0) relative = 3.0 .. 7.0 eV absolute; the deck freezes from 4.0 eV.
    _, verdict, record = _score(
        _construction(
            CLEAN, target=_target(-2.0, 2.0), dis_froz_min=4.0, dis_froz_max=8.0
        )
    )
    assert verdict.passed is False
    assert verdict.refusal == REFUSAL
    assert "4.00" in verdict.note and "3.00" in verdict.note
    assert record["deck_frozen_window_ev"] == [4.0, 8.0]
    assert record["reproduction_window_ev"] == [3.0, 7.0]


def test_a_deck_covering_a_finite_lower_bound_is_scored_from_that_bound():
    _, verdict, record = _score(
        _construction(
            CLEAN, target=_target(-2.0, 2.0), dis_froz_min=3.0, dis_froz_max=8.0
        )
    )
    assert verdict.refusal is None
    assert verdict.passed is True
    # 2.0 eV lies below the window and is not counted: 3 model bands at each of 4 k.
    assert verdict.note.startswith("0 of 12 inside 3.00..7.00 eV")
    assert record["reproduction_window_ev"] == [3.0, 7.0]


def test_an_unbounded_target_does_not_judge_the_decks_lower_edge():
    _, verdict, _ = _score(
        _construction(
            CLEAN, target=_target(-math.inf, 2.0), dis_froz_min=4.0, dis_froz_max=8.0
        )
    )
    assert verdict.refusal is None
    assert verdict.passed is True


# ---------------------------------------------------------------------------
# the narrowed deck still carries the band-reproduction diagnostic over nu
# ---------------------------------------------------------------------------


def test_a_narrowed_deck_that_still_reproduces_nu_is_refused_by_contract_not_by_bands():
    _, verdict, record = _score(
        _construction(CLEAN, target=_target(-math.inf, 2.0), dis_froz_max=6.0)
    )
    assert verdict.refusal == REFUSAL
    assert verdict.passed is False
    # Over nu (2.0 .. 7.0 eV) every model band has a parent partner.
    assert verdict.metric_value == 0.0
    assert verdict.threshold_value == pytest.approx(0.20)
    assert record["metric_within_threshold"] is True
    assert "0 of 16" in verdict.note


def test_a_narrowed_deck_that_also_fails_nu_carries_both_the_reason_and_the_failure():
    _, verdict, record = _score(
        _construction(
            SPURIOUS_INSIDE_NU, target=_target(-math.inf, 2.0), dis_froz_max=6.0
        )
    )
    assert verdict.refusal == REFUSAL
    assert verdict.passed is False
    # Scored over nu: the 6.8 eV band lies inside 7.0 eV and matches nothing. Over the
    # deck's own window (to 6.0 eV) it would have scored 0 of 12.
    assert verdict.metric_value == pytest.approx(0.25)
    assert record["metric_within_threshold"] is False
    assert "4 of 16" in verdict.note


def test_a_narrowed_deck_with_no_model_band_inside_nu_is_still_refused():
    _, verdict, record = _score(
        _construction((7.5, 7.8), target=_target(-math.inf, 2.0), dis_froz_max=6.0)
    )
    assert verdict.evaluable is True
    assert verdict.passed is False
    assert verdict.refusal == REFUSAL
    assert record["metric_within_threshold"] is None


# ---------------------------------------------------------------------------
# a deck at least as wide as the target is scored over the target
# ---------------------------------------------------------------------------


def test_a_deck_wider_than_the_target_is_scored_over_the_target_not_its_own_window():
    _, verdict, record = _score(
        _construction(
            SPURIOUS_ABOVE_NU, target=_target(-math.inf, 2.0), dis_froz_max=8.0
        )
    )
    assert verdict.refusal is None
    assert "refusal" not in record
    assert verdict.passed is True
    assert verdict.metric_value == 0.0
    assert "0 of 12 inside 2.00..7.00 eV" in verdict.note
    assert record["reproduction_window_ev"] == [2.0, 7.0]
    assert record["target_window_source"] == "target"

    # The same construction with no target is graded on the deck's own window, where
    # the two bands above E_F + 2 eV count against it.
    _, own, own_record = _score(_construction(SPURIOUS_ABOVE_NU, dis_froz_max=8.0))
    assert own.metric_value == pytest.approx(0.4)
    assert own.passed is False
    assert own_record["target_window_source"] == "deck"


@pytest.mark.parametrize("froz", [7.0, "7.0", "7.0d0", "0.7D+01"])
def test_a_deck_equal_to_the_target_is_not_refused(froz):
    _, verdict, record = _score(
        _construction(
            SPURIOUS_ABOVE_NU, target=_target(-math.inf, 2.0), dis_froz_max=froz
        )
    )
    assert verdict.refusal is None
    assert verdict.passed is True
    assert verdict.note.startswith("0 of 12 inside 2.00..7.00 eV")


def test_a_deck_without_a_frozen_window_is_scored_over_the_target_and_not_refused():
    _, verdict, record = _score(
        _construction(SPURIOUS_INSIDE_NU, target=_target(-math.inf, 2.0))
    )
    assert verdict.refusal is None
    assert record["deck_frozen_window_ev"] == [None, None]
    assert record["reproduction_window_ev"] == [2.0, 7.0]
    assert verdict.metric_value == pytest.approx(0.25)
    assert verdict.passed is False  # 0.25 > 0.20: a band failure, not a contract one


def test_a_target_equal_to_the_decks_own_window_reproduces_the_untargeted_verdict():
    # The frontier-plus-rule arm's negative control: nu set to each construction's own
    # dis_froz_max must give back exactly the verdict the construction had without one.
    untargeted = _score(_construction(SPURIOUS_ABOVE_NU, dis_froz_max=8.0))
    targeted = _score(
        _construction(
            SPURIOUS_ABOVE_NU, target=_target(-math.inf, 8.0 - EF), dis_froz_max=8.0
        )
    )
    assert targeted[0].passed == untargeted[0].passed
    assert targeted[0].failed_check_ids == untargeted[0].failed_check_ids
    assert {k: targeted[2][k] for k in BASE_KEYS} == {
        k: untargeted[2][k] for k in BASE_KEYS
    }
    assert targeted[1].refusal is None


# ---------------------------------------------------------------------------
# no target: today's behaviour, byte for byte, and the record says so
# ---------------------------------------------------------------------------

_TODAY = {
    # dis_froz_max as the deck spells it -> today's (value, passed, note)
    "absolute float": (
        {"dis_froz_max": 8.0},
        0.4,
        False,
        "8 of 20 inside 2.00..8.00 eV sit further than 20 meV from any band the "
        "parent calculation has",
    ),
    "fortran string": (
        {"dis_froz_max": "8.0d0"},
        0.4,
        False,
        "8 of 20 inside 2.00..8.00 eV sit further than 20 meV from any band the "
        "parent calculation has",
    ),
    "absent: E_F + 2 eV": (
        {},
        0.0,
        True,
        "0 of 12 inside 2.00..7.00 eV sit further than 20 meV from any band the "
        "parent calculation has",
    ),
    "unparseable: E_F + 2 eV": (
        {"dis_froz_max": "not-a-number"},
        0.0,
        True,
        "0 of 12 inside 2.00..7.00 eV sit further than 20 meV from any band the "
        "parent calculation has",
    ),
}


@pytest.mark.parametrize("case", sorted(_TODAY))
def test_no_target_keeps_todays_verdict_and_records_the_deck_as_the_source(case):
    deck, value, passed, note = _TODAY[case]
    report, verdict, record = _score(_construction(SPURIOUS_ABOVE_NU, **deck))
    assert (verdict.evaluable, verdict.applies, verdict.passed) == (True, True, passed)
    assert verdict.metric_value == pytest.approx(value, abs=0)
    assert verdict.metric_units == "fraction of model bands"
    assert verdict.threshold_value == pytest.approx(0.20, abs=0)
    assert verdict.note == note
    assert verdict.refusal is None
    # The seven keys every verdict has always carried come first and are unchanged;
    # the window record is added after them, never in their place.
    assert list(record)[: len(BASE_KEYS)] == BASE_KEYS
    assert record["target_window_source"] == "deck"
    assert record["target_window_ev"] is None
    assert "refusal" not in record and "metric_within_threshold" not in record
    # Every other check's measurement keeps exactly the keys it had.
    for check_id, other in report.measurements().items():
        if check_id != CHECK:
            assert list(other) == BASE_KEYS, check_id
    # The verdict rows a results notebook loads keep their schema.
    rows = verdict_rows(report, CertificationGate(CORE_CHECKS))
    assert all(tuple(row) == VERDICT_COLUMNS for row in rows)


def test_an_undeclared_construction_keeps_the_default_target_with_no_window():
    from autowan.gate import WHOLE_OCCUPIED_MANIFOLD

    construction = _construction(CLEAN, dis_froz_max=8.0)
    assert construction.target is WHOLE_OCCUPIED_MANIFOLD
    assert construction.target.name == "whole_occupied_manifold"
    assert construction.target.window_ev is None


# ---------------------------------------------------------------------------
# the target object
# ---------------------------------------------------------------------------


def test_a_target_window_is_stored_as_two_floats():
    target = _target(np.float64(-np.inf), 2)
    assert target.window_ev == (-math.inf, 2.0)
    assert all(type(bound) is float for bound in target.window_ev)


@pytest.mark.parametrize(
    "window",
    [
        (1.0,),
        (1.0, 2.0, 3.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (-math.inf, math.inf),
        (math.nan, 2.0),
        (-math.inf, math.nan),
        ("low", "high"),
        2.0,
    ],
)
def test_an_inadmissible_target_window_is_refused(window):
    from autowan.gate import CertificationTarget

    with pytest.raises(GateDefinitionError):
        CertificationTarget("whole_occupied_manifold", window)


def test_an_unknown_target_name_is_refused():
    from autowan.gate import CertificationTarget

    with pytest.raises(GateDefinitionError):
        CertificationTarget("whatever_feels_right", (-math.inf, 2.0))


# ---------------------------------------------------------------------------
# plumbing: the workflow carries the target into the construction it scores
# ---------------------------------------------------------------------------


def test_the_workflow_config_carries_the_target_into_the_scored_construction():
    from autowan.gate import WHOLE_OCCUPIED_MANIFOLD
    from autowan.workflow import WorkflowConfig, _gate_construction

    assert WorkflowConfig().target is WHOLE_OCCUPIED_MANIFOLD
    target = _target(-math.inf, 2.0)
    construction = _gate_construction(
        WorkflowConfig(target=target),
        "seed_r1",
        parent=_parent(),
        wannier=_wannier(CLEAN),
        material_class="gapped",
        inputs={"spin_index": 0, "dis_froz_max": 6.0},
    )
    assert construction.construction_id == "seed_r1"
    assert construction.target is target
    assert construction.inputs == {"spin_index": 0, "dis_froz_max": 6.0}
    assert construction.material_class == "gapped"


def test_the_gate_inputs_carry_both_edges_of_the_decks_frozen_window(tmp_path):
    from autowan.inputs import ConstructionInputs

    inputs = ConstructionInputs(
        seed="s",
        win=tmp_path / "s.win",
        pw2wan=None,
        values={"dis_froz_min": "3.5", "dis_froz_max": "8.0"},
    )
    gate_inputs = inputs.gate_inputs()
    assert gate_inputs["dis_froz_min"] == "3.5"
    assert gate_inputs["dis_froz_max"] == "8.0"


# ---------------------------------------------------------------------------
# the QE frontier rebuild: real constructions, scored without and with a target
# ---------------------------------------------------------------------------

_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "results"
    / "qe_frontier_baseline_rebuild"
)
#: The rebuild's raw tree is DVC-tracked (``raw.dvc``); point AUTOWAN_QE_REBUILD_RAW at
#: any copy of it. The committed per-channel records beside it were written by a gate
#: whose ``core_checks.py`` and ``gate.py`` are identical to ``autowan/dev`` bcc77a60d.
_RAW = Path(os.environ.get("AUTOWAN_QE_REBUILD_RAW", _RESULTS / "raw"))


def _rebuild_channels():
    if not _RAW.is_dir():
        return []
    channels = []
    for path in sorted(_RESULTS.glob("*/*/gate_verdicts.json")):
        material, arm = path.parent.parent.name, path.parent.name
        for channel, record in json.loads(path.read_text()).get("channels", {}).items():
            gate = record.get("gate") or {}
            if gate.get("ran"):
                channels.append((material, arm, channel, gate))
    return channels


def _rebuild_construction(material, arm, channel, gate, target=None):
    from autowan.inputs import read_construction
    from autowan.qe_outputs import read_parent
    from autowan.w90_outputs import read_run

    run = _RAW / material / arm
    channel_dir = run / channel if channel in ("up", "down") else run
    parent = read_parent(run / f"{material}.save" / "data-file-schema.xml")
    inputs = read_construction(channel_dir, "aiida").gate_inputs(
        1 if channel == "down" else 0
    )
    extra = {} if target is None else {"target": target(parent, inputs)}
    return Construction(
        f"{material}-{arm}-{channel}",
        parent=parent,
        wannier=read_run(channel_dir / "aiida.wout"),
        material_class=gate["material_class"],
        inputs=inputs,
        **extra,
    )


def _same_up_to_rounding(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-9, abs=1e-15)
    return a == b


@pytest.mark.skipif(not _RAW.is_dir(), reason=f"no QE rebuild raw tree at {_RAW}")
def test_no_target_reproduces_every_rebuild_verdict():
    channels = _rebuild_channels()
    assert len(channels) >= 50
    for material, arm, channel, gate in channels:
        report = CertificationGate(CORE_CHECKS).run(
            _rebuild_construction(material, arm, channel, gate)
        )
        where = f"{material}/{arm}/{channel}"
        assert bool(report.passed) == gate["certified"], where
        assert list(report.failed_check_ids) == gate["failed_checks"], where
        assert (
            list(report.non_evaluable_check_ids) == gate["non_evaluable_checks"]
        ), where
        measurements = report.measurements()
        # The check this change touches: byte for byte, not merely close.
        assert {k: measurements[CHECK][k] for k in BASE_KEYS} == gate["measurements"][
            CHECK
        ], where
        assert measurements[CHECK]["target_window_source"] == "deck", where
        # The rest is untouched code; the committed record was written on another node,
        # whose BLAS can differ in the last bit of a norm or a root-mean-square.
        for check_id, recorded in gate["measurements"].items():
            for key, value in recorded.items():
                assert _same_up_to_rounding(measurements[check_id][key], value), (
                    where,
                    check_id,
                    key,
                )


def _own_window(parent, inputs):
    from autowan.gate import CertificationTarget

    froz = float(str(inputs["dis_froz_max"]).lower().replace("d", "e"))
    return CertificationTarget(
        "whole_occupied_manifold", (-math.inf, froz - parent.fermi_energy_ev)
    )


@pytest.mark.skipif(not _RAW.is_dir(), reason=f"no QE rebuild raw tree at {_RAW}")
def test_a_target_equal_to_each_rebuild_decks_own_window_changes_no_verdict():
    for material, arm, channel, gate in _rebuild_channels():
        untargeted = CertificationGate(CORE_CHECKS).run(
            _rebuild_construction(material, arm, channel, gate)
        )
        targeted = CertificationGate(CORE_CHECKS).run(
            _rebuild_construction(material, arm, channel, gate, target=_own_window)
        )
        where = f"{material}/{arm}/{channel}"
        assert targeted.passed == untargeted.passed, where
        assert targeted.failed_check_ids == untargeted.failed_check_ids, where
        a, b = targeted.measurements()[CHECK], untargeted.measurements()[CHECK]
        assert {k: a[k] for k in BASE_KEYS} == {k: b[k] for k in BASE_KEYS}, where
        assert a.get("refusal") is None, where
