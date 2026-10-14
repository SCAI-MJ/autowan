"""When a Wannierisation fails because of the DFT, say what to change -- never change it.

The band-distance check measures a construction against the parent bands, so a tool
that could re-run or edit the parent could move its own reference. autowan therefore
never alters the DFT. What it does instead is judge whether the parent can support the
deck at all, before the first build (pre-flight), and, when a run ends uncertified,
whether a parent limit stood in the way (terminal diagnosis). In both cases the output
is a message stating what to change in the DFT, plus the same change as structured
data a *caller* may apply to a copy.

One engine-agnostic judgement object decides; each backend's knowledge partition
parses its engine's files and words the prescription in its engine's grammar
(``nbnd`` / ``nosym`` / ``nspin`` for Quantum ESPRESSO).

Severity is part of the finding, because the three kinds ask different things of the
caller:

* BLOCKING -- the deck cannot run on this parent (too few bands for what the deck
  draws from, a k-list that is not the deck's full grid, a declared magnet computed
  without spin). Only these establish a parent cause.
* LIMITING -- one construction route is limited by the parent (too few pseudo-atomic
  functions for the states at the Fermi level); another route on the same parent may
  work, so this is never reported as the parent's fault.
* ADVISORY -- evidence that warrants a convergence check (hoppings still large at the
  edge of the supercell the mesh can represent).
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan import parent_adequacy as pa
from autowan.actions import EDITABLE_PARAMETERS
from autowan.backends import Backend, get_backend
from autowan.execution import ConstructionRunner, StepResult
from autowan.inputs import read_construction
from autowan.parent_adequacy import (
    ADEQUATE,
    ADVISORY,
    BLOCKING,
    INADEQUATE,
    LIMITING,
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    PARENT_ADEQUACY_CHECKS,
    AttemptEvidence,
    Channel,
    DeckDemand,
    DftChange,
    Finding,
    ParentAdequacyReport,
    ParentFacts,
    Prescription,
    TerminalDiagnosis,
    assess_parent,
    diagnose_terminal,
)
from autowan.providers import ModelResponse

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


FIXTURES = Path(__file__).parent / "fixtures" / "parent_adequacy_qe"
EXAMPLES = Path(__file__).parent.parent / "examples"
FROZEN = _stored("AUTOWAN_TEST_FROZEN")

BAND_COUNT = "band_count_covers_demand"
KLIST = "kpoints_are_the_decks_full_grid"
SPIN = "spin_treatment_matches_magnetic_state"
INVENTORY = "projector_inventory_covers_fermi_level"
HOPPING = "mesh_resolves_hopping_decay"
BAND_PARTITION = "band_partition_matches_deck"


# ---------------------------------------------------------------------------
# synthetic parents for the engine-agnostic object
# ---------------------------------------------------------------------------


def full_grid(n1, n2=None, n3=None) -> np.ndarray:
    """The Gamma-centred grid in Wannier90's (kmesh.pl) order: first index slowest."""
    n2 = n1 if n2 is None else n2
    n3 = n1 if n3 is None else n3
    return np.array(
        [
            [i / n1, j / n2, k / n3]
            for i in range(n1)
            for j in range(n2)
            for k in range(n3)
        ]
    )


CUBIC = np.eye(3) * 3.0


def facts(
    *,
    n_bands=16,
    kpoints=None,
    n_spin=1,
    fermi=0.0,
    eigen=None,
    n_symmetries=1,
    spin_polarised=None,
    moment=None,
    projector_functions=8,
    channels=None,
    species=("A", "B"),
    species_order=("A", "B"),
    requested_moments=None,
) -> ParentFacts:
    """A parent whose band ``b`` sits at ``b - 4`` eV at every k: four bands below E_F.

    Sixteen bands by default, so the default deck (num_bands = 16, nothing excluded)
    partitions it exactly; a larger default would be a surplus pair (2026-10-01)."""
    kpoints = full_grid(2) if kpoints is None else np.asarray(kpoints, dtype=float)
    if eigen is None:
        eigen = np.broadcast_to(
            np.arange(n_bands, dtype=float) - 3.5, (n_spin, len(kpoints), n_bands)
        ).copy()
    return ParentFacts(
        engine="synthetic",
        eigenvalues_ev=np.asarray(eigen, dtype=float),
        fermi_energy_ev=fermi,
        kpoints_frac=kpoints,
        lattice_ang=CUBIC,
        species=species,
        n_symmetries=n_symmetries,
        spin_polarised=(n_spin == 2) if spin_polarised is None else spin_polarised,
        noncollinear=False,
        moment_bohr=moment,
        projector_functions=projector_functions,
        channels=channels or {},
        species_order=species_order,
        requested_moments=requested_moments or {},
        source="synthetic",
    )


def demand(
    *,
    num_wann=8,
    num_bands=16,
    excluded=(),
    mp_grid=(2, 2, 2),
    kpoints="grid",
    frozen=(None, 2.0),
    route="pseudo_atomic",
    spin_channel=None,
) -> DeckDemand:
    k = full_grid(*mp_grid) if (kpoints == "grid" and mp_grid) else kpoints
    return DeckDemand(
        num_wann=num_wann,
        num_bands=num_bands,
        excluded_bands=tuple(excluded),
        mp_grid=mp_grid,
        kpoints_frac=None if k is None else np.asarray(k, dtype=float),
        frozen_window_ev=frozen,
        projector_route=route,
        spin_channel=spin_channel,
    )


def fake_prescribe(finding, parent, deck):
    """A partition that is not any engine, so the wording is visibly the partition's."""
    return Prescription(
        check_id=finding.check_id,
        message=f"FAKE-ENGINE: fix {finding.check_id}",
        changes=(DftChange("run", "input", finding.check_id, "fixed"),),
    )


def assess(f, d, **kw):
    kw.setdefault("prescribe", fake_prescribe)
    return assess_parent(f, d, **kw)


# ---------------------------------------------------------------------------
# the object: shape and vocabulary
# ---------------------------------------------------------------------------


def test_the_checks_are_the_six_the_design_names_each_with_its_severity():
    """The band-partition check (2026-10-01) is appended, so the five earlier
    findings keep their positions in every serialised report."""
    assert PARENT_ADEQUACY_CHECKS == (
        BAND_COUNT,
        KLIST,
        SPIN,
        INVENTORY,
        HOPPING,
        BAND_PARTITION,
    )
    assert pa.SEVERITY == {
        BAND_COUNT: BLOCKING,
        KLIST: BLOCKING,
        SPIN: BLOCKING,
        INVENTORY: LIMITING,
        HOPPING: ADVISORY,
        BAND_PARTITION: BLOCKING,
    }


def test_every_check_reports_a_finding_even_when_it_cannot_be_evaluated():
    report = assess(facts(), demand())
    assert tuple(f.check_id for f in report.findings) == PARENT_ADEQUACY_CHECKS
    for f in report.findings:
        assert f.status in (ADEQUATE, INADEQUATE, NOT_EVALUABLE, NOT_APPLICABLE)
        assert f.severity == pa.SEVERITY[f.check_id]
        assert isinstance(f.reason, str) and f.reason


def test_an_adequate_parent_raises_no_alarm():
    report = assess(facts(), demand(), declared_moments={})
    assert report.alarms == ()
    assert report.blocking == ()
    assert report.adequate
    assert report.finding(BAND_COUNT).status == ADEQUATE
    assert report.finding(KLIST).status == ADEQUATE
    assert report.finding(SPIN).status == ADEQUATE
    assert report.finding(INVENTORY).status == ADEQUATE
    assert report.prescription_text() == ""


def test_the_report_serialises_to_json():
    report = assess(facts(n_bands=12), demand(), declared_moments={})
    blob = json.loads(json.dumps(report.to_dict()))
    ids = [f["check_id"] for f in blob["findings"]]
    assert ids == list(PARENT_ADEQUACY_CHECKS)
    band = next(f for f in blob["findings"] if f["check_id"] == BAND_COUNT)
    assert band["status"] == INADEQUATE and band["severity"] == BLOCKING
    assert band["prescription"]["message"].startswith("FAKE-ENGINE")
    assert band["prescription"]["changes"][0]["key"] == BAND_COUNT


# ---------------------------------------------------------------------------
# band count: the parent must carry what the deck draws from
# ---------------------------------------------------------------------------


def test_too_few_bands_for_the_decks_disentanglement_is_blocking():
    """Si's control: 8 functions, a deck drawing from 16 bands, a parent with 12."""
    report = assess(facts(n_bands=12), demand(num_wann=8, num_bands=16))
    f = report.finding(BAND_COUNT)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    assert f.measured["parent_bands"] == 12
    assert f.measured["required_bands"] == 16
    assert f in report.blocking
    assert f.prescription is not None


def test_the_band_count_the_deck_needs_includes_the_bands_it_excludes():
    """A deck excluding 35-68 was written for a 68-band parent: num_bands + excluded."""
    excluded = tuple(range(35, 69))
    report = assess(
        facts(n_bands=50), demand(num_wann=17, num_bands=34, excluded=excluded)
    )
    f = report.finding(BAND_COUNT)
    assert f.status == INADEQUATE
    assert f.measured["required_bands"] == 68


def test_more_bands_than_the_deck_expects_is_not_the_parents_fault():
    report = assess(facts(n_bands=24), demand(num_wann=8, num_bands=16))
    f = report.finding(BAND_COUNT)
    assert f.status == ADEQUATE
    assert f not in report.alarms


# ---------------------------------------------------------------------------
# band partition: the parent's bands must split exactly into the deck's num_bands
# plus its excluded bands, or the overlaps span a different band set
# ---------------------------------------------------------------------------


def test_a_surplus_parent_under_a_fixed_num_bands_deck_raises_the_partition_alarm():
    """The 2026-09-29 VASP blind spot: 24 bands under a deck fixing num_bands = 16 with
    no exclude_bands. The overlaps span all 24, the deck asks for 16, and autowan's
    deck-overlap check refuses the construction -- so the pre-flight must say so."""
    report = assess(facts(n_bands=24), demand(num_wann=8, num_bands=16))
    f = report.finding(BAND_PARTITION)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    assert f.alarm and f in report.blocking
    assert f.measured["parent_bands"] == 24
    assert f.measured["num_bands"] == 16
    assert f.measured["excluded"] == 0
    assert f.measured["deck_bands"] == 16
    assert f.measured["overlap_bands"] == 24
    assert f.measured["surplus_bands"] == 8
    assert f.measured["surplus_band_range"] == [17, 24]
    assert "24" in f.reason and "16" in f.reason
    assert f.prescription is not None
    # The band-count threshold is unchanged: a surplus is still not a shortfall.
    assert report.finding(BAND_COUNT).status == ADEQUATE
    assert [x.check_id for x in report.blocking] == [BAND_PARTITION]


def test_the_partition_counts_the_bands_the_deck_already_excludes():
    """A deck excluding 1-4 and fixing num_bands = 16 partitions 20 bands; a 26-band
    parent leaves six over, and the overlaps would span 22."""
    report = assess(
        facts(n_bands=26),
        demand(num_wann=8, num_bands=16, excluded=(1, 2, 3, 4)),
    )
    f = report.finding(BAND_PARTITION)
    assert f.status == INADEQUATE
    assert f.measured["deck_bands"] == 20
    assert f.measured["overlap_bands"] == 22
    assert f.measured["surplus_bands"] == 6
    assert f.measured["surplus_band_range"] == [21, 26]


@pytest.mark.parametrize(
    "n_bands, num_bands, excluded",
    [
        (16, 16, ()),  # the deck written for its own parent
        (20, 10, tuple(range(11, 21))),  # Ba's automatic deck: the surplus excluded
        (9, 2, tuple(range(3, 10))),  # Na's automatic deck
        (24, 16, tuple(range(17, 25))),  # the surplus parent with the surplus excluded
        (24, 24, ()),  # the surplus parent under a deck written for it
    ],
)
def test_a_correctly_partitioned_parent_raises_no_partition_alarm(
    n_bands, num_bands, excluded
):
    report = assess(
        facts(n_bands=n_bands),
        demand(num_wann=min(8, num_bands), num_bands=num_bands, excluded=excluded),
    )
    f = report.finding(BAND_PARTITION)
    assert f.status == ADEQUATE
    assert f not in report.alarms
    assert f.measured["surplus_bands"] == 0


def test_a_parent_short_of_the_deck_is_the_band_count_finding_not_a_partition_alarm():
    """The 2026-09-29 band-count causal pair (Si: 12 bands under the 16-band deck) must
    keep exactly its one BLOCKING alarm."""
    report = assess(facts(n_bands=12), demand(num_wann=8, num_bands=16))
    assert [f.check_id for f in report.blocking] == [BAND_COUNT]
    f = report.finding(BAND_PARTITION)
    assert f.status == NOT_APPLICABLE
    assert f not in report.alarms


def test_a_deck_that_does_not_fix_num_bands_raises_no_partition_alarm():
    report = assess(facts(n_bands=24), demand(num_wann=8, num_bands=None))
    f = report.finding(BAND_PARTITION)
    assert f.status == NOT_APPLICABLE
    assert f not in report.alarms


def test_the_partition_alarm_names_whether_the_deck_bands_close_the_frozen_window():
    """Dropping the surplus from the parent is only a remedy if the bands the deck
    partitions still close its frozen window at every point."""
    closes = assess(facts(n_bands=24), demand(num_bands=16, frozen=(None, 2.0)))
    f = closes.finding(BAND_PARTITION)
    assert f.measured["frozen_window_closed_by_deck_bands"] is True
    assert f.measured["prescribed_bands"] == 16

    eig = facts(n_bands=24).eigenvalues_ev.copy()
    eig[0, 2, :16] = np.linspace(-3.5, 1.5, 16)  # the first 16 all below 2 eV here
    open_ = assess(
        facts(n_bands=24, eigen=eig), demand(num_bands=16, frozen=(None, 2.0))
    )
    g = open_.finding(BAND_PARTITION)
    assert g.status == INADEQUATE
    assert g.measured["frozen_window_closed_by_deck_bands"] is False
    assert g.measured["prescribed_bands"] is None

    unknown = assess(facts(n_bands=24), demand(num_bands=16, frozen=(None, None)))
    h = unknown.finding(BAND_PARTITION)
    assert h.measured["frozen_window_closed_by_deck_bands"] is None
    assert h.measured["prescribed_bands"] == 16


def test_a_partition_alarm_establishes_a_contract_violation_in_the_diagnosis():
    report = assess(facts(n_bands=24), demand(num_wann=8, num_bands=16))
    dx = diagnose_terminal(report, [])
    assert dx.cause_established is True
    assert [e.check_id for e in dx.events] == [BAND_PARTITION]
    assert f"FAKE-ENGINE: fix {BAND_PARTITION}" in dx.message()


def test_a_parent_spectrum_that_ends_inside_the_frozen_window_is_blocking():
    """Enough bands for the deck, but at one k-point every computed band lies below
    the top of the frozen window: the parent never computed the states that close it."""
    eig = facts(n_bands=16).eigenvalues_ev.copy()
    eig[0, 3, :] = np.linspace(-3.5, 1.0, 16)  # all sixteen below dis_froz_max = 2 eV
    report = assess(
        facts(n_bands=16, eigen=eig), demand(num_bands=16, frozen=(None, 2.0))
    )
    f = report.finding(BAND_COUNT)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    assert f.measured["prescribed_bands"] > 16


def test_the_frozen_window_is_closed_by_the_parent_not_by_the_decks_included_bands():
    """Na's automatic deck keeps 2 of 9 bands, and both lie inside its frozen window at
    some k-point. Bands 3-9 exist in the parent; the deck excluded them. Not a parent
    shortage."""
    eig = facts(n_bands=9).eigenvalues_ev.copy()
    eig[0, :, 0] = -1.0
    eig[0, :, 1] = 1.5  # below dis_froz_max = 2 everywhere
    report = assess(
        facts(n_bands=9, eigen=eig, projector_functions=1),
        demand(num_wann=1, num_bands=2, excluded=range(3, 10), frozen=(None, 2.0)),
    )
    assert report.finding(BAND_COUNT).status == ADEQUATE


# ---------------------------------------------------------------------------
# k-list: the parent must hold the deck's full grid, in the deck's order
# ---------------------------------------------------------------------------


def test_a_symmetry_reduced_klist_is_blocking():
    """Cu's control: the automatic grid folded by the point group -- a few of the
    deck's points, not all of them."""
    reduced = full_grid(2)[[0, 1, 3, 7]]
    report = assess(facts(kpoints=reduced, n_symmetries=48), demand(mp_grid=(2, 2, 2)))
    f = report.finding(KLIST)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    assert f.measured["parent_kpoints"] == 4
    assert f.measured["deck_kpoints"] == 8
    assert f.measured["missing"] == 4
    assert f.measured["symmetry_reduced"] is True


def test_a_parent_on_another_grid_is_blocking():
    report = assess(facts(kpoints=full_grid(3)), demand(mp_grid=(2, 2, 2)))
    assert report.finding(KLIST).status == INADEQUATE


def test_a_shifted_grid_of_the_right_size_is_not_the_decks_grid():
    shifted = full_grid(2) + 0.25
    report = assess(facts(kpoints=shifted), demand(mp_grid=(2, 2, 2)))
    f = report.finding(KLIST)
    assert f.status == INADEQUATE
    assert f.measured["missing"] == 8


def test_points_equal_modulo_a_reciprocal_vector_are_the_same_point():
    wrapped = full_grid(2)
    wrapped[wrapped > 0.4] -= 1.0  # 0.5 written as -0.5
    report = assess(facts(kpoints=wrapped), demand(mp_grid=(2, 2, 2)))
    assert report.finding(KLIST).status == ADEQUATE


def test_the_decks_points_in_another_order_are_not_the_decks_klist():
    """The overlaps are indexed by position; the same set in another order pairs each
    k-point's states with another k-point's neighbours."""
    reordered = full_grid(2)[::-1]
    report = assess(facts(kpoints=reordered), demand(mp_grid=(2, 2, 2)))
    f = report.finding(KLIST)
    assert f.status == INADEQUATE
    assert f.measured["order_matches"] is False


def test_without_a_declared_mesh_the_klist_cannot_be_judged():
    report = assess(facts(), demand(mp_grid=None, kpoints=None))
    assert report.finding(KLIST).status == NOT_EVALUABLE
    assert report.finding(KLIST) not in report.alarms


# ---------------------------------------------------------------------------
# spin: the parent's spin treatment against the declared magnetic state
# ---------------------------------------------------------------------------


def test_a_declared_magnet_computed_without_spin_is_blocking():
    """MnO's control: the ferromagnet computed spin-unpolarised."""
    report = assess(
        facts(species=("Mn", "O"), species_order=("Mn", "O")),
        demand(),
        declared_moments={"Mn": 5.0, "O": 0.0},
    )
    f = report.finding(SPIN)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    assert f.measured["spin_polarised"] is False


def test_a_declared_magnet_with_its_moment_is_adequate():
    report = assess(
        facts(n_spin=2, moment=4.85, species=("Mn", "O"), species_order=("Mn", "O")),
        demand(spin_channel=0),
        declared_moments={"Mn": 5.0, "O": 0.0},
    )
    assert report.finding(SPIN).status == ADEQUATE


def test_a_spin_polarised_parent_that_lost_its_declared_moment_is_blocking():
    report = assess(
        facts(n_spin=2, moment=0.02, species=("Mn", "O"), species_order=("Mn", "O")),
        demand(spin_channel=0),
        declared_moments={"Mn": 5.0, "O": 0.0},
    )
    f = report.finding(SPIN)
    assert f.status == INADEQUATE
    assert f.measured["moment_bohr"] == pytest.approx(0.02)


def test_a_compensated_declaration_does_not_judge_the_net_moment():
    """Equal and opposite moments sum to zero: a zero net moment is then no evidence of
    a collapse, so the moment branch is not applied."""
    report = assess(
        facts(n_spin=2, moment=0.0, species=("Fe", "Mn"), species_order=("Fe", "Mn")),
        demand(spin_channel=0),
        declared_moments={"Fe": 3.0, "Mn": -3.0},
    )
    assert report.finding(SPIN).status == ADEQUATE


def test_without_a_declaration_the_spin_check_is_not_evaluable_not_passed():
    report = assess(facts(), demand(), declared_moments=None)
    f = report.finding(SPIN)
    assert f.status == NOT_EVALUABLE
    assert f in report.not_evaluable
    assert f not in report.alarms


def test_the_parents_own_request_for_magnetism_counts_as_a_declaration():
    """A parent whose input asked for a starting moment and ran without spin anyway."""
    report = assess(
        facts(
            species=("Mn", "O"),
            species_order=("Mn", "O"),
            requested_moments={"Mn": 0.6},
        ),
        demand(),
        declared_moments=None,
    )
    assert report.finding(SPIN).status == INADEQUATE


def test_a_starting_magnetisation_is_not_a_moment_to_judge_a_collapse_against():
    """Quantum ESPRESSO's starting_magnetization is a fraction of the valence charge,
    not Bohr magnetons: a parent-input request says the material is meant to be
    magnetic, and nothing about how large its converged moment should be."""
    report = assess(
        facts(
            n_spin=2,
            moment=0.1,
            species=("Mn", "O"),
            species_order=("Mn", "O"),
            requested_moments={"Mn": 0.6},
        ),
        demand(spin_channel=0),
        declared_moments=None,
    )
    f = report.finding(SPIN)
    assert f.status == ADEQUATE
    assert f.measured["source"] == "parent input"


def test_no_declaration_is_recorded_as_none_not_as_the_parents():
    report = assess(facts(), demand(), declared_moments=None)
    assert report.finding(SPIN).measured["source"] == "none"


def test_a_declared_non_magnet_is_adequate_whatever_the_spin_treatment():
    report = assess(facts(n_spin=2, moment=0.0), demand(), declared_moments={})
    assert report.finding(SPIN).status == ADEQUATE


# ---------------------------------------------------------------------------
# projector inventory: a limit of the pseudo-atomic route, never a parent cause
# ---------------------------------------------------------------------------


def _ba_like_eigen():
    """Five pseudo-atomic functions; at one k-point six states lie at or below E_F."""
    eig = np.broadcast_to(np.arange(20, dtype=float) + 1.0, (1, 8, 20)).copy()
    eig[0, :, :4] = [-30.0, -15.0, -15.0, -15.0]  # 5s, 5p
    eig[0, :, 4] = -1.0
    eig[0, 0, 5] = -0.2  # a second valence band dips below E_F at one k-point
    return eig


def test_fewer_pseudo_atomic_functions_than_states_at_the_fermi_level_is_limiting():
    report = assess(
        facts(
            n_bands=20,
            eigen=_ba_like_eigen(),
            projector_functions=5,
            channels={"Ba": (Channel("5S", 0), Channel("5P", 1), Channel("6S", 0))},
            species=("Ba",),
            species_order=("Ba",),
        ),
        demand(num_wann=5, num_bands=10, excluded=range(11, 21), frozen=(None, 2.0)),
        declared_moments={},
    )
    f = report.finding(INVENTORY)
    assert f.status == INADEQUATE
    assert f.severity == LIMITING
    assert f.measured["projector_functions"] == 5
    assert f.measured["states_at_fermi_level"] == 6
    assert report.blocking == ()  # a route limit: the parent itself can still be used


def test_a_frozen_window_that_over_demands_an_adequate_inventory_is_deck_side():
    """Na: one pseudo-atomic function, one state at E_F, two inside E_F + 2 eV. Lowering
    the frozen window is a deck change; the parent is not at fault."""
    eig = np.broadcast_to(np.arange(9, dtype=float) + 3.0, (1, 8, 9)).copy()
    eig[0, :, 0] = -0.5
    eig[0, 0, 1] = 1.0  # inside the frozen window at one k-point
    report = assess(
        facts(
            n_bands=9,
            eigen=eig,
            projector_functions=1,
            species=("Na",),
            species_order=("Na",),
        ),
        demand(num_wann=1, num_bands=2, excluded=range(3, 10), frozen=(None, 2.0)),
        declared_moments={},
    )
    f = report.finding(INVENTORY)
    assert f.status == ADEQUATE
    assert f.measured["states_in_frozen_window"] == 2
    assert f.measured["states_at_fermi_level"] == 1
    assert report.alarms == ()


def test_the_inventory_does_not_apply_to_named_analytic_projectors():
    report = assess(facts(projector_functions=1), demand(route="analytic"))
    assert report.finding(INVENTORY).status == NOT_APPLICABLE


def test_an_unknown_inventory_is_not_evaluable():
    report = assess(facts(projector_functions=None), demand())
    assert report.finding(INVENTORY).status == NOT_EVALUABLE


def test_the_inventory_counts_the_spin_channel_the_deck_builds():
    eig = np.broadcast_to(np.arange(20, dtype=float) + 1.0, (2, 8, 20)).copy()
    eig[1, :, :9] = -1.0  # nine minority states below E_F; majority has none
    f_up = assess(
        facts(n_spin=2, moment=5.0, eigen=eig, projector_functions=8),
        demand(spin_channel=0),
    )
    f_dn = assess(
        facts(n_spin=2, moment=5.0, eigen=eig, projector_functions=8),
        demand(spin_channel=1),
    )
    assert f_up.finding(INVENTORY).status == ADEQUATE
    assert f_dn.finding(INVENTORY).status == INADEQUATE


# ---------------------------------------------------------------------------
# mesh vs hopping decay: only once a construction exists, and only advisory
# ---------------------------------------------------------------------------


def _hamiltonian(decay_length_ang, mesh=4, a=3.0, flat=False):
    n = mesh // 2
    rvec = np.array(
        [
            [i, j, k]
            for i in range(-n, n + 1)
            for j in range(-n, n + 1)
            for k in range(-n, n + 1)
        ]
    )
    d = np.linalg.norm(rvec * a, axis=1)
    h = np.where(d == 0, 1.0, 0.3 if flat else np.exp(-d / decay_length_ang))
    return SimpleNamespace(
        hr_rvectors=rvec, hr_matrix=h.reshape(-1, 1, 1).astype(complex)
    )


def test_hopping_decay_is_not_evaluable_before_a_construction_exists():
    report = assess(facts(), demand(mp_grid=(4, 4, 4)))
    assert report.finding(HOPPING).status == NOT_EVALUABLE


def test_hoppings_still_large_at_the_supercell_edge_warrant_a_denser_mesh():
    report = assess(
        facts(kpoints=full_grid(4)),
        demand(mp_grid=(4, 4, 4)),
        hamiltonian=_hamiltonian(3.0),
    )
    f = report.finding(HOPPING)
    assert f.status == INADEQUATE and f.severity == ADVISORY
    assert all(n > 4 for n in f.measured["prescribed_mesh"])
    assert f.measured["decay_length_ang"] == pytest.approx(3.0, rel=0.05)
    assert report.blocking == ()


def test_hoppings_that_have_decayed_by_the_supercell_edge_are_adequate():
    report = assess(
        facts(kpoints=full_grid(4)),
        demand(mp_grid=(4, 4, 4)),
        hamiltonian=_hamiltonian(0.8),
    )
    assert report.finding(HOPPING).status == ADEQUATE


def test_hoppings_that_do_not_decay_are_the_constructions_not_the_meshs():
    report = assess(
        facts(kpoints=full_grid(4)),
        demand(mp_grid=(4, 4, 4)),
        hamiltonian=_hamiltonian(3.0, flat=True),
    )
    f = report.finding(HOPPING)
    assert f.status == NOT_EVALUABLE
    assert f not in report.alarms


def test_hoppings_that_decay_over_more_than_the_supercell_are_not_a_mesh_verdict():
    """A decay length longer than the supercell the mesh represents is a construction
    that is not localised, not a mesh that is too coarse; extrapolating it would
    prescribe an absurd mesh."""
    n = 2
    rvec = np.array(
        [
            [i, j, k]
            for i in range(-n, n + 1)
            for j in range(-n, n + 1)
            for k in range(-n, n + 1)
        ]
    )
    d = np.linalg.norm(rvec * 3.0, axis=1)
    h = np.where(d == 0, 1.0, 0.3 * np.exp(-d / 1000.0))
    ham = SimpleNamespace(
        hr_rvectors=rvec, hr_matrix=h.reshape(-1, 1, 1).astype(complex)
    )
    report = assess(
        facts(kpoints=full_grid(4)), demand(mp_grid=(4, 4, 4)), hamiltonian=ham
    )
    f = report.finding(HOPPING)
    assert f.status == NOT_EVALUABLE
    assert f.measured["prescribed_mesh"] is None


def test_the_hopping_threshold_is_the_gates_own_not_a_second_constant():
    from autowan.core_checks import CORE_CHECKS

    gate_fraction = next(
        c.threshold.fraction
        for c in CORE_CHECKS
        if c.check_id == "hopping_decays_with_distance"
    )
    report = assess(
        facts(kpoints=full_grid(4)),
        demand(mp_grid=(4, 4, 4)),
        hamiltonian=_hamiltonian(3.0),
    )
    assert report.finding(HOPPING).measured["threshold_fraction"] == gate_fraction


# ---------------------------------------------------------------------------
# engine-agnostic: the wording is the partition's
# ---------------------------------------------------------------------------


def test_the_prescription_wording_comes_from_the_partition():
    report = assess(facts(n_bands=12), demand())
    f = report.finding(BAND_COUNT)
    assert f.prescription.message == f"FAKE-ENGINE: fix {BAND_COUNT}"
    assert report.prescription_text().count("FAKE-ENGINE") == len(report.alarms)


def test_without_a_partition_findings_carry_no_engine_wording():
    report = assess_parent(facts(n_bands=12), demand(), prescribe=None)
    assert report.finding(BAND_COUNT).status == INADEQUATE
    assert report.finding(BAND_COUNT).prescription is None


def test_the_judgement_object_names_no_engine_parameter():
    source = inspect.getsource(pa)
    for word in (
        "nbnd",
        "nosym",
        "nspin",
        "K_POINTS",
        "starting_magnetization",
        "NBANDS",
        "ISYM",
        "ISPIN",
    ):
        assert word not in source, word


# ---------------------------------------------------------------------------
# never alter the DFT
# ---------------------------------------------------------------------------

PARENT_KNOBS = {
    "nbnd",
    "nosym",
    "noinv",
    "nspin",
    "starting_magnetization",
    "k_points",
    "kpoints",
    "mp_grid",
    "ecutwfc",
    "ecutrho",
    "pseudopotential",
    "pseudo_dir",
    "nbands",
    "isym",
    "ispin",
    "magmom",
}


def test_the_repair_vocabulary_still_has_no_parent_knob():
    assert {p.name.lower() for p in EDITABLE_PARAMETERS}.isdisjoint(PARENT_KNOBS)


def test_a_prescription_is_data_and_has_no_way_to_apply_itself():
    assert {f.name for f in fields(Prescription)} == {"check_id", "message", "changes"}
    assert {f.name for f in fields(DftChange)} == {"step", "section", "key", "value"}
    for cls in (
        Prescription,
        DftChange,
        Finding,
        ParentAdequacyReport,
        TerminalDiagnosis,
    ):
        assert not any(name.startswith("apply") for name in dir(cls)), cls


# ---------------------------------------------------------------------------
# the QE partition: parsing and wording
# ---------------------------------------------------------------------------

ALAT_BOHR = 10.0
FCC = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]) * ALAT_BOHR
HARTREE = 27.211386245988


def _cart_2pi_alat(kfrac):
    """Crystal k to Quantum ESPRESSO's cartesian units of 2 pi / alat."""
    recip = np.linalg.inv(FCC).T  # rows b_i / (2 pi), in 1/bohr
    return np.asarray(kfrac) @ recip * ALAT_BOHR


def write_qe_xml(
    path: Path,
    *,
    kfrac,
    nbnd=16,
    lsda=False,
    nsym=1,
    nosym=True,
    species=(("Mn", "Mn.upf", 0.0), ("O", "O.upf", 0.0)),
    atoms=(("Mn", (0, 0, 0)), ("O", (0.5, 0.5, 0.5))),
    n_atomic_wfc=17,
    pseudo_dir="",
    fermi_ev=0.0,
    up_electrons=None,
    down_electrons=None,
):
    """A Quantum ESPRESSO 7.5 data-file-schema.xml carrying what the partition reads.

    Tags and nesting follow the real files (compare the Ba/Na fixtures): cartesian
    k-points in 2 pi/alat, energies in Hartree, spin channels concatenated along the
    band axis within one <ks_energies> for an LSDA run.
    """
    kfrac = np.asarray(kfrac, dtype=float)
    kcart = _cart_2pi_alat(kfrac)
    n_spin = 2 if lsda else 1
    w = 1.0 / len(kfrac)
    eig = np.arange(nbnd, dtype=float) - 3.5  # eV
    ks = []
    for kc in kcart:
        occ_blocks = []
        for s in range(n_spin):
            filled = (up_electrons if s == 0 else down_electrons) if lsda else None
            if filled is None:
                filled = 4
            occ_blocks.append([1.0 if b < filled else 0.0 for b in range(nbnd)])
        e = np.concatenate([eig] * n_spin) / HARTREE
        o = np.concatenate(occ_blocks)
        ks.append(
            f'<ks_energies><k_point weight="{w * (2 if not lsda else 1)}">'
            f"{kc[0]:.12f} {kc[1]:.12f} {kc[2]:.12f}</k_point>"
            f"<npw>100</npw>"
            f"<eigenvalues size=\"{len(e)}\">{' '.join(f'{x:.12e}' for x in e)}</eigenvalues>"
            f"<occupations size=\"{len(o)}\">{' '.join(f'{x:.6e}' for x in o)}</occupations>"
            f"</ks_energies>"
        )
    sp_in = "".join(
        f'<species name="{n}"><mass>1.0</mass><pseudo_file>{pp}</pseudo_file>'
        + (f"<starting_magnetization>{m}</starting_magnetization>" if m else "")
        + "</species>"
        for n, pp, m in species
    )
    sp_out = "".join(
        f'<species name="{n}"><mass>1.0</mass><pseudo_file>{pp}</pseudo_file></species>'
        for n, pp, _m in species
    )
    cell = "".join(
        f"<a{i + 1}>{' '.join(f'{x:.10f}' for x in FCC[i])}</a{i + 1}>"
        for i in range(3)
    )
    pos = "".join(
        f'<atom name="{n}" index="{i + 1}">'
        + " ".join(f"{x:.10f}" for x in (np.asarray(f, dtype=float) @ FCC))
        + "</atom>"
        for i, (n, f) in enumerate(atoms)
    )
    nbnd_tag = (
        f"<nbnd_up>{nbnd}</nbnd_up><nbnd_dw>{nbnd}</nbnd_dw>"
        if lsda
        else f"<nbnd>{nbnd}</nbnd>"
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0" Units="Hartree atomic units">
<input>
<control_variables><calculation>nscf</calculation><prefix>X</prefix><pseudo_dir>{pseudo_dir}</pseudo_dir></control_variables>
<atomic_species ntyp="{len(species)}">{sp_in}</atomic_species>
<spin><lsda>{str(lsda).lower()}</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit></spin>
<bands><nbnd>{nbnd}</nbnd></bands>
<symmetry_flags><nosym>{str(nosym).lower()}</nosym><noinv>false</noinv></symmetry_flags>
</input>
<output>
<atomic_species ntyp="{len(species)}" pseudo_dir="{pseudo_dir}">{sp_out}</atomic_species>
<atomic_structure nat="{len(atoms)}" num_of_atomic_wfc="{n_atomic_wfc}" alat="{ALAT_BOHR}" bravais_index="2">
<atomic_positions>{pos}</atomic_positions><cell>{cell}</cell></atomic_structure>
<symmetries><nsym>{nsym}</nsym></symmetries>
<magnetization><lsda>{str(lsda).lower()}</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit>
<total>38.8</total><absolute>0.0</absolute></magnetization>
<band_structure><lsda>{str(lsda).lower()}</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit>
{nbnd_tag}<nelec>8.0</nelec><fermi_energy>{fermi_ev / HARTREE}</fermi_energy><nks>{len(kfrac)}</nks>
{''.join(ks)}
</band_structure>
</output>
</qes:espresso>
"""
    path.write_text(xml)
    return path


def write_deck(
    directory: Path,
    seed="X_pdwf",
    *,
    num_wann=8,
    num_bands=16,
    exclude="",
    mesh=2,
    kfrac=None,
    froz_max=2.0,
    projections="",
    spin_component=None,
    atom_proj=True,
):
    kfrac = full_grid(mesh) if kfrac is None else kfrac
    win = [
        f"num_wann = {num_wann}",
        f"num_bands = {num_bands}",
        "auto_projections = .true." if not projections else "",
        "dis_proj_min = 0.01" if not projections else "",
        "dis_proj_max = 0.95" if not projections else "",
        f"dis_froz_max = {froz_max}",
        f"exclude_bands = {exclude}" if exclude else "",
        "conv_window = 3",
        "",
        "begin projections",
        projections,
        "end projections",
        "",
        f"mp_grid = {mesh} {mesh} {mesh}",
        "",
        "begin kpoints",
        *[f" {a:.8f} {b:.8f} {c:.8f}" for a, b, c in kfrac],
        "end kpoints",
    ]
    (directory / f"{seed}.win").write_text("\n".join(win) + "\n")
    pw = ["&inputpp", "  outdir = './'", "  prefix = 'X'", f"  seedname = '{seed}'"]
    if spin_component:
        pw.append(f"  spin_component = '{spin_component}'")
    if atom_proj and not projections:
        pw.append("  atom_proj = .true.")
    pw.append("/")
    (directory / f"{seed}.pw2wan").write_text("\n".join(pw) + "\n")
    return read_construction(directory, seed)


UPF_V1 = """<PP_INFO>
</PP_INFO>
<PP_HEADER>
   0                   Version Number
  Ba                   Element
 Wavefunctions         nl  l   occ
                       5S  0  2.00
                       5P  1  6.00
                       6S  0  2.00
   10.00000000000      Z valence
</PP_HEADER>
"""

UPF_V2 = """<UPF version="2.0.1">
<PP_HEADER element="Cu" z_valence="1.1e1" number_of_wfc="3"/>
<PP_PSWFC>
<PP_CHI.1 type="real" size="4" columns="4" index="1" label="4S" l="0" occupation="1.5" n="1">
0 0 0 0
</PP_CHI.1>
<PP_CHI.2 type="real" size="4" columns="4" index="2" label="4P" l="1" occupation="0.0" n="2">
0 0 0 0
</PP_CHI.2>
<PP_CHI.3 type="real" size="4" columns="4" index="3" label="3D" l="2" occupation="9.5" n="3">
0 0 0 0
</PP_CHI.3>
</PP_PSWFC>
</UPF>
"""


@pytest.fixture
def qe():
    return get_backend("qe")


def test_the_qe_backend_carries_an_adequacy_partition(qe):
    assert qe.adequacy is not None
    assert qe.adequacy.engine == "qe"


def test_a_backend_without_a_partition_assesses_nothing(tmp_path):
    """The VASP partition is its own task; until then its pre-flight is absent, not
    passed -- and the protocol must not require every backend to have one."""
    assert Backend.adequacy is None
    assert "assess_parent" in dict(inspect.getmembers(Backend))

    class _NoPartition(Backend):
        name = "none"
        knowledge = get_backend("qe").knowledge

        def read_parent(self, source):
            raise AssertionError

        def read_construction(self, run_dir, seed):
            raise AssertionError

        def write_derived(self, *a, **k):
            raise AssertionError

        def deck_files(self, inputs):
            return ()

        def make_runner(self, run_dir, **launch):
            raise AssertionError

        def refusal_reason(self, run_dir, seed):
            return None

    assert _NoPartition().assess_parent(tmp_path / "x.xml", None) is None


def test_the_qe_partition_reads_the_parent_facts(tmp_path, qe):
    k = full_grid(2)
    xml = write_qe_xml(
        tmp_path / "p.xml",
        kfrac=k,
        nbnd=16,
        lsda=True,
        nsym=1,
        up_electrons=9,
        down_electrons=4,
        species=(("Mn", "Mn.upf", 0.6), ("O", "O.upf", 0.0)),
    )
    f = qe.adequacy.read_facts(xml)
    assert isinstance(f, ParentFacts)
    assert f.engine == "qe"
    assert f.n_bands == 16
    assert f.eigenvalues_ev.shape == (2, 8, 16)
    assert np.allclose(
        np.mod(f.kpoints_frac + 1e-9, 1.0), np.mod(k + 1e-9, 1.0), atol=1e-6
    )
    assert f.n_symmetries == 1
    assert f.spin_polarised is True
    assert f.moment_bohr == pytest.approx(
        5.0, abs=1e-6
    )  # 9 up - 4 down, from occupations
    assert f.species == ("Mn", "O")
    assert f.species_order == ("Mn", "O")
    assert f.projector_functions == 17
    assert f.requested_moments == {"Mn": pytest.approx(0.6)}


def test_the_moment_is_read_from_occupations_not_the_nscf_magnetization_block(
    tmp_path, qe
):
    """An nscf XML carries a <magnetization> block that is not recomputed -- the frozen
    MnO parent records total 38.86 and absolute 0 for a 4.85 Bohr-magneton cell."""
    xml = write_qe_xml(
        tmp_path / "p.xml",
        kfrac=full_grid(2),
        lsda=True,
        up_electrons=6,
        down_electrons=6,
    )
    assert qe.adequacy.read_facts(xml).moment_bohr == pytest.approx(0.0, abs=1e-9)


def test_the_qe_partition_reads_the_channel_inventory_from_the_pseudopotentials(
    tmp_path, qe
):
    (tmp_path / "Ba.upf").write_text(UPF_V1)
    (tmp_path / "Cu.upf").write_text(UPF_V2)
    xml = write_qe_xml(
        tmp_path / "p.xml",
        kfrac=full_grid(2),
        pseudo_dir=str(tmp_path),
        species=(("Ba", "Ba.upf", 0.0), ("Cu", "Cu.upf", 0.0)),
        atoms=(("Ba", (0, 0, 0)), ("Cu", (0.5, 0.5, 0.5))),
        n_atomic_wfc=14,
    )
    f = qe.adequacy.read_facts(xml)
    assert [c.label for c in f.channels["Ba"]] == ["5S", "5P", "6S"]
    assert [c.l for c in f.channels["Cu"]] == [0, 1, 2]
    assert sum(c.functions for c in f.channels["Ba"]) == 5
    assert sum(c.functions for c in f.channels["Cu"]) == 9


def test_a_pseudopotential_that_cannot_be_found_leaves_the_channels_unknown(
    tmp_path, qe
):
    xml = write_qe_xml(
        tmp_path / "p.xml", kfrac=full_grid(2), pseudo_dir=str(tmp_path / "absent")
    )
    f = qe.adequacy.read_facts(xml)
    assert f.channels == {} or all(v == () for v in f.channels.values())
    assert f.projector_functions == 17  # the XML's own count still stands


def test_the_qe_partition_reads_the_decks_demand(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2), nbnd=68, lsda=True)
    inputs = write_deck(
        tmp_path,
        num_wann=17,
        num_bands=34,
        exclude="35-68",
        froz_max=15.4679,
        spin_component="up",
    )
    d = qe.adequacy.read_demand(inputs, qe.adequacy.read_facts(xml))
    assert isinstance(d, DeckDemand)
    assert (d.num_wann, d.num_bands) == (17, 34)
    assert d.excluded_bands == tuple(range(35, 69))
    assert d.mp_grid == (2, 2, 2)
    assert np.allclose(d.kpoints_frac, full_grid(2))
    assert d.frozen_window_ev == (None, pytest.approx(15.4679))
    assert d.projector_route == "pseudo_atomic"
    assert d.spin_channel == 0


def test_named_projections_are_the_analytic_route(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2))
    inputs = write_deck(tmp_path, projections="Mn:s;p;d\nO:s;p", atom_proj=False)
    d = qe.adequacy.read_demand(inputs, qe.adequacy.read_facts(xml))
    assert d.projector_route == "analytic"
    assert d.spin_channel is None


def test_exclude_bands_lists_and_ranges_are_both_read(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2), nbnd=30)
    inputs = write_deck(tmp_path, num_bands=16, exclude="1-5, 22, 24-30")
    d = qe.adequacy.read_demand(inputs, qe.adequacy.read_facts(xml))
    assert d.excluded_bands == (1, 2, 3, 4, 5, 22, 24, 25, 26, 27, 28, 29, 30)


# -- QE wording ---------------------------------------------------------------


def _changes(finding):
    return {(c.step, c.section, c.key): c.value for c in finding.prescription.changes}


def test_qe_prescribes_the_nscf_band_count_the_deck_needs(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2), nbnd=12)
    inputs = write_deck(tmp_path, num_wann=8, num_bands=16)
    report = qe.assess_parent(xml, inputs, declared_moments={})
    f = report.finding(BAND_COUNT)
    assert f.status == INADEQUATE
    assert _changes(f) == {("nscf", "system", "nbnd"): 16}
    assert "nbnd" in f.prescription.message


def test_qe_words_the_surplus_band_partition_alarm_in_its_own_terms(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2), nbnd=24)
    inputs = write_deck(tmp_path, num_wann=8, num_bands=16)
    report = qe.assess_parent(xml, inputs, declared_moments={})
    f = report.finding(BAND_PARTITION)
    assert f.status == INADEQUATE and f in report.blocking
    assert report.finding(BAND_COUNT).status == ADEQUATE
    message = f.prescription.message
    assert "24" in message and "16" in message
    assert "exclude_bands = 17-24" in message
    assert "nbnd" in message and "NBANDS" not in message
    if f.measured["prescribed_bands"] is not None:
        assert _changes(f) == {("nscf", "system", "nbnd"): 16}
    else:
        assert f.prescription.changes == ()


def test_qe_raises_no_partition_alarm_when_the_deck_excludes_the_surplus(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2), nbnd=24)
    inputs = write_deck(tmp_path, num_wann=8, num_bands=16, exclude="17-24")
    report = qe.assess_parent(xml, inputs, declared_moments={})
    assert report.finding(BAND_PARTITION).status == ADEQUATE
    assert report.blocking == ()


def test_qe_prescribes_nosym_and_the_decks_full_klist(tmp_path, qe):
    reduced = full_grid(2)[[0, 1, 3, 7]]
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=reduced, nsym=48, nosym=False)
    inputs = write_deck(tmp_path)
    report = qe.assess_parent(xml, inputs, declared_moments={})
    f = report.finding(KLIST)
    assert f.status == INADEQUATE and f.measured["symmetry_reduced"] is True
    ch = _changes(f)
    assert ch[("nscf", "system", "nosym")] is True
    points = ch[("nscf", "K_POINTS", "crystal")]
    assert len(points) == 8
    assert np.allclose([p[:3] for p in points], full_grid(2))
    assert sum(p[3] for p in points) == pytest.approx(1.0)
    assert "nosym" in f.prescription.message


def test_qe_prescribes_spin_in_both_calculations_for_the_declared_species(tmp_path, qe):
    xml = write_qe_xml(tmp_path / "p.xml", kfrac=full_grid(2))
    inputs = write_deck(tmp_path)
    report = qe.assess_parent(xml, inputs, declared_moments={"Mn": 5.0, "O": 0.0})
    f = report.finding(SPIN)
    assert f.status == INADEQUATE and f.severity == BLOCKING
    ch = _changes(f)
    assert ch[("scf", "system", "nspin")] == 2
    assert ch[("nscf", "system", "nspin")] == 2
    assert ch[("scf", "system", "starting_magnetization(1)")] > 0  # Mn is species 1
    assert ch[("nscf", "system", "starting_magnetization(1)")] > 0
    assert not any("starting_magnetization(2)" in key for (_s, _n, key) in ch)
    assert (
        "nspin" in f.prescription.message and "spin_component" in f.prescription.message
    )


def test_qe_words_the_inventory_limit_as_a_pseudopotential_matter_not_a_command(
    tmp_path, qe
):
    (tmp_path / "Ba.upf").write_text(UPF_V1)
    xml = write_qe_xml(
        tmp_path / "p.xml",
        kfrac=full_grid(2),
        nbnd=20,
        pseudo_dir=str(tmp_path),
        species=(("Ba", "Ba.upf", 0.0),),
        atoms=(("Ba", (0, 0, 0)),),
        n_atomic_wfc=5,
    )
    inputs = write_deck(
        tmp_path, num_wann=5, num_bands=10, exclude="11-20", froz_max=1.0
    )
    parent = qe.adequacy.read_facts(xml)
    deck = qe.adequacy.read_demand(inputs, parent)
    finding = Finding(
        INVENTORY,
        INADEQUATE,
        LIMITING,
        {"projector_functions": 5, "states_at_fermi_level": 6},
        "synthetic",
    )
    p = qe.adequacy.prescribe(finding, parent, deck)
    assert "pseudopotential" in p.message.lower()
    assert "5S" in p.message and "6S" in p.message
    assert "6" in p.message and "5" in p.message


# -- real Quantum ESPRESSO parents --------------------------------------------


def _fixture(tmp_path, material):
    src = FIXTURES / material
    xml = tmp_path / f"{material}.xml"
    xml.write_bytes(gzip.decompress((src / f"{material}.xml.gz").read_bytes()))
    for name in (f"{material}_pdwf.win", f"{material}_pdwf.pw2wan"):
        shutil.copy(src / name, tmp_path / name)
    return xml, read_construction(tmp_path, f"{material}_pdwf")


def test_ba_is_short_of_pseudo_atomic_functions_at_the_fermi_level(tmp_path, qe):
    """The real Ba parent (QE 7.5, 216 k-points, 20 bands) and its automatic deck:
    five pseudo-atomic functions, six states at or below E_F at some k-point."""
    xml, inputs = _fixture(tmp_path, "Ba")
    report = qe.assess_parent(xml, inputs, declared_moments={})
    inv = report.finding(INVENTORY)
    assert inv.status == INADEQUATE and inv.severity == LIMITING
    assert inv.measured["projector_functions"] == 5
    assert inv.measured["states_at_fermi_level"] == 6
    assert report.blocking == ()
    assert report.finding(BAND_COUNT).status == ADEQUATE
    assert report.finding(KLIST).status == ADEQUATE
    assert report.finding(SPIN).status == ADEQUATE


def test_na_is_not_blamed_on_its_parent(tmp_path, qe):
    """Na could not start either, and was certified after a deck-side repair. Its
    parent supplies the one function the Fermi level needs; the deck's frozen window
    asked for two."""
    xml, inputs = _fixture(tmp_path, "Na")
    report = qe.assess_parent(xml, inputs, declared_moments={})
    assert report.alarms == ()
    inv = report.finding(INVENTORY)
    assert inv.measured["states_at_fermi_level"] == 1
    assert inv.measured["states_in_frozen_window"] == 2


def test_assessing_a_parent_writes_nothing(tmp_path, qe):
    xml, inputs = _fixture(tmp_path, "Ba")

    def snapshot():
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(tmp_path.iterdir())
        }

    before = snapshot()
    qe.assess_parent(xml, inputs, declared_moments={"Ba": 0.0})
    assert snapshot() == before


FROZEN_CERTIFIED_WITH_XML = (
    "Sr",
    "La",
    "TiC",
    "Mn",
    "V3Si",
    "TiO2",
    "Cu2O",
    "MnO",
    "SiC",
)


@pytest.mark.skipif(
    not FROZEN.is_dir(), reason="the frozen campaign tree lives on cluster2"
)
@pytest.mark.parametrize("material", FROZEN_CERTIFIED_WITH_XML)
def test_campaign_parents_that_certified_raise_no_alarm(material, qe):
    d = FROZEN / "campaign2" / material
    declared = {"Mn": 5.0, "O": 0.0} if material == "MnO" else {}
    report = qe.assess_parent(
        d / f"{material}.xml",
        read_construction(d, f"{material}_pdwf"),
        declared_moments=declared,
    )
    assert report.blocking == ()
    assert report.alarms == ()


@pytest.mark.skipif(
    not FROZEN.is_dir(), reason="the frozen campaign tree lives on cluster2"
)
def test_the_ti_parent_raises_no_alarm(qe):
    d = FROZEN / "campaign" / "Ti"
    report = qe.assess_parent(
        d / "Ti.xml", read_construction(d, "Ti_pdwf"), declared_moments={}
    )
    assert report.alarms == ()


# ---------------------------------------------------------------------------
# the terminal diagnosis
# ---------------------------------------------------------------------------

PSEUDO = {"projection_source": "the pseudopotential's own atomic orbitals"}
FROZEN_REFUSAL = "dis_windows: More states in the frozen window than target WFs"


def test_attempt_evidence_is_read_back_from_a_run_log_record():
    record = {
        "index": 3,
        "stage": "generic_repair",
        "parameters": {"num_wann": "5", **PSEUDO},
        "calculation_error": FROZEN_REFUSAL,
        "model_error": None,
        "outcome": None,
        "action": None,
        "requests": [],
    }
    ev = AttemptEvidence.from_record(record)
    assert ev.index == 3
    assert ev.calculation_error == FROZEN_REFUSAL
    assert ev.failed_check_ids == ()
    record["outcome"] = {
        "failed_check_ids": ["centres_sit_on_atomic_sites"],
        "passed": False,
    }
    assert AttemptEvidence.from_record(record).failed_check_ids == (
        "centres_sit_on_atomic_sites",
    )


def test_a_blocking_preflight_finding_establishes_the_parent_as_the_cause():
    pre = assess(facts(n_bands=12), demand())
    dx = diagnose_terminal(pre, [])
    assert dx.cause_established is True
    assert dx.blocked_by_parent is False
    assert dx.returns_prescription is True
    assert [e.check_id for e in dx.events] == [BAND_COUNT]
    assert dx.prescriptions and dx.prescriptions[0].message.startswith("FAKE-ENGINE")


def test_a_repair_that_exceeded_the_band_ceiling_is_reported_as_blocked_not_as_the_cause():
    pre = assess(facts(n_bands=16), demand(num_bands=16), declared_moments={})
    attempts = [
        AttemptEvidence(0, PSEUDO, None, ("occupied_manifold_is_represented",)),
        AttemptEvidence(
            1,
            PSEUDO,
            None,
            ("occupied_manifold_is_represented",),
            parent_limit_requests=(
                {"parameter": "num_bands", "requested": 24, "parent_bands": 16},
            ),
        ),
    ]
    dx = diagnose_terminal(
        pre,
        attempts,
        facts=facts(n_bands=16),
        demand=demand(num_bands=16),
        prescribe=fake_prescribe,
    )
    assert dx.blocked_by_parent is True
    assert dx.cause_established is False
    assert dx.returns_prescription is True
    event = next(e for e in dx.events if e.check_id == BAND_COUNT)
    assert event.attempt == 1
    text = dx.message().lower()
    assert "24" in text and "16" in text
    assert "unresolved" in text


def test_pseudo_route_refusals_are_attributed_to_the_inventory_as_an_advisory():
    """Ba's record: the automatic deck and a num_bands raise both refused for the
    frozen window on the pseudo-atomic route; later analytic-projector attempts failed
    on convergence and centres, which the inventory does not explain."""
    parent = facts(
        n_bands=20,
        eigen=_ba_like_eigen(),
        projector_functions=5,
        species=("Ba",),
        species_order=("Ba",),
    )
    deck = demand(num_wann=5, num_bands=10, excluded=range(11, 21), frozen=(None, 2.0))
    pre = assess(parent, deck, declared_moments={})
    attempts = [
        AttemptEvidence(0, PSEUDO, FROZEN_REFUSAL, ()),
        AttemptEvidence(1, {**PSEUDO, "num_bands": "20"}, FROZEN_REFUSAL, ()),
        AttemptEvidence(
            2, {"projectors": "Ba:s;p;d"}, None, ("spread_minimisation_converged",)
        ),
        AttemptEvidence(3, {"projectors": "Ba:s;d"}, FROZEN_REFUSAL, ()),
    ]
    dx = diagnose_terminal(
        pre, attempts, facts=parent, demand=deck, prescribe=fake_prescribe
    )
    assert dx.cause_established is False
    assert dx.blocked_by_parent is False
    assert dx.returns_prescription is False
    inv = [e for e in dx.events if e.check_id == INVENTORY]
    assert sorted(e.attempt for e in inv) == [0, 1]
    assert all(e.severity == LIMITING for e in inv)
    assert dx.advisories and dx.advisories[0].check_id == INVENTORY
    assert set(dx.unattributed_attempts) == {2, 3}


def test_an_adequate_parent_with_a_deck_side_failure_is_not_blamed():
    """The specificity arms: nothing in the parent, failures from the deck alone."""
    pre = assess(facts(), demand(), declared_moments={})
    attempts = [
        AttemptEvidence(0, PSEUDO, FROZEN_REFUSAL, ()),
        AttemptEvidence(1, PSEUDO, None, ("spread_minimisation_converged",)),
    ]
    dx = diagnose_terminal(
        pre, attempts, facts=facts(), demand=demand(), prescribe=fake_prescribe
    )
    assert not dx.cause_established and not dx.blocked_by_parent
    assert dx.events == ()
    assert set(dx.unattributed_attempts) == {0, 1}


def test_a_coarse_mesh_found_on_the_final_construction_is_an_advisory():
    parent = facts(kpoints=full_grid(4))
    deck = demand(mp_grid=(4, 4, 4))
    pre = assess(parent, deck, declared_moments={})
    dx = diagnose_terminal(
        pre,
        [AttemptEvidence(0, PSEUDO, None, ("hopping_decays_with_distance",))],
        facts=parent,
        demand=deck,
        prescribe=fake_prescribe,
        hamiltonian=_hamiltonian(3.0),
    )
    assert not dx.cause_established and not dx.blocked_by_parent
    assert any(e.check_id == HOPPING and e.severity == ADVISORY for e in dx.events)
    assert any(a.check_id == HOPPING for a in dx.advisories)


def test_the_diagnosis_serialises():
    pre = assess(facts(n_bands=12), demand())
    blob = json.loads(json.dumps(diagnose_terminal(pre, []).to_dict()))
    assert blob["cause_established"] is True
    assert blob["events"][0]["check_id"] == BAND_COUNT


# ---------------------------------------------------------------------------
# the workflow: pre-flight before the first build, diagnosis at the end
# ---------------------------------------------------------------------------


class _LayDownRunner(ConstructionRunner):
    """Runs nothing; lays a finished construction down under the seed it was given."""

    def __init__(self, run_dir, template):
        super().__init__(run_dir=run_dir)
        self.template = template
        self.builds = []

    def run_overlaps(self, seed):  # pragma: no cover - build is overridden
        raise AssertionError("never computes overlaps")

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        shutil.copy(self.template, self.run_dir / f"{seed}.wout")
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


WOUT = EXAMPLES / "bcc_iron" / "fe_up_manual.wout"

#: Wannier90's own box for an applied projectability window (``dis_windows_proj`` in
#: w90dev ``src/disentangle.F90``), with the thresholds ``write_deck`` asks for.
PROJECTABILITY_BOX = (
    " |----------------------------------------------------------------------------|\n"
    " |                          Projectability  Windows                           |\n"
    " |                          -----------------------                           |\n"
    " |               Discarded:    0.00000  to    0.01000                         |\n"
    " |            Disentangled:    0.01000  to    1.00000                         |\n"
    " |                  Frozen:    0.95000  to    1.00000                         |\n"
)


def _wout_with_the_applied_window(directory):
    """The iron output with the box Wannier90 prints when it applied the window.

    ``write_deck`` asks for a projectability window; laid down without the box, the
    construction is invalid (``w90_outputs.projectability_not_applied``) and the loop
    stops before the gate or the model sees it.
    """
    lines = WOUT.read_text().splitlines(keepends=True)
    inner = next(i for i, line in enumerate(lines) if "Inner:" in line)
    path = directory / "fe_up_projectability_applied.wout"
    path.write_text(
        "".join(lines[: inner + 1]) + PROJECTABILITY_BOX + "".join(lines[inner + 1 :])
    )
    return path


def _qe_run_dir(tmp_path, *, parent_bands=14, kfrac=None, nsym=1):
    """An iron-sized QE run: 9 functions from 14 bands on a 2x2x2 grid."""
    k = full_grid(2)
    xml = write_qe_xml(
        tmp_path / "parent.xml",
        kfrac=k if kfrac is None else kfrac,
        nbnd=parent_bands,
        nsym=nsym,
        species=(("Fe", "Fe.upf", 0.0),),
        atoms=(("Fe", (0, 0, 0)),),
        n_atomic_wfc=9,
    )
    write_deck(tmp_path, seed="fe", num_wann=9, num_bands=14, froz_max=2.0)
    return xml


def test_the_workflow_takes_a_declared_magnetic_state(qe):
    from autowan.workflow import WannierisationResult, wannierize

    assert "declared_moments" in inspect.signature(wannierize).parameters
    names = {f.name for f in fields(WannierisationResult)}
    assert {"parent_adequacy", "diagnosis", "prescription"} <= names


def test_a_blocking_parent_returns_a_prescription_and_builds_nothing(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    xml = _qe_run_dir(tmp_path, kfrac=full_grid(2)[[0, 1, 3, 7]], nsym=48)
    runner = _LayDownRunner(tmp_path, WOUT)
    log = tmp_path / "run.json"
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=xml,
        seed="fe",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        log_path=log,
        declared_moments={},
    )
    assert runner.builds == []
    assert result.run.attempts == []
    assert not result.certified
    assert result.status == "DFT prescription returned"
    assert "nosym" in result.prescription
    assert result.parent_adequacy.finding(KLIST).status == INADEQUATE
    assert result.diagnosis.cause_established is True
    assert "parent" in (result.run.terminal_reason or "").lower()
    blob = json.loads(log.read_text())
    assert blob["prescription"] == result.prescription
    assert any(
        f["check_id"] == KLIST and f["status"] == INADEQUATE
        for f in blob["parent_adequacy"]["findings"]
    )
    assert blob["terminal_diagnosis"]["cause_established"] is True


def test_an_adequate_parent_runs_the_loop_exactly_as_before(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    xml = _qe_run_dir(tmp_path)
    runner = _LayDownRunner(tmp_path, WOUT)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=xml,
        seed="fe",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        declared_moments={},
    )
    assert runner.builds == [("fe", None)]
    assert result.parent_adequacy.blocking == ()
    assert result.status != "DFT prescription returned"
    assert result.prescription is None
    if not result.certified:
        assert result.diagnosis is not None
        assert not result.diagnosis.cause_established
        assert not result.diagnosis.blocked_by_parent


def test_the_workflow_never_writes_the_parent(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    xml = _qe_run_dir(tmp_path, kfrac=full_grid(2)[[0, 1, 3, 7]], nsym=48)
    before = hashlib.sha256(xml.read_bytes()).hexdigest()
    wannierize(
        run_dir=tmp_path,
        parent_xml=xml,
        seed="fe",
        runner=_LayDownRunner(tmp_path, WOUT),
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        declared_moments={},
    )
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == before


class _AsksForMoreBands:
    """A model that keeps asking for more bands than the parent computed."""

    def __init__(self, requested):
        self.requested = requested
        self.calls = 0

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        self.calls += 1
        props = schema["properties"]
        evidence = props["evidence"]["items"]["enum"][:1]
        targets = props.get("targets", {}).get("items", {}).get("enum", [])[:1]
        content = {
            "action": "modify_parameters",
            "diagnosis": "the disentanglement needs more of the spectrum",
            "evidence": evidence,
            "expected_effect": "the occupied manifold is represented",
            "confidence": 0.5,
            "changes": [
                {
                    "parameter": "num_bands",
                    "new": self.requested,
                    "reason": "more states",
                }
            ],
        }
        if targets:
            content["targets"] = targets
        return ModelResponse(
            content=content, model="stub", raw_text=json.dumps(content)
        )


def test_a_repair_blocked_by_the_parents_band_ceiling_returns_a_prescription(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    xml = _qe_run_dir(tmp_path, parent_bands=14)
    model = _AsksForMoreBands(requested=24)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=xml,
        seed="fe",
        runner=_LayDownRunner(tmp_path, _wout_with_the_applied_window(tmp_path)),
        provider_factory=lambda _m: model,
        config=WorkflowConfig(draw_figures=False),
        declared_moments={},
    )
    assert model.calls >= 1
    assert not result.certified
    assert result.diagnosis.blocked_by_parent is True
    assert result.diagnosis.cause_established is False
    assert result.status == "DFT prescription returned"
    assert "nbnd" in result.prescription and "24" in result.prescription
    event = next(e for e in result.diagnosis.events if e.check_id == BAND_COUNT)
    assert event.attempt is not None


class _AsksForMoreBandsThenIterates(_AsksForMoreBands):
    """Asks past the parent's band ceiling once, then for an answer the parent allows."""

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        if self.calls == 0:
            return super().propose(
                system=system, payload_text=payload_text, schema=schema, images=images
            )
        self.calls += 1
        props = schema["properties"]
        content = {
            "action": "modify_parameters",
            "diagnosis": "the minimisation needs more iterations",
            "evidence": props["evidence"]["items"]["enum"][:1],
            "expected_effect": "the spread converges",
            "confidence": 0.5,
            "changes": [
                {"parameter": "num_iter", "new": 8000, "reason": "more iterations"}
            ],
        }
        targets = props.get("targets", {}).get("items", {}).get("enum", [])[:1]
        if targets:
            content["targets"] = targets
        return ModelResponse(
            content=content, model="stub", raw_text=json.dumps(content)
        )


class _WindowAppliedOnlyOnTheBaseline(_LayDownRunner):
    """The baseline shows the applied window; every repair's output does not."""

    def __init__(self, run_dir, baseline, repair):
        super().__init__(run_dir, baseline)
        self.repair = repair

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        source = self.template if not self.builds[1:] else self.repair
        shutil.copy(source, self.run_dir / f"{seed}.wout")
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


def test_an_invalid_construction_outranks_a_ceiling_blocked_prescription(tmp_path):
    """Both arms in one run: the label is ``invalid``, the prescription is kept.

    Attempt 0 is a valid, gate-refused construction; the model first asks for more
    bands than the parent carries (recorded as blocked by the parent), then for a
    repair the parent allows. That repair's output lacks the projectability window
    its deck asked for, so it is not a measurement. Invalid outranks the prescription
    in the status; the prescription, the diagnosis and the log keep it.
    """
    from autowan.workflow import WorkflowConfig, wannierize

    xml = _qe_run_dir(tmp_path, parent_bands=14)
    win = tmp_path / "fe.win"
    win.write_text("dis_froz_proj = .true.\n" + win.read_text())
    model = _AsksForMoreBandsThenIterates(requested=24)
    runner = _WindowAppliedOnlyOnTheBaseline(
        tmp_path, _wout_with_the_applied_window(tmp_path), WOUT
    )
    log = tmp_path / "run.json"
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=xml,
        seed="fe",
        runner=runner,
        provider_factory=lambda _m: model,
        config=WorkflowConfig(draw_figures=False),
        log_path=log,
        declared_moments={},
    )
    statuses = [a.status for a in result.run.attempts]
    assert statuses == ["gate_refused", "invalid"]
    assert not result.certified
    assert result.status == "invalid"
    assert result.diagnosis.blocked_by_parent is True
    assert result.prescription is not None
    assert "nbnd" in result.prescription and "24" in result.prescription
    event = next(e for e in result.diagnosis.events if e.check_id == BAND_COUNT)
    assert event.attempt == 0
    blob = json.loads(log.read_text())
    assert blob["prescription"] == result.prescription
    assert blob["terminal_diagnosis"]["blocked_by_parent"] is True
