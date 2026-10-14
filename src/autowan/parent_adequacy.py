"""Read-only judgements of a parent calculation against a construction's demand.

Engine partitions supply facts and prescription wording. Only contract violations
block certification; route limits and convergence suggestions remain advisories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import numpy as np

ADEQUATE = "adequate"
INADEQUATE = "inadequate"
NOT_EVALUABLE = "not_evaluable"
NOT_APPLICABLE = "not_applicable"
BLOCKING = "blocking"
LIMITING = "limiting"
ADVISORY = "advisory"

PARENT_ADEQUACY_CHECKS = (
    "band_count_covers_demand",
    "kpoints_are_the_decks_full_grid",
    "spin_treatment_matches_magnetic_state",
    "projector_inventory_covers_fermi_level",
    "mesh_resolves_hopping_decay",
    "band_partition_matches_deck",
)
BAND_COUNT, KLIST, SPIN, INVENTORY, HOPPING, BAND_PARTITION = PARENT_ADEQUACY_CHECKS
SEVERITY = dict(
    zip(
        PARENT_ADEQUACY_CHECKS,
        (BLOCKING, BLOCKING, BLOCKING, LIMITING, ADVISORY, BLOCKING),
    )
)


@dataclass(frozen=True)
class Channel:
    label: str
    l: int
    j: float | None = None
    """Total angular momentum of a fully relativistic (j-resolved) orbital, else None."""

    @property
    def functions(self) -> int:
        """States this orbital supplies: 2j+1 when j-resolved, else 2l+1 (one spin)."""
        if self.j is not None:
            return int(round(2 * self.j + 1))
        return 2 * self.l + 1

    def spinor_functions(self) -> int:
        """States in a spinor calculation: 2j+1 when j-resolved, else both spins."""
        return self.functions if self.j is not None else 2 * (2 * self.l + 1)


@dataclass(frozen=True, eq=False)
class ParentFacts:
    engine: str
    eigenvalues_ev: np.ndarray
    fermi_energy_ev: float
    kpoints_frac: np.ndarray
    lattice_ang: np.ndarray
    species: tuple[str, ...]
    n_symmetries: int
    spin_polarised: bool
    noncollinear: bool = False
    moment_bohr: float | None = None
    projector_functions: int | None = None
    channels: Mapping[str, tuple[Channel, ...]] = field(default_factory=dict)
    species_order: tuple[str, ...] = ()
    requested_moments: Mapping[str, float] = field(default_factory=dict)
    source: str = ""

    @property
    def n_bands(self) -> int:
        return self.eigenvalues_ev.shape[2]

    @property
    def n_spin(self) -> int:
        return self.eigenvalues_ev.shape[0]

    @property
    def n_kpoints(self) -> int:
        return self.eigenvalues_ev.shape[1]


@dataclass(frozen=True, eq=False)
class DeckDemand:
    num_wann: int
    num_bands: int | None = None
    excluded_bands: tuple[int, ...] = ()
    mp_grid: tuple[int, int, int] | None = None
    kpoints_frac: np.ndarray | None = None
    frozen_window_ev: tuple[float | None, float | None] = (None, None)
    projector_route: str = "unknown"
    spin_channel: int | None = None
    demanded_channels: Mapping[str, tuple[int, ...]] | None = None


@dataclass(frozen=True)
class DftChange:
    step: str
    section: str
    key: str
    value: Any


@dataclass(frozen=True)
class Prescription:
    check_id: str
    message: str
    changes: tuple[DftChange, ...] = ()


def _plain(value):
    """Keep evidence comparable and serialisable even for numerical callers."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_plain(v) for v in value)
    return value


@dataclass(frozen=True)
class Finding:
    check_id: str
    status: str
    severity: str
    measured: Mapping
    reason: str
    prescription: Prescription | None = None

    def __post_init__(self):
        object.__setattr__(self, "measured", _plain(self.measured))

    @property
    def alarm(self) -> bool:
        return self.status == INADEQUATE


@dataclass(frozen=True)
class ParentAdequacyReport:
    findings: tuple[Finding, ...]
    engine: str = ""
    facts: ParentFacts | None = field(default=None, compare=False, repr=False)
    demand: DeckDemand | None = field(default=None, compare=False, repr=False)

    def finding(self, check_id: str) -> Finding:
        return next(f for f in self.findings if f.check_id == check_id)

    @property
    def alarms(self):
        return tuple(f for f in self.findings if f.alarm)

    @property
    def blocking(self):
        return tuple(f for f in self.alarms if f.severity == BLOCKING)

    @property
    def not_evaluable(self):
        return tuple(f for f in self.findings if f.status == NOT_EVALUABLE)

    @property
    def adequate(self) -> bool:
        return not self.alarms

    def prescription_text(self) -> str:
        return "\n".join(f.prescription.message for f in self.alarms if f.prescription)

    def to_dict(self):
        return {
            "engine": self.engine,
            "findings": [_plain(asdict(f)) for f in self.findings],
        }


class AdequacyPartition(ABC):
    engine: str

    @abstractmethod
    def read_facts(self, parent_source) -> ParentFacts:
        raise NotImplementedError

    @abstractmethod
    def read_demand(self, inputs, facts) -> DeckDemand:
        raise NotImplementedError

    @abstractmethod
    def prescribe(self, finding, facts, demand) -> Prescription:
        raise NotImplementedError


def full_grid(mesh):
    """Gamma-centred fractional points, with the first index slowest."""
    a, b, c = mesh
    return np.array(
        [[i / a, j / b, k / c] for i in range(a) for j in range(b) for k in range(c)]
    )


def _finding(check, status, measured, reason):
    return Finding(check, status, SEVERITY[check], measured, reason)


def _spectrum(facts, demand):
    e = facts.eigenvalues_ev
    return e if demand.spin_channel is None else e[[demand.spin_channel]]


def _bands(facts, demand):
    required = (
        demand.num_bands + len(demand.excluded_bands)
        if demand.num_bands is not None
        else demand.num_wann
    )
    top = demand.frozen_window_ev[1]
    closed = (
        None
        if top is None
        else bool(np.all(np.any(_spectrum(facts, demand) > top, axis=2)))
    )
    short = facts.n_bands < required or facts.n_bands < demand.num_wann
    measured = dict(
        parent_bands=facts.n_bands,
        num_wann=demand.num_wann,
        num_bands=demand.num_bands,
        excluded=len(demand.excluded_bands),
        required_bands=required,
        frozen_window_closed=closed,
        prescribed_bands=(
            max(required, facts.n_bands + demand.num_wann)
            if closed is False
            else required
        ),
    )
    if closed is False:
        reason = "The parent spectrum does not close the frozen window at every sampled point."
    elif short:
        reason = "The parent carries fewer bands than the construction requires."
    elif facts.n_bands > required:
        reason = "The parent supplies enough bands; a deck written for a smaller parent is a deck-side matter."
    else:
        reason = "The parent supplies the bands required by the deck."
    return _finding(
        BAND_COUNT,
        INADEQUATE if short or closed is False else ADEQUATE,
        measured,
        reason,
    )


def _partition(facts, demand):
    excluded = len(demand.excluded_bands)
    deck_bands = None if demand.num_bands is None else demand.num_bands + excluded
    short = deck_bands is not None and facts.n_bands < deck_bands
    surplus = 0 if deck_bands is None else max(facts.n_bands - deck_bands, 0)
    top = demand.frozen_window_ev[1]
    closed = (
        None
        if demand.num_bands is None or top is None or short
        else bool(
            np.all(np.any(_spectrum(facts, demand)[:, :, :deck_bands] > top, axis=2))
        )
    )
    measured = dict(
        parent_bands=facts.n_bands,
        num_bands=demand.num_bands,
        excluded=excluded,
        deck_bands=deck_bands,
        overlap_bands=facts.n_bands - excluded,
        surplus_bands=surplus,
        surplus_band_range=[deck_bands + 1, facts.n_bands] if surplus else None,
        frozen_window_closed_by_deck_bands=closed,
        prescribed_bands=deck_bands if surplus and closed is not False else None,
    )
    if demand.num_bands is None:
        return _finding(
            BAND_PARTITION,
            NOT_APPLICABLE,
            measured,
            "The deck does not fix num_bands; the deck-overlap check compares a declared count only.",
        )
    if short:
        return _finding(
            BAND_PARTITION,
            NOT_APPLICABLE,
            measured,
            "A parent short of the deck is the band-count check's finding.",
        )
    if not surplus:
        return _finding(
            BAND_PARTITION,
            ADEQUATE,
            measured,
            "The parent bands partition exactly into the deck's declared bands and exclusions.",
        )
    return _finding(
        BAND_PARTITION,
        INADEQUATE,
        measured,
        f"The parent carries {facts.n_bands} bands but the deck partitions {deck_bands} "
        f"(num_bands = {demand.num_bands} plus {excluded} excluded): the overlaps "
        f"would span {facts.n_bands - excluded} bands against the deck's {demand.num_bands}, "
        "and the deck-overlap check refuses the construction.",
    )


def _kpoints(facts, demand):
    expected = demand.kpoints_frac
    if expected is None and demand.mp_grid is not None:
        expected = full_grid(demand.mp_grid)
    measured = dict(
        parent_kpoints=facts.n_kpoints,
        deck_kpoints=None,
        mp_grid=demand.mp_grid,
        missing=None,
        extra=None,
        order_matches=None,
        n_symmetries=facts.n_symmetries,
        symmetry_reduced=False,
    )
    if expected is None:
        return _finding(
            KLIST,
            NOT_EVALUABLE,
            measured,
            "The deck declares no sampling grid or point list.",
        )
    parent = np.asarray(facts.kpoints_frac)
    expected = np.asarray(expected)
    # Compare one expected point at a time to avoid a quadratic-size 3-D array.
    present = np.zeros(len(parent), dtype=bool)
    missing = 0
    for point in expected:
        delta = parent - point
        matches = np.all(np.abs(delta - np.rint(delta)) <= 1e-5, axis=1)
        missing += int(not matches.any())
        present |= matches
    extra = int((~present).sum())
    order = None
    if demand.kpoints_frac is not None:
        order = False
        if len(parent) == len(expected):
            delta = parent - expected
            order = bool(np.all(np.abs(delta - np.rint(delta)) <= 1e-5))
    measured.update(
        deck_kpoints=len(expected),
        missing=missing,
        extra=extra,
        order_matches=order,
        symmetry_reduced=len(parent) < len(expected) and facts.n_symmetries > 1,
    )
    bad = missing or extra or order is False
    return _finding(
        KLIST,
        INADEQUATE if bad else ADEQUATE,
        measured,
        (
            "The parent point list differs from the deck's full grid or ordering."
            if bad
            else "The parent contains the deck's full sampling grid."
        ),
    )


def _spin(facts, declared_moments):
    declared = declared_moments
    source = "caller" if declared is not None else "none"
    if declared is None and any(facts.requested_moments.values()):
        declared = facts.requested_moments
        source = "parent input"
    # A parent-input request establishes magnetic intent, not a moment in Bohr magnetons.
    total = (
        abs(sum(declared.get(s, 0.0) for s in facts.species))
        if source == "caller"
        else None
    )
    measured = dict(
        spin_polarised=facts.spin_polarised,
        noncollinear=facts.noncollinear,
        moment_bohr=facts.moment_bohr,
        declared_total_bohr=total,
        declared=None if declared is None else dict(declared),
        source=source,
    )
    if declared is None:
        return _finding(
            SPIN, NOT_EVALUABLE, measured, "No magnetic state has been declared."
        )
    magnetic = any(declared.get(s, 0.0) != 0 for s in facts.species)
    if magnetic and not (facts.spin_polarised or facts.noncollinear):
        return _finding(
            SPIN,
            INADEQUATE,
            measured,
            "The declared magnet was computed without spin polarisation.",
        )
    if (
        magnetic
        and facts.spin_polarised
        and total is not None
        and total > 0
        and facts.moment_bohr is not None
        and abs(facts.moment_bohr) < 0.25 * total
    ):
        return _finding(
            SPIN,
            INADEQUATE,
            measured,
            "The parent converged to a non-magnetic solution despite the declared moments.",
        )
    return _finding(
        SPIN,
        ADEQUATE,
        measured,
        "The spin treatment is consistent with the declared magnetic state.",
    )


def _inventory(facts, demand):
    measured = dict(
        projector_functions=facts.projector_functions,
        num_wann=demand.num_wann,
        channels={s: [c.label for c in cs] for s, cs in facts.channels.items()},
    )
    if demand.demanded_channels is not None:
        demanded = {s: sorted(set(ls)) for s, ls in demand.demanded_channels.items()}
        missing = {}
        for species, angulars in demanded.items():
            if species in facts.channels:
                absent = sorted(set(angulars) - {c.l for c in facts.channels[species]})
                if absent:
                    missing[species] = absent
        measured.update(demanded_channels=demanded, missing_channels=missing)
        if any(s not in facts.channels for s in demanded):
            return _finding(
                INVENTORY,
                NOT_EVALUABLE,
                measured,
                "The projector channels for a demanded species are unknown.",
            )
        return _finding(
            INVENTORY,
            INADEQUATE if missing else ADEQUATE,
            measured,
            (
                "The pseudopotentials lack channels demanded by the deck's projections; "
                "this limits the projections and is not a parent cause."
                if missing
                else "The pseudopotentials supply the channels demanded by the deck."
            ),
        )
    if demand.projector_route != "pseudo_atomic":
        return _finding(
            INVENTORY,
            NOT_APPLICABLE,
            measured,
            "This projection route does not use the pseudo-atomic inventory.",
        )
    if facts.projector_functions is None:
        return _finding(
            INVENTORY,
            NOT_EVALUABLE,
            measured,
            "The pseudo-atomic projection route's inventory is unknown.",
        )
    excluded = set(demand.excluded_bands)
    e = _spectrum(facts, demand)[
        :, :, [b for b in range(facts.n_bands) if b + 1 not in excluded]
    ]
    lo, hi = demand.frozen_window_ev
    inside = np.ones(e.shape, dtype=bool) if lo is None else e >= lo
    occupied = int(
        np.count_nonzero(inside & (e <= facts.fermi_energy_ev), axis=2).max()
    )
    frozen = (
        None if hi is None else int(np.count_nonzero(inside & (e <= hi), axis=2).max())
    )
    measured.update(states_at_fermi_level=occupied, states_in_frozen_window=frozen)
    bad = facts.projector_functions < max(occupied, demand.num_wann)
    reason = (
        "The pseudo-atomic projection route lacks functions for the required states; "
        "this is a route limit, not a parent cause."
        if bad
        else "The pseudo-atomic projection route supplies enough functions at the Fermi level."
    )
    if not bad and frozen is not None and frozen > demand.num_wann:
        reason += (
            " The frozen window asks for too many states: this is a deck-side choice."
        )
    return _finding(INVENTORY, INADEQUATE if bad else ADEQUATE, measured, reason)


def _hopping(facts, demand, hamiltonian):
    from .core_checks import CORE_CHECKS

    fraction = next(
        c.threshold.fraction
        for c in CORE_CHECKS
        if c.check_id == "hopping_decays_with_distance"
    )
    measured = dict(
        onsite_ev=None,
        threshold_fraction=fraction,
        threshold_ev=None,
        outer_hopping_ev=None,
        decay_length_ang=None,
        reach_needed_ang=None,
        outer_distance_ang=None,
        mesh=demand.mp_grid,
        prescribed_mesh=None,
    )

    def unknown(reason):
        return _finding(HOPPING, NOT_EVALUABLE, measured, reason)

    if hamiltonian is None or demand.mp_grid is None:
        return unknown(
            "A Hamiltonian and a declared mesh are needed to assess hopping decay."
        )
    r = getattr(hamiltonian, "hr_rvectors", None)
    h = getattr(hamiltonian, "hr_matrix", None)
    if r is None or h is None:
        return unknown("The construction records no real-space Hamiltonian.")
    r, h = np.asarray(r), np.asarray(h)
    if not r.size or not h.size:
        return unknown("The real-space Hamiltonian is empty.")
    d = np.linalg.norm(r @ facts.lattice_ang, axis=1)
    onsite_mask = np.all(r == 0, axis=1)
    if not onsite_mask.any() or d.max() <= 0:
        return unknown("Both on-site and distant hopping blocks are needed.")
    envelope = np.abs(h).max(axis=(1, 2))
    onsite = float(envelope[onsite_mask].max())
    outer = d >= np.quantile(d, 0.9)
    edge = float(d[outer].min())
    outer_h = float(envelope[outer].max())
    threshold = fraction * onsite
    measured.update(
        onsite_ev=onsite,
        threshold_ev=threshold,
        outer_distance_ang=edge,
        outer_hopping_ev=outer_h,
    )
    if outer_h <= threshold:
        return _finding(
            HOPPING,
            ADEQUATE,
            measured,
            "Outer hoppings have decayed below the gate's on-site-relative threshold.",
        )
    shells = np.unique(d[(d > 0) & (d < edge)])
    values = np.array([envelope[d == distance].max() for distance in shells])
    positive = values > 0
    shells, values = shells[positive], values[positive]
    if len(shells) < 3 or threshold <= 0 or not np.all(np.isfinite(values)):
        return unknown(
            "Too few positive inner shells or no on-site scale to establish decay."
        )
    slope, intercept = np.polyfit(shells, np.log(values), 1)
    if slope >= 0 or np.allclose(values, values[0], rtol=1e-12, atol=0):
        return unknown(
            "The construction does not decay in its inner shells; a mesh cause is not established."
        )
    decay_length = float(-1 / slope)
    measured["decay_length_ang"] = decay_length
    if decay_length >= edge:
        return unknown(
            "The construction is not localised within the supercell represented by the mesh: "
            "its decay length reaches or exceeds the outer distance. This does not establish a mesh cause."
        )
    reach = float((np.log(threshold) - intercept) / slope)
    mesh = tuple(max(n + 1, int(np.ceil(n * reach / edge))) for n in demand.mp_grid)
    measured.update(reach_needed_ang=reach, prescribed_mesh=mesh)
    return _finding(
        HOPPING,
        INADEQUATE,
        measured,
        "The fitted decay warrants a mesh-convergence check; it does not establish a parent cause.",
    )


def assess_parent(
    facts, demand, *, prescribe=None, declared_moments=None, hamiltonian=None
):
    findings = (
        _bands(facts, demand),
        _kpoints(facts, demand),
        _spin(facts, declared_moments),
        _inventory(facts, demand),
        _hopping(facts, demand, hamiltonian),
        _partition(facts, demand),
    )
    if prescribe is not None:
        findings = tuple(
            replace(f, prescription=prescribe(f, facts, demand)) if f.alarm else f
            for f in findings
        )
    return ParentAdequacyReport(findings, facts.engine, facts, demand)


@dataclass(frozen=True)
class AttemptEvidence:
    index: int
    parameters: Mapping
    calculation_error: str | None = None
    failed_check_ids: tuple[str, ...] = ()
    parent_limit_requests: tuple[Mapping, ...] = ()

    @classmethod
    def from_record(cls, record):
        return cls(
            record["index"],
            record.get("parameters", {}),
            record.get("calculation_error"),
            tuple((record.get("outcome") or {}).get("failed_check_ids", ())),
            tuple(record.get("parent_limit_requests", ())),
        )


@dataclass(frozen=True)
class DiagnosisEvent:
    check_id: str
    attempt: int | None
    evidence: Mapping
    severity: str


@dataclass(frozen=True)
class TerminalDiagnosis:
    cause_established: bool
    blocked_by_parent: bool
    events: tuple[DiagnosisEvent, ...]
    prescriptions: tuple[Prescription, ...]
    advisories: tuple[Prescription, ...]
    unattributed_attempts: tuple[int, ...]

    @property
    def returns_prescription(self):
        return self.cause_established or self.blocked_by_parent

    def message(self):
        lines = []
        if self.cause_established:
            lines.append(
                "A parent–deck contract violation blocks certification before construction."
            )
        if self.blocked_by_parent:
            for event in self.events:
                if event.check_id == BAND_COUNT and event.attempt is not None:
                    lines.append(
                        f"The repair proposed in attempt {event.attempt} needed "
                        f"{event.evidence['requested']} bands; the parent carries "
                        f"{event.evidence['parent_bands']}. The cause of the failure remains unresolved."
                    )
        if not self.returns_prescription:
            lines.append(
                "The cause of the failure remains unresolved; no blocking parent cause was established."
            )
        lines.extend(p.message for p in self.prescriptions)
        lines.extend("Advisory: " + p.message for p in self.advisories)
        return "\n".join(lines)

    def to_dict(self):
        return _plain(asdict(self))


def diagnose_terminal(
    preflight, attempts, *, facts=None, demand=None, prescribe=None, hamiltonian=None
):
    facts = facts if facts is not None else preflight.facts
    demand = demand if demand is not None else preflight.demand
    events, prescriptions, advisories = [], [], []
    for f in preflight.blocking:
        events.append(DiagnosisEvent(f.check_id, None, f.measured, BLOCKING))
        if f.prescription is not None:
            prescriptions.append(f.prescription)
    requests = [(a.index, r) for a in attempts for r in a.parent_limit_requests]
    for index, request in requests:
        events.append(DiagnosisEvent(BAND_COUNT, index, _plain(request), BLOCKING))
    if requests and prescribe is not None and facts is not None and demand is not None:
        requested = max(int(r["requested"]) for _, r in requests)
        measured = dict(
            parent_bands=facts.n_bands,
            required_bands=requested,
            prescribed_bands=requested,
            num_wann=demand.num_wann,
            num_bands=demand.num_bands,
            excluded=len(demand.excluded_bands),
        )
        prescriptions.append(
            prescribe(
                _finding(
                    BAND_COUNT,
                    INADEQUATE,
                    measured,
                    "A proposed repair exceeds the parent's band ceiling.",
                ),
                facts,
                demand,
            )
        )
    inventory = next((f for f in preflight.alarms if f.check_id == INVENTORY), None)
    if inventory is not None:
        if inventory.measured.get("missing_channels"):
            events.append(DiagnosisEvent(INVENTORY, None, inventory.measured, LIMITING))
            if (
                inventory.prescription is not None
                and inventory.prescription not in advisories
            ):
                advisories.append(inventory.prescription)
        for a in attempts:
            pseudo = (
                not a.parameters.get("projectors")
                and "pseudopotential"
                in str(a.parameters.get("projection_source", "")).lower()
            )
            if pseudo and "frozen window" in (a.calculation_error or "").lower():
                events.append(
                    DiagnosisEvent(
                        INVENTORY,
                        a.index,
                        {
                            "calculation_error": a.calculation_error,
                            **inventory.measured,
                        },
                        LIMITING,
                    )
                )
                if (
                    inventory.prescription is not None
                    and inventory.prescription not in advisories
                ):
                    advisories.append(inventory.prescription)
    if hamiltonian is not None and facts is not None and demand is not None:
        hopping = _hopping(facts, demand, hamiltonian)
        if hopping.alarm:
            events.append(DiagnosisEvent(HOPPING, None, hopping.measured, ADVISORY))
            if prescribe is not None:
                advisories.append(prescribe(hopping, facts, demand))
    attributed = {e.attempt for e in events if e.attempt is not None}
    unattributed = tuple(
        a.index
        for a in attempts
        if (a.calculation_error or a.failed_check_ids) and a.index not in attributed
    )
    return TerminalDiagnosis(
        bool(preflight.blocking),
        bool(requests),
        tuple(events),
        tuple(prescriptions),
        tuple(advisories),
        unattributed,
    )
