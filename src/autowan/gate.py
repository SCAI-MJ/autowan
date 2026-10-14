"""The deterministic gate: what a construction must satisfy, and against what.

Three rules give the gate its authority, and each exists because its absence has
already cost this work a wrong answer.

**Every check resolves against the parent calculation, the crystal structure, or the
construction itself.** Never against another localized representation. A gate that
scored one construction against another would make one arm the reference for the
other, which is exactly the quantity a comparison between them is trying to measure.

**A scalar check may not carry a bare constant.** It carries a dimensionless fraction
of a scale the material itself supplies, together with a record of where that fraction
was calibrated. A threshold that travels without its calibrating material is how a
bcc-iron number ends up deciding whether a halide monolayer passes.

**A check that cannot be evaluated says so.** It does not fall back to a default and
it does not fail. Non-evaluable and failing are different states, and conflating them
turns missing data into evidence of a defect — an inversion that once reported eight
false alarms where there was one.

**What "passed" means is a named, hashed profile.** The checks a gate runs are chosen by
a gate profile (:mod:`.gate_profiles`): ``default`` plus whatever criterion a use needs.
Every check is removable, the default ones included, and there are no tiers: removing a
default check emits a :class:`GateProfileWarning` and nothing else, and a contributed check
that shares an identifier replaces the profile's check (with the same warning) rather than
being refused. Whether a changed profile still answers the experiment's question is the
experiment's concern, not the code's. What the code guarantees is that the change cannot
be silent: every verdict and every report records the profile's name and a hash of the
effective check list, so two verdicts are comparable exactly when their hashes are equal.
Acceptance stays the conjunction of every check in the profile.

A construction declares its certification target. The default holds a model to every
occupied state of its parent, autowan's stated design position. The magnetic_subspace
target is proposed beside it, not as a correction: a declaration names the magnetic
manifold the model is to represent, and an undeclared construction keeps the default.

A whole_occupied_manifold target may also declare a ``window_ev``, relative to the
parent's Fermi level, which is the window the band-reproduction check judges. A target
that declares no window keeps the deck's own window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

VerdictType = Literal["discrete", "scalar_vs_scale"]
Reference = Literal["dft_bands", "structure", "self"]
ALLOWED_REFERENCES: tuple[Reference, ...] = ("dft_bands", "structure", "self")

MaterialClass = Literal["metal", "gapped"]


class NonEvaluable(Exception):
    """This construction does not carry what the check needs.

    Not a failure. The verdict row records ``evaluable=False`` and is excluded from
    both the acceptance decision and any specificity or detection statistic.
    """


class GateDefinitionError(ValueError):
    """A check is not admissible as written."""


class NotApplicable(Exception):
    """This check asks a question the construction's material does not pose.

    Raised by a check that decides its own applicability from measured quantities (an
    insulator rule, for instance) rather than from the declared material class. The
    verdict is recorded as withheld (``applies=False``), never as passed or failed, and
    ``record`` keeps the measurements that decided it.
    """

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = dict(record or {})


class GateProfileWarning(UserWarning):
    """A profile or a contributed check changed which checks define "passed"."""


def check_list_hash(checks: Iterable["Check"]) -> str:
    """A short, order-independent hash of what a list of checks demands.

    It covers each check's identifier, verdict type, reference, threshold (fraction and
    scale), the material classes and routes it applies to, and its origin -- everything
    that decides a verdict except the code of the evaluation itself. Two gates with the
    same hash ask the same questions with the same thresholds.
    """
    import hashlib
    import json

    rows = sorted(
        (
            c.check_id,
            c.verdict_type,
            c.reference,
            (
                None
                if c.threshold is None
                else [repr(float(c.threshold.fraction)), c.threshold.scale]
            ),
            list(c.applies_to),
            c.origin,
            None if c.applies_to_routes is None else list(c.applies_to_routes),
        )
        for c in checks
    )
    blob = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


CERTIFICATION_TARGETS = ("whole_occupied_manifold", "magnetic_subspace")


@dataclass(frozen=True, eq=False)
class CertificationTarget:
    """The manifold a construction declares that it represents.

    Both targets read ``window_ev`` as ``(lo, hi)`` in eV *relative to the parent's
    Fermi level*, so the two declarations share one frame.

    ``whole_occupied_manifold`` may declare ``window_ev`` as the reproduction window ν
    the research use needs. The deck's ``dis_froz_min``/``dis_froz_max`` stay absolute,
    as Wannier90 reads them; the band-reproduction check converts ν with the parent's
    own Fermi level before comparing the two. ``window_ev=None`` declares no window and
    leaves it to the deck. A lower bound of ``-inf`` starts at the bottom of the
    parent's bands, the lower edge the check has always used; the upper bound must be
    finite. The window comes from the target and never from the deck because a deck
    graded on its own frozen window is graded on a choice it made: lowering
    ``dis_froz_max`` would then clear any material trivially.

    ``magnetic_subspace`` supplies both an energy window and the magnetic-d weight of
    every parent state on its construction mesh. The backend that can read those
    weights records their source; the core gate needs only the aligned array, never
    knowledge of that backend's orbitals.
    """

    name: str
    window_ev: tuple[float, float] | None = None
    magnetic_d_weight: np.ndarray | None = None
    weight_source: str = ""

    def __post_init__(self) -> None:
        if self.name not in CERTIFICATION_TARGETS:
            raise GateDefinitionError(f"unknown certification target {self.name!r}")
        if self.name == "whole_occupied_manifold":
            if self.magnetic_d_weight is not None:
                raise GateDefinitionError("whole_occupied_manifold takes no weights")
            if self.window_ev is None:
                return
            try:
                if len(self.window_ev) != 2:
                    raise ValueError("a target window needs exactly two bounds")
                lo, hi = (float(bound) for bound in self.window_ev)
            except (TypeError, ValueError, OverflowError) as exc:
                raise GateDefinitionError("invalid target window bounds") from exc
            if math.isnan(lo) or not math.isfinite(hi) or not lo < hi:
                raise GateDefinitionError(
                    "target window requires lo < hi, a finite hi, and no NaN bounds"
                )
            object.__setattr__(self, "window_ev", (lo, hi))
            return
        try:
            if self.window_ev is None or len(self.window_ev) != 2:
                raise ValueError("expected two window bounds")
            lo, hi = (float(bound) for bound in self.window_ev)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GateDefinitionError(
                "magnetic_subspace needs a two-bound window"
            ) from exc
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise GateDefinitionError(
                "magnetic_subspace needs finite window bounds with lo < hi"
            )
        try:
            weight = np.asarray(self.magnetic_d_weight, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GateDefinitionError(
                "magnetic_subspace needs numerical weights"
            ) from exc
        if weight.ndim != 3 or not np.isfinite(weight).all():
            raise GateDefinitionError(
                "magnetic_subspace needs finite three-dimensional weights"
            )
        if np.any(weight < -1e-9):
            raise GateDefinitionError("magnetic_subspace weights must be nonnegative")
        if not isinstance(self.weight_source, str) or not self.weight_source.strip():
            raise GateDefinitionError("magnetic_subspace needs a weight_source")
        object.__setattr__(self, "window_ev", (lo, hi))
        object.__setattr__(self, "magnetic_d_weight", np.maximum(weight, 0.0))


WHOLE_OCCUPIED_MANIFOLD = CertificationTarget("whole_occupied_manifold")


def magnetic_subspace_target(
    window_ev, magnetic_d_weight, *, weight_source
) -> CertificationTarget:
    """Declare a magnetic manifold by its relative window and parent-state d weights."""
    return CertificationTarget(
        "magnetic_subspace", window_ev, magnetic_d_weight, weight_source
    )


# ---------------------------------------------------------------------------
# the view a check is given
# ---------------------------------------------------------------------------


@dataclass
class Construction:
    """One Wannier construction with the calculation that produced it.

    Checks read this and nothing else; in particular they have no route to another
    construction, so the first rule above is enforced by what is reachable rather
    than by review.
    """

    construction_id: str
    parent: Any
    """A :class:`~.qe_outputs.ParentCalculation`."""
    wannier: Any
    """A :class:`~.w90_outputs.WannierRun`."""
    material_class: MaterialClass
    inputs: dict[str, Any] = field(default_factory=dict)
    """The parameters this construction was run with, for the record."""
    target: CertificationTarget = WHOLE_OCCUPIED_MANIFOLD
    """Chosen when declared; an undeclared construction keeps the default, no window."""
    reference_path: Any = None
    """A :class:`~.criteria.ReferenceBandPath`: DFT bands on a separate high-symmetry path.

    Only the acceptance-criterion checks read it. When the construction carries none,
    the gate supplies the one it was built with, if any; without either, those checks
    are non-evaluable."""

    def require(self, attribute: str) -> Any:
        """Fetch a quantity from the construction or declare the check non-evaluable."""
        value = getattr(self.wannier, attribute, None)
        if value is None:
            raise NonEvaluable(
                f"{self.construction_id}: the run recorded no {attribute}"
            )
        return value


# ---------------------------------------------------------------------------
# scales and thresholds
# ---------------------------------------------------------------------------


#: Two band edges closer than this are one energy, not a gap. It is the numerical
#: resolution of a parent's eigenvalues, not a physical threshold: a symmetry-enforced
#: degeneracy comes out split by the diagonaliser and by the finite precision of the
#: k-point coordinates (graphene's Dirac pair at K: 8.8e-6 eV on the QE 15x15x1 mesh;
#: 8.7e-15 eV on another), and VASP prints eigenvalues to 1e-4 eV. One millielectronvolt
#: is ten times the coarsest of these and three orders of magnitude below the semicore
#: and intra-valence gaps the scale exists to find (eV), so it changes no block boundary
#: that is a real gap. Used only to decide whether band edges touch and whether a block
#: has any energy spread -- never as a tolerance a check applies.
DEGENERACY_RESOLUTION_EV = 1.0e-3


def occupied_block_above_largest_gap(
    eigenvalues_ev, fermi_energy_ev
) -> tuple[float, float]:
    """Occupied bounds above the largest gap in the merged band-edge envelope.

    Each spin band's interval covers all k-points and is clipped at the Fermi level.
    Intervals that touch, or lie within :data:`DEGENERACY_RESOLUTION_EV` of each other,
    belong to one block. Equal gaps choose the one nearest the Fermi level; no
    minimum-gap constant is needed to decide where blocks separate. The top is a
    sampled occupied eigenvalue, never an interpolated Fermi crossing.

    A gap only counts when the occupied states above it have an energy spread: a
    block that is one degenerate level -- a band occupied only where it touches the one
    below, as graphene's pi* band is at K when the Fermi level sits a hair above the
    Dirac point -- is a point of the manifold below it, not a manifold of its own, and
    choosing it would make the scale numerically zero (every occupied state then fails
    a 1 % tolerance). Such a gap is skipped; with no gap left the bounds are the whole
    occupied manifold's.
    """
    e = np.asarray(eigenvalues_ev)
    occupied = e[e <= fermi_energy_ev]
    if occupied.size < 2:
        raise NonEvaluable("parent has no occupied manifold to scale against")
    lower = e.min(axis=1)
    upper = np.minimum(e.max(axis=1), fermi_energy_ev)
    intervals = sorted(
        zip(lower[lower <= fermi_energy_ev], upper[lower <= fermi_energy_ev])
    )
    blocks = []
    for lo, hi in intervals:
        if blocks and lo <= blocks[-1][1] + DEGENERACY_RESOLUTION_EV:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], hi))
        else:
            blocks.append((lo, hi))
    top = float(occupied.max())
    bottom = float(occupied.min())
    candidates = [
        i
        for i in range(1, len(blocks))
        if top - blocks[i][0] > DEGENERACY_RESOLUTION_EV
    ]
    if candidates:
        above = max(
            candidates,
            key=lambda i: (blocks[i][0] - blocks[i - 1][1], i),
        )
        bottom = float(blocks[above][0])
    return bottom, top


def _occupied_bandwidth_ev(c: Construction) -> float:
    """Occupied spread above the largest gap, for either certification target.

    The deepest occupied state measures the pseudopotential or PAO semicore content,
    not the material: it made the scale 54 eV for bcc Fe with Fe_PBE19S and 73 eV for
    CrI3. The band-edge envelope is taken per band over all k and both spins, so a
    band's own coarse sampling cannot split it, and a gap one spin channel fills is
    not a gap in the parent.

    The LARGEST gap is not always the one nearest the valence. A shallower semicore
    shell above it stays inside the scale (Sr_sv 4p; Ti 3p); an intra-valence s–p gap
    in GaAs or SiC can itself be the largest where no semicore gap is larger. These
    are consequences of the rule, not exceptions tuned with a minimum-gap constant.
    With no gap the scale is exactly the old occupied spread. A degenerate level split
    only numerically (graphene's Dirac point) is no gap, and a block with no energy
    spread is never the one chosen (:func:`occupied_block_above_largest_gap`); a
    whole occupied manifold with no spread supplies no energy scale and cannot license
    a tolerance.
    """
    e = c.parent.eigenvalues_ev
    if np.count_nonzero(e <= c.parent.fermi_energy_ev) < 2:
        raise NonEvaluable(
            f"{c.construction_id}: parent has no occupied manifold to scale against"
        )
    bottom, top = occupied_block_above_largest_gap(e, c.parent.fermi_energy_ev)
    if top - bottom <= DEGENERACY_RESOLUTION_EV:
        raise NonEvaluable(f"{c.construction_id}: occupied block has no energy spread")
    return top - bottom


def exclusion_tolerance_ev(eigenvalues_ev, fermi_energy_ev) -> float:
    """The distance ``occupied_manifold_is_represented`` honours an exclusion by.

    1 % of the occupied bandwidth (:func:`_occupied_bandwidth_ev`, computed the same way
    from the parent alone), for the deck check to state the gate's verdict before
    anything runs. Raises :class:`NonEvaluable` where that scale does.
    """
    e = np.asarray(eigenvalues_ev)
    if np.count_nonzero(e <= fermi_energy_ev) < 2:
        raise NonEvaluable("parent has no occupied manifold to scale against")
    bottom, top = occupied_block_above_largest_gap(e, fermi_energy_ev)
    if top - bottom <= DEGENERACY_RESOLUTION_EV:
        raise NonEvaluable("occupied block has no energy spread")
    return 0.01 * (top - bottom)


def magnetic_subspace_mask(c: Construction, spin: int) -> np.ndarray:
    """States inside the declared window with majority magnetic-d character.

    The window, not occupation, bounds this manifold. Its empty states count too;
    a declaration restricted to occupied states places its upper bound at zero.
    """
    if c.target.name != "magnetic_subspace":
        raise NonEvaluable("construction declares no magnetic_subspace target")
    weight = c.target.magnetic_d_weight
    if weight.shape != c.parent.eigenvalues_ev.shape:
        raise NonEvaluable(
            "magnetic-d weight shape differs from the parent's eigenvalues"
        )
    lo, hi = c.target.window_ev
    ef = c.parent.fermi_energy_ev
    e = c.parent.eigenvalues_ev[spin]
    return (
        (e >= ef + lo)
        & (e <= ef + hi)
        & (weight[spin] >= MAGNETIC_D_WEIGHT_FLOOR.value(c))
    )


def _magnetic_manifold_bandwidth_ev(c: Construction) -> float:
    """Spread of the counted magnetic states over all k-points and both spins.

    The tolerance belongs to the declared manifold itself, including its empty
    states when the window reaches above the Fermi level. Semicore and ligand
    states outside that declaration supply no scale for it. Fewer than two counted
    states, or a flat manifold, cannot supply an energy tolerance.
    """
    if c.target.name != "magnetic_subspace":
        raise NonEvaluable("construction declares no magnetic_subspace target")
    states = np.concatenate(
        [
            e[magnetic_subspace_mask(c, spin)]
            for spin, e in enumerate(c.parent.eigenvalues_ev)
        ]
    )
    if states.size < 2:
        raise NonEvaluable("fewer than two magnetic-subspace states to scale against")
    spread = float(np.ptp(states))
    if spread <= 0:
        raise NonEvaluable("magnetic manifold has no energy spread")
    return spread


def _nearest_neighbour_distance_ang(c: Construction) -> float:
    """Shortest interatomic distance, including periodic images — a length the structure sets."""
    lattice = c.parent.lattice_ang
    frac = c.parent.positions_frac
    if frac is None or len(frac) == 0:
        raise NonEvaluable(f"{c.construction_id}: parent records no atomic positions")
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    best = math.inf
    for a in range(len(frac)):
        for b in range(len(frac)):
            for s in shifts:
                if a == b and not s.any():
                    continue
                d = np.linalg.norm((frac[b] + s - frac[a]) @ lattice)
                best = min(best, float(d))
    if not math.isfinite(best):
        raise NonEvaluable(f"{c.construction_id}: no interatomic distance available")
    return best


def _onsite_hopping_scale_ev(c: Construction) -> float:
    """Largest on-site matrix element — the natural scale for a distant hopping."""
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    home = np.all(np.asarray(rvec) == 0, axis=1)
    if not home.any():
        raise NonEvaluable(
            f"{c.construction_id}: the real-space Hamiltonian has no R=0 block"
        )
    return float(np.abs(np.asarray(h)[home][0]).max())


def _occupied_state_fraction(_c: Construction) -> float:
    """Unity: a scale for quantities that are already fractions of a count.

    Present so a dimensionless check still declares a scale and a calibration rather
    than carrying a bare number with no account of where it came from.
    """
    return 1.0


def _published_absolute_unit(_c: Construction) -> float:
    """Unity in the criterion's own unit (meV, or Angstrom), for a published criterion.

    Some criteria are defined by their publication as an absolute number -- eta_2 <= 20
    meV, a centre within 0.1 Angstrom of an atom -- and their value lies in being the
    same number the literature reports its success rates against. The material supplies
    no scale for them by design. Declaring this scale keeps such a threshold inside the
    rule: it is still a fraction of a named scale, and its calibration must cite the
    publication the number comes from.
    """
    return 1.0


SCALES: dict[str, Callable[[Construction], float]] = {
    "occupied_state_fraction": _occupied_state_fraction,
    "published_absolute_mev": _published_absolute_unit,
    "published_absolute_ang": _published_absolute_unit,
    "occupied_bandwidth_ev": _occupied_bandwidth_ev,
    "nearest_neighbour_distance_ang": _nearest_neighbour_distance_ang,
    "onsite_hopping_scale_ev": _onsite_hopping_scale_ev,
    "magnetic_manifold_bandwidth_ev": _magnetic_manifold_bandwidth_ev,
}


# ---------------------------------------------------------------------------
# the bands a construction was asked to represent
# ---------------------------------------------------------------------------


def _band_list(indices: Sequence[int]) -> str:
    """``1-4,7`` for (1, 2, 3, 4, 7): the deck's own spelling, for notes."""
    groups: list[list[int]] = []
    for i in indices:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return ",".join(f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups)


def parent_bands_for_check(c: Construction, spin_index: int) -> tuple[np.ndarray, str]:
    """Share the exclusion decision, using the full parent's matching tolerance.

    With no exclusion, return the original bands without asking for a scale. If
    the scale is unavailable, an occupied exclusion cannot be honoured safely.
    """
    bands = c.parent.eigenvalues_ev[spin_index]
    declared = tuple(int(i) for i in (c.inputs.get("exclude_bands") or ()))
    if not declared:
        return bands, ""
    try:
        tolerance = 0.01 * SCALES["occupied_bandwidth_ev"](c)
    except NonEvaluable as exc:
        valid = [i - 1 for i in declared if 1 <= i <= bands.shape[1]]
        if not np.any(bands[:, valid] <= c.parent.fermi_energy_ev):
            return bands, ""
        return bands, f"; exclude_bands {_band_list(declared)} not honoured: {exc}"
    return represented_parent_bands(c, spin_index, tolerance)


def represented_parent_bands(
    c: Construction, spin_index: int, tolerance_ev: float
) -> tuple[np.ndarray, str]:
    """The parent's bands this construction was asked to represent, and a note saying why.

    A deck's ``exclude_bands`` removes bands before Wannier90 reads anything else, so the
    model is never asked to represent them -- the 2025 frontier construction excludes the
    pseudopotential's semicore shells this way. A check that counts those bands refuses
    the construction by convention, not by construction.

    A declaration is not taken on trust, because a deck can exclude any band, including a
    valence band the model failed to reproduce; honouring that would let the exclusion
    redefine the target and certify the failure. The exclusion is therefore honoured only
    when the excluded bands are an isolated block *below* everything the model represents:
    over the whole mesh, the highest excluded eigenvalue must lie below the lowest retained
    one by more than ``tolerance_ev`` -- the calling check's own matching tolerance (for
    ``occupied_manifold_is_represented``, 1 % of the occupied bandwidth), so the rule
    carries no constant of its own. A gap smaller than the distance at which the check
    calls two states the same cannot separate "excluded" from "represented". When the rule
    fails, the full set of bands is returned and the note says the exclusion was not
    honoured; the check then counts every state, exactly as it did before the gate read
    exclusions. This rule is the guard the 2026-10-01 plan requires (its adversarial
    control): it adds no threshold and changes no check's threshold.

    An exclusion that removes no occupied state cannot change what an occupied-state count
    sees, and returns the full set with an empty note -- byte-identical to a deck that
    excludes nothing. A deck that excludes semicore at the bottom *and* unoccupied
    headroom at the top (the VASP SMV deck) has the headroom set aside and its semicore
    block judged by the rule.
    """
    bands = np.asarray(c.parent.eigenvalues_ev[spin_index])
    declared = tuple(int(i) for i in (c.inputs.get("exclude_bands") or ()))
    verdict = judge_exclusion(bands, declared, c.parent.fermi_energy_ev, tolerance_ev)
    if verdict.status == "honoured":
        return verdict.represented, verdict.note
    return bands, verdict.note


@dataclass(frozen=True)
class ExclusionVerdict:
    """How the gate treats a deck's ``exclude_bands`` on one spin channel.

    ``status`` is ``none`` (nothing excluded), ``irrelevant`` (no occupied state is
    excluded, so no occupied-state count can see the exclusion), ``not_honoured`` or
    ``honoured``. ``note`` is the text the checks append, byte for byte; ``gap_ev`` the
    distance from the highest excluded eigenvalue judged to the lowest retained one
    (negative when they overlap), ``below`` and ``headroom`` the 1-based bands judged
    as the bottom block and set aside as headroom.
    """

    status: str
    note: str = ""
    gap_ev: float | None = None
    below: tuple[int, ...] = ()
    headroom: tuple[int, ...] = ()
    represented: Any = None


def judge_exclusion(
    bands: np.ndarray,
    declared: Sequence[int],
    fermi_energy_ev: float,
    tolerance_ev: float,
) -> ExclusionVerdict:
    """The gate's honour rule for a declared exclusion, on one channel's bands (k, band).

    This is the rule :func:`represented_parent_bands` applies, factored out so the deck
    check (``validate_deck``) states the same verdict before anything runs.
    """
    bands = np.asarray(bands)
    declared = tuple(int(i) for i in declared)
    if not declared:
        return ExclusionVerdict("none")
    n_bands = bands.shape[1]
    label = _band_list(declared)
    if max(declared) > n_bands or min(declared) < 1:
        return ExclusionVerdict(
            "not_honoured",
            f"; exclude_bands {label} not honoured: the parent has {n_bands} bands",
        )
    excluded = np.zeros(n_bands, dtype=bool)
    excluded[[i - 1 for i in declared]] = True
    if not np.any(bands[:, excluded] <= fermi_energy_ev):
        return ExclusionVerdict("irrelevant")
    if excluded.all():
        return ExclusionVerdict(
            "not_honoured",
            f"; exclude_bands {label} not honoured: it removes every band of the parent",
        )
    # A deck may also exclude a headroom block at the top (bands past num_bands, as the
    # VASP SMV deck writes ``exclude_bands = 1-N, M-<last>``). Those bands lie above every
    # represented band at each k-point by index; when they hold no occupied state they are
    # set aside, and the rest of the exclusion is judged by the rule above. A top block
    # holding an occupied state is not headroom, and the whole exclusion is judged as one.
    top = np.zeros(n_bands, dtype=bool)
    last_kept = int(np.flatnonzero(~excluded).max())
    top[last_kept + 1 :] = True
    headroom = bool(top.any()) and not np.any(bands[:, top] <= fermi_energy_ev)
    below = excluded & ~top if headroom else excluded
    gap = float(bands[:, ~excluded].min() - bands[:, below].max())
    below_bands = tuple(int(i) + 1 for i in np.flatnonzero(below))
    top_bands = tuple(int(i) + 1 for i in np.flatnonzero(top)) if headroom else ()
    if gap <= tolerance_ev:
        return ExclusionVerdict(
            "not_honoured",
            f"; exclude_bands {label} not honoured: the excluded bands reach "
            f"{-gap:.2f} eV into the represented ones (an isolated block below them, "
            f"more than {tolerance_ev * 1000:.0f} meV apart, is required)",
            gap,
            below_bands,
            top_bands,
        )
    note = (
        f"; bands {_band_list(list(below_bands))} excluded by the "
        f"deck and not counted (isolated {gap:.2f} eV below the represented manifold)"
    )
    if headroom:
        note += (
            f"; headroom bands {_band_list(list(top_bands))} "
            "(unoccupied, above num_bands) set aside"
        )
    return ExclusionVerdict(
        "honoured", note, gap, below_bands, top_bands, bands[:, ~excluded]
    )


@dataclass(frozen=True)
class ScaleAwareThreshold:
    """A dimensionless fraction of a quantity the material itself supplies.

    ``calibration`` is mandatory and records where the fraction came from *and on
    which material*, because a constant that travels without its calibrating material
    is the defect this class exists to remove.
    """

    fraction: float
    scale: str
    calibration: str

    def __post_init__(self) -> None:
        if not (self.fraction > 0 and math.isfinite(self.fraction)):
            raise GateDefinitionError(
                f"fraction must be positive and finite, got {self.fraction!r}"
            )
        if self.scale not in SCALES:
            raise GateDefinitionError(
                f"unknown scale {self.scale!r}; known: {sorted(SCALES)}"
            )
        if not self.calibration.strip():
            raise GateDefinitionError(
                "a scale-aware threshold must record its calibration: the source of "
                "the fraction and the material it was fitted on"
            )

    def value(self, construction: Construction) -> float:
        return self.fraction * SCALES[self.scale](construction)


MAGNETIC_D_WEIGHT_FLOOR = ScaleAwareThreshold(
    0.5,
    "occupied_state_fraction",
    calibration="A state belongs to the magnetic subspace when at least half of its norm "
    "lies in the span of the magnetic sites' d orbitals: majority character, the one "
    "fraction at which a state is assigned to the d manifold or to the rest by dominance "
    "alone. Set a priori on 2026-09-15 before any construction or fixture was scored, "
    "not fitted. Fixed in the gate rather than per declaration so that 'passed under "
    "magnetic_subspace' means one thing for every user of the tool.",
)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    """What a check measured.

    ``passed`` is an explicit bool for a discrete check and ``None`` for a scalar one:
    there the gate applies the threshold, so a check cannot smuggle a constant past
    the rule by deciding for itself.

    A refusal is a contractual verdict that can only fail a check, never pass one,
    so it cannot relax the gate. The metric and threshold comparison remain recorded
    beside it as a diagnostic.
    """

    passed: bool | None
    metric_value: float
    metric_units: str
    note: str = ""
    refusal: str | None = None
    record: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StandIn:
    """A contributed check's declaration that it stands in for a core check on one route.

    Route-scoped withholding lives here and nowhere else: a core check is withheld on a
    construction route only because a contributed check declares itself its replacement
    there, with the physical reason the core check has no meaning on that route (a
    closest-Wannier construction runs no iterations; its centres are pinned to the guiding
    orbitals). Nothing can be withheld without something added in its place, which keeps
    route scoping inside the rule that contributed checks may only add.
    """

    check_id: str
    route: str
    reason: str

    def __post_init__(self) -> None:
        if not self.route.strip() or not self.reason.strip():
            raise GateDefinitionError(
                f"stand-in for {self.check_id!r} on {self.route!r} needs a route and reason"
            )


@dataclass(frozen=True)
class Check:
    check_id: str
    quantity: str
    verdict_type: VerdictType
    reference: Reference
    evaluate: Callable[[Construction], CheckOutcome]
    threshold: ScaleAwareThreshold | None = None
    applies_to: tuple[MaterialClass, ...] = ("metal", "gapped")
    """Material classes this check is meaningful for.

    A check that does not apply is recorded as such rather than waived: a band
    distance measured against a continuum is not a lenient verdict, it is the wrong
    instrument, and the record must say which.
    """
    origin: str = "core"
    """``core`` or the name of the contributor. Community checks may only add."""
    applies_to_routes: tuple[str, ...] | None = None
    """Construction routes (``mlwf``, ``cwf`` …) this check is meaningful on; ``None`` is
    every route. The route reaches the gate as ``Construction.inputs["route"]`` (default
    ``mlwf``). A core check never sets this — see :class:`StandIn`."""
    stands_in_for: tuple[StandIn, ...] = ()
    """The core checks this *contributed* check replaces, and on which route. Refused on
    a core check, on a non-core target, and when this check itself does not apply on the
    route it withholds."""
    description: str = ""
    """What passing this check means, in one plain clause. A profile's
    :meth:`~.gate_profiles.GateProfile.target_text` is assembled from these; a check
    without one is described by its ``quantity``."""

    def __post_init__(self) -> None:
        if self.reference not in ALLOWED_REFERENCES:
            raise GateDefinitionError(
                f"check {self.check_id!r} resolves against {self.reference!r}; a check "
                f"may only reference {', '.join(ALLOWED_REFERENCES)} — never another "
                "localized representation, which is the quantity under comparison"
            )
        if self.verdict_type == "scalar_vs_scale":
            if not isinstance(self.threshold, ScaleAwareThreshold):
                raise GateDefinitionError(
                    f"check {self.check_id!r}: a scalar check needs a ScaleAwareThreshold, "
                    "not a bare constant"
                )
        elif self.threshold is not None:
            raise GateDefinitionError(
                f"check {self.check_id!r}: a discrete check takes no threshold"
            )


@dataclass
class Verdict:
    """One check's judgement of one construction.

    A refusal is a contractual verdict that can only fail a check, never pass one,
    so it cannot relax the gate. The metric and threshold comparison remain recorded
    beside it as a diagnostic.
    """

    check_id: str
    evaluable: bool
    passed: bool | None
    metric_value: float | None
    metric_units: str
    threshold_value: float | None
    applies: bool
    note: str
    refusal: str | None = None
    metric_within_threshold: bool | None = None
    record: dict[str, Any] = field(default_factory=dict)
    profile: str = ""
    """The gate profile this verdict was made under."""
    profile_hash: str = ""
    """:func:`check_list_hash` of that gate's effective check list."""

    def counts_against_acceptance(self) -> bool:
        return self.applies and self.evaluable and self.passed is False


@dataclass
class GateReport:
    construction_id: str
    material_class: MaterialClass
    verdicts: tuple[Verdict, ...]
    target: str = "whole_occupied_manifold"
    profile: str = ""
    """The gate profile the report was made under (``default`` unless chosen)."""
    profile_hash: str = ""
    """:func:`check_list_hash` of the effective check list; equal hashes, same questions."""

    @property
    def evaluated_check_ids(self) -> tuple[str, ...]:
        return tuple(v.check_id for v in self.verdicts if v.applies and v.evaluable)

    @property
    def failed_check_ids(self) -> tuple[str, ...]:
        return tuple(v.check_id for v in self.verdicts if v.counts_against_acceptance())

    @property
    def non_evaluable_check_ids(self) -> tuple[str, ...]:
        return tuple(v.check_id for v in self.verdicts if v.applies and not v.evaluable)

    @property
    def passed(self) -> bool:
        """Every applicable check was evaluable, and every one of them passed.

        Missing evidence is not a pass. A check that applies to this material but
        could not be evaluated means the construction did not produce what the check
        needs, and certifying it anyway would let a run be accepted on the strength
        of the outputs it failed to write — the first construction this gate scored
        was accepted that way, having answered four of its eight applicable checks.

        The two states stay separate in the report: :attr:`failed_check_ids` is what
        the construction got wrong, :attr:`non_evaluable_check_ids` is what it never
        showed. A repair addresses those differently.
        """
        return (
            bool(self.evaluated_check_ids)
            and not self.failed_check_ids
            and not self.non_evaluable_check_ids
        )

    @property
    def coverage(self) -> float:
        """Fraction of applicable checks that returned a verdict."""
        applicable = [v for v in self.verdicts if v.applies]
        if not applicable:
            return 0.0
        return sum(1 for v in applicable if v.evaluable) / len(applicable)

    def measurements(self) -> dict[str, Any]:
        measurements = {}
        for v in self.verdicts:
            row = {
                "value": v.metric_value,
                "units": v.metric_units,
                "threshold": v.threshold_value,
                "passed": v.passed,
                "evaluable": v.evaluable,
                "applies": v.applies,
                "note": v.note,
            }
            if v.record:
                for key, value in v.record.items():
                    if key not in row:
                        row[key] = value
            if v.refusal is not None:
                row["refusal"] = v.refusal
                row["metric_within_threshold"] = v.metric_within_threshold
            measurements[v.check_id] = row
        return measurements


class CertificationGate:
    """A profile's checks, plus whatever a backend or a user contributes.

    ``core`` is the profile's check list (``CORE_CHECKS``, the ``default`` profile, when
    built without one). A contributed check in ``extra`` that shares an identifier with
    one of them **replaces** it and emits a :class:`GateProfileWarning`; nothing is
    refused for being a replacement. A contributed check still may not claim the
    ``core`` origin -- the provenance of a criterion is part of it -- and the verdicts
    record the profile name and :func:`check_list_hash` of the effective list, so a
    replacement is never invisible.
    """

    def __init__(
        self,
        core: Sequence[Check],
        extra: Sequence[Check] = (),
        *,
        profile_name: str | None = None,
        reference_path: Any = None,
    ) -> None:
        import warnings

        self.targets = CERTIFICATION_TARGETS
        checks = list(core)
        position = {c.check_id: i for i, c in enumerate(checks)}
        contributed: list[Check] = []
        for c in extra:
            if c.origin == "core":
                raise GateDefinitionError(
                    f"added check {c.check_id!r} claims origin 'core'; contributed "
                    "checks must name their contributor"
                )
            if c.check_id in position:
                warnings.warn(
                    f"contributed check {c.check_id!r} ({c.origin}) replaces the "
                    f"profile's check of the same name; verdicts record the changed "
                    "check-list hash",
                    GateProfileWarning,
                    stacklevel=2,
                )
                checks[position[c.check_id]] = c
            else:
                position[c.check_id] = len(checks)
                checks.append(c)
            contributed.append(c)
        self.checks: tuple[Check, ...] = tuple(checks)
        self.reference_path = reference_path
        self.profile_hash = check_list_hash(self.checks)
        if profile_name is None:
            from .core_checks import CORE_CHECKS

            default_ids = tuple(c.check_id for c in CORE_CHECKS)
            profile_name = (
                "default"
                if tuple(c.check_id for c in core) == default_ids
                else "custom"
            )
        self.profile_name = profile_name
        core_ids = {c.check_id for c in self.checks if c.origin == "core"}
        #: Route-scoped withholding, derived from the contributed checks only:
        #: ``(core check id, route) -> (the contributed check that stands in, why)``. A core
        #: check is unconditional on every route unless a contributed check stands in for it
        #: there, so nothing can be withheld without something added in its place.
        self.withholdings: dict[tuple[str, str], tuple[str, str]] = {}
        for check in self.checks:
            if check.origin == "core" and (
                check.stands_in_for or check.applies_to_routes is not None
            ):
                raise GateDefinitionError(
                    f"core check {check.check_id!r} may not be route-scoped or stand in for "
                    "another check; route-scoped withholding belongs to contributed checks"
                )
        for check in contributed:
            for stand_in in check.stands_in_for:
                key = (stand_in.check_id, stand_in.route)
                if stand_in.check_id not in core_ids:
                    raise GateDefinitionError(
                        f"stand-in names non-core check {stand_in.check_id!r}"
                    )
                if (
                    check.applies_to_routes is not None
                    and stand_in.route not in check.applies_to_routes
                ):
                    raise GateDefinitionError(
                        f"{check.check_id!r} does not apply on {stand_in.route!r}"
                    )
                if key in self.withholdings:
                    raise GateDefinitionError(
                        f"duplicate stand-in for {stand_in.check_id!r} on {stand_in.route!r}"
                    )
                self.withholdings[key] = (check.check_id, stand_in.reason)

    def run(self, construction: Construction) -> GateReport:
        verdicts = []
        if (
            getattr(construction, "reference_path", None) is None
            and self.reference_path is not None
        ):
            import dataclasses

            construction = dataclasses.replace(
                construction, reference_path=self.reference_path
            )
        route = construction.inputs.get("route", "mlwf")
        for check in self.checks:
            note = None
            if construction.material_class not in check.applies_to:
                note = f"not meaningful for a {construction.material_class} manifold"
            elif (
                check.applies_to_routes is not None
                and route not in check.applies_to_routes
            ):
                note = f"not meaningful on the {route} route"
            elif (check.check_id, route) in self.withholdings:
                replacement, reason = self.withholdings[(check.check_id, route)]
                note = (
                    f"withheld on the {route} route: {reason}; {replacement} stands in"
                )
            if note is not None:
                verdicts.append(
                    Verdict(
                        check.check_id,
                        evaluable=False,
                        passed=None,
                        metric_value=None,
                        metric_units="",
                        threshold_value=None,
                        applies=False,
                        note=note,
                    )
                )
                continue
            try:
                outcome = check.evaluate(construction)
            except NotApplicable as exc:
                verdicts.append(
                    Verdict(
                        check.check_id,
                        evaluable=False,
                        passed=None,
                        metric_value=None,
                        metric_units="",
                        threshold_value=None,
                        applies=False,
                        note=str(exc),
                        record=dict(exc.record),
                    )
                )
                continue
            except NonEvaluable as exc:
                verdicts.append(
                    Verdict(
                        check.check_id,
                        evaluable=False,
                        passed=None,
                        metric_value=None,
                        metric_units="",
                        threshold_value=None,
                        applies=True,
                        note=str(exc),
                    )
                )
                continue

            threshold_value = None
            passed = outcome.passed
            within = None
            if check.verdict_type == "scalar_vs_scale":
                try:
                    threshold_value = check.threshold.value(construction)
                except NonEvaluable as exc:
                    verdicts.append(
                        Verdict(
                            check.check_id,
                            evaluable=False,
                            passed=None,
                            metric_value=outcome.metric_value,
                            metric_units=outcome.metric_units,
                            threshold_value=None,
                            applies=True,
                            note=f"threshold not derivable: {exc}",
                        )
                    )
                    continue
                passed = bool(outcome.metric_value <= threshold_value)
                if math.isfinite(outcome.metric_value):
                    within = passed

            if outcome.refusal is not None:
                passed = False

            verdicts.append(
                Verdict(
                    check.check_id,
                    evaluable=True,
                    passed=passed,
                    metric_value=outcome.metric_value,
                    metric_units=outcome.metric_units,
                    threshold_value=threshold_value,
                    applies=True,
                    note=outcome.note,
                    refusal=outcome.refusal,
                    metric_within_threshold=(
                        within if outcome.refusal is not None else None
                    ),
                    record=dict(outcome.record),
                )
            )
        for verdict in verdicts:
            verdict.profile = self.profile_name
            verdict.profile_hash = self.profile_hash
        return GateReport(
            construction.construction_id,
            construction.material_class,
            tuple(verdicts),
            getattr(construction, "target", WHOLE_OCCUPIED_MANIFOLD).name,
            profile=self.profile_name,
            profile_hash=self.profile_hash,
        )
