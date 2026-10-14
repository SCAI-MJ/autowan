"""Gate profiles: named, hashed lists of checks that say what "passed" means.

One profile drives everything a run decides (Item 4 decision 13, 2026-10-07): the loop's
stop condition, the repair prompt's target text, and the frontier rescan-grid selection.
The profile's name and :attr:`GateProfile.hash` are recorded with every verdict.

* ``default`` -- :data:`~.core_checks.CORE_CHECKS`: counts, the representable supercell,
  the parent-band checks, hopping decay, a metal's frozen window, and the drift guard
  ``centres_stay_near_home_cell``. Atom-centring is not in it, and neither is convergence
  (since 2026-10-09, decision 22): the gate judges the construction, not how the
  optimiser stopped.
* ``band_interpolation`` -- default + ``band_interpolation_eta2`` (eta_2 <= 20 meV at
  E_F + 2 eV on a separate path).
* ``sparse_tb`` -- default + ``truncated_hopping_eta2`` (the same, hoppings beyond
  L_BvK/4 removed).
* ``downfolded_d`` -- a **d-manifold target** (user decision 2026-10-09): the default
  set without its whole-occupied-manifold checks (:data:`WHOLE_MANIFOLD_CHECKS`), plus
  ``d_only_downfolding`` (a d-only model on the transition-metal sites: 5 functions per
  TM atom and spin, TM d projectors only, centres on TM sites, eta <= 20 meV over
  E_F +- 1 eV against the matched DFT bands). Its ``target_manifold`` is ``d_manifold``,
  which also lifts the repair validator's occupied-band floor and the deck check's
  refusal of an unhonoured occupied exclusion (both exist only for the whole-manifold
  target). Which default checks it keeps, and why: :data:`DOWNFOLDED_D_RATIONALE`.
  It replaced ``conduction_band`` as the third criterion profile on 2026-10-09 (user
  decision); ``conduction_band_eta`` stays registered, and the retired profile is
  rebuilt by name (:data:`RETIRED_PROFILES`, with a warning) or as
  ``default --add-check conduction_band_eta`` -- the same check-list hash.
* ``dmft_subspace`` -- the **literature's DMFT subspace** (user decision 2026-10-12,
  the third criterion from then on): the same sub-manifold base as ``downfolded_d``
  plus ``dmft_subspace_model`` (:mod:`.dmft_subspace`): for SrVO3 and SrTiO3 the
  t2g orbitals, for NiO and MnO the metal 3d with O 2p, for Fe its 4s and 3d; the
  model must seed exactly that set and reproduce the DFT states of the target manifold
  (projected weight >= 0.5) within 20 meV RMS. Withheld for every other material. Its
  target text names the structure's own target (:meth:`GateProfile.target_text` with
  ``species``). ``downfolded_d`` stays a named profile so its stored verdicts reproduce.
  Since the second 2026-10-12 revision (user decision) the check is
  ``dmft_subspace_model_window``: an entangled target manifold is compared only up to
  the frontier protocol's frozen-window top (E_F + 2 eV, or the CBM + 2 eV when
  gapped), an isolated one over its whole width; the target text also says which case
  the structure is, when the run's reference path is given. The first form,
  ``dmft_subspace_model`` (profile hash 941cf34c1c36bce4), stays registered: that hash
  is rebuilt as ``dmft_subspace --drop-check dmft_subspace_model_window --add-check
  dmft_subspace_model``.
* ``atom_centred_basis`` -- default + ``centres_on_atoms_atom_centred`` (centres within
  0.1 Angstrom of an atom).

A custom profile starts from any of these and adds or drops registered checks
(:func:`custom_profile`, ``--gate-profile/--add-check/--drop-check``). A *group* name
adds or drops its checks together: ``convergence`` is ``subspace_selection_converged``
and ``spread_minimisation_converged`` (:data:`CHECK_GROUPS`), so ``default+convergence``
is the default set as it stood until 2026-10-09 (tags v1 to v4). Every check is
removable and there are no tiers: dropping a check of the default set emits a
:class:`~.gate.GateProfileWarning` and nothing more. The registry is open
(:func:`register_check`).

The API the repair loop reads is small and meant to stay put: :class:`GateProfile` with
``name``, ``checks``, ``check_ids``, ``hash`` and :meth:`GateProfile.target_text`;
:func:`get_profile`; :func:`build_gate`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .core_checks import CONVERGENCE_CHECKS, CORE_CHECKS, OPTIONAL_CHECKS
from .criteria import CRITERION_CHECKS
from .dmft_subspace import (
    DMFT_SUBSPACE_CHECK_IDS,
    DMFT_SUBSPACE_MODEL,
    DMFT_SUBSPACE_MODEL_UNWINDOWED,
    comparison_text,
    target_for_species,
)
from .downfolding import D_ONLY_DOWNFOLDING
from .gate import (
    CertificationGate,
    Check,
    GateDefinitionError,
    GateProfileWarning,
    check_list_hash,
)

__all__ = [
    "CHECK_GROUPS",
    "CHECK_REGISTRY",
    "DEFAULT_CHECK_IDS",
    "GateProfile",
    "GateProfileWarning",
    "PROFILES",
    "RETIRED_PROFILES",
    "WHOLE_MANIFOLD_CHECKS",
    "build_gate",
    "custom_profile",
    "get_profile",
    "profile_names",
    "register_check",
]

#: Every check a profile may name, by identifier. Open: see :func:`register_check`.
CHECK_REGISTRY: dict[str, Check] = {
    c.check_id: c
    for c in (
        *CORE_CHECKS,
        *OPTIONAL_CHECKS,
        *CRITERION_CHECKS,
        D_ONLY_DOWNFOLDING,
        DMFT_SUBSPACE_MODEL,
        DMFT_SUBSPACE_MODEL_UNWINDOWED,
    )
}

#: The ``default`` profile's checks, in report order.
DEFAULT_CHECK_IDS: tuple[str, ...] = tuple(c.check_id for c in CORE_CHECKS)

#: Named groups of registered checks that ``--add-check`` / ``--drop-check`` (and
#: :func:`custom_profile`) accept in place of a check identifier; a group adds or drops
#: its checks together and the derived profile is labelled with the group's name.
CHECK_GROUPS: dict[str, tuple[str, ...]] = {
    "convergence": tuple(c.check_id for c in CONVERGENCE_CHECKS),
}

#: Default checks that hold a model to the parent's whole occupied manifold. A profile
#: whose target is a sub-manifold (``downfolded_d``) does not run them.
WHOLE_MANIFOLD_CHECKS: tuple[str, ...] = (
    "frontier_manifold_straddles_fermi_level",
    "occupied_manifold_is_represented",
    "band_distance_from_parent",
)

#: Each default check against the d-manifold target, kept or dropped, in one line.
DOWNFOLDED_D_RATIONALE: dict[str, tuple[str, str]] = {
    "wannier_count_matches_request": (
        "keep",
        "the run built the functions the deck asked for: a property of any construction",
    ),
    "hopping_shells_inside_representable_supercell": (
        "keep",
        "H(R) stays inside the supercell the mesh represents: true of any sub-manifold",
    ),
    "frontier_manifold_straddles_fermi_level": (
        "drop",
        "asks that the model's bands reach the parent's highest occupied (frontier) "
        "states; a d-only model of a compound leaves those ligand states out by design",
    ),
    "centres_stay_near_home_cell": (
        "keep",
        "the drift guard (nearest-atom image, v5): a function attached to no home-cell "
        "atom is broken whatever manifold it spans",
    ),
    "occupied_manifold_is_represented": (
        "drop",
        "every occupied parent state must have a model band: the whole occupied "
        "manifold by definition, which a d-only model does not claim",
    ),
    "model_bands_correspond_to_parent_bands": (
        "keep",
        "one-directional -- every model band inside the window must sit on some parent "
        "band (no invented states); it never asks that parent states be covered, so it "
        "is consistent with a sub-manifold; the occupied bandwidth only sizes its "
        "tolerance",
    ),
    "band_distance_from_parent": (
        "drop",
        "pairs every parent state of the occupied window with its nearest model band, "
        "so every occupied state must be represented: the whole manifold again",
    ),
    "hopping_decays_with_distance": (
        "keep",
        "outermost hoppings small beside the on-site scale: a property of any "
        "localized model",
    ),
    "metal_frozen_window_contains_fermi_level": (
        "keep",
        "a metal's declared frozen window must contain E_F; a d-only model's window "
        "sits around E_F, so the requirement is unchanged",
    ),
}

#: The manifold a named profile holds a construction to, when not the whole occupied one.
PROFILE_TARGETS: dict[str, str] = {
    "downfolded_d": "d_manifold",
    "dmft_subspace": "correlated_subspace",
}

WHOLE_OCCUPIED_MANIFOLD_TARGET = "whole_occupied_manifold"

#: Named profiles: (check identifiers, the use the profile serves).
PROFILES: dict[str, tuple[tuple[str, ...], str]] = {
    "default": (
        DEFAULT_CHECK_IDS,
        "a sound construction of the declared manifold, with no use-specific criterion",
    ),
    "band_interpolation": (
        (*DEFAULT_CHECK_IDS, "band_interpolation_eta2"),
        "a tight-binding Hamiltonian used to interpolate bands near the Fermi level",
    ),
    "sparse_tb": (
        (*DEFAULT_CHECK_IDS, "truncated_hopping_eta2"),
        "a sparse tight-binding model that keeps only hoppings within L_BvK / 4",
    ),
    "downfolded_d": (
        (
            *(i for i in DEFAULT_CHECK_IDS if i not in WHOLE_MANIFOLD_CHECKS),
            "d_only_downfolding",
        ),
        "a minimal correlated subspace -- the d orbitals of the transition-metal sites "
        "only -- for DMFT, cRPA or DFT+U",
    ),
    "dmft_subspace": (
        (
            *(i for i in DEFAULT_CHECK_IDS if i not in WHOLE_MANIFOLD_CHECKS),
            "dmft_subspace_model_window",
        ),
        "the correlated-subspace Wannier model a DMFT calculation of this material is "
        "built on, as the literature builds it",
    ),
    "atom_centred_basis": (
        (*DEFAULT_CHECK_IDS, "centres_on_atoms_atom_centred"),
        "functions labelled by atom and orbital (DFT+U, DMFT, exchange couplings)",
    ),
}


#: Profiles that were named and no longer are: name -> (check identifiers, purpose, why
#: retired). :func:`get_profile` still resolves them, with a
#: :class:`~.gate.GateProfileWarning`, so that verdicts stored under them can be
#: reproduced (same checks, same hash); they are not offered by :func:`profile_names`
#: or the command line.
RETIRED_PROFILES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "conduction_band": (
        (*DEFAULT_CHECK_IDS, "conduction_band_eta"),
        "an insulator's conduction band up to 2 eV above its minimum",
        "retired 2026-10-09 (user decision): the third criterion is the d-only "
        "downfolded model (downfolded_d); conduction_band_eta stays registered",
    ),
}

#: What a repair should aim at under a profile, appended to its target text. The check
#: descriptions say what passing means; this says which moves reach it.
PROFILE_GUIDANCE: dict[str, str] = {
    "downfolded_d": (
        "To reach this target, build the construction on the transition-metal d "
        "orbitals alone: under SMV, name projectors with only the d shell of each "
        "transition-metal species and no projector on any other species; under PDWF, "
        "keep only the transition-metal d orbital in pao_subset and give every other "
        "species an empty list; num_wann then follows as five per transition-metal atom "
        "(ten for spinor functions). This target is the d manifold, not the whole "
        "occupied manifold: occupied ligand or s bands the d functions leave out are "
        "not a failure, and fewer functions than occupied bands is allowed. Place the "
        "energy windows around the Fermi level, where the d bands sit: freeze only "
        "states the d functions can hold (a frozen window that takes in ligand bands "
        "holds more states than there are functions) and keep the outer window wide "
        "enough to contain the whole d manifold. A d-only model reproduces only the "
        "bands of d character; ligand bands inside the window are not compared."
    ),
}


def register_check(check: Check, *, replace: bool = False) -> None:
    """Make a check nameable by profiles and ``--add-check``."""
    if not isinstance(check, Check):
        raise GateDefinitionError(f"{check!r} is not a Check")
    if check.check_id in CHECK_REGISTRY and not replace:
        raise GateDefinitionError(
            f"a check named {check.check_id!r} is already registered; pass "
            "replace=True to replace it"
        )
    CHECK_REGISTRY[check.check_id] = check


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


@dataclass(frozen=True)
class GateProfile:
    """What "passed" means for one run: a name and the checks that define it."""

    name: str
    checks: tuple[Check, ...]
    purpose: str = ""
    base: str = ""
    added: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    guidance: str = ""
    """Which moves reach the target (:data:`PROFILE_GUIDANCE`); not part of the hash."""
    target_manifold: str = WHOLE_OCCUPIED_MANIFOLD_TARGET
    """What the profile holds a model to: the whole occupied manifold, or a sub-manifold
    (``d_manifold``). It follows from the check list (a sub-manifold profile runs none of
    :data:`WHOLE_MANIFOLD_CHECKS`) and is read by the repair validator and the deck
    check, which relax the occupied-band requirements only for a sub-manifold."""

    @property
    def sub_manifold(self) -> bool:
        """Whether the target is a sub-manifold of the occupied states."""
        return self.target_manifold != WHOLE_OCCUPIED_MANIFOLD_TARGET

    def __post_init__(self) -> None:
        ids = [c.check_id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise GateDefinitionError(f"profile {self.name!r} names a check twice")

    @property
    def check_ids(self) -> tuple[str, ...]:
        return tuple(c.check_id for c in self.checks)

    @property
    def hash(self) -> str:
        """:func:`~.gate.check_list_hash` of the checks: recorded with every verdict."""
        return check_list_hash(self.checks)

    def target_text(
        self,
        species: Sequence[str] | None = None,
        *,
        reference_path: Any = None,
        parent: Any = None,
    ) -> str:
        """One plain paragraph saying what passing this profile means.

        With ``species`` (the structure's), a profile that runs a DMFT-subspace check
        also states this structure's own target subspace, which is part of the
        criterion, and -- for ``dmft_subspace_model_window``, given the run's
        ``reference_path`` (with its target weights) and ``parent`` -- whether its target
        manifold is isolated or entangled and so which of its states are compared;
        every other profile's text is the same with or without them.
        """
        clauses = "; ".join(c.description or c.quantity for c in self.checks)
        lead = f'A construction passes the "{self.name}" gate profile'
        if self.purpose:
            lead += f" (for {self.purpose})"
        changes = []
        if self.added:
            changes.append("adds " + ", ".join(self.added))
        if self.dropped:
            changes.append("drops " + ", ".join(self.dropped))
        if changes and self.base:
            lead += f", which {' and '.join(changes)} relative to {self.base!r},"
        text = (
            f"{lead} when every one of these holds: {clauses}. A check that applies "
            "but could not be evaluated counts against acceptance, the same as a "
            "failure; a check that does not apply to this material (a metal under an "
            "insulator-only criterion, for instance) is withheld and does not count."
        )
        if self.guidance:
            text += " " + self.guidance
        if species is not None and any(
            i in self.check_ids for i in DMFT_SUBSPACE_CHECK_IDS
        ):
            target = target_for_species(species)
            if target is None:
                text += (
                    " This structure is not one the DMFT-subspace criterion covers, so "
                    "that check is withheld."
                )
            else:
                text += (
                    f" For this structure the target subspace is {target.description}."
                )
                if DMFT_SUBSPACE_MODEL.check_id in self.check_ids:
                    text += " " + comparison_text(
                        target, species, reference_path=reference_path, parent=parent
                    )
        return text

    def gate(
        self, backend: Any = None, extra: Sequence[Check] = (), *, reference_path=None
    ) -> CertificationGate:
        """The gate this profile defines, plus a backend's and the caller's checks."""
        contributed: tuple[Check, ...] = ()
        if backend is not None:
            from .backends import get_backend

            contributed = tuple(get_backend(backend).contributed_checks())
        return CertificationGate(
            self.checks,
            (*contributed, *extra),
            profile_name=self.name,
            reference_path=reference_path,
        )


def _lookup(check_id: str) -> Check:
    try:
        return CHECK_REGISTRY[check_id]
    except KeyError:
        raise GateDefinitionError(
            f"unknown check {check_id!r}; registered: {', '.join(sorted(CHECK_REGISTRY))}"
        ) from None


def _named(name: str) -> GateProfile:
    if name in RETIRED_PROFILES:
        ids, purpose, why = RETIRED_PROFILES[name]
        warnings.warn(
            f"gate profile {name!r} is {why}; it is rebuilt here so stored verdicts "
            "can be reproduced (same checks, same hash)",
            GateProfileWarning,
            stacklevel=3,
        )
        return GateProfile(name, tuple(_lookup(i) for i in ids), purpose)
    try:
        ids, purpose = PROFILES[name]
    except KeyError:
        raise GateDefinitionError(
            f"unknown gate profile {name!r}; known: {', '.join(PROFILES)}"
        ) from None
    return GateProfile(
        name,
        tuple(_lookup(i) for i in ids),
        purpose,
        guidance=PROFILE_GUIDANCE.get(name, ""),
        target_manifold=PROFILE_TARGETS.get(name, WHOLE_OCCUPIED_MANIFOLD_TARGET),
    )


def _expand(names: Iterable[str]) -> tuple[str, ...]:
    """Check identifiers, with each :data:`CHECK_GROUPS` name replaced by its members."""
    out: list[str] = []
    for name in names:
        out.extend(CHECK_GROUPS.get(name, (name,)))
    return tuple(out)


def custom_profile(
    base: str | GateProfile = "default",
    add: Iterable[str] = (),
    drop: Iterable[str] = (),
    *,
    name: str | None = None,
) -> GateProfile:
    """A profile derived from ``base`` by adding and dropping registered checks.

    A name in :data:`CHECK_GROUPS` stands for all of its checks (``convergence``).

    Dropping a check of the default set emits :class:`~.gate.GateProfileWarning` and
    nothing else: every check is removable. Adding a check already present, dropping
    one that is absent, or naming an unregistered check is an error, because each would
    leave the caller believing in a profile that is not the one that runs.
    """
    start = get_profile(base)
    add, drop = tuple(add), tuple(drop)
    if not add and not drop:
        return start
    ids = list(start.check_ids)
    for check_id in _expand(drop):
        if check_id not in ids:
            raise GateDefinitionError(
                f"cannot drop {check_id!r}: profile {start.name!r} does not run it"
            )
        if check_id in DEFAULT_CHECK_IDS:
            warnings.warn(
                f"dropping {check_id!r}, a check of the default gate profile: 'passed' "
                "under this profile no longer means what it means under 'default' "
                "(the verdicts record the changed check-list hash)",
                GateProfileWarning,
                stacklevel=2,
            )
        ids.remove(check_id)
    for check_id in _expand(add):
        _lookup(check_id)
        if check_id in ids:
            raise GateDefinitionError(
                f"cannot add {check_id!r}: profile {start.name!r} already runs it"
            )
        ids.append(check_id)
    label = name or (
        start.name + "".join(f"+{i}" for i in add) + "".join(f"-{i}" for i in drop)
    )
    return GateProfile(
        label,
        tuple(_lookup(i) for i in ids),
        start.purpose,
        base=start.name,
        added=add,
        dropped=drop,
        guidance=start.guidance,
        target_manifold=(
            start.target_manifold
            if not set(WHOLE_MANIFOLD_CHECKS) & set(ids)
            else WHOLE_OCCUPIED_MANIFOLD_TARGET
        ),
    )


def get_profile(profile: str | GateProfile | None = "default") -> GateProfile:
    """A named profile, or ``profile`` itself when it already is one."""
    if profile is None:
        profile = "default"
    if isinstance(profile, GateProfile):
        return profile
    return _named(str(profile))


def build_gate(
    backend: Any = None,
    profile: str | GateProfile | None = "default",
    extra: Sequence[Check] = (),
    *,
    reference_path=None,
) -> CertificationGate:
    """The gate for ``profile`` with ``backend``'s contributed checks and ``extra``.

    ``reference_path`` (a :class:`~.criteria.ReferenceBandPath`) is what the acceptance
    criteria are scored against; without it they are non-evaluable, so a profile that
    includes one cannot certify.
    """
    return get_profile(profile).gate(backend, extra, reference_path=reference_path)
