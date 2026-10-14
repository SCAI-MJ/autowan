"""The three ways of choosing a Wannier subspace, and which knobs belong to each.

A construction is built by one *method family*, and the family is a first-class
choice rather than something a parameter edit can change by accident:

* **SMV** -- energy-window disentanglement (Souza, Marzari & Vanderbilt 2001) seeded by
  trial projections. Its knobs are the outer and frozen windows and the projector set.
* **PDWF** -- projectability-disentangled Wannier functions (Qiao, Pizzi & Marzari
  2023): the pseudo-atomic orbitals (PAOs) of the pseudopotential seed the functions,
  and a state is frozen or dropped by how well those PAOs describe it. The published
  method is projectability *and* energy disentanglement (PD+ED): a state is frozen when
  its projectability reaches ``dis_proj_max`` or its energy lies below ``dis_froz_max``.
  Its knobs are therefore the PAO subset, the two projectability thresholds and the
  frozen window's top; ``dis_froz_proj`` stays on. (No frontier deck sets an outer
  window, so ``dis_win_max`` is not PDWF's.)
* **SCDM** -- selected columns of the density matrix (Damle, Lin & Ying 2015/2017;
  Vitale et al. 2020), weighted by ``f(e; mu, sigma)``. Its knobs are mu and sigma.

The family of a deck is read from the deck itself (:func:`.inputs.deck_method`). Moving
from one family to another is the ``switch_method`` action, which is logged with where
it came from, where it went and why; no edit inside a family can reach another
family's knobs, and the deck writer refuses one that would.

PDWF is not offered on VASP: VASP's hydrogenic and LOCPROJ projections are not
orthonormal, their projectability reaches values far above one, and a projectability
threshold on them selects nothing meaningful (2026-09-30; autowan item 4 decision 1).
"""

from __future__ import annotations

from enum import Enum


class Method(str, Enum):
    SMV = "smv"
    PDWF = "pdwf"
    SCDM = "scdm"


#: Families each engine can build. Ordered: the first is the engine's frontier family.
AVAILABLE: dict[str, tuple[Method, ...]] = {
    "qe": (Method.PDWF, Method.SCDM, Method.SMV),
    "vasp": (Method.SCDM, Method.SMV),
}

#: The frontier (attempt 0) family on each engine.
FRONTIER: dict[str, Method] = {"qe": Method.PDWF, "vasp": Method.SCDM}

UNAVAILABLE_REASONS: dict[tuple[str, Method], str] = {
    ("vasp", Method.PDWF): (
        "PDWF needs projectors whose projectability stays at or below one; VASP's "
        "hydrogenic and LOCPROJ projections are not orthonormal and reach well above "
        "one, so a projectability window on them freezes or drops the wrong states"
    ),
}

#: Knobs every family shares: they act on the band pool or the minimisers, not on how
#: the subspace is chosen.
COMMON_KNOBS: tuple[str, ...] = (
    "num_bands",
    "semicore_exclusion",
    "num_iter",
    "dis_num_iter",
    "dis_mix_ratio",
    "conv_window",
    "trial_step",
)

FAMILY_KNOBS: dict[Method, tuple[str, ...]] = {
    Method.SMV: (
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
        "projectors",
    ),
    # PD+ED: the energy half of the published method is its frozen window's top.
    Method.PDWF: ("pao_subset", "dis_proj_min", "dis_proj_max", "dis_froz_max"),
    Method.SCDM: ("scdm_mu", "scdm_sigma"),
}

DESCRIPTIONS: dict[Method, str] = {
    Method.SMV: (
        "energy-window disentanglement seeded by trial projections: the frozen window "
        "fixes which states are reproduced exactly, the outer window bounds the pool, "
        "and the projector set decides what the functions can become"
    ),
    Method.PDWF: (
        "projectability plus energy disentanglement on the pseudopotential's own "
        "atomic orbitals: a state is frozen when the kept orbitals describe it well "
        "(p >= dis_proj_max) or its energy lies below dis_froz_max, and states poorly "
        "described (p < dis_proj_min) are dropped"
    ),
    Method.SCDM: (
        "selected columns of the density matrix with an occupation-like weight "
        "f(e) = erfc((e - mu)/sigma)/2: mu sets where states stop counting, sigma how "
        "softly; no trial orbitals and no energy windows"
    ),
}


#: A shared knob a (engine, family) pair has no route for, with the reason.
KNOB_UNAVAILABLE: dict[tuple[str, Method, str], str] = {
    ("vasp", Method.SCDM, "semicore_exclusion"): (
        "the VASP SCDM frontier builds its overlaps inside VASP for bands 1..NBANDS and "
        "takes num_wann = min(NBANDS, n_pseudo), counting every POTCAR channel, semicore "
        "included; the protocol excludes nothing at the bottom and hands VASP no "
        "exclude_bands, so a bottom exclusion has no tested route there. Its semicore "
        "is already included. To exclude semicore on VASP, switch to SMV"
    ),
}


def knob_unavailable(code: str, method, knob: str) -> str | None:
    """Why ``knob`` has no route on this (engine, family), or ``None`` when it has one."""
    return KNOB_UNAVAILABLE.get((code, as_method(method), knob))


def as_method(value) -> Method:
    try:
        return Method(str(getattr(value, "value", value)).lower())
    except ValueError as exc:
        known = ", ".join(m.value for m in Method)
        raise ValueError(f"unknown method {value!r}; known: {known}") from exc


def available(code: str) -> tuple[Method, ...]:
    return AVAILABLE.get(code, ())


def knobs_for(method) -> tuple[str, ...]:
    """Every knob an action under ``method`` may set."""
    return COMMON_KNOBS + FAMILY_KNOBS[as_method(method)]


def families_of(parameter: str) -> tuple[Method, ...]:
    """Every family that owns a knob; empty for a common knob or an unknown name.

    A knob may belong to more than one family: ``dis_froz_max`` is SMV's frozen window
    and the energy half of PDWF's PD+ED.
    """
    return tuple(m for m, knobs in FAMILY_KNOBS.items() if parameter in knobs)


def family_of(parameter: str) -> Method | None:
    """The first family that owns a knob, or ``None`` for a common knob."""
    owners = families_of(parameter)
    return owners[0] if owners else None
