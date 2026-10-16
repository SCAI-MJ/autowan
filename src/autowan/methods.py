"""A deterministic starting deck for every (method, engine) pair.

When a repair switches method family it does not start from nothing, and it does not
start from a deck the model typed: it starts from the published or decided protocol for
that family on that engine, built from this material's own parent and frontier deck.
The deck is handed to the model as information it may edit (``method_defaults``), and
a ``switch_method`` action builds it and applies the model's edits on top.

=======  ======================================================================
QE PDWF  the frontier itself: SSSP pseudo-atomic orbitals, ``dis_proj_min/max``
         0.01/0.95 with ``dis_froz_proj``, the paper's window rule
         (``dis_froz_max`` = E_F + 2 eV for a metal, CBM + 2 eV for an insulator;
         Qiao, Pizzi & Marzari 2023/2025)
QE SCDM  Vitale et al. 2020 (npj Comput. Mater. 6, 66): erfc weight with
         mu = mu_fit - 3 sigma_fit and sigma = sigma_fit, fitted to the
         projectability of the states on the pseudo-atomic orbitals; no windows,
         ``dis_num_iter = 0`` (SCDM in lieu of disentanglement)
QE SMV   the frontier's pseudo-atomic projections with an energy window: the
         projectability window removed, ``dis_froz_max`` by the paper's rule
         capped at the frozen capacity
VASP     SCDM: the frontier, as the Varrassi et al. (2025) GW workflow's Wannier
         step runs it
VASP     SMV: semicore excluded by our POTCAR count, one hydrogenic function per
         valence channel, energy window only
=======  ======================================================================
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .families import FRONTIER, Method, as_method, available
from .inputs import (
    ConstructionInputs,
    InputError,
    _excluded_from_text,
    assignments,
    deck_method,
    method_of,
    pao_excluded,
    pao_layout,
    pao_subset_of,
    read_construction,
    rewrite_pw2wan_text,
    rewrite_win_text,
    to_relative,
)

#: The paper's frozen-window margin above E_F (metal) or the CBM (insulator).
WINDOW_MARGIN_EV = 2.0
#: PDWF thresholds of the frontier (aiida-wannier90-workflows' first attempt).
PDWF_THRESHOLDS = (0.01, 0.95)
#: Vitale et al. 2020: mu sits this many fitted widths below the fitted midpoint.
SCDM_MU_OFFSET_SIGMA = 3.0


class StartingDeckUnavailable(InputError):
    """No starting deck can be built for this (method, engine) on this material."""


@dataclass
class DeckContext:
    """What a starting deck is built from: the parent and the run's frontier deck."""

    code: str
    run_dir: Path
    parent: Any
    parent_source: Path
    frontier: ConstructionInputs
    spin_index: int = 0
    _cache: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StartingDeck:
    method: Method
    code: str
    source: str
    """The protocol it follows, with its citation."""
    win_text: str
    pw2wan_text: str | None
    values: dict
    """Model-facing knob values (energies relative to E_F) and the deck's counts."""
    notes: tuple[str, ...] = ()

    def write(self, run_dir: Path, seed: str) -> ConstructionInputs:
        run_dir = Path(run_dir)
        (run_dir / f"{seed}.win").write_text(self.win_text)
        if self.pw2wan_text is not None:
            (run_dir / f"{seed}.pw2wan").write_text(
                rewrite_pw2wan_text(self.pw2wan_text, {}, seed)
            )
        return read_construction(run_dir, seed)

    def summary(self) -> dict:
        return {
            "method": self.method.value,
            "code": self.code,
            "source": self.source,
            "values": self.values,
            "notes": list(self.notes),
        }


def window_rule_top(parent) -> tuple[float, str]:
    """The paper's frozen-window top: E_F + 2 eV (metal) or CBM + 2 eV (insulator)."""
    e = parent.eigenvalues_ev
    ef = parent.fermi_energy_ev
    if parent.is_metallic():
        return ef + WINDOW_MARGIN_EV, "Fermi level + 2 eV (metal)"
    above = e[e > ef]
    if not above.size:
        return ef + WINDOW_MARGIN_EV, "Fermi level + 2 eV (no conduction band computed)"
    return (
        float(above.min()) + WINDOW_MARGIN_EV,
        "conduction-band minimum + 2 eV (insulator)",
    )


def kept_bands(parent, win_text: str, num_bands: int | None) -> np.ndarray:
    """Parent energies of the bands the deck keeps: (spin, k, band)."""
    excluded = set(_excluded_from_text(win_text))
    keep = [b for b in range(parent.n_bands) if (b + 1) not in excluded]
    if num_bands is not None:
        keep = keep[:num_bands]
    return parent.eigenvalues_ev[..., keep]


def frozen_cap(bands: np.ndarray, num_wann: int) -> float | None:
    """The lowest energy at which some k-point or spin would hold num_wann + 1 states."""
    if bands.shape[-1] <= num_wann:
        return None
    return float(np.sort(bands, axis=-1)[..., num_wann].min())


def _int(value) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _relative(values: dict, fermi: float) -> dict:
    return {k: to_relative(k, v, fermi) for k, v in values.items()}


# ---------------------------------------------------------------------------
# Quantum ESPRESSO
# ---------------------------------------------------------------------------


def _qe_frontier_texts(ctx: DeckContext) -> tuple[str, str]:
    base = ctx.frontier
    if base.pw2wan is None:
        raise StartingDeckUnavailable(
            "the run's first deck has no overlap input to take the pseudo-atomic "
            "orbitals from"
        )
    pw_text = base.pw2wan.read_text()
    if not assignments(pw_text).get("atom_proj", "").lower().startswith((".t", "t")):
        raise StartingDeckUnavailable(
            "the run's first deck does not project on the pseudopotential's atomic "
            "orbitals, so there is no PAO set to start from"
        )
    return base.win.read_text(), pw_text


def _pao_values(ctx: DeckContext, win_text: str, pw_text: str) -> dict:
    try:
        layout = pao_layout(ctx.run_dir, win_text, pw_text)
    except InputError as exc:
        return {"pao_subset": f"not readable: {exc}"}
    kept = pao_subset_of(layout, pao_excluded(pw_text))
    every = pao_subset_of(layout, ())
    return {
        "pao_subset": [{"species": s, "orbitals": list(v)} for s, v in kept.items()],
        "pao_available": {s: list(v) for s, v in every.items()},
    }


def qe_pdwf(ctx: DeckContext) -> StartingDeck:
    win_text, pw_text = _qe_frontier_texts(ctx)
    note = ()
    if deck_method(win_text, pw_text) is not Method.PDWF:
        top, rule = window_rule_top(ctx.parent)
        win_text = rewrite_win_text(
            win_text,
            {
                "auto_projections": ".true.",
                "dis_proj_min": PDWF_THRESHOLDS[0],
                "dis_proj_max": PDWF_THRESHOLDS[1],
                "dis_froz_proj": ".true.",
                "dis_froz_max": top,
            },
            ctx.frontier.seed,
            remove=("dis_win_min", "dis_win_max", "dis_froz_min"),
            projections=False,
            note=f"PDWF starting deck: frontier thresholds, dis_froz_max = {rule}",
        )
        note = (f"built from the first deck's PAOs; dis_froz_max = {rule}",)
    else:
        note = ("the frontier deck of this run, unchanged",)
    values = assignments(win_text)
    shown = {
        k: values.get(k)
        for k in (
            "num_wann",
            "num_bands",
            "exclude_bands",
            "dis_proj_min",
            "dis_proj_max",
            "dis_froz_max",
        )
    }
    return StartingDeck(
        Method.PDWF,
        "qe",
        "the QE frontier: PDWF on SSSP pseudo-atomic orbitals, thresholds 0.01/0.95, "
        "dis_froz_max by the paper's rule (Qiao, Pizzi & Marzari 2023, "
        "arXiv:2303.07877; 2025, arXiv:2507.06840)",
        win_text,
        pw_text,
        {
            **_relative(shown, ctx.parent.fermi_energy_ev),
            **_pao_values(ctx, win_text, pw_text),
        },
        note,
    )


def qe_smv(ctx: DeckContext) -> StartingDeck:
    win_text, pw_text = _qe_frontier_texts(ctx)
    values = assignments(win_text)
    num_wann = _int(values.get("num_wann"))
    if num_wann is None:
        raise StartingDeckUnavailable("the first deck declares no num_wann")
    top, rule = window_rule_top(ctx.parent)
    bands = kept_bands(ctx.parent, win_text, _int(values.get("num_bands")))
    cap = frozen_cap(bands, num_wann)
    capped = top
    if cap is not None and top >= cap:
        capped = min(cap - 1e-3, float(np.nextafter(cap, -np.inf)))
    smv_win = rewrite_win_text(
        win_text,
        {"auto_projections": ".true.", "dis_froz_max": round(capped, 6)},
        ctx.frontier.seed,
        remove=(
            "dis_proj_min",
            "dis_proj_max",
            "dis_proj_auto",
            "dis_froz_proj",
            "dis_froz_min",
            "dis_win_min",
            "dis_win_max",
        ),
        note=(
            f"SMV starting deck: the frontier's PAO projections, energy window only; "
            f"dis_froz_max = {rule}"
            + (
                ""
                if capped == top
                else f", capped below {cap:.6g} eV (frozen capacity)"
            )
        ),
    )
    shown = {
        k: assignments(smv_win).get(k)
        for k in ("num_wann", "num_bands", "exclude_bands", "dis_froz_max")
    }
    notes = [f"dis_froz_max = {rule}"]
    if capped != top:
        notes.append(
            f"capped below {cap:.6g} eV so no k-point holds more frozen states than "
            f"{num_wann} functions"
        )
    return StartingDeck(
        Method.SMV,
        "qe",
        "SMV with the frontier's pseudo-atomic projections and an energy frozen window "
        "(Souza, Marzari & Vanderbilt 2001), the window placed by the paper's rule",
        smv_win,
        pw_text,
        {
            **_relative(shown, ctx.parent.fermi_energy_ev),
            **_pao_values(ctx, win_text, pw_text),
        },
        tuple(notes),
    )


def scdm_fit(ctx: DeckContext):
    """Vitale's erfc fit of projectability on the PAOs, and where it was read from."""
    if "scdm_fit" in ctx._cache:
        return ctx._cache["scdm_fit"]
    from .overlaps import OverlapReadError, read_projections
    from .projectability import (
        ProjectabilityError,
        fit_scdm_parameters,
        read_atomic_proj,
    )

    result, source, errors = None, None, []
    for candidate in (
        ctx.parent_source.parent / "atomic_proj.xml",
        *sorted(ctx.parent_source.parent.glob("*.save/atomic_proj.xml")),
    ):
        if candidate.is_file():
            try:
                proj = read_atomic_proj(
                    candidate, fermi_energy_ev=ctx.parent.fermi_energy_ev
                )
                result = fit_scdm_parameters(
                    proj.energies_ev,
                    proj.projectability,
                    mu_offset_sigma=SCDM_MU_OFFSET_SIGMA,
                )
                source = f"projwfc.x projectability ({candidate.name})"
                break
            except ProjectabilityError as exc:
                errors.append(str(exc))
    if result is None:
        seed = ctx.frontier.seed
        try:
            proj = read_projections(ctx.run_dir, seed)
            if proj.energies_ev is None:
                raise OverlapReadError(f"no {seed}.eig")
            result = fit_scdm_parameters(
                proj.energies_ev.ravel(),
                proj.projectability.ravel(),
                mu_offset_sigma=SCDM_MU_OFFSET_SIGMA,
            )
            source = (
                f"the frontier's own overlaps ({seed}.amn/.eig): projectability on the "
                f"kept pseudo-atomic orbitals, no projwfc.x run being available"
            )
        except (OverlapReadError, ProjectabilityError) as exc:
            errors.append(str(exc))
    ctx._cache["scdm_fit"] = (result, source, errors)
    return ctx._cache["scdm_fit"]


def qe_scdm(ctx: DeckContext) -> StartingDeck:
    win_text, pw_text = _qe_frontier_texts(ctx)
    fit, source, errors = scdm_fit(ctx)
    if fit is None:
        raise StartingDeckUnavailable(
            "the Vitale 2020 fit needs the projectability of the states on the "
            "pseudo-atomic orbitals, and none could be read: " + "; ".join(errors)
        )
    scdm_win = rewrite_win_text(
        win_text,
        {"auto_projections": ".true.", "dis_num_iter": 0},
        ctx.frontier.seed,
        remove=(
            "dis_proj_min",
            "dis_proj_max",
            "dis_proj_auto",
            "dis_froz_proj",
            "dis_froz_min",
            "dis_froz_max",
            "dis_win_min",
            "dis_win_max",
        ),
        projections=False,
        note=(
            f"SCDM starting deck (Vitale et al. 2020): erfc fit mu_fit = "
            f"{fit.mu_fit:.4f} eV, sigma = {fit.sigma:.4f} eV, mu = mu_fit - 3 sigma; "
            f"fit from {source}"
        ),
    )
    scdm_pw = rewrite_pw2wan_text(
        pw_text,
        {
            "scdm_proj": True,
            "scdm_entanglement": "erfc",
            "scdm_mu": round(fit.mu, 4),
            "scdm_sigma": round(fit.sigma, 4),
        },
        ctx.frontier.seed,
        remove=(
            "atom_proj",
            "atom_proj_ext",
            "atom_proj_dir",
            "atom_proj_ortho",
            "atom_proj_exclude",
            "atom_proj_frozen",
        ),
    )
    fermi = ctx.parent.fermi_energy_ev
    values = assignments(scdm_win)
    return StartingDeck(
        Method.SCDM,
        "qe",
        "SCDM with the Vitale et al. 2020 protocol (npj Comput. Mater. 6, 66): "
        "mu = mu_fit - 3 sigma_fit, sigma = sigma_fit from an erfc fit of the "
        "projectability on the pseudo-atomic orbitals; no windows, dis_num_iter = 0",
        scdm_win,
        scdm_pw,
        {
            "scdm_mu": round(fit.mu - fermi, 4),
            "scdm_sigma": round(fit.sigma, 4),
            "num_wann": values.get("num_wann"),
            "num_bands": values.get("num_bands"),
            "exclude_bands": values.get("exclude_bands"),
            "dis_num_iter": 0,
        },
        (f"fit from {source}; residual {fit.residual:.3f} over {fit.n_states} states",),
    )


# ---------------------------------------------------------------------------
# VASP
# ---------------------------------------------------------------------------


def _vasp_build(ctx: DeckContext, builder) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as scratch:
        inputs = builder(ctx.parent_source, Path(scratch), "start")
        return inputs.win.read_text(), dict(inputs.values)


def vasp_scdm(ctx: DeckContext) -> StartingDeck:
    from .backends.vasp.inputs import is_frontier_overlap_step, scdm_frontier_deck

    frontier_text = ctx.frontier.win.read_text()
    if is_frontier_overlap_step(frontier_text):
        text, note = frontier_text, "the frontier deck of this run, unchanged"
    else:
        try:
            text, _ = _vasp_build(ctx, scdm_frontier_deck)
        except InputError as exc:
            raise StartingDeckUnavailable(str(exc)) from exc
        note = "built from the parent by the frontier protocol"
    values = assignments(text)
    return StartingDeck(
        Method.SCDM,
        "vasp",
        "the VASP frontier: SCDM as the Varrassi et al. (2025, arXiv:2410.15948) GW "
        "workflow's Wannier step runs it -- ALGO = None, NELM = 1, ISTART = 1, "
        "LWANNIER90_RUN, LSCDM with VASP's default cutoff, ISMEAR = 0, SIGMA = 0.02, "
        "LORBIT = 11, NUM_WANN unset, no windows",
        text,
        None,
        {
            k: values.get(k)
            for k in (
                "num_wann",
                "num_bands",
                "exclude_bands",
                "dis_num_iter",
                "num_iter",
            )
        }
        | {
            "scdm_mu": "VASP default (0.8 num_wann filling)",
            "scdm_sigma": "VASP default (1 eV)",
        },
        (note,),
    )


def vasp_smv(ctx: DeckContext) -> StartingDeck:
    from .backends.vasp.inputs import semicore_excluded_deck

    try:
        text, values = _vasp_build(ctx, semicore_excluded_deck)
    except InputError as exc:
        raise StartingDeckUnavailable(str(exc)) from exc
    return StartingDeck(
        Method.SMV,
        "vasp",
        "autowan's SMV deck on VASP: semicore bands excluded (counted from our "
        "POTCARs, checked against band character), one hydrogenic function per valence "
        "channel, frozen window 2 eV above the highest occupied state, capped "
        "",
        text,
        None,
        _relative(
            {
                k: values.get(k)
                for k in ("num_wann", "num_bands", "dis_froz_max", "projectors")
            }
            | {"exclude_bands": assignments(text).get("exclude_bands")},
            ctx.parent.fermi_energy_ev,
        ),
        tuple(line[2:] for line in text.splitlines() if line.startswith("!   ")),
    )


BUILDERS = {
    ("qe", Method.PDWF): qe_pdwf,
    ("qe", Method.SMV): qe_smv,
    ("qe", Method.SCDM): qe_scdm,
    ("vasp", Method.SCDM): vasp_scdm,
    ("vasp", Method.SMV): vasp_smv,
}


def starting_deck(method, ctx: DeckContext) -> StartingDeck:
    method = as_method(method)
    if method not in available(ctx.code):
        from .families import UNAVAILABLE_REASONS

        raise StartingDeckUnavailable(
            UNAVAILABLE_REASONS.get(
                (ctx.code, method), f"{ctx.code} offers no {method.value} route"
            )
        )
    key = ("deck", method)
    if key not in ctx._cache:
        try:
            ctx._cache[key] = BUILDERS[(ctx.code, method)](ctx)
        except StartingDeckUnavailable as exc:
            ctx._cache[key] = exc
    found = ctx._cache[key]
    if isinstance(found, Exception):
        raise found
    return found


def frontier_method(code: str) -> Method:
    return FRONTIER[code]


__all__ = [
    "DeckContext",
    "StartingDeck",
    "StartingDeckUnavailable",
    "starting_deck",
    "window_rule_top",
    "kept_bands",
    "frozen_cap",
    "method_of",
]


def derive_deck(
    *,
    backend,
    source: ConstructionInputs,
    ctx: DeckContext,
    run_dir: Path,
    new_seed: str,
    changes: dict,
    method=None,
    fermi_energy_ev: float | None = None,
    species_counts: dict | None = None,
) -> ConstructionInputs:
    """The deck an action produces: an edit in the family, or a switch to another.

    A switch writes the target family's starting deck (``<seed>_start``) and applies
    the model's edits on top; with no edits the starting deck is the next deck. Within
    the family it is the backend's own writer, which refuses another family's knob.
    """
    if method is not None and as_method(method) is not method_of(source):
        start = starting_deck(method, ctx)
        if not changes:
            return start.write(run_dir, new_seed)
        base = start.write(run_dir, f"{new_seed}_start")
        return backend.write_derived(
            base,
            run_dir,
            new_seed,
            resolve_semicore(base, ctx, changes),
            fermi_energy_ev=fermi_energy_ev,
            species_counts=species_counts,
        )
    return backend.write_derived(
        source,
        run_dir,
        new_seed,
        resolve_semicore(source, ctx, changes),
        fermi_energy_ev=fermi_energy_ev,
        species_counts=species_counts,
    )


def resolve_semicore(deck: ConstructionInputs, ctx: DeckContext, changes: dict) -> dict:
    """``changes`` with a ``semicore_exclusion`` shell list resolved against this deck.

    The shell list names orbitals; what it means in bands and in projectors depends on
    the deck (its PAO list, its projections, its bottom and top exclusions) and on the
    parent (its band count, its pseudopotentials), so it is resolved here, where both are
    known, into the plan the deck writers apply (:mod:`.semicore`).
    """
    if "semicore_exclusion" not in changes:
        return changes
    from .semicore import SemicorePlan, catalogue, resolve

    value = changes["semicore_exclusion"]
    if isinstance(value, SemicorePlan):
        return changes
    pw_text = (
        deck.pw2wan.read_text()
        if deck.pw2wan is not None and deck.pw2wan.is_file()
        else None
    )
    cat = catalogue(deck.win.read_text(), pw_text, deck.win.parent, ctx)
    return {**changes, "semicore_exclusion": resolve(cat, value)}


def probe_directory(run_dir: Path, source: ConstructionInputs, scratch: Path) -> None:
    """Make ``scratch`` a place a deck derived from ``source`` can be written and read.

    The deck files are copied; the external projector directory a PDWF deck names is
    linked, because the PAO numbering is read from it.
    """
    import shutil

    for path in (source.win, source.pw2wan):
        if path is not None and path.is_file():
            shutil.copy(path, scratch / path.name)
    if source.pw2wan is not None and source.pw2wan.is_file():
        directory = assignments(source.pw2wan.read_text()).get("atom_proj_dir")
        if directory:
            name = directory.strip("'\"").rstrip("/")
            target = Path(run_dir) / name
            if target.exists() and not (scratch / name).exists():
                (scratch / name).parent.mkdir(parents=True, exist_ok=True)
                (scratch / name).symlink_to(target.resolve())
