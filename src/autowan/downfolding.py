"""The d-only downfolding criterion: a minimal correlated subspace for DMFT, cRPA, DFT+U.

Since 2026-10-09, the third acceptance criterion is no longer
"the conduction band to CBM + 2 eV" but **a d-only downfolded model** -- the minimal
correlated subspace a DMFT, cRPA or DFT+U calculation is built on. The check
``d_only_downfolding`` applies only when the structure contains at least one
transition-metal atom (3d, 4d or 5d; lanthanum counts, and is flagged); otherwise it is
withheld. It passes when all three of these hold:

(i) **Count and projectors.** ``num_wann`` equals 5 x N_TM per spin channel (10 x N_TM
    for spinor functions), and the construction's projectors are d orbitals on the
    transition-metal sites only, read from whatever the family uses: the projections
    block (SMV), the kept pseudo-atomic orbitals (PDWF, and QE SMV started from them),
    or, for SCDM, which has no projectors, the count alone.

(ii) **Band reproduction near E_F.** On the separate reference path, the Wannier bands
    reproduce the DFT bands they correspond to within E_F +- 1 eV, with E_F the scf Fermi
    level (as for ``band_interpolation_eta2``). The distance is the paper's eta with the
    occupation factor replaced by a soft window,

        w_X = f(e_X; E_F + 1, sigma) (1 - f(e_X; E_F - 1, sigma)),   X = DFT, Wannier,
        w   = sqrt(w_DFT w_Wan),   eta = sqrt(sum w (e_DFT - e_Wan)^2 / sum w),

    sigma = 0.1 eV (:func:`~.band_distance_eta.band_distance_eta` with ``nu = 1`` and
    ``lower_edge = E_F - 1``), and passes at <= 20 meV.

(iii) **Assignability.** Every function's centre lies within half the structure's
    nearest-neighbour distance of a transition-metal site: the pre-profile
    ``centres_sit_on_atomic_sites`` rule restricted to transition-metal sites.

**Which DFT band each Wannier band is compared with.** A d-only model has fewer bands
than the DFT reference wherever ligand bands share the window, so the index matching of
eta (Wannier band j against DFT band j after the exclusions) compares the wrong bands:
one ligand band below the d manifold shifts every pair by one. The rule here is, at
every path k-point independently, the **order-preserving assignment of minimal squared
energy distance** (:func:`match_bands`): the model's bands, ascending, are assigned to
an increasing subsequence of the DFT bands, ascending, minimising sum_j (e_Wan,j -
e_DFT,pi(j))^2. Why this rule:

* It is the global optimum of the full assignment problem, not a heuristic. For a
  convex cost of the energy difference on the line, an optimal one-to-one assignment
  can always be taken order-preserving (two crossed pairs can be uncrossed without
  increasing the cost), so the dynamic programme over increasing subsequences returns
  the same total cost as the Hungarian algorithm over all injections -- the tests check
  this on random spectra.
* It is one-to-one: two model bands can never borrow the same DFT band, which a
  per-band nearest-energy rule allows exactly where it matters (a model band that has
  drifted into a gap would be credited to whichever DFT band is closest).
* It leaves every unmatched DFT band out of the distance: the ligand bands a d-only
  model was never asked to represent are skipped wherever they sit, below, above or
  between the d bands, and however they cross them along the path.
* It is deterministic: ties go to the lower DFT band index, and nothing is random.
* It reads only the two sets of energies. The path reference is a band-structure run
  that records energies and no orbital character, so a character weight is not
  available there; the matching therefore never consults the deck either, and a deck's
  ``exclude_bands`` cannot steer which reference bands the model is compared with.

What the rule cannot do is tell a ligand band from a d band of the same energy. A
model band that has wandered onto a ligand band is matched to it and judged against it.
The record therefore states which DFT bands were matched (per Wannier band, the modal
assignment, and the bands compared inside the window) and which DFT bands inside the
window were left unmatched, so the reader can see what the verdict was made against;
the index-matched window eta is recorded beside it as a diagnostic.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from .band_distance_eta import ETA_SIGMA_EV, band_distance_eta
from .criteria import (
    PASS_AT_20_MEV,
    _criterion_fermi,
    _exclude,
    _finite,
    _model,
    _reference,
)
from .gate import (
    Check,
    CheckOutcome,
    Construction,
    NonEvaluable,
    NotApplicable,
    ScaleAwareThreshold,
    _nearest_neighbour_distance_ang,
)

# ---------------------------------------------------------------------------
# which atoms are transition metals
# ---------------------------------------------------------------------------

#: Every element symbol, for reading a species label ("Ni2", "Fe_up", "Co").
ELEMENTS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
    Mc Lv Ts Og""".split()
)

#: The transition metals whose d shell a downfolded model is built on: groups 3 to 11 of
#: the 3d, 4d and 5d rows, with lanthanum as the 5d row's group-3 member.
TRANSITION_METALS_3D = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")
TRANSITION_METALS_4D = ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag")
TRANSITION_METALS_5D = ("La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au")
TRANSITION_METALS = frozenset(
    TRANSITION_METALS_3D + TRANSITION_METALS_4D + TRANSITION_METALS_5D
)

#: Transition metals counted with a caveat the record carries.
FLAGGED_TRANSITION_METALS = {
    "La": "lanthanum is counted as a 5d transition metal (5d1 6s2; "
    "2026-10-09) and flagged: its 4f shell lies just above E_F and a d-only model of it "
    "leaves the 4f states out by construction",
}

#: Elements with a d shell that are deliberately not counted, and why.
NOT_COUNTED = {
    symbol: "group 12: the d shell is full in the atom and in every common cation, so "
    "there is no partly filled d manifold to downfold onto (IUPAC's definition of a "
    "transition element excludes the group for the same reason)"
    for symbol in ("Zn", "Cd", "Hg")
}

#: Functions per transition-metal site and spin channel: one full d shell.
D_FUNCTIONS_PER_SITE = 5

#: The half-width of the Fermi window criterion (ii) compares bands in, in eV.
WINDOW_HALF_WIDTH_EV = 1.0
WINDOW_CALIBRATION = (
    "E_F +- 1 eV: the energy range a d-only model for DMFT, cRPA or DFT+U is asked to "
    "reproduce, with the paper's sigma = 0.1 eV soft edges"
)

#: Criterion (iii): a centre within half the nearest-neighbour distance of a TM site.
CENTRES_ON_TM_SITES = ScaleAwareThreshold(
    0.5,
    "nearest_neighbour_distance_ang",
    calibration="half the nearest-neighbour distance: beyond it a centre is closer to a "
    "different site than to its own, so the function cannot be assigned to a "
    "transition-metal site. The pre-profile centres_sit_on_atomic_sites rule, restricted "
    "to transition-metal sites. Geometric, not fitted.",
)

_LABEL = re.compile(r"([A-Za-z])([a-z]?)")


def element_of(label: str | None) -> str | None:
    """The element a species label names ("Ni2" -> Ni, "Fe_up" -> Fe, "O1" -> O)."""
    if not label:
        return None
    text = str(label).strip()
    match = _LABEL.match(text)
    if match is None:
        return None
    first = match.group(1).upper()
    two = first + match.group(2)
    if match.group(2) and two in ELEMENTS:
        return two
    return first if first in ELEMENTS else None


def transition_metal_sites(c: Construction) -> list[dict[str, Any]]:
    """The structure's transition-metal atoms: index (1-based), label, element."""
    species = getattr(c.parent, "species", None)
    if species is None or len(species) == 0:
        raise NonEvaluable(f"{c.construction_id}: the parent records no species")
    sites = []
    for i, label in enumerate(species):
        element = element_of(label)
        if element in TRANSITION_METALS:
            sites.append({"atom": i + 1, "label": str(label), "element": element})
    return sites


# ---------------------------------------------------------------------------
# (ii) the band matching
# ---------------------------------------------------------------------------


def match_bands(model_ev, dft_ev) -> np.ndarray:
    """The order-preserving, minimal-squared-distance assignment, k-point by k-point.

    ``model_ev`` is ``(n_k, n_w)`` and ``dft_ev`` ``(n_k, n_b)`` with ``n_b >= n_w``.
    Returns ``(n_k, n_w)`` 0-based DFT band indices (columns of ``dft_ev``) such that,
    at every k, the model's bands in ascending order go to DFT bands in ascending order
    of energy, one to one, with the least sum of squared differences. Ties go to the
    lowest DFT index. See the module docstring for why this rule.
    """
    model = np.asarray(model_ev, dtype=float)
    dft = np.asarray(dft_ev, dtype=float)
    if model.ndim != 2 or dft.ndim != 2 or model.shape[0] != dft.shape[0]:
        raise ValueError(
            "model and DFT bands must be (n_k, n_bands) on the same k-list"
        )
    n_k, n_w = model.shape
    n_b = dft.shape[1]
    if n_b < n_w:
        raise ValueError(
            f"the reference has {n_b} bands, fewer than the model's {n_w}: no "
            "one-to-one assignment exists"
        )
    out = np.empty((n_k, n_w), dtype=int)
    positions = np.arange(n_b)
    for k in range(n_k):
        w_order = np.argsort(model[k], kind="stable")
        d_order = np.argsort(dft[k], kind="stable")
        w = model[k, w_order]
        d = dft[k, d_order]
        cost = (w[:, None] - d[None, :]) ** 2
        total = cost[0].copy()
        back = np.zeros((n_w, n_b), dtype=int)
        for j in range(1, n_w):
            best = np.minimum.accumulate(total)
            fresh = np.concatenate([[True], best[1:] < best[:-1]])
            arg = np.maximum.accumulate(np.where(fresh, positions, -1))
            row = np.full(n_b, np.inf)
            row[1:] = cost[j, 1:] + best[:-1]
            back[j, 1:] = arg[:-1]
            total = row
        chosen = np.empty(n_w, dtype=int)
        chosen[-1] = int(np.argmin(total))
        for j in range(n_w - 1, 0, -1):
            chosen[j - 1] = back[j, chosen[j]]
        out[k, w_order] = d_order[chosen]
    return out


def _matching_summary(
    matched: np.ndarray, dft: np.ndarray, wan: np.ndarray, lo: float, hi: float
) -> dict[str, Any]:
    """Which DFT bands (1-based) the model was compared with, said several ways."""
    n_k, n_w = matched.shape
    one_based = matched + 1
    rows = [tuple(int(x) for x in row) for row in one_based]
    modal, modal_count = Counter(rows).most_common(1)[0]
    pair_dft = np.take_along_axis(dft, matched, axis=1)
    inside = (pair_dft >= lo) & (pair_dft <= hi) & (wan >= lo) & (wan <= hi)
    compared = sorted({int(x) for x in one_based[inside]})
    in_window = (dft >= lo) & (dft <= hi)
    taken = np.zeros_like(in_window)
    np.put_along_axis(taken, matched, True, axis=1)
    unmatched = sorted({int(i) + 1 for i in np.nonzero(in_window & ~taken)[1]})
    per_band = {
        str(j + 1): sorted({int(x) for x in one_based[:, j]}) for j in range(n_w)
    }
    return {
        "matched_dft_bands_modal": list(modal),
        "n_k_matched_as_modal": int(modal_count),
        "n_k_matched_otherwise": int(n_k - modal_count),
        "matched_dft_bands_per_wannier_band": per_band,
        "matched_dft_bands_compared_in_window": compared,
        "n_matched_states_in_window": int(inside.sum()),
        "dft_bands_in_window_unmatched": unmatched,
    }


def fermi_window_eta(c: Construction) -> dict[str, Any]:
    """Criterion (ii): the window eta over matched bands, and how they were matched."""
    ref = _reference(c)
    ef, ef_source = _criterion_fermi(c, ref)
    dft = ref.channel(int(c.inputs.get("spin_index", 0)))
    wan = _model(c).bands(ref.kpoints_frac)
    try:
        matched = match_bands(wan, dft)
    except ValueError as exc:
        raise NonEvaluable(f"{c.construction_id}: {exc}") from exc
    pair_dft = np.take_along_axis(dft, matched, axis=1)
    lo, hi = ef - WINDOW_HALF_WIDTH_EV, ef + WINDOW_HALF_WIDTH_EV
    result = band_distance_eta(
        pair_dft,
        wan,
        ef,
        nu_ev=WINDOW_HALF_WIDTH_EV,
        sigma_ev=ETA_SIGMA_EV,
        lower_edge_ev=lo,
    )
    record: dict[str, Any] = {
        "window_eta_mev": _finite(result.eta_mev),
        "window_eta_max_mev": _finite(result.eta_max_mev),
        "window_ev": [lo, hi],
        "window_definition": "w = sqrt(w_DFT w_Wan), w_X = f(e_X; E_F+1, sigma) "
        "(1 - f(e_X; E_F-1, sigma)), sigma 0.1 eV; bands matched by the "
        "order-preserving minimal-squared-distance assignment per k",
        "fermi_energy_scf_ev": ef,
        "fermi_energy_source": ef_source,
        "n_kpoints": int(dft.shape[0]),
        "n_dft_bands": int(dft.shape[1]),
        "n_wannier_bands": int(wan.shape[1]),
        "reference": ref.source,
    }
    record.update(_matching_summary(matched, dft, wan, lo, hi))
    try:
        indexed = band_distance_eta(
            dft,
            wan,
            ef,
            nu_ev=WINDOW_HALF_WIDTH_EV,
            sigma_ev=ETA_SIGMA_EV,
            exclude_bands=_exclude(c),
            lower_edge_ev=lo,
        ).eta_mev
    except ValueError:
        indexed = float("nan")
    record["window_eta_index_matched_mev"] = _finite(indexed)
    return record


# ---------------------------------------------------------------------------
# (i) count and projectors, (iii) centres
# ---------------------------------------------------------------------------


def _site_of_position(c: Construction, group: dict[str, Any]) -> int | None:
    """The atom (0-based) a ``c=``/``f=`` projection site sits on, if any."""
    lattice = np.asarray(c.parent.lattice_ang, dtype=float)
    frac_atoms = np.asarray(c.parent.positions_frac, dtype=float)
    position = np.asarray(group["position"], dtype=float)
    if group.get("frame") == "fractional":
        frac = position
    else:
        scale = 0.529177210903 if group.get("frame") == "cartesian_bohr" else 1.0
        frac = np.linalg.solve(lattice.T, position * scale)
    delta = frac_atoms - frac
    delta -= np.round(delta)
    distance = np.linalg.norm(delta @ lattice, axis=1)
    best = int(np.argmin(distance))
    # A projection centre written to the deck's own precision sits on its atom.
    return best if distance[best] < 1e-3 else None


def count_and_projectors(
    c: Construction, sites: list[dict[str, Any]]
) -> dict[str, Any]:
    """Criterion (i): the Wannier count, and what its projectors are seeded on."""
    spinor = bool(getattr(c.parent, "spinor", False))
    groups_record = c.inputs.get("projector_groups") or {}
    if groups_record.get("spinors"):
        spinor = True
    components = 2 if spinor else 1
    expected = D_FUNCTIONS_PER_SITE * len(sites) * components
    actual = getattr(c.wannier, "num_wann", None)
    requested = c.inputs.get("num_wann")
    if actual is None:
        actual = requested
    record: dict[str, Any] = {
        "n_tm": len(sites),
        "tm_sites": [f"{s['label']}#{s['atom']}" for s in sites],
        "tm_elements": sorted({s["element"] for s in sites}),
        "spinor": spinor,
        "num_wann_expected": expected,
        "num_wann_actual": None if actual is None else int(actual),
        "num_wann_requested": None if requested is None else int(requested),
        "projector_route": groups_record.get("route", "not recorded"),
        "projector_method": groups_record.get("method"),
    }
    flags = [
        FLAGGED_TRANSITION_METALS[e]
        for e in record["tm_elements"]
        if e in FLAGGED_TRANSITION_METALS
    ]
    if flags:
        record["flags"] = flags
    count_ok = None if actual is None else int(actual) == expected
    record["count_ok"] = count_ok

    route = record["projector_route"]
    projectors_ok: bool | None = None
    if route == "scdm":
        record["projector_note"] = "SCDM has no projectors: the count alone is judged"
    elif groups_record.get("error"):
        record["projector_note"] = f"projectors unreadable: {groups_record['error']}"
    elif not groups_record:
        record["projector_note"] = "the construction's projectors were not recorded"
    else:
        tm_atoms = {s["atom"] - 1 for s in sites}
        species = list(getattr(c.parent, "species", ()))
        described, wrong = [], []
        implied = 0
        for group in groups_record.get("groups", []):
            label = group.get("species")
            if label is None and "position" in group:
                atom = _site_of_position(c, group)
                element = None if atom is None else element_of(species[atom])
                on_tm = atom is not None and atom in tm_atoms
            else:
                element = element_of(label)
                on_tm = element in TRANSITION_METALS
            full_d = (
                group.get("l") == [2]
                and int(group.get("functions_per_site", 0))
                == D_FUNCTIONS_PER_SITE * components
            )
            n_sites = int(group.get("n_sites", 1))
            implied += int(group.get("functions_per_site", 0)) * n_sites
            text = (
                f"{group.get('site')} x{n_sites}: l={group.get('l')}, "
                f"{group.get('functions_per_site')} per site"
            )
            described.append(text)
            if not (on_tm and full_d):
                why = (
                    "not a transition-metal site"
                    if not on_tm
                    else "not one full d shell"
                )
                wrong.append(f"{text} ({element or 'unassigned'}; {why})")
        record["projector_groups"] = described
        record["projector_functions_implied"] = implied
        record["projectors_not_tm_d"] = wrong
        projectors_ok = not wrong and implied == expected
    record["projectors_ok"] = projectors_ok
    return record


def centres_on_tm_sites(c: Construction, sites: list[dict[str, Any]]) -> dict[str, Any]:
    """Criterion (iii): each centre's distance to its nearest transition-metal site."""
    centres = np.asarray(c.require("centres_ang"), dtype=float).reshape(-1, 3)
    lattice = np.asarray(c.parent.lattice_ang, dtype=float)
    frac_atoms = np.asarray(c.parent.positions_frac, dtype=float)
    tm = [s["atom"] - 1 for s in sites]
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    centres_frac = np.linalg.solve(lattice.T, centres.T).T
    distances = np.full((len(centres), len(tm)), np.inf)
    for column, atom in enumerate(tm):
        delta = centres_frac - frac_atoms[atom]
        delta -= np.round(delta)
        images = delta[:, None, :] + shifts[None, :, :]
        distances[:, column] = np.linalg.norm(images @ lattice, axis=2).min(axis=1)
    nearest = distances.argmin(axis=1)
    best = distances.min(axis=1)
    nn = _nearest_neighbour_distance_ang(c)
    limit = CENTRES_ON_TM_SITES.value(c)
    return {
        "centre_tm_distance_ang": [round(float(d), 4) for d in best],
        "centre_nearest_tm_site": [
            f"{sites[i]['label']}#{sites[i]['atom']}" for i in nearest
        ],
        "centre_tm_distance_max_ang": float(best.max()) if len(best) else None,
        "centre_tm_limit_ang": limit,
        "nearest_neighbour_ang": nn,
        "centres_ok": bool(len(best) and best.max() <= limit),
    }


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

REFUSE_COUNT = "d_only_count_mismatch"
REFUSE_PROJECTORS = "d_only_projectors_not_tm_d"
REFUSE_CENTRES = "d_only_centres_off_tm_sites"


def _d_only_downfolding(c: Construction) -> CheckOutcome:
    sites = transition_metal_sites(c)
    if not sites:
        species = list(getattr(c.parent, "species", ()))
        raise NotApplicable(
            "no transition-metal atom in the structure (3d, 4d or 5d; La counted): a "
            "d-only downfolded model has nothing to downfold onto",
            record={
                "n_tm": 0,
                "elements": sorted({e for e in map(element_of, species) if e}),
                "not_counted": {
                    e: NOT_COUNTED[e]
                    for e in map(element_of, species)
                    if e in NOT_COUNTED
                },
            },
        )
    record = count_and_projectors(c, sites)
    missing: list[str] = []
    try:
        record.update(centres_on_tm_sites(c, sites))
    except NonEvaluable as exc:
        record["centres_ok"] = None
        missing.append(f"centres: {exc}")
    try:
        record.update(fermi_window_eta(c))
    except NonEvaluable as exc:
        record["window_eta_mev"] = None
        missing.append(f"window eta: {exc}")
    eta = record.get("window_eta_mev")
    if eta is not None and record.get("n_matched_states_in_window", 0) == 0:
        missing.append(
            "window eta: no model band and its matched DFT band lie together inside "
            "E_F +- 1 eV on the path; the d manifold does not reach the Fermi window, "
            "so the weights there are Fermi-function tails only"
        )
        eta = None

    refusals = []
    if record["count_ok"] is False:
        refusals.append(REFUSE_COUNT)
    if record["projectors_ok"] is False:
        refusals.append(REFUSE_PROJECTORS)
    if record.get("centres_ok") is False:
        refusals.append(REFUSE_CENTRES)
    if record["count_ok"] is None:
        missing.append("count: the construction recorded no num_wann")
    if record["projectors_ok"] is None and record["projector_route"] != "scdm":
        missing.append(record.get("projector_note", "projectors not judged"))

    note = f"n_TM {record['n_tm']} ({', '.join(record['tm_elements'])}): num_wann " f"{record['num_wann_actual']} against {record['num_wann_expected']} expected" + (
        ""
        if record.get("centre_tm_distance_max_ang") is None
        else f"; furthest centre {record['centre_tm_distance_max_ang']:.3f} Ang from "
        f"a TM site (limit {record['centre_tm_limit_ang']:.3f})"
    ) + (
        "; window eta n/a"
        if eta is None
        else f"; window eta {eta:.2f} meV over E_F +- 1 eV against DFT bands "
        f"{_bands_text(record.get('matched_dft_bands_compared_in_window', []))}"
    )
    if refusals:
        return CheckOutcome(
            passed=None,
            metric_value=float("nan") if eta is None else float(eta),
            metric_units="meV",
            note=f"{', '.join(refusals)}: {note}",
            refusal=", ".join(refusals),
            record=record,
        )
    if missing:
        raise NonEvaluable(f"{c.construction_id}: {'; '.join(missing)} ({note})")
    return CheckOutcome(
        passed=None,
        metric_value=float(eta),
        metric_units="meV",
        note=note,
        record=record,
    )


def _bands_text(bands) -> str:
    bands = sorted(bands)
    if not bands:
        return "(none)"
    groups: list[list[int]] = []
    for b in bands:
        if groups and b == groups[-1][-1] + 1:
            groups[-1].append(b)
        else:
            groups.append([b])
    return ",".join(f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups)


D_ONLY_DOWNFOLDING = Check(
    "d_only_downfolding",
    "eta over E_F +- 1 eV of the model's bands against the DFT bands they match, for a "
    "d-only model on the transition-metal sites",
    "scalar_vs_scale",
    "dft_bands",
    _d_only_downfolding,
    threshold=PASS_AT_20_MEV,
    description="for a structure with transition-metal atoms (3d, 4d or 5d; La "
    "included), the construction is a d-only downfolded model: num_wann is 5 per "
    "transition-metal atom and spin channel (10 for spinor functions) and its projectors "
    "are the d orbitals of the transition-metal sites and nothing else; every Wannier "
    "centre lies within half a nearest-neighbour distance of a transition-metal site; "
    "and on the separate high-symmetry path its bands reproduce the DFT bands they "
    "correspond to (each model band matched one-to-one, in energy order, to the DFT band "
    "nearest it; ligand bands are skipped) within E_F(scf) +- 1 eV with eta <= 20 meV "
    "(soft window edges, sigma 0.1 eV); withheld when the structure has no "
    "transition-metal atom",
)
