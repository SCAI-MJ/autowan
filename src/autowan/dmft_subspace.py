"""The DMFT-subspace criterion: the correlated-subspace Wannier model the literature builds.

User decision 2026-10-12 (autowan item 4, replacing ``downfolded_d``): the third
acceptance criterion is **the Wannier model a DMFT calculation of this material is built
on, as the literature builds it**. ``downfolded_d`` asked every transition-metal compound
for five d functions per metal site, judged over E_F +- 1 eV; neither matches practice.
Practice picks, material by material, the correlated subspace and judges the model over
that whole manifold: an isolated manifold is reproduced exactly (by construction) and an
entangled one inside its frozen window. So the criterion names a target per material and
applies to the five materials whose target the literature fixes beyond doubt:

========  ===============================  ==========================================
material  target subspace (per spin)       literature
========  ===============================  ==========================================
SrVO3     V t2g (dxy, dyz, dxz): 3 per V   Pavarini et al., PRL 92, 176403 (2004);
                                           Lechermann et al., PRB 74, 125120 (2006);
                                           Amadon et al., PRB 77, 205112 (2008)
SrTiO3    Ti t2g, the empty conduction     Zhong, Toth & Held, PRB 87, 161102(R)
          band: 3 per Ti                   (2013); Souto-Casares, Spaldin & Ederer,
                                           PRResearch 3, 023027 (2021)
NiO       Ni 3d + O 2p: 5 per Ni, 3 per O  Kunes et al., PRL 99, 156404 (2007);
                                           PRB 75, 165115 (2007)
MnO       Mn 3d + O 2p: 5 per Mn, 3 per O  Kunes et al., Nat. Mater. 7, 198 (2008)
Fe        Fe 4s + 3d: 6 per Fe             Sasioglu et al., PRB 83, 121101(R) (2011);
                                           Leonov et al., PRL 106, 106405 (2011)
========  ===============================  ==========================================

Every other material: withheld (not applicable). The structure is recognised by its
reduced composition, whatever magnetic cell the parent uses (NiO's AFM-II cell has two
Ni and two O). The parent DFT is the campaign's, unchanged; whether it is the material's
experimental ground state is not part of the criterion.

The check ``dmft_subspace_model`` passes when all three hold:

(i) **Count and seeds.** ``num_wann`` equals the target's functions per spin channel,
    and the projectors seed exactly the target orbitals, species by species: under an
    analytic projections block the named orbitals (``V:dxy;dyz;dxz``, ``Ni:d``,
    ``O:p``...), under pseudo-atomic orbitals the kept ones (a valence shell is the
    outermost orbital of its angular momentum in the species' list, so Fe 4s is the
    second s and not the 3s semicore). SCDM has no seeds and cannot name a subspace, so
    it is refused.

(ii) **Correlated functions on their sites.** Every correlated site (the transition-metal
    atoms) carries at least its correlated functions -- 3 for a t2g target, 5 for a d
    shell -- with centres within half the nearest-neighbour distance of it (each centre
    counted once, at its nearest correlated site). Ligand and s functions are not
    constrained beyond the default drift guard.

(iii) **The target manifold, reproduced.** On the separate high-symmetry path, the DFT
    states (k, n) whose Loewdin-projected weight on the target orbitals is at least 0.5
    -- the target manifold, read from projwfc on the path's own pw.x ``bands`` run; at a
    k-point where more states pass than the model has functions (a target orbital shared
    by two hybridised states of an entangled manifold), the ones with the largest weight,
    as many as there are functions -- are
    each compared with a model band: the one the order-preserving minimal-squared-
    distance assignment (:func:`~.downfolding.match_bands`) pairs with it, or, for a
    target state that assignment leaves unpaired, the model band nearest it in energy.
    The band distance is the root mean square of those differences over all target
    states, and passes at <= 20 meV. The literature judges this agreement by eye over
    the whole isolated manifold (Lechermann 2006: "identical to the LDA bands") or inside
    the frozen window of an entangled one (Souza, Marzari & Vanderbilt 2001) and states
    no number; 20 meV is this campaign's, the same as criteria 1 and 2. Selecting the
    states by target weight lets states of an entangled manifold that have mixed out of
    the target (MoS2-like Gamma points, SrTiO3's t2g top against e_g) leave the
    comparison by themselves.

    **Isolated or entangled, and the comparison window** (user decision 2026-10-12,
    second revision; check ``dmft_subspace_model_window``). The target manifold is
    *isolated* when, on the path and in every spin channel, the bands that carry its
    states (the lowest to the highest band index holding a state of target weight >=
    0.5) are exactly as many as the target has functions and an energy gap of more than
    1 meV separates them from the band below and the band above at every k-point;
    otherwise it is *entangled*. :func:`manifold_isolation` decides it from the
    reference bands and the weights, never by hand. An isolated manifold is compared
    over its whole width, as before. For an entangled one only the target states at or
    below the top of the frontier protocol's frozen window are compared:
    ``dis_froz_max`` = E_F + 2 eV for a metal, the conduction-band minimum + 2 eV for a
    gapped parent (aiida-wannier90-workflows 3.1.0 ``Wannier90WorkChain``: +2.0 eV
    relative to the Fermi energy, shifted to the LUMO when the bands are gapped;
    :data:`~.methods.WINDOW_MARGIN_EV` and :func:`~.methods.window_rule_top` here) --
    the window inside which the literature judges an entangled s-d or d-p model. Fe's
    s-d antibonding states 3-22 eV above E_F (target weight up to 0.97) are reproduced
    by no six-function model and leave the comparison; the per-k cap then acts on the
    states inside the window. E_F is the scf level of the reference
    (:func:`~.criteria._criterion_fermi`); metal or gapped is the parent's class on the
    construction mesh, and the conduction-band minimum the lowest parent state above
    E_F. The retired form without the window stays registered as
    ``dmft_subspace_model`` so verdicts stored under hash 941cf34c1c36bce4 reproduce.

The weights live beside the reference, not in it: a band-structure run records energies
only. :func:`write_weights` writes them; the reference carries the file's path in
``provenance["dmft_subspace_weights"]`` and the check verifies that the file's k-points,
band count and target agree with the reference before reading it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .criteria import PASS_AT_20_MEV, _criterion_fermi, _model, _reference
from .downfolding import element_of, match_bands
from .gate import (
    Check,
    CheckOutcome,
    Construction,
    NonEvaluable,
    NotApplicable,
    ScaleAwareThreshold,
    _nearest_neighbour_distance_ang,
)

__all__ = [
    "COMPARISON_WINDOW_MARGIN_EV",
    "DMFT_SUBSPACE_MODEL",
    "DMFT_SUBSPACE_MODEL_UNWINDOWED",
    "DMFT_SUBSPACE_CHECK_IDS",
    "SUBSPACES",
    "Shell",
    "SubspaceTarget",
    "TARGET_WEIGHT_MIN",
    "comparison_text",
    "comparison_window",
    "manifold_isolation",
    "reduced_formula",
    "target_for_species",
    "write_weights",
]

# ---------------------------------------------------------------------------
# the targets
# ---------------------------------------------------------------------------

#: Wannier90's names of the real harmonics, by l (mr order of its user guide Table 3.1).
M_NAMES: dict[int, tuple[str, ...]] = {
    0: ("s",),
    1: ("pz", "px", "py"),
    2: ("dz2", "dxz", "dyz", "dx2-y2", "dxy"),
}
SHELL_TOKENS = {"s": 0, "p": 1, "d": 2}
T2G = ("dxy", "dyz", "dxz")


@dataclass(frozen=True)
class Shell:
    """One target shell of one element: angular momentum, principal n, kept m."""

    element: str
    l: int
    n: int
    m: tuple[str, ...] = ()
    """The real harmonics kept (Wannier90 names); empty for the whole shell."""
    correlated: bool = False

    @property
    def functions(self) -> int:
        return len(self.m) if self.m else 2 * self.l + 1

    @property
    def orbitals(self) -> frozenset[tuple[int, str]]:
        return frozenset((self.l, name) for name in (self.m or M_NAMES[self.l]))

    @property
    def label(self) -> str:
        text = f"{self.element} {self.n}{'spdf'[self.l]}"
        return text + (f" ({', '.join(self.m)})" if self.m else "")


@dataclass(frozen=True)
class SubspaceTarget:
    """A material's correlated subspace, as the DMFT literature builds it."""

    formula: str
    shells: tuple[Shell, ...]
    description: str
    """The target in physical terms: what the repair prompt states."""
    references: tuple[str, ...]

    def functions_per_site(self, element: str) -> int:
        return sum(s.functions for s in self.shells if s.element == element)

    def correlated_per_site(self, element: str) -> int:
        return sum(
            s.functions for s in self.shells if s.element == element and s.correlated
        )

    @property
    def correlated_elements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.element for s in self.shells if s.correlated))

    def orbitals(self, element: str) -> frozenset[tuple[int, str]]:
        out: set[tuple[int, str]] = set()
        for s in self.shells:
            if s.element == element:
                out |= s.orbitals
        return frozenset(out)

    @property
    def text(self) -> str:
        return "; ".join(s.label for s in self.shells)


#: Reduced composition -> target. The keys are :func:`reduced_formula` strings.
SUBSPACES: dict[str, SubspaceTarget] = {
    t.formula: t
    for t in (
        SubspaceTarget(
            "O3SrV",
            (Shell("V", 2, 3, T2G, correlated=True),),
            "SrVO3: the three V t2g orbitals (dxy, dyz, dxz in the cubic axes) of each V "
            "atom -- the partly filled t2g band manifold at the Fermi level, between the O "
            "2p bands below and the e_g bands above",
            (
                "Pavarini et al., PRL 92, 176403 (2004)",
                "Lechermann et al., PRB 74, 125120 (2006)",
                "Amadon et al., PRB 77, 205112 (2008)",
            ),
        ),
        SubspaceTarget(
            "O3SrTi",
            (Shell("Ti", 2, 3, T2G, correlated=True),),
            "SrTiO3: the three Ti t2g orbitals (dxy, dyz, dxz in the cubic axes) of each "
            "Ti atom -- the empty t2g conduction-band manifold above the gap",
            (
                "Zhong, Toth & Held, PRB 87, 161102(R) (2013)",
                "Souto-Casares, Spaldin & Ederer, PRResearch 3, 023027 (2021)",
            ),
        ),
        SubspaceTarget(
            "NiO",
            (Shell("Ni", 2, 3, correlated=True), Shell("O", 1, 2)),
            "NiO: the five Ni 3d orbitals of each Ni atom and the three O 2p orbitals of "
            "each O atom -- the d-p (charge-transfer) manifold spanned by the O 2p and "
            "Ni 3d bands",
            (
                "Kunes et al., PRL 99, 156404 (2007)",
                "Kunes et al., PRB 75, 165115 (2007)",
            ),
        ),
        SubspaceTarget(
            "MnO",
            (Shell("Mn", 2, 3, correlated=True), Shell("O", 1, 2)),
            "MnO: the five Mn 3d orbitals of each Mn atom and the three O 2p orbitals of "
            "each O atom -- the d-p manifold spanned by the O 2p and Mn 3d bands",
            ("Kunes et al., Nat. Mater. 7, 198 (2008)",),
        ),
        SubspaceTarget(
            "Fe",
            (Shell("Fe", 0, 4), Shell("Fe", 2, 3, correlated=True)),
            "Fe: the five Fe 3d orbitals and the Fe 4s orbital of each Fe atom -- the "
            "s-d manifold of the metal's valence bands",
            (
                "Sasioglu et al., PRB 83, 121101(R) (2011)",
                "Leonov et al., PRL 106, 106405 (2011)",
            ),
        ),
    )
}

#: Criterion (iii): a DFT state belongs to the target manifold at this projected weight.
TARGET_WEIGHT_MIN = 0.5

#: Criterion (iii), entangled targets: the frontier protocol's frozen-window margin above
#: E_F (metal) or the conduction-band minimum (gapped) -- aiida-wannier90-workflows'
#: ``dis_froz_max = +2.0`` relative to the Fermi energy (the same value as
#: :data:`~.methods.WINDOW_MARGIN_EV`, asserted by the tests).
COMPARISON_WINDOW_MARGIN_EV = 2.0

#: An energy gap separates two bands when it exceeds this (the frontier package's own
#: test for a gapped band structure, ``bandgap > 1e-3``).
GAP_MIN_EV = 1e-3

#: Criterion (ii): a centre within half the nearest-neighbour distance of its site.
CENTRES_ON_CORRELATED_SITES = ScaleAwareThreshold(
    0.5,
    "nearest_neighbour_distance_ang",
    calibration="half the nearest-neighbour distance: beyond it a centre is closer to "
    "another site than to its own, so the function cannot be assigned to the correlated "
    "atom a DMFT local self-energy lives on. Geometric, not fitted (as downfolded_d).",
)


def reduced_formula(species: Sequence[str]) -> str | None:
    """Elements sorted alphabetically with their counts divided by the common factor."""
    elements = [element_of(s) for s in species]
    if not elements or any(e is None for e in elements):
        return None
    counts = Counter(elements)
    g = reduce(math.gcd, counts.values())
    return "".join(
        f"{e}{counts[e] // g if counts[e] // g > 1 else ''}" for e in sorted(counts)
    )


def target_for_species(species: Sequence[str]) -> SubspaceTarget | None:
    """The target for a structure's species list, or ``None`` when withheld."""
    formula = reduced_formula(species)
    return None if formula is None else SUBSPACES.get(formula)


# ---------------------------------------------------------------------------
# (i) count and seeds
# ---------------------------------------------------------------------------

_MR = re.compile(r"mr=([\d,]+)")


def _token_orbitals(token: str) -> set[tuple[int, str]] | None:
    """The (l, m-name) pairs one projections token names; ``None`` for a hybrid."""
    t = token.strip().lower()
    if t in SHELL_TOKENS:
        l = SHELL_TOKENS[t]
        return {(l, m) for m in M_NAMES[l]}
    for l, names in M_NAMES.items():
        if t in names:
            return {(l, t)}
    if t.startswith("l="):
        try:
            l = int(t.split(",")[0][2:])
        except ValueError:
            return None
        if l not in M_NAMES:
            return None
        found = _MR.search(t.replace(" ", ""))
        if not found:
            return {(l, m) for m in M_NAMES[l]}
        out = set()
        for part in found.group(1).split(","):
            if part:
                idx = int(part)
                if not 1 <= idx <= len(M_NAMES[l]):
                    return None
                out.add((l, M_NAMES[l][idx - 1]))
        return out
    return None


def _pao_orbitals(group: Mapping[str, Any]) -> tuple[set[tuple[int, str]], list[str]]:
    """(l, m-name) pairs a PAO group keeps, and the kept orbitals that are not valence.

    A valence shell is the outermost orbital of its l in the species' list: with the
    generic labels (``s1``, ``s2``...) the highest index, with spectroscopic labels
    (``3D``, ``4S``) the highest n. A kept inner orbital (semicore) is reported.
    """
    kept = [str(x) for x in group.get("labels", [])]
    every = [str(x) for x in group.get("all_labels", kept)]

    def parsed(label: str) -> tuple[int, int] | None:
        m = re.fullmatch(r"(\d)([spdfSPDF])", label.strip())
        if m:
            return "spdf".index(m.group(2).lower()), int(m.group(1))
        m = re.fullmatch(r"([spdfSPDF])(\d+)", label.strip())
        if m:
            return "spdf".index(m.group(1).lower()), int(m.group(2))
        return None

    outermost: dict[int, int] = {}
    for label in every:
        p = parsed(label)
        if p is not None:
            outermost[p[0]] = max(outermost.get(p[0], 0), p[1])
    orbitals: set[tuple[int, str]] = set()
    inner: list[str] = []
    for label in kept:
        p = parsed(label)
        if p is None:
            inner.append(label)
            continue
        l, rank = p
        if rank != outermost.get(l):
            inner.append(label)
        if l in M_NAMES:
            orbitals |= {(l, m) for m in M_NAMES[l]}
    return orbitals, inner


def expected_functions(species: Sequence[str], target: SubspaceTarget) -> int:
    """The target's functions per spin channel for a structure's species list."""
    return sum(target.functions_per_site(element_of(s)) for s in species)


def count_and_seeds(c: Construction, target: SubspaceTarget) -> dict[str, Any]:
    """Criterion (i): the Wannier count, and whether the seeds are the target orbitals."""
    species = list(getattr(c.parent, "species", ()))
    elements = [element_of(s) for s in species]
    expected = expected_functions(species, target)
    actual = getattr(c.wannier, "num_wann", None)
    requested = c.inputs.get("num_wann")
    if actual is None:
        actual = requested
    groups_record = c.inputs.get("projector_groups") or {}
    record: dict[str, Any] = {
        "target": target.formula,
        "target_shells": [s.label for s in target.shells],
        "num_wann_expected": expected,
        "num_wann_actual": None if actual is None else int(actual),
        "num_wann_requested": None if requested is None else int(requested),
        "projector_route": groups_record.get("route", "not recorded"),
        "count_ok": None if actual is None else int(actual) == expected,
    }
    route = record["projector_route"]
    seeds_ok: bool | None
    if bool(getattr(c.parent, "spinor", False)) or groups_record.get("spinors"):
        seeds_ok = False
        record["seed_note"] = "spinor functions: no target of this criterion is spinor"
    elif route == "scdm":
        seeds_ok = False
        record["seed_note"] = (
            "SCDM has no seeds, so the construction does not name the target orbitals"
        )
    elif groups_record.get("error"):
        seeds_ok = None
        record["seed_note"] = f"projectors unreadable: {groups_record['error']}"
    elif not groups_record:
        seeds_ok = None
        record["seed_note"] = "the construction's projectors were not recorded"
    else:
        lattice = np.asarray(c.parent.lattice_ang, dtype=float)
        frac_atoms = np.asarray(c.parent.positions_frac, dtype=float)
        seeded: dict[str, set[tuple[int, str]]] = {}
        problems: list[str] = []
        for group in groups_record.get("groups", []):
            label = group.get("species")
            if label is None and "position" in group:
                position = np.asarray(group["position"], dtype=float)
                frac = (
                    position
                    if group.get("frame") == "fractional"
                    else np.linalg.solve(
                        lattice.T,
                        position
                        * (
                            0.529177210903
                            if group.get("frame") == "cartesian_bohr"
                            else 1.0
                        ),
                    )
                )
                delta = frac_atoms - frac
                delta -= np.round(delta)
                dist = np.linalg.norm(delta @ lattice, axis=1)
                atom = int(np.argmin(dist))
                element = element_of(species[atom]) if dist[atom] < 1e-3 else None
            else:
                element = element_of(label)
            if element is None:
                problems.append(f"{group.get('site')}: not on an atom")
                continue
            if route == "projections":
                tokens = group.get("orbitals")
                if tokens is None:
                    problems.append(f"{group.get('site')}: orbital names not recorded")
                    continue
                got: set[tuple[int, str]] = set()
                for token in tokens:
                    pairs = _token_orbitals(token)
                    if pairs is None:
                        problems.append(
                            f"{group.get('site')}: {token!r} is not a target orbital"
                        )
                        continue
                    got |= pairs
            else:
                got, inner = _pao_orbitals(group)
                if inner:
                    problems.append(
                        f"{group.get('site')}: keeps {', '.join(inner)}, not the valence shell"
                    )
            seeded.setdefault(element, set()).update(got)
        for element in sorted(
            set(seeded) | {e for e in elements if target.orbitals(e)}
        ):
            want = target.orbitals(element)
            have = seeded.get(element, set())
            if have != want:
                missing = sorted(want - have)
                extra = sorted(have - want)
                problems.append(
                    f"{element}: "
                    + (
                        "; ".join(
                            x
                            for x in (
                                (
                                    f"missing {', '.join(m for _, m in missing)}"
                                    if missing
                                    else ""
                                ),
                                (
                                    f"not in the target {', '.join(m for _, m in extra)}"
                                    if extra
                                    else ""
                                ),
                            )
                            if x
                        )
                    )
                )
        record["seeded"] = {
            e: sorted(m for _, m in v) for e, v in sorted(seeded.items())
        }
        record["seed_problems"] = problems
        seeds_ok = not problems
    record["seeds_ok"] = seeds_ok
    return record


# ---------------------------------------------------------------------------
# (ii) correlated functions on their sites
# ---------------------------------------------------------------------------


def centres_on_correlated_sites(
    c: Construction, target: SubspaceTarget
) -> dict[str, Any]:
    """Criterion (ii): each correlated site carries its correlated functions."""
    species = list(getattr(c.parent, "species", ()))
    sites = [
        (i, species[i], element_of(species[i]))
        for i in range(len(species))
        if element_of(species[i]) in target.correlated_elements
    ]
    centres = np.asarray(c.require("centres_ang"), dtype=float).reshape(-1, 3)
    lattice = np.asarray(c.parent.lattice_ang, dtype=float)
    frac_atoms = np.asarray(c.parent.positions_frac, dtype=float)
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    centres_frac = np.linalg.solve(lattice.T, centres.T).T
    distances = np.full((len(centres), len(sites)), np.inf)
    for column, (atom, _, _) in enumerate(sites):
        delta = centres_frac - frac_atoms[atom]
        delta -= np.round(delta)
        images = delta[:, None, :] + shifts[None, :, :]
        distances[:, column] = np.linalg.norm(images @ lattice, axis=2).min(axis=1)
    limit = CENTRES_ON_CORRELATED_SITES.value(c)
    nearest = distances.argmin(axis=1) if len(sites) else np.zeros(len(centres), int)
    best = distances.min(axis=1) if len(sites) else np.full(len(centres), np.inf)
    per_site = []
    ok = bool(sites)
    for column, (atom, label, element) in enumerate(sites):
        on = int(np.sum((nearest == column) & (best <= limit)))
        need = target.correlated_per_site(element)
        per_site.append(
            {"site": f"{label}#{atom + 1}", "functions_on_site": on, "needed": need}
        )
        ok = ok and on >= need
    return {
        "correlated_sites": per_site,
        "centre_site_limit_ang": limit,
        "nearest_neighbour_ang": _nearest_neighbour_distance_ang(c),
        "centres_ok": ok,
    }


# ---------------------------------------------------------------------------
# (iii) the target manifold
# ---------------------------------------------------------------------------


def write_weights(
    path: str | Path,
    *,
    weights: np.ndarray,
    kpoints_frac: np.ndarray,
    target: SubspaceTarget,
    source: str,
    bands_ev: np.ndarray | None = None,
) -> Path:
    """Store target-orbital weights ``(n_spin, n_k, n_bands)`` for a reference path."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 2:
        w = w[None]
    path = Path(path)
    payload = {
        "weights": w,
        "kpoints": np.asarray(kpoints_frac, dtype=float),
        "target": np.array(target.formula),
        "shells": np.array(json.dumps([s.label for s in target.shells])),
        "source": np.array(source),
    }
    if bands_ev is not None:
        payload["bands_ev"] = np.asarray(bands_ev, dtype=float)
    np.savez_compressed(path, **payload)
    return path


def _read_weights(source: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(source) as f:
        return (
            np.asarray(f["weights"], dtype=float),
            np.asarray(f["kpoints"], dtype=float),
            str(f["target"]),
        )


def _weights(c: Construction, target: SubspaceTarget, ref) -> tuple[np.ndarray, str]:
    source = (getattr(ref, "provenance", None) or {}).get("dmft_subspace_weights")
    if not source:
        raise NonEvaluable(
            f"{c.construction_id}: the reference path carries no target-orbital weights "
            "(provenance['dmft_subspace_weights']), so the target manifold is unknown"
        )
    w, k, formula = _read_weights(source)
    if formula != target.formula:
        raise NonEvaluable(
            f"{c.construction_id}: the weights are for {formula}, the structure is "
            f"{target.formula}"
        )
    if k.shape != ref.kpoints_frac.shape or np.max(np.abs(k - ref.kpoints_frac)) > 1e-5:
        raise NonEvaluable(
            f"{c.construction_id}: the weights' k-points are not the path's"
        )
    if w.shape != ref.bands_ev.shape:
        raise NonEvaluable(
            f"{c.construction_id}: weights {w.shape} do not match the reference bands "
            f"{ref.bands_ev.shape}"
        )
    return w, str(source)


def manifold_isolation(bands_ev, weights, n_functions: int) -> dict[str, Any]:
    """Whether the target manifold is isolated on the path, decided from the band data.

    Per spin channel: the target bands run from the lowest to the highest band index
    holding a state of target weight >= :data:`TARGET_WEIGHT_MIN`. The channel is
    isolated when those bands are exactly ``n_functions`` and a gap of more than
    :data:`GAP_MIN_EV` separates them, over the whole path, from the band below (none
    computed counts as a gap) and from the band above (none computed does not: what lies
    above is unknown). The manifold is isolated when every channel is.
    """
    bands = np.asarray(bands_ev, dtype=float)
    w = np.asarray(weights, dtype=float)
    if bands.ndim == 2:
        bands = bands[None]
    if w.ndim == 2:
        w = w[None]
    channels = []
    for spin in range(bands.shape[0]):
        e = bands[spin]
        in_t = w[spin] >= TARGET_WEIGHT_MIN
        index = np.nonzero(in_t.any(axis=0))[0]
        if not len(index):
            channels.append(
                {
                    "spin": spin,
                    "isolated": False,
                    "reasons": [f"no state of target weight >= {TARGET_WEIGHT_MIN}"],
                }
            )
            continue
        lo, hi = int(index.min()), int(index.max())
        width = hi - lo + 1
        gap_below = None if lo == 0 else float(e[:, lo].min() - e[:, lo - 1].max())
        gap_above = (
            None if hi + 1 >= e.shape[1] else float(e[:, hi + 1].min() - e[:, hi].max())
        )
        reasons = []
        if width != n_functions:
            reasons.append(
                f"bands {lo + 1}-{hi + 1} carry target states: {width} bands for "
                f"{n_functions} target functions"
            )
        if gap_below is not None and gap_below <= GAP_MIN_EV:
            reasons.append(
                f"no gap to the band below (band {lo} reaches {-gap_below * 1000:.0f} "
                f"meV above the bottom of band {lo + 1})"
            )
        if gap_above is None:
            reasons.append(
                "the highest computed band carries target states, so what lies above "
                "is unknown"
            )
        elif gap_above <= GAP_MIN_EV:
            reasons.append(
                f"no gap to the band above (band {hi + 2} dips {-gap_above * 1000:.0f} "
                f"meV below the top of band {hi + 1})"
            )
        channels.append(
            {
                "spin": spin,
                "target_bands_1based": [lo + 1, hi + 1],
                "n_target_bands": width,
                "n_functions": int(n_functions),
                "gap_below_ev": gap_below,
                "gap_above_ev": gap_above,
                "isolated": not reasons,
                "reasons": reasons,
            }
        )
    return {
        "isolated": bool(channels) and all(ch["isolated"] for ch in channels),
        "channels": channels,
        "gap_min_ev": GAP_MIN_EV,
        "target_weight_min": TARGET_WEIGHT_MIN,
    }


def _window(
    metal: bool, fermi_ev: float, eigenvalues_ev, source: str
) -> dict[str, Any]:
    """The frontier protocol's frozen-window top: E_F + 2 eV, or the CBM + 2 eV."""
    anchor, anchor_ev = "Fermi level", float(fermi_ev)
    if not metal and eigenvalues_ev is not None:
        e = np.asarray(eigenvalues_ev, dtype=float)
        above = e[e > fermi_ev]
        if above.size:
            anchor, anchor_ev = "conduction-band minimum", float(above.min())
        else:
            anchor = "Fermi level (no conduction band computed)"
    return {
        "comparison_window_top_ev": anchor_ev + COMPARISON_WINDOW_MARGIN_EV,
        "comparison_window_anchor": anchor,
        "comparison_window_anchor_ev": anchor_ev,
        "comparison_window_margin_ev": COMPARISON_WINDOW_MARGIN_EV,
        "comparison_window_fermi_ev": float(fermi_ev),
        "comparison_window_bands": source,
    }


def comparison_window(c: Construction, ref) -> dict[str, Any]:
    """The comparison window of an entangled target for this construction."""
    fermi, fermi_source = _criterion_fermi(c, ref)
    metal = c.material_class == "metal"
    eigen = getattr(c.parent, "eigenvalues_ev", None)
    source = "construction mesh"
    if eigen is None:
        eigen, source = ref.bands_ev, "reference path"
    out = _window(metal, fermi, eigen, source)
    out.update(material_class=c.material_class, fermi_source=fermi_source)
    return out


def comparison_text(
    target: SubspaceTarget,
    species: Sequence[str],
    reference_path=None,
    parent=None,
) -> str:
    """The model-facing statement of which target states are compared, for a structure.

    With the run's reference path (and its weights) and parent, the statement names this
    manifold's own case; without them, the rule.
    """
    rule = (
        "When the target manifold is isolated (in every spin channel an energy gap "
        "separates its bands from all other bands) all of its states are compared; when "
        "it is entangled with other bands, only its states at or below 2 eV above the "
        "Fermi level (2 eV above the conduction-band minimum for a gapped material) are "
        "compared."
    )
    if reference_path is None:
        return rule
    try:
        source = (reference_path.provenance or {}).get("dmft_subspace_weights")
        w, _, formula = _read_weights(source)
        if formula != target.formula or w.shape != reference_path.bands_ev.shape:
            return rule
        iso = manifold_isolation(
            reference_path.bands_ev, w, expected_functions(species, target)
        )
    except Exception:  # noqa: BLE001 -- the rule is stated instead
        return rule
    if iso["isolated"]:
        return (
            "On this calculation's reference band path the target manifold is isolated "
            "(in every spin channel an energy gap separates its bands from all other "
            "bands), so all of its states are compared."
        )
    fermi = reference_path.fermi_energy_ev
    if fermi is None:
        return rule
    metal = (
        bool(parent.is_metallic())
        if parent is not None and hasattr(parent, "is_metallic")
        else None
    )
    if metal is None:
        return (
            "On this calculation's reference band path the target manifold is entangled "
            "with other bands (in at least one spin channel no energy gap separates it "
            "from them), so only its states at or below 2 eV above the Fermi level (2 "
            "eV above the conduction-band minimum for a gapped material) are compared."
        )
    win = _window(
        metal,
        fermi,
        getattr(parent, "eigenvalues_ev", None),
        "construction mesh",
    )
    anchor = win["comparison_window_anchor"].split(" (")[0]
    short = "E_F" if anchor == "Fermi level" else "CBM"
    return (
        "On this calculation's reference band path the target manifold is entangled "
        "with other bands (in at least one spin channel no energy gap separates it from "
        f"them), so it is judged on its target states up to 2 eV above the {anchor}: "
        f"those at or below {win['comparison_window_top_ev']:.2f} eV on this "
        f"calculation's energy scale ({short} = "
        f"{win['comparison_window_anchor_ev']:.2f} eV) are compared, and target states "
        "above that energy are not."
    )


def target_manifold_distance(
    c: Construction, target: SubspaceTarget, *, window: bool = True
) -> dict[str, Any]:
    """Criterion (iii): RMS over the target states of the model band paired with each.

    With ``window`` (the criterion since the second 2026-10-12 revision), an entangled
    target manifold is compared only at or below :func:`comparison_window`'s top.
    """
    ref = _reference(c)
    spin = int(c.inputs.get("spin_index", 0))
    dft = ref.channel(spin)
    weights, source = _weights(c, target, ref)
    weight = weights[spin]
    wan = _model(c).bands(ref.kpoints_frac)
    try:
        matched = match_bands(wan, dft)
    except ValueError as exc:
        raise NonEvaluable(f"{c.construction_id}: {exc}") from exc
    in_target = weight >= TARGET_WEIGHT_MIN
    extra: dict[str, Any] = {}
    if window:
        species = list(getattr(c.parent, "species", ()))
        iso = manifold_isolation(
            ref.bands_ev, weights, expected_functions(species, target)
        )
        extra["manifold_isolated"] = iso["isolated"]
        extra["manifold_isolation"] = iso
        extra["n_target_states_above_window"] = 0
        if not iso["isolated"]:
            win = comparison_window(c, ref)
            extra.update(win)
            above = in_target & (dft > win["comparison_window_top_ev"])
            extra["n_target_states_above_window"] = int(above.sum())
            in_target = in_target & ~above
    n_functions = int(wan.shape[1])
    capped = 0
    for k in np.nonzero(in_target.sum(axis=1) > n_functions)[0]:
        passing = np.nonzero(in_target[k])[0]
        keep = passing[np.argsort(-weight[k, passing], kind="stable")[:n_functions]]
        capped += len(passing) - len(keep)
        in_target[k] = False
        in_target[k, keep] = True
    n_states = int(in_target.sum())
    if n_states == 0:
        raise NonEvaluable(
            f"{c.construction_id}: no DFT state on the path has target weight >= "
            f"{TARGET_WEIGHT_MIN}"
            + (
                f" at or below {extra['comparison_window_top_ev']:.2f} eV"
                if "comparison_window_top_ev" in extra
                else ""
            )
        )
    diffs = []
    unpaired = 0
    for k in range(dft.shape[0]):
        paired = {int(n): j for j, n in enumerate(matched[k])}
        for n in np.nonzero(in_target[k])[0]:
            j = paired.get(int(n))
            if j is None:
                unpaired += 1
                diffs.append(float(np.min(np.abs(wan[k] - dft[k, n]))))
            else:
                diffs.append(float(dft[k, n] - wan[k, j]))
    diffs_arr = np.asarray(diffs)
    rms = float(np.sqrt(np.mean(diffs_arr**2)) * 1000.0)
    bands_in = sorted({int(n) + 1 for n in np.nonzero(in_target)[1]})
    energies = dft[in_target]
    return {
        "target_rms_mev": rms,
        "target_max_mev": float(np.max(np.abs(diffs_arr)) * 1000.0),
        "n_target_states": n_states,
        "n_target_states_unpaired": unpaired,
        "n_target_states_beyond_function_count": capped,
        "target_states_per_k": [
            int(in_target.sum(axis=1).min()),
            int(in_target.sum(axis=1).max()),
        ],
        "target_bands": bands_in,
        "target_energy_range_ev": [float(energies.min()), float(energies.max())],
        "target_weight_min": TARGET_WEIGHT_MIN,
        "weights_source": source,
        "n_kpoints": int(dft.shape[0]),
        "n_dft_bands": int(dft.shape[1]),
        "n_wannier_bands": int(wan.shape[1]),
        "reference": ref.source,
        **extra,
    }


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

REFUSE_COUNT = "dmft_subspace_count_mismatch"
REFUSE_SEEDS = "dmft_subspace_seeds_not_target"
REFUSE_CENTRES = "dmft_subspace_correlated_centres_off_site"


def _dmft_subspace_model_window(c: Construction) -> CheckOutcome:
    return _evaluate(c, window=True)


def _dmft_subspace_model(c: Construction) -> CheckOutcome:
    return _evaluate(c, window=False)


def _evaluate(c: Construction, *, window: bool) -> CheckOutcome:
    species = list(getattr(c.parent, "species", ()))
    target = target_for_species(species)
    if target is None:
        raise NotApplicable(
            "the criterion names a literature DMFT subspace for SrVO3, SrTiO3, NiO, MnO "
            "and Fe only",
            record={"formula": reduced_formula(species), "covered": sorted(SUBSPACES)},
        )
    record = count_and_seeds(c, target)
    missing: list[str] = []
    try:
        record.update(centres_on_correlated_sites(c, target))
    except NonEvaluable as exc:
        record["centres_ok"] = None
        missing.append(f"centres: {exc}")
    try:
        record.update(target_manifold_distance(c, target, window=window))
    except NonEvaluable as exc:
        record["target_rms_mev"] = None
        missing.append(f"target manifold: {exc}")
    rms = record.get("target_rms_mev")
    refusals = []
    if record["count_ok"] is False:
        refusals.append(REFUSE_COUNT)
    if record["seeds_ok"] is False:
        refusals.append(REFUSE_SEEDS)
    if record.get("centres_ok") is False:
        refusals.append(REFUSE_CENTRES)
    if record["count_ok"] is None:
        missing.append("count: the construction recorded no num_wann")
    if record["seeds_ok"] is None:
        missing.append(record.get("seed_note", "seeds not judged"))
    note = (
        f"target {target.text}: num_wann {record['num_wann_actual']} against "
        f"{record['num_wann_expected']}"
        + (
            ""
            if rms is None
            else f"; RMS {rms:.2f} meV over {record['n_target_states']} target states "
            f"(bands {_bands_text(record['target_bands'])}, "
            f"{record['target_energy_range_ev'][0]:.2f} to "
            f"{record['target_energy_range_ev'][1]:.2f} eV)"
        )
        + _window_note(record)
    )
    if refusals:
        return CheckOutcome(
            passed=None,
            metric_value=float("nan") if rms is None else float(rms),
            metric_units="meV",
            note=f"{', '.join(refusals)}: {note}",
            refusal=", ".join(refusals),
            record=record,
        )
    if missing:
        raise NonEvaluable(f"{c.construction_id}: {'; '.join(missing)} ({note})")
    return CheckOutcome(
        passed=None,
        metric_value=float(rms),
        metric_units="meV",
        note=note,
        record=record,
    )


def _window_note(record: Mapping[str, Any]) -> str:
    if "manifold_isolated" not in record:
        return ""
    if record["manifold_isolated"]:
        return "; isolated target manifold, compared over its whole width"
    if "comparison_window_top_ev" not in record:
        return "; entangled target manifold"
    return (
        f"; entangled target manifold, compared at or below "
        f"{record['comparison_window_top_ev']:.2f} eV ("
        f"{record['comparison_window_anchor'].split(' (')[0]} + "
        f"{record['comparison_window_margin_ev']:g} eV), "
        f"{record['n_target_states_above_window']} target states above it not compared"
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


#: The criterion (second 2026-10-12 revision): an entangled target manifold is compared
#: inside the frontier protocol's frozen window only.
DMFT_SUBSPACE_MODEL = Check(
    "dmft_subspace_model_window",
    "RMS energy difference over the target manifold's DFT states of the model bands "
    "paired with them, for the literature's DMFT subspace of the material, over the "
    "whole manifold when it is isolated and up to 2 eV above E_F (or the conduction-band "
    "minimum) when it is entangled",
    "scalar_vs_scale",
    "dft_bands",
    _dmft_subspace_model_window,
    threshold=PASS_AT_20_MEV,
    description="for SrVO3, SrTiO3, NiO, MnO and Fe, the construction is the "
    "correlated-subspace Wannier model the DMFT literature builds for that material (the "
    "target stated for this structure below): num_wann equals the target's functions "
    "per spin channel and the projectors seed exactly the target orbitals; every "
    "correlated (transition-metal) site carries its correlated functions, centred within "
    "half a nearest-neighbour distance of it; and on the separate high-symmetry path, "
    "every compared DFT state whose projected weight on the target orbitals is at least "
    "0.5 is reproduced by the model band paired with it, with a root-mean-square energy "
    "difference <= 20 meV over all those states. The compared states are the whole "
    "target manifold when it is isolated (in every spin channel an energy gap separates "
    "its bands from all other bands) and, when it is entangled with other bands, its "
    "states at or below 2 eV above the Fermi level (2 eV above the conduction-band "
    "minimum for a gapped material), where the literature judges an entangled model "
    "(it states no number; 20 meV is this campaign's, as for the other criteria); "
    "withheld for any other material",
)

#: The first form (hash 941cf34c1c36bce4 as the dmft_subspace profile): every target
#: state compared, entangled or not. Registered so its stored verdicts reproduce.
DMFT_SUBSPACE_MODEL_UNWINDOWED = Check(
    "dmft_subspace_model",
    "RMS energy difference over the target manifold's DFT states of the model bands "
    "paired with them, for the literature's DMFT subspace of the material",
    "scalar_vs_scale",
    "dft_bands",
    _dmft_subspace_model,
    threshold=PASS_AT_20_MEV,
    description="for SrVO3, SrTiO3, NiO, MnO and Fe, the construction is the "
    "correlated-subspace Wannier model the DMFT literature builds for that material (the "
    "target stated for this structure below): num_wann equals the target's functions "
    "per spin channel and the projectors seed exactly the target orbitals; every "
    "correlated (transition-metal) site carries its correlated functions, centred within "
    "half a nearest-neighbour distance of it; and on the separate high-symmetry path, "
    "every DFT state whose projected weight on the target orbitals is at least 0.5 is "
    "reproduced by the model band paired with it, with a root-mean-square energy "
    "difference <= 20 meV over all those states (the literature judges this agreement "
    "over the whole target manifold and states no number; 20 meV is this campaign's, "
    "as for the other criteria); withheld for any other material",
)

#: The check identifiers that carry a per-structure target.
DMFT_SUBSPACE_CHECK_IDS = (
    DMFT_SUBSPACE_MODEL.check_id,
    DMFT_SUBSPACE_MODEL_UNWINDOWED.check_id,
)
