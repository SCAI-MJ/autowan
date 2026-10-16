"""The checks a construction must satisfy, and which material classes they apply to.

The set is split by what each check needs rather than by what it is about. Some
questions are *same-or-not* — did the minimisation converge, do the Wannier centres
sit on atoms, does the real-space Hamiltonian stay inside the supercell the k-mesh
can represent — and those carry no threshold at all, so no constant can travel with
them. The rest are distances, and each of those measures itself against a scale the
material supplies.

One check is withheld from metals on purpose. Band distance asks how faithfully the
model reproduces the parent's bands, which presumes the construction was meant to
reproduce a particular set of them. A metal has no isolated manifold: its subspace is
carved out of a continuum, so the states left outside are not an error the
construction made. Judging a metal by that distance answers a question its physics
never posed — and widening the threshold until it passes would be worse, because then
the number reports nothing at all. The metal is judged on the constant-free checks
instead.

The metal_frozen_window_contains_fermi_level check asks a separate, discrete question:
whether a declared frozen window actually contains the metal's Fermi level. A window
that misses it releases the frontier states to the disentanglement, even when every
check measured inside that window is satisfied.
"""

from __future__ import annotations

import math

import numpy as np

from .gate import (
    MAGNETIC_D_WEIGHT_FLOOR,
    SCALES,
    Check,
    CheckOutcome,
    Construction,
    NonEvaluable,
    ScaleAwareThreshold,
    magnetic_subspace_mask,
    parent_bands_for_check,
)
from .interpolation import cartesian_to_fractional_k, compare_to_parent, interpolate

#: How far above the Fermi level a construction is held to when it declares no frozen
#: window of its own. Two electronvolts is the range the projectability literature
#: reports its band distances over, so a verdict here is comparable with a published one.
REPRODUCED_MARGIN_EV = 2.0
FROZEN_WINDOW_NARROWER_THAN_TARGET = "frozen_window_narrower_than_target"
PARENT_BAND_CHECKS = (
    "frontier_manifold_straddles_fermi_level",
    "occupied_manifold_is_represented",
    "model_bands_correspond_to_parent_bands",
    "band_distance_from_parent",
)


MAGNETIC_SUBSPACE_TOLERANCE = ScaleAwareThreshold(
    0.01,
    "magnetic_manifold_bandwidth_ev",
    calibration="One percent of the declared magnetic manifold's own bandwidth: the "
    "spread of the counted states over all k-points and both spins. The same relative "
    "tolerance the default target takes of the occupied bandwidth; set a priori, "
    "with no energy imported from another material.",
)


# ---------------------------------------------------------------------------
# discrete checks — no threshold, nothing to transplant
# ---------------------------------------------------------------------------


def _subspace_selection_converged(c: Construction) -> CheckOutcome:
    stage = c.wannier.disentanglement
    if not c.wannier.disentanglement_used:
        return CheckOutcome(True, 0.0, "iterations", "no disentanglement was required")
    if stage is None:
        raise NonEvaluable(f"{c.construction_id}: no disentanglement record in the run")
    return CheckOutcome(
        stage.converged,
        float(stage.n_iterations),
        "iterations",
        (
            ""
            if stage.converged
            else "stopped at the iteration limit, not at the criterion"
        ),
    )


#: The convergence window a spread minimisation is judged over when the run itself
#: tested nothing (``conv_window`` unset or <= 1, which Wannier90 reads as "no test").
#: It is the window autowan writes into every repaired deck
#: (:data:`autowan.execution.DEFAULT_CONV_WINDOW`), so a frontier construction read
#: from its trajectory meets the criterion a repaired one meets inside the engine. Every
#: verdict it decides records it as the reference window.
REFERENCE_CONV_WINDOW = 3

#: Wannier90's own defaults (``parameters.F90``, v3.1 and develop) for the keys the
#: trajectory reading needs when neither the deck nor the ``.wout`` header gives them:
#: ``conv_tol`` in Å² and ``num_iter``.
WANNIER90_DEFAULT_CONV_TOL_ANG2 = 1.0e-10
WANNIER90_DEFAULT_NUM_ITER = 100

#: How a spread-convergence verdict read from the trajectory names its evidence.
EVALUATED_FROM_TRAJECTORY = "evaluated from trajectory"


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _spread_minimisation_converged(c: Construction) -> CheckOutcome:
    """Whether the spread minimisation met its own convergence criterion.

    Wannier90 runs its convergence test only when ``conv_window`` > 1; it leaves the
    window at -1 unless asked, and then runs ``num_iter`` iterations with no test, so the
    run neither converges nor fails to by its own account. Reading the absent flag as a
    failure would condemn every run that did not opt in -- the canonical iron example in
    the Wannier90 manual and the published VASP SCDM frontier deck among them.

    Such a run is judged from its trajectory instead: the
    question is unchanged, only the evidence is read differently. Wannier90's own
    criterion -- |ΔΩ| <= ``conv_tol`` on each of ``conv_window`` consecutive iterations
    -- is applied to the per-iteration total-spread changes printed in the ``.wout``,
    with the deck's ``conv_tol`` (Wannier90's default if unset), the reference window
    :data:`REFERENCE_CONV_WINDOW`, and the window ending at the run's last iteration. A
    run whose ``.wout`` does not print those iterations is non-evaluable, with the reason.
    """
    stage = c.wannier.wannierisation
    if stage is None:
        raise NonEvaluable(
            f"{c.construction_id}: no spread-minimisation record in the run"
        )
    settings = getattr(c.wannier, "wannierise_settings", None) or {}
    deck_window = _number(c.inputs.get("conv_window"))
    window = deck_window
    if window is None:
        window = _number(settings.get("conv_window"))
    if window is None or int(window) <= 1:
        return _spread_converged_from_trajectory(c, stage, settings, deck_window)
    return CheckOutcome(
        stage.converged,
        float(stage.n_iterations),
        "iterations",
        (
            ""
            if stage.converged
            else f"stopped at the iteration limit with delta {stage.final_delta:.2e}"
        ),
    )


def _spread_converged_from_trajectory(
    c: Construction, stage, settings: dict, deck_window: float | None
) -> CheckOutcome:
    """Wannier90's spread-convergence test, applied to the printed trajectory."""
    window = REFERENCE_CONV_WINDOW
    tol = _number(c.inputs.get("conv_tol"))
    tol_source = "deck"
    if tol is None:
        tol, tol_source = WANNIER90_DEFAULT_CONV_TOL_ANG2, "Wannier90 default"
    num_iter = _number(settings.get("num_iter"))
    if num_iter is None:
        num_iter = _number(c.inputs.get("num_iter"))
    if num_iter is None:
        num_iter = WANNIER90_DEFAULT_NUM_ITER
    num_iter = int(num_iter)
    no_test = (
        "the deck sets no conv_window"
        if deck_window is None
        else f"the deck sets conv_window = {int(deck_window)}"
    ) + ", so Wannier90 ran no convergence test on the spread functional"
    record = {
        "evidence": EVALUATED_FROM_TRAJECTORY,
        "conv_window_used": window,
        "conv_window_source": "reference window (the conv_window autowan writes into repaired decks)",
        "conv_tol_ang2": tol,
        "conv_tol_source": tol_source,
        "deck_conv_window": None if deck_window is None else int(deck_window),
        "num_iter": num_iter,
    }
    for key in ("conv_window", "conv_tol", "iprint", "num_print_cycles"):
        if key in settings:
            record[f"wout_{key}"] = settings[key]

    def unevaluable(reason: str) -> NonEvaluable:
        return NonEvaluable(
            f"{c.construction_id}: {no_test}, and the spread trajectory cannot be "
            f"judged over the reference window of {window} instead: {reason}"
        )

    steps = [s for s in getattr(stage, "trajectory", ()) or () if s[0] >= 1]
    iprint = settings.get("iprint")
    cycles = settings.get("num_print_cycles")
    if not steps:
        reason = "the .wout prints no per-iteration spread changes"
        if iprint is not None and iprint < 1:
            reason += f" (iprint = {int(iprint)})"
        raise unevaluable(reason)
    if num_iter < window:
        raise unevaluable(f"the run has only num_iter = {num_iter} iterations")
    expected = list(range(num_iter - window + 1, num_iter + 1))
    if steps[-1][0] != num_iter:
        raise unevaluable(
            f"the printed trajectory ends at iteration {steps[-1][0]}, not at the "
            f"run's last iteration {num_iter}"
        )
    tail = steps[-window:]
    printed = [s[0] for s in tail]
    if printed != expected:
        every = (
            f"every {int(cycles)} iterations (num_print_cycles = {int(cycles)})"
            if cycles is not None
            else "not at every iteration"
        )
        raise unevaluable(
            f"the .wout prints the trajectory {every}, so iterations "
            f"{expected[0]}-{expected[-1]} are not all printed (printed: {printed})"
        )
    deltas = [float(s[1]) for s in tail]
    figures = min(int(s[2]) for s in tail)
    # Half a unit in the last printed digit, relative to the value: a mantissa in
    # [0.1, 1) with n digits is uncertain by 0.5e-n absolute, at most 5e-n relative.
    uncertainty = [5.0 * 10.0 ** (-figures) * abs(d) for d in deltas]
    record.update(
        {
            "window_iterations": printed,
            "window_deltas_ang2": deltas,
            "max_abs_delta_ang2": max(abs(d) for d in deltas),
            "delta_significant_figures": figures,
        }
    )
    clearly_above = any(abs(d) - u > tol for d, u in zip(deltas, uncertainty))
    clearly_within = all(abs(d) + u <= tol for d, u in zip(deltas, uncertainty))
    if not clearly_above and not clearly_within:
        raise unevaluable(
            f"|delta Omega| over iterations {expected[0]}-{expected[-1]} "
            f"({', '.join(f'{d:.3e}' for d in deltas)} Å²) sits at conv_tol = "
            f"{tol:.3e} Å² within the {figures} significant figures the .wout prints"
        )
    passed = all(abs(d) <= tol for d in deltas)
    largest = max(abs(d) for d in deltas)
    note = (
        f"{EVALUATED_FROM_TRAJECTORY} ({no_test.split(', so')[0]}): "
        f"max |delta Omega| over iterations {expected[0]}-{expected[-1]} = "
        f"{largest:.2e} Å² {'<=' if passed else '>'} conv_tol {tol:.1e} Å² "
        f"({tol_source}), reference window {window}"
    )
    return CheckOutcome(
        passed, float(stage.n_iterations), "iterations", note, record=record
    )


def _wannier_count_matches_request(c: Construction) -> CheckOutcome:
    got = c.wannier.num_wann
    want = c.inputs.get("num_wann")
    if got is None or want is None:
        raise NonEvaluable(
            f"{c.construction_id}: the requested Wannier count is not recorded"
        )
    return CheckOutcome(int(got) == int(want), float(got), "functions")


def _metal_frozen_window_contains_fermi_level(c: Construction) -> CheckOutcome:
    """Whether a metal's declared frozen window contains its Fermi level.

    A frozen window below or above E_F releases the states at E_F to the
    disentanglement. The 2026-09-13 construction
    defective_fe_bcc_mlwf_frozen_window_below_ef, with a frozen window of -4..-2 eV,
    flipped no check: every window-reading check judged inside the declaration and
    none asked where it sat. Containment is discrete and carries no constant. It is
    meaningful for metals only; an insulator's window routinely stops at the valence
    maximum, with no states at E_F to protect.

    Read absolute eV from the run's frozen_window_ev when present, otherwise from
    dis_froz_max and dis_froz_min in the construction inputs, including Fortran d
    exponents. Without a minimum, Wannier90 takes the outer window's lower bound,
    below E_F, represented here by -infinity. With no frozen window there is no
    claim to contradict, as there is no convergence to demand when disentanglement
    was not needed; an unreadable declaration instead supplies no evaluable evidence.
    """
    ef = c.parent.fermi_energy_ev
    bounds = getattr(c.wannier, "frozen_window_ev", None)
    if bounds is None:
        if c.inputs.get("dis_froz_max") is None:
            # Wannier90 defines a frozen window only through dis_froz_max; a lone
            # dis_froz_min freezes nothing.
            return CheckOutcome(True, 0.0, "eV", "no frozen window was declared")
        bounds = (c.inputs.get("dis_froz_min", -np.inf), c.inputs["dis_froz_max"])
    try:
        lo, hi = bounds
        lo = float(str(lo).lower().replace("d", "e"))
        hi = float(str(hi).lower().replace("d", "e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise NonEvaluable("frozen window has an unreadable bound") from exc
    if np.isnan(lo) or np.isnan(hi):
        raise NonEvaluable("frozen window has an unreadable bound")
    passed = bool(lo <= ef <= hi)
    return CheckOutcome(
        passed,
        max(lo - ef, ef - hi, 0.0),
        "eV",
        (
            ""
            if passed
            else f"frozen window {lo:.2f}..{hi:.2f} eV does not contain E_F = {ef:.2f} eV"
        ),
    )


def _hopping_shells_inside_representable_supercell(c: Construction) -> CheckOutcome:
    """A hopping outside the Wigner-Seitz supercell of the k-mesh is an image.

    The mesh fixes how far the real-space Hamiltonian can reach: a lattice vector
    beyond the supercell conjugate to it is not independent information but a
    periodic repeat, and keeping it double-counts. The bound is discrete and set by
    the mesh alone, which is what makes it the right instrument where a scalar band
    error is not.

    The test is a genuine Wigner-Seitz one — is this vector at least as close to the
    origin as to any other superlattice point — and not a bound on the components of
    ``R``. Those differ whenever the primitive vectors are not orthogonal: on a
    body-centred cubic lattice the componentwise box rejects a large share of the
    shells Wannier90 itself constructed, which reads as a defect in the construction
    and is a defect in the test.
    """
    rvec = np.asarray(c.require("hr_rvectors"), dtype=float)
    mesh = c.inputs.get("mp_grid")
    if mesh is None:
        raise NonEvaluable(f"{c.construction_id}: the k-mesh is not recorded")

    lattice = c.parent.lattice_ang
    superlattice = np.asarray(mesh, dtype=float)[:, None] * lattice
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    translations = shifts @ superlattice  # (27, 3), including the null shift
    cart = rvec @ lattice  # (n_R, 3)
    distances = np.linalg.norm(cart[:, None, :] - translations[None, :, :], axis=2)
    tolerance = 1e-6 * max(1.0, float(np.linalg.norm(superlattice, axis=1).max()))
    outside = int(
        np.count_nonzero(
            distances.min(axis=1)
            < distances[:, shifts.any(axis=1) == False][:, 0] - tolerance
        )
    )
    return CheckOutcome(
        outside == 0,
        float(outside),
        "shells",
        (
            ""
            if outside == 0
            else f"{outside} lattice vectors sit nearer another {np.asarray(mesh).tolist()} "
            "superlattice point than the origin"
        ),
    )


def _frontier_manifold_straddles_fermi_level(c: Construction) -> CheckOutcome:
    """The subspace must bracket the Fermi level wherever the parent does.

    A construction that misses the frontier cannot describe anything the Fermi surface
    governs, whatever its spread or its band distance says.

    Measured against the parent rather than against unity, because bracketing
    everywhere is not a property every parent has. A nearly-free-electron metal whose
    single valence band rises above the Fermi level over part of the zone has k-points
    with no occupied state at all, and demanding that the model bracket there asks it
    to invent one. Sodium's parent brackets at 0.625 of its k-points; the repair that
    reproduced it exactly was refused for scoring 0.625, which is the gate failing and
    not the construction.
    """
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    deg = c.require("hr_degeneracies")
    kf = cartesian_to_fractional_k(
        c.parent.kpoints_cart, c.parent.lattice_ang, c.parent.alat_ang
    )
    bands = interpolate(h, rvec, deg, kf)
    ef = c.parent.fermi_energy_ev
    model = np.any(bands <= ef, axis=1) & np.any(bands >= ef, axis=1)
    parent_bands, exclusion_note = parent_bands_for_check(
        c, c.inputs.get("spin_index", 0)
    )
    parent = np.any(parent_bands <= ef, axis=1) & np.any(parent_bands >= ef, axis=1)
    expected = int(np.count_nonzero(parent))
    if expected == 0:
        raise NonEvaluable(
            f"{c.construction_id}: the parent brackets the Fermi level nowhere, so "
            f"there is no frontier for the model to be judged against"
        )
    missed = int(np.count_nonzero(parent & ~model))
    fraction = 1.0 - missed / expected
    return CheckOutcome(
        missed == 0,
        fraction,
        "fraction of the parent's frontier k-points",
        (
            exclusion_note.removeprefix("; ")
            if missed == 0
            else (
                f"the model does not bracket the Fermi level at {missed} of the "
                f"{expected} k-points where the parent does" + exclusion_note
            )
        ),
    )


#: Lattice-vector range searched for the nearest atom image of a centre, after both are
#: wrapped into the unit cell. One shell suffices for a reduced cell; two cover oblique
#: cells (the search is a few thousand distances, so the margin costs nothing).
_IMAGE_SEARCH = 2


def nearest_atom_distance_ang(c: Construction) -> np.ndarray:
    """Each Wannier centre's distance to the nearest atom, over every lattice image.

    Centres and atoms are both wrapped into the unit cell ``[0, 1)`` before the
    nearest image is searched, so neither the frame a parent writes its atoms in (a
    deck may place an atom at ``-0.25``, or at an image two cells away) nor the image
    of the cell a centre was reported in (Wannier90 reports centres unwrapped) can
    change the distance: a function sitting on an atom measures 0 whichever image
    either was written in.
    """
    centres = np.asarray(c.require("centres_ang"), dtype=float)
    lattice = np.asarray(c.parent.lattice_ang, dtype=float)
    frac = c.parent.positions_frac
    if frac is None or len(frac) == 0:
        raise NonEvaluable(f"{c.construction_id}: parent records no atomic positions")
    atoms = np.asarray(frac, dtype=float)
    atoms = atoms - np.floor(atoms)
    fc = np.linalg.solve(lattice.T, centres.T).T
    fc = fc - np.floor(fc)
    r = range(-_IMAGE_SEARCH, _IMAGE_SEARCH + 1)
    shifts = np.array([[i, j, k] for i in r for j in r for k in r], dtype=float)
    images = (atoms[:, None, :] + shifts[None, :, :]).reshape(-1, 3) @ lattice
    cart = fc @ lattice
    return np.min(np.linalg.norm(cart[:, None, :] - images[None, :, :], axis=2), axis=1)


def _centres_sit_on_atomic_sites(c: Construction) -> CheckOutcome:
    """The furthest Wannier centre's distance to its nearest atomic site.

    Two optional checks read this one measurement at different thresholds: the
    pre-profile ``centres_sit_on_atomic_sites`` (half the nearest-neighbour distance)
    and ``centres_on_atoms_atom_centred`` (0.1 Angstrom). Neither is in the default
    profile: atom-centring is a requirement of uses that label functions by atom and
    orbital, not of band interpolation, and a bond-centred function (SCDM's, typically)
    sits at exactly half the nearest-neighbour distance by construction. The distance
    is to the nearest lattice image (:func:`nearest_atom_distance_ang`).
    """
    worst = float(np.max(nearest_atom_distance_ang(c)))
    return CheckOutcome(passed=None, metric_value=worst, metric_units="Ang")


def _home_cell_box(c: Construction) -> tuple[np.ndarray, np.ndarray]:
    """Fractional bounds of the home cell, widened to hold every atom as given.

    The home cell is the parallelepiped spanned by the lattice vectors, ``[0, 1]`` along
    each, extended along an axis wherever the parent places an atom outside it -- a deck
    is free to write positions as ``-0.25`` or ``1.0``, and a function centred on such
    an atom has not drifted anywhere.
    """
    frac = c.parent.positions_frac
    if frac is None or len(frac) == 0:
        raise NonEvaluable(f"{c.construction_id}: parent records no atomic positions")
    frac = np.asarray(frac, dtype=float)
    return np.minimum(0.0, frac.min(axis=0)), np.maximum(1.0, frac.max(axis=0))


def distance_outside_home_cell_ang(c: Construction) -> np.ndarray:
    """Each Wannier centre's Cartesian distance from the home cell; zero inside it.

    The exact distance from a point to a (possibly oblique) parallelepiped, computed as
    a bounded linear least-squares problem: the nearest point of the cell is
    ``f @ lattice`` with ``f`` inside the fractional box.
    """
    from scipy.optimize import lsq_linear

    centres = np.asarray(c.require("centres_ang"), dtype=float)
    lattice = np.asarray(c.parent.lattice_ang, dtype=float)
    lo, hi = _home_cell_box(c)
    out = []
    for centre in centres:
        frac = np.linalg.solve(lattice.T, centre)
        if np.all(frac >= lo - 1e-12) and np.all(frac <= hi + 1e-12):
            out.append(0.0)
            continue
        fit = lsq_linear(lattice.T, centre, bounds=(lo, hi), method="bvls")
        out.append(float(np.linalg.norm(lattice.T @ fit.x - centre)))
    return np.asarray(out)


def _centres_stay_near_home_cell(c: Construction) -> CheckOutcome:
    """How far the furthest Wannier centre has wandered from every atom.

    A weak drift guard, in the default profile in place of atom-centring. A function
    seeded on an atom, or on a bond or interstice next to one, has no reason to wander
    away from the atoms; a centre further than one nearest-neighbour distance from
    every atom (every lattice image included) belongs to no atom or bond, which is what
    a spread minimisation that has run away looks like -- into a slab's vacuum, for
    instance. The threshold (one nearest-neighbour distance) is generous on purpose: a
    bond-centred function sits half a bond from its atoms and must pass.

    Measured to the nearest atom image since v5 (2026-10-09). The v1-v4 guard measured
    the distance outside the home cell, widened to the atoms as the parent wrote them;
    a function sitting exactly on an atom whose image the parent or Wannier90 reported
    two cells away (VASP TiC, SrO, SrTiO3, TiO2, V3Si, MnO) failed it at 0.000 Angstrom
    from the atom. That distance is kept in the record as a diagnostic.
    """
    distances = nearest_atom_distance_ang(c)
    worst = int(np.argmax(distances))
    try:
        outside = distance_outside_home_cell_ang(c)
        outside_max = float(np.max(outside))
    except NonEvaluable:  # pragma: no cover - same positions requirement as above
        outside_max = None
    return CheckOutcome(
        passed=None,
        metric_value=float(distances[worst]),
        metric_units="Ang",
        note=(
            f"Wannier function {worst + 1} lies {distances[worst]:.3f} Ang from the "
            "nearest atom (any lattice image)"
        ),
        record={
            "measure": "distance to the nearest atom, every lattice image",
            "furthest_outside_home_cell_ang": outside_max,
        },
    )


# ---------------------------------------------------------------------------
# scalar checks — each against a scale the material supplies
# ---------------------------------------------------------------------------


def _occupied_manifold_is_represented(c: Construction) -> CheckOutcome:
    """Fraction of occupied parent states with no band of the model near them.

    This applies to metals, and has to. Withholding the band-distance check from a
    metal is right about *where*: above the region a construction was constrained to
    reproduce, the states it left outside are a choice and not an error. It is wrong
    about the occupied manifold. Whatever a Wannier model is later used for — an
    exchange coupling, a transport integral — is built from the states below the Fermi
    level, and a model with no band anywhere near four fifths of them is not describing
    the material, however localized its functions are and however smoothly its own
    bands disperse.

    Measured as coverage rather than as an average distance. A mean over nearest
    neighbours cannot tell a model that reproduces every occupied state to a
    hundredth of a millielectronvolt from one that misses most of them entirely: the
    unmatched states dominate the mean, and two very different constructions return
    almost the same number. Counting how many states have a partner separates them at
    once.

    Under the magnetic_subspace target the declaration instead identifies the
    manifold by a relative energy window and majority magnetic-d character. Every
    state it selects, occupied or empty, must have a nearby model band, with distance
    measured against that magnetic manifold's own bandwidth over all k and both
    spins. This is the same coverage check and missing-state fraction threshold;
    only the declared set of parent states and its energy scale change.
    """
    if c.target.name == "magnetic_subspace":
        return _magnetic_subspace_is_represented(c)
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    deg = c.require("hr_degeneracies")
    spin = c.inputs.get("spin_index", 0)
    parent, exclusion_note = parent_bands_for_check(c, spin)
    ef = c.parent.fermi_energy_ev

    occupied_mask = parent <= ef
    if not occupied_mask.any():
        # An honoured exclusion can leave an originally occupied parent with no
        # occupied states to represent; retain the distinction in the refusal.
        reason = (
            "no occupied states to represent"
            if exclusion_note.startswith("; bands ")
            else "parent has no occupied states"
        )
        raise NonEvaluable(f"{c.construction_id}: {reason}")

    kf = cartesian_to_fractional_k(
        c.parent.kpoints_cart, c.parent.lattice_ang, c.parent.alat_ang
    )
    model = interpolate(h, rvec, deg, kf)

    # A partner must be nearer than a fixed fraction of this material's own occupied
    # bandwidth: an absolute tolerance in electronvolts would be a constant borrowed
    # from whichever material it was chosen on.
    tolerance = 0.01 * SCALES["occupied_bandwidth_ev"](c)
    # Only the bands the model was asked to represent are counted: a deck that excludes
    # an isolated block below them (the frontier's semicore shells) is not asked for
    # those. Any other exclusion is not honoured -- see represented_parent_bands.
    # The tolerance stays 1 % of the full occupied bandwidth either way.
    missing = 0
    total = 0
    for model_k, parent_k in zip(model, parent):
        below = parent_k[parent_k <= ef]
        if not below.size:
            continue
        distance = np.min(np.abs(below[:, None] - model_k[None, :]), axis=1)
        total += below.size
        missing += int((distance > tolerance).sum())
    if total == 0:
        raise NonEvaluable(f"{c.construction_id}: no occupied states to represent")

    fraction = missing / total
    return CheckOutcome(
        passed=None,
        metric_value=fraction,
        metric_units="fraction of occupied states",
        note=f"{missing} of {total} lie further than {tolerance*1000:.0f} meV from any "
        "band of the model" + exclusion_note,
    )


def _deck_energy_ev(value) -> float | None:
    """Read a deck energy, including Wannier90's Fortran exponent spelling."""
    if value is None:
        return None
    try:
        return float(str(value).lower().replace("d", "e"))
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    """Show a frozen-window edge, or a dash when the deck declares none."""
    return "-" if value is None else f"{value:.4f}"


def _magnetic_subspace_is_represented(c: Construction) -> CheckOutcome:
    """Count missing partners in the declared magnetic manifold on the parent mesh.

    A d-only model may legitimately omit ligand and semicore states under this
    target, but it must still reproduce the dispersion of the states it declares.
    The magnetic-d floor fixes membership for every declaration, and the tolerance
    follows that counted manifold's own spread rather than the depth of other shells.
    """
    spin = c.inputs.get("spin_index", 0)
    mask = magnetic_subspace_mask(c, spin)
    if not mask.any():
        raise NonEvaluable(
            f"{c.construction_id}: no parent state inside the declared window "
            "clears the magnetic-d floor"
        )
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    deg = c.require("hr_degeneracies")
    parent = c.parent.eigenvalues_ev[spin]
    kf = cartesian_to_fractional_k(
        c.parent.kpoints_cart, c.parent.lattice_ang, c.parent.alat_ang
    )
    model = interpolate(h, rvec, deg, kf)
    tolerance = MAGNETIC_SUBSPACE_TOLERANCE.value(c)
    missing = total = 0
    for model_k, parent_k, mask_k in zip(model, parent, mask):
        selected = parent_k[mask_k]
        if not selected.size:
            continue
        distance = np.min(np.abs(selected[:, None] - model_k[None, :]), axis=1)
        total += selected.size
        missing += int((distance > tolerance).sum())
    if total == 0:
        raise NonEvaluable(
            f"{c.construction_id}: no parent state inside the declared window "
            "clears the magnetic-d floor"
        )
    lo, hi = c.target.window_ev
    ef = c.parent.fermi_energy_ev
    floor = MAGNETIC_D_WEIGHT_FLOOR.value(c)
    return CheckOutcome(
        passed=None,
        metric_value=missing / total,
        metric_units="fraction of magnetic-subspace states",
        note=f"{missing} of {total} magnetic-subspace states with magnetic-d weight "
        f">= {floor:g} inside {ef + lo:.2f}..{ef + hi:.2f} eV lie further than "
        f"{tolerance * 1000:.0f} meV from any band of the model",
    )


def _model_bands_correspond_to_parent_bands(c: Construction) -> CheckOutcome:
    """Fraction of the model's own bands that sit where the parent has nothing.

    The companion to the coverage check, and the direction that catches what a reader
    of a band plot notices first. Coverage asks whether the parent's states survived
    into the model; this asks whether the model invented states the parent does not
    have. A construction can score perfectly on the first and badly on the second: a
    subspace pinned exactly over the occupied manifold and left free above it will
    reproduce every occupied state and then send its remaining freedom wherever the
    functional prefers, which is nowhere in particular.

    Restricted to the target's nu when a whole_occupied_manifold construction declares
    one, relative to the parent's Fermi level. Otherwise it uses the deck's own frozen window, or
    E_F + REPRODUCED_MARGIN_EV when no upper edge is declared. A deck narrower than
    nu is refused with the contractual reason frozen_window_narrower_than_target,
    kept apart from a band-reproduction failure; the comparison over nu is still
    recorded as a diagnostic. Above the window a disentangled subspace is
    free by construction: its bands there are projections rather than eigenvalues and
    are not claimed to match anything. Judged over the whole computed range instead,
    every disentangled construction looks broken, including ones nobody disputes — the
    first version of this check refused seven of eight campaign materials, silicon and
    diamond among them.
    """
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    deg = c.require("hr_degeneracies")
    spin = c.inputs.get("spin_index", 0)
    parent, exclusion_note = parent_bands_for_check(c, spin)

    kf = cartesian_to_fractional_k(
        c.parent.kpoints_cart, c.parent.lattice_ang, c.parent.alat_ang
    )
    model = interpolate(h, rvec, deg, kf)

    tolerance = 0.01 * SCALES["occupied_bandwidth_ev"](c)
    ef = c.parent.fermi_energy_ev
    target = getattr(c, "target", None)
    # Only a whole_occupied_manifold target declares a reproduction window. A
    # magnetic_subspace window bounds the declared manifold for the coverage check and
    # leaves this check on the deck's own window, as it was before the two merged.
    window = (
        target.window_ev
        if target is not None and target.name == "whole_occupied_manifold"
        else None
    )
    froz = c.inputs.get("dis_froz_max")
    deck_hi = _deck_energy_ev(froz)
    deck_lo = _deck_energy_ev(c.inputs.get("dis_froz_min"))
    narrower = False
    if window is None:
        upper = deck_hi if deck_hi is not None else ef + REPRODUCED_MARGIN_EV
        lo, hi = float(parent.min()), upper
        record = {
            "target_window_source": "deck",
            "target_window_ev": None,
            "reproduction_window_ev": [lo, hi],
            "deck_frozen_window_ev": [deck_lo, deck_hi],
        }
    else:
        t_lo, t_hi = window
        hi = ef + t_hi
        lo = float(parent.min()) if t_lo == -math.inf else ef + t_lo
        # Without dis_froz_max the deck declares no energy frozen window.
        narrower = froz is not None and (
            (deck_hi is not None and deck_hi < hi)
            or (math.isfinite(t_lo) and deck_lo is not None and deck_lo > lo)
        )
        record = {
            "target_window_source": "target",
            "target_window_ev": [None if t_lo == -math.inf else t_lo, t_hi],
            "reproduction_window_ev": [lo, hi],
            "deck_frozen_window_ev": [deck_lo, deck_hi],
        }
    spurious = 0
    total = 0
    for model_k, parent_k in zip(model, parent):
        inside = model_k[(model_k >= lo) & (model_k <= hi)]
        if not inside.size:
            continue
        distance = np.min(np.abs(inside[:, None] - parent_k[None, :]), axis=1)
        total += inside.size
        spurious += int((distance > tolerance).sum())
    if narrower:
        diagnostic = (
            f"{spurious} of {total} model bands sit further than "
            f"{tolerance*1000:.0f} meV from any band the parent calculation has"
            if total
            else "no model band falls inside it"
        )
        # The shortfall is stated exactly: a deck printed a hair below the target by
        # its writer's rounding is refused too, and the reason has to show that it was
        # a hair rather than leave two equal-looking numbers side by side.
        short = []
        if deck_hi is not None and deck_hi < hi:
            short.append(f"{hi - deck_hi:.3g} eV short at the top")
        if math.isfinite(t_lo) and deck_lo is not None and deck_lo > lo:
            short.append(f"{deck_lo - lo:.3g} eV short at the bottom")
        return CheckOutcome(
            passed=None,
            metric_value=spurious / total if total else float("nan"),
            metric_units="fraction of model bands",
            note=(
                f"{FROZEN_WINDOW_NARROWER_THAN_TARGET}: the deck's frozen window "
                f"{_fmt(deck_lo)}..{_fmt(deck_hi)} eV is narrower than the target "
                f"window {lo:.4f}..{hi:.4f} eV (E_F {t_hi:+.4f} eV at the top; "
                f"{', '.join(short)}); over the target window, {diagnostic}"
                + exclusion_note
            ),
            refusal=FROZEN_WINDOW_NARROWER_THAN_TARGET,
            record=record,
        )
    if total == 0:
        raise NonEvaluable(
            f"{c.construction_id}: no model band falls inside the reproduced window"
        )

    fraction = spurious / total
    return CheckOutcome(
        passed=None,
        metric_value=fraction,
        metric_units="fraction of model bands",
        note=f"{spurious} of {total} inside {lo:.2f}..{hi:.2f} eV sit further than "
        f"{tolerance*1000:.0f} meV from any band the parent calculation has"
        + exclusion_note,
        record=record,
    )


def _band_distance_from_parent(c: Construction) -> CheckOutcome:
    """Distance between the interpolated bands and the parent's, near the Fermi level.

    Restricted to the occupied manifold and a little above it: the construction was
    never asked to reproduce states above its outer window, so a distance measured
    over the whole band set reports the disentanglement's freedom as error.
    """
    h = c.require("hr_matrix")
    rvec = c.require("hr_rvectors")
    deg = c.require("hr_degeneracies")
    spin = c.inputs.get("spin_index", 0)
    parent, exclusion_note = parent_bands_for_check(c, spin)
    ef = c.parent.fermi_energy_ev
    occupied = parent[parent <= ef]
    if occupied.size == 0:
        raise NonEvaluable(f"{c.construction_id}: parent has no occupied states")
    window = (
        float(occupied.min()),
        ef + 0.1 * float(np.ptp(parent[parent <= ef + 1e-9])),
    )

    kf = cartesian_to_fractional_k(
        c.parent.kpoints_cart, c.parent.lattice_ang, c.parent.alat_ang
    )
    bands = interpolate(h, rvec, deg, kf)
    comparison = compare_to_parent(bands, parent, window_ev=window)
    if comparison.n_states_compared == 0:
        raise NonEvaluable(
            f"{c.construction_id}: no parent state falls in {window[0]:.2f}..{window[1]:.2f} eV"
        )
    return CheckOutcome(
        passed=None,
        metric_value=comparison.rms_error_ev,
        metric_units="eV",
        note=f"{comparison.n_states_compared} states compared over "
        f"{window[0]:.2f}..{window[1]:.2f} eV" + exclusion_note,
    )


def _hopping_decays_with_distance(c: Construction) -> CheckOutcome:
    """The outermost hoppings must be small beside the on-site scale.

    This is what licenses truncating the model at all, and it is self-referential:
    the construction supplies both the distant hopping and the scale it is compared
    against, so nothing is imported from another material.
    """
    h = np.asarray(c.require("hr_matrix"))
    rvec = np.asarray(c.require("hr_rvectors"), dtype=float)
    lattice = c.parent.lattice_ang
    distances = np.linalg.norm(rvec @ lattice, axis=1)
    if distances.max() <= 0:
        raise NonEvaluable(
            f"{c.construction_id}: the Hamiltonian has only the on-site block"
        )
    outer = distances >= np.quantile(distances, 0.9)
    return CheckOutcome(
        passed=None,
        metric_value=float(np.abs(h[outer]).max()),
        metric_units="eV",
        note=f"largest element among the outermost {int(outer.sum())} shells",
    )


# ---------------------------------------------------------------------------
# the core set
# ---------------------------------------------------------------------------

#: The two convergence checks: whether each minimisation met its own stopping criterion.
#: They left the ``default`` profile on 2026-10-09 and are
#: the optional ``convergence`` group (``--add-check convergence``). The gate judges the
#: construction it is handed -- its bands, its hoppings, its centres -- not how the
#: optimiser stopped: a protocol that stops at a fixed iteration count by design (the
#: JARVIS VASP deck, ``dis_num_iter = num_iter = 100``; the VASP SCDM protocol, no
#: ``conv_window``) can still produce a construction every quality check accepts.
CONVERGENCE_CHECKS: tuple[Check, ...] = (
    Check(
        "subspace_selection_converged",
        "whether disentanglement met its criterion",
        "discrete",
        "self",
        _subspace_selection_converged,
        description="the disentanglement (subspace selection) met its own convergence criterion, or was not needed",
    ),
    Check(
        "spread_minimisation_converged",
        "whether the spread functional met its criterion",
        "discrete",
        "self",
        _spread_minimisation_converged,
        description="the spread minimisation met its own convergence criterion: Wannier90's own test when the deck sets a convergence window, otherwise the same test (|change of spread| <= conv_tol over 3 consecutive iterations at the end of the run) applied to the printed trajectory",
    ),
)


#: The ``default`` gate profile, in report order. ``centres_sit_on_atomic_sites`` left it
#: on 2026-10-07 for the drift guard ``centres_stay_near_home_cell``;
#: it stays registered as an optional check (:data:`CENTRES_SIT_ON_ATOMIC_SITES`). The
#: two convergence checks left it on 2026-10-09 (:data:`CONVERGENCE_CHECKS`).
CORE_CHECKS: tuple[Check, ...] = (
    Check(
        "wannier_count_matches_request",
        "constructed functions against those asked for",
        "discrete",
        "structure",
        _wannier_count_matches_request,
        description="the run built as many Wannier functions as the deck asked for",
    ),
    Check(
        "hopping_shells_inside_representable_supercell",
        "lattice vectors beyond the supercell the k-mesh can represent",
        "discrete",
        "structure",
        _hopping_shells_inside_representable_supercell,
        description="every hopping vector of the real-space Hamiltonian lies inside the Wigner-Seitz supercell the k-mesh can represent",
    ),
    Check(
        "frontier_manifold_straddles_fermi_level",
        "fraction of k-points where the model brackets the Fermi level",
        "discrete",
        "dft_bands",
        _frontier_manifold_straddles_fermi_level,
        description="the model's bands bracket the Fermi level at every mesh k-point where the parent's bands do",
    ),
    Check(
        "centres_stay_near_home_cell",
        "furthest any Wannier centre lies from every atom (any lattice image)",
        "scalar_vs_scale",
        "structure",
        _centres_stay_near_home_cell,
        threshold=ScaleAwareThreshold(
            fraction=1.0,
            scale="nearest_neighbour_distance_ang",
            calibration="one nearest-neighbour distance of this structure, geometric and "
            "set a priori (2026-10-07), not fitted. An atom-centred function lies on its "
            "atom (0); a bond-centred one lies half a bond from its atoms; one full bond "
            "is twice that margin, so only a centre attached to no atom or bond fails. "
            "Measured to the nearest atom image since 2026-10-09 (until then: outside "
            "the home cell, which failed functions on atoms written as distant images).",
        ),
        description="no Wannier centre lies more than one nearest-neighbour distance "
        "from every atom, lattice images included (the drift guard)",
    ),
    Check(
        "occupied_manifold_is_represented",
        "occupied parent states with no band of the model near them",
        "scalar_vs_scale",
        "dft_bands",
        _occupied_manifold_is_represented,
        threshold=ScaleAwareThreshold(
            fraction=0.05,
            scale="occupied_state_fraction",
            calibration="at most five percent of occupied states may be unrepresented; "
            "a fraction of a count, so nothing is borrowed from another "
            "material. Set a priori, not fitted.",
        ),
        description="at most 5 % of the occupied parent states (on the construction mesh) lack a model band within 1 % of the occupied bandwidth",
    ),
    Check(
        "model_bands_correspond_to_parent_bands",
        "model bands sitting where the parent calculation has none",
        "scalar_vs_scale",
        "dft_bands",
        _model_bands_correspond_to_parent_bands,
        threshold=ScaleAwareThreshold(
            fraction=0.20,
            scale="occupied_state_fraction",
            calibration="at most a fifth of the model's bands may correspond to "
            "nothing in the parent; a fraction of a count, set a priori. "
            "Loose on purpose: a disentangled subspace is free above the "
            "frozen window by construction, and the check is for a model "
            "that has largely stopped describing the material, not for "
            "one exercising the freedom it was given.",
        ),
        description="at most 20 % of the model's bands inside the reproduction window sit away from every parent band, and the deck's frozen window is not narrower than a declared target window",
    ),
    Check(
        "band_distance_from_parent",
        "band distance from the parent calculation near the Fermi level",
        "scalar_vs_scale",
        "dft_bands",
        _band_distance_from_parent,
        applies_to=("gapped",),
        threshold=ScaleAwareThreshold(
            fraction=0.01,
            scale="occupied_bandwidth_ev",
            calibration="one percent of this material's own occupied bandwidth; "
            "fixed a priori as a relative tolerance, not fitted to any material",
        ),
        description="for a gapped parent, the RMS distance between model and parent bands over the occupied manifold is within 1 % of the occupied bandwidth",
    ),
    Check(
        "hopping_decays_with_distance",
        "largest hopping in the outermost shells",
        "scalar_vs_scale",
        "self",
        _hopping_decays_with_distance,
        threshold=ScaleAwareThreshold(
            fraction=0.01,
            scale="onsite_hopping_scale_ev",
            calibration="one percent of this construction's own on-site scale; "
            "self-referential, so nothing is imported from another material",
        ),
        description="the largest hopping in the outermost 10 % of shells is within 1 % of the largest on-site element",
    ),
    Check(
        "metal_frozen_window_contains_fermi_level",
        "whether a declared frozen window contains the Fermi level",
        "discrete",
        "dft_bands",
        _metal_frozen_window_contains_fermi_level,
        applies_to=("metal",),
        description="for a metal, a declared frozen window contains the Fermi level",
    ),
)


#: The pre-profile atom-centring check at half the nearest-neighbour distance. Optional:
#: in no named profile, kept so the gate that issued the stored verdicts can be rebuilt
#: (``--add-check centres_sit_on_atomic_sites``).
CENTRES_SIT_ON_ATOMIC_SITES = Check(
    "centres_sit_on_atomic_sites",
    "furthest Wannier centre from any atom",
    "scalar_vs_scale",
    "structure",
    _centres_sit_on_atomic_sites,
    threshold=ScaleAwareThreshold(
        fraction=0.5,
        scale="nearest_neighbour_distance_ang",
        calibration="half the nearest-neighbour distance: beyond it a centre is "
        "closer to a different site than to its own. Geometric, not fitted.",
    ),
    description="every Wannier centre lies within half a nearest-neighbour distance of "
    "an atom",
)

#: Atom-centred basis: every centre within 0.1 Angstrom of an atom (the ``atom_centred_basis``
#: profile). The requirement of uses that label functions by atom and orbital.
CENTRES_ON_ATOMS_ATOM_CENTRED = Check(
    "centres_on_atoms_atom_centred",
    "furthest Wannier centre from any atom",
    "scalar_vs_scale",
    "structure",
    _centres_sit_on_atomic_sites,
    threshold=ScaleAwareThreshold(
        fraction=0.1,
        scale="published_absolute_ang",
        calibration="0.1 Angstrom: the distance d_NN from a Wannier centre to its nearest "
        "atom under which Qiao, Pizzi & Marzari 2023 (arXiv:2303.07877) call a function "
        "atom-centred. A published absolute criterion, adopted with its source so that "
        "'atom-centred' means what the paper means; not fitted here.",
    ),
    description="every Wannier centre lies within 0.1 Angstrom of an atom, so each "
    "function can be labelled by its atom (an atom-centred basis, as DFT+U, DMFT or "
    "exchange-coupling uses need)",
)

#: The ``default`` profile as it stood from 2026-10-07 to 2026-10-09 (tags v1 to v4): the
#: convergence checks first, then :data:`CORE_CHECKS`. It is no profile -- it exists so
#: that verdicts stored under those tags can be reproduced (the same check-list hash).
PRE_V5_CORE_CHECKS: tuple[Check, ...] = (*CONVERGENCE_CHECKS, *CORE_CHECKS)

#: The core set as it stood before gate profiles (to 2026-10-07): the pre-v5 default with
#: ``centres_sit_on_atomic_sites`` in the drift guard's report position. It is no profile
#: -- it exists so that stored verdicts can be reproduced byte for byte.
PRE_PROFILE_CORE_CHECKS: tuple[Check, ...] = tuple(
    CENTRES_SIT_ON_ATOMIC_SITES if c.check_id == "centres_stay_near_home_cell" else c
    for c in PRE_V5_CORE_CHECKS
)

#: Every optional check this module ships, by identifier.
OPTIONAL_CHECKS: tuple[Check, ...] = (
    *CONVERGENCE_CHECKS,
    CENTRES_ON_ATOMS_ATOM_CENTRED,
    CENTRES_SIT_ON_ATOMIC_SITES,
)
