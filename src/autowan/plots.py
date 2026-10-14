"""Figures a model is shown, drawn so that what matters is visible in the image.

A scalar band distance answers "how far, on average". It cannot say *where* a
construction went wrong, and some failures have no scalar at all: a manifold that
detaches from the parent bands over one region of the zone, a spurious flat band, an
interpolation that is faithful below the Fermi level and free above it. Those are
visible immediately and are why a review stage looks at pictures.

The drawing is therefore not decoration. The parent bands and the model's own are
plotted on one axis with the Fermi level marked and the disentanglement windows shaded,
because a reviewer — human or otherwise — cannot judge a deviation without knowing
whether the construction was ever asked to reproduce that region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .interpolation import cartesian_to_fractional_k, interpolate


@dataclass(frozen=True)
class BandFigure:
    path: Path
    caption: str


#: A walk through the zone in fractional coordinates. These points exist for every
#: lattice, so the path is drawn without claiming crystallographic labels the code has
#: not verified — a mislabelled high-symmetry point is worse than an unlabelled one.
DEFAULT_PATH: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("(0,0,0)", (0.0, 0.0, 0.0)),
    ("(½,0,0)", (0.5, 0.0, 0.0)),
    ("(½,½,0)", (0.5, 0.5, 0.0)),
    ("(½,½,½)", (0.5, 0.5, 0.5)),
    ("(0,0,0)", (0.0, 0.0, 0.0)),
)


def _sample_path(path, per_segment: int = 80):
    """Dense fractional k-points along the walk, with the corner positions."""
    points, ticks = [], [0]
    for (_, start), (_, end) in zip(path[:-1], path[1:]):
        seg = np.linspace(start, end, per_segment, endpoint=False)
        points.append(seg)
        ticks.append(ticks[-1] + per_segment)
    points.append(np.array([path[-1][1]]))
    return np.concatenate(points), ticks, [label for label, _ in path]


def draw_band_comparison(
    parent,
    wannier,
    out_path: str | Path,
    *,
    spin_index: int = 0,
    dis_win_max: float | None = None,
    dis_froz_max: float | None = None,
    title: str = "",
) -> BandFigure:
    """Two panels: what the model's bands look like, and where they leave the parent.

    The parent calculation was run on a uniform mesh, so there are no parent bands
    along a path to draw beside the model's. The two questions are therefore separated
    rather than forced onto one axis. The left panel is the model alone along a walk
    through the zone, which is what "does this look like a band structure" asks of. The
    right panel is the comparison, done where the parent actually exists: for every
    parent state on the mesh, how far the nearest model band sits, against the energy it
    sits at. Deviation resolved by energy is what says whether a construction is faithful
    where it was constrained and free where it was not.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ef = parent.fermi_energy_ev
    parent_bands = parent.eigenvalues_ev[spin_index]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                      gridspec_kw={"width_ratios": [1.25, 1.0]})

    if wannier is None or wannier.hr_matrix is None:
        for axis in (left, right):
            axis.text(0.5, 0.5, "no real-space Hamiltonian:\nthe model has no bands to draw",
                      ha="center", va="center", transform=axis.transAxes, fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])
        fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
        return BandFigure(out_path, "The construction wrote no real-space Hamiltonian, "
                                    "so neither its bands nor its deviation can be shown.")

    kpath, ticks, labels = _sample_path(DEFAULT_PATH)
    model_path = interpolate(wannier.hr_matrix, wannier.hr_rvectors,
                             wannier.hr_degeneracies, kpath) - ef
    left.plot(np.arange(model_path.shape[0]), model_path, color="tab:red", lw=1.0)
    left.axhline(0.0, color="tab:blue", lw=0.9)
    for edge, colour, name in ((dis_froz_max, "tab:green", "frozen"),
                               (dis_win_max, "tab:orange", "outer")):
        if edge is not None:
            left.axhline(float(edge) - ef, color=colour, lw=0.8, ls=":")
            left.text(0, float(edge) - ef, f" {name} window", color=colour,
                      va="bottom", fontsize=8)
    left.set_xticks(ticks); left.set_xticklabels(labels, fontsize=8)
    for tick in ticks[1:-1]:
        left.axvline(tick, color="0.85", lw=0.7)
    left.set_xlim(0, model_path.shape[0] - 1)
    left.set_ylabel("E - E$_F$  (eV)")
    left.set_title(f"interpolated bands ({model_path.shape[1]} of them)", fontsize=10)
    left.tick_params(labelsize=8)

    kmesh = cartesian_to_fractional_k(parent.kpoints_cart, parent.lattice_ang, parent.alat_ang)
    model_mesh = interpolate(wannier.hr_matrix, wannier.hr_rvectors,
                             wannier.hr_degeneracies, kmesh)
    deviation = np.min(np.abs(parent_bands[:, :, None] - model_mesh[:, None, :]), axis=2)
    energies = (parent_bands - ef).ravel()
    right.scatter(energies, np.maximum(deviation.ravel(), 1e-6), s=2, alpha=0.25,
                  color="tab:purple", edgecolors="none")
    right.set_yscale("log")
    right.axvline(0.0, color="tab:blue", lw=0.9)
    right.text(0.0, right.get_ylim()[1], " E$_F$", color="tab:blue", va="top", fontsize=8)
    for edge, colour, name in ((dis_froz_max, "tab:green", "frozen"),
                               (dis_win_max, "tab:orange", "outer")):
        if edge is not None:
            right.axvline(float(edge) - ef, color=colour, lw=0.8, ls=":")
            right.text(float(edge) - ef, right.get_ylim()[0], f" {name}", color=colour,
                       va="bottom", fontsize=8, rotation=90)
    right.set_xlabel("parent state energy, E - E$_F$  (eV)")
    right.set_ylabel("distance to the nearest model band (eV)")
    right.set_title("where the model leaves the parent", fontsize=10)
    right.tick_params(labelsize=8)

    fig.suptitle(title or "the construction against the calculation it came from", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    inside = energies <= (float(dis_froz_max) - ef if dis_froz_max is not None else 0.0)
    worst_inside = float(deviation.ravel()[inside].max()) if inside.any() else float("nan")
    caption = (
        "Left: the bands the construction interpolates, along a walk through the zone, "
        "relative to the Fermi level; the dotted lines mark the windows that constrained "
        "it. Right: for every state of the parent calculation on its own mesh, the "
        "distance to the nearest band of the model, against the energy that state sits "
        f"at. Largest deviation below the frozen edge: {worst_inside:.4g} eV. A "
        "construction should track the parent closely where it was constrained and may "
        "depart above it."
    )
    return BandFigure(path=out_path, caption=caption)


def draw_spread_profile(wannier, out_path: str | Path, *, title: str = "") -> BandFigure:
    """Final spreads, sorted — the picture that separates a projector problem.

    A construction whose functions fall into a tight group and a diffuse one has a
    projector set mixing shells of very different extent, and the descent then trades
    the localized members against the extended ones. That shape is obvious at a glance
    and invisible in the total spread.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spreads = np.sort(np.asarray(wannier.spreads_ang2))

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(np.arange(spreads.size), spreads, color="tab:purple", alpha=0.85)
    ax.set_xlabel("Wannier function, sorted by spread")
    ax.set_ylabel(r"spread  ($\AA^2$)")
    ratio = spreads.max() / spreads.min() if spreads.min() > 0 else float("inf")
    ax.set_title(title or f"final spreads (widest / narrowest = {ratio:.2f})", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return BandFigure(
        path=out_path,
        caption=(
            f"Final spread of each Wannier function, sorted. The widest is {ratio:.2f} "
            "times the narrowest; a large ratio means the projector set mixes shells of "
            "very different extent."
        ),
    )
