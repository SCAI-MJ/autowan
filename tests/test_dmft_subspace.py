"""The DMFT-subspace criterion and the dmft_subspace profile.

Synthetic SrVO3-like constructions: a cubic cell (a = 3.9 Angstrom) with Sr, V and three
O. The model is three decoupled t2g chains along x, so its bands are known in closed
form; the DFT reference is those bands plus other bands that cross them, and the target
weights mark the t2g bands as the target manifold.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autowan.criteria import ReferenceBandPath
from autowan.dmft_subspace import (
    COMPARISON_WINDOW_MARGIN_EV,
    DMFT_SUBSPACE_MODEL,
    DMFT_SUBSPACE_MODEL_UNWINDOWED,
    REFUSE_CENTRES,
    REFUSE_COUNT,
    REFUSE_SEEDS,
    SUBSPACES,
    comparison_window,
    manifold_isolation,
    reduced_formula,
    target_for_species,
    write_weights,
)
from autowan.gate import CertificationGate, Construction
from autowan.gate_profiles import custom_profile, get_profile, profile_names
from autowan.inputs import projector_groups
from autowan.interpolation import interpolate

A = 3.9
LATTICE = A * np.eye(3)
EF = 5.0
PATH = np.array([[s, 0.0, 0.0] for s in np.linspace(0.0, 0.5, 41)])
MESH = np.array([[i / 8, 0.0, 0.0] for i in range(8)])
T2G_ONSITE = (-0.4, 0.0, 0.35)
T2G_HOP = (0.25, -0.2, 0.3)
OTHER_ONSITE = (-0.2, 0.6)
OTHER_HOP = 0.9
SPECIES = ("Sr", "V", "O", "O", "O")
POSITIONS = (
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)


def _hr(onsite, hop):
    n = len(onsite)
    h0 = np.diag(np.asarray(onsite, dtype=float) + EF)
    h1 = np.diag(np.asarray(hop, dtype=float))
    tail = 1e-5 * np.eye(n)
    rvec = np.array([[r, 0, 0] for r in range(-3, 4)])
    h = np.array([tail, tail, h1.T, h0, h1, tail, tail]).astype(complex)
    return rvec, np.ones(7, dtype=int), h.reshape(7, n, n)


def _channel(t2g_onsite=T2G_ONSITE, other_onsite=OTHER_ONSITE, other_weight=0.0):
    """One spin channel's path bands (sorted) and target weights (t2g 0.9)."""
    rvec, deg, h = _hr(t2g_onsite, T2G_HOP)
    t2g = interpolate(h, rvec, deg, PATH)
    other = np.stack(
        [EF + e + 2 * OTHER_HOP * np.cos(2 * np.pi * PATH[:, 0]) for e in other_onsite],
        axis=1,
    )
    below = np.full((len(PATH), 2), EF - 6.0)
    bands = np.concatenate([below, t2g, other], axis=1)
    weight = np.zeros_like(bands)
    weight[:, 2:5] = 0.9
    weight[:, 5:] = other_weight
    order = np.argsort(bands, axis=1, kind="stable")
    return (
        np.take_along_axis(bands, order, axis=1),
        np.take_along_axis(weight, order, axis=1),
    )


def _reference(
    tmp_path,
    *,
    weights_target="O3SrV",
    with_weights=True,
    channels=None,
    **channel_kwargs,
):
    """A one-channel reference, or one per entry of ``channels`` (kwargs of _channel)."""
    pairs = [_channel(**kw) for kw in (channels or [channel_kwargs])]
    bands = np.stack([b for b, _ in pairs])
    weights = np.stack([w for _, w in pairs])
    provenance = {}
    if with_weights:
        target = SUBSPACES[weights_target]
        path = write_weights(
            tmp_path / "w.npz",
            weights=weights,
            kpoints_frac=PATH,
            target=target,
            source="synthetic",
        )
        provenance["dmft_subspace_weights"] = str(path)
    return ReferenceBandPath(
        PATH, bands, EF, EF, source="synthetic", provenance=provenance
    )


def _groups(lines, route="projections"):
    counts = {s: SPECIES.count(s) for s in set(SPECIES)}
    return {
        "method": "smv",
        "route": route,
        "spinors": False,
        "groups": [
            {
                "site": site,
                "species": site,
                "l": [2],
                "functions_per_site": len(tokens),
                "orbitals": list(tokens),
                "n_sites": counts.get(site, 1),
            }
            for site, tokens in lines
        ],
        "error": None,
    }


def _construction(
    tmp_path,
    *,
    shift=0.0,
    onsite=T2G_ONSITE,
    hop=T2G_HOP,
    projectors=(("V", ("dxy", "dyz", "dxz")),),
    centres=None,
    species=SPECIES,
    positions=POSITIONS,
    material_class="metal",
    spin_index=0,
    **ref_kwargs,
):
    rvec, deg, h = _hr(np.asarray(onsite) + shift, hop)
    nw = h.shape[1]
    parent = SimpleNamespace(
        eigenvalues_ev=interpolate(h, rvec, deg, MESH)[None],
        fermi_energy_ev=EF,
        lattice_ang=LATTICE,
        positions_frac=np.array(positions, dtype=float),
        kpoints_cart=MESH,
        alat_ang=A,
        species=tuple(species),
        spinor=False,
    )
    wannier = SimpleNamespace(
        hr_matrix=h,
        hr_rvectors=rvec,
        hr_degeneracies=deg,
        hr_ws_shifts=None,
        num_wann=nw,
        centres_ang=np.zeros((nw, 3)) if centres is None else np.asarray(centres),
    )
    inputs = {
        "spin_index": spin_index,
        "num_wann": nw,
        "projector_groups": _groups(projectors),
    }
    return Construction(
        "synthetic",
        parent=parent,
        wannier=wannier,
        material_class=material_class,
        inputs=inputs,
        reference_path=_reference(tmp_path, **ref_kwargs),
    )


def _verdict(c, check=DMFT_SUBSPACE_MODEL):
    return CertificationGate([check], profile_name="test").run(c).verdicts[0]


def test_targets_are_found_by_reduced_composition_whatever_the_magnetic_cell():
    assert reduced_formula(("Ni1", "Ni2", "O", "O")) == "NiO"
    assert target_for_species(("Ni1", "Ni2", "O", "O")).formula == "NiO"
    assert target_for_species(SPECIES).formula == "O3SrV"
    assert target_for_species(("Fe",)).correlated_per_site("Fe") == 5
    assert target_for_species(("Fe",)).functions_per_site("Fe") == 6
    assert target_for_species(("Cu", "Cu", "O")) is None
    assert set(SUBSPACES) == {"O3SrV", "O3SrTi", "NiO", "MnO", "Fe"}


def test_the_exact_t2g_model_passes(tmp_path):
    v = _verdict(_construction(tmp_path))
    assert v.passed is True, v.note
    assert v.record["target_rms_mev"] < 1e-6
    assert v.record["target_states_per_k"] == [3, 3]
    assert v.record["n_target_states_unpaired"] == 0


def test_a_shifted_model_fails_on_the_target_manifold(tmp_path):
    v = _verdict(_construction(tmp_path, shift=0.05))
    assert v.passed is False
    assert v.record["target_rms_mev"] == pytest.approx(50.0, abs=1.0)


def test_non_target_bands_are_not_compared(tmp_path):
    """The other bands cross the t2g ones; only the weighted states enter the RMS."""
    v = _verdict(_construction(tmp_path))
    assert v.record["n_target_states"] == 3 * len(PATH)


def test_a_full_d_shell_is_not_the_t2g_target(tmp_path):
    c = _construction(
        tmp_path, projectors=(("V", ("dz2", "dxz", "dyz", "dx2-y2", "dxy")),)
    )
    v = _verdict(c)
    assert v.passed is not True
    assert REFUSE_SEEDS in (v.refusal or "")
    assert any("dz2" in p for p in v.record["seed_problems"])


def test_a_ligand_seed_is_refused_and_a_wrong_count_too(tmp_path):
    c = _construction(
        tmp_path, projectors=(("V", ("dxy", "dyz", "dxz")), ("O", ("pz",)))
    )
    v = _verdict(c)
    assert REFUSE_SEEDS in (v.refusal or "")
    c.inputs["num_wann"] = 4
    c.wannier.num_wann = 4
    v = _verdict(c)
    assert REFUSE_COUNT in (v.refusal or "")


def test_correlated_functions_must_sit_on_their_site(tmp_path):
    off = np.zeros((3, 3))
    off[0] = [A / 2, A / 2, 0.0]  # between sites, beyond half the V-O distance
    v = _verdict(_construction(tmp_path, centres=off))
    assert REFUSE_CENTRES in (v.refusal or "")
    assert v.record["correlated_sites"][0]["functions_on_site"] == 2


def test_other_materials_are_withheld(tmp_path):
    c = _construction(
        tmp_path,
        species=("Cu",),
        positions=((0.0, 0.0, 0.0),),
        projectors=(("Cu", ("d",)),),
    )
    v = _verdict(c)
    assert v.applies is False


def test_weights_for_another_target_or_none_are_non_evaluable(tmp_path):
    v = _verdict(_construction(tmp_path, weights_target="O3SrTi"))
    assert v.passed is not True and v.evaluable is False
    v = _verdict(_construction(tmp_path, with_weights=False))
    assert v.evaluable is False


def test_pao_seeds_count_the_outermost_orbital_as_valence():
    from autowan.dmft_subspace import _pao_orbitals

    valence, inner = _pao_orbitals(
        {"labels": ["s2", "d1"], "all_labels": ["s1", "s2", "p1", "p2", "d1"]}
    )
    assert inner == [] and (0, "s") in valence and (2, "dxy") in valence
    _, inner = _pao_orbitals(
        {"labels": ["s1", "d1"], "all_labels": ["s1", "s2", "p1", "p2", "d1"]}
    )
    assert inner == ["s1"]


def test_projections_groups_record_their_orbital_names(tmp_path):
    win = (
        "num_wann = 3\nbegin atoms_frac\nSr 0.5 0.5 0.5\nV 0 0 0\nO 0.5 0 0\n"
        "end atoms_frac\nbegin projections\nV:dxy;dyz;dxz\nend projections\n"
    )
    out = projector_groups(win, None, tmp_path)
    assert out["groups"][0]["orbitals"] == ["dxy", "dyz", "dxz"]
    assert out["groups"][0]["functions_per_site"] == 3


def test_the_profile_states_the_structures_own_target():
    profile = get_profile("dmft_subspace")
    assert profile.sub_manifold and profile.target_manifold == "correlated_subspace"
    text = profile.target_text(SPECIES)
    assert "three V t2g orbitals" in text
    assert "NiO:" in get_profile("dmft_subspace").target_text(("Ni1", "Ni2", "O", "O"))
    assert "not one the DMFT-subspace criterion covers" in profile.target_text(("Cu",))
    for name in profile_names():
        if name != "dmft_subspace":
            p = get_profile(name)
            assert p.target_text(SPECIES) == p.target_text()


def test_more_target_states_than_functions_keeps_the_largest_weights(tmp_path):
    """An entangled manifold can lend a target orbital to two hybridised states."""
    c = _construction(tmp_path)
    path = c.reference_path.provenance["dmft_subspace_weights"]
    with np.load(path) as f:
        w = np.array(f["weights"])
    w[0, :, 0] = 0.55  # a far band now also passes the weight cut
    write_weights(
        path, weights=w, kpoints_frac=PATH, target=SUBSPACES["O3SrV"], source="t"
    )
    v = _verdict(c)
    assert v.passed is True, v.note
    assert v.record["n_target_states_beyond_function_count"] == len(PATH)
    assert v.record["target_states_per_k"] == [3, 3]


# ---------------------------------------------------------------------------
# isolated or entangled: the comparison window (second 2026-10-12 revision)
# ---------------------------------------------------------------------------

#: The third t2g band entirely above E_F + 2 eV (EF + 2.4 to EF + 3.6).
HIGH_T2G = (-0.4, 0.0, 3.0)
#: The model's third band off by 50 meV, there only.
HIGH_T2G_OFF = (-0.4, 0.0, 3.05)
#: Other bands far above the t2g ones (EF + 8.2 and up): the t2g manifold isolated.
FAR_OTHERS = (10.0, 12.0)


def test_the_window_margin_is_the_frontier_protocols():
    from autowan.methods import WINDOW_MARGIN_EV

    assert COMPARISON_WINDOW_MARGIN_EV == WINDOW_MARGIN_EV == 2.0


def test_isolation_is_decided_from_the_band_data(tmp_path):
    crossing = _reference(tmp_path)
    iso = manifold_isolation(crossing.bands_ev, _weights_of(crossing), 3)
    assert iso["isolated"] is False
    assert iso["channels"][0]["n_target_bands"] > 3
    far = _reference(tmp_path, t2g_onsite=HIGH_T2G, other_onsite=FAR_OTHERS)
    iso = manifold_isolation(far.bands_ev, _weights_of(far), 3)
    assert iso["isolated"] is True
    ch = iso["channels"][0]
    assert ch["target_bands_1based"] == [3, 5]
    assert ch["gap_above_ev"] > 4.0 and ch["gap_below_ev"] > 4.0
    # the same bands judged against a four-function target are not isolated
    assert manifold_isolation(far.bands_ev, _weights_of(far), 4)["isolated"] is False


def _weights_of(ref):
    with np.load(ref.provenance["dmft_subspace_weights"]) as f:
        return np.asarray(f["weights"])


def test_an_entangled_target_is_compared_only_up_to_ef_plus_2(tmp_path):
    """The model misses a t2g band lying wholly above E_F + 2 eV: not compared."""
    c = _construction(tmp_path, onsite=HIGH_T2G_OFF, t2g_onsite=HIGH_T2G)
    v = _verdict(c)
    assert v.passed is True, v.note
    r = v.record
    assert r["manifold_isolated"] is False
    assert r["comparison_window_anchor"] == "Fermi level"
    assert r["comparison_window_top_ev"] == pytest.approx(EF + 2.0)
    assert r["n_target_states_above_window"] == len(PATH)
    assert r["n_target_states"] == 2 * len(PATH)
    assert r["target_energy_range_ev"][1] <= EF + 2.0
    assert "entangled target manifold" in v.note
    # the first form compares every target state and fails it
    old = _verdict(c, DMFT_SUBSPACE_MODEL_UNWINDOWED)
    assert old.passed is False
    assert old.record["target_rms_mev"] == pytest.approx(50.0 / np.sqrt(3), rel=0.05)
    assert "manifold_isolated" not in old.record


def test_an_isolated_target_is_compared_over_its_whole_width(tmp_path):
    c = _construction(
        tmp_path, onsite=HIGH_T2G_OFF, t2g_onsite=HIGH_T2G, other_onsite=FAR_OTHERS
    )
    v = _verdict(c)
    assert v.passed is False
    assert v.record["manifold_isolated"] is True
    assert "comparison_window_top_ev" not in v.record
    assert v.record["n_target_states_above_window"] == 0
    assert v.record["n_target_states"] == 3 * len(PATH)
    assert "isolated target manifold" in v.note
    # unchanged from the first form
    old = _verdict(c, DMFT_SUBSPACE_MODEL_UNWINDOWED)
    assert old.record["target_rms_mev"] == pytest.approx(v.record["target_rms_mev"])
    # and the exact model of an isolated manifold passes
    exact = _construction(
        tmp_path, onsite=HIGH_T2G, t2g_onsite=HIGH_T2G, other_onsite=FAR_OTHERS
    )
    assert _verdict(exact).passed is True


def test_a_gapped_parent_anchors_the_window_at_the_conduction_band_minimum(tmp_path):
    c = _construction(
        tmp_path, onsite=HIGH_T2G_OFF, t2g_onsite=HIGH_T2G, material_class="gapped"
    )
    c.parent.eigenvalues_ev = np.array([[[EF - 1.0, EF + 0.5, EF + 3.0]]])
    win = comparison_window(c, c.reference_path)
    assert win["comparison_window_anchor"] == "conduction-band minimum"
    assert win["comparison_window_anchor_ev"] == pytest.approx(EF + 0.5)
    assert win["comparison_window_top_ev"] == pytest.approx(EF + 2.5)
    assert win["comparison_window_bands"] == "construction mesh"
    v = _verdict(c)
    # the window now reaches EF + 2.5: the off band's states up to there are compared
    assert v.record["target_rms_mev"] > 1.0
    assert 0 < v.record["n_target_states_above_window"] < len(PATH)


def test_the_window_acts_before_the_per_k_cap(tmp_path):
    """States above the window with the largest weights do not displace those inside."""
    c = _construction(tmp_path, other_onsite=(4.5, 5.0), other_weight=0.99)
    v = _verdict(c)
    assert v.passed is True, v.note
    assert v.record["n_target_states_above_window"] == 2 * len(PATH)
    assert v.record["n_target_states_beyond_function_count"] == 0
    assert v.record["target_states_per_k"] == [3, 3]
    # without the window the cap keeps the far, heavier states and the model fails
    assert _verdict(c, DMFT_SUBSPACE_MODEL_UNWINDOWED).passed is False


def test_the_per_k_cap_still_acts_inside_the_window(tmp_path):
    c = _construction(tmp_path)
    path = c.reference_path.provenance["dmft_subspace_weights"]
    w = _weights_of(c.reference_path)
    w[0, :, 0] = 0.55  # a band at EF - 6 eV, inside the window, also passes the cut
    write_weights(
        path, weights=w, kpoints_frac=PATH, target=SUBSPACES["O3SrV"], source="t"
    )
    v = _verdict(c)
    assert v.passed is True, v.note
    assert v.record["manifold_isolated"] is False
    assert v.record["n_target_states_above_window"] == 0
    assert v.record["n_target_states_beyond_function_count"] == len(PATH)
    assert v.record["target_states_per_k"] == [3, 3]


@pytest.mark.parametrize("spin", [0, 1])
def test_one_entangled_spin_channel_makes_the_manifold_entangled(tmp_path, spin):
    """Spin 0 isolated, spin 1 entangled: both channels are judged inside the window."""
    channels = [
        {"t2g_onsite": HIGH_T2G, "other_onsite": FAR_OTHERS},
        {"t2g_onsite": HIGH_T2G},
    ]
    c = _construction(tmp_path, onsite=HIGH_T2G_OFF, channels=channels, spin_index=spin)
    iso = manifold_isolation(
        c.reference_path.bands_ev, _weights_of(c.reference_path), 3
    )
    assert [ch["isolated"] for ch in iso["channels"]] == [True, False]
    v = _verdict(c)
    assert v.record["manifold_isolated"] is False
    assert v.passed is True, v.note
    assert v.record["n_target_states_above_window"] == len(PATH)
    # both channels isolated: the whole width again, in either channel
    both = [{"t2g_onsite": HIGH_T2G, "other_onsite": FAR_OTHERS}] * 2
    c = _construction(tmp_path, onsite=HIGH_T2G_OFF, channels=both, spin_index=spin)
    v = _verdict(c)
    assert v.record["manifold_isolated"] is True and v.passed is False


def test_the_target_text_states_which_states_are_compared(tmp_path):
    profile = get_profile("dmft_subspace")
    rule = profile.target_text(SPECIES)
    assert "isolated" in rule and "entangled" in rule
    assert "2 eV above the Fermi level" in rule
    c = _construction(tmp_path, t2g_onsite=HIGH_T2G)
    parent = c.parent
    parent.is_metallic = lambda: True
    text = profile.target_text(SPECIES, reference_path=c.reference_path, parent=parent)
    assert "is entangled with other bands" in text
    assert "up to 2 eV above the Fermi level" in text
    assert f"{EF + 2.0:.2f} eV" in text
    far = _reference(tmp_path, t2g_onsite=HIGH_T2G, other_onsite=FAR_OTHERS)
    text = profile.target_text(SPECIES, reference_path=far, parent=parent)
    assert "is isolated" in text and "all of its states are compared" in text
    # the structure's own statement says nothing about how to build the model
    own = text.split("For this structure the target subspace is")[1]
    for word in ("dis_froz", "frozen", "window", "projection", "pao_subset"):
        assert word not in own


def test_the_profile_hashes_new_and_retired():
    assert get_profile("dmft_subspace").hash == "333e859de17049d0"
    assert get_profile("dmft_subspace").check_ids[-1] == "dmft_subspace_model_window"
    first = custom_profile(
        "dmft_subspace",
        add=["dmft_subspace_model"],
        drop=["dmft_subspace_model_window"],
    )
    assert first.hash == "941cf34c1c36bce4"
    assert get_profile("downfolded_d").hash == "109693a2b2a8f92d"
