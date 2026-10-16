"""The d-only downfolding criterion and the downfolded_d profile.

Synthetic constructions: a cubic cell (a = 3 Angstrom) with a transition-metal atom at the
origin and a ligand at (1/2, 1/2, 1/2). The model is five decoupled d chains along x, so
its bands are known in closed form; the DFT reference is those bands plus ligand bands
that cross them inside E_F +- 1 eV, so plain index matching compares the wrong bands and
the order-preserving assignment must find the right ones.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import repair_fixtures as fx
from scipy.optimize import linear_sum_assignment

from autowan.actions import ActionRejected, Stage, validate_action
from autowan.band_distance_eta import band_distance_eta
from autowan.criteria import ReferenceBandPath
from autowan.downfolding import (
    D_ONLY_DOWNFOLDING,
    REFUSE_CENTRES,
    REFUSE_COUNT,
    REFUSE_PROJECTORS,
    element_of,
    match_bands,
)
from autowan.gate import CertificationGate, Construction, GateProfileWarning
from autowan.gate_profiles import (
    DEFAULT_CHECK_IDS,
    DOWNFOLDED_D_RATIONALE,
    PROFILES,
    WHOLE_MANIFOLD_CHECKS,
    custom_profile,
    get_profile,
    profile_names,
)
from autowan.inputs import (
    InputError,
    pao_excluded,
    projector_groups,
    read_construction,
    read_win,
    write_derived,
)
from autowan.interpolation import interpolate

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


A = 3.0
LATTICE = A * np.eye(3)
EF = 5.0
PATH = np.array([[s, 0.0, 0.0] for s in np.linspace(0.0, 0.5, 41)])
MESH = np.array([[i / 8, 0.0, 0.0] for i in range(8)])
#: Five d chains: on-site offsets from E_F (eV) and nearest-neighbour hoppings.
D_ONSITE = (-0.7, -0.35, 0.0, 0.3, 0.6)
D_HOP = (0.10, -0.08, 0.12, -0.05, 0.07)
#: Three ligand bands, wide (2 t = 1.6 eV), centred inside the window: they cross the d
#: bands along the path.
LIGAND_ONSITE = (-0.5, 0.1, 0.45)
LIGAND_HOP = 0.8


def _hr(onsite, hop, *, copies=1):
    n = len(onsite) * copies
    h0 = np.diag(np.repeat(np.asarray(onsite, dtype=float) + EF, copies))
    h1 = np.diag(np.repeat(np.asarray(hop, dtype=float), copies))
    # two further shells with a negligible hopping, so the outermost shells of the
    # model decay as a real construction's do (hopping_decays_with_distance)
    tail = 1e-5 * np.eye(n)
    rvec = np.array([[r, 0, 0] for r in range(-3, 4)])
    h = np.array([tail, tail, h1.T, h0, h1, tail, tail]).astype(complex)
    return rvec, np.ones(7, dtype=int), h.reshape(7, n, n)


def _ligand_bands(k):
    return np.stack(
        [EF + e + 2 * LIGAND_HOP * np.cos(2 * np.pi * k[:, 0]) for e in LIGAND_ONSITE],
        axis=1,
    )


def _groups(lines, *, spinors=False, route="projections", method="smv", species=None):
    species = species or {"Ni": 1, "O": 1}
    groups = []
    for site, ls, per_site in lines:
        groups.append(
            {
                "site": site,
                "species": site,
                "l": ls,
                "functions_per_site": per_site,
                "n_sites": species.get(site, 1),
            }
        )
    return {
        "method": method,
        "route": route,
        "spinors": spinors,
        "groups": groups,
        "error": None,
    }


def _construction(
    *,
    model_shift=0.0,
    species=("Ni", "O"),
    positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    copies=1,
    spinor=False,
    projector_record="d",
    centres=None,
    extra_model_bands=(),
    reference=True,
):
    onsite = list(D_ONSITE) + [o for o, _ in extra_model_bands]
    hop = list(D_HOP) + [t for _, t in extra_model_bands]
    rvec, deg, h = _hr(np.asarray(onsite) + model_shift, hop, copies=copies)
    exact_rvec, exact_deg, exact_h = _hr(D_ONSITE, D_HOP, copies=copies)
    d_path = interpolate(exact_h, exact_rvec, exact_deg, PATH)
    dft = np.sort(
        np.concatenate(
            [
                np.full((len(PATH), 2), EF - 20.0),  # semicore, far below
                d_path,
                _ligand_bands(PATH),
                np.full((len(PATH), 2), EF + 15.0),
            ],
            axis=1,
        ),
        axis=1,
    )
    mesh = interpolate(exact_h, exact_rvec, exact_deg, MESH)
    parent = SimpleNamespace(
        eigenvalues_ev=mesh[None],
        fermi_energy_ev=EF,
        lattice_ang=LATTICE,
        positions_frac=np.array(positions, dtype=float),
        kpoints_cart=MESH,
        alat_ang=A,
        species=tuple(species),
        spinor=spinor,
    )
    nw = h.shape[1]
    wannier = SimpleNamespace(
        hr_matrix=h,
        hr_rvectors=rvec,
        hr_degeneracies=deg,
        hr_ws_shifts=None,
        num_wann=nw,
        centres_ang=np.zeros((nw, 3)) if centres is None else np.asarray(centres),
    )
    if projector_record == "d":
        record = _groups([("Ni", [2], 5 * (2 if spinor else 1))], spinors=spinor)
    else:
        record = projector_record
    inputs = {"spin_index": 0, "mp_grid": (8, 8, 8), "num_wann": nw}
    if record is not None:
        inputs["projector_groups"] = record
    ref = (
        ReferenceBandPath(PATH, dft, EF, EF, source="synthetic") if reference else None
    )
    return Construction(
        "synthetic",
        parent=parent,
        wannier=wannier,
        material_class="metal",
        inputs=inputs,
        reference_path=ref,
    )


def _verdict(c):
    return (
        CertificationGate([D_ONLY_DOWNFOLDING], profile_name="test").run(c).verdicts[0]
    )


# ---------------------------------------------------------------------------
# the band matching
# ---------------------------------------------------------------------------


def test_matching_is_the_global_optimum_of_the_assignment_problem():
    """Order-preserving DP = Hungarian over all injections, for a convex cost."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n_w, n_b = rng.integers(1, 7), rng.integers(7, 14)
        model = np.sort(rng.normal(0, 1, (1, n_w)), axis=1)
        dft = np.sort(rng.normal(0, 1, (1, n_b)), axis=1)
        chosen = match_bands(model, dft)[0]
        assert len(set(chosen)) == n_w
        cost = (model[0][:, None] - dft[0][None, :]) ** 2
        rows, cols = linear_sum_assignment(cost)
        assert cost[np.arange(n_w), chosen].sum() == pytest.approx(
            cost[rows, cols].sum(), abs=1e-12
        )


def test_matching_skips_ligand_bands_below_between_and_above():
    d = np.array([[-0.5, 0.0, 0.4]])
    dft = np.array([[-3.0, -0.52, -0.2, 0.01, 0.2, 0.41, 2.0]])
    assert match_bands(d, dft)[0].tolist() == [1, 3, 5]


def test_matching_is_deterministic_on_a_degenerate_reference():
    model = np.array([[0.0]])
    dft = np.array([[-1.0, 0.0, 0.0, 1.0]])
    assert match_bands(model, dft)[0].tolist() == [1]
    assert match_bands(model, dft)[0].tolist() == [1]


def test_matching_maps_back_through_an_unsorted_input():
    model = np.array([[0.4, -0.5]])
    dft = np.array([[2.0, 0.41, -0.52, -3.0]])
    assert match_bands(model, dft)[0].tolist() == [1, 2]


def test_matching_refuses_a_reference_with_fewer_bands():
    with pytest.raises(ValueError, match="fewer than the model"):
        match_bands(np.zeros((1, 3)), np.zeros((1, 2)))


def test_soft_lower_edge_keeps_the_published_eta_when_absent():
    dft = np.array([[1.0, 2.0, 3.0]])
    wan = dft + 0.01
    plain = band_distance_eta(dft, wan, 0.0)
    assert plain.lower_edge_ev is None
    windowed = band_distance_eta(dft, wan, 2.0, nu_ev=1.0, lower_edge_ev=1.0)
    assert windowed.eta_mev == pytest.approx(10.0)
    assert windowed.lower_edge_ev == 1.0


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


def test_a_correct_d_only_model_with_entangled_ligands_passes():
    v = _verdict(_construction())
    assert v.applies and v.evaluable and v.passed, v.note
    assert v.metric_value == pytest.approx(0.0, abs=1e-6)
    r = v.record
    assert r["n_tm"] == 1 and r["num_wann_expected"] == 5 == r["num_wann_actual"]
    # the ligand bands cross the d manifold inside the window: index matching compares
    # the wrong bands, the assignment does not
    assert r["window_eta_index_matched_mev"] > 100.0
    assert r["dft_bands_in_window_unmatched"]
    assert set(r["matched_dft_bands_compared_in_window"]).isdisjoint(
        set(range(1, 3)) | {11, 12}
    )
    assert (
        r["n_k_matched_otherwise"] > 0
    )  # the d bands' DFT indices move along the path
    assert r["centre_tm_distance_max_ang"] == pytest.approx(0.0)


def test_a_model_off_by_30_mev_fails_on_the_window_eta():
    v = _verdict(_construction(model_shift=0.030))
    assert v.evaluable and not v.passed and v.refusal is None
    assert v.metric_value == pytest.approx(30.0, rel=0.02)


def test_a_model_off_by_10_mev_passes():
    v = _verdict(_construction(model_shift=0.010))
    assert v.passed and v.metric_value == pytest.approx(10.0, rel=0.02)


def test_count_mismatch_fails_even_with_a_perfect_window():
    """A d + ligand-p model: num_wann 8 against 5, projectors on O too."""
    record = _groups([("Ni", [2], 5), ("O", [1], 3)])
    c = _construction(
        projector_record=record,
        extra_model_bands=[(e, LIGAND_HOP) for e in LIGAND_ONSITE],
    )
    v = _verdict(c)
    assert v.evaluable and v.passed is False
    assert REFUSE_COUNT in v.refusal and REFUSE_PROJECTORS in v.refusal
    assert v.record["num_wann_expected"] == 5 and v.record["num_wann_actual"] == 8


def test_right_count_wrong_projectors_fails():
    """Five functions on the ligand (O:p;sp) count right and are not d on a TM site."""
    record = _groups([("O", [-1, 1], 5)])
    v = _verdict(_construction(projector_record=record))
    assert v.passed is False and v.refusal == REFUSE_PROJECTORS


def test_a_partial_d_shell_is_not_a_d_only_model():
    record = _groups([("Ni", [2], 3), ("Ni", [2], 2)])
    v = _verdict(_construction(projector_record=record))
    assert v.passed is False and REFUSE_PROJECTORS in v.refusal


def test_spinor_counting_doubles_the_expected_count():
    one = _verdict(_construction(spinor=True, projector_record=None))
    assert one.record["num_wann_expected"] == 10
    assert one.passed is False and REFUSE_COUNT in one.refusal
    two = _verdict(_construction(spinor=True, copies=2))
    assert two.record["num_wann_actual"] == 10 and two.passed, two.note


def test_two_transition_metal_atoms_expect_ten_functions():
    c = _construction(
        species=("Ni1", "Ni2", "O"),
        positions=((0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0.5)),
        projector_record=_groups([("Ni1", [2], 5)], species={"Ni1": 1}),
    )
    v = _verdict(c)
    assert v.record["n_tm"] == 2 and v.record["num_wann_expected"] == 10
    assert REFUSE_COUNT in v.refusal


def test_centres_off_the_transition_metal_sites_fail():
    centres = np.tile([[1.5, 1.5, 1.5]], (5, 1))  # on the ligand
    v = _verdict(_construction(centres=centres))
    assert v.passed is False and v.refusal == REFUSE_CENTRES
    assert v.record["centre_tm_distance_max_ang"] == pytest.approx(1.5 * np.sqrt(3))
    assert v.record["centre_tm_limit_ang"] == pytest.approx(0.5 * 1.5 * np.sqrt(3))


def test_no_transition_metal_withholds_the_check():
    v = _verdict(_construction(species=("Si", "O")))
    assert v.applies is False and v.evaluable is False
    assert "no transition-metal atom" in v.note


def test_group_12_is_not_counted():
    v = _verdict(_construction(species=("Zn", "O")))
    assert v.applies is False and "Zn" in v.record["not_counted"]


def test_lanthanum_counts_and_is_flagged():
    v = _verdict(_construction(species=("La", "O")))
    assert v.applies and v.record["tm_elements"] == ["La"]
    assert any("lanthanum" in f for f in v.record["flags"])


def test_scdm_is_judged_on_the_count_alone():
    record = {
        "method": "scdm",
        "route": "scdm",
        "spinors": False,
        "groups": [],
        "error": None,
    }
    v = _verdict(_construction(projector_record=record))
    assert v.passed, v.note
    assert v.record["projectors_ok"] is None and v.record["projector_route"] == "scdm"


def test_unreadable_projectors_with_the_right_count_are_non_evaluable():
    record = {"method": "pdwf", "route": "pao", "spinors": False, "groups": [],
              "error": "the deck's projector directory is missing"}  # fmt: skip
    v = _verdict(_construction(projector_record=record))
    assert v.applies and not v.evaluable and "projectors unreadable" in v.note


def test_a_count_failure_is_a_failure_without_a_reference_path():
    record = _groups([("Ni", [2], 5), ("O", [1], 3)])
    c = _construction(
        projector_record=record,
        extra_model_bands=[(e, LIGAND_HOP) for e in LIGAND_ONSITE],
        reference=False,
    )
    v = _verdict(c)
    assert v.evaluable and v.passed is False and REFUSE_COUNT in v.refusal


def test_a_correct_model_without_a_reference_path_is_non_evaluable():
    v = _verdict(_construction(reference=False))
    assert v.applies and not v.evaluable


def test_a_d_manifold_outside_the_window_is_non_evaluable():
    """A d manifold 3 eV below E_F: no matched pair inside the window, tails only."""
    v = _verdict(_construction(model_shift=-3.0))
    assert v.applies and not v.evaluable and "Fermi-function tails" in v.note


@pytest.mark.parametrize(
    "label, element",
    [
        ("Ni2", "Ni"),
        ("Fe_up", "Fe"),
        ("O1", "O"),
        ("Co", "Co"),
        ("C", "C"),
        ("Sn", "Sn"),
    ],
)
def test_species_labels_name_their_element(label, element):
    assert element_of(label) == element


# ---------------------------------------------------------------------------
# reading a deck's projectors
# ---------------------------------------------------------------------------


def _atoms(*rows):
    return "begin atoms_frac\n" + "\n".join(rows) + "\nend atoms_frac\n"


def test_projections_block_is_read_per_species_with_site_counts(tmp_path):
    win = "num_wann = 10\n" + _atoms("Ni 0 0 0", "Ni 0.5 0 0", "O 0.5 0.5 0.5")
    win += "begin projections\nNi:d\nend projections\n"
    out = projector_groups(win, None, tmp_path)
    assert out["route"] == "projections" and out["error"] is None
    assert out["groups"] == [
        {
            "site": "Ni",
            "species": "Ni",
            "l": [2],
            "functions_per_site": 5,
            "orbitals": ["d"],
            "n_sites": 2,
        }
    ]


def test_projections_block_reads_spinors_positions_and_hybrids(tmp_path):
    win = "spinors = .true.\n" + _atoms("Ni 0 0 0", "O 0.5 0.5 0.5")
    win += "begin projections\nang\nf=0,0,0:l=2\nO:sp3\nend projections\n"
    out = projector_groups(win, None, tmp_path)
    first, second = out["groups"]
    assert first["position"] == [0.0, 0.0, 0.0] and first["functions_per_site"] == 10
    assert second["l"] == [-1] and second["functions_per_site"] == 8


def test_an_unreadable_projections_line_is_reported_not_guessed(tmp_path):
    win = _atoms("Ni 0 0 0") + "begin projections\nrandom\nend projections\n"
    out = projector_groups(win, None, tmp_path)
    assert out["error"] and out["groups"] == []


def test_scdm_deck_has_no_projectors(tmp_path):
    out = projector_groups(
        "num_wann = 5\n", "&inputpp\n scdm_proj = .true.\n/\n", tmp_path
    )
    assert out["route"] == "scdm"


def test_pao_deck_lists_the_kept_orbitals_per_atom(tmp_path):
    fx.write_deck(tmp_path)
    out = read_construction(tmp_path, fx.SEED).gate_inputs()["projector_groups"]
    assert out["route"] == "pao" and out["method"] == "pdwf"
    sr, o = out["groups"]
    assert (
        sr["species"] == "Sr" and sr["l"] == [0, 1, 2] and sr["functions_per_site"] == 9
    )
    assert o["l"] == [0, 1] and o["functions_per_site"] == 4


def test_a_pao_deck_without_its_projector_directory_reports_why(tmp_path):
    fx.write_deck(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "external_projectors")
    out = read_construction(tmp_path, fx.SEED).gate_inputs()["projector_groups"]
    assert out["route"] == "pao" and "missing" in out["error"]


# ---------------------------------------------------------------------------
# what the vocabulary can write
# ---------------------------------------------------------------------------


def test_an_empty_pao_subset_drops_a_species(tmp_path):
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    derived = write_derived(
        source,
        tmp_path,
        "SrO_r1",
        {
            "pao_subset": [
                {"species": "Sr", "orbitals": ("4d",)},
                {"species": "O", "orbitals": ()},
            ]
        },
    )
    assert read_win(derived.win)["num_wann"] == "5"
    assert pao_excluded(derived.pw2wan.read_text()) == set(range(1, 9)) | {
        14,
        15,
        16,
        17,
    }
    groups = derived.gate_inputs()["projector_groups"]["groups"]
    assert [(g["species"], g["l"], g["functions_per_site"]) for g in groups] == [
        ("Sr", [2], 5)
    ]


def test_a_pao_subset_that_keeps_nothing_is_refused(tmp_path):
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    with pytest.raises(InputError, match="keeps no pseudo-atomic orbital"):
        write_derived(
            source,
            tmp_path,
            "SrO_r1",
            {
                "pao_subset": [
                    {"species": "Sr", "orbitals": ()},
                    {"species": "O", "orbitals": ()},
                ]
            },
        )


def test_the_validator_accepts_an_empty_pao_subset_entry():
    action = validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "d",
            "evidence": ["c"],
            "targets": ["c"],
            "expected_effect": "e",
            "confidence": 0.5,
            "changes": [
                {
                    "parameter": "pao_subset",
                    "new": [
                        {"species": "Sr", "orbitals": ["4d"]},
                        {"species": "O", "orbitals": []},
                    ],
                    "reason": "r",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=("c",),
        unresolved_check_ids=("c",),
        allowed_species=("O", "Sr"),
        method="pdwf",
    )
    assert action.changes[0].new[1] == {"species": "O", "orbitals": ()}


def test_an_explicit_tm_d_projector_set_writes_a_d_only_smv_deck(tmp_path):
    """QE SMV: ``projectors`` [{TM, [d]}] replaces the block and sets num_wann = 5 N_TM."""
    win = "num_wann = 9\n" + _atoms("Ni 0 0 0", "O 0.5 0.5 0.5")
    win += "begin projections\nNi:s;d\nO:p\nend projections\n"
    (tmp_path / "s.win").write_text(win)
    (tmp_path / "s.pw2wan").write_text("&inputpp\n  seedname = 's'\n/\n")
    source = read_construction(tmp_path, "s")
    derived = write_derived(
        source,
        tmp_path,
        "s_r1",
        {"projectors": [{"species": "Ni", "orbitals": ("d",)}]},
        species_counts={"Ni": 1, "O": 1},
    )
    assert read_win(derived.win)["num_wann"] == "5"
    groups = derived.gate_inputs()["projector_groups"]
    assert groups["route"] == "projections" and groups["groups"][0]["species"] == "Ni"
    assert len(groups["groups"]) == 1


# ---------------------------------------------------------------------------
# the profiles
# ---------------------------------------------------------------------------


def test_downfolded_d_is_the_d_manifold_target():
    """Default minus the whole-occupied-manifold checks, plus the d-only check."""
    p = get_profile("downfolded_d")
    assert p.check_ids == (
        "wannier_count_matches_request",
        "hopping_shells_inside_representable_supercell",
        "centres_stay_near_home_cell",
        "model_bands_correspond_to_parent_bands",
        "hopping_decays_with_distance",
        "metal_frozen_window_contains_fermi_level",
        "d_only_downfolding",
    )
    assert not set(WHOLE_MANIFOLD_CHECKS) & set(p.check_ids)
    assert p.target_manifold == "d_manifold" and p.sub_manifold
    assert p.hash == "109693a2b2a8f92d"
    # every default check is accounted for, kept or dropped, with a reason
    assert set(DOWNFOLDED_D_RATIONALE) == set(DEFAULT_CHECK_IDS)
    kept = {k for k, (v, _) in DOWNFOLDED_D_RATIONALE.items() if v == "keep"}
    assert kept == set(p.check_ids) - {"d_only_downfolding"}
    text = p.target_text()
    assert "five per transition-metal atom" in text and "d orbitals" in text
    assert "pao_subset" in text and "Fermi level" in text
    assert "not the whole occupied manifold" in text


def test_the_other_profiles_are_unchanged():
    sub_manifold = ("downfolded_d", "dmft_subspace")
    assert {
        n: get_profile(n).hash for n in profile_names() if n not in sub_manifold
    } == {
        "default": "6d07e441e0930d6c",
        "band_interpolation": "03e038df686b1d80",
        "sparse_tb": "1856955f699ba0c4",
        "atom_centred_basis": "e2381471219a230b",
    }
    for name in profile_names():
        if name not in sub_manifold:
            assert get_profile(name).target_manifold == "whole_occupied_manifold"


def test_a_custom_profile_keeps_the_target_only_without_whole_manifold_checks():
    derived = custom_profile("downfolded_d", add=["band_interpolation_eta2"])
    assert derived.sub_manifold
    back = custom_profile("downfolded_d", add=["occupied_manifold_is_represented"])
    assert not back.sub_manifold
    assert not custom_profile("default", add=["d_only_downfolding"]).sub_manifold


def test_conduction_band_left_the_named_profiles_but_reproduces():
    assert (
        "conduction_band" not in profile_names() and "conduction_band" not in PROFILES
    )
    rebuilt = custom_profile("default", add=["conduction_band_eta"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GateProfileWarning)
        retired = get_profile("conduction_band")
    assert caught and "retired" in str(caught[0].message)
    assert retired.hash == rebuilt.hash
    assert retired.check_ids == (*DEFAULT_CHECK_IDS, "conduction_band_eta")


def test_the_d_only_check_is_addable_to_any_profile():
    p = custom_profile("band_interpolation", add=["d_only_downfolding"])
    assert p.check_ids[-1] == "d_only_downfolding"


def _with_occupied_ligand(c):
    occupied_ligand = np.full((1, len(MESH), 1), EF - 3.0)
    c.parent.eigenvalues_ev = np.concatenate(
        [occupied_ligand, c.parent.eigenvalues_ev], axis=2
    )
    return c


def test_downfolded_d_certifies_a_d_only_model_of_a_parent_with_occupied_ligands():
    """The d-manifold target: the occupied ligand band the d model leaves out is not a
    failure under downfolded_d, and is under the whole-manifold default (the v6 issue).
    """
    c = _with_occupied_ligand(_construction())
    report = get_profile("downfolded_d").gate(None).run(c)
    assert report.passed, report.failed_check_ids
    whole = custom_profile("default", add=["d_only_downfolding"]).gate(None).run(c)
    by_id = {v.check_id: v for v in whole.verdicts}
    assert by_id["d_only_downfolding"].passed
    assert by_id["occupied_manifold_is_represented"].passed is False
    assert not whole.passed


def test_a_full_valence_model_still_fails_downfolded_d_on_count():
    record = _groups([("Ni", [2], 5), ("O", [1], 3)])
    c = _with_occupied_ligand(
        _construction(
            projector_record=record,
            extra_model_bands=[(e, LIGAND_HOP) for e in LIGAND_ONSITE],
        )
    )
    report = get_profile("downfolded_d").gate(None).run(c)
    # (the toy parent's mesh carries no ligand bands, so model_bands_correspond also
    # objects here; the point is the count)
    assert "d_only_downfolding" in report.failed_check_ids and not report.passed
    v = {v.check_id: v for v in report.verdicts}["d_only_downfolding"]
    assert REFUSE_COUNT in v.refusal


# ---------------------------------------------------------------------------
# the occupied-band requirements lifted only for the sub-manifold target
# ---------------------------------------------------------------------------


def _projector_action(**kwargs):
    return validate_action(
        {
            "action": "modify_parameters",
            "diagnosis": "d",
            "evidence": ["c"],
            "targets": ["c"],
            "expected_effect": "e",
            "confidence": 0.5,
            "changes": [
                {
                    "parameter": "projectors",
                    "new": [{"species": "Ni", "orbitals": ["d"]}],
                    "reason": "r",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=("c",),
        unresolved_check_ids=("c",),
        allowed_species=("Ni", "O"),
        species_counts={"Ni": 2, "O": 2},
        band_count=40,
        occupied_bands=16,
        method="smv",
        **kwargs,
    )


def test_the_validator_keeps_the_occupied_floor_for_a_whole_manifold_profile():
    with pytest.raises(ActionRejected, match="whole occupied manifold") as exc:
        _projector_action()
    assert "downfolded_d" in str(exc.value)


def test_the_validator_lifts_the_occupied_floor_for_the_d_manifold_target():
    action = _projector_action(sub_manifold_target=True)
    assert action.changes[0].new == ({"species": "Ni", "orbitals": ("d",)},)


def test_the_deck_check_states_rather_than_refuses_an_exclusion_under_a_sub_manifold(
    tmp_path,
):
    import test_semicore_exclusion_knob as sk

    from autowan.deckcheck import check_deck

    ctx = sk._ctx(tmp_path)
    deck = sk._derive(ctx, ctx.frontier, sk._semicore(("Sr", ["4s"])))
    back = sk._derive(ctx, deck, sk._semicore(("Sr", ["4s", "4p"])), seed="SrO_r2")
    whole = check_deck(back, parent=ctx.parent, code="qe", ctx=ctx)
    assert not whole.ok and "not honoured" in whole.reason()
    sub = check_deck(
        back, parent=ctx.parent, code="qe", ctx=ctx, whole_occupied_target=False
    )
    assert sub.ok and any("sub-manifold" in n for n in sub.notes)


# ---------------------------------------------------------------------------
# stored constructions (not shipped; skipped when absent)
# ---------------------------------------------------------------------------

V6_RUNS = _stored("AUTOWAN_TEST_V6_RUNS")
SRVO3_PARENT = _stored("AUTOWAN_TEST_SRVO3_PARENT")
SRVO3_NPZ = SRVO3_PARENT / "work/8f/6f/SrVO3_bands_path.npz"
SRVO3_FRONTIER = _stored("AUTOWAN_TEST_SRVO3_FRONTIER")


def _score_stored(run_dir: Path, profile: str):
    from autowan.backends import get_backend
    from autowan.criteria import read_qe_reference_band_path
    from autowan.gate_profiles import build_gate
    from autowan.w90_outputs import read_run
    from autowan.workflow import WorkflowConfig, _Scorer

    backend = get_backend("qe")
    parent = backend.read_parent(
        run_dir.joinpath("out").resolve() / "aiida.save" / "data-file-schema.xml"
    )
    ref = read_qe_reference_band_path(SRVO3_NPZ, fermi_energy_ev=11.8776)
    material_class = "metal" if parent.is_metallic() else "gapped"
    config = WorkflowConfig(
        gate_profile=profile, reference_path=ref, draw_figures=False
    )
    gate = build_gate(backend, profile, (), reference_path=ref)
    scorer = _Scorer(config, gate, backend, parent, material_class, run_dir)
    return scorer.score(
        "aiida",
        backend.read_construction(run_dir, "aiida"),
        read_run(run_dir / "aiida.wout"),
    )


@pytest.mark.skipif(
    not (V6_RUNS / "demo/SrVO3_Vd_smv/aiida_hr.dat").is_file()
    or not SRVO3_NPZ.is_file(),
    reason="the stored SrVO3 V:d construction is not shipped",
)
def test_the_stored_srvo3_v_d_model_certifies_under_downfolded_d():
    report = _score_stored(V6_RUNS / "demo/SrVO3_Vd_smv", "downfolded_d")
    assert report.passed, (report.failed_check_ids, report.non_evaluable_check_ids)
    d = report.measurements()["d_only_downfolding"]
    assert d["value"] == pytest.approx(1.32, abs=0.01)
    assert d["matched_dft_bands_modal"] == [21, 22, 23, 24, 25]
    # the whole-manifold default refuses the same model
    assert not _score_stored(V6_RUNS / "demo/SrVO3_Vd_smv", "default").passed


@pytest.mark.skipif(
    not (SRVO3_FRONTIER / "aiida_hr.dat").is_file() or not SRVO3_NPZ.is_file(),
    reason="the stored SrVO3 frontier construction is not shipped",
)
def test_the_stored_full_valence_srvo3_frontier_fails_downfolded_d_on_count():
    report = _score_stored(SRVO3_FRONTIER, "downfolded_d")
    assert "d_only_downfolding" in report.failed_check_ids
    d = report.measurements()["d_only_downfolding"]
    assert "d_only_count_mismatch" in d["refusal"]
    assert d["num_wann_actual"] == 30 and d["num_wann_expected"] == 5
