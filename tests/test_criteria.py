"""The three acceptance-criterion checks, the reference-path readers and the drift guard.

Synthetic constructions throughout: a cubic cell (a = 3 Angstrom) carrying a few decoupled
chains along x, so every band is known in closed form and every expected eta is exact.
The regression against the stored QE results is ``test_criteria_regression_qe.py``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan.band_distance_eta import band_distance_eta
from autowan.core_checks import (
    CENTRES_ON_ATOMS_ATOM_CENTRED,
    CORE_CHECKS,
    distance_outside_home_cell_ang,
    nearest_atom_distance_ang,
)
from autowan.criteria import (
    BAND_INTERPOLATION_ETA2,
    CONDUCTION_BAND_ETA,
    TRUNCATED_HOPPING_ETA2,
    ReferenceBandPath,
    bvk_length_ang,
    decay_lengths_ang,
    insulator_classification,
    read_eigenval,
    read_qe_reference_band_path,
    read_vasp_reference_band_path,
    real_space_model,
)
from autowan.gate import CertificationGate, Construction, NonEvaluable
from autowan.interpolation import interpolate
from autowan.w90_outputs import WannierOutputError, WsShifts, read_wsvec

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


A = 3.0
LATTICE = A * np.eye(3)
EF = 5.0
PATH = np.array([[s, 0.0, 0.0] for s in np.linspace(0.0, 0.5, 21)])
MESH = np.array([[i / 8, 0.0, 0.0] for i in range(8)])


def _hr(blocks: dict[int, np.ndarray]):
    """H(R) for R = n x from ``{n: block}``, Hermitian partners added."""
    full = {}
    for n, block in blocks.items():
        full[n] = np.asarray(block, dtype=complex)
        if n:
            full[-n] = np.conjugate(np.asarray(block, dtype=complex)).T
    ns = sorted(full)
    rvec = np.array([[n, 0, 0] for n in ns])
    return rvec, np.ones(len(ns), dtype=int), np.array([full[n] for n in ns])


def _bands(rvec, deg, h, k):
    return interpolate(h, rvec, deg, k)


def _construction(
    blocks,
    *,
    dft_shift=0.0,
    reference=True,
    fermi_scf=EF,
    fermi_nscf=EF,
    extra_dft_band=True,
    mp_grid=(8, 8, 8),
    exclude=(),
    centres=None,
    positions=((0.0, 0.0, 0.0),),
):
    rvec, deg, h = _hr(blocks)
    exact_path = _bands(rvec, deg, h, PATH)
    exact_mesh = _bands(rvec, deg, h, MESH)
    dft = exact_path + dft_shift
    if extra_dft_band:
        dft = np.concatenate([dft, np.full((len(PATH), 1), EF + 30.0)], axis=1)
    if exclude:
        dft = np.concatenate(
            [np.full((len(PATH), len(exclude)), EF - 40.0), dft], axis=1
        )
    parent = SimpleNamespace(
        eigenvalues_ev=exact_mesh[None],
        fermi_energy_ev=fermi_nscf,
        lattice_ang=LATTICE,
        positions_frac=np.array(positions, dtype=float),
        kpoints_cart=MESH,
        alat_ang=A,
    )
    nw = h.shape[1]
    wannier = SimpleNamespace(
        hr_matrix=h,
        hr_rvectors=rvec,
        hr_degeneracies=deg,
        hr_ws_shifts=None,
        centres_ang=np.zeros((nw, 3)) if centres is None else np.asarray(centres),
    )
    ref = (
        ReferenceBandPath(PATH, dft, fermi_scf, fermi_nscf, source="synthetic")
        if reference
        else None
    )
    inputs = {"spin_index": 0, "mp_grid": mp_grid}
    if exclude:
        inputs["exclude_bands"] = tuple(exclude)
    return Construction(
        "synthetic",
        parent=parent,
        wannier=wannier,
        material_class="gapped",
        inputs=inputs,
        reference_path=ref,
    )


def _verdict(check, c):
    report = CertificationGate([check], profile_name="test").run(c)
    return report.verdicts[0]


#: One metallic band crossing E_F: e = E_F + 2 t cos(2 pi k), t = 1 eV.
METAL = {0: [[EF]], 1: [[1.0]]}
#: An insulator: valence centred at E_F - 2, conduction at E_F + 2, each 1 eV wide.
INSULATOR = {0: np.diag([EF - 2.0, EF + 2.0]), 1: np.diag([0.25, -0.25])}


# ---------------------------------------------------------------------------
# criterion: band interpolation
# ---------------------------------------------------------------------------


def test_an_exact_model_has_zero_eta2_and_passes():
    v = _verdict(BAND_INTERPOLATION_ETA2, _construction(METAL))
    assert v.applies and v.evaluable and v.passed
    assert v.metric_value == pytest.approx(0.0, abs=1e-9)
    assert v.threshold_value == 20.0
    assert v.metric_units == "meV"


@pytest.mark.parametrize("shift_mev, passed", [(15.0, True), (30.0, False)])
def test_eta2_is_the_rigid_shift_and_is_judged_at_twenty_mev(shift_mev, passed):
    v = _verdict(
        BAND_INTERPOLATION_ETA2, _construction(METAL, dft_shift=shift_mev / 1e3)
    )
    assert v.metric_value == pytest.approx(shift_mev, rel=1e-6)
    assert v.passed is passed


def test_the_scf_fermi_level_is_used_and_the_nscf_one_recorded():
    # Only the scf level reaches mu; a valence-only error just above the nscf window
    # is weighted at the scf mu and not at the nscf one.
    c = _construction(METAL, fermi_scf=EF, fermi_nscf=EF - 1.5, dft_shift=0.01)
    v = _verdict(BAND_INTERPOLATION_ETA2, c)
    assert v.record["fermi_energy_scf_ev"] == EF
    assert v.record["fermi_energy_nscf_ev"] == EF - 1.5
    assert v.record["mu_ev"] == EF + 2.0
    rvec, deg, h = _hr(METAL)
    dft = c.reference_path.channel(0)
    wan = _bands(rvec, deg, h, PATH)
    assert v.metric_value == pytest.approx(band_distance_eta(dft, wan, EF).eta_mev)
    assert v.record["eta2_mev_at_nscf_fermi"] == pytest.approx(
        band_distance_eta(dft, wan, EF - 1.5).eta_mev
    )


def test_without_a_reference_path_the_criteria_are_non_evaluable_not_failed():
    c = _construction(METAL, reference=False)
    for check in (BAND_INTERPOLATION_ETA2, TRUNCATED_HOPPING_ETA2, CONDUCTION_BAND_ETA):
        v = _verdict(check, c)
        assert v.applies and not v.evaluable and v.passed is None
        assert "reference band path" in v.note


def test_the_gate_supplies_its_reference_to_a_construction_that_has_none():
    c = _construction(METAL, reference=False)
    ref = _construction(METAL).reference_path
    report = CertificationGate(
        [BAND_INTERPOLATION_ETA2], profile_name="t", reference_path=ref
    ).run(c)
    assert report.passed


def test_excluded_dft_bands_are_removed_before_index_matching():
    c = _construction(METAL, exclude=(1, 2))
    v = _verdict(BAND_INTERPOLATION_ETA2, c)
    assert v.metric_value == pytest.approx(0.0, abs=1e-9)
    assert v.record["excluded_dft_bands"] == [1, 2]


def test_a_missing_scf_fermi_level_falls_to_the_decks_fermi_energy_and_says_so():
    c = _construction(METAL, fermi_scf=None)
    c.inputs["fermi_energy"] = "5.0d0"
    v = _verdict(BAND_INTERPOLATION_ETA2, c)
    assert v.evaluable and v.record["fermi_energy_source"] == "deck fermi_energy"
    del c.inputs["fermi_energy"]
    v = _verdict(BAND_INTERPOLATION_ETA2, c)
    assert not v.evaluable and "no scf Fermi level" in v.note


# ---------------------------------------------------------------------------
# criterion: sparse tight binding
# ---------------------------------------------------------------------------


def test_bvk_length_is_the_shortest_superlattice_vector():
    assert bvk_length_ang(LATTICE, (5, 5, 5)) == pytest.approx(15.0)
    assert bvk_length_ang(LATTICE, (8, 8, 2)) == pytest.approx(6.0)
    fcc = 0.5 * 5.43 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert bvk_length_ang(fcc, (11, 11, 11)) == pytest.approx(11 * 5.43 / math.sqrt(2))
    with pytest.raises(ValueError):
        bvk_length_ang(LATTICE, (0, 1, 1))


def test_hoppings_beyond_a_quarter_of_l_bvk_are_dropped_and_the_model_rediagonalised():
    # L = 24 Ang on the 8x8x8 mesh, r_c = 6 Ang: R = 1, 2 (3, 6 Ang) kept, R = 3 dropped.
    blocks = {0: [[EF]], 1: [[1.0]], 2: [[0.1]], 3: [[0.05]]}
    v = _verdict(TRUNCATED_HOPPING_ETA2, _construction(blocks))
    rec = v.record
    assert rec["l_bvk_ang"] == pytest.approx(24.0)
    assert rec["r_c_ang"] == pytest.approx(6.0)
    assert rec["n_vectors_kept"] == 5 and rec["n_vectors_total"] == 7
    assert rec["eta2_untruncated_mev"] == pytest.approx(0.0, abs=1e-9)
    # the dropped 2 * 0.05 cos(6 pi k) term is the whole error
    rvec, deg, h = _hr({0: [[EF]], 1: [[1.0]], 2: [[0.1]]})
    truncated = _bands(rvec, deg, h, PATH)
    dft = _construction(blocks).reference_path.channel(0)
    assert v.metric_value == pytest.approx(
        band_distance_eta(dft, truncated, EF).eta_mev
    )
    assert v.metric_value > 20.0 and v.passed is False
    # L/3 = 8 Ang still drops R = 3; L/2 = 12 Ang keeps it
    assert rec["eta_trunc_L3_mev"] == pytest.approx(v.metric_value)
    assert rec["eta_trunc_L2_mev"] == pytest.approx(0.0, abs=1e-9)


def test_a_short_ranged_model_is_unchanged_by_truncation():
    v = _verdict(TRUNCATED_HOPPING_ETA2, _construction(METAL))
    assert v.passed and v.metric_value == pytest.approx(0.0, abs=1e-9)


def test_truncation_cuts_on_the_hopping_length_not_the_lattice_vector():
    """NiO's elongated AFM cell: two centres far apart inside the cell are bonded across
    the cell face, so their nearest-neighbour hopping sits at R = -1 (|R| = A) while the
    hopping itself is short. Cutting on |R| removed it (v1-v4); v5 keeps it."""
    # centres at x = 0.25 and x = 2.75 Ang in a 3 Ang cell: bonded across x = 0 by 0.5 Ang
    centres = [[0.25, 0.0, 0.0], [A - 0.25, 0.0, 0.0]]
    bond = {0: np.diag([EF, EF]), -1: [[0.0, 0.8], [0.0, 0.0]]}
    c = _construction(bond, centres=centres, mp_grid=(2, 2, 2))
    # L_BvK = 2 A = 6 Ang, r_c = 1.5 Ang < |R| = 3 Ang, > the 0.5 Ang hopping
    model = real_space_model(c.wannier, LATTICE, centres)
    at_minus_one = int(np.flatnonzero((model.vectors == [-1, 0, 0]).all(axis=1))[0])
    assert model.lengths_ang[at_minus_one] == pytest.approx(A)
    assert model.element_lengths_ang[at_minus_one, 0, 1] == pytest.approx(0.5)
    v = _verdict(TRUNCATED_HOPPING_ETA2, c)
    assert v.record["r_c_ang"] == pytest.approx(1.5)
    assert v.metric_value == pytest.approx(0.0, abs=1e-9) and v.passed
    assert "tau_j - tau_i" in v.record["truncation_length"]
    # the same model judged on |R| (no centres) loses the bond and fails
    old = _construction(bond, centres=centres, mp_grid=(2, 2, 2))
    old.wannier.centres_ang = None
    v_old = _verdict(TRUNCATED_HOPPING_ETA2, old)
    assert v_old.metric_value > 20.0 and v_old.passed is False
    assert "|R|" in v_old.record["truncation_length"]


def test_a_long_intracell_hopping_is_cut_even_at_r_zero():
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    far = {0: [[EF, 0.6], [0.6, EF]]}
    c = _construction(far, centres=centres, mp_grid=(2, 2, 2))  # r_c = 1.5 < 2 Ang
    v = _verdict(TRUNCATED_HOPPING_ETA2, c)
    assert v.passed is False and v.record["n_elements_kept"] == 2


def test_the_decay_length_fit_recovers_an_exponential():
    tau, amp = 2.5, 0.8
    blocks = {0: [[EF + amp]]}
    for n in range(1, 6):
        blocks[n] = [[amp * math.exp(-n * A / tau)]]
    c = _construction(blocks)
    fit = decay_lengths_ang(real_space_model(c.wannier, LATTICE), EF)
    assert fit["tau_ang"] == pytest.approx(tau, rel=1e-9)
    assert fit["tau_free_intercept_ang"] == pytest.approx(tau, rel=1e-9)
    v = _verdict(TRUNCATED_HOPPING_ETA2, c)
    assert v.record["tau_ang"] == pytest.approx(tau, rel=1e-9)


def test_a_missing_mesh_leaves_the_sparse_criterion_non_evaluable():
    c = _construction(METAL)
    del c.inputs["mp_grid"]
    v = _verdict(TRUNCATED_HOPPING_ETA2, c)
    assert not v.evaluable and "k-mesh" in v.note


# ---------------------------------------------------------------------------
# criterion: conduction band of insulators
# ---------------------------------------------------------------------------


def test_a_metal_is_withheld_from_the_conduction_band_criterion_with_its_reason():
    v = _verdict(CONDUCTION_BAND_ETA, _construction(METAL))
    assert not v.applies and v.passed is None
    assert v.record["is_insulator"] is False
    assert v.record["crossing_bands_mesh"] == 1
    report = CertificationGate([CONDUCTION_BAND_ETA], profile_name="t").run(
        _construction(METAL)
    )
    assert report.failed_check_ids == () and report.non_evaluable_check_ids == ()


def test_a_gap_below_sigma_is_not_an_insulator():
    # conduction band bottom 0.05 eV above the valence top, nothing crossing E_F
    narrow = {0: np.diag([EF, EF + 0.05]), 1: np.diag([0.0, 0.0])}
    v = _verdict(CONDUCTION_BAND_ETA, _construction(narrow, fermi_scf=EF + 0.02))
    assert not v.applies
    assert v.record["crossing_bands_mesh"] == 0
    assert v.record["gap_ev"] == pytest.approx(0.05)


def test_the_conduction_criterion_is_cumulative_eta_at_cbm_plus_two():
    c = _construction(INSULATOR, dft_shift=0.012)
    v = _verdict(CONDUCTION_BAND_ETA, c)
    assert v.applies and v.evaluable
    cbm = EF + 2.0 - 0.5  # conduction band minimum of e_c - 0.5 cos(2 pi k)
    assert v.record["cbm_ev"] == pytest.approx(cbm)
    assert v.record["mu_ev"] == pytest.approx(cbm + 2.0)
    rvec, deg, h = _hr(INSULATOR)
    wan = _bands(rvec, deg, h, PATH)
    dft = c.reference_path.channel(0)
    expected = band_distance_eta(dft, wan, cbm, nu_ev=2.0, sigma_ev=0.1).eta_mev
    assert v.metric_value == pytest.approx(expected)
    assert v.metric_value == pytest.approx(12.0, rel=1e-6) and v.passed


def test_the_window_only_diagnostic_sees_the_conduction_band_alone():
    # error only in the valence band: cumulative eta sees it, the window-only one not
    rvec, deg, h = _hr(INSULATOR)
    c = _construction(INSULATOR)
    dft = c.reference_path.bands_ev.copy()
    dft[0, :, 0] += 0.05
    c.reference_path = ReferenceBandPath(PATH, dft, EF, EF)
    v = _verdict(CONDUCTION_BAND_ETA, c)
    assert v.metric_value > 20.0
    assert v.record["eta_window_only_mev"] == pytest.approx(0.0, abs=1e-6)


def test_insulator_classification_records_where_e_f_sits():
    c = _construction(INSULATOR, fermi_scf=EF + 0.3)
    rec = insulator_classification(c, c.reference_path, EF + 0.3)
    assert rec["is_insulator"]
    assert rec["vbm_ev"] == pytest.approx(EF - 2.0 + 0.5)
    assert rec["cbm_minus_ef_ev"] == pytest.approx(1.5 - 0.3)
    assert rec["eta2_conduction_coverage_ev"] == pytest.approx(2.0 - 1.2)


# ---------------------------------------------------------------------------
# the minimal-image (wsvec) interpolation
# ---------------------------------------------------------------------------


def test_without_shifts_the_model_is_the_plain_fourier_sum():
    c = _construction({0: np.diag([EF, EF + 1]), 1: [[0.5, 0.1], [0.2, -0.3]]})
    model = real_space_model(c.wannier, LATTICE)
    assert not model.ws_distance_applied
    w = c.wannier
    np.testing.assert_allclose(
        model.bands(PATH),
        interpolate(w.hr_matrix, w.hr_rvectors, w.hr_degeneracies, PATH),
    )


def test_an_element_split_over_images_is_averaged_over_their_phases():
    # one orbital, H(R=1) = t, shifted to R + T with T in {0, -2}: the element sits at
    # R = 1 and R = -1 with half weight each, so H(k) = e0 + t cos(2 pi k) per image pair
    h = np.array([[[2.0]], [[1.0]]], dtype=complex)
    rvec = np.array([[0, 0, 0], [1, 0, 0]])
    deg = np.array([1, 1])
    ws = WsShifts(
        np.array([1, 2]).reshape(2, 1, 1), np.array([[0, 0, 0], [0, 0, 0], [-2, 0, 0]])
    )
    wannier = SimpleNamespace(
        hr_matrix=h, hr_rvectors=rvec, hr_degeneracies=deg, hr_ws_shifts=ws
    )
    model = real_space_model(wannier, LATTICE)
    assert model.ws_distance_applied
    expected = 2.0 + 0.5 * (
        np.exp(2j * np.pi * PATH[:, 0]) + np.exp(-2j * np.pi * PATH[:, 0])
    )
    np.testing.assert_allclose(model.bands(PATH)[:, 0], expected.real, atol=1e-12)


def test_wsvec_is_read_in_the_hamiltonians_order(tmp_path):
    rvec = np.array([[0, 0, 0], [1, 0, 0]])
    lines = ["## written on  4Oct2026 with use_ws_distance=.true."]
    for r in rvec:
        for i in (1, 2):
            for j in (1, 2):
                lines.append(f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{i:5d}{j:5d}")
                n = 2 if (r[0], i, j) == (1, 1, 2) else 1
                lines.append(f"{n:5d}")
                lines += [f"{-2 * k:5d}    0    0" for k in range(n)]
    path = tmp_path / "aiida_wsvec.dat"
    path.write_text("\n".join(lines) + "\n")
    ws = read_wsvec(path, rvec, 2)
    assert ws.use_ws_distance is True
    assert ws.counts.shape == (2, 2, 2) and ws.counts[1, 0, 1] == 2
    assert ws.shifts.shape == (9, 3)
    assert ws.shifts[6].tolist() == [-2, 0, 0]
    with pytest.raises(WannierOutputError, match="order"):
        read_wsvec(path, rvec[::-1], 2)


# ---------------------------------------------------------------------------
# the readers
# ---------------------------------------------------------------------------


def test_the_qe_reader_takes_the_scf_fermi_level_from_scf_out(tmp_path):
    bands = np.arange(12.0).reshape(2, 2, 3)
    np.savez(tmp_path / "dft_bands_path.npz", bands=bands, kpoints=np.zeros((2, 3)))
    (tmp_path / "scf.out").write_text("     the Fermi energy is    13.4187 ev\n")
    (tmp_path / "nscf.out").write_text("     the Fermi energy is    13.4190 ev\n")
    ref = read_qe_reference_band_path(tmp_path / "dft_bands_path.npz")
    assert ref.fermi_energy_ev == 13.4187 and ref.nscf_fermi_energy_ev == 13.4190
    assert ref.n_spin == 2 and ref.channel(1).tolist() == bands[1].tolist()
    with pytest.raises(NonEvaluable):
        ref.channel(2)
    given = read_qe_reference_band_path(
        tmp_path / "dft_bands_path.npz", fermi_energy_ev=1.0
    )
    assert given.fermi_energy_ev == 1.0 and given.fermi_source == "given"


def _eigenval(path, kpoints, bands):
    n_spin, n_k, n_b = bands.shape
    out = [f"    2    2    1    {n_spin}", "  x", "  1e-4", "  CAR", " test"]
    out.append(f"      8    {n_k}     {n_b}")
    for ik, k in enumerate(kpoints):
        out.append("")
        out.append("  " + "  ".join(f"{x:.7E}" for x in k) + "  0.1E+00")
        for ib in range(n_b):
            es = "  ".join(f"{bands[s, ik, ib]:.6f}" for s in range(n_spin))
            occ = "  ".join("1.0" for _ in range(n_spin))
            out.append(f"  {ib + 1}  {es}  {occ}")
    path.write_text("\n".join(out) + "\n")


@pytest.mark.parametrize("n_spin", [1, 2])
def test_the_vasp_reader_takes_the_parents_scf_fermi_level(tmp_path, n_spin):
    parent = tmp_path / "parent"
    (parent / "scf").mkdir(parents=True)
    (parent / "nscf").mkdir()
    (parent / "scf" / "OUTCAR").write_text(
        " E-fermi :   6.0000\n E-fermi :   6.0201  XC\n"
    )
    (parent / "nscf" / "OUTCAR").write_text(" E-fermi :   6.0970  XC\n")
    run = tmp_path / "Si"
    run.mkdir()
    k = np.array([[0.0, 0.0, 0.0], [0.0, 0.0111111111, 0.0111111111]])
    bands = np.arange(n_spin * 2 * 3, dtype=float).reshape(n_spin, 2, 3)
    _eigenval(run / "EIGENVAL", k, bands)
    (run / "settings.json").write_text(json.dumps({"parent_dir": str(parent)}))
    (run / "path.json").write_text(
        json.dumps({"kpoints_frac": k.tolist(), "labels": []})
    )
    ref = read_vasp_reference_band_path(run)
    assert ref.fermi_energy_ev == 6.0201 and ref.nscf_fermi_energy_ev == 6.0970
    np.testing.assert_allclose(ref.bands_ev, bands)
    np.testing.assert_allclose(ref.kpoints_frac, k, atol=1e-9)
    (run / "path.json").write_text(json.dumps({"kpoints_frac": (k + 0.1).tolist()}))
    with pytest.raises(ValueError, match="path.json"):
        read_vasp_reference_band_path(run)


def test_read_eigenval_shapes(tmp_path):
    k = np.zeros((3, 3))
    bands = np.random.default_rng(0).normal(size=(2, 3, 4))
    _eigenval(tmp_path / "EIGENVAL", k, bands)
    kk, bb = read_eigenval(tmp_path / "EIGENVAL")
    assert kk.shape == (3, 3) and bb.shape == (2, 3, 4)
    np.testing.assert_allclose(bb, np.round(bands, 6))


# ---------------------------------------------------------------------------
# the drift guard and the atom-centred check
# ---------------------------------------------------------------------------


DRIFT = next(c for c in CORE_CHECKS if c.check_id == "centres_stay_near_home_cell")


def _centred(centres, positions=((0.0, 0.0, 0.0),)):
    return _construction(METAL, centres=np.atleast_2d(centres), positions=positions)


def test_a_centre_on_its_atom_has_not_drifted():
    v = _verdict(DRIFT, _centred([[0.0, 0.0, 0.0]]))
    assert v.metric_value == pytest.approx(0.0) and v.passed
    assert v.threshold_value == pytest.approx(A)  # one nearest-neighbour distance


def test_bond_and_interstitial_centres_pass():
    bond = _centred([[-0.5 * A, 0.0, 0.0]])  # half a bond, across the face x = 0
    v = _verdict(DRIFT, bond)
    assert v.metric_value == pytest.approx(0.5 * A) and v.passed
    body = _centred([[0.5 * A, 0.5 * A, 0.5 * A]])  # the cube's centre
    v = _verdict(DRIFT, body)
    assert v.metric_value == pytest.approx(0.5 * A * math.sqrt(3.0)) and v.passed


def test_a_centre_run_away_into_a_slab_vacuum_fails():
    c = _centred([[0.0, 0.0, 10.0]])
    c.parent.lattice_ang = np.diag([A, A, 20.0])
    v = _verdict(DRIFT, c)
    assert v.metric_value == pytest.approx(10.0) and v.passed is False


@pytest.mark.parametrize(
    "centre, position",
    [
        ([-2.0 * A, 0.0, 0.0], (0.0, 0.0, 0.0)),  # centre reported two cells away
        (
            [-0.75 * A, 0.0, 0.0],
            (-0.75, 0.0, 0.0),
        ),  # atom written at a negative coordinate
        ([0.25 * A, 0.0, 0.0], (-0.75, 0.0, 0.0)),  # the same atom, wrapped centre
        ([-0.75 * A, 0.0, 0.0], (1.25, 0.0, 0.0)),  # atom written one cell up
    ],
)
def test_a_centre_on_an_atom_image_measures_zero_in_any_frame(centre, position):
    v = _verdict(DRIFT, _centred([centre], positions=(position,)))
    assert v.metric_value == pytest.approx(0.0, abs=1e-9) and v.passed


def test_the_tic_case_an_image_two_cells_away_on_an_fcc_cell():
    """VASP TiC (runs_v2): C functions at fractional (-1.5, 0.5, 0.5), an image of C at
    (0.5, 0.5, 0.5); the v1-v4 guard refused them, at 0.000 Angstrom from C."""
    a = 4.33
    fcc = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    c = _centred(
        [np.array([-1.5, 0.5, 0.5]) @ fcc, [0.0, 0.0, 0.0]],
        positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )
    c.parent.lattice_ang = fcc
    assert distance_outside_home_cell_ang(c)[0] > a / 2  # the old measure
    v = _verdict(DRIFT, c)
    assert v.metric_value == pytest.approx(0.0, abs=1e-9) and v.passed
    assert v.threshold_value == pytest.approx(a / 2)  # Ti-C nearest neighbours
    assert nearest_atom_distance_ang(c) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_the_distance_to_an_oblique_cell_is_exact():
    fcc = 0.5 * 4.0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    c = _centred([[-1.0, -1.0, -1.0]])
    c.parent.lattice_ang = fcc
    # the nearest point of the cell to (-1, -1, -1) is the corner at the origin
    assert distance_outside_home_cell_ang(c)[0] == pytest.approx(math.sqrt(3.0))


@pytest.mark.parametrize("offset, passed", [(0.05, True), (0.2, False)])
def test_the_atom_centred_check_is_judged_at_a_tenth_of_an_angstrom(offset, passed):
    v = _verdict(CENTRES_ON_ATOMS_ATOM_CENTRED, _centred([[offset, 0.0, 0.0]]))
    assert v.threshold_value == pytest.approx(0.1)
    assert v.metric_value == pytest.approx(offset)
    assert v.passed is passed


# ---------------------------------------------------------------------------
# the stored campaign trees (not shipped; skipped when absent)
# ---------------------------------------------------------------------------

VASP_BAND_PATHS = _stored("AUTOWAN_TEST_VASP_BAND_PATHS")
QE_BASELINE_RAW = _stored("AUTOWAN_TEST_QE_BASELINE_RAW")


@pytest.mark.skipif(
    not (VASP_BAND_PATHS / "Si" / "EIGENVAL").is_file()
    or not (QE_BASELINE_RAW / "Si").is_dir(),
    reason="the VASP band-path run tree and the QE baseline are not shipped",
)
def test_the_vasp_band_path_sits_on_the_qe_reference_k_points():
    vasp = read_vasp_reference_band_path(VASP_BAND_PATHS / "Si")
    qe = read_qe_reference_band_path(
        QE_BASELINE_RAW / "Si" / "A_external_fixed" / "dft_bands_path.npz"
    )
    assert vasp.bands_ev.shape == (1, 202, 32)
    np.testing.assert_allclose(vasp.kpoints_frac, qe.kpoints_frac, atol=1e-6)
    assert vasp.fermi_energy_ev == pytest.approx(6.0201)
    assert qe.fermi_energy_ev == pytest.approx(6.5717)
    # the two codes' valence bands, aligned at the VBM, agree to tens of meV
    v = vasp.channel(0)[:, :4] - vasp.channel(0)[:, :4].max()
    q = qe.channel(0)[:, :4] - qe.channel(0)[:, :4].max()
    assert np.abs(v - q).max() < 0.05
