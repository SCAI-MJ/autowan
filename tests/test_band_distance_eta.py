"""The paper's band-interpolation error eta_nu, reported beside the gate verdict.

Definition (Jiang, Qiao, Paulish, Zhao, Marzari, Pizzi, arXiv:2507.06840, eqs. 6-7, after
Vitale et al.; computed in the paper by the authors' open-source workflows,
``aiida_wannier90_workflows.utils.bands.distance``):

    eta_nu     = sqrt( sum_nk f_nk (e_DFT_nk - e_Wan_nk)^2 / sum_nk f_nk )
    eta_max_nu = max_nk ( f_nk |e_DFT_nk - e_Wan_nk| )
    f_nk       = sqrt( f_FD(e_DFT_nk; E_F + nu, sigma) * f_FD(e_Wan_nk; E_F + nu, sigma) )

with nu = 2 eV and sigma = 0.1 eV; success is eta_2 <= 20 meV. Bands are matched by index:
the DFT bands the deck excluded (1-based ``exclude_bands``) are removed, and the lowest
min(n_DFT_kept, num_wann) bands of each set are paired in ascending order at every k-point.
The authors' code also gives zero weight to states whose DFT energy is at or below an
absolute -30 eV (``lower_cutoff=-30``), a convention the paper's text does not state; it
is the default here, named and switchable, so the paper's numbers and ours are the same
computation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autowan.band_distance_eta import (
    ETA2_SUCCESS_MEV,
    ETA_NU_EV,
    ETA_SIGMA_EV,
    REFERENCE_LOWER_CUTOFF_EV,
    band_distance_eta,
    eta2_record,
    path_points_on_mesh,
    read_wannier_path_bands,
    read_wannier_path_kpoints,
)

EF = 5.0


def _fd(e, mu, sigma=0.1):
    return 1.0 / (np.exp((e - mu) / sigma) + 1.0)


def _valence(nk: int = 6) -> np.ndarray:
    """Two well-separated bands below E_F, ``(nk, 2)``."""
    k = np.linspace(0.0, 1.0, nk)
    return np.stack(
        [EF - 4.0 + 0.3 * np.cos(2 * np.pi * k), EF - 1.5 + 0.2 * k], axis=1
    )


def test_the_published_parameters():
    assert ETA_NU_EV == 2.0
    assert ETA_SIGMA_EV == 0.1
    assert ETA2_SUCCESS_MEV == 20.0
    assert REFERENCE_LOWER_CUTOFF_EV == -30.0


@pytest.mark.parametrize("shift_mev", [10.0, 30.0])
def test_a_rigid_shift_has_eta_equal_to_the_shift(shift_mev):
    # sum f d^2 / sum f with a constant d is d^2, whatever the weights.
    dft = _valence()
    wan = dft + shift_mev / 1000.0
    result = band_distance_eta(dft, wan, EF)
    assert result.eta_mev == pytest.approx(shift_mev, rel=1e-9)
    assert result.passes == (shift_mev <= 20.0)


def test_pass_and_fail_either_side_of_twenty_millielectronvolts():
    dft = _valence()
    assert eta2_record(dft, dft + 0.0199, EF)["eta2_pass_20mev"] is True
    assert eta2_record(dft, dft + 0.0201, EF)["eta2_pass_20mev"] is False


def test_the_effective_weight_is_the_geometric_mean_of_the_two_fermi_functions():
    # One exact band deep below E_F and one band straddling mu = E_F + 2 eV with a
    # 10 meV error: the known answer follows from the definition directly.
    mu = EF + 2.0
    dft = np.array([[EF - 1.0, mu - 0.005]])
    wan = np.array([[EF - 1.0, mu + 0.005]])
    w1 = math.sqrt(_fd(EF - 1.0, mu) ** 2)
    w2 = math.sqrt(_fd(mu - 0.005, mu) * _fd(mu + 0.005, mu))
    expected = math.sqrt(w2 * 0.010**2 / (w1 + w2)) * 1000.0
    result = band_distance_eta(dft, wan, EF)
    assert result.eta_mev == pytest.approx(expected, rel=1e-12)
    assert result.eta_max_mev == pytest.approx(w2 * 10.0, rel=1e-12)


def test_errors_far_above_the_window_carry_no_weight():
    dft = np.concatenate([_valence(), np.full((6, 1), EF + 10.0)], axis=1)
    wan = dft.copy()
    wan[:, 2] += 1.0  # a 1 eV error 8 eV above E_F + 2 eV
    assert band_distance_eta(dft, wan, EF).eta_mev < 1e-6


def test_excluded_dft_bands_are_removed_before_matching_by_index():
    semicore = np.full((6, 1), -20.0)
    dft = np.concatenate([semicore, _valence()], axis=1)
    wan = _valence()
    matched = band_distance_eta(dft, wan, EF, exclude_bands=(1,))
    assert matched.eta_mev == 0.0
    assert matched.n_bands_compared == 2
    unmatched = band_distance_eta(dft, wan, EF)
    assert (
        unmatched.eta_mev > 1000.0
    )  # the semicore is paired with the first valence band


def test_only_the_lowest_num_wann_dft_bands_are_compared():
    dft = np.concatenate([_valence(), np.full((6, 1), EF + 1.0)], axis=1)
    wan = _valence()  # two Wannier functions; the third DFT band is not represented
    result = band_distance_eta(dft, wan, EF)
    assert result.eta_mev == 0.0
    assert result.n_bands_compared == 2


def test_states_at_or_below_the_reference_lower_cutoff_carry_no_weight():
    deep = np.full((6, 1), -40.0)
    dft = np.concatenate([deep, _valence()], axis=1)
    wan = dft.copy()
    wan[:, 0] += 0.5  # a 500 meV error on a band below -30 eV absolute
    assert band_distance_eta(dft, wan, EF).eta_mev == 0.0
    no_cutoff = band_distance_eta(dft, wan, EF, lower_cutoff_ev=None)
    assert no_cutoff.eta_mev == pytest.approx(500.0 / math.sqrt(3), rel=1e-6)


def test_non_finite_wannier_bands_give_no_eta_and_no_verdict():
    dft = _valence()
    wan = dft.copy()
    wan[2, 1] = np.nan
    record = eta2_record(dft, wan, EF)
    assert record["eta2_mev"] is None
    assert record["eta2_pass_20mev"] is None


def test_a_k_point_count_mismatch_is_an_error():
    with pytest.raises(ValueError):
        band_distance_eta(_valence(6), _valence(5), EF)


def test_the_record_names_its_conventions():
    record = eta2_record(_valence(), _valence(), EF, exclude_bands=())
    for key in (
        "eta2_mev",
        "eta2_max_mev",
        "eta2_pass_20mev",
        "nu_ev",
        "sigma_ev",
        "fermi_energy_ev",
        "lower_cutoff_ev",
        "n_kpoints",
        "n_bands_compared",
        "excluded_dft_bands",
    ):
        assert key in record, key
    assert record["fermi_energy_ev"] == EF
    assert record["eta2_mev"] == 0.0 and record["eta2_pass_20mev"] is True


def test_eta_equals_the_authors_reference_implementation():
    ref = pytest.importorskip("aiida_wannier90_workflows.utils.bands.distance")
    rng = np.random.default_rng(7)
    nk, nb, nw = 40, 12, 8
    dft = np.sort(rng.uniform(-35.0, 15.0, size=(nk, nb)), axis=1)
    wan = np.sort(dft[:, 2 : 2 + nw] + rng.normal(0.0, 0.05, size=(nk, nw)), axis=1)
    exclude = [1, 2]
    table = ref.bands_distance(dft, wan, fermi_energy=EF, exclude_list_dft=exclude)
    row = table[np.argmin(np.abs(table[:, 0] - (EF + 2.0)))]
    ours = band_distance_eta(dft, wan, EF, exclude_bands=tuple(exclude))
    assert ours.eta_mev == pytest.approx(row[1] * 1000.0, rel=1e-12)
    assert ours.eta_max_mev == pytest.approx(row[3] * 1000.0, rel=1e-12)


# ---------------------------------------------------------------------------
# the k-set: Wannier90's band path, and how much of it sits on the construction mesh
# ---------------------------------------------------------------------------


def test_path_points_on_a_gamma_centred_mesh_are_counted():
    path = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma: on every Gamma-centred mesh
            [0.5, 0.0, 0.0],  # on an even mesh along x
            [0.25, 0.0, 0.0],  # on 4x4x4, not on 2x2x2
            [0.1, 0.2, 0.3],  # on neither
            [1.0, -0.5, 0.0],  # equivalent to (0, 0.5, 0) modulo a reciprocal vector
        ]
    )
    assert path_points_on_mesh(path, (2, 2, 2)) == 3
    assert path_points_on_mesh(path, (4, 4, 4)) == 4


def test_wannier90_band_files_are_read(tmp_path):
    (tmp_path / "aiida_band.dat").write_text(
        "  0.0 -1.0\n  0.5 -0.9\n  1.0 -0.8\n\n  0.0 2.0\n  0.5 2.1\n  1.0 2.2\n\n"
    )
    (tmp_path / "aiida_band.kpt").write_text(
        "      3\n 0.0 0.0 0.0 1.0\n 0.25 0.0 0.0 1.0\n 0.5 0.0 0.0 1.0\n"
    )
    bands = read_wannier_path_bands(tmp_path / "aiida_band.dat")
    assert bands.shape == (3, 2)
    assert bands[1].tolist() == [-0.9, 2.1]
    kpts = read_wannier_path_kpoints(tmp_path / "aiida_band.kpt")
    assert kpts.shape == (3, 3)
    assert kpts[2].tolist() == [0.5, 0.0, 0.0]
