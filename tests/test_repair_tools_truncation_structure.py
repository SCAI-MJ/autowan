"""The second-wave read-only tools: truncation_diagnostics, band_error_by_energy, structure.

Why they exist (poster campaign, 2026-10-09): a sparse_tb run that fails criterion 2 had
no way to see *which* hoppings beyond r_c = L_BvK/4 carried the error, and spent its
attempts cycling projector sets blind; other runs needed structure facts (atoms stored
outside the home cell, a slab's vacuum, neighbour distances) that no tool returned.

Synthetic constructions reuse ``test_criteria``'s chain model (cubic, a = 3 Angstrom),
so every number is known in closed form. The structure fixtures are a NiO-like
rhombohedral antiferromagnetic cell (elongated along [111], one oxygen stored outside the
cell) and a monolayer slab with 16 Angstrom of vacuum.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_criteria import EF, INSULATOR, METAL, _construction

from autowan.criteria import ReferenceBandPath
from autowan.orchestrator import Run
from autowan.tools import (
    ENERGY_WINDOWS_EV,
    RepairTools,
    describe_structure,
    nearest_site,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

#: NiO in its type-II antiferromagnetic cell: rhombohedral, the fcc conventional cell
#: (a = 4.17 Angstrom) doubled along [111]. Ni-O is a/2 = 2.085 Angstrom, Ni-Ni a/sqrt 2.
NIO_A = 4.17
NIO_LATTICE = (NIO_A / 2) * np.array(
    [[1.0, 1.0, 2.0], [1.0, 2.0, 1.0], [2.0, 1.0, 1.0]]
)
NIO_SPECIES = ("Ni1", "Ni2", "O", "O")
#: The second oxygen is stored one lattice translation outside the cell.
NIO_FRAC = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [-0.25, -0.25, -0.25]]
)

#: A MoS2-like monolayer: hexagonal plane, 3.1 Angstrom thick, centred in c = 19.1.
SLAB_C = 19.1
SLAB_LATTICE = np.array([[3.19, 0.0, 0.0], [-1.595, 2.7626, 0.0], [0.0, 0.0, SLAB_C]])
SLAB_SPECIES = ("Mo", "S", "S")
SLAB_FRAC = np.array(
    [
        [0.0, 0.0, 0.5],
        [1 / 3, 2 / 3, 0.5 - 1.55 / SLAB_C],
        [1 / 3, 2 / 3, 0.5 + 1.55 / SLAB_C],
    ]
)


def _tools_on(c, *, profile="sparse_tb", parent=None):
    """RepairTools whose last judged construction is ``c``."""
    run = Run("synthetic")
    attempt = run.start_attempt({"x": 1}, seed="synthetic", method="smv")
    attempt.outcome = SimpleNamespace(failed_check_ids=())
    run.gate_profile = {"name": profile}
    parent = parent or c.parent
    if not hasattr(parent, "species"):
        parent.species = tuple("X" for _ in parent.positions_frac)
    return RepairTools(
        ctx=SimpleNamespace(parent=parent, code="qe", spin_index=0),
        backend=None,
        run=run,
        inputs=SimpleNamespace(win=Path("/nonexistent")),
        seed="synthetic",
        construction_of=lambda seed: c,
    )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_an_atom_stored_outside_the_cell_is_flagged_and_wrapped():
    s = describe_structure(NIO_LATTICE, NIO_SPECIES, NIO_FRAC, (6, 6, 6))
    assert s["atoms_stored_outside_cell"] == 1
    o2 = s["atoms"][3]
    assert o2["site"] == "O#4" and o2["stored_outside_cell"]
    assert o2["frac_stored"] == [-0.25, -0.25, -0.25]
    assert o2["frac_wrapped"] == [0.75, 0.75, 0.75]
    assert o2["lattice_translation"] == [-1, -1, -1]
    assert not any(a["stored_outside_cell"] for a in s["atoms"][:3])


def test_nio_like_elongated_cell_neighbours_are_the_rock_salt_shells():
    s = describe_structure(NIO_LATTICE, NIO_SPECIES, NIO_FRAC, (6, 6, 6))
    assert s["nearest_neighbour_ang"] == pytest.approx(NIO_A / 2, abs=1e-4)
    by_site = {row["site"]: row for row in s["neighbours"]}
    for ni in ("Ni1#1", "Ni2#2"):
        first, second = by_site[ni]["shells"]
        assert first["distance_ang"] == pytest.approx(NIO_A / 2, abs=1e-4)
        assert first["neighbours"] == {"O": 6}
        assert second["distance_ang"] == pytest.approx(NIO_A / np.sqrt(2), abs=1e-4)
        # 12 Ni neighbours: 6 of each sublattice in the type-II order
        assert second["neighbours"] == {"Ni1": 6, "Ni2": 6}
    assert by_site["O#3"]["shells"][0]["neighbours"] == {"Ni1": 3, "Ni2": 3}
    # bulk: no direction is a vacuum direction, every axis periodic
    assert s["periodicity"]["periodic_dims"] == 3
    assert all(a["periodic"] for a in s["periodicity"]["axes"])
    assert s["species_counts"] == {"Ni1": 1, "Ni2": 1, "O": 2}


def test_the_structure_reading_does_not_depend_on_how_skewed_the_basis_is():
    # the same lattice in a much more elongated basis: a3' = a3 + 2 a1 + 2 a2
    transform = np.array([[1, 0, 0], [0, 1, 0], [2, 2, 1]])
    skewed = transform @ NIO_LATTICE
    cart = NIO_FRAC @ NIO_LATTICE
    frac = np.linalg.solve(skewed.T, cart.T).T
    plain = describe_structure(NIO_LATTICE, NIO_SPECIES, NIO_FRAC)
    sheared = describe_structure(skewed, NIO_SPECIES, frac)
    for a, b in zip(plain["neighbours"], sheared["neighbours"]):
        assert a["shells"] == b["shells"]
    # and a point's nearest atom is the same atom, at the same distance
    point = np.array([0.3, -0.2, 9.0])
    assert nearest_site(NIO_LATTICE, NIO_FRAC, point) == pytest.approx(
        nearest_site(skewed, frac, point)
    )


def test_a_slab_reports_its_vacuum_and_two_periodic_directions():
    s = describe_structure(SLAB_LATTICE, SLAB_SPECIES, SLAB_FRAC, (8, 8, 1))
    per = s["periodicity"]
    assert per["periodic_dims"] == 2
    axes = per["axes"]
    assert [a["periodic"] for a in axes] == [True, True, False]
    assert axes[2]["widest_empty_layer_ang"] == pytest.approx(SLAB_C - 3.1, abs=1e-3)
    assert axes[2]["empty_layer_over_nn"] > 3
    assert "mp_grid = 1" in per["rule"]
    assert s["angles_deg"]["gamma"] == pytest.approx(120.0, abs=1e-3)


def test_a_slab_sampled_along_c_is_not_a_slab_and_no_mesh_says_unknown():
    sampled = describe_structure(SLAB_LATTICE, SLAB_SPECIES, SLAB_FRAC, (8, 8, 2))
    assert sampled["periodicity"]["periodic_dims"] == 3
    unknown = describe_structure(SLAB_LATTICE, SLAB_SPECIES, SLAB_FRAC)
    assert "periodic_dims" not in unknown["periodicity"]
    assert {a["periodic"] for a in unknown["periodicity"]["axes"]} == {None}
    # the empty layer is still measured without a mesh
    assert unknown["periodicity"]["axes"][2]["widest_empty_layer_ang"] > 15


def test_the_structure_tool_names_no_material(tmp_path):
    win = tmp_path / "wan.win"
    win.write_text("num_wann = 4\nmp_grid = 6 6 6\n")
    parent = SimpleNamespace(
        lattice_ang=NIO_LATTICE, species=NIO_SPECIES, positions_frac=NIO_FRAC
    )
    tools = RepairTools(
        ctx=SimpleNamespace(parent=parent),
        backend=None,
        run=Run("NiO_afm2"),
        inputs=SimpleNamespace(win=win),
        seed="NiO_afm2",
    )
    text = tools("structure", {})
    data = json.loads(text)
    assert data["periodicity"]["mp_grid"] == [6, 6, 6]
    assert "NiO" not in text and "formula" not in text


# ---------------------------------------------------------------------------
# band_error_by_energy
# ---------------------------------------------------------------------------


def test_a_rigid_shift_is_the_same_eta_in_every_window():
    c = _construction(METAL, dft_shift=0.03)
    data = json.loads(_tools_on(c)("band_error_by_energy", {}))
    assert [w["nu_ev"] for w in data["windows"]] == list(ENERGY_WINDOWS_EV)
    for w in data["windows"]:
        assert w["eta_mev"] == pytest.approx(30.0, abs=1e-3)
        assert w["worst_bands"][0]["band_dft"] == 1
        assert w["worst_bands"][0]["largest_abs_difference_mev"] == pytest.approx(30.0)
    assert data["construction_attempt"] == 0
    assert data["fermi_energy_ev"] == EF


def _with_conduction_error(c, error_ev):
    ref = c.reference_path
    bands = ref.bands_ev[0].copy()
    bands[:, 1] += error_ev  # the conduction band only
    c.reference_path = ReferenceBandPath(
        ref.kpoints_frac, bands, ref.fermi_energy_ev, ref.nscf_fermi_energy_ev
    )
    return c


def test_an_error_in_the_conduction_band_shows_only_in_the_upper_windows():
    c = _with_conduction_error(_construction(INSULATOR), 0.05)
    data = json.loads(_tools_on(c)("band_error_by_energy", {}))
    eta = {w["nu_ev"]: w["eta_mev"] for w in data["windows"]}
    # the conduction band spans E_F + 1.5 .. 2.5 eV: invisible at E_F + 0 and + 1
    assert eta[0.0] < 0.05 and eta[1.0] < 5.0 < 20.0 < eta[3.0]
    assert eta[3.0] > 20.0 and eta[5.0] > eta[2.0]
    top = data["windows"][-1]
    assert top["worst_bands"][0]["band_dft"] == 2
    assert top["worst_bands"][0]["dft_span_minus_ef_ev"][0] == pytest.approx(
        1.55, abs=0.06
    )
    k = top["worst_kpoints"][0]
    assert k["worst_band_dft"] == 2 and k["e_wan_minus_dft_mev"] == pytest.approx(-50.0)
    # at E_F the occupation tail leaves the error all but invisible
    assert all(r["eta_max_mev"] < 0.01 for r in data["windows"][0]["worst_kpoints"])


def test_without_a_judged_construction_the_criterion_tools_say_so():
    tools = RepairTools(ctx=None, backend=None, run=Run("x"), inputs=None, seed="x")
    for name in ("band_error_by_energy", "truncation_diagnostics"):
        assert "no construction has run far enough" in tools(name, {})


def test_without_a_reference_path_the_error_is_named_not_raised():
    c = _construction(METAL, reference=False)
    data = json.loads(_tools_on(c)("band_error_by_energy", {}))
    assert "no reference band path" in data["error"]


# ---------------------------------------------------------------------------
# truncation_diagnostics
# ---------------------------------------------------------------------------

#: Two functions per cell along the chain, centred 1.4 Angstrom apart (as an elongated
#: cell's functions are): mp_grid 8 x 8 x 8 on a = 3 gives L_BvK = 24, r_c = 6 Angstrom.
#: R = 2 couples function 1 to function 2 of the cell two along: |R| = 6 is inside
#: r_c, but the hopping itself is |R + tau_2 - tau_1| = 7.4, beyond it. R = 3 couples
#: function 1 to itself at 9 Angstrom.
TWO_CENTRES = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
LONG_RANGED = {
    0: [[EF, 0.5], [0.5, EF + 0.3]],
    1: [[0.3, 0.0], [0.4, 0.2]],
    2: [[0.0, 0.02], [0.0, 0.0]],
    3: [[0.05, 0.0], [0.0, 0.0]],
}
#: |H|^2 without the on-site energies: R = 0 off-diagonal, then R = +-1, +-2, +-3.
TOTAL_H2 = 2 * 0.25 + 2 * (0.09 + 0.16 + 0.04) + 2 * 0.02**2 + 2 * 0.05**2
BEYOND_H2 = 2 * 0.02**2 + 2 * 0.05**2


def _long_ranged():
    c = _construction(LONG_RANGED, centres=TWO_CENTRES)
    c.parent.species = ("X",)
    return c


def test_truncation_diagnostics_reports_the_range_and_the_criterion_verdict():
    data = json.loads(_tools_on(_long_ranged())("truncation_diagnostics", {}))
    assert data["profile"] == "sparse_tb" and data["judged_by_profile"]
    assert data["criterion"] == "truncated_hopping_eta2" and data["evaluable"]
    rng = data["range"]
    assert rng["l_bvk_ang"] == pytest.approx(24.0)
    assert rng["r_c_ang"] == pytest.approx(6.0)
    assert rng["periodic_dims"] == 3
    assert "shortest non-zero vector" in rng["l_bvk_definition"]
    assert data["mp_grid"] == [8, 8, 8]
    # the reference is the exact model: untruncated eta is zero, truncated is not
    assert data["eta2_untruncated_mev"] == pytest.approx(0.0, abs=1e-6)
    assert data["eta2_truncated_mev"] > 0 and data["passed"] is (
        data["eta2_truncated_mev"] <= 20.0
    )
    assert data["decay"]["tau_ang"] is not None


def test_the_hopping_norm_is_binned_by_the_real_hopping_distance():
    data = json.loads(_tools_on(_long_ranged())("truncation_diagnostics", {}))
    bins = data["hopping_norm_by_distance"]
    assert bins["shell_width_ang"] == pytest.approx(1.5)
    shells = {tuple(s["d_ang"]): s for s in bins["shells"]}
    # R = 2's hopping sits at 7.4 Angstrom (beyond r_c), not at |R| = 6 (inside it)
    beyond_74 = shells[(6.0, 7.5)]
    assert beyond_74["beyond_r_c"] and beyond_74["largest_ev"] == pytest.approx(0.02)
    assert shells[(7.5, 9.0)]["largest_ev"] == pytest.approx(0.05)
    assert not shells[(0.0, 1.5)]["beyond_r_c"]
    assert sum(s["share"] for s in bins["shells"]) == pytest.approx(1.0, abs=1e-5)
    beyond = data["beyond_r_c"]
    assert beyond["share_of_total_h2"] == pytest.approx(BEYOND_H2 / TOTAL_H2, rel=1e-4)
    assert beyond["largest_element_ev"] == pytest.approx(0.05)
    assert beyond["largest_element_distance_ang"] == pytest.approx(9.0)


def test_the_functions_carrying_the_beyond_r_c_weight_are_named_with_centres():
    data = json.loads(_tools_on(_long_ranged())("truncation_diagnostics", {}))
    rows = data["functions_carrying_beyond_r_c_weight"]
    # function 1 owns all of R = 3 and half of R = 2
    first = rows[0]
    assert first["wf"] == 1
    expected = (2 * 0.05**2 + 2 * 0.02**2 / 2) / BEYOND_H2
    assert first["share_of_beyond_r_c_h2"] == pytest.approx(expected, abs=1e-4)
    assert first["nearest_atom"] == "X#1" and first["distance_ang"] == pytest.approx(0)
    # the list stops once 90 % of the beyond-r_c weight is covered
    assert len(rows) == 1


def test_the_truncated_models_error_is_located_by_window():
    data = json.loads(_tools_on(_long_ranged())("truncation_diagnostics", {}))
    err = data["truncated_error"]
    assert [w["nu_ev"] for w in err["windows"]] == [0.0, 1.0, 2.0]
    # eta at E_F + 2 eV of the truncated model is the criterion's own number
    assert err["windows"][2]["eta_mev"] == pytest.approx(
        data["eta2_truncated_mev"], rel=1e-3
    )
    assert err["windows"][2]["worst_kpoints"] and err["windows"][2]["worst_bands"]


def test_another_profile_reads_the_criterion_without_judging_it():
    data = json.loads(
        _tools_on(_long_ranged())(
            "truncation_diagnostics", {"profile": "band_interpolation"}
        )
    )
    assert data["profile"] == "band_interpolation"
    assert data["judged_by_profile"] is False
    assert data["eta2_truncated_mev"] > 0
    d_only = json.loads(
        _tools_on(_long_ranged())("truncation_diagnostics", {"profile": "downfolded_d"})
    )
    assert d_only["judged_by_profile"] is False
    bad = json.loads(
        _tools_on(_long_ranged())("truncation_diagnostics", {"profile": "x"})
    )
    assert "gate profile 'x'" in bad["error"]


def test_without_a_mesh_the_criterion_is_non_evaluable_and_the_tool_says_why():
    c = _long_ranged()
    del c.inputs["mp_grid"]
    data = json.loads(_tools_on(c)("truncation_diagnostics", {}))
    assert data["evaluable"] is False and "k-mesh" in data["note"]
    assert "hopping_norm_by_distance" not in data


def test_a_short_ranged_model_has_nothing_beyond_r_c():
    c = _construction({0: [[EF]], 1: [[1.0]]})
    c.parent.species = ("X",)
    data = json.loads(_tools_on(c)("truncation_diagnostics", {}))
    assert data["beyond_r_c"]["share_of_total_h2"] == 0.0
    assert data["functions_carrying_beyond_r_c_weight"] == []
    assert data["eta2_truncated_mev"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# registration: offered in the loop, counted against the per-attempt cap
# ---------------------------------------------------------------------------


def test_the_new_tools_are_offered_and_counted_in_the_repair_loop(tmp_path):
    import repair_fixtures as fx

    from autowan.orchestrator import RunConfig
    from autowan.prompts import GENERIC_PRINCIPLES
    from autowan.workflow import WorkflowConfig, wannierize

    calls = [
        ("structure", {}),
        ("truncation_diagnostics", {"profile": None}),
        ("band_error_by_energy", {}),
    ]
    fx.write_deck(tmp_path)
    model = fx.ToolScript([(calls, fx.answer("escalate"))])
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "x.xml",
        seed=fx.SEED,
        runner=fx.RefusingEngine(tmp_path),
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=3, tool_calls_per_attempt=2),
            draw_figures=False,
        ),
        backend=fx.StoredParentQe(fx.parent()),
    )
    offered = model.requests[0]["tools"]
    assert {"truncation_diagnostics", "band_error_by_energy", "structure"} <= set(
        offered
    )
    called = result.run.trailing_proposal["tools_called"]
    assert [c["counted"] for c in called] == [True, True, False]
    structure = json.loads(model.results[0][1])
    assert structure["species_counts"] == {"Sr": 1, "O": 1}
    assert structure["nearest_neighbour_ang"] == pytest.approx(
        5.16 * np.sqrt(3) / 2, abs=1e-4
    )
    # the frontier never produced a construction (the engine refuses), and it says so
    assert "no construction has run far enough" in model.results[1][1]
    for name in ("truncation_diagnostics", "band_error_by_energy", "structure"):
        assert f"- {name}:" in GENERIC_PRINCIPLES
    assert "call it\n  before changing projectors" in GENERIC_PRINCIPLES
