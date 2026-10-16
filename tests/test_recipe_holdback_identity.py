"""The same-material hold-back must recognise the material, not its seed's spelling.

An entry distilled from an earlier run on this material is a memory of it, and the
reference stage holds such entries back so that what it measures is transfer between
materials and not recall of one. The hold-back compared the run's seed to the entry's
recorded seed by string prefix: ``Ti_pdwf`` held back ``Ti_pdwf-2``. A replicate
(``Ti_pdwf_rep2``), a run on another engine (``Ti_vasp``) or any other spelling of the
same material was shown that material's own earlier repairs. Until the retrieved
entries actually reached the model this was harmless; from the reference fix on, it
turns every repeat run into a recall measurement labelled as transfer.
(Named by the 2026-09-17 scientific audit of the repair loop, finding C2.)

The material's identity is its composition, which every entry already records in
``signature["species"]`` and every run derives from its parent. These tests drive the
retrieval directly and, once, the ``search_references`` tool with an on-disk corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from autowan import recipes
from autowan.qe_outputs import ParentCalculation
from autowan.recipes import Recipe, composition_of, search

METAL = {
    "manifold_is_isolated": False,
    "band_gap_ev": 0.0,
    "bands_crossing_fermi_level": 3,
    "atoms_in_cell": 2,
    "occupied_bandwidth_ev": 5.0,
}


def _entry(recipe_id: str, material: str, species: dict[str, int] | None) -> Recipe:
    signature = dict(METAL)
    if species is not None:
        signature["species"] = species
    return Recipe(
        recipe_id=recipe_id,
        signature=signature,
        resolved=("centres",),
        still_unresolved=(),
        changes=({"parameter": "dis_froz_max", "old": 1.0, "new": 2.0, "reason": "r"},),
        diagnosis="d",
        provenance={"material": material, "attempt": 1, "certified": True},
    )


def test_composition_is_the_reduced_species_count():
    assert composition_of({"Ti": 2}) == {"Ti": 1}
    assert composition_of({"Si": 8}) == {"Si": 1}
    assert composition_of({"O": 3, "Sr": 1, "Ti": 1}) == {"O": 3, "Sr": 1, "Ti": 1}
    assert composition_of({"O": 6, "Sr": 2, "Ti": 2}) == {"O": 3, "Sr": 1, "Ti": 1}
    assert composition_of({}) == {}


@pytest.mark.parametrize("seed", ["Ti_pdwf_rep2", "Ti_vasp", "titanium-hcp", "Ti_pdwf"])
def test_a_run_on_the_same_material_under_any_seed_is_held_back(seed):
    mine = _entry("Ti_pdwf-2", "Ti_pdwf", {"Ti": 2})
    theirs = _entry("Ba_pdwf-4", "Ba_pdwf", {"Ba": 1})
    found = search(
        [mine, theirs],
        signature=METAL,
        unresolved=["centres"],
        exclude_material=seed,
        exclude_composition={"Ti": 2},
    )
    assert [r.recipe_id for r, _ in found] == ["Ba_pdwf-4"]


def test_a_supercell_of_the_same_material_is_the_same_material():
    mine = _entry("Si_conv-1", "Si_conventional", {"Si": 8})
    theirs = _entry("C_prim-1", "C_diamond", {"C": 2})
    found = search(
        [mine, theirs],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="Si_vasp",
        exclude_composition={"Si": 2},
    )
    assert [r.recipe_id for r, _ in found] == ["C_prim-1"]


def test_a_different_material_sharing_an_element_is_not_held_back():
    sro = _entry("SrO_pdwf-1", "SrO_pdwf", {"Sr": 1, "O": 1})
    found = search(
        [sro],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="SrTiO3_pdwf",
        exclude_composition={"Sr": 1, "Ti": 1, "O": 3},
    )
    assert [r.recipe_id for r, _ in found] == ["SrO_pdwf-1"]


def test_an_entry_that_recorded_no_species_falls_back_to_the_seed_rule():
    # A corpus written before the species were part of the signature has only the
    # seed to go on. The old rule still applies to it, and only to it.
    legacy_mine = _entry("Ti_pdwf-2", "Ti_pdwf", None)
    legacy_theirs = _entry("Ba_pdwf-4", "Ba_pdwf", None)
    found = search(
        [legacy_mine, legacy_theirs],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="Ti_pdwf",
        exclude_composition={"Ti": 2},
    )
    assert [r.recipe_id for r, _ in found] == ["Ba_pdwf-4"]


def test_a_seed_prefix_shared_by_a_different_material_does_not_hold_it_back():
    # Silicon carbide seeded "Si" would, under the seed rule alone, be denied every
    # silicon entry. With both compositions recorded, the composition decides.
    silicon = _entry("Si_conv-1", "Si_conventional", {"Si": 8})
    found = search(
        [silicon],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="Si",
        exclude_composition={"Si": 2, "C": 2},
    )
    assert [r.recipe_id for r, _ in found] == ["Si_conv-1"]


def test_a_malformed_species_field_records_no_composition_and_does_not_raise():
    assert composition_of("Ti2") == {}
    assert composition_of(["Ti", "Ti"]) == {}
    assert composition_of({"Ti": "two", "O": None}) == {}
    assert composition_of({"Ti": "2", "O": 0, "Sr": -1}) == {"Ti": 1}
    broken = _entry("odd-1", "odd_seed", None)
    broken.signature["species"] = "not a mapping"
    theirs = _entry("Ba_pdwf-4", "Ba_pdwf", {"Ba": 1})
    found = search(
        [broken, theirs],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="Ti_vasp",
        exclude_composition={"Ti": 2},
    )
    assert sorted(r.recipe_id for r, _ in found) == ["Ba_pdwf-4", "odd-1"]


def _titanium_parent() -> ParentCalculation:
    eigenvalues = np.array([[[-6.0, -1.0, 2.0, 5.0], [-5.5, 1.0, 3.0, 6.0]]])
    return ParentCalculation(
        eigenvalues_ev=eigenvalues,
        occupations=np.zeros_like(eigenvalues),
        kpoints_cart=np.zeros((2, 3)),
        fermi_energy_ev=0.0,
        lattice_ang=np.eye(3) * 2.95,
        alat_ang=2.95,
        species=("Ti", "Ti"),
        positions_frac=np.zeros((2, 3)),
        n_symmetries=24,
        n_atomic_wfc=12,
        converged=True,
        calculation="scf",
    )


FAILING = "spread_minimisation_converged"


def test_the_tool_holds_back_this_materials_entries_whatever_the_seed_says(tmp_path):
    # The decisive case: a titanium run seeded ``Ti_vasp`` searches a corpus holding a
    # titanium entry recorded under ``Ti_pdwf`` and a barium entry. What reaches the
    # model is the barium entry alone.
    from types import SimpleNamespace

    from autowan.methods import DeckContext
    from autowan.orchestrator import Run
    from autowan.tools import RepairTools

    mine = Recipe(
        recipe_id="hcp_ti_a2_spread_converged",
        signature={**METAL, "species": {"Ti": 2}},
        resolved=(FAILING,),
        still_unresolved=(),
        changes=({"parameter": "num_iter", "old": 400, "new": 4000, "reason": "r"},),
        diagnosis="d",
        provenance={"material": "Ti_pdwf", "attempt": 2, "certified": True},
    )
    theirs = Recipe(
        recipe_id="bcc_ba_a3_spread_converged",
        signature={**METAL, "species": {"Ba": 1}},
        resolved=(FAILING,),
        still_unresolved=(),
        changes=({"parameter": "dis_froz_max", "old": 1.0, "new": 2.5, "reason": "r"},),
        diagnosis="d",
        provenance={"material": "Ba_pdwf", "attempt": 3, "certified": False},
    )
    corpus = tmp_path / "corpus"
    for recipe in (mine, theirs):
        recipes.write(recipe, corpus)
    run = Run("Ti_vasp")
    run.start_attempt({"num_iter": 400})
    run._unresolved = (FAILING,)
    parent = _titanium_parent()
    tools = RepairTools(
        ctx=DeckContext(
            code="vasp",
            run_dir=tmp_path,
            parent=parent,
            parent_source=tmp_path / "vasprun.xml",
            frontier=SimpleNamespace(),
        ),
        backend=None,
        run=run,
        inputs=SimpleNamespace(),
        seed="Ti_vasp",
        recipe_dir=corpus,
        recipe_same_material_holdback=True,
    )
    answer = tools("search_references", {"query": "spread converged"})
    assert run.searches[-1]["retrieved"] == ["bcc_ba_a3_spread_converged"]
    assert "bcc_ba_a3_spread_converged" in answer
    assert (
        "hcp_ti_a2_spread_converged" not in answer
    ), "the run's own material reached the model under another seed"
    spec = next(s for s in tools.specs() if s.name == "search_references")
    assert "held back" in spec.description
    # The package default (2026-10-13): no hold-back -- public use may draw on
    # an earlier repair of the same material, as frontier_dataset may.
    tools.recipe_same_material_holdback = False
    answer = tools("search_references", {"query": "spread converged"})
    assert "hcp_ti_a2_spread_converged" in answer
    spec = next(s for s in tools.specs() if s.name == "search_references")
    assert "included" in spec.description
