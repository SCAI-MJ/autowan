"""Projectors placed on a point of the cell rather than on every atom of a species.

A covalent bond is described most compactly by one s-like function on the bond, and a
pi channel by one real component on each atom. Neither was expressible while the
repair vocabulary named projectors per species and per whole shell, although Wannier90
and the deck readers have always taken ``f=x,y,z:s`` and ``C:pz``. These tests pin the
whole path: the action accepts and normalises a positioned entry, counts it once, the
writer emits it as Wannier90 takes it, and the deck reader counts the written block the
same way the action did.
"""

from __future__ import annotations

import json

import pytest

from autowan.actions import (
    ActionRejected,
    Stage,
    action_schema,
    implied_wannier_count,
    validate_action,
)
from autowan.deckcheck import implied_by_block
from autowan.inputs import read_construction, write_derived

BASE = {
    "diagnosis": "the functions built from states above the judged region left the atoms",
    "evidence": ["centres_stay_near_home_cell"],
    "expected_effect": "bond-centred functions stay between their atoms",
    "confidence": 0.6,
}


def _validate(projectors, species=("C",)):
    return validate_action(
        {
            **BASE,
            "action": "modify_parameters",
            "changes": [
                {
                    "parameter": "projectors",
                    "old": None,
                    "new": projectors,
                    "reason": "seed the bonds and the out-of-plane orbitals",
                }
            ],
        },
        stage=Stage.REPAIR,
        evaluated_check_ids=["centres_stay_near_home_cell"],
        allowed_species=list(species),
    )


BONDS_AND_PZ = [
    {"site": "f=0.5, 0.0, 0.0", "orbitals": ["s"]},
    {"site": "f=0.0,0.5,0.0", "orbitals": ["s"]},
    {"site": "f=0.5,0.5,0.0", "orbitals": ["s"]},
    {"species": "C", "orbitals": ["pz"]},
]


def test_a_positioned_entry_is_accepted_and_normalised():
    action = _validate(BONDS_AND_PZ)
    new = action.changes[0].new
    assert {"species": "C", "orbitals": ("pz",)} in new
    sites = [e["site"] for e in new if "site" in e]
    assert sites == ["f=0.5,0,0", "f=0,0.5,0", "f=0.5,0.5,0"]


def test_a_positioned_entry_counts_once_and_a_component_counts_one():
    new = _validate(BONDS_AND_PZ).changes[0].new
    # three bond sites x s, plus pz on each of two atoms
    assert implied_wannier_count(new, {"C": 2}) == 5
    assert implied_wannier_count(new, {"C": 2}, spinors=True) == 10
    # a whole p shell is still three per atom
    assert implied_wannier_count([{"species": "C", "orbitals": ("p",)}], {"C": 2}) == 6


@pytest.mark.parametrize(
    "bad, match",
    [
        ([{"site": "x=0.5,0,0", "orbitals": ["s"]}], "f=x,y,z"),
        ([{"site": "f=0.5,0", "orbitals": ["s"]}], "f=x,y,z"),
        ([{"site": "f=a,0,0", "orbitals": ["s"]}], "f=x,y,z"),
        ([{"site": "f=0.5,0,0", "orbitals": []}], "non-empty"),
        ([{"site": "f=0.5,0,0", "orbitals": ["q"]}], "not an orbital label"),
        (
            [
                {"site": "f=0.5,0,0", "orbitals": ["s"]},
                {"site": "f=0.50, 0.0, 0", "orbitals": ["s"]},
            ],
            "twice",
        ),
        ([{"site": "f=0.5,0,0", "species": "C", "orbitals": ["s"]}], "exactly"),
        ([{"species": "Si", "orbitals": ["s"]}], "not in this structure"),
    ],
)
def test_malformed_positioned_entries_are_refused_with_a_reason(bad, match):
    with pytest.raises(ActionRejected, match=match):
        _validate(bad)


def test_the_schema_offers_both_entry_shapes_and_the_components():
    schema = json.dumps(
        action_schema(
            Stage.REPAIR, ["centres_stay_near_home_cell"], allowed_species=["C"]
        )
    )
    assert '"site"' in schema
    assert '"pz"' in schema and '"dxy"' in schema


def test_the_writer_emits_sites_and_the_reader_counts_them_the_same_way(tmp_path):
    (tmp_path / "deck.win").write_text(
        "num_wann = 8\nnum_bands = 32\ndis_froz_max = 1.0\n"
        "begin projections\nC:s;p\nend projections\n"
    )
    (tmp_path / "deck.pw2wan").write_text(
        "&inputpp\n  prefix = 'graphene'\n  seedname = 'deck'\n  write_amn = .true.\n/\n"
    )
    source = read_construction(tmp_path, "deck")
    new = _validate(BONDS_AND_PZ).changes[0].new
    write_derived(
        source, tmp_path, "deck_r1", {"projectors": list(new)}, species_counts={"C": 2}
    )
    win = (tmp_path / "deck_r1.win").read_text()
    block = win.split("begin projections", 1)[1].split("end projections", 1)[0]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert lines == ["C:pz", "f=0.5,0,0:s", "f=0,0.5,0:s", "f=0.5,0.5,0:s"]
    assert "num_wann = 5" in win, win
    assert implied_by_block(lines, {"C": 2}, False) == 5
