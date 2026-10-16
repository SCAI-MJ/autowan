"""The energy-window first deck as a recipe of the VASP deck writer.

The VASP default first deck is the hydrogenic all-channel deck disentangled by an
energy window alone: Wannier90's projectability window cannot run on VASP's raw,
non-orthonormal projections (p > 1 stops it before the gate;
2026-09-30). ``deck_from_parent`` offered no recipe that writes it, so
the 2026-09-29 baseline (13/20 on the fixed-mesh parents) removed the three window lines
from the hydrogenic deck after writing it. The recipe ``hydrogenic_energy_window`` writes
that same deck directly: the hydrogenic deck, line for line, without ``dis_proj_min``,
``dis_proj_max`` and ``dis_froz_proj``. Every other recipe is unchanged.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from autowan.backends.vasp.inputs import (
    DECK_RECIPES,
    ENERGY_WINDOW_RECIPE,
    RECIPE_OVERLAP_TAGS,
    RECIPES,
    deck_from_parent,
    overlap_step_incar_tags,
)
from autowan.inputs import read_win

FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
WINDOW_KEYS = ("dis_proj_min", "dis_proj_max", "dis_froz_proj")
RECIPE = "hydrogenic_energy_window"


def _parent(tmp_path: Path, material: str, bands: int | None = None) -> Path:
    d = tmp_path / f"{material}_{bands or 'own'}"
    d.mkdir()
    for f in ("vasprun.xml", "OUTCAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(FIXTURES / material / f, d / f)
    if bands:
        tree = ET.parse(d / "vasprun.xml")
        root = tree.getroot()
        for spin in root.find("calculation/eigenvalues/array/set").findall("set"):
            for kpoint in spin.findall("set"):
                rows = kpoint.findall("r")
                top = float(rows[-1].text.split()[0])
                for i in range(bands - len(rows)):
                    r = ET.SubElement(kpoint, "r")
                    r.text = f" {top + 1.0 + i:.4f} 0.0000 "
        for item in root.iter("i"):
            if item.get("name") == "NBANDS":
                item.text = f" {bands} "
        tree.write(d / "vasprun.xml", encoding="unicode")
    return d


def _deck(parent: Path, tmp_path: Path, recipe: str, name: str) -> str:
    run = tmp_path / name
    run.mkdir()
    deck_from_parent(parent / "vasprun.xml", run, "deck", recipe=recipe)
    return (run / "deck.win").read_text()


def _key(line: str) -> str:
    return line.split("!")[0].split("=")[0].strip().lower()


def test_the_energy_window_deck_is_a_recipe_of_the_writer():
    assert ENERGY_WINDOW_RECIPE == RECIPE
    assert RECIPE in DECK_RECIPES
    assert RECIPE_OVERLAP_TAGS[RECIPE] == {}
    # the 2026-09-29 benchmark's three recipes are unchanged
    assert RECIPES == ("hydrogenic", "locproj_auto", "scdm")


@pytest.mark.parametrize("material", ["Si", "SrO"])
def test_it_is_the_hydrogenic_deck_without_the_three_window_lines(tmp_path, material):
    parent = _parent(tmp_path, material)
    hydrogenic = _deck(parent, tmp_path, "hydrogenic", "h").splitlines()
    energy = _deck(parent, tmp_path, RECIPE, "e").splitlines()
    dropped = [ln for ln in hydrogenic if _key(ln) in WINDOW_KEYS]
    assert sorted(_key(ln) for ln in dropped) == sorted(WINDOW_KEYS)
    assert [ln for ln in hydrogenic if ln not in dropped] == energy


def test_it_keeps_the_first_deck_band_set_and_excludes_a_headroom_surplus(tmp_path):
    parent = _parent(tmp_path, "Si", bands=24)
    text = _deck(parent, tmp_path, RECIPE, "e")
    values = read_win(tmp_path / "e" / "deck.win")
    assert values["num_bands"] == "16"
    assert "exclude_bands = 17-24" in text.splitlines()
    assert not [ln for ln in text.splitlines() if _key(ln) in WINDOW_KEYS]
    assert "begin projections" in text


def test_it_asks_the_overlap_step_for_no_incar_tags(tmp_path):
    text = _deck(_parent(tmp_path, "Si"), tmp_path, RECIPE, "e")
    assert overlap_step_incar_tags(text) == {}


def test_the_hydrogenic_default_still_writes_its_projectability_window(tmp_path):
    text = _deck(_parent(tmp_path, "Si"), tmp_path, "hydrogenic", "h")
    assert sorted(_key(ln) for ln in text.splitlines() if _key(ln) in WINDOW_KEYS) == (
        sorted(WINDOW_KEYS)
    )
