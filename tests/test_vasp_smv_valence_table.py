"""The VASP SMV deck's valence table covers the 24-material set and guesses nothing.

Semicore = our POTCAR's ZVAL minus a checked valence count, held by the deepest shells
of that POTCAR. An element with no row is refused (the generic shell rule that used to
stand in is gone). The rows for the extension set are checked against the POTCARs the
extension parents actually use, when they are on disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.inputs import (
    VALENCE_TABLE,
    NoValenceRow,
    _species_shells,
    semicore_excluded_deck,
    semicore_shells,
)
from autowan.backends.vasp.outputs import potcar_valences
from autowan.methods import StartingDeckUnavailable

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


SRO = Path(__file__).parent / "fixtures" / "vasp" / "SrO"
EXTENSION = _stored("AUTOWAN_TEST_EXTENSION")

#: The 24 materials of the poster campaign, by element.
TEST_SET_ELEMENTS = {
    "Ba", "C", "Cu", "O", "Ga", "As", "La", "Mn", "Na", "Sr", "Ti", "V", "Si", "W",
    "Fe", "Ni", "Mo", "S", "Bi", "Se", "Pt", "Co", "Sn",
}  # fmt: skip

#: (parent, element) -> (our POTCAR's ZVAL, the semicore shells it must give).
EXTENSION_ROWS = {
    ("Fe_fm", "Fe"): (16, ("3s", "3p")),
    ("NiO_afm2", "Ni"): (18, ("3s", "3p")),
    ("NiO_afm2", "O"): (6, ("2s",)),
    ("Co3Sn2S2", "Co"): (17, ("3s", "3p")),
    ("Co3Sn2S2", "Sn"): (14, ("4d",)),
    ("Co3Sn2S2", "S"): (6, ("3s",)),
    ("MoS2", "Mo"): (14, ("4s", "4p")),
    ("Bi2Se3", "Bi"): (15, ("5d",)),
    ("Bi2Se3", "Se"): (6, ()),
    ("Pt", "Pt"): (18, ("5s", "5p")),
    ("SrVO3", "V"): (13, ("3s", "3p")),
}


def test_the_table_covers_every_element_of_the_test_set():
    assert TEST_SET_ELEMENTS <= set(VALENCE_TABLE), TEST_SET_ELEMENTS - set(
        VALENCE_TABLE
    )


def test_an_element_without_a_row_is_refused_not_guessed(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    shutil.copytree(SRO, parent)
    monkeypatch.delitem(VALENCE_TABLE, "Sr")
    with pytest.raises(NoValenceRow, match="Sr: no row in the VASP SMV valence table"):
        semicore_excluded_deck(parent / "vasprun.xml", tmp_path, "SrO")
    assert not hasattr(vasp_inputs, "_generic_valence")


def test_the_missing_row_reaches_the_model_as_an_unavailable_starting_deck(
    tmp_path, monkeypatch
):
    from autowan.backends.vasp.outputs import read_parent
    from autowan.inputs import ConstructionInputs
    from autowan.methods import DeckContext, starting_deck

    parent = tmp_path / "parent"
    shutil.copytree(SRO, parent)
    monkeypatch.delitem(VALENCE_TABLE, "O")
    win = tmp_path / "SrO.win"
    win.write_text("num_wann = 1\n")
    ctx = DeckContext(
        code="vasp",
        run_dir=tmp_path,
        parent=read_parent(parent / "vasprun.xml"),
        parent_source=parent / "vasprun.xml",
        frontier=ConstructionInputs("SrO", win, None, {}),
    )
    with pytest.raises(StartingDeckUnavailable, match="O: no row"):
        starting_deck("smv", ctx)


def test_semicore_is_the_deepest_shells_even_without_an_outer_s_shell():
    # Ni_sv_GW's reference is 3d10 with no 4s channel: 3s must still be semicore.
    ni = {"3s": (0, 2.0, -102.7), "3p": (1, 6.0, -63.2), "3d": (2, 10.0, -1.97)}
    assert semicore_shells("Ni", 8, ni) == ("3s", "3p")
    # SrO's O 2s and Sr_sv's 4s 4p, as before
    sr = {"4s": (0, 2.0, -40.9), "5s": (0, 2.0, -3.5), "4p": (1, 6.0, -22.8)}
    assert semicore_shells("Sr", 8, sr) == ("4s", "4p")
    with pytest.raises(Exception, match="no set of whole shells"):
        semicore_shells("Sr", 4, sr)


@pytest.mark.parametrize("parent, element", sorted(EXTENSION_ROWS))
def test_the_extension_rows_against_the_extension_parents_potcars(parent, element):
    potcar = EXTENSION / parent / "scf" / "POTCAR"
    if not potcar.is_file():
        pytest.skip("the extension parents are not shipped")
    zval_expected, shells_expected = EXTENSION_ROWS[(parent, element)]
    zval = potcar_valences(potcar)[element]
    assert zval == zval_expected
    semicore_e = zval - VALENCE_TABLE[element][0]
    shells = _species_shells(potcar, None)[element]
    assert semicore_shells(element, semicore_e, shells) == shells_expected
