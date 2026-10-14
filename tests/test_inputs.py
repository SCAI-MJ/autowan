"""The gate's input reader accepts every spelling of ``mp_grid`` the writers emit.

Wannier90 reads ``mp_grid`` as three integers separated by whitespace and/or commas.
The aiida writer emits ``mp_grid = 6, 6, 6``; hand-written decks use ``mp_grid = 6 6 6``.
Both must reach the gate as the tuple ``(6, 6, 6)``; anything that is not three positive
integers is refused with an :class:`InputError` naming the offending value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autowan.inputs import ConstructionInputs, InputError, read_win


def _inputs(mp_grid: str, tmp_path: Path) -> ConstructionInputs:
    return ConstructionInputs(
        seed="s", win=tmp_path / "s.win", pw2wan=None, values={"mp_grid": mp_grid}
    )


@pytest.mark.parametrize(
    "text",
    ["6 6 6", "6, 6, 6", "6,6,6", "6,6 6", "6 6,6", " 6 ,  6 , 6 ", "6\t6,\t6"],
)
def test_mp_grid_spellings_parse_to_the_same_triple(text, tmp_path):
    assert _inputs(text, tmp_path).gate_inputs()["mp_grid"] == (6, 6, 6)


def test_mp_grid_keeps_distinct_axes(tmp_path):
    assert _inputs("4, 5, 6", tmp_path).gate_inputs()["mp_grid"] == (4, 5, 6)


@pytest.mark.parametrize(
    "text", ["6 6", "6, 6", "6 6 6 6", "6, 6, x", "0 6 6", "-1, 6, 6", "6.5 6 6", ""]
)
def test_mp_grid_that_is_not_three_positive_integers_is_refused(text, tmp_path):
    with pytest.raises(InputError, match="mp_grid"):
        _inputs(text, tmp_path).gate_inputs()


@pytest.mark.parametrize(
    "line", ["mp_grid = 6, 6, 6", "mp_grid = 6 6 6", "mp_grid 6,6,6"]
)
def test_win_file_line_reaches_the_gate_as_a_triple(line, tmp_path):
    win = tmp_path / "s.win"
    win.write_text(f"num_wann = 4\n{line}\n")
    values = read_win(win)
    inputs = ConstructionInputs(seed="s", win=win, pw2wan=None, values=values)
    assert inputs.gate_inputs()["mp_grid"] == (6, 6, 6)
