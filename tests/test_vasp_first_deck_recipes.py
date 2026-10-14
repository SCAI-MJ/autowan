"""The three automatic VASP first decks, and the pseudopotential readers they rest on.

The recipe benchmark scores three automatic first decks on the same parents, unrepaired:

- ``hydrogenic``: a hydrogenic trial function for every non-local POTCAR channel, with
  projectability disentanglement (the prototype ``deck_from_parent`` already wrote);
- ``locproj_auto``: VASP's own pseudopotential projections (``LOCPROJ_AUTO``: the first
  bound PS partial wave of each l), with the same projectability disentanglement;
- ``scdm``: VASP's SCDM (``LSCDM``), with the energy frozen window alone.

Two readers carry them. The channel inventory must not count a ``Description`` row that
is the local potential, whichever ``TYP`` it is marked with: ten of the twenty campaign
parents carry such a row with ``TYP = 23`` (Na, Cu, V_sv, Ti_sv, Ga_d), and a reader
that took it as a channel refused them all. And ``LOCPROJ_AUTO``'s selection has to be
known before VASP runs, because the deck states ``num_wann``: VASP takes, per l, the first
channel whose atomic occupancy is non-zero (``locproj.F`` ``LPRJ_FROM_PSEUDO``), which on
sodium is one s function where the all-channel deck has four.

The recipe tags the overlap step needs (``LOCPROJ_AUTO``, ``LSCDM``) are INCAR tags, not
Wannier90 keywords. The deck records them on a comment line so the construction carries
its own overlap-step recipe.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from autowan.backends.vasp.inputs import (
    RECIPES,
    deck_from_parent,
    overlap_step_incar_tags,
    overlap_step_win,
)
from autowan.backends.vasp.outputs import (
    VaspOutputError,
    locproj_auto_channels,
    potcar_channels,
)
from autowan.inputs import InputError, requests_projectability

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
SYNTHETIC = FIXTURES / "synthetic"


# ---------------------------------------------------------------------------
# synthetic pseudopotential text: the fields the readers read, nothing licensed
# ---------------------------------------------------------------------------


def _fortran(values) -> list[str]:
    """Five values a line, the way a POTCAR prints its (5E20.12) blocks."""
    cells = [f"{v:20.12E}" for v in values]
    return ["".join(cells[i : i + 5]) for i in range(0, len(cells), 5)]


def _potcar(symbol, rows, parts, occupancies=None, configuration=None) -> str:
    """One POTCAR dataset: header table, ``Non local Part`` blocks, atomic occupancies.

    rows: (l, reference energy, TYP) as the ``Description`` table lists them.
    parts: (l, number of projectors) per ``Non local Part`` block, in file order.
    occupancies: the diagonal of the channel occupancy matrix, one per projector.
    configuration: (n, l, eigenvalue, occupation) rows of the ``Atomic configuration``
    table, which the automatic deck's function count reads (one per occupied shell).
    """
    lines = [
        f"  PAW_PBE {symbol} 01Jan2000",
        "   1.00000000000000000",
        " parameters from PSCTR are:",
        f"   VRHFIN ={symbol}: synthetic",
        "   LEXCH  = PE",
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000",
    ]
    if configuration is not None:
        lines += ["   Atomic configuration", f"    {len(configuration)} entries"]
        lines += ["     n  l   j            E        occ."]
        lines += [
            f"     {n}  {l}  {l + 0.5 if l else 0.5:.2f}  {e:14.4f}   {occ:.4f}"
            for n, l, e, occ in configuration
        ]
    lines += [
        "   Description",
        "     l       E           TYP  RCUT    TYP  RCUT",
    ]
    lines += [f"     {l}    {e:.7f}     {typ}  2.200" for l, e, typ in rows]
    lines += ["   Error from kinetic energy argument (eV)", "   NDATA  =      100"]
    lines += ["END of PSCTR-controll parameters", " local part"]
    for l, nproj in parts:
        lines += [" Non local Part", f"           {l}           {nproj}   2.2000000000"]
        lines += ["  0.100000000000E+01  0.200000000000E+01"]
    if occupancies is not None:
        channels = sum(n for _, n in parts)
        assert len(occupancies) == channels
        matrix = [0.0] * (channels * channels)
        for i, value in enumerate(occupancies):
            matrix[i * channels + i] = value
        lines += [" PAW radial sets", "         371   1.12", "(5E20.12)"]
        lines += [" augmentation charges (non sperical)"]
        lines += _fortran([0.001] * (channels * channels))
        lines += [" uccopancies in atom"]
        lines += _fortran(matrix)
        lines += [" grid", "  0.1E-04"]
    lines += [" End of Dataset"]
    return "\n".join(lines) + "\n"


def _outcar_echo(symbol, rows, lmmax) -> str:
    """The same header as an OUTCAR echoes it: the table and VASP's LMMAX, no body."""
    lines = [
        f" POTCAR:    PAW_PBE {symbol} 01Jan2000",
        f"   VRHFIN ={symbol}: synthetic",
        "   Description",
        "     l       E           TYP  RCUT    TYP  RCUT",
    ]
    lines += [f"     {l}    {e:.7f}     {typ}  2.200" for l, e, typ in rows]
    lines += ["   Error from kinetic energy argument (eV)"]
    lines += ["   number of l-projection  operators is LMAX  =           3"]
    lines += [f"   number of lm-projection operators is LMMAX =    {lmmax:8d}"]
    return "\n".join(lines) + "\n"


# Row layouts measured on the 2026-09-29 campaign POTCARs (report 07, Task 5): the
# local potential is the table's LAST row, and here it is marked TYP 23, not 7.
NA_ROWS = [
    (0, -2.7439560, 23),
    (0, 27.2116520, 23),
    (1, -1.3605826, 23),
    (2, -1.36, 23),
]
NA_PARTS = [(0, 2), (1, 1)]
NA_OCCUPANCIES = [1.0, 0.0, 0.0]
CU_ROWS = [
    (2, -5.39, 23),
    (2, -1.36, 23),
    (0, -4.86, 23),
    (0, 1.08, 23),
    (1, -0.83, 23),
    (1, 1.36, 23),
    (3, -1.36, 23),
]
CU_PARTS = [(2, 2), (0, 2), (1, 2)]
CU_OCCUPANCIES = [2.0, 0.0, 1.0, 0.0, 0.0, 0.0]
# Sr_sv: semicore 4s and valence 5s both occupied, d occupied only 2e-4 -- still bound.
SR_ROWS = [(0, -40.0, 23), (0, -3.0, 23), (1, -22.0, 23), (1, 1.0, 23)]
SR_ROWS += [(2, -1.0, 23), (2, 0.2, 23)]
SR_PARTS = [(0, 2), (1, 2), (2, 2)]
SR_OCCUPANCIES = [2.0, 1.999, 2.0, 0.0, 2.0e-4, 0.0]
# W_sv_GW: three s, two p, two d and one f projector; f carries no atomic occupancy,
# and every row, the f one included, is a projector (no local row at all).
W_ROWS = [(0, -76.0, 23), (0, -5.0, 23), (0, 10.0, 23), (1, -38.0, 23), (1, 5.0, 23)]
W_ROWS += [(2, -3.0, 23), (2, 1.0, 23), (3, 1.0, 23)]
W_PARTS = [(0, 3), (1, 2), (2, 2), (3, 1)]
W_OCCUPANCIES = [2.0, 0.0, 0.0, 2.0, 0.0, 1.2, 0.0, 0.0]
# Si: the local potential is a TYP 7 d row, as the reader always handled.
SI_ROWS = [(0, -10.81, 23), (0, -0.3, 23), (1, -4.08, 23), (1, 0.7, 23), (2, 0.0, 7)]
SI_PARTS = [(0, 2), (1, 2)]
SI_OCCUPANCIES = [2.0, 0.0, 0.666666, 0.0]
# Si's occupied shells: 3s and 3p, one per channel -- the s + p deck, eight functions.
SI_CONFIGURATION = [(1, 0, -1785.88, 2.0), (3, 0, -10.81, 2.0), (3, 1, -4.08, 2.0)]


# ---------------------------------------------------------------------------
# potcar_channels: a local-potential row marked TYP 23 is not a channel
# ---------------------------------------------------------------------------


def test_a_local_row_marked_typ23_is_not_a_channel_in_the_potcar(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar("Na", NA_ROWS, NA_PARTS) + _potcar("Cu", CU_ROWS, CU_PARTS)
    )
    assert potcar_channels(potcar) == {"Na": ("s", "p"), "Cu": ("s", "p", "d")}


def test_the_outcar_echo_drops_the_same_row_because_that_reproduces_lmmax(tmp_path):
    """An OUTCAR has no ``Non local Part`` blocks: VASP's LMMAX is what identifies it.

    Na: s + s + p = 5 = LMMAX, while the four rows would give 10.
    Cu: 5 + 5 + 1 + 1 + 3 + 3 = 18 = LMMAX, while the seven rows would give 25.
    """
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_outcar_echo("Na", NA_ROWS, 5) + _outcar_echo("Cu", CU_ROWS, 18))
    assert potcar_channels(outcar) == {"Na": ("s", "p"), "Cu": ("s", "p", "d")}


def test_a_last_row_that_does_not_reconcile_lmmax_still_refuses(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_outcar_echo("Na", NA_ROWS, 7))
    with pytest.raises(VaspOutputError, match="Na Description disagrees with LMMAX"):
        potcar_channels(outcar)


def test_a_last_row_that_is_a_projector_is_kept(tmp_path):
    """W_sv_GW's f row is a real projector: LMMAX 26 counts it, and so must the deck."""
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_potcar("W", W_ROWS, W_PARTS))
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_outcar_echo("W", W_ROWS, 26))
    assert potcar_channels(potcar) == {"W": ("s", "p", "d", "f")}
    assert potcar_channels(outcar) == {"W": ("s", "p", "d", "f")}


def test_a_potcar_whose_trailing_row_is_not_the_local_one_still_refuses(tmp_path):
    """Dropping the last row is not a licence: it must make the table agree."""
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_potcar("Na", NA_ROWS, [(0, 2)]))
    with pytest.raises(VaspOutputError, match="Non local"):
        potcar_channels(potcar)


CAMPAIGN_RUNS = _stored("AUTOWAN_TEST_CAMPAIGN_RUNS")
CAMPAIGN = [
    "Ba", "C", "Cu", "Cu2O", "GaAs", "La", "Mn", "MnO", "Na", "Si",
    "SiC", "Sr", "SrO", "SrTiO3", "Ti", "TiC", "TiO2", "V", "V3Si", "W",
]  # fmt: skip


@pytest.mark.skipif(
    not CAMPAIGN_RUNS.is_dir(), reason="the campaign parents are machine-local scratch"
)
@pytest.mark.parametrize("material", CAMPAIGN)
def test_every_campaign_parent_reads_the_same_channels_from_potcar_and_outcar(
    material,
):
    parent = CAMPAIGN_RUNS / material / "nscf"
    from_potcar = potcar_channels(parent / "POTCAR")
    assert potcar_channels(parent / "OUTCAR") == from_potcar


# ---------------------------------------------------------------------------
# locproj_auto_channels: VASP's own selection, known before VASP runs
# ---------------------------------------------------------------------------


def test_locproj_auto_takes_the_first_bound_partial_wave_of_each_l(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar("Na", NA_ROWS, NA_PARTS, NA_OCCUPANCIES)
        + _potcar("Cu", CU_ROWS, CU_PARTS, CU_OCCUPANCIES)
        + _potcar("Sr", SR_ROWS, SR_PARTS, SR_OCCUPANCIES)
        + _potcar("W", W_ROWS, W_PARTS, W_OCCUPANCIES)
    )
    assert locproj_auto_channels(potcar) == {
        "Na": ("s",),
        "Cu": ("d", "s"),
        "Sr": ("s", "p", "d"),
        "W": ("s", "p", "d"),
    }


def test_locproj_auto_needs_the_occupancy_table(tmp_path):
    """An OUTCAR echo, or a POTCAR without its body, cannot say what VASP will pick."""
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_potcar("Na", NA_ROWS, NA_PARTS))
    with pytest.raises(VaspOutputError, match="occupanc"):
        locproj_auto_channels(potcar)
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_outcar_echo("Na", NA_ROWS, 5))
    with pytest.raises(VaspOutputError, match="occupanc"):
        locproj_auto_channels(outcar)


# ---------------------------------------------------------------------------
# the three decks
# ---------------------------------------------------------------------------


@pytest.fixture
def parent(tmp_path):
    """The synthetic Si parent (4 bands, 2x1x1 mesh, highest occupied state 5.0 eV)."""
    parent = tmp_path / "parent"
    parent.mkdir()
    shutil.copy(SYNTHETIC / "vasprun.xml", parent / "vasprun.xml")
    shutil.copy(SYNTHETIC / "OUTCAR.converged", parent / "OUTCAR")
    (parent / "POTCAR").write_text(
        _potcar("Si", SI_ROWS, SI_PARTS, SI_OCCUPANCIES, SI_CONFIGURATION)
    )
    return parent


def _assignments(text: str) -> dict[str, str]:
    """Every ``key = value`` line of a .win outside its blocks, comments dropped."""
    values, block = {}, False
    for line in text.splitlines():
        cleaned = line.split("!", 1)[0].strip()
        lowered = cleaned.lower()
        if lowered.startswith("begin"):
            block = True
        elif lowered.startswith("end"):
            block = False
        elif not block and "=" in cleaned:
            key, _, value = cleaned.partition("=")
            values[key.strip().lower()] = value.strip()
    return values


def _deck(parent, tmp_path, recipe, **kwargs):
    run_dir = tmp_path / recipe
    run_dir.mkdir()
    inputs = deck_from_parent(
        parent / "vasprun.xml", run_dir, "si", recipe=recipe, **kwargs
    )
    return inputs, (run_dir / "si.win").read_text()


def test_the_benchmark_knows_exactly_three_recipes():
    assert RECIPES == ("hydrogenic", "locproj_auto", "scdm")


def test_the_hydrogenic_deck_is_the_prototype_and_asks_nothing_of_the_overlap_step(
    parent, tmp_path
):
    inputs, text = _deck(parent, tmp_path, "hydrogenic")
    default = tmp_path / "default"
    default.mkdir()
    deck_from_parent(parent / "vasprun.xml", default, "si", recipe="hydrogenic")
    assert (default / "si.win").read_text() == text
    assert inputs.values["projectors"] == "Si:s;p"
    assert overlap_step_incar_tags(text) == {}


def test_the_locproj_auto_deck_names_vasps_own_selection_and_asks_for_it(
    parent, tmp_path
):
    inputs, text = _deck(parent, tmp_path, "locproj_auto")
    assert inputs.values["projectors"] == "Si:s;p"
    assert inputs.values["num_wann"] == "8"
    assert inputs.values["dis_proj_min"] == "0.01"
    assert inputs.values["dis_proj_max"] == "0.95"
    assert _assignments(text)["dis_froz_proj"] == ".true."
    assert overlap_step_incar_tags(text) == {"LOCPROJ_AUTO": ".TRUE."}
    assert "LOCPROJ_AUTO" in text.split("begin projections")[0]
    assert "dis_froz_proj" not in overlap_step_win(text).lower()


def test_locproj_auto_is_held_to_the_same_window_rule_as_the_other_recipes(
    parent, tmp_path
):
    """With no p occupancy VASP selects s alone: two functions for two atoms.

    That selection is VASP's automatic choice, not an override, so it meets the rule
    every automatic deck meets: the frozen window must hold the occupied states. Two
    functions cannot hold the synthetic parent's four occupied bands, and the deck is
    refused at writing, exactly as an all-channel deck of that size would be. (Held
    to an override's capped window instead, it would run where the other recipes are
    refused: on the campaign that differs only for MnO.)
    """
    (parent / "POTCAR").write_text(
        _potcar("Si", SI_ROWS, SI_PARTS, [2.0, 0.0, 0.0, 0.0])
    )
    # a third occupied band (5.5 and 6.5 eV) that two functions cannot hold
    xml = (parent / "vasprun.xml").read_text()
    for energy in ("5.5000", "6.5000"):
        xml = xml.replace(
            f"<r>    {energy}    0.0000 </r>", f"<r>    {energy}    1.0000 </r>"
        )
    xml = xml.replace(
        '<i name="efermi">      5.20000000 </i>',
        '<i name="efermi">      6.80000000 </i>',
    )
    (parent / "vasprun.xml").write_text(xml)
    with pytest.raises(InputError, match="2 Wannier functions cannot hold"):
        _deck(parent, tmp_path, "locproj_auto")
    # an override of the same size is capped instead, which is the asymmetry avoided
    override = tmp_path / "override"
    override.mkdir()
    inputs = deck_from_parent(
        parent / "vasprun.xml",
        override,
        "si",
        projectors=[{"species": "Si", "orbitals": ("s",)}],
        recipe="hydrogenic",
    )
    assert inputs.values["num_wann"] == "2"


def test_the_scdm_deck_has_no_projectors_and_no_projectability_window(parent, tmp_path):
    inputs, text = _deck(parent, tmp_path, "scdm")
    values = _assignments(text)
    assert values["num_wann"] == "8"  # the all-channel count: Si s + p, two atoms
    assert "begin projections" not in text.lower()
    for key in ("dis_proj_min", "dis_proj_max", "dis_froz_proj"):
        assert key not in values
    assert not requests_projectability(text)
    # the same energy window as the hydrogenic deck: highest occupied 5.0 eV + 2 eV
    assert float(values["dis_froz_max"]) == pytest.approx(7.0)
    assert overlap_step_incar_tags(text) == {"LSCDM": ".TRUE."}


def test_an_unknown_recipe_or_a_hand_picked_locproj_set_is_refused(parent, tmp_path):
    with pytest.raises(InputError, match="recipe"):
        _deck(parent, tmp_path, "pdwf")
    with pytest.raises(InputError, match="LOCPROJ_AUTO"):
        _deck(
            parent,
            tmp_path,
            "locproj_auto",
            projectors=[{"species": "Si", "orbitals": ("s",)}],
        )


def test_overlap_step_tags_are_read_from_the_comment_the_deck_writes():
    text = (
        "num_wann = 4\n"
        "! overlap-step INCAR: lscdm = .TRUE.\n"
        "! an unrelated comment: X = 1\n"
        "!overlap-step INCAR:  CUTOFF_TYPE = erfc\n"
    )
    assert overlap_step_incar_tags(text) == {"LSCDM": ".TRUE.", "CUTOFF_TYPE": "erfc"}


# ---------------------------------------------------------------------------
# the knowledge entry: what the LSCDM run showed (SLURM 18683, amd1, 2026-09-29)
# ---------------------------------------------------------------------------


def test_the_vasp_knowledge_repairs_the_scdm_cutoff_through_overlap_step_tags():
    """``vasp.6.4.1-w90-avx2`` ran ``LSCDM = .TRUE.``: "SCDM mode", an erfc cutoff,
    and an AMN Wannier90 localised. The cutoff is an overlap-step INCAR setting, which
    a repair now writes on the deck's ``! overlap-step INCAR:`` lines (2026-10-07)."""
    from autowan.backends.vasp.knowledge import VASP_KNOWLEDGE

    for name in ("scdm_mu", "scdm_sigma"):
        assert VASP_KNOWLEDGE.supports(name)
        assert "INCAR" in VASP_KNOWLEDGE.deck_grammar[name].spelling
    assert set(VASP_KNOWLEDGE.unsupported_parameters) == {"pao_subset"}


def test_vasps_scdm_refusal_is_explained():
    from autowan.backends.vasp.knowledge import VASP_KNOWLEDGE

    line = (
        " CALC_WANNIER_TRAFO_SCDM: Number of Wannier functions must not be larger "
        "than number of bands."
    )
    assert VASP_KNOWLEDGE.explain(line)
