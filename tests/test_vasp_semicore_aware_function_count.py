"""One Wannier function per occupied radial channel: the VASP deck writer's function count.

2026-09-30 autowan discussion, item 1, decision 4 (a standing convention since): the
automatic VASP deck holds one Wannier function per (species, l, occupied radial
channel). A channel whose pseudopotential was built at two occupied atomic shells --
Mn_sv's 3s semicore and 4s valence -- gets one function per shell. A channel with one
occupied shell, or none (Cu p, W_sv_GW f), keeps the single function the all-channel
deck gave it, so no channel the POTCAR carries is dropped. On Mn this is 10 functions
per spin (3s, 4s, 3p x3, 3d x5) where the all-channel deck had 9, and on MnO 14 where it
had 13: the 13-function MnO deck could not hold the 14 states the nominal window
freezes, under any window.

An occupied radial channel is read off the pseudopotential header (POTCAR, or the
OUTCAR's echo of it): a non-local partial wave whose reference energy is the
eigenvalue of an occupied shell with the same l in the header's ``Atomic
configuration`` table. The second shell of a channel is written as Wannier90's
``r=2`` hydrogenic radial function, which VASP's overlap step honours (radial
quantum numbers 1-3).

The plan predicted "Mn 10, MnO 14, the others unchanged". The POTCARs say otherwise:
Ti_sv, V_sv, Sr_sv, Ba_sv and La also carry an occupied semicore s shell beside the
valence s, so ten of the other eighteen materials gain functions too (plan-guardian
PROCEED, 2026-10-01). The table below is the POTCARs' answer, measured by an
independent reader before this implementation; the counts are pinned to it.

Synthetic headers carry only the fields the readers read, with rounded energies:
no licensed POTCAR text enters git.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from autowan.backends.vasp.inputs import (
    MAX_RADIAL_FUNCTIONS,
    deck_from_parent,
    functions_per_atom,
    occupied_radial_channels,
    projection_lines,
    semicore_aware_projectors,
)
from autowan.inputs import InputError, wannier_count

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
TABLE = json.loads(
    (FIXTURES / "campaign_potcar_occupied_radial_channels.json").read_text()
)
POTCAR_LIBRARY = _stored("AUTOWAN_POTCAR_LIBRARY")
CAMPAIGN_RUNS = _stored("AUTOWAN_TEST_CAMPAIGN_RUNS")


# ---------------------------------------------------------------------------
# synthetic headers: the fields the reader reads, nothing licensed
# ---------------------------------------------------------------------------


def _config_lines(configuration) -> list[str]:
    lines = ["   Atomic configuration", f"    {len(configuration)} entries"]
    lines += ["     n  l   j            E        occ."]
    for n, l, energy, occ in configuration:
        j = l + 0.5 if l else 0.5
        lines.append(f"     {n}  {l}  {j:.2f}  {energy:14.4f}   {occ:.4f}")
    return lines


def _description_lines(rows) -> list[str]:
    lines = ["   Description", "     l       E           TYP  RCUT    TYP  RCUT"]
    lines += [f"     {l}  {e:14.7f}     {typ}  2.000" for l, e, typ in rows]
    return lines


def _potcar(symbol, configuration, rows, parts) -> str:
    """One POTCAR dataset: Atomic configuration, Description, Non local Part blocks."""
    lines = [
        f"  PAW_PBE {symbol} 01Jan2000",
        "   1.00000000000000000",
        " parameters from PSCTR are:",
        f"   VRHFIN ={symbol}: synthetic",
        "   LEXCH  = PE",
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000",
    ]
    if configuration is not None:
        lines += _config_lines(configuration)
    lines += _description_lines(rows)
    lines += ["   Error from kinetic energy argument (eV)", "   NDATA  =      100"]
    lines += ["END of PSCTR-controll parameters", " local part"]
    for l, nproj in parts:
        lines += [" Non local Part", f"           {l}           {nproj}   2.0000000000"]
        lines += ["  0.100000000000E+01  0.200000000000E+01"]
    lines += [" End of Dataset"]
    return "\n".join(lines) + "\n"


def _outcar_echo(symbol, configuration, rows, lmmax) -> str:
    """The same header as an OUTCAR echoes it: no body, VASP's LMMAX instead."""
    lines = [
        f" POTCAR:    PAW_PBE {symbol} 01Jan2000",
        f"   VRHFIN ={symbol}: synthetic",
    ]
    lines += _config_lines(configuration)
    lines += _description_lines(rows)
    lines += ["   Error from kinetic energy argument (eV)"]
    lines += [f"   number of lm-projection operators is LMMAX =    {lmmax:8d}"]
    return "\n".join(lines) + "\n"


# Mn_sv's layout (rounded energies): 3s and 4s both occupied and both a partial wave,
# a third, unbound s; 3p and an unbound p; 3d and a second d reference energy.
MN_CONFIG = [
    (1, 0, -6424.83, 2.0),
    (2, 0, -739.12, 2.0),
    (2, 1, -625.39, 6.0),
    (3, 0, -81.61, 2.0),
    (4, 0, -4.133, 1.0),
    (3, 1, -50.467, 6.0),
    (3, 2, -3.3745, 6.0),
    (4, 3, -1.3606, 0.0),
]
MN_ROWS = [
    (0, -81.61, 23),
    (0, -4.133, 23),
    (0, 108.85, 23),
    (1, -50.467, 23),
    (1, 20.41, 23),
    (2, -3.3745, 23),
    (2, -4.735, 23),
]
MN_PARTS = [(0, 3), (1, 2), (2, 2)]
MN_LMMAX = 3 * 1 + 2 * 3 + 2 * 5
# O: the d row is the local potential (TYP 7); the second s reference energy equals
# the unoccupied 3d level -- a different l, so it is not an s shell.
O_CONFIG = [(1, 0, -514.69, 2.0), (2, 0, -23.96, 2.0), (2, 1, -9.03, 4.0)]
O_CONFIG += [(3, 2, -9.524, 0.0)]
O_ROWS = [(0, -23.96, 23), (0, -9.524, 23), (1, -9.03, 23), (1, 8.16, 23)]
O_ROWS += [(2, -9.524, 7)]
O_PARTS = [(0, 2), (1, 2)]
# Cu: an unoccupied p channel (its reference energy is no atomic level) and the local
# f row marked TYP 23 -- the case potcar_channels already reconciles.
CU_CONFIG = [(3, 0, -114.69, 2.0), (3, 1, -72.13, 6.0), (3, 2, -5.0394, 10.0)]
CU_CONFIG += [(4, 0, -4.6097, 1.0), (4, 1, -4.0817, 0.0), (4, 3, -1.3606, 0.0)]
CU_ROWS = [(2, -5.0394, 23), (2, 10.88, 23), (0, -4.6097, 23), (0, 8.25, 23)]
CU_ROWS += [(1, -2.72, 23), (1, 21.71, 23), (3, 2.72, 23)]
CU_PARTS = [(2, 2), (0, 2), (1, 2)]


def _write(tmp_path, *datasets, name="POTCAR") -> Path:
    path = tmp_path / name
    path.write_text("".join(_potcar(*d) for d in datasets))
    return path


MN = ("Mn", MN_CONFIG, MN_ROWS, MN_PARTS)
OX = ("O", O_CONFIG, O_ROWS, O_PARTS)
CU = ("Cu", CU_CONFIG, CU_ROWS, CU_PARTS)


# ---------------------------------------------------------------------------
# TDD item 1: Mn_sv -> 10 functions per spin; MnO -> 14
# ---------------------------------------------------------------------------


def test_mn_sv_carries_two_occupied_s_shells_and_one_each_of_p_and_d(tmp_path):
    potcar = _write(tmp_path, MN)
    assert occupied_radial_channels(potcar) == {
        "Mn": {"s": ("3s", "4s"), "p": ("3p",), "d": ("3d",)}
    }


def test_mn_sv_takes_ten_functions_per_spin(tmp_path):
    channels = occupied_radial_channels(_write(tmp_path, MN))
    assert functions_per_atom(channels["Mn"]) == 10  # 3s, 4s, 3p x3, 3d x5
    projectors = semicore_aware_projectors(channels, ("Mn",))
    assert wannier_count(projectors, {"Mn": 1}) == 10
    assert projection_lines(projectors) == ["Mn:s;p;d", "Mn:s:r=2"]


def test_mno_takes_fourteen_functions_per_spin(tmp_path):
    channels = occupied_radial_channels(_write(tmp_path, MN, OX))
    assert channels["O"] == {"s": ("2s",), "p": ("2p",)}
    projectors = semicore_aware_projectors(channels, ("Mn", "O"))
    assert wannier_count(projectors, {"Mn": 1, "O": 1}) == 14
    assert projection_lines(projectors) == ["Mn:s;p;d", "Mn:s:r=2", "O:s;p"]


def test_the_outcar_echo_gives_the_count_the_potcar_gives(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        _outcar_echo("Mn", MN_CONFIG, MN_ROWS, MN_LMMAX)
        + _outcar_echo("O", O_CONFIG, O_ROWS, 8)
    )
    assert occupied_radial_channels(outcar) == occupied_radial_channels(
        _write(tmp_path, MN, OX)
    )


def test_a_channel_with_no_occupied_shell_keeps_its_one_function(tmp_path):
    """Cu p is unoccupied in the atom; the all-channel deck named it, and so does this."""
    channels = occupied_radial_channels(_write(tmp_path, CU))
    assert channels["Cu"] == {"s": ("4s",), "p": (), "d": ("3d",)}
    assert functions_per_atom(channels["Cu"]) == 9
    projectors = semicore_aware_projectors(channels, ("Cu",))
    assert projection_lines(projectors) == ["Cu:s;p;d"]


def test_a_reference_energy_equal_to_a_level_of_another_l_is_not_a_shell(tmp_path):
    """O's second s partial wave sits at the unoccupied 3d level: neither s nor occupied."""
    channels = occupied_radial_channels(_write(tmp_path, OX))
    assert channels["O"]["s"] == ("2s",)


def test_an_occupied_level_with_no_partial_wave_is_not_a_channel(tmp_path):
    """Core shells sit in the configuration table but no partial wave was built there."""
    channels = occupied_radial_channels(_write(tmp_path, MN))
    assert "1s" not in channels["Mn"]["s"] and "2p" not in channels["Mn"]["p"]


def test_a_header_without_its_atomic_configuration_is_refused_by_name(tmp_path):
    potcar = _write(tmp_path, ("Mn", None, MN_ROWS, MN_PARTS))
    with pytest.raises(InputError, match="Atomic configuration"):
        occupied_radial_channels(potcar)


def test_more_shells_than_vasp_has_radial_functions_for_is_refused(tmp_path):
    config = [(n, 0, -100.0 / n, 2.0) for n in range(1, 5)]
    rows = [(0, -100.0 / n, 23) for n in range(1, 5)]
    channels = occupied_radial_channels(_write(tmp_path, ("X", config, rows, [(0, 4)])))
    assert len(channels["X"]["s"]) == 4 > MAX_RADIAL_FUNCTIONS
    with pytest.raises(InputError, match="r="):
        semicore_aware_projectors(channels, ("X",))


# ---------------------------------------------------------------------------
# the deck writer: the automatic hydrogenic and SCDM decks take the new count
# ---------------------------------------------------------------------------


def _sro_parent(tmp_path) -> Path:
    for name in ("vasprun.xml", "OUTCAR", "EIGENVAL"):
        shutil.copy(FIXTURES / "SrO" / name, tmp_path / name)
    return tmp_path / "vasprun.xml"


def _projection_block(win: Path) -> list[str]:
    lines = [line.strip() for line in win.read_text().splitlines()]
    start = lines.index("begin projections")
    return [l for l in lines[start + 1 : lines.index("end projections")] if l]


def test_the_sro_deck_gives_sr_sv_its_semicore_s_function(tmp_path):
    """Sr_sv's 4s and 5s are both occupied partial waves (the fixture's OUTCAR echo)."""
    inputs = deck_from_parent(
        _sro_parent(tmp_path), tmp_path, "SrO", recipe="hydrogenic"
    )
    assert _projection_block(tmp_path / "SrO.win") == ["Sr:s;p;d", "Sr:s:r=2", "O:s;p"]
    assert int(inputs.values["num_wann"]) == 14


def test_the_scdm_deck_counts_the_same_functions(tmp_path):
    inputs = deck_from_parent(_sro_parent(tmp_path), tmp_path, "SrO", recipe="scdm")
    assert int(inputs.values["num_wann"]) == 14


def test_named_projectors_are_still_written_as_named(tmp_path):
    projectors = [
        {"species": "Sr", "orbitals": ("s", "p")},
        {"species": "O", "orbitals": ("p",)},
    ]
    inputs = deck_from_parent(
        _sro_parent(tmp_path),
        tmp_path,
        "SrO",
        projectors=projectors,
        recipe="hydrogenic",
    )
    assert _projection_block(tmp_path / "SrO.win") == ["Sr:s;p", "O:p"]
    assert int(inputs.values["num_wann"]) == 7


# ---------------------------------------------------------------------------
# TDD item 2: the twenty campaign parents, from their POTCARs
# ---------------------------------------------------------------------------

UNCHANGED = {"Na", "Cu", "Cu2O", "GaAs", "W", "Si", "C", "SiC"}
GAINED = {  # functions per spin gained over the 2026-09-29 all-channel count
    "Mn": 1,
    "MnO": 1,
    "Ba": 1,
    "Sr": 1,
    "SrO": 1,
    "SrTiO3": 2,
    "Ti": 2,
    "TiC": 1,
    "TiO2": 2,
    "V": 1,
    "V3Si": 6,
    "La": 1,
}


def _table_count(material: str) -> int:
    entry = TABLE["materials"][material]
    return sum(
        s["atoms"]
        * functions_per_atom(
            {
                letter: tuple(shells)
                for letter, shells in TABLE["potcars"][s["potcar"]][
                    "occupied_radial_channels"
                ].items()
            }
        )
        for s in entry["species"]
    )


def test_the_table_covers_the_twenty_campaign_materials():
    assert len(TABLE["materials"]) == 20
    assert set(TABLE["materials"]) == UNCHANGED | set(GAINED)


@pytest.mark.parametrize("material", sorted(TABLE["materials"]))
def test_each_campaign_count_follows_from_its_potcars(material):
    entry = TABLE["materials"][material]
    assert _table_count(material) == entry["num_wann_semicore_aware"]
    assert entry["num_wann_semicore_aware"] == entry[
        "num_wann_all_channel"
    ] + GAINED.get(material, 0)


def test_mn_and_mno_take_ten_and_fourteen_on_the_campaign_potcars():
    assert _table_count("Mn") == 10
    assert _table_count("MnO") == 14


def test_eight_materials_keep_their_count_and_ten_others_gain_semicore_s_functions():
    """The plan's "others unchanged" holds for eight; the POTCARs name the other ten."""
    unchanged = {
        m
        for m, e in TABLE["materials"].items()
        if _table_count(m) == e["num_wann_all_channel"]
    }
    assert unchanged == UNCHANGED
    assert set(TABLE["materials"]) - unchanged - {"Mn", "MnO"} == {
        "Ba", "Sr", "SrO", "SrTiO3", "Ti", "TiC", "TiO2", "V", "V3Si", "La",
    }  # fmt: skip


@pytest.mark.skipif(
    not POTCAR_LIBRARY.is_dir(), reason="the licensed POTCAR library is cluster-local"
)
@pytest.mark.parametrize("label", sorted(TABLE["potcars"]))
def test_every_campaign_potcar_reads_the_tabled_shells(label):
    channels = occupied_radial_channels(POTCAR_LIBRARY / label / "POTCAR")
    (symbol,) = channels
    expected = TABLE["potcars"][label]["occupied_radial_channels"]
    assert {k: list(v) for k, v in channels[symbol].items()} == expected


CAMPAIGN = sorted(TABLE["materials"])


@pytest.mark.skipif(
    not CAMPAIGN_RUNS.is_dir(), reason="the campaign parents are machine-local scratch"
)
@pytest.mark.parametrize("material", CAMPAIGN)
def test_every_campaign_parent_reads_the_same_shells_from_potcar_and_outcar(material):
    parent = CAMPAIGN_RUNS / material / "nscf"
    assert occupied_radial_channels(parent / "OUTCAR") == occupied_radial_channels(
        parent / "POTCAR"
    )


@pytest.mark.skipif(
    not CAMPAIGN_RUNS.is_dir(), reason="the campaign parents are machine-local scratch"
)
@pytest.mark.parametrize("material, count", [("Mn", 10), ("MnO", 14)])
def test_the_mn_decks_now_hold_the_nominal_window_on_their_six_cubed_parents(
    tmp_path, material, count
):
    """On the 2026-09-29 parents the 9- and 13-function decks overfilled by one band:
    Mn's window was capped below the nominal top and MnO's was refused outright. With
    one function per occupied radial channel the nominal window is placed uncapped."""
    parent = CAMPAIGN_RUNS / material / "nscf"
    for name in ("vasprun.xml", "OUTCAR", "POTCAR"):
        shutil.copy(parent / name, tmp_path / name)
    inputs = deck_from_parent(
        tmp_path / "vasprun.xml", tmp_path, material, recipe="hydrogenic"
    )
    assert int(inputs.values["num_wann"]) == count
    text = (tmp_path / f"{material}.win").read_text()
    assert "not binding" in text
    assert _projection_block(tmp_path / f"{material}.win")[:2] == [
        "Mn:s;p;d",
        "Mn:s:r=2",
    ]
