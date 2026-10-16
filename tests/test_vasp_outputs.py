"""The VASP parsers: parent bands, Fermi level, structure and projector supply.

The gate resolves every check against the parent calculation, so the VASP reader has
to deliver the same ``ParentCalculation`` the Quantum ESPRESSO reader does -- same
units (eV, Angstrom), same axis order, same k-point convention -- or a threshold that
is a fraction of the occupied bandwidth would silently be a fraction of something else.
``vasprun.xml`` carries bands, Fermi level and structure; ``EIGENVAL`` is the same
bands in a second file and is cross-checked rather than trusted; ``PROCAR`` is the
projector supply -- how much of each state the atomic channels carry -- which is what
the Quantum ESPRESSO path reads from ``projwfc.x``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from autowan.backends.vasp.outputs import (
    VaspOutputError,
    VaspProjections,
    potcar_channels,
    projector_supply,
    read_eigenval,
    read_kmesh,
    read_parent,
    read_procar,
)
from autowan.interpolation import cartesian_to_fractional_k
from autowan.qe_outputs import ParentCalculation

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


SYNTHETIC = Path(__file__).parent / "fixtures" / "vasp" / "synthetic"
FIXTURES = Path(__file__).parent / "fixtures" / "vasp"


@pytest.fixture
def parent_dir(tmp_path):
    """The synthetic Si parent: vasprun.xml alone, companions added per test."""
    shutil.copy(SYNTHETIC / "vasprun.xml", tmp_path / "vasprun.xml")
    return tmp_path


# ---------------------------------------------------------------------------
# vasprun.xml
# ---------------------------------------------------------------------------


def test_the_parent_is_the_one_class_the_gate_consumes(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    assert isinstance(parent, ParentCalculation)


def test_bands_arrive_in_ev_with_the_spin_k_band_axis_order(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    assert parent.eigenvalues_ev.shape == (1, 2, 4)
    assert parent.occupations.shape == (1, 2, 4)
    assert parent.n_spin == 1 and parent.n_bands == 4
    np.testing.assert_allclose(parent.eigenvalues_ev[0, 0], [-6.0, 5.0, 5.5, 8.0])
    np.testing.assert_allclose(parent.eigenvalues_ev[0, 1], [-4.0, 3.0, 6.5, 9.0])
    np.testing.assert_allclose(parent.occupations[0, 1], [1.0, 1.0, 0.0, 0.0])


def test_the_fermi_level_is_read_from_the_file_in_ev(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    assert parent.fermi_energy_ev == pytest.approx(5.2)


def test_the_structure_is_in_angstrom_with_fractional_positions(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    np.testing.assert_allclose(
        parent.lattice_ang,
        [
            [-2.71534056, 0.0, 2.71534056],
            [0.0, 2.71534056, 2.71534056],
            [-2.71534056, 2.71534056, 0.0],
        ],
    )
    assert parent.species == ("Si", "Si")
    np.testing.assert_allclose(parent.positions_frac, [[0, 0, 0], [0.25, 0.25, 0.25]])
    assert parent.alat_ang == pytest.approx(np.linalg.norm(parent.lattice_ang[0]))


def test_kpoints_round_trip_through_the_fractional_conversion_the_checks_use(
    parent_dir,
):
    """VASP writes k in fractional coordinates; the gate's checks convert *from* the
    2*pi/alat cartesian convention Quantum ESPRESSO writes. The reader must therefore
    hand over cartesian k such that the conversion returns exactly what VASP wrote."""
    parent = read_parent(parent_dir / "vasprun.xml")
    frac = cartesian_to_fractional_k(
        parent.kpoints_cart, parent.lattice_ang, parent.alat_ang
    )
    np.testing.assert_allclose(frac, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], atol=1e-10)


def test_a_gapped_parent_is_classified_from_its_own_bands(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    assert not parent.is_metallic()
    assert parent.crossing_bands() == 0
    assert parent.occupied_bands() == 2
    assert parent.band_gap_ev() == pytest.approx(0.5)


def test_the_calculation_kind_is_read_off_the_charge_density_setting(parent_dir):
    parent = read_parent(parent_dir / "vasprun.xml")
    assert parent.calculation == "scf"
    text = (parent_dir / "vasprun.xml").read_text()
    (parent_dir / "vasprun.xml").write_text(
        text.replace(
            '<i type="int" name="ICHARG">     2</i>',
            '<i type="int" name="ICHARG">    11</i>',
        )
    )
    assert read_parent(parent_dir / "vasprun.xml").calculation == "nscf"


def test_convergence_is_read_from_the_outcar_beside_the_xml_and_is_unknown_without_one(
    parent_dir,
):
    assert read_parent(parent_dir / "vasprun.xml").converged is None
    shutil.copy(SYNTHETIC / "OUTCAR.converged", parent_dir / "OUTCAR")
    parent = read_parent(parent_dir / "vasprun.xml")
    assert parent.converged is True
    assert parent.n_symmetries == 1
    shutil.copy(SYNTHETIC / "OUTCAR.unconverged", parent_dir / "OUTCAR")
    assert read_parent(parent_dir / "vasprun.xml").converged is False


def test_vasp_records_no_atomic_wavefunction_table(parent_dir):
    assert read_parent(parent_dir / "vasprun.xml").n_atomic_wfc is None


def test_a_spin_polarised_run_splits_the_spin_axis_not_the_band_axis(parent_dir):
    import copy
    import xml.etree.ElementTree as ET

    path = parent_dir / "vasprun.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    outer = root.find("./calculation/eigenvalues/array/set")
    spin1 = outer.find("set")
    spin2 = copy.deepcopy(spin1)
    spin2.set("comment", "spin 2")
    for row in spin2.iter("r"):
        energy, occupation = row.text.split()
        row.text = f" {float(energy) + 1.0:.4f} {occupation} "
    outer.append(spin2)
    for item in root.iter("i"):
        if item.get("name") == "ISPIN":
            item.text = " 2 "
    tree.write(path)

    parent = read_parent(path)
    assert parent.eigenvalues_ev.shape == (2, 2, 4)
    assert parent.eigenvalues_ev[0, 0, 0] == pytest.approx(-6.0)
    assert parent.eigenvalues_ev[1, 0, 0] == pytest.approx(-5.0)
    assert parent.n_spin == 2


def test_a_missing_fermi_level_is_an_error_not_a_default(parent_dir):
    text = (parent_dir / "vasprun.xml").read_text()
    start = text.index("  <dos>")
    end = text.index("  </dos>") + len("  </dos>\n")
    (parent_dir / "vasprun.xml").write_text(text[:start] + text[end:])
    with pytest.raises(VaspOutputError, match="Fermi"):
        read_parent(parent_dir / "vasprun.xml")


def test_a_fermi_level_the_xml_lacks_is_taken_from_the_outcar_beside_it(parent_dir):
    text = (parent_dir / "vasprun.xml").read_text()
    start = text.index("  <dos>")
    end = text.index("  </dos>") + len("  </dos>\n")
    (parent_dir / "vasprun.xml").write_text(text[:start] + text[end:])
    shutil.copy(SYNTHETIC / "OUTCAR.converged", parent_dir / "OUTCAR")
    assert read_parent(parent_dir / "vasprun.xml").fermi_energy_ev == pytest.approx(5.2)


def test_an_absent_or_bandless_file_is_refused_by_name(tmp_path):
    with pytest.raises(VaspOutputError, match="absent"):
        read_parent(tmp_path / "absent" / "vasprun.xml")
    (tmp_path / "vasprun.xml").write_text(
        "<modeling><calculation></calculation></modeling>"
    )
    with pytest.raises(VaspOutputError, match="eigenvalues"):
        read_parent(tmp_path / "vasprun.xml")


# ---------------------------------------------------------------------------
# EIGENVAL: the same bands in a second file, cross-checked
# ---------------------------------------------------------------------------


def test_eigenval_reads_the_same_bands_and_kpoints():
    eigenvalues, kpoints = read_eigenval(SYNTHETIC / "EIGENVAL")
    assert eigenvalues.shape == (1, 2, 4)
    np.testing.assert_allclose(eigenvalues[0, 0], [-6.0, 5.0, 5.5, 8.0])
    np.testing.assert_allclose(kpoints, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def test_an_eigenval_that_disagrees_with_the_xml_is_an_error(parent_dir):
    shutil.copy(SYNTHETIC / "EIGENVAL", parent_dir / "EIGENVAL")
    read_parent(parent_dir / "vasprun.xml")  # agreeing companions are silent
    text = (parent_dir / "EIGENVAL").read_text().replace("-6.000000", "-7.000000")
    (parent_dir / "EIGENVAL").write_text(text)
    with pytest.raises(VaspOutputError, match="EIGENVAL"):
        read_parent(parent_dir / "vasprun.xml")


# ---------------------------------------------------------------------------
# PROCAR: the projector supply
# ---------------------------------------------------------------------------


def test_procar_flattens_projectability_over_k_and_band_in_ev():
    proj = read_procar(SYNTHETIC / "PROCAR")
    assert isinstance(proj, VaspProjections)
    assert proj.n_kpoints == 2 and proj.n_bands == 4 and proj.n_ions == 2
    assert proj.energies_ev.shape == (8,)
    assert proj.projectability.shape == (8,)
    np.testing.assert_allclose(proj.energies_ev[:4], [-6.0, 5.0, 5.5, 8.0])
    # the total over ions and orbitals, per state
    np.testing.assert_allclose(proj.projectability[:4], [0.8, 0.6, 0.3, 0.1])
    assert proj.orbitals[:4] == ("s", "py", "pz", "px")


def test_the_projector_supply_names_the_channels_the_occupied_states_actually_use():
    proj = read_procar(SYNTHETIC / "PROCAR")
    supply = projector_supply(proj, species=("Si", "Si"), fermi_energy_ev=5.2)
    assert supply == {"Si": ("s", "p")}


def test_the_projector_supply_implies_the_wannier_count_the_deck_needs():
    from autowan.actions import implied_wannier_count

    proj = read_procar(SYNTHETIC / "PROCAR")
    supply = projector_supply(proj, species=("Si", "Si"), fermi_energy_ev=5.2)
    projectors = [{"species": s, "orbitals": o} for s, o in supply.items()]
    assert implied_wannier_count(projectors, {"Si": 2}) == 8


def test_a_channel_below_the_floor_is_not_supplied():
    proj = read_procar(SYNTHETIC / "PROCAR")
    # raising the floor above the largest p weight (0.72) but below s (0.80) leaves s alone
    supply = projector_supply(
        proj, species=("Si", "Si"), fermi_energy_ev=5.2, floor=0.75
    )
    assert supply == {"Si": ("s",)}


def test_a_procar_whose_ion_count_disagrees_with_the_species_list_is_refused():
    proj = read_procar(SYNTHETIC / "PROCAR")
    with pytest.raises(VaspOutputError, match="ions"):
        projector_supply(proj, species=("Si",), fermi_energy_ev=5.2)


# ---------------------------------------------------------------------------
# the real fixture parents, when the engine run has been done
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", ["Si", "SrO"])
def test_the_fixture_parent_reads_and_is_the_material_it_says(material):
    xml = FIXTURES / material / "vasprun.xml"
    if not xml.is_file():
        pytest.skip(f"no engine output at {xml}: the fixture SCF has not been run")
    parent = read_parent(xml)
    assert parent.calculation == "scf"
    assert parent.converged is True
    assert parent.n_spin == 1
    assert parent.kpoints_cart.shape == (8, 3)  # Gamma-centred 2x2x2, ISYM = -1
    assert not parent.is_metallic()
    expected = {"Si": ("Si", "Si"), "SrO": ("Sr", "O")}[material]
    assert parent.species == expected
    proj = read_procar(FIXTURES / material / "PROCAR")
    assert proj.n_kpoints == 8 and proj.n_bands == parent.n_bands
    supply = projector_supply(
        proj, species=parent.species, fermi_energy_ev=parent.fermi_energy_ev
    )
    assert set(supply) == set(expected)
    assert "p" in supply["Si" if material == "Si" else "O"]


# ---------------------------------------------------------------------------
# POTCAR channels: which angular momenta each species carries a projector for
# ---------------------------------------------------------------------------
#
# The automatic VASP deck names every channel the POTCARs carry (since
# 2026-09-14). The source is the pseudopotential's own description table:
# one row per partial wave, ``l  E  TYP  RCUT``. ``TYP 7`` marks the channel used as the
# local potential; it has no non-local projector and is not a channel a Wannier
# function can be projected on (O and Si list a d row of this kind). The same table is
# echoed into the OUTCAR of every run, with the lm-projection count ``LMMAX`` VASP
# derived from it -- which is what lets a committed fixture (no POTCAR: licensed) be
# read the same way as the library file.

POTCAR_LIBRARY = _stored("AUTOWAN_POTCAR_LIBRARY")


def _potcar_block(symbol, rows, nonlocal_ls=None):
    """A POTCAR header with the fields the channel reader reads, nothing licensed.

    rows: (l, reference energy, TYP). ``nonlocal_ls`` overrides the l values written
    as ``Non local Part`` blocks (default: every l with a non-local row).
    """
    lines = [
        f"  PAW_PBE {symbol} 01Jan2000",
        "   1.00000000000000000",
        " parameters from PSCTR are:",
        f"   VRHFIN ={symbol}: synthetic",
        "   LEXCH  = PE",
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000",
        "   Description",
        "     l       E           TYP  RCUT    TYP  RCUT",
    ]
    lines += [f"     {l}    {e:.7f}     {typ}  1.500" for l, e, typ in rows]
    lines += ["   Error from kinetic energy argument (eV)", "   NDATA  =      100"]
    ls = (
        nonlocal_ls
        if nonlocal_ls is not None
        else sorted({l for l, _, typ in rows if typ != 7})
    )
    for l in ls:
        lines += [" Non local Part", f"           {l}           2   1.50000000000000"]
        lines += ["  0.100000000000E+01  0.200000000000E+01"]
    lines += [" End of Dataset"]
    return "\n".join(lines) + "\n"


SEMICORE_CATION = [
    (0, -40.0, 23),
    (0, -3.0, 23),
    (1, -22.0, 23),
    (1, -24.0, 23),
    (2, -1.0, 23),
    (2, 0.2, 23),
]
ANION_WITH_LOCAL_D = [
    (0, -24.0, 23),
    (0, -9.5, 23),
    (1, -9.0, 23),
    (1, 8.0, 23),
    (2, -9.5, 7),
]


def test_potcar_channels_are_every_l_with_a_projector_per_species(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar_block("Sr", SEMICORE_CATION) + _potcar_block("O", ANION_WITH_LOCAL_D)
    )
    channels = potcar_channels(potcar)
    assert channels == {"Sr": ("s", "p", "d"), "O": ("s", "p")}
    assert list(channels) == ["Sr", "O"], "file order is the POSCAR species order"


def test_the_local_channel_is_not_a_projector_channel(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_potcar_block("O", ANION_WITH_LOCAL_D))
    assert "d" not in potcar_channels(potcar)["O"]


@pytest.mark.parametrize(
    "material, expected",
    [
        ("SrO", {"Sr": ("s", "p", "d"), "O": ("s", "p")}),
        ("Si", {"Si": ("s", "p")}),
    ],
)
def test_the_same_tables_are_read_from_the_outcar_the_parent_wrote(material, expected):
    assert potcar_channels(FIXTURES / material / "OUTCAR") == expected


def test_an_outcar_whose_table_disagrees_with_its_projector_count_is_refused(tmp_path):
    text = (FIXTURES / "SrO" / "OUTCAR").read_text()
    assert "LMMAX =          18" in text
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(text.replace("LMMAX =          18", "LMMAX =          10"))
    with pytest.raises(VaspOutputError, match="LMMAX"):
        potcar_channels(outcar)


# An OUTCAR echoes each pseudopotential's header in turn, and each header ends with the
# ``LMMAX`` VASP derived from that pseudopotential's table. Everything after the last
# header -- the rest of the run -- belongs to no pseudopotential, yet splitting the
# file on ``VRHFIN`` leaves the last species' block running to end-of-file. A later
# ``LMMAX`` line would then be checked against the last species' table. VASP 6.4.1
# does print per-species lm counts there: a LOCPROJ_AUTO run's atomic calculations
# write ``LMMAX:        18`` for Sr after O's header (measured on the matched-mesh
# SrO run, 2026-09-15). That form uses a colon, which the ``=`` pattern misses by
# accident; the check must not rest on the accident. Sr's count, written with ``=``
# after O's header, is the two-species case where an unscoped check misfires.
LATER_SR_LMMAX = (
    "\n Calculating: Sr\n l-properties of potential\n   LMMAX =          18\n"
)


def test_a_later_lmmax_line_is_not_charged_to_the_last_pseudopotential(tmp_path):
    text = (FIXTURES / "SrO" / "OUTCAR").read_text()
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(text + LATER_SR_LMMAX)
    assert potcar_channels(outcar) == {"Sr": ("s", "p", "d"), "O": ("s", "p")}


def test_the_last_pseudopotentials_own_lmmax_is_still_checked(tmp_path):
    """Scoping keeps each header's own count: O's, altered, is refused.

    A later line carrying O's true count must not rescue the header it follows.
    """
    text = (FIXTURES / "SrO" / "OUTCAR").read_text()
    assert text.count("LMMAX =           8") == 1
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        text.replace("LMMAX =           8", "LMMAX =           9")
        + "\n   LMMAX =           8\n"
    )
    with pytest.raises(VaspOutputError, match="O Description disagrees with LMMAX"):
        potcar_channels(outcar)


def test_a_potcar_whose_nonlocal_parts_disagree_with_its_table_is_refused(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_potcar_block("Sr", SEMICORE_CATION, nonlocal_ls=[0, 1]))
    with pytest.raises(VaspOutputError, match="Non local"):
        potcar_channels(potcar)


def test_a_file_with_no_pseudopotential_table_is_refused_by_name(tmp_path):
    empty = tmp_path / "POTCAR"
    empty.write_text("nothing here\n")
    with pytest.raises(VaspOutputError, match=str(empty)):
        potcar_channels(empty)
    with pytest.raises(VaspOutputError, match="no VASP output"):
        potcar_channels(tmp_path / "absent")


def test_one_element_listed_twice_with_different_channels_is_refused(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar_block("Sr", SEMICORE_CATION)
        + _potcar_block("Sr", [(0, -3.0, 23), (1, -2.0, 23)])
    )
    with pytest.raises(VaspOutputError, match="Sr"):
        potcar_channels(potcar)


@pytest.mark.parametrize(
    "material, entries",
    [("SrO", ("Sr_sv", "O")), ("Si", ("Si",))],
)
def test_the_library_potcars_agree_with_the_outcar_echo(tmp_path, material, entries):
    if not all((POTCAR_LIBRARY / e / "POTCAR").is_file() for e in entries):
        pytest.skip(f"no POTCAR library at {POTCAR_LIBRARY} on this machine")
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "".join((POTCAR_LIBRARY / e / "POTCAR").read_text() for e in entries)
    )
    assert potcar_channels(potcar) == potcar_channels(FIXTURES / material / "OUTCAR")


# ---------------------------------------------------------------------------
# the k-mesh VASP actually ran
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", ["SrO", "Si"])
def test_the_mesh_is_read_from_the_generation_block_of_the_run(material):
    assert read_kmesh(FIXTURES / material / "vasprun.xml") == (2, 2, 2)


def test_a_mesh_whose_kpoint_list_is_symmetry_reduced_is_refused(tmp_path):
    text = (SYNTHETIC / "vasprun.xml").read_text()
    reduced = tmp_path / "vasprun.xml"
    reduced.write_text(text.replace("2        1        1", "2        2        1", 1))
    with pytest.raises(VaspOutputError, match="ISYM"):
        read_kmesh(reduced)


def test_a_run_with_no_mesh_divisions_is_refused(tmp_path):
    text = (SYNTHETIC / "vasprun.xml").read_text()
    listed = tmp_path / "vasprun.xml"
    start = text.index("<generation")
    end = text.index("</generation>") + len("</generation>")
    listed.write_text(text[:start] + text[end:])
    with pytest.raises(VaspOutputError, match="divisions"):
        read_kmesh(listed)
