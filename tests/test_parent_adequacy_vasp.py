"""The parent-adequacy check on VASP parents: the VASP partition of the one object.

The judgement object (``autowan.parent_adequacy``) is engine-agnostic; this file pins the
VASP knowledge partition that feeds it -- facts read from ``vasprun.xml`` and the
pseudopotential tables beside it, the deck's demand read from the ``.win`` VASP's
automatic first deck writes, and prescriptions worded in VASP's own terms (``NBANDS``,
``ISYM``/``KPOINTS``, ``ISPIN``/``MAGMOM``, a POTCAR variant). autowan never alters the
parent: every prescription is data a caller may apply to a copy.

One engine-agnostic addition is pinned here as well. VASP computes a hydrogenic
projection's one-centre part from the POTCAR's own projectors, so a deck that names an
angular momentum the parent's pseudopotential has no projector for cannot be projected
the way the deck assumes. The object therefore accepts, per species, the channels a
deck takes from the parent's pseudopotentials (``DeckDemand.demanded_channels``) and
reports a shortfall under the inventory check -- LIMITING, like every inventory finding:
it limits the projections, it does not establish a parent cause.

Degraded parents here are copies of the committed 2x2x2 silicon fixture with one defect
written into the copy. Tests that read the twenty real VASP parents of the 2026-09-29
campaign (read in place, never written) skip where those files are absent.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autowan import parent_adequacy as pa
from autowan.backends import get_backend
from autowan.backends.vasp.inputs import deck_from_parent
from autowan.backends.vasp.knowledge import VaspAdequacyPartition
from autowan.backends.vasp.outputs import (
    VaspOutputError,
    potcar_channels,
    potcar_valences,
)
from autowan.execution import ConstructionRunner, StepResult
from autowan.inputs import read_construction
from autowan.parent_adequacy import (
    ADEQUATE,
    BLOCKING,
    INADEQUATE,
    LIMITING,
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    Channel,
    DeckDemand,
    DftChange,
    ParentFacts,
    Prescription,
    assess_parent,
    diagnose_terminal,
)


def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
TASK5 = _stored("AUTOWAN_TEST_VASP_PARENTS")
TWENTY = (
    "Ba C Cu Cu2O GaAs La Mn MnO Na Si SiC Sr SrO SrTiO3 Ti TiC TiO2 V V3Si W".split()
)
#: The campaign's own magnetic set-up: only MnO is a (ferromagnetic) magnet.
DECLARED = {m: {} for m in TWENTY}
DECLARED["MnO"] = {"Mn": 5.0, "O": 0.0}

BAND_COUNT = "band_count_covers_demand"
KLIST = "kpoints_are_the_decks_full_grid"
SPIN = "spin_treatment_matches_magnetic_state"
INVENTORY = "projector_inventory_covers_fermi_level"
HOPPING = "mesh_resolves_hopping_decay"
BAND_PARTITION = "band_partition_matches_deck"

#: The 2026-09-29 specificity arms (VASP 6.4.1, 216 k-points, read in place): a Si parent
#: computed with NBANDS = 24 under the fixed 16-band all-channel deck, which autowan's
#: deck-overlap check refused, and the 12-band parent under the deck written for it.
SPECIFICITY = _stored("AUTOWAN_TEST_SPECIFICITY")


@pytest.fixture
def vasp():
    return get_backend("vasp")


# ---------------------------------------------------------------------------
# helpers: copies of the silicon fixture with one defect written in
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent_copy(tmp_path: Path, name: str = "parent", material: str = "Si") -> Path:
    """The fixture parent's XML and OUTCAR (no EIGENVAL: edits go to the XML only)."""
    d = tmp_path / name
    d.mkdir()
    for f in ("vasprun.xml", "OUTCAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(FIXTURES / material / f, d / f)
    return d


def _edit_xml(path: Path, edit) -> None:
    tree = ET.parse(path)
    edit(tree.getroot())
    tree.write(path, encoding="unicode")


def _spin_sets(root):
    return root.find("calculation/eigenvalues/array/set")


def _set_incar(root, name, value, tag="i"):
    incar = root.find("incar")
    item = incar.find(f"*[@name='{name}']")
    if item is None:
        item = ET.SubElement(incar, tag, {"name": name})
    item.text = f" {value} "


def _set_parameter(root, name, value):
    for item in root.iter("i"):
        if item.get("name") == name and item is not None:
            item.text = f" {value} "


def truncate_bands(root, n):
    for spin in _spin_sets(root).findall("set"):
        for kpoint in spin.findall("set"):
            for r in kpoint.findall("r")[n:]:
                kpoint.remove(r)
    _set_incar(root, "NBANDS", n)


def pad_bands(root, n):
    """Extend every k-point's spectrum to ``n`` bands, empty, 1 eV apart above the top."""
    for spin in _spin_sets(root).findall("set"):
        for kpoint in spin.findall("set"):
            rows = kpoint.findall("r")
            top = float(rows[-1].text.split()[0])
            for i in range(n - len(rows)):
                r = ET.SubElement(kpoint, "r")
                r.text = f" {top + 1.0 + i:.4f} 0.0000 "
    _set_incar(root, "NBANDS", n)
    _set_parameter(root, "NBANDS", n)


def reduce_kpoints(root, keep):
    kpoints = root.find("kpoints")
    for name in ("kpointlist", "weights"):
        varray = kpoints.find(f"varray[@name='{name}']")
        rows = varray.findall("v")
        for i, v in enumerate(rows):
            if i not in keep:
                varray.remove(v)
    for spin in _spin_sets(root).findall("set"):
        for i, kpoint in enumerate(spin.findall("set")):
            if i not in keep:
                spin.remove(kpoint)
    _set_incar(root, "ISYM", 2)
    _set_parameter(root, "ISYM", 2)


def spin_polarise(root):
    """Make the fixture an ISPIN = 2 parent with 5 up and 3 down electrons per cell."""
    outer = _spin_sets(root)
    up = outer.find("set")
    down = copy.deepcopy(up)
    down.set("comment", "spin 2")
    outer.append(down)
    for spin, band, occ in ((up, 4, 1.0), (down, 3, 0.0)):
        for kpoint in spin.findall("set"):
            r = kpoint.findall("r")[band]
            energy = r.text.split()[0]
            r.text = f" {energy} {occ:.4f} "
    _set_parameter(root, "ISPIN", 2)
    _set_incar(root, "ISPIN", 2)


def write_deck(parent: Path, run_dir: Path, seed: str = "si", **kw) -> Path:
    run_dir.mkdir(exist_ok=True)
    kw.setdefault("recipe", "hydrogenic")
    deck_from_parent(parent / "vasprun.xml", run_dir, seed, **kw)
    return run_dir


def _potcar_entry(symbol, zval, rows, nonlocal_parts):
    """A POTCAR header with only the fields the readers use; nothing licensed.

    rows: (l, TYP) Description rows; nonlocal_parts: (l, nproj) Non local Part blocks.
    """
    element = symbol.split("_")[0]
    lines = [
        f"  PAW_PBE {symbol} 01Jan2000",
        f"   {zval:.8f}",
        " parameters from PSCTR are:",
        f"   VRHFIN ={element}: synthetic",
        "   LEXCH  = PE",
        f"   POMASS =   10.000; ZVAL   =   {zval:7.3f}    mass and valenz",
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000",
        "   Description",
        "     l       E           TYP  RCUT    TYP  RCUT",
    ]
    lines += [
        f"     {l}    {-1.0 - i:.7f}     {typ}  1.500"
        for i, (l, typ) in enumerate(rows)
    ]
    lines += ["   Error from kinetic energy argument (eV)", "   NDATA  =      100"]
    for l, nproj in nonlocal_parts:
        lines += [
            " Non local Part",
            f"           {l}           {nproj}   1.50000000000000",
            "  0.100000000000E+01  0.200000000000E+01",
        ]
    lines += [" End of Dataset"]
    return "\n".join(lines) + "\n"


def potcar_library(root: Path) -> Path:
    """Three silicon variants: two at ZVAL 4 (with and without d), one at ZVAL 12."""
    entries = {
        "Si": (4.0, [(0, 23), (0, 23), (1, 23), (1, 23), (2, 7)], [(0, 2), (1, 2)]),
        "Si_GW": (
            4.0,
            [(0, 23), (0, 23), (1, 23), (1, 23), (2, 23), (2, 23), (3, 23)],
            [(0, 2), (1, 2), (2, 2)],
        ),
        "Si_sv_GW": (
            12.0,
            [(0, 23), (1, 23), (2, 23), (3, 23)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
        ),
        "Sr_sv": (10.0, [(0, 23), (1, 23), (2, 23)], [(0, 1), (1, 1), (2, 1)]),
    }
    for name, (zval, rows, parts) in entries.items():
        (root / name).mkdir(parents=True)
        (root / name / "POTCAR").write_text(_potcar_entry(name, zval, rows, parts))
    return root


# ---------------------------------------------------------------------------
# the engine-agnostic addition: the channels a deck takes from the pseudopotential
# ---------------------------------------------------------------------------


def _facts(channels=None, n_bands=16):
    k = pa.full_grid((2, 2, 2))
    return ParentFacts(
        engine="synthetic",
        eigenvalues_ev=np.broadcast_to(
            np.arange(n_bands, dtype=float) - 3.5, (1, len(k), n_bands)
        ).copy(),
        fermi_energy_ev=0.0,
        kpoints_frac=k,
        lattice_ang=np.eye(3) * 3.0,
        species=("A", "B"),
        n_symmetries=1,
        spin_polarised=False,
        channels=channels if channels is not None else {},
        species_order=("A", "B"),
    )


def _demand(demanded=None, route="analytic"):
    return DeckDemand(
        num_wann=8,
        num_bands=16,
        mp_grid=(2, 2, 2),
        kpoints_frac=pa.full_grid((2, 2, 2)),
        frozen_window_ev=(None, 2.0),
        projector_route=route,
        demanded_channels=demanded,
    )


SP = (Channel("s", 0), Channel("p", 1))


def test_a_deck_that_demands_a_channel_the_pseudopotential_lacks_is_limiting():
    report = assess_parent(
        _facts({"A": SP, "B": SP}), _demand({"A": (0, 1, 2), "B": (0, 1)})
    )
    f = report.finding(INVENTORY)
    assert f.status == INADEQUATE
    assert f.severity == LIMITING
    assert f.measured["missing_channels"] == {"A": [2]}
    assert f.measured["demanded_channels"] == {"A": [0, 1, 2], "B": [0, 1]}
    assert report.blocking == ()
    assert "not a parent cause" in f.reason


def test_channels_the_pseudopotential_carries_are_adequate():
    report = assess_parent(
        _facts({"A": SP, "B": SP}), _demand({"A": (0, 1), "B": (0,)})
    )
    f = report.finding(INVENTORY)
    assert f.status == ADEQUATE
    assert f.measured["missing_channels"] == {}
    assert report.alarms == ()


def test_unknown_channels_leave_the_coverage_not_evaluable_not_passed():
    report = assess_parent(_facts({"A": SP}), _demand({"A": (0,), "B": (0,)}))
    assert report.finding(INVENTORY).status == NOT_EVALUABLE
    report = assess_parent(_facts({}), _demand({"A": (0,)}))
    assert report.finding(INVENTORY).status == NOT_EVALUABLE


def test_without_demanded_channels_an_analytic_deck_is_still_not_applicable():
    """The QE behaviour: analytic projections there do not use the pseudopotential."""
    report = assess_parent(_facts({"A": SP}), _demand(None, route="analytic"))
    assert report.finding(INVENTORY).status == NOT_APPLICABLE


def test_a_coverage_limit_reaches_the_terminal_diagnosis_as_an_advisory_not_a_cause():
    def prescribe(finding, facts, demand):
        return Prescription(finding.check_id, "use another pseudopotential", ())

    preflight = assess_parent(
        _facts({"A": SP, "B": SP}),
        _demand({"A": (0, 1, 2), "B": (0, 1)}),
        prescribe=prescribe,
    )
    dx = diagnose_terminal(
        preflight,
        [pa.AttemptEvidence(0, {}, None, ("centres_sit_on_atomic_sites",))],
        prescribe=prescribe,
    )
    assert dx.cause_established is False
    assert dx.blocked_by_parent is False
    assert not dx.returns_prescription
    assert [a.check_id for a in dx.advisories] == [INVENTORY]
    assert any(e.check_id == INVENTORY and e.severity == LIMITING for e in dx.events)
    assert "use another pseudopotential" in dx.message()


def test_the_judgement_object_still_names_no_engine_parameter():
    source = inspect.getsource(pa)
    for word in (
        "NBANDS",
        "ISYM",
        "ISPIN",
        "MAGMOM",
        "POTCAR",
        "KPOINTS",
        "INCAR",
        "nbnd",
        "nosym",
    ):
        assert word not in source, word


# ---------------------------------------------------------------------------
# the POTCAR reader: a local-potential row typed 23 is not a projector channel
# ---------------------------------------------------------------------------
#
# Several standard POTCARs list the local-potential reference as the Description
# table's last row with TYP = 23, not 7 (Cu l = 3, Na l = 2, V_sv, Ti_sv, Ga_d). The
# projector inventory is what VASP reads as non-local: the ``Non local Part`` blocks of
# a POTCAR, and the ``non local Contribution for L=`` lines the OUTCAR echo prints, one
# per projector. Their sum of 2l+1 is LMMAX.

CU_LIKE_OUTCAR = """\
 POTCAR:    PAW_PBE Cu 01Jan2000
   VRHFIN =Cu: synthetic
   LEXCH  = PE
   POMASS =   63.546; ZVAL   =   11.000    mass and valenz
   TITEL  = PAW_PBE Cu 01Jan2000
   Description
     l       E           TYP  RCUT    TYP  RCUT
     2     -5.0000000     23  2.200
     2     10.0000000     23  2.200
     0     -4.0000000     23  2.200
     0      8.0000000     23  2.200
     1     -2.0000000     23  2.200
     1     20.0000000     23  2.200
     3      2.0000000     23  2.300
  local pseudopotential read in
  partial core-charges read in
  atomic valenz-charges read in
  non local Contribution for L=           2  read in
    real space projection operators read in
  non local Contribution for L=           2  read in
    real space projection operators read in
  non local Contribution for L=           0  read in
    real space projection operators read in
  non local Contribution for L=           0  read in
    real space projection operators read in
  non local Contribution for L=           1  read in
    real space projection operators read in
  non local Contribution for L=           1  read in
    real space projection operators read in
    PAW grid and wavefunctions read in

   number of l-projection  operators is LMAX  =           6
   number of lm-projection operators is LMMAX =          18

"""


def test_a_local_row_typed_23_is_not_a_channel_in_the_outcar_echo(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(CU_LIKE_OUTCAR)
    assert potcar_channels(outcar) == {"Cu": ("s", "p", "d")}


def test_a_local_row_typed_23_is_not_a_channel_in_the_potcar(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar_entry("Na", 1.0, [(0, 23), (0, 23), (1, 23), (2, 23)], [(0, 2), (1, 1)])
    )
    assert potcar_channels(potcar) == {"Na": ("s", "p")}


def test_the_outcar_echo_is_still_checked_against_its_own_lmmax(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        CU_LIKE_OUTCAR.replace("LMMAX =          18", "LMMAX =          25")
    )
    with pytest.raises(VaspOutputError, match="Cu Description disagrees with LMMAX"):
        potcar_channels(outcar)


def test_more_than_the_trailing_row_left_unmatched_is_still_refused(tmp_path):
    """Two d rows with no d projector are not a local reference; refuse the table."""
    text = CU_LIKE_OUTCAR.replace(
        "  non local Contribution for L=           2  read in\n"
        "    real space projection operators read in\n",
        "",
    ).replace("LMMAX =          18", "LMMAX =           8")
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(text)
    with pytest.raises(VaspOutputError, match="Cu"):
        potcar_channels(outcar)


LATER_ECHO = (
    "\n Calculating: Sr\n"
    "  non local Contribution for L=           2  read in\n"
    "  non local Contribution for L=           2  read in\n"
)


def test_projector_echo_lines_after_a_header_are_not_charged_to_it(tmp_path):
    """Like LMMAX (Task 4's scoping), only the header's own echo counts: text after
    the last pseudopotential's LMMAX line belongs to the rest of the run."""
    text = (FIXTURES / "SrO" / "OUTCAR").read_text()
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(text + LATER_ECHO)
    assert potcar_channels(outcar) == {"Sr": ("s", "p", "d"), "O": ("s", "p")}


def test_the_valence_is_read_per_species_from_the_same_tables(tmp_path):
    assert potcar_valences(FIXTURES / "Si" / "OUTCAR") == {"Si": 4.0}
    assert potcar_valences(FIXTURES / "SrO" / "OUTCAR") == {"Sr": 10.0, "O": 6.0}
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(CU_LIKE_OUTCAR)
    assert potcar_valences(outcar) == {"Cu": 11.0}


@pytest.mark.skipif(not TASK5.is_dir(), reason="the 2026-09-29 VASP parents are absent")
@pytest.mark.parametrize("material", TWENTY)
def test_every_campaign_parent_s_outcar_reads_the_non_local_inventory(material):
    """Independent reference: the campaign's own record of each Non local Part set."""
    outcar = TASK5 / material / "OUTCAR"
    provenance = json.loads((TASK5 / material / "provenance.json").read_text())
    expected = {
        s: tuple(h["channels"]) for s, h in provenance["settings"]["headers"].items()
    }
    before = _sha(outcar)
    assert potcar_channels(outcar) == expected
    assert _sha(outcar) == before


# ---------------------------------------------------------------------------
# the VASP partition: the backend carries it; facts from the parent
# ---------------------------------------------------------------------------


def test_the_vasp_backend_carries_an_adequacy_partition(vasp):
    assert isinstance(vasp.adequacy, VaspAdequacyPartition)
    assert vasp.adequacy.engine == "vasp"


def test_the_vasp_partition_reads_the_parent_facts(vasp):
    xml = FIXTURES / "Si" / "vasprun.xml"
    f = vasp.adequacy.read_facts(xml)
    root = ET.parse(xml).getroot()
    kfrac = np.array(
        [
            [float(x) for x in v.text.split()]
            for v in root.findall("kpoints/varray[@name='kpointlist']/v")
        ]
    )
    assert f.engine == "vasp"
    assert f.eigenvalues_ev.shape == (1, 8, 16)
    np.testing.assert_allclose(f.kpoints_frac, kfrac)
    assert f.n_symmetries == 1
    assert f.spin_polarised is False and f.noncollinear is False
    assert f.moment_bohr is None
    assert f.species == ("Si", "Si")
    assert f.species_order == ("Si",)
    assert f.channels == {"Si": (Channel("s", 0), Channel("p", 1))}
    assert f.projector_functions == 8
    assert f.fermi_energy_ev == pytest.approx(6.37156489)
    assert f.source == str(xml)


def test_a_default_magmom_in_the_run_parameters_is_not_a_request(vasp):
    """vasprun lists MAGMOM = 1 per atom under parameters even for ISPIN = 1."""
    xml = FIXTURES / "Si" / "vasprun.xml"
    assert 'name="MAGMOM"' in xml.read_text()
    assert vasp.adequacy.read_facts(xml).requested_moments == {}


def test_a_magmom_written_in_the_input_is_the_parents_own_declaration(tmp_path, vasp):
    parent = _parent_copy(tmp_path)
    _edit_xml(
        parent / "vasprun.xml",
        lambda root: _set_incar(root, "MAGMOM", "1.0 1.0", tag="v"),
    )
    f = vasp.adequacy.read_facts(parent / "vasprun.xml")
    assert f.requested_moments == {"Si": 1.0}
    assert f.spin_polarised is False


def test_the_moment_is_read_from_the_spin_resolved_occupations(tmp_path, vasp):
    parent = _parent_copy(tmp_path)
    _edit_xml(parent / "vasprun.xml", spin_polarise)
    f = vasp.adequacy.read_facts(parent / "vasprun.xml")
    assert f.spin_polarised is True
    assert f.eigenvalues_ev.shape == (2, 8, 16)
    assert f.moment_bohr == pytest.approx(2.0)


def test_without_a_pseudopotential_table_the_channels_are_unknown(tmp_path, vasp):
    parent = _parent_copy(tmp_path)
    (parent / "OUTCAR").unlink()
    f = vasp.adequacy.read_facts(parent / "vasprun.xml")
    assert f.channels == {}
    assert f.projector_functions is None


def test_a_potcar_beside_the_parent_is_read_before_the_outcar_echo(tmp_path, vasp):
    parent = _parent_copy(tmp_path)
    (parent / "POTCAR").write_text(
        _potcar_entry(
            "Si_GW",
            4.0,
            [(0, 23), (0, 23), (1, 23), (1, 23), (2, 23), (2, 23), (3, 23)],
            [(0, 2), (1, 2), (2, 2)],
        )
    )
    f = vasp.adequacy.read_facts(parent / "vasprun.xml")
    assert [c.l for c in f.channels["Si"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# the VASP partition: the deck's demand
# ---------------------------------------------------------------------------


def test_the_demand_is_read_from_the_deck_the_automatic_writer_makes(tmp_path, vasp):
    parent = FIXTURES / "Si"
    run_dir = write_deck(parent, tmp_path / "run")
    inputs = read_construction(run_dir, "si")
    f = vasp.adequacy.read_facts(parent / "vasprun.xml")
    d = vasp.adequacy.read_demand(inputs, f)
    win = (run_dir / "si.win").read_text()
    top = float(
        next(
            ln.split("=")[1] for ln in win.splitlines() if ln.startswith("dis_froz_max")
        )
    )
    assert d.num_wann == 8
    assert d.num_bands == 16
    assert d.excluded_bands == ()
    assert d.mp_grid == (2, 2, 2)
    np.testing.assert_allclose(d.kpoints_frac, f.kpoints_frac, atol=1e-9)
    assert d.frozen_window_ev == (None, pytest.approx(top))
    assert d.projector_route == "analytic"
    assert d.demanded_channels == {"Si": (0, 1)}
    assert d.spin_channel is None


def test_exclude_bands_and_composite_orbitals_are_read(tmp_path, vasp):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    win = run_dir / "si.win"
    text = win.read_text().replace(
        "begin projections\nSi:s;p\nend projections",
        "begin projections\nSi: sp3\nSi:l=2\nend projections",
    )
    assert "Si: sp3" in text
    win.write_text("exclude_bands = 1-2, 5\n" + text)
    f = vasp.adequacy.read_facts(FIXTURES / "Si" / "vasprun.xml")
    d = vasp.adequacy.read_demand(read_construction(run_dir, "si"), f)
    assert d.excluded_bands == (1, 2, 5)
    assert d.demanded_channels == {"Si": (0, 1, 2)}


def test_a_deck_without_named_projections_demands_no_channels(tmp_path, vasp):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    win = run_dir / "si.win"
    lines = win.read_text().splitlines()
    start = lines.index("begin projections")
    end = lines.index("end projections")
    win.write_text("\n".join(lines[:start] + lines[end + 1 :]) + "\n")
    f = vasp.adequacy.read_facts(FIXTURES / "Si" / "vasprun.xml")
    d = vasp.adequacy.read_demand(read_construction(run_dir, "si"), f)
    assert d.projector_route == "unknown"
    assert d.demanded_channels is None


# ---------------------------------------------------------------------------
# positive and negative controls on the fixture, and the VASP prescriptions
# ---------------------------------------------------------------------------


def _assess(vasp, parent: Path, run_dir: Path, seed="si", declared=None):
    return vasp.assess_parent(
        parent / "vasprun.xml",
        read_construction(run_dir, seed),
        declared_moments={} if declared is None else declared,
    )


@pytest.mark.parametrize("material, seed", [("Si", "si"), ("SrO", "sro")])
def test_a_clean_parent_with_its_own_all_channel_deck_raises_no_alarm(
    tmp_path, vasp, material, seed
):
    parent = FIXTURES / material
    run_dir = write_deck(parent, tmp_path / "run", seed)
    report = _assess(vasp, parent, run_dir, seed)
    assert report.alarms == ()
    assert report.finding(INVENTORY).status == ADEQUATE


def test_a_truncated_band_count_is_blocking_and_prescribes_nbands(tmp_path, vasp):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "truncated")
    _edit_xml(parent / "vasprun.xml", lambda root: truncate_bands(root, 12))
    report = _assess(vasp, parent, run_dir)
    assert [f.check_id for f in report.blocking] == [BAND_COUNT]
    f = report.finding(BAND_COUNT)
    assert f.measured["parent_bands"] == 12
    assert f.measured["required_bands"] == 16
    p = f.prescription
    assert p.changes == (DftChange("nscf", "INCAR", "NBANDS", 16),)
    assert "NBANDS = 16" in p.message
    assert "nbnd" not in p.message


def test_a_surplus_band_parent_under_the_fixed_deck_raises_the_partition_alarm(
    tmp_path, vasp
):
    """The 2026-09-29 blind spot on the fixture: the 16-band deck written for the
    16-band parent, then the parent recomputed with NBANDS = 24. VASP's overlap step
    inherits NBANDS, so the overlaps span 24 bands against the deck's 16."""
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "surplus")
    _edit_xml(parent / "vasprun.xml", lambda root: pad_bands(root, 24))
    report = _assess(vasp, parent, run_dir)
    assert [f.check_id for f in report.blocking] == [BAND_PARTITION]
    assert report.finding(BAND_COUNT).status == ADEQUATE
    f = report.finding(BAND_PARTITION)
    assert f.measured["parent_bands"] == 24
    assert f.measured["num_bands"] == 16
    assert f.measured["overlap_bands"] == 24
    assert f.measured["surplus_bands"] == 8
    message = f.prescription.message
    assert "24" in message and "16" in message
    assert "exclude_bands = 17-24" in message
    assert "NBANDS" in message and "nbnd" not in message
    assert f.measured["frozen_window_closed_by_deck_bands"] is True
    assert f.prescription.changes == (DftChange("nscf", "INCAR", "NBANDS", 16),)


def test_the_partition_alarm_stops_the_workflow_before_the_deck_overlap_check(
    tmp_path,
):
    """Not 'pass pre-flight, then be refused at the deck-overlap check': the pre-flight
    returns its prescription and nothing is built."""
    from autowan.workflow import WorkflowConfig, wannierize

    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "surplus")
    _edit_xml(parent / "vasprun.xml", lambda root: pad_bands(root, 24))
    runner = _NeverBuilds(run_dir)
    result = wannierize(
        run_dir=run_dir,
        parent_xml=parent / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
        declared_moments={},
    )
    assert runner.builds == []
    assert result.run.attempts == []
    assert [f.check_id for f in result.parent_adequacy.blocking] == [BAND_PARTITION]
    assert result.diagnosis.cause_established is True
    assert "exclude_bands = 17-24" in result.prescription


@pytest.mark.skipif(
    not (SPECIFICITY / "si_surplus_bands").is_dir(),
    reason="the 2026-09-29 VASP specificity arms are absent",
)
def test_the_real_surplus_band_arm_now_raises_the_partition_alarm(vasp):
    arm = SPECIFICITY / "si_surplus_bands"
    xml = arm / "parent" / "nscf" / "vasprun.xml"
    before = _sha(xml)
    report = vasp.assess_parent(
        xml, read_construction(arm / "construction", "si"), declared_moments={}
    )
    assert [f.check_id for f in report.blocking] == [BAND_PARTITION]
    f = report.finding(BAND_PARTITION)
    assert (f.measured["parent_bands"], f.measured["num_bands"]) == (24, 16)
    assert "24" in f.prescription.message and "16" in f.prescription.message
    assert _sha(xml) == before


@pytest.mark.skipif(
    not (SPECIFICITY / "si_consistent_deck").is_dir(),
    reason="the 2026-09-29 VASP specificity arms are absent",
)
def test_the_real_consistent_deck_arm_still_raises_no_alarm(vasp):
    arm = SPECIFICITY / "si_consistent_deck"
    report = vasp.assess_parent(
        arm / "parent" / "nscf" / "vasprun.xml",
        read_construction(arm / "deck", "si"),
        declared_moments={},
    )
    assert report.alarms == ()
    assert report.finding(BAND_PARTITION).status == ADEQUATE


def test_a_symmetry_reduced_parent_is_blocking_and_prescribes_the_full_mesh(
    tmp_path, vasp
):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "reduced")
    _edit_xml(parent / "vasprun.xml", lambda root: reduce_kpoints(root, {0, 1, 3}))
    with (parent / "OUTCAR").open("a") as handle:
        handle.write("\n Subroutine INISYM returns: Found 48 space group operations\n")
    report = _assess(vasp, parent, run_dir)
    assert [f.check_id for f in report.blocking] == [KLIST]
    f = report.finding(KLIST)
    assert f.measured["missing"] == 5
    assert f.measured["symmetry_reduced"] is True
    changes = f.prescription.changes
    assert DftChange("nscf", "INCAR", "ISYM", -1) in changes
    assert DftChange("nscf", "KPOINTS", "Gamma", [2, 2, 2]) in changes
    assert "ISYM = -1" in f.prescription.message
    assert "nosym" not in f.prescription.message


def test_a_declared_magnet_computed_without_spin_is_blocking_and_prescribes_ispin(
    tmp_path, vasp
):
    parent = FIXTURES / "Si"
    run_dir = write_deck(parent, tmp_path / "run")
    report = _assess(vasp, parent, run_dir, declared={"Si": 1.0})
    assert [f.check_id for f in report.blocking] == [SPIN]
    changes = report.finding(SPIN).prescription.changes
    for step in ("scf", "nscf"):
        assert DftChange(step, "INCAR", "ISPIN", 2) in changes
        assert DftChange(step, "INCAR", "MAGMOM", [1.0, 1.0]) in changes
    message = report.finding(SPIN).prescription.message
    assert "ISPIN = 2" in message and "MAGMOM = 1.0 1.0" in message


def test_a_spin_polarised_parent_keeping_its_declared_moment_is_adequate(
    tmp_path, vasp
):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "polarised")
    _edit_xml(parent / "vasprun.xml", spin_polarise)
    report = _assess(vasp, parent, run_dir, declared={"Si": 1.0})
    assert report.finding(SPIN).status == ADEQUATE
    assert report.finding(SPIN).measured["moment_bohr"] == pytest.approx(2.0)


def _missing_d_deck(tmp_path):
    """The fixture parent (Si: s, p only) under a deck that names Si d."""
    run_dir = write_deck(
        FIXTURES / "Si",
        tmp_path / "run",
        projectors=[{"species": "Si", "orbitals": ("s", "d")}],
    )
    return run_dir


def test_a_pseudopotential_missing_a_demanded_channel_is_limiting_and_names_a_variant(
    tmp_path,
):
    library = potcar_library(tmp_path / "lib")
    partition = VaspAdequacyPartition(potcar_library=library)
    run_dir = _missing_d_deck(tmp_path)
    xml = FIXTURES / "Si" / "vasprun.xml"
    facts = partition.read_facts(xml)
    demand = partition.read_demand(read_construction(run_dir, "si"), facts)
    assert demand.demanded_channels == {"Si": (0, 2)}
    report = assess_parent(
        facts, demand, prescribe=partition.prescribe, declared_moments={}
    )
    assert report.blocking == ()
    f = report.finding(INVENTORY)
    assert (f.status, f.severity) == (INADEQUATE, LIMITING)
    assert f.measured["missing_channels"] == {"Si": [2]}
    assert f.prescription.changes == (
        DftChange("scf", "POTCAR", "Si", "Si_GW"),
        DftChange("nscf", "POTCAR", "Si", "Si_GW"),
    )
    message = f.prescription.message
    assert "Si_GW" in message and "Si_sv_GW" not in message
    assert "ZVAL" in message and "d" in message


def test_the_library_may_come_from_the_environment(tmp_path, vasp, monkeypatch):
    monkeypatch.setenv("AUTOWAN_POTCAR_LIBRARY", str(potcar_library(tmp_path / "lib")))
    run_dir = _missing_d_deck(tmp_path)
    report = _assess(vasp, FIXTURES / "Si", run_dir)
    changes = report.finding(INVENTORY).prescription.changes
    assert DftChange("nscf", "POTCAR", "Si", "Si_GW") in changes


def test_without_a_library_the_variant_prescription_is_words_only(
    tmp_path, vasp, monkeypatch
):
    monkeypatch.delenv("AUTOWAN_POTCAR_LIBRARY", raising=False)
    run_dir = _missing_d_deck(tmp_path)
    report = _assess(vasp, FIXTURES / "Si", run_dir)
    p = report.finding(INVENTORY).prescription
    assert p.changes == ()
    assert "ZVAL" in p.message and "d" in p.message


# ---------------------------------------------------------------------------
# never alter the parent; the workflow's pre-flight on VASP
# ---------------------------------------------------------------------------


def test_assessing_a_vasp_parent_writes_nothing(tmp_path, vasp):
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "truncated")
    _edit_xml(parent / "vasprun.xml", lambda root: truncate_bands(root, 12))
    watched = sorted(p for d in (parent, run_dir) for p in d.iterdir())
    before = {p: _sha(p) for p in watched}
    _assess(vasp, parent, run_dir)
    assert sorted(p for d in (parent, run_dir) for p in d.iterdir()) == watched
    assert {p: _sha(p) for p in watched} == before


class _NeverBuilds(ConstructionRunner):
    def __init__(self, run_dir):
        super().__init__(run_dir=run_dir)
        self.builds = []

    def run_overlaps(self, seed):  # pragma: no cover - build is overridden
        raise AssertionError("never computes overlaps")

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append(seed)
        return [StepResult(step=f"{seed}:localise", returncode=1, seconds=0.0)]


def test_the_workflow_returns_a_vasp_prescription_and_builds_nothing(tmp_path):
    from autowan.workflow import WorkflowConfig, wannierize

    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    parent = _parent_copy(tmp_path, "truncated")
    _edit_xml(parent / "vasprun.xml", lambda root: truncate_bands(root, 12))
    runner = _NeverBuilds(run_dir)
    result = wannierize(
        run_dir=run_dir,
        parent_xml=parent / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
        declared_moments={},
    )
    assert runner.builds == []
    assert result.status == "DFT prescription returned"
    assert "NBANDS = 16" in result.prescription
    assert result.diagnosis.cause_established is True


# ---------------------------------------------------------------------------
# the twenty real VASP parents (read in place; skipped where absent)
# ---------------------------------------------------------------------------


def _task5_deck(material: str, run_dir: Path) -> Path:
    """The parent's all-channel deck; MnO's automatic window refuses, so its deck
    names the same channels through the explicit-table path (capped window)."""
    run_dir.mkdir()
    xml = TASK5 / material / "vasprun.xml"
    if material == "MnO":
        deck_from_parent(
            xml,
            run_dir,
            material,
            potcar=TASK5 / material / "OUTCAR",
            recipe="hydrogenic",
        )
    else:
        deck_from_parent(xml, run_dir, material, recipe="hydrogenic")
    return run_dir


@pytest.mark.skipif(not TASK5.is_dir(), reason="the 2026-09-29 VASP parents are absent")
@pytest.mark.parametrize("material", TWENTY)
def test_the_campaign_parents_raise_no_blocking_or_limiting_alarm(
    tmp_path, vasp, material
):
    watched = [TASK5 / material / f for f in ("vasprun.xml", "OUTCAR")]
    before = [_sha(p) for p in watched]
    run_dir = _task5_deck(material, tmp_path / "run")
    report = vasp.assess_parent(
        TASK5 / material / "vasprun.xml",
        read_construction(run_dir, material),
        declared_moments=DECLARED[material],
    )
    assert report.blocking == ()
    assert [f.check_id for f in report.alarms if f.severity == LIMITING] == []
    assert report.finding(SPIN).status == ADEQUATE
    assert report.finding(INVENTORY).status == ADEQUATE
    assert report.finding(BAND_PARTITION).status == ADEQUATE
    assert [_sha(p) for p in watched] == before


@pytest.mark.skipif(not TASK5.is_dir(), reason="the 2026-09-29 VASP parents are absent")
def test_the_ferromagnetic_mno_parent_keeps_its_declared_moment(vasp):
    f = vasp.adequacy.read_facts(TASK5 / "MnO" / "vasprun.xml")
    assert f.spin_polarised is True
    assert 4.5 < f.moment_bohr < 5.0
    assert f.requested_moments == {"Mn": 9.0}


# ---------------------------------------------------------------------------
# the overlap step on a spin-polarised parent: VASP 6 numbers its products
# ---------------------------------------------------------------------------
#
# Measured on the real vasp.6.4.1 LWANNIER90 build (2026-09-29): an ISPIN = 2
# run writes wannier90.1.{amn,mmn,eig} and wannier90.2.{...} (mlwf.F: SEED_NAME//'.'//
# str(ISP)), not wannier90.up.* / .dn.*. A runner that looks only for the latter stops
# every spin-polarised construction with "missing VASP overlap products", whatever the
# parent -- which would make the non-spin MnO control's clean arm fail like its degraded one.

VASP6_SPIN_STUB = """#!/usr/bin/env python3
from pathlib import Path
Path("OUTCAR").write_text(" vasp.6.4.1 stub\\n aborting loop because EDIFF is reached\\n")
for isp in (1, 2):
    for ext in ("amn", "mmn", "eig"):
        Path(f"wannier90.{isp}.{ext}").write_text(f"spin {isp} {ext}\\n   16    8    8\\n")
"""


@pytest.mark.parametrize("channel, number", [("up", 1), ("dn", 2)])
def test_the_runner_reads_vasp6_numbered_spin_products(tmp_path, channel, number):
    from autowan.backends.vasp.execution import VaspWannierRunner

    parent = tmp_path / "parent"
    parent.mkdir()
    for name in ("INCAR", "POSCAR", "KPOINTS"):
        shutil.copy(FIXTURES / "Si" / name, parent / name)
    (parent / "POTCAR").write_text("stub\n")
    (parent / "CHGCAR").write_text("stub\n")
    run_dir = write_deck(FIXTURES / "Si", tmp_path / "run")
    stub = tmp_path / "vasp_stub"
    stub.write_text(VASP6_SPIN_STUB)
    stub.chmod(0o755)
    runner = VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent,
        vasp=str(stub),
        wannier90="true",
        mpi_launcher=(),
        spin_channel=channel,
    )
    results = runner.run_overlaps("si")
    assert results[-1].ok, results[-1].stderr_tail
    for ext in ("amn", "mmn", "eig"):
        assert (run_dir / f"si.{ext}").read_text().startswith(f"spin {number} {ext}")
