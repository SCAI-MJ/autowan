"""Spinor (spin-orbit / noncollinear) parents and decks, on small synthetic fixtures.

What a spinor parent changes, and where autowan reads it:

* **Readers.** QE ``noncolin`` (with or without ``spinorbit``) and VASP ``LNONCOLLINEAR``
  give one channel of two-component spinor bands holding one electron each; the parent
  records ``spinor`` and its electron count. A ``vasp_ncl`` PROCAR carries three
  magnetisation tables after each band's charge table, which are kept apart and never
  read as more states.
* **Counting.** With ``spinors = .true.`` every named orbital is two Wannier functions;
  a j-resolved pseudo-atomic orbital is 2j+1, numbered as pw2wannier90 numbers them.
* **Writers.** The deck carries ``spinors = .true.``; the VASP SMV deck doubles its
  functions and excludes one band per semicore electron; the VASP SCDM frontier refuses
  a spinor parent (no published SOC protocol); the VASP overlap step runs ``vasp_ncl``.
* **Gate.** Occupation is one electron per spinor band: a gapped parent's filled bands
  must hold exactly its electrons, and the gate's checks give a spinor construction the
  verdicts they give its scalar twin.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autowan.actions import (
    ActionRejected,
    Stage,
    implied_wannier_count,
    validate_action,
)
from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.outputs import VaspOutputError
from autowan.backends.vasp.outputs import read_parent as read_vasp
from autowan.backends.vasp.outputs import read_procar
from autowan.core_checks import CORE_CHECKS
from autowan.criteria import (
    BAND_INTERPOLATION_ETA2,
    CONDUCTION_BAND_ETA,
    ReferenceBandPath,
    occupation_count,
)
from autowan.deckcheck import check_deck, implied_by_block
from autowan.execution import CalculationError
from autowan.gate import CertificationGate, Construction
from autowan.inputs import (
    InputError,
    deck_spinors,
    exclusion_for_subset,
    pao_excluded,
    pao_layout,
    read_construction,
    read_excluded_bands,
    write_derived,
)
from autowan.interpolation import interpolate
from autowan.parent_adequacy import Channel
from autowan.qe_outputs import ParentCalculation, QeOutputError, read_parent

FIXTURES = Path(__file__).parent / "fixtures"
SRO = FIXTURES / "vasp" / "SrO"
HA = 27.211386245988


# ---------------------------------------------------------------------------
# QE: the noncollinear parent
# ---------------------------------------------------------------------------


def _qe_xml(
    path: Path,
    eigenvalues_ev,
    *,
    noncolin: bool,
    lsda: bool = False,
    nelec: float | None = 2.0,
    fermi_ev: float = 0.0,
) -> Path:
    """A minimal ``data-file-schema.xml``: one atom in a 4 Angstrom cube, one channel or two
    (lsda: the channels concatenated within each ks_energies block, as pw.x writes)."""
    eig = np.asarray(eigenvalues_ev, dtype=float)  # (n_spin, n_k, n_bands)
    nk = eig.shape[1]
    bohr = 0.529177210903
    a = 4.0 / bohr
    ks = []
    for k in range(nk):
        values = np.concatenate([eig[s, k] for s in range(eig.shape[0])]) / HA
        occ = (values * HA <= fermi_ev).astype(float)
        ks.append(
            f"<ks_energies><k_point weight='1'>{k / nk} 0 0</k_point>"
            f"<eigenvalues size='{values.size}'>{' '.join(map(str, values))}</eigenvalues>"
            f"<occupations>{' '.join(map(str, occ))}</occupations></ks_energies>"
        )
    nelec_tag = "" if nelec is None else f"<nelec>{nelec}</nelec>"
    text = f"""<?xml version="1.0"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
 <input><control_variables><calculation>nscf</calculation></control_variables></input>
 <output>
  <atomic_structure nat="1" alat="{a}" num_of_atomic_wfc="4">
   <atomic_positions><atom name="Pt" index="1">0 0 0</atom></atomic_positions>
   <cell><a1>{a} 0 0</a1><a2>0 {a} 0</a2><a3>0 0 {a}</a3></cell>
  </atomic_structure>
  <symmetries><nsym>1</nsym></symmetries>
  <band_structure>
   <lsda>{str(lsda).lower()}</lsda><noncolin>{str(noncolin).lower()}</noncolin>
   <spinorbit>{str(noncolin).lower()}</spinorbit>
   <nbnd>{eig.shape[2]}</nbnd>{nelec_tag}
   <fermi_energy>{fermi_ev / HA}</fermi_energy>
   {''.join(ks)}
  </band_structure>
 </output>
</qes:espresso>
"""
    path.write_text(text)
    return path


# Two Kramers pairs on a four-point line: a filled pair at -2 eV, an empty one at +1 eV.
KRAMERS_INSULATOR = np.array(
    [[[-2.0 - 0.1 * k, -2.0 - 0.1 * k, 1.0 + 0.1 * k, 1.0 + 0.1 * k] for k in range(4)]]
)


def test_a_qe_noncollinear_parent_is_one_channel_of_one_electron_bands(tmp_path):
    parent = read_parent(_qe_xml(tmp_path / "x.xml", KRAMERS_INSULATOR, noncolin=True))
    assert parent.spinor and parent.n_spin == 1 and parent.n_bands == 4
    assert parent.electrons_per_band == 1
    assert parent.n_electrons == 2.0
    assert not parent.is_metallic() and parent.occupied_bands() == 2
    # two filled spinor bands hold the two electrons; read as scalar bands they would hold 4
    assert parent.filled_band_electrons() == 2.0
    assert parent.occupation_consistent() is True


def test_a_scalar_parent_keeps_two_electrons_per_band(tmp_path):
    scalar = KRAMERS_INSULATOR[:, :, ::2]  # one band of each pair, two electrons each
    parent = read_parent(_qe_xml(tmp_path / "x.xml", scalar, noncolin=False, nelec=2.0))
    assert not parent.spinor and parent.electrons_per_band == 2
    assert parent.occupation_consistent() is True


def test_a_collinear_parent_holds_one_electron_per_band_per_channel(tmp_path):
    up = KRAMERS_INSULATOR[0, :, ::2]
    eig = np.array([up, up])
    parent = read_parent(
        _qe_xml(tmp_path / "x.xml", eig, noncolin=False, lsda=True, nelec=2.0)
    )
    assert parent.n_spin == 2 and not parent.spinor
    assert parent.electrons_per_band == 1
    assert parent.occupation_consistent() is True


def test_lsda_and_noncolin_together_are_refused(tmp_path):
    with pytest.raises(QeOutputError, match="one channel"):
        read_parent(
            _qe_xml(tmp_path / "x.xml", KRAMERS_INSULATOR, noncolin=True, lsda=True)
        )


def test_a_parent_without_an_electron_count_cannot_be_asked(tmp_path):
    parent = read_parent(
        _qe_xml(tmp_path / "x.xml", KRAMERS_INSULATOR, noncolin=True, nelec=None)
    )
    assert parent.n_electrons is None and parent.occupation_consistent() is None


def test_fully_relativistic_upf_channels_count_two_j_plus_one(tmp_path):
    from autowan.backends.qe import _channels

    upf = tmp_path / "Pt.upf"
    upf.write_text(
        """<UPF version="2.0.1">
<PP_CHI.1 type="real" label="5S" l="0"/>
<PP_CHI.2 type="real" label="5P" l="1"/>
<PP_CHI.3 type="real" label="5P" l="1"/>
<PP_CHI.4 type="real" label="5D" l="2"/>
<PP_CHI.5 type="real" label="5D" l="2"/>
<PP_CHI.6 type="real" label="6S" l="0"/>
<PP_SPIN_ORB>
<PP_RELWFC.1  index="1"  lchi="0" jchi="0.5" nn="1"/>
<PP_RELWFC.2  index="2"  lchi="1" jchi="1.5" nn="2"/>
<PP_RELWFC.3  index="3"  lchi="1" jchi="0.5" nn="2"/>
<PP_RELWFC.4  index="4"  lchi="2" jchi="2.5" nn="3"/>
<PP_RELWFC.5  index="5"  lchi="2" jchi="1.5" nn="3"/>
<PP_RELWFC.6  index="6"  lchi="0" jchi="0.5" nn="4"/>
</PP_SPIN_ORB>
</UPF>
"""
    )
    channels = _channels(upf)
    # pw.x's num_of_atomic_wfc for this file: 2 + 4 + 2 + 6 + 4 + 2 = 20
    assert sum(c.functions for c in channels) == 20
    assert sum(c.spinor_functions() for c in channels) == 20
    # a scalar channel in a spinor run is both spin components
    assert Channel("3d", 2).spinor_functions() == 10 and Channel("3d", 2).functions == 5


# ---------------------------------------------------------------------------
# VASP: the vasp_ncl parent
# ---------------------------------------------------------------------------


def _spinorised_sro(tmp_path: Path) -> Path:
    """The SrO fixture rewritten as a vasp_ncl parent: every band a Kramers pair.

    LNONCOLLINEAR = T, NBANDS doubled, each eigenvalue listed twice (occupation 0..1 per
    spinor band, as vasp_ncl writes it), NELECT unchanged; the PROCAR carries each band
    twice with three zero magnetisation tables after its charge table.
    """
    parent = tmp_path / "parent"
    shutil.copytree(SRO, parent)
    (parent / "EIGENVAL").unlink()
    tree = ET.parse(parent / "vasprun.xml")
    root = tree.getroot()
    for item in root.iter("i"):
        if item.get("name") == "LNONCOLLINEAR":
            item.text = " T  "
        if item.get("name") == "NBANDS":
            item.text = f"    {2 * int(item.text)}"
    for calc in root.findall("calculation"):
        for kset in calc.findall("eigenvalues/array/set/set/set"):
            rows = list(kset.findall("r"))
            for r in rows:
                kset.remove(r)
            for r in rows:
                for _ in range(2):
                    copy = ET.SubElement(kset, "r")
                    copy.text = r.text
    tree.write(parent / "vasprun.xml")

    proj = read_procar(SRO / "PROCAR")
    nb, ni = proj.n_bands, proj.n_ions
    head = " ".join(["ion", *proj.orbitals, "tot"])
    lines = [
        "PROCAR lm decomposed",
        f"# of k-points:  {proj.n_kpoints}         # of bands:  {2 * nb}         "
        f"# of ions:  {ni}",
    ]
    state = 0
    for k in range(proj.n_kpoints):
        lines += [
            "",
            f" k-point {k + 1} :    0 0 0     weight = {proj.kpoint_weights[k]}",
            "",
        ]
        for b in range(nb):
            w = proj.weights[state]
            e = proj.energies_ev[state]
            state += 1
            for copy in range(2):
                lines += [
                    "",
                    f"band {2 * b + copy + 1} # energy {e:.8f} # occ.  1.0",
                    "",
                    head,
                ]
                for table in range(4):
                    values = w if table == 0 else np.zeros_like(w)
                    for ion in range(ni):
                        row = values[ion]
                        lines.append(
                            f"{ion + 1:5d} "
                            + " ".join(f"{x:6.3f}" for x in row)
                            + f" {row.sum():6.3f}"
                        )
                    tot = values.sum(axis=0)
                    lines.append(
                        "tot   "
                        + " ".join(f"{x:6.3f}" for x in tot)
                        + f" {tot.sum():6.3f}"
                    )
    (parent / "PROCAR").write_text("\n".join(lines) + "\n")
    return parent


def test_a_vasp_ncl_parent_is_one_channel_of_spinor_bands(tmp_path):
    parent_dir = _spinorised_sro(tmp_path)
    scalar = read_vasp(SRO / "vasprun.xml")
    spinor = read_vasp(parent_dir / "vasprun.xml")
    assert not scalar.spinor and scalar.electrons_per_band == 2
    assert spinor.spinor and spinor.n_spin == 1 and spinor.electrons_per_band == 1
    assert spinor.n_bands == 2 * scalar.n_bands
    assert spinor.n_electrons == scalar.n_electrons == 16.0
    # both read their own electrons into the same filled manifold
    assert spinor.filled_band_electrons() == scalar.filled_band_electrons() == 16.0
    assert spinor.occupation_consistent() and scalar.occupation_consistent()


def test_a_vasp_ncl_procar_keeps_its_magnetisation_out_of_the_states(tmp_path):
    parent_dir = _spinorised_sro(tmp_path)
    scalar = read_procar(SRO / "PROCAR")
    ncl = read_procar(parent_dir / "PROCAR")
    assert ncl.noncollinear and not scalar.noncollinear
    assert ncl.n_spin == 1 and ncl.n_bands == 2 * scalar.n_bands
    assert ncl.weights.shape[0] == ncl.n_kpoints * ncl.n_bands
    assert ncl.magnetisation.shape == (ncl.weights.shape[0], 3, *ncl.weights.shape[1:])
    np.testing.assert_allclose(
        ncl.projectability[::2], scalar.projectability, atol=2e-3
    )


def test_a_single_ion_procar_without_tot_rows_is_read(tmp_path):
    procar = tmp_path / "PROCAR"
    rows = []
    for b, (e, s, p) in enumerate(((-50.0, 0.98, 0.0), (-30.0, 0.0, 0.97))):
        rows += [
            f"band {b + 1} # energy {e} # occ. 1.0",
            "ion      s     py     pz     px    tot",
            f"    1  {s}  {p / 3:.3f}  {p / 3:.3f}  {p / 3:.3f}  {s + p:.3f}",
            "    1  0.000  0.000  0.000  0.000  0.000",
            "    1  0.000  0.000  0.000  0.000  0.000",
            "    1  0.100  0.000  0.000  0.000  0.100",
        ]
    procar.write_text(
        "PROCAR lm decomposed\n# of k-points:    1         # of bands:    2         "
        "# of ions:    1\n\n k-point     1 :    0 0 0     weight = 1.0\n\n"
        + "\n".join(rows)
        + "\n"
    )
    proj = read_procar(procar)
    assert proj.noncollinear and proj.n_bands == 2 and proj.weights.shape == (2, 1, 4)
    np.testing.assert_allclose(proj.projectability, [0.98, 0.97], atol=2e-3)
    assert proj.magnetisation[0, 2, 0, 0] == pytest.approx(0.1)


def test_a_vasp_ncl_with_two_eigenvalue_channels_is_refused(tmp_path):
    parent_dir = _spinorised_sro(tmp_path)
    tree = ET.parse(parent_dir / "vasprun.xml")
    for spins in tree.getroot().findall("calculation/eigenvalues/array/set"):
        first = spins.find("set")
        twin = ET.SubElement(spins, "set", {"comment": "spin 2"})
        for k in first:
            twin.append(k)
    tree.write(parent_dir / "vasprun.xml")
    with pytest.raises(VaspOutputError, match="noncollinear"):
        read_vasp(parent_dir / "vasprun.xml")


# ---------------------------------------------------------------------------
# counting: projections, PAOs, actions
# ---------------------------------------------------------------------------


def test_a_spinor_projector_set_implies_two_functions_per_orbital():
    projectors = [{"species": "Pt", "orbitals": ("s", "d")}]
    assert implied_wannier_count(projectors, {"Pt": 1}) == 6
    assert implied_wannier_count(projectors, {"Pt": 1}, spinors=True) == 12


@pytest.mark.parametrize(
    ("line", "spinors", "count"),
    [
        ("Pt:s;d", False, 6),
        ("Pt:s;d", True, 12),
        ("Fe:d(u)", True, 5),
        ("Fe:d(d)[0,0,1]", True, 5),
        ("Fe:d(u,d)[1,0,0]", True, 10),
        ("Fe:l=2,mr=1,4", False, 2),
        ("Fe:l=2,mr=1,4", True, 4),
        ("Fe:d(u)", False, None),  # a spin qualifier without spinors is no deck
    ],
)
def test_a_projections_block_counts_spinor_components(line, spinors, count):
    assert implied_by_block([line], {"Pt": 1, "Fe": 1}, spinors) == count


def _fr_projector_dir(root: Path, with_j: bool = True) -> Path:
    """A PseudoDojo-FR-like external projector set for Pt (the shipped labels and l, j)."""
    directory = root / "external_projectors"
    directory.mkdir()
    ls = [0, 0, 1, 1, 2, 2, 1, 1]
    js = [0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 0.5, 1.5]
    labels = ["5s", "6s", "5p", "5p", "5d", "5d", "6p", "6p"]
    lines = ["3 8", " ".join(map(str, ls))]
    if with_j:
        lines.append(" ".join(map(str, js)))
    lines += [" ".join(["0.1", "0.2"] + ["0.0"] * 8) for _ in range(3)]
    (directory / "Pt.dat").write_text("\n".join(lines) + "\n")
    table = [
        {"label": lab, "l": l, **({"j": j} if with_j else {})}
        for lab, l, j in zip(labels, ls, js)
    ]
    import json

    (directory / "projectors.json").write_text(json.dumps({"Pt": table}))
    return directory


def _pdwf_deck(root: Path, spinors: bool, exclude=(1, 2, 5, 6, 7, 8, 9, 10)) -> Path:
    win = [
        "num_wann = 18",
        "num_bands = 64",
        "exclude_bands = 1-8",
        "auto_projections = .true.",
        "dis_froz_proj = .true.",
        "dis_proj_min = 0.01",
        "dis_proj_max = 0.95",
        "dis_froz_max = 25.0",
        "mp_grid = 2 2 2",
    ]
    if spinors:
        win.append("spinors = .true.")
    win += ["begin atoms_frac", "Pt 0 0 0", "end atoms_frac"]
    (root / "pt.win").write_text("\n".join(win) + "\n")
    pw = ["&inputpp", "  outdir = './out/'", "  prefix = 'aiida'", "  seedname = 'pt'"]
    pw += [
        "  atom_proj = .true.",
        "  atom_proj_ext = .true.",
        "  atom_proj_dir = 'external_projectors/'",
    ]
    pw += [f"  atom_proj_exclude({i}) = {v}" for i, v in enumerate(exclude, start=1)]
    pw.append("/")
    (root / "pt.pw2wan").write_text("\n".join(pw) + "\n")
    return root


def test_j_resolved_paos_are_numbered_as_pw2wannier90_numbers_them(tmp_path):
    _fr_projector_dir(tmp_path)
    _pdwf_deck(tmp_path, spinors=True)
    win, pw = (tmp_path / "pt.win").read_text(), (tmp_path / "pt.pw2wan").read_text()
    layout = pao_layout(tmp_path, win, pw)
    # 2 + 2 + 2 + 4 + 4 + 6 + 2 + 4 = 26 states, m_j running within each (l, j)
    assert len(layout) == 26
    assert [(s.label, s.j) for s in layout[4:10]] == [("5p", 0.5)] * 2 + [
        ("5p", 1.5)
    ] * 4
    # aiida-wannier90-workflows' semicore exclusion for Pt (5s, 5p) is exactly these
    excluded = exclusion_for_subset(
        layout, (), [{"species": "Pt", "orbitals": ["6s", "5d", "6p"]}]
    )
    assert excluded == (1, 2, 5, 6, 7, 8, 9, 10)
    assert len(layout) - len(excluded) == 18


def test_a_spinor_deck_without_j_doubles_each_orbital_up_block_first(tmp_path):
    _fr_projector_dir(tmp_path, with_j=False)
    _pdwf_deck(tmp_path, spinors=True)
    layout = pao_layout(
        tmp_path,
        (tmp_path / "pt.win").read_text(),
        (tmp_path / "pt.pw2wan").read_text(),
    )
    assert len(layout) == 2 * (1 + 1 + 3 + 3 + 5 + 5 + 3 + 3)
    assert [s.spin for s in layout[:2]] == ["up", "down"]  # 5s up, then 5s down
    assert [s.spin for s in layout[2:4]] == ["up", "down"]  # 6s


def test_a_scalar_deck_ignores_any_j_line(tmp_path):
    _fr_projector_dir(tmp_path)
    _pdwf_deck(tmp_path, spinors=False, exclude=(1,))
    layout = pao_layout(
        tmp_path,
        (tmp_path / "pt.win").read_text(),
        (tmp_path / "pt.pw2wan").read_text(),
    )
    assert len(layout) == 24 and all(s.j is None for s in layout)


def test_a_pao_subset_on_a_spinor_pdwf_deck_keeps_its_spinors_and_counts_2j_plus_1(
    tmp_path,
):
    _fr_projector_dir(tmp_path)
    _pdwf_deck(tmp_path, spinors=True)
    source = read_construction(tmp_path, "pt")
    derived = write_derived(
        source,
        tmp_path,
        "pt_r1",
        {"pao_subset": [{"species": "Pt", "orbitals": ["5d", "6s"]}]},
    )
    assert deck_spinors(derived.win.read_text())
    assert int(derived.values["num_wann"]) == 12  # 6s (2) + 5d (4 + 6)
    assert pao_excluded(derived.pw2wan.read_text()) == {
        1,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        21,
        22,
        23,
        24,
        25,
        26,
    }


def test_named_projectors_on_a_spinor_smv_deck_set_a_doubled_count(tmp_path):
    (tmp_path / "pt.win").write_text(
        "num_wann = 12\nnum_bands = 64\nspinors = .true.\ndis_froz_max = 25.0\n"
        "begin projections\nPt:s;d\nend projections\n"
        "begin atoms_frac\nPt 0 0 0\nend atoms_frac\n"
    )
    (tmp_path / "pt.pw2wan").write_text("&inputpp\n  seedname = 'pt'\n/\n")
    source = read_construction(tmp_path, "pt")
    derived = write_derived(
        source,
        tmp_path,
        "pt_r1",
        {"projectors": [{"species": "Pt", "orbitals": ["s", "p", "d"]}]},
        species_counts={"Pt": 1},
    )
    assert int(derived.values["num_wann"]) == 18


def _spinor_parent(n_bands=72, spinor=True):
    e = np.sort(np.random.default_rng(0).uniform(-10, 40, (1, 8, n_bands)), axis=-1)
    # The eight lowest bands are the semicore block the decks below exclude: isolated
    # far below the rest, as Pt 5s 5p is (38 eV), so the gate would honour the cut and
    # the deck check judges only what each test is about.
    e[..., :8] -= 60.0
    return SimpleNamespace(
        eigenvalues_ev=e,
        n_bands=n_bands,
        species=("Pt",),
        spinor=spinor,
        fermi_energy_ev=23.0,
    )


def test_the_deck_check_refuses_a_spinor_flag_the_parent_does_not_share(tmp_path):
    (tmp_path / "pt.win").write_text(
        "num_wann = 12\nnum_bands = 64\nexclude_bands = 1-8\n"
        "begin projections\nPt:s;d\nend projections\n"
        "begin atoms_frac\nPt 0 0 0\nend atoms_frac\n"
    )
    deck = read_construction(tmp_path, "pt")
    refused = check_deck(deck, parent=_spinor_parent(), code="vasp")
    assert not refused.ok and "spinors = .true." in refused.reason()
    # the projections imply 6 scalar functions: also a count disagreement
    assert "imply 6" in refused.reason()
    text = (
        (tmp_path / "pt.win")
        .read_text()
        .replace("num_wann", "spinors = .true.\nnum_wann")
    )
    (tmp_path / "pt.win").write_text(text)
    accepted = check_deck(
        read_construction(tmp_path, "pt"), parent=_spinor_parent(), code="vasp"
    )
    assert accepted.ok, accepted.reason()
    assert accepted.facts["projectors_imply"] == 12
    scalar = check_deck(
        read_construction(tmp_path, "pt"),
        parent=_spinor_parent(spinor=False),
        code="vasp",
    )
    assert not scalar.ok and "not spinors" in scalar.reason()


def test_an_action_naming_projectors_on_a_spinor_parent_counts_both_components():
    payload = {
        "action": "modify_parameters",
        "diagnosis": "d",
        "evidence": ["x"],
        "expected_effect": "e",
        "confidence": 0.5,
        "changes": [
            {
                "parameter": "projectors",
                "new": [{"species": "Pt", "orbitals": ["s", "p", "d"]}],
                "reason": "r",
            }
        ],
    }
    kwargs = dict(
        stage=Stage.REPAIR,
        evaluated_check_ids=["x"],
        allowed_species=["Pt"],
        species_counts={"Pt": 1},
        band_count=16,
        method="smv",
        code="vasp",
    )
    validate_action(payload, **kwargs)  # 9 scalar functions from 16 bands
    with pytest.raises(ActionRejected, match="imply 18"):
        validate_action(payload, spinors=True, **kwargs)


# ---------------------------------------------------------------------------
# VASP writers and the overlap step
# ---------------------------------------------------------------------------


def test_the_vasp_smv_deck_of_a_spinor_parent_doubles_functions_and_excludes_by_electron(
    tmp_path,
):
    parent_dir = _spinorised_sro(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    deck = vasp_inputs.semicore_excluded_deck(
        parent_dir / "vasprun.xml", run_dir, "SrO"
    )
    text = deck.win.read_text()
    assert deck_spinors(text)
    assert deck.values["projectors"] == "Sr:s;d;O:p"
    assert deck.values["num_wann"] == "18"  # 2 x (Sr s+d 6 + O p 3)
    excluded = read_excluded_bands(deck.win)
    # 10 semicore electrons (Sr 4s2 4p6, O 2s2): ten spinor bands, not five
    assert excluded[:10] == tuple(range(1, 11)) and 11 not in excluded
    assert int(deck.values["num_bands"]) + len(excluded) == 64
    assert "semicore bands excluded: 10 (PROCAR channel character" in text
    parent = read_vasp(parent_dir / "vasprun.xml")
    check = check_deck(deck, parent=parent, code="vasp")
    assert check.ok, check.reason()


def test_the_vasp_scdm_frontier_refuses_a_spinor_parent(tmp_path):
    parent_dir = _spinorised_sro(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(InputError, match="no spin-orbit"):
        vasp_inputs.scdm_frontier_deck(parent_dir / "vasprun.xml", run_dir, "SrO")
    with pytest.raises(InputError, match="no VASP frontier"):
        vasp_inputs.deck_from_parent(parent_dir / "vasprun.xml", run_dir, "SrO")


def test_the_vasp_starting_scdm_deck_is_unavailable_for_a_spinor_parent(tmp_path):
    from autowan.methods import DeckContext, StartingDeckUnavailable, starting_deck

    parent_dir = _spinorised_sro(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    smv = vasp_inputs.semicore_excluded_deck(parent_dir / "vasprun.xml", run_dir, "SrO")
    ctx = DeckContext(
        code="vasp",
        run_dir=run_dir,
        parent=read_vasp(parent_dir / "vasprun.xml"),
        parent_source=parent_dir / "vasprun.xml",
        frontier=smv,
    )
    with pytest.raises(StartingDeckUnavailable, match="no spin-orbit"):
        starting_deck("scdm", ctx)
    assert deck_spinors(starting_deck("smv", ctx).win_text)


def test_the_vasp_overlap_step_runs_vasp_ncl_for_a_spinor_parent(tmp_path):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("vasp_std", "vasp_ncl"):
        (bins / name).write_text("#!/bin/sh\n")
    runner = VaspWannierRunner(
        run_dir=tmp_path, parent_dir=parent_dir, vasp=str(bins / "vasp_std")
    )
    assert runner.binary_for({"LSORBIT": ".TRUE."}) == str(bins / "vasp_ncl")
    assert runner.binary_for({"LNONCOLLINEAR": "T"}) == str(bins / "vasp_ncl")
    assert runner.binary_for({"ISPIN": "2"}) == str(bins / "vasp_std")
    explicit = VaspWannierRunner(
        run_dir=tmp_path, parent_dir=parent_dir, vasp="vasp_std", vasp_ncl="/x/vasp_ncl"
    )
    assert explicit.binary_for({"LSORBIT": "T"}) == "/x/vasp_ncl"
    (bins / "vasp_ncl").unlink()
    lone = VaspWannierRunner(
        run_dir=tmp_path, parent_dir=parent_dir, vasp=str(bins / "vasp_std")
    )
    with pytest.raises(CalculationError, match="vasp_ncl"):
        lone.binary_for({"LSORBIT": ".TRUE."})


def test_the_vasp_overlap_step_launches_the_ncl_binary(tmp_path, monkeypatch):
    parent_dir = _spinorised_sro(tmp_path)
    (parent_dir / "INCAR").write_text(
        (parent_dir / "INCAR").read_text()
        + "LSORBIT = .TRUE.\nLNONCOLLINEAR = .TRUE.\n"
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vasp_inputs.semicore_excluded_deck(parent_dir / "vasprun.xml", run_dir, "SrO")
    for name in ("CHGCAR", "POTCAR"):
        (parent_dir / name).write_text("stub\n")
    launched = {}

    def fake_run(self, step, command, **kw):
        from autowan.execution import StepResult

        launched["command"] = list(command)
        stage = kw["cwd"]
        launched["win"] = (stage / "wannier90.win").read_text()
        return StepResult(step, 0, 0.0)

    monkeypatch.setattr(VaspWannierRunner, "_run", fake_run)
    runner = VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent_dir,
        vasp_ncl="/opt/vasp_ncl",
        mpi_launcher=(),
    )
    runner.run_overlaps("SrO")
    assert launched["command"] == ["/opt/vasp_ncl"]
    assert "spinors = .true." in launched["win"]


# ---------------------------------------------------------------------------
# the gate on spinor bands
# ---------------------------------------------------------------------------

A = 3.0
EF = 5.0
PATH = np.array([[s, 0.0, 0.0] for s in np.linspace(0.0, 0.5, 21)])
MESH = np.array([[i / 8, 0.0, 0.0] for i in range(8)])
#: A scalar insulator: valence centred at E_F - 2, conduction at E_F + 2, each 1 eV wide.
SCALAR = {0: np.diag([EF - 2.0, EF + 2.0]), 1: np.diag([0.25, -0.25])}
#: Its spinor twin: every band a Kramers pair, four functions.
SPINOR = {
    0: np.diag([EF - 2.0] * 2 + [EF + 2.0] * 2),
    1: np.diag([0.25] * 2 + [-0.25] * 2),
}


def _hr(blocks):
    full = {}
    for n, block in blocks.items():
        full[n] = np.asarray(block, dtype=complex)
        if n:
            full[-n] = np.conjugate(full[n]).T
    ns = sorted(full)
    return (
        np.array([[n, 0, 0] for n in ns]),
        np.ones(len(ns), dtype=int),
        np.array([full[n] for n in ns]),
    )


def _construction(blocks, *, spinor, n_electrons, dft_shift=0.0):
    rvec, deg, h = _hr(blocks)
    path = interpolate(h, rvec, deg, PATH) + dft_shift
    mesh = interpolate(h, rvec, deg, MESH)
    parent = ParentCalculation(
        eigenvalues_ev=mesh[None],
        occupations=(mesh[None] <= EF).astype(float),
        kpoints_cart=MESH,
        fermi_energy_ev=EF,
        lattice_ang=A * np.eye(3),
        alat_ang=A,
        species=("Pt",),
        positions_frac=np.zeros((1, 3)),
        n_symmetries=1,
        n_atomic_wfc=None,
        converged=True,
        calculation="nscf",
        spinor=spinor,
        n_electrons=n_electrons,
    )
    nw = h.shape[1]
    wannier = SimpleNamespace(
        num_wann=nw,
        hr_matrix=h,
        hr_rvectors=rvec,
        hr_degeneracies=deg,
        hr_ws_shifts=None,
        centres_ang=np.zeros((nw, 3)),
        spreads_ang2=np.full(nw, 1.0),
        disentanglement_used=False,
        disentanglement=None,
        wannierisation=None,
    )
    return Construction(
        "synthetic",
        parent=parent,
        wannier=wannier,
        material_class="gapped",
        inputs={"spin_index": 0, "mp_grid": (8, 8, 8), "num_wann": nw},
        reference_path=ReferenceBandPath(PATH, path, EF, EF, source="synthetic"),
    )


def _verdict(check, c):
    return CertificationGate([check], profile_name="t").run(c).verdicts[0]


def test_a_spinor_insulator_holds_one_electron_per_filled_band():
    c = _construction(SPINOR, spinor=True, n_electrons=2.0)
    count = occupation_count(c.parent, EF, crossing=0)
    assert count["electrons_per_band"] == 1 and count["filled_bands_mesh"] == 2
    assert count["occupation_consistent"] is True
    # the same spinor bands read as two-electron bands would hold four electrons
    misread = _construction(SPINOR, spinor=False, n_electrons=2.0)
    assert (
        occupation_count(misread.parent, EF, crossing=0)["occupation_consistent"]
        is False
    )


def test_the_conduction_band_criterion_scores_a_spinor_insulator_like_its_scalar_twin():
    shift = 0.012
    scalar = _verdict(
        CONDUCTION_BAND_ETA,
        _construction(SCALAR, spinor=False, n_electrons=2.0, dft_shift=shift),
    )
    spinor = _verdict(
        CONDUCTION_BAND_ETA,
        _construction(SPINOR, spinor=True, n_electrons=2.0, dft_shift=shift),
    )
    assert scalar.applies and spinor.applies and scalar.passed and spinor.passed
    assert spinor.metric_value == pytest.approx(scalar.metric_value)
    assert spinor.record["cbm_ev"] == pytest.approx(scalar.record["cbm_ev"])
    assert spinor.record["electrons_per_band"] == 1
    assert scalar.record["electrons_per_band"] == 2


def test_criterion_one_on_spinor_bands_equals_its_scalar_twin():
    shift = 0.015
    scalar = _verdict(
        BAND_INTERPOLATION_ETA2,
        _construction(SCALAR, spinor=False, n_electrons=2.0, dft_shift=shift),
    )
    spinor = _verdict(
        BAND_INTERPOLATION_ETA2,
        _construction(SPINOR, spinor=True, n_electrons=2.0, dft_shift=shift),
    )
    assert spinor.metric_value == pytest.approx(scalar.metric_value)
    assert spinor.metric_value == pytest.approx(15.0, rel=1e-6) and spinor.passed
    assert spinor.record["occupation_consistent"] is True


def test_spinor_bands_misread_as_two_electron_bands_leave_the_cbm_unplaced():
    c = _construction(SPINOR, spinor=False, n_electrons=2.0)
    report = CertificationGate([CONDUCTION_BAND_ETA], profile_name="t").run(c)
    assert report.non_evaluable_check_ids == ("conduction_band_eta",)
    assert "4 electrons at 2 per band" in report.verdicts[0].note


def test_the_core_checks_give_a_spinor_construction_its_scalar_twins_verdicts():
    scalar = CertificationGate(CORE_CHECKS, profile_name="t").run(
        _construction(SCALAR, spinor=False, n_electrons=2.0)
    )
    spinor = CertificationGate(CORE_CHECKS, profile_name="t").run(
        _construction(SPINOR, spinor=True, n_electrons=2.0)
    )
    for a, b in zip(scalar.verdicts, spinor.verdicts):
        assert (a.check_id, a.evaluable, a.applies, a.passed) == (
            b.check_id,
            b.evaluable,
            b.applies,
            b.passed,
        ), a.check_id
        if (
            a.evaluable
            and a.metric_value is not None
            and "fraction" in (a.metric_units or "")
        ):
            assert b.metric_value == pytest.approx(a.metric_value), a.check_id


def test_the_qe_adequacy_facts_of_a_spinor_parent_are_noncollinear_spinor_counts(
    tmp_path,
):
    from autowan.backends.qe import QeAdequacyPartition

    xml = _qe_xml(tmp_path / "x.xml", KRAMERS_INSULATOR, noncolin=True)
    facts = QeAdequacyPartition().read_facts(xml)
    assert facts.noncollinear and not facts.spin_polarised
    assert facts.n_spin == 1 and facts.moment_bohr is None
    # pw.x's own atomic-wavefunction count of a noncollinear run is already in spinor states
    assert facts.projector_functions == 4


# ---------------------------------------------------------------------------
# an exclusion of semicore below and headroom above (the VASP SMV deck's shape)
# ---------------------------------------------------------------------------


def _analytic():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analytic_constructions", FIXTURES / "analytic" / "analytic_constructions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semicore_below_and_unoccupied_headroom_above_are_both_honoured():
    from autowan.gate import parent_bands_for_check

    ac = _analytic()
    parent = ac.make_parent(
        [[ac.flat(ac.SEMICORE_EV), ac.s_band(), ac.d_band(), ac.flat(20.0)]]
    )
    model = ac.hr_diagonal([(-4.0, -1.0), (-1.25, -0.875)])  # s and d, exactly
    c = ac.construction(parent, model, inputs={"exclude_bands": (1, 4)})
    bands, note = parent_bands_for_check(c, 0)
    assert bands.shape[1] == 2
    assert "bands 1 excluded" in note and "headroom bands 4" in note
    report = CertificationGate(CORE_CHECKS, profile_name="t").run(c)
    occupied = next(
        v for v in report.verdicts if v.check_id == "occupied_manifold_is_represented"
    )
    assert occupied.passed and occupied.metric_value == 0.0


def test_a_top_exclusion_holding_occupied_states_is_not_headroom():
    from autowan.gate import parent_bands_for_check

    ac = _analytic()
    # the top band dips below E_F: excluding it would hide an occupied state
    parent = ac.make_parent(
        [[ac.flat(ac.SEMICORE_EV), ac.s_band(), ac.d_band(), ac.d_band() + 0.6]]
    )
    c = ac.construction(
        parent,
        ac.hr_diagonal([(-4.0, -1.0), (-1.25, -0.875)]),
        inputs={"exclude_bands": (1, 4)},
    )
    bands, note = parent_bands_for_check(c, 0)
    assert bands.shape[1] == 4 and "not honoured" in note
