"""The VASP first deck names every channel its POTCARs carry, and freezes by the parent's bands.

The automatic VASP deck used to name only the channels some state below E_F + 2 eV
carried at least 0.1 of (``projector_supply``). On SrO -- a 3 eV-gap charge-transfer
insulator -- that window closes below the empty Sr 4d/5s manifold, so the deck named
eight valence functions for eight occupied bands and the model had nothing above the
Fermi level: the gate refused it on the frontier and on band distance, correctly.

Two changes, pinned here (2026-09-14):

1. **Channels.** The deck names every angular-momentum channel each species' POTCAR
   carries a projector for -- SrO: Sr s, p, d and O s, p, thirteen functions; silicon:
   Si s, p, eight -- the VASP counterpart of taking every pseudo-atomic orbital a
   pseudopotential ships. A POTCAR's local channel (``TYP 7`` in its description
   table, e.g. the d line of O and Si) carries no projector and is not a channel. The
   occupied-channel supply may still be computed as a diagnostic; it no longer decides
   the deck, and the deck no longer needs a PROCAR at all. Since 2026-10-01 a channel
   built at two occupied shells is named once per shell (Sr_sv s: 4s and 5s), so SrO
   takes fourteen functions, not thirteen
   (``test_vasp_semicore_aware_function_count.py``).

2. **Frozen window.** ``dis_froz_max`` is no longer E_F + a margin shared in spirit with
   the channel selection. It is read off the parent's own bands: 2 eV above the highest
   occupied state (above E_F for a metallic parent), capped just below the lowest
   energy at which any k-point would hold more states than the deck has functions, so
   Wannier90 is never handed more frozen states than targets. Anchoring on the highest
   occupied state rather than on the Fermi level matters for VASP: with Gaussian
   smearing it places E_F inside an insulator's gap (0.39 eV above the SrO valence
   maximum on the fixture), and the Quantum ESPRESSO first deck this is compared with
   freezes to its highest occupied level + 2 eV.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp import outputs as vasp_outputs
from autowan.backends.vasp.inputs import (
    FROZEN_WINDOW_ABOVE_OCCUPIED_EV,
    deck_from_parent,
    frozen_window_top,
)
from autowan.backends.vasp.outputs import read_parent
from autowan.inputs import InputError
from autowan.qe_outputs import ParentCalculation

FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
PARENT_FILES = ("vasprun.xml", "OUTCAR", "PROCAR", "EIGENVAL")


def _parent(tmp_path: Path, material: str, files=PARENT_FILES) -> Path:
    """A copy of a fixture parent in its own directory (the deck is written beside it)."""
    for name in files:
        shutil.copy(FIXTURES / material / name, tmp_path / name)
    return tmp_path / "vasprun.xml"


def _projection_lines(win: Path) -> list[str]:
    lines = [line.strip() for line in win.read_text().splitlines()]
    start = next(i for i, l in enumerate(lines) if l.lower() == "begin projections")
    end = next(i for i, l in enumerate(lines) if l.lower() == "end projections")
    return [l for l in lines[start + 1 : end] if l]


def _number(value) -> float:
    return float(str(value).lower().replace("d", "e"))


# ---------------------------------------------------------------------------
# the channel set: every channel the POTCARs carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material, lines, count",
    [
        # Sr_sv's s channel carries two occupied shells (4s semicore, 5s valence):
        # one function each since the semicore-aware count (2026-09-30).
        ("SrO", ["Sr:s;p;d", "Sr:s:r=2", "O:s;p"], 14),
        ("Si", ["Si:s;p"], 8),
    ],
)
def test_the_first_deck_names_every_channel_the_potcars_carry(
    tmp_path, material, lines, count
):
    xml = _parent(tmp_path, material)
    inputs = deck_from_parent(xml, tmp_path, material, recipe="hydrogenic")
    assert _projection_lines(tmp_path / f"{material}.win") == lines
    assert int(inputs.values["num_wann"]) == count


def test_the_sro_deck_now_reaches_the_conduction_channel_the_supply_left_out(tmp_path):
    """The supply named Sr s, p only (eight functions); the Sr d channel is the 4d band."""
    xml = _parent(tmp_path, "SrO")
    parent = read_parent(xml)
    supply = vasp_outputs.projector_supply(
        vasp_outputs.read_procar(tmp_path / "PROCAR"),
        parent.species,
        parent.fermi_energy_ev,
    )
    assert "d" not in supply["Sr"], "the fixture no longer shows the old defect"
    deck_from_parent(xml, tmp_path, "SrO", recipe="hydrogenic")
    sr_line = _projection_lines(tmp_path / "SrO.win")[0]
    assert sr_line.split(":")[1].split(";") == ["s", "p", "d"]


def test_the_channel_set_does_not_consult_the_occupied_channel_supply(
    tmp_path, monkeypatch
):
    xml = _parent(tmp_path, "SrO", files=("vasprun.xml", "OUTCAR", "EIGENVAL"))

    def _forbidden(*args, **kwargs):
        raise AssertionError("the deck consulted the occupied-channel supply")

    monkeypatch.setattr(vasp_outputs, "projector_supply", _forbidden)
    monkeypatch.setattr(vasp_inputs, "projector_supply", _forbidden, raising=False)
    inputs = deck_from_parent(
        xml, tmp_path, "SrO", recipe="hydrogenic"
    )  # no PROCAR in the directory
    assert int(inputs.values["num_wann"]) == 14


def test_an_explicit_potcar_is_the_channel_source_it_names(tmp_path):
    """A POTCAR passed in is read instead of the parent's OUTCAR echo."""
    xml = _parent(tmp_path, "Si")
    potcar = tmp_path / "elsewhere" / "POTCAR"
    potcar.parent.mkdir()
    potcar.write_text(
        _potcar_block(
            "Si", [(0, -10.0, 23), (0, -7.0, 23), (2, -4.0, 7)], [(3, 0, -10.0, 2.0)]
        )
    )  # an s-only silicon: no p projector, one occupied s shell
    inputs = deck_from_parent(xml, tmp_path, "Si", potcar=potcar, recipe="hydrogenic")
    assert _projection_lines(tmp_path / "Si.win") == ["Si:s"]
    assert int(inputs.values["num_wann"]) == 2


def test_a_species_the_potcars_do_not_describe_is_refused_by_name(tmp_path):
    xml = _parent(tmp_path, "SrO")
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        _potcar_block(
            "Sr",
            [(0, -40.0, 23), (1, -22.0, 23), (2, -1.0, 23)],
            [(4, 0, -40.0, 2.0), (4, 1, -22.0, 6.0)],
        )
    )  # no oxygen
    with pytest.raises(InputError, match="O"):
        deck_from_parent(xml, tmp_path, "SrO", potcar=potcar, recipe="hydrogenic")


def test_without_a_potcar_or_an_outcar_the_deck_is_refused_by_name(tmp_path):
    xml = _parent(tmp_path, "SrO", files=("vasprun.xml",))
    with pytest.raises(InputError, match="POTCAR"):
        deck_from_parent(xml, tmp_path, "SrO", recipe="hydrogenic")


def test_named_projectors_still_override_the_automatic_set(tmp_path):
    xml = _parent(tmp_path, "Si")
    inputs = deck_from_parent(
        xml,
        tmp_path,
        "Si",
        projectors=[{"species": "Si", "orbitals": ("s",)}],
        recipe="hydrogenic",
    )
    assert _projection_lines(tmp_path / "Si.win") == ["Si:s"]
    assert int(inputs.values["num_wann"]) == 2


def test_the_deck_mesh_is_the_mesh_the_parent_ran(tmp_path):
    xml = _parent(tmp_path, "SrO")
    inputs = deck_from_parent(xml, tmp_path, "SrO", recipe="hydrogenic")
    assert str(inputs.values["mp_grid"]).split() == ["2", "2", "2"]
    assert vasp_outputs.read_kmesh(xml) == (2, 2, 2)


# ---------------------------------------------------------------------------
# the frozen window: from the parent's own bands, not from a margin
# ---------------------------------------------------------------------------


def test_the_deck_takes_no_margin_argument():
    parameters = inspect.signature(deck_from_parent).parameters
    assert "frozen_margin_ev" not in parameters
    assert "margin_ev" not in parameters
    assert FROZEN_WINDOW_ABOVE_OCCUPIED_EV == 2.0


@pytest.mark.parametrize("material", ["SrO", "Si"])
def test_the_frozen_window_is_two_ev_above_the_highest_occupied_state(
    tmp_path, material
):
    xml = _parent(tmp_path, material)
    parent = read_parent(xml)
    inputs = deck_from_parent(xml, tmp_path, material, recipe="hydrogenic")
    bands = parent.eigenvalues_ev
    highest_occupied = float(bands[bands <= parent.fermi_energy_ev].max())
    assert _number(inputs.values["dis_froz_max"]) == pytest.approx(
        highest_occupied + 2.0, abs=1e-6
    )


def test_on_sro_the_window_no_longer_follows_the_fermi_level_into_the_gap(tmp_path):
    xml = _parent(tmp_path, "SrO")
    parent = read_parent(xml)
    inputs = deck_from_parent(xml, tmp_path, "SrO", recipe="hydrogenic")
    # VASP's smeared E_F sits 0.39 eV above the valence maximum; E_F + 2 eV was the old rule
    assert _number(inputs.values["dis_froz_max"]) < parent.fermi_energy_ev + 2.0 - 0.3


@pytest.mark.parametrize("margin", [0.5, 2.0, 5.0, 12.0])
def test_the_frozen_window_does_not_move_when_the_channel_margin_does(
    tmp_path, monkeypatch, margin
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = deck_from_parent(
        _parent(reference_dir, "SrO"),
        reference_dir,
        "SrO",
        recipe="hydrogenic",
    ).values

    original = vasp_outputs.projector_supply

    def _supply_at(proj, species, fermi_energy_ev, margin_ev=margin, floor=0.1):
        return original(proj, species, fermi_energy_ev, margin_ev=margin, floor=floor)

    monkeypatch.setattr(vasp_outputs, "projector_supply", _supply_at)
    monkeypatch.setattr(vasp_inputs, "projector_supply", _supply_at, raising=False)
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    moved = deck_from_parent(
        _parent(moved_dir, "SrO"), moved_dir, "SrO", recipe="hydrogenic"
    ).values

    assert _number(moved["dis_froz_max"]) == _number(reference["dis_froz_max"])
    assert int(moved["num_wann"]) == int(reference["num_wann"]) == 14
    assert _projection_lines(moved_dir / "SrO.win") == _projection_lines(
        reference_dir / "SrO.win"
    )


def _parent_with(bands, fermi, spins=None):
    """A minimal parent: bands[k][b] (or spins[s][k][b]) in eV."""
    values = np.asarray(spins if spins is not None else [bands], dtype=float)
    n_k = values.shape[1]
    return ParentCalculation(
        eigenvalues_ev=values,
        occupations=(values <= fermi).astype(float),
        kpoints_cart=np.zeros((n_k, 3)),
        fermi_energy_ev=fermi,
        lattice_ang=np.eye(3) * 3.0,
        alat_ang=3.0,
        species=("X",),
        positions_frac=np.zeros((1, 3)),
        n_symmetries=1,
        n_atomic_wfc=None,
        converged=True,
        calculation="scf",
    )


def test_a_gapped_parent_freezes_above_its_valence_maximum_not_its_fermi_level():
    parent = _parent_with([[-5.0, 0.0, 6.0, 7.0], [-4.0, 1.0, 5.5, 8.0]], fermi=3.0)
    assert not parent.is_metallic()
    assert frozen_window_top(parent, num_wann=3) == pytest.approx(3.0)


def test_a_metallic_parent_freezes_two_ev_above_its_fermi_level():
    parent = _parent_with([[-5.0, 0.5, 6.0, 9.0], [-4.0, 2.5, 7.0, 10.0]], fermi=1.5)
    assert parent.is_metallic()
    assert frozen_window_top(parent, num_wann=3) == pytest.approx(3.5)


def test_the_window_stops_below_the_first_k_point_that_would_overfill_the_model():
    parent = _parent_with([[-5.0, 0.0, 1.5, 7.0], [-4.0, 1.0, 2.2, 8.0]], fermi=1.0)
    top = frozen_window_top(parent, num_wann=2)
    assert 1.0 < top < 1.5
    assert (np.sum(parent.eigenvalues_ev[0] <= top, axis=1) <= 2).all()


def test_the_cap_is_taken_over_both_spin_channels():
    spins = [
        [[-5.0, 0.0, 6.0, 7.0], [-4.0, 1.0, 5.5, 8.0]],
        [[-5.0, 0.0, 1.8, 7.0], [-4.0, 1.0, 5.5, 8.0]],
    ]
    parent = _parent_with(None, fermi=1.0, spins=spins)
    top = frozen_window_top(parent, num_wann=2)
    assert 1.0 < top < 1.8


def test_a_model_with_no_more_bands_than_functions_is_not_capped():
    parent = _parent_with([[-5.0, 0.0, 6.0], [-4.0, 1.0, 5.5]], fermi=3.0)
    assert frozen_window_top(parent, num_wann=3) == pytest.approx(3.0)


def test_functions_too_few_for_the_occupied_manifold_are_refused():
    parent = _parent_with([[-5.0, 0.0, 0.5, 7.0], [-4.0, 0.2, 0.8, 8.0]], fermi=1.0)
    with pytest.raises(InputError, match="occupied"):
        frozen_window_top(parent, num_wann=2)


# ---------------------------------------------------------------------------
# a synthetic POTCAR header: the fields the channel reader reads, nothing licensed
# ---------------------------------------------------------------------------


def _potcar_block(symbol: str, rows, configuration) -> str:
    """rows: (l, reference energy, TYP); TYP 7 is the local channel (no projector).

    configuration: (n, l, eigenvalue, occupation) -- the Atomic configuration table
    the function count reads its occupied shells from.
    """
    lines = [
        f"  PAW_PBE {symbol} 01Jan2000",
        "   1.00000000000000000",
        " parameters from PSCTR are:",
        f"   VRHFIN ={symbol}: synthetic",
        "   LEXCH  = PE",
        f"   TITEL  = PAW_PBE {symbol} 01Jan2000",
        "   Atomic configuration",
        f"    {len(configuration)} entries",
        "     n  l   j            E        occ.",
    ]
    lines += [
        f"     {n}  {l}  {l + 0.5 if l else 0.5:.2f}  {e:14.4f}   {occ:.4f}"
        for n, l, e, occ in configuration
    ]
    lines += [
        "   Description",
        "     l       E           TYP  RCUT    TYP  RCUT",
    ]
    lines += [f"     {l}    {e:.7f}     {typ}  1.500" for l, e, typ in rows]
    lines += ["   Error from kinetic energy argument (eV)", "   NDATA  =      100"]
    for l in sorted({l for l, _, typ in rows if typ != 7}):
        lines += [" Non local Part", f"           {l}           2   1.50000000000000"]
        lines += ["  0.100000000000E+01  0.200000000000E+01"]
    lines += [" End of Dataset"]
    return "\n".join(lines) + "\n"
