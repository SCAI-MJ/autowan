"""The bottom (semicore) exclusion is a knob: ``semicore_exclusion``, named by shell.

On the v1 campaign code the QE SrO runs ended after one or two attempts: the frontier
deck's ``exclude_bands 1-4`` removes Sr 4s and Sr 4p, Sr 4p overlaps O 2s by 0.35 eV, so
the cut runs through a connected manifold and the gate does not honour it -- and no knob
reached the bottom exclusion (``num_bands`` edits move only the top one). The model said
so and escalated (the v1 campaign's stored SrO records).

What these tests hold:

* the knob exists in every family that has a route for it (QE PDWF, SMV, SCDM; VASP
  SMV) and is refused, with the reason, where it has none (the VASP SCDM frontier);
* ``validate_deck`` states the gate's own honour rule: it refuses ``exclude 1-4`` on an
  SrO-like parent as a cut through a connected manifold, lists the semicore shells and
  every cut they imply, and accepts the semicore-inclusive decks with the right counts;
* each family's consequence is written by the deck writer: PDWF's pseudo-atomic orbitals
  (and the method stays PDWF), analytic SMV's semicore trial functions (``r=2`` where an
  angular momentum repeats), SCDM's count; spinor bands count one band per electron;
* the stored SrO construction replays: ``exclude 1-4`` refused, Sr 4s alone or nothing
  excluded accepted, and one such deck built offline without a Wannier90 run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import repair_fixtures as fx

from autowan.actions import (
    BY_NAME,
    ActionRejected,
    Stage,
    action_schema,
    validate_action,
)
from autowan.backends import get_backend
from autowan.deckcheck import check_deck
from autowan.families import AVAILABLE, Method, knob_unavailable, knobs_for
from autowan.gate import exclusion_tolerance_ev, represented_parent_bands
from autowan.inputs import (
    InputError,
    assignments,
    method_of,
    pao_excluded,
    read_construction,
    read_excluded_bands,
    read_win,
    write_derived,
)
from autowan.methods import DeckContext, derive_deck
from autowan.orchestrator import Run, RunConfig
from autowan.qe_outputs import ParentCalculation
from autowan.semicore import (
    RECORD,
    catalogue,
    read_record,
    read_upf_orbitals,
)
from autowan.tools import RepairTools
from autowan.workflow import CALCULATION_FAILED, WorkflowConfig, wannierize

def _stored(env: str) -> Path:
    """A stored run tree outside the repository, named by ``$env``.

    Unset, it is a path that does not exist, so the tests that read it skip.
    """
    return Path(os.environ.get(env, "/nonexistent/" + env))


HERE = Path(__file__).parent
VASP_SRO = HERE / "fixtures" / "vasp" / "SrO"
KNOB = "semicore_exclusion"
CHECKS = (CALCULATION_FAILED,)


# ---------------------------------------------------------------------------
# an SrO-like parent: Sr 4s isolated, Sr 4p overlapping O 2s, a gapped valence
# ---------------------------------------------------------------------------


def overlapping_parent() -> ParentCalculation:
    """Two k-points. Band 1 (Sr 4s) sits 15 eV below a block of four bands (Sr 4p x3 and
    O 2s) in which band 4 reaches 0.3 eV above the bottom of band 5 -- the SrO overlap --
    and that block 10 eV below four occupied valence bands; twelve empty bands above."""
    k1 = [-35.0, -19.3, -17.7, -17.3, -17.1]
    k2 = [-34.9, -17.8, -16.9, -16.8, -15.6]
    valence = [-5.0, -3.5, -2.0, -1.0]
    empty = list(np.linspace(1.0, 30.0, 11))
    rows = [sorted(k + valence + empty) for k in (k1, k2)]
    eigenvalues = np.array([rows], dtype=float)
    return ParentCalculation(
        eigenvalues_ev=eigenvalues,
        occupations=(eigenvalues <= 0.0).astype(float),
        kpoints_cart=np.zeros((2, 3)),
        fermi_energy_ev=0.0,
        lattice_ang=np.eye(3) * 5.16,
        alat_ang=5.16,
        species=("Sr", "O"),
        positions_frac=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        n_symmetries=48,
        n_atomic_wfc=None,
        converged=None,
        calculation="nscf",
    )


UPF = {
    # (label, l, occupation, pseudo_energy in Ry), in UPF order
    "Sr": [
        ("4S", 0, 2.0, -2.934),
        ("4P", 1, 6.0, -1.612),
        ("4D", 2, 1.0, -0.067),
        ("5S", 0, 1.0, -0.242),
        ("5P", 1, 0.0, -0.090),
    ],
    "O": [("2S", 0, 2.0, -1.761), ("2P", 1, 4.0, -0.664)],
}


def write_pseudopotentials(directory: Path) -> Path:
    """A ``data-file-schema.xml`` naming two UPF v2 files that carry PP_CHI headers."""
    directory.mkdir(parents=True, exist_ok=True)
    species = []
    for symbol, orbitals in UPF.items():
        chi = "\n".join(
            f'  <PP_CHI.{i} type="real" size="2" index="{i}" label="{label}" l="{l}" '
            f'occupation="{occ}" n="{i}"\n  pseudo_energy="{e}" cutoff_radius="1.0">'
            f"\n  0.0 0.0\n  </PP_CHI.{i}>"
            for i, (label, l, occ, e) in enumerate(orbitals, start=1)
        )
        (directory / f"{symbol}.upf").write_text(
            f'<UPF version="2.0.1">\n<PP_PSWFC>\n{chi}\n</PP_PSWFC>\n</UPF>\n'
        )
        species.append(
            f'<species name="{symbol}"><pseudo_file>{symbol}.upf</pseudo_file></species>'
        )
    xml = directory / "data-file-schema.xml"
    xml.write_text(
        "<qes><input><atomic_species>"
        + "".join(species)
        + "</atomic_species></input></qes>\n"
    )
    return xml


class OverlapQe(fx.StoredParentQe):
    pass


def _ctx(tmp_path, parent=None, with_upf=True):
    fx.write_deck(tmp_path)
    source = (
        write_pseudopotentials(tmp_path / "parent")
        if with_upf
        else tmp_path / "parent.xml"
    )
    return DeckContext(
        code="qe",
        run_dir=tmp_path,
        parent=parent or overlapping_parent(),
        parent_source=source,
        frontier=read_construction(tmp_path, fx.SEED),
    )


def _tools(ctx, inputs=None):
    run = Run(fx.SEED)
    run.start_attempt({"x": 1}, seed=fx.SEED, method="pdwf")
    run._unresolved = ("occupied_manifold_is_represented",)
    return RepairTools(
        ctx=ctx,
        backend=get_backend("qe"),
        run=run,
        inputs=inputs or ctx.frontier,
        seed=fx.SEED,
        species=("O", "Sr"),
    )


def _semicore(*entries):
    return [{"species": s, "shells": list(v)} for s, v in entries]


def _validate(tools, value, method=None):
    args = {"changes": [{"parameter": KNOB, "new": value}]}
    if method:
        args["method"] = method
    return json.loads(tools("validate_deck", args))


def _derive(ctx, source, value, seed="SrO_r1", method=None, extra=None):
    changes = {
        KNOB: tuple(
            {"species": e["species"], "shells": tuple(e["shells"])} for e in value
        )
    }
    changes.update(extra or {})
    return derive_deck(
        backend=get_backend("qe"),
        source=source,
        ctx=ctx,
        run_dir=ctx.run_dir,
        new_seed=seed,
        changes=changes,
        method=method,
        fermi_energy_ev=0.0,
        species_counts={"Sr": 1, "O": 1},
    )


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------


BASE = {
    "action": "modify_parameters",
    "diagnosis": "d",
    "evidence": list(CHECKS),
    "targets": list(CHECKS),
    "expected_effect": "e",
    "confidence": 0.5,
}


def _action(value, method="pdwf", code="qe"):
    return validate_action(
        {**BASE, "changes": [{"parameter": KNOB, "new": value, "reason": "r"}]},
        stage=Stage.REPAIR,
        evaluated_check_ids=CHECKS,
        unresolved_check_ids=CHECKS,
        allowed_species=("O", "Sr"),
        method=method,
        code=code,
    )


def test_the_knob_is_shared_by_every_family_and_refused_where_it_has_no_route():
    for method in Method:
        assert KNOB in knobs_for(method)
    for code, families in AVAILABLE.items():
        for method in families:
            reason = knob_unavailable(code, method, KNOB)
            if (code, method) == ("vasp", Method.SCDM):
                assert reason and "NBANDS" in reason and "switch to SMV" in reason
                with pytest.raises(ActionRejected, match="no route on vasp SCDM"):
                    _action(_semicore(("Sr", [])), method="scdm", code="vasp")
            else:
                assert reason is None
                action = _action(_semicore(("Sr", ["4s"])), method=method, code=code)
                assert action.changes[0].new == ({"species": "Sr", "shells": ("4s",)},)
    assert BY_NAME[KNOB].rebuilds_overlaps, "a new band pool needs new overlaps"
    schema = action_schema(Stage.REPAIR, CHECKS, CHECKS, ("O", "Sr"), code="qe")
    assert (
        KNOB
        in schema["properties"]["changes"]["items"]["properties"]["parameter"]["enum"]
    )


def test_the_value_is_a_per_species_shell_list_and_an_empty_list_includes_all():
    assert _action(_semicore(("Sr", []), ("O", ["2s"]))).changes[0].new == (
        {"species": "Sr", "shells": ()},
        {"species": "O", "shells": ("2s",)},
    )
    for bad, message in (
        ([], "non-empty list"),
        ("4s", "non-empty list"),
        (_semicore(("Sr", ["4s"]), ("Sr", [])), "twice"),
        (_semicore(("Fe", ["3s"])), "not in this structure"),
        (_semicore(("Sr", ["4s", "4s"])), "names a shell twice"),
        ([{"species": "Sr", "orbitals": ["4s"]}], "exactly 'species' and 'shells'"),
    ):
        with pytest.raises(ActionRejected, match=message):
            _action(bad)


# ---------------------------------------------------------------------------
# validate_deck states the gate's honour rule
# ---------------------------------------------------------------------------


def test_validate_deck_refuses_a_bottom_cut_through_a_connected_manifold(tmp_path):
    ctx = _ctx(tmp_path)
    verdict = json.loads(_tools(ctx)("validate_deck", {}))
    assert verdict["ok"] is False
    problem = verdict["problems"][0]
    assert "exclude_bands 1-4 not honoured" in problem
    assert "reach 0.30 eV into the represented ones" in problem
    assert "semicore_exclusion" in problem
    assert verdict["facts"]["exclusion"]["gate_verdict"] == "not_honoured"
    assert verdict["facts"]["exclusion"]["gap_ev"] == pytest.approx(-0.3)
    # The same words the gate writes into its verdict, from the same rule.
    tolerance = exclusion_tolerance_ev(ctx.parent.eigenvalues_ev, 0.0)
    c = SimpleNamespace(parent=ctx.parent, inputs={"exclude_bands": (1, 2, 3, 4)})
    bands, note = represented_parent_bands(c, 0, tolerance)
    assert bands.shape[1] == 20, "not honoured: every band stays the target"
    assert problem.startswith(note.lstrip("; "))


def test_validate_deck_lists_the_shells_and_every_cut_they_imply(tmp_path):
    verdict = json.loads(_tools(_ctx(tmp_path))("validate_deck", {}))
    info = verdict["semicore"]
    assert info["available"] and info["route"] == "pao"
    shells = {(s["species"], s["shell"]): s for s in info["shells"]}
    assert set(shells) == {("Sr", "4s"), ("Sr", "4p"), ("O", "2s")}
    assert shells[("Sr", "4p")]["bands"] == 3 and shells[("Sr", "4p")]["excluded_now"]
    assert (
        shells[("O", "2s")]["occupation"] == 2.0
        and not shells[("O", "2s")]["excluded_now"]
    )
    assert info["current"] == [
        {"species": "Sr", "shells": ["4s", "4p"]},
        {"species": "O", "shells": []},
    ]
    cuts = {row["exclude_lowest"]: row for row in info["cuts"]}
    assert cuts[0]["honoured_by_gate"] and cuts[1]["honoured_by_gate"]
    assert cuts[5]["honoured_by_gate"] and cuts[5]["gap_above_ev"] == pytest.approx(
        10.6
    )
    assert not cuts[4]["honoured_by_gate"] and cuts[4]["current"]
    assert cuts[4]["gap_above_ev"] == pytest.approx(-0.3)
    # Sr 4s is the deepest shell of the pseudo-atoms, so it heads the one-band cut.
    assert cuts[1]["shell_sets"][0] == {
        "shells": ["Sr 4s"],
        "deepest_by_pseudo_atom_energy": True,
    }
    lowest = {row["band"]: row for row in info["lowest_bands"]}
    assert lowest[1]["excluded_by_deck"] and not lowest[5]["excluded_by_deck"]
    assert lowest[1]["gap_to_next_ev"] == pytest.approx(15.6)


@pytest.mark.parametrize(
    "value, num_wann, num_bands, exclude",
    [
        (_semicore(("Sr", ["4s"])), 16, 19, "1"),
        (_semicore(("Sr", [])), 17, 20, None),
        (_semicore(("Sr", ["4s", "4p"]), ("O", ["2s"])), 12, 15, "1-5"),
    ],
)
def test_a_semicore_inclusive_deck_is_accepted_with_the_right_counts(
    tmp_path, value, num_wann, num_bands, exclude
):
    verdict = _validate(_tools(_ctx(tmp_path)), value)
    assert verdict["ok"] is True, verdict["problems"]
    assert int(verdict["deck"]["num_wann"]) == num_wann
    assert int(verdict["deck"]["num_bands"]) == num_bands
    assert verdict["deck"]["exclude_bands"] == exclude
    assert verdict["facts"]["paos_kept"] == num_wann
    assert verdict["method"] == "pdwf"
    if exclude:
        assert verdict["facts"]["exclusion"]["gate_verdict"] == "honoured"


def test_moving_the_cut_back_into_the_manifold_is_refused(tmp_path):
    ctx = _ctx(tmp_path)
    included = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])))
    back = _validate(_tools(ctx, included), _semicore(("Sr", ["4s", "4p"])))
    assert back["ok"] is False and "reach 0.30 eV" in back["problems"][0]
    o_only = _validate(_tools(ctx), _semicore(("Sr", []), ("O", ["2s"])))
    # One band at the bottom is honoured by index; the pseudo-atoms say the lowest
    # shell is Sr 4s, not O 2s, and the check says so.
    assert o_only["ok"] is True
    assert any("Sr 4s" in note and "deeper" in note for note in o_only["notes"])


# ---------------------------------------------------------------------------
# the writers: each family's consequence follows from the shell list
# ---------------------------------------------------------------------------


def test_a_pdwf_deck_keeps_its_method_its_window_and_records_the_shells(tmp_path):
    ctx = _ctx(tmp_path)
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])))
    text = deck.win.read_text()
    assert method_of(deck) is Method.PDWF
    assert "dis_froz_proj = .true." in text and "dis_proj_max = 0.95" in text
    assert read_excluded_bands(deck.win) == (1,)
    assert (
        read_win(deck.win)["num_bands"] == "19"
        and read_win(deck.win)["num_wann"] == "16"
    )
    assert pao_excluded(deck.pw2wan.read_text()) == {1}, "Sr 4p's orbitals are back"
    assert f"{RECORD} Sr: 4s; O: none" in text
    assert read_record(text) == {"Sr": ("4s",), "O": ()}
    again = catalogue(text, deck.pw2wan.read_text(), tmp_path, ctx)
    assert again.known and again.current_value()[0] == {
        "species": "Sr",
        "shells": ["4s"],
    }
    assert check_deck(deck, parent=ctx.parent, code="qe", ctx=ctx).ok
    # and back again: the frontier's own assignment, refused as it was
    back = _derive(ctx, deck, _semicore(("Sr", ["4s", "4p"])), seed="SrO_r2")
    assert read_excluded_bands(back.win) == (1, 2, 3, 4)
    assert pao_excluded(back.pw2wan.read_text()) == {1, 3, 4, 5}
    assert read_win(back.win)["num_wann"] == "13"
    refused = check_deck(back, parent=ctx.parent, code="qe", ctx=ctx)
    assert not refused.ok and "not honoured" in refused.reason()


def test_a_kept_orbital_of_an_excluded_shell_is_refused(tmp_path):
    ctx = _ctx(tmp_path)
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])))
    # Hand-edit the PAO list so Sr 4s's orbital is kept while its band stays excluded.
    pw = deck.pw2wan.read_text().replace("atom_proj_exclude(1) = 1\n", "")
    deck.pw2wan.write_text(pw)
    deck.win.write_text(deck.win.read_text().replace("num_wann = 16", "num_wann = 17"))
    refused = check_deck(
        read_construction(tmp_path, "SrO_r1"), parent=ctx.parent, code="qe", ctx=ctx
    )
    assert not refused.ok
    assert "keeps the pseudo-atomic orbitals of Sr 4s" in refused.reason()


def test_pao_subset_and_semicore_exclusion_must_agree(tmp_path):
    ctx = _ctx(tmp_path)
    keeps_excluded = [{"species": "Sr", "orbitals": ("4p", "5s", "4d")}]
    with pytest.raises(InputError, match="the subset keeps Sr 4p"):
        _derive(
            ctx,
            ctx.frontier,
            _semicore(("Sr", ["4s", "4p"]), ("O", ["2s"])),
            extra={"pao_subset": keeps_excluded},
        )
    leaves_out = [{"species": "Sr", "orbitals": ("5s", "4d")}]
    with pytest.raises(InputError, match="the subset leaves out Sr 4p"):
        _derive(
            ctx,
            ctx.frontier,
            _semicore(("Sr", ["4s"])),
            extra={"pao_subset": leaves_out},
        )
    agreed = [{"species": "Sr", "orbitals": ("4p", "5s", "4d")}]
    deck = _derive(
        ctx, ctx.frontier, _semicore(("Sr", ["4s"])), extra={"pao_subset": agreed}
    )
    assert read_win(deck.win)["num_wann"] == "13", "Sr 4p in, Sr 5p out"


def test_a_shell_the_deck_does_not_carry_is_refused_with_the_list(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(InputError, match="no semicore shell '3d'.*Sr 4s.*Sr 4p"):
        _derive(ctx, ctx.frontier, _semicore(("Sr", ["3d"])))
    verdict = _validate(_tools(ctx), _semicore(("Sr", ["3d"])))
    assert verdict["ok"] is False and "no semicore shell '3d'" in verdict["problems"][0]


def test_without_pseudopotential_files_the_excluded_shells_are_still_named(tmp_path):
    ctx = _ctx(tmp_path, with_upf=False)
    cat = catalogue(fx.WIN, fx.PW2WAN, tmp_path, ctx)
    assert cat.known and [(s.species, s.label) for s in cat.shells] == [
        ("Sr", "4s"),
        ("Sr", "4p"),
    ]
    assert "closed shells unknown" in cat.source


def test_an_scdm_deck_moves_its_count_by_the_bands_brought_in(tmp_path):
    ctx = _ctx(tmp_path)
    fx.write_overlaps(tmp_path)
    ctx = _ctx(tmp_path)
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])), method="scdm")
    assert method_of(deck) is Method.SCDM
    values = read_win(deck.win)
    assert values["num_wann"] == "16" and values["num_bands"] == "19"
    assert read_excluded_bands(deck.win) == (1,)
    assert "atom_proj" not in deck.pw2wan.read_text()
    assert check_deck(deck, parent=ctx.parent, code="qe", ctx=ctx).ok


def test_a_switch_carries_a_semicore_edit_onto_the_target_starting_deck(tmp_path):
    ctx = _ctx(tmp_path)
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])), method="smv")
    assert method_of(deck) is Method.SMV and "dis_proj" not in deck.win.read_text()
    assert read_excluded_bands(deck.win) == (1,)
    assert read_win(deck.win)["num_wann"] == "16"
    assert pao_excluded(deck.pw2wan.read_text()) == {1}
    verdict = _validate(_tools(ctx), _semicore(("Sr", ["4s"])), method="smv")
    assert verdict["switch"] == "pdwf -> smv" and int(verdict["deck"]["num_wann"]) == 16


def test_analytic_smv_adds_a_trial_function_per_shell_and_removes_its_own(tmp_path):
    ctx = _ctx(tmp_path)
    named = [
        {"species": "Sr", "orbitals": ("s", "d")},
        {"species": "O", "orbitals": ("s", "p")},
    ]
    smv = derive_deck(
        backend=get_backend("qe"),
        source=ctx.frontier,
        ctx=ctx,
        run_dir=tmp_path,
        new_seed="SrO_smv",
        changes={"projectors": named},
        method="smv",
        fermi_energy_ev=0.0,
        species_counts={"Sr": 1, "O": 1},
    )
    assert method_of(smv) is Method.SMV and read_win(smv.win)["num_wann"] == "10"
    deck = _derive(ctx, smv, _semicore(("Sr", ["4s"])), seed="SrO_r1")
    assert method_of(deck) is Method.SMV
    assert read_win(deck.win)["projectors"] == "Sr:s;d;O:s;p;Sr:p"
    assert read_win(deck.win)["num_wann"] == "13" and read_excluded_bands(deck.win) == (
        1,
    )
    assert "! semicore_projections: Sr:p" in deck.win.read_text()
    # Sr 4s in as well: Sr already carries an s function, so this one is r=2.
    both = _derive(ctx, deck, _semicore(("Sr", [])), seed="SrO_r2")
    # semicore lines follow the shells deepest first: Sr 4s, then Sr 4p
    assert read_win(both.win)["projectors"] == "Sr:s;d;O:s;p;Sr:s:r=2;Sr:p"
    assert (
        read_win(both.win)["num_wann"] == "14" and read_excluded_bands(both.win) == ()
    )
    # and out again: only the lines the knob wrote go
    out = _derive(ctx, both, _semicore(("Sr", ["4s", "4p"])), seed="SrO_r3")
    assert read_win(out.win)["projectors"] == "Sr:s;d;O:s;p"
    assert read_excluded_bands(out.win) == (1, 2, 3, 4)


def test_spinor_paos_count_one_band_per_electron(tmp_path):
    """A j-resolved PAO list: 4p is p1/2 (two bands) and p3/2 (four)."""
    run = tmp_path
    directory = run / "external_projectors"
    directory.mkdir(parents=True)
    (directory / "Sr.dat").write_text("10 4\n0 1 1 2\n0.5 0.5 1.5 1.5\n")
    (directory / "projectors.json").write_text(
        json.dumps(
            {
                "Sr": [
                    {"label": "4s", "l": 0},
                    {"label": "4p1/2", "l": 1},
                    {"label": "4p3/2", "l": 1},
                    {"label": "4d3/2", "l": 2},
                ]
            }
        )
    )
    win = (
        "spinors = .true.\nnum_wann = 4\nnum_bands = 20\nexclude_bands = 1-8\n"
        "dis_froz_proj = .true.\ndis_proj_min = 0.01\ndis_proj_max = 0.95\n"
        "begin atoms_cart\nang\nSr 0 0 0\nend atoms_cart\n"
    )
    pw = (
        "&inputpp\n  atom_proj = .true.\n  atom_proj_ext = .true.\n"
        "  atom_proj_dir = 'external_projectors/'\n"
        + "".join(f"  atom_proj_exclude({i}) = {i}\n" for i in range(1, 9))
        + "/\n"
    )
    (run / "pt.win").write_text(win)
    (run / "pt.pw2wan").write_text(pw)
    bands = np.concatenate(
        [np.full(2, -60.0), np.full(6, -40.0), np.linspace(-5, 20, 20)]
    )
    parent = SimpleNamespace(
        eigenvalues_ev=np.tile(bands, (1, 2, 1)),
        n_bands=28,
        spinor=True,
        fermi_energy_ev=0.0,
        species=("Sr",),
    )
    ctx = DeckContext("qe", run, parent, run / "none.xml", read_construction(run, "pt"))
    cat = catalogue(win, pw, run, ctx)
    per = {s.label: s.bands for s in cat.shells}
    assert per == {"4s": 2, "4p1/2": 2, "4p3/2": 4} and cat.known
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])), seed="pt_r1")
    assert read_excluded_bands(deck.win) == (1, 2)
    assert (
        read_win(deck.win)["num_wann"] == "10"
        and read_win(deck.win)["num_bands"] == "26"
    )


# ---------------------------------------------------------------------------
# VASP: SMV has a route, the SCDM frontier none
# ---------------------------------------------------------------------------


@pytest.fixture
def vasp_sro(tmp_path):
    from autowan.backends.vasp.inputs import semicore_excluded_deck
    from autowan.backends.vasp.outputs import read_parent

    parent_dir = tmp_path / "parent"
    shutil.copytree(VASP_SRO, parent_dir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    deck = semicore_excluded_deck(parent_dir / "vasprun.xml", run_dir, "SrO")
    parent = read_parent(parent_dir / "vasprun.xml")
    ctx = DeckContext("vasp", run_dir, parent, parent_dir / "vasprun.xml", deck)
    return ctx, deck


def _vderive(ctx, source, value, seed="SrO_r1"):
    return derive_deck(
        backend=get_backend("vasp"),
        source=source,
        ctx=ctx,
        run_dir=ctx.run_dir,
        new_seed=seed,
        changes={KNOB: tuple({"species": s, "shells": tuple(v)} for s, v in value)},
        fermi_energy_ev=ctx.parent.fermi_energy_ev,
        species_counts={"Sr": 1, "O": 1},
    )


def test_vasp_smv_brings_shells_in_with_their_own_trial_functions(vasp_sro):
    ctx, deck = vasp_sro
    cat = catalogue(deck.win.read_text(), None, ctx.run_dir, ctx)
    assert cat.known and cat.bottom == 5
    assert {(s.species, s.label, s.bands) for s in cat.shells} == {
        ("Sr", "4s", 1),
        ("Sr", "4p", 3),
        ("O", "2s", 1),
    }
    top = [b for b in read_excluded_bands(deck.win) if b > 5]
    out = _vderive(ctx, deck, [("Sr", ["4s"]), ("O", [])])
    # one line per shell brought in, deepest first by the POTCAR's atomic levels
    assert read_win(out.win)["projectors"] == "Sr:s;d;O:p;O:s;Sr:p"
    assert read_win(out.win)["num_wann"] == "13"
    excluded = read_excluded_bands(out.win)
    assert excluded == (1, *top), "the headroom block stays where it was"
    assert int(read_win(out.win)["num_bands"]) + len(excluded) == 32
    check = check_deck(out, parent=ctx.parent, code="vasp", ctx=ctx)
    assert check.ok, check.reason()
    everything = _vderive(ctx, deck, [("Sr", []), ("O", [])], seed="SrO_r2")
    assert read_win(everything.win)["projectors"] == "Sr:s;d;O:p;Sr:s:r=2;O:s;Sr:p"


def test_vasp_spinor_bands_hold_one_electron_each(vasp_sro):
    ctx, deck = vasp_sro
    spinor = dataclasses.replace(ctx.parent, spinor=True)
    sctx = DeckContext("vasp", ctx.run_dir, spinor, ctx.parent_source, deck)
    text = deck.win.read_text().replace("exclude_bands = 1-5", "exclude_bands = 1-10")
    cat = catalogue(text, None, ctx.run_dir, sctx)
    assert {(s.label, s.bands) for s in cat.shells} == {("4s", 2), ("4p", 6), ("2s", 2)}
    assert cat.known and cat.bottom == 10, "all three shells: 10 spinor bands"


def test_vasp_smv_refuses_a_cut_through_the_manifold_or_of_the_wrong_character(
    vasp_sro,
):
    ctx, deck = vasp_sro
    o_in = _vderive(ctx, deck, [("O", [])])  # Sr 4s 4p out: bands 1-4
    refused = check_deck(o_in, parent=ctx.parent, code="vasp", ctx=ctx)
    assert not refused.ok and "not honoured" in refused.reason()
    s_in = _vderive(ctx, deck, [("Sr", ["4p"])], seed="SrO_r2")  # Sr 4p + O 2s out
    refused = check_deck(s_in, parent=ctx.parent, code="vasp", ctx=ctx)
    assert "band 1 (Sr s" in refused.reason(), "band 1 is Sr 4s, which stays in"


def test_the_vasp_scdm_frontier_has_no_bottom_exclusion_route(tmp_path, monkeypatch):
    from autowan.backends.vasp import inputs as vasp_inputs
    from autowan.backends.vasp.inputs import scdm_frontier_deck
    from autowan.semicore import SemicoreUnavailable

    parent_dir = tmp_path / "parent"
    shutil.copytree(VASP_SRO, parent_dir)
    monkeypatch.setattr(vasp_inputs, "pseudo_wannier_count", lambda potcar, species: 13)
    deck = scdm_frontier_deck(parent_dir / "vasprun.xml", tmp_path, "SrO")
    ctx = DeckContext("vasp", tmp_path, None, parent_dir / "vasprun.xml", deck)
    with pytest.raises(SemicoreUnavailable, match="switch to SMV"):
        catalogue(deck.win.read_text(), None, tmp_path, ctx)
    with pytest.raises(InputError, match="resolved against the run's parent"):
        get_backend("vasp").write_derived(deck, tmp_path, "x", {KNOB: ()})


# ---------------------------------------------------------------------------
# band_character shows the excluded bands
# ---------------------------------------------------------------------------


def test_band_character_lists_the_excluded_bands_with_their_gaps(tmp_path):
    out = json.loads(
        _tools(_ctx(tmp_path))("band_character", {"e_min": -40, "e_max": -10})
    )
    rows = {r["band"]: r for r in out["bands"]}
    assert set(rows) == {1, 2, 3, 4, 5}
    assert [rows[b]["excluded_by_deck"] for b in range(1, 6)] == [True] * 4 + [False]
    assert rows[4]["gap_to_next_ev"] == pytest.approx(-0.3)


def test_on_vasp_band_character_names_each_excluded_bands_character(vasp_sro):
    ctx, deck = vasp_sro
    tools = RepairTools(
        ctx=ctx, backend=get_backend("vasp"), run=Run("x"), inputs=deck, seed="SrO"
    )
    rows = json.loads(tools("band_character", {"e_min": -40, "e_max": -12}))["bands"]
    assert rows[0]["band"] == 1 and rows[0]["character"].startswith("Sr s")
    assert all(r["excluded_by_deck"] for r in rows)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def _loop(tmp_path, script):
    write_pseudopotentials(tmp_path / "parent")
    fx.write_deck(tmp_path)
    runner = fx.RefusingEngine(tmp_path)
    model = fx.ToolScript(script)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "parent" / "data-file-schema.xml",
        seed=fx.SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=4), draw_figures=False
        ),
        backend=OverlapQe(overlapping_parent()),
        log_path=tmp_path / "run.json",
    )
    return result, runner, model


def test_the_loop_writes_and_runs_a_semicore_edit_and_refuses_a_doomed_one(tmp_path):
    include = fx.answer(changes=[(KNOB, _semicore(("Sr", ["4s"])))])
    untouched = fx.answer(changes=[("dis_proj_max", 0.9)])
    result, runner, model = _loop(
        tmp_path,
        [
            ([("validate_deck", {})], untouched),
            ([], include),
            ([], fx.answer("escalate")),
        ],
    )
    attempts = result.run.attempts
    # An edit that leaves the cut through the manifold is refused before it runs.
    assert attempts[1].status == "deck_refused"
    assert "not honoured" in attempts[1].calculation_error
    # The semicore edit is written, checked and built.
    deck = read_construction(tmp_path, attempts[2].seed)
    assert read_excluded_bands(deck.win) == (1,)
    assert read_win(deck.win)["num_wann"] == "16"
    assert [b[0] for b in runner.builds] == [fx.SEED, attempts[2].seed]
    assert runner.builds[1][1] is None, "a new band pool recomputes the overlaps"
    payload = model.requests[0]["payload_text"]
    assert (
        "semicore_exclusion" in payload
        and "THE BOTTOM OF THE SPECTRUM" in model.requests[0]["system"]
    )
    assert "'4s', '4p'" in payload or '"4s", "4p"' in payload


# ---------------------------------------------------------------------------
# replay: the stored SrO construction of the v1 campaign (not shipped; skipped when absent)
# ---------------------------------------------------------------------------


STORED = _stored("AUTOWAN_TEST_STORED")
W90DEV = _stored("AUTOWAN_TEST_W90DEV")
stored = pytest.mark.skipif(
    not (STORED / "wan.win").is_file(), reason="the v1 SrO run tree is not shipped"
)


@pytest.fixture(scope="module")
def stored_sro(tmp_path_factory):
    """The stored frontier deck, copied; projectors and the parent linked read-only."""
    work = tmp_path_factory.mktemp("sro_replay")
    for name in ("wan.win", "wan.pw2wan"):
        shutil.copy(STORED / name, work / name)
    (work / "external_projectors").symlink_to(STORED / "external_projectors")
    (work / "out").mkdir()
    (work / "out" / "aiida.save").symlink_to((STORED / "out" / "aiida.save").resolve())
    qe = get_backend("qe")
    parent_xml = work / "out" / "aiida.save" / "data-file-schema.xml"
    parent = qe.read_parent(parent_xml)
    frontier = qe.read_construction(work, "wan")
    ctx = DeckContext("qe", work, parent, parent_xml, frontier)
    record = json.loads((STORED / "record.json").read_text())
    return work, ctx, record


@stored
def test_replay_the_v1_escalation_was_the_bottom_exclusion(stored_sro):
    _, ctx, record = stored_sro
    assert record["code_sha"].startswith("58871ea60")
    assert "exclude_bands" in record["terminal_reason"]
    assert "0.35 eV" in record["terminal_reason"]


@stored
def test_replay_validate_deck_refuses_exclude_1_4_as_not_isolated(stored_sro):
    _, ctx, _ = stored_sro
    tools = _tools(ctx)
    verdict = json.loads(tools("validate_deck", {}))
    assert verdict["ok"] is False
    assert "exclude_bands 1-4 not honoured" in verdict["problems"][0]
    assert "reach 0.35 eV" in verdict["problems"][0]
    cuts = {r["exclude_lowest"]: r for r in verdict["semicore"]["cuts"]}
    assert not cuts[4]["honoured_by_gate"] and cuts[4]["current"]
    assert cuts[1]["honoured_by_gate"] and cuts[0]["honoured_by_gate"]
    assert cuts[5]["honoured_by_gate"]
    names = {
        (s["species"], s["shell"], s.get("name")) for s in verdict["semicore"]["shells"]
    }
    assert names == {("Sr", "s1", "4s"), ("Sr", "p1", "4p"), ("O", "s1", "2s")}
    # The v1 model's own test deck (Sr 4p's orbitals kept, exclude 1-4) is refused too.
    kept = _validate_changes(
        tools,
        [
            {
                "parameter": "pao_subset",
                "new": [{"species": "Sr", "orbitals": ["p1", "s2", "p2", "d1"]}],
            }
        ],
    )
    assert kept["ok"] is False and "not honoured" in kept["problems"][0]


def _validate_changes(tools, changes):
    return json.loads(tools("validate_deck", {"changes": changes}))


@stored
@pytest.mark.parametrize(
    "value, num_wann, num_bands, exclude",
    [
        (_semicore(("Sr", ["4s"])), 16, 51, "1"),
        (_semicore(("Sr", ["s1"])), 16, 51, "1"),
        (_semicore(("Sr", [])), 17, 52, None),
    ],
)
def test_replay_a_semicore_inclusive_deck_is_accepted(
    stored_sro, value, num_wann, num_bands, exclude
):
    _, ctx, _ = stored_sro
    verdict = _validate(_tools(ctx), value)
    assert verdict["ok"] is True, verdict["problems"]
    assert int(verdict["deck"]["num_wann"]) == num_wann == verdict["facts"]["paos_kept"]
    assert int(verdict["deck"]["num_bands"]) == num_bands
    assert verdict["deck"]["exclude_bands"] == exclude


@stored
def test_replay_one_inclusive_deck_builds_offline_without_a_wannier90_run(stored_sro):
    work, ctx, _ = stored_sro
    deck = _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])), seed="wan_r1")
    assert method_of(deck) is Method.PDWF
    check = check_deck(deck, parent=ctx.parent, code="qe", run_dir=work, ctx=ctx)
    assert check.ok, check.reason()
    assert check.facts["exclusion"]["gate_verdict"] == "honoured"
    assert check.facts["exclusion"]["gap_ev"] == pytest.approx(15.66, abs=0.01)
    values = assignments(deck.win.read_text())
    assert (values["num_wann"], values["num_bands"], values["exclude_bands"]) == (
        "16",
        "51",
        "1",
    )
    assert pao_excluded(deck.pw2wan.read_text()) == {1}
    # attempt 0 is untouched: the stored deck is still the published one
    assert (work / "wan.win").read_text() == (STORED / "wan.win").read_text()
    assert not (work / "wan_r1.wout").exists() and not (work / "wan_r1.chk").exists()


@stored
@pytest.mark.skipif(not W90DEV.is_file(), reason="the Wannier90 development build is not installed")
def test_replay_wannier90_preprocessing_accepts_the_inclusive_deck(stored_sro):
    """``wannier90.x -pp`` only parses the deck and writes the .nnkp: no overlaps, no
    disentanglement, no minimisation."""
    work, ctx, _ = stored_sro
    _derive(ctx, ctx.frontier, _semicore(("Sr", ["4s"])), seed="wan_pp")
    done = subprocess.run(
        [str(W90DEV), "-pp", "wan_pp"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout[-400:] + done.stderr[-400:]
    wout = (work / "wan_pp.wout").read_text()
    assert "Number of Wannier Functions               :                16" in wout
    assert "Number of input Bloch states              :                51" in wout
    nnkp = (work / "wan_pp.nnkp").read_text()
    assert "begin exclude_bands\n   1\n   1\nend exclude_bands" in nnkp
