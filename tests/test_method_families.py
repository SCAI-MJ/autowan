"""The construction method is a first-class, logged choice.

Three families -- SMV, PDWF, SCDM -- each with only its own knobs; QE offers all three,
VASP SMV and SCDM. A knob from another family is refused with the reason, no writer
path changes a deck's family, and ``switch_method`` is an explicit action that costs an
attempt, starts from the family's deterministic starting deck, and is recorded with
where it came from, where it went and why.
"""

from __future__ import annotations

import json

import pytest
import repair_fixtures as fx

from autowan.actions import ActionRejected, Stage, action_schema, validate_action
from autowan.families import AVAILABLE, FAMILY_KNOBS, Method, families_of, knobs_for
from autowan.inputs import (
    InputError,
    assignments,
    deck_method,
    method_of,
    pao_excluded,
    pao_layout,
    read_construction,
    read_win,
    write_derived,
)
from autowan.methods import DeckContext, StartingDeckUnavailable, starting_deck
from autowan.orchestrator import RunConfig
from autowan.workflow import CALCULATION_FAILED, WorkflowConfig, wannierize

CHECKS = (CALCULATION_FAILED,)
BASE = {
    "diagnosis": "d",
    "evidence": list(CHECKS),
    "targets": list(CHECKS),
    "expected_effect": "e",
    "confidence": 0.5,
}


def _validate(payload, method="pdwf", code="qe"):
    return validate_action(
        {**BASE, **payload},
        stage=Stage.REPAIR,
        evaluated_check_ids=CHECKS,
        unresolved_check_ids=CHECKS,
        allowed_species=("O", "Sr"),
        method=method,
        code=code,
    )


def _change(parameter, new):
    return {"parameter": parameter, "new": new, "reason": "r"}


# ---------------------------------------------------------------------------
# the families and their knobs
# ---------------------------------------------------------------------------


def test_each_engine_offers_its_families_and_vasp_has_no_pdwf():
    assert set(AVAILABLE["qe"]) == {Method.SMV, Method.PDWF, Method.SCDM}
    assert set(AVAILABLE["vasp"]) == {Method.SMV, Method.SCDM}


def test_each_family_exposes_only_its_own_knobs():
    assert FAMILY_KNOBS[Method.SMV] == (
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
        "projectors",
    )
    # PD+ED, as published: the frozen window's top is PDWF's as well as SMV's
    assert FAMILY_KNOBS[Method.PDWF] == (
        "pao_subset",
        "dis_proj_min",
        "dis_proj_max",
        "dis_froz_max",
    )
    assert FAMILY_KNOBS[Method.SCDM] == ("scdm_mu", "scdm_sigma")
    assert families_of("dis_froz_max") == (Method.SMV, Method.PDWF)
    # dis_froz_max is the only knob two families share
    for method in Method:
        others = {k for m, ks in FAMILY_KNOBS.items() if m is not method for k in ks}
        assert others & set(knobs_for(method)) <= {"dis_froz_max"}
    assert "dis_froz_max" not in knobs_for(Method.SCDM)


@pytest.mark.parametrize(
    "method, knob, owner",
    [
        ("pdwf", "projectors", "SMV"),
        ("pdwf", "dis_win_max", "SMV"),
        ("pdwf", "dis_froz_min", "SMV"),
        ("scdm", "dis_froz_max", "SMV or PDWF"),
        ("smv", "dis_proj_max", "PDWF"),
        ("scdm", "dis_win_max", "SMV"),
        ("smv", "scdm_mu", "SCDM"),
    ],
)
def test_a_cross_family_knob_is_refused_with_the_reason(method, knob, owner):
    value = [{"species": "Sr", "orbitals": ["d"]}] if knob == "projectors" else 0.5
    with pytest.raises(ActionRejected) as info:
        _validate(
            {"action": "modify_parameters", "changes": [_change(knob, value)]},
            method=method,
        )
    message = str(info.value)
    assert f"{owner} knob" in message and "switch_method" in message


def test_switch_method_needs_a_family_this_engine_offers_and_a_reason():
    with pytest.raises(ActionRejected, match="not available on vasp"):
        _validate(
            {"action": "switch_method", "method": "pdwf", "reason": "r"},
            method="scdm",
            code="vasp",
        )
    with pytest.raises(ActionRejected, match="already built by PDWF"):
        _validate({"action": "switch_method", "method": "pdwf", "reason": "r"})
    with pytest.raises(ActionRejected, match="non-empty reason"):
        _validate({"action": "switch_method", "method": "smv"})
    with pytest.raises(ActionRejected, match="switch_method's"):
        _validate(
            {
                "action": "modify_parameters",
                "method": "smv",
                "changes": [_change("num_iter", 10)],
            }
        )


def test_switch_edits_are_judged_against_the_target_family():
    action = _validate(
        {
            "action": "switch_method",
            "method": "smv",
            "reason": "the PAO window cannot hold the frozen states",
            "changes": [_change("dis_froz_max", 1.5)],
        }
    )
    assert (action.method_from, action.method, action.kind) == (
        "pdwf",
        "smv",
        "switch_method",
    )
    with pytest.raises(ActionRejected, match="PDWF knob"):
        _validate(
            {
                "action": "switch_method",
                "method": "smv",
                "reason": "r",
                "changes": [_change("dis_proj_max", 0.9)],
            }
        )


def test_the_schema_lists_only_the_engines_families():
    schema = action_schema(Stage.REPAIR, CHECKS, CHECKS, code="vasp")
    assert schema["properties"]["method"]["enum"] == ["scdm", "smv"]
    names = schema["properties"]["changes"]["items"]["properties"]["parameter"]["enum"]
    assert "pao_subset" not in names and "scdm_mu" in names


# ---------------------------------------------------------------------------
# reading the family off a deck, and writers that keep it
# ---------------------------------------------------------------------------


def test_the_family_is_read_from_the_deck(tmp_path):
    fx.write_deck(tmp_path)
    assert method_of(read_construction(tmp_path, fx.SEED)) is Method.PDWF
    assert deck_method("num_wann = 4\n", "&i\n scdm_proj = .true.\n/\n") is Method.SCDM
    assert (
        deck_method("! overlap-step INCAR: LSCDM = .TRUE.\nnum_wann = 4\n")
        is Method.SCDM
    )
    assert deck_method("num_wann = 4\ndis_froz_max = 3\n") is Method.SMV


def test_the_pao_layout_numbers_states_as_pw2wannier90_does(tmp_path):
    fx.write_deck(tmp_path)
    layout = pao_layout(tmp_path, fx.WIN, fx.PW2WAN)
    assert len(layout) == 17
    assert [(s.index, s.species, s.label) for s in layout[:3]] == [
        (1, "Sr", "4s"),
        (2, "Sr", "5s"),
        (3, "Sr", "4p"),
    ]
    assert layout[13].species == "O" and layout[13].label == "2s"
    assert pao_excluded(fx.PW2WAN) == {1, 3, 4, 5}


def test_a_pao_subset_keeps_pdwf_and_rewrites_the_exclusion_and_count(tmp_path):
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    derived = write_derived(
        source,
        tmp_path,
        "SrO_r1",
        {"pao_subset": [{"species": "Sr", "orbitals": ("5s", "4d")}]},
    )
    assert method_of(derived) is Method.PDWF
    text = derived.win.read_text()
    assert "dis_froz_proj = .true." in text and "dis_proj_max" in text
    pw = derived.pw2wan.read_text()
    # Sr 4s, 4p and 5p out (1, 3-8); O untouched
    assert pao_excluded(pw) == {1, 3, 4, 5, 6, 7, 8}
    assert read_win(derived.win)["num_wann"] == "10"


def test_a_pao_label_the_projector_list_lacks_is_refused(tmp_path):
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    with pytest.raises(InputError, match="does not supply"):
        write_derived(
            source,
            tmp_path,
            "SrO_r1",
            {"pao_subset": [{"species": "Sr", "orbitals": ("6s",)}]},
        )


def test_no_writer_path_changes_the_family(tmp_path):
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    for knob, value in (
        ("projectors", [{"species": "Sr", "orbitals": ("d",)}]),
        ("dis_win_max", 6.0),
        ("scdm_mu", 3.0),
    ):
        with pytest.raises(InputError, match="switch_method"):
            write_derived(source, tmp_path, "SrO_r1", {knob: value})
    assert not (tmp_path / "SrO_r1.win").exists()


def test_a_pdwf_repair_moves_the_frozen_window_and_stays_pdwf(tmp_path):
    """The energy half of PD+ED is reachable without switch_method."""
    fx.write_deck(tmp_path)
    source = read_construction(tmp_path, fx.SEED)
    action = _validate(
        {"action": "modify_parameters", "changes": [_change("dis_froz_max", 1.5)]}
    )
    assert action.method is None and action.changes[0].parameter == "dis_froz_max"
    derived = write_derived(
        source,
        tmp_path,
        "SrO_r1",
        {"dis_froz_max": 1.5},
        fermi_energy_ev=0.0,
    )
    assert method_of(derived) is Method.PDWF
    values = assignments(derived.win.read_text())
    assert float(values["dis_froz_max"]) == 1.5
    assert values["dis_froz_proj"] == ".true."
    assert values["dis_proj_max"] == "0.95" and values["dis_proj_min"] == "0.01"


# ---------------------------------------------------------------------------
# starting decks
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    fx.write_deck(tmp_path)
    return DeckContext(
        code="qe",
        run_dir=tmp_path,
        parent=fx.parent(),
        parent_source=tmp_path / "parent.xml",
        frontier=read_construction(tmp_path, fx.SEED),
    )


def test_the_qe_pdwf_starting_deck_is_the_frontier_itself(tmp_path):
    ctx = _ctx(tmp_path)
    deck = starting_deck("pdwf", ctx)
    assert deck.win_text == fx.WIN
    assert (
        deck.values["dis_proj_min"] == "0.01" and deck.values["dis_proj_max"] == "0.95"
    )
    assert deck.values["pao_available"]["Sr"] == ["4s", "5s", "4p", "5p", "4d"]
    assert deck.values["pao_subset"][0] == {
        "species": "Sr",
        "orbitals": ["5s", "5p", "4d"],
    }


def test_the_qe_smv_starting_deck_keeps_the_paos_and_places_the_window_by_the_rule(
    tmp_path,
):
    ctx = _ctx(tmp_path)
    deck = starting_deck("smv", ctx)
    assert deck_method(deck.win_text, deck.pw2wan_text) is Method.SMV
    assert "atom_proj = .true." in deck.pw2wan_text
    assert "dis_proj_m" not in deck.win_text and "dis_froz_proj" not in deck.win_text
    # gapped parent: CBM (1 eV) + 2 eV, relative to E_F = 0
    assert deck.values["dis_froz_max"] == pytest.approx(3.0)
    assert any("conduction-band minimum + 2 eV" in n for n in deck.notes)


def test_the_qe_scdm_starting_deck_fits_mu_and_sigma_from_the_pao_projectability(
    tmp_path,
):
    ctx = _ctx(tmp_path)
    with pytest.raises(StartingDeckUnavailable, match="projectability"):
        starting_deck("scdm", ctx)
    fx.write_overlaps(tmp_path)
    ctx = _ctx(tmp_path)
    deck = starting_deck("scdm", ctx)
    assert deck_method(deck.win_text, deck.pw2wan_text) is Method.SCDM
    assert "scdm_entanglement = 'erfc'" in deck.pw2wan_text
    assert "atom_proj" not in deck.pw2wan_text
    assert "dis_num_iter = 0" in deck.win_text and "dis_froz_max" not in deck.win_text
    # the synthetic step: erfc((e - 8)/2)/2, so mu = 8 - 3*2 = 2 eV above E_F = 0
    assert deck.values["scdm_sigma"] == pytest.approx(2.0, abs=0.05)
    assert deck.values["scdm_mu"] == pytest.approx(2.0, abs=0.2)
    assert "frontier's own overlaps" in deck.notes[0]


def test_vasp_has_no_pdwf_starting_deck(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.code = "vasp"
    with pytest.raises(StartingDeckUnavailable, match="projectability"):
        starting_deck("pdwf", ctx)


# ---------------------------------------------------------------------------
# the loop: a switch is an attempt with a record, and a bad deck is refused first
# ---------------------------------------------------------------------------


def _run(tmp_path, script, total=3):
    fx.write_deck(tmp_path)
    runner = fx.RefusingEngine(tmp_path)
    model = fx.ToolScript(script)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "unused.xml",
        seed=fx.SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=total), draw_figures=False
        ),
        backend=fx.StoredParentQe(fx.parent()),
        log_path=tmp_path / "run.json",
    )
    return result, runner, model


def test_a_switch_is_an_attempt_built_from_the_starting_deck_and_recorded(tmp_path):
    switch = fx.answer(
        "switch_method",
        changes=[("dis_froz_max", 2.0)],
        method="smv",
        reason="the projectability window freezes too few states",
    )
    result, runner, model = _run(
        tmp_path,
        [
            ([("method_defaults", {"method": "smv"})], switch),
            ([], fx.answer("escalate")),
        ],
    )
    attempts = result.run.attempts
    assert [a.method for a in attempts] == ["pdwf", "smv"]
    assert [a.method_switched for a in attempts] == [False, True]
    assert attempts[1].switch == {
        "from": "pdwf",
        "to": "smv",
        "reason": "the projectability window freezes too few states",
    }
    assert [c["name"] for c in attempts[1].tools_called] == ["method_defaults"]
    deck = read_construction(tmp_path, attempts[1].seed)
    assert method_of(deck) is Method.SMV
    assert float(read_win(deck.win)["dis_froz_max"]) == pytest.approx(2.0)
    assert (tmp_path / f"{attempts[1].seed}_start.win").is_file()
    # a switch always recomputes the overlaps
    assert runner.builds[1] == (attempts[1].seed, None)
    log = json.loads((tmp_path / "run.json").read_text())
    assert log["attempts"][1]["switch"]["to"] == "smv"
    assert log["attempts"][1]["tools_called"][0]["name"] == "method_defaults"
    assert log["gate_profile"]["name"] == "default"


def test_a_deck_the_check_refuses_is_a_refused_attempt_before_anything_runs(tmp_path):
    # SMV with a frozen window holding every kept state: more frozen states than the
    # 13 functions at every k-point.
    overflow = fx.answer(
        "switch_method", changes=[("dis_froz_max", 40.0)], method="smv", reason="wider"
    )
    result, runner, model = _run(
        tmp_path, [([], overflow), ([], fx.answer("escalate"))]
    )
    attempts = result.run.attempts
    assert attempts[1].status == "deck_refused"
    assert "deck check refused it before anything ran" in attempts[1].calculation_error
    assert "more than num_wann = 13" in attempts[1].calculation_error
    assert [b[0] for b in runner.builds] == [
        fx.SEED
    ], "the refused deck reached the engine"
    # and the reason went back with the next request
    assert "more than num_wann = 13" in model.requests[1]["payload_text"]


def test_the_validate_deck_tool_says_the_same_before_the_attempt_is_spent(tmp_path):
    overflow = {
        "method": "smv",
        "changes": [{"parameter": "dis_froz_max", "new": 40.0}],
    }
    fine = {"method": "smv", "changes": [{"parameter": "dis_froz_max", "new": 2.0}]}
    cross = {
        "changes": [
            {"parameter": "projectors", "new": [{"species": "Sr", "orbitals": ["d"]}]}
        ]
    }
    result, runner, model = _run(
        tmp_path,
        [
            (
                [
                    ("validate_deck", overflow),
                    ("validate_deck", fine),
                    ("validate_deck", cross),
                ],
                fx.answer("escalate"),
            )
        ],
    )
    verdicts = [json.loads(text) for name, text in model.results]
    assert verdicts[0]["ok"] is False and "num_wann = 13" in verdicts[0]["problems"][0]
    assert verdicts[1]["ok"] is True and verdicts[1]["switch"] == "pdwf -> smv"
    assert verdicts[1]["gate_profile"] == "default"
    assert verdicts[2]["ok"] is False and "SMV knob" in verdicts[2]["problems"][0]
    assert len(result.run.attempts) == 1, "a tool call spent an attempt"
    assert [c["name"] for c in result.run.trailing_proposal["tools_called"]] == [
        "validate_deck"
    ] * 3


def test_the_prompt_carries_the_profile_target_and_the_family_guidance(tmp_path):
    result, runner, model = _run(tmp_path, [([], fx.answer("escalate"))])
    system = model.requests[0]["system"]
    assert 'passes the "default" gate profile' in system
    assert "THE THREE METHOD FAMILIES" in system and "switch_method" in system
    assert "Reason from the principles above and the evidence first" in system
    payload = model.requests[0]["payload_text"]
    assert "METHOD OF THIS CONSTRUCTION: PDWF" in payload
    editable = payload.split("PARAMETERS YOU MAY CHANGE")[1]
    assert "pao_subset" in editable and "dis_froz_max" in editable
    assert "dis_win_max" not in editable and "projectors" not in editable
    assert set(model.requests[0]["tools"]) >= {
        "validate_deck",
        "method_defaults",
        "frozen_capacity",
        "centres_and_spreads",
        "band_character",
        "reference_pages",
        "frontier_candidates",
        "search_references",
    }


def test_validate_deck_counts_a_pdwf_frozen_window_that_overfills(tmp_path):
    """A PDWF dis_froz_max that freezes more states than functions is refused free."""
    overflow = {"changes": [{"parameter": "dis_froz_max", "new": 40.0}]}
    fine = {"changes": [{"parameter": "dis_froz_max", "new": 1.5}]}
    result, runner, model = _run(
        tmp_path,
        [
            (
                [("validate_deck", overflow), ("validate_deck", fine)],
                fx.answer("escalate"),
            )
        ],
    )
    verdicts = [json.loads(text) for name, text in model.results]
    assert verdicts[0]["ok"] is False
    assert "more than num_wann = 13" in " ".join(verdicts[0]["problems"])
    assert verdicts[1]["ok"] is True, verdicts[1]["problems"]
    assert len(result.run.attempts) == 1
