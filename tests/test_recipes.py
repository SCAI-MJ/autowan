"""Tests for the reference corpus and retrieval over it."""

from __future__ import annotations

import json

from autowan.recipes import Recipe, distil_run, load, render, search, write

METAL = {
    "manifold_is_isolated": False,
    "band_gap_ev": 0.0,
    "bands_crossing_fermi_level": 3,
    "atoms_in_cell": 1,
    "occupied_bandwidth_ev": 5.0,
    "species": {"Na": 1},
}
INSULATOR = {
    "manifold_is_isolated": True,
    "band_gap_ev": 3.3,
    "bands_crossing_fermi_level": 0,
    "atoms_in_cell": 2,
    "occupied_bandwidth_ev": 18.0,
    "species": {"Sr": 1, "O": 1},
}


def _outcome(passed, failed, **evaluable):
    measurements = {
        c: {"applies": True, "evaluable": evaluable.get(c, True), "value": 1.0}
        for c in ("a", "b", "c")
    }
    return {
        "passed": passed,
        "failed_check_ids": list(failed),
        "evaluated_check_ids": ["a", "b", "c"],
        "measurements": measurements,
    }


def _action(param, new, reason="because"):
    return {
        "diagnosis": "d",
        "changes": [{"parameter": param, "old": None, "new": new, "reason": reason}],
    }


def test_only_a_step_that_resolved_something_becomes_an_entry():
    log = {
        "material": "Na",
        "attempts": [
            {"index": 0, "outcome": _outcome(False, ["a", "b"]), "action": None},
            # resolves nothing: still stuck on the same two
            {
                "index": 1,
                "outcome": _outcome(False, ["a", "b"]),
                "action": _action("num_iter", 8000),
            },
            {
                "index": 2,
                "outcome": _outcome(False, ["b"]),
                "action": _action(
                    "projectors", [{"species": "Na", "orbitals": ["s", "p"]}]
                ),
            },
        ],
    }
    entries = distil_run(log, METAL)
    assert len(entries) == 1
    assert entries[0].provenance["attempt"] == 2
    assert entries[0].resolved == ("a",)
    assert entries[0].still_unresolved == ("b",)


def test_a_check_that_could_not_be_evaluated_counts_as_unresolved():
    # A check with no verdict is not a pass; a change that gives it one has resolved it.
    log = {
        "material": "Ti",
        "attempts": [
            {"index": 0, "outcome": _outcome(False, [], b=False), "action": None},
            {
                "index": 1,
                "outcome": _outcome(True, []),
                "action": _action("conv_window", 3),
            },
        ],
    }
    entries = distil_run(log, METAL)
    assert entries[0].resolved == ("b",)


def test_the_comparison_skips_an_attempt_that_never_ran():
    # An attempt whose calculation failed has no verdicts, so the change after it is
    # compared against the last attempt that produced any.
    log = {
        "material": "Ba",
        "attempts": [
            {"index": 0, "outcome": _outcome(False, ["a"]), "action": None},
            {
                "index": 1,
                "outcome": None,
                "action": _action("dis_froz_max", -2.0),
                "calculation_error": "the deck was refused",
            },
            {
                "index": 2,
                "outcome": _outcome(False, []),
                "action": _action("projectors", []),
            },
        ],
    }
    entries = distil_run(log, METAL)
    assert [e.provenance["attempt"] for e in entries] == [2]


def _recipe(rid, sig, resolved, param="projectors"):
    return Recipe(
        recipe_id=rid,
        signature=sig,
        resolved=tuple(resolved),
        still_unresolved=(),
        changes=({"parameter": param, "old": None, "new": "x", "reason": "r"},),
        diagnosis="the projector set was too narrow",
        provenance={"material": rid},
    )


def test_an_entry_that_resolved_nothing_the_caller_is_stuck_on_is_not_a_weaker_match():
    corpus = [
        _recipe("far", INSULATOR, ["centres"]),
        _recipe("wrong-question", METAL, ["convergence"]),
    ]
    found = search(corpus, signature=METAL, unresolved=["centres"])
    assert [r.recipe_id for r, _ in found] == ["far"]


def test_ranking_is_by_the_physics_and_not_by_a_shared_name():
    near = _recipe("near", {**METAL, "species": {"K": 1}}, ["centres"])
    far = _recipe("far", {**INSULATOR, "species": {"Na": 1}}, ["centres"])
    found = search([far, near], signature=METAL, unresolved=["centres"])
    assert [r.recipe_id for r, _ in found] == ["near", "far"]


def test_a_malformed_entry_costs_one_recipe_and_not_the_stage(tmp_path):
    write(_recipe("good", METAL, ["centres"]), tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "incomplete.json").write_text(json.dumps({"signature": {}}))
    assert [r.recipe_id for r in load(tmp_path)] == ["good"]


def test_nothing_retrieved_says_so_rather_than_rendering_an_empty_section():
    assert "nothing in the corpus" in render([])


def test_a_change_that_made_an_unrunnable_deck_run_is_an_entry():
    # The most useful thing the corpus can hold, and the comparison used to discard it:
    # an attempt that produced no verdicts was not a state to compare against, so a
    # deck that went from refused to certified contributed nothing.
    log = {
        "material": "Na",
        "attempts": [
            {
                "index": 0,
                "outcome": None,
                "action": None,
                "calculation_error": "dis_windows: More states in the frozen window",
            },
            {
                "index": 1,
                "outcome": None,
                "action": _action("num_bands", 9),
                "calculation_error": "dis_windows: More states in the frozen window",
            },
            {
                "index": 2,
                "outcome": _outcome(True, []),
                "action": _action(
                    "projectors", [{"species": "Na", "orbitals": ["s", "p"]}]
                ),
            },
        ],
    }
    entries = distil_run(log, METAL)
    assert [e.provenance["attempt"] for e in entries] == [2]
    assert entries[0].resolved == ("calculation_did_not_run",)


def test_a_failed_attempt_adds_to_what_was_known_rather_than_replacing_it():
    # The verdicts from before the failure still stand -- nothing has contradicted
    # them -- so a repair that clears both is recorded as clearing both.
    log = {
        "material": "SrO",
        "attempts": [
            {"index": 0, "outcome": _outcome(False, ["a"]), "action": None},
            {
                "index": 1,
                "outcome": None,
                "action": _action("projectors", []),
                "calculation_error": "the overlaps exited 1",
            },
            {
                "index": 2,
                "outcome": _outcome(True, []),
                "action": _action("dis_froz_max", 3.5),
            },
        ],
    }
    entries = distil_run(log, INSULATOR)
    assert set(entries[0].resolved) == {"a", "calculation_did_not_run"}


def test_an_entry_from_this_material_is_held_back():
    # It is a memory of an earlier run here, not a reference from elsewhere; returning
    # it measures recall where the stage measures transfer.
    mine = _recipe("Ti_pdwf-2", METAL, ["centres"])
    mine.provenance["material"] = "Ti_pdwf"
    theirs = _recipe("Ba_pdwf-4", METAL, ["centres"])
    theirs.provenance["material"] = "Ba_pdwf"
    found = search(
        [mine, theirs],
        signature=METAL,
        unresolved=["centres"],
        exclude_material="Ti_pdwf",
    )
    assert [r.recipe_id for r, _ in found] == ["Ba_pdwf-4"]


def test_what_the_model_reads_never_names_the_material():
    # Shown an identifier with the material in it, the model reasoned from the formula
    # rather than from the signature -- the one thing this stage is meant to avoid.
    log = {
        "material": "SrTiO3_pdwf",
        "attempts": [
            {"index": 0, "outcome": _outcome(False, ["a"]), "action": None},
            {
                "index": 1,
                "outcome": _outcome(True, []),
                "action": _action("projectors", [{"species": "Sr", "orbitals": ["s"]}]),
            },
        ],
    }
    entry = distil_run(log, INSULATOR)[0]
    assert "SrTiO3" not in entry.recipe_id
    assert entry.provenance["material"] == "SrTiO3_pdwf"
    shown = render([(entry, 1.0)])
    assert "SrTiO3" not in shown
    assert "Sr" in shown, "the species a projector names is physics, and stays"


def test_the_stage_records_what_it_searched_for_even_when_nothing_came_back():
    # A retrieval that shaped a repair is part of the repair's provenance, and one that
    # found nothing is part of the record of what the stage could not do.
    from autowan.orchestrator import Run, RunConfig

    run = Run("m", RunConfig())
    run.searches.append({"query": "centres off site", "retrieved": [], "scores": []})
    assert run.to_dict()["searches"] == [
        {"query": "centres off site", "retrieved": [], "scores": []}
    ]
