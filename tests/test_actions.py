"""The action layer is the containment boundary, so its refusals are the test.

Each case here is a way an automated Wannierisation could stop being physics: a model
that moves the acceptance threshold instead of the calculation, one that cites a
measurement nobody took, one that reaches for a tool the current stage withholds, or
one that proposes a window describing an empty subspace. The schema is handed to the
provider as a request; these tests cover what happens when that request is not honoured.
"""

from __future__ import annotations

import json

import pytest

from autowan.actions import (
    ActionRejected,
    Stage,
    action_schema,
    validate_action,
)

CHECKS = ("mlwf_spread_functional_converged", "fermi_weighted_band_distance_vs_dft")
SPECIES = ("Fe",)

BASE = {
    "diagnosis": "the spread functional was still descending at the iteration limit",
    "evidence": ["mlwf_spread_functional_converged"],
    "expected_effect": "the minimisation reaches its convergence window",
    "confidence": 0.7,
}


def _validate(payload, stage=Stage.REPAIR, **kwargs):
    kwargs.setdefault("allowed_species", SPECIES)
    return validate_action(
        payload,
        stage=stage,
        evaluated_check_ids=CHECKS,
        **kwargs,
    )


def _change(parameter, new, old=None):
    return {"parameter": parameter, "old": old, "new": new, "reason": "because"}


def test_a_well_formed_repair_is_accepted_and_routed_to_the_right_file():
    action = _validate(
        {
            **BASE,
            "action": "modify_parameters",
            "changes": [_change("num_iter", 20000, 1000)],
        }
    )
    assert action.kind == "modify_parameters"
    assert action.changes_by_file() == {"win": {"num_iter": 20000}, "pw2wan": {}}


def test_the_repair_stage_offers_three_actions_and_the_tools_are_not_among_them():
    schema = action_schema(Stage.REPAIR, CHECKS)
    assert schema["properties"]["action"]["enum"] == [
        "modify_parameters",
        "switch_method",
        "escalate",
    ]
    # Reading the corpus is a read-only tool now, called before the answer.
    assert "query" not in schema["properties"]


def test_a_tool_returned_as_an_action_is_refused():
    with pytest.raises(ActionRejected, match="not available in stage repair"):
        _validate({**BASE, "action": "search_references"})


def test_the_automatic_stage_consults_no_model():
    with pytest.raises(ActionRejected, match="consults no model"):
        action_schema(Stage.AUTOMATIC, CHECKS)


def test_a_gate_threshold_is_not_reachable_from_the_vocabulary():
    with pytest.raises(ActionRejected, match="not an editable parameter"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [_change("gate_floor_mev", 500)],
            }
        )


def test_evidence_must_name_a_check_the_gate_actually_evaluated():
    with pytest.raises(ActionRejected, match="cites no check the gate evaluated"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "evidence": ["a_check_nobody_ran"],
                "changes": [_change("num_iter", 5000)],
            }
        )


def test_a_frozen_window_outside_the_outer_window_is_refused():
    with pytest.raises(ActionRejected, match="frozen window must sit inside"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [
                    _change("dis_froz_max", 40.0),
                    _change("dis_win_max", 20.0),
                ],
            }
        )


def test_projectors_are_checked_against_the_structure():
    ok = _validate(
        {
            **BASE,
            "action": "modify_parameters",
            "changes": [
                _change("projectors", [{"species": "Fe", "orbitals": ["s", "d"]}])
            ],
        }
    )
    assert ok.changes[0].new == ({"species": "Fe", "orbitals": ("s", "d")},)

    with pytest.raises(ActionRejected, match="not in this structure"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [
                    _change("projectors", [{"species": "Ni", "orbitals": ["d"]}])
                ],
            }
        )


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"action": "modify_parameters", "changes": []}, "carries no change"),
        (
            {"action": "escalate", "changes": [_change("num_iter", 100)]},
            "may not carry parameter changes",
        ),
        (
            {
                "action": "modify_parameters",
                "changes": [_change("num_iter", 100), _change("num_iter", 200)],
            },
            "assigned twice",
        ),
        (
            {
                "action": "modify_parameters",
                "confidence": 1.4,
                "changes": [_change("num_iter", 1)],
            },
            r"outside \[0, 1\]",
        ),
        (
            {
                "action": "modify_parameters",
                "shell": "rm -rf /",
                "changes": [_change("num_iter", 1)],
            },
            "unexpected field",
        ),
        (
            {"action": "modify_parameters", "changes": [_change("num_iter", 1000.5)]},
            "expected an integer",
        ),
        (
            {"action": "modify_parameters", "changes": [_change("dis_mix_ratio", 1.7)]},
            "above the allowed maximum",
        ),
    ],
)
def test_malformed_actions_are_refused(payload, expected):
    merged = {**BASE, **payload}
    with pytest.raises(ActionRejected, match=expected):
        _validate(merged)


UNRESOLVED = ("mlwf_spread_functional_converged",)


def _validate_targeted(payload):
    return validate_action(
        payload,
        stage=Stage.REPAIR,
        evaluated_check_ids=CHECKS,
        allowed_species=SPECIES,
        unresolved_check_ids=UNRESOLVED,
    )


def test_a_repair_must_name_which_unresolved_check_it_addresses():
    """Evidence and target answer different questions and must not collapse together.

    Asked only for evidence, a model returned the six checks that had *passed* — a
    defensible reading of "what supports your diagnosis" and useless as a record of
    what the repair was for.
    """
    action = _validate_targeted(
        {
            **BASE,
            "action": "modify_parameters",
            "targets": ["mlwf_spread_functional_converged"],
            "changes": [_change("num_iter", 20000, 1000)],
        }
    )
    assert action.evidence == ("mlwf_spread_functional_converged",)
    assert action.targets == ("mlwf_spread_functional_converged",)


def test_a_repair_may_not_target_a_check_that_was_never_in_trouble():
    with pytest.raises(ActionRejected, match="did not leave unresolved"):
        _validate_targeted(
            {
                **BASE,
                "action": "modify_parameters",
                "targets": ["fermi_weighted_band_distance_vs_dft"],
                "changes": [_change("num_iter", 20000)],
            }
        )


def test_the_target_is_required_whenever_something_is_unresolved():
    with pytest.raises(ActionRejected, match="targets"):
        _validate_targeted(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [_change("num_iter", 20000)],
            }
        )


def test_the_schema_offers_only_the_unresolved_checks_as_targets():
    schema = action_schema(Stage.REPAIR, CHECKS, unresolved_check_ids=UNRESOLVED)
    assert schema["properties"]["targets"]["items"]["enum"] == list(UNRESOLVED)
    assert "targets" in schema["required"]


def test_a_reading_of_the_numbers_is_kept_as_an_observation_not_refused():
    # The whole action used to be refused when one evidence entry named a relation
    # between measurements rather than a measurement. It is now filed as an
    # observation, provided at least one real check is still cited.
    action = _validate(
        {
            **BASE,
            "action": "modify_parameters",
            "evidence": [CHECKS[0], "spread_ratio_widest_to_narrowest"],
            "changes": [_change("dis_froz_max", 3.0)],
        }
    )
    assert action.evidence == (CHECKS[0],)
    assert "spread_ratio_widest_to_narrowest" in action.observations


def test_the_structures_species_are_enumerated_in_the_schema():
    schema = action_schema(Stage.REPAIR, CHECKS, allowed_species=("Ba",))
    text = json.dumps(schema)
    assert '"enum": ["Ba"]' in text


def test_a_projector_set_wider_than_the_bands_is_refused_with_the_bound():
    with pytest.raises(ActionRejected, match="bands to draw them from"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [
                    _change(
                        "projectors", [{"species": "Fe", "orbitals": ["s", "p", "d"]}]
                    )
                ],
            },
            species_counts={"Fe": 2},
            band_count=8,
            allowed_species=("Fe",),
        )


def test_num_bands_cannot_reach_past_the_parent_spectrum():
    with pytest.raises(ActionRejected, match="more than the parent calculation"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [_change("num_bands", 64)],
            },
            parent_band_count=40,
        )


def test_projectors_and_num_bands_raised_together_are_judged_against_the_new_ceiling():
    action = _validate(
        {
            **BASE,
            "action": "modify_parameters",
            "changes": [
                _change("projectors", [{"species": "Fe", "orbitals": ["s", "p", "d"]}]),
                _change("num_bands", 20),
            ],
        },
        species_counts={"Fe": 2},
        band_count=4,
        parent_band_count=40,
        allowed_species=("Fe",),
    )
    assert {c.parameter for c in action.changes} == {"projectors", "num_bands"}


def test_a_shell_named_twice_on_one_species_is_refused():
    with pytest.raises(ActionRejected, match="names an orbital twice"):
        _validate(
            {
                **BASE,
                "action": "modify_parameters",
                "changes": [
                    _change(
                        "projectors", [{"species": "Fe", "orbitals": ["d", "d", "s"]}]
                    )
                ],
            },
            allowed_species=("Fe",),
        )
