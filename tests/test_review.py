"""The review stage may withhold acceptance; it may not grant it, and it may not shrug.

A language model's reading of a figure is a second opinion on work the deterministic
gate has already accepted. These cover the two rules that keep it that way: a refusal
has to name something a repair can act on, and a gate configured to look at nothing
reviews nothing.
"""

from __future__ import annotations

import pytest

from autowan.providers import ProviderError
from autowan.review_gate import (
    DEFAULT_CONCERNS,
    REVIEW_SCHEMA,
    ReviewGate,
)


class _StubProvider:
    """Returns a fixed answer without leaving the machine."""

    def __init__(self, content):
        self.content = content
        self.seen_images = None

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        from autowan.providers import ModelResponse

        self.seen_images = images
        return ModelResponse(content=self.content, model="stub", raw_text="")


class _Image:
    pass


def _gate(content, concerns=DEFAULT_CONCERNS):
    return ReviewGate(_StubProvider(content), concerns=concerns), None


def test_a_gate_with_no_concerns_is_refused_at_construction():
    with pytest.raises(ValueError, match="reviews nothing"):
        ReviewGate(_StubProvider({}), concerns=())


def test_reviewing_without_figures_is_refused():
    gate, _ = _gate(
        {
            "accept": True,
            "observations": [{"concern": "c", "seen": "satisfied", "where": "w"}],
        }
    )
    with pytest.raises(ValueError, match="no figures"):
        gate.review(figures=[], captions=[], signature={})


def test_a_refusal_that_names_no_fault_is_a_provider_failure_not_a_verdict():
    """A repair loop cannot act on 'something looks wrong'."""
    gate, _ = _gate(
        {
            "accept": False,
            "observations": [{"concern": "c", "seen": "violated", "where": "w"}],
        }
    )
    with pytest.raises(ProviderError, match="without naming a fault"):
        gate.review(figures=[_Image()], captions=["c"], signature={})


def test_a_refusal_naming_a_fault_is_returned_with_its_violations():
    gate, _ = _gate(
        {
            "accept": False,
            "fault": "the interpolated manifold detaches from the parent across half the zone",
            "observations": [
                {
                    "concern": DEFAULT_CONCERNS[1],
                    "seen": "violated",
                    "where": "right panel",
                },
                {
                    "concern": DEFAULT_CONCERNS[0],
                    "seen": "satisfied",
                    "where": "left panel",
                },
            ],
        }
    )
    outcome = gate.review(figures=[_Image()], captions=["c"], signature={})
    assert outcome.accepted is False
    assert outcome.fault
    assert outcome.violations == (DEFAULT_CONCERNS[1],)


def test_acceptance_carries_the_observations_it_rests_on():
    gate, _ = _gate(
        {
            "accept": True,
            "observations": [
                {"concern": c, "seen": "satisfied", "where": "figure"}
                for c in DEFAULT_CONCERNS
            ],
        }
    )
    outcome = gate.review(figures=[_Image()], captions=["c"], signature={})
    assert outcome.accepted is True
    assert outcome.violations == ()
    assert len(outcome.observations) == len(DEFAULT_CONCERNS)


def test_the_schema_forces_an_observation_per_concern_and_a_closed_vocabulary():
    props = REVIEW_SCHEMA["properties"]
    assert REVIEW_SCHEMA["additionalProperties"] is False
    assert props["observations"]["minItems"] == 1
    item = props["observations"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["seen"]["enum"] == [
        "satisfied",
        "violated",
        "not_visible",
    ]


def test_the_figures_are_passed_to_the_model():
    provider = _StubProvider(
        {
            "accept": True,
            "observations": [{"concern": "c", "seen": "satisfied", "where": "w"}],
        }
    )
    gate = ReviewGate(provider)
    images = [_Image(), _Image()]
    gate.review(figures=images, captions=["a", "b"], signature={"x": 1})
    assert provider.seen_images == images
