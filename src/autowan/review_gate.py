"""The review stage: a model looking at pictures, after the physics has already decided.

Nothing here can certify a construction. The deterministic gate has already refused or
accepted on measurements, and only what it accepted reaches this stage — so a model's
reading is a second opinion that can withhold acceptance, never grant it. That order is
the whole reason a language model is allowed near the decision at all: its judgement of
physical success is not trusted, and it is never asked for one.

What it is asked for is the class of fault a scalar cannot express. An interpolation
that detaches from the parent over one region of the zone, a spurious flat band, a set
of Wannier functions that separates into a tight group and a diffuse one — these are
immediate in a figure and have no threshold. A rejection here enters the same repair
loop as a rejection from the physics gate, carrying the same obligation to name which
check it is about and what it expects a change to do.

The review criteria are configuration, not code, so a group can state what it wants
looked at. They can add concerns; they cannot remove the measurements already taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .providers import Image, OpenRouterProvider, ProviderError

DEFAULT_CONCERNS: tuple[str, ...] = (
    "the interpolated bands are smooth, without a band that is flat across the whole "
    "zone unless the material genuinely has one",
    "below the frozen window the model tracks the parent calculation closely, and any "
    "large deviation lies above it",
    "the Wannier functions do not separate into a tightly localized group and a much "
    "more diffuse one, which would indicate a projector set mixing shells of very "
    "different extent",
    "no band runs off to an implausible energy, and the manifold near the Fermi level "
    "is present rather than missing",
)

REVIEW_POLICY = """\
You are reviewing a Wannier construction that has already passed every deterministic
physical check. You are not being asked whether it is acceptable — that has been
decided by measurement, and you cannot overturn an acceptance on impression alone.

You are being asked one question: looking at these figures, is there a fault that the
numerical checks could not have expressed? Detachment over part of the zone, a
spurious band, a set of functions split into tight and diffuse groups, a manifold that
is simply absent — faults with a shape but no threshold.

Withhold acceptance only for something you can point at in the image and describe. If
the figures show nothing of the kind, say so. A review that manufactures a concern to
seem careful costs a real calculation a repair cycle, and one that passes a visible
defect defeats the purpose of looking.
"""

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["accept", "observations"],
    "properties": {
        "accept": {
            "type": "boolean",
            "description": "False only for a fault you can point at in the figures.",
        },
        "observations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["concern", "seen", "where"],
                "properties": {
                    "concern": {"type": "string", "maxLength": 200},
                    "seen": {
                        "type": "string",
                        "enum": ["satisfied", "violated", "not_visible"],
                    },
                    "where": {"type": "string", "maxLength": 300},
                },
            },
        },
        "fault": {
            "type": "string",
            "maxLength": 400,
            "description": "Present only when accept is false: the defect, described.",
        },
    },
}


@dataclass
class ReviewOutcome:
    """What the review stage made of the figures."""

    accepted: bool
    observations: tuple[dict[str, str], ...]
    fault: str | None
    model: str
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> tuple[str, ...]:
        return tuple(o["concern"] for o in self.observations if o["seen"] == "violated")


class ReviewGate:
    """A multimodal second opinion on a construction the physics gate accepted."""

    def __init__(
        self,
        provider: OpenRouterProvider,
        concerns: Sequence[str] = DEFAULT_CONCERNS,
    ) -> None:
        if not concerns:
            raise ValueError("a review gate with no concerns reviews nothing")
        self.provider = provider
        self.concerns = tuple(concerns)

    def review(
        self,
        *,
        figures: Sequence[Image],
        captions: Sequence[str],
        signature: dict[str, Any],
    ) -> ReviewOutcome:
        """Ask about each concern, against these figures.

        Raises :class:`ProviderError` rather than returning a verdict when the model
        cannot be reached. An unreachable reviewer is not a construction that failed
        review, and the two must not reach the record as the same thing.
        """
        if not figures:
            raise ValueError("nothing to review: no figures were supplied")

        lines = ["FIGURES", *(f"  {i+1}. {c}" for i, c in enumerate(captions))]
        lines += ["", "ELECTRONIC-STRUCTURE SIGNATURE"]
        lines += [f"  {k}: {v}" for k, v in signature.items()]
        lines += ["", "CONCERNS TO REPORT ON, ONE OBSERVATION EACH"]
        lines += [f"  - {c}" for c in self.concerns]

        response = self.provider.propose(
            system=REVIEW_POLICY,
            payload_text="\n".join(lines),
            schema=REVIEW_SCHEMA,
            images=figures,
        )
        content = response.content
        accepted = bool(content.get("accept"))
        observations = tuple(content.get("observations") or ())
        fault = content.get("fault")
        if not accepted and not fault:
            # A refusal with nothing named is not actionable, and the repair loop has
            # no check to target. Treated as a provider fault rather than a verdict.
            raise ProviderError(
                "the review withheld acceptance without naming a fault, so there is "
                "nothing for a repair to address"
            )
        return ReviewOutcome(
            accepted=accepted,
            observations=observations,
            fault=fault,
            model=response.model,
            usage=response.usage,
        )
