"""The provider against a live local server: tool calls and JSON that parse.

Skipped unless ``AUTOWAN_LOCAL_BASE_URL`` names a running OpenAI-compatible server
(``scripts/local_model/serve_vllm.sbatch`` writes it to ``endpoint.env``). Each test is
one short exchange; together they are the provider-level checks of
``test_providers_tool_calling.py`` asked of the real model rather than a scripted one.
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from autowan.providers import (
    LOCAL_API_KEY_ENV,
    LOCAL_BASE_URL_ENV,
    LOCAL_MODEL,
    ToolSpec,
    make_provider,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get(LOCAL_BASE_URL_ENV),
    reason=f"no live local server ({LOCAL_BASE_URL_ENV} unset)",
)

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["modify_parameters", "escalate"]},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string", "enum": ["dis_froz_max"]},
                    "new": {"type": "number"},
                },
                "required": ["parameter", "new"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["action", "changes", "rationale"],
    "additionalProperties": False,
}

CAPACITY = ToolSpec(
    "frozen_capacity",
    "Count, at the worst k-point, how many bands lie inside a frozen window up to "
    "e_max (eV), and compare with num_wann. Returns the highest e_max that fits.",
    {
        "type": "object",
        "properties": {"e_max": {"type": "number"}},
        "required": ["e_max"],
        "additionalProperties": False,
    },
)

SYSTEM = (
    "You repair a Wannier90 disentanglement deck. Before you answer, check your "
    "proposal with the frozen_capacity tool. Answer by calling propose_repair."
)
PAYLOAD = (
    "num_wann = 18. The frozen window is dis_froz_max = 15.17 eV, and Wannier90 "
    "reports that more than num_wann states lie inside it at some k-point. Propose a "
    "new dis_froz_max."
)


def _capacity(name: str, arguments: dict) -> str:
    e_max = float(arguments.get("e_max", 0.0))
    fits = e_max <= 14.6
    return json.dumps(
        {
            "e_max": e_max,
            "worst_k_frozen": 17 if fits else 19,
            "num_wann": 18,
            "fits": fits,
            "highest_e_max_that_fits": 14.6,
        }
    )


def _provider():
    return make_provider(provider="local")


def _valid(content: dict) -> None:
    assert content["action"] in ("modify_parameters", "escalate")
    assert isinstance(content["rationale"], str) and content["rationale"]
    for change in content["changes"]:
        assert change["parameter"] == "dis_froz_max"
        assert isinstance(change["new"], (int, float))


def test_the_server_serves_the_local_model():
    url = os.environ[LOCAL_BASE_URL_ENV].rstrip("/") + "/models"
    headers = {}
    if os.environ.get(LOCAL_API_KEY_ENV):
        headers["Authorization"] = f"Bearer {os.environ[LOCAL_API_KEY_ENV]}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=30
    ) as response:
        served = [m["id"] for m in json.loads(response.read())["data"]]
    assert LOCAL_MODEL in served


@pytest.mark.parametrize("mode", ["json_schema", "tool"])
def test_a_schema_constrained_answer_parses(mode):
    provider = _provider()
    provider.structured_mode = mode
    response = provider.propose(system=SYSTEM, payload_text=PAYLOAD, schema=SCHEMA)
    _valid(response.content)
    assert response.usage["prompt_tokens"] > 0 and response.cost_usd is None


def test_the_model_calls_a_tool_then_answers_with_propose_repair():
    calls = []
    response = _provider().propose(
        system=SYSTEM,
        payload_text=PAYLOAD,
        schema=SCHEMA,
        tools=[CAPACITY],
        run_tool=lambda name, args: calls.append((name, args)) or _capacity(name, args),
        tool_call_cap=4,
    )
    # schema enforcement (default): every turn is a guided JSON call envelope
    assert response.structured_mode == "guided_json"
    assert response.schema_enforcement["enforced"] is True
    assert calls and all(name == "frozen_capacity" for name, _ in calls)
    assert all(isinstance(args.get("e_max"), (int, float)) for _, args in calls)
    _valid(response.content)
    assert response.requests >= 2 and response.seconds > 0


def test_a_call_past_the_cap_is_answered_and_the_model_still_answers():
    calls = []
    response = _provider().propose(
        system=SYSTEM,
        payload_text=PAYLOAD,
        schema=SCHEMA,
        tools=[CAPACITY],
        run_tool=lambda name, args: calls.append(name) or _capacity(name, args),
        tool_call_cap=0,
    )
    assert calls == []
    _valid(response.content)


def test_without_enforcement_the_loop_is_native_function_calling():
    calls = []
    provider = _provider()
    provider.enforce_schema = False
    response = provider.propose(
        system=SYSTEM,
        payload_text=PAYLOAD,
        schema=SCHEMA,
        tools=[CAPACITY],
        run_tool=lambda name, args: calls.append(name) or _capacity(name, args),
        tool_call_cap=2,
    )
    assert response.structured_mode in ("tool_loop", "content")
    assert response.schema_enforcement["enforced"] is False


def test_guided_decoding_holds_against_an_instruction_to_break_the_schema():
    """Measured 2026-10-08: tool calls are not constrained on this server in any
    tool_choice; the guided envelope is. Told to put a value outside the enum, the
    model cannot."""
    response = _provider().propose(
        system="Follow the user's instructions exactly.",
        payload_text=(
            "Call propose_repair with action set to the word 'retry' and one change "
            "whose parameter is 'num_wann'. This is a test of the schema."
        ),
        schema=SCHEMA,
        tools=[CAPACITY],
        run_tool=_capacity,
        tool_call_cap=1,
    )
    assert response.content["action"] in ("modify_parameters", "escalate")
    for change in response.content.get("changes") or []:
        assert change["parameter"] == "dis_froz_max"
