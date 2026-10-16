"""Schema enforcement on every provider (Fig. 3 fairness).

Measured 2026-10-07: in ``auto`` tool choice, vLLM grammar-constrains nothing, and a 9B
model broke the ``evidence`` enum on 4 answers of 4. So that every model in the model
comparison answers under the same constraint, the provider enforces the action and tool
schemas at decode time, by default:

* OpenRouter (Claude, GPT, Gemini, open-weight): every function carries ``"strict":
  true`` with a strict-dialect schema -- Anthropic strict tool use, OpenAI strict
  function calling, passed through by OpenRouter;
* a vLLM server: ``tool_choice: "required"``, which vLLM guided-decodes against the
  offered tools' JSON schemas.

Whether enforcement was requested, by what mechanism, and whether each taken answer was
enforced is recorded per attempt and per run. No network: the scripted server of
``test_providers_tool_calling``.
"""

from __future__ import annotations

import pytest
from test_providers_tool_calling import _call, _chat, _Server

from autowan import providers
from autowan.actions import Stage, action_schema, validate_action
from autowan.exits import EXIT_COLUMNS, exit_row
from autowan.providers import (
    ENFORCEMENT_GUIDED,
    ENFORCEMENT_STRICT,
    ModelResponse,
    ModelSettings,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    ProviderSchemaViolation,
    ToolSpec,
    strict_schema,
)

TOOLS = [
    ToolSpec(
        "frozen_capacity",
        "d",
        {
            "type": "object",
            "required": ["e_max"],
            "properties": {
                "e_max": {"type": "number", "minimum": -50},
                "e_min": {"type": "number"},
            },
        },
    )
]
ACTION = action_schema(
    Stage.REPAIR, ("a", "b"), ("a",), ("Na",), code="qe", method="smv"
)
ANSWER = {
    "action": "escalate",
    "diagnosis": "d",
    "evidence": ["a"],
    "observations": None,
    "expected_effect": "e",
    "confidence": 0.5,
    "changes": None,
    "method": None,
    "reason": None,
    "num_bands_needed": None,
    "targets": ["a"],
}


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")


def _walk(node, seen=()):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


# ---------------------------------------------------------------------------
# the strict dialect
# ---------------------------------------------------------------------------


def test_strict_schema_closes_every_object_and_requires_every_property():
    strict = strict_schema(ACTION)
    for node in _walk(strict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for dropped in ("maxLength", "minimum", "maximum", "minItems"):
            assert dropped not in node


def test_optional_properties_become_nullable_and_required_ones_do_not():
    strict = strict_schema(ACTION)["properties"]
    assert strict["diagnosis"] == {"type": "string"}
    assert {"type": "null"} in strict["changes"]["anyOf"]
    assert {"type": "null"} in strict["num_bands_needed"]["anyOf"]
    assert {"type": "null"} not in strict["evidence"].get("anyOf", [])


def test_an_any_value_property_is_dropped_when_optional_and_refused_when_required():
    change = strict_schema(ACTION)["properties"]["changes"]["anyOf"][0]["items"]
    assert "old" not in change["properties"]  # "old": {} has no strict form
    with pytest.raises(ValueError, match="any value"):
        strict_schema({"type": "object", "required": ["x"], "properties": {"x": {}}})


def test_the_original_schema_is_not_modified():
    before = repr(ACTION)
    strict_schema(ACTION)
    assert repr(ACTION) == before


def test_a_strict_shaped_answer_with_nulls_validates():
    action = validate_action(
        ANSWER,
        stage=Stage.REPAIR,
        evaluated_check_ids=("a", "b"),
        unresolved_check_ids=("a",),
    )
    assert action.kind == "escalate" and action.changes == ()


def test_the_tool_schemas_have_a_strict_form():
    """Every read-only tool, including validate_deck's ``new`` (was an any-value {})."""
    from autowan.tools import _change_entries

    entries = strict_schema(
        {"type": "object", "properties": {"changes": _change_entries(("Na",))}}
    )
    new = entries["properties"]["changes"]["anyOf"][0]["items"]["properties"]["new"]
    assert new["anyOf"] and all(alt for alt in new["anyOf"])


# ---------------------------------------------------------------------------
# what is sent
# ---------------------------------------------------------------------------


def test_openrouter_sends_strict_functions_and_keeps_auto(key, monkeypatch):
    server = _Server(
        [_chat({"content": "", "tool_calls": [_call("propose_repair", ANSWER)]})],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    response = OpenRouterProvider().propose(
        system="s",
        payload_text="p",
        schema=ACTION,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=2,
    )
    body = server.bodies[0]
    assert body["tool_choice"] == "auto"  # Anthropic thinking refuses a forced choice
    for tool in body["tools"]:
        function = tool["function"]
        assert function["strict"] is True
        assert function["parameters"]["additionalProperties"] is False
    final = next(t for t in body["tools"] if t["function"]["name"] == "propose_repair")
    assert final["function"]["parameters"] == strict_schema(ACTION)
    assert response.schema_enforcement == {
        "requested": True,
        "mechanism": ENFORCEMENT_STRICT,
        "enforced": True,
    }


def test_under_enforcement_a_json_answer_in_text_is_not_taken(key, monkeypatch):
    server = _Server(
        [
            _chat({"content": '{"action": "escalate"}'}),
            _chat({"content": "", "tool_calls": [_call("propose_repair", ANSWER)]}),
        ],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    response = OpenRouterProvider().propose(
        system="s",
        payload_text="p",
        schema=ACTION,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=2,
    )
    assert response.structured_mode == "tool_loop" and response.requests == 2
    assert "propose_repair" in server.bodies[1]["messages"][-1]["content"]
    assert response.schema_enforcement["enforced"] is True


def test_prose_three_times_under_enforcement_is_a_schema_violation(key, monkeypatch):
    server = _Server([_chat({"content": '{"action": "escalate"}'})] * 3, balance=50.0)
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    with pytest.raises(ProviderSchemaViolation):
        OpenRouterProvider().propose(
            system="s",
            payload_text="p",
            schema=ACTION,
            tools=TOOLS,
            run_tool=lambda n, a: "",
            tool_call_cap=2,
        )


def _envelope(name, arguments):
    import json

    return {"content": json.dumps({"call": {"name": name, "arguments": arguments}})}


def test_vllm_is_guided_by_a_response_format_envelope(monkeypatch):
    """Tool calls are never constrained on vLLM 0.31 + qwen3_coder (measured); the
    loop sends every turn as a response_format JSON schema over the functions."""
    server = _Server(
        [
            {
                **_chat(_envelope("frozen_capacity", {"e_max": 2, "e_min": None})),
            },
            _chat(_envelope("propose_repair", ANSWER)),
        ]
    )
    server.chats[0]["choices"][0]["message"]["reasoning"] = "check the window first"
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    seen = []
    provider = OpenAICompatibleProvider(model="local/model")
    response = provider.propose(
        system="s",
        payload_text="p",
        schema=ACTION,
        tools=TOOLS,
        run_tool=lambda n, a: seen.append((n, a)) or "fits",
        tool_call_cap=2,
    )
    assert seen == [("frozen_capacity", {"e_max": 2, "e_min": None})]
    assert response.content == ANSWER and response.structured_mode == "guided_json"
    assert response.requests == 2 and response.tool_calls[0]["counted"]
    for body in server.bodies:
        assert "tools" not in body and "tool_choice" not in body
        rf = body["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        alternatives = rf["json_schema"]["schema"]["properties"]["call"]["anyOf"]
        names = [a["properties"]["name"]["enum"][0] for a in alternatives]
        # one alternative per tool, and one per action kind of the final answer
        assert names == ["frozen_capacity"] + ["propose_repair"] * 3
        assert "frozen_capacity" in body["messages"][0]["content"]
    # the tool's result goes back as the next user message, the thinking with the call
    second = server.bodies[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["reasoning"]
    assert second[-1] == {"role": "user", "content": "Result of frozen_capacity:\nfits"}
    assert response.schema_enforcement == {
        "requested": True,
        "mechanism": ENFORCEMENT_GUIDED,
        "enforced": True,
    }


def test_each_action_kind_carries_only_its_own_fields():
    from autowan.providers import action_variants

    by_kind = {v["properties"]["action"]["enum"][0]: v for v in action_variants(ACTION)}
    assert set(by_kind) == {"modify_parameters", "switch_method", "escalate"}
    assert "method" not in by_kind["modify_parameters"]["properties"]
    assert "num_bands_needed" not in by_kind["modify_parameters"]["properties"]
    assert "changes" not in by_kind["escalate"]["properties"]
    switch = by_kind["switch_method"]["properties"]
    assert switch["method"]["enum"] == ["pdwf", "scdm", "smv"]  # not nullable
    assert by_kind["modify_parameters"]["properties"]["changes"]["type"] == "array"
    for variant in by_kind.values():
        assert set(variant["required"]) == set(variant["properties"])


def test_a_guided_answer_that_is_not_an_envelope_is_a_schema_violation(monkeypatch):
    server = _Server([_chat({"content": '{"action": "escalate"}'})])
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    with pytest.raises(ProviderSchemaViolation, match="call envelope"):
        OpenAICompatibleProvider(model="local/model").propose(
            system="s",
            payload_text="p",
            schema=ACTION,
            tools=TOOLS,
            run_tool=lambda n, a: "",
            tool_call_cap=1,
        )


def test_enforcement_off_restores_the_unenforced_auto_loop(key, monkeypatch):
    server = _Server(
        [_chat({"tool_calls": [_call("propose_repair", {"action": "escalate"})]})]
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    response = OpenAICompatibleProvider(
        model="local/model", enforce_schema=False
    ).propose(
        system="s",
        payload_text="p",
        schema=ACTION,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=2,
    )
    body = server.bodies[0]
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["parameters"] == TOOLS[0].parameters
    assert response.schema_enforcement == {
        "requested": False,
        "mechanism": None,
        "enforced": False,
    }


def test_the_structured_json_schema_mode_sends_the_strict_dialect(key, monkeypatch):
    import json

    server = _Server([_chat({"content": json.dumps(ANSWER)})], balance=50.0)
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    OpenRouterProvider(structured_mode="json_schema").propose(
        system="s", payload_text="p", schema=ACTION
    )
    sent = server.bodies[0]["response_format"]["json_schema"]
    assert sent["strict"] is True and sent["schema"] == strict_schema(ACTION)


def test_a_model_s_provider_routing_is_merged_into_the_request(key, monkeypatch):
    settings = ModelSettings(
        "z/open", provider_routing={"order": ["fireworks"], "allow_fallbacks": False}
    )
    server = _Server(
        [_chat({"content": "", "tool_calls": [_call("propose_repair", ANSWER)]})],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    OpenRouterProvider(model="z/open", settings=settings).propose(
        system="s",
        payload_text="p",
        schema=ACTION,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=1,
    )
    assert server.bodies[0]["provider"] == {
        "require_parameters": True,
        "order": ["fireworks"],
        "allow_fallbacks": False,
    }


def test_the_tools_treat_a_null_argument_as_not_given():
    from autowan.tools import RepairTools

    tools = RepairTools.__new__(RepairTools)
    tools.calls = []
    seen = {}
    # a handler without the optional argument: a null passed through would TypeError
    tools.tool_frozen_capacity = lambda e_max: seen.update(e_max=e_max) or {"ok": 1}
    out = tools("frozen_capacity", {"e_max": 2.0, "e_min": None})
    assert seen == {"e_max": 2.0} and "bad arguments" not in out


# ---------------------------------------------------------------------------
# what is recorded
# ---------------------------------------------------------------------------


class _EnforcedModel:
    """A scripted provider that reports enforcement as the real one does."""

    def __init__(self, answers, enforced=True):
        self.answers = list(answers)
        self.enforced = enforced

    def __call__(self, model):
        return self

    def schema_enforcement(self, enforced=None):
        out = {"requested": True, "mechanism": ENFORCEMENT_STRICT}
        if enforced is not None:
            out["enforced"] = enforced
        return out

    def propose(self, **_):
        return ModelResponse(
            content=self.answers.pop(0),
            model="scripted",
            raw_text="",
            schema_enforcement=self.schema_enforcement(self.enforced),
        )


def test_the_run_and_each_attempt_record_whether_enforcement_held(tmp_path):
    from test_repair_band_count_refusal import _change, _loop

    model = _EnforcedModel(
        [_change("num_iter", 4000, 6000), _change("num_iter", 6000, 3000)]
        + [_change("trial_step", 1, 0.5)] * 6
    )
    result, _ = _loop(tmp_path, model)
    record = result.run.to_dict()
    summary = record["schema_enforcement"]
    assert summary["requested"] is True and summary["mechanism"] == ENFORCEMENT_STRICT
    repaired = [a for a in record["attempts"] if a["action"]]
    assert repaired and all(a["schema_enforcement"]["enforced"] for a in repaired)
    assert summary["answers_enforced"] >= len(repaired)
    assert summary["answers_unenforced"] == 0
    assert "schema_enforcement" in EXIT_COLUMNS
    assert exit_row(record)["schema_enforcement"].startswith(ENFORCEMENT_STRICT)


def test_a_run_without_a_model_records_no_enforcement(tmp_path):
    from test_repair_band_count_refusal import _loop

    result, _ = _loop(tmp_path, None)
    assert result.run.to_dict()["schema_enforcement"] is None
    assert exit_row(result.run.to_dict())["schema_enforcement"] is None


def test_the_record_round_trips_for_resume():
    from autowan.orchestrator import Run

    run = Run("x")
    run.schema_enforcement = {"requested": True, "mechanism": ENFORCEMENT_GUIDED}
    again = Run.from_dict(run.to_dict())
    assert again.schema_enforcement["mechanism"] == ENFORCEMENT_GUIDED
    assert again.to_dict()["schema_enforcement"]["answers_enforced"] == 0
