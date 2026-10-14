"""The provider speaks function calling, records its settings and cost, and stops early.

No network: ``urllib.request.urlopen`` is replaced by a scripted server that answers
the chat completions endpoint and OpenRouter's key and credits endpoints.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from autowan import providers
from autowan.providers import (
    DEFAULT_MODEL,
    MODEL_SETTINGS,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    ProviderCreditsExhausted,
    ProviderLowBalance,
    ProviderSchemaViolation,
    ToolSpec,
    make_provider,
)

SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}}}
TOOLS = [ToolSpec("frozen_capacity", "d", {"type": "object", "properties": {}})]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Server:
    def __init__(self, chats, balance=None, credits=None):
        self.chats = list(chats)
        self.balance = balance
        self.credits = credits
        self.bodies: list[dict] = []
        self.paths: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.paths.append(url)
        if url.endswith("/key"):
            return _Response(
                json.dumps({"data": {"limit_remaining": self.balance}}).encode()
            )
        if url.endswith("/credits"):
            return _Response(json.dumps({"data": self.credits or {}}).encode())
        self.bodies.append(json.loads(request.data.decode()))
        answer = self.chats.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(json.dumps(answer).encode())


def _chat(message, usage=None):
    return {
        "id": "gen-1",
        "model": DEFAULT_MODEL,
        "choices": [{"message": message}],
        "usage": usage
        or {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.006},
    }


def _call(name, arguments, ident="c1"):
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")


def test_the_default_model_is_the_latest_opus_with_explicit_high_effort():
    settings = MODEL_SETTINGS[DEFAULT_MODEL]
    assert DEFAULT_MODEL == "anthropic/claude-opus-5.5"
    assert settings.reasoning == {"effort": "high"}
    assert settings.temperature is None and settings.max_tokens == 32000
    assert settings.reservation_usd() == pytest.approx(0.64)


def test_a_tool_call_runs_and_its_result_goes_back_before_the_final_answer(
    key, monkeypatch
):
    server = _Server(
        [
            _chat(
                {
                    "content": "",
                    "tool_calls": [_call("frozen_capacity", {"e_max": 2})],
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": "x"}],
                }
            ),
            _chat(
                {
                    "content": "",
                    "tool_calls": [
                        _call("propose_repair", {"action": "escalate"}, "c2")
                    ],
                }
            ),
        ],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    seen = []
    provider = OpenRouterProvider()
    response = provider.propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda name, args: seen.append((name, args)) or "fits",
        tool_call_cap=3,
    )
    assert response.content == {"action": "escalate"}
    assert seen == [("frozen_capacity", {"e_max": 2})]
    assert (
        response.tool_calls[0]["name"] == "frozen_capacity"
        and response.tool_calls[0]["counted"]
    )
    assert response.requests == 2 and response.cost_usd == pytest.approx(0.012)
    assert response.usage["prompt_tokens"] == 2000
    first, second = server.bodies
    assert first["reasoning"] == {"effort": "high"} and first["max_tokens"] == 32000
    assert "temperature" not in first and first["tool_choice"] == "auto"
    assert first["usage"] == {"include": True}
    assert [t["function"]["name"] for t in first["tools"]] == [
        "frozen_capacity",
        "propose_repair",
    ]
    # the tool result and the reasoning blocks go back unmodified
    assert second["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "fits",
    }
    assert second["messages"][-2]["reasoning_details"][0]["data"] == "x"
    assert response.settings["reasoning"] == {"effort": "high"}


def test_calls_past_the_cap_are_answered_but_not_run(key, monkeypatch):
    server = _Server(
        [
            _chat(
                {
                    "tool_calls": [
                        _call("frozen_capacity", {}, "a"),
                        _call("frozen_capacity", {}, "b"),
                    ]
                }
            ),
            _chat({"tool_calls": [_call("propose_repair", {"action": "escalate"})]}),
        ],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    ran = []
    response = OpenRouterProvider().propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: ran.append(n) or "ok",
        tool_call_cap=1,
    )
    assert ran == ["frozen_capacity"]
    assert [c["counted"] for c in response.tool_calls] == [True, False]
    assert "budget" in server.bodies[1]["messages"][-1]["content"]


def test_a_prose_answer_is_asked_again_then_refused(key, monkeypatch):
    server = _Server([_chat({"content": "I think..."})] * 3, balance=50.0)
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    with pytest.raises(ProviderSchemaViolation, match="prose"):
        OpenRouterProvider().propose(
            system="s",
            payload_text="p",
            schema=SCHEMA,
            tools=TOOLS,
            run_tool=lambda n, a: "",
            tool_call_cap=2,
        )
    assert "propose_repair" in server.bodies[1]["messages"][-1]["content"]


def test_a_json_answer_in_the_content_is_taken(key, monkeypatch):
    server = _Server(
        [_chat({"content": '```json\n{"action": "escalate"}\n```'})], balance=50.0
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    # Without schema enforcement a JSON object in the text is taken (and recorded as
    # unenforced); with it, see test_schema_enforcement.py.
    response = OpenRouterProvider(enforce_schema=False).propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=2,
    )
    assert (
        response.content == {"action": "escalate"}
        and response.structured_mode == "content"
    )
    assert response.schema_enforcement == {
        "requested": False,
        "mechanism": None,
        "enforced": False,
    }


def test_a_low_balance_stops_before_the_request_is_sent(key, monkeypatch):
    server = _Server(
        [], balance=4.0, credits={"total_credits": 200, "total_usage": 190}
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    with pytest.raises(ProviderLowBalance, match="USD 4.00") as info:
        OpenRouterProvider().propose(system="s", payload_text="p", schema=SCHEMA)
    assert info.value.stop and info.value.kind == "low_balance"
    assert server.bodies == [], "a request was sent below the alarm"


def test_an_unknown_balance_is_not_low(key, monkeypatch):
    def failing(request, timeout=None):
        if request.full_url.endswith(("/key", "/credits")):
            raise urllib.error.URLError("unreachable")
        return _Response(json.dumps(_chat({"content": '{"action": "x"}'})).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", failing)
    provider = OpenRouterProvider()
    provider.propose(system="s", payload_text="p", schema=SCHEMA)
    assert provider.balance_checks[-1]["remaining_usd"] is None
    assert provider.balance_checks[-1]["errors"]


def test_a_402_is_a_provider_stop(key, monkeypatch):
    error = urllib.error.HTTPError(
        "u", 402, "pay", {}, io.BytesIO(b"Insufficient credits")
    )
    server = _Server([error], balance=50.0)
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    with pytest.raises(ProviderCreditsExhausted) as info:
        OpenRouterProvider().propose(system="s", payload_text="p", schema=SCHEMA)
    assert info.value.stop


def test_an_openai_compatible_server_needs_no_key_and_reports_tokens_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # under schema enforcement (default) the answer is a guided JSON call envelope
    server = _Server(
        [
            _chat(
                {
                    "content": json.dumps(
                        {
                            "call": {
                                "name": "propose_repair",
                                "arguments": {"action": "escalate"},
                            }
                        }
                    )
                }
            )
        ]
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    provider = make_provider("Qwen/Qwen3-32B", base_url="http://node10:8000/v1")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "http://node10:8000/v1/chat/completions"
    response = provider.propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=1,
    )
    assert response.cost_usd is None and response.usage["prompt_tokens"] == 1000
    body = server.bodies[0]
    assert "reasoning" not in body and "provider" not in body
    assert server.paths == ["http://node10:8000/v1/chat/completions"]


def test_the_key_never_reaches_the_record(key):
    provider = OpenRouterProvider()
    assert "sk-test-not-real" not in repr(provider)
    assert "sk-test-not-real" not in json.dumps(provider.settings.to_dict())


# ---------------------------------------------------------------------------
# the local preset (item 4 decisions 6-7)
# ---------------------------------------------------------------------------


def test_the_local_preset_needs_a_url(monkeypatch):
    monkeypatch.delenv(providers.LOCAL_BASE_URL_ENV, raising=False)
    with pytest.raises(providers.ProviderError, match="AUTOWAN_LOCAL_BASE_URL"):
        make_provider(provider="local")


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider(provider="somewhere")


def test_the_local_preset_reads_the_url_and_key_from_the_environment(monkeypatch):
    monkeypatch.setenv(providers.LOCAL_BASE_URL_ENV, "http://llm-host:8000/v1/")
    monkeypatch.setenv(providers.LOCAL_API_KEY_ENV, "local-test-key")
    server = _Server(
        [
            _chat(
                {
                    "content": json.dumps(
                        {
                            "call": {
                                "name": "propose_repair",
                                "arguments": {"action": "escalate"},
                            }
                        }
                    )
                }
            )
        ]
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    provider = make_provider(provider="local", min_balance_usd=10.0)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == providers.LOCAL_MODEL
    assert provider.base_url == "http://llm-host:8000/v1/chat/completions"
    assert provider.timeout_s == 900.0
    assert provider.settings is MODEL_SETTINGS[providers.LOCAL_MODEL]
    assert "local-test-key" not in repr(provider)
    response = provider.propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=1,
    )
    assert response.content == {"action": "escalate"}
    assert response.cost_usd is None and response.seconds is not None
    body = server.bodies[0]
    # provider defaults: no reasoning object, no temperature, the recorded max_tokens
    assert "reasoning" not in body and "temperature" not in body
    # schema enforcement (on by default) is vLLM guided decoding via response_format
    assert body["max_tokens"] == 16384 and "tool_choice" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert response.schema_enforcement["mechanism"] == "guided_json_response_format"


def test_an_explicit_url_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(providers.LOCAL_BASE_URL_ENV, "http://elsewhere:1/v1")
    provider = make_provider(
        "some/other-model", provider="local", base_url="http://llm-host:8000/v1"
    )
    assert provider.base_url == "http://llm-host:8000/v1/chat/completions"
    assert provider.model == "some/other-model"


def test_the_local_model_s_thinking_goes_back_with_its_tool_calls(monkeypatch):
    server = _Server(
        [
            _chat(
                {
                    "content": None,
                    "reasoning": "the frozen window may hold too many states",
                    "tool_calls": [_call("frozen_capacity", {"e_max": 2})],
                }
            ),
            _chat(
                {
                    "content": None,
                    "tool_calls": [
                        _call("propose_repair", {"action": "escalate"}, "c2")
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    # the native function-calling loop (enforcement off); the guided loop's echo is
    # tested in test_schema_enforcement.py
    provider = make_provider(
        provider="local", base_url="http://llm-host:8000/v1", enforce_schema=False
    )
    response = provider.propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: "fits",
        tool_call_cap=2,
    )
    assert response.requests == 2
    assistant = server.bodies[1]["messages"][-2]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning"] == "the frozen window may hold too many states"
    assert assistant["reasoning_content"] == assistant["reasoning"]
    assert "reasoning_details" not in assistant


def test_openrouter_does_not_get_a_reasoning_echo(key, monkeypatch):
    server = _Server(
        [
            _chat(
                {
                    "content": "",
                    "reasoning": "plain text reasoning",
                    "tool_calls": [_call("frozen_capacity", {})],
                }
            ),
            _chat({"tool_calls": [_call("propose_repair", {"action": "escalate"})]}),
        ],
        balance=50.0,
    )
    monkeypatch.setattr(providers.urllib.request, "urlopen", server)
    OpenRouterProvider().propose(
        system="s",
        payload_text="p",
        schema=SCHEMA,
        tools=TOOLS,
        run_tool=lambda n, a: "",
        tool_call_cap=2,
    )
    assistant = server.bodies[1]["messages"][-2]
    assert "reasoning_content" not in assistant and "reasoning" not in assistant


def test_the_cli_offers_the_local_preset(tmp_path, monkeypatch, capsys):
    from autowan import cli

    args = cli.build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--parent-xml",
            str(tmp_path / "data-file-schema.xml"),
            "--seed",
            "Ti",
            "--provider",
            "local",
        ]
    )
    assert args.provider == "local" and args.model is None
    monkeypatch.delenv(providers.LOCAL_BASE_URL_ENV, raising=False)
    assert (
        cli.main(
            [
                "--run-dir",
                str(tmp_path),
                "--parent-xml",
                str(tmp_path / "data-file-schema.xml"),
                "--seed",
                "Ti",
                "--provider",
                "local",
            ]
        )
        == 2
    )
    assert "AUTOWAN_LOCAL_BASE_URL" in capsys.readouterr().err
