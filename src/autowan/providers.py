"""Calling a model with the user's own key, and refusing to guess when it fails.

The workflow runs on a key its user supplies, against whichever model they choose, so
the provider layer is deliberately thin: it speaks the OpenAI-compatible chat
completions protocol (OpenRouter, or any base URL such as a local vLLM server), offers
the read-only repair tools through function calling, executes each tool call with a
callback the workflow supplies, and hands back the one final answer -- a call of the
``propose_repair`` function -- as a dictionary. Everything about what may be asked and
what may be returned lives in :mod:`actions` and :mod:`tools`, not here.

Per model it records what it sent: ``max_tokens``, the reasoning setting, the
temperature (or that none was sent), and per request the tokens and cost. Claude runs
with an explicit reasoning effort of ``high``; every other
model at its provider's default, which is recorded as such.

Failures are typed so the run can tell them apart. A provider that cannot be reached,
refuses payment, or is about to (the low-balance alarm, raised *before* a request that
would reserve more than the account holds) is a **provider stop**: the run records it
and can be resumed from the same attempt. An answer that is not the schema-shaped call
asked for is a schema violation, handed back to the model like any refused answer. The
key is read from the environment or passed in, used, and never written anywhere.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

OPENROUTER_API = "https://openrouter.ai/api/v1"
DEFAULT_BASE_URL = f"{OPENROUTER_API}/chat/completions"

#: The latest Claude Opus on OpenRouter, verified against GET /api/v1/models on
#: 2026-10-07 (created 2026-09-22; reasoning mandatory, efforts max..low).
DEFAULT_MODEL = "anthropic/claude-opus-5.5"

#: The locally hosted open-weight model: Qwen3.5-9B, RedHat's INT8
#: weight-and-activation quantisation, served by vLLM (or any OpenAI-compatible server
#: with tool calling). The served name is the Hugging Face ID,
#: so the record names the exact weights.
LOCAL_MODEL = "RedHatAI/Qwen3.5-9B-quantized.w8a8"

#: Fig. 3 models, verified on GET /api/v1/models and
#: /endpoints on 2026-10-08/09: each lists tools, tool_choice and structured_outputs.
#: Every one runs at its provider-default reasoning (nothing sent). GPT is GPT-6.1 Sol
#: (chosen 2026-10-08): GPT-6 Astra at USD 10/50 does not fit the cap.
GPT_MODEL = "openai/gpt-6.1-sol"
GEMINI_FLASH_MODEL = "google/gemini-3.8-flash"
OPEN_WEIGHT_MODEL = "z-ai/glm-5.3"

#: Where the local endpoint is read from when ``--provider local`` is given no URL.
LOCAL_BASE_URL_ENV = "AUTOWAN_LOCAL_BASE_URL"
#: The local server's key, if it was started with one (``vllm serve --api-key``).
LOCAL_API_KEY_ENV = "AUTOWAN_LOCAL_API_KEY"

#: Provider presets the CLI and drivers accept.
PROVIDERS = ("openrouter", "local")

#: Name of the function a model calls to give its one final answer.
TOOL_NAME = "propose_repair"

#: Order in which the two structured-output mechanisms are tried when no tools are
#: offered (the review gate).
STRUCTURED_MODES = ("json_schema", "tool")

#: Status codes worth another attempt: transient congestion and server faults.
#: 402 is absent on purpose -- more attempts do not buy credit.
RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)

#: How long a balance reading is trusted before it is read again, in seconds.
BALANCE_CACHE_S = 60.0


# ---------------------------------------------------------------------------
# schema enforcement (Fig. 3 fairness)
# ---------------------------------------------------------------------------

#: How the answer's schema is enforced at decode time, per provider class.
#:
#: * ``strict_function_calling`` -- every function definition carries
#:   ``"strict": true`` with a strict-dialect schema. OpenAI enforces it as strict
#:   function calling (structured outputs); Anthropic as strict tool use; OpenRouter
#:   passes the flag through to the backend it routes to.
#: * ``guided_json_response_format`` -- a vLLM server is sent, every turn, a
#:   ``response_format`` JSON schema whose one property ``call`` is an ``anyOf`` over
#:   the offered functions (``{"name": <tool>, "arguments": <its strict schema>}``);
#:   vLLM grammar-decodes it after the reasoning block. Measured on vLLM 0.31 with the
#:   ``qwen3_coder`` tool parser (2026-10-08): tool calls are parsed, never
#:   constrained, in every ``tool_choice`` (``auto``, ``required``, a named function --
#:   a model told to put ``"green"`` in a red/blue enum and ``"big"`` in an integer did
#:   so in all three), while ``response_format`` held every constraint. Report
#:   2026-10-07/12 saw the 9B model break the evidence enum 4 times in 4 in ``auto``.
ENFORCEMENT_STRICT = "strict_function_calling"
ENFORCEMENT_GUIDED = "guided_json_response_format"

#: Keywords the strict dialects do not accept. Anthropic strict tool use refuses numeric
#: and string-length constraints and array bounds beyond minItems 0/1; OpenAI strict
#: mode refuses string lengths. They are dropped from what is *sent*; the validator
#: (:func:`autowan.actions.validate_action`) still applies every one of them.
_STRICT_DROPPED = frozenset(
    {
        "maxLength",
        "minLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "pattern",
        "format",
        "default",
    }
)


#: Fields only one action kind may carry, and fields an action kind may not carry
#: (the rules :func:`autowan.actions.validate_action` applies). Used to split the final
#: answer's schema into one alternative per action kind, so a constrained decoder never
#: has to fill a field that belongs to another action (a 9B model given one all-fields
#: schema put ``method: "scdm"`` on a modify_parameters 3 times in 3, 2026-10-08).
_ACTION_ONLY = {"method": "switch_method", "reason": "switch_method"}
_ACTION_ONLY_ESCALATE = ("num_bands_needed",)
_ACTION_FORBIDS = {"escalate": ("changes",)}


def action_variants(schema: Mapping[str, Any]) -> list[dict]:
    """The strict form of an action schema, split into one schema per action kind.

    A schema without an ``action`` enum is returned whole (as the one alternative).
    """
    strict = strict_schema(schema)
    action = (strict.get("properties") or {}).get("action") or {}
    kinds = action.get("enum")
    if not kinds:
        return [strict]
    variants = []
    for kind in kinds:
        props = {}
        for name, sub in strict["properties"].items():
            owner = _ACTION_ONLY.get(name)
            if owner is not None and owner != kind:
                continue
            if name in _ACTION_ONLY_ESCALATE and kind != "escalate":
                continue
            if name in _ACTION_FORBIDS.get(kind, ()):
                continue
            if name == "action":
                sub = {"type": "string", "enum": [kind]}
            elif owner == kind or (kind == "modify_parameters" and name == "changes"):
                # required for this kind: drop the null alternative
                alts = [a for a in sub.get("anyOf", [sub]) if a != {"type": "null"}]
                sub = alts[0] if len(alts) == 1 else {"anyOf": alts}
            props[name] = sub
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(props),
                "properties": props,
            }
        )
    return variants


def guided_envelope(functions: Sequence["ToolSpec"]) -> dict:
    """One JSON schema whose instances are exactly the calls of ``functions``.

    The final answer (:data:`TOOL_NAME`) contributes one alternative per action kind.
    """
    alternatives = []
    for f in functions:
        arguments = (
            action_variants(f.parameters)
            if f.name == TOOL_NAME
            else [strict_schema(f.parameters)]
        )
        for variant in arguments:
            alternatives.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string", "enum": [f.name]},
                        "arguments": variant,
                    },
                }
            )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["call"],
        "properties": {"call": {"anyOf": alternatives}},
    }


def guided_instructions(functions: Sequence["ToolSpec"]) -> str:
    """How to call the functions when the reply itself is the (constrained) call."""
    lines = [
        "HOW TO CALL FUNCTIONS IN THIS SESSION",
        "",
        'Every reply is one JSON object {"call": {"name": ..., "arguments": {...}}} '
        "naming one of the functions below. A read-only tool's result comes back in "
        f"the next message; {TOOL_NAME} gives your final answer and ends your turn. "
        "An optional argument you do not use is null.",
        "",
    ]
    for f in functions:
        lines.append(f"- {f.name}: {f.description}")
    return "\n".join(lines)


def strict_schema(schema: Mapping[str, Any]) -> dict:
    """``schema`` rewritten in the dialect every strict mechanism accepts.

    Every object closes (``additionalProperties: false``) and lists all its properties
    as required, as OpenAI strict mode demands; a property that was optional becomes
    nullable instead (``anyOf`` with ``null``), so leaving it out is still expressible.
    An empty (any-value) schema has no strict form, so an optional property carrying
    one is dropped. Unsupported constraint keywords are removed (see
    :data:`_STRICT_DROPPED`).
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, Mapping):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRICT_DROPPED:
                continue
            if key == "properties":
                continue
            if key in ("items", "anyOf", "allOf", "oneOf"):
                out[key] = walk(value)
            else:
                out[key] = value
        if node.get("type") == "object" or "properties" in node:
            required = set(node.get("required") or ())
            props: dict[str, Any] = {}
            for name, sub in (node.get("properties") or {}).items():
                if isinstance(sub, Mapping) and not sub:
                    if name in required:
                        raise ValueError(
                            f"property {name!r} is required and accepts any value; "
                            "a strict schema cannot express it"
                        )
                    continue
                converted = walk(sub)
                if name not in required:
                    converted = {"anyOf": [converted, {"type": "null"}]}
                props[name] = converted
            out["type"] = "object"
            out["properties"] = props
            out["required"] = list(props)
            out["additionalProperties"] = False
        return out

    return walk(schema)


class ProviderError(RuntimeError):
    """The model could not be reached, or answered with something unusable.

    Distinct from every physics failure in this package. ``stop`` is true for the
    failures that end a run as a provider stop -- resumable from the same attempt --
    rather than as something the model said.
    """

    stop = True
    kind = "provider_error"


class ProviderCreditsExhausted(ProviderError):
    """The account cannot pay for this request (HTTP 402)."""

    kind = "credits_exhausted"


class ProviderLowBalance(ProviderError):
    """The balance is below the alarm: the next request could be refused for payment."""

    kind = "low_balance"


class ProviderUnreachable(ProviderError):
    """The network or the provider failed on every retry."""

    kind = "unreachable"


class ProviderAuthError(ProviderError):
    """The key was refused."""

    kind = "credentials"


class ProviderSchemaViolation(ProviderError):
    """The answer was not the object the schema asked for. Not a stop."""

    stop = False
    kind = "schema_violation"


class ProviderUnroutable(ProviderError):
    """No backend offers this structured-output mechanism for this model."""

    kind = "unroutable"


# ---------------------------------------------------------------------------
# per-model settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSettings:
    """What is sent for one model, and what it costs. Recorded with every run."""

    model: str
    max_tokens: int = 16000
    reasoning: dict | None = None
    """Sent as OpenRouter's unified ``reasoning`` object; ``None`` sends nothing, so the
    provider's default applies."""
    temperature: float | None = None
    """``None`` sends no temperature (the provider default)."""
    usd_per_mtok_in: float | None = None
    usd_per_mtok_out: float | None = None
    note: str = ""
    provider_routing: dict | None = None
    """OpenRouter ``provider`` preferences for this model (e.g. pinning one backend so
    every run of an open-weight model meets the same weights and quantisation);
    merged into the request's ``provider`` object."""

    def reservation_usd(self, prompt_tokens: int = 0) -> float | None:
        """The most one request can cost: what a provider reserves before running it."""
        if self.usd_per_mtok_out is None:
            return None
        return (
            self.max_tokens * self.usd_per_mtok_out
            + prompt_tokens * (self.usd_per_mtok_in or 0.0)
        ) / 1e6

    def cost_usd(self, usage: Mapping[str, Any]) -> float | None:
        if self.usd_per_mtok_in is None or self.usd_per_mtok_out is None:
            return None
        return (
            float(usage.get("prompt_tokens") or 0) * self.usd_per_mtok_in
            + float(usage.get("completion_tokens") or 0) * self.usd_per_mtok_out
        ) / 1e6

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_SETTINGS: dict[str, ModelSettings] = {
    DEFAULT_MODEL: ModelSettings(
        DEFAULT_MODEL,
        max_tokens=32000,
        reasoning={"effort": "high"},
        temperature=None,
        usd_per_mtok_in=4.0,
        usd_per_mtok_out=20.0,
        note=(
            "OpenRouter lists reasoning as mandatory for this model (efforts max, xhigh, "
            "high, medium, low); effort set explicitly to high. No "
            "temperature is sent: Anthropic's extended thinking takes only the default. "
            "max_tokens 32000 keeps the per-request reservation at USD 0.64."
        ),
    ),
    GPT_MODEL: ModelSettings(
        GPT_MODEL,
        max_tokens=32000,
        reasoning=None,
        temperature=None,
        usd_per_mtok_in=2.0,
        usd_per_mtok_out=10.0,
        note=(
            "OpenAI's GPT-6.1 Sol (created 2026-09-29), below the GPT-6 Astra flagship; "
            "chosen for cost (USD 2/10 against Astra's 10/50). Reasoning "
            "mandatory, efforts max..low; nothing sent, so OpenRouter's default_effort "
            "medium applies. max_tokens 32000 reserves up to USD 0.32 per request."
        ),
    ),
    GEMINI_FLASH_MODEL: ModelSettings(
        GEMINI_FLASH_MODEL,
        max_tokens=32000,
        reasoning=None,
        temperature=None,
        usd_per_mtok_in=0.75,
        usd_per_mtok_out=3.75,
        note=(
            "Google's newest Flash tier (created 2026-09-02). Reasoning mandatory, "
            "efforts high/medium/low; nothing sent, so the default medium applies. "
            "Thinking bills as output (internal_reasoning 3.75/M)."
        ),
    ),
    OPEN_WEIGHT_MODEL: ModelSettings(
        OPEN_WEIGHT_MODEL,
        max_tokens=32000,
        reasoning=None,
        temperature=None,
        usd_per_mtok_in=1.4,
        usd_per_mtok_out=4.4,
        provider_routing={"order": ["fireworks"], "allow_fallbacks": False},
        note=(
            "Z.ai GLM-5.3 (open weights, created 2026-08-19): highest Artificial "
            "Analysis agentic index of the open-weight models listed (53.1); model card "
            "Toolathlon Verified 73.0, AutomationBench 48.2. Reasoning mandatory, "
            "efforts max/high/low; nothing sent, so OpenRouter's default_effort max "
            "applies. Pinned to one backend (no fallbacks) so every run meets the same "
            "weights and quantisation; the first-party Z.AI endpoint lists no "
            "structured_outputs, so a backend that does is pinned."
        ),
    ),
    LOCAL_MODEL: ModelSettings(
        LOCAL_MODEL,
        max_tokens=16384,
        reasoning=None,
        temperature=None,
        note=(
            "Local vLLM, provider defaults: nothing is sent, so thinking is on (the "
            "Qwen3.5 chat template's default) and sampling is the server's default, set "
            "by serve_vllm.sbatch to Qwen's documented thinking-mode setting "
            "(temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5). max_tokens "
            "16384 leaves the prompt 49k of the server's 65536-token context. No cost: "
            "tokens only."
        ),
    ),
}


def settings_for(model: str) -> ModelSettings:
    """The recorded settings for ``model``; an unknown model runs at provider defaults."""
    if model in MODEL_SETTINGS:
        return MODEL_SETTINGS[model]
    return ModelSettings(
        model,
        note="no entry in MODEL_SETTINGS: provider-default reasoning and temperature",
    )


# ---------------------------------------------------------------------------
# messages and answers
# ---------------------------------------------------------------------------


@dataclass
class Image:
    """An image sent alongside the payload."""

    path: Path
    media_type: str = "image/png"

    def as_data_url(self) -> str:
        data = base64.b64encode(Path(self.path).read_bytes()).decode("ascii")
        return f"data:{self.media_type};base64,{data}"


#: Calls per attempt of the tools that do not count against the budget (``uncapped``).
UNCAPPED_CALLS = 8


def tool_call_gate(name, names, free, used, free_used, cap, final_hint):
    """Whether a call may run, and how it is counted: ``(result_or_None, counted,
    uncapped)``. ``result`` is the refusal text when the call may not run."""
    if name not in names:
        return f"there is no tool named {name!r}", False, False
    if name in free:
        if free_used >= UNCAPPED_CALLS:
            return (
                f"{name} has been called {UNCAPPED_CALLS} times for this attempt; "
                f"its answer does not change until a new deck has run",
                False,
                True,
            )
        return None, False, True
    if used >= cap:
        return (
            f"the tool budget for this attempt ({cap} calls) is spent; {final_hint}",
            False,
            False,
        )
    return None, True, False


@dataclass(frozen=True)
class ToolSpec:
    """One read-only tool offered through function calling."""

    name: str
    description: str
    parameters: dict
    uncapped: bool = False
    """A tool whose calls do not count against the per-attempt tool-call budget (they
    have their own bound, :data:`UNCAPPED_CALLS`): the frontier's next-proposal tool,
    which costs nothing and answers a question the model should never be priced out
    of asking."""

    def as_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ModelResponse:
    """What came back, with what it cost and which tools were called on the way."""

    content: dict[str, Any]
    model: str
    raw_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    generation_id: str | None = None
    attempts: int = 1
    structured_mode: str = "json_schema"
    """Which mechanism carried the final answer."""
    tool_calls: list[dict] = field(default_factory=list)
    """Every read-only tool call made on the way: name, arguments, a result excerpt."""
    requests: int = 1
    cost_usd: float | None = None
    settings: dict = field(default_factory=dict)
    seconds: float | None = None
    """Wall time spent waiting on the model for this answer, summed over requests."""
    schema_enforcement: dict = field(default_factory=dict)
    """Whether this answer was schema-enforced at decode time, and by what mechanism:
    ``{"requested", "mechanism", "enforced"}``. An answer read from free text (mode
    ``content``) was not, whatever was requested."""


ToolRunner = Callable[[str, dict], str]


def _short(blob: Any, limit: int = 400) -> str:
    text = (
        blob.decode(errors="replace")
        if isinstance(blob, (bytes, bytearray))
        else str(blob)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _json_object(text: str) -> dict | None:
    """A JSON object in a reply, bare or in a fenced block; ``None`` if there is none."""
    candidates = [text]
    candidates += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _add_usage(total: dict, usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if usage.get(key) is not None:
            total[key] = total.get(key, 0) + int(usage[key])
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") is not None:
        total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + int(
            details["reasoning_tokens"]
        )
    if usage.get("cost") is not None:
        total["cost"] = round(total.get("cost", 0.0) + float(usage["cost"]), 8)


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------


@dataclass
class ChatCompletionsProvider:
    """A model behind an OpenAI-compatible ``/chat/completions`` endpoint."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 300.0
    max_attempts: int = 4
    structured_mode: str | None = None
    settings: ModelSettings | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    require_key: bool = True
    max_turns: int = 16
    """Requests one tool loop may make before it must have an answer."""
    enforce_schema: bool = True
    """Enforce the action and tool schemas at decode time (Fig. 3 fairness). On
    by default, so every campaign provider runs with it; ``False``
    restores the unenforced ``auto`` loop. Recorded with every answer."""
    enforcement_mechanism: str = ENFORCEMENT_STRICT

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key and self.require_key:
            raise ProviderError(
                f"no API key: pass api_key= or set {self.api_key_env}. This tool runs "
                "on a key you supply and never ships one."
            )
        self._key = key
        # Cleared so the key cannot reach a repr, a traceback frame dump, or a log line.
        self.api_key = None
        if self.settings is None:
            self.settings = settings_for(self.model)

    # -- hooks -------------------------------------------------------------

    def _extra_body(self) -> dict:
        return {}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def before_request(self, body: dict) -> None:
        """Called before every request; the OpenRouter provider checks the balance."""

    # -- schema enforcement ----------------------------------------------------

    def schema_enforcement(self, enforced: bool | None = None) -> dict:
        """What this provider requests, and (per answer) whether it held."""
        record = {
            "requested": bool(self.enforce_schema),
            "mechanism": self.enforcement_mechanism if self.enforce_schema else None,
        }
        if enforced is not None:
            record["enforced"] = bool(self.enforce_schema and enforced)
        return record

    def _function(self, spec: "ToolSpec") -> dict:
        """One function definition as sent: strict-dialect and flagged when enforcing."""
        function = spec.as_function()
        if self.enforce_schema and self.enforcement_mechanism == ENFORCEMENT_STRICT:
            function["function"]["parameters"] = strict_schema(spec.parameters)
            function["function"]["strict"] = True
        elif self.enforce_schema:
            # Guided decoding compiles the schema into a grammar: send the same
            # closed, constraint-free dialect so every backend sees one schema.
            function["function"]["parameters"] = strict_schema(spec.parameters)
        return function

    def _loop_tool_choice(self) -> Any:
        """``auto`` throughout: Anthropic's extended thinking refuses a forced choice,
        and a strict function definition is enforced whenever it is called."""
        return "auto"

    def request_cost(self, usage: Mapping[str, Any]) -> float | None:
        if usage.get("cost") is not None:
            return float(usage["cost"])
        return self.settings.cost_usd(usage)

    def _assistant_echo(self, message: Mapping[str, Any]) -> dict:
        """What of an assistant turn must go back with it for the model to continue."""
        if message.get("reasoning_details"):
            # Claude continues its reasoning across tool turns only if its thinking
            # blocks come back unmodified.
            return {"reasoning_details": message["reasoning_details"]}
        return {}

    # -- the request ---------------------------------------------------------

    def _messages(
        self, system: str, payload_text: str, images: Sequence[Image]
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": payload_text}]
        for image in images:
            content.append(
                {"type": "image_url", "image_url": {"url": image.as_data_url()}}
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def _base_body(self, messages: list, temperature: float | None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.reasoning is not None:
            body["reasoning"] = dict(self.settings.reasoning)
        chosen = temperature if temperature is not None else self.settings.temperature
        if chosen is not None:
            body["temperature"] = chosen
        body.update(self._extra_body())
        return body

    def _post(self, body: dict) -> dict:
        self.before_request(body)
        data = json.dumps(body).encode()
        started = time.monotonic()
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                self.base_url, data=data, headers=self._headers()
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    parsed = json.loads(response.read().decode())
                self._raise_reported_error(parsed)
                parsed["_attempts"] = attempt
                parsed["_seconds"] = round(time.monotonic() - started, 3)
                return parsed
            except urllib.error.HTTPError as exc:
                detail = _short(exc.read())
                if exc.code == 402:
                    raise ProviderCreditsExhausted(
                        f"the provider refused the request for payment (HTTP 402): {detail}. "
                        "This is not a result -- nothing downstream may read it as the model "
                        "having no suggestion."
                    ) from exc
                if exc.code in (401, 403):
                    raise ProviderAuthError(
                        f"the provider rejected the credentials (HTTP {exc.code}). "
                        "Check the key this run was given; its value is not logged."
                    ) from exc
                if exc.code == 404 and "no endpoints found" in detail.lower():
                    raise ProviderUnroutable(detail) from exc
                if exc.code not in RETRYABLE_STATUS:
                    raise ProviderError(
                        f"HTTP {exc.code} from the provider: {detail}"
                    ) from exc
                last = exc
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                ConnectionError,
            ) as exc:
                last = exc
            if attempt < self.max_attempts:
                time.sleep(min(2.0**attempt, 30.0))
        raise ProviderUnreachable(
            f"no usable answer after {self.max_attempts} attempts; last failure: {last}"
        )

    @staticmethod
    def _raise_reported_error(parsed: dict) -> None:
        if "error" in parsed and parsed["error"]:
            message = parsed["error"].get("message", "")
            if (
                "402" in str(parsed["error"].get("code", ""))
                or "credit" in message.lower()
            ):
                raise ProviderCreditsExhausted(f"the provider reports: {message}")
            raise ProviderError(f"the provider reports: {message}")
        if not parsed.get("choices"):
            raise ProviderError(
                "the response carried no choices. An empty answer is not a proposal, "
                "and is recorded as a provider failure rather than as a model that "
                "declined to act."
            )

    # -- one answer ----------------------------------------------------------

    def propose(
        self,
        *,
        system: str,
        payload_text: str,
        schema: dict[str, Any],
        images: Sequence[Image] = (),
        tools: Sequence[ToolSpec] = (),
        run_tool: ToolRunner | None = None,
        tool_call_cap: int = 0,
        temperature: float | None = None,
    ) -> ModelResponse:
        """Ask for one answer conforming to ``schema``.

        With ``tools``, the model may call them (each executed by ``run_tool``, at most
        ``tool_call_cap`` times) before it answers by calling ``propose_repair``.
        Without, the answer is constrained by a JSON schema or a forced function call,
        whichever the route offers.
        """
        if tools:
            if run_tool is None:
                raise ValueError("tools were offered with nothing to run them")
            return self._tool_loop(
                system,
                payload_text,
                schema,
                images,
                tools,
                run_tool,
                tool_call_cap,
                temperature,
            )
        modes = (self.structured_mode,) if self.structured_mode else STRUCTURED_MODES
        unroutable: list[str] = []
        for mode in modes:
            try:
                response = self._structured(
                    system, payload_text, schema, images, temperature, mode
                )
            except ProviderUnroutable as exc:
                unroutable.append(f"{mode}: {exc}")
                continue
            self.structured_mode = mode
            return response
        raise ProviderError(
            "this model offers neither structured-output mechanism through the "
            "configured route, so its answer could not be constrained to the schema. "
            "An unconstrained answer is not accepted here. Tried -- "
            + " | ".join(unroutable)
        )

    def _final_tool(self, schema: dict) -> dict:
        return self._function(
            ToolSpec(
                TOOL_NAME,
                "Return exactly one action for this construction. Call this once, "
                "when you are ready; it ends your turn.",
                schema,
            )
        )

    def _structured(
        self, system, payload_text, schema, images, temperature, mode
    ) -> ModelResponse:
        body = self._base_body(
            self._messages(system, payload_text, images), temperature
        )
        if mode == "tool":
            body["tools"] = [self._final_tool(schema)]
            body["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "repair_action",
                    "strict": True,
                    "schema": strict_schema(schema) if self.enforce_schema else schema,
                },
            }
        parsed = self._post(body)
        message = parsed["choices"][0].get("message") or {}
        if mode == "tool":
            calls = message.get("tool_calls") or []
            if not calls:
                raise ProviderSchemaViolation(
                    "the tool call was forced but the answer carried none, so the "
                    "output was never constrained to the schema"
                )
            text = (calls[0].get("function") or {}).get("arguments") or ""
        else:
            text = message.get("content") or ""
        content = self._parse_object(text)
        usage = parsed.get("usage") or {}
        return ModelResponse(
            content=content,
            model=parsed.get("model", ""),
            raw_text=text,
            usage=dict(usage),
            generation_id=parsed.get("id"),
            attempts=parsed.get("_attempts", 1),
            structured_mode=mode,
            cost_usd=self.request_cost(usage),
            settings=self.settings.to_dict(),
            seconds=parsed.get("_seconds"),
            schema_enforcement=self.schema_enforcement(enforced=True),
        )

    @staticmethod
    def _parse_object(text: str) -> dict:
        if not text.strip():
            raise ProviderSchemaViolation("the response carried empty content")
        try:
            content = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderSchemaViolation(
                f"the answer is not JSON despite a schema being requested: {_short(text)}"
            ) from exc
        if not isinstance(content, dict):
            raise ProviderSchemaViolation(
                f"the answer is a {type(content).__name__}, not the object the schema asked for"
            )
        return content

    def _tool_loop(
        self, system, payload_text, schema, images, tools, run_tool, cap, temperature
    ) -> ModelResponse:
        messages = self._messages(system, payload_text, images)
        offered = [self._function(t) for t in tools] + [self._final_tool(schema)]
        names = {t.name for t in tools}
        free = {t.name for t in tools if getattr(t, "uncapped", False)}
        free_used = 0
        records: list[dict] = []
        usage_total: dict[str, Any] = {}
        cost_total: float | None = None
        used = 0
        nudges = 0
        model_name = ""
        last_id = None
        seconds = 0.0
        for request_number in range(1, self.max_turns + 1):
            body = self._base_body(messages, temperature)
            body["tools"] = offered
            # A model that answers in text is told how to answer instead.
            body["tool_choice"] = self._loop_tool_choice()
            try:
                parsed = self._post(body)
            except ProviderError as exc:
                # What was spent before the failure stays on the record.
                exc.tool_calls = records
                exc.cost_usd = cost_total
                exc.requests = request_number - 1
                raise
            seconds += float(parsed.get("_seconds") or 0.0)
            usage = parsed.get("usage") or {}
            _add_usage(usage_total, usage)
            cost = self.request_cost(usage)
            if cost is not None:
                cost_total = (cost_total or 0.0) + cost
            model_name = parsed.get("model", model_name)
            last_id = parsed.get("id", last_id)
            message = parsed["choices"][0].get("message") or {}
            calls = message.get("tool_calls") or []
            final = next(
                (
                    c
                    for c in calls
                    if (c.get("function") or {}).get("name") == TOOL_NAME
                ),
                None,
            )
            if final is not None:
                text = (final.get("function") or {}).get("arguments") or ""
                return ModelResponse(
                    content=self._parse_object(text),
                    model=model_name,
                    raw_text=text,
                    usage=usage_total,
                    generation_id=last_id,
                    structured_mode="tool_loop",
                    tool_calls=records,
                    requests=request_number,
                    cost_usd=cost_total,
                    settings=self.settings.to_dict(),
                    seconds=round(seconds, 3),
                    schema_enforcement=self.schema_enforcement(enforced=True),
                )
            if not calls:
                text = message.get("content") or ""
                found = _json_object(text) if text else None
                # Under enforcement a JSON object in free text is not taken: it was
                # never constrained, so the model is asked for the function call.
                if found is not None and "action" in found and not self.enforce_schema:
                    return ModelResponse(
                        content=found,
                        model=model_name,
                        raw_text=text,
                        usage=usage_total,
                        generation_id=last_id,
                        structured_mode="content",
                        tool_calls=records,
                        requests=request_number,
                        cost_usd=cost_total,
                        settings=self.settings.to_dict(),
                        seconds=round(seconds, 3),
                        schema_enforcement=self.schema_enforcement(enforced=False),
                    )
                nudges += 1
                if nudges > 2:
                    raise ProviderSchemaViolation(
                        "the model answered in prose three times without calling "
                        f"{TOOL_NAME}: {_short(text)}"
                    )
                messages.append({"role": "assistant", "content": text or "(empty)"})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Give your answer by calling the {TOOL_NAME} function.",
                    }
                )
                continue
            assistant = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            }
            assistant.update(self._assistant_echo(message))
            messages.append(assistant)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    arguments = {"unparsed": function.get("arguments")}
                result, counted, uncapped = tool_call_gate(
                    name, names, free, used, free_used, cap,
                    f"give your answer now by calling {TOOL_NAME}",
                )  # fmt: skip
                if result is None:
                    used += int(counted)
                    free_used += int(uncapped)
                    try:
                        result = run_tool(
                            name, arguments if isinstance(arguments, dict) else {}
                        )
                    except (
                        Exception
                    ) as exc:  # a tool failure is information, not a crash
                        result = f"the tool failed: {exc}"
                records.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": _short(result, 600),
                        "counted": counted,
                        **({"uncapped": True} if uncapped else {}),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )
        raise ProviderSchemaViolation(
            f"no {TOOL_NAME} call within {self.max_turns} requests"
        )


@dataclass
class OpenRouterProvider(ChatCompletionsProvider):
    """A model reached through OpenRouter, using a key the user supplies.

    ``require_parameters`` is set so OpenRouter only routes to a backend that honours
    the tools and response format; usage accounting is requested so each response
    carries its cost. Before every request the account balance is compared with the
    alarm (``min_balance_usd``, raised to twice the request's own reservation), and a
    balance below it stops the run before a 402 can.
    """

    min_balance_usd: float = 10.0
    api_url: str = OPENROUTER_API
    referer: str = "https://github.com/"
    title: str = "physics-gated-wannier"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._balance: tuple[float, float | None] | None = None
        self.balance_checks: list[dict] = []

    def _headers(self) -> dict:
        headers = super()._headers()
        headers["HTTP-Referer"] = self.referer
        headers["X-Title"] = self.title
        return headers

    def _extra_body(self) -> dict:
        routing = {"require_parameters": True}
        routing.update(self.settings.provider_routing or {})
        return {"provider": routing, "usage": {"include": True}}

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            f"{self.api_url}{path}", headers=self._headers()
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode()).get("data") or {}

    def balance_usd(self) -> float | None:
        """What the key can still spend: the lower of key limit and account credit."""
        now = time.monotonic()
        if self._balance is not None and now - self._balance[0] < BALANCE_CACHE_S:
            return self._balance[1]
        known: list[float] = []
        errors = []
        try:
            key = self._get("/key")
            if key.get("limit_remaining") is not None:
                known.append(float(key["limit_remaining"]))
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
            errors.append(f"/key: {exc}")
        try:
            credits = self._get("/credits")
            if credits.get("total_credits") is not None:
                known.append(
                    float(credits["total_credits"])
                    - float(credits.get("total_usage") or 0)
                )
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
            errors.append(f"/credits: {exc}")
        remaining = min(known) if known else None
        self._balance = (now, remaining)
        self.balance_checks.append(
            {"remaining_usd": remaining, "errors": errors, "checked_at": time.time()}
        )
        return remaining

    def before_request(self, body: dict) -> None:
        if self.min_balance_usd <= 0:
            return
        remaining = self.balance_usd()
        if remaining is None:
            return  # unknown is not low; the read failures are on the record
        reservation = self.settings.reservation_usd() or 0.0
        alarm = max(self.min_balance_usd, 2.0 * reservation)
        if remaining < alarm:
            raise ProviderLowBalance(
                f"the OpenRouter balance is USD {remaining:.2f}, below the alarm of "
                f"USD {alarm:.2f} (a request to {self.model} reserves up to USD "
                f"{reservation:.2f}); top up and resume the run"
            )


@dataclass
class OpenAICompatibleProvider(ChatCompletionsProvider):
    """Any OpenAI-compatible server by base URL -- e.g. a local vLLM with tool calling.

    The key is optional (read from ``api_key_env`` if set); cost is not known, so only
    tokens are recorded. A reasoning model's thinking (vLLM's ``reasoning`` /
    ``reasoning_content``) goes back with its tool calls, so a chat template that keeps
    the thinking after the last user turn (Qwen3.5's does) sees it on the next request.
    """

    base_url: str = "http://localhost:8000/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    require_key: bool = False
    enforcement_mechanism: str = ENFORCEMENT_GUIDED

    def _tool_loop(
        self, system, payload_text, schema, images, tools, run_tool, cap, temperature
    ) -> ModelResponse:
        if not (
            self.enforce_schema and self.enforcement_mechanism == ENFORCEMENT_GUIDED
        ):
            return super()._tool_loop(
                system, payload_text, schema, images, tools, run_tool, cap, temperature
            )
        return self._guided_loop(
            system, payload_text, schema, images, tools, run_tool, cap, temperature
        )

    def _guided_loop(
        self, system, payload_text, schema, images, tools, run_tool, cap, temperature
    ) -> ModelResponse:
        """The tool loop with every turn grammar-constrained by ``response_format``.

        The functions are described in the system text and offered as one JSON
        envelope: each reply is ``{"call": {"name", "arguments"}}``, a read-only tool
        (its result comes back as the next user message) or ``propose_repair`` (the
        answer). Same cap, records and cost columns as the function-calling loop.
        """
        final = ToolSpec(
            TOOL_NAME,
            "Return exactly one action for this construction; it ends your turn.",
            schema,
        )
        offered = list(tools) + [final]
        names = {t.name for t in tools}
        free = {t.name for t in tools if getattr(t, "uncapped", False)}
        free_used = 0
        envelope = guided_envelope(offered)
        messages = self._messages(
            system + "\n\n" + guided_instructions(offered), payload_text, images
        )
        records: list[dict] = []
        usage_total: dict[str, Any] = {}
        cost_total: float | None = None
        used = 0
        model_name = ""
        last_id = None
        seconds = 0.0
        for request_number in range(1, self.max_turns + 1):
            body = self._base_body(messages, temperature)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "next_call",
                    "strict": True,
                    "schema": envelope,
                },
            }
            try:
                parsed = self._post(body)
            except ProviderError as exc:
                exc.tool_calls = records
                exc.cost_usd = cost_total
                exc.requests = request_number - 1
                raise
            seconds += float(parsed.get("_seconds") or 0.0)
            usage = parsed.get("usage") or {}
            _add_usage(usage_total, usage)
            cost = self.request_cost(usage)
            if cost is not None:
                cost_total = (cost_total or 0.0) + cost
            model_name = parsed.get("model", model_name)
            last_id = parsed.get("id", last_id)
            message = parsed["choices"][0].get("message") or {}
            text = message.get("content") or ""
            try:
                call = self._parse_object(text)["call"]
                name, arguments = call["name"], call["arguments"]
            except (KeyError, TypeError) as exc:
                raise ProviderSchemaViolation(
                    f"the guided answer is not a call envelope: {_short(text)}"
                ) from exc
            if name == TOOL_NAME:
                if not isinstance(arguments, dict):
                    raise ProviderSchemaViolation(
                        f"propose_repair arguments are not an object: {_short(text)}"
                    )
                return ModelResponse(
                    content=arguments,
                    model=model_name,
                    raw_text=json.dumps(arguments),
                    usage=usage_total,
                    generation_id=last_id,
                    structured_mode="guided_json",
                    tool_calls=records,
                    requests=request_number,
                    cost_usd=cost_total,
                    settings=self.settings.to_dict(),
                    seconds=round(seconds, 3),
                    schema_enforcement=self.schema_enforcement(enforced=True),
                )
            result, counted, uncapped = tool_call_gate(
                name, names, free, used, free_used, cap,
                f"give your answer now by calling {TOOL_NAME}",
            )  # fmt: skip
            if result is None:
                used += int(counted)
                free_used += int(uncapped)
                try:
                    result = run_tool(
                        name, arguments if isinstance(arguments, dict) else {}
                    )
                except Exception as exc:  # a tool failure is information, not a crash
                    result = f"the tool failed: {exc}"
            records.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": _short(result, 600),
                    "counted": counted,
                    **({"uncapped": True} if uncapped else {}),
                }
            )
            assistant = {"role": "assistant", "content": text}
            assistant.update(self._assistant_echo(message))
            messages.append(assistant)
            messages.append({"role": "user", "content": f"Result of {name}:\n{result}"})
        raise ProviderSchemaViolation(
            f"no {TOOL_NAME} call within {self.max_turns} requests"
        )

    def request_cost(self, usage: Mapping[str, Any]) -> float | None:
        return None

    def _assistant_echo(self, message: Mapping[str, Any]) -> dict:
        echo = super()._assistant_echo(message)
        thinking = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(thinking, str) and thinking:
            # vLLM 0.31 reads ``reasoning`` back (chat_utils); older vLLM and SGLang
            # read ``reasoning_content``. Both carry the same text.
            echo["reasoning"] = thinking
            echo["reasoning_content"] = thinking
        return echo


def chat_completions_url(base_url: str) -> str:
    """``base_url`` as a ``/chat/completions`` URL (a bare ``.../v1`` is completed)."""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return url


def make_provider(
    model: str | None = None,
    *,
    base_url: str | None = None,
    provider: str | None = None,
    **options,
) -> ChatCompletionsProvider:
    """OpenRouter for a model ID, or an OpenAI-compatible server.

    ``provider="local"`` is the preset for the locally hosted model: the model defaults
    to :data:`LOCAL_MODEL`, the URL to ``$AUTOWAN_LOCAL_BASE_URL``, the key (optional)
    to ``$AUTOWAN_LOCAL_API_KEY``, and a request may run 15 minutes (a 9B model on one
    RTX 3090 writes ~16k thinking tokens in about four). A ``base_url`` without a
    preset is any OpenAI-compatible server.
    """
    if provider not in (None, *PROVIDERS):
        raise ValueError(f"unknown provider {provider!r}; one of {PROVIDERS}")
    if provider == "local":
        base_url = base_url or os.environ.get(LOCAL_BASE_URL_ENV)
        if not base_url:
            raise ProviderError(
                f"--provider local needs the server's URL: pass --base-url or set "
                f"{LOCAL_BASE_URL_ENV} (e.g. http://localhost:8000/v1)"
            )
        options.setdefault("api_key_env", LOCAL_API_KEY_ENV)
        options.setdefault("timeout_s", 900.0)
        options.setdefault("max_attempts", 2)
        options.pop("min_balance_usd", None)
        return OpenAICompatibleProvider(
            model=model or LOCAL_MODEL,
            base_url=chat_completions_url(base_url),
            **options,
        )
    if base_url:
        options.pop("min_balance_usd", None)
        return OpenAICompatibleProvider(
            model=model or DEFAULT_MODEL,
            base_url=chat_completions_url(base_url),
            **options,
        )
    return OpenRouterProvider(model=model or DEFAULT_MODEL, **options)
