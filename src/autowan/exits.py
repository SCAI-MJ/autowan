"""One row per run, one row per attempt: how each construction ended, and how it got there.

The exits table is what a campaign reports. Besides the verdict it carries the method
family the ending construction was built by, whether any attempt switched family, the
read-only tools the model called, and the stage the run ended in -- so a repair that
needed a reference page, or a family switch, stays distinguishable from one that did not.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

EXIT_COLUMNS = (
    "material",
    "code",
    "exit",
    "stage",
    "attempts",
    "method",
    "method_switched",
    "switches",
    "tools_called",
    "tool_calls",
    "requests",
    "usd",
    "model",
    "reasoning",
    "max_tokens",
    "gate_profile",
    "gate_profile_hash",
    "provider_stops",
    "terminal_reason",
    "schema_enforcement",
    "stalled_attempts",
    "infrastructure_stops",
)

ATTEMPT_COLUMNS = (
    "material",
    "code",
    "attempt",
    "seed",
    "stage",
    "status",
    "method",
    "method_switched",
    "switch_from",
    "switch_to",
    "switch_reason",
    "tools_called",
    "tool_calls",
    "requests",
    "usd",
    "failed_checks",
    "gate_profile",
    "gate_profile_hash",
)


def _tool_names(calls: Iterable[Mapping[str, Any]]) -> str:
    return ";".join(str(c.get("name")) for c in calls)


def _exit_of(record: Mapping[str, Any]) -> str:
    if record.get("passed"):
        attempts = record.get("attempts") or []
        passing = next(
            (a for a in attempts if (a.get("outcome") or {}).get("passed")), None
        )
        if passing and passing.get("stage") == "automatic":
            return "certified automatically"
        return "certified after repair"
    if record.get("stopped_by_provider"):
        return "stopped by the provider"
    if record.get("stopped_by_infrastructure"):
        return "stopped by the infrastructure"
    if any(a.get("invalid_reason") for a in record.get("attempts") or []):
        return "invalid"
    if record.get("report"):
        return "report returned"
    if record.get("prescription"):
        return "DFT prescription returned"
    return "not certified"


def exit_row(
    record: Mapping[str, Any], *, material: str | None = None
) -> dict[str, Any]:
    """The exits-table row for one run record (``Run.to_dict()`` or the log JSON)."""
    attempts = list(record.get("attempts") or [])
    ending = next(
        (a for a in attempts if (a.get("outcome") or {}).get("passed")),
        attempts[-1] if attempts else {},
    )
    calls = [c for a in attempts for c in a.get("tools_called") or []]
    settings = record.get("model_settings") or {}
    profile = record.get("gate_profile") or {}
    costs = [a.get("cost_usd") for a in attempts if a.get("cost_usd") is not None]
    usd = record.get("cost_usd")
    if usd is None and costs:
        usd = round(sum(costs), 6)
    return {
        "material": material or record.get("material"),
        "code": record.get("backend"),
        "exit": _exit_of(record),
        "stage": ending.get("stage"),
        "attempts": len(attempts),
        "method": ending.get("method"),
        "method_switched": any(a.get("method_switched") for a in attempts),
        "switches": ";".join(
            f"{a['switch']['from']}->{a['switch']['to']}@{a['index']}"
            for a in attempts
            if a.get("switch")
        ),
        "tools_called": _tool_names(calls),
        "tool_calls": len(calls),
        "requests": sum(int(a.get("model_requests") or 0) for a in attempts),
        "usd": usd,
        "model": record.get("model"),
        "reasoning": (
            (settings.get("reasoning") or {}).get("effort")
            if settings.get("reasoning")
            else ("provider default" if record.get("model") else None)
        ),
        "max_tokens": settings.get("max_tokens"),
        "gate_profile": profile.get("name"),
        "gate_profile_hash": profile.get("hash"),
        "provider_stops": len(record.get("provider_stops") or []),
        "terminal_reason": record.get("terminal_reason"),
        "schema_enforcement": _enforcement(record.get("schema_enforcement")),
        "stalled_attempts": ";".join(
            f"{a.get('index')}:{(a.get('stalled') or {}).get('step')}"
            for a in attempts
            if a.get("stalled")
        ),
        "infrastructure_stops": len(record.get("infrastructure_stops") or []),
    }


def _enforcement(summary: Mapping[str, Any] | None) -> str | None:
    """``off``, or the mechanism with how many answers it held for (e.g. ``3/3``)."""
    if not summary:
        return None
    if not summary.get("requested"):
        return "off"
    held = int(summary.get("answers_enforced") or 0)
    total = held + int(summary.get("answers_unenforced") or 0)
    return f"{summary.get('mechanism')} {held}/{total}"


def attempt_rows(
    record: Mapping[str, Any], *, material: str | None = None
) -> list[dict]:
    rows = []
    for a in record.get("attempts") or []:
        switch = a.get("switch") or {}
        rows.append(
            {
                "material": material or record.get("material"),
                "code": record.get("backend"),
                "attempt": a.get("index"),
                "seed": a.get("seed"),
                "stage": a.get("stage"),
                "status": a.get("status"),
                "method": a.get("method"),
                "method_switched": bool(a.get("method_switched")),
                "switch_from": switch.get("from"),
                "switch_to": switch.get("to"),
                "switch_reason": switch.get("reason"),
                "tools_called": _tool_names(a.get("tools_called") or []),
                "tool_calls": len(a.get("tools_called") or []),
                "requests": a.get("model_requests"),
                "usd": a.get("cost_usd"),
                "failed_checks": ";".join(
                    (a.get("outcome") or {}).get("failed_check_ids") or ()
                ),
                "gate_profile": a.get("gate_profile"),
                "gate_profile_hash": a.get("gate_profile_hash"),
            }
        )
    return rows


def write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], columns) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})
    return path
