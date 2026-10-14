"""Export comparable verdict rows without losing their material scales.

The same schema serves every engine. Provenance lives beside the rows so a results
reader can load a table directly and still identify the construction that was scored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .gate import CertificationGate, GateReport

VERDICT_COLUMNS = (
    "check_id",
    "status",
    "value",
    "threshold",
    "scale",
    "applies",
    "profile",
    "profile_hash",
)
#: The schema before gate profiles (to 2026-10-07): rows written then lack the last two.
PRE_PROFILE_VERDICT_COLUMNS = VERDICT_COLUMNS[:6]


def verdict_rows(report: GateReport, gate: CertificationGate) -> list[dict]:
    scales = {
        c.check_id: c.threshold.scale if c.threshold else None for c in gate.checks
    }
    return [
        {
            "check_id": v.check_id,
            "status": (
                "withheld"
                if not v.applies
                else (
                    "non_evaluable"
                    if not v.evaluable
                    else "passed" if v.passed else "failed"
                )
            ),
            "value": v.metric_value,
            "threshold": v.threshold_value,
            "scale": scales.get(v.check_id),
            "applies": bool(v.applies),
            "profile": v.profile or report.profile,
            "profile_hash": v.profile_hash or report.profile_hash,
        }
        for v in report.verdicts
    ]


def write_gate_verdicts(
    path: str | Path,
    report: GateReport,
    gate: CertificationGate,
    *,
    material: str,
    backend: str,
    seed: str,
    extra_provenance=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict_rows(report, gate), indent=2) + "\n")
    provenance = {
        **(extra_provenance or {}),
        "material": material,
        "backend": backend,
        "seed": seed,
        "construction_id": report.construction_id,
        "material_class": report.material_class,
        "passed": report.passed,
        "gate_profile": report.profile,
        "gate_profile_hash": report.profile_hash,
        "gate_check_ids": [c.check_id for c in gate.checks],
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    path.with_name("provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return path
