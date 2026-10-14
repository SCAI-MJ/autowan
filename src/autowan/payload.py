"""What is sent to the model, assembled so a person can read it first.

Two properties matter here and they pull in the same direction. A user running this
tool with their own API key is entitled to see exactly what leaves their machine, and
a claim about what a model recovered from general principles is only checkable if the
evidence it was given is on the record. So the payload is built as data, rendered for
a human, and written to the run log — never assembled inline at the call site where it
would exist only in the request.

The payload carries evidence and no policy. Policy lives in :mod:`prompts` and is the
same text for every material; everything here is specific to this construction and is
what the model is supposed to reason from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .actions import EDITABLE_PARAMETERS, STAGE_ACTIONS, Stage
from .families import (
    DESCRIPTIONS,
    Method,
    as_method,
    available,
    knob_unavailable,
    knobs_for,
)


@dataclass
class Artifact:
    """A file the model may be shown, named so it can ask for it by name."""

    name: str
    path: Path
    media_type: str
    caption: str


@dataclass
class RepairPayload:
    """Everything the model sees about one refused construction."""

    material_signature: dict[str, Any]
    """Electronic-structure facts, not an identity.

    The manifold is described by what it does — how many bands cross the Fermi level,
    how wide the occupied manifold is, where the projectability step sits — rather
    than by the material's name. A model that recognises "iron" can retrieve an answer
    it already knows; one given only the signature has to reason from the physics, and
    the difference between those two is what this stage measures.
    """

    gate_report: dict[str, Any]
    current_parameters: dict[str, Any]
    convergence_trajectory: list[dict[str, float]] = field(default_factory=list)
    """The last iterations of each minimisation stage.

    Supplied because the endpoint alone cannot distinguish a descent that ran out of
    budget from one that was oscillating, and those need different repairs.
    """

    artifacts: tuple[Artifact, ...] = ()
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    """Earlier attempts in this run, so a repair is not proposed twice."""

    fermi_energy_ev: float | None = None
    """Needed to state the energy knobs as offsets, which is how they are proposed."""

    method: str = "smv"
    """The family the current deck is built by."""
    code: str = "qe"
    """The engine, which decides which families exist."""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Read-only tool calls made while this payload was answered (the record only)."""
    frontier_history: dict[str, Any] | None = None
    """A frontier handoff: the frontier's own runs before this run took over
    (:meth:`~.workflow.FrontierHandoff.model_view`), none of which certified."""

    def editable_parameters(self, stage: Stage) -> list[dict[str, Any]]:
        from .inputs import to_relative

        own = knobs_for(self.method)
        out = []
        for p in EDITABLE_PARAMETERS:
            if p.name not in own or knob_unavailable(self.code, self.method, p.name):
                continue
            current = self.current_parameters.get(p.name)
            if p.relative_to_fermi and self.fermi_energy_ev is not None:
                current = to_relative(p.name, current, self.fermi_energy_ev)
            out.append(
                {
                    "name": p.name,
                    "type": p.kind,
                    "unit": (
                        f"eV relative to the Fermi level"
                        if p.relative_to_fermi
                        else p.unit
                    ),
                    "what_it_does": p.what_it_does,
                    "minimum": p.minimum,
                    "maximum": p.maximum,
                    "current": current,
                    "relative_to_fermi_level": p.relative_to_fermi,
                }
            )
        return out

    def to_dict(self, stage: Stage) -> dict[str, Any]:
        method = as_method(self.method)
        return {
            "stage": stage.value,
            "available_actions": list(STAGE_ACTIONS[stage]),
            "method": method.value,
            "methods_on_this_engine": {
                m.value: DESCRIPTIONS[m] for m in available(self.code)
            },
            "material_signature": self.material_signature,
            "gate_report": self.gate_report,
            "current_parameters": self.current_parameters,
            "editable_parameters": self.editable_parameters(stage),
            "convergence_trajectory": self.convergence_trajectory,
            **(
                {"frontier_runs_before_handoff": self.frontier_history}
                if self.frontier_history is not None
                else {}
            ),
            "attempts_so_far": self.attempt_history,
            "tool_calls": self.tool_calls,
            "artifacts": [
                {"name": a.name, "media_type": a.media_type, "caption": a.caption}
                for a in self.artifacts
            ],
        }

    def render(self, stage: Stage) -> str:
        """The payload as prose, for the person who is paying for the call."""
        d = self.to_dict(stage)
        lines = [
            f"STAGE: {d['stage']}",
            f"ACTIONS AVAILABLE: {', '.join(d['available_actions'])}",
            f"METHOD OF THIS CONSTRUCTION: {d['method'].upper()}",
            "METHODS THIS ENGINE OFFERS (switch_method; method_defaults shows each "
            "starting deck):",
            *[
                f"  {m.upper()}: {text}"
                for m, text in d["methods_on_this_engine"].items()
            ],
            "",
            "ELECTRONIC-STRUCTURE SIGNATURE",
        ]
        for key, value in d["material_signature"].items():
            lines.append(f"  {key}: {value}")

        lines += ["", "GATE REPORT"]
        for check_id, r in d["gate_report"].items():
            if not r.get("applies", True):
                verdict = "not applicable"
            elif not r.get("evaluable", True):
                verdict = "no verdict"
            else:
                verdict = "passed" if r.get("passed") else "FAILED"
            detail = ""
            if r.get("value") is not None:
                detail = f"  = {r['value']:.6g} {r.get('units','')}"
                if r.get("threshold") is not None:
                    detail += f"  (limit {r['threshold']:.6g})"
            note = f"   [{r['note']}]" if r.get("note") else ""
            lines.append(f"  {check_id}: {verdict}{detail}{note}")

        if d["convergence_trajectory"]:
            lines += ["", "CONVERGENCE TRAJECTORY (final iterations)"]
            for row in d["convergence_trajectory"]:
                deltas = row.get("recent_deltas") or []
                lines.append(
                    "  "
                    + "  ".join(
                        f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()
                        if k != "recent_deltas"
                    )
                )
                if deltas:
                    # The steps themselves, not just the last one: a minimisation
                    # nearly there and one wandering both end on a small number, and
                    # only the sequence tells them apart.
                    lines.append(
                        "    last steps (oldest first): "
                        + ", ".join(f"{v:.2e}" for v in deltas)
                    )
                    # Only where the stage did not converge: a stage that met its
                    # criterion is not being stopped by anything, and annotating it
                    # with why it is stuck says something untrue.
                    if not row.get("converged"):
                        lines.append(f"    these steps are {_trend(deltas)}")

        lines += ["", f"PARAMETERS YOU MAY CHANGE ({d['method'].upper()} and shared)"]
        for p in d["editable_parameters"]:
            bounds = ""
            if p["minimum"] is not None or p["maximum"] is not None:
                bounds = f"  [{p['minimum']}, {p['maximum']}]"
            fermi = (
                "  — state this as an offset from the Fermi level, positive above"
                if p["relative_to_fermi_level"]
                else ""
            )
            lines.append(
                f"  {p['name']} = {p['current']!r} {p['unit']}{bounds}{fermi}\n"
                f"      {p['what_it_does']}"
            )

        handoff = d.get("frontier_runs_before_handoff")
        if handoff is not None:
            runs = handoff.get("runs") or []
            lines += [
                "",
                f"THE FRONTIER'S OWN RUNS BEFORE YOU TOOK OVER ({len(runs)} Wannier90 "
                f"runs, executed and judged under this run's criterion)",
                f"  The frontier method ran these {len(runs)} constructions itself: its "
                f"default deck (run 0) and its own retry ladder, points 1 to "
                f"{handoff.get('point')}. None of them certified. Your current deck is "
                f"its last one (ladder point {handoff.get('point')}), and its Wannier90 "
                f"outputs are what the tools read as attempt 0. These are the frontier's "
                f"results, not yours: they spend none of your attempts. Energies are in "
                f"eV relative to the Fermi level; 'deck' gives run 0's values and, for "
                f"later runs, only what changed from run 0.",
            ]
            for entry in runs:
                lines.append(f"  {json.dumps(entry, default=str)}")

        if d["attempts_so_far"]:
            lines += ["", "ALREADY ATTEMPTED IN THIS RUN"]
            for a in d["attempts_so_far"]:
                lines.append(f"  {a}")

        if d["artifacts"]:
            lines += ["", "ARTIFACTS YOU MAY INSPECT"]
            for a in d["artifacts"]:
                lines.append(f"  {a['name']} ({a['media_type']}): {a['caption']}")
        return "\n".join(lines)

    def write(self, path: str | Path, stage: Stage) -> Path:
        """Record the payload beside the run log, so the call is reconstructible."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(stage), indent=2, default=str) + "\n")
        return path


def signature_of(parent, wannier) -> dict[str, Any]:
    """Describe the manifold by its behaviour rather than by the material's name."""
    import numpy as np

    e = parent.eigenvalues_ev
    ef = parent.fermi_energy_ev
    occupied = e[e <= ef]
    signature: dict[str, Any] = {
        "bands_crossing_fermi_level": parent.crossing_bands(),
        "manifold_is_isolated": not parent.is_metallic(),
        "band_gap_ev": round(parent.band_gap_ev(), 4),
        "occupied_bandwidth_ev": (
            round(float(np.ptp(occupied)), 4) if occupied.size else None
        ),
        "bands_available": parent.n_bands,
        "spin_channels": parent.n_spin,
        # One electron per spinor band (spin-orbit / noncollinear), two per band
        # otherwise: what turns a projector set or an electron count into bands.
        "spinor_bands": bool(getattr(parent, "spinor", False)),
        "atoms_in_cell": len(parent.species),
        # Named, not just counted. A repair that names projectors has to name the
        # species carrying them, and a model given only a count invents a placeholder —
        # 'auto' and 'X' were both offered before this line existed.
        "species": {s: parent.species.count(s) for s in dict.fromkeys(parent.species)},
    }
    if wannier is not None and wannier.spreads_ang2 is not None:
        spreads = np.sort(wannier.spreads_ang2)
        signature["wannier_functions"] = int(spreads.size)
        signature["spread_range_ang2"] = [
            round(float(spreads.min()), 4),
            round(float(spreads.max()), 4),
        ]
        signature["spread_ratio_widest_to_narrowest"] = (
            round(float(spreads.max() / spreads.min()), 2)
            if spreads.min() > 0
            else None
        )
    return signature


def _trend(deltas: list[float]) -> str:
    """Whether a run of steps is heading anywhere, said as a measurement.

    The prose policy told the model how to read the sequence and the model went on
    reading the last number: shown eight steps of one magnitude alternating in sign it
    still asked for more iterations, five times out of five. So the comparison is made
    here, where it is a fact about the numbers rather than an inference left to be
    drawn -- the halves of the window, by mean magnitude.
    """
    if len(deltas) < 4:
        return "too few to judge"
    half = len(deltas) // 2
    early = sum(abs(d) for d in deltas[:half]) / half
    late = sum(abs(d) for d in deltas[len(deltas) - half :]) / half
    if early == 0:
        return "flat"
    ratio = late / early
    if ratio < 0.5:
        return (
            f"shrinking ({ratio:.2f}x over the window): more iterations may finish it"
        )
    if ratio > 2.0:
        return f"growing ({ratio:.2f}x over the window): the minimisation is diverging"
    return (
        f"not shrinking ({ratio:.2f}x over the window): the iteration limit is not "
        f"what is stopping this, and raising it will not converge the run"
    )


def trajectory_of(wannier, *, last: int = 8) -> list[dict[str, float]]:
    """The tail of each minimisation, as the model needs it to tell the cases apart."""
    rows: list[dict[str, float]] = []
    for stage in (wannier.disentanglement, wannier.wannierisation):
        if stage is None:
            continue
        rows.append(
            {
                "stage": stage.name,
                "iterations": float(stage.n_iterations),
                "final_value": (
                    float(stage.final_value)
                    if stage.final_value is not None
                    else float("nan")
                ),
                "final_delta": (
                    float(stage.final_delta)
                    if stage.final_delta is not None
                    else float("nan")
                ),
                "converged": float(stage.converged),
                "recent_deltas": [float(d) for d in stage.recent_deltas],
            }
        )
    return rows
