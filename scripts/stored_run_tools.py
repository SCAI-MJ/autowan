"""The read-only repair tools on a stored run, offline; optionally one live tool-call smoke.

Binds :class:`~autowan.tools.RepairTools` to a finished run directory exactly as the
repair loop binds it (``workflow._construction_reader``), with the run's record cut back
to a chosen attempt, so "the last construction that ran" is that attempt. Nothing is
built or run: the tools read the stored ``.wout``/``_hr.dat``/``_wsvec.dat``, the parent
and the reference band path.

    python scripts/stored_run_tools.py RUN_DIR --attempt 0 --out tools.json \
        [--tools truncation_diagnostics band_error_by_energy structure]

``--live`` then asks the default model, through autowan's own provider and schema, to
call ``truncation_diagnostics`` once and answer with ``propose_repair``: a smoke of the
tool loop on a real construction. The request is kept small (a short system text, only
that tool offered, reasoning effort low). The 2026-10-09 smoke on the stored Ba
frontier (attempt 0) spent USD 0.052 over two requests and was cut off at
``--max-tokens 800`` mid-answer (the tool output itself, about 2.2k tokens, was read and
summarised correctly); the default is therefore 2000, which costs a little more.

``RUN_DIR`` is a poster-campaign run directory: it holds ``staging.json`` (parent XML,
reference band path, spin channel) and ``autowan_log.json`` (the run record).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from autowan.actions import Stage, action_schema, validate_action
from autowan.backends import get_backend
from autowan.criteria import read_qe_reference_band_path
from autowan.methods import DeckContext
from autowan.orchestrator import Run
from autowan.providers import (
    DEFAULT_MODEL,
    OpenRouterProvider,
    ProviderError,
    settings_for,
)
from autowan.tools import RepairTools
from autowan.workflow import WorkflowConfig, _construction_reader

TOOLS = ("truncation_diagnostics", "band_error_by_energy", "structure")
SMOKE_SYSTEM = (
    "You repair Wannier-function constructions. You may call the read-only tool offered "
    "before answering; answer only with one propose_repair call, never in prose."
)
SMOKE_PROMPT = (
    "The sparse tight-binding criterion (truncated_hopping_eta2: eta_2 of the model with "
    "every hopping beyond r_c = L_BvK/4 removed must be <= 20 meV) failed on this "
    "construction. Call truncation_diagnostics exactly once. Then call propose_repair "
    "with action 'escalate': in the diagnosis say, in two sentences, which hopping "
    "distances and which functions carry the weight beyond r_c and in which energy "
    "window the truncated model's error is largest; evidence "
    "['truncated_hopping_eta2']; targets []; confidence 0.5."
)


def bind(run_dir: Path, attempt: int, profile: str | None = None) -> RepairTools:
    """The tools of ``run_dir`` as the loop would hand them out after ``attempt``."""
    staging = json.loads((run_dir / "staging.json").read_text())
    record = json.loads((run_dir / "autowan_log.json").read_text())
    run = Run.from_dict(record)
    run.attempts = [a for a in run.attempts if a.index <= attempt]
    if not run.attempts:
        raise SystemExit(f"{run_dir}: the record has no attempt {attempt}")
    if profile:
        run.gate_profile = {**(run.gate_profile or {}), "name": profile}
    seed = run.attempts[-1].seed
    backend = get_backend("qe")
    parent_xml = Path(staging["parent_xml"])
    parent = backend.read_parent(parent_xml)
    reference = read_qe_reference_band_path(
        Path(staging["reference_npz"]),
        fermi_energy_ev=staging.get("reference_fermi_scf_ev"),
        scf_output=staging.get("scf_out") or None,
        nscf_output=staging.get("nscf_out") or None,
    )
    config = WorkflowConfig(
        reference_path=reference,
        gate_profile=(run.gate_profile or {}).get("name", "default"),
        spin_index=int(staging.get("spin_index", 0)),
    )
    inputs = backend.read_construction(run_dir, seed)
    ctx = DeckContext(
        code="qe",
        run_dir=run_dir,
        parent=parent,
        parent_source=parent_xml,
        frontier=inputs,
        spin_index=config.spin_index,
    )
    return RepairTools(
        ctx=ctx,
        backend=backend,
        run=run,
        inputs=inputs,
        seed=seed,
        species=tuple(sorted(set(parent.species))),
        construction_of=_construction_reader(config, backend, parent, run_dir),
    )


def live_smoke(tools: RepairTools, max_tokens: int) -> dict:
    settings = dataclasses.replace(
        settings_for(DEFAULT_MODEL), max_tokens=max_tokens, reasoning={"effort": "low"}
    )
    provider = OpenRouterProvider(
        model=DEFAULT_MODEL, settings=settings, min_balance_usd=10.0
    )
    checks = ("truncated_hopping_eta2",)
    schema = action_schema(
        Stage.REPAIR, checks, checks, tools.species, code="qe", method="smv"
    )
    spec = next(s for s in tools.specs() if s.name == "truncation_diagnostics")
    out: dict = {"model": DEFAULT_MODEL, "settings": settings.to_dict()}
    started = time.time()
    try:
        response = provider.propose(
            system=SMOKE_SYSTEM,
            payload_text=SMOKE_PROMPT,
            schema=schema,
            tools=[spec],
            run_tool=tools,
            tool_call_cap=1,
        )
    except ProviderError as exc:
        out.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            requests=getattr(exc, "requests", None),
            cost_usd=getattr(exc, "cost_usd", None),
        )
        return out
    try:
        action = validate_action(
            response.content,
            stage=Stage.REPAIR,
            evaluated_check_ids=checks,
            unresolved_check_ids=checks,
            allowed_species=tools.species,
            method="smv",
        )
        valid = action.kind
    except Exception as exc:  # recorded, not raised: this is a smoke
        valid = f"refused: {exc}"
    out.update(
        ok=True,
        served_by=response.model,
        tool_calls=[
            {k: c.get(k) for k in ("name", "arguments", "counted")}
            for c in response.tool_calls
        ],
        tools_run=tools.calls,
        final_answer=response.content,
        validates=valid,
        requests=response.requests,
        usage=response.usage,
        cost_usd=response.cost_usd,
        seconds=round(time.time() - started, 1),
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--tools", nargs="+", default=list(TOOLS))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    tools = bind(args.run_dir.resolve(), args.attempt, args.profile)
    result: dict = {
        "run_dir": str(args.run_dir),
        "attempt": args.attempt,
        "seed": tools.seed,
        "outputs": {},
    }
    for name in args.tools:
        text = tools(name, {})
        result["outputs"][name] = json.loads(text) if text.startswith("{") else text
        print(f"{name}: {len(text)} characters", flush=True)
    tools.calls.clear()
    if args.live:
        result["live_smoke"] = live_smoke(tools, args.max_tokens)
        smoke = result["live_smoke"]
        print(
            json.dumps(
                {k: smoke.get(k) for k in ("ok", "error", "validates", "requests")}
            )
        )
        print(f"smoke cost USD {smoke.get('cost_usd')}")
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
