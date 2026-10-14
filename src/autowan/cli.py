"""Command-line entry point.

The workflow is meant to be usable by someone who did not write it, from a directory
containing a Wannier90 deck and the parent calculation it came from. Everything the
run does is printed as it happens — which step was re-run, what the gate measured,
what the model proposed and what it cost — because a tool that spends a user's own API
budget and returns a verdict has to show its work.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

from .backends import get_backend
from .gate import GateDefinitionError, GateProfileWarning
from .gate_profiles import (
    CHECK_GROUPS,
    CHECK_REGISTRY,
    DEFAULT_CHECK_IDS,
    build_gate,
    custom_profile,
    get_profile,
    profile_names,
)
from .orchestrator import RunConfig
from .payload import signature_of
from .plugins import PluginError, load
from .providers import (
    DEFAULT_MODEL,
    LOCAL_BASE_URL_ENV,
    LOCAL_MODEL,
    PROVIDERS,
    make_provider,
)
from .recipes import DEFAULT_CORPUS, distil_run
from .recipes import write as write_recipe
from .verdicts import write_gate_verdicts
from .workflow import WorkflowConfig, wannierize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autowan",
        description="Build a Wannier construction and repair it until a physics gate "
        "accepts it, or say plainly that it did not.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="directory holding <seed>.win and the engine's overlap inputs",
    )
    parser.add_argument(
        "--parent-xml",
        type=Path,
        required=True,
        help="path to <prefix>.save/data-file-schema.xml (QE) or vasprun.xml (VASP)",
    )
    parser.add_argument(
        "--seed", required=True, help="seedname of the automatic method's own deck"
    )
    parser.add_argument("--backend", choices=("qe", "vasp"), default="qe")
    parser.add_argument(
        "--material", help="material name recorded with verdicts; defaults to seed"
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        help="write gate verdict rows and provenance beside them",
    )
    parser.add_argument(
        "--spin-index",
        type=int,
        default=0,
        help="which spin channel of the parent this deck describes",
    )

    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openrouter",
        help="where the model runs: OpenRouter, or the locally hosted open-weight "
        f"model (default model {LOCAL_MODEL}; URL from --base-url or "
        f"${LOCAL_BASE_URL_ENV})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model to consult for repairs, as your provider names it (default "
        f"{DEFAULT_MODEL} on OpenRouter, {LOCAL_MODEL} with --provider local)",
    )
    parser.add_argument(
        "--base-url",
        help="an OpenAI-compatible endpoint (e.g. a local vLLM server) instead of "
        "OpenRouter",
    )
    parser.add_argument(
        "--no-schema-enforcement",
        action="store_true",
        help="do not enforce the action and tool schemas at decode time (strict "
        "function calling on OpenRouter, guided tool_choice=required on vLLM); "
        "enforcement is on by default and recorded per run",
    )
    parser.add_argument(
        "--min-balance",
        type=float,
        default=10.0,
        help="stop (resumably) before a request when the OpenRouter balance is below "
        "this many USD; 0 disables the check",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run the provider stopped, from the attempt it stopped at "
        "(needs --log pointing at its record)",
    )
    parser.add_argument(
        "--frontier-rescan",
        type=Path,
        help="the frontier's retry ladder (CSV/JSON); the frontier_candidates tool reads only its thresholds and order",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        help="the DFT bands on a high-symmetry path the acceptance criteria are scored "
        "on: a QE dft_bands_path.npz or a VASP band-path NSCF directory. Without it a "
        "criterion profile (band_interpolation, sparse_tb, downfolded_d) cannot "
        "certify, because its criterion is non-evaluable",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="score the construction and stop; consult no model",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="after the physics gate accepts, ask a multimodal model "
        "to look at the figures; it may withhold acceptance "
        "but can never grant it",
    )

    parser.add_argument(
        "--total-attempts",
        type=int,
        default=8,
        help="attempts in all, the frontier's included",
    )
    parser.add_argument(
        "--tool-calls", type=int, default=8, help="tool calls per attempt"
    )

    parser.add_argument(
        "--wannier90", default=os.environ.get("WANNIER90", "wannier90.x")
    )
    parser.add_argument(
        "--pw2wannier90", default=os.environ.get("PW2WANNIER90", "pw2wannier90.x")
    )
    parser.add_argument("--vasp", default=os.environ.get("VASP", "vasp_std"))
    parser.add_argument(
        "--mpi",
        default="mpirun -np 1",
        help="launcher for the overlap engine, as a single string",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="MODULE_OR_FILE",
        help="a contributed check module to add to the gate; repeatable. A "
        "contributed check that shares a profile check's name replaces it, with a "
        "warning; acceptance is the conjunction over all of them",
    )
    _add_profile_arguments(parser)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--log", type=Path, help="write the run record here")
    parser.add_argument(
        "--recipes",
        type=Path,
        metavar="DIR",
        default=DEFAULT_CORPUS,
        help="the corpus search_references searches (default: the shipped one)",
    )
    parser.add_argument(
        "--contribute-recipes",
        type=Path,
        metavar="DIR",
        help="distil this run's successful repairs into DIR when it finishes",
    )
    return parser


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """The gate-profile options: what "certified" means for this run."""
    group = parser.add_argument_group("gate profile")
    group.add_argument(
        "--gate-profile",
        default="default",
        choices=profile_names(),
        help="the named gate profile that defines certification (default: default). "
        "See --list-gate-profiles",
    )
    group.add_argument(
        "--add-check",
        action="append",
        default=[],
        metavar="CHECK_ID",
        help="add a registered check, or a check group (convergence), to the "
        "profile; repeatable",
    )
    group.add_argument(
        "--drop-check",
        action="append",
        default=[],
        metavar="CHECK_ID",
        help="drop a check, or a check group, from the profile; repeatable. Dropping "
        "a default check prints a warning and nothing else",
    )
    group.add_argument(
        "--list-gate-profiles",
        action="store_true",
        help="print every gate profile, its checks, hash and target text, and exit",
    )


def resolve_gate_profile(args: argparse.Namespace):
    """The profile ``--gate-profile/--add-check/--drop-check`` describe; warnings printed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GateProfileWarning)
        profile = custom_profile(args.gate_profile, args.add_check, args.drop_check)
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)
    return profile


def list_gate_profiles(stream=None) -> None:
    stream = stream or sys.stdout
    for name in profile_names():
        profile = get_profile(name)
        print(f"{name}  [hash {profile.hash}]", file=stream)
        for check_id in profile.check_ids:
            print(f"  - {check_id}", file=stream)
        print(f"  {profile.target_text()}\n", file=stream)
    optional = sorted(set(CHECK_REGISTRY) - set(DEFAULT_CHECK_IDS))
    print("registered checks outside the default: " + ", ".join(optional), file=stream)
    for group, members in CHECK_GROUPS.items():
        print(f"check group {group}: " + ", ".join(members), file=stream)


def main(argv: list[str] | None = None) -> int:
    if "--list-gate-profiles" in (sys.argv[1:] if argv is None else argv):
        list_gate_profiles()
        return 0
    args = build_parser().parse_args(argv)
    try:
        gate_profile = resolve_gate_profile(args)
    except GateDefinitionError as exc:
        print(f"gate profile: {exc}", file=sys.stderr)
        return 2
    print(f"gate profile: {gate_profile.name} [hash {gate_profile.hash}]")

    backend = get_backend(args.backend)
    launch = (
        {"pw2wannier90": args.pw2wannier90}
        if args.backend == "qe"
        else {"vasp": args.vasp, "parent_dir": args.parent_xml.parent}
    )
    if args.backend == "vasp":
        from .backends.vasp.execution import spin_channel_for
        from .execution import CalculationError

        try:
            launch["spin_channel"] = spin_channel_for(
                args.parent_xml.parent, args.spin_index
            )
        except CalculationError as exc:
            print(exc, file=sys.stderr)
            return 2
    runner = backend.make_runner(
        run_dir=args.run_dir,
        wannier90=args.wannier90,
        mpi_launcher=tuple(args.mpi.split()),
        **launch,
    )
    reference = None
    if args.reference_path is not None:
        from .workflow import _reference_path

        if not args.reference_path.exists():
            print(
                f"--reference-path: {args.reference_path} does not exist",
                file=sys.stderr,
            )
            return 2
        try:
            # read once here, so a bad path fails before any engine runs
            reference = _reference_path(
                WorkflowConfig(reference_path=args.reference_path)
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"--reference-path: {exc}", file=sys.stderr)
            return 2
        print(f"reference band path: {args.reference_path}")
    config = WorkflowConfig(
        run_config=RunConfig(
            total_attempts=args.total_attempts,
            tool_calls_per_attempt=args.tool_calls,
        ),
        model=args.model
        or (LOCAL_MODEL if args.provider == "local" else DEFAULT_MODEL),
        spin_index=args.spin_index,
        draw_figures=not args.no_figures,
        review=args.review,
        recipe_dir=args.recipes,
        gate_profile=gate_profile,
        frontier_rescan=args.frontier_rescan,
        reference_path=reference,
    )

    try:
        extra_checks = load(args.check) if args.check else ()
    except PluginError as exc:
        print(f"contributed check: {exc}", file=sys.stderr)
        return 2
    if extra_checks:
        print(
            f"gate: {len(extra_checks)} contributed check(s) — "
            + ", ".join(f"{c.check_id} ({c.origin})" for c in extra_checks)
        )

    factory = None
    if not args.no_repair and args.provider == "local":
        if not (args.base_url or os.environ.get(LOCAL_BASE_URL_ENV)):
            print(
                f"--provider local needs the server's URL: pass --base-url or set "
                f"{LOCAL_BASE_URL_ENV}",
                file=sys.stderr,
            )
            return 2
        factory = lambda model: make_provider(  # noqa: E731
            model,
            base_url=args.base_url,
            provider="local",
            enforce_schema=not args.no_schema_enforcement,
        )
    elif not args.no_repair and args.base_url:
        factory = lambda model: make_provider(  # noqa: E731
            model,
            base_url=args.base_url,
            enforce_schema=not args.no_schema_enforcement,
        )
    elif not args.no_repair:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print(
                "no OPENROUTER_API_KEY in the environment. This tool runs on a key "
                "you supply and ships none; pass --no-repair to score only.",
                file=sys.stderr,
            )
            return 2
        factory = lambda model: make_provider(  # noqa: E731
            model,
            min_balance_usd=args.min_balance,
            enforce_schema=not args.no_schema_enforcement,
        )

    from .execution import InfrastructureStop, stop_on_termination_signals

    try:
        # SIGTERM (a scheduler wall-time limit, scancel) stops the run as an
        # infrastructure stop: recorded in --log, not a result, resumable (--resume).
        with stop_on_termination_signals():
            result = wannierize(
                run_dir=args.run_dir,
                parent_xml=args.parent_xml,
                seed=args.seed,
                runner=runner,
                provider_factory=factory,
                config=config,
                extra_checks=extra_checks,
                log_path=args.log,
                backend=backend,
                resume=args.resume,
            )
    except InfrastructureStop as exc:
        print(
            f"stopped from outside: {exc}. Not a result; resume with --resume"
            + (f" (record: {args.log})" if args.log else ""),
            file=sys.stderr,
        )
        return 75

    if args.verdicts and result.report is not None:
        scored = result.seed or args.seed
        write_gate_verdicts(
            args.verdicts,
            result.report,
            build_gate(backend, gate_profile, extra_checks, reference_path=reference),
            material=args.material or args.seed,
            backend=backend.name,
            seed=scored,
            # Enough to say which calculation the rows describe: a verdict table
            # without its parent, mesh and engine is a number without a material.
            extra_provenance={
                "parent_xml": str(args.parent_xml),
                "run_dir": str(args.run_dir),
                "mp_grid": backend.read_construction(args.run_dir, scored).values.get(
                    "mp_grid"
                ),
                "engine_binary": (
                    args.vasp if args.backend == "vasp" else args.pw2wannier90
                ),
                "wannier90": args.wannier90,
                "mpi_launcher": args.mpi,
                "reference_path": (
                    str(args.reference_path) if args.reference_path else None
                ),
                "attempts": len(result.run.attempts),
                "terminal_reason": result.run.terminal_reason,
            },
        )

    if args.contribute_recipes:
        # Every run is a candidate entry. Asking people to write recipes by hand
        # collects nothing, so the corpus grows from what the loop already recorded --
        # and only from the steps where a check actually stopped failing.
        entries = distil_run(
            result.run.to_dict(), signature_of(result.parent, None), engine=backend.name
        )
        for entry in entries:
            write_recipe(entry, args.contribute_recipes)
        print(f"\ncontributed {len(entries)} recipe(s) to {args.contribute_recipes}")

    print(f"\n{result.status.upper()}")
    print(f"  construction : {result.seed}")
    print(f"  attempts     : {len(result.run.attempts)}")
    if result.run.terminal_reason:
        print(f"  ended because: {result.run.terminal_reason}")
    for attempt in result.run.attempts:
        verdict = (
            "certified"
            if attempt.outcome and attempt.outcome.passed
            else "refused" if attempt.outcome else "not scored"
        )
        note = attempt.calculation_error or attempt.model_error or ""
        switched = " switched" if attempt.method_switched else ""
        print(
            f"   [{attempt.index}] {attempt.stage:<9} {attempt.method or '?':<5}{switched:<9} "
            f"{verdict:<10} tools {len(attempt.tools_called):<2} {note[:50]}"
        )
    if result.report and not result.certified:
        unresolved = (
            result.report.failed_check_ids + result.report.non_evaluable_check_ids
        )
        print(f"  unresolved   : {', '.join(unresolved)}")
    for figure in result.figures:
        print(f"  figure       : {figure}")
    return 0 if result.certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
