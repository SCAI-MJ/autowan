"""A physics-gated workflow for building Wannier functions without a human in the loop.

An automatic Wannierisation fails often enough that recovery, not the first attempt,
is the hard part — and recovery is where an unconstrained language model would decide
for itself whether a calculation is physically sound. This package is arranged so it
cannot: a deterministic gate judges every construction against the calculation that
produced it, the model's authority expands only after simpler approaches have been
tried and failed, and every action it may take is drawn from a closed vocabulary that
contains no acceptance threshold.

Reading order follows the data:

* :mod:`qe_outputs` and :mod:`w90_outputs` — the measurement, read from the parent
  calculation and from the construction.
* :mod:`projectability` — the deterministic first pass, which derives its SCDM
  parameters from the material's own projectability curve rather than inheriting them.
* :mod:`interpolation` — the model's own bands, Fourier-summed from its real-space
  Hamiltonian, so it can be compared against the parent at the same k-points.
* :mod:`gate`, :mod:`core_checks`, :mod:`criteria` and :mod:`gate_profiles` — what a
  construction must satisfy. Thresholds are fractions of scales the material supplies;
  a named, hashed gate profile says which checks define "passed" for a run.
* :mod:`prompts` and :mod:`payload` — the policy handed to a model, which names no
  material, and the evidence handed with it, which is nothing but this one.
* :mod:`actions` — what a model may propose, and which stage permits it.
* :mod:`plots` and :mod:`review_gate` — the figures, and the second opinion that may
  withhold an acceptance the measurements already granted but can never grant one.
* :mod:`providers` — the call itself, made with a key the user supplies, constrained
  to the schema by whichever mechanism the chosen model offers.
* :mod:`inputs` and :mod:`execution` — writing a derived deck and running only the
  step a change invalidated.
* :mod:`orchestrator` — the state machine that owns the loop, the budgets and the log.
* :mod:`workflow` — :func:`wannierize`, the callable that runs the ladder end to end.
"""

from .actions import (
    Action,
    ActionRejected,
    Stage,
    action_schema,
    validate_action,
)
from .backends import (
    BACKENDS,
    Backend,
    BackendError,
    DeckRule,
    EngineKnowledge,
    FailureSignature,
    gate_for,
    get_backend,
    register_backend,
)
from .core_checks import CONVERGENCE_CHECKS, CORE_CHECKS
from .execution import (
    CalculationError,
    ConstructionRunner,
    DeckRefused,
    InfrastructureStop,
    StepStalled,
    WannierRunner,
    check_band_count,
    stop_on_termination_signals,
)
from .families import Method
from .gate import (
    CertificationGate,
    Check,
    Construction,
    GateReport,
    NonEvaluable,
    ScaleAwareThreshold,
)
from .inputs import ConstructionInputs, read_construction, write_derived
from .interpolation import compare_to_parent, interpolate
from .orchestrator import BudgetExhausted, GateOutcome, Run, RunConfig
from .payload import Artifact, RepairPayload, signature_of, trajectory_of
from .plots import BandFigure, draw_band_comparison, draw_spread_profile
from .plugins import PluginError, load_from_module, load_from_path
from .progress import StallPolicy
from .projectability import ScdmParameters, scdm_parameters_from_projwfc
from .prompts import GENERIC_PRINCIPLES, REPAIR_TASK, system_prompt
from .providers import (
    Image,
    ModelResponse,
    ModelSettings,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    ProviderCreditsExhausted,
    ProviderError,
    ProviderLowBalance,
    ProviderSchemaViolation,
    ToolSpec,
    make_provider,
)
from .qe_outputs import ParentCalculation, read_parent
from .review_gate import DEFAULT_CONCERNS, ReviewGate, ReviewOutcome
from .verdicts import VERDICT_COLUMNS, verdict_rows, write_gate_verdicts
from .w90_outputs import WannierRun, read_run
from .workflow import WannierisationResult, WorkflowConfig, wannierize

__all__ = [
    "BACKENDS",
    "Backend",
    "BackendError",
    "ConstructionRunner",
    "DeckRule",
    "EngineKnowledge",
    "FailureSignature",
    "VERDICT_COLUMNS",
    "gate_for",
    "get_backend",
    "register_backend",
    "verdict_rows",
    "write_gate_verdicts",
    "Action",
    "ActionRejected",
    "Artifact",
    "BandFigure",
    "BudgetExhausted",
    "CONVERGENCE_CHECKS",
    "CORE_CHECKS",
    "CalculationError",
    "DeckRefused",
    "InfrastructureStop",
    "StallPolicy",
    "StepStalled",
    "stop_on_termination_signals",
    "check_band_count",
    "CertificationGate",
    "Check",
    "Construction",
    "ConstructionInputs",
    "DEFAULT_CONCERNS",
    "GENERIC_PRINCIPLES",
    "GateOutcome",
    "GateReport",
    "Image",
    "ModelResponse",
    "NonEvaluable",
    "OpenRouterProvider",
    "ParentCalculation",
    "PluginError",
    "ProviderCreditsExhausted",
    "ProviderError",
    "ProviderSchemaViolation",
    "RepairPayload",
    "ReviewGate",
    "ReviewOutcome",
    "Run",
    "RunConfig",
    "REPAIR_TASK",
    "Method",
    "OpenAICompatibleProvider",
    "ModelSettings",
    "ProviderLowBalance",
    "ToolSpec",
    "make_provider",
    "ScaleAwareThreshold",
    "ScdmParameters",
    "Stage",
    "WannierRun",
    "WannierRunner",
    "WannierisationResult",
    "WorkflowConfig",
    "action_schema",
    "compare_to_parent",
    "draw_band_comparison",
    "draw_spread_profile",
    "interpolate",
    "load_from_module",
    "load_from_path",
    "read_construction",
    "read_parent",
    "read_run",
    "scdm_parameters_from_projwfc",
    "signature_of",
    "system_prompt",
    "trajectory_of",
    "validate_action",
    "wannierize",
    "write_derived",
]
