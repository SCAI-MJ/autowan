"""Keep engine knowledge and execution outside the certification definition.

A backend translates a parent's files into the quantities the gate already reads.
It may contribute checks, but the gate itself enforces that none replace the core.

The seams, and what a third engine hangs off each of them:

* :meth:`Backend.read_parent` returns the one :class:`~.qe_outputs.ParentCalculation`
  the checks consume -- eV, Angstrom, k-points in the convention the interpolation
  converts from -- so a threshold that is a fraction of the occupied bandwidth is a
  fraction of the same quantity whichever engine produced it.
* :meth:`Backend.read_construction` / :meth:`Backend.write_derived` /
  :meth:`Backend.deck_files` own the deck grammar; :attr:`Backend.knowledge` records
  that grammar and the engine's failure signatures as data, so the modifier's
  knowledge of an engine is a partition, not a second prompt.
* :meth:`Backend.make_runner` supplies a :class:`~.execution.ConstructionRunner`
  whose overlap half is the engine's and whose localisation half is shared.
* :meth:`Backend.refusal_reason` turns an engine's own refusal into the message the
  loop reports under ``calculation_did_not_run`` -- a deck the engine declines is
  never a pass, and never silently a failure of some other check.
* :meth:`Backend.contributed_checks` adds checks to the gate profile's set. A
  contributed check that shares a profile check's identifier replaces it, with a
  :class:`~.gate.GateProfileWarning`, and every verdict records the changed check-list
  hash; acceptance stays the conjunction of them all.
* :meth:`Backend.route` names the construction route (``mlwf`` for both shipped
  engines); it reaches the checks as ``inputs["route"]`` and is where route-scoped
  withholding attaches when an engine offers a second route. The gate stays
  class-scoped here.

Because a backend is one module set behind these seams, a public branch can carry the
package without one of them and the gate does not notice.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..core_checks import CORE_CHECKS
from ..execution import ConstructionRunner
from ..gate import CertificationGate, Check
from ..inputs import ConstructionInputs
from ..parent_adequacy import AdequacyPartition, assess_parent
from ..qe_outputs import ParentCalculation


class BackendError(RuntimeError):
    """The requested engine or launch option is not registered."""


@dataclass(frozen=True)
class DeckRule:
    file: str
    spelling: str
    note: str = ""


@dataclass(frozen=True)
class FailureSignature:
    pattern: str
    meaning: str
    where: str = ""


@dataclass(frozen=True)
class EngineKnowledge:
    engine: str
    deck_grammar: Mapping[str, DeckRule] = field(default_factory=dict)
    unsupported_parameters: tuple[str, ...] = ()
    unsupported_reasons: Mapping[str, str] = field(default_factory=dict)
    failure_signatures: tuple[FailureSignature, ...] = ()
    overlap_step_keys: tuple[str, ...] = ()

    def supports(self, parameter: str) -> bool:
        return (
            parameter in self.deck_grammar
            and parameter not in self.unsupported_parameters
        )

    def explain(self, text: str) -> str | None:
        for signature in self.failure_signatures:
            if text and re.search(signature.pattern, text, re.IGNORECASE):
                return signature.meaning
        return None


class Backend(ABC):
    name: str
    knowledge: EngineKnowledge
    adequacy: AdequacyPartition | None = None

    def assess_parent(
        self, parent_source, inputs, *, declared_moments=None, hamiltonian=None
    ):
        if self.adequacy is None:
            return None
        facts = self.adequacy.read_facts(parent_source)
        demand = self.adequacy.read_demand(inputs, facts)
        return assess_parent(
            facts,
            demand,
            prescribe=self.adequacy.prescribe,
            declared_moments=declared_moments,
            hamiltonian=hamiltonian,
        )

    @abstractmethod
    def read_parent(self, source: str | Path) -> ParentCalculation:
        raise NotImplementedError

    @abstractmethod
    def read_construction(self, run_dir: Path, seed: str) -> ConstructionInputs:
        raise NotImplementedError

    @abstractmethod
    def write_derived(
        self,
        source: ConstructionInputs,
        run_dir: Path,
        new_seed: str,
        changes: Mapping[str, Any],
        fermi_energy_ev: float | None = None,
        species_counts: Mapping[str, int] | None = None,
    ) -> ConstructionInputs:
        raise NotImplementedError

    @abstractmethod
    def deck_files(self, inputs: ConstructionInputs) -> tuple[Path, ...]:
        raise NotImplementedError

    @abstractmethod
    def make_runner(self, run_dir: Path, **launch) -> ConstructionRunner:
        raise NotImplementedError

    @abstractmethod
    def refusal_reason(self, run_dir: Path, seed: str) -> str | None:
        raise NotImplementedError

    def contributed_checks(self) -> tuple[Check, ...]:
        return ()

    def route(self, inputs: ConstructionInputs) -> str:
        return "mlwf"


BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend, name: str | None = None) -> None:
    name = backend.name if name is None else name
    if name in BACKENDS:
        raise BackendError(f"backend {name!r} is already registered")
    BACKENDS[name] = backend


def get_backend(name_or_instance: str | Backend) -> Backend:
    if isinstance(name_or_instance, Backend):
        return name_or_instance
    if name_or_instance not in BACKENDS:
        raise BackendError(
            f"unknown backend {name_or_instance!r}; known: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[name_or_instance]


def gate_for(backend: str | Backend, extra=()) -> CertificationGate:
    backend = get_backend(backend)
    return CertificationGate(CORE_CHECKS, (*backend.contributed_checks(), *extra))
