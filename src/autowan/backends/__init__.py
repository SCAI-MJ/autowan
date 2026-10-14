"""Register the shipped engines at import so callers need no setup ordering."""

from .protocol import (
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
from .qe import QeBackend
from .vasp import VaspBackend

register_backend(QeBackend())
register_backend(VaspBackend())

__all__ = [
    "Backend",
    "BackendError",
    "EngineKnowledge",
    "DeckRule",
    "FailureSignature",
    "BACKENDS",
    "get_backend",
    "register_backend",
    "gate_for",
]
