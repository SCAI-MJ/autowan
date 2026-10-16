"""Loading checks somebody else wrote.

A contributed check is executable code, which makes this the one place in the workflow
where trust runs outward rather than inward: everywhere else the model's output is
data to be validated, and here a file decides what "certified" means. Two things
follow.

Loading is explicit. Nothing is imported because it happens to sit in a directory —
a path or a module has to be named, so a check cannot arrive by being dropped
somewhere. A loaded check is added to the run's gate profile, or, when it shares an
identifier with one of the profile's checks, replaces it with a
:class:`~.gate.GateProfileWarning` (every check is removable).
Either way the verdicts record the profile and the hash of the effective check list,
so a contribution can change what "passed" means but never silently.

A module contributes by exposing ``CHECKS``, a sequence of :class:`~.gate.Check`. Each
carries its contributor in ``origin``, and a check claiming ``core`` is refused: the
provenance of a criterion is part of the criterion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .gate import Check, GateDefinitionError

#: The name a contributed module must expose.
ENTRY_POINT = "CHECKS"


class PluginError(RuntimeError):
    """A contributed module could not be loaded, or does not contribute checks."""


def load_from_path(path: str | Path) -> tuple[Check, ...]:
    """Load contributed checks from one Python file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise PluginError(f"no such file: {path}")

    spec = importlib.util.spec_from_file_location(f"autowan_contrib_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PluginError(f"{path} is not importable as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PluginError(f"{path} raised while being imported: {exc}") from exc

    return _collect(module, str(path))


def load_from_module(dotted: str) -> tuple[Check, ...]:
    """Load contributed checks from an installed module."""
    try:
        module = importlib.import_module(dotted)
    except Exception as exc:
        raise PluginError(f"{dotted} could not be imported: {exc}") from exc
    return _collect(module, dotted)


def _collect(module, where: str) -> tuple[Check, ...]:
    checks = getattr(module, ENTRY_POINT, None)
    if checks is None:
        raise PluginError(
            f"{where} exposes no {ENTRY_POINT}; a contributed module names the checks "
            f"it adds in a sequence called {ENTRY_POINT}"
        )
    if isinstance(checks, Check):
        checks = (checks,)
    if not isinstance(checks, Sequence):
        raise PluginError(f"{where}: {ENTRY_POINT} is not a sequence of checks")

    out = []
    for item in checks:
        if not isinstance(item, Check):
            raise PluginError(
                f"{where}: {ENTRY_POINT} holds a {type(item).__name__}, not a Check"
            )
        if item.origin == "core":
            raise GateDefinitionError(
                f"{where}: check {item.check_id!r} claims origin 'core'. A contributed "
                "check names its contributor; the provenance of a criterion is part of it"
            )
        out.append(item)
    if not out:
        raise PluginError(f"{where}: {ENTRY_POINT} is empty")
    return tuple(out)


def load(sources: Iterable[str]) -> tuple[Check, ...]:
    """Load from a mixed list of file paths and dotted module names."""
    loaded: list[Check] = []
    for source in sources:
        text = str(source)
        found = (
            load_from_path(text)
            if text.endswith(".py") or Path(text).is_file()
            else load_from_module(text)
        )
        loaded.extend(found)
    seen: set[str] = set()
    for check in loaded:
        if check.check_id in seen:
            raise GateDefinitionError(
                f"two contributed checks share the identifier {check.check_id!r}; a "
                "verdict has to say which criterion produced it"
            )
        seen.add(check.check_id)
    return tuple(loaded)
