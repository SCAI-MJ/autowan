"""Watching an engine step for progress, and stopping it only once progress has stopped.

User decision (2026-10-08): the package carries no fixed wall-time limit. Some
materials take a long time, and a limit fixed in seconds cannot tell a slow material
from a stuck calculation. A Wannier90 step is already bounded by the deck's own
iteration counts (``dis_num_iter``, ``num_iter``, capped by the action vocabulary), so
the remaining risk is a step that stops making progress. That is what this module
detects.

The rule. A step is stopped only when it has shown **no progress** for longer than

* inside an iteration loop (the disentanglement or wannierisation loop of Wannier90,
  the k-point loop of pw2wannier90, VASP's electronic steps):
  ``max(STALL_FLOOR_S, STALL_FACTOR x median interval between iterations)``, with the
  median taken over the intervals observed so far in that loop of this step (or, while
  the loop has none yet, over the step's earlier loops);
* outside a loop (start-up, between the two loops, writing the results), or inside the
  step's first loop before an interval has been observed:
  ``max(STALL_FLOOR_S, REFERENCE_FACTOR x attempt 0's time for the same step x S)``,
  where ``S`` scales attempt 0's time to this deck's size (:func:`size_scale`). Without
  an attempt-0 time for that step (attempt 0 itself, or a step attempt 0 never ran)
  there is **no limit** there.

Progress is anything the step writes: a new iteration line, any other new line in its
log, or any growth of the files it produces. Only an iteration line advances the
per-iteration clock; everything else just resets the idle time.

The interval is measured between successive *observations* of new iteration lines, not
from the per-line timestamps Wannier90 prints. An engine that buffers its output makes
iterations visible in blocks, and the idle time a buffered run legitimately shows is a
block, not an iteration. On cluster2, Wannier90 3.1 flushes every line (V3Si
disentanglement: one 80-byte line every 36 s; SrVO3 wannierisation: one 2.8 kB cycle
block every 5 s), so there the interval is the iteration.
"""

from __future__ import annotations

import glob
import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

#: N, dimensionless. How many median iteration intervals a step may go without
#: progress inside a loop. Calibration (2026-10-08, every ``.wout`` under
#: the development campaigns' run trees: 5824 loops, 2877 disentanglement and 2947
#: wannierisation, read from Wannier90's own per-iteration time column and applied
#: online, i.e. against the running median at each iteration):
#:
#: * per-iteration time is stationary -- both loops repeat the same linear algebra on
#:   matrices of fixed size -- and the 99th-percentile interval of a loop never exceeds
#:   8.7x its median (loops with a median of at least 0.2 s);
#: * the largest excursions are absolute, not proportional: the longest single interval
#:   anywhere is 96 s, against medians of 0.2-1.7 s (one-off file-system or scheduling
#:   pauses), which is what the floor absorbs;
#: * with this factor and the floor, **no** historical loop would have been stopped;
#:   with the factor alone (no floor) 401 would have.
#:
#: 20 is a little over twice the largest proportional spread observed, so a node
#: twice as loaded as the worst one seen still leaves a slow but progressing loop alone,
#: while a loop that has truly stopped is caught after 20 iterations' worth of silence
#: (V3Si disentanglement at 36 s per iteration: 12 minutes, against a 6 h fixed limit).
STALL_FACTOR = 20.0

#: Floor on every limit, in seconds. A few minutes, so start-up, a checkpoint write or a
#: file-system pause is never read as a stall: three times the longest single interval
#: in the calibration set (96 s), and longer than any idle time a fast loop (median
#: 0.01-1 s) can show while it is still running.
STALL_FLOOR_S = 300.0

#: M, dimensionless. Outside a loop, the limit is M x attempt 0's time for the same
#: step, scaled to this deck's size. Generous on purpose: it covers a phase that has no
#: iterations to measure (reading a 3.8 GB .mmn, writing the Hamiltonian), and it
#: compares a *phase* of this step with the *whole* of attempt 0's step, so a
#: legitimate phase stays below one reference step even before the factor. The factor
#: absorbs node load: cluster2 pw2wannier90 logs show wall/CPU of 1.2-1.3 over a step
#: and up to 14 in its I/O-bound start-up (V3Si: init_pw2wan 7 s CPU, 95 s wall).
REFERENCE_FACTOR = 10.0

#: Seconds between looks at the step's files. Not a limit: it sets how promptly a stall
#: is seen and how finely intervals are resolved, and it is far below the floor.
POLL_S = 5.0


@dataclass(frozen=True)
class StallPolicy:
    """The stall rule's constants, recorded with every stopped step."""

    factor: float = STALL_FACTOR
    floor_s: float = STALL_FLOOR_S
    reference_factor: float = REFERENCE_FACTOR
    poll_s: float = POLL_S

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# what a line of output means
# ---------------------------------------------------------------------------

#: A line is an iteration (with how many iterations it stands for), part of an
#: iteration's printout (neutral), or anything else (a phase boundary).
ITERATION, NEUTRAL, BOUNDARY = "iteration", "neutral", "boundary"

_W90_ITERATION = re.compile(r"^\s*\d+\s+\S.*<--\s*(DIS|CONV)\s*$")
_W90_NEUTRAL = re.compile(
    r"<--\s*(SPRD|DLTA|CONV|DIS)\s*$"  # unnumbered: the loop's header box
    r"|^\s*Cycle:\s*\d+"
    r"|^\s*WF centre and spread"
    r"|^\s*Sum of centres and spreads"
    r"|^\s*[-+=|]*\s*$"
)


def wannier90_line(line: str) -> tuple[str, int]:
    """Wannier90's ``.wout``: a numbered ``<-- DIS`` or ``<-- CONV`` line is an iteration.

    The rest of a wannierisation cycle's printout (centres, ``<-- SPRD``, ``<-- DLTA``,
    the separator) is neutral. Anything else -- ``Final Omega_I``, ``<<< ... >>>``,
    ``Final State``, timing lines -- ends the loop it follows.
    """
    if _W90_ITERATION.match(line):
        return ITERATION, 1
    if _W90_NEUTRAL.search(line):
        return NEUTRAL, 0
    return BOUNDARY, 0


_INTEGERS = re.compile(r"^\s*\d+(\s+\d+)*\s*$")


def pw2wannier90_line(line: str) -> tuple[str, int]:
    """pw2wannier90's report: the k-point loops print each k-point index, ten a line."""
    if not line.strip():
        return NEUTRAL, 0
    if _INTEGERS.match(line):
        return ITERATION, len(line.split())
    return BOUNDARY, 0


_VASP_ITERATION = re.compile(r"^\s*(DAV|RMM|DIA|CG|SDA|EDD\w*):\s*\d+")


def vasp_line(line: str) -> tuple[str, int]:
    """VASP's standard output: each electronic step (``DAV:``, ``RMM:``, ``DIA:``) is one."""
    if not line.strip():
        return NEUTRAL, 0
    if _VASP_ITERATION.match(line):
        return ITERATION, 1
    return BOUNDARY, 0


Classifier = Callable[[str], "tuple[str, int]"]


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


@dataclass
class _Followed:
    signature: tuple | None = None
    offset: int = 0
    partial: str = ""
    partial_counted: int = 0


class LineSource:
    """New lines appended to every file matching ``pattern`` (a path or a glob).

    A file that is replaced or truncated is read again from the start: Wannier90
    rewrites its ``.wout`` when it starts, and the one left by an earlier run of the
    same seedname must not be read as this run's progress.
    """

    def __init__(self, pattern, classify: Classifier, *, count_partial=False):
        # an open file (a step's standard output) is followed through its descriptor
        self.handle = pattern if hasattr(pattern, "fileno") else None
        self.pattern = None if self.handle is not None else str(pattern)
        self.classify = classify
        self.count_partial = count_partial
        self._files: dict[str, _Followed] = {}

    def _targets(self) -> list[str]:
        if self.handle is not None:
            return ["<handle>"]
        return _matches(self.pattern)

    def _signature(self, target: str) -> tuple | None:
        if self.handle is None:
            return _signature(target)
        try:
            st = os.fstat(self.handle.fileno())
        except (OSError, ValueError):
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    def _read(self, target: str, offset: int, size: int) -> bytes:
        if self.handle is not None:
            return os.pread(self.handle.fileno(), size, offset)
        with open(target, "rb") as handle:
            handle.seek(offset)
            return handle.read(size)

    def prime(self) -> None:
        """Skip whatever is already there when the step starts."""
        for target in self._targets():
            sig = self._signature(target)
            if sig is not None:
                self._files[target] = _Followed(sig, sig[1])

    def poll(self) -> list[tuple[str, int]]:
        events: list[tuple[str, int]] = []
        for path in self._targets():
            sig = self._signature(path)
            if sig is None:
                continue
            state = self._files.setdefault(path, _Followed())
            if state.signature is not None and (
                sig[0] != state.signature[0] or sig[1] < state.offset
            ):
                state.offset, state.partial, state.partial_counted = 0, "", 0
            state.signature = sig
            if sig[1] <= state.offset:
                continue
            try:
                chunk = self._read(path, state.offset, sig[1] - state.offset)
            except (OSError, ValueError):
                continue
            state.offset += len(chunk)
            text = state.partial + chunk.decode("utf-8", errors="replace")
            *lines, state.partial = text.split("\n")
            if lines:
                # the partial line, now complete, is classified whole; whatever of it
                # was already counted as progress is not counted again
                first_kind, first_count = self.classify(lines[0])
                events.append((first_kind, max(0, first_count - state.partial_counted)))
                events.extend(self.classify(line) for line in lines[1:])
                state.partial_counted = 0
            if chunk:
                events.append(("activity", 0))
            if self.count_partial and state.partial.strip():
                kind, count = self.classify(state.partial)
                if kind == ITERATION and count > state.partial_counted:
                    events.append((ITERATION, count - state.partial_counted))
                    state.partial_counted = count
        return events


class ActivitySource:
    """Any change to the files matching these patterns: the step is doing something."""

    def __init__(self, patterns: Iterable[str | Path]):
        self.patterns = [str(p) for p in patterns]
        self._seen: dict[str, tuple] = {}

    def prime(self) -> None:
        self._seen = self._scan()

    def _scan(self) -> dict[str, tuple]:
        seen = {}
        for pattern in self.patterns:
            for path in _matches(pattern):
                sig = _signature(path)
                if sig is not None:
                    seen[path] = sig
        return seen

    def poll(self) -> list[tuple[str, int]]:
        now = self._scan()
        changed = any(self._seen.get(path) != sig for path, sig in now.items())
        self._seen = now
        return [("activity", 0)] if changed else []


def _matches(pattern: str) -> list[str]:
    if any(ch in pattern for ch in "*?["):
        return sorted(glob.glob(pattern))
    return [pattern] if os.path.exists(pattern) else []


def _signature(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


# ---------------------------------------------------------------------------
# deck size, for scaling attempt 0's time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeckSize:
    """The sizes a Wannier step's cost depends on: bands, functions, k-points."""

    num_bands: int | None = None
    num_wann: int | None = None
    num_kpts: int | None = None

    @classmethod
    def from_values(cls, values: Mapping[str, object]) -> "DeckSize":
        def count(key):
            try:
                return int(float(str(values.get(key)).split()[0]))
            except (TypeError, ValueError, IndexError):
                return None

        kpts = None
        grid = values.get("mp_grid")
        if grid is not None:
            try:
                parts = [int(x) for x in str(grid).replace(",", " ").split()]
                kpts = math.prod(parts) if len(parts) == 3 else None
            except ValueError:
                kpts = None
        return cls(count("num_bands"), count("num_wann"), kpts)


def size_scale(size: DeckSize, reference: DeckSize) -> float:
    """How much more a step on ``size`` may cost than on ``reference`` (at least 1).

    ``(k-points ratio) x (largest of the band and function ratios)^3``: every step is
    linear in the k-points (the k-mesh is the parent's), and at most cubic in the
    matrix dimension -- reading the ``.mmn`` is quadratic in the bands, a
    disentanglement iteration diagonalises a bands-sized matrix per k-point, and a
    wannierisation cycle multiplies function-sized ones. A smaller deck keeps
    attempt 0's bound rather than a tighter one: the bound is meant to be generous.
    """

    def ratio(new, old):
        if not new or not old:
            return 1.0
        return max(1.0, new / old)

    dimension = max(
        ratio(size.num_bands, reference.num_bands),
        ratio(size.num_wann, reference.num_wann),
    )
    return ratio(size.num_kpts, reference.num_kpts) * dimension**3


# ---------------------------------------------------------------------------
# the monitor
# ---------------------------------------------------------------------------


@dataclass
class Stall:
    """Why a step was stopped, as recorded and as handed back to the model."""

    step: str
    idle_s: float
    limit_s: float
    rule: str
    """``loop``: N x the median iteration interval; ``reference``: M x attempt 0's
    time for the step, scaled; both under the floor."""
    iterations: int
    median_interval_s: float | None
    reference_s: float | None
    scale: float | None
    elapsed_s: float
    policy: dict

    def message(self) -> str:
        n = self.policy["factor"]
        floor = self.policy["floor_s"]
        if self.rule == "loop":
            ratio = self.idle_s / self.median_interval_s
            return (
                f"{self.step} stalled: no progress for {self.idle_s:.0f} s, "
                f"{ratio:.0f}x the median iteration time ({self.median_interval_s:.3g} s "
                f"over {self.iterations} iterations; the limit is {n:g}x the median, "
                f"at least {floor:.0f} s). It was stopped before its iteration count "
                f"(num_iter / dis_num_iter) was reached"
            )
        m = self.policy["reference_factor"]
        return (
            f"{self.step} stalled: no progress for {self.idle_s:.0f} s outside an "
            f"iteration loop, beyond {m:g}x attempt 0's {self.reference_s:.0f} s for this "
            f"step scaled by {self.scale:.3g} for this deck's size (limit "
            f"{self.limit_s:.0f} s, at least {floor:.0f} s). It was stopped after "
            f"{self.elapsed_s:.0f} s and {self.iterations} iterations"
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "message": self.message()}


@dataclass
class ProgressMonitor:
    """Follows a step's output and says when it has stopped making progress."""

    step: str
    sources: Sequence[LineSource | ActivitySource]
    policy: StallPolicy = field(default_factory=StallPolicy)
    reference_s: float | None = None
    """Attempt 0's wall time for this step, already multiplied by the size scale."""
    scale: float | None = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.started = self.clock()
        self.last_progress = self.started
        self.iterations = 0
        self.in_loop = False
        self.loops = 0
        self._last_iteration: float | None = None
        self._loop_intervals: list[float] = []
        self._step_intervals: list[float] = []
        for source in self.sources:
            source.prime()

    # -- observing ---------------------------------------------------------

    def observe(self) -> None:
        now = self.clock()
        for source in self.sources:
            for kind, count in source.poll():
                self.last_progress = now
                if kind == ITERATION and count > 0:
                    if not self.in_loop:
                        self.in_loop, self._loop_intervals = True, []
                        self.loops += 1
                    elif (
                        self._last_iteration is not None and now > self._last_iteration
                    ):
                        interval = now - self._last_iteration
                        self._loop_intervals.append(interval)
                        self._step_intervals.append(interval)
                    self._last_iteration = now
                    self.iterations += count
                elif kind == BOUNDARY:
                    self.in_loop, self._last_iteration = False, None

    # -- judging -----------------------------------------------------------

    def median_interval(self) -> float | None:
        intervals = self._loop_intervals or self._step_intervals
        return statistics.median(intervals) if intervals else None

    def limit(self) -> tuple[float | None, str]:
        median = self.median_interval() if self.in_loop else None
        if median is not None:
            return max(self.policy.floor_s, self.policy.factor * median), "loop"
        if self.reference_s is not None:
            return (
                max(
                    self.policy.floor_s, self.policy.reference_factor * self.reference_s
                ),
                "reference",
            )
        return None, "none"

    def stalled(self) -> Stall | None:
        now = self.clock()
        limit, rule = self.limit()
        idle = now - self.last_progress
        if limit is None or idle <= limit:
            return None
        return Stall(
            step=self.step,
            idle_s=round(idle, 1),
            limit_s=round(limit, 1),
            rule=rule,
            iterations=self.iterations,
            median_interval_s=self.median_interval() if rule == "loop" else None,
            reference_s=(
                round(self.reference_s / (self.scale or 1.0), 1)
                if rule == "reference"
                else None
            ),
            scale=self.scale if rule == "reference" else None,
            elapsed_s=round(now - self.started, 1),
            policy=self.policy.to_dict(),
        )

    def summary(self) -> dict:
        """What the step showed, kept with the step's record (and for recalibration)."""
        intervals = self._step_intervals
        return {
            "iterations": self.iterations,
            "loops": self.loops,
            "median_interval_s": (
                round(statistics.median(intervals), 4) if intervals else None
            ),
            "max_interval_s": round(max(intervals), 3) if intervals else None,
        }
