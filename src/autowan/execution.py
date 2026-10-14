"""Running the two calculation steps, and re-running only what a change invalidated.

A construction is built in two steps with very different costs. The overlaps and
projections come from ``pw2wannier90`` reading the parent wavefunctions, which on a
dense mesh is tens of minutes. The localisation is ``wannier90`` acting on files
already written, and is seconds. Most repairs touch only the second, so re-running
both would spend twenty minutes recomputing something the change could not have
altered.

Which step a change invalidates is not decided here. Each editable parameter already
declares the file it belongs to, so the dependency is read off the vocabulary rather
than maintained as a second list that can drift away from it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Iterable, Mapping, Sequence

from .actions import BY_NAME
from .inputs import InputError, read_excluded_bands, read_win
from .progress import (
    NEUTRAL,
    ActivitySource,
    DeckSize,
    LineSource,
    ProgressMonitor,
    Stall,
    StallPolicy,
    pw2wannier90_line,
    size_scale,
    wannier90_line,
)

#: Files the localisation step needs that the overlap step produces.
OVERLAP_PRODUCTS = ("amn", "mmn", "eig", "nnkp")

#: How many consecutive iterations must meet the tolerance before Wannier90 calls the
#: spread minimisation converged. Wannier90's own default is -1, meaning it runs no
#: convergence test at all — so a deck that does not set this neither converges nor
#: fails to, it simply never reports, and every construction built from it arrives
#: untestable. Enabling the test is deck preparation, not a repair: whether a run met
#: its criterion is a property of the run, and asking a model to switch the criterion on
#: turns our own omission into a verdict about the method being measured.
#:
#: It is preparation of *repaired* decks only. The automatic (frontier) construction is
#: the published method's own deck and runs exactly as its builder wrote it: the QE
#: PDWF deck of aiida-wannier90-workflows sets ``conv_window = 3`` itself, and the VASP
#: SCDM frontier (Varrassi et al. 2025) sets none, so its run is not given a criterion
#: the protocol lacks; the gate judges its spread minimisation from the printed
#: trajectory over this same window (``core_checks.REFERENCE_CONV_WINDOW``).
DEFAULT_CONV_WINDOW = 3

#: The .win keys the workflow writes into a deck that its builder or the model did not:
#: added to every repaired deck, never to the frontier deck (attempt 0).
REPAIR_DECK_PREPARATION = ("conv_window",)


def ensure_convergence_test(win: Path, window: int = DEFAULT_CONV_WINDOW) -> bool:
    """Give a deck a convergence test if it has none. Returns whether one was added."""
    win = Path(win)
    if not win.is_file():
        return False
    text = win.read_text()
    for line in text.splitlines():
        stripped = line.split("!")[0].split("#")[0].strip()
        if stripped.lower().replace(" ", "").startswith("conv_window="):
            return False
    win.write_text(
        f"! convergence test enabled by autowan: Wannier90 runs none by default\n"
        f"conv_window = {window}\n\n" + text
    )
    return True


class CalculationError(RuntimeError):
    """A calculation ran and failed.

    Distinct from :class:`~.providers.ProviderError`: this one is about the physics
    or the deck, and belongs in a different column of the run record.
    """


class DeckRefused(CalculationError):
    """A deck the stored parent or the engine cannot run, for the next proposal to repair.

    A subclass, so every caller that stops on :class:`CalculationError` still stops on
    this. The repair loop is the one caller that tells them apart: a deck it cannot run
    is information for the next attempt, while a step that runs out of a caller's fixed
    time limit, an executable that cannot be launched or overlap files that are missing
    say nothing about the deck and still end the run.
    """


class StepStalled(DeckRefused):
    """A step that stopped making progress and was stopped (:mod:`.progress`).

    A deck refusal: the deck is what made the step stall, so a model-proposed deck
    becomes a refused attempt, its message goes back to the model, and the run goes on.
    """

    def __init__(self, stall: Stall):
        super().__init__(stall.message())
        self.stall = stall

    @property
    def record(self) -> dict:
        return self.stall.to_dict()

    @property
    def kind(self) -> str:
        return self.stall.step.rpartition(":")[2]


class InfrastructureStop(BaseException):
    """The job running the workflow is being stopped from outside.

    A scheduler wall-time limit, a cancellation, a preemption: none of them is a result
    about the deck or the method, and the run is resumable from where it stood. A
    ``BaseException`` (like ``KeyboardInterrupt``), so the many ``except Exception``
    clauses that turn a provider's failure into a recorded answer cannot swallow it.
    """

    def __init__(self, reason: str, signum: int | None = None):
        super().__init__(reason)
        self.signum = signum


@contextlib.contextmanager
def stop_on_termination_signals(signals: Sequence[int] = (signal.SIGTERM,)):
    """Turn SIGTERM into :class:`InfrastructureStop` while the block runs.

    SLURM sends SIGTERM to every process of a job at its wall-time limit, on
    ``scancel`` and on preemption, and SIGKILL ``KillWait`` seconds later (30 s on
    cluster2): time enough for the workflow to stop its engine step and record the run
    as stopped rather than finished. The handler fires once; a repeated signal while
    the record is being written is ignored. Outside the main thread it does nothing.
    """

    def handler(signum, frame):
        signal.signal(signum, signal.SIG_IGN)
        raise InfrastructureStop(
            f"received {signal.Signals(signum).name}: the job was stopped from outside "
            f"(a scheduler wall-time limit, a cancellation or a preemption)",
            signum,
        )

    previous = {}
    try:
        for sig in signals:
            previous[sig] = signal.signal(sig, handler)
    except ValueError:  # not the main thread
        pass
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def check_band_count(win: Path, raw_bands: int) -> None:
    """Refuse a deck whose band count the stored parent cannot satisfy.

    pw2wannier90 writes the overlaps over every stored band the deck does not exclude,
    and Wannier90 requires ``num_bands`` to be exactly that many. With ``raw_bands``
    stored and the exclusion set ``E``: every member of ``E`` lies in
    ``1..raw_bands``, and ``num_bands == raw_bands - |E|``. Strontium oxide stored 52
    bands and excluded 1-4, so it takes 48; a proposal of 26 there cannot be run, and
    finding that out after the overlap step spent tens of minutes and ended the run.
    """
    declared = _as_count(read_win(win).get("num_bands"))
    if declared is None:
        return
    try:
        excluded = read_excluded_bands(win)
    except InputError as exc:
        raise DeckRefused(
            f"{win.name} has invalid exclude_bands for raw_bands = {raw_bands}: {exc}"
        ) from exc
    outside = [band for band in excluded if not 1 <= band <= raw_bands]
    if outside:
        raise DeckRefused(
            f"{win.name} has exclude_bands indices {outside} outside "
            f"1..{raw_bands} (raw_bands = {raw_bands})"
        )
    expected = raw_bands - len(excluded)
    if declared != expected:
        exclusion = _ranges(excluded) or "none"
        raise DeckRefused(
            f"{win.name} asks for num_bands = {declared}, but the parent stored "
            f"{raw_bands} bands and the deck excludes {len(excluded)} of them "
            f"(exclude_bands {exclusion}), so the overlaps will carry {expected}; "
            f"num_bands must be {expected} for this exclusion"
        )


@dataclass
class StepResult:
    step: str
    returncode: int
    seconds: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    progress: dict | None = None
    """What the step showed while it ran: iterations, loops, interval statistics."""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_record(self) -> dict:
        return {
            "step": self.step,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 2),
            "progress": self.progress,
        }


@dataclass
class ConstructionRunner:
    """Share reuse, validation and localisation across overlap engines."""

    run_dir: Path
    wannier90: str = "wannier90.x"
    mpi_launcher: Sequence[str] = ("mpirun", "-np", "1")
    timeout_s: float | None = None
    """A fixed wall-time limit per step. ``None`` (the default): none. The steps are
    bounded by the deck's own iteration counts, and a step that stops progressing is
    caught by :attr:`stall_policy`. A number here is honoured, with a warning: it is
    not scale-aware, and stops a slow material exactly as it stops a stuck one."""
    env: Mapping[str, str] = field(default_factory=dict)
    stall_policy: StallPolicy | None = field(default_factory=StallPolicy)
    """The stall rule (:mod:`.progress`). ``None`` switches stall detection off."""
    reference_steps: dict = field(default_factory=dict)
    """Attempt 0's wall time per step kind, with the deck size it was measured on:
    ``{"localise": {"seconds": ..., "size": {...}}}``. Set by the workflow once the
    frontier construction has run; it bounds the phases of later steps that have no
    iterations to measure."""

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if not self.run_dir.is_dir():
            raise CalculationError(f"no working directory at {self.run_dir}")
        if self.timeout_s is not None:
            warnings.warn(
                f"a fixed step limit of {self.timeout_s:g} s is set; it is not "
                f"scale-aware and stops a slow but progressing material like a stuck "
                f"one. Leave timeout_s unset to rely on the stall detector",
                UserWarning,
                stacklevel=3,
            )

    # -- progress and stalls -----------------------------------------------

    def remember_reference(self, seed: str, steps: Sequence[StepResult]) -> None:
        """Keep attempt 0's step times, for bounding later steps' silent phases."""
        size = asdict(self._deck_size(seed))
        for result in steps:
            if result.ok:
                kind = result.step.rpartition(":")[2]
                self.reference_steps[kind] = {
                    "seconds": float(result.seconds),
                    "size": size,
                }

    def _deck_size(self, seed: str) -> DeckSize:
        try:
            return DeckSize.from_values(read_win(self.run_dir / f"{seed}.win"))
        except Exception:
            return DeckSize()

    def _reference_for(self, step: str) -> tuple[float | None, float | None]:
        """Attempt 0's time for this kind of step, scaled to this deck; or nothing."""
        seed, _, kind = step.rpartition(":")
        reference = self.reference_steps.get(kind)
        if not reference or not reference.get("seconds"):
            return None, None
        scale = size_scale(
            self._deck_size(seed), DeckSize(**(reference.get("size") or {}))
        )
        return float(reference["seconds"]) * scale, scale

    def _progress_sources(self, step: str, cwd: Path, stdout: IO[bytes]) -> list:
        """What shows this step's progress. The overlap engines override ``overlaps``."""
        seed, _, kind = step.rpartition(":")
        directory = self.run_dir
        quiet = LineSource(stdout, lambda line: (NEUTRAL, 0))
        if kind in ("localise", "nnkp"):
            return [
                LineSource(directory / f"{seed}.wout", wannier90_line),
                quiet,
                ActivitySource(
                    [
                        directory / f"{seed}.chk",
                        directory / f"{seed}.nnkp",
                        directory / f"{seed}.werr",
                        directory / f"{seed}_*",
                    ]
                ),
            ]
        if kind == "overlaps":
            return [
                LineSource(stdout, pw2wannier90_line, count_partial=True),
                ActivitySource(
                    [directory / f"{seed}.{ext}" for ext in ("amn", "mmn", "eig")]
                ),
            ]
        return [quiet]

    # -- what a change invalidates ---------------------------------------

    @staticmethod
    def invalidates_overlaps(changes: Iterable[str]) -> bool:
        """Whether these parameters change what ``pw2wannier90`` produces."""
        return any(
            BY_NAME[name].stage_file == "pw2wan" or BY_NAME[name].rebuilds_overlaps
            for name in changes
            if name in BY_NAME
        )

    def overlaps_present(self, seed: str) -> bool:
        return all(
            (self.run_dir / f"{seed}.{ext}").is_file() for ext in ("amn", "mmn", "eig")
        )

    def discard_overlaps(self, seed: str) -> None:
        """Remove any overlap files left under this seedname by an earlier run."""
        for ext in ("amn", "mmn", "eig", "nnkp"):
            self.run_dir.joinpath(f"{seed}.{ext}").unlink(missing_ok=True)

    def reuse_overlaps(self, source_seed: str, target_seed: str) -> list[str]:
        """Copy one construction's overlaps onto another seedname.

        Copied rather than linked: a later attempt that reruns the overlap step must
        not silently rewrite the inputs of an attempt already recorded as finished.
        """
        copied = []
        for ext in OVERLAP_PRODUCTS:
            source = self.run_dir / f"{source_seed}.{ext}"
            if source.is_file():
                shutil.copyfile(source, self.run_dir / f"{target_seed}.{ext}")
                copied.append(ext)
        if not {"amn", "mmn", "eig"}.issubset(copied):
            raise CalculationError(
                f"{source_seed} has no complete set of overlaps to reuse "
                f"(found {', '.join(copied) or 'none'})"
            )
        return copied

    def amn_header(self, seed: str) -> tuple[int, int, int] | None:
        """``num_bands``, ``num_kpts``, ``num_projections`` off the ``.amn``.

        Wannier90 writes these three counts on the second line and reads them back
        the same way. ``None`` when there is no file or the line is not three
        integers: an unreadable header is not evidence of a disagreement.
        """
        path = self.run_dir / f"{seed}.amn"
        if not path.is_file():
            return None
        with path.open() as handle:
            handle.readline()  # the provenance comment
            counts = handle.readline().split()
        if len(counts) < 3:
            return None
        try:
            return int(counts[0]), int(counts[1]), int(counts[2])
        except ValueError:
            return None

    def check_deck_matches_its_projections(
        self, seed: str, *, fresh: bool = False
    ) -> None:
        """Refuse a construction whose deck and overlaps describe different sets.

        Wannier90's own report is ``<seed>.amn has not the right number of
        projections``, which names neither count nor where the file came from, and
        arrives as a failed calculation -- so the loop records an attempt that was
        never scored, spends it, and hands a model a message it cannot act on. Both
        counts are on disk before the localisation starts, so say which is which.

        Equality, not an inequality: across thirty-six constructions on ten materials
        in the 2026-08-29 campaign the ``.amn`` projection count equalled the deck's
        ``num_wann`` in every case Wannier90 accepted, including every construction
        built through projectability disentanglement, and differed in the single case
        it refused.
        """
        header = self.amn_header(seed)
        if header is None:
            return
        bands, _, projections = header
        win = self.run_dir / f"{seed}.win"
        if not win.is_file():
            return
        declared = read_win(win)
        expected_wann = _as_count(declared.get("num_wann"))
        expected_bands = _as_count(declared.get("num_bands"))
        error = DeckRefused if fresh else CalculationError
        if expected_wann is not None and projections != expected_wann:
            raise error(
                f"{seed}.amn holds {projections} projections but {seed}.win asks for "
                f"{expected_wann} Wannier functions; the deck and its overlaps "
                f"describe different projector sets, so the construction was not run"
            )
        if expected_bands is not None and bands != expected_bands:
            raise error(
                f"{seed}.amn was built over {bands} bands but {seed}.win asks for "
                f"{expected_bands}; the deck and its overlaps describe different "
                f"band sets, so the construction was not run"
            )

    # -- the steps -------------------------------------------------------

    def _run(
        self,
        step: str,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> StepResult:
        """Run one step, watching it, and stop it only if it stalls (:mod:`.progress`).

        Standard output and error go to files rather than pipes: an engine that writes
        more than a pipe holds would block on a pipe nobody reads while it is watched,
        and its progress is read from them.
        """
        environment = {**os.environ, **self.env}
        workdir = self.run_dir if cwd is None else Path(cwd)
        policy = self.stall_policy
        with contextlib.ExitStack() as stack:
            stdout = stack.enter_context(
                open(stdout_path, "w+b")
                if stdout_path is not None
                else tempfile.TemporaryFile(dir=workdir)
            )
            stderr = stack.enter_context(
                open(stderr_path, "w+b")
                if stderr_path is not None
                else tempfile.TemporaryFile(dir=workdir)
            )
            started = time.monotonic()
            monitor = None
            if policy is not None:
                reference, scale = self._reference_for(step)
                monitor = ProgressMonitor(
                    step,
                    self._progress_sources(step, workdir, stdout),
                    policy,
                    reference_s=reference,
                    scale=scale,
                )
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=workdir,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    # its own process group, so a stopped step takes mpirun's ranks
                    # with it
                    start_new_session=True,
                )
            except OSError as exc:
                raise CalculationError(f"{step} could not be launched: {exc}") from exc
            poll_s = policy.poll_s if policy is not None else 5.0
            deadline = None if self.timeout_s is None else started + self.timeout_s
            try:
                while True:
                    wait = poll_s
                    if deadline is not None:
                        wait = max(0.0, min(wait, deadline - time.monotonic()))
                    try:
                        returncode = process.wait(timeout=wait)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if monitor is not None:
                        monitor.observe()
                        stall = monitor.stalled()
                        if stall is not None:
                            _stop(process)
                            raise StepStalled(stall)
                    if deadline is not None and time.monotonic() >= deadline:
                        _stop(process)
                        raise CalculationError(
                            f"{step} exceeded {self.timeout_s:g} s and was stopped "
                            f"(a fixed wall-time limit set by the caller, not "
                            f"scale-aware)"
                        )
            finally:
                if process.poll() is None:
                    # an exception -- a stall, the job being stopped -- never leaves
                    # the engine running behind it
                    _stop(process)
            if monitor is not None:
                monitor.observe()
            return StepResult(
                step=step,
                returncode=returncode,
                seconds=time.monotonic() - started,
                stdout_tail=_tail(_read_tail(stdout)),
                stderr_tail=_tail(_read_tail(stderr)),
                progress=monitor.summary() if monitor is not None else None,
            )

    def run_overlaps(self, seed: str) -> list[StepResult]:
        """Produce overlaps through the selected engine."""
        raise NotImplementedError

    def run_localisation(self, seed: str) -> StepResult:
        """``wannier90`` — the cheap half, and the one most repairs touch."""
        result = self._run(f"{seed}:localise", [self.wannier90, seed])
        error_file = self.run_dir / f"{seed}.werr"
        if not result.ok and error_file.is_file():
            result.stderr_tail = _tail(error_file.read_text())
        elif not result.ok:
            wout = self.run_dir / f"{seed}.wout"
            if wout.is_file():
                # Wannier90 reports a refused deck in its output, not on stderr.
                lines = [ln for ln in wout.read_text().splitlines() if "ERROR" in ln]
                if lines:
                    result.stderr_tail = _tail("\n".join(lines))
        return result

    def build(
        self, seed: str, *, reuse_overlaps_from: str | None = None
    ) -> list[StepResult]:
        """Produce one construction, doing as little as the change requires."""
        results: list[StepResult] = []
        # the steps completed so far, readable by the caller when a later one raises
        self.last_steps = results
        # The deck is run as written. A convergence test is added to repaired decks by
        # the workflow when it writes them (REPAIR_DECK_PREPARATION), never here: this
        # step also runs the frontier deck, which must stay the published one.
        # Whether the overlaps may be reused is the caller's decision, never a
        # question about what happens to be on disk. Adopting files because their
        # names match let a second run of the same material inherit the first run's
        # overlaps for a deck it had since changed: the .amn held five atomic
        # projections while the deck named nine, and Wannier90 refused it as though
        # the projector set were wrong.
        if reuse_overlaps_from and reuse_overlaps_from != seed:
            # Not "copy if nothing is there": whatever is already there may be an
            # earlier run's. Runs of the same material reuse the same `_r1`, `_r2`
            # seednames, so the presence of a file says nothing about which deck it
            # belongs to -- on sodium the file left under this name held four
            # projections while the deck the repair had just written named one.
            self.discard_overlaps(seed)
            self.reuse_overlaps(reuse_overlaps_from, seed)
        elif not reuse_overlaps_from:
            self.discard_overlaps(seed)
            results.extend(self.run_overlaps(seed))
            if not results[-1].ok:
                return results
        self.check_deck_matches_its_projections(seed, fresh=not reuse_overlaps_from)
        results.append(self.run_localisation(seed))
        return results


@dataclass
class WannierRunner(ConstructionRunner):
    """Keep the QE overlap sequence while sharing its localisation half."""

    pw2wannier90: str = "pw2wannier90.x"
    #: pw2wannier90 diagonalises a matrix whose size is the projector count, which is
    #: routinely smaller than the rank count; serial linear algebra avoids an abort
    #: that has nothing to do with the physics.
    pw2wannier90_args: Sequence[str] = ("-ndiag", "1")

    def run_overlaps(self, seed: str) -> list[StepResult]:
        """``wannier90 -pp`` then ``pw2wannier90`` — the expensive half."""
        results = [self._run(f"{seed}:nnkp", [self.wannier90, "-pp", seed])]
        if not results[-1].ok:
            return results
        command = [
            *self.mpi_launcher,
            self.pw2wannier90,
            *self.pw2wannier90_args,
            "-in",
            f"{seed}.pw2wan",
        ]
        result = self._run(f"{seed}:overlaps", command)
        # pw2wannier90 writes its report to stdout; keep it beside the run.
        (self.run_dir / f"{seed}.pw2wan.out").write_text(result.stdout_tail)
        results.append(result)
        return results


def _ranges(bands: Sequence[int]) -> str:
    """``1, 2, 3, 4, 9`` as Wannier90 spells it, ``1-4, 9``."""
    spans: list[list[int]] = []
    for band in sorted(set(bands)):
        if spans and band == spans[-1][1] + 1:
            spans[-1][1] = band
        else:
            spans.append([band, band])
    return ", ".join(f"{lo}-{hi}" if hi > lo else f"{lo}" for lo, hi in spans)


def _as_count(value: object) -> int | None:
    """A count as the deck spells it: read back from a file these are strings."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _stop(process: subprocess.Popen, grace_s: float = 5.0) -> None:
    """Stop a step and everything it started: SIGTERM, then SIGKILL after a grace."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _read_tail(handle: IO[bytes], limit: int = 16000) -> str:
    """The end of a step's output file, read back without loading all of it."""
    try:
        handle.flush()
        size = os.fstat(handle.fileno()).st_size
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _tail(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else "…" + text[-limit:]
