"""Stage VASP's fixed seedname without changing the construction's identity.

VASP reads wannier90.win and may exit zero after refusing it. Each stage starts
clean, and success requires fresh overlaps as well as a successful process status.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ...execution import CalculationError, ConstructionRunner, StepResult
from ...inputs import assignments, read_win
from ...progress import ActivitySource, LineSource, vasp_line, wannier90_line
from .inputs import (
    FRONTIER_WIN_KEYS,
    is_frontier_overlap_step,
    is_jarvis_overlap_step,
    overlap_step_incar_tags,
    overlap_step_win,
    read_incar,
    write_incar,
)
from .knowledge import VASP_KNOWLEDGE


def spin_channel_for(parent_dir: Path, spin_index: int) -> str | None:
    """Use one channel for the gate and overlap step: 0 = up = wannier90.1.*,
    1 = dn = wannier90.2.*. An unpolarised parent's channel 0 is unnumbered;
    an explicit channel 1 request is passed on so the runner can refuse it.
    """
    if spin_index not in (0, 1):
        raise CalculationError("VASP spin_index must be 0 or 1")
    incar = Path(parent_dir) / "INCAR"
    tags = read_incar(incar.read_text()) if incar.is_file() else {}
    if spin_index == 1:
        return "dn"
    return "up" if tags.get("ISPIN", "1") == "2" else None


def refusal_reason(run_dir: Path, seed: str) -> str | None:
    """Prefer the engine's own refusal, then its library, then localisation."""
    from ...workflow import _refusal_reason

    directory = Path(run_dir)
    stage = directory / f"{seed}.vasp2wannier90"
    reasons = []
    for path in (stage / "OUTCAR", stage / "vasp.log", stage / "vasp.err"):
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                meaning = VASP_KNOWLEDGE.explain(line)
                if meaning:
                    reasons.append(f"{line.strip()} ({meaning})")
    library = _refusal_reason(stage / "wannier90.wout")
    if library:
        meaning = VASP_KNOWLEDGE.explain(library)
        reasons.append(f"{library} ({meaning})" if meaning else library)
    if reasons:
        return " | ".join(reasons)
    return _refusal_reason(directory / f"{seed}.wout")


#: Parent INCAR tags the SCDM frontier's overlap step drops: ICHARG (the frontier reads
#: the WAVECAR, ``ALGO = None``), NUM_WANN (VASP chooses it), LWANNIER90 (it runs
#: ``LWANNIER90_RUN``), and NSW/IBRION, which the protocol leaves at VASP's defaults.
FRONTIER_PARENT_TAGS_DROPPED = ("ICHARG", "NUM_WANN", "LWANNIER90", "NSW", "IBRION")


def launcher_ranks(launcher) -> int | None:
    """The rank count an ``mpirun -np N`` / ``-n N`` launcher asks for, else ``None``."""
    args = list(launcher or ())
    for flag in ("-np", "-n", "--np"):
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                try:
                    return int(args[index + 1])
                except ValueError:
                    return None
    return None


def _has_content(path: Path) -> bool:
    """A WAVECAR VASP declined to write is left as an empty file, not no file."""
    return path.is_file() and path.stat().st_size > 0


def parent_is_noncollinear(tags) -> bool:
    """A vasp_ncl parent: LNONCOLLINEAR, or LSORBIT (which implies it)."""
    return any(
        str(tags.get(key, "")).strip().upper().lstrip(".").startswith("T")
        for key in ("LNONCOLLINEAR", "LSORBIT")
    )


@dataclass
class VaspWannierRunner(ConstructionRunner):
    vasp: str = "vasp_std"
    parent_dir: Path | None = None
    spin_channel: str | None = None
    vasp_ncl: str | None = None
    """The noncollinear binary for a spinor parent. Unset: ``vasp`` itself when its name
    says ncl, else ``vasp_ncl`` beside it. A vasp_std or vasp_gam binary cannot read a
    spinor WAVECAR, so the overlap step refuses rather than launch one."""

    def binary_for(self, tags) -> str:
        """The executable the overlap step runs for a parent with these INCAR tags."""
        if not parent_is_noncollinear(tags):
            return self.vasp
        if self.vasp_ncl:
            return self.vasp_ncl
        name = Path(self.vasp).name
        if "ncl" in name:
            return self.vasp
        sibling = Path(self.vasp).with_name("vasp_ncl")
        if Path(self.vasp).parent != Path(".") and sibling.is_file():
            return str(sibling)
        if Path(self.vasp).parent == Path(".") and shutil.which("vasp_ncl"):
            return "vasp_ncl"
        raise CalculationError(
            f"the parent is noncollinear (LNONCOLLINEAR/LSORBIT) and its spinor "
            f"wavefunctions need vasp_ncl; {self.vasp} is not one and no vasp_ncl was "
            "given (launch option vasp_ncl)"
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.parent_dir = (
            Path(self.parent_dir) if self.parent_dir is not None else self.run_dir
        )
        if self.spin_channel not in (None, "up", "dn"):
            raise CalculationError("VASP spin_channel must be 'up' or 'dn'")

    def stage_dir(self, seed: str) -> Path:
        return self.run_dir / f"{seed}.vasp2wannier90"

    def _progress_sources(self, step, cwd, stdout):
        """VASP's electronic steps, then the Wannier90 library's loops when it runs one
        (``LWANNIER90_RUN``, the SCDM frontier); the products' growth in between."""
        if step.rpartition(":")[2] != "overlaps":
            return super()._progress_sources(step, cwd, stdout)
        stage = Path(cwd)
        return [
            LineSource(stdout, vasp_line),
            LineSource(stage / "wannier90*.wout", wannier90_line),
            ActivitySource(
                [stage / f"wannier90*.{ext}" for ext in ("amn", "mmn", "eig")]
                + [stage / "OUTCAR", stage / "OSZICAR"]
            ),
        ]

    def run_overlaps(self, seed: str) -> list[StepResult]:
        required = ("CHGCAR", "POSCAR", "POTCAR", "KPOINTS", "INCAR")
        for name in required:
            path = self.parent_dir / name
            if not path.is_file():
                raise CalculationError(f"VASP overlap step needs {path}")
        win = self.run_dir / f"{seed}.win"
        if not win.is_file():
            raise CalculationError(f"VASP overlap step needs {win}")
        stage = self.stage_dir(seed)
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        # A previous successful call must not supply products missing from this one.
        self.discard_overlaps(seed)
        # The parent's files are read, never written, by the overlap step, so the
        # heavy ones (CHGCAR, WAVECAR) are linked rather than copied: a production
        # density is gigabytes, and every repair that reaches the projections stages
        # it again. Linking fails only across filesystems, and then a copy is fine.
        for name in ("POSCAR", "POTCAR", "KPOINTS", "CHGCAR", "WAVECAR"):
            source = self.parent_dir / name
            if source.is_file():
                try:
                    (stage / name).symlink_to(source.resolve())
                except OSError:
                    shutil.copyfile(source, stage / name)
        tags = read_incar((self.parent_dir / "INCAR").read_text())
        binary = self.binary_for(tags)
        deck_text = win.read_text()
        deck_tags = overlap_step_incar_tags(deck_text)
        if is_frontier_overlap_step(deck_text):
            # The SCDM frontier: read the parent's WAVECAR
            # with no diagonalisation, let VASP choose NUM_WANN, and hand the .win body
            # over as WANNIER90_WIN so VASP writes its own wannier90.win around it.
            # VASP's Wannier interface (the PEAD routines) refuses NCORE != 1, so
            # NPAR must equal the rank count; the deck rounded NBANDS to its NPAR, so a
            # different rank count is a different deck, not a launch detail to adjust.
            ranks = launcher_ranks(self.mpi_launcher)
            npar = deck_tags.get("NPAR")
            if ranks is not None and npar is not None and int(npar) != ranks:
                raise CalculationError(
                    f"the SCDM frontier deck sets NPAR = {npar} (its NBANDS is rounded "
                    f"to that), but the overlap step runs on {ranks} MPI ranks; VASP's "
                    f"Wannier interface needs NCORE = 1, i.e. NPAR equal to the rank "
                    f"count. Build the deck with npar={ranks}"
                )
            if not _has_content(stage / "WAVECAR"):
                raise CalculationError(
                    f"the SCDM frontier reads the parent WAVECAR, and "
                    f"{self.parent_dir / 'WAVECAR'} is missing or empty"
                )
            # The protocol's INCAR is the parent's overlaid with the deck's tags and
            # nothing else: no NSW/IBRION is written (VASP's defaults, 0 and -1, are
            # what the protocol runs), and the parent's tags that would contradict it
            # are dropped rather than overridden.
            for key in FRONTIER_PARENT_TAGS_DROPPED:
                tags.pop(key, None)
            tags.update(deck_tags)
            declared = assignments(deck_text)
            body = [
                f"{key} = {declared[key]}"
                for key in FRONTIER_WIN_KEYS
                if key in declared
            ]
            tags["WANNIER90_WIN"] = '"' + "\n".join(body) + '"'
            (stage / "INCAR").write_text(write_incar(tags))
        else:
            tags.update(
                {
                    "ICHARG": "11",
                    "LWANNIER90": ".TRUE.",
                    "NSW": "0",
                    "IBRION": "-1",
                    "LWAVE": ".FALSE.",
                    "LCHARG": ".FALSE.",
                    "ISTART": "1" if _has_content(stage / "WAVECAR") else "0",
                }
            )
            # The deck's own overlap-step tags (LSCDM, LOCPROJ_AUTO, CUTOFF_*): a
            # construction carries its overlap recipe, and a repair of mu or sigma
            # reaches VASP only through them.
            tags.update(deck_tags)
            if is_jarvis_overlap_step(deck_text) and not _has_content(
                stage / "WAVECAR"
            ):
                # The JARVIS frontier (ALGO = None) reads the parent's orbitals as
                # they are; without them VASP would start from random ones.
                raise CalculationError(
                    f"the JARVIS frontier reads the parent WAVECAR, and "
                    f"{self.parent_dir / 'WAVECAR'} is missing or empty"
                )
            # VASP does not take the Wannier count from the .win. Without NUM_WANN in
            # the INCAR it estimates one from the POTCARs' projector tables (mlwf.F,
            # LPRJ_FROM_PSEUDO) and reads that many entries out of the library's
            # projection arrays: on SrO the deck named 8 functions, the estimate was
            # 13, and entries 9-13 were zeros (2026-09-12). The deck is the source.
            declared = read_win(win).get("num_wann")
            if declared is None:
                raise CalculationError(
                    f"{win} declares no num_wann; VASP needs it as NUM_WANN"
                )
            tags["NUM_WANN"] = str(int(float(declared)))
            (stage / "INCAR").write_text(write_incar(tags))
            (stage / "wannier90.win").write_text(overlap_step_win(deck_text))
        result = self._run(
            f"{seed}:overlaps",
            [*self.mpi_launcher, binary],
            cwd=stage,
            stdout_path=stage / "vasp.log",
            stderr_path=stage / "vasp.err",
        )
        prefix = "wannier90"
        if self.spin_channel:
            # VASP 6 mlwf.F uses SEED_NAME//'.'//str(ISP); VASP 5 used up/dn.
            prefix += ".1" if self.spin_channel == "up" else ".2"
            legacy_prefix = f"wannier90.{self.spin_channel}"
            if not all(
                (stage / f"{prefix}.{ext}").is_file() for ext in ("amn", "mmn", "eig")
            ) and all(
                (stage / f"{legacy_prefix}.{ext}").is_file()
                for ext in ("amn", "mmn", "eig")
            ):
                prefix = legacy_prefix
        missing = []
        for ext in ("amn", "mmn", "eig"):
            source = stage / f"{prefix}.{ext}"
            if source.is_file():
                shutil.copyfile(source, self.run_dir / f"{seed}.{ext}")
            else:
                missing.append(source.name)
        if missing:
            if result.ok:
                result.returncode = 1
            numbered = [
                f"wannier90.{number}.{ext}"
                for number in (1, 2)
                for ext in ("amn", "mmn", "eig")
                if (stage / f"wannier90.{number}.{ext}").is_file()
            ]
            diagnosis = None
            if self.spin_channel is None and numbered:
                diagnosis = (
                    "the overlap step ran spin-polarised (ISPIN = 2) and wrote "
                    f"{', '.join(numbered)}; name the channel with "
                    "spin_channel = 'up' or 'dn'"
                )
            elif (
                self.spin_channel
                and not numbered
                and not any(
                    (stage / f"wannier90.{channel}.{ext}").is_file()
                    for channel in ("up", "dn")
                    for ext in ("amn", "mmn", "eig")
                )
                and all(
                    (stage / f"wannier90.{ext}").is_file()
                    for ext in ("amn", "mmn", "eig")
                )
            ):
                diagnosis = (
                    "VASP ran with ISPIN = 1 and wrote the unnumbered products "
                    "wannier90.amn, wannier90.mmn, wannier90.eig; "
                    f"there is no spin channel '{self.spin_channel}' to read"
                )
            result.stderr_tail = " | ".join(
                filter(
                    None,
                    (
                        result.stderr_tail,
                        f"missing VASP overlap products: {', '.join(missing)}",
                        diagnosis,
                        refusal_reason(self.run_dir, seed),
                    ),
                )
            )
        elif not result.ok:
            reason = refusal_reason(self.run_dir, seed)
            if reason:
                result.stderr_tail = " | ".join(
                    filter(None, (result.stderr_tail, reason))
                )
        return [result]
