"""Shared synthetic material for the method-family, tool-loop and resume tests.

A rocksalt-like two-atom cell with Quantum ESPRESSO's PDWF frontier deck as
aiida-wannier90-workflows writes it: an external projector list (Sr: 4s 5s 4p 5p 4d,
O: 2s 2p), the Sr semicore PAOs excluded (atom_proj_exclude 1, 3, 4, 5), the four
semicore bands excluded, num_wann = 13. The parent is gapped: four semicore bands, four
occupied valence bands, twelve empty ones, E_F = 0, CBM = 1 eV. Nothing here is a real
calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autowan.backends.qe import QeBackend
from autowan.execution import StepResult, WannierRunner
from autowan.providers import ModelResponse, ProviderCreditsExhausted
from autowan.qe_outputs import ParentCalculation

SEED = "SrO"
N_K = 2

WIN = """\
num_wann = 13
num_bands = 16
exclude_bands = 1-4
auto_projections = .true.
dis_froz_max = 3.0
dis_froz_proj = .true.
dis_proj_min = 0.01
dis_proj_max = 0.95
conv_window = 3
num_iter = 4000
dis_num_iter = 4000
write_hr = .true.
mp_grid = 1 1 2

begin unit_cell_cart
ang
5.16 0 0
0 5.16 0
0 0 5.16
end unit_cell_cart

begin atoms_cart
ang
Sr 0.0 0.0 0.0
O 2.58 2.58 2.58
end atoms_cart

begin projections
end projections

begin kpoints
0.0 0.0 0.0
0.0 0.0 0.5
end kpoints
"""

PW2WAN = """\
&inputpp
  outdir = './out/'
  prefix = 'aiida'
  seedname = 'SrO'
  atom_proj = .true.
  atom_proj_ext = .true.
  atom_proj_dir = 'external_projectors/'
  atom_proj_exclude(1) = 1
  atom_proj_exclude(2) = 3
  atom_proj_exclude(3) = 4
  atom_proj_exclude(4) = 5
/
"""

LABELS = {
    "Sr": [("4s", 0), ("5s", 0), ("4p", 1), ("5p", 1), ("4d", 2)],
    "O": [("2s", 0), ("2p", 1)],
}


def write_projectors(run_dir: Path) -> None:
    directory = run_dir / "external_projectors"
    directory.mkdir(parents=True, exist_ok=True)
    table = {}
    for species, entries in LABELS.items():
        ls = " ".join(str(l) for _, l in entries)
        (directory / f"{species}.dat").write_text(f"10 {len(entries)}\n{ls}\n")
        table[species] = [
            {"label": label, "l": l, "alpha": "UPF"} for label, l in entries
        ]
    (directory / "projectors.json").write_text(json.dumps(table))


def write_deck(run_dir: Path, seed: str = SEED) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_projectors(run_dir)
    (run_dir / f"{seed}.win").write_text(WIN)
    (run_dir / f"{seed}.pw2wan").write_text(PW2WAN.replace("'SrO'", f"'{seed}'"))
    return run_dir


def parent() -> ParentCalculation:
    semicore = np.linspace(-20.0, -15.0, 4)
    valence = np.linspace(-5.0, -1.0, 4)
    empty = np.linspace(1.0, 30.0, 12)
    bands = np.concatenate([semicore, valence, empty])
    eigenvalues = np.tile(bands, (1, N_K, 1)).astype(float)
    return ParentCalculation(
        eigenvalues_ev=eigenvalues,
        occupations=(eigenvalues <= 0.0).astype(float),
        kpoints_cart=np.zeros((N_K, 3)),
        fermi_energy_ev=0.0,
        lattice_ang=np.eye(3) * 5.16,
        alat_ang=5.16,
        species=("Sr", "O"),
        positions_frac=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        n_symmetries=48,
        n_atomic_wfc=None,
        converged=None,
        calculation="nscf",
    )


def write_overlaps(run_dir: Path, seed: str = SEED, nw: int = 13) -> None:
    """``.amn``/``.eig`` over the 16 kept bands with an erfc-shaped projectability."""
    from scipy.special import erfc

    energies = parent().eigenvalues_ev[0, :, 4:]  # (k, 16)
    lines_amn = [" synthetic", f"{16:12d}{N_K:12d}{nw:12d}"]
    lines_eig = []
    for k in range(N_K):
        for m in range(16):
            e = energies[k, m]
            p = 0.5 * erfc((e - 8.0) / 2.0)
            amp = np.sqrt(p / nw)
            for n in range(nw):
                lines_amn.append(f"{m + 1:5d}{n + 1:5d}{k + 1:5d} {amp:.10f} 0.0")
            lines_eig.append(f"{m + 1:5d}{k + 1:5d} {e:.8f}")
    (run_dir / f"{seed}.amn").write_text("\n".join(lines_amn) + "\n")
    (run_dir / f"{seed}.eig").write_text("\n".join(lines_eig) + "\n")


class StoredParentQe(QeBackend):
    """The QE backend with the parent read from memory instead of an XML file."""

    def __init__(self, stored):
        self._parent = stored

    def read_parent(self, source):
        return self._parent

    def assess_parent(self, *a, **k):
        return None


class RefusingEngine(WannierRunner):
    """Records every build and fails the localisation, so no gate is ever run."""

    def __init__(self, run_dir):
        super().__init__(run_dir=run_dir, wannier90="wannier90.x")
        self.builds: list[tuple[str, str | None]] = []

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        return [StepResult(f"{seed}:localise", 1, 0.0, stderr_tail="offline: not run")]


class ToolScript:
    """A provider that calls tools and then answers, as scripted per request.

    Each script entry is ``(tool_calls, answer)``: ``tool_calls`` a list of
    ``(name, arguments)`` run through the workflow's own tool runner (honouring the
    cap), ``answer`` the final ``propose_repair`` arguments -- or an exception to raise.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []
        self.results: list[tuple[str, str]] = []

    def __call__(self, model):
        return self

    def propose(
        self,
        *,
        system,
        payload_text,
        schema,
        images=(),
        tools=(),
        run_tool=None,
        tool_call_cap=0,
        temperature=None,
    ):
        self.requests.append(
            {
                "system": system,
                "payload_text": payload_text,
                "schema": schema,
                "tools": [t.name for t in tools],
                "cap": tool_call_cap,
            }
        )
        if not self.script:
            raise ProviderCreditsExhausted("scripted model has no further answers")
        calls, answer = self.script.pop(0)
        records = []
        used = 0
        for name, arguments in calls:
            if used >= tool_call_cap:
                result = "the tool budget for this attempt is spent"
                counted = False
            else:
                used += 1
                result = run_tool(name, arguments)
                counted = True
            self.results.append((name, result))
            records.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": result[:600],
                    "counted": counted,
                }
            )
        if isinstance(answer, Exception):
            answer.tool_calls = records
            answer.requests = len(calls)
            raise answer
        return ModelResponse(
            content=answer,
            model="scripted",
            raw_text=json.dumps(answer),
            tool_calls=records,
            requests=1 + len(calls),
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            cost_usd=0.001,
        )


def answer(kind="modify_parameters", changes=(), **extra):
    from autowan.workflow import CALCULATION_FAILED

    out = {
        "action": kind,
        "diagnosis": "the construction did not run as written",
        "evidence": [CALCULATION_FAILED],
        "targets": [CALCULATION_FAILED],
        "expected_effect": "a construction that runs",
        "confidence": 0.5,
        "changes": [
            {"parameter": p, "new": v, "reason": "scripted"} for p, v in changes
        ],
    }
    out.update(extra)
    return out
