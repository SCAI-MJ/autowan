"""Expose the established QE path without duplicating its readers or deck writer.

Keeping these as delegates means an engine switch cannot quietly fork the behaviour
of the existing QE implementation. Failure explanations are data beside that seam.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import numpy as np

from .. import inputs, qe_outputs
from ..actions import EDITABLE_PARAMETERS
from ..execution import WannierRunner
from ..interpolation import cartesian_to_fractional_k
from ..parent_adequacy import (
    BAND_COUNT,
    BAND_PARTITION,
    HOPPING,
    INVENTORY,
    KLIST,
    SPIN,
    AdequacyPartition,
    Channel,
    DeckDemand,
    DftChange,
    ParentFacts,
    Prescription,
    full_grid,
)
from .protocol import Backend, BackendError, DeckRule, EngineKnowledge, FailureSignature

QE_KNOWLEDGE = EngineKnowledge(
    engine="qe",
    deck_grammar={
        p.name: DeckRule(
            p.stage_file,
            (
                "begin projections ... end projections"
                if p.name == "projectors"
                else f"{p.name} = <{p.kind}>"
            ),
        )
        for p in EDITABLE_PARAMETERS
    },
    failure_signatures=tuple(
        FailureSignature(pattern, meaning, where)
        for pattern, meaning, where in (
            (
                "blk2cyc_zredist",
                "pw2wannier90 needs -ndiag 1 for a small projector matrix",
                "stdout",
            ),
            (
                "Wrong number of k-points",
                "The parent and the deck must use the same k-point mesh",
                "stdout",
            ),
            (
                "S matrix not positive definite",
                "The overlap matrix is not positive definite",
                "stdout",
            ),
            (
                r"mismatch in .*\.eig",
                "The band count and excluded bands disagree with the eigenvalue file",
                "wannier90.werr",
            ),
            (
                "has not the right number of projections",
                "The deck and overlaps describe different projector sets",
                "wannier90.werr",
            ),
            (
                "Energy window contains fewer states",
                "The energy window cannot supply the requested Wannier count",
                "wannier90.werr",
            ),
            (
                "not converged",
                "The iteration stopped before its convergence criterion was met",
                "stdout",
            ),
            ("SIGSEGV", "The executable aborted with a segmentation fault", "stderr"),
        )
    ),
)


def _number(value):
    return float(str(value).strip().lower().replace("d", "e"))


def _channels(path):
    """Read atomic wavefunction channels from either UPF generation.

    A fully relativistic UPF lists one PP_CHI per (l, j) and records j in its
    ``PP_SPIN_ORB/PP_RELWFC.<i>`` entries (``jchi``); the channel then supplies 2j+1
    spinor states, which is what pw.x counts as atomic wavefunctions.
    """
    try:
        text = path.read_text()
        tags = re.findall(r"<PP_CHI\.\d+\b[^>]*>", text, re.I)
        if tags:
            jchi = {}
            for rel in re.findall(r"<PP_RELWFC\.(\d+)\b([^>]*)>", text, re.I):
                attrs = dict(re.findall(r"""(\w+)\s*=\s*["']([^"']*)["']""", rel[1]))
                if "jchi" in attrs:
                    jchi[int(rel[0])] = float(attrs["jchi"])
            result = []
            for number, tag in enumerate(tags, start=1):
                attrs = dict(re.findall(r"""(\w+)\s*=\s*["']([^"']*)["']""", tag))
                result.append(
                    Channel(attrs["label"], int(attrs["l"]), jchi.get(number))
                )
            return tuple(result)
        header = re.search(r"<PP_HEADER>(.*?)</PP_HEADER>", text, re.S | re.I)
        if header is not None:
            table = re.split(r"Wavefunctions\s+nl\s+l\s+occ", header[1], flags=re.I)
            if len(table) == 2:
                return tuple(
                    Channel(label, int(l))
                    for label, l in re.findall(
                        r"^\s*(\d+[A-Za-z]+)\s+(\d+)\s+[-+\d.]", table[1], re.M
                    )
                )
    except (OSError, UnicodeError, ValueError, KeyError):
        pass
    return ()


def _deck_text(path):
    if path is None:
        return ""
    return "\n".join(
        line.split("!")[0].split("#")[0].strip()
        for line in path.read_text().splitlines()
    )


def _block(text, name):
    match = re.search(rf"\bbegin\s+{name}\b(.*?)\bend\s+{name}\b", text, re.I | re.S)
    return match[1].strip() if match else ""


class QeAdequacyPartition(AdequacyPartition):
    engine = "qe"

    def read_facts(self, parent_source):
        path = Path(parent_source)
        parent = qe_outputs.read_parent(path)
        root = ET.parse(path).getroot()
        bs = root.find(".//band_structure")
        polarised = "true" in bs.findtext("lsda", "false").lower()
        noncollinear = "true" in bs.findtext("noncolin", "false").lower()
        moment = None
        if polarised:
            weights = np.array(
                [
                    _number(k.get("weight", "1"))
                    for k in bs.findall("ks_energies/k_point")
                ]
            )
            if weights.sum() != 0:
                difference = parent.occupations[0].sum(axis=1) - parent.occupations[
                    1
                ].sum(axis=1)
                moment = float(np.dot(weights, difference) / weights.sum())
        species = root.find("output/atomic_species")
        if species is None:
            species = root.find("input/atomic_species")
        entries = [] if species is None else species.findall("species")
        order = tuple(s.get("name") for s in entries)
        pseudo_dir = species.get("pseudo_dir", "") if species is not None else ""
        pseudo_dir = pseudo_dir or root.findtext(
            "input/control_variables/pseudo_dir", ""
        )
        directory = Path(pseudo_dir.strip())
        if not directory.is_absolute():
            directory = path.parent / directory
        channels = {}
        for entry in entries:
            filename = entry.findtext("pseudo_file", "").strip()
            if filename:
                parsed = _channels(directory / filename)
                if parsed:
                    channels[entry.get("name")] = parsed
        requested = {}
        for entry in root.findall("input/atomic_species/species"):
            value = _number(entry.findtext("starting_magnetization", "0"))
            if value:
                requested[entry.get("name")] = value
        count = parent.n_atomic_wfc  # pw.x counts spinor states when noncolin
        if (
            count is None
            and parent.species
            and all(s in channels for s in parent.species)
        ):
            count = sum(
                c.spinor_functions() if noncollinear else c.functions
                for s in parent.species
                for c in channels[s]
            )
        return ParentFacts(
            engine=self.engine,
            eigenvalues_ev=parent.eigenvalues_ev,
            fermi_energy_ev=parent.fermi_energy_ev,
            kpoints_frac=cartesian_to_fractional_k(
                parent.kpoints_cart, parent.lattice_ang, parent.alat_ang
            ),
            lattice_ang=parent.lattice_ang,
            species=parent.species,
            n_symmetries=parent.n_symmetries,
            spin_polarised=polarised,
            noncollinear=noncollinear,
            moment_bohr=moment,
            projector_functions=count,
            channels=channels,
            species_order=order,
            requested_moments=requested,
            source=str(path),
        )

    def read_demand(self, inputs, facts):
        values = inputs.values
        win, pw = _deck_text(inputs.win), _deck_text(inputs.pw2wan)
        excluded = []
        match = re.search(r"^\s*exclude_bands\s*[=:]\s*(.*)$", win, re.I | re.M)
        if match:
            for item in match[1].split(","):
                if not item.strip():
                    continue
                bounds = [int(v.strip()) for v in item.split("-")]
                excluded.extend(
                    range(bounds[0], bounds[1] + 1) if len(bounds) == 2 else bounds
                )
        mesh = values.get("mp_grid")
        if mesh is not None:
            mesh = tuple(
                int(v)
                for v in (
                    mesh.replace(",", " ").split() if isinstance(mesh, str) else mesh
                )
            )
        points = _block(win, "kpoints")
        k = (
            np.array(
                [
                    [_number(v) for v in line.split()[:3]]
                    for line in points.splitlines()
                    if line.strip()
                ]
            )
            if points
            else None
        )

        def enabled(text, key):
            return bool(
                re.search(rf"\b{key}\s*=\s*(?:\.true\.|true\b|t\b)", text, re.I)
            )

        if _block(win, "projections"):
            route = "analytic"
        elif enabled(pw, "scdm_proj"):
            route = "scdm"
        elif enabled(pw, "atom_proj") or enabled(win, "auto_projections"):
            route = "pseudo_atomic"
        else:
            route = "unknown"
        spin = re.search(r"""\bspin_component\s*=\s*["'](up|down)["']""", pw, re.I)
        return DeckDemand(
            num_wann=int(_number(values["num_wann"])),
            num_bands=(
                int(_number(values["num_bands"])) if "num_bands" in values else None
            ),
            excluded_bands=tuple(sorted(set(excluded))),
            mp_grid=mesh,
            kpoints_frac=k,
            frozen_window_ev=tuple(
                _number(values[key]) if key in values else None
                for key in ("dis_froz_min", "dis_froz_max")
            ),
            projector_route=route,
            spin_channel={"up": 0, "down": 1}[spin[1].lower()] if spin else None,
        )

    def prescribe(self, finding, facts, demand):
        m, check = finding.measured, finding.check_id
        changes = []
        if check == BAND_COUNT:
            count = m["prescribed_bands"]
            changes.append(DftChange("nscf", "system", "nbnd", count))
            message = (
                f"Set nbnd = {count} in the nscf calculation. The parent carries {m['parent_bands']} bands; "
                f"the deck draws from {demand.num_bands if demand.num_bands is not None else demand.num_wann}, "
                f"and exclude_bands excludes {len(demand.excluded_bands)} additional bands. "
                "The computed spectrum must also close the frozen window."
            )
        elif check == BAND_PARTITION:
            lo, hi = m["surplus_band_range"]
            message = (
                f"The parent carries {m['parent_bands']} bands; the deck fixes num_bands = "
                f"{m['num_bands']} and excludes {m['excluded']} more, so it partitions "
                f"{m['deck_bands']}. pw2wannier90 writes nbnd minus the excluded bands, so the "
                f"overlaps span {m['overlap_bands']} bands and the deck-overlap check refuses "
                f"the construction. On the deck side, exclude the surplus bands {lo}-{hi} as well "
                f"(exclude_bands = {lo}-{hi}, added to any exclusion already listed), or write "
                f"the deck for this parent (num_bands = {m['overlap_bands']}). "
            )
            if m["prescribed_bands"] is not None:
                count = m["prescribed_bands"]
                changes.append(DftChange("nscf", "system", "nbnd", count))
                message += (
                    f"On the parent side, set nbnd = {count} in the nscf calculation."
                )
            else:
                message += (
                    "The bands the deck partitions do not close its frozen window at every "
                    "point, so only the deck-side remedy applies."
                )
        elif check in (KLIST, HOPPING):
            if check == KLIST:
                points = (
                    demand.kpoints_frac
                    if demand.kpoints_frac is not None
                    else full_grid(demand.mp_grid)
                )
                changes.append(DftChange("nscf", "system", "nosym", True))
                if facts.noncollinear:
                    changes.append(DftChange("nscf", "system", "noinv", True))
                message = "Use nosym = .true. in nscf with the deck's full explicit crystal list in its order, not K_POINTS automatic."
            else:
                mesh = m["prescribed_mesh"]
                points = full_grid(mesh)
                message = (
                    f"Check mesh convergence at {tuple(mesh)}: nscf with nosym = .true. and the full "
                    "explicit crystal list, with the deck mp_grid matching. This is a convergence advisory."
                )
            weighted = [[float(x) for x in p] + [1.0 / len(points)] for p in points]
            changes.append(DftChange("nscf", "K_POINTS", "crystal", weighted))
        elif check == SPIN:
            declared = m["declared"]
            for step in ("scf", "nscf"):
                changes.append(DftChange(step, "system", "nspin", 2))
                for i, symbol in enumerate(facts.species_order, 1):
                    moment = declared.get(symbol, 0)
                    if symbol in facts.species and moment:
                        changes.append(
                            DftChange(
                                step,
                                "system",
                                f"starting_magnetization({i})",
                                0.5 if moment > 0 else -0.5,
                            )
                        )
            message = (
                "Use nspin = 2 in both scf and nscf with starting_magnetization set by the declared moments. "
                "Build one construction per spin channel with spin_component = 'up'/'down'."
            )
            if "converged to a non-magnetic solution" in finding.reason:
                message = (
                    "The SCF converged to a non-magnetic solution; restart from the declared moments. "
                    + message
                )
        elif check == INVENTORY:
            needed = max(m["states_at_fermi_level"], demand.num_wann)
            channels = "; ".join(
                f"{s}: {' '.join(c.label for c in cs)}"
                for s, cs in facts.channels.items()
            )
            message = (
                f"The pseudo-atomic route needs at least {needed} functions; the parent's pseudopotentials "
                f"supply {m['projector_functions']} ({channels or 'channels unavailable'}). "
                "Re-run with a pseudopotential whose PP_CHI set includes the missing valence channel, "
                "or use a deck-side projection route that does not depend on the pseudopotential's orbitals."
            )
        else:
            raise ValueError(f"Unknown parent adequacy check: {check}")
        return Prescription(check, message, tuple(changes))


class QeBackend(Backend):
    name = "qe"
    knowledge = QE_KNOWLEDGE
    adequacy = QeAdequacyPartition()

    def read_parent(self, source):
        return qe_outputs.read_parent(source)

    def read_construction(self, run_dir, seed):
        return inputs.read_construction(Path(run_dir), seed)

    def write_derived(
        self,
        source,
        run_dir,
        new_seed,
        changes,
        fermi_energy_ev=None,
        species_counts=None,
    ):
        return inputs.write_derived(
            source,
            Path(run_dir),
            new_seed,
            changes,
            fermi_energy_ev=fermi_energy_ev,
            species_counts=species_counts,
        )

    def deck_files(self, inputs):
        return (inputs.win,) + ((inputs.pw2wan,) if inputs.pw2wan is not None else ())

    def make_runner(self, run_dir, **launch):
        known = {f.name for f in fields(WannierRunner)} - {"run_dir"}
        for option in launch:
            if option not in known:
                raise BackendError(f"QE does not know launch option {option!r}")
        return WannierRunner(run_dir=run_dir, **launch)

    def refusal_reason(self, run_dir, seed):
        from ..workflow import _refusal_reason

        return _refusal_reason(Path(run_dir) / f"{seed}.wout")
