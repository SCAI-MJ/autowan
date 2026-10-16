"""Record VASP's grammar and refusal signatures separately from the repair policy.

The overlap library accepts fewer Wannier90 keywords than the standalone program.
An explicit allowlist keeps localisation settings from making that setup abort.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ...actions import EDITABLE_PARAMETERS
from ...parent_adequacy import (
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
)
from ..protocol import DeckRule, EngineKnowledge, FailureSignature
from . import outputs

VASP_KNOWLEDGE = EngineKnowledge(
    engine="vasp",
    deck_grammar={
        p.name: DeckRule(
            "win",
            (
                "begin projections ... end projections"
                if p.name == "projectors"
                else (
                    "! overlap-step INCAR: CUTOFF_MU / CUTOFF_SIGMA (CUTOFF_TYPE = erfc)"
                    if p.name in ("scdm_mu", "scdm_sigma")
                    else f"{p.name} = <{p.kind}>"
                )
            ),
            note=(
                "the implied num_wann is also written to the overlap step's INCAR as "
                "NUM_WANN, which VASP reads instead of the .win"
                if p.name == "projectors"
                else (
                    "an overlap-step INCAR tag: VASP's SCDM weight, applied when the "
                    "overlaps are recomputed"
                    if p.name in ("scdm_mu", "scdm_sigma")
                    else ""
                )
            ),
        )
        for p in EDITABLE_PARAMETERS
        if p.name != "pao_subset"
    },
    unsupported_parameters=("pao_subset",),
    # VASP 6.4.1 runs SCDM: vasp.6.4.1-w90-avx2 with LSCDM = .TRUE. printed "SCDM mode"
    # and an erfc cutoff (CUTOFF_MU / CUTOFF_SIGMA defaulted from NUM_WANN) and wrote an
    # AMN that Wannier90 localised (2026-09-29). The cutoff is an
    # overlap-step INCAR tag, carried on the deck's ``! overlap-step INCAR:`` lines.
    unsupported_reasons={
        "pao_subset": (
            "PDWF is not available on VASP: its hydrogenic and LOCPROJ projections are "
            "not orthonormal, so projectability exceeds one and no PAO subset applies"
        ),
    },
    failure_signatures=tuple(
        FailureSignature(pattern, meaning, where)
        for pattern, meaning, where in (
            (
                "compiled without the VASP2WANNIER90 interface",
                "Use a VASP binary compiled with the VASP2WANNIER90 interface",
                "stdout",
            ),
            (
                r"The file wannier90\.win can not be found",
                "The overlap stage needs wannier90.win under VASP's fixed seedname",
                "stdout",
            ),
            (
                "aborting loop.*EDIFF was not reached",
                "The electronic loop did not reach EDIFF",
                "OUTCAR",
            ),
            (
                "I REFUSE TO CONTINUE",
                "VASP refused the job; the accompanying lines identify the cause",
                "stdout",
            ),
            ("ZBRENT: fatal error", "VASP's line search failed", "OUTCAR / stdout"),
            (
                "EDDDAV: Call to ZHEGV failed",
                "The electronic eigensolver failed",
                "OUTCAR / stdout",
            ),
            (
                "Sub-Space-Matrix is not hermitian",
                "The electronic subspace matrix lost Hermiticity",
                "OUTCAR / stdout",
            ),
            (
                "VERY BAD NEWS! internal error in subroutine",
                "VASP encountered an internal error",
                "OUTCAR / stdout",
            ),
            (
                "charge density could not be read from file CHGCAR",
                "The overlap step cannot read the parent's charge density",
                "OUTCAR / stdout",
            ),
            (
                "CALC_WANNIER_TRAFO_SCDM: Number of Wannier functions must not be larger",
                "SCDM needs at least as many bands as Wannier functions",
                "OUTCAR / stdout",
            ),
            (
                "NUM_WANN must be different from zero",
                "Declare a nonzero Wannier count in the overlap deck",
                "OUTCAR / stdout",
            ),
            (
                r"wannier interface is returning \d+ projections while NUM_WANN is",
                "INCAR's NUM_WANN disagrees with the deck's projection count; the overlap step must write NUM_WANN from num_wann",
                "stdout",
            ),
            (
                r"ORBITAL_DEFINITIONS: ERROR: L=\s*\d+\s*M=\s*\d+ not implemented",
                "VASP read a projection entry the library did not fill: NUM_WANN exceeds the deck's projections, or an orbital type VASP lacks",
                "stdout",
            ),
            (
                r"num_bands\s*<\s*num_wann",
                "The band manifold is smaller than the requested Wannier count",
                "wannier90.werr",
            ),
            (
                r"forrtl: severe \(174\): SIGSEGV",
                "The VASP process segfaulted; check the stack limit",
                "stderr",
            ),
            (
                r"forrtl: severe \(168\).*illegal instruction",
                "The binary was built for a different CPU (-xHOST); run it on the node class it was compiled on",
                "stderr",
            ),
            (
                r"forrtl: severe \(\d+\)",
                "The VASP process died in the Fortran runtime; the line above names the exception",
                "stderr",
            ),
            (
                "Error reading item",
                "An input item does not match the engine's grammar",
                "OUTCAR / stdout",
            ),
            (
                "kmesh_get|B1 condition",
                "The overlap library could not construct neighbours satisfying the B1 condition",
                "wannier90.werr",
            ),
        )
    ),
    overlap_step_keys=(
        "num_wann",
        "num_bands",
        "exclude_bands",
        "spinors",
        "gamma_only",
        "skip_B1_tests",
        "search_shells",
        "kmesh_tol",
        "mp_grid",
        "projections",
        "unit_cell_cart",
        "atoms_frac",
        "atoms_cart",
        "kpoints",
    ),
)


def _number(value):
    return float(str(value).strip().lower().replace("d", "e"))


def _block(text, name):
    match = re.search(rf"\bbegin\s+{name}\b(.*?)\bend\s+{name}\b", text, re.I | re.S)
    return match[1].strip() if match else ""


def _table_path(path):
    potcar = path.with_name("POTCAR")
    return potcar if potcar.is_file() else path.with_name("OUTCAR")


def _demanded_channels(projections):
    # VASP computes each hydrogenic projection's one-centre part from the
    # POTCAR's own projectors (CALC_OVERLAP_GN, LAUG), so a channel the POTCAR
    # lacks has no one-centre part.
    demanded = {}
    hybrids = {
        "sp": (0, 1),
        "sp2": (0, 1),
        "sp3": (0, 1),
        "sp3d": (0, 1, 2),
        "sp3d2": (0, 1, 2),
    }
    for line in projections.splitlines():
        fields = line.split(":")
        if len(fields) < 2 or not re.fullmatch(r"[A-Za-z]+", fields[0].strip()):
            continue
        symbol = fields[0].strip()
        angulars = demanded.setdefault(symbol, set())
        orbitals = [
            orbital
            for group in fields[1].split(";")
            for orbital in re.split(r",\s*mr\s*=", group, maxsplit=1, flags=re.I)[
                0
            ].split(",")
        ]
        for orbital in orbitals:
            orbital = orbital.strip().lower()
            explicit = re.fullmatch(r"l\s*=\s*([-+]?\d+)", orbital)
            if explicit:
                angular = int(explicit[1])
                if angular >= 0:
                    angulars.add(angular)
                elif angular in (-1, -2, -3):
                    angulars.update((0, 1))
                elif angular in (-4, -5):
                    angulars.update((0, 1, 2))
            elif orbital in hybrids:
                angulars.update(hybrids[orbital])
            elif orbital and orbital[0] in "spdf":
                angulars.add("spdf".index(orbital[0]))
    return {s: tuple(sorted(ls)) for s, ls in demanded.items()}


class VaspAdequacyPartition(AdequacyPartition):
    """Read VASP parent facts and express remedies as data for the caller."""

    engine = "vasp"

    def __init__(self, potcar_library=None):
        self.potcar_library = (
            Path(potcar_library) if potcar_library is not None else None
        )

    def read_facts(self, parent_source):
        path = Path(parent_source)
        parent = outputs.read_parent(path)
        root = ET.parse(path).getroot()
        polarised = parent.eigenvalues_ev.shape[0] == 2
        noncollinear = parent.spinor or any(
            (item.text or "").strip().upper() in ("T", "TRUE", ".TRUE.")
            for item in root.iter()
            if item.get("name") == "LNONCOLLINEAR"
        )
        moment = None
        if polarised:
            weights = outputs._vectors(root, "kpoints/varray[@name='weights']").ravel()
            if weights.sum() != 0:
                difference = parent.occupations[0].sum(axis=1) - parent.occupations[
                    1
                ].sum(axis=1)
                moment = float(np.dot(weights, difference) / weights.sum())
        try:
            channels = {
                s: tuple(Channel(letter, "spdf".index(letter)) for letter in letters)
                for s, letters in outputs.potcar_channels(_table_path(path)).items()
            }
        except (outputs.VaspOutputError, OSError):
            channels = {}
        # One function per (l, m) and, for vasp_ncl, per spinor component.
        count = (
            sum(
                c.spinor_functions() if noncollinear else c.functions
                for s in parent.species
                for c in channels[s]
            )
            if all(s in channels for s in parent.species)
            else None
        )
        moments = []
        for token in root.findtext("incar/v[@name='MAGMOM']", "").split():
            if "*" in token:
                repeat, value = token.split("*", 1)
                moments.extend([_number(value)] * int(repeat))
            else:
                moments.append(_number(token))
        requested = {}
        for symbol, value in zip(parent.species, moments):
            if value and symbol not in requested:
                requested[symbol] = value
        return ParentFacts(
            engine=self.engine,
            eigenvalues_ev=parent.eigenvalues_ev,
            fermi_energy_ev=parent.fermi_energy_ev,
            kpoints_frac=outputs._vectors(root, "kpoints/varray[@name='kpointlist']"),
            lattice_ang=parent.lattice_ang,
            species=parent.species,
            n_symmetries=parent.n_symmetries,
            spin_polarised=polarised,
            noncollinear=noncollinear,
            moment_bohr=moment,
            projector_functions=count,
            channels=channels,
            species_order=tuple(dict.fromkeys(parent.species)),
            requested_moments=requested,
            source=str(path),
        )

    def read_demand(self, inputs, facts):
        values = inputs.values
        win = "\n".join(
            re.split(r"[!#]", line, maxsplit=1)[0].strip()
            for line in inputs.win.read_text().splitlines()
        )
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
        kpoints = (
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
        projections = _block(win, "projections")
        return DeckDemand(
            num_wann=int(_number(values["num_wann"])),
            num_bands=(
                int(_number(values["num_bands"])) if "num_bands" in values else None
            ),
            excluded_bands=tuple(sorted(set(excluded))),
            mp_grid=mesh,
            kpoints_frac=kpoints,
            frozen_window_ev=tuple(
                _number(values[key]) if key in values else None
                for key in ("dis_froz_min", "dis_froz_max")
            ),
            projector_route="analytic" if projections else "unknown",
            spin_channel=None,
            demanded_channels=_demanded_channels(projections) if projections else None,
        )

    def _inventory_prescription(self, missing, facts):
        library = self.potcar_library or os.environ.get("AUTOWAN_POTCAR_LIBRARY")
        try:
            variants = (
                sorted(p for p in Path(library).iterdir() if p.is_dir())
                if library
                else []
            )
        except OSError:
            variants = []
        try:
            valences = outputs.potcar_valences(_table_path(Path(facts.source)))
        except (outputs.VaspOutputError, OSError, ValueError):
            valences = {}
        messages, changes = [], []
        for symbol, angulars in missing.items():
            zval = valences.get(symbol)
            required = {c.l for c in facts.channels.get(symbol, ())} | set(angulars)
            candidates = []
            for variant in variants:
                if zval is None or not (
                    variant.name == symbol or variant.name.startswith(symbol + "_")
                ):
                    continue
                try:
                    candidate_valence = outputs.potcar_valences(variant / "POTCAR").get(
                        symbol
                    )
                    if (
                        candidate_valence is None
                        or not abs(candidate_valence - zval) < 1e-6
                    ):
                        continue
                    channels = outputs.potcar_channels(variant / "POTCAR").get(
                        symbol, ()
                    )
                    if required <= {"spdf".index(letter) for letter in channels}:
                        candidates.append(variant.name)
                except (outputs.VaspOutputError, OSError, ValueError):
                    continue
            labels = ", ".join(
                f"{'spdfghiklm'[l] if l < 10 else 'channel'} (l = {l})"
                for l in angulars
            )
            projector = "projector" if len(angulars) == 1 else "projectors"
            pronoun = "it" if len(angulars) == 1 else "them"
            message = (
                f"The {symbol} POTCAR has no non-local {labels} {projector} "
                "that the deck's projections demand; use a variant with the same "
                f"ZVAL = {zval if zval is not None else 'unknown'} that carries {pronoun}. "
            )
            if candidates:
                message += f"Compatible POTCAR variants: {', '.join(candidates)}. Use {candidates[0]} in SCF and non-self-consistent runs. "
                changes.extend(
                    DftChange(step, "POTCAR", symbol, candidates[0])
                    for step in ("scf", "nscf")
                )
            else:
                message += (
                    "No compatible POTCAR variant was found in the supplied library. "
                )
            messages.append(
                message
                + "Alternatively, drop those channels from the deck's projections."
            )
        return " ".join(messages), changes

    def prescribe(self, finding, facts, demand):
        m, check = finding.measured, finding.check_id
        changes = []
        if check == BAND_COUNT:
            count = m["prescribed_bands"]
            changes.append(DftChange("nscf", "INCAR", "NBANDS", count))
            message = (
                f"Set NBANDS = {count} in the non-self-consistent run (INCAR). "
                f"The parent carries {m['parent_bands']} bands; the deck draws from num_bands = "
                f"{demand.num_bands if demand.num_bands is not None else demand.num_wann} "
                f"and excludes {len(demand.excluded_bands)} more. "
                "VASP rounds NBANDS up to a multiple of its band-parallel group, "
                f"so the run must use a rank layout that keeps {count} exactly. "
                "The computed spectrum must close the frozen window."
            )
        elif check == BAND_PARTITION:
            lo, hi = m["surplus_band_range"]
            message = (
                f"The parent carries {m['parent_bands']} bands; the deck fixes num_bands = "
                f"{m['num_bands']} and excludes {m['excluded']} more, so it partitions "
                f"{m['deck_bands']}. VASP's overlap step inherits NBANDS, so the overlaps span "
                f"{m['overlap_bands']} bands and the deck-overlap check refuses the "
                f"construction. On the deck side, exclude the surplus bands {lo}-{hi} as well "
                f"(exclude_bands = {lo}-{hi}, added to any exclusion already listed), or write "
                f"the deck for this parent (num_bands = {m['overlap_bands']}). "
            )
            if m["prescribed_bands"] is not None:
                count = m["prescribed_bands"]
                changes.append(DftChange("nscf", "INCAR", "NBANDS", count))
                message += (
                    f"On the parent side, set NBANDS = {count} in the non-self-consistent run "
                    "with a rank layout that keeps it exact."
                )
            else:
                message += (
                    "The bands the deck partitions do not close its frozen window at every "
                    "point, so only the deck-side remedy applies."
                )
        elif check in (KLIST, HOPPING):
            mesh = demand.mp_grid if check == KLIST else m["prescribed_mesh"]
            changes.append(DftChange("nscf", "INCAR", "ISYM", -1))
            if mesh is not None:
                changes.append(
                    DftChange("nscf", "KPOINTS", "Gamma", [int(n) for n in mesh])
                )
            mesh_text = (
                " ".join(map(str, mesh)) if mesh is not None else "matching the deck"
            )
            message = (
                f"Use ISYM = -1 with a Gamma-centred {mesh_text} mesh with zero shift "
                "in the non-self-consistent run, so VASP keeps the full grid in the order the deck lists it."
            )
            if check == HOPPING:
                message += " Match the deck's mp_grid to this mesh. This is a convergence advisory."
        elif check == SPIN:
            moments = [float(m["declared"].get(s, 0.0)) for s in facts.species]
            for step in ("scf", "nscf"):
                changes.extend(
                    (
                        DftChange(step, "INCAR", "ISPIN", 2),
                        DftChange(step, "INCAR", "MAGMOM", moments),
                    )
                )
            message = (
                f"Use ISPIN = 2 and MAGMOM = {' '.join(map(str, moments))} "
                "in POSCAR atom order in both SCF and non-self-consistent runs. "
                "The SCF density (CHGCAR) must be recomputed before the non-self-consistent run. "
                "The overlap step then writes one set of products per spin channel."
            )
            if "converged to a non-magnetic solution" in finding.reason:
                message = (
                    "The SCF converged to a non-magnetic solution; restart it from the declared moments. "
                    + message
                )
        elif check == INVENTORY:
            if m.get("missing_channels"):
                message, changes = self._inventory_prescription(
                    m["missing_channels"], facts
                )
            else:
                message = (
                    f"The pseudo-atomic route supplies {m['projector_functions']} functions. "
                    "Use a POTCAR with sufficient projector functions or change the deck's projections; "
                    "this is a projection limit, not a parent cause."
                )
        else:
            raise ValueError(f"Unknown parent adequacy check: {check}")
        return Prescription(check, message, tuple(changes))
