"""The published PDWF frontier, as a lookup: how it Wannierised similar materials.

The table (``data/frontier_wannier_lookup.csv.gz``, provenance and licence in
``data/FRONTIER_DATASET.md``) holds one row per published Wannierisation workflow of
Qiao, Pizzi & Marzari (2023) and Jiang et al. (2025): formula, space group, method arm,
spin treatment, pseudopotential family, the deck actually used (num_wann,
exclude_bands, the projectability thresholds after any rescan, dis_froz_max, k-mesh),
eta2 and exit status.

Rows are ranked by element overlap with the query, an exact composition match first,
then by how alike the cells are. A same-material record is allowed by default (autowan
item 4 decision 14); ``exclude_composition`` holds back every record of one
composition -- the run's own -- whatever the query asks for (user rule 2026-10-13: the
only frontier information an agent may receive is what its deck proposes next). Rows
are grouped by pseudopotential family -- SSSP 1.1 PBE, SSSP PBEsol, PseudoDojo FR,
pslibrary -- and every row names its family, so families are never mixed without
saying so. The number of rows returned is capped.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TABLE = Path(__file__).parent / "data" / "frontier_wannier_lookup.csv.gz"
#: Most rows one lookup returns, over every family.
MAX_ROWS = 12

#: The family each dataset's index label means, normalised.
FAMILY_LABELS = {
    "SSSP 1.1 PBE efficiency": "SSSP 1.1 PBE",
    "SSSP/PBE": "SSSP 1.1 PBE",
    "SSSP/PBEsol": "SSSP PBEsol",
    "pseudoDojo": "PseudoDojo FR (SOC)",
    "ps_mix": "pslibrary mix (SOC)",
}
#: Used only where a row's index carries no label, and said so on the row.
DATASET_FAMILY = {
    "qiao2023_x0-yf": "SSSP 1.1 PBE",
    "jiang2025_ck-br_200": "SSSP 1.1 PBE",
    "jiang2025_ck-br_magnetic": "SSSP 1.1 PBE",
    "jiang2025_ck-br_40largest": "SSSP PBEsol",
    "jiang2025_9g-ds_v1_soc": "unlabelled SOC set (PseudoDojo FR or pslibrary)",
}

RETURNED = (
    "formula",
    "spacegroup",
    "n_sites",
    "dataset",
    "method",
    "spin_type",
    "pseudo_family",
    "num_wann",
    "num_bands",
    "exclude_bands",
    "atom_proj_exclude",
    "dis_proj_min_used",
    "dis_proj_max_used",
    "n_w90_runs",
    "dis_froz_max_input_rel",
    "dis_froz_max_used_abs",
    "fermi_energy_used",
    "mp_grid",
    "eta2_mev",
    "exit_status",
    "success_eta2_le_20mev",
)


def parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        counts[symbol] = counts.get(symbol, 0) + (int(n) if n else 1)
    return counts


def _reduced(counts: Mapping[str, int]) -> dict[str, int]:
    from .recipes import composition_of

    return composition_of(counts)


@lru_cache(maxsize=4)
def load_table(path: str | Path = DEFAULT_TABLE) -> tuple[dict, ...]:
    path = Path(path)
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        label = row.get("pseudo_family_from_index") or ""
        if label in FAMILY_LABELS:
            row["pseudo_family"] = FAMILY_LABELS[label]
        elif label:
            row["pseudo_family"] = label
        else:
            row["pseudo_family"] = DATASET_FAMILY.get(row["dataset"], "unlabelled")
            row["pseudo_family_note"] = "no index label; family from the dataset"
        row["_elements"] = frozenset(parse_formula(row["reduced_formula"]))
        row["_composition"] = _reduced(parse_formula(row["reduced_formula"]))
        rows.append(row)
    return tuple(rows)


def families(path: str | Path = DEFAULT_TABLE) -> tuple[str, ...]:
    return tuple(sorted({r["pseudo_family"] for r in load_table(str(path))}))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lookup(
    *,
    elements=(),
    formula: str | None = None,
    pseudo_family: str | None = None,
    n_sites: int | None = None,
    magnetic: bool | None = None,
    limit: int = MAX_ROWS,
    path: str | Path = DEFAULT_TABLE,
    spinor: bool | None = None,
    same_composition: bool = True,
    exclude_composition: Mapping[str, int] | str | None = None,
) -> dict[str, Any]:
    """The nearest published frontier constructions, grouped by pseudopotential family.

    ``exclude_composition`` (a formula or species counts; the run's own material) holds
    back every record whose reduced composition equals it, for every query form --
    elements only, any formula, any pseudo_family, any limit -- and the answer says so.
    ``same_composition=False`` without ``exclude_composition`` holds back the queried
    formula's composition.
    """
    limit = max(1, min(int(limit or MAX_ROWS), MAX_ROWS))
    wanted = set(elements or ())
    composition = _reduced(parse_formula(formula)) if formula else {}
    wanted |= set(composition)
    if not wanted:
        return {"error": "give elements or a formula"}
    if isinstance(exclude_composition, str):
        held_back = _reduced(parse_formula(exclude_composition))
    elif exclude_composition:
        held_back = _reduced(exclude_composition)
    elif not same_composition:
        held_back = composition
    else:
        held_back = {}
    rows = load_table(str(path))
    if held_back:
        # Before anything else reads the table: no held-back row is ranked, counted or
        # returned, whatever the query.
        rows = [r for r in rows if r["_composition"] != held_back]
    if pseudo_family:
        known = {r["pseudo_family"] for r in rows}
        if pseudo_family not in known:
            return {
                "error": f"unknown pseudo_family {pseudo_family!r}; known: {sorted(known)}"
            }
        rows = [r for r in rows if r["pseudo_family"] == pseudo_family]
    scored = []
    for row in rows:
        shared = wanted & row["_elements"]
        if not shared:
            continue
        overlap = len(shared) / len(wanted | row["_elements"])
        same = bool(composition) and row["_composition"] == composition
        score = overlap + (2.0 if same else 0.0)
        if n_sites and _float(row.get("n_sites")):
            score += 0.25 * max(
                0.0, 1.0 - abs(float(row["n_sites"]) - n_sites) / max(n_sites, 1)
            )
        if magnetic is not None:
            row_magnetic = row.get("spin_type") in ("Col", "NonCol")
            score += 0.25 if row_magnetic == magnetic else 0.0
        if spinor is not None:
            # A spinor (SOC / noncollinear) deck counts 2j+1 functions per orbital and
            # one electron per band; a scalar row's num_wann is not comparable to it.
            row_spinor = row.get("spin_type") in ("SOC", "NonCol")
            score += 0.5 if row_spinor == spinor else 0.0
        if _float(row.get("eta2_mev")) is not None:
            score += 0.1  # an Optimize workflow with a measured eta2
        scored.append((score, same, row))
    scored.sort(key=lambda item: -item[0])
    picked = scored[:limit]
    grouped: dict[str, list] = {}
    for score, same, row in picked:
        entry = {k: row.get(k) for k in RETURNED}
        entry["formula"] = row["reduced_formula"]
        entry["same_composition"] = same
        if row.get("pseudo_family_note"):
            entry["pseudo_family_note"] = row["pseudo_family_note"]
        grouped.setdefault(row["pseudo_family"], []).append(entry)
    answer = {
        "query": {
            "elements": sorted(wanted),
            "formula": formula,
            "pseudo_family": pseudo_family,
        },
        "rows_returned": len(picked),
        "rows_matching": len(scored),
        "cap": limit,
        "by_pseudo_family": grouped,
    }
    if held_back:
        name = "".join(
            f"{e}{n if n != 1 else ''}" for e, n in sorted(held_back.items())
        )
        answer["held_back"] = (
            f"Every record of composition {name} (this run's own material) is held "
            "back from every query, whatever elements, formula, pseudo_family or limit "
            "is asked for; none is ranked, counted or returned."
        )
    answer["read_this"] = (
        "Published PDWF frontier runs (Qiao 2023; Jiang 2025), CC-BY-4.0. Each "
        "group is one pseudopotential family; values from different families are "
        "not interchangeable. Phase and magnetic state are matched only through "
        "spacegroup and spin_type: check them. dis_proj_*_used are the thresholds "
        "after the workflow's own rescan; dis_froz_max_input_rel is relative to E_F "
        "(insulators: the CBM), dis_froz_max_used_abs absolute. eta2 in meV; "
        "success means exit 0 and eta2 <= 20 meV. A same-composition row is the "
        "frontier's own record of the queried composition, for reference."
    )
    return answer
