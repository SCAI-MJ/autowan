"""The read-only tools: deterministic answers, no attempt spent, a cap per attempt.

Each tool reads what is on disk and returns text the model reads; none names the
material. The loop logs every call with the attempt it served.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import repair_fixtures as fx

from autowan import recipes
from autowan.inputs import read_construction
from autowan.methods import DeckContext
from autowan.orchestrator import Run, RunConfig
from autowan.tools import RepairTools, load_rescan, reference_topics
from autowan.workflow import WorkflowConfig, wannierize

HERE = Path(__file__).parent
VASP = HERE / "fixtures" / "vasp"
W90 = HERE / "fixtures" / "wannier90"


def _tools(tmp_path, **kw):
    fx.write_deck(tmp_path)
    inputs = read_construction(tmp_path, fx.SEED)
    run = Run(fx.SEED)
    run.start_attempt({"x": 1}, seed=fx.SEED, method="pdwf")
    run._unresolved = ("calculation_did_not_run",)
    from autowan.backends import get_backend

    return RepairTools(
        ctx=DeckContext(
            code="qe",
            run_dir=tmp_path,
            parent=fx.parent(),
            parent_source=tmp_path / "parent.xml",
            frontier=inputs,
        ),
        backend=get_backend("qe"),
        run=run,
        inputs=inputs,
        seed=fx.SEED,
        species=("O", "Sr"),
        **kw,
    )


def test_frozen_capacity_counts_the_kept_states_against_num_wann(tmp_path):
    tools = _tools(tmp_path)
    fits = json.loads(tools("frozen_capacity", {"e_max": 2.0}))
    # kept bands: 4 occupied (-5..-1) and the empty ones from 1 eV; 2 eV holds one of them
    assert fits["frozen_states_max"] == 5 and fits["num_wann"] == 13 and fits["fits"]
    over = json.loads(tools("frozen_capacity", {"e_max": 40.0}))
    assert over["frozen_states_max"] == 16 and not over["fits"]
    # the 14th kept state (13 functions) sits at the 10th empty level
    assert over["highest_e_max_that_fits"] < 30.0


def test_band_character_falls_back_to_the_deck_projections_and_says_so(tmp_path):
    tools = _tools(tmp_path)
    fx.write_overlaps(tmp_path)
    tools.run.attempts[-1].outcome = SimpleNamespace(failed_check_ids=())
    answer = tools("band_character", {"e_min": -6.0, "e_max": 0.0})
    data = json.loads(answer)
    assert data["states"] == 8  # 4 occupied kept bands x 2 k-points
    assert "not an independent decomposition" in data["source"]
    assert set(data["composition"]) <= {"Sr s", "Sr p", "Sr d", "O s", "O p"}
    assert abs(sum(data["composition"].values()) - 1.0) < 0.02


def test_band_character_on_vasp_reads_the_parent_procar(tmp_path):
    from autowan.backends.vasp.outputs import read_parent

    parent_dir = tmp_path / "parent"
    shutil.copytree(VASP / "SrO", parent_dir)
    parent = read_parent(parent_dir / "vasprun.xml")
    run = Run("x")
    tools = RepairTools(
        ctx=DeckContext(
            "vasp", tmp_path, parent, parent_dir / "vasprun.xml", SimpleNamespace()
        ),
        backend=None,
        run=run,
        inputs=SimpleNamespace(),
        seed="x",
    )
    deep = json.loads(tools("band_character", {"e_min": -40.0, "e_max": -12.0}))
    assert deep["source"].startswith("PROCAR")
    top = max(deep["composition"], key=deep["composition"].get)
    assert top in ("Sr p", "O s", "Sr s")


def test_centres_and_spreads_name_the_nearest_atom_and_distance(tmp_path):
    tools = _tools(tmp_path)
    shutil.copy(
        W90 / "graphene_projectability_fixed.wout", tmp_path / f"{fx.SEED}.wout"
    )
    tools.run.attempts[-1].outcome = SimpleNamespace(failed_check_ids=())
    data = json.loads(tools("centres_and_spreads", {}))
    assert data["construction_attempt"] == 0
    row = data["functions"][0]
    assert set(row) == {
        "wf",
        "centre_ang",
        "spread_ang2",
        "nearest_atom",
        "distance_ang",
    }
    assert row["nearest_atom"].split("#")[0] in ("Sr", "O")


def test_reference_pages_are_short_paraphrases_with_their_sources():
    topics = reference_topics()
    assert {
        "pdwf",
        "scdm",
        "vasp_scdm",
        "windows",
        "minimisation",
        "projections",
        "exclude_bands",
    } <= set(topics)
    tools = RepairTools(ctx=None, backend=None, run=None, inputs=None, seed="x")
    for topic in topics:
        page = tools("reference_pages", {"topic": topic})
        assert "Sources:" in page and len(page) < 4000, topic
    assert "arXiv:2303.07877" in tools("reference_pages", {"topic": "pdwf"})
    assert "npj Comput. Mater. 6, 66" in tools("reference_pages", {"topic": "scdm"})
    assert "no page" in tools("reference_pages", {"topic": "nope"})


def _ladder(tmp_path):
    """A stored ladder carrying outcome columns the tool must never pass on."""
    grid = tmp_path / "ladder.csv"
    fields = [
        "material",
        "point",
        "dis_proj_min",
        "dis_proj_max",
        "status",
        "default_pass",
        "band_interpolation_pass",
        "eta2_meV",
        "sparse_tb_pass",
        "eta_sparse_tb_meV",
    ]
    points = [(0.02, 0.99), (0.02, 0.95), (0.01, 0.99), (0.01, 0.95), (0.005, 0.99)]
    with grid.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        # written out of order: the tool must follow the ladder's point order
        for i in (3, 1, 5, 2, 4):
            pmin, pmax = points[i - 1]
            writer.writerow(
                {
                    "material": fx.SEED,
                    "point": i,
                    "dis_proj_min": pmin,
                    "dis_proj_max": pmax,
                    "status": "finished",
                    "default_pass": i == 3,
                    "band_interpolation_pass": i == 5,
                    "eta2_meV": 99.5 - i,
                    "sparse_tb_pass": i == 5,
                    "eta_sparse_tb_meV": 77.25 + i,
                }
            )
        writer.writerow(
            {
                "material": "other",
                "point": 1,
                "dis_proj_min": 0.3,
                "dis_proj_max": 0.7,
                "status": "finished",
                "default_pass": True,
                "band_interpolation_pass": True,
                "eta2_meV": 1,
                "sparse_tb_pass": True,
                "eta_sparse_tb_meV": 1,
            }
        )
    return grid


def test_frontier_candidates_says_none_when_no_ladder_is_stored(tmp_path):
    tools = _tools(tmp_path)
    assert tools("frontier_candidates", {}).startswith("not available")
    assert "there is no tool named" in tools("frontier_rescan", {})


def test_frontier_candidates_returns_only_untested_thresholds_in_ladder_order(tmp_path):
    tools = _tools(tmp_path, frontier_rescan_source=_ladder(tmp_path))
    text = tools("frontier_candidates", {})
    data = json.loads(text)
    assert set(data) == {"note", "candidates"}
    assert "untested" in data["note"] and "PDWF" in data["note"]
    # the current deck (0.01 / 0.95, point 4) has run: it is not proposed again
    assert [c["ladder_point"] for c in data["candidates"]] == ["1", "2", "3", "5"]
    assert data["candidates"][0] == {
        "ladder_point": "1",
        "dis_proj_min": 0.02,
        "dis_proj_max": 0.99,
    }
    for c in data["candidates"]:
        assert set(c) == {"ladder_point", "dis_proj_min", "dis_proj_max"}
    # nothing the ladder file scored reaches the model
    for leak in (
        "pass",
        "eta",
        "status",
        "meV",
        "finished",
        "first",
        "True",
        "99.",
        "77.",
        "0.3",
        "0.7",
    ):
        assert leak not in text, leak


def test_frontier_candidates_drops_every_threshold_pair_this_run_has_tried(tmp_path):
    tools = _tools(tmp_path, frontier_rescan_source=_ladder(tmp_path))

    def ran(pmin, pmax):
        tools.run.attempts.append(
            SimpleNamespace(parameters={"dis_proj_min": pmin, "dis_proj_max": pmax})
        )

    ran(
        "2.0000000000d-02", "9.9000000000d-01"
    )  # Fortran exponents, as decks write them
    ran("0.005", "0.99")
    ran(None, None)  # a non-PDWF attempt carries no thresholds
    data = json.loads(tools("frontier_candidates", {}))
    assert [c["ladder_point"] for c in data["candidates"]] == ["2", "3"]
    ran("0.02", "0.95")
    ran("0.01", "0.99")
    assert tools("frontier_candidates", {}).startswith("not available")


def test_frontier_dataset_can_hold_back_the_runs_own_composition(tmp_path):
    from autowan.frontier_dataset import lookup

    kept = lookup(elements=["O", "Sr", "V"], formula="O3SrV", n_sites=5)
    rows = [r for g in kept["by_pseudo_family"].values() for r in g]
    assert any(r["same_composition"] for r in rows)
    held = lookup(
        elements=["O", "Sr", "V"], formula="O3SrV", n_sites=5, same_composition=False
    )
    rows = [r for g in held["by_pseudo_family"].values() for r in g]
    assert rows and not any(r["same_composition"] for r in rows)
    assert all(r["formula"] != "SrVO3" for r in rows)
    tools = _tools(tmp_path, frontier_dataset_same_composition=False)
    spec = next(s for s in tools.specs() if s.name == "frontier_dataset")
    assert "held back" in spec.description


def _rows(answer):
    return [r for g in answer["by_pseudo_family"].values() for r in g]


def _held_tools(tmp_path, species, same=False):
    """RepairTools on a parent of ``species`` (graphene: C, C)."""
    import dataclasses

    tools = _tools(tmp_path, frontier_dataset_same_composition=same)
    tools.ctx.parent = dataclasses.replace(
        tools.ctx.parent,
        species=tuple(species),
        positions_frac=np.zeros((len(species), 3)),
    )
    return tools


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # the default query: the run's own species and formula
        {"elements": ["C"]},  # elements only (the v8 leak)
        {"elements": ["C"], "formula": None},
        {"formula": "C"},
        {"formula": "C2"},  # a cell multiple of the same composition
        {"elements": ["C", "Si"], "formula": "C"},
        {"elements": ["C"], "pseudo_family": "SSSP 1.1 PBE", "limit": 12},
        {"elements": ["C"], "limit": 1},
    ],
)
def test_frontier_dataset_holds_back_the_parents_composition_for_every_query(
    tmp_path, arguments
):
    from autowan.frontier_dataset import load_table

    assert any(r["reduced_formula"] == "C" for r in load_table())  # rows exist to leak
    tools = _held_tools(tmp_path, ("C", "C"))
    answer = tools.tool_frontier_dataset(**arguments)
    rows = _rows(answer)
    assert all(r["formula"] != "C" for r in rows)
    assert "composition C " in answer["held_back"]
    # the serialised answer the model reads carries no record of graphene either
    text = tools("frontier_dataset", arguments)
    assert "18.4674" not in text and "0.8663" not in text
    assert all(r["formula"] != "C" for r in _rows(json.loads(text)))


def test_frontier_dataset_still_returns_other_compositions_when_holding_back(tmp_path):
    tools = _held_tools(tmp_path, ("C", "C"))
    rows = _rows(tools.tool_frontier_dataset(elements=["C"]))
    assert rows and {r["formula"] for r in rows} <= {
        "AgCN",
        "Be2C",
        "CdCO3",
        "CuCO3",
        "Mg2C3",
        "MgC2",
        "NaCN",
        "RhC2ClO2",
        "SrCO3",
        "Ta2C",
        "Ti4C",
        "TiC",
    }
    # another composition named by formula comes back in full, flagged as the query's
    other = _rows(tools.tool_frontier_dataset(formula="TiC"))
    assert other and other[0]["formula"] == "TiC" and other[0]["same_composition"]


def test_frontier_dataset_holds_back_a_compound_parent_under_any_formula(tmp_path):
    tools = _held_tools(tmp_path, ("Sr", "V", "O", "O", "O"))
    for arguments in (
        {},
        {"elements": ["V"]},
        {"elements": ["O", "Sr", "V"]},
        {"formula": "SrVO3"},
        {"formula": "O3SrV"},
        {"formula": "Sr2V2O6"},
        {"formula": "SrTiO3"},  # another formula sharing species
    ):
        rows = _rows(tools.tool_frontier_dataset(**arguments))
        assert rows and all(r["formula"] != "SrVO3" for r in rows), arguments


def test_frontier_dataset_with_same_composition_on_returns_the_parents_records(
    tmp_path,
):
    tools = _held_tools(tmp_path, ("C", "C"), same=True)
    answer = tools.tool_frontier_dataset(elements=["C"])
    assert "held_back" not in answer
    assert any(r["formula"] == "C" for r in _rows(answer))


def test_lookup_exclude_composition_overrides_the_query():
    from autowan.frontier_dataset import lookup

    for kw in ({"elements": ["W"]}, {"formula": "W"}, {"formula": "W", "limit": 1}):
        rows = _rows(lookup(exclude_composition="W", **kw))
        assert all(r["formula"] != "W" for r in rows)
        rows = _rows(lookup(exclude_composition={"W": 2}, **kw))
        assert all(r["formula"] != "W" for r in rows)


def test_load_rescan_keeps_one_constructions_rows(tmp_path):
    grid = _ladder(tmp_path)
    assert len(load_rescan(grid, "other")) == 1
    assert [r["point"] for r in load_rescan(grid, fx.SEED)] == ["1", "2", "3", "4", "5"]


def test_search_references_uses_the_shipped_corpus_and_leaves_out_obsolete_entries():
    shipped = recipes.load(recipes.DEFAULT_CORPUS, include_obsolete=True)
    obsolete = [r for r in shipped if r.regime == "obsolete"]
    current = recipes.load(recipes.DEFAULT_CORPUS)
    assert len(obsolete) == 12 and len(current) == 60
    assert all(r.engine == "qe" for r in shipped)
    assert all(r.regime == "current" for r in current)
    assert not {r.recipe_id for r in obsolete} & {r.recipe_id for r in current}
    assert WorkflowConfig().recipe_dir == recipes.DEFAULT_CORPUS


def test_a_distilled_entry_records_its_engine_and_method():
    log = {
        "material": "m",
        "backend": "vasp",
        "attempts": [
            {"index": 0, "outcome": {"failed_check_ids": ["a"], "passed": False}},
            {
                "index": 1,
                "method": "scdm",
                "outcome": {"failed_check_ids": [], "passed": True},
                "action": {
                    "changes": [{"parameter": "scdm_mu", "new": 1.0}],
                    "diagnosis": "d",
                },
            },
        ],
    }
    (entry,) = recipes.distil_run(log, {"species": {"Si": 2}})
    assert (
        entry.engine == "vasp" and entry.method == "scdm" and entry.regime == "current"
    )


def test_the_tool_call_cap_is_per_attempt_and_calls_spend_no_attempt(tmp_path):
    calls = [("frozen_capacity", {"e_max": 1.0})] * 5
    fx.write_deck(tmp_path)
    model = fx.ToolScript(
        [
            (calls, fx.answer(changes=[("num_iter", 5000)])),
            (calls, fx.answer("escalate")),
        ]
    )
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "x.xml",
        seed=fx.SEED,
        runner=fx.RefusingEngine(tmp_path),
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=4, tool_calls_per_attempt=3),
            draw_figures=False,
        ),
        backend=fx.StoredParentQe(fx.parent()),
    )
    attempts = result.run.attempts
    assert len(attempts) == 2
    assert model.requests[0]["cap"] == 3
    counted = [c["counted"] for c in attempts[1].tools_called]
    assert counted == [True, True, True, False, False]
    assert attempts[1].model_requests == 6 and attempts[1].cost_usd == pytest.approx(
        0.001
    )


# ---------------------------------------------------------------------------
# frontier_dataset: the published PDWF runs, keyed on pseudopotential family
# ---------------------------------------------------------------------------


def test_the_frontier_table_ships_with_its_provenance_and_families():
    from autowan.frontier_dataset import DEFAULT_TABLE, families, load_table

    assert len(load_table(str(DEFAULT_TABLE))) == 2172
    note = (DEFAULT_TABLE.parent / "FRONTIER_DATASET.md").read_text()
    for doi in ("materialscloud:x0-yf", "materialscloud:ck-br", "materialscloud:9g-ds"):
        assert doi in note
    assert "CC-BY-4.0" in note
    assert {
        "SSSP 1.1 PBE",
        "SSSP PBEsol",
        "PseudoDojo FR (SOC)",
        "pslibrary mix (SOC)",
    } <= set(families())


def test_a_lookup_ranks_the_same_composition_first_and_labels_every_family():
    from autowan.frontier_dataset import MAX_ROWS, lookup

    found = lookup(formula="W", n_sites=1)
    assert found["rows_returned"] == MAX_ROWS < found["rows_matching"]
    for family, rows in found["by_pseudo_family"].items():
        assert all(r["pseudo_family"] == family for r in rows)
        assert rows[0]["formula"] == "W" and rows[0]["same_composition"]
        flags = [r["same_composition"] for r in rows]
        assert flags == sorted(flags, reverse=True), "a same-composition row ranked low"
    first = found["by_pseudo_family"]["SSSP 1.1 PBE"][0]
    assert first["num_wann"] == "9" and first["exclude_bands"] == "1-4"
    assert first["dis_proj_max_used"] and first["eta2_mev"] and first["mp_grid"]


def test_a_family_filter_never_returns_another_family_and_pbesol_is_its_own():
    from autowan.frontier_dataset import lookup

    pbesol = lookup(elements=["Na"], pseudo_family="SSSP PBEsol")
    assert set(pbesol["by_pseudo_family"]) == {"SSSP PBEsol"}
    rows = pbesol["by_pseudo_family"]["SSSP PBEsol"]
    assert all(r["dataset"] == "jiang2025_ck-br_40largest" for r in rows)
    assert "error" in lookup(elements=["Na"], pseudo_family="PBE whatever")
    capped = lookup(elements=["O"], limit=50)
    assert capped["cap"] == 12 and capped["rows_returned"] <= 12


def test_the_frontier_dataset_tool_defaults_to_this_runs_composition(tmp_path):
    tools = _tools(tmp_path)
    data = json.loads(tools("frontier_dataset", {}))
    assert data["query"]["elements"] == ["O", "Sr"]
    assert data["rows_returned"] >= 1
    for family, rows in data["by_pseudo_family"].items():
        assert all(
            {"Sr", "O"} & set(__import__("re").findall(r"[A-Z][a-z]?", r["formula"]))
            for r in rows
        )
