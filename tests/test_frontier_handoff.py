"""Frontier handoff: autowan takes over from the frontier's last ladder state.

User decision 2026-10-13 (autowan item 4, the KPS 2026 poster): the frontier gets a
budget of ten Wannier90 runs -- its default deck (run 0) and its own retry ladder points
1..9 -- each judged under the run's criterion. When none certifies, autowan starts from
ladder point 9: the deck in the run directory is point 9's, with point 9's Wannier90
outputs beside it (scored as attempt 0, never rebuilt), and the model is shown the
frontier's ten runs with their outcomes (the frontier's own executed work). Its
``frontier_candidates`` tool then offers only the frontier's next proposal when it
stopped -- ladder point 10, untested -- and nothing after it (coordinator correction,
same day: the frontier would choose point 11 only after seeing point 10's result). The
frontier's runs are not autowan's attempts: autowan gets eight of its own on top.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
from test_repair_band_count_refusal import (
    SEED,
    SRO_PW2WAN,
    SRO_WIN,
    _change,
    _OfflineEngine,
    _ScriptedModel,
    _sro_parent,
    _StoredParentQe,
)

import autowan.workflow as workflow
from autowan.gate import GateReport, Verdict
from autowan.gate_profiles import get_profile
from autowan.orchestrator import Run, RunConfig
from autowan.workflow import FrontierHandoff, WorkflowConfig, wannierize

#: the 2025 archive ladder's first twelve points, in package order
LADDER = [
    (0.02, 0.99), (0.02, 0.95), (0.02, 0.90), (0.02, 0.85),
    (0.01, 0.99), (0.01, 0.95), (0.01, 0.90), (0.01, 0.85),
    (0.005, 0.99), (0.005, 0.95), (0.005, 0.90), (0.005, 0.85),
]  # fmt: skip
#: run 0's thresholds (the frontier's default deck); not on points 1..9 here
RUN0 = (0.0025, 0.95)
HIDDEN = "hidden-marker-never-shown"


class _Model(_ScriptedModel):
    """The scripted provider, keeping the system prompt it was sent."""

    def __init__(self, answers):
        super().__init__(answers)
        self.systems: list[str] = []
        self.tool_names: list[list[str]] = []

    def propose(self, *, system, tools=None, **kw):
        self.systems.append(system)
        self.tool_names.append([t.name for t in tools or ()])
        return super().propose(system=system, **kw)


def _handoff_deck(run_dir):
    text = SRO_WIN.replace(
        "dis_proj_max =   9.5000000000d-01", "dis_proj_max = 0.99"
    ).replace("dis_proj_min =   1.0000000000d-02", "dis_proj_min = 0.005")
    (run_dir / f"{SEED}.win").write_text(text)
    (run_dir / f"{SEED}.pw2wan").write_text(SRO_PW2WAN)
    # point 9's overlaps, as the driver links them beside the deck
    for ext in ("amn", "mmn", "eig"):
        (run_dir / f"{SEED}.{ext}").write_text("frontier point 9\n")
    (run_dir / f"{SEED}.amn").write_text(
        f" frontier point 9 overlaps\n{48:12d}{8:12d}{13:12d}\n"
    )
    return run_dir


def _history(run0=RUN0):
    runs = []
    for run in range(10):
        pmin, pmax = run0 if run == 0 else LADDER[run - 1]
        deck = {"dis_proj_min": pmin, "dis_proj_max": pmax}
        if run == 0:
            deck.update(method="pdwf", num_wann=13, num_bands=48)
        runs.append(
            {
                "run": run,
                "ladder_point": None if run == 0 else run,
                "deck": deck,
                "certified": False,
                "failing_checks": ["band_interpolation_eta2"],
                "measurements": {
                    "band_interpolation_eta2": {
                        "value": 100.0 + run,
                        "threshold": 20.0,
                        "units": "meV",
                    }
                },
                "calculation_error": None,
                "deck_values": {"dis_proj_min": str(pmin), "dis_proj_max": str(pmax)},
                "source": HIDDEN,
            }
        )
    return tuple(runs)


def _ladder_file(tmp_path):
    """The stored ladder with outcome columns (and later points) that must not leak."""
    path = tmp_path / "ladder.csv"
    fields = ["material", "point", "dis_proj_min", "dis_proj_max", "status",
              "band_interpolation_pass", "eta2_meV"]  # fmt: skip
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (pmin, pmax) in enumerate(LADDER, start=1):
            writer.writerow(
                {"material": SEED, "point": i, "dis_proj_min": pmin,
                 "dis_proj_max": pmax, "status": "finished",
                 "band_interpolation_pass": i == 11, "eta2_meV": 77.25 + i}  # fmt: skip
            )
    return path


def _handoff(history=None):
    return FrontierHandoff(
        point=9,
        history=history or _history(),
        next_proposal={"ladder_point": 10, "dis_proj_min": 0.005, "dis_proj_max": 0.95},
    )


def _run(tmp_path, answers, *, refused=True, handoff=None, total=9, monkeypatch=None):
    """A handoff run on the offline engine. ``refused``: point 9's Wannier90 stopped
    (a ``.werr``, as V sparse_tb's point 9); otherwise point 9 is scored (a stub
    construction, gate refusing the criterion)."""
    _handoff_deck(tmp_path)
    if refused:
        (tmp_path / f"{SEED}.werr").write_text(
            " Wannier90: Execution started\n Exiting.......\n dis_windows_proj: Energy "
            "window contains fewer states than number of target WFs\n (rank: 0)\n"
        )
    else:
        (tmp_path / f"{SEED}.wout").write_text("stub\n")
        monkeypatch.setattr(
            workflow,
            "read_run",
            lambda path: SimpleNamespace(
                spreads_ang2=None, disentanglement=None, wannierisation=None
            ),
        )
        monkeypatch.setattr(
            workflow, "projectability_not_applied", lambda text, wannier: None
        )

        def score(self, seed, inputs, wannier):
            return GateReport(
                seed,
                "gapped",
                (
                    Verdict(
                        "band_interpolation_eta2",
                        evaluable=True,
                        passed=False,
                        metric_value=108.0,
                        metric_units="meV",
                        threshold_value=20.0,
                        applies=True,
                        note="",
                    ),
                ),
            )

        monkeypatch.setattr(workflow._Scorer, "score", score)
    runner = _OfflineEngine(tmp_path)
    model = _Model(answers)
    result = wannierize(
        run_dir=tmp_path,
        parent_xml=tmp_path / "unused.xml",
        seed=SEED,
        runner=runner,
        provider_factory=model,
        config=WorkflowConfig(
            run_config=RunConfig(total_attempts=total),
            draw_figures=False,
            frontier_handoff=handoff or _handoff(),
            frontier_rescan=_ladder_file(tmp_path),
            rescan_key=SEED,
        ),
        backend=_StoredParentQe(_sro_parent()),
        log_path=tmp_path / "log.json",
    )
    return result, runner, model


def _decreasing(n):
    return [_change("num_iter", 4000, 3900 - 100 * i) for i in range(n)]


# ---------------------------------------------------------------------------
# attempt 0 is the frontier's point 9, read from disk and never rebuilt
# ---------------------------------------------------------------------------


def test_attempt_zero_is_the_frontiers_point_nine_and_is_not_rebuilt(tmp_path):
    result, runner, _ = _run(tmp_path, _decreasing(1))
    first = result.run.attempts[0]
    assert first.frontier_point == 9 and first.index == 0
    assert "ladder point 9" in first.calculation_error
    assert "fewer states than number of target WFs" in first.calculation_error
    assert first.steps == []
    # nothing was built under the frontier's seed: no overlaps, no localisation
    assert SEED not in runner.overlaps_built and SEED not in runner.localised
    assert runner.localised == [f"{SEED}_r1"]
    # the next deck reuses point 9's overlaps (only num_iter changed)
    assert result.run.attempts[1].overlaps_from == SEED


def test_a_scored_point_nine_is_attempt_zero_with_the_gates_outcome(
    tmp_path, monkeypatch
):
    result, _, model = _run(
        tmp_path, _decreasing(1), refused=False, monkeypatch=monkeypatch
    )
    first = result.run.attempts[0]
    assert first.frontier_point == 9 and first.outcome is not None
    assert not first.outcome.passed
    assert first.outcome.failed_check_ids == ("band_interpolation_eta2",)
    assert "band_interpolation_eta2: FAILED  = 108 meV" in model.payloads[0]


# ---------------------------------------------------------------------------
# what the model sees
# ---------------------------------------------------------------------------


def test_the_model_sees_the_frontiers_ten_runs_as_the_frontiers(tmp_path):
    _, _, model = _run(tmp_path, _decreasing(1))
    text = model.payloads[0]
    assert "THE FRONTIER'S OWN RUNS BEFORE YOU TOOK OVER (10 Wannier90 runs" in text
    assert "None of them certified" in text
    assert "ladder point 9" in text and "spend none of your attempts" in text
    system = model.systems[0]
    assert "This run took over from the frontier method" in system
    assert "frontier_next_proposal" in system and "frontier_candidates" not in system
    assert "frontier_next_proposal" in model.tool_names[0]
    assert "frontier_candidates" not in model.tool_names[0]
    section = text.split("THE FRONTIER'S OWN RUNS")[1].split("ALREADY ATTEMPTED")[0]
    rows = [json.loads(ln) for ln in section.splitlines() if ln.strip().startswith("{")]
    assert [r["run"] for r in rows] == list(range(10))
    assert rows[0]["ladder_point"] is None and rows[9]["ladder_point"] == 9
    assert rows[3]["measurements"]["band_interpolation_eta2"]["value"] == 103.0
    for row in rows:
        assert set(row) <= set(workflow.HANDOFF_MODEL_FIELDS)
    # the record-only fields never reach the model
    assert (
        HIDDEN not in text and "deck_values" not in text and "parameters" not in section
    )
    # attempt 0 in its own history is labelled as the frontier's, not the model's
    attempted = text.split("ALREADY ATTEMPTED IN THIS RUN")[1]
    assert "not yours: the frontier's own run at its ladder point 9" in attempted


def test_the_handoff_is_kept_on_the_record_with_the_parameter_sets(tmp_path):
    result, _, _ = _run(tmp_path, _decreasing(1))
    record = json.loads((tmp_path / "log.json").read_text())
    handoff = record["frontier_handoff"]
    assert handoff["point"] == 9 and len(handoff["history"]) == 10
    run3 = handoff["history"][3]["parameters"]
    assert run3["dis_proj_min"] == "0.02" and run3["dis_proj_max"] == "0.9"
    assert run3["num_wann"] == result.run.attempts[0].parameters["num_wann"]


# ---------------------------------------------------------------------------
# frontier_next_proposal: the frontier's retry rule on the latest run deck
# ---------------------------------------------------------------------------

BASE = {"method": "pdwf", "num_wann": "13", "num_bands": "48", "num_iter": "4000"}


def _tools(tmp_path, history=None, attempts=()):
    """RepairTools after a handoff at point 9; ``attempts``: (parameter changes, ran)."""
    from autowan.tools import RepairTools

    run = Run(SEED, RunConfig(total_attempts=9))
    point9 = {**BASE, "dis_proj_min": "0.005", "dis_proj_max": "0.99"}
    run.start_attempt(point9, seed=SEED, method="pdwf")
    run.attempts[0].frontier_point = 9
    handoff = _handoff(history)
    run.start_from_frontier(handoff.to_record(point9))
    run.advance_after_failure()
    for i, (changes, ran) in enumerate(attempts, start=1):
        a = run.start_attempt({**point9, **changes}, seed=f"{SEED}_r{i}", method="pdwf")
        if ran:
            a.steps = [{"step": f"{SEED}_r{i}:localise", "returncode": 1}]
        else:
            a.refused = True
    return RepairTools(
        ctx=SimpleNamespace(code="qe"),
        backend=None,
        run=run,
        inputs=None,
        seed=SEED,
        frontier_rescan_source=_ladder_file(tmp_path),
        rescan_key=SEED,
        frontier_handoff=handoff,
    )


def _proposal(tools):
    return json.loads(tools("frontier_next_proposal", {}))


def test_at_the_handoff_the_proposal_is_ladder_point_ten(tmp_path):
    tools = _tools(tmp_path)
    data = _proposal(tools)
    p = data["proposal"]
    assert p["ladder_point"] == 10
    assert p["changes"] == {
        "dis_proj_min": {"from": 0.005, "to": 0.005},
        "dis_proj_max": {"from": 0.99, "to": 0.95},
    }
    assert "the frontier's ladder point 9" in p["from_deck_of"]
    assert "Untested" in data["note"]
    names = [s.name for s in tools.specs()]
    assert "frontier_next_proposal" in names and "frontier_candidates" not in names
    spec = next(s for s in tools.specs() if s.name == "frontier_next_proposal")
    assert (
        "what the frontier's own retry rule would run next from your latest run deck"
        in (spec.description.lower())
    )
    assert "untested; changes only dis_proj_min/max" in spec.description
    assert "there is no tool named" in tools("frontier_candidates", {})


def test_no_outcome_of_anything_unrun_reaches_the_model(tmp_path):
    text = _tools(tmp_path)("frontier_next_proposal", {})
    for leak in ("pass", "eta", "77.", "finished", "True", "meV", "status"):
        assert leak not in text, leak


def test_the_untried_ladder_from_the_latest_deck_follows_in_order(tmp_path):
    """Round 2: the rule's proposals may be shown (never their outcomes), the next one
    flagged."""
    data = _proposal(_tools(tmp_path))
    seq = data["untried_ladder_from_that_deck"]
    assert [c["ladder_point"] for c in seq] == [10, 11, 12]
    assert seq[0] == {"ladder_point": 10, "dis_proj_min": 0.005, "dis_proj_max": 0.95,
                      "next": True}  # fmt: skip
    assert not any(c.get("next") for c in seq[1:])
    assert data["proposal"]["ladder_point"] == 10
    # a pair that rebuilds a frontier run is left out of the sequence too
    na = _proposal(_tools(tmp_path, history=_history(run0=(0.005, 0.95))))
    assert [c["ladder_point"] for c in na["untried_ladder_from_that_deck"]] == [11, 12]
    # off the ladder: every unused pair, in ladder order
    off = ({"dis_proj_min": "0.003", "dis_proj_max": "0.97"}, True)
    seq = _proposal(_tools(tmp_path, attempts=[off]))["untried_ladder_from_that_deck"]
    assert [c["ladder_point"] for c in seq] == [10, 11, 12]


def test_the_next_proposal_tool_does_not_spend_the_tool_budget(tmp_path):
    from autowan.providers import UNCAPPED_CALLS, tool_call_gate

    tools = _tools(tmp_path)
    spec = next(s for s in tools.specs() if s.name == "frontier_next_proposal")
    assert (
        spec.uncapped and "does not count against your tool budget" in spec.description
    )
    names, free = {"frontier_next_proposal", "band_character"}, {
        "frontier_next_proposal"
    }
    # the budget spent (cap 8, 8 used): the uncapped tool still answers, uncounted
    assert tool_call_gate("frontier_next_proposal", names, free, 8, 0, 8, "x") == (
        None, False, True,
    )  # fmt: skip
    refused, counted, _ = tool_call_gate("band_character", names, free, 8, 0, 8, "x")
    assert "budget for this attempt (8 calls) is spent" in refused and not counted
    assert tool_call_gate("band_character", names, free, 7, 0, 8, "x") == (
        None,
        True,
        False,
    )
    # its own bound
    over, _, _ = tool_call_gate(
        "frontier_next_proposal", names, free, 0, UNCAPPED_CALLS, 8, "x"
    )
    assert "does not change until a new deck has run" in over


def test_reference_page_states_the_archive_ladder_without_outcomes():
    from autowan.tools import REFERENCE_DIR

    page = (REFERENCE_DIR / "pdwf.md").read_text()
    for text in ("0.02,\n  0.01, 0.005, 0.0025, 0.00125, 0", "0.99, 0.95, 0.90, 0.85",
                 "first construction that passes", "lowering dis_proj_min"):  # fmt: skip
        assert text in page, text
    import re

    assert not re.search(r"meV|\beta(_2)?\b|certif", page)


def test_a_repeat_call_without_a_new_run_gives_the_same_proposal(tmp_path):
    tools = _tools(tmp_path)
    assert tools("frontier_next_proposal", {}) == tools("frontier_next_proposal", {})


def test_it_advances_only_after_a_new_deck_has_run(tmp_path):
    # autowan ran point 10's pair with another num_wann: the next is point 11, kept change
    ran = ({"dis_proj_min": "0.005", "dis_proj_max": "0.95", "num_wann": "9"}, True)
    p = _proposal(_tools(tmp_path, attempts=[ran]))["proposal"]
    assert p["ladder_point"] == 11 and p["from_deck_of"] == "attempt 1"
    assert p["kept_from_that_deck"]["num_wann"] == "9"
    # a deck that was refused before it ran does not move the rule
    refused = ({"dis_proj_min": "0.005", "dis_proj_max": "0.90"}, False)
    p2 = _proposal(_tools(tmp_path, attempts=[ran, refused]))["proposal"]
    assert p2 == p


def test_an_off_ladder_deck_gets_the_first_unused_pair_and_keeps_its_changes(tmp_path):
    off = (
        {"dis_proj_min": "3.0000000000d-03", "dis_proj_max": "0.97", "num_wann": "9",
         "num_bands": "40"},
        True,
    )  # fmt: skip
    p = _proposal(_tools(tmp_path, attempts=[off]))["proposal"]
    # points 1..9 and run 0's pair were used by the frontier's runs: point 10 is first
    assert p["ladder_point"] == 10
    assert p["changes"]["dis_proj_min"] == {"from": 0.003, "to": 0.005}
    assert p["kept_from_that_deck"] == {
        "method": "pdwf",
        "num_wann": "9",
        "num_bands": "40",
    }


def test_fortran_spellings_of_a_ladder_pair_are_that_pair(tmp_path):
    on = (
        {"dis_proj_min": "5.0000000000d-03", "dis_proj_max": "9.0000000000d-01"},
        True,
    )
    p = _proposal(_tools(tmp_path, attempts=[on]))["proposal"]
    assert p["ladder_point"] == 12


def test_a_pair_that_rebuilds_a_frontier_run_is_passed_over(tmp_path):
    # Na, V3Si: run 0 (the default deck) already ran point 10's pair on this deck
    data = _proposal(_tools(tmp_path, history=_history(run0=(0.005, 0.95))))
    assert data["proposal"]["ladder_point"] == 11
    assert "ladder point 10 (it would rebuild the frontier's run 0)" in data["note"]


def test_an_exhausted_ladder_gives_no_proposal(tmp_path):
    last = ({"dis_proj_min": "0.005", "dis_proj_max": "0.85"}, True)
    data = _proposal(_tools(tmp_path, attempts=[last]))
    assert data["proposal"] is None and "exhausted" in data["note"]


def test_a_deck_without_thresholds_gets_no_proposal(tmp_path):
    smv = ({"dis_proj_min": None, "dis_proj_max": None, "method": "smv"}, True)
    data = _proposal(_tools(tmp_path, attempts=[smv]))
    assert data["proposal"] is None and "no projectability thresholds" in data["note"]


def test_the_ordinary_ladder_mode_is_unchanged_without_a_handoff(tmp_path):
    tools = _tools(tmp_path)
    tools.frontier_handoff = None
    data = json.loads(tools("frontier_candidates", {}))
    assert [c["ladder_point"] for c in data["candidates"]][:3] == ["1", "2", "3"]
    assert "there is no tool named" in tools("frontier_next_proposal", {})
    assert "frontier_next_proposal" not in [s.name for s in tools.specs()]


# ---------------------------------------------------------------------------
# accounting: ten frontier runs, eight of autowan's own
# ---------------------------------------------------------------------------


def test_autowan_gets_eight_attempts_of_its_own_beside_the_frontiers_ten(tmp_path):
    result, runner, _ = _run(tmp_path, _decreasing(12))
    attempts = result.run.attempts
    own = [a for a in attempts if a.frontier_point is None]
    assert len(own) == 8 and [a.index for a in own] == list(range(1, 9))
    assert len(runner.localised) == 8  # every one of them ran Wannier90
    record = json.loads((tmp_path / "log.json").read_text())
    acc = record["accounting"]
    assert acc == {
        "frontier_attempts": 10,
        "handoff_point": 9,
        "autowan_attempts": 8,
        "autowan_attempt_budget": 8,
        "autowan_wannier90_runs": 8,
        "w90_runs_total": 18,
        "certified_at_autowan_attempt": None,
        "certified_at_w90_run": None,
    }
    assert "attempt budget of 9 spent" in result.run.terminal_reason


def test_certified_at_counts_the_frontiers_runs_first(tmp_path):
    run = Run(SEED, RunConfig(total_attempts=9))
    run.start_attempt({"k": 0}, seed=SEED)
    run.attempts[0].frontier_point = 9
    run.start_from_frontier(_handoff().to_record({"k": 0}))
    from autowan.orchestrator import GateOutcome

    run.finish_attempt(run.attempts[0], calculation_error="stopped")
    run.advance_after_failure()
    for k in (1, 2, 3):
        a = run.start_attempt({"k": k}, seed=f"{SEED}_r{k}")
        a.steps = [{"step": f"{SEED}_r{k}:localise", "returncode": 0}]
        run.finish_attempt(
            a,
            outcome=GateOutcome(
                passed=k == 3, evaluated_check_ids=(), failed_check_ids=()
            ),
        )
    acc = run.accounting()
    assert acc["autowan_attempts"] == 3 and acc["certified_at_autowan_attempt"] == 3
    assert acc["certified_at_w90_run"] == 13 and acc["w90_runs_total"] == 13


def test_a_proposal_repeating_a_frontier_run_is_refused_without_spending(tmp_path):
    repeat = {
        "action": "modify_parameters",
        "diagnosis": "try the frontier's point 3 thresholds",
        "evidence": [workflow.CALCULATION_FAILED],
        "targets": [workflow.CALCULATION_FAILED],
        "expected_effect": "a construction",
        "confidence": 0.3,
        "changes": [
            {"parameter": "dis_proj_min", "old": 0.005, "new": 0.02, "reason": "x"},
            {"parameter": "dis_proj_max", "old": 0.99, "new": 0.9, "reason": "x"},
        ],
    }
    result, _, model = _run(tmp_path, [repeat] + _decreasing(1))
    assert "the frontier's own run 3 before the handoff" in model.payloads[1]
    own = [a for a in result.run.attempts if a.frontier_point is None]
    assert int(float(own[0].parameters["num_iter"])) == 3900


def test_resume_restores_the_handoff_and_its_repeat_guard(tmp_path):
    _run(tmp_path, _decreasing(1))
    record = json.loads((tmp_path / "log.json").read_text())
    run = Run.from_dict(record)
    assert run.frontier_handoff["point"] == 9
    params = dict(record["frontier_handoff"]["history"][4]["parameters"])
    assert run.frontier_run_of(params) == 4 and run.would_repeat(params)
    assert run.attempts[0].frontier_point == 9


@pytest.mark.parametrize("spin_index", [0, 1])
def test_each_spin_channel_is_handed_over_with_its_own_history(tmp_path, spin_index):
    """C3 MnO/Fe: each channel is its own run, with its own channel's frontier runs."""
    history = tuple(
        {
            **e,
            "measurements": {
                "dmft_subspace_model_window": {"value": 10.0 * spin_index + e["run"]}
            },
        }
        for e in _history()
    )
    handoff = _handoff(history)
    view = handoff.model_view()
    assert [
        r["measurements"]["dmft_subspace_model_window"]["value"] for r in view["runs"]
    ] == [10.0 * spin_index + r for r in range(10)]


# ---------------------------------------------------------------------------
# the gate itself is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile, digest",
    [
        ("default", "6d07e441e0930d6c"),
        ("band_interpolation", "03e038df686b1d80"),
        ("sparse_tb", "1856955f699ba0c4"),
        ("downfolded_d", "109693a2b2a8f92d"),
        ("dmft_subspace", "333e859de17049d0"),
    ],
)
def test_gate_profile_hashes_are_unchanged(profile, digest):
    assert get_profile(profile).hash == digest


# ---------------------------------------------------------------------------
# the repeat guard sees the overlap input (SCDM knobs live in the .pw2wan)
# ---------------------------------------------------------------------------

SCDM_PW2WAN = """\
&inputpp
  outdir = './out/'
  prefix = 'aiida'
  scdm_proj = .true.
  scdm_entanglement = 'erfc'
  seedname = '{seed}'
  scdm_mu = 0.13760361441681868
  scdm_sigma = {sigma}
/
"""
SCDM_WIN = "num_wann = 8\nnum_bands = 32\nnum_iter = 4000\ndis_num_iter = 0\n"


def _scdm_deck(directory, seed, sigma):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{seed}.win").write_text(SCDM_WIN)
    (directory / f"{seed}.pw2wan").write_text(
        SCDM_PW2WAN.format(seed=seed, sigma=sigma)
    )
    from autowan.inputs import read_construction

    return read_construction(directory, seed)


def test_two_scdm_decks_differing_only_in_sigma_are_not_a_repeat(tmp_path):
    """runs_v11 graphene C1 (GPT): every sigma-only proposal after an SCDM attempt was
    refused 16 times as "an attempt already made" -- the .pw2wan was not in the set."""
    from autowan.orchestrator import _state_key

    a = workflow._parameters(_scdm_deck(tmp_path / "a", "g_r7", "3.0"))
    b = workflow._parameters(_scdm_deck(tmp_path / "b", "g_r8", "6.0"))
    same = workflow._parameters(_scdm_deck(tmp_path / "c", "g_r9", "3.0"))
    assert a["method"] == "scdm" and a["pw2wan.scdm_sigma"] == "3.0"
    assert "pw2wan.seedname" not in a and "pw2wan.outdir" not in a
    assert _state_key(a) != _state_key(b)
    assert _state_key(a) == _state_key(same)  # the seedname is not the construction
    run = Run("g", RunConfig())
    run.start_attempt(a, seed="g_r7", method="scdm")
    assert not run.would_repeat(b) and run.would_repeat(same)


def test_a_sigma_only_proposal_on_an_scdm_deck_is_not_refused_as_a_repeat(tmp_path):
    from autowan.actions import Stage, validate_action
    from autowan.backends.qe import QeBackend

    inputs = _scdm_deck(tmp_path, "g_r7", "3.0")
    run = Run("g", RunConfig())
    run.start_attempt(workflow._parameters(inputs), seed="g_r7", method="scdm")
    run.advance_after_failure()
    check = workflow.CALCULATION_FAILED
    action = validate_action(
        {"action": "modify_parameters", "diagnosis": "d", "evidence": [check],
         "targets": [check], "expected_effect": "e", "confidence": 0.5,
         "changes": [{"parameter": "scdm_sigma", "old": 3.0, "new": 6.0, "reason": "r"}]},
        stage=Stage.REPAIR, evaluated_check_ids=(check,), method="scdm",
        unresolved_check_ids=(check,),
    )  # fmt: skip
    prospective = workflow._prospective(
        action, inputs, _sro_parent(), run, backend=QeBackend()
    )
    assert prospective["pw2wan.scdm_sigma"] == "6.0"
    assert not run.would_repeat(prospective)
