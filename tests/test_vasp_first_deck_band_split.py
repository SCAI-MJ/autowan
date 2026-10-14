"""The VASP first deck on a parent with band headroom: same deck, surplus excluded.

The campaign's VASP parents carry Quantum ESPRESSO-level empty-band headroom, so a
repair that widens the disentanglement has bands to draw on (discussion 2026-10-02,
autowan item 1, Decision 2). The first deck must not see that headroom, or the
unrepaired measurement changes definition. So the deck stays at the QE ratio,

    num_bands = min(parent bands,
                    round_up_8(max(2 num_wann, num_wann + frozen_max + 2)))

with ``frozen_max`` the most states at or below the deck's ``dis_froz_max`` at any
spin and k-point -- the parent-side rule the 2026-10-01 parents were computed with
(``parent_inputs.nscf_nbands``) -- and ``exclude_bands`` covers every parent band above
it. The bands are excluded, not absent: a repair that raises ``num_bands`` uncovers
them through the existing ``exclude_bands`` rewrite, with fresh overlaps from the
stored parent and no DFT rerun. A request past the parent's count is the DFT
prescription, as before.

VASP 6.4.1 honours the exclusion in its overlap step (``mlwf.F``: ``EXCLUDE_BAND``,
``NB_TOT - NEXCLB == num_bands``, the ``.eig`` renumbered over the kept bands); the
stand-in engine below writes exactly that layout.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autowan.backends import get_backend
from autowan.backends.vasp import inputs as vasp_inputs
from autowan.backends.vasp.execution import VaspWannierRunner
from autowan.backends.vasp.inputs import deck_from_parent, read_construction
from autowan.backends.vasp.outputs import read_parent
from autowan.execution import ConstructionRunner, StepResult
from autowan.inputs import read_win
from autowan.parent_adequacy import ADEQUATE
from autowan.providers import ModelResponse

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "vasp"
W90 = HERE / "fixtures" / "wannier90"
PARENT_INPUTS = (
    HERE.parent
    / "experiments"
    / "vasp_parents_twenty_campaign_materials"
    / "parent_inputs.py"
)
BAND_PARTITION = "band_partition_matches_deck"

#: (material, seed, the fixture's band count, the same parent with headroom)
PAIRS = [("Si", "si", 16, 24), ("SrO", "sro", 32, 40)]


# ---------------------------------------------------------------------------
# helpers: copies of a fixture parent with bands appended above its top
# ---------------------------------------------------------------------------


def _spin_sets(root):
    return root.find("calculation/eigenvalues/array/set")


def _set_band_count(root, n):
    for item in root.iter("i"):
        if item.get("name") == "NBANDS":
            item.text = f" {n} "


def pad_bands(root, n):
    """Extend every k-point's spectrum to ``n`` bands, empty, 1 eV apart above the top."""
    for spin in _spin_sets(root).findall("set"):
        for kpoint in spin.findall("set"):
            rows = kpoint.findall("r")
            top = float(rows[-1].text.split()[0])
            for i in range(n - len(rows)):
                r = ET.SubElement(kpoint, "r")
                r.text = f" {top + 1.0 + i:.4f} 0.0000 "
    _set_band_count(root, n)


def lower_bands(root, kpoint_index, first, energies):
    """Move empty bands ``first...`` at one k-point to the given (ascending) energies."""
    kpoint = _spin_sets(root).findall("set")[0].findall("set")[kpoint_index]
    rows = kpoint.findall("r")
    for r, energy in zip(rows[first:], energies):
        r.text = f" {energy:.4f} 0.0000 "


def _parent(tmp_path, material, name, *edits):
    d = tmp_path / name
    d.mkdir()
    for f in ("vasprun.xml", "OUTCAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(FIXTURES / material / f, d / f)
    if edits:
        tree = ET.parse(d / "vasprun.xml")
        for edit in edits:
            edit(tree.getroot())
        tree.write(d / "vasprun.xml", encoding="unicode")
    return d


def _deck(parent, run_dir, seed):
    """The deck's text, with the parent's directory written as ``<parent>``.

    The deck names the file its projector channels came from, and two parents are two
    directories; that provenance line is compared up to the directory, every other line
    verbatim.
    """
    run_dir.mkdir(exist_ok=True)
    # The SMV (energy-window) deck: the projectability deck is PDWF, which VASP cannot run.
    deck_from_parent(
        parent / "vasprun.xml", run_dir, seed, recipe="hydrogenic_energy_window"
    )
    return (run_dir / f"{seed}.win").read_text().replace(str(parent), "<parent>")


def _without_exclusion(text):
    return [line for line in text.splitlines() if not line.startswith("exclude_bands")]


def _assigned(win, name):
    """A deck's own ``name = value``, read off the text (None when it sets none).

    Read from the deck rather than from the parsed construction values, which carry
    only the keys the repair model is shown -- and ``exclude_bands`` is not one of them,
    on either backend.
    """
    for line in Path(win).read_text().splitlines():
        key, sep, value = line.split("!")[0].partition("=")
        if sep and key.strip().lower() == name:
            return value.strip()
    return None


def _parent_side_rule():
    """The experiment's ``parent_inputs`` loaded by path (it is not a package)."""
    name = "vasp_parents_twenty_campaign_materials_parent_inputs"
    if not PARENT_INPUTS.is_file():
        pytest.skip("the campaign's parent-input driver is not part of the package")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PARENT_INPUTS)
        module = importlib.util.module_from_spec(spec)
        # registered first: its dataclasses resolve their module through sys.modules
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def test_the_first_deck_band_count_is_the_parent_side_rule():
    """One rule on both sides: the band set a headroom-less parent was computed with
    is the band set the first deck takes from a parent with headroom."""
    pi = _parent_side_rule()
    assert vasp_inputs.FROZEN_COUNT_MARGIN == pi.FROZEN_COUNT_MARGIN
    assert vasp_inputs.BAND_MULTIPLE == pi.BAND_MULTIPLE
    for num_wann in range(1, 49):
        for frozen_max in range(0, num_wann + 1):
            rule = pi.nscf_nbands(num_wann, frozen_max)
            assert (
                vasp_inputs.first_deck_band_count(num_wann, frozen_max, 10_000) == rule
            )
            # a parent short of the rule keeps every band it has
            assert (
                vasp_inputs.first_deck_band_count(num_wann, frozen_max, rule - 1)
                == rule - 1
            )


@pytest.mark.parametrize("material, seed, bands, headroom", PAIRS)
def test_a_headroom_parent_gets_the_same_first_deck_with_its_surplus_excluded(
    tmp_path, material, seed, bands, headroom
):
    lean = _parent(tmp_path, material, "lean")
    roomy = _parent(tmp_path, material, "roomy", lambda r: pad_bands(r, headroom))
    lean_deck = _deck(lean, tmp_path / "lean_run", seed)
    roomy_deck = _deck(roomy, tmp_path / "roomy_run", seed)

    assert "exclude_bands" not in lean_deck
    assert read_win(tmp_path / "lean_run" / f"{seed}.win")["num_bands"] == str(bands)
    excluded = [
        line for line in roomy_deck.splitlines() if line.startswith("exclude_bands")
    ]
    assert excluded == [f"exclude_bands = {bands + 1}-{headroom}"]
    # every other line -- num_bands, num_wann, dis_froz_max and its cap note, the
    # projections, the k-points -- is the headroom-less parent's deck, verbatim
    assert _without_exclusion(roomy_deck) == lean_deck.splitlines()


def test_the_frozen_window_term_sets_the_band_count_when_it_dominates(tmp_path):
    """Silicon with four empty bands pulled under the window at Gamma: the window
    there holds all 8 functions' worth of states, so the deck needs
    round_up_8(8 + 8 + 2) = 24 bands, not 2 x 8 = 16."""
    crowd = lambda r: lower_bands(r, 0, 4, (7.0, 7.2, 7.4, 7.6))  # noqa: E731
    lean = _parent(tmp_path, "Si", "lean", crowd, lambda r: pad_bands(r, 24))
    roomy = _parent(tmp_path, "Si", "roomy", crowd, lambda r: pad_bands(r, 32))
    short = _parent(tmp_path, "Si", "short", crowd)
    lean_deck = _deck(lean, tmp_path / "lean_run", "si")
    roomy_deck = _deck(roomy, tmp_path / "roomy_run", "si")
    short_deck = _deck(short, tmp_path / "short_run", "si")

    deck = read_win(tmp_path / "roomy_run" / "si.win")
    top = float(deck["dis_froz_max"])
    bands = read_parent(roomy / "vasprun.xml").eigenvalues_ev
    frozen_max = int((bands <= top).sum(axis=-1).max())
    assert (int(deck["num_wann"]), frozen_max) == (8, 8)
    assert deck["num_bands"] == "24"
    assert "exclude_bands = 25-32" in roomy_deck.splitlines()
    assert _without_exclusion(roomy_deck) == lean_deck.splitlines()
    # a parent short of the rule: every band it carries, nothing to exclude
    assert read_win(tmp_path / "short_run" / "si.win")["num_bands"] == "16"
    assert "exclude_bands" not in short_deck


# ---------------------------------------------------------------------------
# the surplus-band pre-flight alarm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material, seed, bands, headroom", PAIRS)
def test_the_partition_check_is_silent_on_a_deck_written_from_its_own_parent(
    tmp_path, material, seed, bands, headroom
):
    roomy = _parent(tmp_path, material, "roomy", lambda r: pad_bands(r, headroom))
    run_dir = tmp_path / "run"
    _deck(roomy, run_dir, seed)
    report = get_backend("vasp").assess_parent(
        roomy / "vasprun.xml", read_construction(run_dir, seed), declared_moments={}
    )
    f = report.finding(BAND_PARTITION)
    assert f.status == ADEQUATE
    assert (f.measured["num_bands"], f.measured["excluded"]) == (
        bands,
        headroom - bands,
    )
    assert f.measured["surplus_bands"] == 0
    assert report.alarms == ()


def test_the_partition_check_still_fires_on_a_hand_fixed_deck_without_exclusions(
    tmp_path,
):
    """The 2026-09-29 blind spot stays covered: the 16-band deck written for the
    16-band silicon parent, read against the same parent recomputed with 24 bands."""
    lean = _parent(tmp_path, "Si", "lean")
    roomy = _parent(tmp_path, "Si", "roomy", lambda r: pad_bands(r, 24))
    run_dir = tmp_path / "run"
    assert "exclude_bands" not in _deck(lean, run_dir, "si")
    report = get_backend("vasp").assess_parent(
        roomy / "vasprun.xml", read_construction(run_dir, "si"), declared_moments={}
    )
    assert [f.check_id for f in report.blocking] == [BAND_PARTITION]
    assert report.finding(BAND_PARTITION).measured["surplus_bands"] == 8


# ---------------------------------------------------------------------------
# a repair reaches the headroom without a DFT rerun
# ---------------------------------------------------------------------------

#: VASP's overlap step reduced to what this test needs from it: it reads the staged
#: wannier90.win and writes the .eig for the bands the deck keeps, renumbered over
#: them, in mlwf.F's own format (2I12, F22.12). The energies are the parent's.
STAND_IN_VASP = """\
#!{python}
import json
import re
from pathlib import Path

win = Path("wannier90.win").read_text()
num_bands = int(re.search(r"^num_bands\\s*=\\s*(\\d+)", win, re.M).group(1))
excluded = set()
found = re.search(r"^exclude_bands\\s*=\\s*(\\d+)-(\\d+)", win, re.M)
if found:
    excluded = set(range(int(found.group(1)), int(found.group(2)) + 1))
bands = json.loads(Path({bands!r}).read_text())
kept = [n for n in range(1, len(bands[0]) + 1) if n not in excluded]
assert len(kept) == num_bands, "num_bands seems corrupt"
with open("wannier90.eig", "w") as eig:
    for k, energies in enumerate(bands, start=1):
        for m, n in enumerate(kept, start=1):
            eig.write(f"{{m:12d}}{{k:12d}}{{energies[n - 1]:22.12f}}\\n")
for name in ("wannier90.amn", "wannier90.mmn"):
    Path(name).write_text(f"stand-in over bands {{kept[0]}}-{{kept[-1]}}\\n")
"""


def _stand_in_runner(tmp_path, parent, run_dir):
    bands = read_parent(parent / "vasprun.xml").eigenvalues_ev[0]
    (tmp_path / "bands.json").write_text(json.dumps(bands.tolist()))
    (parent / "POTCAR").write_text("stand-in: the licensed POTCAR is not committed\n")
    (parent / "CHGCAR").write_text("stand-in\n")
    stub = tmp_path / "vasp_stand_in"
    stub.write_text(
        STAND_IN_VASP.format(python=sys.executable, bands=str(tmp_path / "bands.json"))
    )
    stub.chmod(0o755)
    return VaspWannierRunner(
        run_dir=run_dir,
        parent_dir=parent,
        vasp=str(stub),
        wannier90="true",
        mpi_launcher=(),
    )


def _eig(path):
    rows = np.loadtxt(path)
    return {int(k): rows[rows[:, 1] == k][:, 2] for k in np.unique(rows[:, 1])}


def test_a_repair_within_the_headroom_uncovers_excluded_bands_with_fresh_overlaps(
    tmp_path,
):
    roomy = _parent(tmp_path, "Si", "roomy", lambda r: pad_bands(r, 24))
    run_dir = tmp_path / "run"
    _deck(roomy, run_dir, "si")
    vasp = get_backend("vasp")
    source = read_construction(run_dir, "si")
    assert _assigned(source.win, "exclude_bands") == "17-24"

    widened = vasp.write_derived(source, run_dir, "si_r1", {"num_bands": 20})
    assert widened.values["num_bands"] == "20"
    # 8 excluded -> 4: by exactly the 4 bands the repair adds
    assert _assigned(widened.win, "exclude_bands") == "21-24"
    whole = vasp.write_derived(source, run_dir, "si_r2", {"num_bands": 24})
    assert _assigned(whole.win, "exclude_bands") is None

    runner = _stand_in_runner(tmp_path, roomy, run_dir)
    assert runner.invalidates_overlaps({"num_bands": 20}) is True
    for seed in ("si", "si_r1"):
        [result] = runner.run_overlaps(seed)
        assert result.ok, result.stderr_tail
    staged = runner.stage_dir("si_r1") / "wannier90.win"
    assert (_assigned(staged, "num_bands"), _assigned(staged, "exclude_bands")) == (
        "20",
        "21-24",
    )

    parent_bands = read_parent(roomy / "vasprun.xml").eigenvalues_ev[0]
    first, repaired = _eig(run_dir / "si.eig"), _eig(run_dir / "si_r1.eig")
    for k, energies in repaired.items():
        assert len(first[k]) == 16 and len(energies) == 20
        # the four bands the repair uncovered are the parent's own bands 17-20
        np.testing.assert_allclose(energies[16:], parent_bands[k - 1, 16:20])
        np.testing.assert_allclose(energies[:16], first[k])


class _LaysDown(ConstructionRunner):
    """Runs nothing; lays a finished output down under each seed, recording reuse."""

    def __init__(self, run_dir, wout):
        super().__init__(run_dir=run_dir)
        self.wout = wout
        self.builds = []

    def run_overlaps(self, seed):  # pragma: no cover - build is overridden
        raise AssertionError("never computes overlaps")

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        shutil.copy(self.wout, self.run_dir / f"{seed}.wout")
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


class _AsksForBands:
    """A model that always asks for ``requested`` bands."""

    def __init__(self, requested):
        self.requested = requested
        self.calls = 0

    def propose(self, *, system, payload_text, schema, images=(), temperature=0.0, **_):
        self.calls += 1
        props = schema["properties"]
        content = {
            "action": "modify_parameters",
            "diagnosis": "the disentanglement needs more of the spectrum",
            "evidence": props["evidence"]["items"]["enum"][:1],
            "expected_effect": "the occupied manifold is represented",
            "confidence": 0.5,
            "changes": [
                {
                    "parameter": "num_bands",
                    "new": self.requested,
                    "reason": "more states",
                }
            ],
        }
        targets = props.get("targets", {}).get("items", {}).get("enum", [])[:1]
        if targets:
            content["targets"] = targets
        return ModelResponse(
            content=content, model="stub", raw_text=json.dumps(content)
        )


def _loop(tmp_path, requested):
    from autowan.workflow import WorkflowConfig, wannierize

    roomy = _parent(tmp_path, "Si", "roomy", lambda r: pad_bands(r, 24))
    run_dir = tmp_path / "run"
    _deck(roomy, run_dir, "si")
    runner = _LaysDown(run_dir, W90 / "graphene_projectability_fixed.wout")
    model = _AsksForBands(requested)
    result = wannierize(
        run_dir=run_dir,
        parent_xml=roomy / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=lambda _m: model,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
        declared_moments={},
    )
    return result, runner, model, run_dir


def test_the_loop_widens_into_the_headroom_and_rebuilds_the_overlaps(tmp_path):
    result, runner, model, run_dir = _loop(tmp_path, requested=20)
    assert result.parent_adequacy.blocking == ()
    assert runner.builds[0] == ("si", None)
    assert runner.builds[1] == ("si_r1", None), "overlaps reused across a band change"
    widened = run_dir / "si_r1.win"
    assert (_assigned(widened, "num_bands"), _assigned(widened, "exclude_bands")) == (
        "20",
        "21-24",
    )
    assert result.status != "DFT prescription returned"
    assert result.prescription is None


def test_a_request_one_band_past_the_parent_returns_the_prescription(tmp_path):
    result, runner, model, run_dir = _loop(tmp_path, requested=25)
    assert model.calls >= 1
    assert runner.builds == [("si", None)]
    assert result.diagnosis.blocked_by_parent is True
    assert result.status == "DFT prescription returned"
    assert "NBANDS = 25" in result.prescription
    assert "nbnd" not in result.prescription
