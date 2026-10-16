"""The backend protocol: runner, deck writer and parsers as one swappable object.

Two engines have to satisfy the same contract for the gate's claim -- that no threshold
changes between engines -- to be testable at all. The contract is deliberately narrow:
a backend reads the parent calculation into the one ``ParentCalculation`` the gate
already consumes, reads and rewrites the deck, supplies a runner whose localisation half
is shared code rather than a copy, explains its own engine's refusal lines, and may
*add* checks to the gate but never remove or replace one. Everything a backend knows
about its engine's deck grammar and failure signatures is data, so a third engine is
one module set that can be dropped from a public branch without touching the gate.
"""

from __future__ import annotations

import inspect
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from autowan.actions import EDITABLE_PARAMETERS
from autowan.backends import (
    BACKENDS,
    Backend,
    BackendError,
    EngineKnowledge,
    gate_for,
    get_backend,
    register_backend,
)
from autowan.core_checks import CORE_CHECKS
from autowan.execution import ConstructionRunner, StepResult, WannierRunner
from autowan.gate import (
    Check,
    CheckOutcome,
    GateDefinitionError,
)
from autowan.inputs import InputError, read_construction
from autowan.qe_outputs import ParentCalculation, QeOutputError

FIXTURES = Path(__file__).parent / "fixtures" / "vasp"
SYNTHETIC = FIXTURES / "synthetic"

REQUIRED_METHODS = (
    "read_parent",
    "read_construction",
    "write_derived",
    "deck_files",
    "make_runner",
    "refusal_reason",
    "contributed_checks",
    "route",
)

WIN = """\
num_wann  = 9
num_bands = 14
num_iter  = 400
conv_window = 3
dis_froz_max = 30.0
write_hr = .true.

begin projections
Fe:s;p;d
end projections

mp_grid 8 8 8

begin kpoints
 0.0 0.0 0.0
 0.5 0.5 0.5
end kpoints
"""

PW2WAN = """\
&inputpp
  outdir = './'
  prefix = 'Fe'
  seedname = 'base'
  spin_component = 'up'
  atom_proj = .true.
/
"""

VASP_WIN = """\
num_wann  = 8
num_bands = 16
num_iter  = 400
conv_window = 3
dis_froz_max = 7.0
dis_froz_proj = .true.
dis_proj_min = 0.01
dis_proj_max = 0.95
write_hr = .true.

begin projections
Si:s;p
end projections

begin unit_cell_cart
Ang
-2.71534056  0.00000000  2.71534056
 0.00000000  2.71534056  2.71534056
-2.71534056  2.71534056  0.00000000
end unit_cell_cart

begin atoms_frac
Si  0.00 0.00 0.00
Si  0.25 0.25 0.25
end atoms_frac

mp_grid 2 1 1

begin kpoints
 0.0 0.0 0.0
 0.5 0.0 0.0
end kpoints
"""


#: The same deck with no projectability window asked for. The loop test below lays down
#: an energy-window ``.wout``; paired with a deck that asked for a projectability window,
#: that construction is invalid and never reaches the gate
#: (``test_projectability_window.py``), which is not what that test is about.
VASP_ENERGY_WINDOW_WIN = "".join(
    line
    for line in VASP_WIN.splitlines(keepends=True)
    if not line.startswith(("dis_proj_", "dis_froz_proj"))
)


@pytest.fixture
def qe_deck(tmp_path):
    (tmp_path / "base.win").write_text(WIN)
    (tmp_path / "base.pw2wan").write_text(PW2WAN)
    return tmp_path


@pytest.fixture
def vasp_deck(tmp_path):
    (tmp_path / "si.win").write_text(VASP_WIN)
    return tmp_path


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def test_the_two_shipped_backends_are_registered_under_their_engine_names():
    assert {"qe", "vasp"} <= set(BACKENDS)
    for name in ("qe", "vasp"):
        backend = get_backend(name)
        assert isinstance(backend, Backend)
        assert backend.name == name
        assert isinstance(backend.knowledge, EngineKnowledge)
        assert backend.knowledge.engine == name
        for method in REQUIRED_METHODS:
            assert callable(getattr(backend, method)), (name, method)


def test_get_backend_accepts_an_instance_and_returns_it_unchanged():
    backend = get_backend("vasp")
    assert get_backend(backend) is backend


def test_an_unknown_backend_is_refused_by_name_with_the_known_ones_listed():
    with pytest.raises(BackendError, match="siesta") as info:
        get_backend("siesta")
    assert "qe" in str(info.value) and "vasp" in str(info.value)


# ---------------------------------------------------------------------------
# QE is the first instance: nothing it does changes
# ---------------------------------------------------------------------------


def test_the_qe_backend_reads_and_writes_the_deck_exactly_as_the_module_functions_do(
    qe_deck,
):
    backend = get_backend("qe")
    via_backend = backend.read_construction(qe_deck, "base")
    direct = read_construction(qe_deck, "base")
    assert via_backend == direct
    assert backend.deck_files(via_backend) == (
        qe_deck / "base.win",
        qe_deck / "base.pw2wan",
    )

    derived = backend.write_derived(via_backend, qe_deck, "base_r1", {"num_iter": 4000})
    text = (qe_deck / "base_r1.win").read_text()
    assert derived.values["num_iter"] == "4000"
    assert derived.pw2wan == qe_deck / "base_r1.pw2wan"
    for preserved in ("begin kpoints", "0.5 0.5 0.5", "mp_grid 8 8 8", "Fe:s;p;d"):
        assert preserved in text


def test_the_qe_backend_parent_reader_is_the_quantum_espresso_one(tmp_path):
    with pytest.raises(QeOutputError):
        get_backend("qe").read_parent(tmp_path / "absent.xml")


def test_the_qe_backend_supplies_the_runner_the_package_always_had(tmp_path):
    runner = get_backend("qe").make_runner(
        tmp_path,
        wannier90="w90",
        pw2wannier90="p2w",
        mpi_launcher=("mpirun", "-np", "2"),
    )
    assert isinstance(runner, WannierRunner)
    assert runner.pw2wannier90 == "p2w"
    assert runner.wannier90 == "w90"
    assert tuple(runner.mpi_launcher) == ("mpirun", "-np", "2")


# ---------------------------------------------------------------------------
# the runner: one shared localisation half, one engine-specific overlap half
# ---------------------------------------------------------------------------


def test_each_backend_supplies_a_runner_that_shares_the_localisation_half(tmp_path):
    qe = get_backend("qe").make_runner(tmp_path)
    vasp = get_backend("vasp").make_runner(tmp_path)
    assert isinstance(qe, ConstructionRunner)
    assert isinstance(vasp, ConstructionRunner)
    assert not isinstance(vasp, WannierRunner)
    for runner in (qe, vasp):
        # the invalidation rule is read off the vocabulary, so it is the same rule
        assert runner.invalidates_overlaps(["projectors"])
        assert runner.invalidates_overlaps(["num_bands"])
        assert not runner.invalidates_overlaps(["num_iter", "conv_window"])
        # shared code, not a copy: the engine-independent half is the base class's
        assert type(runner).run_localisation is ConstructionRunner.run_localisation
        assert type(runner).build is ConstructionRunner.build
        assert (
            type(runner).check_deck_matches_its_projections
            is ConstructionRunner.check_deck_matches_its_projections
        )
    # and the overlap half is the engine's own
    assert type(qe).run_overlaps is not ConstructionRunner.run_overlaps
    assert type(vasp).run_overlaps is not ConstructionRunner.run_overlaps
    assert type(vasp).run_overlaps is not type(qe).run_overlaps


def test_a_launch_option_of_the_other_engine_is_refused_rather_than_ignored(tmp_path):
    vasp = get_backend("vasp").make_runner(
        tmp_path,
        wannier90="w90",
        vasp="/opt/vasp_std",
        mpi_launcher=("mpirun", "-np", "4"),
    )
    assert vasp.vasp == "/opt/vasp_std"
    assert vasp.wannier90 == "w90"
    with pytest.raises(BackendError, match="pw2wannier90"):
        get_backend("vasp").make_runner(tmp_path, pw2wannier90="p2w")
    with pytest.raises(BackendError, match="vasp"):
        get_backend("qe").make_runner(tmp_path, vasp="/opt/vasp_std")


# ---------------------------------------------------------------------------
# the knowledge partition: what the modifier knows about an engine, as data
# ---------------------------------------------------------------------------


def test_the_knowledge_partition_covers_every_editable_parameter_for_every_backend():
    for name in ("qe", "vasp"):
        knowledge = get_backend(name).knowledge
        for parameter in EDITABLE_PARAMETERS:
            in_grammar = parameter.name in knowledge.deck_grammar
            unsupported = parameter.name in knowledge.unsupported_parameters
            assert in_grammar != unsupported, (name, parameter.name)
        for rule in knowledge.deck_grammar.values():
            assert rule.file
            assert rule.spelling


def test_vasp_has_no_pdwf_and_writes_its_scdm_weight_as_overlap_step_tags():
    vasp = get_backend("vasp").knowledge
    assert set(vasp.unsupported_parameters) == {"pao_subset"}
    assert vasp.deck_grammar["projectors"].file == "win"
    assert vasp.supports("num_iter")
    assert (
        vasp.supports("scdm_mu")
        and "CUTOFF_MU" in vasp.deck_grammar["scdm_mu"].spelling
    )
    assert not vasp.supports("pao_subset")
    qe = get_backend("qe").knowledge
    assert qe.unsupported_parameters == ()
    assert qe.supports("scdm_mu") and qe.supports("pao_subset")


def test_the_knowledge_partition_explains_an_engine_failure_line_and_stays_silent_otherwise():
    vasp = get_backend("vasp").knowledge
    assert vasp.failure_signatures
    meaning = vasp.explain(
        "|     VASP was compiled without the VASP2WANNIER90 interface.     |\n"
        "|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |"
    )
    assert meaning is not None
    assert "VASP2WANNIER90" in meaning
    assert vasp.explain("aborting loop EDIFF was not reached (unconverged)") is not None
    assert vasp.explain("The file wannier90.win can not be found.") is not None
    assert (
        vasp.explain(
            " reached required accuracy - stopping structural energy minimisation"
        )
        is None
    )
    assert vasp.explain("") is None

    qe = get_backend("qe").knowledge
    assert (
        qe.explain("blk2cyc_zredist (1): nb less than the number of proc") is not None
    )


# ---------------------------------------------------------------------------
# the VASP deck writer: every editable knob lands in the .win, or is refused
# ---------------------------------------------------------------------------


def test_the_vasp_backend_reads_a_deck_with_no_overlap_input_file(vasp_deck):
    backend = get_backend("vasp")
    inputs = backend.read_construction(vasp_deck, "si")
    assert inputs.seed == "si"
    assert inputs.pw2wan is None
    assert inputs.values["num_wann"] == "8"
    assert inputs.values["projectors"] == "Si:s;p"
    assert inputs.values["projection_source"] == "analytic orbitals named in the input"
    assert backend.deck_files(inputs) == (vasp_deck / "si.win",)


def test_a_vasp_localisation_change_is_written_into_the_derived_win(vasp_deck):
    backend = get_backend("vasp")
    source = backend.read_construction(vasp_deck, "si")
    derived = backend.write_derived(source, vasp_deck, "si_r1", {"num_iter": 4000})
    text = (vasp_deck / "si_r1.win").read_text()
    assert derived.values["num_iter"] == "4000"
    assert derived.pw2wan is None
    assert not (vasp_deck / "si_r1.pw2wan").exists()
    for preserved in (
        "begin kpoints",
        "0.5 0.0 0.0",
        "mp_grid 2 1 1",
        "Si:s;p",
        "end unit_cell_cart",
    ):
        assert preserved in text


def test_naming_projectors_under_vasp_rewrites_the_block_and_drops_the_automatic_keys(
    vasp_deck,
):
    # An SMV deck: the projectability window of the old hydrogenic first deck removed.
    smv = "\n".join(
        line
        for line in (vasp_deck / "si.win").read_text().splitlines()
        if not line.startswith(("dis_proj_", "dis_froz_proj"))
    )
    (vasp_deck / "si.win").write_text(smv + "\n")
    backend = get_backend("vasp")
    source = backend.read_construction(vasp_deck, "si")
    derived = backend.write_derived(
        source,
        vasp_deck,
        "si_r1",
        {"projectors": [{"species": "Si", "orbitals": ("sp3",)}]},
        species_counts={"Si": 2},
    )
    text = (vasp_deck / "si_r1.win").read_text()
    assert derived.values["projectors"] == "Si:sp3"
    assert derived.values["num_wann"] == "8"
    assert "Si:s;p" not in text
    assert "dis_proj_min" not in text and "dis_proj_max" not in text
    assert "dis_froz_proj" not in text


def test_a_density_matrix_knob_is_refused_by_the_vasp_writer_outside_scdm(vasp_deck):
    backend = get_backend("vasp")
    source = backend.read_construction(vasp_deck, "si")
    with pytest.raises(InputError, match="SCDM knob"):
        backend.write_derived(source, vasp_deck, "si_r1", {"scdm_mu": 4.0})
    assert not (vasp_deck / "si_r1.win").exists()


def test_naming_projectors_on_the_vasp_pdwf_deck_is_refused(vasp_deck):
    backend = get_backend("vasp")
    source = backend.read_construction(vasp_deck, "si")
    with pytest.raises(InputError, match="SMV knob"):
        backend.write_derived(
            source,
            vasp_deck,
            "si_r1",
            {"projectors": [{"species": "Si", "orbitals": ("sp3",)}]},
            species_counts={"Si": 2},
        )


def test_the_vasp_scdm_weight_is_written_as_cutoff_tags_in_absolute_ev(tmp_path):
    from autowan.backends.vasp.inputs import overlap_step_incar_tags

    (tmp_path / "si.win").write_text(
        "! overlap-step INCAR: LSCDM = .TRUE.\nnum_wann = 8\nnum_bands = 8\n"
    )
    backend = get_backend("vasp")
    source = backend.read_construction(tmp_path, "si")
    backend.write_derived(
        source,
        tmp_path,
        "si_r1",
        {"scdm_mu": 1.5, "scdm_sigma": 0.7},
        fermi_energy_ev=6.0,
    )
    tags = overlap_step_incar_tags((tmp_path / "si_r1.win").read_text())
    assert tags["LSCDM"] == ".TRUE."
    assert float(tags["CUTOFF_MU"]) == 7.5
    assert float(tags["CUTOFF_SIGMA"]) == 0.7
    assert tags["CUTOFF_TYPE"] == "erfc"


def test_the_vasp_backend_reads_the_parent_into_the_one_parent_class():
    parent = get_backend("vasp").read_parent(SYNTHETIC / "vasprun.xml")
    assert isinstance(parent, ParentCalculation)
    assert parent.n_bands == 4
    assert parent.species == ("Si", "Si")


def test_both_backends_route_a_wannier90_deck_as_a_maximally_localised_construction(
    qe_deck, vasp_deck
):
    """The seam a third backend extends: which route a deck takes through the gate.

    Both shipped backends build maximally-localised functions through Wannier90, so
    both answer ``mlwf``. An engine with a second construction route (a closest-Wannier
    scheme, say) answers differently, and route-scoped withholding of a check hangs off
    this answer -- outside the gate, which stays class-scoped and unchanged here.
    """
    qe_inputs = get_backend("qe").read_construction(qe_deck, "base")
    vasp_inputs = get_backend("vasp").read_construction(vasp_deck, "si")
    assert get_backend("qe").route(qe_inputs) == "mlwf"
    assert get_backend("vasp").route(vasp_inputs) == "mlwf"


# ---------------------------------------------------------------------------
# contributed checks only add, and a third backend is one separable module set
# ---------------------------------------------------------------------------


def _stub_check(check_id: str, origin: str = "stub") -> Check:
    return Check(
        check_id,
        "a stub quantity",
        "discrete",
        "self",
        lambda c: CheckOutcome(True, 0.0, "unit"),
        origin=origin,
    )


class _StubBackend(Backend):
    """A third engine as one module set: registers itself, contributes one check."""

    name = "stub"
    knowledge = EngineKnowledge(engine="stub")

    def __init__(self, contributed=()):
        self._contributed = tuple(contributed)

    def contributed_checks(self):
        return self._contributed

    def read_parent(self, source):  # pragma: no cover - the stub never runs
        raise NotImplementedError

    def read_construction(self, run_dir, seed):  # pragma: no cover
        raise NotImplementedError

    def write_derived(
        self, source, run_dir, new_seed, changes, **kw
    ):  # pragma: no cover
        raise NotImplementedError

    def deck_files(self, inputs):  # pragma: no cover
        raise NotImplementedError

    def make_runner(self, run_dir, **launch):  # pragma: no cover
        raise NotImplementedError

    def refusal_reason(self, run_dir, seed):  # pragma: no cover
        return None


@pytest.fixture
def clean_registry():
    before = dict(BACKENDS)
    yield
    BACKENDS.clear()
    BACKENDS.update(before)


def test_a_backend_may_add_a_check_and_the_gate_is_the_core_gate_plus_it(
    clean_registry,
):
    register_backend(_StubBackend(contributed=(_stub_check("stub_extra"),)))
    assert "stub" in BACKENDS
    gate = gate_for(get_backend("stub"))
    assert gate.checks[: len(CORE_CHECKS)] == CORE_CHECKS
    assert gate.checks[-1].check_id == "stub_extra"
    assert len(gate.checks) == len(CORE_CHECKS) + 1


def test_a_backend_may_replace_but_not_impersonate_a_core_check(clean_registry):
    # A shadowing contribution replaces the core check with a warning;
    # claiming the core origin is still refused.
    from autowan.gate import GateProfileWarning

    register_backend(
        _StubBackend(contributed=(_stub_check("band_distance_from_parent"),))
    )
    with pytest.warns(GateProfileWarning, match="replaces"):
        gate = gate_for(get_backend("stub"))
    replaced = [c for c in gate.checks if c.check_id == "band_distance_from_parent"]
    assert len(replaced) == 1 and replaced[0].origin != "core"

    BACKENDS.pop("stub")
    register_backend(
        _StubBackend(contributed=(_stub_check("looks_core", origin="core"),))
    )
    with pytest.raises(GateDefinitionError, match="must name their contributor"):
        gate_for(get_backend("stub"))


def test_registering_a_backend_under_a_taken_name_is_refused(clean_registry):
    with pytest.raises(BackendError, match="vasp"):
        register_backend(_StubBackend(), name="vasp")


def test_gate_for_appends_the_caller_extras_after_the_backend_contributions(
    clean_registry,
):
    register_backend(_StubBackend(contributed=(_stub_check("stub_extra"),)))
    mine = _stub_check("my_extra", origin="me")
    gate = gate_for(get_backend("stub"), extra=(mine,))
    assert [c.check_id for c in gate.checks[-2:]] == ["stub_extra", "my_extra"]


# ---------------------------------------------------------------------------
# the workflow takes the backend, and runs the VASP one end to end
# ---------------------------------------------------------------------------


def test_the_workflow_takes_the_backend_by_name_and_defaults_to_qe():
    from autowan.workflow import wannierize

    signature = inspect.signature(wannierize)
    assert "backend" in signature.parameters
    assert signature.parameters["backend"].default == "qe"


class _RecordingRunner(ConstructionRunner):
    """Runs nothing; lays a finished construction down under the seed it was given."""

    def __init__(self, run_dir, template):
        super().__init__(run_dir=run_dir)
        self.template = template
        self.builds = []

    def run_overlaps(self, seed):  # pragma: no cover - build is overridden
        raise AssertionError("the recording runner never computes overlaps")

    def build(self, seed, *, reuse_overlaps_from=None):
        self.builds.append((seed, reuse_overlaps_from))
        shutil.copy(self.template, self.run_dir / f"{seed}.wout")
        return [StepResult(step=f"{seed}:localise", returncode=0, seconds=0.1)]


def test_the_vasp_backend_drives_the_loop_from_a_vasprun_to_a_scored_construction(
    tmp_path,
):
    """No repair, one attempt: the parent is read from vasprun.xml, the deck from the
    .win alone, and the gate scores what the runner laid down."""
    from autowan.workflow import WorkflowConfig, wannierize

    example = (
        Path(__file__).parent.parent / "examples" / "bcc_iron" / "fe_up_manual.wout"
    )
    # The synthetic parent computes 4 bands; the parent-adequacy pre-flight blocks a
    # deck that draws on more bands than the parent carries, so the deck here asks
    # for what this parent can supply. It also asks for no projectability window: the
    # loop lays down an energy-window .wout, and that pairing would be invalid.
    (tmp_path / "si.win").write_text(
        VASP_ENERGY_WINDOW_WIN.replace("num_wann  = 8", "num_wann  = 2")
        .replace("num_bands = 16", "num_bands = 4")
        .replace("Si:s;p", "Si:s")
    )
    runner = _RecordingRunner(tmp_path, example)

    result = wannierize(
        run_dir=tmp_path,
        parent_xml=SYNTHETIC / "vasprun.xml",
        seed="si",
        runner=runner,
        provider_factory=None,
        config=WorkflowConfig(draw_figures=False),
        backend="vasp",
    )

    assert runner.builds == [("si", None)]
    assert isinstance(result.parent, ParentCalculation)
    assert result.parent.species == ("Si", "Si")
    assert len(result.run.attempts) == 1
    assert result.run.attempts[0].outcome is not None
    assert result.report is not None
    assert result.report.construction_id == "si"
    # every core check has a verdict row, whether or not it could be evaluated here
    assert {v.check_id for v in result.report.verdicts} == {
        c.check_id for c in CORE_CHECKS
    }
    assert result.run.attempts[0].parameters.get("projection_source") == (
        "analytic orbitals named in the input"
    )


def test_the_parent_contract_has_no_engine_specific_field():
    names = {f.name for f in fields(ParentCalculation)}
    assert "eigenvalues_ev" in names and "fermi_energy_ev" in names
    assert not any("qe" in n or "vasp" in n or "xml" in n for n in names)
