"""Gate profiles: named, hashed check lists.

Every check is removable and there are no tiers; dropping a default check warns and does
nothing else; a contributed check may replace a profile's check; every verdict records the
profile and the hash of the effective check list.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from autowan.cli import build_parser, list_gate_profiles, main, resolve_gate_profile
from autowan.core_checks import (
    CONVERGENCE_CHECKS,
    CORE_CHECKS,
    PRE_PROFILE_CORE_CHECKS,
    PRE_V5_CORE_CHECKS,
)
from autowan.gate import (
    CertificationGate,
    Check,
    CheckOutcome,
    Construction,
    GateDefinitionError,
    GateProfileWarning,
    check_list_hash,
)
from autowan.gate_profiles import (
    CHECK_REGISTRY,
    DEFAULT_CHECK_IDS,
    PROFILES,
    GateProfile,
    build_gate,
    custom_profile,
    get_profile,
    register_check,
)
from autowan.workflow import WorkflowConfig


def _always(result):
    return lambda c: CheckOutcome(result, 0.0, "")


def _construction():
    return Construction("c", parent=None, wannier=None, material_class="metal")


def test_the_named_profiles_are_the_default_plus_one_criterion():
    assert set(PROFILES) == {
        "default",
        "band_interpolation",
        "sparse_tb",
        "downfolded_d",
        "dmft_subspace",
        "atom_centred_basis",
    }
    assert get_profile("default").check_ids == DEFAULT_CHECK_IDS
    added = {
        "band_interpolation": "band_interpolation_eta2",
        "sparse_tb": "truncated_hopping_eta2",
        "atom_centred_basis": "centres_on_atoms_atom_centred",
    }
    for name, check_id in added.items():
        assert get_profile(name).check_ids == (*DEFAULT_CHECK_IDS, check_id)
    # downfolded_d is the d-manifold target: the default without its whole-occupied-
    # manifold checks, plus the d-only check (tests/test_downfolding.py)
    from autowan.gate_profiles import WHOLE_MANIFOLD_CHECKS

    assert get_profile("downfolded_d").check_ids == (
        *(i for i in DEFAULT_CHECK_IDS if i not in WHOLE_MANIFOLD_CHECKS),
        "d_only_downfolding",
    )
    # dmft_subspace (2026-10-12) shares that base, plus the literature-subspace check
    # (tests/test_dmft_subspace.py)
    assert get_profile("dmft_subspace").check_ids == (
        *(i for i in DEFAULT_CHECK_IDS if i not in WHOLE_MANIFOLD_CHECKS),
        "dmft_subspace_model_window",
    )


def test_atom_centring_left_the_default_and_the_drift_guard_joined_it():
    assert "centres_sit_on_atomic_sites" not in DEFAULT_CHECK_IDS
    assert "centres_on_atoms_atom_centred" not in DEFAULT_CHECK_IDS
    assert "centres_stay_near_home_cell" in DEFAULT_CHECK_IDS
    assert "centres_sit_on_atomic_sites" in CHECK_REGISTRY  # still buildable
    # the pre-profile set is the pre-v5 default with the old check in the guard's position
    old = [c.check_id for c in PRE_PROFILE_CORE_CHECKS]
    new = [c.check_id for c in PRE_V5_CORE_CHECKS]
    assert len(old) == len(new)
    assert [o for o, n in zip(old, new) if o != n] == ["centres_sit_on_atomic_sites"]


CONVERGENCE_IDS = ("subspace_selection_converged", "spread_minimisation_converged")


def test_convergence_left_the_default_and_every_named_profile():
    assert tuple(c.check_id for c in CONVERGENCE_CHECKS) == CONVERGENCE_IDS
    for name in PROFILES:
        assert not set(CONVERGENCE_IDS) & set(get_profile(name).check_ids), name
    # still registered, so a profile can name them one by one
    assert set(CONVERGENCE_IDS) <= set(CHECK_REGISTRY)
    # every named criterion profile is the new default plus its criterion
    for name in ("band_interpolation", "sparse_tb"):
        assert (
            get_profile(name).check_ids[: len(DEFAULT_CHECK_IDS)] == DEFAULT_CHECK_IDS
        )


def test_the_convergence_group_rebuilds_the_pre_v5_default():
    profile = custom_profile("default", add=["convergence"])
    assert profile.name == "default+convergence"
    assert set(profile.check_ids) == set(DEFAULT_CHECK_IDS) | set(CONVERGENCE_IDS)
    # the same check-list hash the v1-v4 default recorded with its verdicts
    assert profile.hash == check_list_hash(PRE_V5_CORE_CHECKS)
    assert profile.hash != get_profile("default").hash
    criterion = custom_profile("band_interpolation", add=["convergence"])
    assert criterion.name == "band_interpolation+convergence"
    assert criterion.hash == check_list_hash(
        (*PRE_V5_CORE_CHECKS, CHECK_REGISTRY["band_interpolation_eta2"])
    )


def test_the_group_drops_together_and_one_member_adds_alone():
    with_both = custom_profile("default", add=["convergence"])
    assert custom_profile(with_both, drop=["convergence"]).hash == (
        get_profile("default").hash
    )
    one = custom_profile("default", add=["spread_minimisation_converged"])
    assert "spread_minimisation_converged" in one.check_ids
    assert "subspace_selection_converged" not in one.check_ids
    with pytest.raises(GateDefinitionError):
        custom_profile("default", drop=["convergence"])  # the default does not run it
    with pytest.raises(GateDefinitionError):
        custom_profile(with_both, add=["convergence"])  # already there


def test_the_hash_is_order_independent_and_moves_with_any_change():
    checks = get_profile("default").checks
    assert check_list_hash(checks) == check_list_hash(tuple(reversed(checks)))
    assert get_profile("default").hash != get_profile("band_interpolation").hash
    assert len({get_profile(n).hash for n in PROFILES}) == len(PROFILES)
    assert get_profile("default").hash == CertificationGate(CORE_CHECKS).profile_hash


def test_every_verdict_records_the_profile_name_and_hash():
    gate = build_gate("qe", "band_interpolation")
    assert gate.profile_name == "band_interpolation"
    assert gate.profile_hash == get_profile("band_interpolation").hash
    core = Check("only", "q", "discrete", "self", _always(True))
    report = GateProfile("mine", (core,)).gate().run(_construction())
    assert report.profile == "mine"
    assert report.profile_hash == check_list_hash((core,))
    assert all(
        (v.profile, v.profile_hash) == ("mine", report.profile_hash)
        for v in report.verdicts
    )


def test_a_gate_built_without_a_profile_names_the_default_or_custom():
    assert CertificationGate(CORE_CHECKS).profile_name == "default"
    assert CertificationGate(CORE_CHECKS[:3]).profile_name == "custom"


def test_dropping_a_default_check_warns_and_does_nothing_else():
    with pytest.warns(GateProfileWarning, match="hopping_decays_with_distance"):
        profile = custom_profile("default", drop=["hopping_decays_with_distance"])
    assert "hopping_decays_with_distance" not in profile.check_ids
    assert profile.base == "default"
    assert profile.dropped == ("hopping_decays_with_distance",)
    assert profile.hash != get_profile("default").hash
    profile.gate()  # builds; nothing is refused


def test_every_check_is_removable_including_the_whole_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        empty = custom_profile("default", drop=DEFAULT_CHECK_IDS)
    assert len(caught) == len(DEFAULT_CHECK_IDS)
    assert empty.checks == ()
    # nothing evaluated is still not a pass
    assert not empty.gate().run(_construction()).passed


def test_dropping_a_criterion_is_silent_and_adding_one_is_allowed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = custom_profile("band_interpolation", drop=["band_interpolation_eta2"])
        both = custom_profile(
            "default", add=["band_interpolation_eta2", "conduction_band_eta"]
        )
    assert back.check_ids == DEFAULT_CHECK_IDS
    assert both.name == "default+band_interpolation_eta2+conduction_band_eta"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"add": ["no_such_check"]}, "unknown check"),
        ({"add": ["occupied_manifold_is_represented"]}, "already runs it"),
        ({"drop": ["band_interpolation_eta2"]}, "does not run it"),
    ],
)
def test_a_profile_edit_that_would_not_be_the_profile_that_runs_is_refused(
    kwargs, match
):
    with pytest.raises(GateDefinitionError, match=match):
        custom_profile("default", **kwargs)


def test_an_unknown_profile_is_refused():
    with pytest.raises(GateDefinitionError, match="unknown gate profile"):
        get_profile("lenient")


def test_the_registry_is_open():
    extra = Check("my_criterion", "q", "discrete", "self", _always(True), origin="me")
    register_check(extra)
    try:
        assert (
            "my_criterion" in custom_profile("default", add=["my_criterion"]).check_ids
        )
        with pytest.raises(GateDefinitionError, match="already registered"):
            register_check(extra)
    finally:
        CHECK_REGISTRY.pop("my_criterion")


def test_target_text_is_one_paragraph_naming_what_passing_means():
    for name in PROFILES:
        profile = get_profile(name)
        text = profile.target_text()
        assert "\n" not in text
        assert f'"{name}"' in text
        for check in profile.checks:
            assert check.description in text
    assert "eta_2 <= 20 meV" in get_profile("band_interpolation").target_text()
    with pytest.warns(GateProfileWarning):
        custom = custom_profile("sparse_tb", drop=["hopping_decays_with_distance"])
    assert "drops hopping_decays_with_distance" in custom.target_text()


def test_the_workflow_reads_its_profile_from_the_config():
    assert WorkflowConfig().gate_profile == "default"
    assert WorkflowConfig(gate_profile="sparse_tb").gate_profile == "sparse_tb"


def test_the_cli_resolves_a_custom_profile_and_prints_the_drop_warning(capsys):
    args = build_parser().parse_args(
        [
            "--run-dir", ".", "--parent-xml", "x.xml", "--seed", "s",
            "--gate-profile", "downfolded_d",
            "--add-check", "centres_on_atoms_atom_centred",
            "--drop-check", "hopping_decays_with_distance",
        ]
    )  # fmt: skip
    profile = resolve_gate_profile(args)
    assert profile.base == "downfolded_d"
    assert "centres_on_atoms_atom_centred" in profile.check_ids
    assert "warning: dropping 'hopping_decays_with_distance'" in capsys.readouterr().err


def test_the_cli_lists_the_profiles(capsys):
    assert main(["--list-gate-profiles"]) == 0
    out = capsys.readouterr().out
    for name in PROFILES:
        assert f"{name}  [hash {get_profile(name).hash}]" in out


def test_the_cli_refuses_an_unregistered_check(capsys):
    code = main(
        [
            "--run-dir",
            ".",
            "--parent-xml",
            "x.xml",
            "--seed",
            "s",
            "--add-check",
            "nope",
        ]
    )
    assert code == 2 and "unknown check" in capsys.readouterr().err


def test_list_gate_profiles_mentions_the_optional_checks():
    stream = SimpleNamespace(lines=[])
    stream.write = stream.lines.append
    list_gate_profiles(stream)
    text = "".join(stream.lines)
    assert "centres_sit_on_atomic_sites" in text and "truncated_hopping_eta2" in text


def test_the_cli_reference_path_reaches_the_workflow_and_a_bad_one_stops_early(
    monkeypatch, tmp_path
):
    """``--reference-path`` is read once, before any engine runs, and handed on."""
    import numpy as np

    from autowan import cli
    from autowan.criteria import ReferenceBandPath

    seen = {}

    class _Launched(Exception):
        pass

    def _record(**kwargs):
        seen.update(kwargs)
        raise _Launched

    monkeypatch.setattr(cli, "wannierize", _record)
    npz = tmp_path / "dft_bands_path.npz"
    np.savez(npz, bands=np.zeros((3, 4)), kpoints=np.zeros((3, 3)))
    (tmp_path / "scf.out").write_text("     the Fermi energy is     6.5717 ev\n")
    base = [
        "--run-dir",
        str(tmp_path),
        "--parent-xml",
        str(tmp_path / "x.xml"),
        "--seed",
        "Si",
        "--no-repair",
        "--no-figures",
        "--gate-profile",
        "band_interpolation",
    ]
    with pytest.raises(_Launched):
        main(base + ["--reference-path", str(npz)])
    reference = seen["config"].reference_path
    assert isinstance(reference, ReferenceBandPath)
    assert reference.fermi_energy_ev == pytest.approx(6.5717)
    assert main(base + ["--reference-path", str(tmp_path / "absent.npz")]) == 2
    assert build_parser().parse_args(base).reference_path is None


def test_the_default_target_text_and_the_policy_do_not_require_convergence():
    from autowan.prompts import GENERIC_PRINCIPLES

    assert "convergence" not in get_profile("default").target_text()
    assert (
        "convergence criterion"
        in custom_profile("default", add=["convergence"]).target_text()
    )
    assert "is not, by default, something the gate judges" in " ".join(
        GENERIC_PRINCIPLES.split()
    )
