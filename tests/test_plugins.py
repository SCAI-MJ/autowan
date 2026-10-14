"""Contributed checks are executable code, so what the loader refuses is the contract.

Everywhere else in this workflow trust runs inward: a model's answer is data to be
validated. Here a file someone else wrote decides what "certified" means. Loading is
therefore explicit — nothing arrives by sitting in a directory — and whatever arrives
can only be added, since acceptance is the conjunction over every check.
"""

from __future__ import annotations

import textwrap

import pytest

from autowan.gate import CertificationGate, Check, CheckOutcome, GateDefinitionError
from autowan.plugins import PluginError, load, load_from_path

CONTRIBUTED = '''
from autowan.gate import Check, CheckOutcome

CHECKS = [
    Check("contributed_check", "something a group cares about", "discrete", "self",
          lambda c: CheckOutcome(True, 1.0, "unit"), origin="a_contributor"),
]
'''


def _write(tmp_path, body, name="contrib.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_a_contributed_module_is_loaded_from_a_path(tmp_path):
    checks = load_from_path(_write(tmp_path, CONTRIBUTED))
    assert len(checks) == 1
    assert checks[0].check_id == "contributed_check"
    assert checks[0].origin == "a_contributor"


def test_a_module_exposing_nothing_is_refused(tmp_path):
    with pytest.raises(PluginError, match="exposes no CHECKS"):
        load_from_path(_write(tmp_path, "SOMETHING_ELSE = []"))


def test_an_empty_contribution_is_refused(tmp_path):
    with pytest.raises(PluginError, match="empty"):
        load_from_path(_write(tmp_path, "CHECKS = []"))


def test_a_module_contributing_the_wrong_type_is_refused(tmp_path):
    with pytest.raises(PluginError, match="not a Check"):
        load_from_path(_write(tmp_path, "CHECKS = ['not a check']"))


def test_a_contribution_claiming_to_be_core_is_refused(tmp_path):
    body = CONTRIBUTED.replace('origin="a_contributor"', 'origin="core"')
    with pytest.raises(GateDefinitionError, match="provenance"):
        load_from_path(_write(tmp_path, body))


def test_two_contributions_may_not_share_an_identifier(tmp_path):
    first = _write(tmp_path, CONTRIBUTED, "one.py")
    second = _write(tmp_path, CONTRIBUTED, "two.py")
    with pytest.raises(GateDefinitionError, match="share the identifier"):
        load([str(first), str(second)])


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(PluginError, match="no such file"):
        load_from_path(tmp_path / "absent.py")


def test_a_module_that_raises_on_import_is_refused(tmp_path):
    with pytest.raises(PluginError, match="raised while being imported"):
        load_from_path(_write(tmp_path, "raise ValueError('boom')"))


def test_a_loaded_check_can_only_tighten_the_gate(tmp_path):
    """The property the whole extension point rests on."""
    core = Check("core_check", "q", "discrete", "self",
                 lambda c: CheckOutcome(True, 1.0, "unit"))
    body = CONTRIBUTED.replace("CheckOutcome(True, 1.0", "CheckOutcome(False, 1.0")
    contributed = load_from_path(_write(tmp_path, body))

    class _Parent:
        import numpy as _np
        eigenvalues_ev = _np.linspace(-5, 5, 40).reshape(1, 4, 10)
        fermi_energy_ev = 0.0
        lattice_ang = _np.eye(3) * 3.0
        alat_ang = 3.0
        positions_frac = _np.zeros((1, 3))
        kpoints_cart = _np.zeros((4, 3))

    from autowan.gate import Construction

    construction = Construction("t", parent=_Parent(), wannier=object(),
                                material_class="gapped")
    assert CertificationGate([core]).run(construction).passed
    assert not CertificationGate([core], contributed).run(construction).passed
