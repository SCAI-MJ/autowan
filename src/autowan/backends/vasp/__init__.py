"""Connect VASP's files and runner to the engine-independent workflow contract."""

from dataclasses import fields

from ..protocol import Backend, BackendError
from . import inputs, outputs
from .execution import VaspWannierRunner, refusal_reason
from .knowledge import VASP_KNOWLEDGE, VaspAdequacyPartition


class VaspBackend(Backend):
    name = "vasp"
    knowledge = VASP_KNOWLEDGE
    adequacy = VaspAdequacyPartition()

    def read_parent(self, source):
        return outputs.read_parent(source)

    def read_construction(self, run_dir, seed):
        return inputs.read_construction(run_dir, seed)

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
            run_dir,
            new_seed,
            changes,
            fermi_energy_ev=fermi_energy_ev,
            species_counts=species_counts,
        )

    def deck_files(self, inputs):
        return (inputs.win,)

    def make_runner(self, run_dir, **launch):
        known = {f.name for f in fields(VaspWannierRunner)} - {"run_dir"}
        for option in launch:
            if option not in known:
                raise BackendError(f"VASP does not know launch option {option!r}")
        return VaspWannierRunner(run_dir=run_dir, **launch)

    def refusal_reason(self, run_dir, seed):
        return refusal_reason(run_dir, seed)
