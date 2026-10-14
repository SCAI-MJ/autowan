#!/bin/bash
#SBATCH --job-name=autowan_vasp_fixture_scf
#SBATCH --partition=orte
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
# engine-bypass-allowed: autowan is a standalone public package that cannot import major; debugging-scale fixture SCF only; expires 2026-10-12
#
# The two parent self-consistent runs behind autowan/tests/fixtures/vasp/{Si,SrO}: one SCF per
# material on a Gamma-centred 2x2x2 mesh (seconds on a 2-atom cell). The overlap step is NOT run
# here -- that is autowan's own VASP runner (backends/vasp/execution.py) reading these products.
#
#   sbatch run_parent_scf.slurm <scratch_root> <fixture_dir>
#   e.g. sbatch run_parent_scf.slurm $AUTOWAN_WORK/vasp_smoke "$PWD"   (from this directory)
#
# Products copied back into the fixture directory afterwards are the small text files only
# (INCAR, POSCAR, KPOINTS, OUTCAR, OSZICAR, vasprun.xml, EIGENVAL, PROCAR); CHGCAR stays in the
# scratch tree and POTCAR (licensed) is never copied -- POTCAR.spec names the library entries.
set -uo pipefail
# Under sbatch BASH_SOURCE is the spool copy, so the fixture directory is an argument.
SCRATCH="${1:?usage: sbatch run_parent_scf.slurm <scratch_root> <fixture_dir>}"
FIXTURES="${2:?usage: sbatch run_parent_scf.slurm <scratch_root> <fixture_dir>}"
VASP="${VASP_STD:?set VASP_STD to vasp_std}"

# Intel MPI runtime for the production vasp_std (oneAPI 2025.3); libfabric is in its own subtree.
ONEAPI_ROOT="${ONEAPI_ROOT:?set ONEAPI_ROOT}"
export PATH="$ONEAPI_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ONEAPI_ROOT/lib:$ONEAPI_ROOT/libfabric/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER_PATH="$ONEAPI_ROOT/libfabric/lib/prov"
export OMP_NUM_THREADS=1
# A bounded stack, never unlimited (memory: project_vasp_stack_limit_blocked_davidson).
ulimit -s 262144

for M in Si SrO; do
  RUN="$SCRATCH/$M"
  mkdir -p "$RUN"
  cp "$FIXTURES/$M/INCAR" "$FIXTURES/$M/POSCAR" "$FIXTURES/$M/KPOINTS" "$RUN/"
  : > "$RUN/POTCAR"
  while read -r species path; do
    [ -z "$species" ] && continue
    cat "$path" >> "$RUN/POTCAR"
  done < "$FIXTURES/$M/POTCAR.spec"
  echo "[scf] $M host=$(hostname) binary=$VASP np=$SLURM_NTASKS start=$(date -Is)"
  ( cd "$RUN" && mpirun -np "$SLURM_NTASKS" "$VASP" > vasp.log 2>&1 )
  rc=$?
  echo "[scf] $M exit=$rc end=$(date -Is)"
  grep -c "aborting loop because EDIFF is reached" "$RUN/OUTCAR" 2>/dev/null | sed "s/^/[scf] $M EDIFF-reached lines: /"
  grep -E "E-fermi|NBANDS|NKPTS" "$RUN/OUTCAR" 2>/dev/null | head -3
done
