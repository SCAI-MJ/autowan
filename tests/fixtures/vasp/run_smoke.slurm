#!/bin/bash
#SBATCH --job-name=autowan_vasp_backend_smoke
# Add an #SBATCH --partition line for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
# engine-bypass-allowed: autowan is a standalone public package that cannot import major; debugging-scale fixture SCF only; expires 2026-10-12
#
# The VASP backend smoke behind autowan/data/results/vasp_backend_smoke/<material>/: for each
# fixture parent (the SCFs of run_parent_scf.slurm), autowan's own VASP runner performs the
# vasp2wannier90 overlap step (one non-self-consistent VASP call from the parent's CHGCAR),
# Wannier90 localises, and the gate scores the construction through `autowan --backend vasp
# --no-repair`. Also runs the real-binary execution test once, on the compute node.
#
#   sbatch run_smoke.slurm <worktree> <scratch_root>
#   scratch_root/<material>/ holds the finished parent (CHGCAR, POTCAR, vasprun.xml, ...).
set -uo pipefail
WORKTREE="${1:?usage: sbatch run_smoke.slurm <worktree> <scratch_root>}"
SCRATCH="${2:?usage: sbatch run_smoke.slurm <worktree> <scratch_root>}"
PKG="$WORKTREE/autowan"
PY="$PKG/.venv/bin/python"
# A vasp_std built with LWANNIER90. A -xHOST build (e.g. AVX-512) dies at start on older
# CPUs with "forrtl: severe (168): illegal instruction": run it on the nodes it was built for.
VASP="${VASP_W90:?set VASP_W90 to a vasp_std built with LWANNIER90}"
# Wannier90 develop: the only build that knows the projectability keywords (dis_proj_min/max).
W90="${WANNIER90:?set WANNIER90 to wannier90.x}"
NP="${SLURM_NTASKS:-4}"

# The -w90 build links the oneapi_2022 MKL and Intel MPI; setvars.sh trips `set -u`, so it
# is sourced before strict mode.
set +u
[ -n "${ONEAPI_SETVARS:-}" ] && source "$ONEAPI_SETVARS" --force > /dev/null
set -u
export OMP_NUM_THREADS=1
ulimit -s 262144   # bounded, never unlimited (memory: project_vasp_stack_limit_blocked_davidson)

echo "[smoke] host=$(hostname) vasp=$VASP w90=$W90 np=$NP start=$(date -Is)"
echo "[smoke] git=$(git -C "$WORKTREE" rev-parse --short HEAD) branch=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)"

# 1. the real-binary execution test, skipped everywhere else by name
cd "$PKG"
AUTOWAN_VASP="$VASP" AUTOWAN_VASP_PARENT_DIR="$SCRATCH/Si" AUTOWAN_VASP_SMOKE=1 \
AUTOWAN_MPI="mpirun -np $NP" \
  "$PY" -m pytest -q tests/test_vasp_execution.py -k real_binary -rs 2>&1 | tail -8

# 2. score both fixture materials through the command line, VASP backend, no repair
for M in Si SrO; do
  RUN="$SCRATCH/$M"                      # the deck lives beside the parent it came from
  OUT="$PKG/data/results/vasp_backend_smoke/$M"
  mkdir -p "$OUT"
  "$PY" - <<EOF
from autowan.backends.vasp.inputs import deck_from_parent
inputs = deck_from_parent("$RUN/vasprun.xml", "$RUN", "$M")
print("[deck] $M num_wann", inputs.values.get("num_wann"), "num_bands", inputs.values.get("num_bands"),
      "projectors", inputs.values.get("projectors"), "mp_grid", inputs.values.get("mp_grid"))
EOF
  cp "$RUN/$M.win" "$OUT/$M.win"
  echo "[smoke] $M autowan --backend vasp --no-repair start=$(date -Is)"
  ( cd "$RUN" && "$PY" -m autowan.cli --backend vasp --no-repair \
      --run-dir "$RUN" --parent-xml "$RUN/vasprun.xml" --seed "$M" \
      --vasp "$VASP" --wannier90 "$W90" --mpi "mpirun -np $NP" --no-figures \
      --log "$OUT/run_record.json" --verdicts "$OUT/gate_verdicts.json" --material "$M" ) \
      > "$OUT/autowan.out" 2>&1
  echo "[smoke] $M exit=$? end=$(date -Is)"
  tail -20 "$OUT/autowan.out"
  for f in "$M.wout" "${M}_hr.dat" "$M.eig" "$M.werr"; do
    [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/$f"
  done
  ls -la "$RUN/$M.vasp2wannier90" 2>/dev/null | head -20
  cp "$RUN/$M.vasp2wannier90/OUTCAR" "$OUT/overlap_step_OUTCAR" 2>/dev/null
  cp "$RUN/$M.vasp2wannier90/vasp.log" "$OUT/overlap_step_vasp.log" 2>/dev/null
  cp "$RUN/$M.vasp2wannier90/wannier90.win" "$OUT/overlap_step_wannier90.win" 2>/dev/null
done
echo "[smoke] end=$(date -Is)"
