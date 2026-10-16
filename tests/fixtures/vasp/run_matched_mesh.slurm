#!/bin/bash
#SBATCH --job-name=autowan_vasp_matched_mesh
# Add an #SBATCH --partition line for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
# engine-bypass-allowed: autowan is a standalone public package that cannot import major; the VASP parent SCF and overlap step at the QE campaign mesh for the VASP-vs-QE gate-verdict comparison (SrO, Si) only; expires 2026-10-15
#
# The VASP side of autowan/data/results/vasp_all_channel_deck_matched_mesh/<material>/.
# For each material, on the k-mesh its Quantum ESPRESSO campaign construction ran on:
#   1. one VASP SCF (Gamma-centred mesh read from the QE deck's mp_grid, ISYM = -1,
#      NBANDS = the QE construction's num_bands, MPI ranks chosen to divide NBANDS --
#      VASP silently rounds NBANDS up to a multiple of the band-parallel rank count;
#      LMAXMIX = 4 when a species carries a d projector, so the ICHARG = 11 overlap step
#      reads the one-centre densities the SCF had);
#   2. the automatic VASP first deck (deck_from_parent: every POTCAR channel, frozen
#      window 2 eV above the parent's highest occupied state, capped by the band count);
#   3. autowan --backend vasp --no-repair: autowan's own VaspWannierRunner performs the
#      vasp2wannier90 overlap step (one non-self-consistent VASP call from the SCF
#      density), Wannier90 develop localises, the gate scores the construction;
#   4. provenance.json is completed from what VASP actually ran: the mesh from the
#      overlap step's own vasprun.xml, cross-checked against the .amn header, and
#      compared with the QE deck's mp_grid (the Wannier90 library ignores the .win mesh).
# Sub-analyses, labelled and outside the contract paths (Rule 38 audit, 2026-09-15):
#   - vasp_old_deck_same_mesh/: the superseded automatic deck (channels occupied below
#     E_F + 2 eV, dis_froz_max = E_F + 2 eV) on the SAME parent -- separates the deck change
#     from the 2x2x2 -> 6x6x6 mesh change;
#   - qe_control_13_hydrogenic/ (SrO): the QE campaign's 13-function hydrogenic overlaps
#     (Sr:s;p;d O:s;p, from attempt 4) re-localised unrepaired with the first deck's
#     frozen rule (highest occupied + 2 eV) -- the same representation as the VASP deck,
#     on the QE engine;
#   - an SCF at 1.3 x ENCUT (cutoff evidence) and diagnostics.json (SCF/overlap-step
#     eigenvalue hand-off, k-point sets, frozen-state counts per k, A-matrix singular
#     values and occupied capture, per-function centres, and the VASP construction
#     re-scored with its E_F placed at the valence maximum, QE's convention).
# Then, report-only: VASP 6.4's LOCPROJ_AUTO route (LPRJ_FROM_PSEUDO) on the SrO parent.
#
#   sbatch run_matched_mesh.slurm <worktree> <scratch_root> [material ...]
#   SMOKE=1 sbatch run_matched_mesh.slurm <worktree> <scratch_root>   # launch-context preflight only
#
# A -xHOST (AVX-512) vasp_std dies at start with "illegal instruction" on older CPUs.
# A portable build of the same source (-march=core-avx2, no Intel-vendor check) runs
# elsewhere: sbatch -p <partition> --export=ALL,VASP_W90=<that vasp_std>,
# REQUIRE_AVX512=0,VASP_BUILD=<note> ... The binary, node and CPU land in provenance.json.
# CHGCAR, WAVECAR and POTCAR stay in <scratch_root>; only small text products are copied.
set -uo pipefail
WORKTREE="${1:?usage: sbatch run_matched_mesh.slurm <worktree> <scratch_root> [material ...]}"
SCRATCH="${2:?usage: sbatch run_matched_mesh.slurm <worktree> <scratch_root> [material ...]}"
shift 2
MATERIALS=("$@")
[ "${#MATERIALS[@]}" -eq 0 ] && MATERIALS=(Si SrO)

PKG="$WORKTREE/autowan"
PY="$PKG/.venv/bin/python"
FIX="$PKG/tests/fixtures/vasp"
RESULTS="$PKG/data/results/vasp_all_channel_deck_matched_mesh"
VASP="${VASP_W90:?set VASP_W90 to a vasp_std built with LWANNIER90}"
W90="${WANNIER90:?set WANNIER90 to wannier90.x}"
TASKS="${SLURM_NTASKS:-4}"
ENCUT_FACTOR=1.3

# The QE campaign construction each VASP run is compared with (the automatic first deck,
# attempt 0 of each campaign -- the stage-matched counterpart of a --no-repair VASP run).
declare -A QE_DIR=(
  [Si]=$AUTOWAN_WORK/campaign/Si
  [SrO]=$AUTOWAN_WORK/campaign2/SrO
)
declare -A QE_SEED=([Si]=Si_pdwf [SrO]=SrO_pdwf)

# The -w90 build links the oneapi_2022 MKL and Intel MPI; setvars.sh trips `set -u`.
set +u
[ -n "${ONEAPI_SETVARS:-}" ] && source "$ONEAPI_SETVARS" --force > /dev/null
set -u
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
ulimit -s 262144   # bounded, never unlimited (memory: project_vasp_stack_limit_blocked_davidson)

GIT_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
GIT_BRANCH="$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)"
W90_REV="$(git -C "$(dirname "$W90")" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[matched] host=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') start=$(date -Is)"
echo "[matched] vasp=$VASP w90=$W90 (git $W90_REV) tasks=$TASKS git=$GIT_SHA branch=$GIT_BRANCH"
echo "[matched] mpi launch: I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-default} I_MPI_HYDRA_TOPOLIB=${I_MPI_HYDRA_TOPOLIB:-default}"

win_value() {  # win_value <file.win> <key> -> the value of `key = value`, comments stripped
  sed -e 's/[!#].*//' "$1" | awk -F'[=:]' -v k="$2" 'tolower($1) ~ "^[ \t]*"k"[ \t]*$" {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}'
}

ranks_for() {  # ranks_for <nbands> -> the largest rank count <= TASKS that divides nbands
  local nb="$1" n
  for ((n = TASKS; n >= 1; n--)); do
    if (( nb % n == 0 )); then echo "$n"; return; fi
  done
}

stage_parent() {  # stage_parent <material> <run_dir> <mesh> <nbands> [encut]
  local m="$1" run="$2" mesh="$3" nb="$4" encut="${5:-}"
  mkdir -p "$run"
  cp "$FIX/$m/POSCAR" "$run/POSCAR"
  : > "$run/POTCAR"
  while read -r species path; do
    [ -z "$species" ] && continue
    cat "$path" >> "$run/POTCAR"
  done < "$FIX/$m/POTCAR.spec"
  sed -e "s/^SYSTEM .*/SYSTEM = $m, autowan VASP all-channel first deck at the QE campaign mesh ($mesh)/" \
      -e "s/^NBANDS .*/NBANDS = $nb      ! the QE construction's num_bands; divisible by the rank count/" \
      "$FIX/$m/INCAR" > "$run/INCAR"
  if [ -n "$encut" ]; then
    sed -i "s/^ENCUT .*/ENCUT  = $encut   ! cutoff check: $ENCUT_FACTOR x the production cutoff/" "$run/INCAR"
  fi
  # LMAXMIX = 4 when any species carries a d projector (VASP: d elements), 2 otherwise
  local lmaxmix
  lmaxmix="$("$PY" -c "from autowan.backends.vasp.outputs import potcar_channels as c; print(4 if any('d' in v for v in c('$run/POTCAR').values()) else 2)")"
  printf 'LMAXMIX = %s     ! one-centre densities kept to l = %s for the ICHARG = 11 overlap step\n' \
      "$lmaxmix" "$lmaxmix" >> "$run/INCAR"
  printf 'Gamma-centred %s, the mesh of the QE campaign construction %s\n0\nGamma\n %s\n 0 0 0\n' \
      "$mesh" "${QE_SEED[$m]}" "$mesh" > "$run/KPOINTS"
}

run_scf() {  # run_scf <label> <run_dir> <ranks> -> 0 when EDIFF was reached with the NBANDS asked for
  local label="$1" run="$2" np="$3" t0 t1 rc nb_ask nb_run
  t0=$(date +%s)
  echo "[scf] $label start=$(date -Is) ranks=$np"
  ( cd "$run" && mpirun -np "$np" "$VASP" > vasp.log 2>&1 )
  rc=$?
  t1=$(date +%s)
  echo "[scf] $label exit=$rc seconds=$((t1 - t0)) end=$(date -Is)"
  echo "$((t1 - t0))" > "$run/scf_seconds"
  grep -E "E-fermi|NBANDS=|NKPTS" "$run/OUTCAR" 2>/dev/null | head -3
  if ! grep -q "aborting loop because EDIFF is reached" "$run/OUTCAR" 2>/dev/null; then
    echo "[scf] $label did NOT reach EDIFF"
    tail -20 "$run/vasp.log"
    return 1
  fi
  nb_ask="$(sed -n 's/^NBANDS *= *\([0-9]*\).*/\1/p' "$run/INCAR")"
  nb_run="$(sed -n 's/.*NBANDS= *\([0-9]*\).*/\1/p' "$run/OUTCAR" | head -1)"
  if [ "$nb_run" != "$nb_ask" ]; then
    echo "[scf] $label VASP ran NBANDS=$nb_run, not $nb_ask (rank rounding)"
    return 1
  fi
  return 0
}

score_vasp() {  # score_vasp <material> <run_dir> <seed> <out_dir> <ranks>
  local m="$1" run="$2" seed="$3" out="$4" np="$5"
  mkdir -p "$out"
  cp "$run/$seed.win" "$out/$seed.win"
  echo "[matched] $m $seed autowan --backend vasp --no-repair start=$(date -Is)"
  ( cd "$run" && "$PY" -m autowan.cli --backend vasp --no-repair \
      --run-dir "$run" --parent-xml "$run/vasprun.xml" --seed "$seed" \
      --vasp "$VASP" --wannier90 "$W90" --mpi "mpirun -np $np" --no-figures \
      --log "$out/run_record.json" --verdicts "$out/gate_verdicts.json" --material "$m" ) \
      > "$out/autowan.out" 2>&1
  echo "[matched] $m $seed exit=$? end=$(date -Is)"
  tail -14 "$out/autowan.out"
  for f in "$seed.wout" "${seed}_hr.dat" "$seed.eig" "$seed.werr"; do
    [ -f "$run/$f" ] && cp "$run/$f" "$out/$f"
  done
  local stage="$run/$seed.vasp2wannier90"
  cp "$stage/INCAR" "$out/overlap_step_INCAR" 2>/dev/null
  cp "$stage/vasp.log" "$out/overlap_step_vasp.log" 2>/dev/null
  cp "$stage/wannier90.win" "$out/overlap_step_wannier90.win" 2>/dev/null
}

# ---------------------------------------------------------------------------------------
# launch-context preflight (Rule 41): everything the production run binds at launch,
# no engine call
# ---------------------------------------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
  rc=0
  [ -x "$VASP" ] || { echo "[smoke] missing VASP $VASP"; rc=1; }
  [ -x "$W90" ] || { echo "[smoke] missing Wannier90 $W90"; rc=1; }
  if [ "${REQUIRE_AVX512:-1}" = "1" ]; then
    lscpu | grep -q avx512 || { echo "[smoke] no AVX-512 on $(hostname): the -xHOST build will not start"; rc=1; }
  else
    lscpu | grep -q avx2 || { echo "[smoke] no AVX2 on $(hostname): the portable build will not start"; rc=1; }
  fi
  ( cd "$(mktemp -d)" && timeout 60 "$VASP" < /dev/null > vasp.probe 2>&1; grep -q -i "illegal instruction\|not built to run" vasp.probe && { echo "[smoke] $VASP does not start on $(hostname)"; exit 1; }; echo "[smoke] $VASP starts on $(hostname): $(head -2 vasp.probe | tr '\n' ' ')" ) || rc=1
  command -v mpirun > /dev/null || { echo "[smoke] no mpirun on PATH"; rc=1; }
  # A real MPI launch, not a singleton start: on one partition the hydra proxy segfaulted
  # at launch while a singleton VASP start passed the preflight.
  n_up="$(timeout 60 mpirun -np 2 /bin/hostname 2>/dev/null | grep -c "$(hostname)")"
  [ "$n_up" = "2" ] && echo "[smoke] mpirun -np 2 launches on $(hostname)" \
    || { echo "[smoke] mpirun -np 2 did not launch on $(hostname) (got $n_up of 2)"; rc=1; }
  "$PY" -c "import autowan.backends.vasp.inputs as i, autowan.backends.vasp.outputs as o; print('[smoke] imports', i.frozen_window_top.__name__, o.potcar_channels.__name__, o.read_kmesh.__name__)" || rc=1
  "$PY" -m autowan.cli --help > /dev/null || { echo "[smoke] autowan.cli --help failed"; rc=1; }
  "$PY" "$FIX/score_qe_reference.py" --help > /dev/null || { echo "[smoke] score_qe_reference --help failed"; rc=1; }
  for M in "${MATERIALS[@]}"; do
    QE_WIN="${QE_DIR[$M]}/${QE_SEED[$M]}.win"
    MESH="$(win_value "$QE_WIN" mp_grid)"; NB="$(win_value "$QE_WIN" num_bands)"
    NP="$(ranks_for "$NB")"
    TMP="$(mktemp -d)"
    stage_parent "$M" "$TMP" "$MESH" "$NB"
    "$PY" -c "from autowan.backends.vasp.outputs import potcar_channels; print('[smoke] $M channels', potcar_channels('$TMP/POTCAR'))" || rc=1
    echo "[smoke] $M mesh='$MESH' nbands=$NB ranks=$NP kpoints='$(sed -n 4p "$TMP/KPOINTS")' $(grep -E '^(LMAXMIX|NBANDS)' "$TMP/INCAR" | tr '\n' ' ')"
    [ -n "$MESH" ] && [ -n "$NB" ] && [ -n "$NP" ] || rc=1
    rm -rf "$TMP"
  done
  [ -f $AUTOWAN_WORK/campaign2/SrO/SrO_pdwf_r4.amn ] || { echo "[smoke] missing QE r4 overlaps"; rc=1; }
  [ "$rc" -eq 0 ] && echo "SMOKE OK" || echo "SMOKE FAILED"
  exit "$rc"
fi

# ---------------------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------------------
cd "$PKG"

for M in "${MATERIALS[@]}"; do
  QE_WIN="${QE_DIR[$M]}/${QE_SEED[$M]}.win"
  MESH="$(win_value "$QE_WIN" mp_grid)"
  NB="$(win_value "$QE_WIN" num_bands)"
  NP="$(ranks_for "$NB")"
  RUN="$SCRATCH/$M"
  OUT="$RESULTS/$M"
  mkdir -p "$OUT"
  echo "[matched] $M QE deck $QE_WIN: mp_grid='$MESH' num_bands=$NB -> ranks=$NP"

  # 1. the parent SCF on the QE campaign mesh
  stage_parent "$M" "$RUN" "$MESH" "$NB"
  run_scf "$M" "$RUN" "$NP" || { echo "[matched] $M: no construction is built on this parent"; continue; }
  for f in INCAR KPOINTS OSZICAR vasp.log; do cp "$RUN/$f" "$OUT/parent_$f"; done

  # 2. the automatic first deck
  "$PY" - <<EOF
from autowan.backends.vasp.inputs import deck_from_parent
v = deck_from_parent("$RUN/vasprun.xml", "$RUN", "$M").values
print("[deck] $M num_wann", v.get("num_wann"), "num_bands", v.get("num_bands"),
      "projectors", v.get("projectors"), "dis_froz_max", v.get("dis_froz_max"), "mp_grid", v.get("mp_grid"))
EOF

  # 3. overlap step + localisation + gate, one attempt, no repair
  score_vasp "$M" "$RUN" "$M" "$OUT" "$NP"

  # 4. provenance from what VASP ran
  "$PY" - <<EOF
import json, pathlib
import numpy as np
from autowan.backends.vasp.inputs import frozen_window_top, FROZEN_WINDOW_ABOVE_OCCUPIED_EV
from autowan.backends.vasp.outputs import read_kmesh, read_parent, potcar_channels

out = pathlib.Path("$OUT"); run = pathlib.Path("$RUN"); stage = run / "$M.vasp2wannier90"
prov_path = out / "provenance.json"
prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
ran = read_kmesh(stage / "vasprun.xml")          # the overlap step VASP actually ran
parent_mesh = read_kmesh(run / "vasprun.xml")    # the SCF
header = None
amn = run / "$M.amn"
if amn.is_file():
    header = tuple(int(x) for x in amn.read_text().splitlines()[1].split()[:3])
qe_mesh = " ".join("$MESH".split())
measured = " ".join(map(str, ran))
problems = []
if ran != parent_mesh: problems.append(f"overlap step mesh {ran} != SCF mesh {parent_mesh}")
if header and header[1] != ran[0] * ran[1] * ran[2]: problems.append(f".amn k-count {header[1]} != {ran}")
if measured != qe_mesh: problems.append(f"VASP mesh '{measured}' != QE mesh '{qe_mesh}'")
parent = read_parent(run / "vasprun.xml")
nw = next(int(l.split("=")[1]) for l in (run / "$M.win").read_text().splitlines()
          if l.strip().lower().startswith("num_wann"))
bands = parent.eigenvalues_ev
occupied = float(bands[bands <= parent.fermi_energy_ev].max())
cap = float(np.sort(bands, axis=-1)[..., nw].min()) if parent.n_bands > nw else None
prov.update({
    "mp_grid": measured,
    "mp_grid_source": "kpoints/generation divisions of the overlap step's own vasprun.xml (what VASP ran), "
                      "equal to the SCF's; k-count cross-checked against the .amn header",
    "mp_grid_qe_reference": qe_mesh,
    "mp_grid_matches_qe_reference": measured == qe_mesh,
    "amn_header_bands_kpoints_functions": list(header) if header else None,
    "mesh_problems": problems,
    "num_bands": int("$NB"),
    "num_bands_rule": "NBANDS = the QE construction's num_bands ($NB, ${QE_SEED[$M]}.win), so the band pool the "
                      "disentanglement draws from matches across engines; MPI ranks ($NP) chosen to divide it "
                      "(VASP rounds NBANDS up to a multiple of the band-parallel rank count)",
    "mpi_ranks": int("$NP"),
    "num_wann": nw,
    "channel_rule": "every angular-momentum channel each species' POTCAR carries a non-local projector for "
                    "(the TYP 7 local channel excluded); Sr semicore 4s4p kept (Sr_sv), matching the semicore "
                    "policy of the QE constructions compared",
    "channels": {k: list(v) for k, v in potcar_channels(run / "POTCAR").items()},
    "potcars": [l.split()[1] for l in pathlib.Path("$FIX/$M/POTCAR.spec").read_text().splitlines() if l.strip()],
    "frozen_window_rule": f"dis_froz_max = highest occupied parent state + {FROZEN_WINDOW_ABOVE_OCCUPIED_EV} eV "
                          "(E_F + that for a metal), capped just below the lowest energy at which any k-point would "
                          "hold more than num_wann states; the QE first deck uses its highest occupied level + 2 eV",
    "fermi_energy_ev": parent.fermi_energy_ev,
    "highest_occupied_ev": occupied,
    "frozen_window_cap_ev": cap,
    "frozen_window_cap_binding": bool(cap is not None and occupied + FROZEN_WINDOW_ABOVE_OCCUPIED_EV >= cap),
    "dis_froz_max": frozen_window_top(parent, nw),
    "parent_scf_seconds": int((run / "scf_seconds").read_text()),
    "vasp_binary": "$VASP",
    "vasp_build": "${VASP_BUILD:--xHOST build (AVX-512)}",
    "compute_node": "$(hostname)",
    "mpi_launch": "I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-default} I_MPI_HYDRA_TOPOLIB=${I_MPI_HYDRA_TOPOLIB:-default}",
    "cpu_model": "$(lscpu | sed -n 's/^Model name: *//p')",
    "wannier90": "$W90 (git $W90_REV)",
    "autowan_git": "$GIT_SHA",
    "autowan_branch": "$GIT_BRANCH",
    "slurm_job": "${SLURM_JOB_ID:-}",
    "qe_reference_seed": "${QE_SEED[$M]}",
    "incar_parent": (run / "INCAR").read_text(),
})
prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
print("[provenance] $M mp_grid", measured, "qe", qe_mesh, "problems", problems or "none")
EOF

  # 5. sub-analysis: the superseded automatic deck on the same parent
  "$PY" - <<EOF
import re
from autowan.backends.vasp.inputs import deck_from_parent
from autowan.backends.vasp.outputs import projector_supply, read_parent, read_procar
parent = read_parent("$RUN/vasprun.xml")
supply = projector_supply(read_procar("$RUN/PROCAR"), parent.species, parent.fermi_energy_ev)
projectors = [{"species": s, "orbitals": o} for s, o in supply.items()]
deck_from_parent("$RUN/vasprun.xml", "$RUN", "${M}_old_deck", projectors=projectors)
win = open("$RUN/${M}_old_deck.win").read()
old = parent.fermi_energy_ev + 2.0
win = re.sub(r"(?m)^! dis_froz_max:.*\n", "", win)
win = re.sub(r"(?m)^dis_froz_max = .*$",
             f"! dis_froz_max: the superseded rule, E_F + 2 eV (projector_supply channels below E_F + 2 eV)\n"
             f"dis_froz_max = {old!r}", win)
open("$RUN/${M}_old_deck.win", "w").write(win)
print("[old deck] $M projectors", projectors, "dis_froz_max", old)
EOF
  score_vasp "$M" "$RUN" "${M}_old_deck" "$OUT/vasp_old_deck_same_mesh" "$NP"

  # 6. cutoff evidence: the same parent at 1.3 x ENCUT
  ENCUT0="$(sed -n 's/^ENCUT *= *\([0-9.]*\).*/\1/p' "$FIX/$M/INCAR")"
  ENCUT1="$(python3 -c "print(round($ENCUT0 * $ENCUT_FACTOR))")"
  stage_parent "$M" "${RUN}_encut" "$MESH" "$NB" "$ENCUT1"
  run_scf "$M ENCUT=$ENCUT1" "${RUN}_encut" "$NP" || echo "[matched] $M cutoff check did not finish"

  # the real-binary execution test, once
  if [ "$M" = "Si" ]; then
    AUTOWAN_VASP="$VASP" AUTOWAN_VASP_PARENT_DIR="$RUN" AUTOWAN_VASP_SMOKE=1 \
    AUTOWAN_MPI="mpirun -np $NP" \
      "$PY" -m pytest -q -p no:cacheprovider tests/test_vasp_execution.py -k real_binary -rs 2>&1 | tail -4
  fi
done

# ---------------------------------------------------------------------------------------
# sub-analysis: the QE representation control (SrO) -- 13 hydrogenic functions,
# unrepaired, frozen to the highest occupied level + 2 eV, from the campaign's r4 overlaps
# ---------------------------------------------------------------------------------------
QD=$AUTOWAN_WORK/campaign2/SrO
QC="$SCRATCH/qe_control_SrO"
QOUT="$RESULTS/SrO/qe_control_13_hydrogenic"
rm -rf "$QC"; mkdir -p "$QC" "$QOUT"
for ext in amn mmn eig; do cp "$QD/SrO_pdwf_r4.$ext" "$QC/SrO_qe13.$ext"; done
"$PY" - <<EOF
import re
from autowan.backends import get_backend
from autowan.backends.vasp.inputs import frozen_window_top
parent = get_backend("qe").read_parent("$QD/SrO.save/data-file-schema.xml")
top = frozen_window_top(parent, 13)
win = open("$QD/SrO_pdwf_r4.win").read()
win = re.sub(r"(?m)^! derived from.*\n", "", win)
win = re.sub(r"(?m)^dis_froz_max\s*=.*$",
             f"! QE representation control: the campaign's attempt-4 overlaps (Sr:s;p;d O:s;p hydrogenic),\n"
             f"! re-localised unrepaired with the first deck's frozen rule: highest occupied + 2 eV (capped)\n"
             f"dis_froz_max = {top!r}", win)
open("$QC/SrO_qe13.win", "w").write(win)
print("[qe control] E_F", parent.fermi_energy_ev, "dis_froz_max", top)
EOF
echo "[qe control] wannier90 start=$(date -Is)"
( cd "$QC" && "$W90" SrO_qe13 > w90.log 2>&1 )   # wannier90.x links no MPI: one serial process
echo "[qe control] wannier90 exit=$? end=$(date -Is)"
for f in SrO_qe13.win SrO_qe13.wout SrO_qe13_hr.dat SrO_qe13.werr; do [ -f "$QC/$f" ] && cp "$QC/$f" "$QOUT/$f"; done
"$PY" "$FIX/score_qe_reference.py" --run-dir "$QC" --seed SrO_qe13 \
    --parent-xml "$QD/SrO.save/data-file-schema.xml" --material SrO \
    --out "$QOUT/gate_verdicts.json" --role control \
    --choice "CONTROL (Rule 38 audit): the QE engine with the VASP deck's representation -- 13 hydrogenic functions Sr:s;p;d O:s;p (the campaign's attempt-4 overlaps; overlaps do not depend on the frozen window), unrepaired, dis_froz_max = highest occupied level + 2 eV (the first deck's rule), otherwise the first deck's Wannier90 settings" \
    2>&1 | head -14

# ---------------------------------------------------------------------------------------
# diagnostics (report-only; the same gate object throughout)
# ---------------------------------------------------------------------------------------
for M in "${MATERIALS[@]}"; do
  "$PY" - <<EOF
import dataclasses, json, pathlib
import numpy as np
from autowan.backends import gate_for, get_backend
from autowan.backends.vasp.outputs import read_parent as read_vasp_parent
from autowan.gate import Construction
from autowan.verdicts import verdict_rows
from autowan.w90_outputs import read_run

M = "$M"; run = pathlib.Path("$SCRATCH") / M; out = pathlib.Path("$RESULTS") / M
qdir = pathlib.Path({"Si": "$AUTOWAN_WORK/campaign/Si",
                     "SrO": "$AUTOWAN_WORK/campaign2/SrO"}[M])
qseed = {"Si": "Si_pdwf", "SrO": "SrO_pdwf"}[M]
diag = {"material": M, "note": "report-only diagnostics; no verdict here replaces the contract rows"}

def win_float(path, key):
    for line in open(path):
        s = line.split("!")[0].strip()
        if s.lower().startswith(key) and "=" in s:
            return float(s.split("=")[1])
    return None

def windows(parent, froz):
    b = parent.eigenvalues_ev[0]; ef = parent.fermi_energy_ev
    occ = b[b <= ef]; vbm = float(occ.max()); cbm = float(b[b > ef].min()) if (b > ef).any() else None
    return {"fermi_energy_ev": ef, "vbm_ev": vbm, "cbm_ev": cbm, "gap_ev": (cbm - vbm) if cbm else None,
            "fermi_minus_vbm_ev": ef - vbm, "occupied_min_ev": float(occ.min()),
            "band_distance_window_ev": [float(occ.min()), ef + 0.1 * float(np.ptp(b[b <= ef + 1e-9]))],
            "model_bands_window_ev": [float(b.min()), froz],
            "dis_froz_max_minus_vbm_ev": (froz - vbm) if froz is not None else None,
            "frozen_states_per_k_min_max": [int((b <= froz).sum(axis=1).min()), int((b <= froz).sum(axis=1).max())] if froz is not None else None,
            "frozen_conduction_states_max": int(((b <= froz) & (b > ef)).sum(axis=1).max()) if froz is not None else None}

def amn_stats(path, n_occ):
    lines = open(path).read().splitlines()
    nb, nk, nw = (int(x) for x in lines[1].split()[:3])
    a = np.zeros((nk, nb, nw), complex)
    for line in lines[2:]:
        p = line.split()
        if len(p) >= 5:
            a[int(p[2]) - 1, int(p[0]) - 1, int(p[1]) - 1] = float(p[3]) + 1j * float(p[4])
    s = np.linalg.svd(a, compute_uv=False)
    w = np.abs(a) ** 2
    occ = w[:, :n_occ, :].sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    per_function_occ = w[:, :n_occ, :].sum(axis=1).mean(axis=0) / w.sum(axis=1).mean(axis=0)
    return {"bands_kpoints_functions": [nb, nk, nw],
            "singular_value_min_over_k": float(s.min()), "singular_value_max_over_k": float(s.max()),
            "condition_number_max_over_k": float((s.max(axis=1) / s.min(axis=1)).max()),
            "occupied_weight_fraction_min_mean_max": [float(occ.min()), float(occ.mean()), float(occ.max())],
            "per_function_occupied_weight_fraction": [round(float(x), 4) for x in per_function_occ]}

def centres(wannier, parent):
    c = np.asarray(wannier.centres_ang); lat = parent.lattice_ang; frac = parent.positions_frac
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    sites = np.concatenate([(frac + s) @ lat for s in shifts])
    return [round(float(x), 4) for x in np.min(np.linalg.norm(c[:, None, :] - sites[None], axis=2), axis=1)]

vasp = get_backend("vasp"); qe = get_backend("qe")
vparent = read_vasp_parent(run / "vasprun.xml")
qparent = qe.read_parent(qdir / f"{qseed.split('_')[0]}.save/data-file-schema.xml")
vfroz = win_float(run / f"{M}.win", "dis_froz_max"); qfroz = win_float(qdir / f"{qseed}.win", "dis_froz_max")
diag["windows"] = {"vasp": windows(vparent, vfroz), "qe_reference": windows(qparent, qfroz)}
old_win = run / f"{M}_old_deck.win"
ofroz = win_float(old_win, "dis_froz_max") if old_win.is_file() else None
if ofroz is not None:
    diag["windows"]["vasp_old_deck"] = windows(vparent, ofroz)
    vb = vparent.eigenvalues_ev[0]
    diag["old_vs_new_deck_same_frozen_membership"] = bool(np.array_equal(vb <= ofroz, vb <= vfroz))
n_occ = vparent.occupied_bands()
diag["occupied_bands"] = {"vasp": n_occ, "qe": qparent.occupied_bands()}

# SCF -> overlap step hand-off: the overlap step's own eigenvalues on the identical list
stage = run / f"{M}.vasp2wannier90" / "vasprun.xml"
if stage.is_file():
    so = read_vasp_parent(stage)
    diag["scf_vs_overlap_step_max_abs_ev"] = float(np.abs(so.eigenvalues_ev - vparent.eigenvalues_ev).max())
    diag["scf_vs_overlap_step_fermi_shift_ev"] = so.fermi_energy_ev - vparent.fermi_energy_ev

# k-point sets, as sets: VASP's list vs the QE construction's .win list, wrapped to [0,1)
import xml.etree.ElementTree as ET
root = ET.parse(run / "vasprun.xml").getroot()
vk = np.array([[float(x) for x in v.text.split()] for v in root.findall("kpoints/varray[@name='kpointlist']/v")])
vw = np.array([float(v.text) for v in root.findall("kpoints/varray[@name='weights']/v")])
qtext = open(qdir / f"{qseed}.win").read().lower()
qblock = qtext.split("begin kpoints")[1].split("end kpoints")[0].split()
qk = np.array(qblock, float).reshape(-1, 3)
wrap = lambda k: {tuple(np.round(np.mod(r, 1.0), 6) % 1.0) for r in k}
diag["kpoints"] = {"vasp_count": len(vk), "qe_count": len(qk), "same_set_mod_G": wrap(vk) == wrap(qk),
                   "vasp_weights_uniform": bool(np.allclose(vw, vw[0])),
                   "vasp_usershift": root.findtext("kpoints/generation/v[@name='usershift']"),
                   "vasp_order_first_three": vk[:3].tolist(), "qe_order_first_three": qk[:3].tolist()}

# the cutoff check: eigenvalues at 1.3 x ENCUT, aligned on the valence maximum
enc = run.parent / f"{M}_encut" / "vasprun.xml"
if enc.is_file():
    ep = read_vasp_parent(enc)
    a = vparent.eigenvalues_ev[0] - windows(vparent, None)["vbm_ev"]
    b = ep.eigenvalues_ev[0] - windows(ep, None)["vbm_ev"]
    lo, hi = diag["windows"]["vasp"]["band_distance_window_ev"]
    vb = windows(vparent, None)["vbm_ev"]
    mask = (vparent.eigenvalues_ev[0] >= lo) & (vparent.eigenvalues_ev[0] <= hi)
    diag["cutoff_check"] = {"encut_factor": $ENCUT_FACTOR,
                            "max_abs_ev_in_band_distance_window_vbm_aligned": float(np.abs(a - b)[mask].max()),
                            "max_abs_ev_all_bands_vbm_aligned": float(np.abs(a - b).max()),
                            "gap_change_ev": (windows(ep, None)["gap_ev"] or 0) - (windows(vparent, None)["gap_ev"] or 0)}

# A-matrix: projection singular values and occupied capture
diag["amn"] = {}
for label, path, occ in (("vasp", run / f"{M}.amn", n_occ), ("qe_reference", qdir / f"{qseed}.amn", qparent.occupied_bands())):
    if path.is_file():
        diag["amn"][label] = amn_stats(path, occ)
if M == "SrO":
    for label, path in (("qe_r4", qdir / "SrO_pdwf_r4.amn"), ("vasp_old_deck", run / "SrO_old_deck.amn")):
        if path.is_file():
            diag["amn"][label] = amn_stats(path, qparent.occupied_bands() if label.startswith("qe") else n_occ)

# per-function centre distances to the nearest atom
diag["centre_distances_ang"] = {}
for label, wout, parent in (("vasp", run / f"{M}.wout", vparent), ("qe_reference", qdir / f"{qseed}.wout", qparent)):
    if wout.is_file():
        diag["centre_distances_ang"][label] = centres(read_run(wout), parent)

# the VASP construction re-scored natively (must equal the contract rows) and with its E_F
# placed at the valence maximum (Quantum ESPRESSO's convention for a gapped parent)
inputs = vasp.read_construction(run, M)
wannier = read_run(run / f"{M}.wout")
gate = gate_for(vasp)
def score(parent):
    c = Construction(M, parent=parent, wannier=wannier,
                     material_class="metal" if parent.is_metallic() else "gapped",
                     inputs={**inputs.gate_inputs(0), "route": vasp.route(inputs)})
    return gate.run(c)
native = score(vparent)
contract = json.loads((out / "gate_verdicts.json").read_text())
native_rows = verdict_rows(native, gate)
diag["native_rescore_equals_contract_rows"] = [
    (r["check_id"], r["status"]) for r in native_rows] == [(r["check_id"], r["status"]) for r in contract]
shifted = dataclasses.replace(vparent, fermi_energy_ev=windows(vparent, None)["vbm_ev"])
report = score(shifted)
diag["common_reference_rescore"] = {
    "fermi_energy_ev_used": shifted.fermi_energy_ev, "passed": report.passed,
    "rows": verdict_rows(report, gate)}
handoff = diag.get("scf_vs_overlap_step_max_abs_ev")
cutoff = (diag.get("cutoff_check") or {}).get("max_abs_ev_in_band_distance_window_vbm_aligned")
diag["acceptance"] = {
    "rule": "pre-registered 2026-09-15 before the production job: a construction whose SCF -> overlap-step "
            "eigenvalue hand-off exceeds 5 meV, or whose band-distance-window eigenvalues move more than 10 meV "
            "at 1.3 x ENCUT, has its verdict flagged (not discarded); a missing check reads as not established",
    "handoff_tolerance_ev": 0.005, "handoff_ok": None if handoff is None else handoff <= 0.005,
    "cutoff_tolerance_ev": 0.010, "cutoff_ok": None if cutoff is None else cutoff <= 0.010,
}
(out / "diagnostics.json").write_text(json.dumps(diag, indent=2, default=str) + "\n")
print("[diagnostics]", M, "native==contract", diag["native_rescore_equals_contract_rows"],
      "E_F-at-VBM passed", report.passed, "kset", diag["kpoints"]["same_set_mod_G"],
      "handoff", diag.get("scf_vs_overlap_step_max_abs_ev"), "cutoff", diag.get("cutoff_check"))
EOF
done

# ---------------------------------------------------------------------------------------
# report-only: VASP 6.4's own pseudopotential projections (LOCPROJ_AUTO -> LPRJ_FROM_PSEUDO)
# ---------------------------------------------------------------------------------------
M=SrO
RUN="$SCRATCH/$M"
if [ -f "$RUN/CHGCAR" ] && [ -f "$RUN/$M.win" ]; then
  LP="$SCRATCH/locproj_auto_$M"
  rm -rf "$LP"; mkdir -p "$LP"
  for f in POSCAR POTCAR KPOINTS CHGCAR; do ln -s "$RUN/$f" "$LP/$f"; done
  NB="$(sed -n 's/^NBANDS *= *\([0-9]*\).*/\1/p' "$RUN/INCAR")"
  NP="$(ranks_for "$NB")"
  { grep -v -E '^(LCHARG|LWAVE|ICHARG|LORBIT)' "$RUN/INCAR"
    printf 'ICHARG = 11\nLWANNIER90 = .TRUE.\nLOCPROJ_AUTO = .TRUE.\nNSW = 0\nIBRION = -1\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n'
  } > "$LP/INCAR"
  # the deck's library-facing lines, without a projections block: the functions come from the POTCARs
  "$PY" - <<EOF
from autowan.backends.vasp.inputs import overlap_step_win
text = open("$RUN/$M.win").read()
kept, skip = [], False
for line in overlap_step_win(text).splitlines():
    low = line.strip().lower()
    if low.startswith("begin projections"): skip = True; continue
    if skip:
        if low.startswith("end projections"): skip = False
        continue
    kept.append(line)
open("$LP/wannier90.win", "w").write("\n".join(kept) + "\n")
EOF
  echo "[locproj_auto] $M start=$(date -Is) ranks=$NP"
  ( cd "$LP" && mpirun -np "$NP" "$VASP" > vasp.log 2>&1 )
  echo "[locproj_auto] $M exit=$? end=$(date -Is)"
  grep -i -E "mode|NUM_WANN|LOCPROJ|error|not implemented|refuse" "$LP/vasp.log" | head -20
  ls -la "$LP" | grep -E "amn|mmn|eig|wout" || echo "[locproj_auto] no overlap files written"
  head -2 "$LP/wannier90.amn" 2>/dev/null
fi
echo "[matched] end=$(date -Is)"
