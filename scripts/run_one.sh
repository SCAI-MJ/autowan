#!/bin/bash
# One material end to end: scf, then the band count read back from what the
# pseudopotentials actually provide, then nscf, projwfc, and a projectability deck:
# the pseudopotentials' own atomic orbitals as projectors (atom_proj +
# auto_projections), projectability thresholds 0.01 / 0.95 with dis_froz_proj set,
# and an energy frozen window up to E_F + 2 eV.
#
# dis_froz_proj is what makes Wannier90 apply the thresholds at all. It defaults to
# .false., and without it the thresholds are read and silently ignored: every deck of
# the 2026-08-28 campaign ran that way, as energy-window disentanglement, and was
# reported as projectability disentanglement. The deck writer below refuses such a deck.
set -uo pipefail
M="$1"; ROOT="$2"
# Quantum ESPRESSO's bin directory and wannier90.x come from the environment; any
# runtime libraries they need (MPI, MKL) are the caller's to put on LD_LIBRARY_PATH.
QEBIN="${QE_BIN:?set QE_BIN to the directory holding pw.x, projwfc.x and pw2wannier90.x}"
W90="${WANNIER90:?set WANNIER90 to wannier90.x}"
export OMP_NUM_THREADS=1 I_MPI_HYDRA_BOOTSTRAP=fork
ulimit -s unlimited 2>/dev/null || true
NP=${NP:-8}

# In place, not in a scratch directory. A repair that renames the projectors has to
# recompute the overlaps, and pw2wannier90 then needs the charge density and the
# wavefunctions -- which a scratch run deletes on the way out. Copying back only the
# two XML files left every projector repair failing on a missing charge density
# rather than on physics (2026-08-28).
SUB="$ROOT/$M"
cd "$SUB" || exit 1
rm -rf ${M}.save ${M}.wfc* 2>/dev/null
MESH=$(cat "$SUB/mesh.txt")

step () { l=$1; shift; a=$(date +%s.%N); "$@"; rc=$?; b=$(date +%s.%N)
  printf "  %-10s rc=%d %.0fs\n" "$l" $rc "$(echo "$b-$a"|bc)"; return $rc; }

step scf sh -c "mpirun -np $NP $QEBIN/pw.x -nk $NP -in ${M}.scf > ${M}.scf.out 2>&1" || exit 1

# num_wann is however many atomic orbitals the pseudopotentials supply; the protocol
# asks for twice that many bands.
NW=$(grep -oE "Starting wfcs are +[0-9]+" ${M}.scf.out | head -1 | grep -oE "[0-9]+$")
[ -z "$NW" ] && NW=$(grep -oE "number of atomic wfc *= *[0-9]+" ${M}.scf.out | grep -oE "[0-9]+$")
[ -z "$NW" ] && { echo "  could not read the atomic-orbital count"; exit 1; }
NB=$((NW*2))
# The parent is given room the protocol's own deck does not use. PDWF asks for twice
# the function count and that is what the .win declares, but a repair that widens the
# projectors needs bands to draw the new functions from, and those have to exist in the
# parent -- num_bands is a ceiling on what the calculation produced, not a request to
# it. Sized generously here so the ceiling is not the protocol's guess.
# Relative to the function count, not an absolute floor: an atom supplying one
# atomic orbital asked for two dozen bands puts most of them far above anything
# physical, and the diagonaliser fails to converge them at all.
NB_PARENT=$(( NW*4 > NW+8 ? NW*4 : NW+8 ))
# Set the count rather than filling a placeholder: a deck that has already been run
# once carries a number there, the placeholder is gone, and the substitution silently
# does nothing. The .win then declares a band count the .eig does not have, and
# Wannier90 stops with "mismatch in <seed>.eig" -- which reads as a physics failure.
sed -i -E "s/nbnd *= *[A-Za-z_0-9]+/nbnd = $NB_PARENT/" ${M}.nscf

if ! step nscf sh -c "mpirun -np $NP $QEBIN/pw.x -nk $NP -in ${M}.nscf > ${M}.nscf.out 2>&1"; then
  # Davidson can break down on a small cell asked for many empty states -- it reports
  # "S matrix not positive definite" out of cdiaghg, which reads like a broken
  # pseudopotential rather than a diagonaliser choice. Conjugate gradient is slower and
  # does not have that failure, so it is worth one retry before giving up on the
  # material. Not the default: cg costs real time on the larger cells.
  if grep -q "S matrix not positive definite" ${M}.nscf.out 2>/dev/null; then
    echo "  nscf: Davidson broke down; retrying with conjugate gradient"
    sed -i "s/ &electrons/ \&electrons\n    diagonalization = 'cg'/" ${M}.nscf
    step nscf-cg sh -c "mpirun -np $NP $QEBIN/pw.x -nk $NP -in ${M}.nscf > ${M}.nscf.out 2>&1" || exit 1
  else
    exit 1
  fi
fi
# What the run actually produced, not what was asked for: pw.x rounds the count up to
# fill a spin channel and will not go below the occupied manifold.
NB_ACTUAL=$(grep -oE "number of Kohn-Sham states *= *[0-9]+" ${M}.nscf.out | tail -1 | grep -oE "[0-9]+$")
[ -n "$NB_ACTUAL" ] && NB_PARENT=$NB_ACTUAL
[ "$NB" -gt "$NB_PARENT" ] && NB=$NB_PARENT
echo "  num_wann=$NW num_bands=$NB (parent carries $NB_PARENT)"
EF=$(grep -oE "the Fermi energy is +[-0-9.]+" ${M}.nscf.out | tail -1 | grep -oE "[-0-9.]+$")
[ -z "$EF" ] && EF=$(grep -oE "highest occupied.*: *[-0-9.]+" ${M}.nscf.out | tail -1 | grep -oE "[-0-9.]+$")
echo "  E_F=$EF eV"

step projwfc sh -c "mpirun -np $NP $QEBIN/projwfc.x -nk $NP -ndiag 1 -in ${M}.projwfc > ${M}.projwfc.out 2>&1"

python3 - "$M" "$NW" "$NB" "$EF" "$MESH" "$NB_PARENT" <<'PYEOF' || { echo "  deck refused: dis_proj_min/max without dis_froz_proj"; exit 1; }
import re
import sys
from pathlib import Path
M, NW, NB, EF, MESH, NBP = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                            float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
k = [(i/MESH, j/MESH, l/MESH) for i in range(MESH) for j in range(MESH) for l in range(MESH)]
cell = Path(f"{M}.scf").read_text()
struct = cell.split("ATOMIC_SPECIES")[1]
deck = "\n".join([
    "! Pseudo-atomic projectors via atom_proj/auto_projections; projectability "
    "thresholds 0.01/0.95 with dis_froz_proj set; energy frozen window at E_F + 2 eV",
    f"num_wann     = {NW}", f"num_bands    = {NB}",
    "auto_projections = .true.", "dis_proj_min = 0.01", "dis_proj_max = 0.95",
    "dis_froz_proj = .true.",
    f"dis_froz_max = {EF + 2.0:.4f}",
    # pw2wannier90 writes every band the parent produced into the .eig, so a deck that
    # declares fewer has to say which ones it is leaving out; without this Wannier90
    # stops with "mismatch in <seed>.eig".
    (f"exclude_bands = {NB+1}-{NBP}" if NBP > NB else ""),
    "conv_window  = 3", "conv_tol = 1.0e-7", "dis_conv_tol = 1.0e-7",
    "num_iter = 4000", "dis_num_iter = 4000", "num_cg_steps = 200",
    "write_hr = .true.", "", "begin projections", "end projections", "",
    f"mp_grid = {MESH} {MESH} {MESH}", "", "begin kpoints",
    "\n".join(f" {a:.8f} {b:.8f} {c:.8f}" for a, b, c in k), "end kpoints", ""])
# Parse the finished deck: a missing flag silently disables the requested window.
values = {}
in_block = False
for line in deck.splitlines():
    cleaned = re.split(r"[!#]", line, maxsplit=1)[0].strip().lower()
    if re.match(r"begin(?:\s|:|=)", cleaned):
        in_block = True
        continue
    if re.match(r"end(?:\s|:|=)", cleaned):
        in_block = False
        continue
    if in_block:
        continue
    match = re.match(
        r"(dis_proj_min|dis_proj_max|dis_proj_auto|dis_froz_proj)"
        r"(?:\s*[=:]\s*|\s+)(.*)$", cleaned
    )
    if match:
        values[match.group(1)] = match.group(2).strip()

def logical_true(value):
    return re.match(r"\.?t", value, re.I) is not None

if (
    "dis_proj_min" in values
    or "dis_proj_max" in values
    or logical_true(values.get("dis_proj_auto", ""))
) and not logical_true(values.get("dis_froz_proj", "")):
    print(
        f"{M}_pdwf.win: dis_froz_proj must be true; "
        "Wannier90 would parse and ignore the projectability window",
        file=sys.stderr,
    )
    sys.exit(1)
Path(f"{M}_pdwf.win").write_text(deck)
# A spin-polarised parent holds two spin channels in one calculation, so the overlap
# step has to be told which one it is building. Left out, pw2wannier90 counts twice as
# many k-points as the deck declares and stops on "Wrong number of k-points".
spin = "  spin_component = 'up'\n" if "nspin" in cell and "2" in cell.split("nspin")[1][:6] else ""
Path(f"{M}_pdwf.pw2wan").write_text(
    f"&inputpp\n  outdir = './'\n  prefix = '{M}'\n  seedname = '{M}_pdwf'\n"
    f"{spin}"
    "  write_amn = .true.\n  write_mmn = .true.\n  write_eig = .true.\n  atom_proj = .true.\n/\n")
PYEOF

# wannier90 -pp needs the cell and atoms; take them from the nscf output it can read
python3 - "$M" <<'PYEOF'
import sys, re
from pathlib import Path
M = sys.argv[1]
out = Path(f"{M}.nscf.out").read_text()
alat = float(re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)", out).group(1))
BOHR = 0.529177210903
vec = re.findall(r"a\(\d\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", out)[:3]
cell = "\n".join(" ".join(f"{float(c)*alat*BOHR: .8f}" for c in v) for v in vec)
# verbosity='high' prints the atom positions twice -- once in alat units and once in
# crystal units -- and taking every match listed each atom twice, the second copy
# converted from crystal coordinates as though they were alat. Wannier90 then read a
# cell with twice the atoms, so a projector set named per species implied twice the
# functions it should. auto_projections never reads this block, which is why the
# automatic baselines were unaffected and only repairs saw it.
nat = int(re.search(r"number of atoms/cell\s*=\s*(\d+)", out).group(1))
atoms = re.findall(r"\d+\s+(\w+)\s+tau\(\s*\d+\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", out)[:nat]
inv = None
import numpy as np
A = np.array([[float(c)*alat*BOHR for c in v] for v in vec])
lines = []
for sym, x, y, z in atoms:
    cart = np.array([float(x), float(y), float(z)])*alat*BOHR
    frac = cart @ np.linalg.inv(A)
    lines.append(f"{sym} {frac[0]: .8f} {frac[1]: .8f} {frac[2]: .8f}")
win = Path(f"{M}_pdwf.win")
win.write_text(win.read_text().replace("begin projections", "\n".join(
    ["begin unit_cell_cart", "Ang", cell, "end unit_cell_cart", "",
     "begin atoms_frac", "\n".join(lines), "end atoms_frac", "", "begin projections"]), 1))
PYEOF

step w90pp $W90 -pp ${M}_pdwf || exit 1
step p2w   sh -c "mpirun -np $NP $QEBIN/pw2wannier90.x -ndiag 1 -in ${M}_pdwf.pw2wan > ${M}_pdwf.pw2wan.out 2>&1" || exit 1
step w90   $W90 ${M}_pdwf || true
echo "  DONE $M"
