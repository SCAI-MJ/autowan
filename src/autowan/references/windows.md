# Energy windows in disentanglement (SMV)

When more bands are available than Wannier functions are wanted, Wannier90 first picks,
at every k-point, an N-dimensional subspace (N = num_wann) and only then localises it.

- **Outer window** (`dis_win_min`, `dis_win_max`): the pool of Bloch states the
  subspace may be built from. Unset, it spans every band the overlaps carry. It must
  hold at least num_wann states at every k-point, or the run stops before it starts.
  Setting it can only narrow the pool.
- **Frozen (inner) window** (`dis_froz_min`, `dis_froz_max`): states inside it are kept
  exactly, so the model reproduces the parent bands there. It must lie inside the outer
  window, and at every k-point (and spin) it may hold at most num_wann states; Wannier90
  refuses a deck where it holds more. Unset, nothing is frozen.
- Above the frozen window the subspace is chosen by minimising the gauge-invariant
  spread, so bands there are interpolated smoothly but not reproduced exactly.
- **Iteration**: `dis_num_iter` (default 200) iterations, converged when the change
  stays below `dis_conv_tol` (default 1e-10) over `dis_conv_window` (default 3)
  iterations. `dis_mix_ratio` (default 0.5) mixes successive subspaces; a smaller value
  damps an oscillating selection, 1.0 is undamped and fastest.

Practical reading: widening the frozen window buys fidelity over a wider energy range
but constrains the subspace more; if the frozen count reaches num_wann at some k-point,
either more functions are needed or the window has to stop below that point.

Sources: Souza, Marzari & Vanderbilt, Phys. Rev. B 65, 035109 (2001); Wannier90 User
Guide (v3.1), "Disentanglement" parameters; Pizzi et al., J. Phys.: Condens. Matter 32,
165902 (2020).
