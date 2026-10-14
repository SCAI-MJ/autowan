# Frontier Wannierisation lookup table (`frontier_wannier_lookup.csv.gz`)

One row per top-level Wannierisation workflow (2,172 rows, 42 columns) of the published
projectability-disentangled (PDWF) frontier, extracted from the SQL databases inside the
Materials Cloud AiiDA archives below. Read by the repair model's read-only
`frontier_dataset` tool. Built 2026-10-07 by plain `sqlite3` over the archives'
`db.sqlite3` members (range-fetched, CRC-checked).

Licence: **CC-BY-4.0** (all records). Cite the papers and the archive DOIs:

| `dataset` value | archive (DOI) | paper | pseudopotentials |
|---|---|---|---|
| `qiao2023_x0-yf` | Materials Cloud `10.24435/materialscloud:x0-yf` (v2 of `v4-e9`) | J. Qiao, G. Pizzi, N. Marzari, npj Comput. Mater. 9, 208 (2023), arXiv:2303.07877 | SSSP 1.1 PBE efficiency |
| `jiang2025_ck-br_200` | `10.24435/materialscloud:ck-br` (v2; v1 `9g-ds`) | Y. Jiang et al., npj Comput. Mater. 11, 353 (2025), arXiv:2507.06840 | SSSP 1.1 PBE |
| `jiang2025_ck-br_magnetic` | `10.24435/materialscloud:ck-br` | same | SSSP PBE (index label) |
| `jiang2025_ck-br_40largest` | `10.24435/materialscloud:ck-br` | same; the 40 worst-eta2 materials of the 2023 high-throughput run | SSSP **PBEsol** |
| `jiang2025_9g-ds_v1_soc` | `10.24435/materialscloud:9g-ds` (v1 SOC member) | same | PseudoDojo FR (`pseudoDojo`) or pslibrary mix (`ps_mix`), spin-orbit |

Caveats: only the 200-material benchmarks, the test materials, the 40
largest-eta2 materials and the magnetic and SOC sets are published per workflow, not the
21,737-material run; eta2 exists for Optimize workflows only; `n_w90_runs` counts the
plotting rerun; the 2023 `dis_froz_max_used_abs` is empty where projectability alone was
used; phase and magnetic state are not matched by formula (e.g. the archive's NiO is
non-magnetic, its Na is hcp or cI16). Pseudopotential families are never mixed silently:
every row the tool returns names its family.

sha256 of the uncompressed CSV: 5a79e2bfbdf67118cfb28ccc1ab54b6ce321f09319860607442695e678432af4
