# multidomain (serial port of MFEM miniapps/multidomain, H1 version)

1:1-port of `mfem/miniapps/multidomain/multidomain.cpp` (H1 version) as a
**serial** fem-rs miniapp: two PDEs on two subdomains of one mesh, coupled
one-way through a shared interface.

- outer box (attr 2): heat equation `dT/dt = κΔT`, κ=1, Dirichlet `T=1` on the
  four outer walls (bdr attrs 1-4), insulated on the cylinder wall (attr 9).
- inner cylinder (attr 1): convection-diffusion `dT/dt = κΔT - α∇·(bT)`,
  κ=0.1, α=1, inflow cap (attr 8) `T=0`, interface (attr 9) `T = T_block`.
- order-2 H1 on both submeshes; SSP-RK3 (RK3SSP); mass matrix inverted with
  CG + Jacobi (rtol 1e-8, max 100) inside every `Mult`.

## Files

| file | purpose |
|------|---------|
| `multidomain.rs` | the H1 miniapp (this port) |
| `multidomain_nd.rs` | H(curl) variant (`multidomain_nd.cpp`): magnetic diffusion (outer box) + convection-diffusion (cylinder), ND2. Ported round 64 (D657) |
| `multidomain_rt.rs` | H(div) variant (`multidomain_rt.cpp`): same split scheme with RT1 fluxes. Ported round 64 (D657) |

Upstream `multidomain_nd.cpp` / `multidomain_rt.cpp` are PAR-only; the ND/RT
ports mirror them serially (see each file's header for the mechanism table).

## Running

Once wired as `[[example]] name = "multidomain"` in `examples/Cargo.toml`:

```text
cargo run --example multidomain -- -tf 0.5 -dt 2e-4      # short stable run
cargo run --example multidomain --                       # upstream defaults (dt=1e-5, tf=5)
```

Options: `-o` FE order (2), `-tf` (5.0), `-dt` (1e-5), `-vs` print stride,
`-m` mesh (`data/multidomain-hex.mesh`), plus verification aids `-nr 1`
(mesh already refined), `-qp N` shared quadrature order, `-dump 1` final
dof dumps.

Note: explicit SSP-RK3 needs a diffusion-stable dt; `dt ≳ 5e-4` blows up on
this mesh.

## Mechanism mapping (C++ PAR → serial fem-rs)

| MFEM                                          | this port                                        | note |
|-----------------------------------------------|--------------------------------------------------|------|
| `ParSubMesh::CreateFromDomain(parent, attrs)` | `fem_mesh::submesh::extract_submesh_3d`          | NE/NV/ndofs/bdr-attrs match C++ exactly (288/399/2717, 192/336/2080; attrs 6..9 / 1..9) |
| `ParFiniteElementSpace`, true dofs            | `H1Space<Mesh<3>>`                               | conforming: vdofs == tdofs |
| `ParSubMesh::CreateTransferMap().Transfer`    | interface dof copy by coordinate match           | SubMesh→SubMesh via root parent ≡ "copy block values onto cylinder interface dofs, keep the rest" because the two domains partition the parent |
| `GetEssentialTrueDofs(marker array)`          | local `hex_boundary_dofs` (marker expanded to tag list) | see kernel gap (b) |
| `ProjectBdrCoefficient(ConstantCoefficient)`  | direct dof assignment                            | constant ⇒ identical result |
| `CGSolver` + `HypreSmoother(Jacobi)`          | `solve_pcg_dsmoother`                            | bit-level port of CG+DSmoother(DIAG) |
| `BilinearForm::FormSystemMatrix(ess, M)`      | `fem_space::apply_dirichlet` (DIAG_KEEP)         | MFEM default diag policy |
| `RK3SSPSolver`                                | hand-coded SSP-RK3                               | identical stage coefficients; `t` advances **twice per loop** (both `Step` calls share `t`) like upstream |
| GLVis socketstream                            | trimmed                                          | upstream `-no-vis` behaviour |

## Kernel gaps found (crates/** — not touched, reported for upstream fixing)

1. **`refine_hex8_uniform` child template differs from MFEM
   `UniformRefinement3D`.** On `data/multidomain-hex.mesh` only **9 / 480**
   refined hexes have matching corner sets vs MFEM's output (evidence:
   `tmp/multidomain/{cpp,rust}_hexes.txt`). The port therefore cannot be
   compared 1:1 with a C++ run that refines the same parent; validation runs
   both codes on the *same* (MFEM-refined) mesh via `-nr 1`.
   *Round-70 note:* that figure complicates two effects — the dumps were
   written at different precisions (17 vs 10 significant digits) and the parent
   mesh carries P2 geometry (curved refinement moves vertices off the straight
   lattice), so a corner-set comparison is not a like-for-like test. Rounds
   69/70 show the two refinements are *effectively equivalent for these
   spaces*: rt/nd trajectories are byte-equal on the internally refined mesh
   (round 69) and the round-70 straight-config H1 block rows are byte-equal
   too. The gap note needs re-derivation before it is quoted again.
2. **`refine_uniform_3d` panics on Hex8 meshes with quad boundary faces.**
   `rebuild_3d_boundary` looks up new quad-face centers by exact-bit
   coordinates built as (boundary-vertex-order sum)/4, while the refinement
   creates them in MFEM element-face vertex order → 1-ulp miss. Worked around
   with `refine_hex8_uniform` (emits boundary from its own face-center ids).
3. **`extract_submesh_3d` writes quad boundary faces in sorted-vertex order**
   (a dedup key, not a cyclic Quad4 connectivity). fem-space's
   `boundary_dofs` derives face edges from that order and hits face diagonals
   (`miss_edge = 192/384` on the block walls) → wrong essential sets. MFEM
   solves this with SubMesh face orientations. Worked around with a local
   `hex_boundary_dofs` that walks the element face table.
4. **`write_mfem` corrupts the boundary section** for `Mesh<3>` with mixed
   face types (emits triangles with repeated ids). The refined parent mesh
   cannot be round-tripped to MFEM. Workaround: hand-rolled writer in the
   tmp validation harness.
5. **`dof_coord` reports vertex-averages** for quad-face/body-center dofs
   instead of the mapped positions; differs from MFEM by up to ~2e-2 on
   non-affine hexes (affects reporting/inter-code matching only).
6. **Curved-hex (P2 nodes) geometry path suspected wrong.** On the identical
   curved refined mesh the assembled K still differs ~3% (L1) from MFEM at
   matching high-order quadrature, while a *single straight* hex matches
   bit-for-bit; the cylinder sub-solution diverges (see validation). Prime
   suspect: `build_h1_geometry` node→reference-position assignment for hexes.
7. (design, not a bug) fem-space `boundary_dofs` takes an explicit tag list
   while MFEM uses `bdr_attr_is_ess` marker arrays — the port expands markers.

## ND / RT variants (round 64, D657)

Both variants were ported 1:1 and verified against serial C++ harnesses
(MFEM 4.10 serial tree, same `ParX → X` mirroring as the H1 harness; sources
`tmp/d657/multidomain_{nd,rt}_serial.cpp`, binaries `$HOME/work/d657/`):

- dof / essential-dof counts match MFEM exactly on the default gear:
  ND 7708/5664 ndofs, 1168/800 essential; RT 7296/5120 ndofs, 576/640
  essential (= `(k+1)²` per quad face — the full face blocks).
- initial condition = `interpolate_vector(square_xy)` restricted to the wall
  dofs, equal to MFEM's `ProjectBdrCoefficient{Tangent,Normal}` for this
  linear field: ND IC sum −4.000000 vs C++ −4 (exact); RT ≈ 1e-17 both.
- ND trajectory (dt=1e-5): block dof-sum agrees to ~1e-5 relative per step,
  block/dof-energy (Σdof²) to 0.1–0.3% through t=0.005; cylinder energy to
  a few %.
- RT trajectory: block dof-energy agrees to ~3e-6 (t=4e-5) … ~2% (t=0.005).
  The RT cylinder diverges (fills faster than C++): traced (round 64) to the
  *free* block-interface dof values after one step differing ~2.2x in
  Σv² from MFEM (absolute level ~1e-8 on a 5e-2 field — the transfer itself
  is verified to copy values exactly, `cyl_if == blk_if` in both codes).
  Prime suspect: the RT1-hex interior-basis divergence / `MixedWeakGradDot`
  evaluation on the non-parallelepiped facet hexes — core territory
  (crates/space hdiv, crates/assembly), reported as debt, not worked around
  in the miniapp.

The ND/RT transfer map cannot reuse the H1 coordinate trick: ND edge dofs
sit at Gauss-Legendre points along the *canonical* (min node id → max) edge
direction and ND face dofs take the face-creating element's anchors, so the
same physical dof carries different coordinates/indices in the two
differently-numbered submeshes. Both ports pair dofs per *physical entity*
(sorted corner coordinates) and fix the orientation with each dof's
"geometric factor" `g(d)` recovered from three constant-field interpolations
(`interpolate_vector(e_i)[d] = e_i·g(d)`): `g_cyl = ±g_blk`, and
`dst[c] = sign(g_c·g_b)·src[b]`. The pairing is verified at construction
against an interpolated linear field.

Quadrature orders mirror MFEM (bilininteg.cpp/hpp): mass
`Trans.OrderW() + 2p` (= 2p+2 on the trilinear hex map), CurlCurl on Qk `2p`
(integrator-selected), MixedWeakCurlCross `2p+2`, DivDiv `2p−2`,
MixedWeakGradDot `2p−1`. Note MFEM's `MixedWeakGradDotIntegrator` negates
the test shape internally (`shape *= -1.0`), so the C++ `-alpha` coefficient
fold is a *net* `+alpha` — the RT port passes `+alpha` to fem-rs'
non-negating integrator.

## Validation status (vs serial C++ harness, MFEM 4.9 serial, same mesh)

With both codes solving the identical MFEM-refined mesh (`-nr 1`),
order 2, `-qp 20`, tf=0.5, dt=2e-4:

- cylinder/block submesh NE, NV, ndofs, bdr attrs, essential-dof counts
  (593 / 416) and wall-BC counts (416): **exact match**.
- block (heat) equation final sum: C++ 2079.89 vs Rust 2079.876
  (**7e-6 relative**); min/max agree to 6 digits. Mid-transient (~t=0.1)
  differs ~2% — consistent with the ~3% K-matrix difference below.
- assembled M/K on the same mesh: M relL1 ≈ 0.19, K relL1 ≈ 0.03
  (see gap 6; on a single *straight* hex M/K match **bit-for-bit**).
- cylinder (convection-diffusion) final sum: C++ 454.6 vs Rust 2146.7 —
  **diverges**; prime suspect gap 6 (the pure-diffusion block on the same
  mesh matches, and a single-element convection-matrix check matches after
  the sign relation `fem_rs(b) = -MFEM(b, -1)`).

Repro assets (temporary, `tmp/multidomain/`): C++ serial harness
`serial_multidomain.cpp` (1:1 serial mirror + dumps), comparison scripts,
run logs.

## D667 (round 65): the RT/ND numeric residual family — root causes

The round-64 registered residuals (RT cylinder diverging ~2.2x per step-1
interface dof, ND trajectory 0.1–2.8% off) decompose into three independent
findings, all pinned against MFEM 4.10 probes (`tmp/d667/d667_probe.cpp`,
`d667_rt_ser.cpp`, `d667_twotet_probe.cpp`):

1. **Signed integration weight in the vector assembler (FIXED, assembly
   crate).** `VectorAssembler`'s volume loops multiplied the quadrature
   weight by `det_j.abs()`; MFEM 4.10 carries the **signed** `det J` through
   `Trans.Weight()` (`EvalWeight`, D652 precedent in the scalar assembler).
   Mass-type forms are quadratic in the Piola shape (one `1/det` unpaired),
   so a locally folded curved element contributes a *negated* block in MFEM
   and a repaired positive one in fem-rs: on the folded single-hex fixture
   `data/d667_curved_hex.mesh` the fem-rs mass matrix was off by up to 23%
   per entry; after the one-line fix the element matrices (mass / DivDiv /
   MixedWeakGradDot, incl. amplified-warp variants) match MFEM's raw
   `AssembleElementMatrix` output to ≤1e-11 relative at identical rules
   (`crates/assembly/tests/d667_rt1_curved_hex_mfem_parity.rs`).  Div-type
   forms cancel `det²` and were never sensitive.  The fix is bitwise-inert
   for `det > 0`, which is why the round-64 straight-hex runs matched.

2. **The miniapp's quadrature-order table is wrong (registered, miniapps).**
   The RT1 hex reports `FiniteElement::GetOrder() = 2` (fe_rt.cpp passes
   `p + 1`), not the collection order `p = 1`, and on the curved P2 map
   `Trans.OrderW() = 5` (Qk: `3g − 1`), so MFEM's true per-element defaults
   are mass `OrderW + 2·GetOrder` (affine 6 / curved 9), DivDiv
   `2·GetOrder − 2` = `2k` (RT1: an 8-point rule — never the round-64
   assumed 1-point rule), WeakGradDot `2·GetOrder + OrderW` (6 / 9).  The
   port's `2p+2 / 2p−2 / 2p−1` (4 / 0 / 1) under-integrate every hex.
   Exact helpers now live in `fem_assembly::standard`
   (`mfem_vector_mass_quad_order_rt_hex`, `mfem_div_div_quad_order_rt_hex`,
   `mfem_weak_grad_dot_quad_order_rt_hex`, `mfem_weak_curl_cross_quad_order_nd_hex`,
   `mfem_vector_mass_quad_order_nd_hex`, `mfem_hex_order_w`); the TDO should
   pass those per form.  With `-qp 9` (mass stays at its own 2k+3, divdiv and
   wgrad at 9) the block trajectory matches C++ to 3e-4 relative at t=0.005
   and the cylinder ssq gap collapses from ~870x to ~16%.

3. **`extract_submesh_3d` drops the curved geometry (registered, mesh
   crate).** The submesh builder hard-codes `geometry: None`, so both
   subdomains solve on straightened (trilinear) hexes while MFEM's
   `SubMesh::CreateFromDomain` carries the parent's P2 field.  This is the
   dominant remaining trajectory term (the cylinder, a fully curved
   annulus, is far more sensitive than the mostly-straight block; fem-rs
   also never probes the P2-only geometry detail at its under-integrated
   rules).  Evidence:
   `d667_refined_mesh_curvature_and_folds` (fem-rs's read of the refined
   mesh matches MFEM's own Weight survey exactly — gmin 4.0277e-4, zero
   folded hexes at order-9 points), so the reader and assembler are exact;
   only the submesh hand-off loses the field.

Validation numbers (tf=0.005, dt=1e-5, MFEM-refined mesh `-nr 1`, C++
reference `d667_rt_ser`): block ssq 0.26366 (C++) vs 0.26359 (fem-rs,
`-qp 9`); cyl ssq 1.431e-6 vs 1.196e-6.  After items 2+3 land, the RT/ND
trajectories should close to the straight-mesh bit-level.

## Round 69: stdout byte-alignment (BIT) — canonical surface and residual

Round 68 (D708) landed the gf shadow and made the rt/nd trajectories
C++-exact. Round 69 raised the miniapps' stdout to byte-level comparison
against a fresh print-reference harness (`tmp/dbit/multidomain_{rt,nd}_printref.cpp`,
compiled on the serial MFEM 4.10 tree at `$HOME/mfem410_ser`; numeric core =
`tmp/d657/multidomain_{rt,nd}_serial.cpp` verbatim, round-64 probes stripped).

Adjudication: upstream `multidomain_rt.cpp` / `multidomain_nd.cpp` are
PAR-only and fail to compile on the serial tree (evidence:
`$HOME/work/dbit/official_{rt,nd}_compile_err.txt`, `ParBilinearForm` /
`HypreSmoother` / `ParFiniteElementSpace` undefined) — the round-64 precedent
(harness as reference, compile-fail on record) applies.

Canonical stdout surface (both sides now emit it 1:1):

1. `Options used:` + `   --order/--t-final/--time-step/--visualization-steps`
   rows — MFEM 4.10's `OptionsParser::ParseCheck` ends with `PrintOptions(out)`
   on success (optparser.cpp:270), so this block is part of the official
   stdout; values render in C++ default ostream form (`%g`, 6 significant
   digits), reproduced by the miniapps' `cxx_ostream_f64` (cross-checked
   against `printf %.6g` on 19 values incl. fixed/exponential boundary cases).
2. banner rows (parent/submesh NE, NV, ndofs; essential-dof counts; NEW
   `Block interface tdofs:` row = `GetEssentialTrueDofs(attr 9)` on the block
   submesh — rt 384, nd 800, fem-rs equal);
3. trajectory rows `step <ti>, t = <t>  block: sum=… ssq=…  cyl: sum=… ssq=…`
   — the literal prefix `step <ti>, t = <t>` is the official upstream print
   (multidomain_rt.cpp:440 / multidomain_nd.cpp:388); the sum/ssq diagnostics
   and the float conventions (Rust `{:.6e}` 7-sig, `t` shortest-round-trip)
   are the round-68 no-regression bytes, applied on the C++ side too.

Verification (all three profiles, byte diffs vs the print-ref):
`-tf 0.005 -dt 1e-5` (251 rows), `-tf 0.05 -dt 1e-5` (2501 rows) and
`-tf 0.0002 -dt 1e-5 -vs 2` — every row byte-identical except ONE exempted
row: `Block initial BC dofs (nonzero): …  IC sum: …`.  Root cause (round-64
known, now precisely characterized): MFEM's `ProjectBdrCoefficient{Normal,
Tangent}` integration leaves ±1e-18 dust where the exact trace of the linear
IC vanishes, fem-rs' `interpolate_vector` yields exact 0.0 there — so the
nonzero counts differ (rt 640 vs 512, nd 512 vs 416) and the rt IC sum is
dust-on-zero (−4.796555e-17 vs −3.808677e-17; nd −4.000000e0 matches
exactly). Both are ~22 orders below the field norm; the evolved trajectories
are unaffected (byte-exact everywhere). Core-level (projection equivalence up
to roundoff) — registered D735, not worked around.

⚠️ **`-vs` discipline (round 71 D763 — mandatory for every comparison):** the
print stride is now part of the *solve trajectory*, not just of the output.
Round 71 D749 landed the upstream data flow: inside the `last_step || ti % vs ==
0` block the port refreshes `gf_state` from the RK3 state (`SetFromTrueDofs` in
the upstream sources, `multidomain_rt.cpp:386-387` / `multidomain_nd.cpp:391-392`),
and `Transfer` then reads/writes that refreshed field — so the step at which a
print happens feeds the next transfer. Two runs with different `-vs` therefore
produce *different numbers*, not merely different amounts of output. **Every
port-vs-harness run, every regression profile and every quoted trajectory value
must state `-vs` explicitly** (default 10 on both sides, upstream
`multidomain_rt.cpp:217`); any historical record that omits it is downgraded to
non-reproducible.

Data-flow no-regression: the round-68 pinned step-250 bytes are preserved —
rt `cyl: sum=-6.420799e-6 ssq=1.430509e-6`, nd
`cyl: sum=1.337180e-5 ssq=1.594132e-4` (now proven 7-significant-digit exact
against C++ doubles, stricter than the round-68 6-digit check).

Not run: the full-default profile (`-tf 5`, 5·10^5 rows) — infeasible
in-session (extrapolated ≈2.4 h rt / ≈7 h nd at measured 7.3 s / 21.8 s per
251 rows; C++ print-ref 4.6 s). Equality there is a HYPOTHESIS (D736)
extrapolating from the 2501-row profile.

## Round 70: H1 joins the canonical surface (D737) and the long window (D736)

**D737 — `multidomain.rs` (H1).** Upstream `multidomain.cpp` is PAR-only like
the rt/nd variants (rc=2 on the serial MFEM 4.10 tree, error list in
`tmp/d737/official_h1_compile_err.txt`), so the comparison object is the
serial print-reference harness `tmp/d737/multidomain_h1_printref.cpp` (numeric
core = the official source's serial-feasible path; print surface 1:1 with the
Rust file). The format gaps aligned:

| row | before | after (canonical) |
|-----|--------|-------------------|
| opening block | absent | `Options used:` + `--order/--t-final/--time-step/--visualization-steps` |
| submesh banners | `… ndofs=N bdr_attrs=a..b` probe suffix | `Cylinder submesh: NE=… NV=… ndofs=…` (probe suffix dropped) |
| interface count | absent | `Block interface tdofs: 416` |
| IC row | `Block initial BC dofs (nonzero): 416` | `… 416  IC sum: 4.160000e2` |
| trajectory row | `block: sum=… min=… max=…  cyl: sum=… min=… max=…` | `block: sum=… ssq=…  cyl: sum=… ssq=…` (family form, `{:.6e}` 7-sig) |

Two numeric defects were fixed with it (both D667/D678-class, previously
registered against the rt/nd ports only):

* **Quadrature rules.** The round-64 hardcodes (mass `2p`, K `2p-1`) are not
  MFEM's defaults. MFEM's per-integrator defaults (`bilininteg.cpp`) are mass
  `2p + Trans.OrderW()`, diffusion `2p + dim - 1` and convection
  `OrderGrad + Order + p = 3k + 2p - 1` (`OrderW = 3k-1`, `eltrans.cpp:493-530`).
  The submeshes carry the parent's P2 geometry (D677), so `k = 2` and the
  print-ref `-rules` probe reads **mass 125 / convection 125 / diffusion 64**
  points per hex (orders 9 / 9 / 6) on the curved map, and 64 / 64 / 64
  (order 6) once the geometry is straightened. The two stiffness integrators
  need different rules and are therefore assembled separately and added.
* **Data flow (upstream `multidomain.cpp:381-382`).** The cylinder GridFunction
  is the transfer destination; the upstream loop re-seeds the RK3 vector from
  it every step *and refreshes it from the RK3 output at print steps*. That
  refresh changes the trajectory (the `-norefresh` harness variant, i.e. the
  round-69 rt/nd model, drifts to `cyl sum = 2.782465e0` vs `2.783957e0` at
  step 250 — 5.4e-4 relative), so it is reproduced in the port
  (`cyl_gf_state`).

Verification (print-ref vs port, `-straight` harness variant so that both
sides read the mesh as straight-sided — see the re-run caveat below): the alt
profile (`-tf 0.0002 -vs 2`, 6 rows) is **byte-identical**; the t005 profile
(`-tf 0.005`, 26 rows) is byte-identical in every block row and differs in
10/26 cylinder rows from step 100 on, always in the last printed digit
(~3e-7 relative, registered D750 with candidate causes). Unlike rt/nd there is
**no IC exemption row**: the constant Dirichlet IC projects to exactly 1.0 on
all 416 wall dofs in both codes.

Re-run caveat: this round's runs were made while lane A's `[0,1]³` hex
migration was in flight; that state trips the mesh reader's D41 safety net on
this mesh (P2 nodes refused → straight read, plus a stdout
`Elements with wrong orientation` line). The curved-map configuration must be
re-run once A's migration lands (the curved rules 9/9/6 are already pinned on
the C++ side by the `-rules` probe).

**D736 — long window.** rt at `-tf 0.5 -dt 1e-5` (50001 steps, 2513 printed
rows, C++ print-ref 731 s) is byte-identical apart from the single D735 IC
row; the final row is
`block: sum=-1.558563e0 ssq=4.112349e-1  cyl: sum=7.995288e-6 ssq=4.711721e-4`
on both sides. No divergence and no format drift over a 10×-longer window than
round 69. The full default (`-tf 5`, ≈10× longer again) remains a HYPOTHESIS.

**D749 — the rt/nd harnesses dropped an upstream line.** The round-64/69
print-refs (and the rt/nd ports) omit the print-block `SetFromTrueDofs` pair
that upstream has (`multidomain_rt.cpp:386-387`, `multidomain_nd.cpp:391-392`),
so the D708 "GF interior never refreshed" model reproduces the harness, not
the official source. Measured at t005 step 250 with fresh C++ runs: rt
`cyl sum` −6.420799e-6 (harness/port) vs **−2.137667e-4** (upstream-faithful),
nd 1.337180e-5 vs **6.932270e-5**; the ssq values move by 0.6 % / 0.006 %.
Ready-made variants define the fix
(`tmp/d737/multidomain_{rt,nd}_printref_refresh.cpp`, binaries
`$HOME/work/d737/{rt,nd}_printref_refresh`): add the two lines to the
harnesses, mirror the refresh in the rt/nd ports, then re-gate (the round-69
red-line numbers are harness-model values).
