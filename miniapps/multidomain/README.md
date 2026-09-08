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

| file          | purpose                                        |
|---------------|------------------------------------------------|
| `multidomain.rs` | the miniapp (this port)                     |

Upstream `multidomain_nd.cpp` / `multidomain_rt.cpp` (ND/RT interface
variants) are **not** ported — see "kernel gaps" below.

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
