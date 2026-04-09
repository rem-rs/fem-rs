# fem-rs

A general-purpose finite element method (FEM) library in Rust, targeting
feature parity with [MFEM](https://mfem.org/). Designed for clarity,
extensibility, MPI/AMG parallelism, and WASM compilation.

---

## Crate Structure

```
fem-rs/
├── crates/
│   ├── core/       fem-core     — scalar types, index aliases, FemError
│   ├── mesh/       fem-mesh     — SimplexMesh<D>, AMR, NCMesh, CurvedMesh, periodic, generators
│   ├── element/    fem-element  — Lagrange P1–P3 (Seg/Tri/Tet/Quad/Hex), Nedelec ND1, RT0, quadrature
│   ├── space/      fem-space    — H1/L2/HCurl/HDiv/VectorH1/H1Trace spaces, DOF management, hanging nodes
│   ├── assembly/   fem-assembly — bilinear/linear/mixed/DG/nonlinear/partial assembly, coefficients
│   ├── linalg/     fem-linalg   — CsrMatrix, CooMatrix, Vector, BlockMatrix, DenseTensor
│   ├── solver/     fem-solver   — CG/PCG/GMRES/BiCGSTAB/IDR(s)/TFQMR, direct solvers, LOBPCG/KrylovSchur, ODE
│   ├── amg/        fem-amg      — SA-AMG + RS-AMG, Chebyshev smoother, V/W/F cycles (via linger)
│   ├── parallel/   fem-parallel — thread/MPI backends, METIS partitioning, ghost exchange, WASM Workers
│   ├── io/         fem-io       — GMSH .msh v2/v4 reader, VTK .vtu writer/reader, Matrix Market .mtx
│   ├── wasm/       fem-wasm     — wasm-bindgen Poisson solver, multi-Worker parallel
│   └── ceed/       fem-ceed     — libCEED-style partial assembly (matrix-free mass/diffusion)
└── examples/       fem-examples — MFEM-style examples + EM simulations + parallel examples
```

### MFEM-Style Examples

| Example | PDE | Method | Notes |
|---------|-----|--------|-------|
| `ex1_poisson` | −Δu = f | H¹ P1, PCG+Jacobi | O(h²) verified |
| `ex2_elasticity` | −∇·σ = f | VectorH1 P1, PCG | Working |
| `ex3_maxwell` | ∇×∇×E + E = f | H(curl) ND1/ND2, PCG+AMS | AMS preconditioner available |
| `ex4_darcy` | −∇·u = f, u = −κ∇p | H(div) RT0/RT1 + L², MINRES+ADS | ADS preconditioner available |
| `ex5_mixed_darcy` | Saddle-point Darcy/Stokes | Block PGMRES | Working |
| `ex7_neumann_mixed_bc` | −Δu = f, mixed BCs | H¹ P1, Neumann + Dirichlet | Working |
| `ex9_dg_advection` | −Δu = f (DG) | SIP-DG P1, GMRES | O(h²) verified |
| `ex10_heat_equation` | ∂u/∂t − Δu = 0 | SDIRK-2 + PCG | Working |
| `ex10_wave_equation` | ∂²u/∂t² − Δu = 0 | Newmark-β + PCG | Working |
| `ex13_eigenvalue` | Kx = λMx | LOBPCG | 1-D Laplacian eigenvalues verified |
| `ex14_dc_current` | −∇·(σ∇φ) = 0 | H¹ P1, DC current distribution | Working |
| `ex15_dg_amr` | −Δu = f (AMR+DG) | P1 + ZZ estimator + Dörfler | O(h²) with refinement |
| `ex15_tet_nc_amr` | 3-D NC AMR | Tet4 NC refinement + hanging face constraints | Working |
| `ex16_nonlinear_heat` | −∇·(κ(u)∇u) = f | Newton + GMRES | O(h²) verified |
| `ex_convergence` | −Δu = f | P1/P2/P3 convergence sweep | O(h²)/O(h³)/O(h⁴) |
| `ex_stokes` | Stokes lid-driven cavity | Taylor-Hood P2/P1 + Schur GMRES | Verified |
| `ex_navier_stokes` | Navier-Stokes (Kovasznay) | P2/P1 Oseen/Picard, Re=40 | Converged |

### Parallel Examples

| Example | Problem | Notes |
|---------|---------|-------|
| `pex1_poisson` | Parallel Poisson (P1/P2) | PCG+AMG, contiguous/METIS/streaming |
| `pex2_mixed_darcy` | Parallel mixed Poisson | H(div) × L², block GMRES |
| `pex3_maxwell` | Parallel Maxwell | H(curl) ND1, PCG |
| `pex4_parallel_heat` | Parallel heat equation | Parallel SDIRK-2 |
| `pex5_darcy` | Parallel Darcy | H(div) × L², saddle-point |

Dependency order (each crate depends only on crates listed above it):
`core → mesh/linalg/element → space → assembly → solver/amg → parallel/io/wasm`

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Rust | ≥ 1.75 stable | `rustup update stable` |
| wasm32 target | optional | `rustup target add wasm32-unknown-unknown` |
| GMSH | optional | only needed to generate custom meshes |
| ParaView / VisIt | optional | to visualise `.vtk` output |

---

## Quick Start

```bash
git clone <repo>
cd fem-rs
git submodule update --init --recursive

# build + test everything
cargo test --workspace

# run the electrostatics example (built-in unit-square mesh, 32×32)
cargo run --example em_electrostatics

# run the magnetostatics example
cargo run --example em_magnetostatics_2d

# P1/P2/P3 convergence comparison
cargo run --example ex_convergence

# Stokes lid-driven cavity (Taylor-Hood P2/P1)
cargo run --example ex_stokes

# Navier-Stokes Kovasznay flow (Re=40)
cargo run --example ex_navier_stokes
```

---

## EM Simulation Examples

All examples are in `examples/` and share a common library (`examples/src/lib.rs`)
that provides:

- **P1 (linear triangle) assembly** — diffusion operator `∫ κ ∇u·∇v dx`
- **Neumann load** — boundary flux `∫ g v ds`
- **Reduced-system PCG solver** — solves on free DOFs, avoiding
  Dirichlet-scale artefacts
- **Gradient recovery** — element-averaged `∇u` from nodal DOFs
- **VTK Legacy ASCII writer** — direct ParaView/VisIt input

### 1. Electrostatics (`em_electrostatics`)

Solves `-∇·(ε ∇φ) = ρ` for the electric potential φ.

```
cargo run --example em_electrostatics [-- OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--case <name>` | `parallel_plate` | Test case (see below) |
| `--n <N>` | `32` | Mesh refinement: N×N squares |
| `--mesh <file.msh>` | — | Load a GMSH v4 mesh instead |
| `--tol <f>` | `1e-10` | PCG relative tolerance |
| `--max-iter <N>` | `10000` | PCG maximum iterations |
| `--voltage <V>` | `1.0` | Applied voltage (coaxial case) |
| `--dirichlet-tags <1,2>` | — | Override Dirichlet boundary tags |

#### Built-in cases

**`parallel_plate`** (default) — parallel plate capacitor

```
φ = 0  on y = 0   (bottom, tag 1)
φ = 1  on y = 1   (top,    tag 3)
∂φ/∂n = 0 on x = 0, 1  (left/right, tags 2, 4)

Exact solution: φ(x,y) = y   →   L2 error ≈ machine ε for P1
```

```bash
cargo run --example em_electrostatics -- --case parallel_plate --n 64
```

**`point_charge`** — point charge at domain centre

```
-ε₀ ∇²φ = δ(x-0.5, y-0.5)  (approximated by uniform disc)
φ = 0 on all boundaries
```

```bash
cargo run --example em_electrostatics -- --case point_charge --n 64
```

**`coaxial`** — coaxial cable cross-section

```
φ = V_inner on inner circle  (tag 1)
φ = 0       on outer circle  (tag 2)

Exact: φ(r) = V·ln(r/r_outer) / ln(r_inner/r_outer)
```

```bash
# with built-in mesh (polygonal approximation):
cargo run --example em_electrostatics -- --case coaxial

# with a proper GMSH mesh:
gmsh examples/meshes/coaxial.geo -2 -o examples/meshes/coaxial.msh -format msh4
cargo run --example em_electrostatics -- \
    --case coaxial --mesh examples/meshes/coaxial.msh
```

#### Output

`output/electrostatics.vtk` — open with ParaView or VisIt.
Fields: `potential_V` (nodal scalar), `E_field_Vm` (element vector).

---

### 2. 2-D Magnetostatics (`em_magnetostatics_2d`)

Solves `-∇·(ν ∇A_z) = J_z` for the z-component of magnetic vector potential.
Magnetic flux density recovered as `B_x = ∂A_z/∂y`, `B_y = -∂A_z/∂x`.

```
cargo run --example em_magnetostatics_2d [-- OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--case <name>` | `square_conductor` | Test case (see below) |
| `--n <N>` | `32` | Mesh refinement: N×N squares |
| `--mesh <file.msh>` | — | Load a GMSH v4 mesh instead |
| `--J <A/m²>` | `1e6` | Current density magnitude |
| `--tol <f>` | `1e-10` | PCG relative tolerance |
| `--max-iter <N>` | `10000` | PCG maximum iterations |

#### Built-in cases

**`square_conductor`** (default) — single square conductor in free space

```
ν = ν₀ = 1/μ₀  everywhere
J_z = 1 MA/m²   in [0.3, 0.7]²
A_z = 0          on all boundaries
```

```bash
cargo run --example em_magnetostatics_2d -- --case square_conductor --n 64
```

**`two_conductors`** — two anti-parallel conductors (demonstrates field cancellation)

```
+J_z in [0.1,0.3]×[0.3,0.7]    (current out of page)
-J_z in [0.7,0.9]×[0.3,0.7]    (current into page)
A_z = 0 on boundary
```

```bash
cargo run --example em_magnetostatics_2d -- --case two_conductors --n 64
```

**`transformer`** — transformer cross-section with iron core

```
Iron core: μ_r = 1000  (ν = ν₀/1000)
Primary winding:   +J_z in left window
Secondary winding: -J_z in right window
A_z = 0 on outer boundary
```

```bash
cargo run --example em_magnetostatics_2d -- --case transformer --n 64

# with a proper GMSH mesh:
gmsh examples/meshes/transformer.geo -2 -o examples/meshes/transformer.msh -format msh4
cargo run --example em_magnetostatics_2d -- \
    --case transformer --mesh examples/meshes/transformer.msh
```

#### Output

`output/magnetostatics.vtk` — open with ParaView or VisIt.
Fields: `Az_Wb_per_m` (nodal scalar), `B_field_T` (element vector).

---

## Using Custom GMSH Meshes

Any GMSH **v2 ASCII** or **v4.1 ASCII/binary** mesh is supported.  Save with:

```
gmsh your_geometry.geo -2 -o mesh.msh -format msh4
```

Physical group tags in the `.geo` file map directly to boundary condition
selectors in the examples (`--dirichlet-tags`, `--neumann-tags`).

Sample geometry files are in `examples/meshes/`:

| File | Problem |
|------|---------|
| `coaxial.geo` | Coaxial cable (annular domain) |
| `square_conductor.geo` | Single square conductor |
| `transformer.geo` | Transformer cross-section with iron core |

---

## Viewing Results in ParaView

1. Open ParaView (≥ 5.10 recommended).
2. **File → Open** → select `output/electrostatics.vtk` or `output/magnetostatics.vtk`.
3. Click **Apply**.
4. Select a field in the toolbar dropdown (`potential_V`, `E_field_Vm`, etc.).
5. For vector fields: **Filters → Glyph** to show arrows.

---

## Convergence Test

The parallel plate example has a known exact solution `φ(x,y) = y`.
Run at different refinements to confirm O(h²) L2 convergence for P1 elements:

```bash
for N in 4 8 16 32 64; do
  cargo run -q --example em_electrostatics -- --n $N 2>&1 | grep "L2 error"
done
```

Run the high-order convergence sweep (P1/P2/P3 on 2-D Poisson):

```bash
cargo run --example ex_convergence
```

Expected rates: P1 → 2, P2 → 3, P3 → 4.

---

## Architecture Reference

See [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) for:
- Complete trait interface definitions (`MeshTopology`, `ReferenceElement`, `FESpace`, `LinearSolver`, …)
- Assembly pipeline (8-step reference → physical coordinate transformation)
- AMG hierarchy design
- MPI parallel mesh and parallel CSR matrix specs
- WASM target rules and JS API

See [DESIGN_PLAN.md](DESIGN_PLAN.md) for the full phase-by-phase implementation log (Phases 0–48).

See [MFEM_MAPPING.md](MFEM_MAPPING.md) for a feature-by-feature correspondence with MFEM.

---

## Development

```bash
# check entire workspace
cargo check --workspace

# run all tests
cargo test --workspace

# clippy (zero warnings policy)
cargo clippy --workspace -- -D warnings

# build for WASM (requires wasm32 target)
cargo wasm-build
```

The workspace `Cargo.toml` defines two alias shortcuts:

```toml
[alias]
wasm-build = "build --target wasm32-unknown-unknown -p wasm --no-default-features"
check-all  = "check --workspace --all-features"
```

---

## Implementation Status

Phase 49 complete. 515+ tests passing across the workspace.

| Crate | Status | Highlights |
|-------|--------|------------|
| `fem-core` | ✅ Complete | Scalar traits, FemError, NodeId/DofId, coord aliases |
| `fem-mesh` | ✅ Complete | SimplexMesh, uniform/adaptive AMR, NCMesh (Tri3+Tet4 hanging constraints), CurvedMesh P2 isoparametric, periodic mesh, bounding box |
| `fem-element` | ✅ Complete | Lagrange P1–P3 (Seg, Tri, Tet), Q1/Q2 (Quad), Q1 (Hex); Nédélec ND1/ND2 (Tri, Tet); Raviart-Thomas RT0/RT1 (Tri, Tet); Gauss/Lobatto/Grundmann-Moller quadrature |
| `fem-linalg` | ✅ Complete | CsrMatrix, CooMatrix, Vector, SparsityPattern, dense LU, BlockMatrix/BlockVector, DenseTensor |
| `fem-space` | ✅ Complete | H1Space (P1–P3), L2Space (P0/P1), VectorH1Space, HCurlSpace (ND1/ND2), HDivSpace (RT0/RT1), H1TraceSpace (P1–P3), DOF manager, hanging-node constraints |
| `fem-assembly` | ✅ Complete | Scalar + vector assemblers; 15+ integrators; MixedAssembler; SIP-DG; NonlinearForm + Newton; partial assembly (matrix-free); coefficient system (PWConst, GridFunction, composition) |
| `fem-solver` | ✅ Complete | CG/PCG+Jacobi/ILU0/ILDLt, GMRES, BiCGSTAB, IDR(s), TFQMR, FGMRES; sparse direct: LU/Cholesky/LDLᵀ; LOBPCG + KrylovSchur; MINRES; Schur complement; ODE: Euler/RK4/RK45/SDIRK-2/BDF-2/Newmark-β/Generalized-α/IMEX-ARK3 |
| `fem-amg` | ✅ Complete | SA-AMG + RS-AMG, Chebyshev smoother, V/W/F cycles, reusable hierarchy (via linger) |
| `fem-io` | ✅ Complete | GMSH v2/v4.1 ASCII+binary reader; VTK .vtu XML writer + reader; Matrix Market .mtx reader/writer |
| `fem-parallel` | ✅ Complete | ChannelBackend (multi-thread), NativeMPI backend, GhostExchange, METIS k-way partitioning, streaming partition, WASM multi-Worker |
| `fem-wasm` | ✅ Complete | WasmSolver (unit-square P1 Poisson), multi-Worker parallel solver, wasm-bindgen JS API |
| `fem-ceed` | ✅ Complete | PA operators (mass, diffusion, lumped mass), MatFreeOperator trait |
