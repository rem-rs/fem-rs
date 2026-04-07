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
│   ├── mesh/       fem-mesh     — SimplexMesh<D>, AMR, CurvedMesh, generators
│   ├── element/    fem-element  — Lagrange P1–P2, Nedelec, Raviart-Thomas, quadrature
│   ├── space/      fem-space    — H1/L2/HCurl/HDiv/VectorH1 spaces, DOF management
│   ├── assembly/   fem-assembly — bilinear/linear/mixed/DG/nonlinear assembly
│   ├── linalg/     fem-linalg   — CsrMatrix, CooMatrix, Vector, BlockMatrix
│   ├── solver/     fem-solver   — CG, GMRES, BiCGSTAB, LOBPCG, ODE integrators
│   ├── amg/        fem-amg      — SA-AMG + RS-AMG (via linger)
│   ├── parallel/   fem-parallel — thread/MPI backends, METIS partitioning, ghost exchange
│   ├── io/         fem-io       — GMSH .msh v4 reader, VTK .vtu XML writer
│   ├── wasm/       fem-wasm     — wasm-bindgen Poisson solver
│   └── ceed/       fem-ceed     — libCEED-style partial assembly operators
└── examples/       fem-examples — MFEM-style examples + EM simulations
```

### MFEM-Style Examples

| Example | PDE | Method | Status |
|---------|-----|--------|--------|
| `ex1_poisson` | -Δu = f | H¹ P1, PCG+Jacobi | O(h²) verified |
| `ex2_elasticity` | -∇·σ = f | VectorH1, PCG+Jacobi | Working |
| `ex3_maxwell` | ∇×∇×E + E = f | H(curl) ND1, PCG+Jacobi | O(h) verified |
| `ex5_mixed_darcy` | Saddle-point [A,Bᵀ;B,0] | Block GMRES | Working |
| `ex9_dg_advection` | -Δu = f (DG) | SIP-DG P1, GMRES | O(h²) verified |
| `ex10_heat_equation` | ∂u/∂t - Δu = 0 | SDIRK-2 + PCG | Working |
| `ex13_eigenvalue` | Kx = λMx | LOBPCG | Working |
| `ex16_nonlinear_heat` | -∇·(κ(u)∇u) = f | Newton + GMRES | O(h²) verified |

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

# build + test everything
cargo test --workspace

# run the electrostatics example (built-in unit-square mesh, 32×32)
cargo run --example em_electrostatics

# run the magnetostatics example
cargo run --example em_magnetostatics_2d
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

Any GMSH **v4.1 ASCII** mesh is supported.  Save with:

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

Expected output (L2 error halves roughly every time N doubles → rate ≈ 2):

```
h ≈ 2.5000e-1,  L2 error ≈ 1e-15   (P1 is exact for linear φ)
h ≈ 1.1111e-1,  L2 error ≈ 1e-15
...
```

> P1 reproduces linear polynomials exactly; the error is at machine precision
> for this case.  Use a quadratic exact solution to observe O(h²) in practice.

---

## Architecture Reference

See [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) for:
- Complete trait interface definitions (`MeshTopology`, `ReferenceElement`, `FESpace`, `LinearSolver`, …)
- Assembly pipeline (8-step reference → physical coordinate transformation)
- AMG hierarchy design
- MPI parallel mesh and parallel CSR matrix specs
- WASM target rules and JS API

See [DESIGN_PLAN.md](DESIGN_PLAN.md) for:
- Phase-by-phase implementation roadmap (Phases 0–25, all complete)
- Module-level file trees for each crate
- Per-phase acceptance criteria and convergence tests

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

All 25 phases are complete. 220+ tests passing across the workspace.

| Crate | Status | Notes |
|-------|--------|-------|
| `fem-core` | ✅ Complete | Scalar, FemError, index types, coord aliases |
| `fem-mesh` | ✅ Complete | SimplexMesh, AMR (red refinement, ZZ estimator, Dörfler marking), CurvedMesh (P2 isoparametric) |
| `fem-element` | ✅ Complete | Lagrange P1/P2 (Seg, Tri, Tet, Quad, Hex); Nedelec ND1 (Tri, Tet); Raviart-Thomas RT0 (Tri, Tet); quadrature rules |
| `fem-linalg` | ✅ Complete | CsrMatrix, CooMatrix, Vector, SparsityPattern, dense LU, BlockMatrix/BlockVector |
| `fem-space` | ✅ Complete | H1Space (P1/P2), L2Space (P0/P1), VectorH1Space, HCurlSpace (Nédélec ND1), HDivSpace (Raviart-Thomas RT0), DofManager, boundary DOFs, Dirichlet BC |
| `fem-assembly` | ✅ Complete | Assembler, Diffusion/Mass/Source/Neumann/Elasticity integrators; MixedAssembler; DG SIP; NonlinearForm + Newton; partial assembly (PA mass/diffusion, matrix-free); VectorAssembler (Piola transforms); CurlCurlIntegrator; VectorMassIntegrator |
| `fem-solver` | ✅ Complete | CG, PCG+Jacobi/ILU0, GMRES, BiCGSTAB (via linger); ODE: ForwardEuler, RK4, RK45, ImplicitEuler, SDIRK-2, BDF-2; LOBPCG eigenvalue solver; MINRES (Choi-Paige-Saunders); SchurComplement (preconditioned GMRES + block-diagonal precond) |
| `fem-amg` | ✅ Complete | SA-AMG + RS-AMG (via linger), reusable AmgSolver hierarchy |
| `fem-io` | ✅ Complete | GMSH v4.1 ASCII reader, VTK .vtu XML writer, unit_cube_tet generator |
| `fem-parallel` | ✅ Complete | ChannelBackend (multi-thread), GhostExchange, METIS k-way partitioning (pure-Rust) |
| `fem-wasm` | ✅ Complete | WasmSolver (unit-square P1 Poisson), wasm-bindgen JS API |
| `fem-ceed` | ✅ Complete | PA operators (mass, diffusion, lumped mass), MatFreeOperator trait |
