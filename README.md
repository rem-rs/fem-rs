# fem-rs

A general-purpose finite element method (FEM) library in Rust, targeting
feature parity with [MFEM](https://mfem.org/). Designed for clarity,
extensibility, MPI/AMG parallelism, and WASM compilation.

---

## Crate Structure

```
fem-rs/
├── crates/
�?  ├── core/       fem-core     �?scalar types, index aliases, FemError
�?  ├── mesh/       fem-mesh     �?SimplexMesh<D>, AMR, NCMesh, CurvedMesh, periodic, generators
�?  ├── element/    fem-element  �?Lagrange P1–P3 (Seg/Tri/Tet/Quad/Hex), Nedelec ND1, RT0, quadrature
�?  ├── space/      fem-space    �?H1/L2/HCurl/HDiv/VectorH1/H1Trace spaces, DOF management, hanging nodes
�?  ├── assembly/   fem-assembly �?bilinear/linear/mixed/DG/nonlinear/partial assembly, coefficients
�?  ├── linalg/     fem-linalg   �?CsrMatrix, CooMatrix, Vector, BlockMatrix, DenseTensor
�?  ├── solver/     fem-solver   �?CG/PCG/GMRES/BiCGSTAB/IDR(s)/TFQMR, direct solvers, LOBPCG/KrylovSchur, ODE
�?  ├── amg/        fem-amg      �?SA-AMG + RS-AMG, Chebyshev smoother, V/W/F cycles (via linger)
�?  ├── parallel/   fem-parallel �?thread/MPI backends, METIS partitioning, ghost exchange, WASM Workers
�?  ├── io/         fem-io       �?GMSH .msh v2/v4 reader, VTK .vtu writer/reader, Matrix Market .mtx
�?  ├── wasm/       fem-wasm     �?wasm-bindgen Poisson solver, multi-Worker parallel
�?  └── ceed/       fem-ceed     �?libCEED-style partial assembly (matrix-free mass/diffusion)
└── examples/       fem-examples �?MFEM-style examples + EM simulations + parallel examples
```

### MFEM-Style Examples

All examples listed in this section are intended to have a one-to-one correspondence
with MFEM examples.

| Example | PDE | Method | Notes |
|---------|-----|--------|-------|
| `mfem_ex1_poisson` | −Δu = f | H¹ P1, PCG+Jacobi | O(h²) verified |
| `mfem_ex2_elasticity` | −∇·σ = f | VectorH1 P1, PCG | Working |
| `mfem_ex3` | ∇×∇×E + E = f | H(curl) ND1/ND2, PCG+AMS | AMS preconditioner available |
| `mfem_ex4_darcy` | −∇·u = f, u = −κ∇p | H(div) RT0/RT1 + L², MINRES+ADS | ADS preconditioner available |
| `mfem_ex5_mixed_darcy` | Saddle-point Darcy/Stokes | Block PGMRES | Working |
| `mfem_ex7_neumann_mixed_bc` | −Δu = f, mixed BCs | H¹ P1, Neumann + Dirichlet | Working |
| `mfem_ex9_dg_advection` | −Δu = f (DG) | SIP-DG P1, GMRES | O(h²) verified |
| `mfem_ex10_heat_equation` | ∂u/∂t �?Δu = 0 | SDIRK-2 + PCG | Working |
| `mfem_ex10_wave_equation` | ∂²u/∂t² �?Δu = 0 | Newmark-β + PCG | Working |
| `mfem_ex13` | Kx = λMx | LOBPCG | 1-D Laplacian eigenvalues verified |
| `mfem_ex14_dc_current` | −∇·(σ∇�? = 0 | H¹ P1, DC current distribution | Working |
| `mfem_ex15_dg_amr` | −Δu = f (AMR+DG) | P1 + ZZ estimator + Dörfler | O(h²) with refinement |
| `mfem_ex15_tet_nc_amr` | 3-D NC AMR | Tet4 NC refinement + hanging face constraints | Working |
| `mfem_ex16_nonlinear_heat` | −∇·(κ(u)∇u) = f | Newton + GMRES | O(h²) verified |
| `mfem_ex40` | Stokes lid-driven cavity | Taylor-Hood P2/P1 + Schur GMRES | Verified |
| `mfem_ex19` | Navier-Stokes (Kovasznay) | P2/P1 Oseen/Picard, Re=40 | Converged |

### Parallel Examples

| Example | Problem | Notes |
|---------|---------|-------|
| `mfem_pex1_poisson` | Parallel Poisson (P1/P2) | PCG+AMG, contiguous/METIS/streaming |
| `mfem_pex2_mixed_darcy` | Parallel mixed Poisson | H(div) × L², block GMRES |
| `mfem_pex3_maxwell` | Parallel Maxwell | H(curl) ND1, PCG |
| `mfem_pex4_parallel_heat` | Parallel heat equation | Parallel SDIRK-2 |
| `mfem_pex5_darcy` | Parallel Darcy | H(div) × L², saddle-point |

Dependency order (each crate depends only on crates listed above it):
`core �?mesh/linalg/element �?space �?assembly �?solver/amg �?parallel/io/wasm`

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Rust | �?1.75 stable | `rustup update stable` |
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

# Stokes lid-driven cavity (Taylor-Hood P2/P1)
cargo run --example mfem_ex40

# Navier-Stokes Kovasznay flow (Re=40)
cargo run --example mfem_ex19
```

---

## Architecture Reference

See [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) for:
- Complete trait interface definitions (`MeshTopology`, `ReferenceElement`, `FESpace`, `LinearSolver`, �?
- Assembly pipeline (8-step reference �?physical coordinate transformation)
- AMG hierarchy design
- MPI parallel mesh and parallel CSR matrix specs
- WASM target rules and JS API

See [DESIGN_PLAN.md](DESIGN_PLAN.md) for the full phase-by-phase implementation log (Phases 0�?8).

See [MFEM_MAPPING.md](MFEM_MAPPING.md) for a feature-by-feature correspondence with MFEM.

See [MFEM_ALIGNMENT_TRACKER.md](MFEM_ALIGNMENT_TRACKER.md) for the unified parity tracker and priority gaps.

See [docs/mfem-parity-matrix-template.md](docs/mfem-parity-matrix-template.md) for measurable parity acceptance gates.

See [docs/mfem-6week-plan-estimates.md](docs/mfem-6week-plan-estimates.md) for the current 6-week execution and effort plan.

See [docs/mfem-baseline-snapshot-2026-04-18.md](docs/mfem-baseline-snapshot-2026-04-18.md) for the latest command-backed baseline snapshot.

See [docs/ras-ddm-status-2026-04-19.md](docs/ras-ddm-status-2026-04-19.md) for the current Domain Decomposition (RAS) implementation status, benchmark commands, and latest results.

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

Phase 52 complete. 520+ tests passing across the workspace.

| Crate | Status | Highlights |
|-------|--------|------------|
| `fem-core` | �?Complete | Scalar traits, FemError, NodeId/DofId, coord aliases |
| `fem-mesh` | �?Complete | SimplexMesh, uniform/adaptive AMR, NCMesh (Tri3+Tet4 hanging constraints), CurvedMesh P2 isoparametric, periodic mesh, bounding box |
| `fem-element` | �?Complete | Lagrange P1–P3 (Seg, Tri, Tet), Q1/Q2 (Quad), Q1 (Hex); Nédélec ND1/ND2 (Tri, Tet); Raviart-Thomas RT0/RT1 (Tri, Tet); Gauss/Lobatto/Grundmann-Moller quadrature |
| `fem-linalg` | �?Complete | CsrMatrix, CooMatrix, Vector, SparsityPattern, dense LU, BlockMatrix/BlockVector, DenseTensor |
| `fem-space` | �?Complete | H1Space (P1–P3), L2Space (P0/P1/P2), VectorH1Space, HCurlSpace (ND1/ND2, including 3D ND2 shared face DOFs), HDivSpace (RT0/RT1), H1TraceSpace (P1–P3), DOF manager, hanging-node constraints |
| `fem-assembly` | �?Complete | Scalar + vector assemblers; 15+ integrators; MixedAssembler; SIP-DG; NonlinearForm + Newton; partial assembly (matrix-free); coefficient system (PWConst, GridFunction, composition); DiscreteLinearOperator supports ND2->L2(P2), RT1->L2(P2), and 3D high-order curl (ND2->RT1) with strict de Rham verification |
| `fem-solver` | �?Complete | CG/PCG+Jacobi/ILU0/ILDLt, GMRES, BiCGSTAB, IDR(s), TFQMR, FGMRES; sparse direct: LU/Cholesky/LDLᵀ; LOBPCG + KrylovSchur; MINRES; Schur complement; ODE: Euler/RK4/RK45/SDIRK-2/BDF-2/Newmark-β/Generalized-α/IMEX-Euler/IMEX-SSP2/IMEX-ARK3 + ImexOperator/ImexTimeStepper |
| `fem-amg` | �?Complete | SA-AMG + RS-AMG, Chebyshev smoother, V/W/F cycles, reusable hierarchy (via linger) |
| `fem-io` | �?Complete | GMSH v2/v4.1 ASCII+binary reader; VTK .vtu XML writer + reader; Matrix Market .mtx reader/writer |
| `fem-parallel` | �?Complete | ChannelBackend (multi-thread), NativeMPI backend, GhostExchange, METIS k-way partitioning, streaming partition, WASM multi-Worker, RAS preconditioning (PCG/GMRES, overlap 0/1, Diag/ILU0 local solves) |
| `fem-wasm` | �?Complete | WasmSolver (unit-square P1 Poisson), multi-Worker parallel solver, wasm-bindgen JS API |
| `fem-ceed` | �?Complete | PA operators (mass, diffusion, lumped mass), MatFreeOperator trait |

