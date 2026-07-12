# fem-rs

A general-purpose finite element method (FEM) library in Rust, targeting
feature parity with [MFEM](https://mfem.org/). Designed for clarity,
extensibility, MPI/AMG parallelism, and WASM compilation.

MFEM correspondence in this repository is defined by feature coverage,
example/workflow parity, and compatible public APIs. External solver names
such as `hypre`, `mumps`, and `mkl` are treated as compatibility contracts,
not as required external FFI deliverables.

---

## Crate Structure

```
fem-rs/
├── crates/
│   ├── core/       fem-core     — scalar types, index aliases, FemError
│   ├── mesh/       fem-mesh     — Mesh<D>, AMR, NCMesh, CurvedMesh, periodic, generators
│   ├── element/    fem-element  — Lagrange P1–P3 (Seg/Tri/Tet/Quad/Hex), Nedelec ND1, RT0, quadrature
│   ├── space/      fem-space    — H1/L2/HCurl/HDiv/VectorH1/H1Trace spaces, DOF management, hanging nodes
│   ├── assembly/   fem-assembly — bilinear/linear/mixed/DG/nonlinear/partial assembly, coefficients; optional `--features reed` helpers
│   ├── linalg/     fem-linalg   — CsrMatrix, CooMatrix, Vector, BlockMatrix, DenseTensor
│   ├── solver/     fem-solver   — CG/PCG/GMRES/BiCGSTAB/IDR(s)/TFQMR, direct solvers, LOBPCG/KrylovSchur, ODE, compatibility solver APIs
│   ├── amg/        fem-amg      — SA-AMG + RS-AMG, Chebyshev smoother, V/W/F cycles (via linger)
│   ├── parallel/   fem-parallel — thread/MPI backends, METIS partitioning, ghost exchange, WASM Workers
│   ├── io/         fem-io       — GMSH .msh v2/v4 reader, VTK .vtu writer/reader, Matrix Market .mtx, HDF5 serial/parallel I/O, XDMF
│   ├── wasm/       fem-wasm     — wasm-bindgen Poisson solver, multi-Worker parallel
│   └── ceed/       fem-ceed     — libCEED-style partial assembly (matrix-free mass/diffusion)
└── examples/       fem-examples — MFEM-style examples + EM simulations + parallel examples
```

---

## Editions

fem-rs is available in two editions:

| Edition | License | Repository | Includes |
|---------|---------|------------|----------|
| **Community** (this repo) | Apache 2.0 | Public GitHub | Complete MFEM-comparable FEM library with WGSL GPU acceleration |
| **Professional** (fem-pro) | Proprietary | Private | GPU (CUDA), T-spline IGA, trimmed NURBS, CAD integration (STEP/IGES), advanced solvers (FETI/BDDC, JDQZ), industry physics, enterprise features |

The community edition is a **fully functional FEM library** — no crippleware.
The professional edition builds on the community via plugin API traits (see [`EDITION_SPLIT.md`](./EDITION_SPLIT.md)).

### MFEM-Style Examples

All examples listed in this section are intended to have a one-to-one correspondence
with MFEM examples.

| Example | PDE | Method | Notes |
|---------|-----|--------|-------|
| `mfem_ex1_poisson` | −Δu = f | H¹ P1, PCG+Jacobi | O(h²) verified |
| `mfem_ex2_elasticity` | −∇·σ = f | VectorH1 P1, PCG | Working |
| `mfem_ex3_maxwell_cavity` | ∇×∇×E + E = f | H(curl) ND1/ND2, PCG+AMS | AMS preconditioner available |
| `mfem_ex4_darcy` | −∇·u = f, u = −κ∇p | H(div) RT0/RT1 + L², MINRES+ADS | ADS preconditioner available |
| `mfem_ex5_mixed_darcy` | Saddle-point Darcy/Stokes | Block PGMRES | Working |
| `mfem_ex7_neumann_mixed_bc` | −Δu = f, mixed BCs | H¹ P1, Neumann + Dirichlet | Working |
| `mfem_ex9_dg_advection` | −Δu = f (DG) | SIP-DG P1, GMRES | O(h²) verified |
| `mfem_ex10_heat_equation` | ∂u/∂t - Δu = 0 | SDIRK-2 + PCG | Working |
| `mfem_ex10_wave_equation` | ∂²u/∂t² - Δu = 0 | Newmark-β + PCG | Working |
| `mfem_ex13_laplacian_eigen` | Kx = λMx (Laplacian) | LOBPCG | 1-D Laplacian eigenvalues verified |
| `mfem_ex13_eigenvalue` | Kx = λMx (Maxwell cavity) | LOBPCG | Cavity resonance verified |
| `mfem_ex14_dc_current` | −∇·(σ∇φ) = 0 | H¹ P1, DC current distribution | Working |
| `mfem_ex15_dg_amr` | −Δu = f (AMR+DG) | P1 + ZZ estimator + Dörfler | O(h²) with refinement |
| `mfem_ex15_tet_nc_amr` | 3-D NC AMR | Tet4 NC refinement + hanging face constraints | Working |
| `mfem_ex16_nonlinear_heat` | −∇·(κ(u)∇u) = f | Newton + GMRES | O(h²) verified |
| `mfem_ex19_navier_stokes` | Navier-Stokes (Kovasznay) | P2/P1 Oseen/Picard, Re=40 | Converged |
| `mfem_ex22_complex_helmholtz` | Complex time-harmonic Helmholtz | Block 2×2 system | Working |
| `mfem_ex23_wave_equation` | ∂²u/∂t² = c²∇²u | Newmark-β / RK4 | Both schemes |
| `mfem_ex24_discrete_ops` | Mixed discrete operators | ∇/∇×/∇· via DiscreteLinearOperator | Gradient/Curl/Div |
| `mfem_ex25_pml_helmholtz` | PML-like damped Helmholtz | Complex damping layer | Working |
| `mfem_ex27_robin_bc` | −Δu = 0, Robin/Neumann/Dirichlet | H¹ P1, mixed BCs | All 4 BC types |
| `mfem_ex31_anisotropic_maxwell` | Anisotropic H(curl) problem | ND1/ND2, PCG+AMS | Working |
| `mfem_ex32_impedance_maxwell` | H(curl) with impedance BC | ND1, impedance boundary | Working |
| `mfem_ex34_absorbing_maxwell` | H(curl) absorbing BC | ND1, first-order ABC | Working |
| `mfem_ex40_stokes` | Stokes lid-driven cavity | Taylor-Hood P2/P1 + Schur GMRES | Verified |

### Parallel Examples

| Example | Problem | Notes |
|---------|---------|-------|
| `mfem_pex1_parallel_poisson` | Parallel Poisson (P1/P2) | PCG+AMG, contiguous/METIS/streaming |
| `mfem_pex2_mixed_darcy` | Parallel mixed Poisson | H(div) × L², block GMRES |
| `mfem_pex3_maxwell_cavity` | Parallel Maxwell | H(curl) ND1, PCG |
| `mfem_pex4_parallel_heat` | Parallel heat equation | Parallel SDIRK-2 |
| `mfem_pex5_hdiv_darcy` | Parallel Darcy | H(div) × L², saddle-point |

**Real MPI (multi-process):** the `mfem_pex*` targets use [`ThreadLauncher`](crates/parallel/src/launcher/native.rs) (in-process “ranks”) so they run without installing MPI. For **OS processes** (`mpiexec` / `mpirun`), build your binary with `fem-parallel`’s `mpi` feature, call [`MpiLauncher`](crates/parallel/src/launcher/native.rs) in each process, and keep the launcher alive for the lifetime of [`Comm`](crates/parallel/src/comm.rs). Partition → [`GhostExchange`](crates/parallel/src/ghost.rs) → [`ParAssembler`](crates/parallel/src/par_assembler.rs) / [`ParCsrMatrix`](crates/parallel/src/par_csr.rs) is unchanged; see the **“MPI (multi-process) scenarios”** section in `cargo doc -p fem-parallel`. Parallel HDF5 checkpoints use `fem-io-hdf5-parallel` with the `hdf5-mpi` feature (`io_hdf5_mpi` on examples).

Dependency order (each crate depends only on crates listed above it):
`core -> mesh/linalg/element -> space -> assembly -> solver/amg -> parallel/io/wasm`

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Rust | >=1.75 stable | `rustup update stable` |
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

# Stokes lid-driven cavity (Taylor-Hood P2/P1)
cargo run --example mfem_ex40_stokes

# Navier-Stokes Kovasznay flow (Re=40)
cargo run --example mfem_ex19_navier_stokes
```

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

Core FEM pipeline (Poisson → Elasticity → Maxwell → Darcy → Stokes → Navier-Stokes →
multi-physics coupling) is complete for simplex meshes.

Known gaps:

| Crate | Status | Highlights |
|-------|--------|------------|
| `fem-core` | ✅ | Scalar traits, FemError, NodeId/DofId, coord aliases |
| `fem-mesh` | ✅ | Mesh, AMR, NCMesh (Tri3/Tet4/Quad4/Hex8), CurvedMesh P2, periodic |
| `fem-element` | ✅ | Lagrange P1–P3 (Seg/Tri/Tet/Quad/Hex + Pk/Qk), Nédélec ND1/ND2 (Tri/Quad/Tet/Hex), RT0/RT1/RT2 (Tri), RT0/RT1 (Tet), NURBS/IGA basis |
| `fem-linalg` | ✅ | CsrMatrix, CooMatrix, Vector, BlockMatrix, DenseTensor |
| `fem-space` | ✅ | H1, L2, VectorH1, HCurl, HDiv, H1Trace, IGA spaces, p-refinement |
| `fem-assembly` | ✅ | Bilinear/linear/mixed/DG/nonlinear/partial assembly, complex systems, Navier-Stokes, phasefield, FSI, thermoelastic, static condensation, de Rham discrete ops |
| `fem-solver` | ✅ | CG/GMRES/BiCGSTAB/IDR(s)/TFQMR/FGMRES, AMS/ADS, sparse direct, LOBPCG/KrylovSchur, ODE (RK/IMEX/BDF/symplectic), DAE, multi-rate, adjoint, coupled Newton |
| `fem-amg` | ✅ | SA-AMG, RS-AMG, Chebyshev smoother, V/W/F cycles |
| `fem-io` | ✅ | GMSH v2/v4, Netgen, Abaqus, VTK, Matrix Market, HDF5/XDMF |
| `fem-parallel` | ✅ | MPI/Thread/WASM backends, METIS, ghost exchange, ParCSR, ParAMG, RAS, checkpoint |
| `fem-wasm` | ✅ | Browser Poisson solver, multi-Worker parallel |
