# fem-rs Edition Split: Open-Source Community vs Professional

**Date:** 2026-07-12
**Status:** Approved
**Author:** User + Claude

## Motivation

Split fem-rs into two editions:

- **Open-Source Edition** (Apache 2.0, GitHub public) — a complete, MFEM-comparable FEM library for research, education, and community use
- **Professional Edition** (private repo) — industrial-grade extensions for commercial CAE, built on top of the open-source core via a plugin/submodule architecture

This enables a dual-track strategy: **business model** (open core drives adoption, pro edition generates revenue) + **dual-track R&D** (pro can iterate fast without destabilizing the stable OSS core, and innovations can migrate down when mature).

## Relationship between Editions

```
fem-rs (GitHub, Apache 2.0)
  ├── Complete MFEM-equivalent FEM library
  ├── All crates fully open source
  ├── Clean, standalone repo
  └── Plugin API traits (no-op when empty)

fem-pro (private repo)
  ├── fem-rs/                    # git submodule
  ├── Cargo.toml                 # workspace includes fem-rs + pro crates
  ├── crates/
  │   ├── pro-gpu/               # GPU-accelerated assembly & solvers
  │   ├── pro-iga/               # T-spline, trimmed NURBS, C¹/C², IGA contact
  │   ├── pro-solver/            # FETI/BDDC, JDQZ, GPU-native solvers
  │   ├── pro-cad/               # OpenCASCADE bindings, STEP/IGES import
  │   ├── pro-physics/           # Forming, NVH, fatigue, EDA mult-physics
  │   ├── pro-enterprise/        # REST API, cloud HPC, workflow, license mgmt
  │   └── pro-plugin-host/       # Registry init, .so/.dll loader
  └── examples/
      └── pro-*/                 # Pro edition examples
```

Building fem-pro: `git clone`, `git submodule update --init`, `cargo build --workspace`.

Building fem-rs standalone (anyone): `git clone`, `cargo test --workspace`.

## Licensing

| Edition | License | Distribution |
|---------|---------|-------------|
| fem-rs  | Apache 2.0 | Public GitHub |
| fem-pro | Proprietary, per-seat or site license | Private repository, binary distribution via crate registry or .so/.dll |

## Function Division

### Open-Source Edition (fem-rs): Everything Currently Implemented Except GPU

The open-source edition is a **complete, usable FEM library** — an MFEM-equivalent in Rust. Any researcher or engineer can clone it and solve real PDE problems.

| Crate | Status | Notes |
|-------|--------|-------|
| `fem-core` | ✅ Open | Types, indices, traits, error |
| `fem-mesh` | ✅ Open | Simplex/Quad/Hex/Curved/AMR/NCMesh, periodic, generators |
| `fem-element` | ✅ Open | Lagrange P1–Pk, Nédélec ND1/ND2, RT0–RT2, BDM, serendipity, hierarchical, Crouzeix-Raviart, Bernstein, NURBS basis, quadrature |
| `fem-space` | ✅ Open | H¹, L2, HCurl, HDiv, VectorH1, H1Trace, IGA FE space, p-refinement |
| `fem-linalg` | ✅ Open | CsrMatrix, CooMatrix, Vector, BlockMatrix, DenseTensor |
| `fem-solver` | ✅ Open | CG/PCG, GMRES/FGMRES, BiCGSTAB, IDR(s), TFQMR, MINRES, LOBPCG, KrylovSchur, ODE (RK/BDF/SDIRK/IMEX/symplectic), block solvers, AMS/ADS, direct solvers |
| `fem-amg` | ✅ Open | SA-AMG, RS-AMG, Chebyshev smoother, V/W/F cycles |
| `fem-parallel` | ✅ Open | MPI/Thread/WASM backends, METIS partitioning, ghost exchange, ParCSR, checkpoint |
| `fem-io` | ✅ Open | GMSH v2/v4, Netgen, Abaqus, VTK, Matrix Market, HDF5, XDMF |
| `fem-wasm` | ✅ Open | Browser solver, multi-Worker parallel |
| `fem-py` | ✅ Open | Python bindings via PyO3 |
| `fem-stochastic` | ✅ Open | Stochastic FEM |
| `fem-regression` | ✅ Open | Regression testing framework |
| `linalg-gpu` | ❌ **Move to pro** | GPU assembly is a key pro differentiator |

#### IGA — Partial Open (MFEM-Equivalent Level)

| Feature | Edition | Reasoning |
|---------|---------|-----------|
| NURBS basis, DegreeElevate | ✅ Open | Textbook material; MFEM has this |
| Knot insertion (h-refinement) | ✅ Open | MFEM has this |
| NURBS .mesh parser | ✅ Open | File format parsing |
| IgaFESpace 1D/2D/3D | ✅ Open | FE space collection |
| Single-patch assembly | ✅ Open | Basic diffusion/mass/load |
| Bezier extraction math | ✅ Open | Published algorithm |
| C⁰ multi-patch stitching | ✅ Open | Basic connectivity |
| T-spline engine | ❌ Pro | Coreform-level differentiator |
| Trimmed NURBS + CAD kernel | ❌ Pro | Industry CAD integration |
| Multi-patch C¹/C² coupling | ❌ Pro | High-order continuity |
| IGA contact (frictional) | ❌ Pro | Industrial forming |
| IGA shape/topology optimization | ❌ Pro | Design-through-analysis |
| IGA shell (Kirchhoff-Love) | ❌ Pro | Thin-shell analysis |
| Non-uniform Bezier extraction (C≠I) | ❌ Pro | General case |

#### Examples

| Category | Edition | Count |
|----------|---------|-------|
| Serial MFEM examples (ex1–ex53) | ✅ Open | 53 |
| Parallel MFEM examples (pex*) | ✅ Open | 27 |
| Basic IGA examples (single-patch) | ✅ Open | 5 |
| Pro IGA examples | ❌ Pro | T-spline, trimmed, optimization |
| Pro physics examples | ❌ Pro | Forming, NVH, fatigue |

### Professional Edition (fem-pro): Value-Add Extensions

Sold to commercial/industrial users (SMEs, aerospace, automotive, EDA/semiconductor). Not for academic teams — academic users get the full open-source edition with research-level capabilities.

#### pro-gpu — GPU Acceleration

| Feature | Status in Prototype |
|---------|-------------------|
| WGSL compute shaders for 2D/3D diffusion+mass assembly | ✅ Already prototyped (move from linalg-gpu) |
| WGSL shaders for 2D/3D elasticity assembly | Pending |
| CUDA backend via `cust` | Feature-gated (already exists) |
| GPU-native AMG | Pending |
| GPU batched solvers (many-RHS) | Pending |

#### pro-iga — Advanced IGA

All features beyond basic NURBS that MFEM does not provide.

#### pro-solver — Advanced Solvers

| Solver | Notes |
|--------|-------|
| FETI/BDDC domain decomposition | Scalable to 100k+ cores |
| JDQZ contour integral eigenvalue | Robust for large Maxwell eigenproblems |
| Parallel-in-time (ParaReal, MGRIT) | Speed up transient problems |
| GPU-accelerated AMG | AmgX-class performance |
| Nonlinear hyper-reduction | Real-time capable ROM |

#### pro-cad — CAD Integration

| Feature | Notes |
|---------|-------|
| OpenCASCADE bindings | STEP/IGES direct import |
| Geometry defeaturing & cleanup | Remove small features automatically |
| Associative parametric updates | CAD change → remesh → resolve |
| Mesh morphing | Shape optimization without remeshing |

#### pro-physics — Industry Physics Modules

| Module | Target Industry |
|--------|----------------|
| Advanced plasticity + damage + fracture | Forming, crashworthiness |
| NVH/Acoustics (frequency sweep, modal superposition, SEA) | Automotive |
| Fatigue life prediction (S-N, ε-N, DVS) | Aerospace, automotive |
| Forming simulation (contact + large deformation + plasticity) | Manufacturing |
| EDA: EM-thermal-stress coupled | Semiconductor |
| XFEM/GFEM for crack propagation | Fracture mechanics |

#### pro-enterprise — Enterprise Features

| Feature | Notes |
|---------|-------|
| REST/gRPC server | Remote solve API |
| Cloud HPC orchestration (AWS Batch, Slurm) | Auto-scaling |
| Workflow pipeline engine | Multi-step simulation |
| Result database & analytics | Compare runs |
| License management | Floating/Node-locked |
| Commercial support SLA | Priority support channel |

## Plugin API Design

The open-source edition (fem-rs) defines trait-based extension points. These are compiled into the OSS library but have no-op behavior when no pro plugins are registered.

### Solver Extension (in `fem-solver`)

```rust
/// Trait for externally-registered solvers.
pub trait ProSolver: Send + Sync {
    fn name(&self) -> &str;
    fn solve(&self, matrix: &CsrMatrix<f64>, rhs: &Vector<f64>)
        -> Result<Vector<f64>, SolverError>;
}

/// Global solver registry.
pub struct SolverRegistry { /* HashMap<String, Box<dyn ProSolver>> */ }

impl SolverRegistry {
    pub fn global() -> &'static Mutex<Self>;
    pub fn register(&mut self, solver: Box<dyn ProSolver>);
    pub fn get(&self, name: &str) -> Option<&dyn ProSolver>;
}
```

### Integrator Extension (in `fem-assembly`)

```rust
/// Trait for externally-registered element integrators.
pub trait ProIntegrator: Send + Sync {
    fn name(&self) -> &str;
    fn assemble_element_matrix(
        &self,
        element: &ElementInfo,
        trial: &FiniteElement,
        test: &FiniteElement,
    ) -> Result<DMatrix<f64>, AssemblyError>;
}

/// Trait for externally-registered mesh modifiers (CAD defeaturing, morphing).
pub trait ProMeshModifier: Send + Sync {
    fn name(&self) -> &str;
    fn modify(&self, mesh: &mut Mesh<3>) -> Result<(), MeshError>;
}
```

### Mesh Import Extension (in `fem-io`)

```rust
/// Trait for externally-registered mesh/geometry importers.
pub trait ProMeshImporter: Send + Sync {
    fn name(&self) -> &str;
    fn can_import(&self, path: &Path) -> bool;
    fn import(&self, path: &Path) -> Result<MeshData, IoError>;
}
```

### Physics Extension (in `fem-assembly`)

```rust
/// Trait for externally-registered nonlinear physics models.
pub trait ProPhysicsModel: Send + Sync {
    fn name(&self) -> &str;
    fn compute_residual(&self, state: &Vector<f64>) -> Result<Vector<f64>>;
    fn compute_tangent(&self, state: &Vector<f64>) -> Result<CsrMatrix<f64>>;
}
```

### User-Facing API

```rust
// Open-source edition: built-in solvers only.
use fem_rs::prelude::*;
let solver = CGSolver::new(pc);
solver.solve(&matrix, &rhs)?;

// Professional edition: fem-pro prelude re-exports fem-rs + pro.
use fem_pro::prelude::*;
let solver = SolverRegistry::global().get("feti-bddc")?;
solver.solve(&matrix, &rhs)?;
```

## Migration Plan

### Step 1: Create fem-pro Repository
- Initialize private repo
- Add fem-rs as git submodule
- Create workspace Cargo.toml referencing both fem-rs crates and pro crates

### Step 2: Move GPU Crate to Pro
- `crates/linalg-gpu/` → `fem-pro/crates/pro-gpu/`
- `crates/assembly/src/iga/iga_gpu.rs` → `fem-pro/crates/pro-gpu/`

### Step 3: Move Advanced IGA to Pro
- `crates/element/src/tmesh.rs` → `fem-pro/crates/pro-iga/`
- `crates/assembly/src/iga/iga_tspline.rs` → `fem-pro/crates/pro-iga/`
- `crates/assembly/src/contact/contact_iga.rs` → `fem-pro/crates/pro-iga/`
- `crates/assembly/src/iga/iga_trim.rs` + `step_iges.rs` → `fem-pro/crates/pro-cad/`

### Step 4: Add Plugin API Traits to fem-rs
- Add `plugin.rs` modules to `fem-solver`, `fem-assembly`, `fem-io`
- Each defines traits + empty registry
- No new dependencies

### Step 5: Implement Pro Crate Registrations
- `pro-plugin-host` calls `SolverRegistry::global().register(...)` at startup
- All pro crates depend only on fem-rs traits + their own implementation

### Step 6: Update Open-Source Edition
- Remove moved files from workspace members
- Update README with edition information
- Keep all 2491+ tests passing
- Document plugin API for community contributors

### Timeline Estimate

| Step | Effort |
|------|--------|
| 1. Repo + submodule setup | 1-2 hours |
| 2. Move GPU crate | 2-3 hours |
| 3. Move advanced IGA | 1-2 hours |
| 4. Plugin API traits | 4-6 hours (design + test) |
| 5. Pro registrations | 2-3 hours |
| 6. OSS edition cleanup | 2 hours |
| **Total** | **~12-16 hours** |

## Key Design Principles

1. **Open-source edition is complete and useful alone.** No "crippleware" — Apache 2.0 users get a full MFEM-comparable FEM library. The pro edition *adds* industrial value, it doesn't *fix* a broken OSS.

2. **Plugin API is zero-overhead.** Trait dispatch is a single vtable lookup. When no pro plugins are registered, behavior is identical to today's fem-rs.

3. **Clean IP boundary.** fem-rs contains zero pro code. fem-pro references fem-rs via submodule + path dependencies. No feature gates or `#[cfg]` conditional compilation leaking pro features into OSS.

4. **Dual-track R&D.** Pro can try experimental features (new solvers, new physics) without destabilizing the community build. Mature innovations can be migrated to OSS later.

5. **Market positioning.** Open-source: "Rust FEM library, MFEM-comparable, Apache 2.0." Professional: "Industrial CAE solver with GPU acceleration, CAD integration, and commercial support."
