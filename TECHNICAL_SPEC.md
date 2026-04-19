# fem-rs Technical Specification & Programming Guidelines
> Version: 0.1.0 | Target: AI Agent Reference Document

---

## 1. Project Identity

| Item | Value |
|------|-------|
| Language | Rust (edition 2021) |
| Build System | Cargo workspace |
| Comparable to | MFEM (C++), FEniCS (Python), deal.II (C++) |
| Primary targets | native (x86_64/aarch64), wasm32-unknown-unknown |
| MPI library | `mpi` crate (wraps OpenMPI/MPICH) |
| Minimum Rust | 1.75 (stable, for const generics + async traits) |

---

## 2. Workspace Layout

```
fem-rs/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── core/               # fundamental traits, numeric types, error
│   ├── mesh/               # mesh topology, geometry, AMR, CurvedMesh
│   ├── element/            # FE basis functions (Lagrange, Nedelec, RT), quadrature
│   ├── space/              # DOF management, FE spaces (H1, L2, HCurl, HDiv, VectorH1)
│   ├── assembly/           # bilinear/linear/mixed/DG/nonlinear form assembly
│   ├── linalg/             # CSR/COO matrix, dense vector, BlockMatrix/BlockVector
│   ├── solver/             # iterative solvers (CG, GMRES, BiCGSTAB), Newton, ODE, LOBPCG
│   ├── amg/                # algebraic multigrid (SA-AMG + RS-AMG via linger)
│   ├── parallel/           # thread/MPI backends, METIS partitioning, ghost exchange
│   ├── io/                 # GMSH .msh v4 reader, VTK .vtu XML writer
│   ├── wasm/               # wasm32 bindings (wasm-bindgen), Poisson solver
│   └── ceed/               # libCEED-style partial assembly operators (via reed)
├── examples/               # 11 runnable FEM examples
├── vendor/
│   ├── linger/             # Krylov solvers + AMG engine
│   ├── reed/               # libCEED analogue (operator decomposition)
│   └── rmetis/             # pure-Rust METIS-compatible graph partitioner
├── benches/
└── tests/                      # integration tests
```

**Dependency DAG** (→ means "depends on"):
```
fem-wasm      → fem-assembly, fem-io, fem-solver
fem-parallel  → fem-assembly, fem-linalg, fem-amg
fem-ceed      → fem-assembly, fem-element, fem-linalg (bridges to vendor/reed)
fem-amg       → fem-linalg (bridges to vendor/linger)
fem-solver    → fem-linalg (bridges to vendor/linger)
fem-assembly  → fem-element, fem-space, fem-linalg
fem-space     → fem-element, fem-mesh, fem-core
fem-element   → fem-core
fem-mesh      → fem-core
fem-linalg    → fem-core
fem-io        → fem-mesh
fem-core      (no internal deps)
```

---

## 3. Feature Flags

Every crate that has conditional compilation must declare its features in Cargo.toml. **Agents must respect these flags** when writing code.

```toml
# parallel/Cargo.toml
[features]
default = []
mpi     = ["dep:mpi", "fem-assembly/parallel"]

# amg/Cargo.toml
[features]
default = []
suitesparse = ["dep:suitesparse-sys"]

# wasm/Cargo.toml — NEVER enable mpi here
[features]
default = ["wasm-bindgen"]
```

Guard rules in code:
```rust
#[cfg(feature = "mpi")]
use fem_parallel::Communicator;

#[cfg(target_arch = "wasm32")]
compile_error!("`fem-parallel` cannot target wasm32");
```

---

## 4. Core Numeric Conventions

### 4.1 Scalar Type
```rust
// core/src/scalar.rs
pub trait Scalar:
    Copy + Clone + Send + Sync + 'static
    + std::fmt::Debug
    + num_traits::Float
    + num_traits::NumAssign
{}
impl Scalar for f32 {}
impl Scalar for f64 {}

// Default throughout: f64. Generic over <T: Scalar> only when compile-time polymorphism is needed.
// Do NOT use f32 unless targeting WASM performance-critical paths.
```

### 4.2 Coordinate / Vector Types
Use `nalgebra` for small fixed-size vectors (reference coordinates, Jacobians).
Use custom `Vector<T>` (wrapping `Vec<T>`) for global DOF vectors.

```rust
use nalgebra::{Point2, Point3, SMatrix, SVector};

pub type Coord2 = Point2<f64>;
pub type Coord3 = Point3<f64>;
pub type Mat3x3 = SMatrix<f64, 3, 3>;
```

**Do NOT use ndarray for FEM computation** — nalgebra's stack-allocated types have zero overhead for small matrices.

### 4.3 Index Types
```rust
pub type NodeId  = u32;   // mesh node index
pub type ElemId  = u32;   // mesh element index
pub type DofId   = u32;   // degree of freedom index
pub type FaceId  = u32;
```
Use `u32` not `usize` for mesh indices to halve memory in large meshes and allow serialization parity across 32/64-bit.

---

## 5. Key Trait Interfaces

> Agents: implement concrete types by satisfying these traits. Do not change trait signatures without updating this spec.

### 5.1 Mesh Topology
```rust
// mesh/src/topology.rs
pub trait MeshTopology: Send + Sync {
    fn dim(&self) -> u8;                          // spatial dimension
    fn n_nodes(&self) -> usize;
    fn n_elements(&self) -> usize;
    fn element_nodes(&self, elem: ElemId) -> &[NodeId];
    fn element_type(&self, elem: ElemId) -> ElementType;
    fn node_coords(&self, node: NodeId) -> &[f64]; // len == dim
    fn boundary_faces(&self) -> &[FaceId];
    fn face_nodes(&self, face: FaceId) -> &[NodeId];
    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>); // (interior, exterior)
}
```

### 5.2 Reference Element
```rust
// element/src/lib.rs
pub trait ReferenceElement: Send + Sync {
    fn dim(&self) -> u8;
    fn order(&self) -> u8;
    fn n_dofs(&self) -> usize;
    // xi: reference coordinates (len == dim)
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]);
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]); // row-major [n_dofs × dim]
    fn quadrature(&self, order: u8) -> QuadratureRule;
}

pub struct QuadratureRule {
    pub points:  Vec<Vec<f64>>,  // [n_pts][dim]
    pub weights: Vec<f64>,
}
```

### 5.3 Finite Element Space
```rust
// fem-space/src/lib.rs
pub trait FESpace: Send + Sync {
    type Mesh: MeshTopology;
    fn mesh(&self) -> &Self::Mesh;
    fn n_dofs(&self) -> usize;
    fn element_dofs(&self, elem: ElemId) -> &[DofId];
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64>;
    fn space_type(&self) -> SpaceType;
    fn order(&self) -> u8;
    /// Orientation signs (±1.0) for H(curl)/H(div) DOFs; None for H1/L2.
    fn element_signs(&self, _elem: u32) -> Option<&[f64]> { None }
}

pub enum SpaceType { H1, Hdiv, Hcurl, L2, VectorH1(u8) }
```

### 5.4 Integrators (Forms)
```rust
// assembly/src/integrator.rs

/// Contributes to the element stiffness matrix K_e (n_dofs × n_dofs)
pub trait BilinearIntegrator: Send + Sync {
    fn assemble_element(
        &self,
        elem: ElemId,
        space: &dyn FESpace<Mesh = dyn MeshTopology>,
        k_elem: &mut [f64],   // row-major, n_dofs_u × n_dofs_v
    );
}

/// Contributes to the element load vector f_e (n_dofs)
pub trait LinearIntegrator: Send + Sync {
    fn assemble_element(
        &self,
        elem: ElemId,
        space: &dyn FESpace<Mesh = dyn MeshTopology>,
        f_elem: &mut [f64],
    );
}
```

### 5.5 Sparse Matrix
```rust
// linalg/src/csr.rs
pub struct CsrMatrix<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,   // len = nrows + 1
    pub col_idx: Vec<u32>,
    pub values:  Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn spmv(&self, x: &[T], y: &mut [T]);       // y = Ax
    pub fn spmv_add(&self, alpha: T, x: &[T], beta: T, y: &mut [T]); // y = αAx + βy
    pub fn transpose(&self) -> Self;
    pub fn to_dense(&self) -> Vec<T>;                // debug only
}
```

### 5.6 Linear Solver
```rust
// solver/src/lib.rs
pub trait LinearSolver: Send + Sync {
    fn solve(
        &mut self,
        mat: &CsrMatrix<f64>,
        rhs: &[f64],
        sol: &mut [f64],
        tol: f64,
        max_iter: usize,
    ) -> SolverResult;
}

pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}
```

---

## 6. Assembly Pipeline

Agents implementing assembly must follow this exact pipeline:

```
1. Mesh iteration       for elem in 0..mesh.n_elements()
2. DOF extraction       dofs = space.element_dofs(elem)
3. Quadrature loop      for (xi, w) in rule.iter()
4. Jacobian             J = ∂x/∂ξ  (physical ← reference)
5. Basis evaluation     φ_i(ξ), ∇_ξ φ_i(ξ)
6. Physical gradient    ∇_x φ_i = J^{-T} ∇_ξ φ_i
7. Integrator call      k_ij += w * det(J) * integrator_kernel(φ_i, φ_j, ...)
8. Global scatter       K[dofs[i], dofs[j]] += k_ij
```

The scatter step uses a `Assembler` struct that owns a `SparsityPattern` built once before assembly:
```rust
// assembly/src/assembler.rs
pub struct Assembler {
    sparsity: SparsityPattern,
}
impl Assembler {
    pub fn new(space: &dyn FESpace<Mesh=dyn MeshTopology>) -> Self;
    pub fn assemble_bilinear(
        &self, integrators: &[&dyn BilinearIntegrator]
    ) -> CsrMatrix<f64>;
    pub fn assemble_linear(
        &self, integrators: &[&dyn LinearIntegrator]
    ) -> Vec<f64>;
}
```

---

## 7. AMG Design (fem-amg)

Native AMG pure-Rust path. Structure:

```
AmgSolver
├── setup_phase(mat: &CsrMatrix) → AmgHierarchy
│   ├── strength_of_connection(theta=0.25) → ConnectionGraph
│   ├── coarsening: Ruge-Stüben C/F splitting
│   ├── interpolation: classical or smoothed aggregation
│   └── Galerkin coarse: A_c = R A P
└── solve_phase(hierarchy, rhs) → sol
    └── V-cycle / W-cycle / F-cycle
```

Key types:
```rust
pub struct AmgHierarchy {
    pub levels: Vec<AmgLevel>,
}
pub struct AmgLevel {
    pub a: CsrMatrix<f64>,   // system matrix at this level
    pub p: CsrMatrix<f64>,   // prolongation
    pub r: CsrMatrix<f64>,   // restriction = P^T
    pub smoother: SmootherKind,
}
pub enum SmootherKind {
    Jacobi { omega: f64 },
    GaussSeidel,
    ChebyshevJacobi { degree: u8 },
}
```

---

## 8. MPI Parallelism (fem-parallel)

### Distributed Mesh
```rust
// parallel/src/mesh.rs
pub struct ParallelMesh<M: MeshTopology> {
    local_mesh: M,
    comm: Communicator,
    // global-to-local and local-to-global index maps
    node_global_ids: Vec<u32>,
    elem_global_ids: Vec<u32>,
    shared_nodes: HashMap<NodeId, Vec<Rank>>,  // ghost info
}
```

### Parallel Assembly Pattern
1. Each rank assembles its local elements.
2. Exchange ghost DOF contributions via `AllReduce` / point-to-point.
3. Build `ParCsrMatrix` with local row ownership.

```rust
pub struct ParCsrMatrix {
    diag:    CsrMatrix<f64>,   // owned rows × owned cols
    off_diag: CsrMatrix<f64>, // owned rows × ghost cols
    col_map: Vec<u32>,         // ghost local → global
    comm: Communicator,
}
impl ParCsrMatrix {
    pub fn par_spmv(&self, x: &ParVector, y: &mut ParVector);
}
```

### Parallel AMG
Wrap `fem-amg` logic with `ParCsrMatrix`; coarse-grid solve on rank 0 or via recursive partitioning.

---

## 9. WASM Target Rules

**Mandatory restrictions** for code in `fem-wasm` and any code compiled for wasm32:
- NO `std::thread`, NO `rayon`, NO `mpi`
- NO `libc` or OS filesystem calls — use `web-sys` for I/O
- Panic handler must be set: `console_error_panic_hook`
- Use `wasm-bindgen` for JS interop
- Serialization via `serde-wasm-bindgen` + `serde`
- Float: prefer `f64` (JS numbers are f64 anyway)
- Mesh/solution data exchange: `Float64Array` / `Uint32Array`

```rust
// wasm/src/lib.rs
#[wasm_bindgen]
pub struct WasmSolver {
    assembler: Assembler,
    solver: Box<dyn LinearSolver>,
}

#[wasm_bindgen]
impl WasmSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(mesh_json: &str) -> Result<WasmSolver, JsValue>;
    pub fn solve(&mut self) -> Float64Array;
}
```

Build command: `cargo build --target wasm32-unknown-unknown -p fem-wasm --no-default-features`

---

## 10. Error Handling

```rust
// core/src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum FemError {
    #[error("mesh error: {0}")]
    Mesh(String),
    #[error("DOF mapping inconsistency: elem {elem}, dof {dof}")]
    DofMapping { elem: usize, dof: usize },
    #[error("solver did not converge after {0} iterations")]
    SolverDivergence(usize),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
}

pub type FemResult<T> = Result<T, FemError>;
```

Rules:
- All public API functions return `FemResult<T>`.
- Use `?` propagation; never `.unwrap()` in library code.
- `.expect()` is allowed only in tests and examples.

---

## 11. Testing Requirements

| Layer | Test type | Location |
|-------|-----------|----------|
| `fem-core` | unit | `crates/core/src/` inline |
| `fem-element` | convergence order test | `crates/element/tests/` |
| `fem-assembly` | patch test (constant strain) | `crates/assembly/tests/` |
| `fem-solver` | solve Laplacian on unit square, check L2 error | `crates/solver/tests/` |
| `fem-amg` | compare with direct solver on SPD systems | `crates/amg/tests/` |
| integration | Poisson, elasticity, Stokes on 2D/3D meshes | `tests/` |

**Convergence test pattern** (mandatory for element and solver crates):
```rust
#[test]
fn h_convergence_laplacian() {
    let errors: Vec<f64> = [4, 8, 16, 32].map(|n| solve_and_compute_error(n));
    let rates = convergence_rates(&errors);
    for rate in &rates { assert!(*rate > 1.9, "expected O(h^2), got {rate}"); }
}
```

---

## 12. Performance Rules

1. **No allocation in hot paths.** Assembly inner loop must not allocate. Pre-allocate element buffers before the loop.
2. **SIMD via `std::simd` or `packed_simd2`** for basis function evaluation if order ≤ 4.
3. **`rayon` par_iter** over elements in non-MPI builds. Partition elements into chunks ≥ 256 to amortize thread overhead.
4. **Sparsity pattern built once.** `Assembler::new` builds the pattern; `assemble_bilinear` only fills values.
5. **Profile-guided**: bench targets in `benches/` using `criterion`. Minimum benches: assembly (Poisson, 3D hex, 1M DOFs), AMG setup, CG solve.

---

## 13. Dependency Policy

| Crate | Allowed | Forbidden |
|-------|---------|-----------|
| all | `thiserror`, `log`, `num-traits` | `anyhow` in library code |
| fem-core | `nalgebra`, `bytemuck` | — |
| fem-linalg | `rayon` (non-wasm) | `ndarray` |
| fem-solver | — | direct LAPACK calls (use trait) |
| fem-parallel | `mpi` (feature-gated) | anything without `Send+Sync` |
| fem-wasm | `wasm-bindgen`, `web-sys`, `serde-wasm-bindgen` | `mpi`, `rayon`, `libc` |
| fem-io | `hdf5`, `quick-xml` | — |

---

## 14. Coding Conventions for Agents

1. **Trait objects**: prefer `Box<dyn Trait>` for owned singletons; `&dyn Trait` for borrowing; `Arc<dyn Trait>` only when shared across threads.
2. **Generics vs trait objects**: use generics (`impl Trait`) for integrators in assembly inner loop (monomorphization = zero overhead); use trait objects for solvers (runtime selection).
3. **Module structure**: each module = one file; re-export public API from `lib.rs`. No `mod.rs` files.
4. **No implicit panics**: replace `.unwrap()` with `.ok_or(FemError::...)?`
5. **Lifetime naming**: `'mesh` for mesh borrows, `'a` for generic short lifetimes.
6. **Documentation**: every public trait and struct gets a `///` doc comment explaining its mathematical meaning, not just its code meaning.
7. **const generics**: use `const D: usize` for spatial dimension when the entire computation is dimension-generic. Example: `struct SimplexMesh<const D: usize>`.

---
