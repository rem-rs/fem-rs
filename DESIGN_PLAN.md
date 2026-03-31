# rs Design Plan
> Version: 0.1.0 | AI Agent Implementation Roadmap

---

## Implementation Status

| Phase | Crate | Status | Date | Notes |
|-------|-------|--------|------|-------|
| 0 | workspace | ✅ Done | 2026-03-30 | 11-crate workspace, toolchain, cargo aliases |
| 1 | core | ✅ Done | 2026-03-30 | Scalar, FemError, NodeId/DofId, nalgebra re-exports |
| 2 | mesh | ✅ Done | 2026-03-30 | SimplexMesh\<D\>, MeshTopology, unit_square_tri generator |
| 3 | element | ✅ Done | 2026-03-31 | ReferenceElement trait; SegP1/P2, TriP1/P2, TetP1, QuadQ1, HexQ1; 26 tests |
| 4 | linalg | ✅ Done | 2026-03-31 | CsrMatrix, CooMatrix, Vector; + SparsityPattern, dense LU; 16 tests |
| 5 | space | ✅ Done | 2026-03-31 | H1Space(P1/P2), L2Space(P0/P1), DofManager, apply_dirichlet, boundary_dofs; 18 tests |
| 6 | assembly | ✅ Done | 2026-03-31 | Assembler, BilinearIntegrator/LinearIntegrator/BoundaryLinearIntegrator; DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator, NeumannIntegrator; P1/P2 Poisson verified (rate=2.0/3.0); 12 tests |
| 7 | solver | ✅ Done | 2026-03-31 | Backed by linger: CG, PCG+Jacobi, PCG+ILU0, GMRES, BiCGSTAB; end-to-end Poisson solve verified (all 5 solvers) |
| 8 | amg | ✅ Done | 2026-03-31 | Backed by linger: SA-AMG + RS-AMG, AmgSolver (reusable hierarchy); AMG-CG fewer iters than CG on 64×64 mesh (38 vs 84) |
| 9 | io | ✅ Done | 2026-03-31 | GMSH v4.1 ASCII reader + VTK .vtu XML writer; unit_cube_tet generator added to fem-mesh; 12 tests |
| 10 | parallel | ✅ Done | 2026-03-31 | ChannelBackend (in-process multi-threading), ThreadLauncher (n-worker), GhostExchange (alltoallv + forward/reverse), NativeMpiBackend::alltoallv_bytes; 20 tests (12 thread + 8 ghost) |
| 11 | wasm | ✅ Done | 2026-03-31 | WasmSolver (unit-square P1 Poisson, wasm-bindgen optional), assemble_constant_rhs / assemble_nodal_rhs / solve / node_coords / connectivity; 7 native tests |

### Vendor submodules
| Submodule | URL | Role |
|-----------|-----|------|
| `vendor/reed` | javagg/reed | libCEED analogue; bridged via `crates/ceed` |
| `vendor/linger` | javagg/linger | Krylov solvers + AMG; drives `fem-solver` and `fem-amg` |

---

## Phase 0: Workspace Bootstrap

**Goal**: Compilable workspace skeleton, all crates registered, CI green.

### Tasks
1. Create `Cargo.toml` workspace root listing all 11 crates.
2. Create stub `lib.rs` for each crate (empty `pub mod` + re-exports).
3. Configure `rust-toolchain.toml` (stable, 1.75+).
4. Add `.cargo/config.toml` with target aliases:
   ```toml
   [alias]
   wasm-build = "build --target wasm32-unknown-unknown -p wasm --no-default-features"
   check-all  = "check --workspace --all-features"
   ```
5. Set up GitHub Actions / local CI: `cargo check --workspace`, `cargo test --workspace`, `cargo clippy`.

### Deliverables
- `Cargo.toml` (workspace)
- `crates/*/Cargo.toml` (each crate with correct inter-crate deps)
- `rust-toolchain.toml`

---

## Phase 1: core

**Depends on**: nothing internal

### Modules to implement
```
core/src/
├── lib.rs
├── scalar.rs        # Scalar trait (f32/f64)
├── error.rs         # FemError enum, FemResult<T>
├── types.rs         # NodeId, ElemId, DofId, FaceId type aliases
└── point.rs         # re-export nalgebra Point2/Point3 with convenience impls
```

### Acceptance criteria
- `Scalar` trait implemented for `f32` and `f64`
- All error variants compile
- Zero warnings with `clippy`

---

## Phase 2: mesh

**Depends on**: `core`

### Modules
```
mesh/src/
├── lib.rs
├── topology.rs      # MeshTopology trait
├── element_type.rs  # ElementType enum: Tri3, Tri6, Quad4, Tet4, Hex8, ...
├── simplex.rs       # SimplexMesh<const D: usize>: concrete unstructured mesh
├── structured.rs    # StructuredMesh: uniform Cartesian grid (fast prototyping)
├── refine.rs        # uniform refinement (bisection for simplex)
└── boundary.rs      # BoundaryCondition marker, face group labeling
```

### Key design: `SimplexMesh<D>`
```rust
pub struct SimplexMesh<const D: usize> {
    coords:     Vec<f64>,          // flat: [x0,y0,..., x1,y1,...]
    conn:       Vec<NodeId>,       // flat element connectivity
    elem_type:  ElementType,       // uniform type per mesh
    face_conn:  Vec<NodeId>,
    face_bc:    Vec<BoundaryTag>,
}
```
- `D=2`: 2D triangular/quad mesh
- `D=3`: 3D tetrahedral/hex mesh

### Acceptance criteria
- Can build a unit-square triangular mesh (2×2 squares split into triangles)
- Can build a unit-cube tetrahedral mesh
- Uniform refinement halves `h_max`
- All `MeshTopology` trait methods implemented and tested

---

## Phase 3: element

**Depends on**: `core`

### Modules
```
element/src/
├── lib.rs
├── reference.rs     # ReferenceElement trait, QuadratureRule
├── quadrature.rs    # Gauss-Legendre tables (orders 1–10) for line/tri/tet/quad/hex
├── lagrange/
│   ├── mod.rs
│   ├── seg.rs       # P1, P2 on [0,1]
│   ├── tri.rs       # P1, P2, P3 on reference triangle
│   ├── tet.rs       # P1, P2 on reference tetrahedron
│   ├── quad.rs      # Q1, Q2 (tensor product)
│   └── hex.rs       # Q1, Q2 (tensor product)
├── nedelec/         # H(curl) elements (Phase 5)
└── raviart_thomas/  # H(div) elements (Phase 5)
```

### Acceptance criteria
- Partition of unity: `sum_i φ_i(xi) == 1` for all quadrature points
- Reproducing polynomials: for P_k element, any degree-k polynomial is exactly represented
- Convergence test: `h_convergence_laplacian()` passes for P1 and P2

---

## Phase 4: linalg

**Depends on**: `core`

### Modules
```
linalg/src/
├── lib.rs
├── csr.rs           # CsrMatrix<T>: spmv, transpose, add, scale
├── coo.rs           # CooMatrix<T>: for incremental assembly → convert to CSR
├── vector.rs        # Vector<T> = newtype over Vec<T>, with axpy, dot, norm
├── sparsity.rs      # SparsityPattern: build from DOF connectivity
└── dense.rs         # small dense ops needed by AMG (local LU, etc.)
```

### CooMatrix → CsrMatrix assembly pattern
```
1. Collect (i, j, v) triples in CooMatrix during assembly
2. Sort by (i, j)
3. Merge duplicate (i,j) entries by summing values
4. Compress to CSR
```
This is the canonical FEM assembly approach; do not deviate.

### Acceptance criteria
- `spmv` correctness verified against dense multiply for 10×10 random SPD
- Assembly of Laplacian on 4-element mesh matches analytical stiffness matrix
- `rayon` feature: parallel `spmv` over rows, speedup on 8+ cores

---

## Phase 5: space

**Depends on**: `core`, `mesh`, `element`

### Modules
```
space/src/
├── lib.rs
├── fe_space.rs      # FESpace trait
├── h1.rs            # H1Space: scalar Lagrange, CG continuity
├── l2.rs            # L2Space: discontinuous Lagrange, DG
├── dof_manager.rs   # DOF numbering: local → global map, boundary DOF identification
└── constraints.rs   # EssentialBC application (zero/nonzero Dirichlet)
```

### DOF Manager Algorithm
1. For each element, assign local DOFs (per vertex, edge, face, interior — based on element order).
2. For shared entities (vertices/edges/faces between elements), assign the same global DOF.
3. For vector spaces, interleave or block-layout DOFs (configurable).

### Acceptance criteria
- Unit-square mesh P1: n_dofs == n_nodes
- Unit-square mesh P2: n_dofs == n_nodes + n_edges
- After applying Dirichlet BC on full boundary, free_dofs count is correct
- Reproducing polynomial test via interpolation

---

## Phase 6: assembly

**Depends on**: `core`, `mesh`, `element`, `space`, `linalg`

### Modules
```
assembly/src/
├── lib.rs
├── assembler.rs      # Assembler: builds SparsityPattern, drives assembly loops
├── integrator.rs     # BilinearIntegrator + LinearIntegrator traits
├── standard/
│   ├── diffusion.rs  # DiffusionIntegrator: ∫ κ ∇u·∇v dx
│   ├── mass.rs       # MassIntegrator:      ∫ ρ u v dx
│   ├── source.rs     # DomainSourceIntegrator: ∫ f v dx
│   ├── neumann.rs    # NeumannIntegrator:   ∫_Γ g v ds
│   └── elasticity.rs # ElasticityIntegrator: ∫ σ(u):ε(v) dx
└── bc.rs             # Apply Dirichlet BCs to assembled system (row zeroing + diagonal 1)
```

### DiffusionIntegrator kernel (reference implementation)
```rust
// k_ij += w * det(J) * (J^{-T} ∇φ_i) · (J^{-T} ∇φ_j)
for q in 0..n_qp {
    let (jac, det_j) = compute_jacobian(elem, qp[q], mesh);
    let j_inv_t = jac.try_inverse().unwrap().transpose();
    for i in 0..n_dofs {
        let grad_i_phys = j_inv_t * grad_ref[q][i];
        for j in 0..n_dofs {
            let grad_j_phys = j_inv_t * grad_ref[q][j];
            k_elem[i * n_dofs + j] += weights[q] * det_j * grad_i_phys.dot(&grad_j_phys);
        }
    }
}
```

### Acceptance criteria
- **Patch test**: constant strain field exactly reproduced
- Poisson on unit square (P1, 16×16 mesh): L2 error < 5e-3, H1 error < 5e-2
- Poisson on unit square (P2, 8×8 mesh): L2 error < 1e-4

---

## Phase 7: solver

**Depends on**: `linalg`

### Modules
```
solver/src/
├── lib.rs
├── cg.rs            # Conjugate Gradient
├── gmres.rs         # GMRES(m) with restart
├── pcg.rs           # Preconditioned CG (calls preconditioner trait)
├── precond/
│   ├── mod.rs       # Preconditioner trait
│   ├── jacobi.rs    # Diagonal scaling
│   ├── ilu0.rs      # ILU(0) for non-symmetric systems
│   └── amg.rs       # Wraps amg as a preconditioner
└── direct.rs        # Tiny dense LU for coarse-grid (< 1000 DOFs)
```

### Preconditioner trait
```rust
pub trait Preconditioner: Send + Sync {
    /// Apply M^{-1}: z ← M^{-1} r
    fn apply(&self, r: &[f64], z: &mut [f64]);
    fn setup(&mut self, mat: &CsrMatrix<f64>);
}
```

### Acceptance criteria
- CG solves SPD Laplacian to tol=1e-10 in ≤ O(n) iterations when preconditioned with AMG
- GMRES solves non-symmetric convection-diffusion
- Iteration counts match reference (within 10%) for benchmark problems

---

## Phase 8: amg

**Depends on**: `linalg`

### Implementation: Smoothed Aggregation AMG (SA-AMG)
Chosen over classical RS-AMG for better performance on elasticity and vector problems.

```
amg/src/
├── lib.rs
├── setup.rs         # AmgHierarchy construction
├── strength.rs      # strength-of-connection matrix
├── aggregation.rs   # MIS-based aggregation (parallel-friendly)
├── smoother.rs      # Jacobi, Gauss-Seidel, Chebyshev
├── interp.rs        # Smoothed prolongation P
├── coarse.rs        # Galerkin coarse operator A_c = R A P
├── cycle.rs         # V-cycle, W-cycle, F-cycle
└── params.rs        # AmgParams: theta, n_levels, smoother_steps, cycle_type
```

### Default parameters
```rust
pub struct AmgParams {
    pub theta:          f64,  // strength threshold: 0.25
    pub max_levels:     u8,   // 25
    pub coarse_size:    usize, // 100 (direct solve below this)
    pub pre_smooth:     u8,   // 2
    pub post_smooth:    u8,   // 2
    pub cycle:          CycleType, // V
    pub smoother:       SmootherKind, // Jacobi { omega: 0.67 }
}
```

### Acceptance criteria
- Setup + solve time for 3D Poisson (1M DOFs) < 10s on 8-core desktop
- Convergence factor per V-cycle < 0.15 for Laplacian
- `hypre` feature: delegate to hypre BoomerAMG when available

---

## Phase 9: io

**Depends on**: `mesh`

### Formats
| Format | Read | Write | Notes |
|--------|------|-------|-------|
| GMSH .msh v4 | ✓ | — | primary mesh input |
| VTK .vtu (XML) | — | ✓ | visualization output |
| HDF5 | ✓ | ✓ | restart files, large datasets |

```
io/src/
├── lib.rs
├── gmsh.rs          # parse .msh v4 → SimplexMesh
├── vtk.rs           # write solution fields to .vtu
└── hdf5.rs          # read/write mesh + solution (feature-gated)
```

### Acceptance criteria
- Round-trip: write mesh to .vtu, visual inspection in ParaView correct
- GMSH import: L-shaped domain from .msh, element count matches gmsh report
- HDF5: write 1M DOF solution, read back, max error < 1e-15

---

## Phase 10: parallel

**Depends on**: `mesh`, `assembly`, `linalg`, `amg` | **feature**: `mpi`

### Modules
```
parallel/src/
├── lib.rs
├── comm.rs          # Communicator wrapper, collective ops
├── partition.rs     # Graph partitioning via METIS (metis-sys crate)
├── par_mesh.rs      # ParallelMesh: distribute SimplexMesh across ranks
├── par_linalg.rs    # ParCsrMatrix, ParVector
├── par_assembly.rs  # Parallel assembly loop + ghost exchange
└── par_amg.rs       # Parallel AMG (BoomerAMG via hypre or native)
```

### Ghost DOF Communication Pattern
```
1. Each rank owns a contiguous range of global DOFs
2. After local assembly, identify off-rank DOF contributions
3. AllToAll communication: send contributions to owning ranks
4. Owning rank accumulates received values
5. Broadcast back owned values as ghost data for next SpMV
```

### Acceptance criteria
- Weak scaling test: 4 ranks × 250K DOFs ≈ same time as 1 rank × 250K DOFs (within 20%)
- Strong scaling: 1M DOF Poisson, 1→16 ranks, efficiency > 70%

---

## Phase 11: wasm

**Depends on**: `assembly`, `solver`, `io` | **NO** `parallel`

### Modules
```
wasm/src/
├── lib.rs           # wasm_bindgen exports
├── solver.rs        # WasmSolver: JS-facing solve interface
├── mesh_builder.rs  # build mesh from JS Float64Array / Uint32Array
└── result.rs        # solution export as Float64Array
```

### JS API (TypeScript types)
```typescript
class WasmSolver {
  constructor(options: { dim: 2 | 3; meshJson: string });
  solve(rhs?: Float64Array): Float64Array;
  getSolution(): Float64Array;
  free(): void;
}
```

### Acceptance criteria
- Bundle size < 2 MB (wasm-opt -O3)
- Solve 2D Poisson 10K DOFs in < 500ms in Chrome
- No panics: all errors surfaced as `Result<_, JsValue>`

---

## Implementation Order & Parallelism

```
Phase 0  (bootstrap)
    ↓
Phase 1  (core)
    ↓
Phase 2 ──── Phase 3 ──── Phase 4      ← can be parallelized
   (mesh)   (element)   (linalg)
    ↓           ↓           ↓
Phase 5  (space)   Phase 7 (solver)
    ↓                      ↓
Phase 6  (assembly) ← Phase 8 (amg)
    ↓
Phase 9 ──── Phase 10 ──── Phase 11    ← can be parallelized
  (io)      (parallel)     (wasm)
```

---

## Directory Creation Sequence for Agents

When bootstrapping, create files in this order to avoid missing-dependency errors:
1. `Cargo.toml` (workspace)
2. `crates/core/`
3. `crates/linalg/` and `crates/mesh/` (no inter-dep, parallel)
4. `crates/element/`
5. `crates/space/`
6. `crates/assembly/` and `crates/solver/` (parallel)
7. `crates/amg/`
8. `crates/io/`
9. `crates/parallel/`
10. `crates/wasm/`

---

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Crate | kebab-case | `assembly` |
| Module | snake_case | `dof_manager` |
| Trait | PascalCase | `ReferenceElement` |
| Struct | PascalCase | `CsrMatrix` |
| Enum | PascalCase | `ElementType` |
| Enum variant | PascalCase | `Tri3`, `Hex8` |
| Function | snake_case | `assemble_bilinear` |
| Constant | SCREAMING_SNAKE | `MAX_POLY_ORDER` |
| Type alias | PascalCase | `FemResult` |
| Generic param (type) | single cap | `T`, `S` |
| Generic param (const) | single cap | `D`, `N` |

---

## Mathematical Notation in Code

Use these variable name standards in all implementations so AI agents can cross-reference with textbooks:

| Math symbol | Code name | Meaning |
|-------------|-----------|---------|
| ξ, η | `xi`, `eta` | reference coordinates |
| φ_i | `phi[i]` | basis function value |
| ∇φ_i | `grad_phi[i]` | reference gradient |
| J | `jac` | Jacobian ∂x/∂ξ |
| J^{-T} | `jac_inv_t` | inverse transpose Jacobian |
| det(J) | `det_j` | Jacobian determinant |
| w_q | `weight[q]` | quadrature weight |
| K_e | `k_elem` | element stiffness matrix (flat) |
| f_e | `f_elem` | element load vector |

---

## Common Pitfalls (Agent Warnings)

1. **Row-major vs column-major**: `k_elem[i * n_dofs + j]` for row-major storage. nalgebra is column-major — never pass element matrices directly to nalgebra without transposing.
2. **Jacobian sign**: ensure `det_j > 0`; negative means inverted element (mesh quality issue). Emit `FemError::Mesh` not panic.
3. **Boundary DOF handling**: Dirichlet BC must be applied AFTER assembly, not during. Apply by zeroing row/column and setting diagonal to 1 and RHS to prescribed value.
4. **WASM and threads**: `rayon` will compile for wasm32 but panic at runtime. Gate all parallel code with `#[cfg(not(target_arch = "wasm32"))]`.
5. **MPI Init**: `mpi::initialize()` must be called exactly once per process. In library code, accept an external `Communicator` — never call `mpi::initialize()` inside a library.
6. **Integer overflow in index arithmetic**: `NodeId` is `u32`; when computing offsets for large meshes, cast to `usize` before arithmetic: `node as usize * dim`.

---
