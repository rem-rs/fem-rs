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
| 5 | space | ✅ Done | 2026-03-31 | H1Space(P1/P2), L2Space(P0/P1/P2), DofManager, apply_dirichlet, boundary_dofs; 18 tests |
| 6 | assembly | ✅ Done | 2026-03-31 | Assembler, BilinearIntegrator/LinearIntegrator/BoundaryLinearIntegrator; DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator, NeumannIntegrator; P1/P2 Poisson verified (rate=2.0/3.0); 12 tests |
| 7 | solver | ✅ Done | 2026-03-31 | Backed by linger: CG, PCG+Jacobi, PCG+ILU0, GMRES, BiCGSTAB; end-to-end Poisson solve verified (all 5 solvers) |
| 8 | amg | ✅ Done | 2026-03-31 | Backed by linger: SA-AMG + RS-AMG, AmgSolver (reusable hierarchy); AMG-CG fewer iters than CG on 64×64 mesh (38 vs 84) |
| 9 | io | ✅ Done | 2026-03-31 | GMSH v4.1 ASCII reader + VTK .vtu XML writer; unit_cube_tet generator added to fem-mesh; 12 tests |
| 10 | parallel | ✅ Done | 2026-03-31 | ChannelBackend (in-process multi-threading), ThreadLauncher (n-worker), GhostExchange (alltoallv + forward/reverse), NativeMpiBackend::alltoallv_bytes; 20 tests (12 thread + 8 ghost) |
| 11 | wasm | ✅ Done | 2026-03-31 | WasmSolver (unit-square P1 Poisson, wasm-bindgen optional), assemble_constant_rhs / assemble_nodal_rhs / solve / node_coords / connectivity; 7 native tests |
| 12 | element | ✅ Done | 2026-04-02 | Nedelec-I (TriND1, TetND1) + Raviart-Thomas RT0 (TriRT0, TetRT0); VectorReferenceElement trait; 12 tests (nodal basis, constant curl/div, divergence theorem) |
| 13 | space + assembly | ✅ Done | 2026-04-02 | VectorH1Space (interleaved elem DOFs, block global DOFs); BlockMatrix/BlockVector; ElasticityIntegrator; MixedAssembler + PressureDivIntegrator/DivIntegrator; 8 tests |
| 14 | assembly | ✅ Done | 2026-04-02 | SIP-DG (Symmetric Interior Penalty): InteriorFaceList, DgAssembler::assemble_sip; volume + interior face + Dirichlet boundary terms; symmetry + positive diagonal verified; 4 tests |
| 15 | solver + assembly | ✅ Done | 2026-04-02 | NonlinearForm trait; NewtonSolver (GMRES linear solves, configurable atol/rtol/max_iter); NonlinearDiffusionForm (Picard linearisation); Dirichlet BC via elimination; 3 tests |
| 16 | solver | ✅ Done | 2026-04-03 | ODE/time integrators: ForwardEuler, RK4, RK45 (adaptive Dormand-Prince), ImplicitEuler, SDIRK-2, BDF-2; TimeStepper + ImplicitTimeStepper traits; stiffness stability verified (λ=-1000, dt=0.1); 7 tests |
| 17 | mesh | ✅ Done | 2026-04-03 | AMR: red refinement (Tri3→4 children), InteriorFaceList propagation, ZZ gradient-recovery error estimator, Dörfler marking; refine_uniform + refine_marked; 6 tests |
| 18 | parallel | ✅ Done | 2026-04-03 | METIS k-way partitioning via rmetis; dual-graph builder; MetisPartitioner + partition_simplex_metis; balance + coverage verified; 4 tests |
| 19 | mesh + space | ✅ Done | 2026-04-03 | CurvedMesh\<D\>: from_linear (P1), elevate_to_order2 (P2/Tri6) with custom map_fn; isoparametric Jacobian + reference_to_physical; area preserved; 6 tests |
| 20 | solver | ✅ Done | 2026-04-03 | LOBPCG eigenvalue solver; GeneralizedEigenSolver trait; LobpcgSolver; handles standard + generalized A x=λBx; 1-D Laplacian eigenvalues verified; 4 tests |
| 21 | solver + linalg | ✅ Done | 2026-04-03 | BlockSystem (2×2 saddle-point); BlockDiagonalPrecond; SchurComplementSolver (GMRES on flat system); MinresSolver; 4 tests |
| 22 | assembly + ceed | ✅ Done | 2026-04-03 | Partial assembly (matrix-free): PAMassOperator, PADiffusionOperator (spatially varying κ), LumpedMassOperator; MatFreeOperator trait; results match assembled matrix × vector to 1e-11; 5 tests |
| 23 | space | ✅ Done | 2026-04-04 | HCurlSpace (Nédélec ND1 edge DOFs, sign convention, 2D+3D) + HDivSpace (RT0 face DOFs, geometric sign computation, 2D+3D); FESpace::element_signs(); EdgeKey/FaceKey public; boundary_dofs_hcurl/hdiv; 13 tests |
| 24 | assembly | ✅ Done | 2026-04-04 | VectorAssembler (Piola transforms + sign application); VectorQpData + VectorBilinearIntegrator/VectorLinearIntegrator traits; CurlCurlIntegrator (∫ μ curl u · curl v); VectorMassIntegrator (∫ α u·v); H(curl) assembly verified symmetric + PSD; 10 tests |
| 25 | assembly + solver | ✅ Done | 2026-04-04 | Fix SIP-DG interior face normals (single consistent n_L + orient_normal_outward); SchurComplementSolver rewritten with right-preconditioned GMRES + block-diagonal precond; MINRES rewritten (Choi-Paige-Saunders); TriND1 Φ₂ basis orientation fix; all 8 examples passing |
| 26 | assembly | ✅ Done | 2026-04-04 | Coefficient system: ScalarCoeff/VectorCoeff/MatrixCoeff traits, PWConstCoeff, PWCoeff, GridFunctionCoeff, composition |
| 27 | assembly | ✅ Done | 2026-04-04 | Convection, VectorDiffusion, BoundaryMass, GradDiv, Transpose, Sum integrators; VectorDomainLF, BoundaryNormalLF |
| 28 | assembly | ✅ Done | 2026-04-04 | GridFunction wrapper (evaluate, L²/H¹ errors); postprocess (element gradients, curl, divergence, nodal gradient recovery); 10 tests |
| 29 | assembly | ✅ Done | 2026-04-04 | DiscreteLinearOperator: gradient (H1→H(curl)), curl_2d (H(curl)→L2), divergence (H(div)→L2) as sparse matrices; de Rham exact sequence verified; 5 tests |
| 30 | solver | ✅ Done | 2026-04-04 | Newmark-β time integrator (average acceleration, unconditionally stable); ex10_wave_equation example (sin(πx)sin(πy), max error 8.3e-3); 1 test |
| 31 | element | ✅ Done | 2026-04-04 | Gauss-Lobatto quadrature (n=2..5 on [-1,1]); seg_lobatto_rule, quad_lobatto_rule, hex_lobatto_rule; 7 tests |
| 32 | examples | ✅ Done | 2026-04-04 | ex4_darcy (H(div) RT0 grad-div, MINRES); ex15_dg_amr (P1 + ZZ estimator + Dörfler marking + refinement, O(h²) convergence verified) |
| 33a | parallel | ✅ Done | 2026-04-05 | jsmpi crates.io package + JsMpiBackend replacing WasmWorkerBackend stub; WASM MPI via jsmpi Web Workers |
| 33b | parallel | ✅ Done | 2026-04-05 | DofPartition (DOF-level ownership, exclusive scan for global offsets) + ParallelFESpace (wraps serial FESpace + DOF ghost exchange); 5 tests |
| 33c | parallel | ✅ Done | 2026-04-05 | ParVector (distributed vector, global_dot/norm, ghost exchange) + ParCsrMatrix (diag+offd split, parallel SpMV); 7 tests |
| 33d | parallel | ✅ Done | 2026-04-05 | ParAssembler: parallel bilinear/linear form assembly with ghost-row exchange via alltoallv; parallel Dirichlet BCs; 3 tests |
| 33e | parallel + examples | ✅ Done | 2026-04-05 | par_solve_cg + par_solve_pcg_jacobi; pex1_poisson example (2-rank PCG, 21 iters, L² error 1.5e-3); 2 tests |
| 34 | parallel | ✅ Done | 2026-04-05 | P2 parallel spaces: DofPartition::from_dof_manager (edge DOF ownership via min-owner rule), ghost edge ID exchange, DOF permutation (dm↔partition), ParAssembler auto-permute; pex1 --p2 (L² error 5.5e-4); 7 new tests |
| 35 | parallel | ✅ Done | 2026-04-05 | Parallel AMG: ParAmgHierarchy (local smoothed aggregation, Galerkin R*A*P), damped Jacobi smoother, par_solve_pcg_amg; AMG fewer iters than Jacobi verified; 3 tests |
| 36 | parallel | ✅ Done | 2026-04-05 | Comm::split(color,key) sub-communicator; CommBackend::split trait method; ChannelBackend impl (Arc pointer sharing); 2 tests |
| 37 | parallel + wasm | ✅ Done | 2026-04-05 | WASM multi-Worker + streaming mesh partition: WorkerInitMsg::from_jsmpi_env(), spawn_async()+WasmJob, MeshPartition::from_raw(), binary mesh_serde (encode/decode), partition_simplex_streaming() (rank 0 distributes sub-meshes via send/recv), extract_submesh_for_rank() refactor, WasmParSolver + jsmpi_main entry point, wasm-parallel feature; 13 new tests |
| 38 | parallel | ✅ Done | 2026-04-05 | METIS streaming partition + pex1 CLI: extract_submesh_from_partition(elem_part) generalized extractor, partition_simplex_metis_streaming(), metis.rs refactored to shared extractor (−80 lines duplication), pex1 enhanced with --n/--ranks/--metis/--streaming flags; 3 new tests |
| 48 | element+space+assembly+solver+io | ✅ Done | 2026-04-09 | linger submodule update: sparse direct solvers (SparseLu/SparseCholesky/SparseLdlt), IDR(s), TFQMR, ILDLt precond, KrylovSchur eigen, Matrix Market I/O; higher-order elements TriP3/TetP2/TetP3/QuadQ2 registered across all assemblers; H1TraceSpace P2/P3 support; Grundmann-Moller quadrature corrected via linear system solve; 8 new solver tests, 2 io tests |
| 49 | element+space+assembly | ✅ Done | 2026-04-09 | TriND2/TetND2 (Nédélec-I order 2, 8/20 DOFs, Vandermonde inversion, linear curl verified); TriRT1/TetRT1 (Raviart-Thomas order 1, 8/15 DOFs, nodal basis verified); HCurlSpace/HDivSpace relaxed to support orders 1/2 and 0/1 (multiple DOFs per edge/face + interior bubble DOFs); VectorAssembler+postprocess factory updated to dispatch on space.order(); 15 new element tests |
| 50 | solver+linalg | ✅ Done | 2026-04-09 | linger submodule update (rem-rs/linger): AMS (Auxiliary-space Maxwell Solver) for H(curl), ADS (Auxiliary-space Divergence Solver) for H(div), BlrMatrix/BlrBlock (Block Low-Rank compression); fem-solver: solve_pcg_ams, solve_gmres_ams, solve_pcg_ads, solve_gmres_ads; fem-linalg: re-export BlrMatrix, BlrBlock, compress_block, compress_block_adaptive under "direct" feature; reed submodule URL updated to rem-rs/reed |
| 51 | space+assembly | ✅ Done | 2026-04-10 | HCurlSpace 3D ND2 topology corrected to shared face DOFs (global count = 2*n_edges + 2*n_faces) with canonical face-moment interpolation; DiscreteLinearOperator::curl_3d extended to ND2->RT1 local reconstruction; strict high-order de Rham property div(curl)=0 restored and verified; new HCurl 3D ND2 and discrete_op tests added |
| 52 | space+assembly | ✅ Done | 2026-04-10 | L2Space extended to discontinuous P2 (Tri: 6 DOFs/elem, Tet: 10 DOFs/elem); DiscreteLinearOperator extended to ND2->L2(P2) and RT1->L2(P2) in 2D/3D; added dimension + commuting + de Rham tests including 3D RT1->L2(P2) and div(curl)=0 with L2(P2) |
| 53 | vendor | ✅ Done | 2026-04-10 | reed submodule synchronized with upstream main: fixed `basis_h1_simplex` backend dispatch to lock backend (`Arc<Mutex<...>>`) before method call; published in rem-rs/reed and bumped fem-rs submodule pointer to commit `e9772a3` |
| 54 | assembly+examples | ✅ Done | 2026-04-11 | **Full Maxwell completion (MFEM parity):** (a) `CurlCurlTensorIntegrator<MatrixCoeff>` + `VectorMassTensorIntegrator<MatrixCoeff>` — anisotropic μ/ε tensors; (b) `VectorBoundaryAssembler` + `TangentialMassIntegrator` — `∫_Γ γ(n×u)·(n×v) dS` for Silver-Müller ABC / impedance BC; (c) `ex_maxwell_time.rs` — time-domain `εÜ + σU̇ + KU = J(t)` with damped Newmark-β, manufactured solution verified; (d) `ex_maxwell_eigenvalue.rs` — cavity resonance via dense generalized eigensolver on free-DOF subspace, 6 eigenvalues converge at O(h²); exact 2D curl-curl spectrum: single-component modes (m²π², mult 2) + stream-function modes ((m²+n²)π²); 5 new tests (tensor identity, boundary symmetry/PSD/empty-tag); MAXWELL_GAPS.md gap analysis document |

### Week-1 High-Order Discrete-Operator Coverage (Phase 53 kickoff)

| Operator | Pair | Dim | Status | Coverage |
|----------|------|-----|--------|----------|
| `gradient` | P2 -> ND2 | 2D | ✅ Implemented | dimensions + commuting + constant-zero |
| `curl_2d` | ND2 -> L2(P2) | 2D | ✅ Implemented | dimensions + commuting + exact field |
| `divergence` | RT1 -> L2(P2) | 2D | ✅ Implemented | dimensions + commuting |
| `curl_3d` | ND2 -> RT1 | 3D | ✅ Implemented | dimensions + commuting |
| `divergence` | RT1 -> L2(P2) | 3D | ✅ Implemented | dimensions + commuting |
| Chain | ND2 -> RT1 -> L2(P2) | 3D | ✅ Implemented | de Rham `div(curl)=0` (random vectors) |

Week-1 checklist:
- [x] Support matrix documented in design plan
- [x] 3D RT1 -> L2(P2) dimensions + commuting tests
- [x] 3D ND2 -> RT1 commuting test
- [x] Add low/high-order randomized stress tests for commuting errors
- [x] Add CI grouping for high-order discrete-operator suite (`cargo test-high-order-discrete-op`)

### Vendor submodules
| Submodule | URL | Role |
|-----------|-----|------|
| `vendor/reed` | rem-rs/reed | libCEED analogue; bridged via `crates/ceed` |
| `vendor/linger` | rem-rs/linger | Krylov solvers + AMG; drives `fem-solver` and `fem-amg` |
| `vendor/rmetis` | javagg/rmetis | Pure-Rust BFS graph partitioner; drives `fem-parallel` Phase 18 |
| `jsmpi` (crates.io) | crates.io/jsmpi | JavaScript MPI shim for WASM Web Workers; drives `fem-parallel` Phase 33a |

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

---

## Phase 12: H(curl) and H(div) Elements

**Depends on**: `element` (extends Phase 3 stubs)

### Goal
Add Nedelec (first-kind, order 1–2) and Raviart-Thomas (RT0, RT1) elements — required for Maxwell equations, Stokes, and mixed Darcy.

### Modules to add
```
element/src/
├── nedelec/
│   ├── mod.rs
│   ├── tri.rs       # Nedelec1 on triangle (6 DOFs for order 2)
│   └── tet.rs       # Nedelec1 on tetrahedron
└── raviart_thomas/
    ├── mod.rs
    ├── tri.rs       # RT0 / RT1 on triangle
    └── tet.rs       # RT0 / RT1 on tetrahedron
```

### Acceptance criteria
- DOF continuity: tangential (Nedelec) and normal (RT) continuity across faces
- Commuting diagram property: `curl ∘ grad = 0` verified numerically
- Patch test for each element type

---

## Phase 13: Mixed Bilinear Forms and Vector FE Spaces

**Depends on**: `space`, `assembly`, Phase 12 elements

### Goal
Support mixed formulations (e.g., Stokes u∈H(div), p∈L2; Maxwell E∈H(curl), B∈H(div)).

### Additions
- `VectorH1Space`: blocked H1 for elasticity (u∈[H1]^d)
- `HCurlSpace`: DOF manager for Nedelec elements
- `HDivSpace`: DOF manager for RT elements
- `MixedBilinearForm`: assembles off-diagonal coupling blocks
- New integrators: `CurlCurlIntegrator`, `DivDivIntegrator`, `MixedScalarIntegrator`
- `BlockMatrix` in `linalg`: 2×2 / n×n block structure for mixed systems

### Acceptance criteria
- Stokes problem (Taylor-Hood P2/P1): divergence-free velocity to tol 1e-12
- Mixed Darcy: exact pressure projection test

---

## Phase 14: DG Interior Penalty

**Depends on**: `assembly`, `space` (L2Space already done)

### Goal
Discontinuous Galerkin for convection-diffusion, incompressible flow.

### Additions in `assembly`
```
assembly/src/
└── dg/
    ├── face_assembler.rs     # iterate interior + boundary faces
    ├── dg_diffusion.rs       # SIP / NIP / IIP penalty terms
    ├── dg_convection.rs      # upwind flux
    └── dg_integrator.rs      # FaceIntegrator trait
```

### Acceptance criteria
- SIP-DG convergence on smooth Poisson: P1 rate ≥ 2, P2 rate ≥ 3
- Penalty parameter auto-selection (C_IP from inverse estimates)
- Works with L2Space P0/P1/P2

---

## Phase 15: Nonlinear Forms and Newton Solver

**Depends on**: `assembly`, `solver`

### Goal
Nonlinear PDE support: nonlinear diffusion, hyperelasticity, Navier-Stokes.

### Additions
- `NonlinearForm` trait: `compute_residual(u, r)`, `compute_jacobian(u, J)`
- `NewtonSolver`: line-search Newton with pluggable linear solver
- `NonlinearDiffusionIntegrator`: ∫ κ(u) ∇u·∇v dx
- `HyperelasticIntegrator`: neo-Hookean / Saint Venant-Kirchhoff models

### Acceptance criteria
- Nonlinear Poisson (p-Laplacian) converges in ≤ 10 Newton iterations
- Jacobian verified by finite-difference check (relative error < 1e-6)

---

## Phase 16: ODE / Time Integrators

**Depends on**: `assembly`, `solver`

### Goal
Time-dependent PDE: heat equation, wave equation, structural dynamics.

### Additions in `solver`
```
solver/src/
└── ode/
    ├── mod.rs           # TimeStepper trait, OdeProblem
    ├── rk_explicit.rs   # Forward Euler, RK4, RK45 (adaptive)
    ├── sdirk.rs         # SDIRK-2/3, implicit Euler
    └── bdf.rs           # BDF-1/2 (for stiff problems)
```

### TimeStepper trait
```rust
pub trait TimeStepper {
    fn step(&mut self, t: f64, dt: f64, u: &mut Vector<f64>) -> FemResult<()>;
}
```

### Acceptance criteria
- Heat equation: L2 error order matches expected temporal order (RK4 → 4, BDF2 → 2)
- SDIRK unconditionally stable on stiff ODE test (λ = -1000)

---

## Phase 17: Adaptive Mesh Refinement (AMR)

**Depends on**: `mesh`, `space`, `assembly`

### Goal
h-refinement driven by a posteriori error estimators — foundational for production solvers.

### Additions in `mesh`
- `refine.rs`: bisection refinement for Tri3/Tet4; hanging-node registry
- `hanging_node.rs`: constraint equations for hanging DOFs (conforming AMR)

### Additions in `assembly`
- `error_estimator.rs`: `ErrorEstimator` trait; Zienkiewicz-Zhu (ZZ) patch recovery; residual estimator
- `marking.rs`: Dörfler/bulk marking strategy

### AMR loop
```
solve → estimate → mark → refine → update DOFs → repeat
```

### Acceptance criteria
- L-shaped domain: adaptive refinement achieves optimal convergence rate 1.0 (P1) vs 0.66 (uniform)
- No hanging-node DOF constraint violation (patch test on adaptively refined mesh)

---

## Phase 18: Parallel Mesh Partitioning and Parallel AMR

**Depends on**: Phase 10 (parallel), Phase 17 (AMR), METIS

### Goal
Complete the `fem-parallel` crate with METIS-based partitioning and distributed AMR.

### Additions
- `partition.rs`: METIS binding via `metis-sys` crate; k-way partitioning
- `par_mesh.rs`: `ParallelMesh` distributing `SimplexMesh` across MPI ranks
- `par_assembly.rs`: parallel assembly loop with ghost exchange (uses existing GhostExchange)
- `par_amg.rs`: parallel AMG — either native (aggregate across ranks) or BoomerAMG via hypre feature

### Acceptance criteria
- 4-rank Poisson: solution matches serial reference (max diff < 1e-12)
- Weak scaling: 4 ranks × 250K DOFs within 20% of 1 rank × 250K DOFs
- METIS partitioning: edge-cut < 1.5× random partitioning edge-cut

---

## Phase 19: High-Order Curved Meshes

**Depends on**: `mesh`, `element`

### Goal
Geometry represented as a FE field (isoparametric mapping) — needed for high-order accuracy on curved domains.

### Additions
- `curved.rs` in `mesh`: `CurvedMesh<D, Order>` stores node DOF field alongside topology
- Update `Jacobian` computation in assembly: use isoparametric mapping instead of affine
- Mesh-quality check: detect inverted curved elements

### Acceptance criteria
- Circle/sphere domain: P2 geometry + P2 solution achieves O(h^3) L2 convergence
- Jacobian always positive inside each element (verified by sampling)

---

## Phase 20: Eigenvalue Solvers

**Depends on**: `linalg`, `solver`, `assembly`

### Goal
Structural vibration modes, buckling, electromagnetic cavity modes.

### Additions in `solver`
```
solver/src/
└── eigen/
    ├── lobpcg.rs        # Locally Optimal Block Preconditioned CG
    └── arpack.rs        # ARPACK binding (feature = "arpack")
```

### `GeneralizedEigenSolver` trait
```rust
pub trait EigenSolver {
    /// Solve K x = λ M x, return (eigenvalues, eigenvectors)
    fn solve(&mut self, k: &CsrMatrix<f64>, m: &CsrMatrix<f64>, n_eigs: usize)
        -> FemResult<(Vec<f64>, Vec<Vector<f64>>)>;
}
```

### Acceptance criteria
- 2D square membrane: first 6 eigenvalues match analytical within 0.5%
- LOBPCG convergence in ≤ 50 iterations with AMG preconditioner

---

## Phase 21: Block Solvers and Saddle-Point Systems

**Depends on**: `linalg`, `solver`, Phase 13 (mixed forms)

### Goal
Efficient solvers for mixed/saddle-point problems (Stokes, Darcy, incompressible elasticity).

### Additions in `linalg`
- `block_matrix.rs`: `BlockMatrix<T>` — indexable 2D block structure wrapping `CsrMatrix`
- `block_vector.rs`: `BlockVector<T>` — contiguous split into named blocks

### Additions in `solver`
- `schur.rs`: Schur complement preconditioner P = [[A, 0], [B A^{-1} B^T, S]]
- `block_precond.rs`: block-diagonal, block-triangular preconditioners

### Acceptance criteria
- Stokes (Taylor-Hood): block-preconditioned MINRES converges in ≤ 30 iterations on 64×64 mesh
- Condition number estimate: κ(P^{-1} K) < 10 (mesh-independent)

---

## Phase 22: Partial Assembly and Matrix-Free

**Depends on**: `assembly`, `ceed` (fem-ceed / reed), Phase 3 elements

### Goal
High-performance high-order FEM via sum-factorization — avoids explicit matrix formation.

### Approach
- Integrate with `fem-ceed` (reed submodule) for operator application kernels
- `PartialAssembler`: stores quadrature-point data (D-vectors) instead of full matrix
- Sum-factorization `apply(u, v)` for tensor-product elements (QuadQ*, HexQ*)

### Additions
- `assembly/src/partial/`: `PADiffusionOperator`, `PAMassOperator`
- `ceed/src/operator.rs`: bridge fem-assembly integrators → reed CeedOperator

### Acceptance criteria
- PA SpMV throughput ≥ 2× explicit CSR SpMV for Q2 on 100K element mesh
- Results match assembled matrix to tol 1e-13
- Works on CPU; CUDA/HIP backend via reed is optional stretch goal

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

── MFEM gap phases (new) ──────────────────────────────────────────
Phase 12 (Nedelec / RT elements)
    ↓
Phase 13 (mixed forms + vector spaces) ←── also needs Phase 12
    ↓
Phase 14 (DG interior penalty)          ← parallel with Phase 13
Phase 15 (nonlinear forms + Newton)     ← parallel with Phase 13
Phase 16 (ODE time integrators)         ← parallel with Phase 13

Phase 17 (AMR: h-refinement + ZZ estimator)
    ↓
Phase 18 (parallel mesh partitioning + par-AMR)  ← needs Phase 10 + 17

Phase 19 (high-order curved meshes)     ← needs Phase 12

Phase 20 (eigenvalue solvers: LOBPCG)   ← needs Phase 13
Phase 21 (block solvers + Schur)        ← needs Phase 13

Phase 22 (partial assembly / matrix-free)  ← needs Phase 12, ceed
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
