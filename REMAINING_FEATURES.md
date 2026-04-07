# fem-rs Remaining Features Analysis

**Date**: 2026-04-04  
**Project**: fem-rs at `/Users/alex/works/fem-rs`  
**Status Source**: MFEM_MAPPING.md

---

## Executive Summary

fem-rs has implemented **25 major development phases** with extensive MFEM feature coverage. This report identifies all remaining work organized by category and priority.

**Current Examples Implemented**:
- ex1_poisson ✅
- ex2_elasticity ✅
- ex3_maxwell ✅
- ex5_mixed_darcy ✅
- ex9_dg_advection ✅
- ex10_heat_equation ✅
- ex13_eigenvalue ✅
- ex16_nonlinear_heat ✅
- em_electrostatics ✅ (custom)
- em_magnetostatics_2d ✅ (custom)
- ceed_mass (partial assembly demo) ✅
- par_mesh_verify (parallel infrastructure) ✅

---

## REMAINING ITEMS BY CATEGORY

### 1. MESH FEATURES (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED (High Priority for Phase 2+)
1. **Mixed element types per mesh**
   - Issue: Currently `SimplexMesh<D>` enforces uniform element type
   - Target: Support array of `elem_type` per element
   - Impact: Enables real-world meshes with mixed Tri/Quad or Tet/Hex/Wedge

2. **NCMesh (non-conforming mesh / hanging nodes)**
   - Issue: Required for adaptive mesh refinement beyond red refinement
   - Target: Implement hanging node constraints
   - Impact: Phase 2+ feature, essential for AMR with coarsening

3. **ParMesh wrapper**
   - Issue: Placeholder only
   - Target: Full distributed mesh container
   - Blocking: Phase 10 (parallel infrastructure)

4. **Mesh::MakePeriodic()**
   - Issue: Periodic boundary condition mesh generator missing
   - Impact: Needed for periodic domain problems

5. **Netgen `.vol` format reader**
   - Issue: Only GMSH v4.1 ASCII supported
   - Target: Add Netgen mesh input
   - Priority: Phase 9+

6. **Mesh::GetBoundingBox()**
   - Issue: Utility function not implemented
   - Priority: Low

#### 🔨 PARTIAL (In Progress)
1. **Mesh::bdr_attributes dedup utility**
   - Status: Face tags stored but no dedup utility
   - Issue: Need unique boundary tag extraction

2. **ElementTransformation type**
   - Status: Jacobian computed inline during assembly
   - Issue: No first-class `ElementTransformation` type
   - Goal: Explicit transformation objects for clarity

---

### 2. REFERENCE ELEMENTS & QUADRATURE (🔲 Planned)

#### 🔲 PLANNED
1. **Gauss-Lobatto quadrature**
   - Issue: Only Gauss-Legendre implemented
   - Impact: Needed for spectral element methods, nodal integration
   - Quadrature orders 1–10 available but Lobatto missing

---

### 3. FINITE ELEMENT SPACES (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED
1. **H1_Trace_FECollection**
   - Issue: Traces of H¹ on faces (H^{1/2} spaces) not implemented
   - Impact: Boundary element methods, trace spaces
   - Priority: Phase 2+

2. **FES::GetTrueDofs() — Parallel DOF ownership**
   - Issue: Missing for distributed DOF numbering
   - Target: Global DOF mapping across parallel processes
   - Blocking: Phase 10 (parallel assembly)

3. **FES::TransferToTrue() / Transfer()**
   - Issue: Missing DOF transfer operations
   - Impact: Parallel assembly, ghost DOF synchronization
   - Blocking: Phase 10

#### 🔨 PARTIAL
1. **Taylor-Hood P2-P1 mixed space**
   - Status: MixedAssembler exists but complete P2-P1 Stokes not yet verified
   - Issue: Via MixedAssembler only, not standalone FECollection

---

### 4. ASSEMBLY: FORMS & INTEGRATORS (🔲 Planned | 🔨 Partial)

#### 🔨 PARTIAL
1. **ElementTransformation (assembly pipeline)**
   - Status: Jacobian computed inline
   - Issue: No first-class transformation type
   - Goal: Explicit `ElementTransformation` wrapper

---

### 5. LINEAR ALGEBRA (🔲 Planned)

#### 🔲 PLANNED
1. **SparseMatrix::Add(A, B)**
   - Issue: Matrix addition not implemented
   - Target: A + B → sparse result
   - Priority: Lower (workaround: manual construction)

2. **DenseTensor**
   - Issue: Nested matrix structures missing
   - Impact: Needed for tensor-based assembly optimizations
   - Priority: Phase 2+

3. **Vector::SetSubVector()**
   - Issue: Index slice assignment to vector not implemented
   - Target: v[indices] = values pattern
   - Priority: Lower

---

### 6. SOLVERS & PRECONDITIONERS (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED
1. **FGMRESSolver — Flexible GMRES**
   - Issue: Variant with varying preconditioner not implemented
   - Impact: Useful for variable-preconditioner scenarios

2. **SLISolver — Stationary Linear Iteration**
   - Issue: Jacobi/Gauss-Seidel stationary solvers not implemented
   - Priority: Low

3. **Chebyshev smoother**
   - Issue: Chebyshev polynomial-based smoothing missing
   - Impact: AMG smoother option, better convergence on some problems
   - Currently: Only Jacobi and Gauss-Seidel available

4. **BlockTriangularPreconditioner**
   - Issue: Missing block triangular (upper/lower) preconditioner
   - Impact: Block system solutions

#### 🔨 PARTIAL
1. **IterativeSolver::SetPrintLevel() — Logging**
   - Status: Partial; verbose flag exists
   - Issue: Log integration incomplete
   - Target: Structured logging for solver convergence

---

### 7. ALGEBRAIC MULTIGRID (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED
1. **F-cycle**
   - Issue: Only V-cycle and W-cycle implemented
   - Impact: Less common, lower priority
   - Target: Full cycle support

2. **hypre binding feature (`amg/hypre`)**
   - Issue: Feature flag not active
   - Target: Optional hypre integration (external library)
   - Blocking: Feature gate

#### 🔨 PARTIAL
1. **ParCsrMatrix (Parallel CSR)**
   - Status: Implemented via ChannelBackend (thread-based)
   - Issue: Thread-based, not MPI; documented as partial

2. **ParVector (Parallel Vector)**
   - Status: Implemented via ChannelBackend
   - Issue: Thread-based, not MPI; documented as partial

---

### 8. PARALLEL INFRASTRUCTURE (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED
1. **Full MPI support**
   - Currently: ChannelBackend (in-process threading only)
   - Target: Real MPI communicators
   - Blocking: Phase 10 (partially implemented)

#### 🔨 PARTIAL
1. **ParCsrMatrix, ParVector thread-based vs MPI**
   - Status: Functional via ChannelBackend
   - Issue: Not true MPI, documented limitation
   - Note: Sufficient for single-machine parallelism

---

### 9. I/O & VISUALIZATION (🔲 Planned | ❌ Out-of-Scope)

#### 🔲 PLANNED
1. **GMSH `.msh` v2 format reader**
   - Issue: Only v4.1 ASCII supported
   - Priority: Phase 9+

2. **GMSH `.msh` v4.1 binary format reader**
   - Issue: Only ASCII v4.1 supported
   - Priority: Phase 9

3. **Netgen `.vol` format reader**
   - Issue: Not supported
   - Priority: Phase 9+

4. **Abaqus `.inp` format reader**
   - Issue: Not supported
   - Priority: Phase 9+

5. **HDF5 / XDMF support (read/write)**
   - Issue: Feature gate `io/hdf5` not active
   - Priority: Phase 2+
   - Target: Parallel I/O, restart files

6. **GridFunction::Load()**
   - Issue: Solution loading from files missing
   - Target: Deserialize DOF vectors from VTK/HDF5

7. **Restart files (HDF5 mesh + solution)**
   - Issue: No checkpoint/restart infrastructure
   - Priority: Phase 2+

#### ❌ OUT-OF-SCOPE
1. **ParaView GLVis socket protocol**
   - Decision: Out of scope (use offline VTK instead)

---

### 10. GRID FUNCTIONS & POST-PROCESSING (🔲 Planned | 🔨 Partial)

#### 🔲 PLANNED
1. **GridFunction::ComputeH1Error()**
   - Issue: Only L² error implemented
   - Target: H¹ seminorm error: ∫|∇(u-u_h)|² dx

2. **GridFunction::GetCurl()**
   - Issue: Gradient recovery exists (ZZ estimator) but not curl
   - Impact: Needed for Maxwell solution post-processing

3. **GridFunction::GetDivergence()**
   - Issue: Not implemented
   - Impact: Darcy/RT solution analysis

4. **KellyErrorEstimator**
   - Issue: Only Zienkiewicz-Zhu (ZZ) estimator implemented
   - Target: Alternative error estimate based on edge jumps

5. **DiscreteLinearOperator**
   - Issue: Gradient, curl, div operators as matrix operators not exposed
   - Impact: Explicit operator assembly (e.g., for decoupled solvers)

#### 🔨 PARTIAL
1. **GridFunction type**
   - Status: Currently `Vec<f64>` + separate `FESpace` reference
   - Issue: No wrapper type
   - Design: Intentional separation of concerns (DOF vector vs space)

---

### 11. MFEM EXAMPLES NOT YET IMPLEMENTED

#### 🔲 PLANNED / 🔨 PARTIAL

| Example | PDE | FE Space | Status | Notes |
|---------|-----|----------|--------|-------|
| **ex4** (Darcy nonlinear) | −∇·(**u**) = f; **u** = −κ∇p | H(div) RT + L² | 🔨 | RT space done, full ex4 with nonlinearity pending |
| **ex10** (wave equation) | ∂²u/∂t² − ∇²u = 0 | H¹ P1 | 🔲 | Requires Leapfrog or Newmark time integrator (Phase 7+) |
| **ex15** (DG advection + AMR) | ∂u/∂t + b·∇u = 0 with AMR | L² DG + refinement | 🔲 | Phase 6+: Combine ex9_dg_advection + ex6 AMR |
| **ex19** (Navier-Stokes) | ∇²**u** − ∇p = **f**; ∇·**u** = 0 | [H¹]² + L² | 🔲 | Phase 7+: Nonlinear convection + pressure BC |
| **pex1** (Parallel Poisson) | −∇²u = 1 (parallel) | H¹ P1 + MPI | 🔲 | Phase 10: ParallelMesh + ghost exchange |
| **pex2** (Parallel Darcy) | Mixed Poisson (parallel) | H(div)×L² + MPI | 🔲 | Phase 10 |
| **pex3** (Parallel Maxwell) | ∇×∇×**u** + **u** = **f** (parallel) | H(curl) + MPI | 🔲 | Phase 10 |
| **pex5** (Parallel Darcy) | Darcy flow (parallel) | H(div)×L² + MPI | 🔲 | Phase 10 |

---

## SUMMARY BY PRIORITY

### TIER 1 — Critical for Phase Completeness
- [ ] **ex10** (wave equation) — new time integrator (Leapfrog/Newmark)
- [ ] **ex15** (DG+AMR) — combine existing ex9 + ex6 features
- [ ] **ex19** (Navier-Stokes) — nonlinear + pressure constraint
- [ ] **Parallel examples (pex1, pex2, pex3, pex5)** — Phase 10 feature gate
- [ ] **H1Error estimator** — post-processing capability
- [ ] **Mixed element mesh support** — real-world mesh format variety

### TIER 2 — Feature Completeness
- [ ] **Gauss-Lobatto quadrature**
- [ ] **Flexible GMRES (FGMRES)**
- [ ] **Chebyshev smoother (AMG)**
- [ ] **HDF5/XDMF I/O** — parallel restart capability
- [ ] **Additional mesh formats** (GMSH v2, v4 binary, Netgen, Abaqus)
- [ ] **DiscreteLinearOperator** — explicit grad/curl/div operators

### TIER 3 — Polish & Extensions
- [ ] **Matrix addition** (`SparseMatrix::Add`)
- [ ] **SetPrintLevel logging integration**
- [ ] **F-cycle (AMG)**
- [ ] **Block triangular preconditioner**
- [ ] **GetBoundingBox utility**
- [ ] **Vector slice assignment** (`SetSubVector`)
- [ ] **Non-conforming mesh (NCMesh)** — advanced AMR

### TIER 4 — Out-of-Scope (Design Decisions)
- ❌ ParaView GLVis socket protocol
- ❌ NURBS isogeometric FEM
- ❌ Direct MFEM mesh format reading

---

## NOTES BY PHASE

### Phase 10 (Parallel) — Biggest Blocker
**Status**: Partially implemented (threading backend)  
**Blocking**: pex1–pex5 examples  
**Requirements**:
- [ ] Full MPI support (or document threading-only limitation)
- [ ] `FES::GetTrueDofs()` for distributed DOF ownership
- [ ] `FES::TransferToTrue()` for parallel assembly
- [ ] Testing on multi-node systems (currently in-process threads only)

### Phase 2+ (Mesh Features)
**Status**: Simplex meshes only  
**Blocking**: Real-world GMSH mesh variety  
**Requirements**:
- [ ] Mixed element types (Tri+Quad, Tet+Hex+Wedge)
- [ ] Netgen/Abaqus readers
- [ ] GMSH v2 and binary v4 support
- [ ] NCMesh for advanced AMR

### Post-Processing (Grid Functions)
**Status**: L² error only  
**Needed for**: Error analysis workflows  
**Requirements**:
- [ ] H¹ error (`ComputeH1Error`)
- [ ] Curl extraction (`GetCurl`)
- [ ] Divergence extraction (`GetDivergence`)

---

## Implementation Recommendations

1. **Quick Wins** (1–2 phases):
   - ex10 (wave) — implement Newmark/Leapfrog time integrator
   - ex15 (DG+AMR) — combine existing pieces
   - H1Error — parallel construction to L2Error code
   - Gauss-Lobatto — add to quadrature table

2. **Medium Effort** (Phase 9–10):
   - Additional mesh readers (GMSH v2, Netgen)
   - HDF5/XDMF feature (optional gate)
   - Parallel example pex1 (leverages Phase 10 infrastructure)

3. **Long-term** (Phase 2+):
   - Mixed element mesh support
   - NCMesh / advanced AMR
   - Full MPI layer (if needed)

---

## Files Checked

- **Mapping**: `/Users/alex/works/fem-rs/MFEM_MAPPING.md` (520 lines)
- **Examples**: 14 example/debug files in `examples/` directory
- **Analysis Date**: 2026-04-04

