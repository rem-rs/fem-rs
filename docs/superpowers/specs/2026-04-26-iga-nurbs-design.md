# IGA/NURBS Minimal End-to-End Design (Phase 1)

Date: 2026-04-26  
Status: Proposed (approved in chat, pending implementation)

## 1. Goal and Scope

Deliver a minimal, production-style IGA/NURBS path in fem-rs with two milestones:

1. 1D NURBS Poisson end-to-end.
2. 2D single-patch tensor-product NURBS Poisson end-to-end.

This phase emphasizes correctness, testability, and compatibility with current fem-rs assembly/solver patterns. It does not attempt full MFEM NURBS parity in one step.

## 2. Non-Goals (Phase 1)

- Multi-patch coupling and continuity constraints.
- T-splines, LR-splines, hierarchical splines.
- NURBS trimming and CAD kernel interoperability.
- Full MFEM NURBS mesh format compatibility layer.
- GPU-specific kernels.

## 3. User-Facing Outcome

After Phase 1, users can:

- Define clamped knot vectors, control points, and weights.
- Build 1D and 2D (single patch) NURBS spaces.
- Assemble mass/diffusion matrices and solve Poisson-like problems.
- Export sampled solutions for visualization (reuse current VTK path).

## 4. Architecture

### 4.1 `fem-element` (basis/evaluation core)

Add an `iga` module containing:

- `KnotVector`: validates monotone clamped knot vectors.
- `BsplineBasis`: Cox-de Boor basis and first derivatives.
- `NurbsBasis`: rational basis and first derivatives from B-splines + weights.
- Span helpers:
  - `find_span(u, p, knots)`
  - `active_basis_indices(span, p)`

Design rule: this crate owns pure math/evaluation and remains independent of global DOF/mesh assembly concerns.

### 4.2 `fem-space` (DOF and connectivity)

Add:

- `IgaSpace1D`
- `IgaSpace2D` (tensor-product patch)

Responsibilities:

- Global DOF numbering (control-point indexed).
- Element/span iteration over non-empty knot spans.
- Element-to-global DOF map.
- Essential boundary DOF queries.

Data model:

- 1D: `knots_u`, degree `p`, `ctrl_pts`, `weights`.
- 2D: `knots_u`, `knots_v`, degrees `p`, `q`, grid of control points + weights.

### 4.3 `fem-assembly` (IGA assembly driver)

Add a focused `iga_assembler` module:

- `assemble_bilinear_mass_iga`
- `assemble_bilinear_diffusion_iga`
- `assemble_linear_source_iga`

Implementation notes:

- Integrate per knot-span element.
- Use Gaussian quadrature mapped from parent interval/square into knot span.
- Build Jacobian from geometric map (NURBS geometry).
- Scatter local contributions into COO then CSR (consistent with existing pipeline).

### 4.4 Examples

Add two examples:

- `mfem_ex_iga_poisson_1d.rs`
- `mfem_ex_iga_poisson_2d_patch.rs`

Both examples:

- Assemble `K u = f` with Dirichlet BC.
- Solve with existing linear solver path.
- Report basic error/consistency metric.

## 5. Error Handling and Contracts

- Invalid knot vectors return `FemError` (not panic).
- Non-positive weights rejected.
- Degenerate Jacobian at quadrature point returns `FemError`.
- Boundary condition API mirrors existing `apply_dirichlet` style behavior.

## 6. Parallelization Plan

Phase 1 keeps IGA assembly serial-first for clarity and baseline correctness.

Phase 1b (follow-up) adds Rayon-based element/span parallelism using the same threshold strategy as current assembly code:

- local `CooMatrix` per worker
- reduction via append/merge
- optional env threshold control

## 7. Testing Strategy

### 7.1 Unit tests (`fem-element`)

- Partition of unity on valid spans.
- Local support correctness.
- Derivative finite-difference sanity.
- NURBS with all weights = 1 matches B-spline.

### 7.2 Space tests (`fem-space`)

- Correct DOF count in 1D and 2D tensor-product cases.
- Element/span to global DOF mapping correctness.
- Boundary DOF extraction.

### 7.3 Assembly tests (`fem-assembly`)

- Matrix dimensions and symmetry for mass/diffusion.
- Positive diagonal checks for diffusion.
- Constant source integral sanity.

### 7.4 Example-level checks

- 1D Poisson convergence trend under knot refinement.
- 2D single-patch Poisson convergence trend under refinement.

## 8. Incremental Delivery Plan

1. `fem-element` iga basis kernel + tests.
2. `fem-space` `IgaSpace1D` + minimal 1D assembly and test.
3. 1D end-to-end example.
4. `IgaSpace2D` tensor-product + 2D assembly.
5. 2D end-to-end example.
6. Docs updates (`MFEM_MAPPING.md`, crate README notes).

## 9. Risks and Mitigations

- Basis derivative bugs: use FD cross-check tests and known analytical cases.
- Geometry Jacobian sign/scale mistakes: add determinant sanity tests and manufactured solutions.
- Scope creep (multi-patch): explicitly deferred to Phase 2.

## 10. Acceptance Criteria

This design is accepted when all are true:

- 1D and 2D single-patch NURBS Poisson examples compile and run.
- New IGA unit and integration tests pass in CI.
- Existing non-IGA assembly tests remain green.
- `MFEM_MAPPING.md` updated from "out-of-scope" toward "partial/phase-1" for NURBS/IGA.

