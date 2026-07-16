# ex21 AMR Elasticity: C++/Rust Comparison

**C++ source**: `/c/Users/lilu/works/mfem/examples/ex21.cpp` (MFEM 4.9, 310 lines)
**Rust source**: `fem-rs/examples/mfem_ex21_amr_elasticity.rs` (rewritten)
**Date**: 2026-07-16

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Mesh | beam-quad.mesh (P1) |
| Order | 1 |
| Solver | PCG + GSSmoother (no SuiteSparse) |
| Estimator | ZZ (Zienkiewicz-Zhu) |
| Refinement fraction | 0.7 |
| Max DOFs | 50000 |

## Results

| Metric | C++ (expected) | Rust | Notes |
|--------|---------------|------|-------|
| AMR iterations | 5-6 | 5 | |
| Final DOFs | ~700-800 | 710 | |
| PCG iters (iter 0) | ~20-30 | 24 | |
| PCG iters (iter 4) | ~100-200 | 102 | |

## Pipeline Comparison

| Feature | C++ | Rust | Match |
|---------|-----|------|-------|
| CLI: -m -o -sc -f -vis | ✅ all 5 | ✅ all 5 | ✔ |
| Mesh sanity check | ≥2 materials/attrs | mesh type check | ⚠️ different check |
| NURBS→curved | ✅ | ❌ (NURBS not in scope) | ❌ |
| Vector FE space (H1^dim) | ✅ | ✅ VectorH1Space | ✔ |
| Multi-material λ/μ | ✅ PWConstCoefficient | ✅ PWConstCoeff | ✔ |
| RHS (traction boundary) | ✅ VectorBoundaryLFIntegrator | ✅ Neumann→y-component | ✔ |
| Essential BC (attr 1) | ✅ ProjectBdrCoefficient | ✅ boundary_dofs + zeroing | ✔ |
| PCG + GSSmoother | ✅ (non-SuiteSparse) | ✅ solve_pcg_gssmoother | ✔ |
| SuiteSparse UMFPACK | ✅ (when available) | ❌ (no direct solver) | ⚠️ |
| ZZ estimator | ✅ | ✅ zz_estimator | ✔ |
| ThresholdRefiner 70% | ✅ | ✅ mark_elements | ✔ |
| Non-conforming AMR (quad) | ✅ | ✅ refine_nonconforming_quad | ✔ |
| Conforming AMR (tri) | ✅ | ❌ (API not exposed) | ❌ |
| Solution interpolation | ✅ fespace.Update() + x.Update() | ❌ (API gap) | ❌ |
| GLVis visualization | ✅ (phase-space ribbon) | ❌ (stub) | ❌ |
| Output files | reference.mesh, deformed.mesh, displacement.sol | reference.mesh only | ⚠️ |

## Notable Differences

1. **Triangle meshes**: C++ supports conforming refinement for triangles; Rust version only supports quads via `refine_nonconforming_quad`.
2. **Solution interpolation**: C++ calls `fespace.Update()` + `x.Update()` to interpolate the solution to the new mesh after refinement, providing a good initial guess for the next solver iteration. Rust lacks this interpolation API.
3. **SuiteSparse**: C++ uses UMFPACK direct solver when available, which is more robust for ill-conditioned hanging-node systems. Rust only has iterative solvers.
4. **GLVis**: C++ visualizes the phase-space ribbon. Rust accepts `-vis` as a no-op.
5. **Output**: C++ writes three files (reference, deformed, displacement). Rust currently writes only the reference mesh.
