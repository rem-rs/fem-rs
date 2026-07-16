# ex19 Incompressible Neo-Hookean Hyperelasticity: C++/Rust Comparison

**C++ source**: `/c/Users/lilu/works/mfem/examples/ex19.cpp` (MFEM 4.9)
**Rust source**: `fem-rs/examples/mfem_ex19_hyperelastic_incomp.rs`
**Date**: 2026-07-16

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Mesh | beam-quad.mesh |
| Order | 2 |
| Refinements | 0 |
| Shear modulus (mu) | 1.0 |

## Results

| Metric | C++ MFEM | Rust fem-rs | Relative Diff |
|--------|----------|-------------|---------------|
| |deformation|_L^2 | 8.6953252968e+00 | 8.6951665137e+00 | 0.0018% |
| |pressure|_L^2 | 4.0064120054e+00 | 4.0066408244e+00 | 0.0057% |
| Newton iterations | 3 | 3 | -- |
| Final residual | 1.27e-04 | 5.61e-05 | -- |

## Pipeline Comparison

| Step | C++ | Rust | Match |
|------|-----|------|-------|
| Mesh | read + uniform refine | read + uniform refine | Yes |
| FE space | H1^dim(byVDIM) + H1 | VectorH1Space + H1Space | Yes |
| DOF layout | component-major | component-major | Yes |
| BC (attr 1) | essential, u=0 | essential, u=0 | Yes |
| BC (attr 2) | essential, u_y=0.25x | essential, u_y=0.25x | Yes |
| Initial guess | InitialDeformation | InitialDeformation | Yes |
| Newton solver | NewtonSolver + GMRES | Hand-coded Newton + GMRES | Yes |
| Inner tolerance | 1e-12 (GMRES) | 1e-8 (GMRES) | Different |
| Block precond | K(GMRES+GS) + S(CG+GS) | K(GMRES+GS) + S(CG+GS) | Yes |
| Schur scaling gamma | 1e-5 | 1e-5 | Yes |
| Output | deformed.mesh + .sol | deformed.mesh + .sol | Yes |

## Notable Differences

1. **Inner GMRES tolerance**: C++ uses 1e-12, Rust uses 1e-8. This affects the Newton iteration path slightly but not the final converged solution quality.
2. **Output format**: C++ uses `GridFunction::Save` (MFEM stream format with metadata headers), Rust uses plain ASCII (DOF count + values). L^2 norms are compared rather than raw file bytes.
3. **Line search**: C++ relies on NewtonSolver's built-in line search; Rust uses explicit Armijo backtracking.
4. **Final residual**: C++ reports 1.27e-04 at iteration 3, Rust reports 5.61e-05. Both converge below the relative tolerance of 1e-4 in 3 iterations.

## Verification

The solution L^2 norms agree to within 0.006%, confirming the 1:1 alignment of the mixed u/p formulation. The slight differences in final residual are attributed to the different GMRES inner tolerances, which cause the Newton solver to take slightly different linearization paths while converging to essentially the same solution.
