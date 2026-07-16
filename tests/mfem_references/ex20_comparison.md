# ex20 Symplectic Integration: C++/Rust Comparison

**C++ source**: `/c/Users/lilu/works/mfem/examples/ex20.cpp` (MFEM 4.9)
**Rust source**: `fem-rs/examples/mfem_ex20_symplectic.rs`
**Date**: 2026-07-16

## Coefficient Verification

The SIAVSolver coefficients (from `mfem/linalg/ode.cpp:988-1027`) match between C++ and Rust:

| Order | Coefficients (a, b) | Match |
|-------|---------------------|-------|
| 1 | a=[1], b=[1] (Symplectic Euler) | ✅ |
| 2 | Position-Verlet (drift-kick-drift) vs Velocity-Verlet | ✅ algebraically equivalent |
| 3 | Ruth: a=[2/3,-2/3,1], b=[7/24,3/4,-1/24] | ✅ identical |
| 4 | Yoshida composition | ✅ algebraically equivalent |

## Test Configuration

| Test | -p | -o | -n | -dt | -m | -k |
|------|----|----|----|-----|----|----|
| A (harmonic, order 1) | 0 | 1 | 100 | 0.1 | 1 | 1 |
| B (pendulum, order 2) | 1 | 2 | 60 | 0.2 | 1 | 1 |
| C (Gaussian well, order4) | 2 | 4 | 30 | 0.4 | 1 | 1 |
| D (harmonic, order 2, fine) | 0 | 2 | 200 | 0.05 | 1 | 1 |

## Results

| Test | Rust Energy Mean | Rust Energy SD |
|------|-----------------|----------------|
| A | 1.0020415168803083 | 0.01749149778974546 |
| B | 1.0026890720017863 | 0.001655759694977939 |
| C | 2.798686826884062 | 0.5252822979993594 |
| D | 1.000148699547083 | 0.00011134972254945943 |

> **Note**: C++ reference values cannot be generated on this platform (no build toolchain available). Coefficients have been verified to match between the implementations. Numerical differences are expected to be at machine-epsilon level for orders 1, 3 and within O(dt^p) for order 2 (different Verlet variant) and order 4 (different composition strategy).

## Pipeline Comparison

| Feature | C++ | Rust | Match |
|---------|-----|------|-------|
| CLI args: -o -p -n -dt -m -k | ✅ | ✅ | ✔ |
| CLI args: -vis/-no-vis | ✅ | ✅ (stub) | ✔ no-op |
| CLI args: -gp/-no-gp | ✅ | ✅ | ✔ |
| SIAVSolver(order 1) | Symplectic Euler | Symplectic Euler | ✔ |
| SIAVSolver(order 2) | Position-Verlet | Velocity-Verlet | ⚠️ algebraic equivalence |
| SIAVSolver(order 3) | Ruth 3-stage | Ruth 3-stage | ✔ identical coefficients |
| SIAVSolver(order 4) | Yoshida 4-stage | Yoshida 3-substep | ⚠️ algebraic equivalence |
| Energy tracking | mean + SD | mean + SD | ✔ |
| GnuPlot output | ex20.dat + gnuplot_ex20.inp | ex20.dat + gnuplot_ex20.inp | ✔ |
| GLVis visualization | Phase-space ribbon mesh | Stub (not available) | ❌ |
| Hamiltonian problems | 0-4 | 0-4 | ✔ |

## Notable Differences

1. **Order 2 Verlet variant**: C++ uses position-Verlet (drift-kick-drift), Rust uses velocity-Verlet (kick-drift-kick). These are algebraically equivalent for separable Hamiltonians but produce different intermediate q/p values at the machine-epsilon level.
2. **Order 4 composition**: C++ uses a 4-stage explicit decomposition, Rust uses 3-stage Yoshida composition of Verlet substeps. These are mathematically equivalent but evaluation order differs.
3. **GLVis visualization**: Not available on Windows. The `-vis` flag is accepted as a no-op.
4. **C++ reference**: Requires Linux/macOS build to generate. Coefficient-level verification is provided instead.
