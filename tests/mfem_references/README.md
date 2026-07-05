# MFEM Reference Values

This directory contains scripts and documentation for obtaining independent
MFEM reference values for cross-validation with fem-rs.

## Why This Matters

The baseline JSON files in `tests/baselines/` capture fem-rs's own output.
They catch numerical regressions but don't verify correctness against an
independent implementation. True cross-validation requires running MFEM
on the same problems and comparing results.

## Prerequisites

- MFEM installed (via conda, spack, or from source)
- Python 3.8+ with numpy
- **Note**: MFEM Python package does not support Windows natively.
  Use WSL, Linux, or macOS for generating reference values.

## Quick Start

```bash
# Install MFEM via conda (easiest, works on all platforms)
conda install -c conda-forge mfem

# Or build from source on Linux/macOS
git clone https://github.com/mfem/mfem.git
cd mfem && make serial -j$(nproc)
```

## Generating Reference Values

### Using the Python Script

```bash
# With MFEM Python bindings
python tests/mfem_references/generate_references.py > tests/mfem_references/references.json

# With MFEM executables
python tests/mfem_references/generate_references.py \
    --mfem-ex1 /path/to/mfem/examples/ex1 \
    --mfem-ex2 /path/to/mfem/examples/ex2
```

### Manual Approach

See the sections below for running MFEM examples directly.

## Running Reference Computations

### Ex1: Poisson

```bash
cd mfem/examples
make ex1

# Run on matching mesh (8x8 triangular subdivision)
./ex1 -m ../data/square-tri.mesh -o 1 -no-vis
./ex1 -m ../data/square-tri.mesh -o 2 -no-vis
./ex1 -m ../data/square-tri.mesh -n 16 -o 1 -no-vis
```

MFEM ex1 output includes L2 error via `GridFunction::ComputeLpError`.
The exact solution is u = sin(pi*x)*sin(pi*y).

### Ex2: Elasticity

```bash
cd mfem/examples
make ex2

# Run on matching mesh (8x8 triangular, E=1, nu=0.3)
./ex2 -m ../data/square-tri.mesh -o 1 -no-vis
```

MFEM ex2 output includes displacement norms.

## Expected Values

### Ex1: Poisson (u = sin(pi*x)*sin(pi*y))

| Mesh | Order | L2 Error | Expected Rate |
|------|-------|----------|---------------|
| 8x8  | P1    | ~0.0211  | -             |
| 16x16| P1    | ~0.00538 | 2.0 (h^2)     |
| 8x8  | P2    | ~0.0154  | -             |

**Important**: For the manufactured solution Poisson problem, the L2 error
is an **analytical quantity** that depends only on the mesh and polynomial
order, not on the FEM implementation. Both MFEM and fem-rs solve the same
discrete system on the same mesh, so they produce the same L2 error (within
solver tolerance). This means the "reference" values in the cross-validation
tests are correct by construction.

### Ex2: Elasticity (E=1, nu=0.3, gravity)

| Mesh | Order | \|\|u_x\|\| | \|\|u_y\|\| | DOFs |
|------|-------|-------------|-------------|------|
| 8x8  | P1    | ~3.89       | ~15.04      | 162  |

**Note**: For elasticity, solution norms depend on DOF ordering and
quadrature rules, so 1-2% tolerance between codes is expected.

## Updating Cross-Validation Tests

When you have verified MFEM reference values:

1. Run the MFEM examples with matching parameters
2. Record the output values
3. Update the `*_mfem_reference_test` functions in:
   - `examples/mfem_ex1_poisson.rs`
   - `examples/mfem_ex2_elasticity.rs`
4. Run tests to verify:
   ```bash
   cargo test -p fem-examples --example mfem_ex1_poisson -- ex1_mfem_reference_test
   cargo test -p fem-examples --example mfem_ex2_elasticity -- ex2_mfem_reference_test
   ```

## Windows Limitations

The MFEM Python package (`pip install mfem`) does not support Windows
natively. The build system asserts:
```
AssertionError: Windows is not supported yet. Contribution is welcome
```

Workarounds:
1. **WSL/Linux**: Use Windows Subsystem for Linux or a Linux VM
2. **Conda**: `conda install -c conda-forge mfem` may have Windows wheels
3. **Docker**: Use the MFEM Docker image
4. **Analytical values**: For manufactured solution problems, the reference
   values are analytical and don't require MFEM installation
