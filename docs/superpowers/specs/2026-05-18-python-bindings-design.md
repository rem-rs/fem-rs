# Python Bindings for fem-rs

**Date**: 2026-05-18
**Status**: Draft

## Design Summary

Add Python bindings for the fem-rs Rust FEM library using PyO3 + maturin,
exposing the full FEM pipeline (mesh → space → assembly → linalg → solver)
through a single Python package `fem-rs` that can be installed via pip/uv.

## Goals

- Cover the full FEM pipeline: mesh creation, finite element space, bilinear/linear
  assembly, linear solve
- Single Python package (`fem-rs`) with hierarchical API: `fem.Mesh`, `fem.H1Space`, `fem.solve_cg`, etc.
- Python ≥ 3.11, f64 only (MVP)
- NumPy/SciPy interop: vectors as `numpy.ndarray`, matrices convertible to `scipy.sparse.csr_matrix`
- Layered API (user can compose the full pipeline step by step)

## Non-Goals (MVP)

- MPI parallel / GPU solver / HDF5 I/O / WASM
- f32 support
- HCurlSpace / HDivSpace and AMS/ADS preconditioners
- Zero-copy array views
- Custom Pythion exception types

## Build System

- Single PyO3 crate at `crates/python/` inside the existing workspace
- Root-level `pyproject.toml` for maturin configuration
- Build command: `maturin develop` or `uv pip install -e .`

```
fem-rs/
├── pyproject.toml           # maturin build config
├── python/fem/              # Python package directory
│   └── __init__.py
├── crates/python/
│   ├── Cargo.toml           # deps: PyO3 + workspace crates
│   └── src/
│       ├── lib.rs           # pymodule entry, flat re-exports
│       ├── mesh.rs          # Mesh #[pyclass]
│       ├── space.rs         # H1Space #[pyclass]
│       ├── assembly.rs      # assemble_bilinear / assemble_linear
│       ├── linalg.rs        # CsrMatrix wrapper + to_scipy()
│       └── solver.rs        # solve_cg, solve_gmres, solve_sparse_lu
```

## Python API

### Mesh
```python
mesh = fem.Mesh.unit_square_tri(n=4)     # 2D → SimplexMesh<2>
mesh = fem.Mesh.unit_cube_tet(n=2)       # 3D → SimplexMesh<3>
mesh.n_elements()
mesh.n_nodes()
mesh.boundary_nodes(tags=[1, 2, 3, 4])   # → list[int]
```

### FESpace
```python
V = fem.H1Space(mesh, order=1)           # H1 空间
V.n_dofs()
```

### Assembly
```python
A = fem.assemble_bilinear(V, [
    fem.StiffnessIntegrator(),
    fem.MassIntegrator(alpha=1.0),
])
b = fem.assemble_linear(V, fem.DomainLoad(f=1.0))
fem.apply_dirichlet(A, b, boundary_dofs)  # 原地修改
```

### 线性代数
```python
A.to_scipy()                              # → scipy.sparse.csr_matrix
```

### Solvers
```python
x = np.zeros(V.n_dofs())

# CG (SPD)
result = fem.solve_cg(A, b, x, tol=1e-8, max_iter=1000)
# GMRES
result = fem.solve_gmres(A, b, x, restart=30, tol=1e-8)
# Sparse LU direct
x = fem.solve_sparse_lu(A, b)

result.converged      # bool
result.iterations     # int
result.final_residual # float
```

## Rust Implementation

### Crate Layout

`crates/python/Cargo.toml`:
```toml
[package]
name = "fem-py"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = { version = "0.23", package = "pyo3-numpy" }
fem-core = { path = "../../core" }
fem-mesh = { path = "../../mesh" }
fem-space = { path = "../../space" }
fem-element = { path = "../../element" }
fem-assembly = { path = "../../assembly" }
fem-linalg = { path = "../../linalg" }
fem-solver = { path = "../../solver" }
```

### Key Design Decisions

1. **Matrix→SciPy conversion**: `CsrMatrix` gets a `to_scipy()` method that builds
   `(data, indices, indptr)` triple and calls `scipy.sparse.csr_matrix` via PyO3 —
   cheapest interoperability without owning Python memory layout.

2. **Vec → ndarray**: Solver input/output uses `numpy.ndarray`. The Rust solver
   works on `Vec<f64>`, and we copy between ndarray ↔ Vec at the boundary.

3. **apply_dirichlet**: Mutates both matrix and RHS in-place, mirroring the Rust
   `FemCsr::apply_dirichlet_symmetric` behavior.

## Testing

- Python tests in `tests/` (alongside pyproject.toml), run via `pytest`
- 1-D Laplacian benchmark (same as Rust unit tests): assemble + solve CG → verify residual
- 2-D Poisson on unit square: full pipeline test

## Future Extensions (Post-MVP)

- f32 support
- HCurl / HDiv spaces + AMS/ADS
- GPU backend exposure
- MPI-parallel solvers
- HDF5 checkpoint read/write
- Lazy / zero-copy numpy views
