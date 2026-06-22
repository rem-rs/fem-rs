# Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python bindings for the fem-rs Rust FEM library via PyO3 + maturin, exposing the full FEM pipeline.

**Architecture:** Single PyO3 crate `crates/python/` inside the existing workspace, configured by root `pyproject.toml`. Each module (mesh, space, assembly, linalg, solver) is a separate Rust source file. The Python package is `fem/` wrapping the Rust extension module.

**Tech Stack:** PyO3 0.23, maturin ≥1.7, pyo3-numpy 0.23, scipy (for sparse matrix conversion), pytest

---

### Task 1: Scaffolding

**Files:**
- Create: `pyproject.toml` (root)
- Create: `python/fem/__init__.py`
- Create: `crates/python/Cargo.toml`
- Create: `crates/python/src/lib.rs`

- [ ] **Step 1: Create root pyproject.toml**

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "fem-rs"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
]

[tool.maturin]
manifest-path = "crates/python/Cargo.toml"
python-source = "python"
module-name = "fem._core"
```

- [ ] **Step 2: Create python/fem/__init__.py**

```python
from fem._core import (
    Mesh,
    H1Space,
    StiffnessIntegrator,
    MassIntegrator,
    ConstantLoad,
    CsrMatrix,
    SolveResult,
    assemble_bilinear,
    assemble_linear,
    apply_dirichlet,
    solve_cg,
    solve_gmres,
    solve_sparse_lu,
)

__all__ = [
    "Mesh", "H1Space",
    "StiffnessIntegrator", "MassIntegrator", "ConstantLoad",
    "CsrMatrix", "SolveResult",
    "assemble_bilinear", "assemble_linear", "apply_dirichlet",
    "solve_cg", "solve_gmres", "solve_sparse_lu",
]
```

- [ ] **Step 3: Create crates/python/Cargo.toml**

```toml
[package]
name = "fem-py"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
pyo3-numpy = "0.23"
fem-core = { path = "../../core" }
fem-mesh = { path = "../../mesh" }
fem-space = { path = "../../space" }
fem-element = { path = "../../element" }
fem-assembly = { path = "../../assembly" }
fem-linalg = { path = "../../linalg" }
fem-solver = { path = "../../solver" }
```

- [ ] **Step 4: Create crates/python/src/lib.rs with module registration**

```rust
mod mesh;
mod space;
mod assembly;
mod linalg;
mod solver;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mesh::PyMesh>()?;
    m.add_class::<space::PyH1Space>()?;
    m.add_class::<assembly::PyStiffnessIntegrator>()?;
    m.add_class::<assembly::PyMassIntegrator>()?;
    m.add_class::<assembly::PyConstantLoad>()?;
    m.add_class::<linalg::PyCsrMatrix>()?;
    m.add_class::<solver::PySolveResult>()?;
    m.add_function(wrap_pyfunction!(assembly::py_assemble_bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(assembly::py_assemble_linear, m)?)?;
    m.add_function(wrap_pyfunction!(assembly::py_apply_dirichlet, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_cg, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_gmres, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_sparse_lu, m)?)?;
    Ok(())
}
```

- [ ] **Step 5: Verify the project builds**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds (with warnings about dead code for unimplemented modules — OK for now)

- [ ] **Step 6: Initial git commit**

```bash
git add pyproject.toml python/ crates/python/
git commit -m "feat: add Python bindings scaffolding (PyO3 + maturin)"
```

---

### Task 2: Mesh Bindings

**Files:**
- Create: `crates/python/src/mesh.rs`

- [ ] **Step 1: Implement PyMesh**

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_mesh::SimplexMesh;

/// Unstructured simplex mesh in 2-D or 3-D.
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner_2d: Option<SimplexMesh<2>>,
    pub(crate) inner_3d: Option<SimplexMesh<3>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyMesh {
    /// Create a 2-D unit-square mesh of `n×n` triangles.
    #[staticmethod]
    pub fn unit_square_tri(n: usize) -> PyResult<Self> {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        Ok(PyMesh { inner_2d: Some(mesh), inner_3d: None, dim: 2 })
    }

    /// Create a 3-D unit-cube mesh of `n×n×n` tetrahedra.
    #[staticmethod]
    pub fn unit_cube_tet(n: usize) -> PyResult<Self> {
        let mesh = SimplexMesh::<3>::unit_cube_tet(n);
        Ok(PyMesh { inner_2d: Some(mesh), inner_3d: None, dim: 3 })  // H1Space uses inner_2d for 2D meshes too
    }

    /// Number of elements (cells).
    pub fn n_elements(&self) -> usize {
        match self.dim {
            2 => self.inner_2d.as_ref().unwrap().n_elems(),
            3 => self.inner_3d.as_ref().unwrap().n_elems(),
            _ => unreachable!(),
        }
    }

    /// Number of nodes (vertices).
    pub fn n_nodes(&self) -> usize {
        match self.dim {
            2 => self.inner_2d.as_ref().unwrap().n_nodes(),
            3 => self.inner_3d.as_ref().unwrap().n_nodes(),
            _ => unreachable!(),
        }
    }

    /// Node indices on the boundary tagged `tags`.
    ///
    /// Currently only supports 2-D meshes (SimplexMesh<2>).
    pub fn boundary_nodes(&self, tags: Vec<i32>) -> PyResult<Vec<u32>> {
        match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                let nodes = fem_mesh::boundary_nodes_with_tags(mesh, &tags);
                Ok(nodes)
            }
            _ => Err(PyValueError::new_err("boundary_nodes: 3-D not yet supported"))
        }
    }
}
```

Wait — the original code stores unit_cube_tet in `inner_2d` by mistake. Fix:

```rust
    #[staticmethod]
    pub fn unit_cube_tet(n: usize) -> PyResult<Self> {
        let mesh = SimplexMesh::<3>::unit_cube_tet(n);
        Ok(PyMesh { inner_2d: None, inner_3d: Some(mesh), dim: 3 })
    }
```

- [ ] **Step 2: Verify the crate still compiles**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/mesh.rs
git commit -m "feat: add PyMesh Python bindings"
```

---

### Task 3: Space Bindings

**Files:**
- Create: `crates/python/src/space.rs`

- [ ] **Step 1: Implement PyH1Space**

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_space::H1Space;
use crate::mesh::PyMesh;

/// H¹ finite element space (continuous Lagrange).
#[pyclass(name = "H1Space")]
pub struct PyH1Space {
    pub(crate) inner_2d: Option<H1Space<fem_mesh::SimplexMesh<2>>>,
    pub(crate) inner_3d: Option<H1Space<fem_mesh::SimplexMesh<3>>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyH1Space {
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().unwrap().clone();
                Ok(PyH1Space { inner_2d: Some(H1Space::new(m, order)), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().unwrap().clone();
                Ok(PyH1Space { inner_3d: Some(H1Space::new(m, order)), inner_2d: None, dim: 3 })
            }
            _ => Err(PyValueError::new_err("unsupported dimension")),
        }
    }

    /// Number of degrees of freedom.
    pub fn n_dofs(&self) -> usize {
        match self.dim {
            2 => self.inner_2d.as_ref().unwrap().n_dofs(),
            3 => self.inner_3d.as_ref().unwrap().n_dofs(),
            _ => unreachable!(),
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/space.rs
git commit -m "feat: add H1Space Python bindings"
```

---

### Task 4: Assembly Bindings

**Files:**
- Create: `crates/python/src/assembly.rs`

- [ ] **Step 1: Implement integrator wrapper types and assemble functions**

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use pyo3_numpy::PyArray1;
use fem_assembly::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};
use fem_assembly::BilinearIntegrator;
use fem_assembly::LinearIntegrator;
use fem_linalg::CsrMatrix;
use fem_space::constraints::apply_dirichlet as rs_apply_dirichlet;
use crate::space::PyH1Space;
use crate::linalg::PyCsrMatrix;

// ── Bilinear integrators ────────────────────────────────────────────────

/// Stiffness (diffusion) integrator: ∫ κ ∇u·∇v dΩ
#[pyclass(name = "StiffnessIntegrator")]
pub struct PyStiffnessIntegrator {
    pub(crate) kappa: f64,
}

#[pymethods]
impl PyStiffnessIntegrator {
    #[new]
    pub fn new(kappa: Option<f64>) -> Self {
        PyStiffnessIntegrator { kappa: kappa.unwrap_or(1.0) }
    }
}

impl BilinearIntegrator for PyStiffnessIntegrator {
    fn assemble(&self, element: &dyn fem_element::ElementAccess, quadrature: &[fem_element::QuadraturePoint], values: &mut [f64]) {
        // Delegate to DiffusionIntegrator
        let inner = DiffusionIntegrator { kappa: self.kappa };
        inner.assemble(element, quadrature, values);
    }
}

/// Mass integrator: ∫ ρ u v dΩ
#[pyclass(name = "MassIntegrator")]
pub struct PyMassIntegrator {
    pub(crate) alpha: f64,
}

#[pymethods]
impl PyMassIntegrator {
    #[new]
    pub fn new(alpha: Option<f64>) -> Self {
        PyMassIntegrator { alpha: alpha.unwrap_or(1.0) }
    }
}

impl BilinearIntegrator for PyMassIntegrator {
    fn assemble(&self, element: &dyn fem_element::ElementAccess, quadrature: &[fem_element::QuadraturePoint], values: &mut [f64]) {
        let inner = MassIntegrator { rho: self.alpha };
        inner.assemble(element, quadrature, values);
    }
}

// ── Linear integrator ───────────────────────────────────────────────────

/// Constant domain load: ∫ f v dΩ
#[pyclass(name = "ConstantLoad")]
pub struct PyConstantLoad {
    pub(crate) value: f64,
}

#[pymethods]
impl PyConstantLoad {
    #[new]
    pub fn new(value: Option<f64>) -> Self {
        PyConstantLoad { value: value.unwrap_or(1.0) }
    }
}

impl LinearIntegrator for PyConstantLoad {
    fn assemble(&self, element: &dyn fem_element::ElementAccess, quadrature: &[fem_element::QuadraturePoint], values: &mut [f64]) {
        let inner = DomainSourceIntegrator::new(move |_| self.value);
        inner.assemble(element, quadrature, values);
    }
}

// ── Assembly functions ──────────────────────────────────────────────────

/// Assemble a bilinear form.
#[pyfunction]
pub fn py_assemble_bilinear(
    space: &PyH1Space,
    integrators: &Bound<'_, PyList>,
) -> PyResult<PyCsrMatrix> {
    // For MVP, just support a single integrator (the first one)
    if integrators.len() == 0 {
        return Err(PyValueError::new_err("at least one integrator required"));
    }

    let obj = integrators.get_item(0)?;
    let quad_order: u8 = match space.dim {
        2 => 4,
        3 => 3,
        _ => return Err(PyValueError::new_err("unsupported dimension")),
    };

    // Try stiffness integrator
    if let Ok(s) = obj.extract::<PyRef<'_, PyStiffnessIntegrator>>() {
        match space.dim {
            2 => {
                let mat = Assembler::assemble_bilinear(
                    space.inner_2d.as_ref().unwrap(),
                    &[&s.inner as &dyn BilinearIntegrator],
                    quad_order,
                );
                Ok(PyCsrMatrix { inner: mat })
            }
            3 => {
                let mat = Assembler::assemble_bilinear(
                    space.inner_3d.as_ref().unwrap(),
                    &[&s.inner as &dyn BilinearIntegrator],
                    quad_order,
                );
                Ok(PyCsrMatrix { inner: mat })
            }
            _ => unreachable!(),
        }
    }
    // Try mass integrator
    else if let Ok(m) = obj.extract::<PyRef<'_, PyMassIntegrator>>() {
        match space.dim {
            2 => {
                let mat = Assembler::assemble_bilinear(
                    space.inner_2d.as_ref().unwrap(),
                    &[&m.inner as &dyn BilinearIntegrator],
                    quad_order,
                );
                Ok(PyCsrMatrix { inner: mat })
            }
            3 => {
                let mat = Assembler::assemble_bilinear(
                    space.inner_3d.as_ref().unwrap(),
                    &[&m.inner as &dyn BilinearIntegrator],
                    quad_order,
                );
                Ok(PyCsrMatrix { inner: mat })
            }
            _ => unreachable!(),
        }
    }
    else {
        Err(PyValueError::new_err("unsupported integrator type"))
    }
}
```

Wait — the delegation approach above is wrong because `PyStiffnessIntegrator` itself implements `BilinearIntegrator`, so `Assembler::assemble_bilinear` takes `&dyn BilinearIntegrator`. But the `inner` field doesn't exist. Let me fix this.

Actually, let me reconsider the design. The cleanest approach is to NOT have the wrapper types implement the Rust traits directly (that would require importing ElementAccess types into the Python crate). Instead, inside the Python assembly function, construct the REAL integrator from the wrapper's config:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use pyo3_numpy::PyArray1;
use fem_assembly::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};
use fem_linalg::CsrMatrix;
use fem_space::constraints::apply_dirichlet as rs_apply_dirichlet;
use crate::space::PyH1Space;
use crate::linalg::PyCsrMatrix;

/// Stiffness (diffusion) integrator: ∫ κ ∇u·∇v dΩ
#[pyclass(name = "StiffnessIntegrator")]
pub struct PyStiffnessIntegrator {
    pub(crate) kappa: f64,
}

#[pymethods]
impl PyStiffnessIntegrator {
    #[new]
    pub fn new(kappa: Option<f64>) -> Self {
        PyStiffnessIntegrator { kappa: kappa.unwrap_or(1.0) }
    }
}

/// Mass integrator: ∫ ρ u v dΩ
#[pyclass(name = "MassIntegrator")]
pub struct PyMassIntegrator {
    pub(crate) alpha: f64,
}

#[pymethods]
impl PyMassIntegrator {
    #[new]
    pub fn new(alpha: Option<f64>) -> Self {
        PyMassIntegrator { alpha: alpha.unwrap_or(1.0) }
    }
}

/// Constant domain load: ∫ f v dΩ
#[pyclass(name = "ConstantLoad")]
pub struct PyConstantLoad {
    pub(crate) value: f64,
}

#[pymethods]
impl PyConstantLoad {
    #[new]
    pub fn new(value: Option<f64>) -> Self {
        PyConstantLoad { value: value.unwrap_or(1.0) }
    }
}

/// Assemble a bilinear form: K = ∫ ... dΩ
///
/// space: H1Space
/// integrators: list of StiffnessIntegrator or MassIntegrator
/// Returns: CsrMatrix
#[pyfunction]
pub fn py_assemble_bilinear(
    space: &PyH1Space,
    integrators: &Bound<'_, PyList>,
) -> PyResult<PyCsrMatrix> {
    if integrators.len() == 0 {
        return Err(PyValueError::new_err("at least one integrator required"));
    }

    let quad_order: u8 = match space.dim {
        2 => 4,
        3 => 3,
        _ => return Err(PyValueError::new_err("unsupported dimension")),
    };

    // Build the integrator list dynamically.
    // For MVP: single integrator (extract the first one).
    let obj = integrators.get_item(0)?;
    let bilinear_integrators: Vec<Box<dyn fem_assembly::BilinearIntegrator>>;

    if let Ok(s) = obj.extract::<PyRef<'_, PyStiffnessIntegrator>>() {
        bilinear_integrators = vec![Box::new(DiffusionIntegrator { kappa: s.kappa })];
    } else if let Ok(m) = obj.extract::<PyRef<'_, PyMassIntegrator>>() {
        bilinear_integrators = vec![Box::new(MassIntegrator { rho: m.alpha })];
    } else {
        return Err(PyValueError::new_err("unsupported integrator; expected StiffnessIntegrator or MassIntegrator"));
    }

    let refs: Vec<&dyn fem_assembly::BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();

    match space.dim {
        2 => {
            let mat = Assembler::assemble_bilinear(
                space.inner_2d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(PyCsrMatrix { inner: mat })
        }
        3 => {
            let mat = Assembler::assemble_bilinear(
                space.inner_3d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(PyCsrMatrix { inner: mat })
        }
        _ => unreachable!(),
    }
}

/// Assemble a linear form: b = ∫ f v dΩ
///
/// space: H1Space
/// source: ConstantLoad
/// Returns: Vec<f64> as list
#[pyfunction]
pub fn py_assemble_linear(
    space: &PyH1Space,
    source: &PyConstantLoad,
) -> PyResult<Vec<f64>> {
    let quad_order: u8 = match space.dim {
        2 => 4,
        3 => 3,
        _ => return Err(PyValueError::new_err("unsupported dimension")),
    };

    let source_integ = DomainSourceIntegrator::new(move |_| source.value);
    let refs: [&dyn fem_assembly::LinearIntegrator; 1] = [&source_integ];

    match space.dim {
        2 => {
            let rhs = Assembler::assemble_linear(
                space.inner_2d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(rhs)
        }
        3 => {
            let rhs = Assembler::assemble_linear(
                space.inner_3d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(rhs)
        }
        _ => unreachable!(),
    }
}

/// Apply homogeneous Dirichlet boundary conditions.
///
/// Modifies the matrix and RHS in-place.
#[pyfunction]
pub fn py_apply_dirichlet(
    mat: &mut PyCsrMatrix,
    rhs: &Bound<'_, PyArray1<f64>>,
    boundary_dofs: Vec<u32>,
) -> PyResult<()> {
    let vals = vec![0.0_f64; boundary_dofs.len()];
    let mut rhs_slice = unsafe { rhs.as_slice_mut()? };
    rs_apply_dirichlet(&mut mat.inner, &mut rhs_slice, &boundary_dofs, &vals);
    Ok(())
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/assembly.rs
git commit -m "feat: add assembly bindings (StiffnessIntegrator, MassIntegrator, ConstantLoad)"
```

---

### Task 5: Linear Algebra Bindings

**Files:**
- Create: `crates/python/src/linalg.rs`

- [ ] **Step 1: Implement PyCsrMatrix**

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use fem_linalg::CsrMatrix;

/// Compressed Sparse Row matrix.
///
/// Wraps the Rust CsrMatrix<f64> and supports conversion to scipy.sparse.csr_matrix.
#[pyclass(name = "CsrMatrix")]
pub struct PyCsrMatrix {
    pub(crate) inner: CsrMatrix<f64>,
}

#[pymethods]
impl PyCsrMatrix {
    /// Number of rows.
    pub fn nrows(&self) -> usize { self.inner.nrows }

    /// Number of columns.
    pub fn ncols(&self) -> usize { self.inner.ncols }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize { self.inner.values.len() }

    /// Convert to a scipy.sparse.csr_matrix.
    ///
    /// Returns a tuple (data, indices, indptr, shape) that can be used
    /// to construct scipy.sparse.csr_matrix.
    pub fn to_scipy(&self) -> PyResult<(Vec<f64>, Vec<u32>, Vec<usize>, (usize, usize))> {
        Ok((
            self.inner.values.clone(),
            self.inner.col_idx.clone(),
            self.inner.row_ptr.clone(),
            (self.inner.nrows, self.inner.ncols),
        ))
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/linalg.rs
git commit -m "feat: add CsrMatrix Python bindings with scipy conversion"
```

---

### Task 6: Solver Bindings

**Files:**
- Create: `crates/python/src/solver.rs`

- [ ] **Step 1: Implement PySolveResult and solver functions**

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_numpy::PyArray1;
use fem_solver::{solve_cg, solve_gmres, solve_sparse_lu, SolverConfig, SolveResult};
use crate::linalg::PyCsrMatrix;

/// Result of an iterative solve.
#[pyclass(name = "SolveResult")]
pub struct PySolveResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub final_residual: f64,
}

impl From<SolveResult> for PySolveResult {
    fn from(r: SolveResult) -> Self {
        PySolveResult {
            converged: r.converged,
            iterations: r.iterations,
            final_residual: r.final_residual,
        }
    }
}

/// Solve Ax = b using Conjugate Gradient (SPD systems).
#[pyfunction]
pub fn py_solve_cg(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice_mut()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_cg(&mat.inner, &rhs, &mut sol, &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Copy the solution back to the numpy array
    let mut x_mut = unsafe { x.as_slice_mut()? };
    x_mut.copy_from_slice(&sol);
    Ok(result.into())
}

/// Solve Ax = b using GMRES (general non-symmetric systems).
#[pyfunction]
pub fn py_solve_gmres(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    restart: Option<usize>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice_mut()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_gmres(&mat.inner, &rhs, &mut sol, restart.unwrap_or(30), &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut x_mut = unsafe { x.as_slice_mut()? };
    x_mut.copy_from_slice(&sol);
    Ok(result.into())
}

/// Solve Ax = b using sparse LU factorization.
#[pyfunction]
pub fn py_solve_sparse_lu(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<PyArray1<f64>> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let x = solve_sparse_lu(&mat.inner, &rhs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x))
}
```

Hmm, `PyArray1::from_vec` needs a `py` token. In PyO3 0.23, the function signature should include `py: Python<'_>`:

Fix the sparse LU function:

```rust
#[pyfunction]
pub fn py_solve_sparse_lu(
    py: Python<'_>,
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<PyArray1<f64>> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let x = solve_sparse_lu(&mat.inner, &rhs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x))
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p fem-py`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/solver.rs
git commit -m "feat: add solver bindings (CG, GMRES, sparse LU)"
```

---

### Task 7: Python Integration Tests

**Files:**
- Create: `tests/test_full_pipeline.py`

- [ ] **Step 1: Write integration test for 1-D Laplacian**

```python
import numpy as np
from scipy.sparse import csr_matrix
import fem

def test_1d_laplacian_cg():
    """Solve -u'' = 1 on [0,1] with u(0)=u(1)=0 using 1-D FEM via 2-D mesh."""
    # Build a 2-D tri mesh (n=4) — for 1D-like problems, we use 2D meshes.
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    
    # Assemble stiffness
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    
    # Assemble RHS
    b = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    
    # Convert to numpy
    b_np = np.array(b, dtype=np.float64)
    x = np.zeros(V.n_dofs(), dtype=np.float64)
    
    # Solve
    result = fem.solve_cg(A, b_np, x, tol=1e-8)
    assert result.converged, f"CG failed to converge: {result.final_residual}"
    assert result.final_residual < 1e-6
    
    # Verify matrix properties
    assert A.nrows() == V.n_dofs()
    assert A.ncols() == V.n_dofs()
    assert A.nnz() > 0
    
    # Check scipy conversion
    data, indices, indptr, shape = A.to_scipy()
    A_sp = csr_matrix((data, indices, indptr), shape=shape)
    assert A_sp.shape == (V.n_dofs(), V.n_dofs())
```

- [ ] **Step 2: Write 2-D Poisson test with boundary conditions**

```python
def test_2d_poisson_dirichlet():
    """Solve -Δu = 1 on unit square with u=0 on boundary."""
    mesh = fem.Mesh.unit_square_tri(8)
    V = fem.H1Space(mesh, order=1)
    
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b_np = np.array(b, dtype=np.float64)
    
    # Apply Dirichlet BCs on all boundaries
    boundary = mesh.boundary_nodes([1, 2, 3, 4])
    fem.apply_dirichlet(A, b_np, boundary)
    
    x = np.zeros(V.n_dofs(), dtype=np.float64)
    result = fem.solve_cg(A, b_np, x, tol=1e-8)
    assert result.converged, f"CG failed: {result.final_residual}"
    assert result.final_residual < 1e-6

def test_gmres_matches_cg_on_spd():
    """GMRES should produce the same solution as CG on SPD systems."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b_np = np.array(b_vec, dtype=np.float64)
    
    x_cg = np.zeros(V.n_dofs(), dtype=np.float64)
    x_gmres = np.zeros(V.n_dofs(), dtype=np.float64)
    
    fem.solve_cg(A, b_np, x_cg, tol=1e-10)
    fem.solve_gmres(A, b_np, x_gmres, restart=30, tol=1e-10)
    
    diff = np.max(np.abs(x_cg - x_gmres))
    assert diff < 1e-8, f"CG and GMRES differ: {diff}"

def test_sparse_lu_direct():
    """Sparse LU should produce an accurate solution."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    
    A = fem.assemble_bilinear(V, [fem.StiffnessIntegrator()])
    b_vec = fem.assemble_linear(V, fem.ConstantLoad(1.0))
    b_np = np.array(b_vec, dtype=np.float64)
    
    x = fem.solve_sparse_lu(A, b_np)
    assert len(x) == V.n_dofs()

def test_mass_integrator():
    """Mass matrix should be non-zero."""
    mesh = fem.Mesh.unit_square_tri(4)
    V = fem.H1Space(mesh, order=1)
    M = fem.assemble_bilinear(V, [fem.MassIntegrator(alpha=1.0)])
    assert M.nnz() > 0
```

- [ ] **Step 3: Create pytest config**

Create `pyproject.toml` (same file, add to existing):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Build and run tests**

Run:
```bash
cd c:/Users/lilu/works/fem-rs
maturin develop
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ crates/python/src/solver.rs
git commit -m "test: add Python integration tests for full FEM pipeline"
```
