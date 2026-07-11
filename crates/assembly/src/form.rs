//! High-level Form abstractions (MFEM-style BilinearForm / LinearForm).
//!
//! These wrapper types hold a space + integrator list and provide lazy
//! assembly, matrix/vector caching, and `Mult` / `MultTranspose` operations.
//!
//! # Example
//! ```rust,ignore
//! use fem_assembly::form::{BilinearForm, LinearForm};
//! use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
//!
//! let stiffness = BilinearForm::new(&space)
//!     .add_integrator(DiffusionIntegrator { kappa: 1.0 })
//!     .assemble(2);
//! let rhs = LinearForm::new(&space)
//!     .add_integrator(DomainSourceIntegrator::new(f))
//!     .assemble(3);
//! ```

use fem_linalg::CsrMatrix;
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::integrator::{BilinearIntegrator, LinearIntegrator};
use crate::vector_assembler::VectorAssembler;
use crate::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator};

// ─── BilinearForm (scalar-valued) ─────────────────────────────────────────────

pub struct BilinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn BilinearIntegrator>>,
    cached: Option<CsrMatrix<f64>>,
}

impl<S: FESpace> BilinearForm<S> {
    pub fn new(space: S) -> Self {
        BilinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl BilinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &CsrMatrix<f64> {
        let refs: Vec<&dyn BilinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let mat = Assembler::assemble_bilinear(&self.space, &refs, quad_order);
        self.cached = Some(mat);
        self.cached.as_ref().unwrap()
    }
    pub fn mat(&self) -> Option<&CsrMatrix<f64>> { self.cached.as_ref() }
    pub fn space(&self) -> &S { &self.space }

    /// Compute `y += A * x` (MFEM's `BilinearForm::AddMult`).
    ///
    /// # Panics
    /// Panics if `assemble()` has not been called first.
    pub fn add_mult(&self, x: &[f64], y: &mut [f64]) {
        let a = self.cached.as_ref().expect("assemble() must be called first");
        a.spmv_add(1.0, x, 1.0, y);
    }

    /// Eliminate essential (Dirichlet) BCs symmetrically.
    ///
    /// Modifies the cached matrix `A` and `rhs` in-place so that
    /// the solution satisfies `u[d] = bc_vals[i]` for each `d = ess_dofs[i]`.
    pub fn eliminate_essential_bc(&mut self, ess_dofs: &[usize], bc_vals: &[f64], rhs: &mut [f64]) {
        let a = self.cached.as_mut().expect("assemble() must be called first");
        let n = a.nrows;
        let ess: std::collections::HashSet<usize> = ess_dofs.iter().copied().collect();
        for (pos, &d) in ess_dofs.iter().enumerate() {
            let val = bc_vals[pos];
            // Row contributions: A[d,j] * val subtracted from rhs[j]
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                let j = a.col_idx[r] as usize;
                if !ess.contains(&j) {
                    rhs[j] -= a.values[r] * val;
                }
            }
            // Column contributions: A[i,d] * val subtracted from rhs[i]
            // Scan all rows for column d
            for i in 0..n {
                if ess.contains(&i) { continue; }
                for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                    if a.col_idx[r] as usize == d {
                        if i != d {
                            rhs[i] -= a.values[r] * val;
                        }
                        // Zero this entry (row i, column d)
                        a.values[r] = 0.0;
                        break;
                    }
                }
            }
            // Zero row d and set diagonal
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                a.values[r] = 0.0;
            }
            // Find diagonal entry in row d and set to 1
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                if a.col_idx[r] as usize == d {
                    a.values[r] = 1.0;
                    break;
                }
            }
            rhs[d] = val;
        }
    }

    /// Fast diagonal-only BC elimination for SPD systems.
    pub fn eliminate_essential_bc_from_diag(&mut self, ess_dofs: &[usize], bc_vals: &[f64], rhs: &mut [f64]) {
        let a = self.cached.as_mut().expect("assemble() must be called first");
        for (pos, &d) in ess_dofs.iter().enumerate() {
            for r in a.row_ptr[d]..a.row_ptr[d + 1] {
                if a.col_idx[r] as usize == d {
                    a.values[r] = 1.0;
                    break;
                }
            }
            rhs[d] = bc_vals[pos];
        }
    }
}

// ─── LinearForm (scalar-valued) ────────────────────────────────────────────────

pub struct LinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn LinearIntegrator>>,
    cached: Option<Vec<f64>>,
}

impl<S: FESpace> LinearForm<S> {
    pub fn new(space: S) -> Self {
        LinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl LinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &[f64] {
        let refs: Vec<&dyn LinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let rhs = Assembler::assemble_linear(&self.space, &refs, quad_order);
        self.cached = Some(rhs);
        self.cached.as_ref().unwrap()
    }
    pub fn vec(&self) -> Option<&[f64]> { self.cached.as_deref() }
}

// ─── VectorBilinearForm (H(curl) / H(div) valued) ──────────────────────────────

pub struct VectorBilinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn VectorBilinearIntegrator>>,
    cached: Option<CsrMatrix<f64>>,
}

impl<S: FESpace> VectorBilinearForm<S> {
    pub fn new(space: S) -> Self {
        VectorBilinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl VectorBilinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &CsrMatrix<f64> {
        let refs: Vec<&dyn VectorBilinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let mat = VectorAssembler::assemble_bilinear(&self.space, &refs, quad_order);
        self.cached = Some(mat);
        self.cached.as_ref().unwrap()
    }
}

// ─── VectorLinearForm ──────────────────────────────────────────────────────────

pub struct VectorLinearForm<S: FESpace> {
    space: S,
    integrators: Vec<Box<dyn VectorLinearIntegrator>>,
    cached: Option<Vec<f64>>,
}

impl<S: FESpace> VectorLinearForm<S> {
    pub fn new(space: S) -> Self {
        VectorLinearForm { space, integrators: Vec::new(), cached: None }
    }
    pub fn add_integrator(mut self, integ: impl VectorLinearIntegrator + 'static) -> Self {
        self.integrators.push(Box::new(integ));
        self
    }
    pub fn assemble(&mut self, quad_order: u8) -> &[f64] {
        let refs: Vec<&dyn VectorLinearIntegrator> = self.integrators.iter().map(|b| b.as_ref()).collect();
        let rhs = VectorAssembler::assemble_linear(&self.space, &refs, quad_order);
        self.cached = Some(rhs);
        self.cached.as_ref().unwrap()
    }
}
