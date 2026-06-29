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
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

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
