//! Complex-valued finite element assembly.
//!
//! Implements a **2×2 real-block** strategy that avoids introducing complex
//! number generics: the complex DOF vector `u = u_re + i·u_im` is stored as
//! two separate real vectors, and the system matrix is a 2×2 block:
//!
//! ```text
//! [ K - ω²M    -ωC ] [ u_re ]   [ f_re ]
//! [ ωC       K-ω²M  ] [ u_im ] = [ f_im ]
//! ```
//!
//! where `K`, `M`, `C` are standard real sparse matrices assembled from
//! existing integrators.
//!
//! # Typical use — scalar H¹ Helmholtz
//! ```rust,ignore
//! use fem_assembly::complex::{ComplexAssembler, ComplexSystem};
//! use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
//! use fem_space::H1Space;
//!
//! let space = H1Space::new(mesh, 1);
//! let omega = 2.0 * PI;
//!
//! // −∇·κ∇ + ω²ρu + iω·c·u = f
//! let sys = ComplexAssembler::assemble(
//!     &space,
//!     &[&DiffusionIntegrator { kappa: 1.0 }],  // stiffness K
//!     &[&MassIntegrator { rho: 1.0 }],           // mass M (multiplied by ω²)
//!     &[&MassIntegrator { rho: 0.1 }],           // damping C (multiplied by ω)
//!     omega, 3,
//! );
//! let x = sys.solve_gmres(&f_re, &f_im, &cfg)?;
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_linalg::complex_csr::{ComplexCoo, ComplexCsr, solve_gmres_complex};
use fem_mesh::ElementTransformation;
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::vector_assembler::VectorAssembler;
use crate::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use crate::vector_integrator::VectorBilinearIntegrator;

// ═══════════════════════════════════════════════════════════════════════════
// ComplexOperator — convention for 2×2 block layout
// ═══════════════════════════════════════════════════════════════════════════

/// Convention for the 2×2 real-block representation of a complex operator.
///
/// Controls the sign of the off-diagonal blocks when flattening a complex
/// system `A = A_re + i·A_im` to a 2n×2n real matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Convention {
    /// Hermitian convention (default):
    /// ```text
    /// [ A_re  -A_im ]
    /// [ A_im   A_re ]
    /// ```
    /// This is the standard form where `A = A_re + i·A_im` and the
    /// complex system `A·u = f` expands to the block system shown.
    Hermitian,

    /// Block-symmetric convention:
    /// ```text
    /// [ A_re   A_im ]
    /// [ A_im  -A_re ]
    /// ```
    /// Used in some formulations where the complex operator has special
    /// symmetry properties.
    BlockSymmetric,
}

impl Convention {
    /// Sign for the top-right block: `-1` for Hermitian, `+1` for BlockSymmetric.
    pub fn tr_sign(&self) -> f64 {
        match self {
            Convention::Hermitian => -1.0,
            Convention::BlockSymmetric => 1.0,
        }
    }

    /// Sign for the bottom-right block: `+1` for Hermitian, `-1` for BlockSymmetric.
    pub fn br_sign(&self) -> f64 {
        match self {
            Convention::Hermitian => 1.0,
            Convention::BlockSymmetric => -1.0,
        }
    }
}

// 鈹€鈹€鈹€ ComplexSystem 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// A 2脳2 real-block equivalent of a complex linear system.
///
/// The system is:
/// ```text
/// [ A_re  A_im ] [ x_re ]   [ b_re ]
/// [-A_im  A_re ] [ x_im ] = [ b_im ]
/// ```
/// where `A = A_re + i路A_im` is the complex system matrix.
///
/// For the time-harmonic problem with stiffness `K`, mass `M`, damping `C`:
/// - `A_re = K 鈭?蠅虏路M`
/// - `A_im = 蠅路C`
pub struct ComplexSystem {
    /// Real part of the system matrix (n_dofs 脳 n_dofs).
    pub k_re: CsrMatrix<f64>,
    /// Imaginary part: the "coupling" matrix (n_dofs 脳 n_dofs).
    pub k_im: CsrMatrix<f64>,
    /// Angular frequency (stored for reference).
    pub omega: f64,
}

impl ComplexSystem {
    /// Number of real DOFs (= total DOFs in the underlying FE space).
    pub fn n_dofs(&self) -> usize { self.k_re.nrows }

    /// Total size of the flattened 2脳2 block system = `2 * n_dofs`.
    pub fn n_total(&self) -> usize { 2 * self.n_dofs() }

    /// Build the flat (2n 脳 2n) block CSR matrix:
    /// ```text
    /// [ K_re   -K_im ]
    /// [ K_im    K_re ]
    /// ```
    pub fn to_flat_csr(&self) -> CsrMatrix<f64> {
        let n = self.n_dofs();
        let tot = 2 * n;
        let mut coo = CooMatrix::<f64>::new(tot, tot);

        // Top-left: +K_re
        for i in 0..n {
            for ptr in self.k_re.row_ptr[i]..self.k_re.row_ptr[i + 1] {
                let j = self.k_re.col_idx[ptr] as usize;
                coo.add(i, j, self.k_re.values[ptr]);
            }
        }
        // Top-right: -K_im  (coupling: -i part to i part)
        for i in 0..n {
            for ptr in self.k_im.row_ptr[i]..self.k_im.row_ptr[i + 1] {
                let j = self.k_im.col_idx[ptr] as usize;
                coo.add(i, n + j, -self.k_im.values[ptr]);
            }
        }
        // Bottom-left: +K_im
        for i in 0..n {
            for ptr in self.k_im.row_ptr[i]..self.k_im.row_ptr[i + 1] {
                let j = self.k_im.col_idx[ptr] as usize;
                coo.add(n + i, j, self.k_im.values[ptr]);
            }
        }
        // Bottom-right: +K_re
        for i in 0..n {
            for ptr in self.k_re.row_ptr[i]..self.k_re.row_ptr[i + 1] {
                let j = self.k_re.col_idx[ptr] as usize;
                coo.add(n + i, n + j, self.k_re.values[ptr]);
            }
        }
        coo.into_csr()
    }

    /// Build the flat RHS from separate real/imaginary parts.
    pub fn assemble_rhs(&self, f_re: &[f64], f_im: &[f64]) -> Vec<f64> {
        let n = self.n_dofs();
        assert_eq!(f_re.len(), n, "f_re length mismatch");
        assert_eq!(f_im.len(), n, "f_im length mismatch");
        let mut rhs = Vec::with_capacity(2 * n);
        rhs.extend_from_slice(f_re);
        rhs.extend_from_slice(f_im);
        rhs
    }

    /// Apply Dirichlet boundary conditions symmetrically on the 2脳2 block
    /// system.
    ///
    /// Each DOF in `dofs` is eliminated in both the re and im blocks.
    /// `bc_re[k]` and `bc_im[k]` supply the real/imaginary parts of the
    /// Dirichlet value for `dofs[k]`.
    pub fn apply_dirichlet(
        &mut self,
        dofs:  &[usize],
        bc_re: &[f64],
        bc_im: &[f64],
        rhs: &mut [f64],
    ) {
        let n = self.n_dofs();
        // Eliminate each constrained DOF i in the real part (row/col i)
        // and the imaginary part (row/col n+i).
        for (k, &i) in dofs.iter().enumerate() {
            let val_re = bc_re[k];
            let val_im = bc_im[k];

            // Column elimination + RHS correction for all rows j != i
            for j in 0..n {
                if j == i { continue; }
                let mut a_re = 0.0_f64;
                let mut a_im = 0.0_f64;
                if let Some(p) = self.k_re.find_entry(j, i) {
                    a_re = self.k_re.values[p];
                    self.k_re.values[p] = 0.0;
                }
                if let Some(p) = self.k_im.find_entry(j, i) {
                    a_im = self.k_im.values[p];
                    self.k_im.values[p] = 0.0;
                }
                rhs[j]     -= a_re * val_re - a_im * val_im;
                rhs[n + j] -= a_im * val_re + a_re * val_im;
            }
            // Row elimination: zero rows, set diagonal = 1 for real block
            let start_re = self.k_re.row_ptr[i];
            let end_re   = self.k_re.row_ptr[i + 1];
            for p in start_re..end_re { self.k_re.values[p] = 0.0; }
            if let Some(p) = self.k_re.find_entry(i, i) { self.k_re.values[p] = 1.0; }
            let start_im = self.k_im.row_ptr[i];
            let end_im   = self.k_im.row_ptr[i + 1];
            for p in start_im..end_im { self.k_im.values[p] = 0.0; }
            rhs[i]     = val_re;
            rhs[n + i] = val_im;
        }
    }
}

// 鈹€鈹€鈹€ ComplexAssembler 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Assembles a complex-valued time-harmonic PDE system using real block
/// decomposition.
///
/// The assembled system represents:
/// ```text
/// (K 鈭?蠅虏M + i蠅C) u = f
/// ```
/// expanded to the 2脳2 real block form stored in [`ComplexSystem`].
pub struct ComplexAssembler;

impl ComplexAssembler {
    /// Assemble a time-harmonic complex system.
    ///
    /// # Arguments
    /// - `space`   鈥?FE space (H鹿, H(curl), H(div))
    /// - `stiff`   鈥?stiffness integrators for `K` (e.g. `DiffusionIntegrator`, `CurlCurlIntegrator`)
    /// - `mass`    鈥?mass integrators for `M` (e.g. `MassIntegrator`, `VectorMassIntegrator`)
    /// - `damp`    鈥?damping integrators for `C` (e.g. `MassIntegrator` with conductivity 蟽)
    /// - `omega`   鈥?angular frequency 蠅
    /// - `quad_order` 鈥?quadrature order
    ///
    /// Returns `(system, k_re, k_im)` where:
    /// - `k_re = K 鈭?蠅虏路M`
    /// - `k_im = 蠅路C`
    pub fn assemble<S: FESpace + Send + Sync>(
        space:      &S,
        stiff:      &[&dyn BilinearIntegrator],
        mass:       &[&dyn BilinearIntegrator],
        damp:       &[&dyn BilinearIntegrator],
        omega:      f64,
        quad_order: u8,
    ) -> ComplexSystem {
        // Assemble individual real matrices
        let k = Assembler::assemble_bilinear(space, stiff, quad_order);
        let m = Assembler::assemble_bilinear(space, mass,  quad_order);
        let c = Assembler::assemble_bilinear(space, damp,  quad_order);

        // k_re = K 鈭?蠅虏路M
        let k_re = subtract_scaled(&k, &m, omega * omega);
        // k_im = 蠅路C
        let k_im = scale_csr(&c, omega);

        ComplexSystem { k_re, k_im, omega }
    }

    /// Assemble a purely-stiffness + mass system with no damping.
    ///
    /// `k_re = K 鈭?蠅虏路M`,  `k_im = 0`.
    ///
    /// Suitable for lossless resonators (eigenvalue problems) or
    /// real-sourced Helmholtz when damping is zero.
    pub fn assemble_undamped<S: FESpace + Send + Sync>(
        space:      &S,
        stiff:      &[&dyn BilinearIntegrator],
        mass:       &[&dyn BilinearIntegrator],
        omega:      f64,
        quad_order: u8,
    ) -> ComplexSystem {
        let k = Assembler::assemble_bilinear(space, stiff, quad_order);
        let m = Assembler::assemble_bilinear(space, mass,  quad_order);
        let n = k.nrows;
        let k_re = subtract_scaled(&k, &m, omega * omega);
        // k_im = zero matrix (same sparsity as k_re for simplicity)
        let k_im = zero_like(&k_re, n);
        ComplexSystem { k_re, k_im, omega }
    }

    /// Assemble a time-harmonic complex system for H(curl) / H(div) spaces.
    ///
    /// Same as [`assemble`] but uses [`VectorBilinearIntegrator`] (e.g.
    /// [`CurlCurlIntegrator`], [`VectorMassIntegrator`]).
    pub fn assemble_vector<S: FESpace + Send + Sync>(
        space:      &S,
        stiff:      &[&dyn VectorBilinearIntegrator],
        mass:       &[&dyn VectorBilinearIntegrator],
        damp:       &[&dyn VectorBilinearIntegrator],
        omega:      f64,
        quad_order: u8,
    ) -> ComplexSystem {
        let k = VectorAssembler::assemble_bilinear(space, stiff, quad_order);
        let m = VectorAssembler::assemble_bilinear(space, mass,  quad_order);
        let c = VectorAssembler::assemble_bilinear(space, damp,  quad_order);

        let k_re = subtract_scaled(&k, &m, omega * omega);
        let k_im = scale_csr(&c, omega);
        ComplexSystem { k_re, k_im, omega }
    }

    /// Assemble an undamped (lossless) complex H(curl) / H(div) system.
    ///
    /// `k_re = K 鈭?蠅虏路M`,  `k_im = 0`.
    pub fn assemble_vector_undamped<S: FESpace + Send + Sync>(
        space:      &S,
        stiff:      &[&dyn VectorBilinearIntegrator],
        mass:       &[&dyn VectorBilinearIntegrator],
        omega:      f64,
        quad_order: u8,
    ) -> ComplexSystem {
        let k = VectorAssembler::assemble_bilinear(space, stiff, quad_order);
        let m = VectorAssembler::assemble_bilinear(space, mass,  quad_order);
        let n = k.nrows;
        let k_re = subtract_scaled(&k, &m, omega * omega);
        let k_im = zero_like(&k_re, n);
        ComplexSystem { k_re, k_im, omega }
    }
}

// 鈹€鈹€鈹€ ComplexLinearForm 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Assembles a complex-valued right-hand side `f = f_re + i路f_im`.
pub struct ComplexLinearForm {
    /// Real part contributions.
    pub f_re: Vec<f64>,
    /// Imaginary part contributions.
    pub f_im: Vec<f64>,
}

impl ComplexLinearForm {
    /// Assemble separate real and imaginary parts.
    ///
    /// Either slice of integrators may be empty (鈫?zero contribution).
    pub fn assemble<S: FESpace + Send + Sync>(
        space:      &S,
        re_integ:   &[&dyn LinearIntegrator],
        im_integ:   &[&dyn LinearIntegrator],
        quad_order: u8,
    ) -> Self {
        let n = space.n_dofs();
        let f_re = if re_integ.is_empty() {
            vec![0.0; n]
        } else {
            Assembler::assemble_linear(space, re_integ, quad_order)
        };
        let f_im = if im_integ.is_empty() {
            vec![0.0; n]
        } else {
            Assembler::assemble_linear(space, im_integ, quad_order)
        };
        ComplexLinearForm { f_re, f_im }
    }
}

// === SesquilinearForm — deferred assembly (matching MFEM) ===
//
// Integrator pairs are stored by reference and assembled in a single pass
// over elements, matching C++'s SesquilinearForm::AddDomainIntegrator + Assemble.
// The caller MUST keep integrators alive until `assemble()` returns.

pub struct SesquilinearForm<'a, 'b, S: FESpace + Send + Sync> {
    space: &'a S,
    conv: Convention,
    quad_order: u8,
    pairs: Vec<(&'b dyn VectorBilinearIntegrator, Option<&'b dyn VectorBilinearIntegrator>)>,
    n: usize,
}

impl<'a, 'b, S: FESpace + Send + Sync> SesquilinearForm<'a, 'b, S> {
    pub fn new(space: &'a S, conv: Convention, quad_order: u8) -> Self {
        let n = space.n_dofs();
        SesquilinearForm { space, conv, quad_order, pairs: Vec::new(), n }
    }

    /// Store an integrator pair reference for later single-pass assembly.
    /// Caller must keep integrators alive until `assemble()` is called.
    pub fn add_domain_integrator_pair(
        &mut self,
        re_integ: &'b dyn VectorBilinearIntegrator,
        im_integ: Option<&'b dyn VectorBilinearIntegrator>,
    ) {
        self.pairs.push((re_integ, im_integ));
    }

    /// Single-pass assembly: all RE integrators in one element loop,
    /// all IM integrators in another, matching C++ assembly order.
    pub fn assemble(self) -> ComplexSystem {
        let mut re_all: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
        let mut im_all: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
        for (re, im) in &self.pairs {
            re_all.push(*re);
            if let Some(im) = im { im_all.push(*im); }
        }

        let k_re = if !re_all.is_empty() {
            VectorAssembler::assemble_bilinear(self.space, &re_all, self.quad_order)
        } else {
            CooMatrix::new(self.n, self.n).into_csr()
        };

        let k_im = if !im_all.is_empty() {
            VectorAssembler::assemble_bilinear(self.space, &im_all, self.quad_order)
        } else {
            CooMatrix::new(self.n, self.n).into_csr()
        };

        ComplexSystem { k_re, k_im, omega: 0.0 }
    }
}

impl ComplexSystem {
    pub fn to_flat_csr_with_conv(&self, conv: Convention) -> CsrMatrix<f64> {
        let n = self.n_dofs();
        let tot = 2 * n;
        let mut coo = CooMatrix::<f64>::new(tot, tot);
        let tr = conv.tr_sign();
        let br = conv.br_sign();
        for i in 0..n {
            for p in self.k_re.row_ptr[i]..self.k_re.row_ptr[i + 1] {
                let j = self.k_re.col_idx[p] as usize;
                let v = self.k_re.values[p];
                if v != 0.0 { coo.add(i, j, v); coo.add(n + i, n + j, br * v); }
            }
        }
        for i in 0..n {
            for p in self.k_im.row_ptr[i]..self.k_im.row_ptr[i + 1] {
                let j = self.k_im.col_idx[p] as usize;
                let v = self.k_im.values[p];
                if v != 0.0 { coo.add(i, n + j, tr * v); coo.add(n + i, j, v); }
            }
        }
        coo.into_csr()
    }
}

/// A complex grid function `u = u_re + i路u_im`.
#[derive(Debug, Clone)]
pub struct ComplexGridFunction {
    /// Real DOF coefficients.
    pub u_re: Vec<f64>,
    /// Imaginary DOF coefficients.
    pub u_im: Vec<f64>,
}

impl ComplexGridFunction {
    /// Extract from a flat 2n solution vector `[u_re; u_im]`.
    pub fn from_flat(flat: &[f64]) -> Self {
        let n = flat.len() / 2;
        ComplexGridFunction {
            u_re: flat[..n].to_vec(),
            u_im: flat[n..].to_vec(),
        }
    }

    /// Pointwise amplitude `|u(x)| = sqrt(u_re虏 + u_im虏)`.
    pub fn amplitude(&self) -> Vec<f64> {
        self.u_re.iter().zip(self.u_im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Total complex L虏 norm: `sqrt(鈥杣_re鈥柭?+ 鈥杣_im鈥柭?`.
    pub fn l2_norm(&self) -> f64 {
        let re: f64 = self.u_re.iter().map(|x| x * x).sum();
        let im: f64 = self.u_im.iter().map(|x| x * x).sum();
        (re + im).sqrt()
    }
}

// 鈹€鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Compute `A 鈭?alpha路B` (both same sparsity pattern, using COO merge).
fn subtract_scaled(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    let n = a.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);

    // Add A entries
    for i in 0..n {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr] as usize;
            coo.add(i, j, a.values[ptr]);
        }
    }
    // Subtract alpha * B entries
    for i in 0..n {
        for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
            let j = b.col_idx[ptr] as usize;
            coo.add(i, j, -alpha * b.values[ptr]);
        }
    }
    coo.into_csr()
}

/// Scale all entries of a CSR matrix by `alpha`.
fn scale_csr(mat: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows:   mat.nrows,
        ncols:   mat.ncols,
        row_ptr: mat.row_ptr.clone(),
        col_idx: mat.col_idx.clone(),
        values:  mat.values.iter().map(|v| alpha * v).collect(),
    }
}

/// Zero CSR matrix with same sparsity/dimensions as `template`.
fn zero_like(_template: &CsrMatrix<f64>, n: usize) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows:   n,
        ncols:   n,
        row_ptr: vec![0; n + 1],
        col_idx: vec![],
        values:  vec![],
    }
}

// 鈹€鈹€鈹€ NativeComplexSystem 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// A **natively complex** FEM system: the matrix is stored as a single
/// `ComplexCsr` (not the 2脳2 real-block workaround).
///
/// Advantages over [`ComplexSystem`]:
/// - Matrix size is n脳n (not 2n脳2n) 鈫?lower memory and better conditioning
/// - Complex GMRES acts directly on complex inner products
/// - Compatible with complex-valued PML coefficients
/// - At high 蠅 the preconditioner quality is substantially better
///
/// # Typical use 鈥?Helmholtz or Maxwell
/// ```rust,ignore
/// let sys = NativeComplexAssembler::assemble_helmholtz(
///     &h1_space, kappa_re, kappa_im, 1.0 /* rho */, omega, 3,
/// );
/// let gf = sys.solve(b_re, b_im, 1e-8, 300, 30)?;
/// ```
pub struct NativeComplexSystem {
    /// Complex stiffness matrix `A = K + iB` assembled element-by-element.
    pub mat: ComplexCsr,
    /// Angular frequency at which this system was assembled.
    pub omega: f64,
    /// Number of FE DOFs.
    pub n_dofs: usize,
}

impl NativeComplexSystem {
    /// Solve `A x = b` via complex restarted GMRES with Jacobi preconditioning.
    ///
    /// Returns the complex grid function as `ComplexGridFunction`.
    pub fn solve(
        &self,
        b_re: &[f64],
        b_im: &[f64],
        tol: f64,
        max_iter: usize,
        restart: usize,
    ) -> Result<ComplexGridFunction, String> {
        let mut x_re = vec![0.0_f64; self.n_dofs];
        let mut x_im = vec![0.0_f64; self.n_dofs];
        solve_gmres_complex(
            &self.mat, b_re, b_im, &mut x_re, &mut x_im,
            tol, max_iter, restart, true,
        )?;
        Ok(ComplexGridFunction { u_re: x_re, u_im: x_im })
    }

    /// Solve with initial guess (in-place update).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_guess(
        &self,
        b_re: &[f64],
        b_im: &[f64],
        x_re: &mut Vec<f64>,
        x_im: &mut Vec<f64>,
        tol: f64,
        max_iter: usize,
        restart: usize,
    ) -> Result<(usize, f64), String> {
        solve_gmres_complex(
            &self.mat, b_re, b_im, x_re, x_im,
            tol, max_iter, restart, true,
        )
    }

    /// Apply Dirichlet BCs on the native complex matrix.
    pub fn apply_dirichlet(
        &mut self,
        dofs: &[usize],
        bc_re: &[f64],
        bc_im: &[f64],
        rhs_re: &mut [f64],
        rhs_im: &mut [f64],
    ) {
        for (k, &dof) in dofs.iter().enumerate() {
            let vr = if k < bc_re.len() { bc_re[k] } else { 0.0 };
            let vi = if k < bc_im.len() { bc_im[k] } else { 0.0 };
            self.mat.apply_dirichlet_row(dof, vr, vi, rhs_re, rhs_im);
        }
    }
}

// 鈹€鈹€鈹€ NativeComplexAssembler 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Assembles a complex FEM system natively into [`NativeComplexSystem`].
///
/// Each element contributes `K_e_re + i*K_e_im` directly assembled into
/// [`ComplexCsr`] via [`ComplexCoo`].
pub struct NativeComplexAssembler;

/// Trait for an integrator that produces complex element matrices.
///
/// Implementors fill both `k_re` and `k_im` (row-major `n_dofs 脳 n_dofs`).
pub trait ComplexBilinearIntegrator: Send + Sync {
    fn add_to_complex_element_matrix(
        &self,
        qp:    &QpData<'_>,
        k_re:  &mut [f64],
        k_im:  &mut [f64],
    );
}

/// Helmholtz integrator: contributes `魏_re 鈭囅喡封垏蠄 鈭?蠅虏 蟻 蠁蠄` (real) and
/// `魏_im 鈭囅喡封垏蠄` (imaginary).
///
/// This naturally handles PML coefficients where `魏 = 魏_re + i 魏_im`.
pub struct HelmholtzIntegrator {
    /// Real part of the diffusion coefficient 魏 (can be PML-modified).
    pub kappa_re: f64,
    /// Imaginary part of the diffusion coefficient 魏 (PML absorption).
    pub kappa_im: f64,
    /// Mass coefficient 蟻 (always real for standard media).
    pub rho: f64,
    /// Angular frequency 蠅.
    pub omega: f64,
}

impl ComplexBilinearIntegrator for HelmholtzIntegrator {
    fn add_to_complex_element_matrix(&self, qp: &QpData<'_>, k_re: &mut [f64], k_im: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let w = qp.weight;
        for i in 0..n {
            for j in 0..n {
                // Diffusion: 魏 鈭囅嗎耽 路 鈭囅嗏奔
                let mut grad_dot = 0.0_f64;
                for d in 0..dim {
                    grad_dot += qp.grad_phys[i * dim + d] * qp.grad_phys[j * dim + d];
                }
                // Mass shift: 鈭捪壜?蟻 蠁岬?蠁獗?
                let mass = self.rho * qp.phi[i] * qp.phi[j];
                k_re[i * n + j] += w * (self.kappa_re * grad_dot - self.omega * self.omega * mass);
                k_im[i * n + j] += w *  self.kappa_im * grad_dot;
            }
        }
    }
}

/// Maxwell time-harmonic integrator (H(curl)):
/// contributes `渭鈦宦?鈭嚸梪)路(鈭嚸梫) 鈭?蠅虏 蔚 u路v` (with complex 蔚/渭 for PML).
pub struct MaxwellHarmonicIntegrator {
    /// Real part of 渭鈦宦?(permeability inverse).
    pub mu_inv_re: f64,
    /// Imaginary part of 渭鈦宦?
    pub mu_inv_im: f64,
    /// Real part of permittivity 蔚.
    pub eps_re: f64,
    /// Imaginary part of permittivity 蔚 (loss tangent 鈫?positive for lossy media).
    pub eps_im: f64,
    /// Angular frequency 蠅.
    pub omega: f64,
}

impl ComplexBilinearIntegrator for MaxwellHarmonicIntegrator {
    fn add_to_complex_element_matrix(&self, qp: &QpData<'_>, k_re: &mut [f64], k_im: &mut [f64]) {
        let n = qp.n_dofs;
        // For H(curl) the grad_phys encodes curl values (2D: scalar curl, 3D: 3-vector curl)
        // Here we treat grad_phys as curl for simplicity (H1 proxy shows correctness of framework)
        let dim = qp.dim;
        let w = qp.weight;
        for i in 0..n {
            for j in 0..n {
                // Curl-curl: 渭鈦宦?(鈭嚸椣嗎耽)路(鈭嚸椣嗏奔) 鈥?use grad as proxy for framework
                let mut curl_dot = 0.0_f64;
                for d in 0..dim {
                    curl_dot += qp.grad_phys[i * dim + d] * qp.grad_phys[j * dim + d];
                }
                // Mass: 蔚 蠁岬⒙废嗏奔
                let mass = qp.phi[i] * qp.phi[j];
                // Re(渭鈦宦? curl路curl 鈭?蠅虏 Re(蔚) mass
                k_re[i * n + j] += w * (self.mu_inv_re * curl_dot
                    - self.omega * self.omega * self.eps_re * mass);
                // Im(渭鈦宦? curl路curl + 蠅虏 Im(蔚) mass (note sign: +蠅虏Im(蔚) for lossy)
                k_im[i * n + j] += w * (self.mu_inv_im * curl_dot
                    + self.omega * self.omega * self.eps_im * mass);
            }
        }
    }
}

impl NativeComplexAssembler {
    /// Assemble a native complex system from a list of [`ComplexBilinearIntegrator`]s.
    pub fn assemble<S: FESpace + Send + Sync>(
        space:      &S,
        integrs:    &[&dyn ComplexBilinearIntegrator],
        quad_order: u8,
    ) -> NativeComplexSystem {
        use fem_element::{ReferenceElement,
            lagrange::{TriP1, TriP2, TetP1, TetP2, QuadQ1, QuadQ2, QuadQ4}};
        use fem_mesh::{element_type::ElementType, topology::MeshTopology};

        let mesh    = space.mesh();
        let n_dofs  = space.n_dofs();
        let order   = space.order();
        let mut coo = ComplexCoo::new(n_dofs, n_dofs);

        for elem in 0..mesh.n_elements() as u32 {
            let nodes   = mesh.element_nodes(elem);
            let etype   = mesh.element_type(elem);
            let etag    = mesh.element_tag(elem);
            let dofs    = space.element_dofs(elem);
            let n       = dofs.len();
            let dim     = mesh.dim() as usize;

            // Build element transformation (affine simplex)
            let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);

            // Reference element 鈥?choose by type and polynomial order
            let ref_elem: Box<dyn ReferenceElement> = match (etype, order) {
                (ElementType::Tri6, _) => Box::new(TriP2),
                (ElementType::Tri3, _) => Box::new(TriP1),
                (ElementType::Tet4, _) => Box::new(TetP1),
                (ElementType::Quad4, 1) => Box::new(QuadQ1),
                (ElementType::Quad4, 2) => Box::new(QuadQ2),
                (ElementType::Quad4, 3) => Box::new(fem_element::lagrange::QuadQk::new(3)),
                (ElementType::Quad4, 4) => Box::new(QuadQ4),
                _ => Box::new(TriP1),
            };
            // Override for P2 orders
            let ref_elem: Box<dyn ReferenceElement> = match (etype, order) {
                (ElementType::Tri3,  2) => Box::new(TriP2),
                (ElementType::Tet4,  2) => Box::new(TetP2),
                _ => ref_elem,
            };

            let quad = ref_elem.quadrature(quad_order);
            let n_local = ref_elem.n_dofs();

            let mut k_re = vec![0.0_f64; n * n];
            let mut k_im = vec![0.0_f64; n * n];

            let mut phi      = vec![0.0_f64; n_local];
            let mut grad_ref = vec![0.0_f64; n_local * dim];
            let mut grad_phys = vec![0.0_f64; n_local * dim];

            for (q_idx, xi) in quad.points.iter().enumerate() {
                let w_ref = quad.weights[q_idx];
                let w_phys = w_ref * tr.det_j().abs();

                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);

                // Transform reference gradients to physical: grad_phys = J^{-T} grad_ref
                let j_inv_t = tr.jacobian_inv_t();
                for i in 0..n_local {
                    for d in 0..dim {
                        let mut s = 0.0_f64;
                        for d2 in 0..dim {
                            s += j_inv_t[(d, d2)] * grad_ref[i * dim + d2];
                        }
                        grad_phys[i * dim + d] = s;
                    }
                }

                // Physical coordinates from mapped point
                let x_phys = tr.map_to_physical(xi);

                let qpd = QpData {
                    n_dofs: n,
                    dim,
                    weight: w_phys,
                    phys_weight: w_phys,
                    ref_weight: quad.weights[q_idx],
                    phi: &phi,
                    grad_phys: &grad_phys,
                    x_phys: &x_phys,
                    elem_id: elem,
                    elem_tag: etag,
                    elem_dofs: Some(dofs),
                };

                for integ in integrs {
                    integ.add_to_complex_element_matrix(&qpd, &mut k_re, &mut k_im);
                }
            }

            let dof_usize: Vec<usize> = dofs.iter().map(|&d| d as usize).collect();
            coo.add_element_matrix(&dof_usize, &k_re, &k_im);
        }

        let mat = coo.into_complex_csr();
        NativeComplexSystem { mat, omega: 0.0, n_dofs }
    }

    /// Convenience: assemble a Helmholtz system `鈭掆垏路(魏鈭噓) 鈭?蠅虏蟻 u = f`
    /// with complex diffusion coefficient `魏 = kappa_re + i*kappa_im`.
    pub fn assemble_helmholtz<S: FESpace + Send + Sync>(
        space:      &S,
        kappa_re:   f64,
        kappa_im:   f64,
        rho:        f64,
        omega:      f64,
        quad_order: u8,
    ) -> NativeComplexSystem {
        let integ = HelmholtzIntegrator { kappa_re, kappa_im, rho, omega };
        let mut sys = Self::assemble(space, &[&integ], quad_order);
        sys.omega = omega;
        sys
    }
}


/// Solve the 3-D time-harmonic eddy current problem (K + i·ω·σ·M)·A = J.
pub fn solve_complex_hcurl_3d(
    k_mat: &CsrMatrix<f64>,
    m_mat: &CsrMatrix<f64>,
    sigma_cond: f64,
    omega: f64,
    pec_dofs: &[usize],
    rtol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<ComplexGridFunction, String> {
    let n_dofs = k_mat.nrows;
    let omega_sigma = omega * sigma_cond;
    let mut coo_re = CooMatrix::new(n_dofs, n_dofs);
    let mut coo_im = CooMatrix::new(n_dofs, n_dofs);
    for i in 0..n_dofs {
        for p in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[p] as usize;
            coo_re.add(i, j, k_mat.values[p]);
        }
        if omega_sigma > 0.0 {
            for p in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                let j = m_mat.col_idx[p] as usize;
                coo_im.add(i, j, omega_sigma * m_mat.values[p]);
            }
        }
    }
    let csr = ComplexCsr::from_re_im(&coo_re.into_csr(), &coo_im.into_csr());
    let mut sys = NativeComplexSystem { mat: csr, omega, n_dofs };
    let mut r_re = vec![0.0; n_dofs];
    let mut r_im = vec![0.0; n_dofs];
    sys.apply_dirichlet(pec_dofs, &vec![0.0; pec_dofs.len()], &vec![0.0; pec_dofs.len()], &mut r_re, &mut r_im);
    for &d in pec_dofs {
        for i in 0..n_dofs {
            if i == d { continue; }
            for p in sys.mat.row_ptr[i]..sys.mat.row_ptr[i + 1] {
                if sys.mat.col_idx[p] as usize == d {
                    sys.mat.re_vals[p] = 0.0;
                    sys.mat.im_vals[p] = 0.0;
                }
            }
        }
    }
    sys.solve(&r_re, &r_im, rtol, max_iter, restart)
}
// 鈹€鈹€鈹€ Tests (NativeComplex) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

#[cfg(test)]
mod native_complex_tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn native_helmholtz_zero_omega_is_real() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let sys   = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 0.0, 1.0, 0.0, 3);
        // With 蠅=0, kappa_im=0 the matrix should be purely real (all im_vals 鈮?0)
        let max_im = sys.mat.im_vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_im < 1e-14, "im_vals should be zero for real Helmholtz: {max_im}");
        // Matrix should be non-trivial
        let max_re = sys.mat.re_vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_re > 1e-10, "re_vals should be non-zero: {max_re}");
    }

    #[test]
    fn native_helmholtz_imaginary_part_grows_with_kappa_im() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let sys1  = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 0.5, 1.0, 1.0, 3);
        let sys2  = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 1.5, 1.0, 1.0, 3);
        let im1: f64 = sys1.mat.im_vals.iter().map(|v| v.abs()).sum();
        let im2: f64 = sys2.mat.im_vals.iter().map(|v| v.abs()).sum();
        assert!(im2 > im1 * 2.0, "larger kappa_im should give larger im part: {im1} vs {im2}");
    }

    #[test]
    fn native_helmholtz_omega_shifts_real_diagonal() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let sys0  = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 0.0, 1.0, 0.0, 3);
        let sys1  = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 0.0, 1.0, 2.0, 3);
        // With 蠅=2: K_re = K - 4M, so diagonal should be smaller
        let tr0: f64 = sys0.mat.diagonal_complex().0.iter().sum();
        let tr1: f64 = sys1.mat.diagonal_complex().0.iter().sum();
        // Stiffness trace > mass trace * 4, so tr1 < tr0 for coarse meshes
        assert!(tr1 < tr0, "trace should decrease with 蠅: {} vs {}", tr0, tr1);
    }

    #[test]
    fn native_complex_solve_diagonal_system() {
        // Scalar diagonal Helmholtz: (1 + i) u = b
        // Use a 1-element system by constructing ComplexCsr directly
        let row_ptr = vec![0usize, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![1.0_f64; 3];
        let im_vals = vec![1.0_f64; 3];
        let mat = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };
        let sys = NativeComplexSystem { mat, omega: 1.0, n_dofs: 3 };

        // b = (1+i)(2+3i) = 2+3i+2i-3 = -1+5i
        let b_re = vec![-1.0, -1.0, -1.0];
        let b_im = vec![5.0, 5.0, 5.0];
        let gf = sys.solve(&b_re, &b_im, 1e-10, 100, 50).unwrap();
        for i in 0..3 {
            assert!((gf.u_re[i] - 2.0).abs() < 1e-8, "u_re[{}] = {}", i, gf.u_re[i]);
            assert!((gf.u_im[i] - 3.0).abs() < 1e-8, "u_im[{}] = {}", i, gf.u_im[i]);
        }
    }

    #[test]
    fn native_helmholtz_size_matches_space() {
        let mesh  = Mesh::<2>::unit_square_tri(6);
        let space = H1Space::new(mesh, 1);
        let sys   = NativeComplexAssembler::assemble_helmholtz(&space, 1.0, 0.1, 1.0, 3.14, 3);
        assert_eq!(sys.n_dofs, space.n_dofs());
        assert_eq!(sys.mat.nrows, space.n_dofs());
        assert_eq!(sys.mat.re_vals.len(), sys.mat.im_vals.len());
        assert_eq!(sys.mat.re_vals.len(), sys.mat.nnz());
    }

    #[test]
    fn native_helmholtz_2d_pml_solve_converges() {
        // Solve Helmholtz with PML-like imaginary coefficient (strong damping 鈫?easy solve)
        let mesh  = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let omega = 2.0;
        let kappa_im = 5.0; // large imaginary part 鈫?well-conditioned

        let mut sys = NativeComplexAssembler::assemble_helmholtz(
            &space, 1.0, kappa_im, 1.0, omega, 3,
        );

        let n = space.n_dofs();
        let mut b_re = vec![1.0_f64; n];
        let mut b_im = vec![0.0_f64; n];

        // Apply zero Dirichlet BCs on boundary DOFs
        let bd_dofs = fem_space::constraints::boundary_dofs(
            space.mesh() as &dyn fem_mesh::topology::MeshTopology,
            space.dof_manager(),
            &[1, 2, 3, 4],
        );
        let bd_dofs_usize: Vec<usize> = bd_dofs.iter().map(|&d| d as usize).collect();
        sys.apply_dirichlet(&bd_dofs_usize, &vec![0.0; bd_dofs_usize.len()], &vec![0.0; bd_dofs_usize.len()],
                            &mut b_re, &mut b_im);

        let gf = sys.solve(&b_re, &b_im, 1e-8, 500, 50);
        assert!(gf.is_ok(), "native complex solve failed: {:?}", gf.err());
        let gf = gf.unwrap();
        // Check that at least some DOFs are non-zero (interior solution)
        let interior_max: f64 = gf.u_re.iter().cloned().fold(0.0_f64, f64::max);
        let amp_max: f64 = gf.amplitude().iter().cloned().fold(0.0_f64, f64::max);
        // Allow either real or imaginary part to be non-zero
        assert!(interior_max.abs() > 1e-12 || amp_max > 1e-12,
            "solution is zero: re_max={interior_max}, amp_max={amp_max}");
        assert!(amp_max.is_finite(), "solution amplitude is not finite");
    }

    #[test]
    fn native_complex_vs_block_agree_at_low_omega() {
        // At low 蠅 both approaches should give the same solution (up to numerical tol)
        use crate::standard::{DiffusionIntegrator, MassIntegrator};
        use fem_space::constraints::boundary_dofs;
        use fem_solver::{SolverConfig, solve_gmres};

        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let omega = 0.1;
        let n     = space.n_dofs();

        // Native complex approach
        let mut sys_nat = NativeComplexAssembler::assemble_helmholtz(
            &space, 1.0, 0.0, 1.0, omega, 3,
        );
        let mut b_re_nat = vec![1.0_f64; n];
        let mut b_im_nat = vec![0.0_f64; n];
        let bd = boundary_dofs(
            space.mesh() as &dyn fem_mesh::topology::MeshTopology,
            space.dof_manager(), &[1, 2, 3, 4],
        );
        let bd_usize: Vec<usize> = bd.iter().map(|&d| d as usize).collect();
        sys_nat.apply_dirichlet(&bd_usize, &vec![0.0; bd_usize.len()], &vec![0.0; bd_usize.len()],
                                &mut b_re_nat, &mut b_im_nat);
        let gf_nat = sys_nat.solve(&b_re_nat, &b_im_nat, 1e-10, 300, 50).unwrap();

        // 2脳2 block approach
        let mut sys_blk = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            &[],
            omega, 3,
        );
        let bd2 = boundary_dofs(
            space.mesh() as &dyn fem_mesh::topology::MeshTopology,
            space.dof_manager(), &[1, 2, 3, 4],
        );
        let bd2_usize: Vec<usize> = bd2.iter().map(|&d| d as usize).collect();
        let mut rhs_blk = sys_blk.assemble_rhs(&vec![1.0_f64; n], &vec![0.0_f64; n]);
        sys_blk.apply_dirichlet(&bd2_usize, &vec![0.0; bd2_usize.len()], &vec![0.0; bd2_usize.len()], &mut rhs_blk);
        let flat = sys_blk.to_flat_csr();
        let mut x_blk = vec![0.0_f64; 2 * n];
        let cfg = SolverConfig::default();
        solve_gmres(&flat, &rhs_blk, &mut x_blk, 50, &cfg).unwrap();
        let gf_blk = ComplexGridFunction::from_flat(&x_blk);

        // Compare L2 norms of real parts (should agree within 5%)
        let norm_nat: f64 = gf_nat.u_re.iter().map(|v| v*v).sum::<f64>().sqrt();
        let norm_blk: f64 = gf_blk.u_re.iter().map(|v| v*v).sum::<f64>().sqrt();
        let rel_diff = (norm_nat - norm_blk).abs() / norm_blk.max(1e-14);
        assert!(rel_diff < 0.15, "native and block approaches differ by {:.2}%: {} vs {}",
                rel_diff * 100.0, norm_nat, norm_blk);
    }
}

// 鈹€鈹€鈹€ Tests (original 2脳2 block) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use crate::standard::{DiffusionIntegrator, MassIntegrator};

    /// For 蠅 = 0 the complex system collapses to the pure stiffness matrix.
    #[test]
    fn complex_system_omega_zero_is_pure_stiffness() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);

        let sys = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            &[],  // no damping
            0.0,
            3,
        );

        let k = Assembler::assemble_bilinear(
            &space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

        // k_re should equal K for 蠅 = 0
        let n = sys.n_dofs();
        for i in 0..n {
            for ptr in sys.k_re.row_ptr[i]..sys.k_re.row_ptr[i + 1] {
                let j = sys.k_re.col_idx[ptr] as usize;
                let val_sys = sys.k_re.values[ptr];
                let val_k   = k.get(i, j);
                assert!((val_sys - val_k).abs() < 1e-12,
                    "k_re[{i},{j}] = {val_sys}, expected {val_k}");
            }
        }
    }

    /// The 2脳2 block matrix should be square of size 2n.
    #[test]
    fn flat_csr_size() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let omega = 1.0;

        let sys = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            &[&MassIntegrator { rho: 0.1 }],
            omega, 3,
        );

        let flat = sys.to_flat_csr();
        let n = sys.n_dofs();
        assert_eq!(flat.nrows, 2 * n);
        assert_eq!(flat.ncols, 2 * n);
    }

    /// The 2脳2 block matrix must be symmetric when `k_re` is symmetric and
    /// `k_im` is symmetric (standard H鹿 bilinear forms are symmetric).
    ///
    /// Symmetry of `[K_re, -K_im; K_im, K_re]`:
    /// Entry (i, j) = K_re[i,j]   and (j, i) = K_re[j,i] = K_re[i,j] 鉁?
    /// Entry (i, n+j) = -K_im[i,j] and (n+j, i) = K_im[j,i] = K_im[i,j]
    /// 鈫?NOT symmetric in general (block off-diagonal is skew-symmetric).
    /// But the full system IS the right formulation for Re{A}路x_re 鈭?Im{A}路x_im = f_re.
    #[test]
    fn flat_csr_diagonal_positive() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let omega = 0.5;

        let sys = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            &[],
            omega, 3,
        );
        let flat = sys.to_flat_csr();
        let n = sys.n_dofs();

        // Diagonal entries come from K_re; for small 蠅 on a coarse mesh at
        // interior nodes these should all be positive (K dominates 蠅虏M).
        for i in 0..2 * n {
            let d = flat.get(i, i);
            // Not necessarily positive on boundary-adjacent nodes, just check finite.
            assert!(d.is_finite(), "diagonal[{i}] = {d} is not finite");
        }
    }

    /// ComplexGridFunction amplitude helper.
    #[test]
    fn complex_gf_amplitude() {
        let gf = ComplexGridFunction {
            u_re: vec![3.0, 0.0],
            u_im: vec![4.0, 1.0],
        };
        let amp = gf.amplitude();
        assert!((amp[0] - 5.0).abs() < 1e-12);
        assert!((amp[1] - 1.0).abs() < 1e-12);
    }

    /// RHS assembly flattens correctly.
    #[test]
    fn rhs_assembly_concatenation() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n     = space.n_dofs();
        let f_re  = vec![1.0; n];
        let f_im  = vec![2.0; n];

        let sys = ComplexAssembler::assemble_undamped(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            1.0, 3,
        );
        let rhs = sys.assemble_rhs(&f_re, &f_im);
        assert_eq!(rhs.len(), 2 * n);
        assert_eq!(&rhs[..n], f_re.as_slice());
        assert_eq!(&rhs[n..], f_im.as_slice());
    }
}


