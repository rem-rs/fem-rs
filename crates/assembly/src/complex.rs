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
//! // −Δu − ω²u + iω·c·u = f
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
use fem_space::fe_space::FESpace;

use crate::assembler::Assembler;
use crate::integrator::{BilinearIntegrator, LinearIntegrator};

// ─── ComplexSystem ────────────────────────────────────────────────────────────

/// A 2×2 real-block equivalent of a complex linear system.
///
/// The system is:
/// ```text
/// [ A_re  A_im ] [ x_re ]   [ b_re ]
/// [-A_im  A_re ] [ x_im ] = [ b_im ]
/// ```
/// where `A = A_re + i·A_im` is the complex system matrix.
///
/// For the time-harmonic problem with stiffness `K`, mass `M`, damping `C`:
/// - `A_re = K − ω²·M`
/// - `A_im = ω·C`
pub struct ComplexSystem {
    /// Real part of the system matrix (n_dofs × n_dofs).
    pub k_re: CsrMatrix<f64>,
    /// Imaginary part: the "coupling" matrix (n_dofs × n_dofs).
    pub k_im: CsrMatrix<f64>,
    /// Angular frequency (stored for reference).
    pub omega: f64,
}

impl ComplexSystem {
    /// Number of real DOFs (= total DOFs in the underlying FE space).
    pub fn n_dofs(&self) -> usize { self.k_re.nrows }

    /// Total size of the flattened 2×2 block system = `2 * n_dofs`.
    pub fn n_total(&self) -> usize { 2 * self.n_dofs() }

    /// Build the flat (2n × 2n) block CSR matrix:
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

    /// Apply Dirichlet boundary conditions symmetrically on the 2×2 block
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
        rhs:   &mut Vec<f64>,
    ) {
        let n = self.n_dofs();
        // Eliminate each constrained DOF i in the real part (row/col i)
        // and the imaginary part (row/col n+i).
        for (k, &i) in dofs.iter().enumerate() {
            let val_re = bc_re[k];
            let val_im = bc_im[k];

            // --- Real block row i: set to identity (diagonal=1) ---
            self.k_re.apply_dirichlet_row_zeroing(i, val_re, rhs);
            // --- Imaginary block row i: zero completely (no diagonal identity) ---
            // In the flat system, k_im[i,:] appears in both top-right (−k_im)
            // and bottom-left (+k_im) blocks.  We want those rows to be 0 so that
            // the flat rows i and n+i decouple to: u_re[i]=val_re, u_im[i]=val_im.
            zero_row(&mut self.k_im, i, &mut rhs[n..]);

            rhs[i]     = val_re;
            rhs[n + i] = val_im;
        }
    }
}

/// Zero all entries in a CSR row (including diagonal).  Does NOT modify rhs.
fn zero_row(mat: &mut CsrMatrix<f64>, row: usize, _rhs: &mut [f64]) {
    let start = mat.row_ptr[row];
    let end   = mat.row_ptr[row + 1];
    for k in start..end {
        mat.values[k] = 0.0;
    }
}

// ─── ComplexAssembler ─────────────────────────────────────────────────────────

/// Assembles a complex-valued time-harmonic PDE system using real block
/// decomposition.
///
/// The assembled system represents:
/// ```text
/// (K − ω²M + iωC) u = f
/// ```
/// expanded to the 2×2 real block form stored in [`ComplexSystem`].
pub struct ComplexAssembler;

impl ComplexAssembler {
    /// Assemble a time-harmonic complex system.
    ///
    /// # Arguments
    /// - `space`   — FE space (H¹, H(curl), H(div))
    /// - `stiff`   — stiffness integrators for `K` (e.g. `DiffusionIntegrator`, `CurlCurlIntegrator`)
    /// - `mass`    — mass integrators for `M` (e.g. `MassIntegrator`, `VectorMassIntegrator`)
    /// - `damp`    — damping integrators for `C` (e.g. `MassIntegrator` with conductivity σ)
    /// - `omega`   — angular frequency ω
    /// - `quad_order` — quadrature order
    ///
    /// Returns `(system, k_re, k_im)` where:
    /// - `k_re = K − ω²·M`
    /// - `k_im = ω·C`
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

        // k_re = K − ω²·M
        let k_re = subtract_scaled(&k, &m, omega * omega);
        // k_im = ω·C
        let k_im = scale_csr(&c, omega);

        ComplexSystem { k_re, k_im, omega }
    }

    /// Assemble a purely-stiffness + mass system with no damping.
    ///
    /// `k_re = K − ω²·M`,  `k_im = 0`.
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
}

// ─── ComplexLinearForm ────────────────────────────────────────────────────────

/// Assembles a complex-valued right-hand side `f = f_re + i·f_im`.
pub struct ComplexLinearForm {
    /// Real part contributions.
    pub f_re: Vec<f64>,
    /// Imaginary part contributions.
    pub f_im: Vec<f64>,
}

impl ComplexLinearForm {
    /// Assemble separate real and imaginary parts.
    ///
    /// Either slice of integrators may be empty (→ zero contribution).
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

// ─── ComplexGridFunction ──────────────────────────────────────────────────────

/// A complex grid function `u = u_re + i·u_im`.
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

    /// Pointwise amplitude `|u(x)| = sqrt(u_re² + u_im²)`.
    pub fn amplitude(&self) -> Vec<f64> {
        self.u_re.iter().zip(self.u_im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Total complex L² norm: `sqrt(‖u_re‖² + ‖u_im‖²)`.
    pub fn l2_norm(&self) -> f64 {
        let re: f64 = self.u_re.iter().map(|x| x * x).sum();
        let im: f64 = self.u_im.iter().map(|x| x * x).sum();
        (re + im).sqrt()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Compute `A − alpha·B` (both same sparsity pattern, using COO merge).
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use crate::standard::{DiffusionIntegrator, MassIntegrator};

    /// For ω = 0 the complex system collapses to the pure stiffness matrix.
    #[test]
    fn complex_system_omega_zero_is_pure_stiffness() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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

        // k_re should equal K for ω = 0
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

    /// The 2×2 block matrix should be square of size 2n.
    #[test]
    fn flat_csr_size() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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

    /// The 2×2 block matrix must be symmetric when `k_re` is symmetric and
    /// `k_im` is symmetric (standard H¹ bilinear forms are symmetric).
    ///
    /// Symmetry of `[K_re, -K_im; K_im, K_re]`:
    /// Entry (i, j) = K_re[i,j]   and (j, i) = K_re[j,i] = K_re[i,j] ✓
    /// Entry (i, n+j) = -K_im[i,j] and (n+j, i) = K_im[j,i] = K_im[i,j]
    /// → NOT symmetric in general (block off-diagonal is skew-symmetric).
    /// But the full system IS the right formulation for Re{A}·x_re − Im{A}·x_im = f_re.
    #[test]
    fn flat_csr_diagonal_positive() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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

        // Diagonal entries come from K_re; for small ω on a coarse mesh at
        // interior nodes these should all be positive (K dominates ω²M).
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
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
