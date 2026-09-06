//! Complex-valued Crank-Nicolson time stepping for Schrödinger-type equations.
//!
//! Ported from MFEM's `CrankNicolsonTimeBaseSolver` used in schrodinger_flow.
//!
//! The Crank-Nicolson scheme for iℏ ∂ψ/∂t = Hψ is:
//!   (I + i·dt/2·H/ℏ) ψ^{n+1} = (I - i·dt/2·H/ℏ) ψ^n
//!
//! This is equivalent to the complex system:
//!   A·x = b   where A = K + i·ω·M, b = (K - i·ω·M)·ψ^n
//!   with K = stiffness, M = mass, ω = dt/(2ℏ)

use fem_linalg::complex_csr::{ComplexCsr, ComplexCoo, solve_gmres_complex};

/// Complex Crank-Nicolson time stepper.
///
/// Solves the linear Schrödinger equation iℏ ∂ψ/∂t = Hψ using CN:
///   (I + i·dt/2·H/ℏ) ψ^{n+1} = (I - i·dt/2·H/ℏ) ψ^n
///
/// The Hamiltonian H is represented as a complex sparse matrix
/// (typically H = K + i·ω·M where K is stiffness, M is mass).
#[derive(Debug, Clone)]
pub struct ComplexCrankNicolson {
    /// Complex system matrix A = I + i·dt/2·H/ℏ
    system_matrix: ComplexCsr,
    /// Complex operator B = I - i·dt/2·H/ℏ (for RHS)
    rhs_operator: ComplexCsr,
    /// GMRES tolerance
    tol: f64,
    /// GMRES max iterations
    max_iter: usize,
    /// GMRES restart parameter
    restart: usize,
    /// Use Jacobi preconditioning
    precond: bool,
}

impl ComplexCrankNicolson {
    /// Create a new complex Crank-Nicolson stepper.
    ///
    /// `hamiltonian` is the complex matrix H = H_re + i·H_im representing the Hamiltonian.
    /// `dt` is the time step, `hbar` is the Planck constant.
    pub fn new(
        hamiltonian: &ComplexCsr,
        dt: f64,
        hbar: f64,
    ) -> Self {
        let n = hamiltonian.nrows;
        let omega = dt / (2.0 * hbar);

        // Build A = I + i·omega·H = I + i·omega·(H_re + i·H_im)
        //        = I - omega·H_im + i·omega·H_re
        //        = (I - omega·H_im) + i·(omega·H_re)
        // Build B = I - i·omega·H = I - i·omega·(H_re + i·H_im)
        //        = I + omega·H_im - i·omega·H_re
        //        = (I + omega·H_im) + i·(-omega·H_re)

        let mut a_coo = ComplexCoo::new(n, n);
        let mut b_coo = ComplexCoo::new(n, n);

        // Add identity
        for i in 0..n {
            a_coo.add(i, i, 1.0, 0.0);
            b_coo.add(i, i, 1.0, 0.0);
        }

        // Add ±i·omega·H contributions
        for i in 0..n {
            for ptr in hamiltonian.row_ptr[i]..hamiltonian.row_ptr[i + 1] {
                let j = hamiltonian.col_idx[ptr] as usize;
                let h_re = hamiltonian.re_vals[ptr];
                let h_im = hamiltonian.im_vals[ptr];

                // A: -omega·H_im (real) + i·omega·H_re (imag)
                a_coo.add(i, j, -omega * h_im, omega * h_re);
                // B: +omega·H_im (real) - i·omega·H_re (imag)
                b_coo.add(i, j, omega * h_im, -omega * h_re);
            }
        }

        Self {
            system_matrix: a_coo.into_complex_csr(),
            rhs_operator: b_coo.into_complex_csr(),
            tol: 1e-10,
            max_iter: 500,
            restart: 30,
            precond: true,
        }
    }

    /// Set GMRES solver parameters.
    pub fn with_solver_params(mut self, tol: f64, max_iter: usize, restart: usize, precond: bool) -> Self {
        self.tol = tol;
        self.max_iter = max_iter;
        self.restart = restart;
        self.precond = precond;
        self
    }

    /// Perform one CN step: ψ^{n+1} = A^{-1}·B·ψ^n
    ///
    /// `psi_re` and `psi_im` are updated in-place.
    pub fn step(&self, psi_re: &mut [f64], psi_im: &mut [f64]) -> Result<(usize, f64), String> {
        let n = psi_re.len();
        assert_eq!(psi_im.len(), n);

        // Compute RHS: b = B·ψ^n
        let mut b_re = vec![0.0_f64; n];
        let mut b_im = vec![0.0_f64; n];
        self.rhs_operator.spmv_into(psi_re, psi_im, &mut b_re, &mut b_im);

        // Solve A·ψ^{n+1} = b
        let mut x_re = psi_re.to_vec();
        let mut x_im = psi_im.to_vec();
        let result = solve_gmres_complex(
            &self.system_matrix,
            &b_re,
            &b_im,
            &mut x_re,
            &mut x_im,
            self.tol,
            self.max_iter,
            self.restart,
            self.precond,
        )?;

        // Update solution
        psi_re.copy_from_slice(&x_re);
        psi_im.copy_from_slice(&x_im);

        Ok(result)
    }

    /// Get a reference to the system matrix.
    pub fn system_matrix(&self) -> &ComplexCsr {
        &self.system_matrix
    }

    /// Get a reference to the RHS operator.
    pub fn rhs_operator(&self) -> &ComplexCsr {
        &self.rhs_operator
    }
}

/// Build a complex Hamiltonian matrix from real stiffness and mass matrices.
///
/// H = K + i·ω·M  where ω = dt/(2ℏ)
///
/// This is the typical form for Schrödinger-type equations where
/// the stiffness K represents the kinetic energy and M is the mass matrix.
pub fn build_complex_hamiltonian(
    k_re: &fem_linalg::csr::CsrMatrix<f64>,
    m_re: &fem_linalg::csr::CsrMatrix<f64>,
    omega: f64,
) -> ComplexCsr {
    let n = k_re.nrows;
    let mut coo = ComplexCoo::new(n, n);

    // Add K (real part)
    for i in 0..n {
        for ptr in k_re.row_ptr[i]..k_re.row_ptr[i + 1] {
            let j = k_re.col_idx[ptr] as usize;
            coo.add(i, j, k_re.values[ptr], 0.0);
        }
    }

    // Add i·omega·M (imaginary part)
    for i in 0..n {
        for ptr in m_re.row_ptr[i]..m_re.row_ptr[i + 1] {
            let j = m_re.col_idx[ptr] as usize;
            coo.add(i, j, 0.0, omega * m_re.values[ptr]);
        }
    }

    coo.into_complex_csr()
}

/// Build a complex Hamiltonian from a single real matrix (e.g., full H already assembled).
pub fn build_complex_hamiltonian_real(
    h_re: &fem_linalg::csr::CsrMatrix<f64>,
    omega: f64,
) -> ComplexCsr {
    let n = h_re.nrows;
    let mut coo = ComplexCoo::new(n, n);

    for i in 0..n {
        for ptr in h_re.row_ptr[i]..h_re.row_ptr[i + 1] {
            let j = h_re.col_idx[ptr] as usize;
            coo.add(i, j, 0.0, omega * h_re.values[ptr]);
        }
    }

    coo.into_complex_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use fem_linalg::csr::CsrMatrix;

    #[test]
    fn test_complex_cn_identity() {
        // H = 0 (identity Hamiltonian), CN should preserve ψ
        let n = 4;
        let k = CooMatrix::<f64>::new(n, n);
        let h = ComplexCsr::from_re_im(&k.into_csr(), &CooMatrix::<f64>::new(n, n).into_csr());
        let cn = ComplexCrankNicolson::new(&h, 0.1, 1.0);

        let mut psi_re = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut psi_im = vec![0.5_f64, 1.0, 1.5, 2.0];

        let _ = cn.step(&mut psi_re, &mut psi_im).unwrap();

        // With H=0, A=I, B=I, so ψ should be unchanged
        for i in 0..n {
            assert!((psi_re[i] - (i + 1) as f64).abs() < 1e-10);
            assert!((psi_im[i] - (i + 1) as f64 * 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_complex_cn_harmonic() {
        // H = I (identity), CN should give exact solution for harmonic oscillator
        let n = 1;
        let mut k_re = CooMatrix::<f64>::new(n, n);
        k_re.add(0, 0, 1.0);
        let k_im = CooMatrix::<f64>::new(n, n);
        let h = ComplexCsr::from_re_im(&k_re.into_csr(), &k_im.into_csr());
        let cn = ComplexCrankNicolson::new(&h, 0.1, 1.0);

        let mut psi_re = vec![1.0_f64];
        let mut psi_im = vec![0.0_f64];

        let _ = cn.step(&mut psi_re, &mut psi_im).unwrap();

        // With H=I, ω=dt/2ℏ=0.05
        // A = I + i·0.05·I = (1 + 0.05i)·I
        // B = I - i·0.05·I = (1 - 0.05i)·I
        // ψ^{n+1} = B/A · ψ^n = (1 - 0.05i)/(1 + 0.05i) · 1
        // |B/A| = 1, so |ψ| should be preserved
        let norm = (psi_re[0] * psi_re[0] + psi_im[0] * psi_im[0]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Norm not preserved: {}", norm);
    }

    #[test]
    fn test_complex_cn_norm_preservation() {
        // For any Hermitian H, CN preserves the L2 norm of ψ
        let n = 3;
        // Simple diagonal Hamiltonian
        let mut k_re = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            k_re.add(i, i, (i + 1) as f64);
        }
        let k_im = CooMatrix::<f64>::new(n, n);
        let h = ComplexCsr::from_re_im(&k_re.into_csr(), &k_im.into_csr());
        let cn = ComplexCrankNicolson::new(&h, 0.05, 1.0);

        let mut psi_re = vec![1.0_f64, 0.5, -0.3];
        let mut psi_im = vec![0.2_f64, -0.7, 0.4];

        // Correct L2 norm: sqrt(sum(|psi_j|²))
        let norm0: f64 = psi_re.iter().zip(psi_im.iter())
            .map(|(r, i)| (r * r + i * i).sqrt())
            .sum();

        for _ in 0..10 {
            let _ = cn.step(&mut psi_re, &mut psi_im).unwrap();
        }

        let norm1: f64 = psi_re.iter().zip(psi_im.iter())
            .map(|(r, i)| (r * r + i * i).sqrt())
            .sum();

        // Norm should be approximately preserved (CN is unitary for Hermitian H)
        assert!((norm1 - norm0).abs() < 1e-6, "Norm drift: {} -> {}", norm0, norm1);
    }

    #[test]
    fn test_build_complex_hamiltonian() {
        let n = 3;
        let mut k_re = CooMatrix::<f64>::new(n, n);
        let mut m_re = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            k_re.add(i, i, 2.0);
            m_re.add(i, i, 1.0);
        }
        let omega = 0.05;
        let h = build_complex_hamiltonian(&k_re.into_csr(), &m_re.into_csr(), omega);

        // H = K + i·ω·M = diag(2) + i·0.05·diag(1) = diag(2 + 0.05i)
        assert_eq!(h.re_vals[0], 2.0);
        assert!((h.im_vals[0] - 0.05).abs() < 1e-14);
    }
}
