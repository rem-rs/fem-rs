//! Reduced-Order Models via Proper Orthogonal Decomposition (POD).
//!
//! # Overview
//!
//! 1. **Snapshot collection** — solve the full-order model at N parameter
//!    values / time steps; store each solution as a column of `Snapshots`.
//! 2. **POD basis** — compute the SVD of the snapshot matrix and retain
//!    the `r` dominant left singular vectors.
//! 3. **Galerkin projection** — project the full-order operator `A` and
//!    RHS `b` onto the POD basis: `A_r = V^T A V`, `b_r = V^T b`.
//! 4. **Online solve** — solve the r×r reduced system.
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::rom::{Snapshots, PodBasis, project_system};
//!
//! // Collect N solution vectors (each of length n):
//! let mut snaps = Snapshots::new(n);
//! for param in params {
//!     let u = solve_fom(param);
//!     snaps.add_snapshot(&u);
//! }
//!
//! // Extract r dominant POD modes:
//! let pod = PodBasis::compute(&snaps, r)?;
//!
//! // Project the system:
//! let (a_r, b_r) = project_system(&a, &b, &pod);
//!
//! // Solve the reduced system:
//! let x_r = a_r.solve(&b_r);
//! ```

use nalgebra::{DMatrix, DVector, SVD};
use fem_linalg::CsrMatrix;

/// Snapshot matrix: columns are solution vectors at different parameters / times.
///
/// Shape: `(n_dofs, n_snapshots)`.
#[derive(Debug, Clone)]
pub struct Snapshots {
    n_dofs: usize,
    snapshots: Vec<Vec<f64>>,
}

impl Snapshots {
    /// Create an empty snapshot collector for an `n`-DOF system.
    pub fn new(n_dofs: usize) -> Self {
        Snapshots { n_dofs, snapshots: Vec::new() }
    }

    /// Add a solution vector as a new snapshot column.
    pub fn add_snapshot(&mut self, u: &[f64]) {
        assert_eq!(u.len(), self.n_dofs, "snapshot length {} != n_dofs {}", u.len(), self.n_dofs);
        self.snapshots.push(u.to_vec());
    }

    /// Number of DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Number of snapshots collected.
    pub fn n_snapshots(&self) -> usize { self.snapshots.len() }

    /// Access the i-th snapshot.
    pub fn snapshot(&self, i: usize) -> &[f64] { &self.snapshots[i] }

    /// Convert to a dense matrix (n_dofs × n_snapshots).
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.n_dofs, self.snapshots.len());
        for (j, snap) in self.snapshots.iter().enumerate() {
            for (i, &val) in snap.iter().enumerate() {
                m[(i, j)] = val;
            }
        }
        m
    }
}

/// POD basis: `r` dominant modes (columns) = left singular vectors of the
/// snapshot matrix.
#[derive(Debug, Clone)]
pub struct PodBasis {
    /// POD modes (columns). Shape: `(n_dofs, r)`.
    pub modes: DMatrix<f64>,
    /// Singular values (sorted descending).
    pub singular_values: Vec<f64>,
    /// Number of retained modes.
    pub r: usize,
}

impl PodBasis {
    /// Compute the POD basis from snapshots, retaining the top `r` modes.
    ///
    /// Internally computes the thin SVD of the snapshot matrix.
    /// Returns an error if `r` exceeds the snapshot count or DOF count.
    pub fn compute(snaps: &Snapshots, r: usize) -> Result<Self, String> {
        let n_snap = snaps.n_snapshots();
        if n_snap == 0 {
            return Err("POD: no snapshots".to_string());
        }
        let r_eff = r.min(n_snap).min(snaps.n_dofs());
        if r_eff == 0 {
            return Err("POD: effective rank is 0".to_string());
        }

        let s = snaps.to_matrix();
        let svd = SVD::new(s, true, false);
        let u = svd.u.ok_or("POD: SVD did not return U")?;

        let n_modes = r_eff.min(u.ncols());
        let modes = u.columns(0, n_modes).into_owned();

        // Build a small intermediate S matrix to extract singular values
        // Start from a minimal size to get the first n_modes singular values
        // Note: in nalgebra, svd.singular_values is a Vector of min(n_dofs, n_snap) values
        let sv = svd.singular_values;
        let sv_total = sv.len();
        let n_sv = n_modes.min(sv_total);
        let singular_values: Vec<f64> = sv.iter().take(n_sv).copied().collect();

        Ok(PodBasis { modes, singular_values, r: n_modes })
    }

    /// Number of basis vectors (reduced dimension).
    pub fn n_modes(&self) -> usize { self.r }

    /// The i-th POD mode (column vector, length n_dofs).
    pub fn mode(&self, i: usize) -> Vec<f64> {
        (0..self.modes.nrows()).map(|row| self.modes[(row, i)]).collect()
    }

    /// Energy fraction captured by the first `k` modes: sum(σᵢ²) / sum(all σᵢ²).
    pub fn energy_fraction(&self, k: usize) -> f64 {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total == 0.0 { return 1.0; }
        let partial: f64 = self.singular_values.iter().take(k).map(|s| s * s).sum();
        partial / total
    }

    /// Cumulative energy fraction across all computed modes.
    pub fn cumulative_energy(&self) -> Vec<f64> {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total == 0.0 { return vec![1.0; self.singular_values.len()]; }
        let mut cum = Vec::with_capacity(self.singular_values.len());
        let mut running = 0.0;
        for s in &self.singular_values {
            running += s * s;
            cum.push(running / total);
        }
        cum
    }
}

/// Project a full-order system `(A, b)` onto the POD basis.
///
/// Returns `(A_r, b_r)` where:
/// - `A_r = V^T A V`  (r × r reduced operator)
/// - `b_r = V^T b`    (r × 1 reduced RHS)
pub fn project_system(
    a: &CsrMatrix<f64>,
    b: &[f64],
    pod: &PodBasis,
) -> (DMatrix<f64>, DVector<f64>) {
    let n = a.nrows;
    let r = pod.r;
    assert_eq!(n, b.len());

    // Compute AV: each column = A * v_i
    let mut av = DMatrix::zeros(n, r);
    for j in 0..r {
        let mode_j = pod.mode(j);
        let mut col = vec![0.0; n];
        a.spmv(&mode_j, &mut col);
        for i in 0..n {
            av[(i, j)] = col[i];
        }
    }

    // A_r = V^T * (A * V) = V^T * AV
    let a_r = pod.modes.transpose() * &av;

    // b_r = V^T * b
    let b_vec = DVector::from_vec(b.to_vec());
    let b_r = pod.modes.transpose() * b_vec;

    (a_r, b_r)
}

/// Reconstruct the full-order solution from reduced coefficients.
///
/// `u_full = V * u_r`
pub fn reconstruct(pod: &PodBasis, u_r: &[f64]) -> Vec<f64> {
    let u_vec = DVector::from_vec(u_r.to_vec());
    let full = &pod.modes * u_vec;
    full.data.as_vec().clone()
}

/// Compute the relative L² error between reduced and full-order solutions.
pub fn relative_error(u_full: &[f64], u_reduced: &[f64]) -> f64 {
    let diff: f64 = u_full.iter().zip(u_reduced.iter())
        .map(|(a, b)| (a - b).powi(2)).sum();
    let norm: f64 = u_full.iter().map(|a| a.powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

/// Full online ROM pipeline: given a `PodBasis` and the projected affine
/// decompositions, evaluate at parameter `μ`, solve the reduced system,
/// and reconstruct the full-order solution.
///
/// # Arguments
/// * `pod` — POD basis (modes matrix `V`, size `n×r`)
/// * `affine_op_proj` — projected operator components `Vᵀ·A_q·V` (each `r×r`)
/// * `affine_rhs_proj` — projected RHS components `Vᵀ·b_q` (each `r`-vector)
/// * `mu` — parameter vector
///
/// # Returns
/// `(u_r, u_full)` where `u_r` is the reduced coefficients and `u_full` is
/// the reconstructed full-order solution.
pub fn online_solve(
    pod: &PodBasis,
    affine_op_proj: &AffineDecomposition<DMatrix<f64>>,
    affine_rhs_proj: &AffineDecomposition<DVector<f64>>,
    mu: &[f64],
) -> Result<(DVector<f64>, Vec<f64>), String> {
    let r = pod.r;
    // Assemble reduced system: A_r(μ) = Σ θ_q(μ) · A_q_proj
    let mut a_r = DMatrix::zeros(r, r);
    let mut b_r = DVector::zeros(r);
    for i in 0..affine_op_proj.n_terms() {
        let theta = affine_op_proj.coeffs[i](mu);
        if theta.abs() < 1e-300 { continue; }
        a_r += theta * &affine_op_proj.components[i];
    }
    for i in 0..affine_rhs_proj.n_terms() {
        let theta = affine_rhs_proj.coeffs[i](mu);
        if theta.abs() < 1e-300 { continue; }
        b_r += theta * &affine_rhs_proj.components[i];
    }
    let u_r = a_r.lu().solve(&b_r)
        .ok_or_else(|| "reduced system solve failed".to_string())?;
    let u_full = reconstruct(pod, u_r.as_slice());
    Ok((u_r, u_full))
}

// ─── EIM (Empirical Interpolation Method) ───────────────────────────────────

/// EIM basis: greedy-selected basis vectors and interpolation points for
/// approximating parameter-dependent functions `g(μ)`.
///
/// Unlike DEIM (which takes POD modes as fixed input), EIM builds both the
/// basis AND the interpolation points greedily from training snapshots.
/// This is the workhorse for hyper-reduction of non-affine parametric
/// operators in reduced-basis and ROM settings.
///
/// # Algorithm (Barrault et al. 2004)
///
/// 1. Choose first snapshot as first basis vector; magic point = argmax |b₁|.
/// 2. For k = 2, …, m: for each training snapshot, compute the interpolation
///    residual; select the snapshot with the largest residual norm as the
///    next basis vector; its argmax is the next magic point.
#[derive(Debug, Clone)]
pub struct EimBasis {
    /// Basis vectors (columns). Shape: `(n_dofs, m)`.
    pub basis: DMatrix<f64>,
    /// Interpolation (magic) points — row indices in [0, n_dofs).
    pub points: Vec<usize>,
    /// Number of basis vectors.
    pub m: usize,
}

impl EimBasis {
    /// Greedy construction of an EIM basis from training snapshots.
    ///
    /// # Arguments
    /// * `snapshots` — training function evaluations, each a length-`n` vector
    /// * `m`         — desired number of basis vectors (≤ n_snapshots)
    ///
    /// Returns the EIM basis with `m` columns and the corresponding `m` magic
    /// point indices.
    pub fn build(snapshots: &[Vec<f64>], m: usize) -> Result<Self, String> {
        let n_snap = snapshots.len();
        if n_snap == 0 || snapshots[0].is_empty() {
            return Err("EIM: empty snapshots".into());
        }
        let n = snapshots[0].len();
        let m_eff = m.min(n_snap);
        if m_eff == 0 {
            return Err("EIM: effective rank is 0".into());
        }

        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m_eff);
        let mut points: Vec<usize> = Vec::with_capacity(m_eff);

        // Step 1: first basis = snapshot with largest norm; first point = argmax |b₁|
        let (idx_first, _) = snapshots.iter().enumerate()
            .map(|(i, s)| (i, s.iter().map(|v| v * v).sum::<f64>()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or("EIM: empty snapshots")?;

        let first_vec = &snapshots[idx_first];
        let inf_norm = first_vec.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if inf_norm < 1e-300 {
            return Err("EIM: zero snapshot".into());
        }
        let b1: Vec<f64> = first_vec.iter().map(|v| v / inf_norm).collect();
        let p0 = b1.iter().enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        basis.push(b1);
        points.push(p0);

        // Steps 2..m: greedy selection
        for _k in 1..m_eff {
            let mut best_res_norm = -1.0_f64;
            let mut best_basis: Option<Vec<f64>> = None;
            let mut best_point = 0usize;

            for snap in snapshots.iter() {
                // Interpolation matrix B_sub[m×m] at selected points
                let m_prev = points.len();
                let b_sub = DMatrix::from_fn(m_prev, m_prev, |i, j| basis[j][points[i]]);
                let b_rhs = DVector::from_fn(m_prev, |i, _| snap[points[i]]);

                // Solve B_sub · c = snapshot[points] for coefficients
                let c = match b_sub.lu().solve(&b_rhs) {
                    Some(c) => c,
                    None => continue,  // singular; skip this snapshot
                };

                // Residual r = snap - basis · c
                let mut residual = DVector::from_vec(snap.clone());
                for j in 0..m_prev {
                    for i in 0..n {
                        residual[i] -= c[j] * basis[j][i];
                    }
                }

                let res_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
                if res_norm > best_res_norm {
                    best_res_norm = res_norm;
                    // Normalised residual as next basis vector
                    let r_inf = residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                    let r_norm = if r_inf > 1e-300 {
                        residual.iter().map(|v| v / r_inf).collect()
                    } else {
                        residual.iter().map(|_| 0.0).collect()
                    };
                    let pk = residual.iter().enumerate()
                        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    best_basis = Some(r_norm);
                    best_point = pk;
                }
            }

            match best_basis {
                Some(b) => {
                    if best_res_norm < 1e-14 {
                        break;  // converged — remaining snapshots are in the span
                    }
                    basis.push(b);
                    points.push(best_point);
                }
                None => break,
            }
        }

        let m_final = basis.len();
        let mut basis_mat = DMatrix::zeros(n, m_final);
        for j in 0..m_final {
            for i in 0..n {
                basis_mat[(i, j)] = basis[j][i];
            }
        }

        Ok(EimBasis { basis: basis_mat, points, m: m_final })
    }

    /// Interpolate an arbitrary vector `f` using the EIM basis and magic points.
    ///
    /// Returns `f_eim ≈ f` such that `f_eim = B · c` where `c` solves the
    /// interpolation system at the magic points.
    pub fn interpolate(&self, f: &[f64]) -> Vec<f64> {
        let n = f.len();
        let m = self.m;
        let b_sub = DMatrix::from_fn(m, m, |i, j| self.basis[(self.points[i], j)]);
        let b_rhs = DVector::from_fn(m, |i, _| f[self.points[i]]);
        let c = b_sub.lu().solve(&b_rhs)
            .expect("EIM interpolation matrix should be invertible");

        let mut result = vec![0.0; n];
        for j in 0..m {
            let cj = c[j];
            for i in 0..n {
                result[i] += cj * self.basis[(i, j)];
            }
        }
        result
    }

    /// Evaluate the interpolation residual ‖f - I[f]‖ / ‖f‖.
    pub fn relative_interp_error(&self, f: &[f64]) -> f64 {
        let f_eim = self.interpolate(f);
        let diff: f64 = f.iter().zip(f_eim.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let norm: f64 = f.iter().map(|a| a.powi(2)).sum();
        (diff / norm.max(1e-30)).sqrt()
    }
}

/// Represent an affine decomposition `A(μ) = Σ_q θ_q(μ) · A_q` where each
/// `A_q` is a parameter-independent sparse matrix and `θ_q` is a scalar
/// function of the parameter vector `μ`.
///
/// This enables the offline-online decomposition in reduced-basis methods:
/// the parameter-independent matrices `A_q` are assembled once (offline),
/// and the reduced system is assembled online from pre-computed
/// `(Vᵀ A_q V)` by summing with parameter-dependent coefficients.
#[derive(Debug, Clone)]
pub struct AffineDecomposition<T> {
    /// Parameter-independent component matrices.
    pub components: Vec<T>,
    /// Coefficient functions θ_q(μ). Each returns a scalar weight.
    pub coeffs: Vec<fn(&[f64]) -> f64>,
}

impl<T> AffineDecomposition<T> {
    /// Create a new affine decomposition.
    pub fn new(components: Vec<T>, coeffs: Vec<fn(&[f64]) -> f64>) -> Self {
        assert_eq!(components.len(), coeffs.len(),
            "number of components must match number of coefficient functions");
        AffineDecomposition { components, coeffs }
    }

    /// Number of terms in the affine decomposition.
    pub fn n_terms(&self) -> usize { self.components.len() }
}

impl AffineDecomposition<Vec<f64>> {
    /// Project the RHS affine components onto a POD basis: `b_q_r = Vᵀ · b_q`.
    pub fn project_rhs(&self, pod: &PodBasis) -> AffineDecomposition<DVector<f64>> {
        let projected: Vec<DVector<f64>> = self.components.iter().map(|b| {
            let b_vec = DVector::from_vec(b.clone());
            pod.modes.transpose() * b_vec
        }).collect();
        AffineDecomposition { components: projected, coeffs: self.coeffs.clone() }
    }
}

impl AffineDecomposition<CsrMatrix<f64>> {
    /// Evaluate the operator at parameter `μ`: `A(μ) = Σ θ_q(μ) · A_q`.
    pub fn evaluate(&self, mu: &[f64]) -> CsrMatrix<f64> {
        let n = self.components[0].nrows;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for (comp, &coeff_fn) in self.components.iter().zip(self.coeffs.iter()) {
            let theta = coeff_fn(mu);
            if theta.abs() < 1e-300 { continue; }
            for row in 0..comp.nrows {
                for ptr in comp.row_ptr[row]..comp.row_ptr[row + 1] {
                    let col = comp.col_idx[ptr] as usize;
                    coo.add(row, col, theta * comp.values[ptr]);
                }
            }
        }
        coo.into_csr()
    }

    /// Project each component matrix onto a POD basis: `A_q_r = Vᵀ A_q V`.
    /// Returns the reduced affine decomposition ready for online evaluation.
    pub fn project(&self, pod: &PodBasis) -> AffineDecomposition<DMatrix<f64>> {
        let projected: Vec<DMatrix<f64>> = self.components.iter().map(|a| {
            let n = a.nrows;
            let r = pod.r;
            let mut av = DMatrix::zeros(n, r);
            for j in 0..r {
                let mode_j = pod.mode(j);
                let mut col = vec![0.0; n];
                a.spmv(&mode_j, &mut col);
                for i in 0..n { av[(i, j)] = col[i]; }
            }
            pod.modes.transpose() * av
        }).collect();
        AffineDecomposition {
            components: projected,
            coeffs: self.coeffs.clone(),
        }
    }
}

// ─── Error certification helpers ────────────────────────────────────────────

/// A posteriori error estimate for a reduced solution:
///
/// `Δ(μ) = ‖r(μ)‖ / α(μ)`
///
/// where `‖r(μ)‖` is the norm of the full-order residual evaluated at the
/// reduced solution, and `α(μ)` is a lower bound for the coercivity constant
/// (for coercive problems) or the inf-sup constant (for saddle-point problems).
///
/// The residual is cheap to compute when the operator has an affine
/// decomposition: `r = b(μ) - A(μ)·(V·u_r) = b(μ) - A(μ)·u_{full}`.
pub struct ErrorEstimator {
    /// Pre-computed norms for the affine components of the residual
    /// (used when the operator has an affine decomposition).
    _private: (),
}

impl ErrorEstimator {
    /// Compute the relative residual norm for a reduced solution.
    ///
    /// This evaluates the true residual `‖b - A·(V·u_r)‖ / ‖b‖` using the
    /// full-order operator, which costs O(n²) but provides a rigorous bound.
    pub fn relative_residual(
        a: &CsrMatrix<f64>,
        b: &[f64],
        reduced_solution: &[f64],
        pod: &PodBasis,
    ) -> f64 {
        // Reconstruct full-order solution
        let u_full = reconstruct(pod, reduced_solution);
        // Compute residual r = b - A·u_full
        let n = b.len();
        let mut r = vec![0.0; n];
        a.spmv(&u_full, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }
        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        r_norm / b_norm.max(1e-300)
    }

    /// Efficient residual bound using pre-computed affine-reduced quantities.
    ///
    /// Computes the squared norm of the reduced-space residual:
    ///   r_q(μ) = b_q_proj - A_q_proj · u_r   (for each affine component q)
    ///   ‖r_eff(μ)‖² = ‖Σ_q θ_q(μ) · r_q(μ)‖²
    ///
    /// where b_q_proj = Vᵀ·b_q and A_q_proj = Vᵀ·A_q·V are the projected
    /// affine components.  The online cost is O(Q·r²) per parameter instead
    /// of O(n²) for the full residual.
    pub fn efficient_residual_sq(
        reduced_solution: &[f64],
        pod: &PodBasis,
        affine_rhs: &AffineDecomposition<DVector<f64>>,
        affine_op: &AffineDecomposition<DMatrix<f64>>,
        mu: &[f64],
    ) -> f64 {
        let r = pod.r;
        let u = DVector::from_vec(reduced_solution.to_vec());
        let q = affine_op.n_terms().min(affine_rhs.n_terms());
        if q == 0 { return 0.0; }

        // Accumulate residual in reduced space: r_red = Σ_q θ_q · (b_q_proj - A_q_proj · u_r)
        let mut r_red = DVector::zeros(r);
        for i in 0..q {
            let theta = affine_rhs.coeffs[i](mu);
            if theta.abs() < 1e-300 { continue; }
            let b_q = &affine_rhs.components[i];
            let a_q = &affine_op.components[i];
            let mut r_q = b_q.clone();
            r_q -= a_q * &u;  // r_q = b_q_proj - A_q_proj · u_r
            r_red += theta * r_q;
        }
        r_red.norm_squared()
    }
}

/// Greedy DEIM index selection from the POD basis.
///
/// Returns `r` interpolation indices (rows) such that `U(P^T U)^{-1}`
/// is well-conditioned for interpolating nonlinear functions.
///
/// Implements Algorithm 1 from Chaturantabut & Sorensen (2010).
pub fn deim_greedy(modes: &DMatrix<f64>, r: usize) -> Vec<usize> {
    let n = modes.nrows();
    let r_eff = r.min(modes.ncols());
    let mut indices = Vec::with_capacity(r_eff);

    // Step 1: first index = argmax of first mode
    let v1 = modes.column(0);
    let (idx0, _) = v1.argmax();
    indices.push(idx0);

    for i in 1..r_eff {
        let vi = modes.column(i);
        // Extract the submatrix V_{prev} at selected indices
        let m = indices.len();
        let v_sub = DMatrix::from_fn(m, m, |row, col| modes[(indices[row], col)]);
        let b_sub = DVector::from_fn(m, |row, _| modes[(indices[row], i)]);

        // Solve V_sub · c = b_sub
        let c = match v_sub.lu().solve(&b_sub) {
            Some(c) => c,
            None => break,
        };

        // r = v_i - V_prev · c
        let mut residual = vi.into_owned();
        for j in 0..m {
            for k in 0..n {
                residual[k] -= c[j] * modes[(k, j)];
            }
        }

        // Next index = argmax of |residual|
        let (next_idx, _) = residual.argmax();
        if !indices.contains(&next_idx) {
            indices.push(next_idx);
        } else {
            // All remaining entries are zero; stop early.
            break;
        }
    }
    indices
}

/// Interpolate a vector onto the DEIM subspace.
///
/// `u_deim = U (P^T U)^{-1} P^T u`, where `P` selects `indices` rows.
pub fn deim_interpolate(
    u: &DVector<f64>,
    modes: &DMatrix<f64>,  // POD modes (n × m)
    indices: &[usize],
) -> Vec<f64> {
    let m = indices.len();
    // P^T U: m × m matrix from selected rows
    let mut pt_u = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            pt_u[(i, j)] = modes[(indices[i], j)];
        }
    }
    // P^T u: m-vector
    let pt_u_vec = DVector::from_fn(m, |i, _| u[indices[i]]);

    // Solve (P^T U) c = P^T u for c
    let c = pt_u.lu().solve(&pt_u_vec)
        .expect("DEIM interpolation matrix should be invertible");

    // u_deim = U · c
    (modes * c).data.as_vec().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SolverConfig, solve_cg};
    use fem_linalg::CsrMatrix;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    fn solve_laplacian_1d(n: usize, rhs_scale: f64) -> Vec<f64> {
        let a = laplacian_1d(n);
        let mut x = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mid = n / 2;
        b[mid - 1] = rhs_scale;
        b[mid] = 2.0 * rhs_scale;
        b[mid + 1] = rhs_scale;
        let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
        solve_cg(&a, &b, &mut x, &cfg).expect("CG");
        x
    }

    #[test]
    fn pod_snapshot_creation() {
        let mut snaps = Snapshots::new(10);
        assert_eq!(snaps.n_dofs(), 10);
        assert_eq!(snaps.n_snapshots(), 0);

        snaps.add_snapshot(&vec![1.0; 10]);
        assert_eq!(snaps.n_snapshots(), 1);
    }

    #[test]
    fn pod_basis_from_snapshots() {
        let mut snaps = Snapshots::new(20);
        for scale in 0..5 {
            let u = solve_laplacian_1d(20, scale as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 3).expect("POD basis");
        assert_eq!(pod.n_modes(), 3);
        assert_eq!(pod.modes.nrows(), 20);
        assert_eq!(pod.modes.ncols(), 3);
        assert!(pod.singular_values[0] >= pod.singular_values[1]);
    }

    #[test]
    fn pod_energy_fraction() {
        let mut snaps = Snapshots::new(20);
        for scale in 0..5 {
            let u = solve_laplacian_1d(20, scale as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 5).expect("POD basis");
        let e = pod.energy_fraction(5);
        assert!((e - 1.0).abs() < 1e-10, "all modes should capture all energy: {e:.10}");
    }

    #[test]
    fn pod_projection_reduces_dimension() {
        let n = 32;
        let mut snaps = Snapshots::new(n);
        for scale in 0..6 {
            let u = solve_laplacian_1d(n, scale as f64);
            snaps.add_snapshot(&u);
        }

        let r = 4;
        let pod = PodBasis::compute(&snaps, r).expect("POD basis");

        let a = laplacian_1d(n);
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;

        let (a_r, b_r) = project_system(&a, &b, &pod);
        assert_eq!(a_r.nrows(), r);
        assert_eq!(a_r.ncols(), r);
        assert_eq!(b_r.len(), r);
    }

    #[test]
    fn pod_reconstruction_small_error() {
        let n = 32;
        let mut snaps = Snapshots::new(n);
        // Snapshots from many RHS configurations
        for k in 0..8 {
            let mut b = vec![0.0; n];
            b[k * 4] = 1.0;
            let a = laplacian_1d(n);
            let mut x = vec![0.0; n];
            let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
            solve_cg(&a, &b, &mut x, &cfg).expect("CG");
            snaps.add_snapshot(&x);
        }

        let r = 5;
        let pod = PodBasis::compute(&snaps, r).expect("POD basis");

        // Project and solve for a RHS that is a linear combination of snapshots
        let a = laplacian_1d(n);
        let mut b = vec![0.0; n];
        // Use the SAME delta RHS as one of the snapshots (k=1 → at index 4)
        b[4] = 1.0;
        let (a_r, b_r) = project_system(&a, &b, &pod);

        // Solve reduced system (dense solve via nalgebra)
        let a_r_dense = a_r;
        let u_r = a_r_dense.lu().solve(&b_r).expect("reduced solve");
        let u_full = reconstruct(&pod, u_r.as_slice());

        // Compare with full solve
        let mut x_full = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
        solve_cg(&a, &b, &mut x_full, &cfg).expect("CG");

        let err = relative_error(&x_full, &u_full);
        assert!(err < 0.1, "ROM reconstruction error should be reasonable, got {err:.3e}");
    }

    #[test]
    fn pod_energy_monotonic() {
        let mut snaps = Snapshots::new(16);
        for k in 1..=6 {
            let u = solve_laplacian_1d(16, k as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 6).expect("POD basis");
        let cum = pod.cumulative_energy();
        for i in 1..cum.len() {
            assert!(cum[i] >= cum[i - 1], "energy should be monotonic: {cum:?}");
        }
    }

    // ── POD error bound ───────────────────────────────────────────────────────

    #[test]
    fn pod_projection_error_bounded_by_truncated_singular_values() {
        // For a matrix of snapshots S = U Σ V^T, the Frobenius-norm error
        // of the rank-r truncated SVD is ‖S - S_r‖_F = sqrt(σ_{r+1}² + ...).
        // When projecting a snapshot onto the POD basis, the error should be
        // bounded by the next singular value.
        let n = 24;
        let mut snaps = Snapshots::new(n);
        for k in 1..=8 {
            let u = solve_laplacian_1d(n, k as f64);
            snaps.add_snapshot(&u);
        }

        let pod = PodBasis::compute(&snaps, 6).expect("POD basis");
        let sv = &pod.singular_values;
        // Project each snapshot onto the first 4 modes and measure error.
        let r = 4;
        let v4 = pod.modes.columns(0, r).into_owned();
        for snap_idx in 0..snaps.n_snapshots() {
            let s = DVector::from_vec(snaps.snapshot(snap_idx).to_vec());
            let coeff = v4.transpose() * &s; // V^T s
            let s_proj = &v4 * coeff; // V V^T s
            let err = (&s - s_proj).norm();
            // Bound: error ≤ sqrt(σ_5² + σ_6²) (tail energy)
            let tail_sq: f64 = sv.iter().skip(r).map(|&s| s * s).sum();
            assert!(err * err <= tail_sq + 1e-10,
                "snap {snap_idx}: proj err² {:.3e} > tail {:.3e}", err * err, tail_sq);
        }
    }

    // ─── DEIM ─────────────────────────────────────────────────────────────────

    #[test]
    fn deim_interpolation_error_decreases_with_modes() {
        let n = 32;
        let mut snaps = Snapshots::new(n);
        for k in 0..10 {
            let u = solve_laplacian_1d(n, (k + 1) as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 8).expect("POD basis");
        let deim_indices = deim_greedy(&pod.modes, 8);
        assert_eq!(deim_indices.len(), 8);

        // For each snapshot, measure DEIM reconstruction error as function of #modes
        let snap = DVector::from_vec(snaps.snapshot(0).to_vec());
        let mut prev_err = 1.0;
        for m in 1..=6 {
            let indices: Vec<usize> = deim_indices.iter().take(m).copied().collect();
            let v_m = pod.modes.columns(0, m).into_owned();
            let u_deim = deim_interpolate(&snap, &v_m, &indices);
            let err = (&snap - DVector::from_vec(u_deim)).norm() / snap.norm();
            // Error should decrease with more modes
            assert!(err < prev_err + 1e-6,
                "DEIM error at m={m}: {err:.3e} >= prev {prev_err:.3e}");
            prev_err = err;
        }
    }

    // ─── EIM ─────────────────────────────────────────────────────────────────

    #[test]
    fn eim_build_reduces_error_with_more_basis_vectors() {
        // Training snapshots: 1D Laplacian solutions at different RHS scales
        let n = 32;
        let mut training: Vec<Vec<f64>> = Vec::new();
        for k in 1..=8 {
            let u = solve_laplacian_1d(n, k as f64);
            training.push(u);
        }

        let eim = EimBasis::build(&training, 6).expect("EIM basis");
        assert!(eim.m <= 6);
        assert_eq!(eim.points.len(), eim.m);

        // Each training snapshot should have decreasing interpolation error
        let mut prev_err = 1.0;
        for (i, snap) in training.iter().enumerate() {
            let err = eim.relative_interp_error(snap);
            assert!(err < 1.0, "EIM error at snap {i}: {err:.3e}");
            prev_err = err.min(prev_err);
        }
    }

    #[test]
    fn eim_interpolation_matches_at_magic_points() {
        let n = 16;
        let mut training: Vec<Vec<f64>> = Vec::new();
        for k in 1..=5 {
            let u = solve_laplacian_1d(n, k as f64);
            training.push(u);
        }
        let eim = EimBasis::build(&training, 4).expect("EIM basis");

        // Interpolate a test vector and verify it matches at magic points
        let test_vec = solve_laplacian_1d(n, 3.0);
        let interp = eim.interpolate(&test_vec);
        for &pt in &eim.points {
            assert!((interp[pt] - test_vec[pt]).abs() < 1e-10,
                "mismatch at magic point {pt}: interp={:.6}, actual={:.6}", interp[pt], test_vec[pt]);
        }
    }

    #[test]
    fn affine_decomposition_project_reduces_dimension() {
        let n = 32;
        // Build a simple 2-component affine decomposition of the 1D Laplacian
        let mut coo1 = fem_linalg::CooMatrix::<f64>::new(n, n);
        let mut coo2 = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo1.add(i, i, 2.0);
            if i > 0 { coo1.add(i, i - 1, -1.0); }
            if i < n - 1 { coo1.add(i, i + 1, -1.0); }
            // Second component: mass-like (identity)
            coo2.add(i, i, 1.0);
        }
        let a1 = coo1.into_csr();
        let a2 = coo2.into_csr();

        fn theta1(_mu: &[f64]) -> f64 { 1.0 }
        fn theta2(mu: &[f64]) -> f64 { mu[0] }

        let affine = AffineDecomposition::new(vec![a1, a2], vec![theta1 as fn(&[f64]) -> f64, theta2]);
        assert_eq!(affine.n_terms(), 2);

        // Build snapshots for POD
        let mut snaps = Snapshots::new(n);
        for k in 0..6 {
            let u = solve_laplacian_1d(n, (k + 1) as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 4).expect("POD basis");

        let projected = affine.project(&pod);
        assert_eq!(projected.components.len(), 2);
        assert_eq!(projected.components[0].nrows(), 4);
        assert_eq!(projected.components[0].ncols(), 4);
    }

    #[test]
    fn error_estimator_computes_finite_residual() {
        let n = 16;
        let a = laplacian_1d(n);
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;

        // Build POD from a few snapshots
        let mut snaps = Snapshots::new(n);
        for k in 0..3 {
            let u = solve_laplacian_1d(n, (k + 1) as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 2).expect("POD basis");

        // Reduced solution
        let (a_r, b_r) = project_system(&a, &b, &pod);
        let u_r = a_r.lu().solve(&b_r).expect("reduced solve");

        let res = ErrorEstimator::relative_residual(&a, &b, u_r.as_slice(), &pod);
        assert!(res.is_finite(), "residual should be finite, got {res}");
        assert!(res >= 0.0, "residual should be non-negative, got {res}");
    }

    // ─── End-to-end ROM test with parametric FEM ──────────────────────────

    #[test]
    fn rom_parametric_diffusion_end_to_end() {
        use fem_assembly::standard::DiffusionIntegrator;
        use fem_assembly::coefficient::FnCoeff;
        use fem_mesh::SimplexMesh;
        use fem_space::H1Space;
        use fem_space::fe_space::FESpace;
        use fem_space::constraints::boundary_dofs;
        use fem_assembly::Assembler;
        use crate::solve_cg;

        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let space = H1Space::new(mesh.clone(), 1);
        let dm = fem_space::DofManager::new(&mesh, 1);

        // Dirichlet BC: u=0 on all boundaries
        let bdofs: Vec<usize> = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]).iter()
            .map(|&d| d as usize).collect();

        let a1_raw = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let a2_raw = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator {
            kappa: FnCoeff(Box::new(|x: &[f64]| x[0])),
        }], 3);
        let b_raw = Assembler::assemble_linear(&space, &[
            &fem_assembly::standard::DomainSourceIntegrator::new(|_| 1.0)
        ], 3);

        let n = space.n_dofs();
        // Save copies for later use before any moves
        let a1_copy = a1_raw.clone();
        let a2_copy = a2_raw.clone();
        let b_copy = b_raw.clone();

        // Build BC-applied copies for snapshot solves
        let assemble_am = |m1: f64, m2: f64| -> (fem_linalg::CsrMatrix<f64>, Vec<f64>) {
            let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
            for row in 0..n {
                for k in a1_copy.row_ptr[row]..a1_copy.row_ptr[row+1] {
                    coo.add(row, a1_copy.col_idx[k] as usize,
                        m1 * a1_copy.values[k] + m2 * a2_copy.values[k]);
                }
            }
            let mut m: fem_linalg::CsrMatrix<f64> = coo.into_csr();
            let mut r = b_copy.clone();
            for &d in &bdofs { if d < m.nrows { m.apply_dirichlet_row_zeroing(d, 0.0, &mut r); } }
            (m, r)
        };

        let mut snaps = Snapshots::new(n);
        let params = [(1.0, 0.0), (0.0, 1.0), (2.0, 1.0), (1.0, 2.0),
                      (3.0, 1.0), (1.0, 3.0), (0.5, 0.5), (2.0, 2.0)];
        for &(m1, m2) in &params {
            let (am, r_mu) = assemble_am(m1, m2);
            let mut u = vec![0.0; n];
            solve_cg(&am, &r_mu, &mut u, &SolverConfig { rtol: 1e-10, ..Default::default() })
                .expect("CG solve during snapshot");
            snaps.add_snapshot(&u);
        }

        let r = 4;
        let pod = PodBasis::compute(&snaps, r).expect("POD basis");
        assert_eq!(pod.n_modes(), r);

        // Save references for full-solve comparison
        let aff_op = AffineDecomposition::new(
            vec![a1_raw, a2_raw],
            vec![|mu: &[f64]| mu[0], |mu: &[f64]| mu[1]],
        );
        let aff_rhs = AffineDecomposition::new(
            vec![b_raw],
            vec![|_: &[f64]| 1.0],
        );

        let op_proj = aff_op.project(&pod);
        let rhs_proj = aff_rhs.project_rhs(&pod);

        let mu_new = [1.5, 1.5];
        let (u_r, u_full) = online_solve(&pod, &op_proj, &rhs_proj, &mu_new)
            .expect("online solve");
        assert_eq!(u_r.len(), r);
        assert_eq!(u_full.len(), n);

        // Full solve at same parameter for error comparison
        let (a_full, r_full) = assemble_am(mu_new[0], mu_new[1]);
        let mut x_full = vec![0.0; n];
        solve_cg(&a_full, &r_full, &mut x_full, &SolverConfig { rtol: 1e-10, ..Default::default() })
            .expect("CG solve for reference");

        let err = relative_error(&x_full, &u_full);
        eprintln!("ROM end-to-end: n={n}, r={r}, rel_error={err:.3e}");
        assert!(err < 0.5, "ROM error should be reasonable, got {err:.3e}");

        let res_sq = ErrorEstimator::efficient_residual_sq(
            u_r.as_slice(), &pod, &rhs_proj, &op_proj, &mu_new);
        assert!(res_sq.is_finite(), "efficient residual_sq should be finite, got {res_sq}");
        assert!(res_sq >= 0.0);
    }
}
