//! Static condensation (Schur complement elimination) for FEM.
//!
//! Static condensation eliminates **interior** (bubble) degrees of freedom
//! from a linear system, reducing it to a smaller system on the **boundary**
//! (interface) DOFs only.  The interior DOFs are recovered by a cheap
//! back-solve after the condensed system has been solved.
//!
//! # Mathematical background
//!
//! Given a system partitioned into interior (I) and boundary (B) blocks:
//!
//! ```text
//! ┌            ┐ ┌     ┐   ┌     ┐
//! │ K_II  K_IB │ │ u_I │   │ f_I │
//! │ K_BI  K_BB │ │ u_B │ = │ f_B │
//! └            ┘ └     ┘   └     ┘
//! ```
//!
//! The condensed system for boundary DOFs:
//!
//! ```text
//! S u_B = g
//! S = K_BB − K_BI K_II⁻¹ K_IB      (Schur complement)
//! g = f_B  − K_BI K_II⁻¹ f_I
//! ```
//!
//! Back-substitution recovers interior DOFs:
//!
//! ```text
//! u_I = K_II⁻¹ (f_I − K_IB u_B)
//! ```
//!
//! # Usage patterns
//!
//! **Element-level condensation** — use [`StaticCondensation::from_element_matrices`]
//! to condense a single element's local stiffness matrix.  The condensed
//! element matrices are then assembled into the global system as usual.
//!
//! **Global condensation** — use [`condense_global`] to condense a fully
//! assembled global CSR matrix; returns a smaller CSR system plus a
//! [`GlobalBacksolve`] handle for post-processing.
//!
//! # References
//! - Hughes, *The Finite Element Method* (2000), §4.1.
//! - Brezzi & Fortin, *Mixed and Hybrid Finite Element Methods* (1991).

use nalgebra::{DMatrix, DVector};
use fem_linalg::{CooMatrix, CsrMatrix};

/// Element-level static condensation.
///
/// Given an element stiffness matrix `K_e` (size `n × n`) and load vector
/// `f_e` (size `n`), this struct pre-computes the Schur complement matrices
/// so that a condensed element system can be assembled into the global stiffness
/// **without** including the interior DOFs.
///
/// After solving the global condensed system, call [`StaticCondensation::backsolve`]
/// with the boundary-DOF solution to recover the interior DOFs for each element.
#[derive(Debug, Clone)]
pub struct StaticCondensation {
    /// Local indices within the element that are **interior** (to be eliminated).
    pub interior: Vec<usize>,
    /// Local indices within the element that are **boundary** (kept in global system).
    pub boundary: Vec<usize>,

    // ── Precomputed factors ──
    /// K_II⁻¹ K_IB  (n_int × n_bdy)
    k_ii_inv_k_ib: DMatrix<f64>,
    /// K_II⁻¹ f_I   (n_int × 1)
    k_ii_inv_f_i: DVector<f64>,

    /// Condensed element stiffness: K_BB − K_BI K_II⁻¹ K_IB  (n_bdy × n_bdy)
    pub k_condensed: DMatrix<f64>,
    /// Condensed element load: f_B − K_BI K_II⁻¹ f_I  (n_bdy)
    pub f_condensed: DVector<f64>,
}

impl StaticCondensation {
    /// Perform element-level static condensation.
    ///
    /// # Arguments
    /// * `k_e` — element stiffness matrix in row-major order (`ndofs × ndofs`).
    /// * `f_e` — element load vector (`ndofs`).
    /// * `interior` — local DOF indices to eliminate (must be non-empty and disjoint from `boundary`).
    /// * `boundary` — local DOF indices to keep (non-empty).
    ///
    /// # Panics
    /// - If `interior` or `boundary` are empty.
    /// - If any index appears in both `interior` and `boundary`.
    /// - If the combined length of `interior` and `boundary` does not equal `f_e.len()`.
    /// - If `K_II` is numerically singular (LU factorization fails with near-zero pivot).
    pub fn from_element_matrices(
        k_e: &[f64],
        f_e: &[f64],
        interior: &[usize],
        boundary: &[usize],
    ) -> Self {
        let n = f_e.len();
        assert!(n > 0, "element DOF count must be positive");
        assert!(!interior.is_empty(), "interior DOF set must be non-empty");
        assert!(!boundary.is_empty(), "boundary DOF set must be non-empty");
        assert_eq!(
            interior.len() + boundary.len(),
            n,
            "interior + boundary must cover all element DOFs"
        );
        assert_eq!(k_e.len(), n * n, "k_e must be n×n row-major");

        let n_int = interior.len();
        let n_bdy = boundary.len();

        // ── Extract sub-blocks ────────────────────────────────────────────────
        // K_II: n_int × n_int
        let mut k_ii = DMatrix::<f64>::zeros(n_int, n_int);
        for (i, &ri) in interior.iter().enumerate() {
            for (j, &ci) in interior.iter().enumerate() {
                k_ii[(i, j)] = k_e[ri * n + ci];
            }
        }

        // K_IB: n_int × n_bdy
        let mut k_ib = DMatrix::<f64>::zeros(n_int, n_bdy);
        for (i, &ri) in interior.iter().enumerate() {
            for (j, &cj) in boundary.iter().enumerate() {
                k_ib[(i, j)] = k_e[ri * n + cj];
            }
        }

        // K_BI: n_bdy × n_int
        let mut k_bi = DMatrix::<f64>::zeros(n_bdy, n_int);
        for (i, &ri) in boundary.iter().enumerate() {
            for (j, &cj) in interior.iter().enumerate() {
                k_bi[(i, j)] = k_e[ri * n + cj];
            }
        }

        // K_BB: n_bdy × n_bdy
        let mut k_bb = DMatrix::<f64>::zeros(n_bdy, n_bdy);
        for (i, &ri) in boundary.iter().enumerate() {
            for (j, &cj) in boundary.iter().enumerate() {
                k_bb[(i, j)] = k_e[ri * n + cj];
            }
        }

        // f_I and f_B
        let f_i: DVector<f64> = DVector::from_iterator(n_int, interior.iter().map(|&i| f_e[i]));
        let f_b: DVector<f64> = DVector::from_iterator(n_bdy, boundary.iter().map(|&i| f_e[i]));

        // ── Factorise K_II and compute Schur complement ───────────────────────
        // Use LU factorisation (suitable for small dense element blocks).
        let lu = k_ii.lu();

        // K_II⁻¹ K_IB  — solve n_bdy right-hand sides simultaneously.
        let k_ii_inv_k_ib = lu.solve(&k_ib)
            .expect("K_II is singular; cannot perform static condensation");

        // K_II⁻¹ f_I
        let k_ii_inv_f_i = lu.solve(&f_i)
            .expect("K_II is singular; cannot perform static condensation");

        // Schur complement: S = K_BB − K_BI K_II⁻¹ K_IB
        let k_condensed = &k_bb - &k_bi * &k_ii_inv_k_ib;

        // Condensed RHS: g = f_B − K_BI K_II⁻¹ f_I
        let f_condensed = f_b - &k_bi * &k_ii_inv_f_i;

        StaticCondensation {
            interior: interior.to_vec(),
            boundary: boundary.to_vec(),
            k_ii_inv_k_ib,
            k_ii_inv_f_i,
            k_condensed,
            f_condensed,
        }
    }

    /// Recover interior DOF values from the solved boundary DOF values.
    ///
    /// Given `u_B` (the solved values at boundary DOFs, ordered according to
    /// `self.boundary`), returns `u_I` (interior DOF values ordered according
    /// to `self.interior`).
    ///
    /// Formula: `u_I = K_II⁻¹ f_I − K_II⁻¹ K_IB u_B`
    pub fn backsolve(&self, u_b: &[f64]) -> Vec<f64> {
        assert_eq!(
            u_b.len(),
            self.boundary.len(),
            "u_b length must match boundary DOF count"
        );
        let u_bdy = DVector::from_column_slice(u_b);
        // u_I = K_II⁻¹ f_I − (K_II⁻¹ K_IB) u_B
        let u_int = &self.k_ii_inv_f_i - &self.k_ii_inv_k_ib * u_bdy;
        u_int.as_slice().to_vec()
    }

    /// Scatter element DOF values back to a full `ndofs`-length element vector.
    ///
    /// Given the boundary solution `u_b`, computes interior DOFs via [`backsolve`]
    /// and assembles a combined vector `u_e` ordered by the original element DOF
    /// numbering.
    pub fn scatter(&self, u_b: &[f64]) -> Vec<f64> {
        let n = self.interior.len() + self.boundary.len();
        let u_i = self.backsolve(u_b);
        let mut u_e = vec![0.0; n];
        for (k, &idx) in self.interior.iter().enumerate() { u_e[idx] = u_i[k]; }
        for (k, &idx) in self.boundary.iter().enumerate() { u_e[idx] = u_b[k]; }
        u_e
    }

    /// Size of the condensed (boundary) system.
    #[inline]
    pub fn n_boundary(&self) -> usize { self.boundary.len() }

    /// Number of interior DOFs that were eliminated.
    #[inline]
    pub fn n_interior(&self) -> usize { self.interior.len() }
}

// ─── Global static condensation ──────────────────────────────────────────────

/// Handle for recovering the interior DOFs after a global condensation.
///
/// Returned by [`condense_global`] alongside the condensed system.
#[derive(Debug, Clone)]
pub struct GlobalBacksolve {
    /// Global indices of interior DOFs.
    pub interior: Vec<usize>,
    /// Global indices of boundary DOFs (in the condensed system, DOF `k` maps
    /// to global DOF `boundary[k]`).
    pub boundary: Vec<usize>,
    /// K_II block (CSR), needed for back-substitution.
    k_ii: CsrMatrix<f64>,
    /// K_IB block (CSR).
    k_ib: CsrMatrix<f64>,
    /// f_I vector.
    f_i: Vec<f64>,
}

impl GlobalBacksolve {
    /// Recover interior DOF values via iterative solve.
    ///
    /// Given `u_b` (boundary DOF solution in condensed system ordering),
    /// computes `u_I = K_II⁻¹ (f_I − K_IB u_B)` using conjugate gradient with
    /// Jacobi preconditioner.
    ///
    /// Returns `u_i` ordered according to `self.interior`.
    pub fn backsolve(
        &self,
        u_b: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<f64>, String> {
        assert_eq!(u_b.len(), self.boundary.len(), "u_b size mismatch");

        // Compute rhs_i = f_I − K_IB u_B
        let n_int = self.interior.len();
        let mut rhs_i = self.f_i.clone();
        // SpMV: rhs_i -= K_IB * u_b
        for row in 0..n_int {
            let start = self.k_ib.row_ptr[row];
            let end   = self.k_ib.row_ptr[row + 1];
            let mut s = 0.0f64;
            for k in start..end {
                s += self.k_ib.values[k] * u_b[self.k_ib.col_idx[k] as usize];
            }
            rhs_i[row] -= s;
        }

        // Solve K_II u_I = rhs_i using CG with Jacobi preconditioner.
        let diag = self.k_ii.diagonal();
        let mut x = vec![0.0f64; n_int];
        let mut r = rhs_i.clone();
        let mut z = Self::jacobi_apply(&diag, &r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);

        for _iter in 0..max_iter {
            let ap = spmv(&self.k_ii, &p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n_int { x[i] += alpha * p[i]; }
            for i in 0..n_int { r[i] -= alpha * ap[i]; }
            let res_norm = dot(&r, &r).sqrt();
            if res_norm < tol { return Ok(x); }
            z = Self::jacobi_apply(&diag, &r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n_int { p[i] = z[i] + beta * p[i]; }
            rz = rz_new;
        }

        Err(format!(
            "GlobalBacksolve::backsolve: CG did not converge in {} iterations",
            max_iter
        ))
    }

    fn jacobi_apply(diag: &[f64], r: &[f64]) -> Vec<f64> {
        r.iter().zip(diag.iter()).map(|(&ri, &di)| {
            if di.abs() > 1e-300 { ri / di } else { ri }
        }).collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn spmv(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    for row in 0..a.nrows {
        let start = a.row_ptr[row];
        let end   = a.row_ptr[row + 1];
        let mut s = 0.0;
        for k in start..end { s += a.values[k] * x[a.col_idx[k] as usize]; }
        y[row] = s;
    }
    y
}

/// Perform global static condensation of a CSR system.
///
/// Eliminates all `interior_dofs` from the linear system `K u = f`,
/// returning a condensed system `S u_B = g` where the DOF ordering in the
/// condensed system is determined by `boundary_dofs`.
///
/// # Arguments
/// * `k` — global stiffness matrix (must be SPD or at least non-singular on the interior block).
/// * `f` — right-hand side vector.
/// * `interior_dofs` — global DOF indices to eliminate (sorted, no duplicates).
///
/// # Returns
/// `(k_condensed, f_condensed, backsolve_handle)`.
///
/// `k_condensed` is an `n_bdy × n_bdy` CSR matrix; `f_condensed` is the
/// corresponding RHS.  After solving the condensed system, pass the boundary
/// solution to [`GlobalBacksolve::backsolve`] to recover the interior DOFs.
///
/// # Panics
/// Panics if `interior_dofs` is empty or contains out-of-range indices.
pub fn condense_global(
    k: &CsrMatrix<f64>,
    f: &[f64],
    interior_dofs: &[usize],
) -> (CsrMatrix<f64>, Vec<f64>, GlobalBacksolve) {
    let n = k.nrows;
    assert_eq!(n, f.len(), "K and f dimension mismatch");
    assert!(!interior_dofs.is_empty(), "interior_dofs must not be empty");

    // Build complementary boundary set.
    let interior_set: std::collections::HashSet<usize> = interior_dofs.iter().copied().collect();
    let boundary_dofs: Vec<usize> = (0..n).filter(|i| !interior_set.contains(i)).collect();
    let n_int = interior_dofs.len();
    let n_bdy = boundary_dofs.len();

    // Local re-indexing: global DOF → local condensed index.
    let mut int_local  = vec![0usize; n];
    let mut bdy_local  = vec![0usize; n];
    for (k, &i) in interior_dofs.iter().enumerate() { int_local[i] = k; }
    for (k, &i) in boundary_dofs.iter().enumerate() { bdy_local[i] = k; }

    // ── Extract K_II, K_IB, K_BI, K_BB blocks ────────────────────────────────
    let mut coo_ii = CooMatrix::<f64>::new(n_int, n_int);
    let mut coo_ib = CooMatrix::<f64>::new(n_int, n_bdy);
    let mut coo_bi = CooMatrix::<f64>::new(n_bdy, n_int);
    let mut coo_bb = CooMatrix::<f64>::new(n_bdy, n_bdy);

    for row in 0..n {
        let start = k.row_ptr[row];
        let end   = k.row_ptr[row + 1];
        let row_is_int = interior_set.contains(&row);
        for idx in start..end {
            let col = k.col_idx[idx] as usize;
            let val = k.values[idx];
            let col_is_int = interior_set.contains(&col);
            match (row_is_int, col_is_int) {
                (true,  true ) => coo_ii.add(int_local[row], int_local[col], val),
                (true,  false) => coo_ib.add(int_local[row], bdy_local[col], val),
                (false, true ) => coo_bi.add(bdy_local[row], int_local[col], val),
                (false, false) => coo_bb.add(bdy_local[row], bdy_local[col], val),
            }
        }
    }

    let k_ii = coo_ii.into_csr();
    let k_ib = coo_ib.into_csr();
    let k_bi = coo_bi.into_csr();
    let k_bb = coo_bb.into_csr();

    let f_i: Vec<f64> = interior_dofs.iter().map(|&i| f[i]).collect();
    let f_b: Vec<f64> = boundary_dofs.iter().map(|&i| f[i]).collect();

    // ── Compute Schur complement S = K_BB − K_BI K_II⁻¹ K_IB ─────────────────
    // Use CG to apply K_II⁻¹ column-by-column to K_IB (expensive for large n_int;
    // this implementation is designed for moderate interior DOF counts, e.g.,
    // element-level or subdomain-level condensation).
    //
    // For each column j of K_IB:
    //   x_j = K_II⁻¹ K_IB[:,j]   (solve with CG)
    //   S[:,j] += -K_BI x_j
    //   g     +=  (contribution to each boundary DOF)
    //
    // Assemble correction as COO, then add to K_BB.

    let diag_ii = k_ii.diagonal();
    let cg_tol   = 1e-12_f64;
    let cg_iters = 4 * n_int + 200;

    // K_II⁻¹ K_IB column-by-column → result stored as dense n_int × n_bdy.
    // (For large n_int, consider a block CG or a sparse direct solver.)
    let mut k_ii_inv_k_ib = vec![0.0f64; n_int * n_bdy];
    for j in 0..n_bdy {
        // Extract column j of K_IB as a dense vector.
        let mut col_j = vec![0.0; n_int];
        for row in 0..n_int {
            let start = k_ib.row_ptr[row];
            let end   = k_ib.row_ptr[row + 1];
            for kk in start..end {
                if k_ib.col_idx[kk] as usize == j {
                    col_j[row] = k_ib.values[kk];
                    break;
                }
            }
        }
        let x = cg_jacobi_solve(&k_ii, &diag_ii, &col_j, cg_tol, cg_iters);
        for i in 0..n_int { k_ii_inv_k_ib[i * n_bdy + j] = x[i]; }
    }

    // K_II⁻¹ f_I
    let k_ii_inv_f_i = cg_jacobi_solve(&k_ii, &diag_ii, &f_i, cg_tol, cg_iters);

    // Build correction C = K_BI * k_ii_inv_k_ib  (n_bdy × n_bdy, dense for now)
    // and rhs correction d = K_BI * k_ii_inv_f_i  (n_bdy).
    let mut correction = CooMatrix::<f64>::new(n_bdy, n_bdy);
    let mut rhs_correction = vec![0.0f64; n_bdy];

    for i in 0..n_bdy {
        let start = k_bi.row_ptr[i];
        let end   = k_bi.row_ptr[i + 1];
        for kk in start..end {
            let int_col = k_bi.col_idx[kk] as usize;
            let val = k_bi.values[kk];
            // rhs: d[i] += K_BI[i, int_col] * (K_II⁻¹ f_I)[int_col]
            rhs_correction[i] += val * k_ii_inv_f_i[int_col];
            // matrix: C[i, j] += K_BI[i, int_col] * (K_II⁻¹ K_IB)[int_col, j]
            for j in 0..n_bdy {
                let c_ij = val * k_ii_inv_k_ib[int_col * n_bdy + j];
                if c_ij.abs() > 1e-300 {
                    correction.add(i, j, c_ij);
                }
            }
        }
    }

    let correction_csr = correction.into_csr();
    // S = K_BB − correction
    let k_condensed = k_bb.axpby(1.0, &correction_csr, -1.0);

    // g = f_B − rhs_correction
    let f_condensed: Vec<f64> = f_b.iter().zip(rhs_correction.iter())
        .map(|(&fb, &rc)| fb - rc)
        .collect();

    let backsolve = GlobalBacksolve {
        interior: interior_dofs.to_vec(),
        boundary: boundary_dofs,
        k_ii,
        k_ib,
        f_i,
    };

    (k_condensed, f_condensed, backsolve)
}

/// Simple Jacobi-preconditioned CG solver for internal use.
fn cg_jacobi_solve(
    a: &CsrMatrix<f64>,
    diag: &[f64],
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = a.nrows;
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let z: Vec<f64> = r.iter().zip(diag.iter()).map(|(&ri, &di)| {
        if di.abs() > 1e-300 { ri / di } else { ri }
    }).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for _iter in 0..max_iter {
        let ap = spmv(a, &p);
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-300 { break; }
        let alpha = rz / pap;
        for i in 0..n { x[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * ap[i]; }
        if dot(&r, &r).sqrt() < tol { break; }
        let z_new: Vec<f64> = r.iter().zip(diag.iter()).map(|(&ri, &di)| {
            if di.abs() > 1e-300 { ri / di } else { ri }
        }).collect();
        let rz_new = dot(&r, &z_new);
        let beta = rz_new / rz;
        for i in 0..n { p[i] = z_new[i] + beta * p[i]; }
        rz = rz_new;
    }
    x
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small 4×4 SPD system with a known exact solution.
    fn small_system() -> (Vec<f64>, Vec<f64>) {
        // K = tridiagonal(−1, 4, −1) (Poisson-like), 4 DOFs
        #[rustfmt::skip]
        let k = vec![
             4.0, -1.0,  0.0,  0.0,
            -1.0,  4.0, -1.0,  0.0,
             0.0, -1.0,  4.0, -1.0,
             0.0,  0.0, -1.0,  4.0,
        ];
        let f = vec![3.0, 2.0, 2.0, 3.0];
        (k, f)
    }

    #[test]
    fn element_condensation_single_interior_dof() {
        // Condense DOF 1 (interior) from a 3×3 system.
        let k = vec![
            2.0, -1.0,  0.0,
           -1.0,  4.0, -1.0,
            0.0, -1.0,  2.0,
        ];
        let f = vec![1.0, 2.0, 1.0];
        let interior = vec![1]; // DOF 1 is interior
        let boundary = vec![0, 2]; // DOFs 0 and 2 are boundary

        let sc = StaticCondensation::from_element_matrices(&k, &f, &interior, &boundary);

        assert_eq!(sc.n_interior(), 1);
        assert_eq!(sc.n_boundary(), 2);
        assert_eq!(sc.k_condensed.nrows(), 2);
        assert_eq!(sc.f_condensed.len(), 2);

        // The condensed system should give the same boundary solution as the full system.
        // Full solution: K u = f
        // By direct solve: u = K⁻¹ f (use nalgebra)
        let k_mat = DMatrix::from_row_slice(3, 3, &k);
        let f_vec = DVector::from_column_slice(&f);
        let u_full = k_mat.lu().solve(&f_vec).unwrap();

        // Boundary solution from condensed system (also solve with nalgebra)
        let s = sc.k_condensed.clone();
        let g = &sc.f_condensed;
        let u_bdy_from_condensed = s.lu().solve(g).unwrap();

        // Check boundary DOFs match
        for (k, &bdy_idx) in boundary.iter().enumerate() {
            assert!(
                (u_bdy_from_condensed[k] - u_full[bdy_idx]).abs() < 1e-12,
                "boundary DOF {} mismatch: condensed={:.6}, full={:.6}",
                bdy_idx, u_bdy_from_condensed[k], u_full[bdy_idx]
            );
        }

        // Check interior DOF backsolve
        let u_b_slice: Vec<f64> = (0..2).map(|k| u_bdy_from_condensed[k]).collect();
        let u_int = sc.backsolve(&u_b_slice);
        assert_eq!(u_int.len(), 1);
        assert!(
            (u_int[0] - u_full[1]).abs() < 1e-12,
            "interior DOF mismatch: backsolve={:.6}, full={:.6}",
            u_int[0], u_full[1]
        );
    }

    #[test]
    fn element_condensation_scatter_reassembles_full_vector() {
        let k = vec![
            2.0, -1.0,  0.0,
           -1.0,  4.0, -1.0,
            0.0, -1.0,  2.0,
        ];
        let f = vec![1.0, 2.0, 1.0];
        let sc = StaticCondensation::from_element_matrices(&k, &f, &[1], &[0, 2]);

        let k_mat = DMatrix::from_row_slice(3, 3, &k);
        let u_full = k_mat.lu().solve(&DVector::from_column_slice(&f)).unwrap();

        let u_b = vec![u_full[0], u_full[2]];
        let u_e = sc.scatter(&u_b);

        assert_eq!(u_e.len(), 3);
        for i in 0..3 {
            assert!((u_e[i] - u_full[i]).abs() < 1e-12, "DOF {} mismatch", i);
        }
    }

    #[test]
    fn element_condensation_two_interior_dofs() {
        // 4-DOF element; condense interior DOFs {1, 2}, keep boundary {0, 3}.
        let (k, f) = small_system();
        let interior = vec![1, 2];
        let boundary = vec![0, 3];

        let sc = StaticCondensation::from_element_matrices(&k, &f, &interior, &boundary);
        assert_eq!(sc.n_interior(), 2);
        assert_eq!(sc.n_boundary(), 2);

        // Full solution
        let k_mat = DMatrix::from_row_slice(4, 4, &k);
        let u_full = k_mat.lu().solve(&DVector::from_column_slice(&f)).unwrap();

        // Solve condensed
        let u_bdy = sc.k_condensed.clone().lu().solve(&sc.f_condensed).unwrap();

        for (ki, &bi) in boundary.iter().enumerate() {
            assert!((u_bdy[ki] - u_full[bi]).abs() < 1e-11, "bdy DOF {bi} mismatch");
        }

        let u_i = sc.backsolve(u_bdy.as_slice());
        for (ki, &ii) in interior.iter().enumerate() {
            assert!((u_i[ki] - u_full[ii]).abs() < 1e-11, "int DOF {ii} mismatch");
        }
    }

    #[test]
    fn global_condensation_recovers_full_solution() {
        // Build a 6×6 SPD tridiagonal system via COO.
        let n = 6usize;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0);
            if i > 0     { coo.add(i, i-1, -1.0); }
            if i + 1 < n { coo.add(i, i+1, -1.0); }
        }
        let k: CsrMatrix<f64> = coo.into_csr();
        let f: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Reference: full solve using CG.
        let diag = k.diagonal();
        let x_ref = cg_jacobi_solve(&k, &diag, &f, 1e-13, 1000);

        // Condense DOFs 1 and 3 (arbitrary interior).
        let interior_dofs = vec![1, 3];
        let (k_cond, f_cond, bs) = condense_global(&k, &f, &interior_dofs);

        assert_eq!(k_cond.nrows, n - interior_dofs.len());
        assert_eq!(f_cond.len(), n - interior_dofs.len());

        // Solve condensed system.
        let diag_c = k_cond.diagonal();
        let u_b = cg_jacobi_solve(&k_cond, &diag_c, &f_cond, 1e-13, 1000);

        // Backsolve interior
        let u_i = bs.backsolve(&u_b, 1e-13, 1000).unwrap();

        // Assemble full solution vector.
        let mut u_full = vec![0.0; n];
        for (k, &bi) in bs.boundary.iter().enumerate() { u_full[bi] = u_b[k]; }
        for (k, &ii) in bs.interior.iter().enumerate() { u_full[ii] = u_i[k]; }

        for i in 0..n {
            assert!((u_full[i] - x_ref[i]).abs() < 1e-9, "DOF {i} mismatch");
        }
    }
}
