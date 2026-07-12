#![allow(non_snake_case)]
//! Bezier extraction operators for isogeometric analysis (IGA).
//!
//! Converts NURBS basis evaluation from per-point Cox-de Boor recursion to
//! **Bernstein evaluation + matrix multiply** by precomputing the extraction
//! operator `C_e` for each knot-span element.
//!
//! Within element `e`, the B-spline basis functions `N_i` are linear combinations
//! of the Bernstein polynomials `B_j` of the same degree:
//!
//!   N(ξ) = C_e^T · B(ξ)
//!
//! # Reference
//! * Borden et al., "Isogeometric analysis with Bézier extraction" (CMAME 2011)

use crate::bernstein::{bernstein_ders, bernstein_vals};
use crate::nurbs::KnotVector;

/// 1-D Bezier extraction data: one `(p+1)×(p+1)` identity matrix per element.
/// For uniform knot vectors (all interior knots distinct), C_e = I since the
/// B-spline basis on each element IS the Bernstein basis.
pub struct BezierExtraction1D {
    pub matrices: Vec<Vec<f64>>,
    pub degree: usize,
    pub n_elements: usize,
}

/// 2-D Bezier extraction data: Kronecker product C_v ⊗ C_u per element.
pub struct BezierExtraction2D {
    pub matrices: Vec<Vec<f64>>,
    pub degree_u: usize,
    pub degree_v: usize,
    pub n_elements_u: usize,
    pub n_elements_v: usize,
    pub n_local: usize,
}

/// Compute 1-D extraction operators (identity for uniform knot vectors).
pub fn compute_extraction_1d(kv: &KnotVector) -> Option<BezierExtraction1D> {
    let p = kv.degree;
    let knots = &kv.knots;
    let n_basis = kv.n_basis();

    let spans: Vec<usize> = (p..=n_basis - 1)
        .filter(|&s| knots[s + 1] > knots[s])
        .collect();
    if spans.is_empty() { return None; }

    let n_elements = spans.len();
    let np1 = p + 1;
    let mut matrices = Vec::with_capacity(n_elements);

    // For uniform knot vectors (all interior knots distinct), C_e = I
    for _ in 0..n_elements {
        let mut Ce = vec![0.0; np1 * np1];
        for i in 0..np1 { Ce[i * np1 + i] = 1.0; }
        matrices.push(Ce);
    }
    Some(BezierExtraction1D { matrices, degree: p, n_elements })
}

/// Solve (p+1) × (p+1) system A·x = b in-place (Gauss elimination with partial pivot).
/// Supports degree ≤ 7.
pub(super) fn solve_small_system(a: &mut [[f64; 8]; 8], b: &mut [f64; 8], n: usize) {
    for col in 0..n {
        let mut best = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[best][col].abs() {
                best = row;
            }
        }
        if best != col {
            a.swap(col, best);
            b.swap(col, best);
        }
        let pivot = a[col][col];
        if pivot.abs() < 1e-300 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * b[j];
        }
        b[i] = sum / a[i][i];
    }
}

/// Compute 1-D extraction operators via point-matching on Chebyshev-Lobatto
/// points. Handles non-uniform knot vectors by solving a (p+1) × (p+1)
/// linear system per element: evaluates the global B-spline and Bernstein
/// bases at Chebyshev-Lobatto points, then solves for the extraction
/// coefficients.
pub fn compute_extraction_1d_full(kv: &KnotVector) -> Option<BezierExtraction1D> {
    let p = kv.degree;
    let knots = &kv.knots;
    let n_basis = kv.n_basis();

    let span_indices: Vec<usize> = (p..=n_basis - 1)
        .filter(|&s| knots[s + 1] > knots[s])
        .collect();
    if span_indices.is_empty() {
        return None;
    }

    let n_elements = span_indices.len();
    let mut matrices = Vec::with_capacity(n_elements);
    let np1 = p + 1;

    // Create a global B-spline evaluator
    let bspline = crate::nurbs::BSplineBasis1D::new(kv.clone());

    // Chebyshev-Lobatto points on [0,1]
    let mut cheb = vec![0.0; np1];
    if p > 0 {
        for m in 0..np1 {
            let angle = m as f64 * std::f64::consts::PI / p as f64;
            cheb[m] = 0.5 * (1.0 - angle.cos());
        }
    }

    // Bernstein basis B_j(xi) = C(p,j) * xi^j * (1-xi)^(p-j)
    let bernstein = |xi: f64, j: usize| -> f64 {
        if j > p {
            return 0.0;
        }
        let k = j.min(p - j);
        let binom: f64 = (1..=k).fold(1.0_f64, |r, jj| r * (p - k + jj) as f64 / jj as f64);
        binom * xi.powi(j as i32) * (1.0 - xi).powi((p - j) as i32)
    };

    // Pre-compute Bernstein matrix A_mat[m][j] = B_j(cheb[m])
    // A_mat is the (p+1) × (p+1) system matrix
    let mut A_mat = vec![0.0_f64; np1 * np1];
    for m in 0..np1 {
        for j in 0..np1 {
            A_mat[m * np1 + j] = bernstein(cheb[m], j);
        }
    }

    for &span in &span_indices {
        let u0 = knots[span];
        let u1 = knots[span + 1];

        // Evaluate global B-spline basis at Chebyshev-Lobatto points
        // mapped to physical coordinates on this element
        let mut N_mat = vec![0.0_f64; np1 * np1];
        for m in 0..np1 {
            let u = u0 + cheb[m] * (u1 - u0);
            let all_vals = bspline.eval(u);
            for i in 0..np1 {
                N_mat[i * np1 + m] = all_vals[span - p + i];
            }
        }

        // Solve for extraction operator:
        // For each B-spline basis i, we have N_i(xi_m) = Σ_j Ce[i][j] * B_j(xi_m)
        // This gives the linear system A_mat · x = n_i where:
        //   x[j] = Ce[i][j],  n_i[m] = N_mat[i][m]
        let mut Ce = vec![0.0_f64; np1 * np1];
        for i in 0..np1 {
            let mut a_loc = [[0.0_f64; 8]; 8];
            let mut b_loc = [0.0_f64; 8];
            for m in 0..np1 {
                for j in 0..np1 {
                    a_loc[m][j] = A_mat[m * np1 + j];
                }
                b_loc[m] = N_mat[i * np1 + m];
            }
            solve_small_system(&mut a_loc, &mut b_loc, np1);
            for j in 0..np1 {
                Ce[i * np1 + j] = b_loc[j];
            }
        }
        matrices.push(Ce);
    }

    Some(BezierExtraction1D {
        matrices,
        degree: p,
        n_elements,
    })
}

/// Compute 2-D extraction operators (tensor-product of 1-D).
pub fn compute_extraction_2d(pd: &super::nurbs::NurbsPatch2DData) -> Option<BezierExtraction2D> {
    let ext_u = compute_extraction_1d_full(&pd.kv_u)?;
    let ext_v = compute_extraction_1d_full(&pd.kv_v)?;

    let p = ext_u.degree; let q = ext_v.degree;
    let np1 = p + 1; let nq1 = q + 1;
    let n_local = np1 * nq1;

    let mut matrices = Vec::with_capacity(ext_u.n_elements * ext_v.n_elements);
    for ev in 0..ext_v.n_elements {
        let Cv = &ext_v.matrices[ev];
        for eu in 0..ext_u.n_elements {
            let Cu = &ext_u.matrices[eu];
            let mut C = vec![0.0; n_local * n_local];
            for iv in 0..nq1 { for iu in 0..np1 {
                for jv in 0..nq1 { for ju in 0..np1 {
                    let row = iv * np1 + iu; let col = jv * np1 + ju;
                    C[row * n_local + col] = Cu[iu * np1 + ju] * Cv[iv * nq1 + jv];
                }}
            }}
            matrices.push(C);
        }
    }
    Some(BezierExtraction2D { matrices, degree_u: p, degree_v: q,
        n_elements_u: ext_u.n_elements, n_elements_v: ext_v.n_elements, n_local })
}

/// Evaluate 2-D Bernstein basis values and parametric gradients at `(xi, eta)`.
pub fn eval_bernstein_2d(p: usize, q: usize, xi: f64, eta: f64,
    phi: &mut [f64], grads: &mut [f64])
{
    let bu = bernstein_vals(p, xi);
    let bv = bernstein_vals(q, eta);
    let du = bernstein_ders(p, xi);
    let dv = bernstein_ders(q, eta);
    let np1 = p + 1; let nq1 = q + 1;
    for j in 0..nq1 { for i in 0..np1 {
        let idx = j * np1 + i;
        phi[idx] = bu[i] * bv[j];
        grads[idx * 2]     = du[i] * bv[j];
        grads[idx * 2 + 1] = bu[i] * dv[j];
    }}
}

/// Apply 2-D extraction: phi_nurbs = C^T · phi_bernstein, grads_nurbs = C^T · grads_bernstein.
pub fn apply_extraction_2d(C: &[f64], n_local: usize,
    phi_b: &[f64], grads_b: &[f64],
    phi_n: &mut [f64], grads_n: &mut [f64])
{
    for i in 0..n_local {
        let (mut s, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for j in 0..n_local {
            let ct = C[i * n_local + j];
            s  += ct * phi_b[j];
            sx += ct * grads_b[j * 2];
            sy += ct * grads_b[j * 2 + 1];
        }
        phi_n[i] = s;
        grads_n[i * 2] = sx;
        grads_n[i * 2 + 1] = sy;
    }
}

// ── 3-D Bezier extraction ──────────────────────────────────────────────────

/// 3-D Bezier extraction data: Kronecker product C_w ⊗ C_v ⊗ C_u per element.
pub struct BezierExtraction3D {
    pub matrices: Vec<Vec<f64>>,
    pub degree_u: usize,
    pub degree_v: usize,
    pub degree_w: usize,
    pub n_elements_u: usize,
    pub n_elements_v: usize,
    pub n_elements_w: usize,
    pub n_local: usize,
}

/// Compute 3-D extraction operators (tensor-product of 1-D).
pub fn compute_extraction_3d(pd: &super::nurbs::NurbsPatch3DData) -> Option<BezierExtraction3D> {
    let ext_u = compute_extraction_1d_full(&pd.kv_u)?;
    let ext_v = compute_extraction_1d_full(&pd.kv_v)?;
    let ext_w = compute_extraction_1d_full(&pd.kv_w)?;

    let p = ext_u.degree; let q = ext_v.degree; let r = ext_w.degree;
    let np1 = p + 1; let nq1 = q + 1; let nr1 = r + 1;
    let n_local = np1 * nq1 * nr1;

    let mut matrices = Vec::with_capacity(ext_u.n_elements * ext_v.n_elements * ext_w.n_elements);
    for ew in 0..ext_w.n_elements {
        let Cw = &ext_w.matrices[ew];
        for ev in 0..ext_v.n_elements {
            let Cv = &ext_v.matrices[ev];
            for eu in 0..ext_u.n_elements {
                let Cu = &ext_u.matrices[eu];
                let mut C = vec![0.0; n_local * n_local];
                for iw in 0..nr1 { for iv in 0..nq1 { for iu in 0..np1 {
                    for jw in 0..nr1 { for jv in 0..nq1 { for ju in 0..np1 {
                        let row = iw * nq1 * np1 + iv * np1 + iu;
                        let col = jw * nq1 * np1 + jv * np1 + ju;
                        C[row * n_local + col] = Cu[iu * np1 + ju]
                                               * Cv[iv * nq1 + jv]
                                               * Cw[iw * nr1 + jw];
                    }}}
                }}}
                matrices.push(C);
            }
        }
    }
    Some(BezierExtraction3D {
        matrices, degree_u: p, degree_v: q, degree_w: r,
        n_elements_u: ext_u.n_elements,
        n_elements_v: ext_v.n_elements,
        n_elements_w: ext_w.n_elements,
        n_local,
    })
}

/// Evaluate 3-D Bernstein basis values and parametric gradients at (xi, eta, zeta).
pub fn eval_bernstein_3d(p: usize, q: usize, r: usize, xi: f64, eta: f64, zeta: f64,
    phi: &mut [f64], grads: &mut [f64])
{
    let bu = bernstein_vals(p, xi);
    let bv = bernstein_vals(q, eta);
    let bw = bernstein_vals(r, zeta);
    let du = bernstein_ders(p, xi);
    let dv = bernstein_ders(q, eta);
    let dw = bernstein_ders(r, zeta);
    let np1 = p + 1; let nq1 = q + 1; let nr1 = r + 1;
    for k in 0..nr1 { for j in 0..nq1 { for i in 0..np1 {
        let idx = k * nq1 * np1 + j * np1 + i;
        phi[idx] = bu[i] * bv[j] * bw[k];
        grads[idx * 3]     = du[i] * bv[j] * bw[k];
        grads[idx * 3 + 1] = bu[i] * dv[j] * bw[k];
        grads[idx * 3 + 2] = bu[i] * bv[j] * dw[k];
    }}}
}

/// Apply 3-D extraction: phi_nurbs = C^T · phi_bernstein, grads_nurbs = C^T · grads_bernstein.
pub fn apply_extraction_3d(C: &[f64], n_local: usize,
    phi_b: &[f64], grads_b: &[f64],
    phi_n: &mut [f64], grads_n: &mut [f64])
{
    for i in 0..n_local {
        let (mut s, mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..n_local {
            let ct = C[i * n_local + j];
            s  += ct * phi_b[j];
            sx += ct * grads_b[j * 3];
            sy += ct * grads_b[j * 3 + 1];
            sz += ct * grads_b[j * 3 + 2];
        }
        phi_n[i] = s;
        grads_n[i * 3] = sx;
        grads_n[i * 3 + 1] = sy;
        grads_n[i * 3 + 2] = sz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::KnotVector;

    #[test]
    fn ext_1d_nonuniform_extraction_matches_bspline_eval() {
        let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0], 2);
        let ext = compute_extraction_1d_full(&kv).unwrap();
        let p = kv.degree;

        // Bernstein basis on [0,1]
        let bernstein = |xi: f64, j: usize, p: usize| -> f64 {
            if j > p { return 0.0; }
            let k = j.min(p - j);
            let binom: f64 = (1..=k).fold(1.0_f64, |r, jj| r * (p - k + jj) as f64 / jj as f64);
            binom * xi.powi(j as i32) * (1.0 - xi).powi((p - j) as i32)
        };

        // Reference B-spline basis via the iga module
        let bspline = crate::iga::BsplineBasis::new(p,
            crate::iga::KnotVector::new_clamped(kv.knots.clone()).unwrap()).unwrap();

        let knots = &kv.knots;
        let spans: Vec<usize> = (p..kv.n_basis()-1).filter(|&s| knots[s+1] > knots[s]).collect();

        for (ei, &span) in spans.iter().enumerate() {
            let u0 = knots[span]; let u1 = knots[span+1];
            let Ce = &ext.matrices[ei];
            for &xi_frac in &[0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0] {
                let u = u0 + xi_frac * (u1 - u0);
                // Bernstein values on [0,1]
                let mut b = vec![0.0; p+1];
                for j in 0..=p { b[j] = bernstein(xi_frac, j, p); }
                // Apply extraction: N = C_e^T · B
                // Ce stores C_e, computed as N_i = Σ_j Ce[i][j] * B_j
                // so n_extract[i] = Σ_j Ce[i][j] * B_j
                let mut n_extract = vec![0.0; p+1];
                for i in 0..=p {
                    for j in 0..=p {
                        n_extract[i] += Ce[i * (p+1) + j] * b[j];
                    }
                }
                // Direct B-spline evaluation
                let n_direct = bspline.nonzero_values(u).unwrap();
                for (idx, val) in n_direct {
                    let local = idx as i64 - span as i64 + p as i64;
                    if local >= 0 && local <= p as i64 {
                        let li = local as usize;
                        assert!((n_extract[li] - val).abs() < 1e-10,
                            "u={u:.4}, basis {li}: extract={:.10e} direct={:.10e}", n_extract[li], val);
                    }
                }
            }
        }
    }

    #[test]
    fn ext_1d_identity() {
        let kv = KnotVector::uniform(1, 2);
        let ext = compute_extraction_1d(&kv).unwrap();
        assert_eq!(ext.n_elements, 2);
        for Ce in &ext.matrices {
            assert!((Ce[0]-1.0).abs()<1e-14 && (Ce[1]-0.0).abs()<1e-14 &&
                    (Ce[2]-0.0).abs()<1e-14 && (Ce[3]-1.0).abs()<1e-14);
        }
    }

    #[test]
    fn ext_2d_identity() {
        let pd = crate::nurbs::NurbsPatch2DData {
            kv_u: KnotVector::uniform(1, 3), kv_v: KnotVector::uniform(1, 2),
            control_pts: vec![[0.0,0.0];12], weights: vec![1.0;12], tag: 1,
        };
        let ext = compute_extraction_2d(&pd).unwrap();
        assert_eq!(ext.n_elements_u, 3);
        assert_eq!(ext.n_elements_v, 2);
        assert_eq!(ext.matrices.len(), 6);
    }

    #[test]
    fn eval_bernstein_2d_partition_unity() {
        let (mut phi, mut g) = (vec![0.0; 4], vec![0.0; 8]);
        eval_bernstein_2d(1, 1, 0.3, 0.7, &mut phi, &mut g);
        assert!((phi.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        let (mut phi2, mut g2) = (vec![0.0; 9], vec![0.0; 18]);
        eval_bernstein_2d(2, 2, 0.5, 0.5, &mut phi2, &mut g2);
        assert!((phi2.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    // ── 3-D tests ────────────────────────────────────────────────────────

    #[test]
    fn ext_3d_basic() {
        let pd = crate::nurbs::NurbsPatch3DData {
            kv_u: KnotVector::uniform(1, 2), kv_v: KnotVector::uniform(1, 2),
            kv_w: KnotVector::uniform(1, 2),
            control_pts: vec![[0.0; 3]; 8], weights: vec![1.0; 8], tag: 1,
        };
        let ext = compute_extraction_3d(&pd).unwrap();
        assert_eq!(ext.matrices.len(), 8); // 2×2×2 elements
        assert_eq!(ext.n_local, 8);
        // Identity for uniform degree 1
        for C in &ext.matrices {
            for i in 0..8 { assert!((C[i * 8 + i] - 1.0).abs() < 1e-14); }
        }
    }

    #[test]
    fn eval_bernstein_3d_partition_unity() {
        let (mut phi, mut g) = (vec![0.0; 8], vec![0.0; 24]);
        eval_bernstein_3d(1, 1, 1, 0.3, 0.7, 0.2, &mut phi, &mut g);
        assert!((phi.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn apply_extraction_3d_identity_recovers_bernstein() {
        let n_local = 8;
        let mut phi_b = vec![0.0; n_local];
        let mut grads_b = vec![0.0; n_local * 3];
        eval_bernstein_3d(1, 1, 1, 0.4, 0.6, 0.3, &mut phi_b, &mut grads_b);
        let mut C = vec![0.0; n_local * n_local];
        for i in 0..n_local { C[i * n_local + i] = 1.0; }
        let mut phi_n = vec![0.0; n_local];
        let mut grads_n = vec![0.0; n_local * 3];
        apply_extraction_3d(&C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n);
        for i in 0..n_local {
            assert!((phi_n[i] - phi_b[i]).abs() < 1e-14);
            assert!((grads_n[i*3] - grads_b[i*3]).abs() < 1e-14);
        }
    }
}
