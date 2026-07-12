//! IGA automatic h-adaptivity for 2-D NURBS discretisations.
//!
//! Provides:
//! - [`iga_zz_estimator_2d`] — Zienkiewicz-Zhu gradient-recovery error estimator
//! - [`iga_l2_error_2d`] — L² error computation via quadrature
//! - [`iga_collect_refinement_knots`] — collect midpoints of marked knot spans
//! - [`iga_adaptivity_step_2d`] — one complete estimate–mark–refine cycle

use fem_element::iga::{NurbsKnotVector, NurbsMesh2D, NurbsPatch2D, NurbsPatch2DData};
use fem_element::nurbs::h_refine2d;
use fem_element::quadrature::seg_rule;
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};

use crate::iga::iga_gmg::{build_prolongation_1d_between, build_prolongation_2d};
use crate::postproc::error_estimate::ElementIndicators;

// ─── ZZ error estimator ─────────────────────────────────────────────────────

/// ZZ gradient-recovery error estimator for 2-D NURBS discretisations.
///
/// For each knot span, evaluates the finite element solution gradient
/// σ_h = ∇u_h at the span centre, recovers a smoothed gradient **G** at
/// control points via nodal averaging, then integrates the error indicator:
///
/// ```text
/// η_s² = ∫_span ‖σ_h − G‖² dΩ
/// ```
///
/// The returned [`ElementIndicators`] contains one entry per knot span
/// across all patches, stored in patch-major order (row-major within each
/// patch: u-span outer, v-span inner).
pub fn iga_zz_estimator_2d(
    mesh: &NurbsMesh2D,
    sol: &[f64],
    quad_order: u8,
) -> ElementIndicators {
    let mut indicators = Vec::new();
    let mut dof_offset = 0usize;

    for pd in &mesh.patches {
        let n_dof = pd.control_pts.len();
        let patch = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
        let p_u = pd.kv_u.degree;
        let p_v = pd.kv_v.degree;

        // - 1 - Non-empty knot spans -------------------------------------------------
        let spans_u: Vec<(usize, f64, f64)> = pd.kv_u
            .knots
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
            .collect();
        let spans_v: Vec<(usize, f64, f64)> = pd.kv_v
            .knots
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
            .collect();
        let n_spans_u = spans_u.len();
        let n_spans_v = spans_v.len();
        let n_spans = n_spans_u * n_spans_v;

        // - 2 - Evaluate σ_h = ∇u_h at each span centre ------------------------------
        let mut sigma_h = Vec::with_capacity(n_spans);
        for (eu, (_su, u0, u1)) in spans_u.iter().enumerate() {
            for (_ev, (_sv, v0, v1)) in spans_v.iter().enumerate() {
                let uc = (u0 + u1) / 2.0;
                let vc = (v0 + v1) / 2.0;
                let (grads, _det_j) = crate::iga::physical_grads_2d(pd, &[uc, vc]);

                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for a in 0..n_dof {
                    let u_val = sol[dof_offset + a];
                    sx += u_val * grads[a * 2];
                    sy += u_val * grads[a * 2 + 1];
                }
                sigma_h.push([sx, sy]);
            }
        }

        // - 3 - Nodal averaging: recovered gradient G at control points ---------------
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();
        let mut g_sum = vec![[0.0_f64; 2]; n_dof];
        let mut g_count = vec![0usize; n_dof];

        for (eu, (su, _, _)) in spans_u.iter().enumerate() {
            for (ev, (sv, _, _)) in spans_v.iter().enumerate() {
                // sigma_h is stored in row-major order (eu outer, ev inner)
                let span_idx = eu * n_spans_v + ev;
                let sg = sigma_h[span_idx];
                // Active control points: [su-p_u .. su] × [sv-p_v .. sv]
                let u_start = su.saturating_sub(p_u);
                let v_start = sv.saturating_sub(p_v);
                for j in 0..=p_v {
                    for i in 0..=p_u {
                        let local = (v_start + j) * nu + (u_start + i);
                        if local < n_dof {
                            g_sum[local][0] += sg[0];
                            g_sum[local][1] += sg[1];
                            g_count[local] += 1;
                        }
                    }
                }
            }
        }
        let mut g_recovered = vec![[0.0_f64; 2]; n_dof];
        for a in 0..n_dof {
            if g_count[a] > 0 {
                let inv = 1.0 / g_count[a] as f64;
                g_recovered[a][0] = g_sum[a][0] * inv;
                g_recovered[a][1] = g_sum[a][1] * inv;
            }
        }

        // - 4 - Integrate η² = ∫ ‖σ_h − G‖² dΩ over each span -----------------------
        let seg = seg_rule(quad_order);
        let qpts: Vec<f64> = seg.points.iter().map(|p| p[0]).collect();
        let qwts = seg.weights;

        for (eu, (_su, u0, u1)) in spans_u.iter().enumerate() {
            let hu = u1 - u0;
            for (_ev, (_sv, v0, v1)) in spans_v.iter().enumerate() {
                let hv = v1 - v0;
                let mut eta_sq = 0.0_f64;

                for (&qx, &wx) in qpts.iter().zip(&qwts) {
                    let u = u0 + hu * qx;
                    for (&qy, &wy) in qpts.iter().zip(&qwts) {
                        let v = v0 + hv * qy;
                        let (grads, det_j) = crate::iga::physical_grads_2d(pd, &[u, v]);
                        let w = wx * wy * hu * hv * det_j.abs();

                        // σ_h at this quadrature point
                        let mut sx = 0.0_f64;
                        let mut sy = 0.0_f64;
                        for a in 0..n_dof {
                            let u_val = sol[dof_offset + a];
                            sx += u_val * grads[a * 2];
                            sy += u_val * grads[a * 2 + 1];
                        }

                        // Recovered gradient G via NURBS interpolation of nodal values
                        let mut basis = vec![0.0_f64; n_dof];
                        patch.eval_basis(&[u, v], &mut basis);
                        let mut gx = 0.0_f64;
                        let mut gy = 0.0_f64;
                        for a in 0..n_dof {
                            gx += basis[a] * g_recovered[a][0];
                            gy += basis[a] * g_recovered[a][1];
                        }

                        eta_sq += ((sx - gx).powi(2) + (sy - gy).powi(2)) * w;
                    }
                }
                indicators.push(eta_sq.sqrt());
            }
        }
        dof_offset += n_dof;
    }

    ElementIndicators::new(indicators, "IGA-ZZ")
}

// ─── L² error ────────────────────────────────────────────────────────────────

/// Compute the L² error between a NURBS solution and an exact function.
///
/// Iterates over all patches and knot spans, using tensor-product Gauss
/// quadrature to evaluate:
///
/// ```text
/// ‖u_exact − u_h‖_{L²(Ω)}² = ∫_Ω (u_exact − u_h)² dΩ
/// ```
pub fn iga_l2_error_2d(
    mesh: &NurbsMesh2D,
    sol: &[f64],
    exact: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> f64 {
    let mut err_sq = 0.0_f64;
    let mut dof_offset = 0usize;

    for pd in &mesh.patches {
        let n_dof = pd.control_pts.len();
        let elem = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
        let qr = crate::iga::patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = crate::iga::physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let u_exact = exact(&map.x_phys);

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            let u_h: f64 = basis
                .iter()
                .zip(&sol[dof_offset..dof_offset + n_dof])
                .map(|(r, ui)| r * ui)
                .sum();
            err_sq += (u_exact - u_h).powi(2) * w;
        }
        dof_offset += n_dof;
    }

    err_sq.sqrt()
}

// ─── Collect refinement knots ────────────────────────────────────────────────

/// Collect the **midpoint** knot values for each marked span.
///
/// Each marked span (flat index in row-major order: `eu * n_spans_v + ev`)
/// contributes its `u`-midpoint and `v`-midpoint.  Duplicates are removed.
///
/// Returns `(u_knots, v_knots)`, each sorted and deduplicated.
pub fn iga_collect_refinement_knots(
    pd: &NurbsPatch2DData,
    marked: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    if marked.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let spans_u: Vec<(usize, f64, f64)> = pd
        .kv_u
        .knots
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
        .collect();
    let spans_v: Vec<(usize, f64, f64)> = pd
        .kv_v
        .knots
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
        .collect();
    let n_spans_u = spans_u.len();
    let n_spans_v = spans_v.len();

    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();

    for &s in marked {
        let eu = s / n_spans_v;
        let ev = s % n_spans_v;
        if eu >= n_spans_u || ev >= n_spans_v {
            continue;
        }
        let (_sui, u0, u1) = spans_u[eu];
        let (_svi, v0, v1) = spans_v[ev];
        u_vals.push((u0 + u1) / 2.0);
        v_vals.push((v0 + v1) / 2.0);
    }

    u_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    u_vals.dedup();
    v_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v_vals.dedup();

    (u_vals, v_vals)
}

// ─── Adaptive refinement step ────────────────────────────────────────────────

/// Perform one adaptive h-refinement step for a 2-D NURBS mesh.
///
/// 1. Estimate per-span error with [`iga_zz_estimator_2d`].
/// 2. Mark spans using Dörfler marking with fraction `θ`.
/// 3. If no spans are marked, return `None` (converged).
/// 4. For each patch, collect refinement knots from marked spans and build
///    a refined patch via [`h_refine2d`].
/// 5. Build the prolongation operator from coarse to fine for each patch and
///    project the solution vector.
/// 6. Return the refined mesh and projected solution.
pub fn iga_adaptivity_step_2d(
    mesh: &NurbsMesh2D,
    sol: &[f64],
    quad_order: u8,
    theta: f64,
) -> Option<(NurbsMesh2D, Vec<f64>)> {
    let indicators = iga_zz_estimator_2d(mesh, sol, quad_order);
    let marked = indicators.dorfler_mark(theta);
    if marked.is_empty() {
        return None;
    }

    // ── Compute per-patch span offsets ───────────────────────────────────────
    let n_patches = mesh.patches.len();
    let mut span_offsets = Vec::with_capacity(n_patches);
    let mut acc = 0usize;
    for pd in &mesh.patches {
        let nsu = pd
            .kv_u
            .knots
            .windows(2)
            .filter(|w| w[1] > w[0])
            .count();
        let nsv = pd
            .kv_v
            .knots
            .windows(2)
            .filter(|w| w[1] > w[0])
            .count();
        let n_spans = nsu * nsv;
        span_offsets.push((acc, acc + n_spans));
        acc += n_spans;
    }

    // ── Group marked spans by patch ──────────────────────────────────────────
    let mut patch_marked: Vec<Vec<usize>> = vec![Vec::new(); n_patches];
    for &m in &marked {
        let m = m as usize;
        for (p, &(start, end)) in span_offsets.iter().enumerate() {
            if m >= start && m < end {
                patch_marked[p].push(m - start);
                break;
            }
        }
    }

    // ── Refine each patch and project the solution ───────────────────────────
    let mut refined_patches = Vec::with_capacity(n_patches);
    let mut sol_fine = Vec::new();

    for (pi, pd) in mesh.patches.iter().enumerate() {
        let n_dof_c = pd.control_pts.len();
        let nu_c = pd.kv_u.n_basis();
        let nv_c = pd.kv_v.n_basis();

        let (u_knots, v_knots) = iga_collect_refinement_knots(pd, &patch_marked[pi]);

        // Build refined patch via knot insertion
        let ref_pd = h_refine2d(pd, &u_knots, &v_knots);
        let n_dof_f = ref_pd.control_pts.len();
        let nu_f = ref_pd.kv_u.n_basis();
        let nv_f = ref_pd.kv_v.n_basis();

        refined_patches.push(ref_pd);

        // Build prolongation P: coarse → fine
        let p_u = build_prolongation_1d_between(&pd.kv_u, &refined_patches[pi].kv_u);
        let p_v = build_prolongation_1d_between(&pd.kv_v, &refined_patches[pi].kv_v);
        let p_2d = build_prolongation_2d(&p_u, nu_c, nu_f, &p_v, nv_c, nv_f);

        // Project: sol_fine = P * sol_coarse
        let coarse_chunk = &sol[dof_offset_in_patches(mesh, pi)..dof_offset_in_patches(mesh, pi) + n_dof_c];
        let mut fine_chunk = vec![0.0; n_dof_f];
        p_2d.spmv(coarse_chunk, &mut fine_chunk);
        sol_fine.extend(fine_chunk);
    }

    let refined_mesh = NurbsMesh2D {
        patches: refined_patches,
        edge_connectivity: mesh.edge_connectivity.clone(),
    };

    Some((refined_mesh, sol_fine))
}

/// Compute the DOF offset (start index in the global solution vector) for a
/// given patch.
fn dof_offset_in_patches(mesh: &NurbsMesh2D, patch_idx: usize) -> usize {
    mesh.patches[..patch_idx]
        .iter()
        .map(|p| p.control_pts.len())
        .sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};

    const PI: f64 = std::f64::consts::PI;

    /// Manufactured solution: u = sin(πx)·sin(πy) on [0,1]².
    /// Satisfies u=0 on the full boundary.
    fn exact_solution(x: &[f64]) -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// Source: f = 2π² sin(πx)·sin(πy) = -Δu.
    fn source_function(x: &[f64]) -> f64 {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// Build a degree-2 single-patch mesh on [0,1]² with initial 2×2 elements.
    fn make_refinement_test_mesh() -> NurbsMesh2D {
        let kv = NurbsKnotVector::uniform(2, 2); // degree 2, 2 elements per side
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u;
        let ctrl: Vec<[f64; 2]> = (0..n_dof)
            .map(|idx| {
                let i = idx % n_u;
                let j = idx / n_u;
                [
                    i as f64 / (n_u - 1) as f64,
                    j as f64 / (n_u - 1) as f64,
                ]
            })
            .collect();
        NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof])
    }

    /// Solve Poisson −Δu = f on a single-patch NURBS mesh with homogeneous
    /// Dirichlet BCs on the boundary.
    fn solve_iga_poisson(
        mesh: &NurbsMesh2D,
        f: impl Fn(&[f64]) -> f64,
        quad_order: u8,
    ) -> Vec<f64> {
        let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();

        let mut k = crate::iga::assemble_iga_diffusion_2d(mesh, 1.0, quad_order);
        let mut rhs = crate::iga::assemble_iga_load_2d(mesh, &f, quad_order);

        // Identify boundary DOFs (single patch → offset = 0)
        let nu = mesh.patches[0].kv_u.n_basis();
        let nv = mesh.patches[0].kv_v.n_basis();
        let mut bc_dofs = Vec::new();
        for j in 0..nv {
            for i in 0..nu {
                if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 {
                    bc_dofs.push(j * nu + i);
                }
            }
        }

        // Apply Dirichlet BCs (zeroing rows/cols)
        let bc_set: std::collections::HashSet<usize> = bc_dofs.iter().copied().collect();
        for &d in &bc_dofs {
            if d < n_total {
                for ptr in k.row_ptr[d]..k.row_ptr[d + 1] {
                    k.values[ptr] = if k.col_idx[ptr] as usize == d {
                        1.0
                    } else {
                        0.0
                    };
                }
                rhs[d] = 0.0;
            }
        }
        for i in 0..n_total {
            if bc_set.contains(&i) {
                continue;
            }
            for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                if bc_set.contains(&(k.col_idx[ptr] as usize)) {
                    k.values[ptr] = 0.0;
                }
            }
        }

        // Direct solve (nalgebra LU)
        use nalgebra::{DMatrix, DVector};
        let mut dense = DMatrix::<f64>::zeros(n_total, n_total);
        for i in 0..n_total {
            for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                dense[(i, k.col_idx[ptr] as usize)] = k.values[ptr];
            }
        }
        let b = DVector::from_column_slice(&rhs);
        dense
            .lu()
            .solve(&b)
            .map(|x| x.iter().cloned().collect())
            .unwrap_or_else(|| rhs.clone())
    }

    /// The ZZ estimator returns non‑trivial indicators for a non‑polynomial
    /// solution where the IGA space cannot represent sin(πx)sin(πy) exactly.
    #[test]
    fn test_iga_zz_estimator_nonzero() {
        let mesh = make_refinement_test_mesh();
        let sol = solve_iga_poisson(&mesh, source_function, 4);
        let ind = iga_zz_estimator_2d(&mesh, &sol, 4);
        assert!(
            ind.total_error > 0.0,
            "ZZ estimator should be > 0 for non-exact solution"
        );
        // degree-2 uniform(2,2) → knot vector [0,0,0,0.5,1,1,1] → 2 spans per direction
        assert_eq!(ind.eta.len(), 4, "expected 2×2 = 4 spans");
        assert_eq!(ind.estimator_name, "IGA-ZZ");
    }

    /// Dörfler marking selects at least one span for a non‑trivial solution.
    #[test]
    fn test_iga_dorfler_marks_some() {
        let mesh = make_refinement_test_mesh();
        let sol = solve_iga_poisson(&mesh, source_function, 4);
        let ind = iga_zz_estimator_2d(&mesh, &sol, 4);
        let marked = ind.dorfler_mark(0.5);
        assert!(!marked.is_empty(), "at least one span should be marked");
    }

    /// L² error is finite and non‑zero for a non‑exact solve.
    #[test]
    fn test_iga_l2_error_finite() {
        let mesh = make_refinement_test_mesh();
        let sol = solve_iga_poisson(&mesh, source_function, 4);
        let err = iga_l2_error_2d(&mesh, &sol, exact_solution, 5);
        assert!(err.is_finite(), "L² error must be finite");
        assert!(err > 0.0, "L² error must be non-zero");
    }

    /// After one adaptive refinement + re‑solve, the L² error decreases.
    #[test]
    fn test_iga_adaptivity_reduces_error() {
        let mesh = make_refinement_test_mesh();
        let sol0 = solve_iga_poisson(&mesh, source_function, 4);
        let err0 = iga_l2_error_2d(&mesh, &sol0, exact_solution, 5);

        // One adaptive step – refine the mesh, then re‑solve
        let result = iga_adaptivity_step_2d(&mesh, &sol0, 4, 0.5);
        assert!(result.is_some(), "first adaptivity step should produce refined mesh");
        let (ref_mesh, _ref_sol) = result.unwrap();
        let refined_sol = solve_iga_poisson(&ref_mesh, source_function, 4);
        let err1 = iga_l2_error_2d(&ref_mesh, &refined_sol, exact_solution, 5);

        assert!(
            err1 < err0,
            "adaptive refinement should reduce L² error: {:.6e} → {:.6e}",
            err0,
            err1
        );
    }

    /// Three adaptive cycles continually reduce the L² error.
    #[test]
    fn test_iga_adaptivity_three_cycles() {
        let mesh = make_refinement_test_mesh();
        let mut mesh_adapt = mesh;
        let mut prev_err = f64::INFINITY;

        for cycle in 0..3 {
            let sol = solve_iga_poisson(&mesh_adapt, source_function, 4);
            let err = iga_l2_error_2d(&mesh_adapt, &sol, exact_solution, 5);
            assert!(
                err < prev_err,
                "cycle {cycle}: L² error increased: {:.6e} → {:.6e}",
                prev_err,
                err
            );
            prev_err = err;

            let result = iga_adaptivity_step_2d(&mesh_adapt, &sol, 4, 0.4);
            if let Some((m, _)) = result {
                mesh_adapt = m;
            } else {
                break;
            }
        }
    }
}
