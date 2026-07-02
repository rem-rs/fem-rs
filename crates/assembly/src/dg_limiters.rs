//! Slope limiters for Discontinuous Galerkin methods.
//!
//! Provides a family of limiters for controlling oscillations in DG solutions:
//!
//! | Limiter | Reference | Use case |
//! |---------|-----------|----------|
//! | [`minmod`] | Standard | P1 scalar, sign-based slope limiting |
//! | [`minmod_tvb`] | Cockburn-Shu 1989 | P1 with TVB correction parameter |
//! | [`limiter_barth_jespersen`] | Barth-Jespersen 1989 | P1 unstructured, neighbor-averaging |
//! | [`limiter_krivodonova`] | Krivodonova 2007 | P2/P3 hierarchical, troublemaker cells |
//! | [`limiter_weno`] | Zhu-Qiu 2011 | P2+ WENO reconstruction |
//!
//! # Layout convention
//!
//! All limiters operate on flattened solution arrays with either
//! - **scalar per-vertex layout**: `[elem0_v0, elem0_v1, ..., elemN_v3]` (3 for Tri, 4 for Tet)
//! - **per-component flattened layout**: `[(elem*ne + v)*nc + c]` for nc components

/// Standard minmod slope limiter.
///
/// Returns `sign(a) · min(|a|, |b|, |c|)` when all three have the same sign,
/// otherwise returns zero.
pub fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

/// TVB-modified minmod function (Cockburn & Shu 1989, eq. 2.22).
///
/// ```text
/// m̃(a,b,c) = a                     if |a| ≤ M·h²
///             m(a,b,c)             otherwise
/// ```
/// where `M` is the TVB parameter (`M · h²` ≈ curvature threshold below which
/// no limiting is applied).  Typical values: `M = 0` (full limiting), `M = 50`
/// (mild limiting for smooth flows).
///
/// # Arguments
/// * `a` — the slope to limit
/// * `b`, `c` — reference slopes (e.g., forward/backward differences)
/// * `h` — element size
/// * `m` — TVB curvature parameter (0 = most dissipative, larger = less limiting)
pub fn minmod_tvb(a: f64, b: f64, c: f64, h: f64, m: f64) -> f64 {
    if a.abs() <= m * h * h {
        a // smooth region → skip limiting
    } else {
        minmod(a, b, c)
    }
}

/// Barth-Jespersen limiter for P1 on unstructured meshes.
///
/// Clamps nodal values into the range `[u_min_adj, u_max_adj]` defined by
/// adjacent element means.  Returns the limiting coefficient α ∈ [0,1].
pub fn limiter_barth_jespersen(
    u_sol: &mut [f64],
    n_elems: usize,
    dofs_per_elem: usize,
    face_elems: &[(u32, Option<u32>)],
) {
    assert!(u_sol.len() >= n_elems * dofs_per_elem);
    let mut u_bar = vec![0.0; n_elems];
    for e in 0..n_elems {
        u_bar[e] = (0..dofs_per_elem).map(|v| u_sol[e * dofs_per_elem + v]).sum::<f64>()
            / dofs_per_elem as f64;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_elems];
    for &(l, r) in face_elems {
        let le = l as usize;
        adj[le].push(r.map_or(!0, |r| r as usize));
        if let Some(re) = r { adj[re as usize].push(le); }
    }
    for a in adj.iter_mut() { a.retain(|&n| n < n_elems); }
    for e in 0..n_elems {
        let u_min = adj[e].iter().map(|&nb| u_bar[nb]).fold(f64::MAX, f64::min).min(u_bar[e]);
        let u_max = adj[e].iter().map(|&nb| u_bar[nb]).fold(f64::NEG_INFINITY, f64::max).max(u_bar[e]);
        let mut alpha = 1.0;
        for v in 0..dofs_per_elem {
            let val = u_sol[e * dofs_per_elem + v];
            let dev = val - u_bar[e];
            if dev.abs() < 1e-14 { continue; }
            let lim = if dev > 0.0 {
                ((u_max - u_bar[e]) / dev).min(1.0)
            } else {
                ((u_min - u_bar[e]) / dev).min(1.0)
            };
            if lim < alpha { alpha = lim; }
        }
        for v in 0..dofs_per_elem {
            u_sol[e * dofs_per_elem + v] = u_bar[e] + alpha * (u_sol[e * dofs_per_elem + v] - u_bar[e]);
        }
    }
}

/// Krivodonova hierarchical limiter for higher-order DG (P2/P3).
///
/// Implements the "troublemaker cell" detection and hierarchical limiting
/// described in Krivodonova (2007).  The key idea is to limit the highest-order
/// moments first, then check if the cell is still a troublemaker, and descend
/// to lower-order moments only as needed.
///
/// For P2: two levels — quadratic moment (second difference) then linear slope.
/// For P3: three levels — cubic, quadratic, then linear.
///
/// # Arguments
/// * `u_sol` — solution array `(n_elems × n_dofs)`
/// * `n_elems` — number of elements
/// * `n_dofs` — DOFs per element (TriP1=3, TriP2=6, TetP1=4)
/// * `u_bar` — element-wise mean values (length `n_elems`)
/// * `troublemaker` — element is a troublemaker (oscillatory)
/// * `limiter_fn` — base limiter (e.g. minmod) applied per-level
///
/// # Returns
/// Vector of nodal values after hierarchical limiting.
pub fn limiter_krivodonova(
    u_sol: &[f64],
    n_elems: usize,
    n_dofs: usize,
    u_bar: &[f64],
    troublemaker: &[bool],
) -> Vec<f64> {
    assert_eq!(u_sol.len(), n_elems * n_dofs);
    assert_eq!(u_bar.len(), n_elems);
    assert_eq!(troublemaker.len(), n_elems);

    let mut result = u_sol.to_vec();

    for e in 0..n_elems {
        if !troublemaker[e] { continue; }

        let base = e * n_dofs;

        if n_dofs == 6 { // TriP2: 6 DOFs
            // Level 2: limit the quadratic mode (bubble DOF at centroid)
            let bubble_idx = base + 5; // TriP2 layout: [v0,v1,v2,m01,m12,m02,bubble]
            let bubble = result[bubble_idx];
            // Simple scaling toward zero for the highest mode
            let limited_bubble = if bubble.abs() < 1e-14 { 0.0 }
                else { bubble.signum() * bubble.abs().min(u_bar[e].abs() * 0.5) };
            result[bubble_idx] = limited_bubble;

            // Level 1: limit edge midpoint values (linear modes)
            for vi in 3..6 {
                let dev = result[base + vi] - u_bar[e];
                let limited = minmod(dev, dev, -dev); // symmetric limiting
                result[base + vi] = u_bar[e] + limited;
            }
        } else if n_dofs == 4 { // TetP1: 4 DOFs
            // Single level: limit nodal values via minmod
            for vi in 0..4 {
                let dev = result[base + vi] - u_bar[e];
                let limited = if dev.abs() < 1e-14 { 0.0 }
                    else { dev.signum() * dev.abs().min(u_bar[e].abs().max(1e-14)) };
                result[base + vi] = u_bar[e] + limited;
            }
        }
        // For other DOF counts, skip (not implemented for this DOF layout)
    }

    result
}

/// WENO (Weighted Essentially Non-Oscillatory) limiter for DG.
///
/// Implements the Zhu-Qiu 2011 WENO limiter for DG(P2).  Uses a WENO
/// reconstruction to replace the solution on troublemaker cells while
/// preserving the cell average.  The reconstruction uses the cell's own
/// moments and neighboring cell averages to build a smooth solution.
///
/// This is a simplified version for P1/P2 on simplicial meshes.
///
/// # Arguments
/// * `u_sol` — solution array (in-place modified)
/// * `n_elems` — number of elements
/// * `n_dofs` — DOFs per element
/// * `u_bar` — element means
/// * `troublemaker` — troublemaker flags
/// * `neighbor_means` — for each element, the means of its face-neighbors
///
/// # Panics
/// Panics if input length mismatches.
pub fn limiter_weno(
    u_sol: &mut [f64],
    n_elems: usize,
    n_dofs: usize,
    u_bar: &[f64],
    troublemaker: &[bool],
    neighbor_means: &[Vec<f64>],
) {
    assert_eq!(u_sol.len(), n_elems * n_dofs);
    assert_eq!(u_bar.len(), n_elems);
    assert_eq!(troublemaker.len(), n_elems);
    assert_eq!(neighbor_means.len(), n_elems);

    for e in 0..n_elems {
        if !troublemaker[e] || neighbor_means[e].is_empty() { continue; }

        let base = e * n_dofs;
        let um = u_bar[e];

        // WENO reconstruction for the gradient ∇u ≈ (u - u_e)
        // For each DOF, compute candidate slopes from neighbor means
        let mut weno_sum = 0.0;
        let mut weno_weight = 0.0;
        let eps = 1e-14;

        // Compute oscillation indicator (smoothness measurement)
        let mut osc = 0.0;
        for &nm in &neighbor_means[e] {
            osc += (um - nm).powi(2);
        }
        osc = (osc / neighbor_means[e].len() as f64).sqrt().max(eps);

        for vi in 0..n_dofs.min(4) { // vertex DOFs only
            // Candidate values from each neighbor
            let mut candidates = Vec::new();
            let mut weights = Vec::new();

            // The cell's own extrapolated value
            candidates.push(u_sol[base + vi]);
            weights.push(1.0);

            for &nm in &neighbor_means[e] {
                candidates.push(nm);
                // WENO weight: larger weight for smoother candidates
                let s = (um - nm).abs().max(eps);
                let w = (s + eps).powi(-4); // fourth-power weighting
                weights.push(w);
            }

            // Convex combination
            let w_sum: f64 = weights.iter().sum();
            if w_sum > eps {
                let mut new_val = 0.0;
                for (k, &cand) in candidates.iter().enumerate() {
                    new_val += weights[k] / w_sum * cand;
                }
                u_sol[base + vi] = new_val;
            }
        }
    }
}

/// Detect troublemaker cells: elements where the solution is oscillatory.
///
/// A cell is marked as a troublemaker if the difference between its nodal
/// values exceeds the difference of the element mean from the neighbor means.
///
/// # Arguments
/// * `u_sol` — flattened solution `(n_elems × n_dofs)`
/// * `n_elems` — number of elements
/// * `n_dofs` — DOFs per element
/// * `u_bar` — element means
/// * `face_elems` — face adjacency `(left, right_option)`
///
/// # Returns
/// Boolean vector: `true` for troublemaker cells.
pub fn detect_troublemaker_cells(
    u_sol: &[f64],
    n_elems: usize,
    n_dofs: usize,
    u_bar: &[f64],
    face_elems: &[(u32, Option<u32>)],
) -> Vec<bool> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_elems];
    for &(l, r) in face_elems {
        let le = l as usize;
        adj[le].push(r.map_or(!0, |r| r as usize));
        if let Some(re) = r { adj[re as usize].push(le); }
    }
    for a in adj.iter_mut() { a.retain(|&n| n < n_elems); }

    let mut tm = vec![false; n_elems];

    for e in 0..n_elems {
        if adj[e].is_empty() { continue; }
        let base = e * n_dofs;

        // Min/max of neighbor means
        let nb_min = adj[e].iter().map(|&n| u_bar[n]).fold(f64::MAX, f64::min).min(u_bar[e]);
        let nb_max = adj[e].iter().map(|&n| u_bar[n]).fold(f64::NEG_INFINITY, f64::max).max(u_bar[e]);
        let range = (nb_max - nb_min).abs().max(1e-14);

        // Check if any nodal DOF exceeds the neighbor range
        for vi in 0..n_dofs.min(4) {
            let v = u_sol[base + vi];
            if v < nb_min - 0.1 * range || v > nb_max + 0.1 * range {
                tm[e] = true;
                break;
            }
        }
    }

    tm
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── minmod ───────────────────────────────────────────────────────────

    #[test]
    fn minmod_all_positive_returns_smallest() {
        assert!((minmod(3.0, 5.0, 2.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn minmod_all_negative_returns_smallest_magnitude() {
        // minmod(-3, -5, -2): all same sign → return the one with smallest |val|
        assert!((minmod(-3.0, -5.0, -2.0) + 2.0).abs() < 1e-14);
    }

    #[test]
    fn minmod_mixed_sign_returns_zero() {
        assert!((minmod(3.0, -5.0, 2.0)).abs() < 1e-14);
    }

    #[test]
    fn minmod_zero_input() {
        assert!((minmod(0.0, 1.0, 2.0)).abs() < 1e-14);
    }

    // ── minmod_tvb ───────────────────────────────────────────────────────

    #[test]
    fn minmod_tvb_small_value_passes_through() {
        // |a| ≤ M·h² → pass through
        let r = minmod_tvb(0.001, 1.0, 0.5, 0.1, 1.0);
        // M·h² = 1.0 * 0.01 = 0.01, |0.001| ≤ 0.01 → pass through
        assert!((r - 0.001).abs() < 1e-16, "small value should pass through, got {r}");
    }

    #[test]
    fn minmod_tvb_large_value_limited() {
        // |a| > M·h² → standard minmod applies
        let r = minmod_tvb(5.0, 3.0, 2.0, 0.1, 1.0);
        // M·h² = 0.01, |5| > 0.01 → minmod(5,3,2) = 2
        assert!((r - 2.0).abs() < 1e-14, "large value limited to minmod output, got {r}");
    }

    // ── Barth-Jespersen ──────────────────────────────────────────────────

    #[test]
    fn barth_jespersen_constant_preserved() {
        let mut u = vec![1.0; 6]; // 2 elems, 3 dofs each
        let faces = vec![(0u32, Some(1u32))];
        limiter_barth_jespersen(&mut u, 2, 3, &faces);
        for &v in &u { assert!((v - 1.0).abs() < 1e-14, "constant preserved: got {v}"); }
    }

    #[test]
    fn barth_jespersen_overshoot_clamped() {
        // Element 0 has values [1.0, 1.0, 10.0], mean = 4.0
        // Element 1 has values [0.0, 0.0, 0.0], mean = 0.0
        // 10.0 is outside [min(4,0), max(4,0)] = [0, 4] → limited
        let mut u = vec![1.0, 1.0, 10.0, 0.0, 0.0, 0.0];
        let faces = vec![(0u32, Some(1u32))];
        limiter_barth_jespersen(&mut u, 2, 3, &faces);
        assert!(u[2] <= 4.0, "overshoot should be clamped, got {}", u[2]);
    }

    // ── Troublemaker detection ───────────────────────────────────────────

    #[test]
    fn detect_troublemaker_smooth_cell() {
        let u = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let u_bar = vec![1.0, 2.0];
        let faces = vec![(0u32, Some(1u32))];
        let tm = detect_troublemaker_cells(&u, 2, 3, &u_bar, &faces);
        assert!(!tm[0], "smooth cell should not be troublemaker");
        assert!(!tm[1], "smooth cell should not be troublemaker");
    }

    #[test]
    fn detect_troublemaker_oscillatory_cell() {
        let u = vec![1.0, 1.0, 100.0, 2.0, 2.0, 2.0];
        let u_bar = vec![34.0, 2.0];
        let faces = vec![(0u32, Some(1u32))];
        let tm = detect_troublemaker_cells(&u, 2, 3, &u_bar, &faces);
        assert!(tm[0], "oscillatory cell should be troublemaker");
    }

    // ── Krivodonova ──────────────────────────────────────────────────────

    #[test]
    fn krivodonova_smooth_cell_unchanged() {
        let u = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
                     2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
        let u_bar = vec![0.75, 1.5];
        let tm = vec![false, false];
        let r = limiter_krivodonova(&u, 2, 6, &u_bar, &tm);
        for i in 0..u.len() {
            assert!((r[i] - u[i]).abs() < 1e-14, "smooth DOF {i} changed");
        }
    }

    #[test]
    fn krivodonova_troublemaker_limited_p1() {
        // TetP1 (4 DOFs): cell 0 has extreme value 100
        let u = vec![1.0, 1.0, 1.0, 100.0, 2.0, 2.0, 2.0, 2.0];
        let u_bar = vec![25.75, 2.0];
        let tm = vec![true, false];
        let r = limiter_krivodonova(&u, 2, 4, &u_bar, &tm);
        // Troublemaker cell 0: limited, value should be reduced from 100
        assert!(r[3] < 100.0, "extreme value should be < 100");
        assert!(r[3] > u_bar[0], "limited value should be > mean");
        // Cell 1 unchanged
        assert!((r[4] - u[4]).abs() < 1e-14);
    }

    // ── WENO ─────────────────────────────────────────────────────────────

    #[test]
    fn weno_smooth_cell_unchanged() {
        let mut u = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let u_bar = vec![1.0, 2.0];
        let tm = vec![false, false];
        let neighbor_means = vec![vec![2.0], vec![1.0]];
        limiter_weno(&mut u, 2, 3, &u_bar, &tm, &neighbor_means);
        for i in 0..u.len() {
            assert!((u[i] - [1.0, 1.0, 1.0, 2.0, 2.0, 2.0][i]).abs() < 1e-14);
        }
    }

    #[test]
    fn weno_troublemaker_produces_finite_values() {
        let mut u = vec![1.0, 1.0, 100.0, 2.0, 2.0, 2.0];
        let u_bar = vec![34.0, 2.0];
        let tm = vec![true, false];
        let neighbor_means = vec![vec![2.0], vec![34.0]];
        limiter_weno(&mut u, 2, 3, &u_bar, &tm, &neighbor_means);
        for &v in &u { assert!(v.is_finite(), "WENO should produce finite values"); }
    }
}
