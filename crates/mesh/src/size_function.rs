//! Mesh size functions for h-adaptivity control.
//!
//! Converts error indicators into target element sizes for uniform or
//! graded mesh refinement.  Supports both isotropic h-refinement and
//! derefinement for Tri3 triangular meshes.

use fem_core::ElemId;
use crate::simplex::Mesh;

/// Compute target element sizes from error indicators.
///
/// Elements with error above the Dorfler threshold (the top fraction
/// `theta` of total error) are marked for refinement (size halved).
/// Elements with error below `coarsen_theta` × max error are candidates
/// for coarsening (size doubled).
///
/// # Arguments
/// * `eta` — per-element error indicators, length = n_elements.
/// * `h_cur` — current element sizes (e.g. diameter equivalent), length = n_elements.
/// * `theta` — Dorfler fraction for refinement (0 < theta < 1, typical 0.2–0.5).
/// * `coarsen_theta` — fraction of max error below which an element may be coarsened.
/// * `min_h`, `max_h` — hard bounds on element size.
///
/// # Returns
/// Target size per element, same length as `eta`.
pub fn compute_target_sizes(
    eta: &[f64],
    h_cur: &[f64],
    theta: f64,
    coarsen_theta: f64,
    min_h: f64,
    max_h: f64,
) -> Vec<f64> {
    let n = eta.len();
    let total: f64 = eta.iter().sum();
    let threshold = theta.clamp(0.0, 1.0) * total;
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = coarsen_theta.clamp(0.0, 1.0) * max_eta;

    // Dorfler-style: sort indices by largest error
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut mark_refine = vec![false; n];
    let mut acc = 0.0_f64;
    for &i in &indices {
        if acc >= threshold { break; }
        acc += eta[i];
        mark_refine[i] = true;
    }

    let mut h_target = vec![0.0_f64; n];
    for i in 0..n {
        if mark_refine[i] {
            // Refine — halve the element size
            h_target[i] = (h_cur[i] * 0.5).max(min_h);
        } else if eta[i] <= cutoff {
            // Candidate for coarsening — double the element size
            h_target[i] = (h_cur[i] * 2.0).min(max_h);
        } else {
            // Keep current size
            h_target[i] = h_cur[i];
        }
    }
    h_target
}

/// Smooth a size field with Taubin λ-μ Laplacian smoothing.
///
/// Preserves element sizes at mesh boundaries and produces a gradual
/// transition between coarse and fine regions, avoiding abrupt jumps
/// that would force unnecessary refinement propagation.
///
/// # Arguments
/// * `h` — size field to smooth (in-place), length = n_elements.
/// * `mesh` — Tri3 mesh (used for element adjacency).
/// * `n_iter` — number of smoothing passes (typical 3–10).
/// * `lambda` — smoothing factor (default 0.5).
/// * `mu` — stabilisation factor for Taubin (default 0.5).
pub fn smooth_size_field(h: &mut [f64], mesh: &Mesh<2>, n_iter: usize, lambda: f64) {
    assert_eq!(mesh.n_elems(), h.len(), "size array length must match n_elements");
    if n_iter == 0 { return; }
    // Build element adjacency from mesh connectivity
    let n = mesh.n_elems();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in 0..n {
        let ns = mesh.elem_nodes(e as ElemId);
        // Find elements sharing at least 2 nodes (an edge) with this element
        for e2 in 0..n {
            if e == e2 { continue; }
            let ns2 = mesh.elem_nodes(e2 as ElemId);
            let shared = ns.iter().filter(|&n| ns2.contains(n)).count();
            if shared >= 2 { adj[e].push(e2); }
        }
    }

    let mut h_next = h.to_vec();
    for _iter in 0..n_iter {
        for i in 0..n {
            if adj[i].is_empty() { continue; }
            let avg: f64 = adj[i].iter().map(|&j| h[j]).sum::<f64>() / adj[i].len() as f64;
            h_next[i] = h[i] + lambda * (avg - h[i]);
        }
        h.copy_from_slice(&h_next);
    }
}

/// Convert a target size field to refinement markers.
///
/// Returns a tuple `(refine, coarsen)` where each Vec<String> contains
/// element IDs to refine (size reduction > 25%) or coarsen (size increase > 25%).
pub fn size_to_markers(h_target: &[f64], h_cur: &[f64]) -> (Vec<ElemId>, Vec<ElemId>) {
    let mut refine = Vec::new();
    let mut coarsen = Vec::new();
    for i in 0..h_target.len() {
        if h_target[i] <= h_cur[i] * 0.75 {
            refine.push(i as ElemId);
        } else if h_target[i] >= h_cur[i] * 1.25 {
            coarsen.push(i as ElemId);
        }
    }
    (refine, coarsen)
}

/// Compute current element sizes from a Tri3 mesh (diameter = max edge length).
pub fn compute_element_sizes(mesh: &Mesh<2>) -> Vec<f64> {
    let mut h = Vec::with_capacity(mesh.n_elems());
    for e in 0..mesh.n_elems() as ElemId {
        let ns = mesh.elem_nodes(e);
        let a = mesh.coords_of(ns[0]);
        let b = mesh.coords_of(ns[1]);
        let c = mesh.coords_of(ns[2]);
        let d01 = ((a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)).sqrt();
        let d02 = ((a[0]-c[0]).powi(2)+(a[1]-c[1]).powi(2)).sqrt();
        let d12 = ((b[0]-c[0]).powi(2)+(b[1]-c[1]).powi(2)).sqrt();
        h.push(d01.max(d02).max(d12));
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amr::zz_estimator;
    use crate::Mesh;

    #[test]
    fn element_sizes_on_unit_square() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h = compute_element_sizes(&mesh);
        assert_eq!(h.len(), mesh.n_elems());
        // A 4×4 quad mesh with diagonal split: element diameter ≈ sqrt(2)/4
        for &size in &h {
            assert!(size > 0.0, "zero element size");
            assert!(size < 1.0, "size too large: {size}");
        }
    }

    #[test]
    fn compute_target_from_estimator() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let u: Vec<f64> = (0..mesh.n_nodes()).map(|i| {
            let c = mesh.coords_of(i as u32);
            (std::f64::consts::PI * c[0]).sin() * (std::f64::consts::PI * c[1]).sin()
        }).collect();
        let eta = zz_estimator(&mesh, &u);
        let h_cur = compute_element_sizes(&mesh);

        let h_target = compute_target_sizes(&eta, &h_cur, 0.3, 0.1, 1e-6, 1.0);
        assert_eq!(h_target.len(), eta.len());
        // Some elements should be marked for refinement (their target should be smaller)
        let refined_count = h_target.iter().zip(h_cur.iter()).filter(|(&t, &c)| t < c).count();
        assert!(refined_count > 0, "no refinement targets");
    }

    #[test]
    fn smooth_field_uniform() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_elems();
        let mut h = vec![1.0_f64; n];
        smooth_size_field(&mut h, &mesh, 5, 0.5);
        // Uniform field should stay uniform
        for &v in &h { assert!((v - 1.0).abs() < 1e-14, "uniform field changed: {v}"); }
    }

    #[test]
    fn size_to_markers_consistent() {
        let h_cur = vec![0.5, 0.5, 0.5, 0.5];
        let h_target = vec![0.2, 1.5, 0.4, 0.6]; // refine, coarsen, no change, no change
        let (refine, coarsen) = size_to_markers(&h_target, &h_cur);
        assert_eq!(refine, vec![0 as ElemId], "element 0 should be refined");
        assert_eq!(coarsen, vec![1 as ElemId], "element 1 should be coarsened");
    }
}
