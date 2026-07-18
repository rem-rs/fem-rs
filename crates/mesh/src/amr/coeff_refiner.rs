//! Coefficient-driven mesh refinement (matching MFEM CoefficientRefiner).
//!
//! Iteratively refines elements where a user-specified coefficient varies
//! more than a threshold across the element's nodes.

use crate::Mesh;
use crate::amr::refine_uniform;
use crate::topology::MeshTopology;

/// Refine a 2D mesh until the coefficient variation over each element is
/// below `threshold` or `max_iters` refinements have been performed.
///
/// For each element, the coefficient is evaluated at the element's nodes.
/// If `|max - min| > threshold * |center_value|` (or `> threshold` when
/// center_value ≈ 0), the element is flagged for uniform refinement.
///
/// This matches MFEM's `CoefficientRefiner::PreprocessMesh()` in ex30p.
pub fn refine_by_coefficient(
    mesh: &Mesh<2>,
    coeff: &impl Fn(&[f64]) -> f64,
    threshold: f64,
    max_iters: usize,
) -> Mesh<2> {
    let mut m = mesh.clone();

    for _iter in 0..max_iters {
        let nelems = m.n_elems();
        let mut any_flagged = false;

        for e in 0..nelems as u32 {
            let nodes = m.element_nodes(e);
            let npe = nodes.len();

            // Evaluate coefficient at each node
            let mut vals = Vec::with_capacity(npe);
            for &n in nodes {
                vals.push(coeff(m.node_coords(n)));
            }

            let center_val = vals.iter().sum::<f64>() / npe as f64;
            let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let variation = max_val - min_val;

            let allow = if center_val.abs() > 1e-15 {
                threshold * center_val.abs()
            } else {
                threshold
            };

            if variation > allow {
                any_flagged = true;
                break; // at least one element needs refinement
            }
        }

        if !any_flagged {
            break; // converged
        }

        m = refine_uniform(&m);
    }

    m
}
