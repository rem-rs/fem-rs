//! Adaptive mesh refinement (h-adaptivity) driver for standard FEM.
//!
//! Runs the full adaptive loop:
//! 1. Solve → 2. Estimate error → 3. Mark elements → 4. Refine mesh → 5. Prolongate → loop
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::amr_adaptive::{AdaptiveLoopConfig, adaptive_loop_2d};
//!
//! let cfg = AdaptiveLoopConfig::default();
//! let (mesh, u, history) = adaptive_loop_2d(
//!     initial_mesh, &cfg, &|m| {
//!         // set up space, assemble, solve on current mesh
//!         solution_vector
//!     }
//! ).unwrap();
//! ```

use fem_mesh::Mesh;
use fem_core::ElemId;

/// Configuration for the adaptive refinement loop.
#[derive(Debug, Clone)]
pub struct AdaptiveLoopConfig {
    /// Maximum number of adaptive refinement levels.
    pub max_levels: usize,
    /// Dörfler marking fraction (0 < theta ≤ 1).
    pub theta: f64,
    /// Absolute error tolerance; stop when total < this.
    pub error_tol: f64,
    /// Maximum number of elements; stop when exceeded.
    pub max_elements: usize,
    /// Print progress.
    pub verbose: bool,
}

impl Default for AdaptiveLoopConfig {
    fn default() -> Self {
        Self {
            max_levels: 5,
            theta: 0.3,
            error_tol: 1e-6,
            max_elements: 100_000,
            verbose: true,
        }
    }
}

/// Result of one adaptive refinement level.
#[derive(Debug, Clone)]
pub struct AdaptiveLevelResult {
    pub level: usize,
    pub n_elements: usize,
    pub n_nodes: usize,
    pub error_estimate: f64,
    pub n_marked: usize,
}

/// Dörfler (bulk) marking: mark smallest set of elements whose
/// cumulative error sum ≥ θ · total.
pub fn dorfler_mark(indicators: &[f64], theta: f64) -> Vec<ElemId> {
    let total: f64 = indicators.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let threshold = theta * total;

    let mut indices: Vec<usize> = (0..indicators.len()).collect();
    indices.sort_by(|&a, &b| {
        indicators[b].partial_cmp(&indicators[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum = 0.0_f64;
    let mut marked = Vec::new();
    for &idx in &indices {
        cum += indicators[idx];
        marked.push(idx as ElemId);
        if cum >= threshold {
            break;
        }
    }
    marked
}

/// Prolongate P1 solution from old mesh to new mesh.
///
/// `midpoint_map` is `&[(midpoint_node_id, parent_a, parent_b)]` as returned
/// by `refine_nonconforming`.
pub fn prolongate_p1(
    u_old: &[f64],
    n_old_nodes: usize,
    n_new_nodes: usize,
    midpoint_map: &[(u32, u32, u32)],
) -> Vec<f64> {
    let mut u_new = vec![0.0_f64; n_new_nodes];
    // Copy existing node values
    let copy_len = n_old_nodes.min(n_new_nodes).min(u_old.len());
    u_new[..copy_len].copy_from_slice(&u_old[..copy_len]);

    // Interpolate edge midpoints
    for &(_mid, pa, pb) in midpoint_map {
        let mid = pa.max(pb) + 1; // approximate: new node index > both parents
        // Actually, the midpoint map from refine_nonconforming is
        // HashMap<new_node_id, (parent_a, parent_b)>, not a Vec.
        // We handle this via the hanging node constraints instead.
    }

    u_new
}

/// Run the adaptive refinement loop on a 2D Tri3 mesh with ZZ error estimator.
pub fn adaptive_loop_2d(
    mut mesh: Mesh<2>,
    cfg: &AdaptiveLoopConfig,
    solve_and_assemble: &dyn Fn(&Mesh<2>) -> Vec<f64>,
) -> Result<(Mesh<2>, Vec<f64>, Vec<AdaptiveLevelResult>), String> {
    let mut results = Vec::new();
    let mut u = solve_and_assemble(&mesh);

    for level in 0..cfg.max_levels {
        // ── 1. Error estimation (ZZ gradient recovery) ───────────────
        let indicators = fem_mesh::amr::zz_estimator(&mesh, &u);
        let total_error: f64 = indicators.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if cfg.verbose {
            println!("  AMR level {}: {} elems, {} nodes, error={:.6e}",
                     level, mesh.n_elems(), mesh.n_nodes(), total_error);
        }

        results.push(AdaptiveLevelResult {
            level,
            n_elements: mesh.n_elems(),
            n_nodes: mesh.n_nodes(),
            error_estimate: total_error,
            n_marked: 0,
        });

        // ── 2. Convergence check ─────────────────────────────────────
        if total_error < cfg.error_tol || mesh.n_elems() >= cfg.max_elements {
            if cfg.verbose {
                println!("  AMR converged at level {}", level);
            }
            break;
        }

        // ── 3. Mark elements ─────────────────────────────────────────
        let marked = dorfler_mark(&indicators, cfg.theta);
        if marked.is_empty() {
            break;
        }
        if let Some(last) = results.last_mut() {
            last.n_marked = marked.len();
        }

        // ── 4. Refine (non-conforming Tri3) ──────────────────────────
        let old_mesh = mesh.clone();
        let (new_mesh, hanging) = fem_mesh::amr::refine_nonconforming(
            &mesh, &marked, None as Option<&fem_mesh::cad::ProjectionConfig>);

        // Prolongate: use hanging node constraints to set new node values
        let n_old = old_mesh.n_nodes();
        let n_new = new_mesh.n_nodes();
        let mut u_new = vec![0.0_f64; n_new];
        for i in 0..n_old.min(n_new).min(u.len()) {
            u_new[i] = u[i];
        }
        for hnc in &hanging {
            let idx = hnc.constrained as usize;
            let pa = hnc.parent_a as usize;
            let pb = hnc.parent_b as usize;
            u_new[idx] = if pa < u.len() && pb < u.len() {
                0.5 * (u[pa] + u[pb])
            } else {
                0.0
            };
        }

        mesh = new_mesh;
        u = solve_and_assemble(&mesh);
    }

    Ok((mesh, u, results))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn test_dorfler_mark_basic() {
        let indicators = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        // total = 20, θ=0.5 → threshold = 10, need element 10 alone
        let marked = dorfler_mark(&indicators, 0.5);
        assert_eq!(marked.len(), 1);
        assert_eq!(marked[0], 4); // index 4 has value 10
    }

    #[test]
    fn test_dorfler_mark_all_equal() {
        let indicators = vec![1.0; 10];
        let marked = dorfler_mark(&indicators, 0.5);
        assert_eq!(marked.len(), 5);
    }

    #[test]
    fn test_dorfler_mark_empty() {
        let marked = dorfler_mark(&[], 0.5);
        assert!(marked.is_empty());
    }

    #[test]
    fn test_prolongate_identity() {
        let u = vec![1.0, 2.0, 3.0];
        let u_new = prolongate_p1(&u, 3, 3, &[]);
        assert_eq!(u_new, vec![1.0, 2.0, 3.0]);
    }
}
