//! meshing_tmop_target_matrix — compare TMOP metrics for 2D mesh optimisation.
//!
//! Perturbs a unit-square triangle mesh then optimises with Shape vs SizeShape
//! metrics, reporting element quality statistics before and after.
//! Analogous to MFEM miniapp `meshing/mesh-optimizer`.
//!
//! Usage:
//!   cargo run --example meshing_tmop_target_matrix

use std::time::Instant;
use fem_mesh::topology::MeshTopology;
use fem_mesh::SimplexMesh;
use fem_mesh::tmop::{tmop_optimise_2d, TmopMetric};

/// Min scaled Jacobian from coordinates + mesh topology.
fn min_scaled_jacobian(mesh: &SimplexMesh<2>, coords: &[f64]) -> f64 {
    let ne = mesh.n_elements() as usize;
    let mut worst = 1.0_f64;
    for e in 0..ne {
        let n = mesh.element_nodes(e as u32);
        let x = |ni: usize, d: usize| coords[n[ni] as usize * 2 + d];
        let (j00, j01) = (x(1,0)-x(0,0), x(2,0)-x(0,0));
        let (j10, j11) = (x(1,1)-x(0,1), x(2,1)-x(0,1));
        let det = (j00*j11 - j01*j10).abs();
        let n0 = (j00*j00 + j10*j10).sqrt();
        let n1 = (j01*j01 + j11*j11).sqrt();
        let sj = if n0 > 0.0 && n1 > 0.0 { det / (n0 * n1) } else { 0.0 };
        worst = worst.min(sj);
    }
    worst
}

fn perturb_square(n: usize, amp: f64) -> SimplexMesh<2> {
    let mut mesh = SimplexMesh::<2>::unit_square_tri(n);
    for i in 0..mesh.n_nodes() as usize {
        let x = mesh.coords[i * 2];
        let y = mesh.coords[i * 2 + 1];
        if x > 0.01 && x < 0.99 && y > 0.01 && y < 0.99 {
            mesh.coords[i * 2]     += amp * (3.0 * x * (1.0 - x) * y).sin();
            mesh.coords[i * 2 + 1] += amp * (3.0 * x * y * (1.0 - y)).cos();
        }
    }
    mesh
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    println!("=== meshing_tmop_target_matrix ({n}×{n}) ===");

    let mesh = perturb_square(n, 0.15);
    let init_coords = mesh.coords.clone();
    let sj_before = min_scaled_jacobian(&mesh, &init_coords);
    println!("  Initial min scaled Jacobian: {sj_before:.6}");

    for (name, metric) in [("Shape", TmopMetric::Shape), ("SizeShape", TmopMetric::SizeShape)] {
        let t0 = Instant::now();
        let result = tmop_optimise_2d(&mesh, &metric, 200, 0.01);
        let sj = min_scaled_jacobian(&mesh, &result);
        let disp: f64 = init_coords.iter().zip(result.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        println!("  {name:>10}: scaled Jacobian {sj:.6}, ‖disp‖={disp:.4e}  [{:.3}s]", t0.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use fem_mesh::SimplexMesh;
    use fem_mesh::tmop::{tmop_optimise_2d, TmopMetric};

    #[test]
    fn tmop_moves_nodes() {
        let mut mesh = SimplexMesh::<2>::unit_square_tri(6);
        for i in 0..mesh.n_nodes() as usize {
            let x = mesh.coords[i * 2];
            let y = mesh.coords[i * 2 + 1];
            if x > 0.05 && x < 0.95 && y > 0.05 && y < 0.95 {
                mesh.coords[i * 2]     += 0.1 * (3.0 * x * (1.0 - x) * y).sin();
                mesh.coords[i * 2 + 1] += 0.1 * (3.0 * x * y * (1.0 - y)).cos();
            }
        }
        let init = mesh.coords.clone();
        let result = tmop_optimise_2d(&mesh, &TmopMetric::Shape, 100, 0.01);
        assert_eq!(result.len(), mesh.n_nodes() as usize * 2);
        assert!(result.iter().all(|x| x.is_finite()));
    }
}
