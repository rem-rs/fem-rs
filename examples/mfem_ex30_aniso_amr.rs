//! # MFEM Example 30 — Anisotropic AMR
//!
//! Demonstrates anisotropic non-conforming refinement on a 2-D quad mesh
//! using ZZ error estimation + Dörfler marking.  Elements in the left half
//! are split in X, right half in Y.
//!
//! Reference: `mfem/ex30.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex30_aniso_amr [n=4] [cycles=3] [theta=0.4]
//! ```

use std::time::Instant;

use fem_mesh::{
    SimplexMesh, MeshTopology, zz_estimator, dorfler_mark,
    refine_nonconforming_quad, refine_nonconforming_quad_aniso, QuadRefineDir,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize    = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let cycles: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let theta: f64  = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.4);

    println!("=== MFEM Example 30: Anisotropic AMR ===");
    println!("  initial n={n}, cycles={cycles}, θ={theta}");

    let mut mesh = SimplexMesh::<2>::unit_square_quad(n);
    let t0 = Instant::now();

    for cycle in 0..cycles {
        let n0 = mesh.n_nodes();
        let e0 = mesh.n_elems();

        // Synthetic field with strong gradient near (0.25, 0.25)
        let u: Vec<f64> = (0..n0).map(|i| {
            let c = mesh.node_coords(i as u32);
            let dx = c[0] - 0.25_f64;
            let dy = c[1] - 0.25_f64;
            (-40.0_f64 * (dx * dx + dy * dy)).exp()
        }).collect();

        let eta = zz_estimator(&mesh, &u);
        let marked = dorfler_mark(&eta, theta);

        let isotropic: Vec<u32> = marked.iter().filter(|&&e| {
            let c = element_centroid(&mesh, e);
            (c[0] - 0.5).abs() < 0.3 && (c[1] - 0.5).abs() < 0.3
        }).copied().collect();

        let aniso: Vec<(u32, QuadRefineDir)> = marked.iter().filter(|&&e| {
            let c = element_centroid(&mesh, e);
            (c[0] - 0.5).abs() >= 0.3 || (c[1] - 0.5).abs() >= 0.3
        }).map(|&e| {
            let c = element_centroid(&mesh, e);
            (e, if c[0] < 0.5 { QuadRefineDir::X } else { QuadRefineDir::Y })
        }).collect();

        if !isotropic.is_empty() {
            mesh = refine_nonconforming_quad(&mesh, &isotropic).0;
        }
        if !aniso.is_empty() {
            mesh = refine_nonconforming_quad_aniso(&mesh, &aniso).0;
        }

        println!("  Cycle {cycle}: nodes {n0}→{}, elems {e0}→{}, marked {}, aniso {}",
            mesh.n_nodes(), mesh.n_elems(), isotropic.len() + aniso.len(), aniso.len());
    }

    println!("  Final mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

fn element_centroid(mesh: &SimplexMesh<2>, e: u32) -> [f64; 2] {
    let nodes = mesh.elem_nodes(e);
    let npe = nodes.len() as f64;
    let mut c = [0.0_f64; 2];
    for &n in nodes {
        let coord = mesh.node_coords(n);
        c[0] += coord[0];
        c[1] += coord[1];
    }
    [c[0] / npe, c[1] / npe]
}
