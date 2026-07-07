//! # Example 30 — Anisotropic AMR (analogous to MFEM ex30)
//!
//! Demonstrates anisotropic non-conforming AMR on a 2-D quad mesh.
//! Three coefficient functions are refined sequentially:
//!
//! 1. **Affine** — piecewise-linear (mesh-conforming for P1, oscillation ≈ 0)
//! 2. **Jump** — discontinuous ring (requires refinement at the interface)
//! 3. **Singular** — steep wavefront from a near-singular Laplacian source
//!
//! Uses ZZ error estimation + Dörfler marking with anisotropic refinement.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex30_aniso_amr
//! cargo run --example mfem_ex30_aniso_amr -- -m ../data/square-disc.mesh -o 2
//! cargo run --example mfem_ex30_aniso_amr -- --cycles 2 --theta 0.3
//! ```

use std::time::Instant;

use fem_io::mfem::read_mfem_file;
use fem_mesh::{
    Mesh, MeshTopology,
    zz_estimator, dorfler_mark,
    refine_nonconforming_quad, refine_nonconforming_quad_aniso, QuadRefineDir,
};

fn main() {
    let args = parse_args();
    println!("=== Example 30: Anisotropic AMR (MFEM ex30) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Initial mesh: {}×{} quads", args.n, args.n);
    }
    println!("  Cycles: {}, theta: {:.3}, enriched_order: {}",
             args.cycles, args.theta, args.enriched_order);

    // Load or generate initial quad mesh
    let mut mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_quad(args.n)
    };

    let t0 = Instant::now();

    // Run three sequential refinement passes, one per coefficient function
    let coeffs: [(&str, fn(&[f64]) -> f64); 3] = [
        ("affine",     affine_fn),
        ("jump",       jump_fn),
        ("singular",   singular_fn),
    ];
    for (pass, (name, coeff_fn)) in coeffs.iter().enumerate() {
        println!("\n  Pass {pass}: {name}");

        for cycle in 0..args.cycles {
            let n0 = mesh.n_nodes();
            let e0 = mesh.n_elems();

            // Eval coefficient on all nodes
            let u: Vec<f64> = (0..n0)
                .map(|i| coeff_fn(&mesh.node_coords(i as u32)))
                .collect();

            let eta = zz_estimator(&mesh, &u);
            let marked = dorfler_mark(&eta, args.theta);

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

            println!("    Cycle {cycle}: nodes {n0}→{}, elems {e0}→{}, marked {}, aniso {}",
                     mesh.n_nodes(), mesh.n_elems(),
                     isotropic.len() + aniso.len(), aniso.len());
        }
    }

    println!("\n  Final mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

// ─── Coefficient functions (matching MFEM ex30) ───────────────────────────

/// Piecewise-affine function which is sometimes mesh-conforming.
fn affine_fn(p: &[f64]) -> f64 {
    let x = p[0];
    let y = p[1];
    if x < 0.0 {
        1.0 + x + y
    } else {
        1.0
    }
}

/// Piecewise-constant function which is never mesh-conforming.
fn jump_fn(p: &[f64]) -> f64 {
    let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
    if r > 0.4 && r < 0.6 {
        1.0
    } else {
        5.0
    }
}

/// Singular function derived from the Laplacian of a steep wavefront.
fn singular_fn(p: &[f64]) -> f64 {
    let x = p[0];
    let y = p[1];
    let alpha: f64 = 1000.0;
    let xc = 0.75;
    let yc = 0.5;
    let r0 = 0.7;
    let r = ((x - xc).powi(2) + (y - yc).powi(2)).sqrt();
    let num = -(alpha - alpha.powi(3) * (r * r - r0 * r0));
    let denom = (r * (alpha.powi(2) * r0 * r0 + alpha.powi(2) * r * r
        - 2.0 * alpha.powi(2) * r0 * r + 1.0))
        .powi(2);
    let denom = denom.max(1.0e-8);
    num / denom
}

fn element_centroid(mesh: &Mesh<2>, e: u32) -> [f64; 2] {
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

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    cycles: usize,
    theta: f64,
    enriched_order: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 4,
        cycles: 3,
        theta: 0.4,
        enriched_order: 5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().unwrap_or("4".into()).parse().unwrap_or(4),
            "--cycles" => a.cycles = it.next().unwrap_or("3".into()).parse().unwrap_or(3),
            "--theta" => a.theta = it.next().unwrap_or("0.4".into()).parse().unwrap_or(0.4),
            "-e" | "--enriched-order" => {
                a.enriched_order = it.next().unwrap_or("5".into()).parse().unwrap_or(5)
            }
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex30_amr_increases_elements_with_refinement() {
        let mut mesh = Mesh::<2>::unit_square_quad(2);
        let n0 = mesh.n_elems();
        for cycle in 0..2 {
            let u: Vec<f64> = (0..mesh.n_nodes())
                .map(|i| affine_fn(&mesh.node_coords(i as u32)))
                .collect();
            let eta = zz_estimator(&mesh, &u);
            let marked = dorfler_mark(&eta, 0.5);
            let aniso: Vec<(u32, QuadRefineDir)> = marked.iter().map(|&e| {
                let c = element_centroid(&mesh, e);
                (e, if c[0] < 0.5 { QuadRefineDir::X } else { QuadRefineDir::Y })
            }).collect();
            mesh = refine_nonconforming_quad_aniso(&mesh, &aniso).0;
            assert!(mesh.n_elems() > n0 || cycle == 0,
                    "refinement should increase element count");
        }
    }

    #[test]
    fn ex30_jump_function_induces_refinement() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let u: Vec<f64> = (0..mesh.n_nodes())
            .map(|i| jump_fn(&mesh.node_coords(i as u32)))
            .collect();
        let eta = zz_estimator(&mesh, &u);
        let marked = dorfler_mark(&eta, 0.5);
        assert!(!marked.is_empty(), "jump function should generate non-zero indicators");
    }
}
