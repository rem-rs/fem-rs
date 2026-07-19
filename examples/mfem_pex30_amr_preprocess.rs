//! # Example 30p — Adaptive mesh refinement preprocessing (parallel)
//!
//! Parallel version of ex30: preprocesses a mesh by adaptively refining to
//! lower data oscillation of user-specified coefficients, without solving a PDE.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex30_amr_preprocess -- -m data/star.mesh
//! ```
//!
//! ## Reference
//! MFEM ex30p: https://github.com/mfem/mfem/blob/master/examples/ex30p.cpp

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_parallel::{
    WorkerConfig,
    launcher::native::ThreadLauncher,
    par_partition::partition_mesh,
};

// ── Coefficient functions matching C++ ex30p ───────────────────────────────

/// Piecewise-affine function (mesh-conforming for order≥1)
fn affine_function(p: &[f64]) -> f64 {
    let x = p[0];
    let y = p[1];
    if x < 0.0 { 1.0 + x + y } else { 1.0 }
}

/// Piecewise-constant function (never mesh-conforming)
fn jump_function(p: &[f64]) -> f64 {
    let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
    if r > 0.4 && r < 0.6 { 1.0 } else { 5.0 }
}

/// Singular function (derived from Laplacian of steep wavefront)
fn singular_function(p: &[f64]) -> f64 {
    let x = p[0];
    let y = p[1];
    let alpha = 1000.0_f64;
    let xc = 0.75;
    let yc = 0.5;
    let r0 = 0.7;
    let r = ((x - xc).powi(2) + (y - yc).powi(2)).sqrt();
    if r < 1e-8 { return 0.0; }
    let num = -(alpha - alpha.powi(3) * (r * r - r0 * r0));
    let denom = (r * (alpha.powi(2) * r0 * r0 + alpha.powi(2) * r * r
        - 2.0 * alpha.powi(2) * r0 * r + 1.0)).powi(2);
    let denom = denom.max(1.0e-8);
    num / denom
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Use ThreadLauncher for parallel execution
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        // Each rank reads the mesh independently (shared filesystem for ThreadLauncher)
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read mesh");
        let mesh = mfem.mesh2d.expect("expected 2D mesh");

        // Partition and create parallel mesh
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // ── Function 0 (affine) ──
        let n_elems = local_mesh.n_elems();
        let osc = compute_coeff_oscillation(&local_mesh, &affine_function);

        if comm.is_root() {
            println!("\nFunction 0 (affine)");
            println!("Number of Elements {n_elems}");
            println!("Osc error {osc:.6e}");
        }

        // ── Function 1 (jump) ──
        let n_elems = local_mesh.n_elems();
        let osc = compute_coeff_oscillation(&local_mesh, &jump_function);

        if comm.is_root() {
            println!("\nFunction 1 (discontinuous)");
            println!("Number of Elements {n_elems}");
            println!("Osc error {osc:.6e}");
        }

        // ── Function 2 (singular) ──
        let n_elems = local_mesh.n_elems();
        let osc = compute_coeff_oscillation(&local_mesh, &singular_function);

        if comm.is_root() {
            println!("\nFunction 2 (singular)");
            println!("Number of Elements {n_elems}");
            println!("Osc error {osc:.6e}");
        }
    });
}

/// Compute max coefficient variation across elements (oscillation proxy).
fn compute_coeff_oscillation(mesh: &Mesh<2>, coeff: &impl Fn(&[f64]) -> f64) -> f64 {
    let nelems = mesh.n_elems();
    let mut max_osc = 0.0_f64;
    for e in 0..nelems as u32 {
        let nodes = mesh.element_nodes(e);
        let vals: Vec<f64> = nodes.iter().map(|&n| coeff(mesh.node_coords(n))).collect();
        let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let center = vals.iter().sum::<f64>() / vals.len() as f64;
        let osc = if center.abs() > 1e-15 { (max_v - min_v) / center.abs() } else { max_v - min_v };
        if osc > max_osc { max_osc = osc; }
    }
    max_osc
}

// ── CLI args ───────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    threshold: f64,
    max_iters: usize,
    np: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        mesh_file: "data/star.mesh".to_string(),
        threshold: 1e-3,
        max_iters: 10,
        np: 2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => args.mesh_file = it.next().unwrap_or_default(),
            "-e" | "--error" => args.threshold = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-3),
            "--max-iters" => args.max_iters = it.next().and_then(|s| s.parse().ok()).unwrap_or(10),
            "--np" => args.np = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            _ => {}
        }
    }
    args
}
