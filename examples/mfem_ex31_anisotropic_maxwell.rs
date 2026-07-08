//! # Example 31 — Anisotropic Maxwell  (1:1 with MFEM ex31)
//!
//! Solves:
//! ```text
//!   ∇×(∇×E) + Σ E = f
//! ```
//! with anisotropic tensor Σ = diag(σ_x, σ_y) and PEC BC (n×E=0).
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex31_anisotropic_maxwell -- -m data/star.mesh -o 2 -r 2
//! cargo run --example mfem_ex31_anisotropic_maxwell -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::{
    StaticMaxwellBuilder, l2_error_hcurl_exact,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_space::{HCurlSpace, fe_space::FESpace};

const DEFAULT_SIGMA_X: f64 = 4.0;
const DEFAULT_SIGMA_Y: f64 = 1.5;
const DEFAULT_SCALE: f64 = 1.0;

struct Args {
    mesh: Option<String>,
    n: usize,
    ref_levels: usize,
    order: u8,
    sigma_x: f64,
    sigma_y: f64,
    verbose: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None, n: 16, ref_levels: 0, order: 1,
        sigma_x: DEFAULT_SIGMA_X, sigma_y: DEFAULT_SIGMA_Y,
        verbose: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(16),
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(0)
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "--sigma-x" => a.sigma_x = it.next().and_then(|v| v.parse().ok()).unwrap_or(4.0),
            "--sigma-y" => a.sigma_y = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.5),
            "--verbose" => a.verbose = true,
            "--quiet" => a.verbose = false,
            _ => {}
        }
    }
    a
}

fn source_value(x: &[f64], sigma_x: f64, sigma_y: f64, _scale: f64) -> [f64; 2] {
    let fx = (PI * PI + sigma_x) * (PI * x[1]).sin();
    let fy = (PI * PI + sigma_y) * (PI * x[0]).sin();
    [fx, fy]
}

/// Exact solution: E = (sin(πy), sin(πx))
fn exact_e(x: &[f64]) -> [f64; 2] {
    [(PI * x[1]).sin(), (PI * x[0]).sin()]
}

fn main() {
    let args = parse_args();
    println!("Options used:");
    match &args.mesh {
        Some(p) => println!("   --mesh {p}"),
        None => println!("   --mesh (built-in {0}x{0} tri)", args.n),
    }
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --sigma-x {}", args.sigma_x);
    println!("   --sigma-y {}", args.sigma_y);
    println!("   --no-visualization");
    println!();

    // Read or generate mesh
    let base_mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    // Uniform refinement
    eprintln!("  Init mesh: {} elems, {} nodes, {} bdr faces, tags={:?}",
        base_mesh.n_elems(), base_mesh.n_nodes(), base_mesh.n_boundary_faces(),
        base_mesh.unique_boundary_tags());
    let mesh = if args.ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..args.ref_levels { m = refine_uniform(&m); }
        eprintln!("  Refined mesh: {} elems, {} nodes, {} bdr faces",
            m.n_elems(), m.n_nodes(), m.n_boundary_faces());
        m
    } else {
        base_mesh
    };

    // HCurl space
    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("Number of H(Curl) unknowns: {n_dofs}");

    // Build the problem: ∇×(∇×E) + Σ·E = f, n×E=0 on all boundaries
    let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_marker = vec![1; bdr_attrs.len()];

    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_anisotropic_diag(1.0, args.sigma_x, args.sigma_y)
        .with_source_fn(move |x| source_value(x, args.sigma_x, args.sigma_y, DEFAULT_SCALE))
        .add_pec_zero_from_marker(&bdr_attrs, &ess_marker)
        .build();

    let solved = problem.solve();
    let u = &solved.solution;

    // Compute H(Curl) error
    let hcurl_err = l2_error_hcurl_exact(
        &solved.space,
        u,
        exact_e,
    );
    println!("|| E_h - E ||_{{H(Curl)}} = {hcurl_err:.10e}");
}
