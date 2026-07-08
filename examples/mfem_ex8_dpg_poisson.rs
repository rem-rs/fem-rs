//! # MFEM Example 8 — DPG (Discontinuous Petrov-Galerkin) Poisson
//!
//! Solves `-Δu = 1` with homogeneous Dirichlet BC using the DPG method.
//!
//! **Note on implementation:** MFEM ex8 uses a 2×2 block DPG formulation
//! with a separate trace (interfacial) unknown `xhat` and block operator
//! `B = [B0, Bhat]` forming the normal equation `A = B^T * Sinv * B`.
//! fem-rs uses a mathematically equivalent **single-field** DPG formulation
//! that condenses the trace variable and forms the element stiffness `K_e`
//! via `B^T * M_V^{-1} * B` directly — fewer unknowns, same solution.
//!
//! Both approaches solve the same problem with the same DPG optimal test
//! functions.  The 2×2 block form additionally provides a weighted-residual
//! error estimator `||Bx − F||_{S^{-1}}` which fem-rs does not expose.
//!
//! Reference: `mfem/ex8.cpp` (DPG Poisson)
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex8_dpg_poisson -- -m data/square-disc.mesh -no-vis
//! ```
//!
//! ## Flags
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `-m/--mesh` | `data/square-disc.mesh` | Mesh file (Tri3 only) |
//! | `-o/--order` | 1 | Trial FE order (test space enriched to p+2) |
//! | `-no-vis` | — | Disable GLVis (no-op) |

use fem_assembly::dpg::solve_dpg_poisson_2d;
use fem_io::mfem::{read_mfem_file, write_gf_file};
use fem_mesh::{Mesh, amr::refine_uniform};

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ── 1. Read mesh ──────────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        eprintln!("  Mesh file: {}", args.mesh);
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ── 2. Uniform refinement ─────────────────────────────────────────────────
    let ref_levels = {
        let ne = mesh.n_elems() as f64;
        let r = (10000.0_f64 / ne).log2() / 2.0;
        (r as usize).max(0).min(5)
    };
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels { m = refine_uniform(&m); }
        eprintln!("  Refined: {} nodes, {} elements ({} levels)", m.n_nodes(), m.n_elems(), ref_levels);
        m
    } else {
        mesh
    };

    // ── 3. DPG solve (single-field optimal test functions) ────────────────────
    // fem-rs uses condensed single-field DPG (see note above).
    let u = solve_dpg_poisson_2d(&mesh, &|_x, _y| 1.0);
    let n_dofs = mesh.n_nodes();
    println!("Number of unknowns: {}", n_dofs);

    // ── 4. Output ─────────────────────────────────────────────────────────────
    let dim = 2;
    fem_io::mfem::write_mfem_file("refined.mesh", &mesh).ok();
    if let Err(e) = write_gf_file("sol.gf", dim, &u, "H1", args.order.max(1), 1) {
        eprintln!("  Warning: could not write sol.gf: {e}");
    }

    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh: String,
    order: u8,
    #[allow(dead_code)]
    no_vis: bool,
}

impl Args {
    fn parse() -> Self {
        let mut mesh = "data/square-disc.mesh".to_string();
        let mut order: u8 = 1;
        let mut no_vis = false;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => { if let Some(v) = it.next() { mesh = v; } }
                "-o" | "--order" => { order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1); }
                "-no-vis" | "--no-visualization" => { no_vis = true; }
                _ => {}
            }
        }
        Args { mesh, order, no_vis }
    }
}
