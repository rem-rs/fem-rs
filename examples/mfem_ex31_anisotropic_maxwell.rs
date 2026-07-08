//! # Example 31 — Anisotropic Maxwell  (1:1 with MFEM ex31)
//!
//! Solves:
//! ```text
//!   ∇×(∇×E) + Σ·E = f
//! ```
//! with a full 2×2 anisotropic tensor Σ and PEC BC (n×E=0).
//!
//! The exact solution (matching MFEM ex31 2D case) is:
//! ```text
//!   E₀ = a₀·sin(κ·√½·(x+y))
//!   E₁ = a₁·sin(κ·√½·(x+y) + φ₁)
//! ```
//! with a₀=1.1, a₁=1.2, φ₁=0.4π, κ=π·freq.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex31_anisotropic_maxwell -- --n 32
//! cargo run --example mfem_ex31_anisotropic_maxwell -- -m data/star.mesh -o 2 -r 2
//! ```
//!
//! ## Comparison with C++ ex31
//!
//! C++ ex31 uses `ND_R2D_FECollection` (restricted 3-component H(Curl) even in 2D)
//! and a full 3×3 tensor Σ.  Rust uses `HCurlSpace` with 2-component fields and a
//! 2×2 upper-left block of Σ (matching the 2D subset).
//!
//! | Aspect | C++ ex31 | Rust ex31 |
//! |--------|----------|-----------|
//! | Space | ND_R2D (3‑component 2D) | HCurl (2‑component) |
//! | Σ | 3×3 full tensor | 2×2 upper-left block |
//! | Error | H(Curl) = L²+curl | L² only (need `ComputeHCurlError`) |
//! | GLVis | 4‑window component viz | None |
//! | Solver | PCG+GSSmoother | PCG+GSSmoother (via builder) |
//! | Exact | a₀·sin(κ√½(x+y)+φ₀) … | ✓ matches C++ 2D |

use std::f64::consts::{PI, SQRT_2};

use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, amr::refine_uniform};
use fem_space::{HCurlSpace, fe_space::FESpace};

/// C++ MFEM ex31 2D coefficients.
const A0: f64 = 1.1;
const A1: f64 = 1.2;
const PHI1: f64 = 0.4 * PI;
/// σ matrix from C++ upper-left 2×2 block:
/// Σ = [[2.0,  1/√2],
///      [1/√2, 2.0]]
const SIGMA_XX: f64 = 2.0;
const SIGMA_XY: f64 = 1.0 / SQRT_2;
const SIGMA_YY: f64 = 2.0;

struct Args {
    mesh: Option<String>,
    n: usize,
    ref_levels: usize,
    order: u8,
    freq: f64,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None, n: 16, ref_levels: 2, order: 1,
        freq: 1.0, visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(16),
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-f" | "--frequency" => {
                a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}

/// C++ E_exact, 2D: E = (a₀·sin(κ√½(x+y)+φ₀), a₁·sin(κ√½(x+y)+φ₁))
fn exact_e(x: &[f64], kappa: f64) -> [f64; 2] {
    let arg = kappa * (x[0] + x[1]) / SQRT_2;
    [A0 * arg.sin(), A1 * (arg + PHI1).sin()]
}

/// Manufactured source: f = ∇×(∇×E) + Σ·E  for the exact 2-component solution.
///
/// With E = (a₀·sin(u), a₁·sin(u+φ₁)), u = κ/√2·(x+y), Σ = [[2, 1/√2],[1/√2, 2]]:
///   ∇×(∇×E) = (g, -g)  where g = (κ²/2)·(a₀·sin(u) - a₁·sin(u+φ₁))
///   Σ·E = (2·E₀ + E₁/√2, E₀/√2 + 2·E₁)
fn source_value(x: &[f64], kappa: f64) -> [f64; 2] {
    let alpha2 = kappa * kappa / 2.0;
    let u = (kappa / SQRT_2) * (x[0] + x[1]);
    let (s0, s1) = (u.sin(), (u + PHI1).sin());

    let g = alpha2 * (A0 * s0 - A1 * s1);
    let se0 = SIGMA_XX * A0 * s0 + SIGMA_XY * A1 * s1;
    let se1 = SIGMA_XY * A0 * s0 + SIGMA_YY * A1 * s1;
    [g + se0, -g + se1]
}

fn main() {
    let args = parse_args();
    let kappa = args.freq * PI;

    println!("Options used:");
    match &args.mesh {
        Some(p) => println!("   --mesh {p}"),
        None => println!("   --mesh (built-in {0}x{0} tri)", args.n),
    }
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --frequency {}", args.freq);
    println!("   --sigma-xx {SIGMA_XX}, --sigma-xy {SIGMA_XY}, --sigma-yy {SIGMA_YY}");
    if args.visualization { println!("   --visualization"); }
    println!();

    // Read or generate mesh (matching C++: read, then refine).
    let base_mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let mesh = if args.ref_levels > 0 {
        let mut m = base_mesh;
        for _ in 0..args.ref_levels { m = refine_uniform(&m); }
        m
    } else {
        base_mesh
    };

    // H(Curl) Nédélec space.
    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("Number of H(Curl) unknowns: {n_dofs}");

    // Build the problem: ∇×(∇×E) + Σ·E = f, n×E=0.
    //
    // Σ is the full anisotropic 2×2 matrix from C++ ex31 upper-left block:
    //   Σ = [[2.0,  1/√2],
    //        [1/√2, 2.0]]
    //
    // The FnMatrixCoeff expects column-major: [σ₀₀, σ₁₀, σ₀₁, σ₁₁]
    let bdr_attrs: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_marker = vec![1; bdr_attrs.len()];

    let kappa = kappa; // move into closure
    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_anisotropic_matrix_fn(1.0, move |_: &[f64]| {
            // Constant anisotropic 2×2 tensor, column-major.
            [SIGMA_XX, SIGMA_XY, SIGMA_XY, SIGMA_YY]
        })
        .with_source_fn(move |x| source_value(x, kappa))
        .add_pec_zero_from_marker(&bdr_attrs, &ess_marker)
        .build();

    let solved = problem.solve();
    let u = &solved.solution;

    // H(Curl) norm of the error: √(‖E_h-E‖²_L² + ‖∇×E_h-∇×E‖²_L²).
    let curl_exact = |x: &[f64]| -> f64 {
        let u = (kappa / SQRT_2) * (x[0] + x[1]);
        (kappa / SQRT_2) * (A1 * (u + PHI1).cos() - A0 * u.cos())
    };
    let err2 = fem_examples::maxwell::hcurl_error_sq_exact(
        &solved.space, u, |x| exact_e(x, kappa), curl_exact,
    );
    let hcurl_err = err2.sqrt();
    println!("\n|| E_h - E ||_{{H(Curl)}} = {hcurl_err:.10e}");
}
