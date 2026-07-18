//! # Example 30 — Anisotropic AMR data oscillation (1:1 with MFEM ex30)
//!
//! Preprocesses a mesh by adaptively refining to resolve coefficient data.
//! There is no PDE being solved — this is a mesh preprocessor that lowers
//! data oscillation to a user-defined relative threshold.
//!
//! Three coefficients are tested sequentially, each building on the refined
//! mesh from the previous step:
//!   - Function 0: piecewise-affine (mesh-conforming for order > 0)
//!   - Function 1: piecewise-constant ring (never mesh-conforming)
//!   - Function 2: singular (Laplacian of steep wavefront from [2])
//!
//! References:
//!   [1] Morin, P., Nochetto, R. H., & Siebert, K. G. (2000). Data
//!       oscillation and convergence of adaptive FEM. SIAM J. Numer. Anal.
//!   [2] Mitchell, W. F. (2013). A collection of 2D elliptic problems for
//!       testing adaptive grid refinement algorithms. Appl. Math. Comput.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex30_aniso_amr -- -m data/star.mesh -o 1
//! cargo run --example mfem_ex30_aniso_amr -- -m data/star.mesh -o 2 -no-vis
//! cargo run --example mfem_ex30_aniso_amr -- -m data/star.mesh -o 2 -e 1e-3
//! ```

use std::time::Instant;
use fem_assembly::postproc::grid_function::{GridFunction, project_coefficient, compute_coeff_l2_norm};
use fem_element::lagrange::QuadQ1;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType,
    amr::{refine_nonconforming_quad, closure_refine_default}};
use fem_space::L2Space;

// ── Coefficient functions — exact 1:1 with MFEM ex30.cpp ───────────────

/// Piecewise-affine function which is sometimes mesh-conforming (Function 0).
fn affine_function(p: &[f64]) -> f64 {
    if p[0] < 0.0 { 1.0 + p[0] + p[1] } else { 1.0 }
}

/// Piecewise-constant function which is never mesh-conforming (Function 1).
fn jump_function(p: &[f64]) -> f64 {
    let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
    if r > 0.4 && r < 0.6 { 1.0 } else { 5.0 }
}

/// Singular function: Laplacian of steep wavefront from [2] (Function 2).
fn singular_function(p: &[f64]) -> f64 {
    let alpha: f64 = 1000.0;
    let xc: f64 = 0.75;
    let yc: f64 = 0.5;
    let r0: f64 = 0.7;
    let x = p[0] - xc;
    let y = p[1] - yc;
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let a2 = alpha * alpha;
    let num = -(alpha - a2 * alpha * (r2 - r0 * r0));
    let denom_raw = r * (a2 * r0 * r0 + a2 * r2 - 2.0 * a2 * r0 * r + 1.0);
    let denom = denom_raw * denom_raw;
    num / denom.max(1.0e-8)
}

// ── Command-line arguments ────────────────────────────────────────────

struct Args {
    mesh: String,
    order: u8,
    threshold: f64,
    max_elements: usize,
    nc_limit: usize,
    enriched_order: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".into(),
        order: 1,
        threshold: 1e-3,
        max_elements: 100_000,
        nc_limit: 1,
        enriched_order: 5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-e" | "--error" => a.threshold = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-3),
            "-me" | "--max-elems" => a.max_elements = it.next().and_then(|v| v.parse().ok()).unwrap_or(100_000),
            "-l" | "--nc-limit" => a.nc_limit = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-eo" | "--enriched_order" => a.enriched_order = it.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    a
}

// ── Element size: |det(J)|^(1/dim) at element center ──────────────────
// Approximates MFEM GetElementSize(type=1) = J.CalcSingularvalue(Dim-1).

fn element_size<M: MeshTopology>(mesh: &M, elem: u32) -> f64 {
    let dim = mesh.topological_dim() as usize;
    let elem_type = mesh.element_type(elem);
    let nodes = mesh.element_nodes(elem);
    let needs_iso = matches!(elem_type, ElementType::Quad4);
    if needs_iso {
        let geo = QuadQ1;
        let n_geo = geo.n_dofs();
        let xi = [0.5; 2];
        let mut grad = vec![0.0_f64; n_geo * dim];
        geo.eval_grad_basis(&xi[..dim], &mut grad);
        let mut j = nalgebra::DMatrix::<f64>::zeros(dim, dim);
        for k in 0..n_geo {
            let xk = mesh.node_coords(nodes[k]);
            for i in 0..dim {
                for d in 0..dim {
                    j[(i, d)] += xk[i] * grad[k * dim + d];
                }
            }
        }
        j.determinant().abs().powf(1.0 / dim as f64)
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = nalgebra::DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        j.determinant().abs().powf(1.0 / dim as f64)
    }
}

// ── Preprocess mesh: refine to resolve coefficient via L² oscillation ─
// 1:1 with MFEM CoefficientRefiner::PreprocessMesh.

fn preprocess(mesh: &mut Mesh<2>,
              coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
              order: u8,
              threshold: f64,
              max_elements: usize,
              _nc_limit: usize,
              enriched_order: u8) -> (usize, f64) {
    let quad_order = 2 * order + enriched_order;

    for _iter in 0..10 {
        let ne = mesh.n_elems();
        let l2 = L2Space::new(mesh.clone(), order);
        let dofs = project_coefficient(&l2, coeff, quad_order);
        let gf = GridFunction::new(&l2, dofs);

        let norm_of_coeff = compute_coeff_l2_norm(mesh, coeff, quad_order);
        let av_norm = norm_of_coeff / (ne as f64).sqrt();

        let element_norms = gf.compute_element_l2_errors(coeff, quad_order);

        let mut marked = Vec::new();
        let mut osc2 = 0.0;

        for e in 0..ne as u32 {
            let h = element_size(mesh, e);
            let element_osc = h * element_norms[e as usize];
            if element_osc > threshold * av_norm {
                marked.push(e);
            }
            osc2 += element_osc * element_osc;
        }

        let global_osc = osc2.sqrt() / norm_of_coeff.max(1e-30);

        if global_osc < threshold || ne as i64 >= max_elements as i64 || marked.is_empty() {
            return (ne, global_osc);
        }

        match mesh.element_type(0) {
            ElementType::Tri3 => *mesh = closure_refine_default(mesh, &marked, None),
            ElementType::Quad4 => *mesh = refine_nonconforming_quad(mesh, &marked, None).0,
            _ => panic!("unsupported element type"),
        }
    }
    (mesh.n_elems(), 0.0)
}

// ── Main ──────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let mut mesh: Mesh<2> = read_mfem_file(&args.mesh)
        .expect("failed to read MFEM mesh")
        .mesh2d
        .expect("MFEM mesh must be 2D");

    // Note: NURBS meshes (ball-nurbs, square-disc-nurbs, etc.) are not yet
    // supported by Mesh<2>.  The C++ ex30 converts NURBS to polynomial curved
    // via UniformRefinement + SetCurvature(2).

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --nc-limit {}", args.nc_limit);
    println!("   --max-elems {}", args.max_elements);
    println!("   --error {}", args.threshold);
    println!("   --enriched_order {}", args.enriched_order);
    println!("   --no-visualization\n");

    let coeffs: [(&str, &(dyn Fn(&[f64]) -> f64 + Send + Sync)); 3] = [
        ("affine", &affine_function),
        ("discontinuous", &jump_function),
        ("singular", &singular_function),
    ];

    for (idx, (name, coeff)) in coeffs.iter().enumerate() {
        println!("Function {idx} ({name}) ");
        let (n, osc) = preprocess(&mut mesh, *coeff, args.order,
                                   args.threshold, args.max_elements,
                                   args.nc_limit, args.enriched_order);
        println!("Number of Elements {n}");
        println!("Osc error {osc:.6}\n");
    }

    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
}
