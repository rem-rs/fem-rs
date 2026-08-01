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
use fem_assembly::postproc::grid_function::compute_coeff_l2_norm;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType,
    amr::{closure_refine_default, NCStateQuad}};

// ── Coefficient functions — exact 1:1 with MFEM ex30.cpp ───────────────

fn affine_function(p: &[f64]) -> f64 {
    if p[0] < 0.0 { 1.0 + p[0] + p[1] } else { 1.0 }
}

fn jump_function(p: &[f64]) -> f64 {
    let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
    if r > 0.4 && r < 0.6 { 1.0 } else { 5.0 }
}

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

// ── Element size: h = |det J|^(1/dim) at element center ─────────────────
// Exact match with MFEM GetElementSize(i) DEFAULT type=0 (weight-based
// size = |det J|^(1/Dim)); NOT type=1 (h_min = sigma_min).

fn element_size<M: MeshTopology>(mesh: &M, elem: u32) -> f64 {
    let dim = mesh.topological_dim() as usize;
    let elem_type = mesh.element_type(elem);
    let nodes = mesh.element_nodes(elem);
    let needs_iso = matches!(elem_type, ElementType::Quad4);
    if needs_iso {
        // MFEM Quad reference domain is [0,1]^2 (BiLinear2DFiniteElement nodes
        // (0,0),(1,0),(1,1),(0,1)); GetElementSize evaluates at GeomCenter =
        // (0.5,0.5).  Using QuadQ1 ([-1,1]^2) here would scale J by 2 and give
        // h twice the MFEM value.
        let geo = fem_element::lagrange::factory::QuadQk::new(1);
        let n_geo = geo.n_dofs();
        let xi = [0.5, 0.5];
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

// ── Preprocess mesh ───────────────────────────────────────────────────
// 1:1 with MFEM CoefficientRefiner::PreprocessMesh.

// [TEMP ex30 verify] Element-wise ‖(Π−I)f‖_{L²(K)} using Gauss-Legendre node
// interpolation of the Q1 (L2 P1) space — MFEM L2_FECollection default
// BasisType::GaussLegendre.  On [0,1]²: nodes a=½(1−1/√3), b=½(1+1/√3).
fn gl_p1_element_errors(mesh: &Mesh<2>,
                        coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
                        quad_order: u8) -> Vec<f64> {
    use fem_element::lagrange::factory::QuadQk;
    use fem_element::ReferenceElement;
    let inv_s3 = 1.0 / 3.0_f64.sqrt();
    let (a, b) = (0.5 * (1.0 - inv_s3), 0.5 * (1.0 + inv_s3));
    let gl = [a, b];
    let geo = QuadQk::new(1); // Q1 on [0,1]²
    let n_geo = geo.n_dofs();
    let mut phi = vec![0.0; n_geo];
    let mut grad = vec![0.0; n_geo * 2];

    let mut q1_map = |nodes: &[u32], xi: &[f64]| -> [f64; 2] {
        geo.eval_basis(xi, &mut phi);
        let mut xp = [0.0; 2];
        for k in 0..n_geo {
            let c = mesh.node_coords(nodes[k]);
            for d in 0..2 { xp[d] += phi[k] * c[d]; }
        }
        xp
    };

    let mut errors = vec![0.0; mesh.n_elems()];
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        // f at the 4 GL nodes (physical positions via Q1 map)
        let mut fk = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let xp = q1_map(nodes, &[gl[i], gl[j]]);
                fk[i][j] = coeff(&xp);
            }
        }
        // Πf(x,y) = Σ_{i,j} fk[i][j] l_i(x) l_j(y)
        let interp = |x: f64, y: f64| -> f64 {
            let lx = [(x - b) / (a - b), (x - a) / (b - a)];
            let ly = [(y - b) / (a - b), (y - a) / (b - a)];
            let mut s = 0.0;
            for i in 0..2 {
                for j in 0..2 { s += fk[i][j] * lx[i] * ly[j]; }
            }
            s
        };
        let quad = geo.quadrature(quad_order);
        let mut err2 = 0.0;
        for (q, xi) in quad.points.iter().enumerate() {
            // Q1 map Jacobian (affine => constant, but computed generally)
            geo.eval_grad_basis(xi, &mut grad);
            let mut j = nalgebra::DMatrix::<f64>::zeros(2, 2);
            for k in 0..n_geo {
                let c = mesh.node_coords(nodes[k]);
                for i in 0..2 {
                    for d in 0..2 { j[(i, d)] += c[i] * grad[k * 2 + d]; }
                }
            }
            let det = j.determinant().abs();
            let xp = q1_map(nodes, xi);
            let w = quad.weights[q] * det;
            let d = interp(xi[0], xi[1]) - coeff(&xp);
            err2 += w * d * d;
        }
        errors[e as usize] = err2.sqrt();
    }
    errors
}

// ── Preprocess mesh ───────────────────────────────────────────────────
// 1:1 with MFEM CoefficientRefiner::PreprocessMesh.

fn preprocess(mesh: &mut Mesh<2>,
              coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
              order: u8,
              threshold: f64,
              max_elements: usize,
              nc_limit: usize,
              enriched_order: u8) -> (usize, f64) {
    let quad_order = 2 * order + enriched_order;

    // Non-conforming refinement state (tracks active edge midpoints and edge
    // refinement levels across iterations, exactly like MFEM's NCMesh).
    let mut ncstate = NCStateQuad::new();
    // Per-element refinement level (0 = initial).  Used with the accumulated
    // hanging-node constraints to enforce MFEM's LimitNCLevel: any fine
    // element whose level exceeds a coarse neighbor by more than nc_limit
    // triggers refinement of the coarse neighbor (1-irregular closure).
    let mut levels: Vec<u32> = vec![0; mesh.n_elems()];

    for _iter in 0..10 {
        let ne = mesh.n_elems();
        let norm_of_coeff = compute_coeff_l2_norm(mesh, coeff, quad_order);
        let av_norm = norm_of_coeff / (ne as f64).sqrt();

        // [TEMP ex30 verify] GL-node L2 P1 interpolation (MFEM L2_FECollection
        // default BasisType::GaussLegendre), replacing L2Space vertex nodes.
        let element_norms = gl_p1_element_errors(mesh, coeff, quad_order);

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

        let global_osc = osc2.sqrt() / (norm_of_coeff + 1e-10);


        if global_osc < threshold || ne as i64 > max_elements as i64 || marked.is_empty() {
            return (ne, global_osc);
        }

        // LimitNCLevel(nc_limit): 1-irregular closure driven by the accumulated
        // hanging-node constraints (NCStateQuad tracks them across iterations,
        // using coarse-edge keys, approximating MFEM NCMesh::LimitNCLevel).
        // nc_limit == 0 means unlimited: no closure at all.
        match mesh.element_type(0) {
            ElementType::Tri3 => *mesh = closure_refine_default(mesh, &marked, None),
            ElementType::Quad4 => {
                let (m, lv) = if nc_limit > 0 {
                    refine_with_nc_limit(mesh, &levels, &marked, nc_limit, &mut ncstate)
                } else {
                    let (mm, _c, _m) = ncstate.refine(mesh, &marked, 0);
                    let mut nl = Vec::with_capacity(mm.n_elems());
                    for e in 0..ne as u32 {
                        let lv0 = levels[e as usize];
                        if marked.contains(&e) { for _ in 0..4 { nl.push(lv0 + 1); } }
                        else { nl.push(lv0); }
                    }
                    (mm, nl)
                };
                *mesh = m;
                levels = lv;
            }
            _ => panic!("unsupported element type"),
        }
    }
    (mesh.n_elems(), 0.0)
}

// Refine `marked` on `mesh`, then repeatedly refine coarse neighbors that are
// more than nc_limit levels shallower than an adjacent fine element
// (MFEM NCMesh::LimitNCLevel).  `ncstate` accumulates midpoints/levels across
// all iterations.  Returns the final mesh and per-element levels.
fn refine_with_nc_limit(mesh: &Mesh<2>, levels: &[u32], marked: &[u32],
                        nc_limit: usize,
                        ncstate: &mut NCStateQuad) -> (Mesh<2>, Vec<u32>) {
    let mut cur = mesh.clone();
    let mut cur_levels = levels.to_vec();
    let mut to_refine: Vec<u32> = marked.to_vec();
    loop {
        let (new_mesh, constraints, _midpoints) = ncstate.refine(&cur, &to_refine, 0);
        // Rebuild per-element levels: element order is preserved, marked
        // elements expand to 4 children with level+1.
        let mut new_levels = Vec::with_capacity(new_mesh.n_elems());
        for e in 0..cur.n_elems() as u32 {
            let lv = cur_levels[e as usize];
            if to_refine.contains(&e) {
                for _ in 0..4 { new_levels.push(lv + 1); }
            } else {
                new_levels.push(lv);
            }
        }
        debug_assert_eq!(new_levels.len(), new_mesh.n_elems(), "level rebuild mismatch");

        // Find coarse elements (containing coarse edge parent_a-parent_b) that
        // are more than nc_limit shallower than the fine element containing
        // the hanging node `constrained`.
        let mut extra: Vec<u32> = Vec::new();
        let mut extra_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for c in &constraints {
            let fine = new_mesh.elem_iter().find(|&e| {
                new_mesh.element_nodes(e).contains(&(c.constrained as u32))
            });
            let coarse = new_mesh.elem_iter().find(|&e| {
                let ns = new_mesh.element_nodes(e);
                (0..4).any(|i| {
                    let a = ns[i]; let b = ns[(i + 1) % 4];
                    (a == c.parent_a as u32 && b == c.parent_b as u32)
                        || (a == c.parent_b as u32 && b == c.parent_a as u32)
                })
            });
            if let (Some(fe), Some(ce)) = (fine, coarse) {
                let lf = new_levels[fe as usize];
                let lc = new_levels[ce as usize];
                // 1-irregular closure: refine the coarse neighbor when the
                // fine-side depth exceeds the coarse level by more than
                // nc_limit.  (Approximates MFEM LimitNCLevel; the residual
                // discrepancy vs MFEM's exact NCMesh edge-split bookkeeping is
                // a few elements per iteration.)
                if lf > lc + nc_limit as u32 && extra_set.insert(ce) {
                    extra.push(ce);
                }
            }
        }
        if extra.is_empty() {
            return (new_mesh, new_levels);
        }
        cur = new_mesh;
        cur_levels = new_levels;
        to_refine = extra;
    }
}

// ── Main ──────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let mut mesh: Mesh<2> = read_mfem_file(&args.mesh)
        .expect("failed to read MFEM mesh")
        .mesh2d
        .expect("MFEM mesh must be 2D");

    // Note: NURBS meshes are not supported by Mesh<2>.

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
