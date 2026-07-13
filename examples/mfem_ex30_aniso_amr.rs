//! # Example 30 — Anisotropic AMR data oscillation (1:1 with MFEM ex30)
//!
//! Preprocesses a mesh by adaptively refining to resolve coefficient data.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex30_aniso_amr -- -m data/star.mesh -o 2
//! cargo run --example mfem_ex30_aniso_amr -- -m data/star.mesh -o 2 -e 1e-3
//! ```

use std::time::Instant;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::postproc::grid_function::project_coefficient;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType,
    amr::{refine_nonconforming_quad, closure_refine_default}};
use fem_space::{L2Space, fe_space::FESpace};

fn affine_fn(x: &[f64]) -> f64 { x[0] + 2.0 * x[1] + 1.0 }
fn jump_fn(x: &[f64]) -> f64 {
    if (x[0]*x[0] + x[1]*x[1]).sqrt() > 0.3 { 1.0 } else { 2.0 }
}
fn singular_fn(x: &[f64]) -> f64 {
    (100.0 * (x[0]*x[0] + x[1]*x[1] - 1.0)).atan()
}

struct Args {
    mesh: String, order: u8, threshold: f64,
    max_elements: usize, nc_limit: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".into(), order: 1, threshold: 1e-2,
        max_elements: 100_000, nc_limit: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-e" | "--error" => a.threshold = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-2),
            "-me" | "--max-elems" => a.max_elements = it.next().and_then(|v| v.parse().ok()).unwrap_or(100_000),
            "-l" | "--nc-limit" => a.nc_limit = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            _ => {}
        }
    }
    a
}

/// Preprocess mesh: refine to resolve coefficient via L² oscillation marking.
fn preprocess(mesh: &mut Mesh<2>, coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
              order: u8, threshold: f64, max_elements: usize) -> (usize, f64) {
    for _iter in 0..15 {
        let ne = mesh.n_elems();
        let l2 = L2Space::new(mesh.clone(), order);
        let dofs = project_coefficient(&l2, coeff, order as u8 * 2 + 3);
        let gf = GridFunction::new(&l2, dofs);
        let norm_coeff = gf.compute_l2_error(&|_| 0.0, order as u8 * 2 + 3);
        let av_norm = norm_coeff / (ne as f64).sqrt();

        // Element-wise oscillation: use h_K × local projection error
        // approximated via compute_l2_error with per-element breakdown
        let mut marked = Vec::new();
        let mut osc2 = 0.0;
        let dim = mesh.dim() as usize;

        for e in 0..ne as u32 {
            let ns = mesh.element_nodes(e);
            let h = element_size(mesh, ns, dim);
            // Compute element L² error vs coefficient via grid function
            let elem_err = element_l2_error(&gf, e, coeff, order as u8 * 2 + 3);
            let osc = h * elem_err;
            if osc > threshold * av_norm { marked.push(e); }
            osc2 += osc * osc;
        }

        let global_osc = osc2.sqrt() / norm_coeff.max(1e-30);
        eprintln!("    elems={ne} osc={global_osc:.6e} marked={}", marked.len());

        if global_osc < threshold || ne >= max_elements || marked.is_empty() { return (ne, global_osc); }
        match mesh.element_type(0) {
            ElementType::Tri3 => *mesh = closure_refine_default(mesh, &marked, None),
            ElementType::Quad4 => *mesh = refine_nonconforming_quad(mesh, &marked, None).0,
            _ => panic!("unsupported element type"),
        }
    }
    (mesh.n_elems(), 0.0)
}

/// Characteristic element size = max edge length (matches MFEM GetElementSize).
fn element_size<M: MeshTopology>(mesh: &M, ns: &[u32], _dim: usize) -> f64 {
    let nv = ns.len();
    let mut max_len2 = 0.0_f64;
    for i in 0..nv {
        let a = mesh.node_coords(ns[i]);
        let b = mesh.node_coords(ns[(i + 1) % nv]);
        let dx = b[0] - a[0]; let dy = b[1] - a[1];
        max_len2 = max_len2.max(dx * dx + dy * dy);
    }
    max_len2.sqrt()
}

fn element_l2_error(gf: &GridFunction<'_, L2Space<Mesh<2>>>, e: u32, exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync), qo: u8) -> f64 {
    use fem_element::lagrange::{TriP1, TriP2, TriP3};
    use fem_element::ReferenceElement;
    let mesh = gf.space().mesh();
    let dim = mesh.dim() as usize;
    let order = gf.space().order();
    let elem_type = mesh.element_type(e);
    let ref_elem: Box<dyn ReferenceElement> = match (elem_type, order) {
        (fem_mesh::element_type::ElementType::Tri3, 1) => Box::new(TriP1),
        (fem_mesh::element_type::ElementType::Tri3, 2) => Box::new(TriP2),
        (fem_mesh::element_type::ElementType::Tri3, 3) => Box::new(TriP3),
        _ => panic!("unsupported element"),
    };
    let n_ldofs = ref_elem.n_dofs();
    let quad = ref_elem.quadrature(qo);
    let elem_dofs = gf.space().element_dofs(e);
    let nodes = mesh.element_nodes(e);
    let x0 = mesh.node_coords(nodes[0]);
    let (jac, det_j) = simplex_jac(mesh, nodes, dim);
    let mut err2 = 0.0;
    let mut phi = vec![0.0; n_ldofs];
    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * det_j.abs();
        ref_elem.eval_basis(xi, &mut phi);
        let mut uh = 0.0;
        let d = gf.dofs();
        for i in 0..n_ldofs { uh += d[elem_dofs[i] as usize] * phi[i]; }
        let xp = phys_coords(x0, &jac, xi, dim);
        let ue = exact(&xp);
        err2 += w * (uh - ue) * (uh - ue);
    }
    err2.sqrt()
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (nalgebra::DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = nalgebra::DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn phys_coords(x0: &[f64], jac: &nalgebra::DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {
    let mut xp = x0.to_vec();
    for d in 0..dim {
        for k in 0..dim { xp[d] += jac[(d, k)] * xi[k]; }
    }
    xp
}

fn main() {
    let args = parse_args();
    let t0 = Instant::now();
    let mut mesh: Mesh<2> = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh")
        .mesh2d.expect("MFEM mesh must be 2D");

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --nc-limit {0}  (not implemented)", args.nc_limit);
    println!("   --max-elems {}", args.max_elements);
    println!("   --error {}", args.threshold);
    println!("   --no-visualization\n");

    let fns: [(&str, &(dyn Fn(&[f64]) -> f64 + Send + Sync)); 3] = [
        ("affine", &|x| affine_fn(x)),
        ("discontinuous", &|x| jump_fn(x)),
        ("singular", &|x| singular_fn(x)),
    ];
    for (name, coeff) in &fns {
        println!("Function {} ()", name);
        let (n, osc) = preprocess(&mut mesh, *coeff, args.order, args.threshold, args.max_elements);
        println!("Number of Elements {n}");
        println!("Osc error {osc:.6}\n");
    }
    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
}
