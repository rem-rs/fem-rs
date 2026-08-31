//! # Example 30p — Adaptive mesh refinement preprocessing (parallel)
//!
//! Parallel version of ex30 (1:1 with MFEM ex30p): preprocesses a mesh by
//! adaptively refining to lower data oscillation of user-specified
//! coefficients, without solving a PDE.  Three coefficients are tested
//! sequentially, each continuing from the mesh refined by the previous one
//! (like C++ `CoefficientRefiner::PreprocessMesh` on the shared `pmesh`).
//!
//! The oscillation math (L2 projection of the coefficient, per-element L2
//! errors, global norm) lives in `fem_assembly::postproc::grid_function`
//! (shared with the serial ex30); the parallel loop marks locally, refines
//! via `par_refine_marked_ordered` and rebalances via
//! `par_repartition_with_hanging` — matching ex30p's per-iteration
//! refine + `pmesh->Rebalance()`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex30_amr_preprocess -- -m data/star.mesh
//! cargo run --example mfem_pex30_amr_preprocess -- --ranks 4 -e 1e-3
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use fem_assembly::postproc::grid_function::{GridFunction, compute_coeff_l2_norm_first_n};
use fem_core::ElemId;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::amr::limit_nc_level_quad;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_parallel::{
    Comm, WorkerConfig,
    launcher::native::ThreadLauncher,
    par_amr::{par_refine_marked_ordered, par_repartition_with_hanging},
    par_partition::partition_mesh,
};
use fem_space::{L2Space, fe_space::FESpace};

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

    let launcher = ThreadLauncher::new(WorkerConfig::new(args.np));
    launcher.launch(move |comm| {
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read mesh");
        let mesh = mfem.mesh2d.expect("expected 2D mesh");

        let mut par_mesh = partition_mesh(&mesh, &comm);
        // MFEM UpdateVertices node creation order (gid → order), threaded
        // across refine rounds for the parallel NC refinement bookkeeping.
        let mut creation_order: HashMap<u32, u32> =
            (0..mesh.n_nodes() as u32).map(|g| (g, g)).collect();

        let funcs: [(&str, &(dyn Fn(&[f64]) -> f64 + Send + Sync)); 3] = [
            ("affine", &affine_function),
            ("discontinuous", &jump_function),
            ("singular", &singular_function),
        ];
        for (fi, (label, coeff)) in funcs.iter().enumerate() {
            let (ne, osc) = parallel_preprocess(
                &mut par_mesh,
                &comm,
                *coeff,
                args.order,
                args.threshold,
                args.max_elems,
                args.nc_limit,
                args.enriched_order,
                &mut creation_order,
            );
            if comm.is_root() {
                println!("\nFunction {fi} ({label})");
                println!("Number of Elements {ne}");
                println!("Osc error {osc:.6e}");
            }
        }
    });
}

/// Parallel `CoefficientRefiner::PreprocessMesh` (1:1 math with the serial
/// ex30 `preprocess`): mark elements whose local oscillation exceeds the
/// relative threshold, refine them (parallel NC refine), rebalance, repeat
/// until the global oscillation drops below the threshold, the element budget
/// is reached or nothing is marked.
fn parallel_preprocess(
    par_mesh: &mut fem_parallel::par_mesh::ParallelMesh<Mesh<2>>,
    comm: &Comm,
    coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    order: u8,
    threshold: f64,
    max_elements: usize,
    nc_limit: usize,
    enriched_order: u8,
    creation_order: &mut HashMap<u32, u32>,
) -> (usize, f64) {
    let quad_order = 2 * order + enriched_order;

    for _iter in 0..10 {
        let local_mesh = par_mesh.local_mesh();
        let partition = par_mesh.partition();
        let n_owned = partition.n_owned_elems;

        let l2 = L2Space::new(local_mesh.clone(), order);
        // Nodal interpolation (NOT L2 projection) — MFEM CoefficientRefiner
        // interpolates the coefficient; the serial ex30 uses `interpolate`
        // and is bit-consistent with C++ (590/3341/12572).
        let dofs = l2.interpolate(coeff);
        let gf = GridFunction::new(&l2, dofs.as_slice().to_vec());
        // Global L2 norm of the coefficient over OWNED elements only — a
        // partitioned mesh lists owned elements first; the allreduce of the
        // squares then equals the true global norm (ghosts are cross-rank
        // mirrors and would be double-counted by the full-mesh function).
        let local_norm = compute_coeff_l2_norm_first_n(local_mesh, coeff, quad_order, n_owned);
        let global_norm2 = comm.allreduce_sum_f64(local_norm * local_norm);
        let global_ne = comm.allreduce_sum_i64(n_owned as i64) as usize;
        let norm_of_coeff = global_norm2.sqrt();
        let av_norm = norm_of_coeff / (global_ne as f64).sqrt();

        let element_norms = gf.compute_element_l2_errors(coeff, quad_order);

        let mut marked: Vec<ElemId> = Vec::new();
        let mut local_osc2 = 0.0;
        // Owned elements only for the global oscillation sum (ghosts mirror
        // owned elements across ranks and would be double-counted).
        for e in 0..n_owned as u32 {
            let h = element_size(local_mesh, e);
            let element_osc = h * element_norms[e as usize];
            if element_osc > threshold * av_norm {
                marked.push(e as ElemId);
            }
            local_osc2 += element_osc * element_osc;
        }
        let global_osc2 = comm.allreduce_sum_f64(local_osc2);
        let global_osc = global_osc2.sqrt() / (norm_of_coeff + 1e-10);

        if std::env::var("PEX30_TRACE").is_ok() && comm.is_root() {
            eprintln!("[pex30] iter={_iter} ne={global_ne} osc={global_osc:.6e} marked={}",
                marked.len());
        }

        if global_osc < threshold || global_ne > max_elements || marked.is_empty() {
            return (global_ne, global_osc);
        }

        // Parallel non-conforming refinement with nc_limit propagation
        // (MFEM `CoefficientRefiner` → `LimitNCLevel` loop: refine the
        // marked elements, then keep refining any element whose edge-split
        // level exceeds nc_limit — same fixpoint as serial ex30).
        let mut to_refine: Vec<ElemId> = marked.clone();
        loop {
            let r = par_refine_marked_ordered(
                par_mesh, fem_mesh::amr::NCState::new(), &to_refine, None, creation_order,
            )
            .expect("pex30 par_refine_marked_ordered failed");
            *creation_order = r.creation_order;
            let hanging_edges = r.hanging_edges.clone();
            let new_pm = par_repartition_with_hanging(r.par_mesh, &hanging_edges)
                .expect("pex30 par_repartition_with_hanging failed");
            let extra: Vec<ElemId> = limit_nc_level_quad(new_pm.local_mesh(), nc_limit as u32)
                .into_iter()
                .map(|e| e as ElemId)
                .collect();
            *par_mesh = new_pm;
            if extra.is_empty() {
                break;
            }
            to_refine = extra;
        }
    }
    let global_ne =
        comm.allreduce_sum_i64(par_mesh.partition().n_owned_elems as i64) as usize;
    (global_ne, 0.0)
}

/// Element size h (MFEM `GetElementSize`); identical to serial ex30.
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

// ── CLI args ───────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: u8,
    threshold: f64,
    max_elems: usize,
    nc_limit: usize,
    enriched_order: u8,
    np: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        mesh_file: "data/star.mesh".to_string(),
        order: 1,
        threshold: 1e-3,
        max_elems: 100_000,
        nc_limit: 1,
        enriched_order: 5,
        np: 2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => args.mesh_file = it.next().unwrap_or_default(),
            "-o" | "--order" => args.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-e" | "--error" => args.threshold = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-3),
            "--max-elems" => args.max_elems = it.next().and_then(|s| s.parse().ok()).unwrap_or(100_000),
            "--nc-limit" => args.nc_limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "--enriched-order" => args.enriched_order = it.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            "--ranks" | "--np" => args.np = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            _ => {}
        }
    }
    args
}
