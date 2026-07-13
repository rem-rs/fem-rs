//! # deal.II step-6 style: Adaptive Poisson with Kelly error indicator
//!
//! This example solves -Δu = f on a unit square, with a manufactured
//! solution u = arctan(α·(r - r₀)) that has a steep circular front.
//! Adaptive mesh refinement is driven by the Kelly face-jump error
//! indicator, with fixed-fraction marking (refine top 30%, coarsen
//! bottom 3%).
//!
//! This mirrors the approach of deal.II tutorial step-6.
//!
//! ## Usage
//! ```bash
//! cargo run --package fem-examples --example dealii_step6_poisson_amr
//! cargo run --package fem-examples --example dealii_step6_poisson_amr -- -n 5
//! cargo run --package fem-examples --example dealii_step6_poisson_amr -- -n 10 -o 2
//! ```
//!
//! ## Output
//! For each AMR iteration:
//! - Number of elements and DOFs
//! - L² error against the exact solution
//! - Refinement statistics (how many elements refined/coarsened)

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::error_estimate::{kelly_estimator, threshold_mark};
use fem_assembly::postproc::grid_function::GridFunction;
use std::collections::HashMap;

use fem_core::NodeId;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{NCStateQuad, HangingNodeConstraint};
use fem_solver::{solve_cg, SolverConfig, SolveResult};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{
        apply_dirichlet, boundary_dofs,
        apply_hanging_constraints, recover_hanging_values,
    },
};

// ─── Problem parameters ────────────────────────────────────────────────────

const ALPHA: f64 = 50.0;
const R0: f64 = 0.25;

/// Manufactured solution: u = arctan(α·(r - r₀))
fn exact_solution(pt: &[f64]) -> f64 {
    let r = ((pt[0] - 0.5).powi(2) + (pt[1] - 0.5).powi(2)).sqrt();
    (ALPHA * (r - R0)).atan()
}

/// RHS: -Δu
fn rhs_function(pt: &[f64]) -> f64 {
    let x = pt[0] - 0.5;
    let y = pt[1] - 0.5;
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    if r < 1e-15 {
        return 0.0;
    }
    let a2 = ALPHA * ALPHA;
    let denom = (1.0 + a2 * (r - R0).powi(2)).powi(2);
    // Analytical -Δ(arctan(α(r-r₀))) in 2D
    let u_xx = -ALPHA * (1.0 - 2.0 * a2 * (r - R0) * x * x / r) / denom;
    let u_yy = -ALPHA * (1.0 - 2.0 * a2 * (r - R0) * y * y / r) / denom;
    -(u_xx + u_yy)
}

// ─── CLI args ──────────────────────────────────────────────────────────────

struct Args {
    mesh_size: usize,
    order: u8,
    refine_fraction: f64,
    max_dofs: usize,
    max_steps: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mesh_size: 4,
            order: 1,
            refine_fraction: 0.3,
            max_dofs: 100_000,
            max_steps: 10,
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-s" | "--mesh-size" => args.mesh_size = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.mesh_size),
                "-o" | "--order" => args.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.order),
                "-r" | "--refine" => args.refine_fraction = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.refine_fraction),
                "-d" | "--max-dofs" => args.max_dofs = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.max_dofs),
                "-n" | "--max-steps" => args.max_steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.max_steps),
                _ => {}
            }
        }
        args
    }
}

// ─── L² error computation ─────────────────────────────────────────────────

fn compute_l2_error(gf: &GridFunction<'_, H1Space<Mesh<2>>>, exact: &dyn Fn(&[f64]) -> f64, order: u8) -> f64 {
    let quad_order = (order as u8) * 2 + 1;
    gf.compute_l2_error(exact, quad_order)
}

// ─── Merge hanging constraints (from ex15) ─────────────────────────────────

fn merge_hanging_constraints(
    old: &[HangingNodeConstraint],
    new: &[HangingNodeConstraint],
    new_mesh: &Mesh<2>,
) -> Vec<HangingNodeConstraint> {
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<NodeId>> = HashMap::new();
    for e in 0..new_mesh.n_elems() as NodeId {
        let ns = new_mesh.elem_nodes(e);
        let n_vert = ns.len();
        for i in 0..n_vert {
            let a = ns[i];
            let b = ns[(i + 1) % n_vert];
            let key = if a < b { (a, b) } else { (b, a) };
            edge_elems.entry(key).or_default().push(e);
        }
    }

    let mut merged: Vec<HangingNodeConstraint> = old
        .iter()
        .filter(|c| {
            let mid = c.constrained as NodeId;
            let pa = c.parent_a as NodeId;
            let pb = c.parent_b as NodeId;
            let key = if pa < pb { (pa, pb) } else { (pb, pa) };
            edge_elems
                .get(&key)
                .map(|elems| elems.iter().any(|&e| !new_mesh.elem_nodes(e).contains(&mid)))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    for c in new {
        if !merged.iter().any(|oc| oc.constrained == c.constrained) {
            merged.push(c.clone());
        }
    }

    merged.sort_by_key(|c| c.constrained);
    merged
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    let args = Args::parse();

    println!("deal.II step-6 style: Adaptive Poisson with Kelly indicator");
    println!("  mesh-size: {}x{}", args.mesh_size, args.mesh_size);
    println!("  order:    {}", args.order);
    println!("  refine:   {:.1}%", args.refine_fraction * 100.0);
    println!("  max-dofs: {}", args.max_dofs);
    println!("  max-steps: {}", args.max_steps);
    println!();

    // Generate a uniform Quad4 mesh of the unit square [0,1]²
    let mut mesh: Mesh<2> = Mesh::<2>::unit_square_quad(args.mesh_size);
    let elem_type = mesh.element_type(0);
    assert!(
        matches!(elem_type, ElementType::Quad4),
        "only Quad4 meshes supported (got {:?})",
        elem_type
    );

    // NC state for non-conforming refinement tracking
    let mut nc_state = NCStateQuad::new();
    let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();

    println!("Initial mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    let order = args.order;

    // AMR loop: step 0 is just the initial report; steps 1..=max_steps refine
    for step in 0..=args.max_steps {
        if step > 0 {
            // ─── Build FE space and solve ─────────────────────────────────
            let space = H1Space::new(mesh.clone(), order);
            let dofs = space.n_dofs();
            println!("\nStep {}: {} elements, {} DOFs", step, mesh.n_elems(), dofs);

            if dofs > args.max_dofs {
                println!("  Max DOFs ({}) reached, stopping.", args.max_dofs);
                break;
            }

            // Assemble system
            let quad_order = (order as u8) * 2 + 1;
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let source = DomainSourceIntegrator::new(rhs_function);
            let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad_order);
            let mut rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_order);

            // Apply hanging node constraints
            if !hanging_constraints.is_empty() {
                apply_hanging_constraints(&mut mat, &mut rhs_vec, &hanging_constraints);
            }

            // Dirichlet BC on all boundaries
            let dm = space.dof_manager();
            let bnd_tags = space.mesh().unique_boundary_tags();
            let bnd = boundary_dofs(space.mesh(), dm, &bnd_tags);
            let bnd_vals: Vec<f64> = bnd.iter()
                .map(|&dof| exact_solution(dm.dof_coord(dof)))
                .collect();
            apply_dirichlet(&mut mat, &mut rhs_vec, &bnd, &bnd_vals);

            // Fix zero diagonal entries
            for (row, &d) in mat.diagonal().iter().enumerate() {
                if d == 0.0 {
                    mat.eliminate_essential_bc_diag(row, 1.0);
                }
            }

            // Solve
            let mut u_h = vec![0.0; dofs];
            let res: Result<SolveResult, _> = solve_cg(&mat, &rhs_vec, &mut u_h, &SolverConfig {
                rtol: 1e-12,
                max_iter: 5000,
                verbose: false,
                ..SolverConfig::default()
            });

            match res {
                Ok(r) if r.converged => {
                    println!("  CG converged in {} iterations (res={:.3e})", r.iterations, r.final_residual);
                }
                Ok(r) => {
                    println!("  WARNING: CG did not converge (iters={}, res={:.3e})", r.iterations, r.final_residual);
                    continue;
                }
                Err(e) => {
                    eprintln!("  CG solver error: {:?}", e);
                    continue;
                }
            }

            // Recover hanging node values
            if !hanging_constraints.is_empty() {
                recover_hanging_values(&mut u_h, &hanging_constraints);
            }

            // Build GridFunction and compute errors
            let gf = GridFunction::new(&space, u_h);

            // Compute L² error (uses existing high‑order quadrature)
            let l2_err = compute_l2_error(&gf, &exact_solution, order);
            println!("  L² error: {:.6e}", l2_err);

            // Kelly error estimation
            let eta_vec = kelly_estimator(&gf).eta;

            let max_eta = eta_vec.iter().cloned().fold(0.0_f64, f64::max);
            println!("  max η_K: {:.6e}", max_eta);

            // ─── Kelly-driven marking ─────────────────────────────────────
            // Fixed-fraction marking: refine top 30% (by Kelly indicator)
            let n_refine = (eta_vec.len() as f64 * args.refine_fraction).ceil() as usize;
            let mut sorted_eta: Vec<(usize, f64)> = eta_vec.iter().copied().enumerate().collect();
            sorted_eta.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = sorted_eta.get(n_refine.saturating_sub(1)).map(|&(_, v)| v).unwrap_or(0.0);
            let marked = threshold_mark(&eta_vec, threshold);

            println!("  Kelly-AMR: marked {} elements (threshold={:.6e})", marked.len(), threshold);

            if !marked.is_empty() {
                let (new_mesh, new_constraints, _midpoint_map) = nc_state.refine(&mesh, &marked);
                hanging_constraints = merge_hanging_constraints(
                    &hanging_constraints, &new_constraints, &new_mesh,
                );
                mesh = new_mesh;
                println!("  Refined to {} elements", mesh.n_elems());
            } else {
                println!("  No elements marked for refinement, stopping.");
                break;
            }
        } else {
            // step 0: just report and refine uniformly once to give Kelly
            // something to work with on the coarse initial mesh
            println!("\nStep 0: {} elements, {} nodes (initial)", mesh.n_elems(), mesh.n_nodes());

            if step == 0 && args.max_steps > 0 {
                // One initial uniform refinement so the coarse mesh has
                // interior edges for the Kelly estimator to evaluate
                let all: Vec<u32> = (0..mesh.n_elems() as u32).collect();
                let (new_mesh, new_constraints, _) = nc_state.refine(&mesh, &all);
                hanging_constraints = new_constraints;
                mesh = new_mesh;
                println!("  Uniform initial refinement → {} elements", mesh.n_elems());
            }
        }
    }

    println!("\nTotal time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("Done.");
}
