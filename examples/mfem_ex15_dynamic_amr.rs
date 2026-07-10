//! # MFEM Example 15 — Dynamic AMR for Poisson (1:1 translation)
//!
//! Time-dependent Poisson with prescribed solution (spherical front / ball),
//! adaptive mesh refinement (threshold-based), and derefinement.
//!
//! Supports **Quad4** (non-conforming AMR with `NCStateQuad`) and **Tri3**
//! (non-conforming AMR with `NCState`).
//!
//! Reference: `mfem/ex15.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex15_amr_poisson -- -m data/star-hilbert.mesh -no-vis
//! cargo run --example mfem_ex15_amr_poisson -- -m data/amr-quad.mesh -o 2 -e 0.005 -no-vis
//! cargo run --example mfem_ex15_amr_poisson -- -m data/star-hilbert.mesh -o 1 -e 0.01 -no-vis -tf 0.05
//! ````

use std::collections::HashMap;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::error_estimate::{threshold_mark, derefine_mark, kelly_estimator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_core::{ElemId, NodeId};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{
    NCState, NCStateQuad, HangingNodeConstraint,
};

use fem_space::{
    H1Space,
    constraints::{
        apply_hanging_constraints, boundary_dofs, recover_hanging_values,
    },
    fe_space::FESpace,
};

// ─── Problem parameters ────────────────────────────────────────────────────────

const ALPHA: f64 = 0.02;
static mut PROBLEM: i32 = 0;
static mut NFEATURES: i32 = 1;

// Spherical front with a Gaussian cross section and radius t
fn front(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    (-0.5 * ((r - t) / ALPHA).powi(2)).exp()
}

fn front_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let x2 = x * x; let y2 = y * y; let z2 = z * z; let t2 = t * t;
    let r = (x2 + y2 + z2).sqrt();
    let a2 = ALPHA * ALPHA; let a4 = a2 * a2;
    let r_term = if r < 1e-30 { 0.0 } else { -2.0 * t * (x2 + y2 + z2 - (dim as f64 - 1.0) * a2 / 2.0) / r };
    -(-0.5 * ((r - t) / ALPHA).powi(2)).exp() / a4 * (r_term + x2 + y2 + z2 + t2 - dim as f64 * a2)
}

// Smooth step function (arctan-based)
fn ball(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    -(2.0 * (r - t) / ALPHA).atan()
}

fn ball_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    let a2 = ALPHA * ALPHA;
    let t2 = 4.0 * t * t;
    let denom = (-a2 - 4.0 * (x * x + y * y + z * z - 2.0 * r * t) - t2).powi(2);
    if denom.abs() < 1e-30 { return 0.0; }
    if dim == 2 {
        2.0 * ALPHA * (a2 + t2 - 4.0 * x * x - 4.0 * y * y) / r / denom
    } else {
        4.0 * ALPHA * (a2 + t2 - 4.0 * r * t) / r / denom
    }
}

fn composite_func<F0, F1>(pt: &[f64], t: f64, f0: F0, f1: F1) -> f64
where
    F0: Fn(f64, f64, f64, f64) -> f64,
    F1: Fn(f64, f64, f64, f64) -> f64,
{
    let dim = pt.len();
    let x = pt[0]; let y = pt[1]; let z = if dim == 3 { pt[2] } else { 0.0 };
    let problem = unsafe { PROBLEM };
    let nfeatures = unsafe { NFEATURES };

    if problem == 0 {
        if nfeatures <= 1 {
            f0(x, y, z, t)
        } else {
            let mut sum = 0.0;
            let two_pi = 2.0 * std::f64::consts::PI;
            for i in 0..nfeatures {
                let x0 = 0.5 * (two_pi * i as f64 / nfeatures as f64).cos();
                let y0 = 0.5 * (two_pi * i as f64 / nfeatures as f64).sin();
                sum += f0(x - x0, y - y0, z, t);
            }
            sum
        }
    } else {
        let mut sum = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..nfeatures {
            let x0 = 0.5 * (two_pi * i as f64 / nfeatures as f64 + std::f64::consts::PI * t).cos();
            let y0 = 0.5 * (two_pi * i as f64 / nfeatures as f64 + std::f64::consts::PI * t).sin();
            sum += f1(x - x0, y - y0, z, 0.25);
        }
        sum
    }
}

fn bdr_func(pt: &[f64], t: f64) -> f64 {
    composite_func(pt, t, front, ball)
}

fn rhs_func(pt: &[f64], t: f64) -> f64 {
    composite_func(pt, t, |x, y, z, t| front_laplace(x, y, z, t, pt.len() as i32), |x, y, z, t| ball_laplace(x, y, z, t, pt.len() as i32))
}

// ─── Mesh element type check ──────────────────────────────────────────────────

enum NcState2 {
    Tri3(NCState),
    Quad4(NCStateQuad),
}

impl NcState2 {
    fn new(elem_type: ElementType) -> Self {
        match elem_type {
            ElementType::Tri3 => NcState2::Tri3(NCState::new()),
            ElementType::Quad4 => NcState2::Quad4(NCStateQuad::new()),
            _ => panic!("Unsupported element type {:?}: only Tri3 and Quad4 are supported", elem_type),
        }
    }

    #[allow(dead_code)]
    fn constraints(&self) -> &[HangingNodeConstraint] {
        match self {
            NcState2::Tri3(s) => s.constraints(),
            NcState2::Quad4(s) => s.constraints(),
        }
    }

    fn can_derefine(&self) -> bool {
        match self {
            NcState2::Tri3(s) => s.can_derefine(),
            NcState2::Quad4(s) => s.can_derefine(),
        }
    }

    fn refine(&mut self, mesh: &Mesh<2>, marked: &[ElemId]) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
        match self {
            NcState2::Tri3(s) => {
                let (new_mesh, constraints, _) = s.refine(mesh, marked);
                (new_mesh, constraints)
            }
            NcState2::Quad4(s) => {
                let (new_mesh, constraints, _) = s.refine(mesh, marked);
                (new_mesh, constraints)
            }
        }
    }

    fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        match self {
            NcState2::Tri3(s) => s.derefine_last(),
            NcState2::Quad4(s) => s.derefine_last(),
        }
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    let args = Args::parse();
    unsafe { PROBLEM = args.problem; }
    unsafe { NFEATURES = args.nfeatures; }

    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --problem {}", args.problem);
    println!("   --nfeatures {}", args.nfeatures);
    println!("   --order {}", args.order);
    println!("   --max-err {}", args.max_elem_error);
    println!("   --hysteresis {}", args.hysteresis);
    println!("   --ref-levels {}", args.ref_levels);
    println!("   --nc-limit {}", args.nc_limit);
    println!("   --t-final {}", args.t_final);
    println!("   --estimator {}", args.estimator);
    println!("   --no-visualization");
    println!("   --no-visit-datafiles");

    // ─── 1. Read mesh ─────────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    let _dim = fem_mesh::topology::MeshTopology::dim(&mesh);
    let elem_type = mesh.element_type(0);

    println!("\nMesh: {} nodes, {} elements, type = {:?}", mesh.n_nodes(), mesh.n_elems(), elem_type);
    if elem_type != ElementType::Tri3 && elem_type != ElementType::Quad4 {
        panic!("Unsupported element type {:?}: only Tri3 and Quad4", elem_type);
    }

    let mesh = mesh;
    let mut mesh = mesh;

    // ─── 2. Uniform refinement (matches C++: NURBS path + initial refs) ──────
    for _ in 0..args.ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }
    // For NC AMR: we need to work with a conforming mesh first, then refine.
    // The initial mesh needs to be non-conforming-capable.
    let mut nc_state = NcState2::new(elem_type);
    let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();

    println!("Initial mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ─── 3. Solver setup ─────────────────────────────────────────────────────
    let order = args.order;

    // Error estimator selection
    let use_kelly = args.estimator == 1;

    // ─── 4. Outer time loop ──────────────────────────────────────────────────
    let dt = 0.01;
    let t_final = args.t_final;
    let max_elem_error = args.max_elem_error;
    let hysteresis = args.hysteresis;
    let derefine_threshold = hysteresis * max_elem_error;
    let _nc_limit = args.nc_limit;

    let mut x = Vec::new();

    let mut time = 0.0;
    while time < t_final + 1e-10 {
        println!("\nTime {}", time);
        println!("\nRefinement:");

        // ─── 4a. Inner refinement loop ──────────────────────────────────────
        for ref_it in 1.. {
            // Build H1 space on current mesh
            let space = H1Space::new(mesh.clone(), order);
            let cdofs = space.n_dofs();
            println!("Iteration: {}, number of unknowns: {}", ref_it, cdofs);

            // Assemble stiffness matrix
            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let quad_rule = (order as u8) * 2 + 1;

            let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);

            // Assemble RHS (time-dependent)
            let rhs_fn = |pt: &[f64]| rhs_func(pt, time);
            let source = DomainSourceIntegrator::new(rhs_fn);
            let mut rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);

            // Apply hanging-node constraints
            if !hanging_constraints.is_empty() {
                apply_hanging_constraints(&mut mat, &mut rhs_vec, &hanging_constraints);
            }

            // Dirichlet BC on all boundaries (time-dependent)
            let dm = space.dof_manager();
            let bnd_tags = space.mesh().unique_boundary_tags();
            let bnd = boundary_dofs(space.mesh(), dm, &bnd_tags);
            let bnd_vals: Vec<f64> = bnd
                .iter()
                .map(|&dof| {
                    let coord = dm.dof_coord(dof);
                    bdr_func(&coord, time)
                })
                .collect();

            // Apply Dirichlet BC in-place (like MFEM FormLinearSystem)
            fem_space::constraints::apply_dirichlet(&mut mat, &mut rhs_vec, &bnd, &bnd_vals);

            // Fix zero diagonals: after apply_dirichlet, any row without a diagonal
            // entry (e.g. isolated NC DOFs) gets its diag forced to 1.0.
            //
            // We detect these by checking which boundary DOFs still have A[i,i]=0
            // (the only rows that should have zero diag after apply_dirichlet are
            // non-boundary rows that happen to have no element support — rare but
            // possible with NC meshes).
            let diag_vals = mat.diagonal();
            for (row, &d) in diag_vals.iter().enumerate() {
                if d == 0.0 {
                    mat.eliminate_essential_bc_diag(row, 1.0);
                }
            }

            // Solve with GSSmoother (matches MFEM GSSmoother)
            let mut u = vec![0.0_f64; cdofs];
            let res = fem_solver::solve_pcg_gssmoother(
                &mat, &rhs_vec, &mut u,
                &fem_solver::SolverConfig {
                    rtol: 1e-12,
                    max_iter: 500,
                    verbose: false,
                    ..Default::default()
                },
            );

            if let Err(e) = &res {
                eprintln!("  Solver error: {e:?}");
                break;
            }
            if let Ok(r) = &res {
                if !r.converged {
                    eprintln!("  WARNING: solver did not converge (iters={}, res={:.3e})", r.iterations, r.final_residual);
                }
            }

            // Recover hanging-node values
            if !hanging_constraints.is_empty() {
                recover_hanging_values(&mut u, &hanging_constraints);
            }

            x = u;

            // Error estimation (mesh-level zz_estimator — matches MFEM ex6 path)
            let eta = if use_kelly {
                let gf = GridFunction::new(&space, x.clone());
                kelly_estimator(&gf).eta
            } else {
                // Use vertex-only values for mesh-level estimator
                let u_vert: Vec<f64> = (0..mesh.n_nodes()).map(|n| x[n as usize]).collect();
                fem_mesh::amr::zz_estimator(&mesh, &u_vert)
            };

            // Threshold marking: refine if η > max_elem_error
            let marked = threshold_mark(&eta, max_elem_error);

            if marked.is_empty() {
                break;
            }

            // Apply NC refinement
            let (new_mesh, new_constraints) = nc_state.refine(&mesh, &marked);
            hanging_constraints = merge_hanging_constraints(&hanging_constraints, &new_constraints, &new_mesh);
            mesh = new_mesh;
        }

        // ─── 4b. Derefinement step ─────────────────────────────────────────
        if nc_state.can_derefine() {
            // Compute error on final refined mesh
            let space = H1Space::new(mesh.clone(), order);
            let gf = GridFunction::new(&space, x.clone());
            let eta = if use_kelly {
                kelly_estimator(&gf).eta
            } else {
                fem_assembly::postproc::error_estimate::zz_estimator_nodal(&gf).eta
            };

            let derefine_candidates = derefine_mark(&eta, derefine_threshold);
            let frac_below = derefine_candidates.len() as f64 / eta.len().max(1) as f64;

            // Derefine if at least 25% of elements have error below threshold
            // (conservative approximation of C++ ThresholdDerefiner behavior)
            if frac_below > 0.25 && nc_state.can_derefine() {
                if let Some((old_mesh, old_constraints)) = nc_state.derefine_last() {
                    mesh = old_mesh;
                    hanging_constraints = old_constraints;
                    println!("\nDerefined elements.");
                }
            }
        }

        time += dt;
    }

    println!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

// ─── Hanging constraint merge (from ex6) ───────────────────────────────────────

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

// ─── CLI ───────────────────────────────────────────────────────────────────────

struct Args {
    mesh: String,
    problem: i32,
    nfeatures: i32,
    order: u8,
    max_elem_error: f64,
    hysteresis: f64,
    ref_levels: u32,
    nc_limit: u32,
    t_final: f64,
    estimator: i32,
}

impl Args {
    fn parse() -> Self {
        let mut mesh = "data/star-hilbert.mesh".to_string();
        let mut problem = 0;
        let mut nfeatures = 1;
        let mut order: u8 = 2;
        let mut max_elem_error = 5.0e-3;
        let mut hysteresis = 0.15;
        let mut ref_levels = 0;
        let mut nc_limit = 3;
        let mut t_final = 1.0;
        let mut estimator = 0;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => { if let Some(v) = it.next() { mesh = v; } }
                "-p" | "--problem" => { if let Some(v) = it.next() { problem = v.parse().unwrap_or(0); } }
                "-n" | "--nfeatures" => { if let Some(v) = it.next() { nfeatures = v.parse().unwrap_or(1); } }
                "-o" | "--order" => { if let Some(v) = it.next() { order = v.parse().unwrap_or(2); } }
                "-e" | "--max-err" => { if let Some(v) = it.next() { max_elem_error = v.parse().unwrap_or(5.0e-3); } }
                "-y" | "--hysteresis" => { if let Some(v) = it.next() { hysteresis = v.parse().unwrap_or(0.15); } }
                "-r" | "--ref-levels" => { if let Some(v) = it.next() { ref_levels = v.parse().unwrap_or(0); } }
                "-l" | "--nc-limit" => { if let Some(v) = it.next() { nc_limit = v.parse().unwrap_or(3); } }
                "-tf" | "--t-final" => { if let Some(v) = it.next() { t_final = v.parse().unwrap_or(1.0); } }
                "-est" | "--estimator" => { if let Some(v) = it.next() { estimator = v.parse().unwrap_or(0); } }
                "-no-vis" | "--no-visualization" => { /* accepted but ignored */ }
                "-vis" | "--visualization" => { /* ignored, no GLVis */ }
                _ => { /* ignore unknown */ }
            }
        }

        Args {
            mesh,
            problem,
            nfeatures,
            order,
            max_elem_error,
            hysteresis,
            ref_levels,
            nc_limit,
            t_final,
            estimator,
        }
    }
}
