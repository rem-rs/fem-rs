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
//! cargo run --example mfem_ex15_dynamic_amr -- -m ../data/star.mesh -no-vis
//! cargo run --example mfem_ex15_dynamic_amr -- -m ../data/star.mesh -o 1 -e 0.01 -no-vis -tf 0.05
//! ```

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::amr_refiner::{ThresholdRefiner, ThresholdDerefiner};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_core::ElemId;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{
    NcState2D, NCState, NCStateQuad, HangingNodeConstraint,
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
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    let problem = unsafe { PROBLEM };
    let nfeatures = unsafe { NFEATURES };
    if problem == 0 {
        if nfeatures <= 1 { f0(x, y, z, t) } else {
            let mut sum = 0.0; let two_pi = 2.0 * std::f64::consts::PI;
            for i in 0..nfeatures {
                let x0 = 0.5 * (two_pi * i as f64 / nfeatures as f64).cos();
                let y0 = 0.5 * (two_pi * i as f64 / nfeatures as f64).sin();
                sum += f0(x - x0, y - y0, z, t);
            }
            sum
        }
    } else {
        let mut sum = 0.0; let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..nfeatures {
            let x0 = 0.5 * (two_pi * i as f64 / nfeatures as f64 + std::f64::consts::PI * t).cos();
            let y0 = 0.5 * (two_pi * i as f64 / nfeatures as f64 + std::f64::consts::PI * t).sin();
            sum += f1(x - x0, y - y0, z, 0.25);
        }
        sum
    }
}

fn bdr_func(pt: &[f64], t: f64) -> f64 { composite_func(pt, t, front, ball) }
fn rhs_func(pt: &[f64], t: f64) -> f64 { composite_func(pt, t, |x, y, z, t| front_laplace(x, y, z, t, pt.len() as i32), |x, y, z, t| ball_laplace(x, y, z, t, pt.len() as i32)) }

// ─── NcState2 wrapper (unifies Tri3 NCState + Quad4 NCStateQuad) ─────────────

enum NcState2 {
    Tri3(NCState),
    Quad4(NCStateQuad),
}

impl NcState2D for NcState2 {
    fn refine(&mut self, mesh: &Mesh<2>, marked: &[ElemId], nc_limit: u32)
        -> (Mesh<2>, Vec<HangingNodeConstraint>, std::collections::HashMap<(u32, u32), u32>)
    {
        match self {
            NcState2::Tri3(s) => s.refine(mesh, marked, nc_limit),
            NcState2::Quad4(s) => s.refine(mesh, marked, nc_limit),
        }
    }
    fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        match self {
            NcState2::Tri3(s) => s.derefine_last(),
            NcState2::Quad4(s) => s.derefine_last(),
        }
    }
    fn can_derefine(&self) -> bool {
        match self {
            NcState2::Tri3(s) => s.can_derefine(),
            NcState2::Quad4(s) => s.can_derefine(),
        }
    }
    fn constraints(&self) -> &[HangingNodeConstraint] {
        match self {
            NcState2::Tri3(s) => s.constraints(),
            NcState2::Quad4(s) => s.constraints(),
        }
    }
    fn deref_groups(&self) -> Vec<usize> {
        match self {
            NcState2::Tri3(s) => s.deref_groups(),
            NcState2::Quad4(s) => s.deref_groups(),
        }
    }
    fn deref_group_children(&self, node: usize) -> [ElemId; 4] {
        match self {
            NcState2::Tri3(s) => s.deref_group_children(node),
            NcState2::Quad4(s) => s.deref_group_children(node),
        }
    }
    fn derefine_groups(&mut self, mesh: &Mesh<2>, groups: &[usize]) -> Option<Mesh<2>> {
        match self {
            NcState2::Tri3(s) => s.derefine_groups(mesh, groups),
            NcState2::Quad4(s) => s.derefine_groups(mesh, groups),
        }
    }
}

// ─── P2 hanging-node constraint upgrade ───────────────────────────────────────
// Rust's NCState generates P1-style constraints `u[mid] = 0.5(u[a]+u[b])` at the
// mesh level. For P2 (quadratic H1) spaces MFEM instead constrains the
// *fine-edge midpoint DOFs* (the 1/4 and 3/4 points of a coarse edge) with the
// parent-edge P2 basis (GetTransferMatrix):
//     u[S1] = 0.375 u[a] + 0.75 u[E] - 0.125 u[b]   (1/4 point)
//     u[S2] = -0.125 u[a] + 0.75 u[E] + 0.375 u[b]  (3/4 point)
// and ties the coarse-edge-midpoint vertex DOF `mid` to the coarse-edge-mid
// DOF `E` (same physical point).  `E` remains a free DOF.
fn p2_constraints(
    p1: &[HangingNodeConstraint],
    dm: &fem_space::dof_manager::DofManager,
) -> Vec<HangingNodeConstraint> {
    use fem_space::dof_manager::EdgeKey;
    let mut out: Vec<HangingNodeConstraint> = Vec::new();
    for c in p1 {
        let (mid, a, b) = (c.constrained, c.parent_a, c.parent_b);
        // NOTE: edge_dof_map keys are canonicalized via EdgeKey::new (min,max);
        // constructing EdgeKey(a, b) directly would miss reversed edges.
        let e = dm.edge_dof_map.get(&EdgeKey::new(a as u32, b as u32)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        // mid vertex DOF == coarse-edge midpoint DOF (same point).
        if mid != e {
            out.push(HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![]));
        }
        // fine edge (a, mid): midpoint at the 1/4 point of the coarse edge.
        if let Some(&s1) = dm.edge_dof_map.get(&EdgeKey::new(a as u32, mid as u32)) {
            let s1 = s1 as usize;
            if s1 != mid && s1 != e {
                out.push(HangingNodeConstraint::new_weighted(
                    s1, a, b, 0.375, -0.125, vec![(e, 0.75)],
                ));
            }
        }
        // fine edge (mid, b): midpoint at the 3/4 point of the coarse edge.
        if let Some(&s2) = dm.edge_dof_map.get(&EdgeKey::new(mid as u32, b as u32)) {
            let s2 = s2 as usize;
            if s2 != mid && s2 != e {
                out.push(HangingNodeConstraint::new_weighted(
                    s2, a, b, -0.125, 0.375, vec![(e, 0.75)],
                ));
            }
        }
    }
    out
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

    // ─── 1. Read mesh ─────────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    let elem_type = mesh.element_type(0);
    println!("\nMesh: {} nodes, {} elements, type = {:?}", mesh.n_nodes(), mesh.n_elems(), elem_type);

    let mut mesh = mesh;
    let mut nc_state: NcState2 = match elem_type {
        ElementType::Tri3 => NcState2::Tri3(NCState::new()),
        ElementType::Quad4 => NcState2::Quad4(NCStateQuad::new()),
        _ => panic!("unsupported element type {elem_type:?}"),
    };

    // ─── 2. Uniform refinement (matches C++) ────────────────────────────────
    for _ in 0..args.ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }
    let order = args.order;
    println!("Initial mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ─── 3. Threshold refiner / derefiner ────────────────────────────────────
    let mut refiner = ThresholdRefiner::new(args.estimator == 1);
    refiner.set_local_error_goal(args.max_elem_error);
    refiner.set_nc_limit(args.nc_limit);

    let mut derefiner = ThresholdDerefiner::new();
    derefiner.set_threshold(args.hysteresis * args.max_elem_error);

    // ─── 4. Time loop (C++ ex15.cpp:250: `for (time = 0.0; time < t_final + 1e-10; time += 0.01)`) ──
    let dt = 0.01;
    let mut time = 0.0;
    while time < args.t_final + 1e-10 {
        println!("\nTime {}\n", fmt_g6(time));
        println!("Refinement:");
        // C++ ex15.cpp:259-260 — refiner.Reset(); derefiner.Reset();
        refiner.reset();

        // ── 4a. Inner refinement loop (C++ ex15.cpp:264: `for (int ref_it = 1; ; ref_it++)`) ─
        let mut ref_it = 1usize;
        loop {
            let space = H1Space::new(mesh.clone(), order);
            let cdofs = space.n_dofs();
            // C++ ex15.cpp:266-267 — `cout << "Iteration: " << ref_it << ", number of unknowns: "
            //                          << fespace.GetVSize() << endl;`
            println!("Iteration: {}, number of unknowns: {}", ref_it, cdofs);
            let quad_rule = (order as u8) * 2 + 1;

            // Assemble stiffness matrix
            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_rule);

            // Assemble RHS (time-dependent)
            let rhs_fn = |pt: &[f64]| rhs_func(pt, time);
            let source = DomainSourceIntegrator::new(rhs_fn);
            let mut rhs_vec = Assembler::assemble_linear(&space, &[&source], quad_rule);

            // Apply hanging-node constraints.
            // P2: upgrade the mesh-level P1 constraints (0.5 average) to the
            // MFEM P2 transfer-matrix constraints (fine-edge midpoints).
            let dm0 = space.dof_manager();
            let hc = if order == 2 {
                p2_constraints(nc_state.constraints(), dm0)
            } else {
                nc_state.constraints().to_vec()
            };
            // ── True-DOF system (MFEM ConformingAssemble: R=cPᵀ, RA=R·A,
            //    A_true=RA·cP, b_true=R·b).  This reproduces MFEM's
            //    GSSmoother sweep order and PCG history bit-for-bit.
            let (mat_true, rhs_true, true_dofs) =
                fem_space::constraints::conforming_assemble(&mat, &rhs_vec, &hc);

            // Dirichlet BC on all boundaries (time-dependent), restricted to
            // true DOFs (C++: ess_tdof_list = GetEssentialTrueDofs).
            // C++ ex15.cpp:275 — `x.ProjectBdrCoefficient(bdr, ess_bdr)`
            // (DIAG_KEEP: diagonal kept, rhs_i = A_ii · bdr_val)
            let dm = space.dof_manager();
            let bnd_tags = space.mesh().unique_boundary_tags();
            let bnd_all = boundary_dofs(space.mesh(), dm, &bnd_tags);
            let true_set: std::collections::HashSet<usize> = true_dofs.iter().copied().collect();
            let true_idx: std::collections::HashMap<usize, usize> = true_dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| (d, i))
                .collect();
            let mut mat_true = mat_true;
            let mut rhs_true = rhs_true;
            // boundary DOFs, mapped to true-DOF (compressed) indices
            let bnd_vals: Vec<f64> = bnd_all
                .iter()
                .filter(|d| true_set.contains(&(**d as usize)))
                .map(|&dof| {
                    let coord = dm.dof_coord(dof);
                    bdr_func(&coord, time)
                })
                .collect();
            let bnd: Vec<u32> = bnd_all
                .iter()
                .filter(|d| true_set.contains(&(**d as usize)))
                .map(|&d| true_idx[&(d as usize)] as u32)
                .collect();
            fem_space::constraints::apply_dirichlet(&mut mat_true, &mut rhs_true, &bnd, &bnd_vals);

            // Solve (C++ ex15.cpp:286-287 — `GSSmoother M(A); PCG(A, M, B, X, 0, 500, 1e-12, 0.0)`)
            let mut u = vec![0.0_f64; cdofs];
            let mut x_true = vec![0.0_f64; true_dofs.len()];
            let res = fem_solver::solve_pcg_gssmoother(
                &mat_true, &rhs_true, &mut x_true,
                &fem_solver::SolverConfig {
                    rtol: 1e-12, max_iter: 500, verbose: false, ..Default::default()
                },
            );
            if res.is_err() { break; }
            if res.as_ref().is_ok_and(|r| !r.converged) { break; }
            // Expand true-DOF solution back to the full vector (RecoverFEMSolution).
            for (&td, &v) in true_dofs.iter().zip(x_true.iter()) {
                u[td] = v;
            }

            // Recover hanging-node values
            if !hc.is_empty() { recover_hanging_values(&mut u, &hc); }

            // Error estimation + refinement (matches C++ refiner.Apply(*mesh))
            let gf = GridFunction::new(&space, u);
            let ne_before = mesh.n_elems();
            refiner.apply(&mut mesh, &mut nc_state, &gf, &diffusion);
            let ne_after = mesh.n_elems();
            let n_marked = if ne_after > ne_before {
                refiner.last_marked.len()
            } else { 0 };

            // C++ ex15.cpp:317-320 — `if (refiner.Stop()) break;`
            if refiner.stop() { break; }
            ref_it += 1;
        }

        // ── 4b. Derefinement (C++ ex15.cpp:330-336: `if (derefiner.Apply(mesh))`
        //          → `cout << "\nDerefined elements." << endl;`)
        // MFEM ThresholdDerefiner::ApplyImpl → Mesh::DerefineByError: coarsen
        // every derefinement-table group whose children's summed error is
        // below the threshold (uses the eta of the last inner iteration).
        if derefiner.apply(&mut mesh, &mut nc_state, &mut refiner) {
            println!("\nDerefined elements.");
        }

        time += dt;
    }

    println!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

// ─── CLI ───────────────────────────────────────────────────────────────────────

/// Mimic C++ `std::cout << v` under the default `cout.precision(6)`
/// (defaultfloat, i.e. printf-style `%g` with 6 significant digits).
/// C++ ex15.cpp prints `time` with the default stream format.
fn fmt_g6(v: f64) -> String {
    let p = 6usize; // significant digits
    let sci = format!("{:.5e}", v); // 5 decimals = 6 significant digits
    let (mant, exp) = sci.split_once('e').expect("sci format");
    let exp: i32 = exp.parse().expect("exp");
    let neg = mant.starts_with('-');
    let mant = mant.trim_start_matches('-');
    let mut digits: Vec<char> = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    while digits.len() > 1 && digits[digits.len() - 1] == '0' {
        digits.pop();
    }
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp >= -4 && exp < p as i32 {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if int_len >= digits.len() {
                out.push_str(&digits.iter().collect::<String>());
                out.push_str(&"0".repeat(int_len - digits.len()));
            } else {
                out.push_str(&digits[..int_len].iter().collect::<String>());
                out.push('.');
                out.push_str(&digits[int_len..].iter().collect::<String>());
            }
        } else {
            out.push('0');
            out.push('.');
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits.iter().collect::<String>());
        }
    } else {
        out.push(digits[0]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..].iter().collect::<String>());
        }
        out.push('e');
        if exp < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        let e = exp.abs();
        if e < 10 {
            out.push('0');
        }
        out.push_str(&e.to_string());
    }
    out
}

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
        // C++ ex15.cpp:78-89 默认参数
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
                "-r" | "--ref-levels" | "-rs" | "--refine-serial" => { if let Some(v) = it.next() { ref_levels = v.parse().unwrap_or(0); } }
                "-l" | "--nc-limit" => { if let Some(v) = it.next() { nc_limit = v.parse().unwrap_or(3); } }
                "-tf" | "--t-final" => { if let Some(v) = it.next() { t_final = v.parse().unwrap_or(1.0); } }
                "-est" | "--estimator" => { if let Some(v) = it.next() { estimator = v.parse().unwrap_or(0); } }
                "-no-vis" | "--no-visualization" => { /* accepted but ignored */ }
                "-vis" | "--visualization" => { /* ignored, no GLVis */ }
                _ => { /* ignore unknown */ }
            }
        }

        Args { mesh, problem, nfeatures, order, max_elem_error, hysteresis, ref_levels, nc_limit, t_final, estimator }
    }
}
