//! # MFEM Example 6 — Adaptive Mesh Refinement for Poisson
//!
//! Solves `-Δu = 1` with homogeneous Dirichlet BC using an adaptive mesh
//! refinement (AMR) loop with a Zienkiewicz–Zhu (ZZ) error estimator.
//!
//! Supports **Tri3** (conforming closure refinement via longest-edge bisection)
//! and **Quad4** (non-conforming refinement with hanging-node constraints).
//!
//! Reference: `mfem/ex6.cpp`  (AMR Poisson with `f = 1`)
//!
//! ## Usage
//! ```bash
//! # Default: star.mesh, order 1, max DOFs 50000 (Quad4 — non-conforming AMR)
//! cargo run --example mfem_ex6_flux_recovery 2>&1
//!
//! # Triangle mesh (conforming AMR)
//! cargo run --example mfem_ex6_flux_recovery -- -m data/square-disc.mesh -o 1 -no-vis
//!
//! # Custom order and max DOFs
//! cargo run --example mfem_ex6_flux_recovery -- -m data/square-disc.mesh -o 2 -md 100000 -no-vis
//! ```

use std::collections::HashMap;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::error_estimate::zz_estimator_l2;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_core::NodeId;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{
    closure_refine_default, dorfler_mark, prolongate_p1,
    refine_nonconforming_quad, zz_estimator, HangingNodeConstraint,
};
use fem_solver::{fem_to_linlvo_csr, solve_pcg};
use fem_space::{
    H1Space,
    constraints::{
        apply_dirichlet, apply_hanging_constraints, boundary_dofs, recover_hanging_values,
    },
    fe_space::FESpace,
};
use fem_solver::GSSmoother;

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();

    // ─── 1. Read mesh ──────────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        eprintln!("  Mesh file: {}", args.mesh);
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    eprintln!(
        "  Mesh: {} nodes, {} elements, type = {:?}",
        mesh.n_nodes(),
        mesh.n_elems(),
        mesh.element_type(0),
    );

    let elem_type = mesh.element_type(0);
    let is_quad = elem_type == ElementType::Quad4;
    if elem_type != ElementType::Tri3 && elem_type != ElementType::Quad4 {
        panic!(
            "Unsupported element type {:?}: only Tri3 and Quad4 meshes are supported",
            elem_type
        );
    }

    // ─── 2. AMR loop ──────────────────────────────────────────────────────────
    let mut mesh = mesh;
    let mut prev_u: Option<Vec<f64>> = None;
    let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();
    let order = args.order;
    let max_dofs = args.max_dofs;

    for it in 0.. {
        // --- 2a. Build H¹ space ---
        let space = H1Space::new(mesh.clone(), order);
        let cdofs = space.n_dofs();
        println!("\nAMR iteration {}", it);
        println!("Number of unknowns: {}", cdofs);

        // --- 2b. Assemble stiffness + RHS (f = 1) ---
        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let quad = (order as u8) * 2 + 1;

        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

        // --- 2c. Apply hanging-node constraints (Quad4 NC path) ---
        if !hanging_constraints.is_empty() {
            apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        }

        // --- 2d. Homogeneous Dirichlet BC on all boundaries ---
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // --- 2e. Solve: PCG + SSOR preconditioner (matches MFEM GSSmoother) ---
        let mut u = prev_u.take().unwrap_or_else(|| vec![0.0_f64; cdofs]);
        u.resize(cdofs, 0.0);

        let la = fem_to_linlvo_csr(&mat);
        let prec = GSSmoother::from_csr(&la, 1.0).expect("GSSmoother");

        let res = solve_pcg(
            &mat, &rhs, &mut u, &prec,
            1e-12,  // rtol (matches MFEM: 1e-12)
            5000,   // max_iter
            true,   // verbose — prints (B r, r) per iteration + Average reduction factor
        );

        if let Err(e) = &res {
            eprintln!("  Solver error: {e:?}");
            break;
        }
        if let Ok(r) = &res {
            if !r.converged {
                eprintln!(
                    "  WARNING: solver did not converge (iters={}, res={:.3e})",
                    r.iterations, r.final_residual
                );
            }
        }

        // --- 2f. Recover hanging-node DOF values (Quad4 NC path) ---
        if !hanging_constraints.is_empty() {
            recover_hanging_values(&mut u, &hanging_constraints);
        }

        // --- 2g. Check max DOFs termination ---
        if cdofs > max_dofs {
            println!("Reached the maximum number of dofs. Stop.");
            break;
        }

        // --- 2h. ZZ error estimator ---
        // Tri3: L² projection recovery (MFEM-compatible).
        // Quad4: mesh-level nodal-averaging (GridFunction doesn't support Quad4).
        let eta = if is_quad {
            zz_estimator(&mesh, &u)
        } else {
            let gf = GridFunction::new(&space, u.clone());
            zz_estimator_l2(&gf).eta
        };

        // --- 2i. Dörfler marking (θ = 0.7) ---
        // Equivalent to MFEM's ThresholdRefiner::SetTotalErrorFraction(0.7)
        let marked = dorfler_mark(&eta, 0.7);

        if marked.is_empty() {
            println!("Stopping criterion satisfied. Stop.");
            break;
        }

        // --- 2j. Refine mesh ---
        if is_quad {
            // Non-conforming Quad4 refinement
            let (new_mesh, new_constraints) = refine_nonconforming_quad(&mesh, &marked);
            hanging_constraints = merge_hanging_constraints(&hanging_constraints, &new_constraints, &new_mesh);
            mesh = new_mesh;
            // For NC Quad4: no prolongation (solution zeroed on refined elements)
            prev_u = None;
        } else {
            // Conforming Tri3 closure refinement
            let new_mesh = closure_refine_default(&mesh, &marked);
            let mid_map = build_edge_midpoint_map(&mesh, &new_mesh);
            prev_u = Some(prolongate_p1(&u, new_mesh.n_nodes(), &mid_map));
            mesh = new_mesh;
        }
    }

    eprintln!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── Hanging constraint merge ─────────────────────────────────────────────────

/// Merge old hanging-node constraints with new ones, keeping only those that
/// are still valid after refinement.
///
/// A constraint on edge `(pa, pb)` with midpoint `mid` is kept if the new mesh
/// still has at least one element on that edge that does **not** contain the
/// midpoint (i.e., the hanging node has not been resolved by subsequent
/// refinement of the adjacent coarse element).
fn merge_hanging_constraints(
    old: &[HangingNodeConstraint],
    new: &[HangingNodeConstraint],
    new_mesh: &Mesh<2>,
) -> Vec<HangingNodeConstraint> {
    // Build edge → elements map for the new mesh
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

    // Keep old constraints that are still valid
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

    // Add new constraints (avoid duplicates)
    for c in new {
        if !merged.iter().any(|oc| oc.constrained == c.constrained) {
            merged.push(c.clone());
        }
    }

    merged.sort_by_key(|c| c.constrained);
    merged
}

// ─── Midpoint map helper (Tri3 conforming path) ──────────────────────────────

/// Build a map from old-mesh edge → new-mesh node ID for every edge that was
/// bisected during conforming closure refinement.
///
/// Works by matching new-node coordinates against old-edge midpoints to within
/// `1e-12`.  Refinement only appends nodes (never reorders), so every new node
/// with ID ≥ `old.n_nodes()` is a candidate.
fn build_edge_midpoint_map(
    old: &Mesh<2>,
    new: &Mesh<2>,
) -> HashMap<(NodeId, NodeId), NodeId> {
    let old_n = old.n_nodes();
    let mut map = HashMap::new();

    // Collect all unique old-mesh edges.
    let mut old_edges: Vec<(NodeId, NodeId)> = Vec::new();
    for e in 0..old.n_elems() as NodeId {
        let ns = old.elem_nodes(e);
        for &(a, b) in &[(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if !old_edges.contains(&key) {
                old_edges.push(key);
            }
        }
    }

    // Collect all new nodes (only those added by refinement).
    let mut new_nodes: Vec<(NodeId, [f64; 2])> = Vec::new();
    for nid in (old_n as NodeId)..(new.n_nodes() as NodeId) {
        new_nodes.push((nid, new.coords_of(nid)));
    }

    // For each old edge, search for a new node at its midpoint.
    for &(a, b) in &old_edges {
        let pa = old.coords_of(a);
        let pb = old.coords_of(b);
        let mx = 0.5 * (pa[0] + pb[0]);
        let my = 0.5 * (pa[1] + pb[1]);
        for &(nid, p) in &new_nodes {
            if (p[0] - mx).abs() < 1e-12 && (p[1] - my).abs() < 1e-12 {
                map.insert((a, b), nid);
                break;
            }
        }
    }

    map
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh: String,
    order: u8,
    max_dofs: usize,
    #[allow(dead_code)]
    no_vis: bool,
}

impl Args {
    fn parse() -> Self {
        let mut mesh = "data/star.mesh".to_string();
        let mut order: u8 = 1;
        let mut max_dofs: usize = 50000;
        let mut no_vis = false;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => {
                    if let Some(v) = it.next() {
                        mesh = v;
                    }
                }
                "-o" | "--order" => {
                    if let Some(v) = it.next() {
                        order = v.parse().unwrap_or(1);
                    }
                }
                "-md" | "--max-dofs" => {
                    if let Some(v) = it.next() {
                        max_dofs = v.parse().unwrap_or(50000);
                    }
                }
                "-no-vis" | "--no-visualization" => {
                    no_vis = true;
                }
                "-ls" | "--ls-zz" => {
                    // Accepted but not yet implemented.
                    eprintln!("  (LS-ZZ estimator not yet implemented, using ZZ)");
                }
                _ => {
                    // Ignore unknown flags for compatibility.
                }
            }
        }

        Args {
            mesh,
            order,
            max_dofs,
            no_vis,
        }
    }
}

// ─── Tests (MMS exact-solution verification) ───────────────────────────────

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use fem_assembly::postprocess::compute_h1_error;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::Assembler;
    use fem_mesh::Mesh;
    use fem_mesh::topology::MeshTopology;
    use fem_solver::{solve_pcg_jacobi, SolverConfig};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    /// Manufactured exact solution: u = sin(πx) sin(πy)
    fn exact(x: &[f64]) -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// RHS for the manufactured solution: -Δu = 2π² sin(πx) sin(πy)
    fn rhs_mms(x: &[f64]) -> f64 {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// Exact gradient: ∇u
    #[allow(dead_code)]
    fn grad_exact(x: &[f64]) -> Vec<f64> {
        vec![
            PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
            PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
        ]
    }

    fn solve_mms(n: usize, order: u8) -> (Vec<f64>, H1Space<Mesh<2>>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, order);
        let ndofs = space.n_dofs();
        let quad = order as u8 * 2 + 1;

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let src = DomainSourceIntegrator::new(rhs_mms);
        let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad);
        let mut rhs = Assembler::assemble_linear(&space, &[&src], quad);

        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals: Vec<f64> = bnd
            .iter()
            .map(|&dof| {
                let x = dm.dof_coord(dof);
                exact(&x)
            })
            .collect();
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        let mut u = vec![0.0_f64; ndofs];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 5_000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("PCG solve");
        (u, space)
    }

    /// L² error against an exact function.
    fn l2_error<S: FESpace>(
        space: &S,
        dofs: &[f64],
        exact: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        use fem_element::lagrange::TriP1;
        use fem_element::ReferenceElement;

        let mesh = space.mesh();
        let mut err2 = 0.0;
        for e in mesh.elem_iter() {
            let ref_elem = TriP1;
            let n_ldofs = ref_elem.n_dofs();
            let elem_dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                - (x1[1] - x0[1]) * (x2[0] - x0[0]))
            .abs();
            let q = ref_elem.quadrature(6);
            let mut basis = vec![0.0; n_ldofs];
            for (qi, xi) in q.points.iter().enumerate() {
                let w = q.weights[qi] * det_j;
                let x_phys = [
                    x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                    x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
                ];
                ref_elem.eval_basis(xi, &mut basis);
                let mut uh = 0.0;
                for i in 0..n_ldofs {
                    uh += basis[i] * dofs[elem_dofs[i] as usize];
                }
                let ue = exact(&x_phys);
                err2 += w * (uh - ue).powi(2);
            }
        }
        err2.sqrt()
    }

    #[test]
    fn ex6_mms_l2_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let err_c = l2_error(&sp_c, &u_c, exact);
        let err_f = l2_error(&sp_f, &u_f, exact);
        eprintln!("  L2 coarse={:.6e} fine={:.6e}", err_c, err_f);
        assert!(err_f < err_c, "L2 error must decrease on refinement");
        let rate = (err_f / err_c).ln() / (32.0_f64 / 16.0_f64).ln();
        assert!(rate < -1.8, "L2 convergence rate {:.2} too slow", rate);
    }

    #[test]
    fn ex6_mms_h1_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let h1_c = compute_h1_error(&sp_c, &u_c, grad_exact, 6);
        let h1_f = compute_h1_error(&sp_f, &u_f, grad_exact, 6);
        eprintln!("  H1 coarse={:.6e} fine={:.6e}", h1_c, h1_f);
        assert!(h1_f < h1_c, "H1 error must decrease on refinement");
    }

    #[test]
    fn ex6_mms_flux_recovery_error_small() {
        let (u, space) = solve_mms(32, 2);
        let grad_nodal =
            fem_assembly::postprocess::recover_gradient_nodal(&space, &u);
        let nv = space.mesh().n_nodes();
        let flux_err: f64 = (0..nv)
            .map(|node| {
                let x = space.mesh().node_coords(node as u32);
                let ex = PI * (PI * x[0]).cos() * (PI * x[1]).sin();
                let ey = PI * (PI * x[0]).sin() * (PI * x[1]).cos();
                let dx = grad_nodal[0][node] - ex;
                let dy = grad_nodal[1][node] - ey;
                dx * dx + dy * dy
            })
            .sum::<f64>()
            .sqrt()
            / nv as f64;
        eprintln!("  Flux error (nodal RMS) = {flux_err:.6e}");
        assert!(
            flux_err < 1e-2,
            "flux recovery error too large: {flux_err:.6e}"
        );
    }
}
