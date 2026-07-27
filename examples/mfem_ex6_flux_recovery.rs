//                                MFEM Example 6
//
// 1:1 Rust translation of MFEM C++ ex6.cpp — AMR Poisson with ZZ estimator.
//
// Compile: cargo run --example mfem_ex6_flux_recovery -- -m data/square-disc.mesh -o 1 -no-vis
//
// Description: This is a version of Example 1 with a simple adaptive mesh
//              refinement loop. The problem being solved is again the Poisson
//              equation -Delta u = 1 with homogeneous Dirichlet boundary
//              conditions. The problem is solved on a sequence of meshes which
//              are locally refined in a conforming (triangles, tetrahedrons)
//              or non-conforming (quadrilaterals, hexahedra) manner according
//              to a simple ZZ error estimator.
//
// Reference: mfem/examples/ex6.cpp

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{closure_refine_default, refine_nonconforming_quad, HangingNodeConstraint};
use fem_linalg::PrintLevel;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::{
    H1Space,
    constraints::{boundary_dofs, eliminate_dirichlet, expand_from_reduced},
    fe_space::FESpace,
};

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── 1. Read the mesh ──────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    let dim = mesh.dim() as usize;
    let elem_type = mesh.element_type(0);
    let is_quad = matches!(elem_type, ElementType::Quad4);

    // ── 2. Define H1 FE space ────────────────────────────────────────────────
    let order = args.order;
    let mut space = H1Space::new(mesh.clone(), order);

    // ── 3. Set up bilinear form a(u,v) = ∫ ∇u·∇v dx ──────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let quad_stiff = (order as u8) * 2;

    // ── 4. Set up linear form b(v) = ∫ 1·v dx ────────────────────────────────
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let quad_rhs = (order as u8) * 2 + 1;

    // ── 5. Initialize solution vector u (persistent across AMR iterations) ────
    let mut u = vec![0.0; space.n_dofs()];
    let mut prev_mesh: Option<Mesh<2>> = None;

    // ── 6. BCs on all boundaries ─────────────────────────────────────────────

    // ── 7. AMR loop ──────────────────────────────────────────────────────────
    let max_dofs = args.max_dofs;
    let mut mesh = mesh;
    let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();

    for it in 0.. {
        // Build space on current mesh.
        space = H1Space::new(mesh.clone(), order);
        let cdofs = space.n_dofs();

        println!("\nAMR iteration {}", it);
        println!("Number of unknowns: {}", cdofs);

        // Assemble RHS: b(v) = ∫ 1·v dx.
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad_rhs);

        // Get boundary DOFs.
        let dm = space.dof_manager();
        let bnd = boundary_dofs(&mesh, dm, &mesh.unique_boundary_tags());
        let bnd_vals = vec![0.0; bnd.len()];

        // Assemble stiffness matrix.
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_stiff);

        // Apply hanging-node constraints (Quad4 NC path).
        if !hanging_constraints.is_empty() {
            use fem_space::constraints::apply_hanging_constraints;
            apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        }

        // Eliminate Dirichlet DOFs → reduced system (MFEM FormLinearSystem).
        let (red_mat, red_rhs, free_map, constrained_map) =
            eliminate_dirichlet(&mat, &rhs, &bnd, &bnd_vals);

        // Warm-start: prolongate previous solution if available (Tri3).
        let mut u_red = vec![0.0; red_mat.nrows];
        if let Some(ref pmesh) = prev_mesh {
            if !is_quad {
                let mid_map = build_edge_midpoint_map(pmesh, &mesh);
                let prol = fem_mesh::amr::prolongate_p1(&u, mesh.n_nodes(), &mid_map);
                for (ri, &dof) in free_map.iter().enumerate() {
                    u_red[ri] = prol[dof as usize];
                }
            }
        }

        // Solve: PCG + GSSmoother (MFEM: PCG(*A, M, B, X, 3, 200, 1e-12, 0.0)).
        // C++ print_iter=3 → PrintLevel::Iterations in linlvo.
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 200,
            print_level: PrintLevel::Iterations,
            ..SolverConfig::default()
        };
        let res = solve_pcg_gssmoother(&red_mat, &red_rhs, &mut u_red, &cfg);
        let (converged, _iters) = match res {
            Ok(r) => (r.converged, r.iterations),
            Err(_) => (false, 200),
        };
        if !converged {
            // MFEM: prints "No convergence!" and continues with the current X.
        }

        // RecoverFEMSolution: expand to full DOF vector.
        u = expand_from_reduced(&u_red, &free_map, &constrained_map, &bnd_vals, cdofs);

        // Recover hanging-node DOF values (Quad4 NC path).
        if !hanging_constraints.is_empty() {
            use fem_space::constraints::recover_hanging_values;
            recover_hanging_values(&mut u, &hanging_constraints);
        }

        // Print ZZ estimator diagnostics.
        let gf = GridFunction::new(&space, u.clone());
        let indicators = zz_estimator_nodal(&gf);

        // Check max DOFs.
        if cdofs > max_dofs {
            println!("Reached the maximum number of dofs. Stop.");
            break;
        }

        // MFEM ThresholdRefiner with SetTotalErrorFraction(0.7):
        // sort elements by error descending, mark the smallest set whose
        // cumulative error sum reaches ≥ 70 % of total error.
        let mut eta: Vec<(usize, f64)> = indicators.eta.iter().copied().enumerate().collect();
        let eta_total: f64 = eta.iter().map(|(_, e)| e).sum();
        // Sort descending by error.
        eta.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0_f64;
        let threshold = 0.7 * eta_total;
        let marked: Vec<u32> = eta.iter()
            .take_while(|(_, e)| { cum += e; cum <= threshold })
            .map(|(i, _)| *i as u32)
            .collect();

        if marked.is_empty() {
            println!("Stopping criterion satisfied. Stop.");
            break;
        }

        // Apply refiner to modify the mesh (MFEM refiner.Apply(mesh)).
        if is_quad {
            let (new_mesh, new_constraints) = refine_nonconforming_quad(&mesh, &marked.iter().map(|&i| i as u32).collect::<Vec<_>>(), None);
            mesh = new_mesh;
            hanging_constraints = new_constraints;
        } else {
            prev_mesh = Some(mesh.clone());
            mesh = closure_refine_default(&mesh, &marked.iter().map(|&i| i as u32).collect::<Vec<_>>(), None);
        }
        // Resize solution vector for new mesh.
        u.resize(space.n_dofs(), 0.0);
    }

    eprintln!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

fn build_edge_midpoint_map(old: &Mesh<2>, new: &Mesh<2>) -> std::collections::HashMap<(u32, u32), u32> {
    use fem_core::NodeId;
    let mut map = std::collections::HashMap::new();
    let old_n = old.n_nodes();
    let mut old_edges: Vec<(NodeId, NodeId)> = Vec::new();
    for e in 0..old.n_elems() as NodeId {
        let ns = old.elem_nodes(e);
        for &(a, b) in &[(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if !old_edges.contains(&key) { old_edges.push(key); }
        }
    }
    let new_nodes: Vec<(NodeId, [f64; 2])> = (old_n as NodeId..new.n_nodes() as NodeId)
        .map(|nid| (nid, new.coords_of(nid))).collect();
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

struct Args {
    mesh: String, order: u8, max_dofs: usize, _ls_zz: bool, _no_vis: bool,
}
impl Args {
    fn parse() -> Self {
        let mut mesh = "data/star.mesh".to_string();
        let mut order: u8 = 1;
        let mut max_dofs: usize = 50000;
        let mut ls_zz = false;
        let mut no_vis = false;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => mesh = it.next().unwrap_or(mesh),
                "-o" | "--order" => order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
                "-md" | "--max-dofs" => max_dofs = it.next().and_then(|v| v.parse().ok()).unwrap_or(50000),
                "-ls" | "--ls-zz" => ls_zz = true,
                "-no-vis" | "--no-visualization" => no_vis = true,
                _ => {}
            }
        }
        Args { mesh, order, max_dofs, _ls_zz: ls_zz, _no_vis: no_vis }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
    use fem_assembly::postprocess::compute_h1_error;
    use fem_assembly::Assembler;
    use fem_mesh::Mesh;
    use fem_solver::{SolverConfig, solve_pcg_gssmoother};
    use fem_space::constraints::{boundary_dofs, eliminate_dirichlet, expand_from_reduced};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    fn exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
    fn rhs_mms(x: &[f64]) -> f64 { 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() }
    fn grad_exact(x: &[f64]) -> Vec<f64> {
        vec![PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
             PI * (PI * x[0]).sin() * (PI * x[1]).cos()]
    }

    fn solve_mms(n: usize, order: u8) -> (Vec<f64>, H1Space<Mesh<2>>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, order);
        let ndofs = space.n_dofs();
        let quad = (order as u8) * 2 + 2;
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator{kappa:1.0}], quad);
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(rhs_mms)], quad);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals: Vec<f64> = bnd.iter().map(|&d| { let x = dm.dof_coord(d); exact(&x) }).collect();
        let (red_mat, red_rhs, free_map, constrained_map) = eliminate_dirichlet(&mat, &rhs, &bnd, &bnd_vals);
        let mut u_red = vec![0.0; red_mat.nrows];
        let cfg = SolverConfig{rtol:1e-12,max_iter:5000,verbose:false,..SolverConfig::default()};
        solve_pcg_gssmoother(&red_mat, &red_rhs, &mut u_red, &cfg).expect("PCG");
        let u = expand_from_reduced(&u_red, &free_map, &constrained_map, &bnd_vals, ndofs);
        (u, space)
    }

    #[test]
    fn ex6_mms_l2_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let gf_c = GridFunction::new(&sp_c, u_c.clone());
        let gf_f = GridFunction::new(&sp_f, u_f.clone());
        let err_c = gf_c.compute_l2_error(&exact, 6);
        let err_f = gf_f.compute_l2_error(&exact, 6);
        let rate = (err_f / err_c).ln() / (32.0_f64 / 16.0_f64).ln();
        assert!(rate < -1.8, "L2 convergence rate {:.2} too slow", rate);
        fem_regression::regression("mfem_ex6_flux_recovery")
            .check("l2_error_n16_p2", err_c)
            .check("l2_error_n32_p2", err_f)
            .check("l2_convergence_rate", rate)
            .finalize();
    }

    #[test]
    fn ex6_mms_h1_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let h1_c = compute_h1_error(&sp_c, &u_c, grad_exact, 6);
        let h1_f = compute_h1_error(&sp_f, &u_f, grad_exact, 6);
        assert!(h1_f < h1_c);
        fem_regression::regression("mfem_ex6_flux_recovery")
            .check("h1_error_n16_p2", h1_c)
            .check("h1_error_n32_p2", h1_f)
            .finalize();
    }

    #[test]
    fn ex6_zz_estimator_symmetry() {
        let (u, space) = solve_mms(16, 2);
        let gf = GridFunction::new(&space, u);
        let ind = zz_estimator_nodal(&gf);
        assert!(ind.eta.iter().all(|&e| e >= 0.0), "ZZ indicators must be non-negative");
        assert!(ind.total_error > 0.0, "total error must be positive");
    }
}
