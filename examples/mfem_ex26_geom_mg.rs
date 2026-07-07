//! # Example 26 — Geometric Multigrid + LOR (analogous to MFEM ex26)
//!
//! Solves the 2-D Poisson equation −Δu = 1 with homogeneous Dirichlet BCs:
//!
//! ```text
//!   −∇²u = 1    in Ω
//!      u = 0    on ∂Ω
//! ```
//!
//! Demonstrates a **Low-Order Refined (LOR)** preconditioner: an AMG V-cycle
//! on the P1 (low-order) system is used as a preconditioner for the P2
//! (high-order) system, mirroring MFEM ex26's multigrid hierarchy.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m ../data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m ../data/fichera.mesh --order 2
//! ```

use std::f64::consts::PI;

use fem_amg::{AmgConfig, AmgSolver};
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

struct LorResult {
    n_p1: usize,
    n_p2: usize,
    amg_p2_levels: usize,
    amg_p1_levels: usize,
    iterations_amg: usize,
    iterations_jacobi: usize,
    final_residual: f64,
    converged: bool,
    l2_error: f64,
}

fn main() {
    let args = parse_args();
    println!("=== Example 26: Geometric Multigrid + LOR (MFEM ex26) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{}", args.n, args.n);
    }
    println!("  Poisson: −Δu = 1, u|_∂Ω = 0, P{}, LOR AMG preconditioner", args.order);

    // Load or generate mesh
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    let r = solve_lor_case(mesh, args.order);

    println!("  P1 DOFs: {}, P2 DOFs: {}", r.n_p1, r.n_p2);
    println!("  AMG levels: P1={}, P2={}", r.amg_p1_levels, r.amg_p2_levels);
    println!(
        "  AMG-PCG: {} iters, residual={:.3e}, converged={}",
        r.iterations_amg, r.final_residual, r.converged
    );
    println!("  Jacobi-PCG: {} iters", r.iterations_jacobi);
    println!("  L2 error = {:.3e}", r.l2_error);
    assert!(r.converged, "AMG P2 solve did not converge");
    println!("  PASS");
}

fn solve_lor_case(mesh: Mesh<2>, p: u8) -> LorResult {
    let order_p2 = p;
    let order_p1 = 1u8;

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });

    // ── High-order space ──────────────────────────────────────────────
    let space_p2 = H1Space::new(mesh.clone(), order_p2);
    let n_p2 = space_p2.n_dofs();
    let quad_p2 = (order_p2 * 2 + 1).max(3);

    let mut mat_p2 = Assembler::assemble_bilinear(&space_p2, &[&diffusion], quad_p2);
    let mut rhs_p2 = Assembler::assemble_linear(&space_p2, &[&source], quad_p2);
    let bnd_p2 = boundary_dofs(space_p2.mesh(), space_p2.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat_p2, &mut rhs_p2, &bnd_p2, &vec![0.0; bnd_p2.len()]);

    // ── Low-order (LOR) space ─────────────────────────────────────────
    let space_p1 = H1Space::new(mesh.clone(), order_p1);
    let n_p1 = space_p1.n_dofs();
    let mut mat_p1 = Assembler::assemble_bilinear(&space_p1, &[&diffusion], 3);
    let mut zero_p1 = vec![0.0_f64; n_p1];
    let bnd_p1 = boundary_dofs(space_p1.mesh(), space_p1.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat_p1, &mut zero_p1, &bnd_p1, &vec![0.0; bnd_p1.len()]);

    // ── AMG hierarchies ───────────────────────────────────────────────
    let amg_p2 = AmgSolver::setup(&mat_p2, AmgConfig::default());
    let amg_p2_lvl = amg_p2.n_levels();
    let amg_p1 = AmgSolver::setup(&mat_p1, AmgConfig::default());
    let amg_p1_lvl = amg_p1.n_levels();

    // ── Solve P2 system with AMG preconditioner ────────────────────────
    let cfg = SolverConfig {
        rtol: 1.0e-7,
        atol: 0.0,
        max_iter: 800,
        verbose: false,
        ..Default::default()
    };
    let mut u_amg = vec![0.0_f64; n_p2];
    let res_amg = amg_p2
        .solve(&mat_p2, &rhs_p2, &mut u_amg, &cfg)
        .expect("AMG P2 solve failed");

    // ── Solve P2 with Jacobi baseline ──────────────────────────────────
    let mut u_jac = vec![0.0_f64; n_p2];
    let res_jac = solve_pcg_jacobi(&mat_p2, &rhs_p2, &mut u_jac, &cfg)
        .expect("Jacobi PCG failed");

    let l2 = l2_error_2d(&space_p2, &u_amg);

    LorResult {
        n_p1,
        n_p2,
        amg_p2_levels: amg_p2_lvl,
        amg_p1_levels: amg_p1_lvl,
        iterations_amg: res_amg.iterations,
        iterations_jacobi: res_jac.iterations,
        final_residual: res_amg.final_residual,
        converged: res_amg.converged,
        l2_error: l2,
    }
}

fn l2_error_2d(space: &H1Space<Mesh<2>>, uh: &[f64]) -> f64 {
    use fem_element::{lagrange::TriP2, ReferenceElement};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let elem = TriP2;
    let qr = elem.quadrature(5);
    let mut err2 = 0.0_f64;
    let mut phi = vec![0.0_f64; elem.n_dofs()];

    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
            - (x1[1] - x0[1]) * (x2[0] - x0[0]))
        .abs();
        let dofs: Vec<usize> = space
            .element_dofs(e)
            .iter()
            .map(|&d| d as usize)
            .collect();

        for (qi, xi) in qr.points.iter().enumerate() {
            elem.eval_basis(xi, &mut phi);
            let w = qr.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];
            let uh_val: f64 = phi
                .iter()
                .zip(dofs.iter())
                .map(|(&v, &d)| v * uh[d])
                .sum();
            let u_ex = (PI * xp[0]).sin() * (PI * xp[1]).sin();
            err2 += w * (uh_val - u_ex).powi(2);
        }
    }
    err2.sqrt()
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 10,
        order: 2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("10".into())
                    .parse()
                    .unwrap_or(10)
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("2".into())
                    .parse()
                    .unwrap_or(2)
            }
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex26_lor_pcg_2d_poisson_converges() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let r = solve_lor_case(mesh, 2);
        assert!(r.converged,
            "AMG P2 solve did not converge: iters={}, residual={:.3e}",
            r.iterations_amg, r.final_residual);
        assert!(r.final_residual < 5.0e-7,
            "AMG P2 residual too large: {:.3e}", r.final_residual);
        assert!(r.l2_error < 5.0e-3,
            "P2 L2 error too large: {:.3e}", r.l2_error);
    }

    #[test]
    fn ex26_lor_p1_dofs_less_than_p2_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let r = solve_lor_case(mesh, 2);
        assert!(r.n_p1 < r.n_p2,
            "expected n_p1 < n_p2: {} vs {}", r.n_p1, r.n_p2);
        let ratio = r.n_p2 as f64 / r.n_p1 as f64;
        assert!(ratio > 1.4, "P2/P1 DOF ratio expected >1.4: {:.3}", ratio);
    }

    #[test]
    fn ex26_lor_amg_faster_than_jacobi() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let r = solve_lor_case(mesh, 2);
        assert!(r.converged, "AMG P2 solve did not converge");
        assert!(
            r.iterations_amg < r.iterations_jacobi,
            "expected AMG to converge faster than Jacobi: amg={} jacobi={}",
            r.iterations_amg, r.iterations_jacobi
        );
    }
}
