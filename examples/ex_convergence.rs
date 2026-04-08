//! # Example: Convergence rate study — P1 vs P2 elements
//!
//! Systematically verifies the theoretical convergence rates of P1 and P2
//! Lagrange elements for the Poisson equation:
//!
//! ```text
//!   −Δu = f    in Ω = [0,1]²
//!      u = 0    on ∂Ω
//! ```
//!
//! Manufactured solution: `u(x,y) = sin(π x) sin(π y)`.
//!
//! Theoretical convergence in L² norm (Céa's lemma + Aubin-Nitsche):
//! ```text
//!   ‖u − u_h‖_{L²} = O(h^{p+1})   →   P1: O(h²),  P2: O(h³)
//! ```
//!
//! The example:
//! 1. Runs both P1 and P2 on a sequence of meshes (n = 4, 8, 16, 32, 64).
//! 2. Computes L² error at each level.
//! 3. Estimates the convergence rate from consecutive levels.
//! 4. Checks that measured rates are close to the theoretical values.
//!
//! ## Usage
//! ```
//! cargo run --example ex_convergence
//! cargo run --example ex_convergence -- --max-n 64
//! cargo run --example ex_convergence -- --levels 6
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    Assembler,
};
use fem_element::{
    lagrange::{TriP1, TriP2},
    ReferenceElement,
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

// Exact solution and RHS
fn u_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}
fn f_rhs(x: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn main() {
    let args = parse_args();

    // Build mesh sizes: n = 4, 8, 16, ..., 4*2^(levels-1)
    let n_list: Vec<usize> = (0..args.levels).map(|i| 4_usize << i).collect();

    println!("=== fem-rs Convergence Study: P1 vs P2 elements (Poisson on [0,1]²) ===");
    println!("  Manufactured solution: u = sin(πx)sin(πy)");
    println!();

    for &order in &[1u8, 2u8] {
        let expected_rate = (order + 1) as f64;
        println!("── P{order} elements  (expected L² rate ≈ {expected_rate:.1}) ──────────────");
        println!(
            "{:>5}  {:>8}  {:>12}  {:>8}",
            "n", "dofs", "L² error", "rate"
        );
        println!("{}", "-".repeat(40));

        let mut prev_err = None::<f64>;
        let mut prev_h = None::<f64>;
        let mut rates = Vec::new();

        for &n in &n_list {
            let (err, dofs) = solve_poisson(n, order);
            let h = 1.0 / n as f64;

            let rate_str = match (prev_err, prev_h) {
                (Some(e0), Some(h0)) => {
                    let r = (err / e0).ln() / (h / h0).ln();
                    rates.push(r);
                    format!("{r:.3}")
                }
                _ => "  —  ".to_string(),
            };

            println!("  {:>3}  {:>8}  {:>12.4e}  {:>8}", n, dofs, err, rate_str);
            prev_err = Some(err);
            prev_h = Some(h);
        }

        // Summary: average rate over last half of levels.
        if rates.len() >= 2 {
            let avg: f64 = rates[rates.len() / 2..].iter().sum::<f64>()
                / (rates.len() - rates.len() / 2) as f64;
            let pass = (avg - expected_rate).abs() < 0.3;
            println!(
                "  Avg rate (last {} levels): {avg:.3}  {}",
                rates.len() - rates.len() / 2,
                if pass {
                    "✓ PASS"
                } else {
                    "✗ FAIL (expected ≈ {expected_rate})"
                }
            );
        }
        println!();
    }

    println!("Done.");
}

fn solve_poisson(n: usize, order: u8) -> (f64, usize) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, order);
    let ndofs = space.n_dofs();

    let quad_order = order * 2 + 1;

    let mut mat =
        Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order);
    let src = DomainSourceIntegrator::new(f_rhs);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], quad_order);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0_f64; bnd.len()]);

    let mut u = vec![0.0_f64; ndofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

    let l2 = l2_error(&space, &u, order, u_exact);
    (l2, ndofs)
}

// ─── L² error — dispatches to P1 or P2 quadrature ───────────────────────────

fn l2_error<S: FESpace>(space: &S, uh: &[f64], order: u8, u_ex: impl Fn(&[f64]) -> f64) -> f64 {
    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    let quad_order = order * 2 + 3; // sufficient for p+2 accuracy

    for e in 0..mesh.n_elements() as u32 {
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();

        match order {
            1 => {
                let re = TriP1;
                let quad = re.quadrature(quad_order);
                let mut phi = vec![0.0_f64; re.n_dofs()];
                for (qi, xi) in quad.points.iter().enumerate() {
                    re.eval_basis(xi, &mut phi);
                    let w = quad.weights[qi] * det_j;
                    let xp = map_to_phys(x0, x1, x2, xi);
                    let uh_q: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
                    err2 += w * (uh_q - u_ex(&xp)).powi(2);
                }
            }
            2 => {
                let re = TriP2;
                let quad = re.quadrature(quad_order);
                let mut phi = vec![0.0_f64; re.n_dofs()];
                for (qi, xi) in quad.points.iter().enumerate() {
                    re.eval_basis(xi, &mut phi);
                    let w = quad.weights[qi] * det_j;
                    let xp = map_to_phys(x0, x1, x2, xi);
                    let uh_q: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
                    err2 += w * (uh_q - u_ex(&xp)).powi(2);
                }
            }
            _ => panic!("unsupported order {order}"),
        }
    }
    err2.sqrt()
}

#[inline]
fn map_to_phys<'a>(x0: &'a [f64], x1: &'a [f64], x2: &'a [f64], xi: &[f64]) -> [f64; 2] {
    [
        x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
        x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
    ]
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    levels: usize,
}

fn parse_args() -> Args {
    let mut a = Args { levels: 5 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--levels" => {
                a.levels = it.next().unwrap_or("5".into()).parse().unwrap_or(5);
            }
            "--max-n" => {
                if let Some(v) = it.next() {
                    let max_n: usize = v.parse().unwrap_or(64);
                    // Find how many doublings of 4 fit in max_n.
                    let mut lvl = 0;
                    let mut nn = 4;
                    while nn <= max_n {
                        lvl += 1;
                        nn *= 2;
                    }
                    a.levels = lvl.max(2);
                }
            }
            _ => {}
        }
    }
    a
}
