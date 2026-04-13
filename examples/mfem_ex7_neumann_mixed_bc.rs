//! # Example 7 Poisson with mixed Dirichlet/Neumann boundary conditions
//!
//! Solves the scalar Poisson equation with mixed boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on Γ_D (bottom y=0 and left x=0 walls, tags 1,4)
//!    κ ∂u/∂n = g(x)   on Γ_N (top y=1 and right x=1 walls, tags 2,3)
//! ```
//!
//! Manufactured solution: `u(x,y) = x(1-x)y(1-y)`, which satisfies
//! homogeneous Dirichlet on all four walls.  However, we **deliberately**
//! apply Dirichlet only on two walls and Neumann on the other two to
//! demonstrate the assembly of the natural boundary term:
//!
//! ```text
//!   g(x,y) = κ (∂u/∂n)(x,y)   on Γ_N
//! ```
//!
//! The Neumann data is computed from the exact solution's normal derivative,
//! so the error against the exact solution should converge at O(h²).
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex7_neumann_mixed_bc
//! cargo run --example mfem_ex7_neumann_mixed_bc -- --n 32
//! cargo run --example mfem_ex7_neumann_mixed_bc -- --n 8 --n 16 --n 32  # convergence
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    assembler::face_dofs_p1,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, NeumannIntegrator},
    Assembler,
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

// Exact solution: u(x,y) = x(1-x) * y(1-y)
fn u_exact(x: &[f64]) -> f64 {
    x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
}

// Source term: f = -Δu  for u = x(1-x)y(1-y)
//   Δu = ∂²u/∂x² + ∂²u/∂y²
//      = (-2)(y(1-y)) + (-2)(x(1-x))
//      = -2 [x(1-x) + y(1-y)]
// So f = -Δu = 2 [x(1-x) + y(1-y)]
fn f_rhs(x: &[f64]) -> f64 {
    2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))
}

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 7: Poisson with mixed Dirichlet/Neumann BCs ===");
    println!("  BCs: Dirichlet on bottom+left (tags 1,4), Neumann on right+top (tags 2,3)");
    println!("  Exact solution: u = x(1-x)y(1-y)");
    println!();
    println!(
        "{:>5}  {:>8}  {:>8}  {:>12}  {:>8}",
        "n", "nodes", "dofs", "L² error", "rate"
    );
    println!("{}", "-".repeat(55));

    let mut prev_err = None::<f64>;
    let mut prev_h = None::<f64>;

    for &n in &args.n_list {
        let (l2, n_nodes, n_dofs) = solve_one(n, args.kappa);
        let h = 1.0 / n as f64;

        let rate = match (prev_err, prev_h) {
            (Some(e0), Some(h0)) => format!("{:.2}", (l2 / e0).ln() / (h / h0).ln()),
            _ => "  --".to_string(),
        };

        println!(
            "  {:>3}  {:>8}  {:>8}  {:>12.4e}  {:>8}",
            n, n_nodes, n_dofs, l2, rate
        );
        prev_err = Some(l2);
        prev_h = Some(h);
    }

    println!("\n  (Expected O(h²) convergence for P1 elements)");
    println!("Done.");
}

fn solve_one(n: usize, kappa: f64) -> (f64, usize, usize) {
    // ─── 1. Mesh and space ────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    // ─── 2. Assemble stiffness K = κ ∇u·∇v dx ─────────────────────────────
    let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa }], 3);

    // ─── 3. Assemble volume RHS f v dx ─────────────────────────────────────
    let src = DomainSourceIntegrator::new(f_rhs);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    // ─── 4. Assemble Neumann boundary term ∫_{Γ_N} g(x) v ds ────────────────
    //   Mesh tags (unit_square_tri convention):
    //     tag 1 = bottom (y=0), tag 2 = right (x=1),
    //     tag 3 = top (y=1),    tag 4 = left (x=0)
    //
    //   We apply Neumann on tags 2 (right) and 3 (top).
    //   On right wall (x=1): outward normal = (1,0), ∂u/∂n = ∂u/∂x = (1-2x)y(1-y)|_{x=1} = -y(1-y)
    //   On top wall  (y=1): outward normal = (0,1), ∂u/∂n = ∂u/∂y = x(1-x)(1-2y)|_{y=1} = -x(1-x)
    //
    //   The Neumann integrator computes g(x,n) v ds where g receives
    //   the physical coords x and the outward normal n at each face QP.

    let neumann_g = NeumannIntegrator::new(|x: &[f64], _n: &[f64]| {
        // κ * ∂u/∂n at (x,y):
        //   On right (x): κ*(1-2x)*y*(1-y) the assembler calls this only for tagged faces
        //   On top   (y): κ*(1-2y)*x*(1-x)
        //   We return κ*(∂u/∂x * n_x + ∂u/∂y * n_y), but since the faces are
        //   axis-aligned and we know the normal from context, we can compute it
        //   directly. For generality we use the full gradient dotted with the
        //   actual normal passed by the integrator:
        let du_dx = (1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]);
        let du_dy = x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]);
        kappa * (du_dx * _n[0] + du_dy * _n[1])
    });

    let face_dofs = face_dofs_p1(space.mesh());
    let neumann_rhs = Assembler::assemble_boundary_linear(
        ndofs,
        space.mesh(),
        &face_dofs,
        1,
        &[&neumann_g],
        &[2, 3], // right and top walls
        3,
    );
    for i in 0..ndofs {
        rhs[i] += neumann_rhs[i];
    }

    // ─── 5. Apply Dirichlet BCs u = 0 on bottom (tag 1) and left (tag 4) ─────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // ─── 6. Solve ─────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; ndofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

    // ─── 7. L² error ──────────────────────────────────────────────────────────
    let l2 = l2_error(&space, &u, u_exact);

    (l2, space.mesh().n_nodes(), ndofs)
}

// ─── L² error helper ─────────────────────────────────────────────────────────

fn l2_error<S: FESpace>(space: &S, uh: &[f64], u_ex: impl Fn(&[f64]) -> f64) -> f64 {
    use fem_element::{lagrange::TriP1, ReferenceElement};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in 0..mesh.n_elements() as u32 {
        let re = TriP1;
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];
            let uh_q: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_q - u_ex(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n_list: Vec<usize>,
    kappa: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n_list: vec![8, 16, 32],
        kappa: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                if let Some(v) = it.next() {
                    a.n_list = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            "--kappa" => {
                a.kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            _ => {}
        }
    }
    if a.n_list.is_empty() {
        a.n_list = vec![8, 16, 32];
    }
    a
}

