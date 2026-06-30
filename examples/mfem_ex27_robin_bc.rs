use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, NeumannIntegrator, BoundaryMassIntegrator},
};
use fem_element::ReferenceElement;
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 27: Mixed Boundary Conditions (Robin) ===");

    // Unit square mesh. Boundary tags:
    //   tag 1: bottom y=0  — Dirichlet: u = 0
    //   tag 2: right  x=1  — Neumann:   ∂u/∂n = g
    //   tag 3: top    y=1  — Robin:     ∂u/∂n + a·u = b
    //   tag 4: left   x=0  — Natural (homogeneous Neumann)
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let order = 1u8; // P1 only (face_dofs_p1 constraint)
    let space = H1Space::new(mesh.clone(), order);
    let n = space.n_dofs();

    println!("Mesh: {}×{} P1, {} DOFs", args.n, args.n, n);
    println!("BC: Dirichlet(bottom) + Neumann(right) + Robin(top,a={}) + Natural(left)", args.robin_a);

    // Exact: u = sin(πx) sinh(πy) / sinh(π)
    // This satisfies: -Δu = 0, u|y=0 = 0, u|y=1 = sin(πx)
    let inv_sinh_pi = 1.0_f64 / PI.sinh();
    let exact = move |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sinh() * inv_sinh_pi;
    let du_dx = move |x: &[f64]| PI * (PI * x[0]).cos() * (PI * x[1]).sinh() * inv_sinh_pi;
    let du_dy = move |x: &[f64]| PI * (PI * x[0]).sin() * (PI * x[1]).cosh() * inv_sinh_pi;

    let face_dofs = |f: u32| -> Vec<u32> {
        let nodes = mesh.face_nodes(f);
        nodes.iter().copied().collect()
    };

    // Stiffness matrix
    let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    // Robin boundary mass: ∫ a·u·v ds on tag 3
    let robin_mass = Assembler::assemble_boundary_bilinear(
        n, &mesh, &face_dofs, order, &[&BoundaryMassIntegrator { alpha: args.robin_a }], &[3], 3);
    stiff = stiff.add(&robin_mass);

    // RHS
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 0.0)], 3);

    // Neumann BC on tag 2 (right): ∂u/∂n = ∂u/∂x at x=1
    let neumann_g = NeumannIntegrator::new(move |x: &[f64], _n: &[f64]| du_dx(x));
    let neumann_rhs = Assembler::assemble_boundary_linear(
        n, &mesh, &face_dofs, order, &[&neumann_g], &[2], 3);
    for i in 0..n { rhs[i] += neumann_rhs[i]; }

    // Robin RHS on tag 3 (top): b = du/dy + a·u at y=1
    let robin_b = move |x: &[f64], _n: &[f64]| du_dy(x) + args.robin_a * exact(x);
    let robin_src = NeumannIntegrator::new(robin_b);
    let robin_rhs = Assembler::assemble_boundary_linear(
        n, &mesh, &face_dofs, order, &[&robin_src], &[3], 3);
    for i in 0..n { rhs[i] += robin_rhs[i]; }

    // Dirichlet BC on tag 1 (bottom): u = 0
    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1]);
    let bvals = vec![0.0; bdofs.len()];
    apply_dirichlet(&mut stiff, &mut rhs, &bdofs, &bvals);

    // Solve
    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
    let res = solve_pcg_jacobi(&stiff, &rhs, &mut u, &cfg).unwrap();
    println!("  PCG: {} iters, ‖r‖₂ = {:.3e}", res.iterations, res.final_residual);

    // L² error
    let ref_elem = fem_element::lagrange::TriP1;
    let quad_e = ref_elem.quadrature(4);
    let n_ldofs = ref_elem.n_dofs();
    let mut phi = vec![0.0; n_ldofs];
    let mut l2_err: f64 = 0.0;
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
        for (q, xi) in quad_e.points.iter().enumerate() {
            let w = quad_e.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            let uh: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| u[d as usize] * p).sum();
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                      x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
            l2_err += w * (uh - exact(&xp)).powi(2);
        }
    }
    let l2 = l2_err.sqrt();
    println!("  ‖u_h - u‖₂ = {:.4e}", l2);
    println!("Done.");
}

struct Args { n: usize, robin_a: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 8, robin_a: PI };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"       => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(8); }
            "--robin-a" => { a.robin_a = it.next().and_then(|v| v.parse().ok()).unwrap_or(PI); }
            _ => {}
        }
    }
    a
}
