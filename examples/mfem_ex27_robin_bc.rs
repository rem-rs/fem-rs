//! # Example 27 — Mixed Boundary Conditions (analogous to MFEM ex27)
//!
//! Solves −Δu = 0 on a unit square with mixed BCs:
//!
//! ```text
//!   −Δu = 0              in Ω
//!    u = 0               on bottom  (attribute 1, Dirichlet)
//!    ∂u/∂n = g           on right   (attribute 2, Neumann)
//!    ∂u/∂n + a·u = b     on top    (attribute 3, Robin)
//!    ∂u/∂n = 0           on left    (attribute 4, natural/homogeneous Neumann)
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex27_robin_bc
//! cargo run --example mfem_ex27_robin_bc -- --n 16 --dbc 0.0 --nbc 2.0
//! cargo run --example mfem_ex27_robin_bc -- -m ../data/star.mesh --rbc-a 2.0 --rbc-b 0.5
//! ```

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, NeumannIntegrator, BoundaryMassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{MeshTopology, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== Example 27: Mixed BC with Robin (MFEM ex27) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{} P1", args.n, args.n);
    }
    println!(
        "  BC: Dirichlet(bottom,{:.3}) + Neumann(right,{:.3}) + Robin(top,a={:.3},b={:.3}) + Natural(left)",
        args.dbc, args.nbc, args.rbc_a, args.rbc_b
    );

    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    let order = 1u8;
    let space = H1Space::new(mesh.clone(), order);
    let n = space.n_dofs();
    println!("  DOFs: {n}");

    // Face DOF mapping for boundary assembly
    let face_dofs = |f: u32| -> Vec<u32> {
        let nodes = mesh.face_nodes(f);
        nodes.iter().copied().collect()
    };

    // Stiffness matrix: −Δ
    let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    // Robin boundary mass: ∫ a·u·v ds on tag 3 (top)
    let robin_mass = Assembler::assemble_boundary_bilinear(
        n,
        &mesh,
        &face_dofs,
        order,
        &[&BoundaryMassIntegrator {
            alpha: args.rbc_a,
        }],
        &[3],
        3,
    );
    stiff = stiff.add(&robin_mass);

    // RHS
    let mut rhs = vec![0.0_f64; n];

    // Neumann BC on tag 2 (right): ∂u/∂n = g
    let neumann = NeumannIntegrator::new(move |_x: &[f64], _n: &[f64]| args.nbc);
    let neumann_rhs = Assembler::assemble_boundary_linear(
        n, &mesh, &face_dofs, order, &[&neumann], &[2], 3,
    );
    for i in 0..n {
        rhs[i] += neumann_rhs[i];
    }

    // Robin RHS on tag 3 (top): b on the RHS
    let robin_b = NeumannIntegrator::new(move |_x: &[f64], _n: &[f64]| args.rbc_b);
    let robin_rhs = Assembler::assemble_boundary_linear(
        n, &mesh, &face_dofs, order, &[&robin_b], &[3], 3,
    );
    for i in 0..n {
        rhs[i] += robin_rhs[i];
    }

    // Dirichlet BC on tag 1 (bottom): u = dbc
    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1]);
    let bvals = vec![args.dbc; bdofs.len()];
    apply_dirichlet(&mut stiff, &mut rhs, &bdofs, &bvals);

    // Solve
    let mut u = vec![0.0; n];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 5000,
        verbose: false,
        ..Default::default()
    };
    let res = solve_pcg_jacobi(&stiff, &rhs, &mut u, &cfg).unwrap();
    println!("  PCG: {} iters, ‖r‖₂ = {:.3e}", res.iterations, res.final_residual);
    println!("  ‖u‖₂ = {:.6e}", u.iter().map(|v| v * v).sum::<f64>().sqrt());
    println!("Done.");
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    dbc: f64,
    nbc: f64,
    rbc_a: f64,
    rbc_b: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 8,
        dbc: 0.0,
        nbc: 1.0,
        rbc_a: 1.0,
        rbc_b: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--dbc" => a.dbc = it.next().unwrap_or("0.0".into()).parse().unwrap_or(0.0),
            "--nbc" => a.nbc = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--rbc-a" => a.rbc_a = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--rbc-b" => a.rbc_b = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::ReferenceElement;
    use fem_mesh::topology::MeshTopology;

    #[test]
    fn ex27_robin_bc_solve_converges() {
        let args = Args { mesh: None, n: 8, dbc: 0.0, nbc: 1.0, rbc_a: 1.0, rbc_b: 1.0 };
        let mesh = Mesh::<2>::unit_square_tri(args.n);
        let order = 1u8;
        let space = H1Space::new(mesh.clone(), order);
        let n = space.n_dofs();

        let face_dofs = |f: u32| -> Vec<u32> {
            let nodes = mesh.face_nodes(f);
            nodes.iter().copied().collect()
        };

        let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let robin_mass = Assembler::assemble_boundary_bilinear(
            n, &mesh, &face_dofs, order,
            &[&BoundaryMassIntegrator { alpha: args.rbc_a }], &[3], 3,
        );
        stiff = stiff.add(&robin_mass);

        let mut rhs = vec![0.0_f64; n];
        let neumann = NeumannIntegrator::new(move |_: &[f64], _: &[f64]| args.nbc);
        let nr = Assembler::assemble_boundary_linear(n, &mesh, &face_dofs, order, &[&neumann], &[2], 3);
        for i in 0..n { rhs[i] += nr[i]; }

        let robin_b = NeumannIntegrator::new(move |_: &[f64], _: &[f64]| args.rbc_b);
        let rr = Assembler::assemble_boundary_linear(n, &mesh, &face_dofs, order, &[&robin_b], &[3], 3);
        for i in 0..n { rhs[i] += rr[i]; }

        let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1]);
        let bvals = vec![args.dbc; bdofs.len()];
        apply_dirichlet(&mut stiff, &mut rhs, &bdofs, &bvals);

        let mut u = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
        let res = solve_pcg_jacobi(&stiff, &rhs, &mut u, &cfg).unwrap();

        assert!(res.converged);
        assert!(res.final_residual < 1.0e-10);
        assert!(u.iter().any(|&v| v.abs() > 0.0));
    }

    /// Verify that u ≈ u_exact = sin(πx) sinh(πy) / sinh(π) for the
    /// manufactured-solution BC setup.
    #[test]
    fn ex27_robin_bc_mms_agreement() {
        use std::f64::consts::PI;
        let inv_sinh_pi = 1.0_f64 / PI.sinh();
        let exact = move |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sinh() * inv_sinh_pi;
        let du_dy = move |x: &[f64]| PI * (PI * x[0]).sin() * (PI * x[1]).cosh() * inv_sinh_pi;
        let du_dx = move |x: &[f64]| PI * (PI * x[0]).cos() * (PI * x[1]).sinh() * inv_sinh_pi;
        let robin_a = PI;
        let inv_sinh = 1.0_f64 / PI.sinh();

        let n = 16;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let order = 1u8;
        let space = H1Space::new(mesh.clone(), order);
        let n_dofs = space.n_dofs();

        let face_dofs = |f: u32| -> Vec<u32> {
            let nodes = mesh.face_nodes(f);
            nodes.iter().copied().collect()
        };

        let mut stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let robin_mass = Assembler::assemble_boundary_bilinear(
            n_dofs, &mesh, &face_dofs, order,
            &[&BoundaryMassIntegrator { alpha: robin_a }], &[3], 3,
        );
        stiff = stiff.add(&robin_mass);

        let mut rhs = vec![0.0_f64; n_dofs];
        let neumann_g = NeumannIntegrator::new(du_dx);
        let nr = Assembler::assemble_boundary_linear(n_dofs, &mesh, &face_dofs, order, &[&neumann_g], &[2], 3);
        for i in 0..n_dofs { rhs[i] += nr[i]; }

        let robin_b_fn = move |x: &[f64], _n: &[f64]| du_dy(x) + robin_a * exact(x);
        let robin_b = NeumannIntegrator::new(robin_b_fn);
        let rr = Assembler::assemble_boundary_linear(n_dofs, &mesh, &face_dofs, order, &[&robin_b], &[3], 3);
        for i in 0..n_dofs { rhs[i] += rr[i]; }

        let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1]);
        let bvals = vec![0.0; bdofs.len()];
        apply_dirichlet(&mut stiff, &mut rhs, &bdofs, &bvals);

        let mut u = vec![0.0; n_dofs];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
        solve_pcg_jacobi(&stiff, &rhs, &mut u, &cfg).unwrap();

        // L2 error
        let ref_elem = fem_element::lagrange::TriP1;
        let quad = ref_elem.quadrature(4);
        let mut l2_err = 0.0_f64;
        let mut phi = vec![0.0; ref_elem.n_dofs()];
        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let dofs = space.element_dofs(e);
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j;
                ref_elem.eval_basis(xi, &mut phi);
                let uh: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| u[d as usize] * p).sum();
                let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                          x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
                l2_err += w * (uh - exact(&xp)).powi(2);
            }
        }
        l2_err = l2_err.sqrt();
        assert!(l2_err < 0.02, "MMS L2 error too large: {:.4e}", l2_err);
    }
}
