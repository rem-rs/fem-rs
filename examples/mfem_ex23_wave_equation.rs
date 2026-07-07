//! # Example 23 — Wave Equation  (analogous to MFEM ex23)
//!
//! Solves the scalar wave equation:
//!
//! ```text
//!   d²u/dt² = c² ∇²u    in Ω
//!        u  = 0          on ∂Ω  (Dirichlet BC)
//! ```
//!
//! with initial condition `u(x,0) = exp(−30‖x‖²)` and zero initial velocity,
//! matching MFEM ex23.
//!
//! Time integration uses the Newmark-beta scheme (β=¼, γ=½).
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex23_wave_equation
//! cargo run --example mfem_ex23_wave_equation -- -m ../data/star.mesh
//! cargo run --example mfem_ex23_wave_equation -- -m ../data/inline-hex.mesh -c 2.0 --dt 0.005
//! ```

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_solver::{solve_cg, SolverConfig, Newmark, NewmarkState};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

/// Gaussian initial condition (MFEM ex23): u(x,0) = exp(-30·‖x‖²)
fn initial_solution(x: &[f64]) -> f64 {
    let r2 = x.iter().map(|c| c * c).sum::<f64>();
    (-30.0 * r2).exp()
}

fn main() {
    let args = parse_args();
    println!("=== Example 23: Wave Equation (MFEM ex23) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{} P{}", args.n, args.n, args.order);
    }
    println!(
        "  c = {:.4}, dt = {:.4}, T = {:.4}, scheme = Newmark",
        args.c, args.dt, args.t_final
    );

    // Load or generate mesh
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  DOFs: {n}");

    // Assemble stiffness (K = c²·Diffusion) and mass (M) matrices
    let diff_coeff = args.c * args.c;
    let stiff = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator {
            kappa: diff_coeff,
        }],
        args.order * 2 + 1,
    );
    let mass = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: 1.0 }],
        args.order * 2 + 1,
    );
    let zero_rhs =
        Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 0.0)], 3);

    // Dirichlet BC: u = 0 on all boundaries
    let bdofs = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4]);
    let bvals = vec![0.0; bdofs.len()];

    let mut stiff_bc = stiff.clone();
    let mut mass_bc = mass.clone();
    let mut rhs_bc = zero_rhs.clone();
    apply_dirichlet(&mut stiff_bc, &mut rhs_bc, &bdofs, &bvals);
    let mut rhs_bc_mass = zero_rhs.clone();
    apply_dirichlet(&mut mass_bc, &mut rhs_bc_mass, &bdofs, &bvals);

    // Initial condition: u₀ = exp(-30‖x‖²)
    let mut u = vec![0.0; n];
    for dof in 0..n as u32 {
        let coord = space.dof_manager().dof_coord(dof);
        u[dof as usize] = initial_solution(&coord);
    }
    for &d in &bdofs {
        u[d as usize] = 0.0;
    }

    // Initial acceleration: a₀ = M⁻¹(-K u₀)
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 5000,
        verbose: false,
        ..Default::default()
    };
    let mut ku = vec![0.0; n];
    stiff.spmv(&u, &mut ku);
    let mut rhs_a = vec![0.0; n];
    for i in 0..n {
        rhs_a[i] = -ku[i];
    }
    for &d in &bdofs {
        rhs_a[d as usize] = 0.0;
    }
    let mut a0 = vec![0.0; n];
    solve_cg(&mass_bc, &rhs_a, &mut a0, &cfg).unwrap();

    let mut state = NewmarkState::new(n);
    state.acc.copy_from_slice(&a0);

    let t_end = args.t_final;
    let dt = args.dt;
    let n_steps = (t_end / dt).round() as usize;
    let t_final = n_steps as f64 * dt;

    // Time integration loop
    let newmark = Newmark::default();
    let mut u_hist = u.clone();
    for step in 0..n_steps {
        newmark.step(&mass_bc, &stiff_bc, &rhs_bc, dt, &mut u_hist, &mut state, &[]);
        if (step + 1) % std::cmp::max(n_steps / 5, 1) == 0 {
            let t = (step + 1) as f64 * dt;
            println!("  t = {t:.4}, max|u| = {:.6e}", max_abs(&u_hist));
        }
    }

    println!(
        "  Final: max|u| = {:.6e} at t = {t_final}",
        max_abs(&u_hist)
    );
    println!("  ‖u‖₂ = {:.6e}", u_hist.iter().map(|v| v * v).sum::<f64>().sqrt());
    println!("Done.");
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()))
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    c: f64,
    dt: f64,
    t_final: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 20,
        order: 1,
        c: 1.0,
        dt: 0.001,
        t_final: 0.5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("20".into())
                    .parse()
                    .unwrap_or(20)
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("1".into())
                    .parse()
                    .unwrap_or(1)
            }
            "-c" | "--c" | "--speed" => {
                a.c = it
                    .next()
                    .unwrap_or("1.0".into())
                    .parse()
                    .unwrap_or(1.0)
            }
            "--dt" => {
                a.dt = it
                    .next()
                    .unwrap_or("0.001".into())
                    .parse()
                    .unwrap_or(0.001)
            }
            "--T" | "--t-final" => {
                a.t_final = it
                    .next()
                    .unwrap_or("0.5".into())
                    .parse()
                    .unwrap_or(0.5)
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
    fn ex23_wave_coarse_newmark_converges() {
        let n = 8;
        let order = 1;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, order);
        let dofs = space.n_dofs();

        let diff_coeff = 1.0;
        let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: diff_coeff }], 3);
        let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 0.0)], 3);

        let bdofs = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4]);
        let bvals = vec![0.0; bdofs.len()];
        let mut stiff_bc = stiff.clone();
        let mut mass_bc = mass.clone();
        let mut rhs_bc = rhs.clone();
        apply_dirichlet(&mut stiff_bc, &mut rhs_bc, &bdofs, &bvals);
        apply_dirichlet(&mut mass_bc, &mut Vec::new(), &bdofs, &bvals);

        let mut u = vec![0.0; dofs];
        for d in 0..dofs as u32 {
            let coord = space.dof_manager().dof_coord(d);
            u[d as usize] = initial_solution(&coord);
        }
        for &d in &bdofs { u[d as usize] = 0.0; }

        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
        let mut ku = vec![0.0; dofs];
        stiff.spmv(&u, &mut ku);
        let mut rhs_a = vec![0.0; dofs];
        for i in 0..dofs { rhs_a[i] = -ku[i]; }
        for &d in &bdofs { rhs_a[d as usize] = 0.0; }
        let mut a0 = vec![0.0; dofs];
        solve_cg(&mass_bc, &rhs_a, &mut a0, &cfg).unwrap();

        let mut state = NewmarkState::new(dofs);
        state.acc.copy_from_slice(&a0);
        let newmark = Newmark::default();
        let mut u_hist = u.clone();
        let dt = 0.01;
        let n_steps = 10;
        for _ in 0..n_steps {
            newmark.step(&mass_bc, &stiff_bc, &rhs_bc, dt, &mut u_hist, &mut state, &[]);
        }

        let final_norm: f64 = u_hist.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(final_norm.is_finite());
        assert!(final_norm > 0.0, "solution should be non-trivial");
    }
}
