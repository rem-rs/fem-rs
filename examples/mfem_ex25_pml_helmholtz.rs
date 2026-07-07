//! # Example 25 — PML Helmholtz (scalar approximation, analogous to MFEM ex25)
//!
//! Solves the complex Helmholtz equation on a unit square with absorbing
//! (PML-like) boundary layers:
//!
//! ```text
//!   −∇·(∇u) − ω²·u + i·ω·σ(x)·u = 0    in Ω
//! ```
//!
//! where σ(x) is a spatially-varying damping coefficient that grows inside
//! the PML boundary layer.  This scalar approximation mimics MFEM ex25's
//! perfectly matched layer for the Maxwell curl-curl problem.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_helmholtz
//! cargo run --example mfem_ex25_pml_helmholtz -- -m ../data/inline-quad.mesh -o 3
//! cargo run --example mfem_ex25_pml_helmholtz -- --omega 8.0 --sigma-max 4.0
//! ```

use fem_assembly::{
    ComplexAssembler, ComplexGridFunction,
    coefficient::PmlCoeff,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

fn main() {
    let args = parse_args();
    println!("=== Example 25: PML Helmholtz (scalar, MFEM ex25 analog) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{} P{}", args.n, args.n, args.order);
    }
    println!(
        "  ω = {:.4}, PML thickness = {:.3}, σ_max = {:.3}, power = {:.1}",
        args.omega, args.thickness, args.sigma_max, args.power
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

    // Spatially varying PML damping: σ grows from 0 at the inner PML boundary
    // to σ_max at the outer boundary.
    let pml_sigma = PmlCoeff::new(
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        args.thickness,
        args.sigma_max,
    )
    .with_axis_weights(vec![1.0, 1.0])
    .with_power(args.power);

    // Build complex system: (K − ω²M + i·ω·C) u = 0
    //   K = Diffusion  (stiffness, ∇·∇)
    //   M = Mass       (ω²·u  term → mass coef = 1)
    //   C = Mass(σ)    (PML damping:  k_im = ω·Mass(σ))
    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&MassIntegrator { rho: 1.0 }],
        &[&MassIntegrator { rho: pml_sigma }],
        args.omega,
        args.order * 2 + 1,
    );

    // Dirichlet BCs: fixed left-boundary drive (tag 4), zero on top/bottom (1,3)
    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let left: Vec<usize> = boundary_dofs(mesh_ref, dm, &[4])
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let other: Vec<usize> = boundary_dofs(mesh_ref, dm, &[1, 3])
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);
    sys.apply_dirichlet(
        &other,
        &vec![0.0; other.len()],
        &vec![0.0; other.len()],
        &mut rhs,
    );
    sys.apply_dirichlet(
        &left,
        &vec![args.left_drive; left.len()],
        &vec![0.0; left.len()],
        &mut rhs,
    );

    // Solve
    let a = sys.to_flat_csr();
    let mut x = vec![0.0; 2 * n];
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 3000,
        verbose: false,
        ..Default::default()
    };
    let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES did not converge");

    let gf = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::INFINITY, f64::min);

    let right: Vec<usize> = boundary_dofs(mesh_ref, dm, &[2])
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let mean_left_amp = mean(&amp, &left);
    let mean_right_amp = mean(&amp, &right);

    println!(
        "  GMRES: {} iters, residual={:.3e}, converged={}",
        res.iterations, res.final_residual, res.converged
    );
    println!("  |u| ∈ [{:.6e}, {:.6e}]", min_amp, max_amp);
    println!(
        "  Mean |u|: left={:.4e}, right={:.4e}  (reflection proxy = {:.4e})",
        mean_left_amp,
        mean_right_amp,
        mean_right_amp / mean_left_amp.max(1e-14)
    );

    assert!(res.converged, "PML solve did not converge");
    println!("  PASS");
}

fn mean(values: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    idx.iter().map(|&i| values[i]).sum::<f64>() / idx.len() as f64
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    omega: f64,
    thickness: f64,
    sigma_max: f64,
    power: f64,
    left_drive: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 12,
        order: 1,
        omega: 2.0,
        thickness: 0.2,
        sigma_max: 1.0,
        power: 2.0,
        left_drive: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("12".into())
                    .parse()
                    .unwrap_or(12)
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("1".into())
                    .parse()
                    .unwrap_or(1)
            }
            "--omega" | "-f" => {
                a.omega = it
                    .next()
                    .unwrap_or("2.0".into())
                    .parse()
                    .unwrap_or(2.0)
            }
            "--pml-thickness" => {
                a.thickness = it
                    .next()
                    .unwrap_or("0.2".into())
                    .parse()
                    .unwrap_or(0.2)
            }
            "--sigma-max" => {
                a.sigma_max = it
                    .next()
                    .unwrap_or("1.0".into())
                    .parse()
                    .unwrap_or(1.0)
            }
            "--pml-power" => {
                let v: f64 = it
                    .next()
                    .unwrap_or("2.0".into())
                    .parse()
                    .unwrap_or(2.0);
                a.power = v.max(1.0);
            }
            "--left-drive" => {
                a.left_drive = it
                    .next()
                    .unwrap_or("1.0".into())
                    .parse()
                    .unwrap_or(1.0)
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

    fn base_args() -> Args {
        Args {
            mesh: None,
            n: 8,
            order: 1,
            omega: 2.0,
            thickness: 0.2,
            sigma_max: 1.0,
            power: 2.0,
            left_drive: 1.0,
        }
    }

    #[test]
    fn ex25_pml_converges_and_has_finite_metrics() {
        let a = base_args();
        let mesh = Mesh::<2>::unit_square_tri(a.n);
        let space = H1Space::new(mesh, a.order);
        let pml_sigma = PmlCoeff::new(
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            a.thickness,
            a.sigma_max,
        )
        .with_axis_weights(vec![1.0, 1.0])
        .with_power(a.power);

        let mut sys = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            &[&MassIntegrator { rho: 1.0 }],
            &[&MassIntegrator { rho: pml_sigma }],
            a.omega,
            a.order * 2 + 1,
        );
        let n = space.n_dofs();
        let dm = space.dof_manager();
        let mesh_ref = space.mesh();
        let left: Vec<usize> = boundary_dofs(mesh_ref, dm, &[4]).into_iter().map(|d| d as usize).collect();
        let other: Vec<usize> = boundary_dofs(mesh_ref, dm, &[1, 3]).into_iter().map(|d| d as usize).collect();
        let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);
        sys.apply_dirichlet(&other, &vec![0.0; other.len()], &vec![0.0; other.len()], &mut rhs);
        sys.apply_dirichlet(&left, &vec![1.0; left.len()], &vec![0.0; left.len()], &mut rhs);
        let a_mat = sys.to_flat_csr();
        let mut x = vec![0.0; 2 * n];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 3000, verbose: false, ..Default::default() };
        let res = solve_gmres(&a_mat, &rhs, &mut x, 50, &cfg).expect("GMRES failed");

        assert!(res.converged);
        assert!(res.final_residual < 1.0e-6, "residual = {}", res.final_residual);

        let gf = ComplexGridFunction::from_flat(&x);
        let amp = gf.amplitude();
        let max_amp = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_amp = amp.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max_amp.is_finite());
        assert!(min_amp.is_finite());
        assert!(max_amp >= min_amp);
    }

    #[test]
    fn ex25_dof_count_matches_p1_h1_formula() {
        for &n in &[6usize, 10usize] {
            let mut a = base_args();
            a.n = n;
            let mesh = Mesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            assert_eq!(space.n_dofs(), (n + 1) * (n + 1));
        }
    }
}
