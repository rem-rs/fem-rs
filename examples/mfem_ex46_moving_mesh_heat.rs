//! Example 46: moving-mesh transient heat solve (quasi-ALE workflow).
//!
//! Per time step:
//! 1) update mesh geometry (prescribed top-wall motion + interior smoothing)
//! 2) conservatively transfer temperature old-mesh -> new-mesh
//! 3) reassemble on the new mesh and advance one implicit-Euler heat step

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::FnVectorCoeff,
    standard::{DiffusionIntegrator, MassIntegrator},
    standard::ConvectionIntegrator,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
};
use fem_linalg::{CooMatrix, CsrMatrix, Vector};
use fem_mesh::{
    MeshMotionConfig,
    SimplexMesh,
    all_boundary_nodes,
    apply_node_displacement,
    laplacian_smooth_2d,
};
use fem_solver::{SolverConfig, solve_pcg_jacobi};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

struct SolveResult {
    final_time: f64,
    n_dofs: usize,
    final_l2: f64,
    final_checksum: f64,
    max_transfer_abs_int_err: f64,
    l2_history: Vec<f64>,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 46: moving-mesh transient heat ===");
    println!(
        "  n={}, dt={}, T={}, kappa={}, amp={}, omega={}, smooth_iters={}",
        args.n, args.dt, args.t_end, args.kappa, args.amp, args.omega, args.smooth_iters
    );

    let result = solve_case(
        args.n,
        args.dt,
        args.t_end,
        args.kappa,
        args.amp,
        args.omega,
        args.smooth_iters,
        args.mesh_advection,
    );

    println!("  final time      = {:.6e}", result.final_time);
    println!("  dofs            = {}", result.n_dofs);
    println!("  recorded L2 samples = {}", result.l2_history.len());
    println!("  final ||u||_2   = {:.6e}", result.final_l2);
    println!("  final checksum  = {:.8e}", result.final_checksum);
    println!(
        "  max transfer integral error = {:.3e}",
        result.max_transfer_abs_int_err
    );
}

fn solve_case(
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    mesh_advection: bool,
) -> SolveResult {
    let mut mesh = SimplexMesh::<2>::unit_square_tri(n);
    let mut space = H1Space::new(mesh.clone(), 1);
    let mut u = space.interpolate(&|x| (PI * x[0]).sin() * (PI * x[1]).sin());

    let n_steps = (t_end / dt).ceil() as usize;
    let mut t = 0.0_f64;
    let mut prev_shift = 0.0_f64;

    let solve_cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 1200,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut max_transfer_abs_int_err = 0.0_f64;
    let mut l2_history = Vec::<f64>::with_capacity(n_steps + 1);
    l2_history.push(l2_norm(u.as_slice()));

    for step in 1..=n_steps {
        let phase = step as f64 / n_steps as f64;
        let old_mesh = mesh.clone();

        let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
            .into_iter()
            .filter(|&nid| (mesh.coords_of(nid)[1] - 1.0).abs() < 1.0e-12)
            .collect();

        // Use incremental boundary motion to avoid cumulative drift bias.
        let target_shift = amp * (2.0 * PI * phase).sin();
        let delta_shift = target_shift - prev_shift;
        prev_shift = target_shift;

        apply_node_displacement(&mut mesh, &top_nodes, |p| {
            let taper = (PI * p[0]).sin().powi(2);
            [delta_shift * taper, 0.0]
        });

        let fixed = all_boundary_nodes(&mesh);
        let _ = laplacian_smooth_2d(
            &mut mesh,
            &fixed,
            MeshMotionConfig {
                omega,
                max_iters: smooth_iters,
                tol: 1.0e-12,
            },
        );

        let src = H1Space::new(old_mesh, 1);
        let dst = H1Space::new(mesh.clone(), 1);
        let (u_transfer, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &src,
            u.as_slice(),
            &dst,
            1.0e-12,
            4,
        )
        .expect("conservative transfer should succeed");
        max_transfer_abs_int_err =
            max_transfer_abs_int_err.max(report.absolute_integral_error_after);

        space = dst;
        let n_dofs = space.n_dofs();
        let mut u_vec = u_transfer;

        let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa }], 3);

        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let mut rhs = vec![0.0_f64; n_dofs];
        m_mat.spmv(&u_vec, &mut rhs);

        let sys = if mesh_advection {
            let shift_rate = delta_shift / dt.max(1.0e-14);
            let conv = Assembler::assemble_bilinear(
                &space,
                &[&ConvectionIntegrator {
                    velocity: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                        let taper = (PI * x[0]).sin().powi(2);
                        out[0] = shift_rate * taper;
                        out[1] = 0.0;
                    }),
                }],
                3,
            );
            add_csr_scaled3(&m_mat, &k_mat, 1.0, &conv, 1.0, dt)
        } else {
            add_csr_scaled3(&m_mat, &k_mat, 1.0, &k_mat, 0.0, dt)
        };

        let mut sys = sys;
        let vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut sys, &mut rhs, &bnd, &vals);

        let mut u_new = vec![0.0_f64; n_dofs];
        let _ = solve_pcg_jacobi(&sys, &rhs, &mut u_new, &solve_cfg);
        for &d in &bnd {
            u_new[d as usize] = 0.0;
        }

        u_vec = u_new;
        u = Vector::from_vec(u_vec);

        t += dt;
        l2_history.push(l2_norm(u.as_slice()));
    }

    let final_l2 = l2_norm(u.as_slice());
    let final_checksum = checksum(u.as_slice());
    SolveResult {
        final_time: t,
        n_dofs: space.n_dofs(),
        final_l2,
        final_checksum,
        max_transfer_abs_int_err,
        l2_history,
    }
}

fn add_csr_scaled3(
    m: &CsrMatrix<f64>,
    k: &CsrMatrix<f64>,
    k_scale: f64,
    c: &CsrMatrix<f64>,
    c_scale: f64,
    dt: f64,
) -> CsrMatrix<f64> {
    let n = m.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for ptr in m.row_ptr[i]..m.row_ptr[i + 1] {
            coo.add(i, m.col_idx[ptr] as usize, m.values[ptr]);
        }
    }
    for i in 0..n {
        for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
            coo.add(i, k.col_idx[ptr] as usize, dt * k_scale * k.values[ptr]);
        }
    }
    for i in 0..n {
        for ptr in c.row_ptr[i]..c.row_ptr[i + 1] {
            coo.add(i, c.col_idx[ptr] as usize, dt * c_scale * c.values[ptr]);
        }
    }
    coo.into_csr()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn checksum(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, val)| (i as f64 + 1.0) * val)
        .sum::<f64>()
}

struct Args {
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    mesh_advection: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 20,
        dt: 0.01,
        t_end: 0.2,
        kappa: 1.0,
        amp: 0.015,
        omega: 0.7,
        smooth_iters: 25,
        mesh_advection: true,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--dt" => a.dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--T" => a.t_end = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--kappa" => a.kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--amp" => a.amp = it.next().unwrap_or("0.015".into()).parse().unwrap_or(0.015),
            "--omega" => a.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--smooth-iters" => {
                a.smooth_iters = it.next().unwrap_or("25".into()).parse().unwrap_or(25)
            }
            "--mesh-advection" => a.mesh_advection = true,
            "--no-mesh-advection" => a.mesh_advection = false,
            _ => {}
        }
    }

    a.omega = a.omega.clamp(0.05, 0.95);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex46_moving_mesh_heat_has_stable_decay_and_conservative_transfer() {
        let r = solve_case(10, 0.02, 0.2, 1.0, 0.0, 0.7, 15, true);
        assert_eq!(r.n_dofs, 121);
        assert!(r.max_transfer_abs_int_err < 1.0e-10, "transfer drift too large: {}", r.max_transfer_abs_int_err);
        assert!(r.final_l2.is_finite());
        assert!(r.final_l2 > 0.0);

        for i in 1..r.l2_history.len() {
            assert!(
                r.l2_history[i] <= r.l2_history[i - 1] + 1.0e-8,
                "L2 should be non-increasing for implicit heat step: prev={} cur={} at i={}",
                r.l2_history[i - 1],
                r.l2_history[i],
                i
            );
        }
    }

    #[test]
    fn ex46_mesh_advection_changes_solution_when_mesh_moves() {
        let no_adv = solve_case(10, 0.02, 0.2, 1.0, 0.01, 0.7, 15, false);
        let with_adv = solve_case(10, 0.02, 0.2, 1.0, 0.01, 0.7, 15, true);
        let diff = (with_adv.final_checksum - no_adv.final_checksum).abs();
        assert!(
            diff > 1.0e-8,
            "mesh-advection switch should alter solution for moving mesh, checksum diff={diff}"
        );
    }
}
