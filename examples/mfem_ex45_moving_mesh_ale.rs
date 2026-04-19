//! Example 45: quasi-ALE moving mesh with conservative field transfer.
//!
//! Demonstrates a lightweight dynamic mesh-update loop:
//! 1) move a boundary subset (top wall)
//! 2) smooth interior nodes (Laplacian)
//! 3) transfer field from old mesh to new mesh conservatively
//!
//! This is a quasi-ALE baseline intended as a stepping stone toward full ALE.

use std::f64::consts::PI;

use fem_assembly::transfer_h1_p1_nonmatching_l2_projection_conservative;
use fem_linalg::Vector;
use fem_mesh::{
    MeshMotionConfig,
    SimplexMesh,
    all_boundary_nodes,
    apply_node_displacement,
    laplacian_smooth_2d,
};
use fem_space::{H1Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 45: quasi-ALE moving mesh ===");
    println!(
        "  n={}, steps={}, amp={}, omega={}, smooth_iters={}",
        args.n, args.steps, args.amp, args.omega, args.smooth_iters
    );

    let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    });

    let mut max_abs_int_err = 0.0_f64;
    for step in 1..=args.steps {
        let t = step as f64 / args.steps as f64;
        let old_mesh = mesh.clone();

        // Move only top boundary nodes (y ≈ 1) with horizontal oscillation.
        let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
            .into_iter()
            .filter(|&n| {
                let p = mesh.coords_of(n);
                (p[1] - 1.0).abs() < 1.0e-12
            })
            .collect();

        let shift = args.amp * (2.0 * PI * t).sin();
        apply_node_displacement(&mut mesh, &top_nodes, |p| {
            let taper = (PI * p[0]).sin().powi(2);
            [shift * taper, 0.0]
        });

        // Keep boundary fixed during interior smoothing.
        let fixed = all_boundary_nodes(&mesh);
        let _iters_done = laplacian_smooth_2d(
            &mut mesh,
            &fixed,
            MeshMotionConfig {
                omega: args.omega,
                max_iters: args.smooth_iters,
                tol: 1.0e-12,
            },
        );

        // Conservative field transfer old -> new.
        let src = H1Space::new(old_mesh.clone(), 1);
        let dst = H1Space::new(mesh.clone(), 1);
        let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &src,
            values.as_slice(),
            &dst,
            1.0e-12,
            4,
        )
        .expect("conservative transfer should succeed");

        max_abs_int_err = max_abs_int_err.max(report.absolute_integral_error_after);
        values = Vector::from_vec(v_new);
    }

    let final_norm = values.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  final ||u||_2 = {:.6e}", final_norm);
    println!("  max absolute integral error after correction = {:.3e}", max_abs_int_err);
}

struct Args {
    n: usize,
    steps: usize,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 20,
        steps: 20,
        amp: 0.02,
        omega: 0.7,
        smooth_iters: 30,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--steps" => a.steps = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--amp" => a.amp = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02),
            "--omega" => a.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--smooth-iters" => {
                a.smooth_iters = it.next().unwrap_or("30".into()).parse().unwrap_or(30)
            }
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
    fn ex45_conservative_transfer_keeps_integral_near_machine_precision() {
        let args = Args {
            n: 10,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 15,
        };

        let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });

        let mut max_abs_int_err = 0.0_f64;
        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
                .collect();

            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [shift * taper, 0.0]
            });

            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed,
                MeshMotionConfig {
                    omega: args.omega,
                    max_iters: args.smooth_iters,
                    tol: 1.0e-12,
                },
            );

            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src,
                values.as_slice(),
                &dst,
                1.0e-12,
                4,
            )
            .unwrap();

            max_abs_int_err = max_abs_int_err.max(report.absolute_integral_error_after);
            values = Vector::from_vec(v_new);
        }

        assert!(max_abs_int_err < 1.0e-10, "integral drift too large: {max_abs_int_err}");
    }
}
