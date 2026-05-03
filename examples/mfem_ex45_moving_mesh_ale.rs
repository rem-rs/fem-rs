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

    /// With zero amplitude the mesh does not move, so the conservative transfer
    /// should produce zero integral error and the field should be unchanged.
    #[test]
    fn ex45_zero_amplitude_mesh_stays_unchanged_and_transfer_is_exact() {
        let args = Args { n: 8, steps: 3, amp: 0.0, omega: 0.7, smooth_iters: 10 };

        let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let space0 = H1Space::new(mesh.clone(), 1);
        let initial_values: Vec<f64> = space0.interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        }).as_slice().to_vec();
        let initial_norm: f64 = initial_values.iter().map(|v| v*v).sum::<f64>().sqrt();
        let mut values = Vector::from_vec(initial_values.clone());

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin(); // = 0
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [shift * taper, 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });
            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            assert!(report.absolute_integral_error_after < 1.0e-12,
                "step {step}: non-trivial integral error on static mesh: {}", report.absolute_integral_error_after);
            values = Vector::from_vec(v_new);
        }

        // Field norm should be preserved (same mesh, same field).
        let final_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();
        let rel_drift = (final_norm - initial_norm).abs() / initial_norm.max(1.0e-300);
        assert!(rel_drift < 1.0e-10, "field norm drifted on static mesh: rel={rel_drift:.3e}");
    }

    /// After mesh motion with moderate amplitude, all triangle areas must remain
    /// strictly positive (no element inversion due to smoothing).
    #[test]
    fn ex45_mesh_remains_valid_no_inverted_elements_after_motion() {
        let args = Args { n: 14, steps: 6, amp: 0.015, omega: 0.7, smooth_iters: 30 };

        let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [shift * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });

            // Check all element areas are positive.
            for e in 0..mesh.n_elems() as u32 {
                let nodes = mesh.elem_nodes(e);
                let p0 = mesh.coords_of(nodes[0]);
                let p1 = mesh.coords_of(nodes[1]);
                let p2 = mesh.coords_of(nodes[2]);
                let area = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]));
                assert!(area > 0.0, "step {step}: inverted element {e}, area={area:.6e}");
            }

            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, _report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            values = Vector::from_vec(v_new);
        }
    }

    /// The L2 norm of the transferred field should not blow up over many steps
    /// (no exponential growth from repeated conservative transfer).
    #[test]
    fn ex45_field_norm_is_stable_over_many_steps() {
        let args = Args { n: 10, steps: 10, amp: 0.01, omega: 0.7, smooth_iters: 20 };

        let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let initial_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [shift * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });
            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, _) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            values = Vector::from_vec(v_new);
        }

        let final_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();
        // Norm may change slightly due to mesh deformation but must not grow unboundedly.
        assert!(final_norm < 5.0 * initial_norm,
            "field norm grew unexpectedly: initial={initial_norm:.4e} final={final_norm:.4e}");
        assert!(final_norm > 0.0, "field collapsed to zero");
    }

    #[test]
    fn ex45_dof_count_matches_p1_h1_formula_for_multiple_meshes() {
        for &n in &[6usize, 10usize, 14usize] {
            let mesh = SimplexMesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            assert_eq!(space.n_dofs(), (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex45_top_boundary_nodes_are_detected_for_motion() {
        let mesh = SimplexMesh::<2>::unit_square_tri(12);
        let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
            .into_iter()
            .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
            .collect();
        assert!(!top_nodes.is_empty(), "expected non-empty top boundary node set");

        let n_top = top_nodes.len();
        let expected_min = 13usize;
        assert!(n_top >= expected_min,
            "top boundary should have at least n+1 nodes, got {n_top}");
    }

    #[test]
    fn ex45_stronger_motion_still_preserves_integral_after_correction() {
        let args = Args { n: 10, steps: 5, amp: 0.02, omega: 0.7, smooth_iters: 20 };

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
                [shift * (PI * p[0]).sin().powi(2), 0.0]
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

            let src = H1Space::new(old_mesh, 1);
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

        assert!(max_abs_int_err < 5.0e-10,
            "integral correction drift too large under stronger motion: {max_abs_int_err}");
    }

    #[test]
    fn ex45_smoothing_parameter_variation_keeps_mesh_valid() {
        for &omega in &[0.3_f64, 0.7_f64, 0.9_f64] {
            let mut mesh = SimplexMesh::<2>::unit_square_tri(10);
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
                .collect();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [0.01 * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed,
                MeshMotionConfig {
                    omega,
                    max_iters: 25,
                    tol: 1.0e-12,
                },
            );

            for e in 0..mesh.n_elems() as u32 {
                let nodes = mesh.elem_nodes(e);
                let p0 = mesh.coords_of(nodes[0]);
                let p1 = mesh.coords_of(nodes[1]);
                let p2 = mesh.coords_of(nodes[2]);
                let area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
                assert!(area > 0.0, "omega={omega}: inverted element {e}, area={area:.6e}");
            }
        }
    }
}
