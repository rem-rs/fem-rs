//! # Example 14 �?DC current flow in a two-material conductor
//!
//! Solves the electrostatic potential equation with piecewise constant
//! conductivity (two-material domain), analogous to MFEM ex14 (DC current):
//!
//! ```text
//!   −∇·(σ ∇�? = 0    in Ω = [0,1]²
//!           φ = 0    on Γ_left  (x=0, tag 4)
//!           φ = 1    on Γ_right (x=1, tag 2)
//!    σ ∂�?∂n = 0    on Γ_top, Γ_bottom (insulated, tags 1,3) [natural BC]
//! ```
//!
//! The domain is split into two materials:
//!   - Left half  (x �?0.5): σ�?= 1.0  (low conductivity, tag 1 in elems)
//!   - Right half (x  > 0.5): σ�?= 10.0 (high conductivity, tag 2 in elems)
//!
//! Since the mesh does not align with the interface, we use `FnCoeff` to
//! assign σ(x) = σ�?for x[0] �?0.5 and σ�?otherwise, and `PWConstCoeff`
//! to demonstrate the tag-based approach on the same problem.
//!
//! Post-processing: compute the element-wise electric field E = −∇φ
//! and the current density J = σ E, and export to VTK.
//!
//! ## Physics
//! For a 1D cross-section (two layers in series), the exact solution is:
//! ```text
//!   φ(x) = { σ�?(σ�?σ�? * (2x)           for x �?0.5
//!           { 1 - σ�?(σ�?σ�? * 2(1-x)     for x > 0.5
//! ```
//! The normal current J·n = σ ∂�?∂x is continuous at x=0.5 (conservation).
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex14_dc_current
//! cargo run --example mfem_ex14_dc_current -- --n 32 --sigma1 1.0 --sigma2 100.0
//! cargo run --example mfem_ex14_dc_current -- --n 16 --vtk output.vtu
//! ```

use fem_assembly::postprocess::compute_element_gradients;
use fem_assembly::{coefficient::FnCoeff, standard::DiffusionIntegrator, Assembler};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_gmres, solve_pcg_jacobi, SolverConfig};
use fem_space::{
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
    H1Space,
};

struct CaseResult {
    iterations: usize,
    final_residual: f64,
    converged: bool,
    phi_l2: f64,
    phi_rms_error: f64,
    phi_at_interface: f64,
    phi_interface_exact: f64,
    phi_checksum: f64,
}

fn main() {
    let args = parse_args();

    let result = solve_case(&args);

    println!("=== fem-rs Example 14: DC current flow (two-material conductor) ===");
    println!("  Mesh: {}×{}, P1 elements", args.n, args.n);
    println!(
        "  σ�?= {} (x �?0.5),  σ�?= {} (x > 0.5)",
        args.sigma1, args.sigma2
    );
    println!("  BCs: φ=0 on left (tag 4),  φ={} on right (tag 2)", args.voltage);
    println!("       Insulated (natural Neumann) on top+bottom (tags 1,3)");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  ||phi||₂ = {:.6e}", result.phi_l2);
    println!(
        "  Interface check: φ_h ≈ {:.6},  φ_exact(0.5) = {:.6},  diff = {:.2e}",
        result.phi_at_interface,
        result.phi_interface_exact,
        (result.phi_at_interface - result.phi_interface_exact).abs()
    );
    println!("  RMS nodal error vs layered exact profile = {:.3e}", result.phi_rms_error);
    println!("  checksum(phi) = {:.8e}", result.phi_checksum);

    if let Some(ref path) = args.vtk {
        let (_, _, phi, e_field, j_field) = solve_case_with_fields(&args);
        let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let space = H1Space::new(mesh, 1);
        let mut writer = VtkWriter::new(space.mesh());
        writer.add_point_data(DataArray {
            name: "phi".to_string(),
            n_components: 1,
            values: phi,
        });
        writer.add_cell_data(DataArray {
            name: "E".to_string(),
            n_components: 2,
            values: e_field,
        });
        writer.add_cell_data(DataArray {
            name: "J".to_string(),
            n_components: 2,
            values: j_field,
        });
        writer.write_file(path).expect("VTK write failed");
        println!("  VTK written to: {path}");
    } else {
        println!("  (Pass --vtk output.vtu to write VTK file)");
    }

    println!("Done.");
}

fn solve_case(args: &Args) -> CaseResult {
    let (result, _, _, _, _) = solve_case_with_fields(args);
    result
}

fn exact_phi(x: f64, sigma1: f64, sigma2: f64, voltage: f64) -> f64 {
    if x <= 0.5 {
        voltage * (2.0 * sigma2 * x) / (sigma1 + sigma2)
    } else {
        voltage * (1.0 - (2.0 * sigma1 * (1.0 - x)) / (sigma1 + sigma2))
    }
}

fn solve_case_with_fields(args: &Args) -> (CaseResult, H1Space<SimplexMesh<2>>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let sigma1 = args.sigma1;
    let sigma2 = args.sigma2;

    // ─── 1. Mesh and H¹ space ────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    // ─── 2. Piecewise conductivity σ(x) ─────────────────────────────────────
    //   FnCoeff: σ is a closure over the physical coordinate x[0].
    //   No mesh tag needed �?just the coordinate decides the material.
    let sigma_fn = FnCoeff(move |x: &[f64]| if x[0] <= 0.5 { sigma1 } else { sigma2 });

    // ─── 3. Assemble −∇·(σ ∇�? bilinear form ────────────────────────────────
    let mut mat =
        Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: sigma_fn }], 3);

    // ─── 4. RHS = 0 (no volume sources) ─────────────────────────────────────
    let mut rhs = vec![0.0_f64; n];

    // ─── 5. Apply Dirichlet BCs ──────────────────────────────────────────────
    //   φ = 0 on left wall (tag 4), φ = 1 on right wall (tag 2)
    let dm = space.dof_manager();

    let left_bnd = boundary_dofs(space.mesh(), dm, &[4]);
    let right_bnd = boundary_dofs(space.mesh(), dm, &[2]);

    let left_vals: Vec<f64> = vec![0.0; left_bnd.len()];
    let right_vals: Vec<f64> = vec![args.voltage; right_bnd.len()];

    apply_dirichlet(&mut mat, &mut rhs, &left_bnd, &left_vals);
    apply_dirichlet(&mut mat, &mut rhs, &right_bnd, &right_vals);

    // ─── 6. Solve ─────────────────────────────────────────────────────────────
    let mut phi = vec![0.0_f64; n];
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = match solve_pcg_jacobi(&mat, &rhs, &mut phi, &cfg) {
        Ok(r) => r,
        Err(_) => {
            // Dirichlet elimination can produce non-ideal conditioning for CG;
            // use GMRES as a robust fallback for this mixed-BC conductivity case.
            solve_gmres(&mat, &rhs, &mut phi, 50, &cfg).expect("solver failed")
        }
    };

    // ─── 7. Post-processing ───────────────────────────────────────────────────
    // Electric field E = -∇�?(element-wise, at centroid)
    let grads = compute_element_gradients(&space, &phi);
    let n_elems = space.mesh().n_elements();

    // Current density J = σ E = -σ ∇�?(element-wise)
    let mut j_field = vec![0.0_f64; n_elems * 2];
    let mut e_field = vec![0.0_f64; n_elems * 2];
    for e in 0..n_elems {
        // Centroid x-coord (average of triangle vertices)
        let nodes = space.mesh().element_nodes(e as u32);
        let xc: f64 = nodes
            .iter()
            .map(|&nd| space.mesh().node_coords(nd)[0])
            .sum::<f64>()
            / nodes.len() as f64;
        let sigma = if xc <= 0.5 { sigma1 } else { sigma2 };

        e_field[e * 2] = -grads[e][0];
        e_field[e * 2 + 1] = -grads[e][1];
        j_field[e * 2] = sigma * e_field[e * 2];
        j_field[e * 2 + 1] = sigma * e_field[e * 2 + 1];
    }

    // ─── 8. 1D analytical check along y = 0.5 ────────────────────────────────
    // For the 1D layered problem the exact potential at the interface (x=0.5) is:
    //   φ(0.5) = V_right * σ₂ / (σ₁ + σ₂)
    let phi_interface_exact = args.voltage * sigma2 / (sigma1 + sigma2);
    // Find the node closest to (0.5, 0.5) and compare.
    let mut best_dist = f64::MAX;
    let mut phi_at_interface = 0.0;
    for i in 0..space.mesh().n_nodes() {
        let x = space.mesh().node_coords(i as u32);
        let d = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
        if d < best_dist {
            best_dist = d;
            phi_at_interface = phi[i];
        }
    }

    let phi_l2 = phi.iter().map(|v| v * v).sum::<f64>().sqrt();
    let phi_checksum = phi
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum();
    let phi_err2: f64 = (0..space.mesh().n_nodes())
        .map(|i| {
            let x = space.mesh().node_coords(i as u32);
            let exact = exact_phi(x[0], sigma1, sigma2, args.voltage);
            let diff = phi[i] - exact;
            diff * diff
        })
        .sum();
    let phi_rms_error = (phi_err2 / space.mesh().n_nodes() as f64).sqrt();

    (
        CaseResult {
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            phi_l2,
            phi_rms_error,
            phi_at_interface,
            phi_interface_exact,
            phi_checksum,
        },
        space,
        phi,
        e_field,
        j_field,
    )
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    sigma1: f64,
    sigma2: f64,
    voltage: f64,
    vtk: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 16,
        sigma1: 1.0,
        sigma2: 10.0,
        voltage: 1.0,
        vtk: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16);
            }
            "--sigma1" => {
                a.sigma1 = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            "--sigma2" => {
                a.sigma2 = it.next().unwrap_or("10.0".into()).parse().unwrap_or(10.0);
            }
            "--voltage" => {
                a.voltage = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            "--vtk" => {
                a.vtk = it.next();
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex14_dc_current_matches_series_interface_potential() {
        let result = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(result.converged);
        assert!(result.final_residual < 1.0e-10, "residual = {}", result.final_residual);
        assert!(
            (result.phi_at_interface - result.phi_interface_exact).abs() < 1.0e-10,
            "interface potential mismatch: numerical={} exact={}",
            result.phi_at_interface,
            result.phi_interface_exact
        );
        assert!(result.phi_rms_error < 1.0e-10,
            "exact layered profile should be reproduced nodally, got RMS error {}",
            result.phi_rms_error);
    }

    #[test]
    fn ex14_dc_current_solution_scales_linearly_with_applied_voltage() {
        let half = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 0.5,
            vtk: None,
        });
        let full = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(half.converged && full.converged);
        let phi_ratio = full.phi_l2 / half.phi_l2.max(1.0e-30);
        let interface_ratio = full.phi_at_interface / half.phi_at_interface.max(1.0e-30);

        assert!(
            (phi_ratio - 2.0).abs() < 1.0e-10,
            "expected potential norm to scale linearly with voltage, got ratio {}",
            phi_ratio
        );
        assert!(
            (interface_ratio - 2.0).abs() < 1.0e-10,
            "expected interface potential to scale linearly with voltage, got ratio {}",
            interface_ratio
        );
        assert!(
            (full.phi_checksum / half.phi_checksum - 2.0).abs() < 1.0e-10,
            "expected checksum to scale linearly with voltage, got ratio {}",
            full.phi_checksum / half.phi_checksum
        );
    }

    #[test]
    fn ex14_dc_current_higher_right_conductivity_raises_interface_potential() {
        let mild_contrast = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 2.0,
            voltage: 1.0,
            vtk: None,
        });
        let strong_contrast = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(mild_contrast.converged && strong_contrast.converged);
        assert!(
            strong_contrast.phi_at_interface > mild_contrast.phi_at_interface,
            "expected larger right conductivity to push more voltage drop into the left layer: mild={} strong={}",
            mild_contrast.phi_at_interface,
            strong_contrast.phi_at_interface
        );
    }

    #[test]
    fn ex14_dc_current_zero_voltage_gives_trivial_solution() {
        let result = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 0.0,
            vtk: None,
        });

        assert!(result.converged);
        assert!(result.phi_l2 < 1.0e-14, "expected zero potential norm, got {}", result.phi_l2);
        assert!(result.phi_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}", result.phi_checksum);
        assert!(result.phi_rms_error < 1.0e-14,
            "expected zero exact-profile error, got {}", result.phi_rms_error);
    }

    #[test]
    fn ex14_dc_current_refinement_preserves_exact_layered_profile() {
        let coarse = solve_case(&Args {
            n: 4,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });
        let fine = solve_case(&Args {
            n: 16,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(coarse.converged && fine.converged);
        assert!(coarse.phi_rms_error < 5.0e-12,
            "coarse grid should retain machine-precision exact-profile agreement: {}",
            coarse.phi_rms_error);
        assert!(fine.phi_rms_error < 5.0e-12,
            "fine grid should retain machine-precision exact-profile agreement: {}",
            fine.phi_rms_error);
        assert!((fine.phi_at_interface - fine.phi_interface_exact).abs() < 1.0e-10);
    }

    #[test]
    fn ex14_dc_current_equal_conductivities_give_midpoint_half_voltage() {
        let result = solve_case(&Args {
            n: 8,
            sigma1: 3.0,
            sigma2: 3.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(result.converged);
        assert!((result.phi_interface_exact - 0.5).abs() < 1.0e-14);
        assert!((result.phi_at_interface - 0.5).abs() < 1.0e-10,
            "equal conductivities should place interface at half voltage: {}", result.phi_at_interface);
        assert!(result.phi_rms_error < 1.0e-10);
    }

    #[test]
    fn ex14_dc_current_scaling_both_conductivities_keeps_solution_invariant() {
        let base = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });
        let scaled = solve_case(&Args {
            n: 8,
            sigma1: 2.0,
            sigma2: 20.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(base.converged && scaled.converged);
        assert!((base.phi_at_interface - scaled.phi_at_interface).abs() < 1.0e-10);
        assert!((base.phi_l2 - scaled.phi_l2).abs() < 1.0e-10);
        assert!((base.phi_checksum - scaled.phi_checksum).abs() < 1.0e-10);
    }

    #[test]
    fn ex14_dc_current_swapping_layer_conductivities_complements_interface_value() {
        let left_low = solve_case(&Args {
            n: 8,
            sigma1: 1.0,
            sigma2: 10.0,
            voltage: 1.0,
            vtk: None,
        });
        let left_high = solve_case(&Args {
            n: 8,
            sigma1: 10.0,
            sigma2: 1.0,
            voltage: 1.0,
            vtk: None,
        });

        assert!(left_low.converged && left_high.converged);
        assert!((left_low.phi_at_interface + left_high.phi_at_interface - 1.0).abs() < 1.0e-8,
            "swapping layer conductivities should complement interface potential: low={} high={}",
            left_low.phi_at_interface,
            left_high.phi_at_interface);
        assert!((left_low.phi_interface_exact + left_high.phi_interface_exact - 1.0).abs() < 1.0e-14);
    }
}

