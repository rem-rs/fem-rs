//! mfem_ex25 - improved PML-like complex Helmholtz demo.
//!
//! Uses a spatially varying damping coefficient in boundary layers to mimic
//! absorbing behavior and reports a reflection proxy on the opposite boundary.

use fem_assembly::{
    ComplexAssembler, ComplexGridFunction,
    coefficient::PmlCoeff,
    BilinearIntegrator, QpData,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== mfem_ex25: improved complex PML-like damping ===");
    println!(
        "  n={}, omega={}, pml_thickness={}, sigma_max={}, power={}, wx={}, wy={}, stretch_blend={}, left_drive_amp={}",
        args.n,
        args.omega,
        args.thickness,
        args.sigma_max,
        args.power,
        args.wx,
        args.wy,
        args.stretch_blend,
        args.left_drive_amp
    );
    println!(
        "  dofs={}, GMRES iters={}, res={:.3e}, converged={}",
        result.n_dofs,
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!(
        "  |u| mean: left={:.4e}, right={:.4e}, interior={:.4e}",
        result.mean_left_amp,
        result.mean_right_amp,
        result.mean_interior_amp
    );
    println!("  Reflection proxy (right/left) = {:.4e}", result.reflection_ratio());
    println!("  |u| range: [{:.4}, {:.4}]", result.min_amp, result.max_amp);

    assert!(result.converged, "PML solve did not converge");
    println!("  PASS");
}

struct SolveResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    mean_left_amp: f64,
    mean_right_amp: f64,
    mean_interior_amp: f64,
    min_amp: f64,
    max_amp: f64,
}

impl SolveResult {
    fn reflection_ratio(&self) -> f64 {
        self.mean_right_amp / self.mean_left_amp.max(1.0e-14)
    }
}

/// Stretch-inspired anisotropic diffusion used in PML layers.
///
/// We blend identity and a diagonal stretch tensor:
/// A_kk = (1-b) + b/(1 + sigma_k),  b in [0,1].
/// b=0 recovers the baseline isotropic stiffness.
struct PmlStretchDiffusionIntegrator {
    profile: PmlCoeff,
    blend: f64,
}

impl BilinearIntegrator for PmlStretchDiffusionIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let w = qp.weight;

        let mut akk = vec![1.0_f64; d];
        for (k, a) in akk.iter_mut().enumerate().take(d) {
            let s = self.profile.axis_sigma(qp.x_phys, k);
            *a = (1.0 - self.blend) + self.blend / (1.0 + s);
        }

        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0_f64;
                for (k, a) in akk.iter().enumerate().take(d) {
                    val += *a * qp.grad_phys[i * d + k] * qp.grad_phys[j * d + k];
                }
                k_elem[i * n + j] += w * val;
            }
        }
    }
}

fn solve_case(args: &Args) -> SolveResult {
    solve_case_with_field(args).0
}

fn solve_case_with_field(args: &Args) -> (SolveResult, ComplexGridFunction) {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    let pml_sigma = PmlCoeff::new(
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        args.thickness,
        args.sigma_max,
    )
    .with_axis_weights(vec![args.wx, args.wy])
    .with_power(args.power);

    let pml_stretch = PmlStretchDiffusionIntegrator {
        profile: pml_sigma.clone(),
        blend: args.stretch_blend,
    };

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }, &pml_stretch],
        &[&MassIntegrator { rho: 1.0 }],
        &[&MassIntegrator { rho: pml_sigma }],
        args.omega,
        3,
    );

    let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);

    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let left: Vec<usize> = boundary_dofs(mesh_ref, dm, &[4]).into_iter().map(|d| d as usize).collect();
    let right: Vec<usize> = boundary_dofs(mesh_ref, dm, &[2]).into_iter().map(|d| d as usize).collect();
    // Left boundary is driven; top+bottom are clamped; right boundary is left open
    // to observe outgoing-wave attenuation in the PML region.
    let other: Vec<usize> = boundary_dofs(mesh_ref, dm, &[1, 3]).into_iter().map(|d| d as usize).collect();

    sys.apply_dirichlet(&other, &vec![0.0; other.len()], &vec![0.0; other.len()], &mut rhs);
    sys.apply_dirichlet(
        &left,
        &vec![args.left_drive_amp; left.len()],
        &vec![0.0; left.len()],
        &mut rhs,
    );

    let a = sys.to_flat_csr();
    let mut x = vec![0.0; 2 * n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 3000, verbose: false, ..Default::default() };
    let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES failed");

    let gf = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::INFINITY, f64::min);

    let mut is_boundary = vec![false; n];
    for &d in &other {
        is_boundary[d] = true;
    }
    for &d in &left {
        is_boundary[d] = true;
    }

    let mean_left_amp = mean_on_indices(&amp, &left);
    let mean_right_amp = mean_on_indices(&amp, &right);
    let interior_idx: Vec<usize> = (0..n).filter(|&i| !is_boundary[i]).collect();
    let mean_interior_amp = mean_on_indices(&amp, &interior_idx);

    (
        SolveResult {
            n_dofs: n,
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            mean_left_amp,
            mean_right_amp,
            mean_interior_amp,
            min_amp,
            max_amp,
        },
        gf,
    )
}

fn mean_on_indices(values: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    idx.iter().map(|&i| values[i]).sum::<f64>() / idx.len() as f64
}

struct Args {
    n: usize,
    omega: f64,
    thickness: f64,
    sigma_max: f64,
    power: f64,
    wx: f64,
    wy: f64,
    stretch_blend: f64,
    left_drive_amp: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 12,
        omega: 2.0,
        thickness: 0.2,
        sigma_max: 1.0,
        power: 2.0,
        wx: 1.0,
        wy: 1.0,
        stretch_blend: 0.0,
        left_drive_amp: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("12".into()).parse().unwrap_or(12),
            "--omega" => a.omega = it.next().unwrap_or("2.0".into()).parse().unwrap_or(2.0),
            "--pml-thickness" => a.thickness = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--sigma-max" => a.sigma_max = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--pml-power" => a.power = it.next().unwrap_or("2.0".into()).parse().unwrap_or(2.0),
            "--wx" => a.wx = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--wy" => a.wy = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--left-drive-amp" => {
                a.left_drive_amp = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--stretch-blend" => {
                a.stretch_blend = it.next().unwrap_or("0.0".into()).parse().unwrap_or(0.0)
            }
            _ => {}
        }
    }
    a.power = a.power.max(1.0);
    a.stretch_blend = a.stretch_blend.clamp(0.0, 1.0);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            n: 12,
            omega: 2.0,
            thickness: 0.2,
            sigma_max: 1.0,
            power: 2.0,
            wx: 1.0,
            wy: 1.0,
            stretch_blend: 0.0,
            left_drive_amp: 1.0,
        }
    }

    #[test]
    fn ex25_pml_converges_and_has_finite_metrics() {
        let r = solve_case(&base_args());
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-6, "residual = {}", r.final_residual);
        assert!(r.mean_left_amp.is_finite());
        assert!(r.mean_right_amp.is_finite());
        assert!(r.mean_interior_amp.is_finite());
        assert!(r.max_amp >= r.min_amp);
    }

    #[test]
    fn ex25_stronger_pml_reduces_right_boundary_reflection() {
        let mut weak = base_args();
        weak.sigma_max = 0.05;
        let weak_r = solve_case(&weak);

        let mut strong = base_args();
        strong.sigma_max = 4.0;
        strong.power = 3.0;
        let strong_r = solve_case(&strong);

        assert!(weak_r.converged && strong_r.converged);
        assert!(
            strong_r.reflection_ratio() < weak_r.reflection_ratio(),
            "expected stronger PML to lower reflection ratio: weak={} strong={}",
            weak_r.reflection_ratio(),
            strong_r.reflection_ratio()
        );
    }

    #[test]
    fn ex25_stretch_mode_changes_boundary_response() {
        let base = solve_case(&base_args());

        let mut stretch_args = base_args();
        stretch_args.stretch_blend = 0.8;
        let stretch = solve_case(&stretch_args);

        assert!(base.converged && stretch.converged);
        let delta = (base.mean_right_amp - stretch.mean_right_amp).abs();
        assert!(
            delta > 1.0e-4,
            "stretch mode should measurably change right-boundary amplitude, delta={}",
            delta
        );
    }

    #[test]
    fn ex25_thicker_pml_reduces_reflection_and_interior_amplitude() {
        let mut thin = base_args();
        thin.thickness = 0.05;
        thin.sigma_max = 4.0;
        thin.power = 3.0;
        let thin_r = solve_case(&thin);

        let mut thick = base_args();
        thick.thickness = 0.35;
        thick.sigma_max = 4.0;
        thick.power = 3.0;
        let thick_r = solve_case(&thick);

        assert!(thin_r.converged && thick_r.converged);
        assert!(
            thick_r.reflection_ratio() < thin_r.reflection_ratio(),
            "expected thicker PML to lower reflection ratio: thin={} thick={}",
            thin_r.reflection_ratio(),
            thick_r.reflection_ratio()
        );
        assert!(
            thick_r.mean_interior_amp < thin_r.mean_interior_amp,
            "expected thicker PML to lower interior amplitude: thin={} thick={}",
            thin_r.mean_interior_amp,
            thick_r.mean_interior_amp
        );
    }

    #[test]
    fn ex25_solution_scales_linearly_with_left_drive() {
        let mut half = base_args();
        half.left_drive_amp = 0.5;
        let rh = solve_case(&half);

        let full = solve_case(&base_args());

        assert!(rh.converged && full.converged);

        let right_ratio = full.mean_right_amp / rh.mean_right_amp.max(1.0e-30);
        let interior_ratio = full.mean_interior_amp / rh.mean_interior_amp.max(1.0e-30);
        let max_ratio = full.max_amp / rh.max_amp.max(1.0e-30);

        assert!((right_ratio - 2.0).abs() < 1.0e-6, "expected right amplitude to scale linearly, got ratio {}", right_ratio);
        assert!((interior_ratio - 2.0).abs() < 1.0e-6, "expected interior amplitude to scale linearly, got ratio {}", interior_ratio);
        assert!((max_ratio - 2.0).abs() < 1.0e-6, "expected max amplitude to scale linearly, got ratio {}", max_ratio);
    }

    #[test]
    fn ex25_sign_reversed_left_drive_flips_complex_field() {
        let (pos_result, pos_field) = solve_case_with_field(&base_args());

        let mut neg = base_args();
        neg.left_drive_amp = -1.0;
        let (neg_result, neg_field) = solve_case_with_field(&neg);

        assert!(pos_result.converged && neg_result.converged);
        assert_eq!(pos_field.u_re.len(), neg_field.u_re.len());
        assert_eq!(pos_field.u_im.len(), neg_field.u_im.len());

        let re_sym_err = pos_field
            .u_re
            .iter()
            .zip(&neg_field.u_re)
            .map(|(a, b)| (a + b).abs())
            .fold(0.0_f64, f64::max);
        let im_sym_err = pos_field
            .u_im
            .iter()
            .zip(&neg_field.u_im)
            .map(|(a, b)| (a + b).abs())
            .fold(0.0_f64, f64::max);

        assert!(re_sym_err < 1.0e-10, "expected real field to flip sign, got max symmetry error {}", re_sym_err);
        assert!(im_sym_err < 1.0e-10, "expected imaginary field to flip sign, got max symmetry error {}", im_sym_err);
        assert!(
            (pos_result.max_amp - neg_result.max_amp).abs() < 1.0e-10,
            "expected amplitude envelope to be invariant under sign reversal: pos={} neg={}",
            pos_result.max_amp,
            neg_result.max_amp
        );
    }

    #[test]
    fn ex25_dof_count_matches_p1_h1_formula() {
        for &n in &[8usize, 12usize, 16usize] {
            let mut a = base_args();
            a.n = n;
            let r = solve_case(&a);
            assert_eq!(r.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex25_zero_left_drive_gives_near_zero_field() {
        let mut a = base_args();
        a.left_drive_amp = 0.0;
        let r = solve_case(&a);
        assert!(r.converged);
        assert!(r.max_amp < 1.0e-12, "expected near-zero field amplitude, got {}", r.max_amp);
        assert!(r.mean_left_amp < 1.0e-12, "left amplitude should be near zero, got {}", r.mean_left_amp);
        assert!(r.mean_right_amp < 1.0e-12, "right amplitude should be near zero, got {}", r.mean_right_amp);
        assert!(r.mean_interior_amp < 1.0e-12, "interior amplitude should be near zero, got {}", r.mean_interior_amp);
    }
}

