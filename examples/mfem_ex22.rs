//! # Example 22 �?Complex-Valued Time-Harmonic Helmholtz (analogous to MFEM ex22)
//!
//! Solves a damped time-harmonic scalar Helmholtz equation using the
//! **2×2 real-block** strategy in `fem_assembly::complex`:
//!
//! ```text
//!   −∇·(a∇u) �?ω²b·u + iω·c·u = 0    in Ω = [0,1]²
//! ```
//!
//! with Dirichlet BCs:
//! - Left edge  (tag 4): u = 1+0i  (unit amplitude port)
//! - All other edges:    u = 0+0i
//!
//! The 2×2 real block system is:
//! ```text
//! [ K �?ω²M   −ωC ] [ u_re ]   [ 0 ]
//! [ ωC        K−ω²M] [ u_im ] = [ 0 ]
//! ```
//! Solved with GMRES.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex22
//! cargo run --example mfem_ex22 -- --n 16 --omega 2.0
//! cargo run --example mfem_ex22 -- --n 32 --sigma 0.2
//! ```

use fem_assembly::{
    Assembler, ComplexAssembler, ComplexGridFunction, face_dofs_p1,
    standard::{BoundaryMassIntegrator, DiffusionIntegrator, MassIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{SolverConfig, solve_gmres};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    omega: f64,
    sigma: f64,
    abc_alpha_right: f64,
    left_drive_amp: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        omega: 1.5,
        sigma: 0.1,
        abc_alpha_right: 1.0,
        left_drive_amp: 1.0,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--n"     => { i += 1; a.n     = raw[i].parse().unwrap(); }
            "--omega" => { i += 1; a.omega = raw[i].parse().unwrap(); }
            "--sigma" => { i += 1; a.sigma = raw[i].parse().unwrap(); }
            "--abc-alpha-right" => { i += 1; a.abc_alpha_right = raw[i].parse().unwrap(); }
            "--left-drive-amp" => { i += 1; a.left_drive_amp = raw[i].parse().unwrap(); }
            other     => eprintln!("unknown arg: {other}"),
        }
        i += 1;
    }
    a.abc_alpha_right = a.abc_alpha_right.max(0.0);
    a
}

#[derive(Debug, Clone)]
struct SolveResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    min_amp: f64,
    max_amp: f64,
    max_left_bc_err: f64,
    mean_left_amp: f64,
    mean_right_amp: f64,
    mean_interior_amp: f64,
}

impl SolveResult {
    fn transmission_ratio(&self) -> f64 {
        self.mean_right_amp / self.mean_left_amp.max(1.0e-14)
    }
}

fn mean_on_indices(values: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    idx.iter().map(|&i| values[i]).sum::<f64>() / idx.len() as f64
}

fn add_scaled_csr(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>, scale_b: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[p] as usize, a.values[p]);
        }
    }
    for i in 0..b.nrows {
        for p in b.row_ptr[i]..b.row_ptr[i + 1] {
            coo.add(i, b.col_idx[p] as usize, scale_b * b.values[p]);
        }
    }
    coo.into_csr()
}

fn solve_case(args: &Args) -> SolveResult {
    solve_case_with_field(args).0
}

fn solve_case_with_field(args: &Args) -> (SolveResult, ComplexGridFunction) {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&MassIntegrator { rho: 1.0 }],
        &[&MassIntegrator { rho: args.sigma }],
        args.omega,
        3,
    );

    // Add right-boundary first-order absorbing contribution i*omega*alpha*u.
    if args.abc_alpha_right > 0.0 {
        let face_dofs = face_dofs_p1(space.mesh());
        let bnd = Assembler::assemble_boundary_bilinear(
            ndofs,
            space.mesh(),
            &face_dofs,
            1,
            &[&BoundaryMassIntegrator {
                alpha: args.abc_alpha_right,
            }],
            &[2],
            3,
        );
        sys.k_im = add_scaled_csr(&sys.k_im, &bnd, args.omega);
    }

    let f_re = vec![0.0_f64; ndofs];
    let f_im = vec![0.0_f64; ndofs];
    let mut rhs = sys.assemble_rhs(&f_re, &f_im);

    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let left_dofs: Vec<usize> = boundary_dofs(mesh_ref, dm, &[4])
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let right_dofs: Vec<usize> = boundary_dofs(mesh_ref, dm, &[2])
        .into_iter()
        .map(|d| d as usize)
        .collect();
    // Keep right boundary open (ABC handles it weakly); clamp only top and bottom.
    let other_dofs: Vec<usize> = boundary_dofs(mesh_ref, dm, &[1, 3])
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let left_re: Vec<f64> = vec![args.left_drive_amp; left_dofs.len()];
    let left_im: Vec<f64> = vec![0.0; left_dofs.len()];
    let other_re: Vec<f64> = vec![0.0; other_dofs.len()];
    let other_im: Vec<f64> = vec![0.0; other_dofs.len()];

    sys.apply_dirichlet(&other_dofs, &other_re, &other_im, &mut rhs);
    sys.apply_dirichlet(&left_dofs, &left_re, &left_im, &mut rhs);

    let flat = sys.to_flat_csr();
    let mut x = vec![0.0_f64; 2 * ndofs];
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-14,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_gmres(&flat, &rhs, &mut x, 50, &cfg).expect("GMRES did not converge");

    let gf = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::INFINITY, f64::min);

    let max_left_bc_err = left_dofs
        .iter()
        .map(|&i| (gf.u_re[i] - args.left_drive_amp).abs())
        .fold(0.0_f64, f64::max);

    let mut is_boundary = vec![false; ndofs];
    for &i in &left_dofs {
        is_boundary[i] = true;
    }
    for &i in &right_dofs {
        is_boundary[i] = true;
    }
    for &i in &other_dofs {
        is_boundary[i] = true;
    }
    let interior: Vec<usize> = (0..ndofs).filter(|&i| !is_boundary[i]).collect();

    (
        SolveResult {
            n_dofs: ndofs,
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            min_amp,
            max_amp,
            max_left_bc_err,
            mean_left_amp: mean_on_indices(&amp, &left_dofs),
            mean_right_amp: mean_on_indices(&amp, &right_dofs),
            mean_interior_amp: mean_on_indices(&amp, &interior),
        },
        gf,
    )
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 22: Complex Helmholtz (2×2 real-block) ===");
    println!(
        "  Mesh: {}×{},  ω = {:.4},  σ = {:.4},  α_right = {:.4}",
        args.n, args.n, args.omega, args.sigma, args.abc_alpha_right
    );
    println!("  Left drive amplitude: {:.4}", args.left_drive_amp);
    println!("  DOFs: {}  (2×{} flat system)", result.n_dofs, result.n_dofs);
    println!(
        "  GMRES: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  |u| �?[{:.4}, {:.4}]", result.min_amp, result.max_amp);
    println!(
        "  |u| mean: left={:.4e}, right={:.4e}, interior={:.4e}",
        result.mean_left_amp, result.mean_right_amp, result.mean_interior_amp
    );
    println!(
        "  Transmission proxy (right/left) = {:.4e}",
        result.transmission_ratio()
    );
    println!("  Max left-BC error: {:.2e}", result.max_left_bc_err);

    assert!(result.max_left_bc_err < 1e-10, "left Dirichlet BC not satisfied");
    assert!(result.converged, "GMRES did not converge");
    println!("  �?Example 22 passed");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            n: 10,
            omega: 1.5,
            sigma: 0.1,
            abc_alpha_right: 1.0,
            left_drive_amp: 1.0,
        }
    }

    #[test]
    fn ex22_converges_with_absorbing_right_boundary() {
        let r = solve_case(&base_args());
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-6, "residual = {}", r.final_residual);
        assert!(r.max_left_bc_err < 1.0e-10, "bc err = {}", r.max_left_bc_err);
    }

    #[test]
    fn ex22_stronger_right_absorption_reduces_transmission_proxy() {
        let mut weak = base_args();
        weak.abc_alpha_right = 0.05;
        let rw = solve_case(&weak);

        let mut strong = base_args();
        strong.abc_alpha_right = 4.0;
        let rs = solve_case(&strong);

        assert!(rw.converged && rs.converged);
        assert!(
            rs.transmission_ratio() < rw.transmission_ratio(),
            "expected stronger right absorption to lower transmission proxy: weak={} strong={}",
            rw.transmission_ratio(),
            rs.transmission_ratio()
        );
    }

    #[test]
    fn ex22_turning_off_right_absorption_increases_right_boundary_amplitude() {
        let mut open = base_args();
        open.abc_alpha_right = 0.0;
        let ro = solve_case(&open);

        let absorbed = solve_case(&base_args());

        assert!(ro.converged && absorbed.converged);
        assert!(
            ro.mean_right_amp > absorbed.mean_right_amp,
            "expected no-ABC case to have larger right-boundary amplitude: open={} absorbed={}",
            ro.mean_right_amp,
            absorbed.mean_right_amp
        );
        assert!(
            ro.transmission_ratio() > absorbed.transmission_ratio(),
            "expected no-ABC case to have larger transmission proxy: open={} absorbed={}",
            ro.transmission_ratio(),
            absorbed.transmission_ratio()
        );
    }

    #[test]
    fn ex22_stronger_volumetric_damping_reduces_interior_and_transmitted_amplitude() {
        let mut weak = base_args();
        weak.sigma = 0.0;
        let rw = solve_case(&weak);

        let mut strong = base_args();
        strong.sigma = 5.0;
        let rs = solve_case(&strong);

        assert!(rw.converged && rs.converged);
        assert!(
            rs.mean_interior_amp < rw.mean_interior_amp,
            "expected stronger sigma to reduce interior amplitude: weak={} strong={}",
            rw.mean_interior_amp,
            rs.mean_interior_amp
        );
        assert!(
            rs.transmission_ratio() < rw.transmission_ratio(),
            "expected stronger sigma to reduce transmission proxy: weak={} strong={}",
            rw.transmission_ratio(),
            rs.transmission_ratio()
        );
    }

    #[test]
    fn ex22_solution_scales_linearly_with_left_port_drive() {
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
    fn ex22_sign_reversed_left_drive_flips_complex_field() {
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
    fn ex22_dof_count_matches_p1_h1_formula() {
        for &n in &[6usize, 10usize, 14usize] {
            let mut a = base_args();
            a.n = n;
            let r = solve_case(&a);
            assert_eq!(r.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex22_zero_left_drive_gives_near_zero_field() {
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

