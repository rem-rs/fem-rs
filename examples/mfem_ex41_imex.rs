//! mfem_ex41_imex - FEM advection-diffusion with IMEX time integration.
//!
//! Semi-discrete model on H1 space:
//!   M du/dt + C u + K u = 0
//! where
//!   C: convection matrix from (b · grad u, v)
//!   K: diffusion stiffness from kappa * (grad u, grad v)
//!
//! IMEX split:
//!   explicit:  f_E(u) = -M^{-1} C u
//!   implicit:  f_I(u) = -M^{-1} K u
//!
//! This example compares:
//!   - IMEX Euler
//!   - IMEX SSP2
//!   - IMEX RK3 (fixed-step)
//!   - IMEX ARK3 (adaptive)
//! against a fine-step RK4 reference trajectory.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::ConstantVectorCoeff,
    standard::{ConvectionIntegrator, DiffusionIntegrator, MassIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{
    solve_cg,
    ImexArk3, ImexOperator, ImexTimeStepper,
    Rk4, TimeStepper,
    SolverConfig,
};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

#[derive(Clone)]
struct AdvectionDiffusionSplit {
    minv_c: CsrMatrix<f64>,
    minv_k: CsrMatrix<f64>,
    bc_dofs: Vec<u32>,
}

struct MethodResult {
    final_time: f64,
    error: f64,
    solution_norm: f64,
    checksum: f64,
    dt_last: Option<f64>,
}

struct SolveResult {
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    vx: f64,
    vy: f64,
    n_dofs: usize,
    reference_norm: f64,
    euler: MethodResult,
    ssp2: MethodResult,
    rk3: MethodResult,
    ark3: MethodResult,
}

impl ImexOperator for AdvectionDiffusionSplit {
    fn explicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        self.minv_c.spmv(u, out);
        for v in out.iter_mut() {
            *v = -*v;
        }
        for &d in &self.bc_dofs {
            out[d as usize] = 0.0;
        }
    }

    fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        self.minv_k.spmv(u, out);
        for v in out.iter_mut() {
            *v = -*v;
        }
        for &d in &self.bc_dofs {
            out[d as usize] = 0.0;
        }
    }

    fn jac_implicit(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        scale_csr(&self.minv_k, -1.0)
    }
}

fn main() {
    let args = parse_args();

    println!("=== mfem_ex41_imex: FEM advection-diffusion with IMEX ===");
    println!("  mesh n={}, dt={}, T={}", args.n, args.dt, args.t_end);
    println!("  kappa={}, velocity=({}, {})", args.kappa, args.vx, args.vy);

    let result = solve_case(&args);

    println!("  confirmed dofs={}", result.n_dofs);
    println!(
        "  params: n={}, dt={}, T={}, kappa={}, velocity=({}, {})",
        result.n,
        result.dt,
        result.t_end,
        result.kappa,
        result.vx,
        result.vy,
    );
    println!("  ||u_ref||_2 = {:.3e}", result.reference_norm);
    print_method("Euler", &result.euler);
    print_method("SSP2", &result.ssp2);
    print_method("RK3", &result.rk3);
    print_method("ARK3", &result.ark3);

    assert!(
        result.euler.error.is_finite()
            && result.ssp2.error.is_finite()
            && result.rk3.error.is_finite()
            && result.ark3.error.is_finite(),
        "non-finite error detected"
    );
    assert!(
        result.rk3.error <= result.euler.error,
        "RK3 should be more accurate than Euler: rk3={:.3e}, euler={:.3e}",
        result.rk3.error,
        result.euler.error,
    );
    assert!(
        result.ark3.error <= result.rk3.error * 1.2,
        "ARK3 should be comparable to or better than RK3: ark3={:.3e}, rk3={:.3e}",
        result.ark3.error,
        result.rk3.error,
    );

    println!("  PASS");
}

fn solve_case(args: &Args) -> SolveResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
    let diff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: args.kappa }], 3);
    let conv = Assembler::assemble_bilinear(
        &space,
        &[&ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![args.vx, args.vy]) }],
        3,
    );

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);

    let mut m_bc = mass.clone();
    let mut k_bc = diff.clone();
    let mut c_bc = conv.clone();
    let mut dummy = vec![0.0f64; n_dofs];
    let vals = vec![0.0f64; bnd.len()];
    apply_dirichlet(&mut m_bc, &mut dummy, &bnd, &vals);
    apply_dirichlet(&mut k_bc, &mut dummy, &bnd, &vals);
    apply_dirichlet(&mut c_bc, &mut dummy, &bnd, &vals);

    let solve_cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 1200, verbose: false, ..SolverConfig::default() };

    let minv_k = mass_inverse_times(&m_bc, &k_bc, &solve_cfg);
    let minv_c = mass_inverse_times(&m_bc, &c_bc, &solve_cfg);

    let split = AdvectionDiffusionSplit {
        minv_c,
        minv_k,
        bc_dofs: bnd.clone(),
    };

    let u0 = initial_condition(dm, n_dofs, &bnd);

    let u_ref = rk4_reference(&split, &u0, args.t_end, args.dt.min(1.0e-4));
    let reference_norm = vector_norm(&u_ref);

    let imex_driver = ImexTimeStepper;

    let mut u_euler = u0.clone();
    let t_e = imex_driver.integrate_euler(&split, 0.0, args.t_end, &mut u_euler, args.dt);

    let mut u_ssp2 = u0.clone();
    let t_s = imex_driver.integrate_ssp2(&split, 0.0, args.t_end, &mut u_ssp2, args.dt);

    let mut u_rk3 = u0.clone();
    let t_r = imex_driver.integrate_rk3(&split, 0.0, args.t_end, &mut u_rk3, args.dt);

    let mut u_ark3 = u0.clone();
    let ark3 = ImexArk3 { rtol: 1e-6, atol: 1e-9, dt_min: 1e-10, dt_max: args.dt, ..Default::default() };
    let (t_a, dt_last) = imex_driver.integrate_ark3(&split, 0.0, args.t_end, &mut u_ark3, args.dt, &ark3);

    SolveResult {
        n: args.n,
        dt: args.dt,
        t_end: args.t_end,
        kappa: args.kappa,
        vx: args.vx,
        vy: args.vy,
        n_dofs,
        reference_norm,
        euler: build_method_result(&u_euler, &u_ref, t_e, None),
        ssp2: build_method_result(&u_ssp2, &u_ref, t_s, None),
        rk3: build_method_result(&u_rk3, &u_ref, t_r, None),
        ark3: build_method_result(&u_ark3, &u_ref, t_a, Some(dt_last)),
    }
}

fn rk4_reference(op: &AdvectionDiffusionSplit, u0: &[f64], t_end: f64, dt_ref: f64) -> Vec<f64> {
    let rhs = |t: f64, u: &[f64], out: &mut [f64]| {
        let mut fe = vec![0.0f64; u.len()];
        let mut fi = vec![0.0f64; u.len()];
        op.explicit(t, u, &mut fe);
        op.implicit(t, u, &mut fi);
        for i in 0..u.len() {
            out[i] = fe[i] + fi[i];
        }
    };

    let rk4 = Rk4;
    let mut u = u0.to_vec();
    let mut t = 0.0;
    while t < t_end - 1e-14 {
        let h = dt_ref.min(t_end - t);
        rk4.step(t, h, &mut u, &rhs);
        t += h;
    }
    u
}

fn initial_condition(dm: &fem_space::DofManager, n: usize, bnd: &[u32]) -> Vec<f64> {
    let mut u: Vec<f64> = (0..n)
        .map(|i| {
            let x = dm.dof_coord(i as u32);
            (PI * x[0]).sin() * (PI * x[1]).sin()
        })
        .collect();
    for &d in bnd {
        u[d as usize] = 0.0;
    }
    u
}

fn l2_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(1) as f64;
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    (s / n).sqrt()
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn checksum(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>()
}

fn build_method_result(solution: &[f64], reference: &[f64], final_time: f64, dt_last: Option<f64>) -> MethodResult {
    MethodResult {
        final_time,
        error: l2_error(solution, reference),
        solution_norm: vector_norm(solution),
        checksum: checksum(solution),
        dt_last,
    }
}

fn print_method(name: &str, result: &MethodResult) {
    println!(
        "  {name}: t_final={:.6}, err={:.3e}, ||u||_2={:.3e}, checksum={:.8e}",
        result.final_time,
        result.error,
        result.solution_norm,
        result.checksum,
    );
    if let Some(dt_last) = result.dt_last {
        println!("    {name} last dt = {:.3e}", dt_last);
    }
}

fn mass_inverse_times(m: &CsrMatrix<f64>, a: &CsrMatrix<f64>, cfg: &SolverConfig) -> CsrMatrix<f64> {
    let n = m.nrows;

    // Build column-wise RHS vectors from CSR A.
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            cols[j][i] = a.values[p];
        }
    }

    let mut coo = CooMatrix::<f64>::new(n, n);

    for j in 0..n {
        let rhs = &cols[j];
        let mut x = vec![0.0f64; n];
        solve_cg(m, rhs, &mut x, cfg).expect("mass_inverse_times: CG solve failed");
        for (i, &v) in x.iter().enumerate() {
            if v.abs() > 1e-14 {
                coo.add(i, j, v);
            }
        }
    }

    coo.into_csr()
}

fn scale_csr(mat: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: mat.nrows,
        ncols: mat.ncols,
        row_ptr: mat.row_ptr.clone(),
        col_idx: mat.col_idx.clone(),
        values: mat.values.iter().map(|&v| alpha * v).collect(),
    }
}

struct Args {
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    vx: f64,
    vy: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        dt: 0.01,
        t_end: 0.2,
        kappa: 0.01,
        vx: 1.0,
        vy: 0.3,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--dt" => a.dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--T" => a.t_end = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--kappa" => a.kappa = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--vx" => a.vx = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--vy" => a.vy = it.next().unwrap_or("0.3".into()).parse().unwrap_or(0.3),
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex41_imex_default_case_preserves_expected_method_ordering() {
        let args = Args { n: 8, dt: 0.01, t_end: 0.2, kappa: 0.01, vx: 1.0, vy: 0.3 };
        let result = solve_case(&args);
        assert_eq!(result.n_dofs, 81);
        assert!((result.euler.final_time - result.t_end).abs() < 1.0e-12);
        assert!((result.ark3.final_time - result.t_end).abs() < 1.0e-12);
        assert!(result.euler.error < 8.0e-3, "Euler error too large: {}", result.euler.error);
        assert!(result.ssp2.error < result.euler.error, "SSP2 should improve on Euler in the default case");
        assert!(result.rk3.error < 1.0e-4, "RK3 should track the reference closely: {}", result.rk3.error);
        assert!(result.ark3.error < result.rk3.error, "ARK3 should beat RK3 in the default case");
    }

    #[test]
    fn ex41_imex_smaller_dt_improves_euler_and_rk3_accuracy() {
        let coarse = solve_case(&Args { n: 8, dt: 0.02, t_end: 0.2, kappa: 0.01, vx: 1.0, vy: 0.3 });
        let fine = solve_case(&Args { n: 8, dt: 0.005, t_end: 0.2, kappa: 0.01, vx: 1.0, vy: 0.3 });
        assert!(fine.euler.error < coarse.euler.error * 0.5,
            "Euler refinement gain too small: coarse={} fine={}", coarse.euler.error, fine.euler.error);
        assert!(fine.rk3.error < coarse.rk3.error * 0.1,
            "RK3 refinement gain too small: coarse={} fine={}", coarse.rk3.error, fine.rk3.error);
        assert!(fine.ark3.error <= coarse.ark3.error * 1.1,
            "adaptive ARK3 should remain at least as accurate when the user dt shrinks: coarse={} fine={}",
            coarse.ark3.error,
            fine.ark3.error);
    }

    #[test]
    fn ex41_imex_pure_diffusion_limit_favors_high_order_methods() {
        let result = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.2, kappa: 0.01, vx: 0.0, vy: 0.0 });
        assert!(result.euler.error < 5.0e-5, "Euler error too large in pure diffusion: {}", result.euler.error);
        assert!(result.ssp2.error < result.euler.error * 1.0e-2,
            "SSP2 should sharply improve in pure diffusion: euler={} ssp2={}", result.euler.error, result.ssp2.error);
        assert!(result.rk3.error < 1.0e-8, "RK3 should be nearly exact in pure diffusion: {}", result.rk3.error);
        assert!(result.ark3.error < 1.0e-8, "ARK3 should be nearly exact in pure diffusion: {}", result.ark3.error);
    }

    #[test]
    fn ex41_imex_stronger_diffusion_keeps_high_order_schemes_accurate() {
        let result = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.2, kappa: 0.05, vx: 1.0, vy: 0.3 });
        assert!(result.euler.error.is_finite() && result.ssp2.error.is_finite());
        assert!(result.rk3.error < result.euler.error * 1.0e-2,
            "RK3 should remain far more accurate under stronger diffusion: euler={} rk3={}", result.euler.error, result.rk3.error);
        assert!(result.ark3.error < 1.0e-6,
            "ARK3 error too large under stronger diffusion: {}", result.ark3.error);
        assert!(result.ark3.dt_last.unwrap_or(result.dt) <= result.dt + 1.0e-12);
    }

    #[test]
    fn ex41_imex_dof_count_matches_p1_h1_formula() {
        for &n in &[6usize, 8usize, 10usize] {
            let result = solve_case(&Args { n, dt: 0.01, t_end: 0.2, kappa: 0.01, vx: 1.0, vy: 0.3 });
            assert_eq!(result.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex41_imex_higher_kappa_decays_faster_in_pure_diffusion() {
        let low_kappa = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.2, kappa: 0.01, vx: 0.0, vy: 0.0 });
        let high_kappa = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.2, kappa: 0.05, vx: 0.0, vy: 0.0 });
        assert!(low_kappa.rk3.solution_norm > 0.0 && high_kappa.rk3.solution_norm > 0.0);
        assert!(high_kappa.rk3.solution_norm < low_kappa.rk3.solution_norm,
            "higher kappa should increase decay: low={} high={}",
            low_kappa.rk3.solution_norm,
            high_kappa.rk3.solution_norm);
    }

    #[test]
    fn ex41_imex_zero_final_time_is_noop_for_all_methods() {
        let result = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.0, kappa: 0.01, vx: 1.0, vy: 0.3 });
        assert!((result.euler.final_time - 0.0).abs() < 1.0e-14);
        assert!((result.ssp2.final_time - 0.0).abs() < 1.0e-14);
        assert!((result.rk3.final_time - 0.0).abs() < 1.0e-14);
        assert!((result.ark3.final_time - 0.0).abs() < 1.0e-14);
        assert!(result.euler.error < 1.0e-14);
        assert!(result.ssp2.error < 1.0e-14);
        assert!(result.rk3.error < 1.0e-14);
        assert!(result.ark3.error < 1.0e-14);
    }

    #[test]
    fn ex41_imex_ark3_last_dt_is_positive_and_bounded() {
        let result = solve_case(&Args { n: 8, dt: 0.01, t_end: 0.215, kappa: 0.01, vx: 1.0, vy: 0.3 });
        let dt_last = result.ark3.dt_last.expect("ARK3 should report last dt");
        assert!(dt_last > 0.0, "ARK3 last dt must be positive");
        assert!(dt_last <= result.dt + 1.0e-12,
            "ARK3 last dt must not exceed user dt: last={} dt={}", dt_last, result.dt);
    }
}

