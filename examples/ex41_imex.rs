//! ex41_imex - FEM advection-diffusion with IMEX time integration.
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

    println!("=== ex41_imex: FEM advection-diffusion with IMEX ===");
    println!("  mesh n={}, dt={}, T={}", args.n, args.dt, args.t_end);
    println!("  kappa={}, velocity=({}, {})", args.kappa, args.vx, args.vy);

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();
    println!("  dofs={n}");

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
    let mut dummy = vec![0.0f64; n];
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

    let u0 = initial_condition(dm, n, &bnd);

    let u_ref = rk4_reference(&split, &u0, args.t_end, args.dt.min(1.0e-4));

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

    let err_e = l2_error(&u_euler, &u_ref);
    let err_s = l2_error(&u_ssp2, &u_ref);
    let err_r = l2_error(&u_rk3, &u_ref);
    let err_a = l2_error(&u_ark3, &u_ref);

    println!("  final times: Euler={:.6}, SSP2={:.6}, RK3={:.6}, ARK3={:.6} (ark dt_last={:.3e})", t_e, t_s, t_r, t_a, dt_last);
    println!("  ||u_euler - u_ref||_2 = {:.3e}", err_e);
    println!("  ||u_ssp2  - u_ref||_2 = {:.3e}", err_s);
    println!("  ||u_rk3   - u_ref||_2 = {:.3e}", err_r);
    println!("  ||u_ark3  - u_ref||_2 = {:.3e}", err_a);

    assert!(err_e.is_finite() && err_s.is_finite() && err_a.is_finite(), "non-finite error detected");
    assert!(err_s <= err_e * 1.05, "SSP2 should not be worse than Euler by much: euler={err_e:.3e}, ssp2={err_s:.3e}");
    assert!(err_r <= err_s * 1.05, "RK3 should be at least comparable to SSP2: rk3={err_r:.3e}, ssp2={err_s:.3e}");

    println!("  PASS");
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
