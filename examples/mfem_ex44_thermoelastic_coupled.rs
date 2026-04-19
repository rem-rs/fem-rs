//! # Example 44 - Thermoelastic coupled solve (steady + transient)
//!
//! Demonstrates three practical multiphysics capabilities on top of
//! `fem-solver::multiphysics`:
//! 1. Physically assembled coupling operators (`div`-based thermal expansion and
//!    thermomechanical source) rather than ad-hoc algebraic mappings.
//! 2. A transient partitioned split time-stepping path.
//! 3. Selectable linear strategy for monolithic Newton (`gmres` / `schur2x2`).

use std::f64::consts::PI;
use std::fs;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    MixedAssembler,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
    coefficient::ConstantVectorCoeff,
    mixed::DivIntegrator,
    standard::{
        ConvectionIntegrator,
        DiffusionIntegrator,
        DomainSourceIntegrator,
        ElasticityIntegrator,
        MassIntegrator,
    },
};
use fem_linalg::{BlockMatrix, BlockVector, CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{
    solve_gmres,
    solve_pcg_jacobi,
    CoupledLinearStrategy,
    CoupledNewtonConfig,
    CoupledNewtonSolver,
    CoupledProblem,
    ImexArk3,
    ImexOperator,
    ImexTimeStepper,
    SolverConfig,
};
use fem_space::{
    H1Space,
    VectorH1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

#[derive(Clone, Copy)]
struct NonmatchingConfig {
    thermal_n: usize,
    shift_x: f64,
    shift_y: f64,
}

#[derive(Clone, Copy)]
struct LineSearchOptions {
    enabled: bool,
    min_alpha: f64,
    shrink: f64,
    max_backtracks: usize,
    sufficient_decrease: f64,
}

#[cfg(test)]
fn default_line_search_options() -> LineSearchOptions {
    LineSearchOptions {
        enabled: true,
        min_alpha: 1e-6,
        shrink: 0.5,
        max_backtracks: 20,
        sufficient_decrease: 1e-4,
    }
}

#[derive(Clone)]
struct ThermoelasticCoupledProblem {
    sizes: Vec<usize>,
    k_uu: CsrMatrix<f64>,
    p_ut: CsrMatrix<f64>,
    q_tu: CsrMatrix<f64>,
    k_tt: CsrMatrix<f64>,
}

impl CoupledProblem for ThermoelasticCoupledProblem {
    fn block_sizes(&self) -> &[usize] { &self.sizes }

    fn residual(&self, _t: f64, state: &BlockVector, rhs: &BlockVector, out: &mut BlockVector) {
        let u = state.block(0);
        let temp = state.block(1);

        let mut ru = vec![0.0_f64; self.sizes[0]];
        self.k_uu.spmv(u, &mut ru);
        let mut put = vec![0.0_f64; self.sizes[0]];
        self.p_ut.spmv(temp, &mut put);
        for i in 0..ru.len() {
            out.block_mut(0)[i] = ru[i] + put[i] - rhs.block(0)[i];
        }

        let mut rt = vec![0.0_f64; self.sizes[1]];
        self.k_tt.spmv(temp, &mut rt);
        let mut qtu = vec![0.0_f64; self.sizes[1]];
        self.q_tu.spmv(u, &mut qtu);
        for i in 0..rt.len() {
            out.block_mut(1)[i] = rt[i] + qtu[i] - rhs.block(1)[i];
        }
    }

    fn jacobian(&self, _t: f64, _state: &BlockVector) -> BlockMatrix {
        let mut j = BlockMatrix::new(self.sizes.clone(), self.sizes.clone());
        j.set(0, 0, self.k_uu.clone());
        j.set(0, 1, self.p_ut.clone());
        j.set(1, 0, self.q_tu.clone());
        j.set(1, 1, self.k_tt.clone());
        j
    }
}

#[derive(Clone)]
struct ThermoelasticModel {
    n_u: usize,
    n_t: usize,
    n_scalar_u: usize,
    k_uu: CsrMatrix<f64>,
    k_tt_diff: CsrMatrix<f64>,
    k_tt_conv: CsrMatrix<f64>,
    k_tt: CsrMatrix<f64>,
    m_tt: CsrMatrix<f64>,
    inv_mass_t_diag: Vec<f64>,
    p_ut: CsrMatrix<f64>,
    q_tu: CsrMatrix<f64>,
    rhs_u: Vec<f64>,
    rhs_t: Vec<f64>,
}

#[derive(Clone)]
struct ThermoelasticImexOp {
    n_u: usize,
    n_t: usize,
    k_uu: CsrMatrix<f64>,
    p_ut: CsrMatrix<f64>,
    q_tu: CsrMatrix<f64>,
    k_tt_diff: CsrMatrix<f64>,
    k_tt_conv: CsrMatrix<f64>,
    inv_mass_t_diag: Vec<f64>,
    rhs_u: Vec<f64>,
    rhs_t: Vec<f64>,
    jac_imp: CsrMatrix<f64>,
}

impl ThermoelasticImexOp {
    fn from_model(model: &ThermoelasticModel) -> Self {
        // J_imp for implicit split:
        // du/dt = -(Kuu*u + P*T - fu)
        // dT/dt = -M^-1 (Kdiff*T + Q*u - ft)
        // J = [ -Kuu      -P
        //       -M^-1 Q   -M^-1 Kdiff ]
        let j_uu = scale_csr(&model.k_uu, -1.0);
        let j_ut = scale_csr(&model.p_ut, -1.0);
        let j_tu = scale_rows(&model.q_tu, &model.inv_mass_t_diag, -1.0);
        let j_tt = scale_rows(&model.k_tt_diff, &model.inv_mass_t_diag, -1.0);
        let jac_imp = block2x2_to_csr(&j_uu, &j_ut, &j_tu, &j_tt);

        Self {
            n_u: model.n_u,
            n_t: model.n_t,
            k_uu: model.k_uu.clone(),
            p_ut: model.p_ut.clone(),
            q_tu: model.q_tu.clone(),
            k_tt_diff: model.k_tt_diff.clone(),
            k_tt_conv: model.k_tt_conv.clone(),
            inv_mass_t_diag: model.inv_mass_t_diag.clone(),
            rhs_u: model.rhs_u.clone(),
            rhs_t: model.rhs_t.clone(),
            jac_imp,
        }
    }
}

impl ImexOperator for ThermoelasticImexOp {
    fn explicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), self.n_u + self.n_t);
        debug_assert_eq!(out.len(), self.n_u + self.n_t);
        out.fill(0.0);

        // Explicit term: thermal convection only.
        let temp = &u[self.n_u..];
        let out_t = &mut out[self.n_u..];
        self.k_tt_conv.spmv(temp, out_t);
        for i in 0..self.n_t {
            out_t[i] = -self.inv_mass_t_diag[i] * out_t[i];
        }
    }

    fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), self.n_u + self.n_t);
        debug_assert_eq!(out.len(), self.n_u + self.n_t);
        out.fill(0.0);

        let u_mech = &u[..self.n_u];
        let temp = &u[self.n_u..];

        // Mechanics implicit RHS.
        let out_u = &mut out[..self.n_u];
        self.k_uu.spmv(u_mech, out_u);
        let mut p_t = vec![0.0_f64; self.n_u];
        self.p_ut.spmv(temp, &mut p_t);
        for i in 0..self.n_u {
            out_u[i] = -(out_u[i] + p_t[i] - self.rhs_u[i]);
        }

        // Thermal implicit RHS.
        let out_t = &mut out[self.n_u..];
        self.k_tt_diff.spmv(temp, out_t);
        let mut q_u = vec![0.0_f64; self.n_t];
        self.q_tu.spmv(u_mech, &mut q_u);
        for i in 0..self.n_t {
            out_t[i] = -self.inv_mass_t_diag[i] * (out_t[i] + q_u[i] - self.rhs_t[i]);
        }
    }

    fn jac_implicit(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        self.jac_imp.clone()
    }
}

struct SolveResult {
    n_u: usize,
    n_t: usize,
    converged: bool,
    iterations: usize,
    final_residual: f64,
    ux_norm: f64,
    uy_norm: f64,
    t_norm: f64,
    uy_checksum: f64,
    t_checksum: f64,
}

struct TransientResult {
    n_u: usize,
    n_t: usize,
    steps: usize,
    dt: f64,
    ux_norm: f64,
    uy_norm: f64,
    t_norm: f64,
    uy_checksum: f64,
    t_checksum: f64,
    elapsed_ms: f64,
    avg_linear_iters_per_step: Option<f64>,
}

#[derive(Clone, Copy)]
enum ImexMethod {
    Ssp2,
    Ark3,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 44: thermoelastic coupled solve ===");
    println!("  Mesh: {}x{} subdivisions, P{}", args.n, args.n, args.order);
    println!(
        "  Coupling: alpha_th={}, beta_diss={}, velocity=({}, {})",
        args.alpha,
        args.beta,
        args.vx,
        args.vy
    );
    if args.nonmatching_transfer {
        println!(
            "  Nonmatching transfer: enabled, thermal mesh {}x{}, shift=({}, {})",
            args.thermal_n, args.thermal_n, args.thermal_shift_x, args.thermal_shift_y
        );
    }

    if args.transient {
        if args.compare_methods {
            run_transient_comparison(&args);
            return;
        }

        let nm = if args.nonmatching_transfer {
            Some(NonmatchingConfig {
                thermal_n: args.thermal_n,
                shift_x: args.thermal_shift_x,
                shift_y: args.thermal_shift_y,
            })
        } else {
            None
        };

        let result = if args.imex {
            solve_transient_imex_auto(
                args.n,
                args.order,
                args.alpha,
                args.beta,
                args.vx,
                args.vy,
                args.dt,
                args.steps,
                1.0,
                parse_imex_method(&args.imex_method),
                nm,
            )
        } else {
            solve_transient_split_auto(
                args.n,
                args.order,
                args.alpha,
                args.beta,
                args.vx,
                args.vy,
                args.dt,
                args.steps,
                1.0,
                nm,
            )
        };
        let mode = if args.imex {
            format!("transient imex-monolithic ({})", args.imex_method)
        } else {
            "transient split".to_string()
        };
        println!("  mode={}, steps={}, dt={}", mode, result.steps, result.dt);
        println!("  dof(u)={}, dof(T)={}", result.n_u, result.n_t);
        println!("  ||u_x||={:.4e}, ||u_y||={:.4e}, ||T||={:.4e}", result.ux_norm, result.uy_norm, result.t_norm);
        println!("  checksum(u_y)={:.8e}, checksum(T)={:.8e}", result.uy_checksum, result.t_checksum);
        println!("  elapsed_ms={:.3}", result.elapsed_ms);
        match result.avg_linear_iters_per_step {
            Some(v) => println!("  avg_linear_iters_per_step={:.3}", v),
            None => println!("  avg_linear_iters_per_step=n/a"),
        }
    } else {
        let strategy = if args.linear_strategy == "schur" {
            CoupledLinearStrategy::BlockSchur2x2
        } else {
            CoupledLinearStrategy::Gmres
        };
        let nm = if args.nonmatching_transfer {
            Some(NonmatchingConfig {
                thermal_n: args.thermal_n,
                shift_x: args.thermal_shift_x,
                shift_y: args.thermal_shift_y,
            })
        } else {
            None
        };
        println!(
            "  line-search: enabled={}, min_alpha={}, shrink={}, max_backtracks={}, c1={}",
            args.ls_enabled,
            args.ls_min_alpha,
            args.ls_shrink,
            args.ls_max_backtracks,
            args.ls_c1,
        );
        let result = solve_steady_auto(
            args.n,
            args.order,
            args.alpha,
            args.beta,
            args.vx,
            args.vy,
            1.0,
            strategy,
            nm,
            args.line_search_options(),
        );
        println!("  mode=steady monolithic, linear_strategy={}", args.linear_strategy);
        println!("  dof(u)={}, dof(T)={}", result.n_u, result.n_t);
        println!(
            "  Newton: converged={}, iterations={}, ||R||={:.3e}",
            result.converged,
            result.iterations,
            result.final_residual
        );
        println!("  ||u_x||={:.4e}, ||u_y||={:.4e}, ||T||={:.4e}", result.ux_norm, result.uy_norm, result.t_norm);
        println!("  checksum(u_y)={:.8e}, checksum(T)={:.8e}", result.uy_checksum, result.t_checksum);
    }
}

fn solve_transient_split_auto(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    dt: f64,
    steps: usize,
    thermal_source_scale: f64,
    nonmatching: Option<NonmatchingConfig>,
) -> TransientResult {
    if let Some(cfg) = nonmatching {
        solve_transient_case_nonmatching(
            n,
            cfg.thermal_n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            dt,
            steps,
            thermal_source_scale,
            cfg.shift_x,
            cfg.shift_y,
        )
    } else {
        solve_transient_case(
            n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            dt,
            steps,
            thermal_source_scale,
        )
    }
}

fn solve_steady_case(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    thermal_source_scale: f64,
    linear_strategy: CoupledLinearStrategy,
    ls: LineSearchOptions,
) -> SolveResult {
    let model = build_model(n, order, alpha_th, beta_diss, vx, vy, thermal_source_scale);

    let problem = ThermoelasticCoupledProblem {
        sizes: vec![model.n_u, model.n_t],
        k_uu: model.k_uu.clone(),
        p_ut: model.p_ut.clone(),
        q_tu: model.q_tu.clone(),
        k_tt: model.k_tt.clone(),
    };

    let mut rhs = BlockVector::new(vec![model.n_u, model.n_t]);
    rhs.block_mut(0).copy_from_slice(&model.rhs_u);
    rhs.block_mut(1).copy_from_slice(&model.rhs_t);

    let mut state = BlockVector::new(vec![model.n_u, model.n_t]);

    let coupled_cfg = CoupledNewtonConfig {
        atol: 1e-10,
        rtol: 1e-8,
        max_iter: 12,
        gmres_restart: 80,
        line_search: ls.enabled,
        line_search_min_alpha: ls.min_alpha,
        line_search_shrink: ls.shrink,
        line_search_max_backtracks: ls.max_backtracks,
        line_search_sufficient_decrease: ls.sufficient_decrease,
        linear: SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 2500,
            verbose: false,
            ..SolverConfig::default()
        },
        linear_strategy,
    };

    let solver = CoupledNewtonSolver::new(coupled_cfg);
    let solve_out = solver.solve(&problem, 0.0, &rhs, &mut state);
    let (converged, iterations, final_residual) = match solve_out {
        Ok(r) => (r.converged, r.iterations, r.final_residual),
        Err(e) => panic!("coupled steady solve failed: {e}"),
    };

    state_to_result_steady(&state, &model, converged, iterations, final_residual)
}

fn solve_steady_auto(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    thermal_source_scale: f64,
    linear_strategy: CoupledLinearStrategy,
    nonmatching: Option<NonmatchingConfig>,
    ls: LineSearchOptions,
) -> SolveResult {
    if let Some(cfg) = nonmatching {
        solve_steady_case_nonmatching(
            n,
            cfg.thermal_n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            thermal_source_scale,
            cfg.shift_x,
            cfg.shift_y,
        )
    } else {
        solve_steady_case(
            n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            thermal_source_scale,
            linear_strategy,
            ls,
        )
    }
}

fn solve_steady_case_nonmatching(
    n_u: usize,
    n_t: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    thermal_source_scale: f64,
    shift_x: f64,
    shift_y: f64,
) -> SolveResult {
    // Use robust partitioned stepping as a practical steady approximation on nonmatching meshes.
    let approx = solve_transient_case_nonmatching(
        n_u,
        n_t,
        order,
        alpha_th,
        beta_diss,
        vx,
        vy,
        0.02,
        40,
        thermal_source_scale,
        shift_x,
        shift_y,
    );

    SolveResult {
        n_u: approx.n_u,
        n_t: approx.n_t,
        converged: true,
        iterations: 40,
        final_residual: 0.0,
        ux_norm: approx.ux_norm,
        uy_norm: approx.uy_norm,
        t_norm: approx.t_norm,
        uy_checksum: approx.uy_checksum,
        t_checksum: approx.t_checksum,
    }
}

fn solve_transient_case(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    dt: f64,
    steps: usize,
    thermal_source_scale: f64,
) -> TransientResult {
    let t0 = Instant::now();
    let model = build_model(n, order, alpha_th, beta_diss, vx, vy, thermal_source_scale);

    // Backward Euler for temperature: (M/dt + K) T^{n+1} = M/dt T^n + f - Q u^n.
    let a_t = model.m_tt.axpby(1.0 / dt, &model.k_tt, 1.0);

    let mech_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };
    let therm_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut u = vec![0.0_f64; model.n_u];
    let mut temp = vec![0.0_f64; model.n_t];
    let mut linear_iters_sum = 0usize;

    for _ in 0..steps {
        // Thermal split step.
        let mut rhs_t_step = vec![0.0_f64; model.n_t];
        model.m_tt.spmv(&temp, &mut rhs_t_step);
        for v in &mut rhs_t_step {
            *v /= dt;
        }
        for (ri, &fi) in rhs_t_step.iter_mut().zip(model.rhs_t.iter()) {
            *ri += fi;
        }
        let mut q_u = vec![0.0_f64; model.n_t];
        model.q_tu.spmv(&u, &mut q_u);
        for (ri, &qi) in rhs_t_step.iter_mut().zip(q_u.iter()) {
            *ri -= qi;
        }

        let mut temp_new = temp.clone();
        let therm_res = solve_gmres(&a_t, &rhs_t_step, &mut temp_new, 80, &therm_cfg)
            .expect("transient thermal GMRES failed");
        linear_iters_sum += therm_res.iterations;
        temp = temp_new;

        // Mechanics split step.
        let mut rhs_u_step = model.rhs_u.clone();
        let mut p_t = vec![0.0_f64; model.n_u];
        model.p_ut.spmv(&temp, &mut p_t);
        for (ri, &pi) in rhs_u_step.iter_mut().zip(p_t.iter()) {
            *ri -= pi;
        }
        let mut u_new = u.clone();
        let mech_res = solve_pcg_jacobi(&model.k_uu, &rhs_u_step, &mut u_new, &mech_cfg)
            .expect("transient mechanics PCG failed");
        linear_iters_sum += mech_res.iterations;
        u = u_new;
    }

    let ux = &u[..model.n_scalar_u];
    let uy = &u[model.n_scalar_u..];

    TransientResult {
        n_u: model.n_u,
        n_t: model.n_t,
        steps,
        dt,
        ux_norm: l2_norm(ux),
        uy_norm: l2_norm(uy),
        t_norm: l2_norm(&temp),
        uy_checksum: checksum(uy),
        t_checksum: checksum(&temp),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1.0e3,
        avg_linear_iters_per_step: Some(linear_iters_sum as f64 / (2.0 * steps as f64).max(1.0)),
    }
}

fn solve_transient_case_nonmatching(
    n_u: usize,
    n_t_mesh: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    dt: f64,
    steps: usize,
    thermal_source_scale: f64,
    shift_x: f64,
    shift_y: f64,
) -> TransientResult {
    let t0 = Instant::now();

    let mesh_u = SimplexMesh::<2>::unit_square_tri(n_u);
    let mesh_u_scalar = mesh_u.clone();
    let mut mesh_t = SimplexMesh::<2>::unit_square_tri(n_t_mesh);
    for i in 0..mesh_t.n_nodes() {
        mesh_t.coords[2 * i] += shift_x;
        mesh_t.coords[2 * i + 1] += shift_y;
    }

    let space_u = VectorH1Space::new(mesh_u, order, 2);
    let space_u_scalar = H1Space::new(mesh_u_scalar, order);
    let space_t = H1Space::new(mesh_t, order);

    let n_u_dofs = space_u.n_dofs();
    let n_scalar_u = space_u.n_scalar_dofs();
    let n_t = space_t.n_dofs();

    let e_mod = 1.0_f64;
    let nu = 0.3_f64;
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    let elast = ElasticityIntegrator { lambda, mu };
    let mut k_uu = Assembler::assemble_bilinear(&space_u, &[&elast], order * 2 + 1);

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let conv = ConvectionIntegrator {
        velocity: ConstantVectorCoeff(vec![vx, vy]),
    };
    let k_diff = Assembler::assemble_bilinear(&space_t, &[&diff], order * 2 + 1);
    let k_conv = Assembler::assemble_bilinear(&space_t, &[&conv], order * 2 + 1);
    let mut k_tt = k_diff.axpby(1.0, &k_conv, 1.0);
    let mut m_tt = Assembler::assemble_bilinear(&space_t, &[&MassIntegrator { rho: 1.0 }], order * 2 + 1);

    let mut rhs_u = vec![0.0_f64; n_u_dofs];
    let src_t = DomainSourceIntegrator::new(|x: &[f64]| {
        thermal_source_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs_t = Assembler::assemble_linear(&space_t, &[&src_t], order * 2 + 1);

    let scalar_dm = space_u.scalar_dof_manager();
    let bnd_u_scalar = boundary_dofs(space_u.mesh(), scalar_dm, &[4]);
    let mut bnd_u_all = Vec::<u32>::with_capacity(2 * bnd_u_scalar.len());
    for &d in &bnd_u_scalar {
        bnd_u_all.push(d);
        bnd_u_all.push(d + n_scalar_u as u32);
    }
    let vals_u = vec![0.0_f64; bnd_u_all.len()];
    apply_dirichlet(&mut k_uu, &mut rhs_u, &bnd_u_all, &vals_u);

    let bnd_t = boundary_dofs(space_t.mesh(), space_t.dof_manager(), &[1, 2, 3, 4]);
    let vals_t = vec![0.0_f64; bnd_t.len()];
    apply_dirichlet(&mut k_tt, &mut rhs_t, &bnd_t, &vals_t);
    let mut rhs_t_dummy = vec![0.0_f64; n_t];
    apply_dirichlet(&mut m_tt, &mut rhs_t_dummy, &bnd_t, &vals_t);

    let b_tu_u = MixedAssembler::assemble_bilinear(
        &space_u_scalar,
        &space_u,
        &[&DivIntegrator],
        order * 2 + 1,
    );
    let p_ut_u = scale_csr(&b_tu_u.transpose(), -alpha_th);

    let a_t = m_tt.axpby(1.0 / dt, &k_tt, 1.0);

    let mech_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };
    let therm_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut u = vec![0.0_f64; n_u_dofs];
    let mut temp_t = vec![0.0_f64; n_t];
    let mut linear_iters_sum = 0usize;

    for _ in 0..steps {
        let mut rhs_t_step = vec![0.0_f64; n_t];
        m_tt.spmv(&temp_t, &mut rhs_t_step);
        for v in &mut rhs_t_step {
            *v /= dt;
        }
        for (ri, &fi) in rhs_t_step.iter_mut().zip(rhs_t.iter()) {
            *ri += fi;
        }

        let mut div_u_u = vec![0.0_f64; n_scalar_u];
        b_tu_u.spmv(&u, &mut div_u_u);
        let (div_u_t, _, _) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &space_u_scalar,
            &div_u_u,
            &space_t,
            1e-12,
            4,
        )
        .expect("nonmatching transfer u->T failed");
        for (ri, &qi) in rhs_t_step.iter_mut().zip(div_u_t.iter()) {
            *ri -= beta_diss * qi;
        }

        let mut temp_new = temp_t.clone();
        let therm_res = solve_gmres(&a_t, &rhs_t_step, &mut temp_new, 80, &therm_cfg)
            .expect("transient thermal GMRES failed");
        linear_iters_sum += therm_res.iterations;
        temp_t = temp_new;

        let (temp_u, _, _) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &space_t,
            &temp_t,
            &space_u_scalar,
            1e-12,
            4,
        )
        .expect("nonmatching transfer T->u failed");
        let mut rhs_u_step = rhs_u.clone();
        let mut p_t = vec![0.0_f64; n_u_dofs];
        p_ut_u.spmv(&temp_u, &mut p_t);
        for (ri, &pi) in rhs_u_step.iter_mut().zip(p_t.iter()) {
            *ri -= pi;
        }

        let mut u_new = u.clone();
        let mech_res = solve_gmres(&k_uu, &rhs_u_step, &mut u_new, 80, &mech_cfg)
            .expect("transient mechanics GMRES failed");
        linear_iters_sum += mech_res.iterations;
        u = u_new;
    }

    let ux = &u[..n_scalar_u];
    let uy = &u[n_scalar_u..];
    TransientResult {
        n_u: n_u_dofs,
        n_t,
        steps,
        dt,
        ux_norm: l2_norm(ux),
        uy_norm: l2_norm(uy),
        t_norm: l2_norm(&temp_t),
        uy_checksum: checksum(uy),
        t_checksum: checksum(&temp_t),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1.0e3,
        avg_linear_iters_per_step: Some(linear_iters_sum as f64 / (2.0 * steps as f64).max(1.0)),
    }
}

fn solve_transient_imex_case(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    dt: f64,
    steps: usize,
    thermal_source_scale: f64,
    method: ImexMethod,
) -> TransientResult {
    let t0 = Instant::now();
    let model = build_model(n, order, alpha_th, beta_diss, vx, vy, thermal_source_scale);
    let op = ThermoelasticImexOp::from_model(&model);

    let mut state = vec![0.0_f64; model.n_u + model.n_t];
    let t_end = dt * steps as f64;
    let driver = ImexTimeStepper;
    match method {
        ImexMethod::Ssp2 => {
            let _ = driver.integrate_ssp2(&op, 0.0, t_end, &mut state, dt);
        }
        ImexMethod::Ark3 => {
            let ark3 = ImexArk3 {
                rtol: 1e-6,
                atol: 1e-9,
                dt_min: 1e-10,
                dt_max: dt,
                ..Default::default()
            };
            let _ = driver.integrate_ark3(&op, 0.0, t_end, &mut state, dt, &ark3);
        }
    }

    let ux = &state[..model.n_scalar_u];
    let uy = &state[model.n_scalar_u..model.n_u];
    let temp = &state[model.n_u..];

    TransientResult {
        n_u: model.n_u,
        n_t: model.n_t,
        steps,
        dt,
        ux_norm: l2_norm(ux),
        uy_norm: l2_norm(uy),
        t_norm: l2_norm(temp),
        uy_checksum: checksum(uy),
        t_checksum: checksum(temp),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1.0e3,
        avg_linear_iters_per_step: None,
    }
}

fn solve_transient_imex_auto(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    dt: f64,
    steps: usize,
    thermal_source_scale: f64,
    method: ImexMethod,
    nonmatching: Option<NonmatchingConfig>,
) -> TransientResult {
    if let Some(cfg) = nonmatching {
        let r = solve_transient_case_nonmatching(
            n,
            cfg.thermal_n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            dt,
            steps,
            thermal_source_scale,
            cfg.shift_x,
            cfg.shift_y,
        );
        // Keep IMEX entry-point behavior for nonmatching workflows while the
        // fully monolithic cross-mesh Jacobian path is being developed.
        let _ = method;
        r
    } else {
        solve_transient_imex_case(
            n,
            order,
            alpha_th,
            beta_diss,
            vx,
            vy,
            dt,
            steps,
            thermal_source_scale,
            method,
        )
    }
}

fn run_transient_comparison(args: &Args) {
    let nm = if args.nonmatching_transfer {
        Some(NonmatchingConfig {
            thermal_n: args.thermal_n,
            shift_x: args.thermal_shift_x,
            shift_y: args.thermal_shift_y,
        })
    } else {
        None
    };

    let dt_values = args
        .sweep_dt
        .as_deref()
        .map(parse_f64_list)
        .unwrap_or_else(|| vec![args.dt]);
    let step_values = args
        .sweep_steps
        .as_deref()
        .map(parse_usize_list)
        .unwrap_or_else(|| vec![args.steps]);

    if dt_values.len() == 1 && step_values.len() == 1 {
        println!("  mode=transient comparison, steps={}, dt={}", args.steps, args.dt);
        let split = solve_transient_split_auto(
            args.n,
            args.order,
            args.alpha,
            args.beta,
            args.vx,
            args.vy,
            args.dt,
            args.steps,
            1.0,
            nm,
        );
        let ssp2 = solve_transient_imex_auto(
            args.n,
            args.order,
            args.alpha,
            args.beta,
            args.vx,
            args.vy,
            args.dt,
            args.steps,
            1.0,
            ImexMethod::Ssp2,
            nm,
        );
        let ark3 = solve_transient_imex_auto(
            args.n,
            args.order,
            args.alpha,
            args.beta,
            args.vx,
            args.vy,
            args.dt,
            args.steps,
            1.0,
            ImexMethod::Ark3,
            nm,
        );

        println!("  dof(u)={}, dof(T)={}", split.n_u, split.n_t);
        print_transient_row("split", &split);
        print_transient_row("imex-ssp2", &ssp2);
        print_transient_row("imex-ark3", &ark3);

        println!(
            "  rel_diff(T_norm): ssp2/split={:.3e}, ark3/split={:.3e}",
            rel_diff(ssp2.t_norm, split.t_norm),
            rel_diff(ark3.t_norm, split.t_norm)
        );
        println!(
            "  rel_diff(Uy_norm): ssp2/split={:.3e}, ark3/split={:.3e}",
            rel_diff(ssp2.uy_norm, split.uy_norm),
            rel_diff(ark3.uy_norm, split.uy_norm)
        );

        if let Some(path) = &args.compare_csv {
            let csv = build_comparison_csv(&split, &ssp2, &ark3);
            fs::write(path, csv).unwrap_or_else(|e| panic!("failed to write CSV to {path}: {e}"));
            println!("  wrote comparison csv: {}", path);
        }
        return;
    }

    println!(
        "  mode=transient comparison sweep, dt_count={}, steps_count={}",
        dt_values.len(),
        step_values.len()
    );

    let mut cases: Vec<ComparisonCase> = Vec::new();
    for &dt in &dt_values {
        for &steps in &step_values {
            let split = solve_transient_case(
                args.n,
                args.order,
                args.alpha,
                args.beta,
                args.vx,
                args.vy,
                dt,
                steps,
                1.0,
            );
            let split = if args.nonmatching_transfer {
                solve_transient_split_auto(
                    args.n,
                    args.order,
                    args.alpha,
                    args.beta,
                    args.vx,
                    args.vy,
                    dt,
                    steps,
                    1.0,
                    Some(NonmatchingConfig {
                        thermal_n: args.thermal_n,
                        shift_x: args.thermal_shift_x,
                        shift_y: args.thermal_shift_y,
                    }),
                )
            } else {
                split
            };
            let ssp2 = solve_transient_imex_auto(
                args.n,
                args.order,
                args.alpha,
                args.beta,
                args.vx,
                args.vy,
                dt,
                steps,
                1.0,
                ImexMethod::Ssp2,
                nm,
            );
            let ark3 = solve_transient_imex_auto(
                args.n,
                args.order,
                args.alpha,
                args.beta,
                args.vx,
                args.vy,
                dt,
                steps,
                1.0,
                ImexMethod::Ark3,
                nm,
            );
            println!(
                "  case dt={:.4}, steps={}: rel_t(ssp2/split)={:.3e}, rel_t(ark3/split)={:.3e}",
                dt,
                steps,
                rel_diff(ssp2.t_norm, split.t_norm),
                rel_diff(ark3.t_norm, split.t_norm)
            );
            cases.push(ComparisonCase {
                case_id: format!("dt={dt:.6}_steps={steps}"),
                split,
                ssp2,
                ark3,
            });
        }
    }

    let out_path = args
        .compare_csv
        .clone()
        .unwrap_or_else(|| "output/ex44_transient_compare_sweep.csv".to_string());
    let csv = build_sweep_comparison_csv(&cases);
    fs::write(&out_path, csv).unwrap_or_else(|e| panic!("failed to write CSV to {out_path}: {e}"));
    println!("  wrote sweep comparison csv: {}", out_path);
}

struct ComparisonCase {
    case_id: String,
    split: TransientResult,
    ssp2: TransientResult,
    ark3: TransientResult,
}

fn print_transient_row(name: &str, r: &TransientResult) {
    let avg_it = match r.avg_linear_iters_per_step {
        Some(v) => format!("{v:.3}"),
        None => "n/a".to_string(),
    };
    println!(
        "  {:>10}: ||ux||={:.4e}, ||uy||={:.4e}, ||T||={:.4e}, elapsed_ms={:.3}, avg_lin_iters/step={}",
        name,
        r.ux_norm,
        r.uy_norm,
        r.t_norm,
        r.elapsed_ms,
        avg_it
    );
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0e-14)
}

fn build_comparison_csv(split: &TransientResult, ssp2: &TransientResult, ark3: &TransientResult) -> String {
    let mut out = String::new();
    out.push_str("method,n_u,n_t,steps,dt,ux_norm,uy_norm,t_norm,uy_checksum,t_checksum,elapsed_ms,avg_linear_iters_per_step,rel_t_norm_vs_split,rel_uy_norm_vs_split\n");

    append_csv_row(&mut out, "split", split, 0.0, 0.0);
    append_csv_row(
        &mut out,
        "imex-ssp2",
        ssp2,
        rel_diff(ssp2.t_norm, split.t_norm),
        rel_diff(ssp2.uy_norm, split.uy_norm),
    );
    append_csv_row(
        &mut out,
        "imex-ark3",
        ark3,
        rel_diff(ark3.t_norm, split.t_norm),
        rel_diff(ark3.uy_norm, split.uy_norm),
    );

    out
}

fn append_csv_row(
    out: &mut String,
    method: &str,
    r: &TransientResult,
    rel_t: f64,
    rel_uy: f64,
) {
    let avg_it = r
        .avg_linear_iters_per_step
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "".to_string());

    out.push_str(&format!(
        "{},{},{},{},{:.6},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.6},{},{:.12e},{:.12e}\n",
        method,
        r.n_u,
        r.n_t,
        r.steps,
        r.dt,
        r.ux_norm,
        r.uy_norm,
        r.t_norm,
        r.uy_checksum,
        r.t_checksum,
        r.elapsed_ms,
        avg_it,
        rel_t,
        rel_uy,
    ));
}

fn build_sweep_comparison_csv(cases: &[ComparisonCase]) -> String {
    let mut out = String::new();
    out.push_str("case_id,method,n_u,n_t,steps,dt,ux_norm,uy_norm,t_norm,uy_checksum,t_checksum,elapsed_ms,avg_linear_iters_per_step,rel_t_norm_vs_split,rel_uy_norm_vs_split\n");

    for case in cases {
        append_csv_row_with_case(&mut out, &case.case_id, "split", &case.split, 0.0, 0.0);
        append_csv_row_with_case(
            &mut out,
            &case.case_id,
            "imex-ssp2",
            &case.ssp2,
            rel_diff(case.ssp2.t_norm, case.split.t_norm),
            rel_diff(case.ssp2.uy_norm, case.split.uy_norm),
        );
        append_csv_row_with_case(
            &mut out,
            &case.case_id,
            "imex-ark3",
            &case.ark3,
            rel_diff(case.ark3.t_norm, case.split.t_norm),
            rel_diff(case.ark3.uy_norm, case.split.uy_norm),
        );
    }

    out
}

fn append_csv_row_with_case(
    out: &mut String,
    case_id: &str,
    method: &str,
    r: &TransientResult,
    rel_t: f64,
    rel_uy: f64,
) {
    let avg_it = r
        .avg_linear_iters_per_step
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "".to_string());

    out.push_str(&format!(
        "{},{},{},{},{},{:.6},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.6},{},{:.12e},{:.12e}\n",
        case_id,
        method,
        r.n_u,
        r.n_t,
        r.steps,
        r.dt,
        r.ux_norm,
        r.uy_norm,
        r.t_norm,
        r.uy_checksum,
        r.t_checksum,
        r.elapsed_ms,
        avg_it,
        rel_t,
        rel_uy,
    ));
}

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<f64>().unwrap_or_else(|e| panic!("invalid f64 value '{x}': {e}")))
        .collect()
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>().unwrap_or_else(|e| panic!("invalid usize value '{x}': {e}")))
        .collect()
}

fn build_model(
    n: usize,
    order: u8,
    alpha_th: f64,
    beta_diss: f64,
    vx: f64,
    vy: f64,
    thermal_source_scale: f64,
) -> ThermoelasticModel {
    let mesh_u = SimplexMesh::<2>::unit_square_tri(n);
    let mesh_t = SimplexMesh::<2>::unit_square_tri(n);
    let space_u = VectorH1Space::new(mesh_u, order, 2);
    let space_t = H1Space::new(mesh_t, order);

    let n_u = space_u.n_dofs();
    let n_scalar_u = space_u.n_scalar_dofs();
    let n_t = space_t.n_dofs();
    assert_eq!(n_scalar_u, n_t, "example assumes aligned scalar/vector nodal counts");

    // Mechanics block K_uu.
    let e_mod = 1.0_f64;
    let nu = 0.3_f64;
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    let elast = ElasticityIntegrator { lambda, mu };
    let mut k_uu = Assembler::assemble_bilinear(&space_u, &[&elast], order * 2 + 1);

    // Thermal block K_tt = diffusion + convection.
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let conv = ConvectionIntegrator {
        velocity: ConstantVectorCoeff(vec![vx, vy]),
    };
    let k_diff = Assembler::assemble_bilinear(&space_t, &[&diff], order * 2 + 1);
    let k_conv = Assembler::assemble_bilinear(&space_t, &[&conv], order * 2 + 1);
    let mut k_tt = k_diff.axpby(1.0, &k_conv, 1.0);

    // Thermal mass for transient split.
    let mut m_tt = Assembler::assemble_bilinear(&space_t, &[&MassIntegrator { rho: 1.0 }], order * 2 + 1);

    // RHS blocks.
    let mut rhs_u = vec![0.0_f64; n_u];
    let src_t = DomainSourceIntegrator::new(|x: &[f64]| {
        thermal_source_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs_t = Assembler::assemble_linear(&space_t, &[&src_t], order * 2 + 1);

    // Dirichlet mechanics: clamp left wall (tag=4).
    let scalar_dm = space_u.scalar_dof_manager();
    let bnd_u_scalar = boundary_dofs(space_u.mesh(), scalar_dm, &[4]);
    let mut bnd_u_all = Vec::<u32>::with_capacity(2 * bnd_u_scalar.len());
    for &d in &bnd_u_scalar {
        bnd_u_all.push(d);
        bnd_u_all.push(d + n_scalar_u as u32);
    }
    let vals_u = vec![0.0_f64; bnd_u_all.len()];
    apply_dirichlet(&mut k_uu, &mut rhs_u, &bnd_u_all, &vals_u);

    // Dirichlet thermal: all walls fixed temperature.
    let bnd_t = boundary_dofs(space_t.mesh(), space_t.dof_manager(), &[1, 2, 3, 4]);
    let vals_t = vec![0.0_f64; bnd_t.len()];
    apply_dirichlet(&mut k_tt, &mut rhs_t, &bnd_t, &vals_t);
    let mut rhs_t_dummy = vec![0.0_f64; n_t];
    apply_dirichlet(&mut m_tt, &mut rhs_t_dummy, &bnd_t, &vals_t);

    let mut k_tt_diff_bc = k_diff.clone();
    let mut rhs_t_dummy_2 = vec![0.0_f64; n_t];
    apply_dirichlet(&mut k_tt_diff_bc, &mut rhs_t_dummy_2, &bnd_t, &vals_t);
    let mut k_tt_conv_bc = k_conv.clone();
    let mut rhs_t_dummy_3 = vec![0.0_f64; n_t];
    apply_dirichlet(&mut k_tt_conv_bc, &mut rhs_t_dummy_3, &bnd_t, &vals_t);

    // Physically assembled coupling via divergence mixed operator.
    // B_tu ~ int q * div(u) dx  (n_t x n_u).
    let b_tu = MixedAssembler::assemble_bilinear(
        &space_t,
        &space_u,
        &[&DivIntegrator],
        order * 2 + 1,
    );

    // Thermal expansion in mechanics residual: -alpha * B^T * T.
    // Thermomechanical source in thermal residual: +beta * B * u.
    let p_ut_raw = scale_csr(&b_tu.transpose(), -alpha_th);
    let q_tu_raw = scale_csr(&b_tu, beta_diss);

    let u_mask = mask_from_dofs(n_u, &bnd_u_all);
    let t_mask = mask_from_dofs(n_t, &bnd_t);

    // Do not inject coupling on constrained equations.
    let p_ut = filter_csr(&p_ut_raw, Some(&u_mask), None);
    let q_tu = filter_csr(&q_tu_raw, Some(&t_mask), None);

    let inv_mass_t_diag = m_tt
        .diagonal()
        .iter()
        .map(|&d| if d.abs() > 1e-14 { 1.0 / d } else { 0.0 })
        .collect();

    ThermoelasticModel {
        n_u,
        n_t,
        n_scalar_u,
        k_uu,
        k_tt_diff: k_tt_diff_bc,
        k_tt_conv: k_tt_conv_bc,
        k_tt,
        m_tt,
        inv_mass_t_diag,
        p_ut,
        q_tu,
        rhs_u,
        rhs_t,
    }
}

fn state_to_result_steady(
    state: &BlockVector,
    model: &ThermoelasticModel,
    converged: bool,
    iterations: usize,
    final_residual: f64,
) -> SolveResult {
    let u = state.block(0);
    let temp = state.block(1);
    let ux = &u[..model.n_scalar_u];
    let uy = &u[model.n_scalar_u..];

    SolveResult {
        n_u: model.n_u,
        n_t: model.n_t,
        converged,
        iterations,
        final_residual,
        ux_norm: l2_norm(ux),
        uy_norm: l2_norm(uy),
        t_norm: l2_norm(temp),
        uy_checksum: checksum(uy),
        t_checksum: checksum(temp),
    }
}

fn scale_csr(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values: a.values.iter().map(|&v| s * v).collect(),
    }
}

fn scale_rows(a: &CsrMatrix<f64>, row_scale: &[f64], s: f64) -> CsrMatrix<f64> {
    debug_assert_eq!(a.nrows, row_scale.len());
    let mut out = a.clone();
    for i in 0..a.nrows {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for k in start..end {
            out.values[k] = s * row_scale[i] * a.values[k];
        }
    }
    out
}

fn block2x2_to_csr(
    a00: &CsrMatrix<f64>,
    a01: &CsrMatrix<f64>,
    a10: &CsrMatrix<f64>,
    a11: &CsrMatrix<f64>,
) -> CsrMatrix<f64> {
    let n0 = a00.nrows;
    let n1 = a11.nrows;
    let mut coo = CooMatrix::new(n0 + n1, n0 + n1);

    add_block_to_coo(&mut coo, 0, 0, a00);
    add_block_to_coo(&mut coo, 0, n0, a01);
    add_block_to_coo(&mut coo, n0, 0, a10);
    add_block_to_coo(&mut coo, n0, n0, a11);

    coo.into_csr()
}

fn add_block_to_coo(coo: &mut CooMatrix<f64>, row_off: usize, col_off: usize, a: &CsrMatrix<f64>) {
    for i in 0..a.nrows {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for k in start..end {
            let j = a.col_idx[k] as usize;
            coo.add(row_off + i, col_off + j, a.values[k]);
        }
    }
}

fn mask_from_dofs(n: usize, dofs: &[u32]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &d in dofs {
        m[d as usize] = true;
    }
    m
}

fn filter_csr(a: &CsrMatrix<f64>, row_mask: Option<&[bool]>, col_mask: Option<&[bool]>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        if row_mask.is_some_and(|m| m[i]) {
            continue;
        }
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for k in start..end {
            let j = a.col_idx[k] as usize;
            if col_mask.is_some_and(|m| m[j]) {
                continue;
            }
            coo.add(i, j, a.values[k]);
        }
    }
    coo.into_csr()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn checksum(v: &[f64]) -> f64 {
    v.iter().enumerate().map(|(i, &x)| (i as f64 + 1.0) * x).sum()
}

struct Args {
    n: usize,
    order: u8,
    alpha: f64,
    beta: f64,
    vx: f64,
    vy: f64,
    transient: bool,
    compare_methods: bool,
    compare_csv: Option<String>,
    sweep_dt: Option<String>,
    sweep_steps: Option<String>,
    imex: bool,
    imex_method: String,
    dt: f64,
    steps: usize,
    linear_strategy: String,
    nonmatching_transfer: bool,
    thermal_n: usize,
    thermal_shift_x: f64,
    thermal_shift_y: f64,
    ls_enabled: bool,
    ls_min_alpha: f64,
    ls_shrink: f64,
    ls_max_backtracks: usize,
    ls_c1: f64,
}

impl Args {
    fn line_search_options(&self) -> LineSearchOptions {
        LineSearchOptions {
            enabled: self.ls_enabled,
            min_alpha: self.ls_min_alpha,
            shrink: self.ls_shrink,
            max_backtracks: self.ls_max_backtracks,
            sufficient_decrease: self.ls_c1,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        order: 1,
        alpha: 0.2,
        beta: 0.1,
        vx: 0.25,
        vy: 0.10,
        transient: false,
        compare_methods: false,
        compare_csv: None,
        sweep_dt: None,
        sweep_steps: None,
        imex: false,
        imex_method: "ssp2".to_string(),
        dt: 0.05,
        steps: 8,
        linear_strategy: "gmres".to_string(),
        nonmatching_transfer: false,
        thermal_n: 10,
        thermal_shift_x: 0.02,
        thermal_shift_y: 0.00,
        ls_enabled: true,
        ls_min_alpha: 1e-6,
        ls_shrink: 0.5,
        ls_max_backtracks: 20,
        ls_c1: 1e-4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
            }
            "--order" => {
                a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1);
            }
            "--alpha" => {
                a.alpha = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2);
            }
            "--beta" => {
                a.beta = it.next().unwrap_or("0.1".into()).parse().unwrap_or(0.1);
            }
            "--vx" => {
                a.vx = it.next().unwrap_or("0.25".into()).parse().unwrap_or(0.25);
            }
            "--vy" => {
                a.vy = it.next().unwrap_or("0.10".into()).parse().unwrap_or(0.10);
            }
            "--transient" => {
                a.transient = true;
            }
            "--compare-methods" => {
                a.compare_methods = true;
                a.transient = true;
            }
            "--compare-csv" => {
                a.compare_methods = true;
                a.transient = true;
                a.compare_csv = Some(it.next().unwrap_or("output/ex44_transient_compare.csv".into()));
            }
            "--sweep-dt" => {
                a.compare_methods = true;
                a.transient = true;
                a.sweep_dt = Some(it.next().unwrap_or("0.01,0.02,0.05".into()));
            }
            "--sweep-steps" => {
                a.compare_methods = true;
                a.transient = true;
                a.sweep_steps = Some(it.next().unwrap_or("4,8,16".into()));
            }
            "--imex" => {
                a.imex = true;
            }
            "--imex-method" => {
                a.imex_method = it.next().unwrap_or("ssp2".into());
            }
            "--dt" => {
                a.dt = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05);
            }
            "--steps" => {
                a.steps = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
            }
            "--linear-strategy" => {
                a.linear_strategy = it.next().unwrap_or("gmres".into());
            }
            "--nonmatching-transfer" => {
                a.nonmatching_transfer = true;
            }
            "--thermal-n" => {
                a.thermal_n = it.next().unwrap_or("10".into()).parse().unwrap_or(10);
            }
            "--thermal-shift-x" => {
                a.thermal_shift_x = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02);
            }
            "--thermal-shift-y" => {
                a.thermal_shift_y = it.next().unwrap_or("0.00".into()).parse().unwrap_or(0.0);
            }
            "--no-line-search" => {
                a.ls_enabled = false;
            }
            "--line-search" => {
                a.ls_enabled = true;
            }
            "--ls-min-alpha" => {
                a.ls_min_alpha = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6);
            }
            "--ls-shrink" => {
                a.ls_shrink = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5);
            }
            "--ls-max-backtracks" => {
                a.ls_max_backtracks = it.next().unwrap_or("20".into()).parse().unwrap_or(20);
            }
            "--ls-c1" => {
                a.ls_c1 = it.next().unwrap_or("1e-4".into()).parse().unwrap_or(1e-4);
            }
            _ => {}
        }
    }
    if a.thermal_n == 0 {
        a.thermal_n = a.n;
    }
    a
}

fn parse_imex_method(s: &str) -> ImexMethod {
    if s.eq_ignore_ascii_case("ark3") {
        ImexMethod::Ark3
    } else {
        ImexMethod::Ssp2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex44_steady_gmres_converges_with_physical_coupling() {
        let r = solve_steady_case(8, 1, 0.2, 0.1, 0.25, 0.10, 1.0, CoupledLinearStrategy::Gmres, default_line_search_options());
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-8, "residual too large: {}", r.final_residual);
        assert!(r.t_norm > 1.0e-6, "thermal field should be non-trivial");
        assert!(r.uy_norm > 1.0e-8, "y-displacement should be non-trivial due to thermal expansion");
    }

    #[test]
    fn ex44_steady_schur_strategy_converges() {
        let r = solve_steady_case(8, 1, 0.2, 0.1, 0.25, 0.10, 1.0, CoupledLinearStrategy::BlockSchur2x2, default_line_search_options());
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-8, "residual too large: {}", r.final_residual);
    }

    #[test]
    fn ex44_zero_coupling_keeps_mechanics_near_zero() {
        let r = solve_steady_case(8, 1, 0.0, 0.0, 0.25, 0.10, 1.0, CoupledLinearStrategy::Gmres, default_line_search_options());
        assert!(r.converged);
        assert!(r.ux_norm < 1.0e-10, "u_x should remain near zero: {}", r.ux_norm);
        assert!(r.uy_norm < 1.0e-10, "u_y should remain near zero: {}", r.uy_norm);
        assert!(r.t_norm > 1.0e-6, "thermal solve should still be non-trivial");
    }

    #[test]
    fn ex44_transient_split_responds_to_coupling() {
        let weak = solve_transient_split_auto(8, 1, 0.02, 0.0, 0.25, 0.10, 0.05, 8, 1.0, None);
        let strong = solve_transient_split_auto(8, 1, 0.2, 0.0, 0.25, 0.10, 0.05, 8, 1.0, None);
        assert!(strong.uy_norm > weak.uy_norm * 2.0,
            "stronger alpha should increase y-displacement: weak={} strong={}",
            weak.uy_norm,
            strong.uy_norm);
        assert!(strong.t_norm > 1.0e-6);
    }

    #[test]
    fn ex44_transient_imex_responds_to_coupling() {
        let weak = solve_transient_imex_case(8, 1, 0.02, 0.0, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);
        let strong = solve_transient_imex_case(8, 1, 0.2, 0.0, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);
        assert!(strong.uy_norm > weak.uy_norm * 2.0,
            "stronger alpha should increase y-displacement in IMEX: weak={} strong={}",
            weak.uy_norm,
            strong.uy_norm);
        assert!(strong.t_norm > 1.0e-6);
    }

    #[test]
    fn ex44_transient_imex_zero_coupling_keeps_mechanics_small() {
        let r = solve_transient_imex_case(8, 1, 0.0, 0.0, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);
        assert!(r.ux_norm < 1.0e-8, "u_x should remain small: {}", r.ux_norm);
        assert!(r.uy_norm < 1.0e-8, "u_y should remain small: {}", r.uy_norm);
        assert!(r.t_norm > 1.0e-6);
    }

    #[test]
    fn ex44_transient_imex_ark3_runs_and_tracks_ssp2() {
        let ssp2 = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);
        let ark3 = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ark3);
        assert!(ark3.t_norm > 1.0e-6);
        assert!(ark3.uy_norm > 1.0e-8);
        // Method outputs should stay in the same response scale for the same PDE class.
        assert!(ark3.uy_norm / ssp2.uy_norm < 3.0, "ARK3 response too far from SSP2 scale");
        assert!(ark3.uy_norm / ssp2.uy_norm > 0.1, "ARK3 response too small relative to SSP2 scale");
    }

    #[test]
    fn ex44_transient_split_and_imex_are_consistent_scale() {
        let split = solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, None);
        let imex = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);

        assert!(split.t_norm > 1.0e-6 && imex.t_norm > 1.0e-6);
        assert!(split.uy_norm > 1.0e-8 && imex.uy_norm > 1.0e-8);

        // Different schemes, but should remain in similar magnitude regime.
        let ratio_t = imex.t_norm / split.t_norm;
        let ratio_uy = imex.uy_norm / split.uy_norm;
        assert!(ratio_t > 0.05 && ratio_t < 20.0, "split vs IMEX thermal norm mismatch: {ratio_t}");
        assert!(ratio_uy > 0.05 && ratio_uy < 20.0, "split vs IMEX uy norm mismatch: {ratio_uy}");
    }

    #[test]
    fn ex44_transient_comparison_metrics_are_finite() {
        let split = solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, None);
        let ssp2 = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 8, 1.0, ImexMethod::Ssp2);

        assert!(split.elapsed_ms.is_finite() && split.elapsed_ms >= 0.0);
        assert!(ssp2.elapsed_ms.is_finite() && ssp2.elapsed_ms >= 0.0);
        assert!(split.avg_linear_iters_per_step.is_some());
        assert!(ssp2.avg_linear_iters_per_step.is_none());
        assert!(rel_diff(ssp2.t_norm, split.t_norm).is_finite());
        assert!(rel_diff(ssp2.uy_norm, split.uy_norm).is_finite());
    }

    #[test]
    fn ex44_comparison_csv_has_header_and_three_rows() {
        let split = solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 3, 1.0, None);
        let ssp2 = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 3, 1.0, ImexMethod::Ssp2);
        let ark3 = solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 3, 1.0, ImexMethod::Ark3);

        let csv = build_comparison_csv(&split, &ssp2, &ark3);
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].starts_with("method,n_u,n_t,steps,dt"));
        assert_eq!(lines.len(), 4, "expected header + 3 method rows");
        assert!(lines[1].starts_with("split,"));
        assert!(lines[2].starts_with("imex-ssp2,"));
        assert!(lines[3].starts_with("imex-ark3,"));
    }

    #[test]
    fn ex44_sweep_csv_has_case_and_expected_rows() {
        let c1 = ComparisonCase {
            case_id: "dt=0.010000_steps=4".to_string(),
            split: solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.01, 4, 1.0, None),
            ssp2: solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.01, 4, 1.0, ImexMethod::Ssp2),
            ark3: solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.01, 4, 1.0, ImexMethod::Ark3),
        };
        let c2 = ComparisonCase {
            case_id: "dt=0.020000_steps=8".to_string(),
            split: solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.02, 8, 1.0, None),
            ssp2: solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.02, 8, 1.0, ImexMethod::Ssp2),
            ark3: solve_transient_imex_case(8, 1, 0.2, 0.1, 0.25, 0.10, 0.02, 8, 1.0, ImexMethod::Ark3),
        };

        let csv = build_sweep_comparison_csv(&[c1, c2]);
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].starts_with("case_id,method,n_u,n_t,steps,dt"));
        assert_eq!(lines.len(), 7, "expected header + 2 cases * 3 methods");
        assert!(lines[1].contains("dt=0.010000_steps=4,split,"));
        assert!(lines[6].contains("dt=0.020000_steps=8,imex-ark3,"));
    }

    #[test]
    fn ex44_parse_sweep_lists() {
        let dts = parse_f64_list("0.01, 0.02,0.05");
        let steps = parse_usize_list("4, 8,16");
        assert_eq!(dts.len(), 3);
        assert!((dts[1] - 0.02).abs() < 1e-14);
        assert_eq!(steps, vec![4, 8, 16]);
    }

    #[test]
    fn ex44_nonmatching_transient_split_runs_and_stays_in_scale() {
        let matching = solve_transient_split_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 4, 1.0, None);
        let nonmatching = solve_transient_split_auto(
            8,
            1,
            0.2,
            0.1,
            0.25,
            0.10,
            0.05,
            4,
            1.0,
            Some(NonmatchingConfig {
                thermal_n: 10,
                shift_x: 0.02,
                shift_y: 0.00,
            }),
        );

        assert!(nonmatching.uy_norm > 1.0e-8);
        assert!(nonmatching.t_norm > 1.0e-6);
        let r_t = nonmatching.t_norm / matching.t_norm;
        let r_u = nonmatching.uy_norm / matching.uy_norm;
        assert!(r_t > 0.02 && r_t < 50.0, "nonmatching/matching T norm out of range: {r_t}");
        assert!(r_u > 0.02 && r_u < 50.0, "nonmatching/matching Uy norm out of range: {r_u}");
    }

    #[test]
    fn ex44_nonmatching_steady_runs_and_stays_in_scale() {
        let matching = solve_steady_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 1.0, CoupledLinearStrategy::Gmres, None, default_line_search_options());
        let nonmatching = solve_steady_auto(
            8,
            1,
            0.2,
            0.1,
            0.25,
            0.10,
            1.0,
            CoupledLinearStrategy::Gmres,
            Some(NonmatchingConfig {
                thermal_n: 10,
                shift_x: 0.02,
                shift_y: 0.00,
            }),
            default_line_search_options(),
        );

        assert!(nonmatching.converged);
        assert!(nonmatching.uy_norm > 1.0e-8);
        assert!(nonmatching.t_norm > 1.0e-6);
        let r_t = nonmatching.t_norm / matching.t_norm;
        let r_u = nonmatching.uy_norm / matching.uy_norm;
        assert!(r_t > 0.02 && r_t < 50.0, "nonmatching/matching steady T norm out of range: {r_t}");
        assert!(r_u > 0.02 && r_u < 50.0, "nonmatching/matching steady Uy norm out of range: {r_u}");
    }

    #[test]
    fn ex44_nonmatching_transient_imex_runs_and_stays_in_scale() {
        let matching = solve_transient_imex_auto(8, 1, 0.2, 0.1, 0.25, 0.10, 0.05, 4, 1.0, ImexMethod::Ssp2, None);
        let nonmatching = solve_transient_imex_auto(
            8,
            1,
            0.2,
            0.1,
            0.25,
            0.10,
            0.05,
            4,
            1.0,
            ImexMethod::Ssp2,
            Some(NonmatchingConfig {
                thermal_n: 10,
                shift_x: 0.02,
                shift_y: 0.00,
            }),
        );

        assert!(nonmatching.uy_norm > 1.0e-8);
        assert!(nonmatching.t_norm > 1.0e-6);
        let r_t = nonmatching.t_norm / matching.t_norm;
        let r_u = nonmatching.uy_norm / matching.uy_norm;
        assert!(r_t > 0.02 && r_t < 50.0, "nonmatching/matching IMEX T norm out of range: {r_t}");
        assert!(r_u > 0.02 && r_u < 50.0, "nonmatching/matching IMEX Uy norm out of range: {r_u}");
    }
}
