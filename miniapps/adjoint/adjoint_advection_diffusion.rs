//! adjoint_advection_diffusion — serial-subset port of MFEM
//! `miniapps/adjoint/adjoint_advection_diffusion.cpp` (itself a port of the
//! SUNDIALS parallel example `cvsAdvDiff_ASAp_non_p`), using fem-solver's own
//! checkpointed adjoint time-integration kernel (no SUNDIALS).
//!
//! 1-D advection-diffusion, semi-discrete FEM form `M du/dt = K(p) u` on
//! `[0, 2]`, homogeneous Dirichlet BCs, initial condition
//! `u(x,0) = x(2-x)e^{2x}`, `p = (p1, p2) = (1.0, 0.5)`:
//!
//! ```text
//!    K(p) = -p1·Δ + p2·C      (forward;  Δ = FE Laplacian, C = convection)
//!    goal  g = ∫ u dx  at t = t_final
//!    adjoint  M dv/dt = K_adj v,  K_adj = +p1·Δ + p2·C,  v(t_final) = 1 (bnd 0)
//! ```
//!
//! The C++ original is parallel (ParMesh/HYPRE) and requires SUNDIALS; this
//! port reproduces the `-np 1` run with the same 1-D H1(linear) FEM
//! discretization (uniform segment mesh, `mx+1` elements). The element
//! matrices match MFEM's `MassIntegrator`/`DiffusionIntegrator`/
//! `ConvectionIntegrator` exactly on P1 segments (all integrands are exactly
//! integrated by the quadrature rules MFEM selects), so the assembled M/K are
//! identical to the serial limit of the C++ run.
//!
//! Deviations from the C++ (documented):
//! * The forward/backward integrator is the kernel's variable-order BDF
//!   (CV_BDF mode). The C++ default CV_ADAMS stepping is not reproduced; run
//!   like-for-like comparisons against the C++ `-no-a` (CV_BDF) mode.
//! * This port uses the exact discrete-gradient quadrature rate
//!   `qBdot_j = (∂f/∂p_j)ᵀ·yB = zᵀ·(∂K/∂p_j)·y` with `z = M⁻¹·yB`, verified
//!   against central finite differences of the discrete goal (`-fd 1`,
//!   relative difference < 1e-6). The C++ `QuadratureSensitivityMult`
//!   omits the `M⁻¹` (its `yB` there is the FE-function adjoint `v = M⁻¹yB`,
//!   with the mass solve folded into its adjoint operator choice
//!   `yB' = M⁻¹K_adj·yB`); the port keeps the abstract costate
//!   `yB' = −(M⁻¹K)ᵀyB` instead and applies the mass solve inside the
//!   quadrature rate.
//!
//! Run: `adjoint_advection_diffusion -dt 0.01 -tf 2.5` (defaults).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::adjoint::{
    AdjointBdfConfig, AdjointConfig, AdjointSolver, AdjointTolerances, Interpolation,
    TimeDependentAdjointOperator,
};
use fem_solver::solve_sparse_cholesky;

/// Format like C `printf("%.*g", prec, x)` — matches `std::ostream` with
/// `cout.precision(prec)` used by the C++ miniapp.
fn fmt_g_prec(x: f64, p: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{:.*e}", (p - 1) as usize, x);
    let epos = s.find('e').unwrap();
    let exp: i32 = s[epos + 1..].parse().unwrap();
    if exp < -4 || exp >= p {
        let mant = s[..epos].trim_end_matches('0').trim_end_matches('.');
        format!("{}e{}{:02}", mant, if exp < 0 { "-" } else { "+" }, exp.abs())
    } else {
        let digits = (p - 1 - exp).max(0) as usize;
        if digits == 0 {
            // Integer rendering: the trailing zeros are significant digits.
            return format!("{:.0}", x);
        }
        let f = format!("{:.*}", digits, x);
        f.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// MFEM `Vector::Print` analogue: `width` space-separated `%g` values/line.
fn print_vec(vals: &[f64], width: usize) {
    let prec = 8; // C++: cout.precision(8)
    for (i, &v) in vals.iter().enumerate() {
        print!("{}", fmt_g_prec(v, prec));
        if (i + 1) % width == 0 || i + 1 == vals.len() {
            println!();
        } else {
            print!(" ");
        }
    }
}

/// Assemble a CSR matrix from triplets.
fn triplets(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for &(i, j, v) in entries {
        coo.add(i, j, v);
    }
    coo.into_csr()
}

/// MFEM `EliminateRowsCols`: zero the rows and columns of `a` for `ess` and
/// place 1.0 on the diagonal.
fn eliminate_ess(a: &CsrMatrix<f64>, ess: &[usize]) -> CsrMatrix<f64> {
    let n = a.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    let ess_set: std::collections::HashSet<usize> = ess.iter().copied().collect();
    for i in 0..n {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[p] as usize;
            if ess_set.contains(&i) || ess_set.contains(&j) {
                continue;
            }
            coo.add(i, j, a.values[p]);
        }
    }
    for &d in ess {
        coo.add(d, d, 1.0);
    }
    coo.into_csr()
}

/// Scale a CSR matrix by `s` (structure preserved).
fn scale_csr(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values: a.values.iter().map(|&v| s * v).collect(),
    }
}

/// `A⁻¹·b` by sparse Cholesky (exact to machine precision; M is SPD).
fn cholesky_solve(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
    solve_sparse_cholesky(a, b).expect("Cholesky solve failed")
}

/// The evolution operator of the C++ `AdvDiffSUNDIALS` (serial subset):
/// `y' = M⁻¹K y` forward and the abstract costate
/// `yB' = −(M⁻¹K)ᵀ yB = K_adj·(M⁻¹yB)` backward (with `K_adj = −Kᵀ =
/// +p1·Δ + p2·C`, using `Cᵀ = −C` on the interior), plus the exact discrete
/// parameter-sensitivity quadrature rate.
struct AdvDiff {
    /// `M⁻¹·K` (essential BCs eliminated) — forward rate.
    minv_k: CsrMatrix<f64>,
    /// `K_adj·M⁻¹` — backward costate rate `−(M⁻¹K)ᵀ`.
    k_adj_minv: CsrMatrix<f64>,
    /// The eliminated mass matrix `M` (for `z = M⁻¹·yB` in the quadrature rate).
    m_elim: CsrMatrix<f64>,
    /// Eliminated `−Δ` and `C` (`∂K/∂p1`, `∂K/∂p2`).
    dk1: CsrMatrix<f64>,
    dk2: CsrMatrix<f64>,
}

impl TimeDependentAdjointOperator for AdvDiff {
    fn dim(&self) -> usize {
        self.minv_k.nrows
    }

    // AdvDiff rate equation: y' = M⁻¹K y
    fn mult(&self, _t: f64, y: &[f64], ydot: &mut [f64]) {
        self.minv_k.spmv(y, ydot);
    }

    fn jac(&self, _t: f64, _y: &[f64]) -> CsrMatrix<f64> {
        self.minv_k.clone()
    }

    // AdvDiff adjoint rate equation (abstract costate for y' = M⁻¹K y):
    // yB' = −(M⁻¹K)ᵀ yB = (K_adj·M⁻¹)·yB  (the M⁻¹ is inside k_adj_minv).
    fn adjoint_rate_mult(&self, _t: f64, _y: &[f64], yb: &[f64], ybdot: &mut [f64]) {
        self.k_adj_minv.spmv(yb, ybdot);
    }

    fn adjoint_jac(&self, _t: f64, _y: &[f64], _yb: &[f64]) -> CsrMatrix<f64> {
        self.k_adj_minv.clone()
    }

    // AdvDiff quadrature sensitivity rate equation: the exact discrete
    // gradient rate qBdot_j = (∂f/∂p_j)ᵀ·yB = zᵀ·(∂K/∂p_j)·y with
    // z = M⁻¹·yB (see the module docs for the relation to the C++ version).
    fn quadrature_sensitivity_mult(&self, _t: f64, y: &[f64], yb: &[f64], qbdot: &mut [f64]) {
        let n = y.len();
        let z = cholesky_solve(&self.m_elim, yb);
        let mut b1 = vec![0.0; n];
        let mut b2 = vec![0.0; n];
        self.dk1.spmv(y, &mut b1);
        self.dk2.spmv(y, &mut b2);
        qbdot[0] = b1.iter().zip(&z).map(|(&a, &b)| a * b).sum::<f64>(); // zᵀ(−Δ)y
        qbdot[1] = b2.iter().zip(&z).map(|(&a, &b)| a * b).sum::<f64>(); // zᵀ(C)y
    }
}

/// Initial condition `u_init(x) = x(2-x)e^{2x}`.
fn u_init(x: f64) -> f64 {
    x * (2.0 - x) * (2.0 * x).exp()
}

/// Build the serial-subset discrete problem (P1 H1 FEM on `mx+1` uniform
/// elements of [0, 2]; ess BCs at both ends) and return the operator plus
/// mesh data. The element matrices are exactly the MFEM integrator results
/// for P1 segments (see module docs).
fn build(mx: usize, p: [f64; 2]) -> (AdvDiff, Vec<usize>, Vec<f64>, CsrMatrix<f64>) {
    let n_elem = mx + 1;
    let h = 2.0 / n_elem as f64;
    let n_dof = mx + 2;
    let ess: Vec<usize> = vec![0, n_dof - 1];

    // P1 element matrices (uniform h):
    //   Mass_e       = (h/6)·[[2,1],[1,2]]
    //   Lap_e        = (1/h)·[[1,−1],[−1,1]]
    //   Conv_e(β)    = (β/2)·[[−1,+1],[−1,+1]]   (ConvectionIntegrator)
    let mut m_coo = CooMatrix::<f64>::new(n_dof, n_dof);
    let mut lap_coo = CooMatrix::<f64>::new(n_dof, n_dof);
    let mut conv_coo = CooMatrix::<f64>::new(n_dof, n_dof);
    for e in 0..n_elem {
        let i = e;
        let j = e + 1;
        let m_e = [
            [(2.0 * h) / 6.0, h / 6.0],
            [h / 6.0, (2.0 * h) / 6.0],
        ];
        let l_e = [
            [1.0 / h, -1.0 / h],
            [-1.0 / h, 1.0 / h],
        ];
        let c_e = [
            [-p[1] / 2.0, p[1] / 2.0],
            [-p[1] / 2.0, p[1] / 2.0],
        ];
        for (li, &gi) in [i, j].iter().enumerate() {
            for (lj, &gj) in [i, j].iter().enumerate() {
                m_coo.add(gi, gj, m_e[li][lj]);
                lap_coo.add(gi, gj, l_e[li][lj]);
                conv_coo.add(gi, gj, c_e[li][lj]);
            }
        }
    }
    let m = m_coo.into_csr();
    let lap = lap_coo.into_csr();
    let conv = conv_coo.into_csr();

    // K = Diffusion(−p1) + Convection(p2); K_adj = Diffusion(+p1) + Convection(p2)
    let k_fwd = {
        let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
        for i in 0..n_dof {
            for pa in lap.row_ptr[i]..lap.row_ptr[i + 1] {
                let j = lap.col_idx[pa] as usize;
                coo.add(i, j, -p[0] * lap.values[pa]);
            }
            for ca in conv.row_ptr[i]..conv.row_ptr[i + 1] {
                let j = conv.col_idx[ca] as usize;
                coo.add(i, j, conv.values[ca]);
            }
        }
        coo.into_csr()
    };
    let k_adj = {
        let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
        for i in 0..n_dof {
            for pa in lap.row_ptr[i]..lap.row_ptr[i + 1] {
                let j = lap.col_idx[pa] as usize;
                coo.add(i, j, p[0] * lap.values[pa]);
            }
            for ca in conv.row_ptr[i]..conv.row_ptr[i + 1] {
                let j = conv.col_idx[ca] as usize;
                coo.add(i, j, conv.values[ca]);
            }
        }
        coo.into_csr()
    };

    // Essential BC elimination (MFEM EliminateRowsCols: rows/cols zeroed, diag 1).
    let m_elim = eliminate_ess(&m, &ess);
    let k_elim = eliminate_ess(&k_fwd, &ess);
    let k_adj_elim = eliminate_ess(&k_adj, &ess);
    let dk1_elim = eliminate_ess(&scale_csr(&lap, -1.0), &ess); // ∂K/∂p1 = −Δ
    // ∂K/∂p2 = C with UNIT coefficient (the `conv` matrix embeds p2).
    let conv_unit = {
        let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
        for e in 0..n_elem {
            let i = e;
            let j = e + 1;
            let c_e = [[-0.5, 0.5], [-0.5, 0.5]];
            for (li, &gi) in [i, j].iter().enumerate() {
                for (lj, &gj) in [i, j].iter().enumerate() {
                    coo.add(gi, gj, c_e[li][lj]);
                }
            }
        }
        coo.into_csr()
    };
    let dk2_elim = eliminate_ess(&conv_unit, &ess); // ∂K/∂p2 = C (coef 1)

    // Materialize M⁻¹·(each operator) — the kernel solves y' = f(y) in
    // identity form, so ∂f/∂y = M⁻¹K must be explicit. Direct Cholesky
    // solves, exact to machine precision.
    let apply_minv = |a: &CsrMatrix<f64>| -> CsrMatrix<f64> {
        let mut cols: Vec<(usize, usize, f64)> = Vec::new();
        for j in 0..n_dof {
            // COLUMN j of a (CSR column extraction — row_ptr[j] is a ROW;
            // using it here would silently build the transpose operator).
            let mut rhs = vec![0.0; n_dof];
            for i in 0..n_dof {
                for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                    if a.col_idx[p] as usize == j {
                        rhs[i] = a.values[p];
                    }
                }
            }
            let x = cholesky_solve(&m_elim, &rhs);
            for (i, &v) in x.iter().enumerate() {
                if v != 0.0 {
                    cols.push((i, j, v));
                }
            }
        }
        triplets(n_dof, &cols)
    };
    let minv_k = apply_minv(&k_elim);
    // K_adj·M⁻¹ = K_adj·(M⁻¹): explicit M⁻¹ (column-wise solves) times K_adj.
    let identity = triplets(
        n_dof,
        &(0..n_dof).map(|i| (i, i, 1.0_f64)).collect::<Vec<_>>(),
    );
    let minv_m = apply_minv(&identity);
    let k_adj_minv = k_adj_elim.multiply(&minv_m);

    // dof coordinates.
    let coords: Vec<f64> = (0..n_dof).map(|i| i as f64 * h).collect();

    let adv = AdvDiff {
        minv_k,
        k_adj_minv,
        m_elim,
        dk1: dk1_elim,
        dk2: dk2_elim,
    };
    (adv, ess, coords, m)
}

/// Goal functional `g = obj·u`, `obj_i = ∫φ_i dx` (FE test-function integrals).
fn goal(coords: &[f64], u: &[f64]) -> f64 {
    // ∫φ_i dx = h for interior dofs, h/2 at the two ends (uniform mesh).
    let n = u.len();
    let h = coords[1] - coords[0];
    let mut g = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        let w = if i == 0 || i == n - 1 { h / 2.0 } else { h };
        g += w * ui;
    }
    g
}

/// One forward solve returning the goal `g = ∫u dx` at `t_final`.
fn forward_goal(
    adv: &AdvDiff,
    ess: &[usize],
    coords: &[f64],
    p: [f64; 2],
    dt: f64,
    tf: f64,
    rtol: f64,
    atol: f64,
) -> f64 {
    let n = coords.len();
    let cfg = AdjointConfig {
        forward: {
            let mut c = AdjointBdfConfig::new(AdjointTolerances::scalar(rtol, atol, n), dt);
            c.dt_max = dt;
            c
        },
        adjoint: AdjointBdfConfig::new(AdjointTolerances::scalar(rtol, atol, n), dt),
        steps_per_checkpoint: 50,
        interpolation: Interpolation::Hermite,
        store_all_forward_steps: false,
    };
    let _ = ess;
    let mut solver = AdjointSolver::new(adv, cfg);
    let mut u0: Vec<f64> = coords.iter().map(|&x| u_init(x)).collect();
    for &d in ess {
        u0[d] = 0.0;
    }
    solver.run_forward(0.0, tf, &u0, &[]).unwrap();
    let _ = p;
    goal(coords, solver.forward_final())
}

fn main() {
    // Parse command-line options (C++ OptionsParser analog; serial subset).
    let mut mx: usize = 20;
    let mut ser_ref_levels: usize = 0;
    let mut t_final: f64 = 2.5;
    let mut dt: f64 = 0.01;
    let mut fd_check = false;
    let mut reltol_override: Option<f64> = None;
    let mut atol_override: Option<f64> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "-m" | "--mx" => mx = val().parse().unwrap_or(mx),
            "-r" | "--refine" => ser_ref_levels = val().parse().unwrap_or(ser_ref_levels),
            "-tf" | "--t-final" => t_final = val().parse().unwrap_or(t_final),
            "-dt" | "--time-step" => dt = val().parse().unwrap_or(dt),
            "-fd" | "--fd-check" => fd_check = val().parse().unwrap_or(0) != 0,
            "-rtol" | "--rel-tol" => reltol_override = val().parse().ok(),
            "-atol" | "--abs-tol" => atol_override = val().parse().ok(),
            _ => {}
        }
    }
    for _ in 0..ser_ref_levels {
        mx *= 2;
    }

    let precision = 8;
    println!("Options:");
    println!("   -m: {mx}");
    println!("   -r: {ser_ref_levels}");
    println!("   -tf: {}", fmt_g_prec(t_final, precision));
    println!("   -dt: {}", fmt_g_prec(dt, precision));

    // Material properties (fixed): p = (p1, p2) = (1.0, 0.5).
    let p = [1.0_f64, 0.5];

    let (adv, ess, coords, m) = build(mx, p);
    let n_dof = coords.len();

    // Relative and absolute tolerances (C++ SetSStolerances 1e-8, 1e-6 for
    // both the forward and the adjoint problem).
    let reltol = reltol_override.unwrap_or(1e-8);
    let abstol = atol_override.unwrap_or(1e-6);

    println!("Number of unknowns: {}", n_dof - ess.len());

    // Set U with the initial conditions.
    let mut u: Vec<f64> = coords.iter().map(|&x| u_init(x)).collect();
    for &d in &ess {
        u[d] = 0.0;
    }
    println!("Init u:");
    print_vec(&u, 8);

    // Initialize adjoint problem settings: InitAdjointSolve(50, CV_HERMITE).
    let cfg = AdjointConfig {
        forward: {
            let mut c = AdjointBdfConfig::new(AdjointTolerances::scalar(reltol, abstol, n_dof), dt);
            c.dt_max = dt;
            c
        },
        adjoint: {
            let mut c = AdjointBdfConfig::new(AdjointTolerances::scalar(reltol, abstol, n_dof), dt);
            c.dt_max = t_final; // StepB with dt_real = max(dt, t): one call to 0
            c
        },
        steps_per_checkpoint: 50,
        interpolation: Interpolation::Hermite,
        store_all_forward_steps: false,
    };
    let mut cvodes = AdjointSolver::new(&adv, cfg);

    // Forward time-integration.
    let t = 0.0_f64;
    cvodes.run_forward(t, t_final, &u, &[]).expect("forward solve failed");
    u.copy_from_slice(cvodes.forward_final());

    println!(
        "BDF statistics (PrintInfo analog): {}",
        cvodes.forward_stats.summary()
    );
    println!("Final Solution: {}", fmt_g_prec(t_final, precision));
    println!("u (0):");
    print_vec(&u, 8);

    // Goal g = ∫u dx (DomainLFIntegrator(one) · u).
    let g = goal(&coords, &u);
    println!("g: {}", fmt_g_prec(g, precision));

    // Solve the adjoint problem. Abstract-costate terminal value:
    // λ(T) = ∂g/∂y(T) = the FE goal weights obj_i = ∫φ_i dx
    // (= h interior, h/2 at the ends; 0 on the eliminated dofs — for the
    // C++'s FE-function adjoint this corresponds to v = 1 via M·1 = h·1).
    let h_mesh = coords[1] - coords[0];
    let mut v: Vec<f64> = vec![0.0; n_dof];
    for i in 0..n_dof {
        if !ess.contains(&i) {
            v[i] = h_mesh;
        }
    }
    let mut q_bdot = vec![0.0_f64; 2];

    cvodes.init_adjoint(&v, &q_bdot);
    cvodes.step_adjoint_to(0.0).expect("adjoint solve failed");
    v.copy_from_slice(cvodes.adjoint());
    q_bdot.copy_from_slice(cvodes.quad_sensitivity());

    println!("t: {}", fmt_g_prec(cvodes.adjoint_time(), precision));
    println!("v (0):");
    print_vec(&v, 8);
    println!("sensitivity:");
    print_vec(&q_bdot, 8);
    println!(
        "Adjoint BDF statistics (PrintInfo analog): {}",
        cvodes.adjoint_stats.summary()
    );
    println!("Checkpoint re-integration steps: {}", cvodes.n_reint_steps());
    let _ = m;

    // Self-consistency check: adjoint gradient vs central finite differences
    // of the discrete goal g(p) (fem-rs addition; off by default to keep the
    // printed output 1:1 with the C++).
    if fd_check {
        let eps1 = 1e-5 * p[0];
        let eps2 = 1e-5 * p[1];
        let (adv1p, ess, coords, _) = build(mx, [p[0] + eps1, p[1]]);
        let g1p = forward_goal(&adv1p, &ess, &coords, p, dt, t_final, 1e-10, 1e-12);
        let (adv1m, ess, coords, _) = build(mx, [p[0] - eps1, p[1]]);
        let g1m = forward_goal(&adv1m, &ess, &coords, p, dt, t_final, 1e-10, 1e-12);
        let (adv2p, ess, coords, _) = build(mx, [p[0], p[1] + eps2]);
        let g2p = forward_goal(&adv2p, &ess, &coords, p, dt, t_final, 1e-10, 1e-12);
        let (adv2m, ess, coords, _) = build(mx, [p[0], p[1] - eps2]);
        let g2m = forward_goal(&adv2m, &ess, &coords, p, dt, t_final, 1e-10, 1e-12);
        let fd1 = (g1p - g1m) / (2.0 * eps1);
        let fd2 = (g2p - g2m) / (2.0 * eps2);
        // Matching adjoint gradient at the FD tolerances.
        let (adv_t, ess_t, coords_t, _) = build(mx, p);
        let cfg_t = AdjointConfig {
            forward: {
                let mut c =
                    AdjointBdfConfig::new(AdjointTolerances::scalar(1e-10, 1e-12, coords_t.len()), dt);
                c.dt_max = dt;
                c
            },
            adjoint: {
                let mut c =
                    AdjointBdfConfig::new(AdjointTolerances::scalar(1e-10, 1e-12, coords_t.len()), dt);
                c.dt_max = t_final;
                c
            },
            steps_per_checkpoint: 50,
            interpolation: Interpolation::Hermite,
            store_all_forward_steps: false,
        };
        let mut solver = AdjointSolver::new(&adv_t, cfg_t);
        let mut u0: Vec<f64> = coords_t.iter().map(|&x| u_init(x)).collect();
        for &d in &ess_t {
            u0[d] = 0.0;
        }
        solver.run_forward(0.0, t_final, &u0, &[]).unwrap();
        solver.init_adjoint(&{
            // λ(T) = ∂g/∂y(T) = h interior (goal ∫u dx; 0 on eliminated dofs).
            let h_mesh = coords_t[1] - coords_t[0];
            let mut v0 = vec![0.0; coords_t.len()];
            for i in 0..coords_t.len() {
                if !ess_t.contains(&i) {
                    v0[i] = h_mesh;
                }
            }
            v0
        }, &[0.0, 0.0]);
        solver.step_adjoint_to(0.0).unwrap();
        let adj1 = solver.quad_sensitivity()[0];
        let adj2 = solver.quad_sensitivity()[1];
        println!("FD check (rtol 1e-10):");
        println!(
            "   dG/dp1: adjoint = {}, FD = {}, rel diff = {:.2e}",
            fmt_g_prec(adj1, precision),
            fmt_g_prec(fd1, precision),
            ((adj1 - fd1) / fd1).abs()
        );
        println!(
            "   dG/dp2: adjoint = {}, FD = {}, rel diff = {:.2e}",
            fmt_g_prec(adj2, precision),
            fmt_g_prec(fd2, precision),
            ((adj2 - fd2) / fd2).abs()
        );
        assert!(
            ((adj1 - fd1) / fd1).abs() < 1e-6 && ((adj2 - fd2) / fd2).abs() < 1e-6,
            "FD self-consistency check failed (<1e-6 required)"
        );
        println!("   PASS (< 1e-6)");
    }
}
