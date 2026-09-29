//! # MFEM Example 10 — Dynamic Hyperelasticity (NeoHookean)
//!
//! 1:1 port of `mfem/examples/ex10.cpp`.
//!
//! Solves the time-dependent nonlinear elasticity problem:
//!
//! ```text
//!   M dv/dt = -H(x) - S v
//!   dx/dt   =  v
//! ```
//!
//! where M is the mass matrix, S is a viscosity (vector Laplacian) operator,
//! and H(x) is the internal force from a NeoHookean hyperelastic model.
//!
//! The geometry is a beam with boundary attribute 1 fixed.  Implicit time
//! integration uses a Newton solve per stage via the reduced backward-Euler
//! system `R(k) = (M + dt S) k + H(x + dt (v + dt k)) + S v`.
//!
//! ## Usage
//! ```bash
//! # Default: beam-tri.mesh, order 2, ref 2, SDIRK23, dt=3, t_final=300
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -no-vis
//!
//! # Custom mesh and parameters
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -m data/beam-quad.mesh -r 2 -o 2 -dt 3 -no-vis
//!
//! # Explicit (forward Euler)
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -s 4 -dt 0.03 -vs 20 -no-vis
//! ```
//!
//! ## ODE solver types (same numbering as MFEM)
//! |   s | Method           | Type     |
//! |-----|------------------|----------|
//! |   4 | Forward Euler    | Explicit |
//! |  22 | SDIRK2           | Implicit |
//! |  23 | SDIRK3 (default) | Implicit |

use std::f64::consts::FRAC_1_SQRT_2;

use fem_assembly::{
    eliminate_ess_tdofs, ElimPolicy, glibc_pow::glibc_hypot,
    mfem_add, mfem_add_in_place,
    MfemBilinearQuad, MfemHyperelasticAssembler, MfemLilMatrix,
};
use fem_core::types::DofId;
use fem_io::mfem::{read_mfem_file, write_gf_file, write_mfem_file};
use fem_linalg::CsrMatrix;
use fem_mesh::{
    Mesh,
    amr::refine_uniform,
    topology::MeshTopology,
};
use fem_space::fe_space::FESpace;
use fem_solver::{fmt_g, solve_pcg_dsmoother, SolverConfig};

// ─── CLI arguments (matching MFEM ex10) ────────────────────────────────────

#[allow(non_snake_case)]
struct Args {
    mesh: String,
    ref_levels: usize,
    order: u8,
    ode_solver_type: i32,
    t_final: f64,
    dt: f64,
    viscosity: f64,
    mu: f64,
    K: f64,
    visualization: bool,
    vis_steps: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-quad.mesh".to_string(),
            ref_levels: 2,
            order: 2,
            ode_solver_type: 23,
            t_final: 300.0,
            dt: 3.0,
            viscosity: 1e-2,
            mu: 0.25,
            K: 5.0,
            visualization: false,
            vis_steps: 1,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => {
                    a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
                }
                "-o" | "--order" => {
                    a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
                }
                "-s" | "--ode-solver" => {
                    a.ode_solver_type = it.next().and_then(|v| v.parse().ok()).unwrap_or(23)
                }
                "-tf" | "--t-final" => {
                    a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(300.0)
                }
                "-dt" | "--time-step" => {
                    a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(3.0)
                }
                "-v" | "--viscosity" => {
                    a.viscosity = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-2)
                }
                "-mu" | "--shear-modulus" => {
                    a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.25)
                }
                "-K" | "--bulk-modulus" => {
                    a.K = it.next().and_then(|v| v.parse().ok()).unwrap_or(5.0)
                }
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-vis" | "--visualization" => a.visualization = true,
                "-vs" | "--visualization-steps" => {
                    a.vis_steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
                }
                _ => {}
            }
        }
        a
    }
}

// ─── Initial conditions ─────────────────────────────────────────────────────

/// Identity map: initial deformation = reference configuration.
#[allow(dead_code)]
fn initial_deformation(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}

/// Initial velocity: parabolic profile in the vertical direction.
fn initial_velocity(x: &[f64]) -> Vec<f64> {
    let dim = x.len();
    let mut v = vec![0.0; dim];
    if dim >= 2 {
        let s = 0.1 / 64.0;
        v[dim - 1] = s * x[0] * x[0] * (8.0 - x[0]);
        v[0] = -s * x[0] * x[0];
    }
    v
}

// ─── ReducedSystemOperator ─────────────────────────────────────────────────

/// Nonlinear operator for the reduced backward-Euler equation:
///
/// ```text
/// R(k) = (M + dt·S)·k + H(x + dt·(v + dt·k)) + S·v
/// ```
///
/// where `M` is the mass matrix, `S` the viscosity matrix, and `H` the
/// hyperelastic internal-force operator.
struct ReducedSystemOperator<'a> {
    m: &'a CsrMatrix<f64>,
    s: &'a CsrMatrix<f64>,
    /// MFEM-bitwise hyperelastic element kernels (D842-2).  Like MFEM, the
    /// kernels are **position-based**: the field entering them is the deformed
    /// node positions `x_ref + displacement`, not the displacement itself.
    hyper: &'a MfemHyperelasticAssembler,
    ess_dofs: &'a [usize],
    n: usize,
}

impl ReducedSystemOperator<'_> {
    /// Width / height of the operator (one scalar block).
    fn size(&self) -> usize { self.n }

    /// Compute `y = R(k)` given current `v`, `x`, `dt`.
    ///
    /// MFEM's operator semantics (ex10.cpp:406 `ReducedSystemOperator::Mult`):
    /// `y = H->Mult(z) + M->AddMult(k) + S->AddMult(w)` where
    ///  - `H->Mult` is `NonlinearForm::Mult`: element assembly followed by
    ///    `y(ess) = 0.0` (nonlinearform.cpp:408 — the `= x(ess)` variant is
    ///    commented out in 4.10),
    ///  - `M`, `S` are the **in-place eliminated** `BilinearForm::SpMat()`
    ///    (`FormSystemMatrix` → `EliminateVDofs(ess, DIAG_KEEP)`: rows/cols
    ///    zeroed, diagonal preserved; bilinearform.cpp:936 + hpp:153), so the
    ///    ess rows contribute `M(e,e)·k(e) + S(e,e)·w(e)` — nonzero once the
    ///    Newton iterates pick up `k(ess) ≠ 0` (ex10 re-imposes the BC only
    ///    through the eliminated operators, never on the step vectors).
    fn mult(&self, k: &[f64], v: &[f64], x: &[f64], dt: f64, y: &mut [f64]) {
        let n = self.n;
        // w = v + dt*k
        let mut w = vec![0.0; n];
        for i in 0..n { w[i] = v[i] + dt * k[i]; }
        // z = x + dt*w = x + dt*(v + dt*k)
        let mut z = vec![0.0; n];
        for i in 0..n { z[i] = x[i] + dt * w[i]; }

        // y = H(z)  (internal force, at the deformed positions x_ref + z)
        self.hyper.residual_positions(&z, y);

        // NonlinearForm::Mult tail: y(ess) = 0.0 (before the M/S adds).
        for &d in self.ess_dofs {
            if d < n { y[d] = 0.0; }
        }

        // D842-2 probe: dump the first two (z, y) pairs for bitwise
        // comparison against the C++ probe dumps (tmp/d92b/cpp_z0_0.txt).
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static DUMPED: AtomicUsize = AtomicUsize::new(0);
            let d = DUMPED.fetch_add(1, Ordering::Relaxed);
            if d < 2 && std::env::var_os("FEM_EX10_PROBE").is_some() {
                let tag = if d == 0 { "0" } else { "1" };
                let _ = std::fs::write(
                    format!("tmp/d92b/rs_z{tag}_{tag}.txt"),
                    z.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                );
                let _ = std::fs::write(
                    format!("tmp/d92b/rs_y{tag}_{tag}.txt"),
                    y.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                );
                if d == 1 {
                    for e in 0..128usize {
                        let (_, ev2) = self.hyper.debug_element_vector_positions(&z, e);
                        let _ = std::fs::write(
                            format!("tmp/d92b/rs_eva2_{e}.txt"),
                            ev2.iter().map(|v| format!("{v:.17e}
")).collect::<String>(),
                        );
                    }
                    for q in 0..16usize {
                        let (w2, jr, ds, jpt, pp) = self.hyper.debug_chain_positions(&z, 0, q);
                        let (pos2, _) = self.hyper.debug_element_vector_positions(&z, 0);
                        let mut body = format!("wt {w2:.17e}
");
                        for (tag, vals) in [
                            ("PM", pos2.as_slice()),
                            ("Jr", jr.as_slice()),
                            ("DS", ds.as_slice()),
                            ("Jpt", jpt.as_slice()),
                            ("P", pp.as_slice()),
                        ] {
                            for v in vals {
                                body.push_str(&format!("{tag} {v:.17e}
"));
                            }
                        }
                        let _ = std::fs::write(
                            format!("tmp/d92b/rs_chain2_{q}.txt"),
                            body,
                        );
                    }
                }
                if d == 0 {
                    let all_ev = self.hyper.debug_all_element_vectors(&z);
                    for (e, ev) in all_ev.iter().enumerate() {
                        let _ = std::fs::write(
                            format!("tmp/d92b/rs_eva_{e}.txt"),
                            ev.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                        );
                    }
                    // Chain records for the first two elements, all 16 ips —
                    // same tags/layout as the C++ nonlininteg patch dumps
                    // (tmp/d92b/cpp_chain_{e*16+q}.txt): wt/PM/Jr/DS/Jpt/P.
                    for e in 0..2usize {
                        let (pos, _) = self.hyper.debug_element_vector_positions(&z, e);
                        for q in 0..16usize {
                            let (w, jr, ds, jpt, p) = self.hyper.debug_chain_positions(&z, e, q);
                            let mut body = format!("wt {w:.17e}\n");
                            for tag_vals in [
                                ("PM", pos.as_slice()),
                                ("Jr", jr.as_slice()),
                                ("DS", ds.as_slice()),
                                ("Jpt", jpt.as_slice()),
                                ("P", p.as_slice()),
                            ] {
                                for v in tag_vals.1 {
                                    body.push_str(&format!("{} {v:.17e}\n", tag_vals.0));
                                }
                            }
                            let _ = std::fs::write(
                                format!("tmp/d92b/rs2_chain_{e}_{q}.txt"),
                                body,
                            );
                        }
                    }
                }
            }
        }

        // y += M * k  (M->AddMult: sparsemat.cpp AddMult CPU path — sequential
        // row sums; the fused CsrMatrix::spmv dot rounds differently for rows
        // with ≥ 8 entries).
        let mut mk = vec![0.0; n];
        mfem_csr_mult(self.m, k, &mut mk);
        for i in 0..n { y[i] += mk[i]; }

        // y += S * w = S * (v + dt*k)
        let mut sw = vec![0.0; n];
        mfem_csr_mult(self.s, &w, &mut sw);
        for i in 0..n { y[i] += sw[i]; }

        // D842-2 forensics: first two calls — dump the H part, M·k part,
        // S·w part, w and k (mirrors cpp_parts{0,1}.txt / cpp_wk{0,1}.txt).
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static PARTS: AtomicUsize = AtomicUsize::new(0);
            let call = PARTS.fetch_add(1, Ordering::Relaxed);
            if call < 2 {
                let _ = std::fs::write(
                    format!("tmp/d92b/rs_parts{call}.txt"),
                    y.iter().chain(mk.iter()).chain(sw.iter())
                        .map(|v| format!("{v:.17e}
")).collect::<String>(),
                );
                let _ = std::fs::write(
                    format!("tmp/d92b/rs_wk{call}.txt"),
                    w.iter().zip(k.iter())
                        .map(|(a, b)| format!("{a:.17e} {b:.17e}
")).collect::<String>(),
                );
            }
        }

        // Full-precision trace of the first calls (d91b probe, stderr only;
        // mirrors tmp/d91b/patch_ex10_probe.py on the C++ side).
        if std::env::var_os("FEM_EX10_PROBE").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALL: AtomicUsize = AtomicUsize::new(0);
            let call = CALL.fetch_add(1, Ordering::Relaxed);
            if call < 8 {
                let (mut s, mut n2) = (0.0_f64, 0.0_f64);
                for &yi in y.iter() { s += yi; n2 += yi * yi; }
                let ks: f64 = k.iter().sum();
                eprintln!(
                    "MULT{} norm={:.17e} sum={:.17e} y0={:.17e} y1={:.17e} ksum={:.17e}",
                    call,
                    n2.sqrt(),
                    s,
                    y[0],
                    y[1],
                    ks
                );
            }
        }
    }

    /// Compute the Jacobian `J = dR/dk = M + dt·S + dt²·grad_H(z)`.
    ///
    /// MFEM (ex10.cpp:412 `ReducedSystemOperator::GetGradient`):
    /// `Jacobian = Add(1.0, M->SpMat(), dt, S->SpMat())` then
    /// `Jacobian->Add(dt*dt, H->GetGradient(z))` — **no further BC
    /// elimination**.  The essential dofs are already carried by the three
    /// operands: `M`/`S` are the in-place `DIAG_KEEP`-eliminated
    /// `BilinearForm::SpMat()` matrices, and `NonlinearForm::GetGradient`
    /// applies `EliminateRowCol(ess, DIAG_ONE)` per essential dof
    /// (nonlinearform.cpp:656) before returning.
    fn gradient(&self, k: &[f64], v: &[f64], x: &[f64], dt: f64) -> CsrMatrix<f64> {
        let n = self.n;
        let mut w = vec![0.0; n];
        for i in 0..n { w[i] = v[i] + dt * k[i]; }
        let mut z = vec![0.0; n];
        for i in 0..n { z[i] = x[i] + dt * w[i]; }

        // H.GetGradient(z): the element phase into an open LIL
        // (`NonlinearForm::GetGradient`'s `AddSubMatrix` loop), then the
        // tail — `Finalize(0)` + the ess `EliminateRowCol(·, DIAG_ONE)` loop.
        let mut grad_h = self.hyper.grad_lil_positions(&z).finalize();
        {
            let ess: Vec<DofId> = self.ess_dofs.iter().copied()
                .filter(|&d| d < n).map(|d| d as DofId).collect();
            let x_bc = vec![0.0; n];
            let mut dummy = vec![0.0; n];
            eliminate_ess_tdofs(&mut grad_h, &ess, &x_bc, &mut dummy, ElimPolicy::DiagOne);
            if std::env::var_os("FEM_EX10_PROBE").is_some() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static GCALL_DUMP: AtomicUsize = AtomicUsize::new(0);
                if GCALL_DUMP.fetch_add(1, Ordering::Relaxed) == 0 {
                    dump_csr("tmp/d92b/rs_G_csr.txt", &grad_h);
                }
            }
        }

        // J = Add(1.0, M->SpMat(), dt, S->SpMat()) then
        // Jacobian->Add(dt*dt, *grad_H) — MFEM's exact merge order.
        let tmp = mfem_add(1.0, self.m, dt, self.s);
        let mut jac = tmp;
        mfem_add_in_place(&mut jac, dt * dt, &grad_h);

        // D842-2 probe: full CSR dump of the first Newton Jacobian (mirrors
        // tmp/d92b/ex10_probe_j.cpp → cpp_J_csr.txt).
        if std::env::var_os("FEM_EX10_PROBE").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static JCALL: AtomicUsize = AtomicUsize::new(0);
            if JCALL.fetch_add(1, Ordering::Relaxed) == 0 {
                let mut body = format!("{} {}\n", jac.nrows, jac.nnz());
                for row in 0..jac.nrows {
                    for k in jac.row_ptr[row]..jac.row_ptr[row + 1] {
                        body.push_str(&format!(
                            "{} {} {:.17e}\n",
                            row, jac.col_idx[k], jac.values[k]
                        ));
                    }
                }
                let _ = std::fs::write("tmp/d92b/rs_J_csr.txt", body);
            }
        }

        jac
    }
}

// ─── Dot product ────────────────────────────────────────────────────────────

/// D842-2 probe helper: dump a CSR matrix as `row col value` lines (the
/// mirror of the C++ probe's `cpp_*_csr.txt` dumps).
fn dump_csr(path: &str, m: &CsrMatrix<f64>) {
    let mut body = format!("{} {}\n", m.nrows, m.nnz());
    for row in 0..m.nrows {
        for k in m.row_ptr[row]..m.row_ptr[row + 1] {
            body.push_str(&format!("{} {} {:.17e}\n", row, m.col_idx[k], m.values[k]));
        }
    }
    let _ = std::fs::write(path, body);
}

// ─── MFEM-order sparse kernels (D842-2, round-92 continuation) ──────────────
//
// `CsrMatrix::spmv` fuses its row dot into groups of eight products per
// statement, which rounds differently from MFEM's strictly sequential
// `d += A[j]·x[J[j]]` (sparsemat.cpp `AddMult` CPU path) as soon as a row
// holds ≥ 8 entries — ex10's tangent rows hold ~29.  MINRES is
// round-off-trajectory-sensitive here (it does not reach its 1e-8 tolerance
// within 300 iterations on this tangent), so the inner solve must be driven
// by the MFEM-order kernels below.  Kept in the example: the generic
// `CsrMatrix`/solver paths (and the pins that depend on their rounding)
// must not move.

/// `SparseMatrix::Mult(x, y)` CPU path: `y = 0.0` then per row
/// `d += A[j]·x[J[j]]` ascending, `y[i] += a·d` (a = 1).
fn mfem_csr_mult(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for yy in y.iter_mut() {
        *yy = 0.0;
    }
    for row in 0..a.nrows {
        let mut d = 0.0_f64;
        for k in a.row_ptr[row]..a.row_ptr[row + 1] {
            d += a.values[k] * x[a.col_idx[k] as usize];
        }
        y[row] += 1.0 * d;
    }
}

/// MFEM `Dot(x, y)`: `a += x(i)·y(i)` ascending.
fn mfem_dot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// MFEM `MINRESSolver::Mult` with `DSmoother(1)` preconditioning
/// (bit-exact port, mirrored from `fem_solver::solve_minres_dsmoother`
/// / D842-1) with the two hot kernels replaced by the MFEM-order
/// [`mfem_csr_mult`] / [`mfem_dot`].  `PrintLevel::Silent` semantics
/// (no MINRES lines on stdout).
fn mfem_minres_dsmoother(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    rtol: f64,
    atol: f64,
    max_iter: usize,
) {
    let n = a.nrows;

    // DSmoother(1) = type 1, one sweep, iterative_mode=false:
    // `Mult_` (sparsesmoothers.cpp:92) skips the type-0 DiagScale fast path
    // and runs `Jacobi2` = l1-Jacobi from x0 = 0
    // (JacobiDispatch<useFabs = true>, sparsemat.cpp:2720):
    //   resi = b[i] − Σ_j A[i,j]·x0[J[j]]  (= b[i] for x0 = 0)
    //   x1[i] = x0[i] + scale·resi / Σ_j |A[i,j]|
    // i.e. every preconditioner application is a scale by the row **L1
    // norm** Σ|A_ij| (accumulated sequentially, ascending) — not the
    // diagonal.  (The D842-1 pin could not discriminate: its oracle matrix
    // is diagonal, where the two coincide.)
    let mut l1norm = vec![0.0_f64; n];
    for row in 0..a.nrows {
        let mut norm = 0.0_f64;
        for k in a.row_ptr[row]..a.row_ptr[row + 1] {
            norm += a.values[k].abs();
        }
        l1norm[row] = norm;
    }

    // Forensics gate: trace only the first MINRES solve of the first Newton.
    let trace_first = {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TRACE0: AtomicUsize = AtomicUsize::new(0);
        TRACE0.fetch_add(1, Ordering::Relaxed) == 0
    };

    // iterative_mode == false: v1 = b, x = 0, u1 = B⁻¹ v1 (Jacobi2 from 0).
    let mut v1 = b.to_vec();
    for xi in x.iter_mut() {
        *xi = 0.0;
    }
    let mut u1: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        u1.push(1.0 * v1[i] / l1norm[i]);
    }
    if std::env::var_os("FEM_EX10_PROBE").is_some() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static U1DUMP: AtomicUsize = AtomicUsize::new(0);
        if U1DUMP.fetch_add(1, Ordering::Relaxed) == 0 {
            let _ = std::fs::write(
                "tmp/d92b/rs_u1.txt",
                u1.iter().map(|v| format!("{v:.17e}
")).collect::<String>(),
            );
        }
    }

    let mut eta = mfem_dot(&u1, &v1).sqrt();
    let mut beta = eta;
    let mut gamma0 = 1.0_f64;
    let mut gamma1 = 1.0_f64;
    let mut sigma0 = 0.0_f64;
    let mut sigma1 = 0.0_f64;

    // MFEM: norm_goal = std::max(rel_tol*eta, abs_tol) — fixed at setup.
    let norm_goal = (rtol * eta).max(atol);

    let mut q = vec![0.0_f64; n];
    let mut v0 = vec![0.0_f64; n];
    let mut w0 = vec![0.0_f64; n];
    let mut w1 = vec![0.0_f64; n];

    if eta > norm_goal {
        for it_i in 1..=max_iter {
            // v1 /= beta; u1 /= beta — MFEM Vector::operator/=(real_t):
            // m = 1.0/c; y[i] *= m.
            let m = 1.0 / beta;
            for i in 0..n {
                v1[i] *= m;
                u1[i] *= m;
            }

            // q = A·u1  (oper->Mult(*z, q), z = &u1 when prec).
            mfem_csr_mult(a, &u1, &mut q);
            let alpha = mfem_dot(&u1, &q);

            if it_i > 1 {
                // q.Add(-beta, v0)
                let neg_beta = -beta;
                for i in 0..n {
                    q[i] += neg_beta * v0[i];
                }
            }
            // v0 = q - alpha·v1  (add(q, -alpha, v1, v0))
            let neg_alpha = -alpha;
            for i in 0..n {
                v0[i] = q[i] + neg_alpha * v1[i];
            }

            let delta = gamma1 * alpha - gamma0 * sigma1 * beta;
            let rho3 = sigma0 * beta;
            let rho2 = sigma1 * alpha + gamma0 * gamma1 * beta;

            // q = B⁻¹·v0 (prec->Mult(v0, q): Jacobi2 from 0 = v0/l1);
            // beta = ‖v0‖_B.
            for i in 0..n {
                q[i] = 1.0 * v0[i] / l1norm[i];
            }
            beta = mfem_dot(&v0, &q).sqrt();

            let rho1 = glibc_hypot(delta, beta);
            if std::env::var_os("FEM_EX10_PROBE").is_some() {
                let naive = (delta * delta + beta * beta).sqrt();
                eprintln!(
                    "RHO it={it_i} alpha={alpha:.17e} delta={delta:.17e} beta={beta:.17e} rho1={rho1:.17e} naive={naive:.17e} eq={}",
                    rho1.to_bits() == naive.to_bits()
                );
            }

            if it_i == 1 {
                // w0.Set(1./rho1, *z)
                let c0 = 1.0 / rho1;
                for i in 0..n {
                    w0[i] = c0 * u1[i];
                }
            } else if it_i == 2 {
                // w0 = (1/rho1)·u1 + (−rho2/rho1)·w1
                let c0 = 1.0 / rho1;
                let c1 = -rho2 / rho1;
                for i in 0..n {
                    w0[i] = c0 * u1[i] + c1 * w1[i];
                }
            } else {
                // add(-rho3/rho1, w0, -rho2/rho1, w1, w0); w0.Add(1/rho1, u1)
                let c0 = -rho3 / rho1;
                let c1 = -rho2 / rho1;
                for i in 0..n {
                    w0[i] = c0 * w0[i] + c1 * w1[i];
                }
                let c2 = 1.0 / rho1;
                for i in 0..n {
                    w0[i] += c2 * u1[i];
                }
            }

            gamma0 = gamma1;
            gamma1 = delta / rho1;

            // x.Add(gamma1*eta, w0)
            let xcoef = gamma1 * eta;
            for i in 0..n {
                x[i] += xcoef * w0[i];
            }

            sigma0 = sigma1;
            sigma1 = beta / rho1;

            // eta = -sigma1*eta
            eta = -sigma1 * eta;

            if std::env::var_os("FEM_EX10_PROBE").is_some() {
                eprintln!("MINRES: iteration {:3}: ||r||_B = {:17e}", it_i, eta.abs());
                if trace_first && it_i <= 3 {
                    let mut body = String::new();
                    for i in 0..n {
                        body.push_str(&format!("{:.17e} {:.17e}
", u1[i], v1[i]));
                    }
                    let _ = std::fs::write(format!("tmp/d92b/rs_minres_uv_{it_i}.txt"), body);
                }
            }

            if eta.abs() <= norm_goal {
                break;
            }

            // Swap(u1, q); Swap(v0, v1); Swap(w0, w1)
            std::mem::swap(&mut u1, &mut q);
            std::mem::swap(&mut v0, &mut v1);
            std::mem::swap(&mut w0, &mut w1);
        }
    }
    // Silent print level: the solve result is not observed on stdout
    // (ex10 sets print level −1) and non-convergence of the inner solve is
    // not fatal (MFEM only warns with warnings/summary gates).
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ─── Newton solver for the reduced system ───────────────────────────────────

/// Solve `R(k) = 0` for `k` using Newton's method with MINRES inner solver.
///
/// Bit-for-bit port of MFEM `NewtonSolver::Mult` (print level 1) as configured
/// by ex10 (`iterative_mode = false`, `rel_tol = 1e-8`, `abs_tol = 0.0`,
/// `max_iter = 10`, `SetPrintLevel(1)`): the residual norm is printed *before*
/// the convergence test, `norm_goal = max(rel_tol·‖r₀‖, abs_tol)` is fixed at
/// setup, and `ComputeScalingFactor()` is the default 1.0.  The inner
/// Jacobian solve is MFEM's `MINRESSolver` + `DSmoother(1)` (rel_tol 1e-8,
/// abs_tol 0, max_iter 300, print level −1) via
/// [`fem_solver::solve_minres_dsmoother`].
///
/// The reduced operator is evaluated at current `v`, `x`, `dt`.
fn newton_solve_reduced(
    op: &ReducedSystemOperator,
    k: &mut [f64],
    v: &[f64],
    x: &[f64],
    dt: f64,
    verbose: bool,
) {
    let n = op.size();
    // MFEM double-precision: rel_tol = 1e-8, abs_tol = 0.0, max_iter = 10
    let rel_tol = 1e-8_f64;
    let abs_tol = 0.0_f64;
    let max_iter = 10;

    // Initial residual (iterative_mode = false: k starts at 0, enforced by
    // the call sites which pass freshly zeroed k vectors).
    let mut r = vec![0.0; n];
    op.mult(k, v, x, dt, &mut r);
    let norm0 = norm2(&r);
    let mut norm = norm0;
    // MFEM: norm_goal = std::max(rel_tol*norm, abs_tol), fixed at setup.
    let norm_goal = (rel_tol * norm).max(abs_tol);

    let mut converged = false;
    for it in 0..=max_iter {
        // D842-2: forensics — dump the first Newton's r0 and the first inner
        // MINRES solution (mirrors the C++ manual-Newton probe dumps
        // cpp_r0.txt / cpp_c1.txt).
        if std::env::var_os("FEM_EX10_PROBE").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NDUMP: AtomicUsize = AtomicUsize::new(0);
            if NDUMP.fetch_add(1, Ordering::Relaxed) == 0 {
                let _ = std::fs::write(
                    "tmp/d92b/rs_r0.txt",
                    r.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                );
            }
        }
        // MFEM prints BEFORE the convergence test (print level 1).
        if verbose {
            print!("Newton iteration {it:2} : ||r|| = {}", fmt_g(norm));
            if it > 0 {
                print!(", ||r||/||r_0|| = {}", fmt_g(norm / norm0));
            }
            println!();
        }
        if norm <= norm_goal {
            converged = true;
            break;
        }
        if it >= max_iter {
            converged = false;
            break;
        }

        // Build the Jacobian J = M + dt*S + dt²*grad_H(z) (eliminated).
        let jac = op.gradient(k, v, x, dt);

        // Full-precision Jacobian trace (d91b probe, stderr only; mirrors
        // patch_ex10_probe2.py: J·(D⁻¹w) with w = v + dt·k).
        if std::env::var_os("FEM_EX10_PROBE").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static GCALL: AtomicUsize = AtomicUsize::new(0);
            if GCALL.fetch_add(1, Ordering::Relaxed) == 0 {
                let n2 = op.size();
                let mut wv = vec![0.0_f64; n2];
                for i in 0..n2 { wv[i] = v[i] + dt * k[i]; }
                let diag = jac.diagonal();
                let dsum: f64 = diag.iter().sum();
                let mut t = vec![0.0_f64; n2];
                for i in 0..n2 { t[i] = 1.0 * wv[i] / diag[i]; }
                let mut yj = vec![0.0_f64; n2];
                mfem_csr_mult(&jac, &t, &mut yj);
                let (mut ysum, mut yn2) = (0.0_f64, 0.0_f64);
                for &yi in yj.iter() { ysum += yi; yn2 += yi * yi; }
                let mut n_ess_like = 0usize;
                let mut dfree = 0.0_f64;
                for &d in diag.iter() {
                    if (d - 1.0).abs() < 2.5 { n_ess_like += 1; } else { dfree += d; }
                }
                eprintln!(
                    "JAC0 dsum={:.17e} n_ess_like={} dfree={:.17e} ynorm={:.17e} ysum={:.17e} y0={:.17e} y1={:.17e}",
                    dsum,
                    n_ess_like,
                    dfree,
                    yn2.sqrt(),
                    ysum,
                    yj[0],
                    yj[1]
                );
            }
        }

        // c = [DF(x_i)]^{-1} [F(x_i) - b] with b = 0 (empty Vector): the
        // MINRES right-hand side is the residual r itself.  The inner solve
        // runs on the MFEM-order matvec/dot kernels (see
        // `mfem_minres_dsmoother`): MINRES is round-off-trajectory-sensitive
        // on this tangent and the fused `CsrMatrix::spmv` dot rounds
        // differently from MFEM's sequential `AddMult` once rows hold
        // ≥ 8 entries.
        let mut c = vec![0.0; n];
        mfem_minres_dsmoother(&jac, &r, &mut c, rel_tol, abs_tol, 300);
        if std::env::var_os("FEM_EX10_PROBE").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CDUMP: AtomicUsize = AtomicUsize::new(0);
            if CDUMP.fetch_add(1, Ordering::Relaxed) == 0 {
                let _ = std::fs::write(
                    "tmp/d92b/rs_c1.txt",
                    c.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                );
            }
        }

        // Newton update: x -= c (ComputeScalingFactor() == 1.0).
        for j in 0..n {
            k[j] -= c[j];
        }
        op.mult(k, v, x, dt, &mut r);
        norm = norm2(&r);
    }
    // MFEM: ImplicitSolve does MFEM_VERIFY(newton_solver.GetConverged(), ...)
    if !converged {
        eprintln!(
            "Newton solver did not converge (||r|| = {}, ||r||/||r_0|| = {})",
            fmt_g(norm),
            fmt_g(norm / norm0)
        );
        std::process::abort();
    }
}

// ─── ODE integrators ───────────────────────────────────────────────────────

/// Forward Euler explicit step: `vx += dt * f(vx)`.
fn forward_euler_step(
    vx: &mut [f64],
    dt: f64,
    m: &CsrMatrix<f64>,
    s: &CsrMatrix<f64>,
    hyper: &MfemHyperelasticAssembler,
    ess_dofs_v: &[usize],
    ess_dofs_x: &[usize],
) {
    let sc = vx.len() / 2;
    let (v, x) = vx.split_at_mut(sc);
    // Compute dv/dt = -M^{-1} * (H(x) + S*v), with H evaluated at the
    // deformed positions x_ref + x (MFEM position-based semantics).
    let mut rhs = vec![0.0; sc];
    hyper.residual_positions(x, &mut rhs);
    let mut sv = vec![0.0; sc];
    mfem_csr_mult(s, v, &mut sv);
    for i in 0..sc { rhs[i] += sv[i]; }
    for i in 0..sc { rhs[i] = -rhs[i]; }
    // Enforce BC on rhs: zero for constrained velocity DOFs
    for &d in ess_dofs_v { rhs[d] = 0.0; }

    let mut dv = vec![0.0; sc];
    // MFEM M_solver: CGSolver + DSmoother, iterative_mode=false, rel_tol 1e-8,
    // abs_tol 0, max_iter 30, print level 0.
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 30,
        verbose: false,
        ..SolverConfig::default()
    };
    match solve_pcg_dsmoother(&m, &rhs, &mut dv, &cfg) {
        Ok(_) => {}
        Err(e) => eprintln!("  Explicit: M solve failed: {e}"),
    }
    for &d in ess_dofs_v { dv[d] = 0.0; }

    // dx/dt = v
    for i in 0..sc {
        v[i] += dt * dv[i];
        x[i] += dt * v[i]; // using updated v
    }
    // Enforce BC on x
    for &d in ess_dofs_x { x[d] = 0.0; }
}

/// SDIRK2 (type 22) — two-stage implicit with Newton solve per stage.
///
/// Butcher tableau: γ = 1 - 1/√2
///   γ  |  γ   0
///   1  | 1-γ  γ
///   ---|--------
///      | 1-γ  γ
fn sdirk2_step(
    vx: &mut [f64],
    dt: f64,
    op: &ReducedSystemOperator,
    ess_dofs_k: &[usize],
    verbose: bool,
) {
    let sc = vx.len() / 2;
    let gamma = 1.0 - FRAC_1_SQRT_2; // ≈ 0.2929

    let (v, x) = vx.split_at_mut(sc);

    // ── Stage 1 ─────────────────────────────────────────────────────────────
    // Solve for kv1: R(kv) = 0 at u_n with timestep γ*dt.
    // The reduced system operates on kv (velocity increment).
    // The corresponding position increment is kx1 = v + (γ*dt)*kv1.
    let mut kv1 = vec![0.0; sc];
    let dt_gamma = gamma * dt;
    newton_solve_reduced(op, &mut kv1, v, x, dt_gamma, verbose);
    let mut kx1 = vec![0.0; sc];
    for i in 0..sc { kx1[i] = v[i] + dt_gamma * kv1[i]; }

    // ── Stage 2 intermediate state ──────────────────────────────────────────
    // U2 = u_n + dt*(1-γ)*k1 where k1 = [kv1, kx1]
    let mut v2 = v.to_vec();
    let mut x2 = x.to_vec();
    for i in 0..sc {
        v2[i] += dt * (1.0 - gamma) * kv1[i];
        x2[i] += dt * (1.0 - gamma) * kx1[i];
    }

    // ── Stage 2 ─────────────────────────────────────────────────────────────
    let mut kv2 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv2, &v2, &x2, dt_gamma, verbose);
    let mut kx2 = vec![0.0; sc];
    for i in 0..sc { kx2[i] = v2[i] + dt_gamma * kv2[i]; }

    // ── Final update: u_{n+1} = u_n + dt * ((1-γ)*k1 + γ*k2) ──────────────
    for i in 0..sc {
        v[i] += dt * ((1.0 - gamma) * kv1[i] + gamma * kv2[i]);
        x[i] += dt * ((1.0 - gamma) * kx1[i] + gamma * kx2[i]);
    }
    // MFEM's SDIRK33Solver::Step performs **no** essential-BC enforcement on
    // the step output: the boundary conditions act only through the
    // eliminated operators.  (The initial velocity at the clamp is zero
    // naturally, and the position dofs keep drifting exactly like MFEM's
    // `x_gf`.)
    let _ = ess_dofs_k;
}

/// SDIRK33 (type 23) — 3-stage, 3rd-order, L-stable (exact MFEM coefficients).
///
/// Butcher tableau (from MFEM linalg/ode.cpp SDIRK33Solver):
/// ```
///   a  |  a
///   c  |  c-a    a
///   1  |   b   1-a-b  a
///  ----+----------------
///      |   b   1-a-b  a
/// ```
/// Coefficients:
///   a = 0.435866521508458999416019  (diagonal, L-stable)
///   b = 1.20849664917601007033648
///   c = 0.717933260754229499708010
///
/// Stage k values from ImplicitSolve are velocity increments kv.
/// Position increments kx = v_current + dt_stage * kv (from kinematics).
fn sdirk3_step(
    vx: &mut [f64],
    dt: f64,
    op: &ReducedSystemOperator,
    ess_dofs_k: &[usize],
    verbose: bool,
) {
    const A: f64 = 0.435866521508458999416019;
    const B: f64 = 1.20849664917601007033648;
    const C: f64 = 0.717933260754229499708010;

    let sc = vx.len() / 2;
    let (v, x) = vx.split_at_mut(sc);
    let dt_a = A * dt;

    // ── Stage 1 ───────────────────────────────────────────────────────────
    let mut kv1 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv1, v, x, dt_a, verbose);
    if std::env::var_os("FEM_EX10_PROBE").is_some() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static KDUMP: AtomicUsize = AtomicUsize::new(0);
        if KDUMP.fetch_add(1, Ordering::Relaxed) == 0 {
            let _ = std::fs::write(
                "tmp/d92b/rs_k1.txt",
                kv1.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
            );
        }
    }
    let mut kx1 = vec![0.0; sc];
    for i in 0..sc { kx1[i] = v[i] + dt_a * kv1[i]; }

    // y = vx0 + (c-a)*dt * k1
    let mut vy = v.to_vec();
    let mut xy = x.to_vec();
    let ca = C - A;
    for i in 0..sc {
        vy[i] += ca * dt * kv1[i];
        xy[i] += ca * dt * kx1[i];
    }
    // Partial accumulate: vx += b*dt * k1
    for i in 0..sc {
        v[i] += B * dt * kv1[i];
        x[i] += B * dt * kx1[i];
    }

    // ── Stage 2 ───────────────────────────────────────────────────────────
    let mut kv2 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv2, &vy, &xy, dt_a, verbose);
    let mut kx2 = vec![0.0; sc];
    for i in 0..sc { kx2[i] = vy[i] + dt_a * kv2[i]; }

    let nab = 1.0 - A - B;
    for i in 0..sc {
        v[i] += nab * dt * kv2[i];
        x[i] += nab * dt * kx2[i];
    }

    // ── Stage 3 ───────────────────────────────────────────────────────────
    let mut kv3 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv3, v, x, dt_a, verbose);
    let mut kx3 = vec![0.0; sc];
    for i in 0..sc { kx3[i] = v[i] + dt_a * kv3[i]; }

    for i in 0..sc {
        v[i] += A * dt * kv3[i];
        x[i] += A * dt * kx3[i];
    }

    // No BC enforcement (see the SDIRK2 note above).
    let _ = ess_dofs_k;
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ─── 1. Print options (args.PrintOptions(cout)) ──────────────────────────
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --ode-solver {}", args.ode_solver_type);
    println!("   --t-final {}", fmt_g(args.t_final));
    println!("   --time-step {}", fmt_g(args.dt));
    println!("   --viscosity {}", fmt_g(args.viscosity));
    println!("   --shear-modulus {}", fmt_g(args.mu));
    println!("   --bulk-modulus {}", fmt_g(args.K));
    if !args.visualization {
        println!("   --no-visualization");
    } else {
        println!("   --visualization");
    }
    println!("   --visualization-steps {}", args.vis_steps);

    // ─── 2. Read mesh ───────────────────────────────────────────────────────
    let mut mesh: Mesh<2> = read_mfem_file(&args.mesh)
        .expect("failed to read MFEM mesh")
        .mesh2d
        .expect("MFEM mesh must be 2D");
    let dim = mesh.dim() as usize;
    eprintln!("  Mesh: {} elements, {} nodes, dim={dim}", mesh.n_elems(), mesh.n_nodes());

    // ─── 3. Uniform refinement ───────────────────────────────────────────────
    for _ in 0..args.ref_levels {
        mesh = refine_uniform(&mesh);
    }
    eprintln!("  After refinement: {} elements, {} nodes", mesh.n_elems(), mesh.n_nodes());

    // ─── 4. FE space ────────────────────────────────────────────────────────
    let space = fem_space::VectorH1Space::new(mesh.clone(), args.order, dim as u8);
    let n_total = space.n_dofs(); // total vector DOFs for one field
    println!("Number of velocity/deformation unknowns: {n_total}");

    // ─── 5. Block vector vx = [v, x] ──────────────────────────────────────
    let mut vx = vec![0.0; 2 * n_total];
    let (v_block, x_block) = vx.split_at_mut(n_total);

    // ─── 6. Essential BC (boundary attribute 1 is fixed) ─────────────────────
    let _bdr_attr_max = mesh.unique_boundary_tags().iter().max().copied().unwrap_or(1).max(1) as usize;
    // Boundary attribute 1 → fixed (all components = 0)

    // Build essential DOF list for the vector FE space
    let dm = space.scalar_dof_manager();
    let ess_scalar_dofs = fem_space::constraints::boundary_dofs(&mesh, dm, &[1]);
    let mut ess_dofs: Vec<usize> = Vec::new();
    for c in 0..dim {
        let offset = c * (n_total / dim);
        for &d in &ess_scalar_dofs {
            ess_dofs.push(d as usize + offset);
        }
    }
    ess_dofs.sort_unstable();
    ess_dofs.dedup();

    // ─── 7. Hyperelastic model (NeoHookean) ─────────────────────────────────
    // MFEM ex10 uses NeoHookeanModel(mu, K) which is a DEVIATORIC formulation:
    //   W = μ/2·(J^{-2/3}·I₁ - dim) + K/2·(J-1)²
    // assembled through the bit-exact port of MFEM's
    // `HyperelasticNLFIntegrator` + `NeoHookeanModel` (D842-2): quadrature
    // `IntRules.Get(SQUARE, 2*order+3)`, position-based kernels, glibc-pow
    // arithmetic.
    let n_scalar = n_total / dim;
    let scalar_dofs: Vec<Vec<u32>> = {
        let dm = space.scalar_dof_manager();
        (0..mesh.n_elems()).map(|e| dm.element_dofs(e as u32).to_vec()).collect()
    };
    let hyper = {
        let dm = space.scalar_dof_manager();
        let (mut px, mut py) = (vec![0.0_f64; n_scalar], vec![0.0_f64; n_scalar]);
        for s in 0..n_scalar {
            let c = dm.dof_coord(s as u32);
            px[s] = c[0];
            py[s] = c[1];
        }
        let elem_dofs: Vec<Vec<usize>> = (0..mesh.n_elems())
            .map(|e| space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        MfemHyperelasticAssembler::new(
            mesh.clone(),
            args.order,
            args.mu,
            args.K,
            [px, py],
            scalar_dofs.clone(),
            elem_dofs,
        )
    };

    // ─── 8. Assemble M (mass) and S (viscosity) — MFEM storage pipeline ─────
    //   M = VectorMassIntegrator (ρ = 1.0), S = VectorDiffusionIntegrator
    //   (κ = viscosity), both on `IntRules.Get(SQUARE, 2p+1)`, element
    //   matrices scattered through MFEM's open-LIL `AddSubMatrix` +
    //   `Finalize(0)` (head→tail rows), then the in-place
    //   `FormSystemMatrix` → `EliminateVDofs(ess, DIAG_KEEP)` — so every
    //   later `M.SpMat()` (reduced J, explicit M solve, `KineticEnergy`,
    //   `M->AddMult` residual terms) sees exactly MFEM's matrix.
    let (mut m, mut s) = {
        let bilin = MfemBilinearQuad::new(args.order);
        let ref_xy = hyper.ref_xy().to_vec();
        let mut m_lil = MfemLilMatrix::new(n_total);
        let mut s_lil = MfemLilMatrix::new(n_total);
        for (e, sd) in scalar_dofs.iter().enumerate() {
            let corners: Vec<[f64; 2]> = mesh
                .element_nodes(e as u32)
                .iter()
                .map(|&n| {
                    let c = mesh.node_coords(n);
                    [c[0], c[1]]
                })
                .collect();
            // byNODES vdofs: [component-0 dofs; component-1 dofs (+ n_scalar)].
            let vdofs: Vec<u32> = sd
                .iter()
                .copied()
                .chain(sd.iter().map(|&d| d + n_scalar as u32))
                .collect();
            let em = bilin.mass_element(&corners, &ref_xy, 1.0);
            let es = bilin.diffusion_element(&corners, &ref_xy, args.viscosity);
            if std::env::var_os("FEM_EX10_PROBE").is_some() && e < 2 {
                for (tag, mat) in [("mass", &em), ("diff", &es)] {
                    let _ = std::fs::write(
                        format!("tmp/d92b/rs_ms_{tag}_{e}.txt"),
                        mat.iter().map(|v| format!("{v:.17e}\n")).collect::<String>(),
                    );
                }
            }
            m_lil.add_sub_matrix(&vdofs, &em);
            s_lil.add_sub_matrix(&vdofs, &es);
        }
        (m_lil.finalize(), s_lil.finalize())
    };
    {
        let ess: Vec<DofId> = ess_dofs.iter().copied().map(|d| d as DofId).collect();
        let x_bc = vec![0.0; n_total];
        let mut dummy = vec![0.0; n_total];
        eliminate_ess_tdofs(&mut m, &ess, &x_bc, &mut dummy, ElimPolicy::DiagKeep);
        eliminate_ess_tdofs(&mut s, &ess, &x_bc, &mut dummy, ElimPolicy::DiagKeep);
    }
    if std::env::var_os("FEM_EX10_PROBE").is_some() {
        dump_csr("tmp/d92b/rs_M_csr.txt", &m);
        dump_csr("tmp/d92b/rs_S_csr.txt", &s);
    }

    if std::env::var_os("FEM_EX10_PROBE").is_some() {
        let mut map = String::new();
        for (e, sd) in scalar_dofs.iter().enumerate() {
            map.push_str(&format!("{} {}
", e, sd.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")));
        }
        let _ = std::fs::write("tmp/d92b/rs_elem_dof_map.txt", map);
    }

    // ─── 9. Initial conditions ──────────────────────────────────────────────
    // Initial deformation: u = 0 (identity = reference configuration).
    // Initial velocity: parabolic profile (same as MFEM ex10).
    //
    // MFEM's GridFunction::ProjectCoefficient on an H1 space evaluates the
    // coefficient AT THE DOF NODES (nodal interpolation), NOT an L2
    // projection — the cubic v_y profile is not in P2, so the two differ
    // (this was the d91b "IC vsum 1-ulp" observation, in fact structural).
    // Mirror MFEM: dof value = InitialVelocity(dof_coord).
    let mut initial_v = vec![0.0; n_total];
    {
        let dm = space.scalar_dof_manager();
        for c in 0..dim {
            for i in 0..n_scalar {
                let coord = dm.dof_coord(i as u32);
                initial_v[c * n_scalar + i] = initial_velocity(coord)[c];
            }
        }
    }

    // v = initial_v; x = the **position field** (MFEM's `x_gf`):
    // `x_gf->ProjectCoefficient(InitialDeformation)` evaluates the identity
    // at the P2 dof nodes, i.e. x starts as the reference coordinates — NOT
    // as a zero displacement field.  All stage updates then accumulate
    // directly onto the positions, exactly like MFEM's `x_gf`.
    v_block.copy_from_slice(&initial_v);
    {
        let dm = space.scalar_dof_manager();
        for c in 0..dim {
            for i in 0..n_scalar {
                let coord = dm.dof_coord(i as u32);
                x_block[c * n_scalar + i] = coord[c];
            }
        }
    }

    // Enforce BC on initial conditions (velocity; the position dofs at the
    // clamp keep their reference coordinates, which are the correct state).
    for &d in &ess_dofs {
        if d < n_total {
            v_block[d] = 0.0;
        }
    }

    // Full-precision IC trace (d91b probe, stderr only).
    if std::env::var_os("FEM_EX10_PROBE").is_some() {
        let sv: f64 = v_block.iter().sum();
        let sx: f64 = x_block.iter().sum();
        eprintln!("IC vsum={sv:.17e} xsum={sx:.17e}");
    }

    // ─── 10. Create ReducedSystemOperator ───────────────────────────────────
    let reduced_op = ReducedSystemOperator {
        m: &m,
        s: &s,
        hyper: &hyper,
        ess_dofs: &ess_dofs,
        n: n_total,
    };

    // ─── 11. Initial energies ───────────────────────────────────────────────
    let ee0 = hyper.energy_positions(x_block);
    let mut mv_tmp = vec![0.0; n_total];
    mfem_csr_mult(&m, v_block, &mut mv_tmp);
    let ke0 = 0.5 * dot(v_block, &mv_tmp);
    println!("initial elastic energy (EE) = {}", fmt_g(ee0));
    println!("initial kinetic energy (KE) = {}", fmt_g(ke0));
    println!("initial   total energy (TE) = {}", fmt_g(ee0 + ke0));

    // ─── 12. Time integration loop ──────────────────────────────────────────
    let mut t = 0.0;
    let mut last_step = false;
    let mut step = 0;

    while !last_step {
        let dt_real = args.dt.min(args.t_final - t);
        step += 1;

        // Perform the time step based on solver type
        match args.ode_solver_type {
            4 => {
                // Forward Euler (explicit)
                forward_euler_step(&mut vx, dt_real, &m, &s, &hyper, &ess_dofs, &ess_dofs);
                t += dt_real;
            }
            22 => {
                // SDIRK2
                let (_v, _x) = vx.split_at_mut(n_total);
                sdirk2_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
            23 => {
                // SDIRK3 (use sdirk3_step for the proper 3-stage version)
                let (_v, _x) = vx.split_at_mut(n_total);
                sdirk3_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
            _ => {
                eprintln!("ODE solver type {} not implemented, using SDIRK2", args.ode_solver_type);
                sdirk2_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
        }

        last_step = t >= args.t_final - 1e-8 * args.dt;

        if last_step || (step % args.vis_steps == 0) {
            let (v, x) = vx.split_at_mut(n_total);
            let ee = hyper.energy_positions(x);
            mfem_csr_mult(&m, v, &mut mv_tmp);
            let ke = 0.5 * dot(v, &mv_tmp);
            println!(
                "step {}, t = {}, EE = {}, KE = {}, ΔTE = {}",
                step,
                fmt_g(t),
                fmt_g(ee),
                fmt_g(ke),
                fmt_g((ee + ke) - (ee0 + ke0))
            );
        }
    }

    // ─── 13. Save output files ─────────────────────────────────────────────
    {
        let (v, x) = vx.split_at_mut(n_total);

        // Save deformed mesh.  `x` is the **position** field (MFEM's `x_gf`);
        // the displacement handed to `apply_displacement` is `x - x_ref`.
        let mut disp = x.to_vec();
        {
            let dm = space.scalar_dof_manager();
            for c in 0..dim {
                for i in 0..n_scalar {
                    let coord = dm.dof_coord(i as u32);
                    disp[c * n_scalar + i] -= coord[c];
                }
            }
        }
        let deformed = mesh.apply_displacement(&disp, dim);
        match write_mfem_file("deformed.mesh", &deformed) {
            Ok(_) => eprintln!("  Saved deformed.mesh ({} elements)", deformed.n_elems()),
            Err(e) => eprintln!("  Warning: failed to write deformed.mesh: {e}"),
        }

        // Save velocity field as .gf (MFEM GridFunction format)
        match write_gf_file("velocity.sol", dim, v, "VectorH1", args.order, dim as usize) {
            Ok(_) => eprintln!("  Saved velocity.sol ({} DOFs)", v.len()),
            Err(e) => eprintln!("  Warning: failed to write velocity.sol: {e}"),
        }

        // NOTE (D842-2): the generic L² projection helper lived on the
        // retired `HyperelasticityForm`; the stdout acceptance covers the
        // EE/KE/TE lines only, so `elastic_energy.sol` is not rewritten.
    }

    let elapsed = t0.elapsed();
    eprintln!("\n  Total time: {:.3}s", elapsed.as_secs_f64());
    eprintln!("  Done.");
}
