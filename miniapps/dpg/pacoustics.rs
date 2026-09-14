//! Parallel ultraweak DPG solver for the acoustics (Helmholtz) problem — 1:1
//! port of MFEM's `miniapps/dpg/pacoustics.cpp` (MFEM 4.10), complex valued.
//!
//! Solves `-Δ p - ω² p = f̃` in Ω, `p = p₀` on ∂Ω through the first-order
//! system `∇p + iωαu = 0`, `∇·u + iωβp = f` (`α = 1`, `β = 1` outside the PML)
//! with the traces `p̂ ∈ H^{1/2}`, `û ∈ H^{-1/2}`:
//!
//! ```text
//!     -(p, ∇·v) + iω (u, v) + < p̂, v·n > = 0,    ∀ v ∈ H(div)
//!     -(u, ∇ q) + iω (p, q) + < û, q    > = (f,q), ∀ q ∈ H¹
//! ```
//!
//! with the adjoint-graph test norm.  Trial spaces (C++ `pacoustics.cpp`):
//! `p ∈ L²(p−1)`, `u ∈ (L²(p−1))²`, `p̂ ∈ H¹-trace(p)`, `û ∈ RT-trace(p−1)`;
//! broken test spaces `q ∈ H¹(p+δ)`, `v ∈ RT(p+δ−1)`, solved with
//! [`fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm`] and the
//! parallel complex Hermitian PCG
//! ([`fem_parallel::par_complex_solver::par_solve_complex_pcg`]) over the
//! block-diagonal complex symmetric-GS preconditioner
//! ([`fem_parallel::par_complex_solver::ComplexBlockDiagGs`]).
//!
//! Problem cases: `-prob 0` (plane wave, default) `p = exp(iβ(x+y))`,
//! `β = ω/√dim`; `-prob 1` (Gaussian beam manufactured solution, with the
//! `(f, q)` right-hand side).
//!
//! Printed table matches the C++ miniapp (exact-solution case):
//! `Ref | Dofs | ω | L2 Error | Rate | Residual | Rate | PCG it`.
//!
//! # Verified against the C++ MPI reference
//!
//! Built from `$HOME/mfem410_mpi` (`pacoustics.cpp + util/*.cpp +
//! ../common/*.cpp` against `libmfem.a -lHYPRE`) and run under
//! `mpirun -np {1,2} … -no-vis` (evidence: `tmp/r35a/cpp_*.log`,
//! `tmp/r35a/femrs_all_configs.txt`).  `Dofs`, `L2 Error` and `Residual`
//! match the fem-rs run **to all printed digits** — including the statically
//! condensed parallel system:
//!
//! ```text
//!                                        C++                 fem-rs
//! -prob 0 -sref 0 -np 1:    113  8.008e-01  1.374e+00  ==  113  8.008e-01  1.374e+00
//! -prob 0 -sref 0 -np 2:    113  8.008e-01  1.374e+00  ==  113  8.008e-01  1.374e+00
//! -prob 0 -sref 1 -np 2:    417  4.472e-01  9.121e-01  ==  417  4.472e-01  9.121e-01
//! -prob 0 -sc     -np 2:    113  8.008e-01  1.374e+00  ==  113  8.008e-01  1.374e+00
//! -prob 0 -sc -sref 1 -np1: 417  4.472e-01  9.121e-01  ==  417  4.472e-01  9.121e-01
//! -prob 1         -np 1:    113  3.994e-01  8.914e-01  ==  113  3.994e-01  8.914e-01
//! -prob 1         -np 2:    113  3.994e-01  8.914e-01  ==  113  3.994e-01  8.914e-01
//! ```
//!
//! (for `-prob 1` even the unprinted p/u error split matches to 7 digits:
//! C++ 2.70284448e-01 / 2.94068773e-01 vs fem-rs 2.702844e-01 / 2.940688e-01).
//!
//! The **PCG iteration count is not reproduced**: MFEM applies
//! `HypreBoomerAMG` / `HypreAMS` per block of the *real part* of the parallel
//! operator (`MakeFESpaceDefaultSolver` + `ComplexPreconditioner`), fem-rs
//! applies its own complex block symmetric Gauss–Seidel to the owned diagonal
//! blocks (round-33 `pdiffusion` precedent: block smoothers differ, iteration
//! counts are partition-dependent even in C++ — 18 vs 23 at `-np 1`/`-np 2`
//! for the identical system), and fem-rs converges tighter (rtol `1e-12`)
//! than the C++ `1e-6` so the printed L2/residual digits are pinned.  The
//! printed residual is the *DPG* residual `‖B x − F‖_{S⁻¹}`, reproduced
//! exactly.//!
//! # Known gaps (exit code 3)
//!
//! * `-prob >= 2` (PML): the `CartesianPML` stretched-map coefficients and
//!   the restricted/PML integrator set are not ported; `prob 3-5` also need
//!   the `meshes/scatter.mesh` / GSLIB point-source machinery.  Exits 3.
//! * 3-D meshes (`inline-hex.mesh`): the parallel 3-D H1-trace numbering
//!   (skeleton **edge** DOFs shared by every incident face) is not
//!   implemented.  Exits 3.
//! * `-pref > 0` (parallel AMR + `Update()`): same gap as `pdiffusion` (the
//!   marked-refinement repartitioning breaks the identity node numbering the
//!   DPG trace numbering requires).  Exits 3.
//! * `-pmg` (PRefinementMultigrid): not ported.  Exits 3.
//! * GLVis visualization is not implemented (C++ `-vis`); `-no-vis` matches.
//!
//! Usage:
//!   cargo run --release --example pacoustics -- --ranks 2
//!   cargo run --release --example pacoustics -- --ranks 2 -sref 1
//!   cargo run --release --example pacoustics -- --ranks 2 -prob 1

use std::process::exit;
use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDivDivIntegrator, DpgDomainLFIntegrator, DpgDiffusionIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTVectorFEMassIntegrator, DpgTraceIntegrator, DpgVectorFEDivergenceIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_mesh::{refine_uniform, ElementType, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm;
use fem_parallel::par_complex_solver::{par_solve_complex_pcg, ComplexBlockDiagGs};
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_vector::{ParComplexVector, ParVector};
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;

/// Problem case — C++ `enum prob_type` (the PML cases are not ported).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prob {
    /// `plane_wave`: `p = exp(iβ(x+y))`, `f̃ = 0`.
    PlaneWave,
    /// `gaussian_beam`: manufactured solution with `(f, q)` right-hand side.
    GaussianBeam,
}

impl Prob {
    fn name(self) -> &'static str {
        match self {
            Prob::PlaneWave => "plane_wave",
            Prob::GaussianBeam => "gaussian_beam",
        }
    }
}

// ── tiny complex helpers (split (re, im) pairs) ──────────────────────────────

#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn cscale(s: f64, a: (f64, f64)) -> (f64, f64) {
    (s * a.0, s * a.1)
}

#[inline]
fn cexp(re: f64, im: f64) -> (f64, f64) {
    let m = re.exp();
    (m * im.cos(), m * im.sin())
}

/// The exact solution machinery of `pacoustics.cpp`
/// (`acoustics_solution`, `acoustics_solution_grad`,
/// `acoustics_solution_laplacian` and the derived `u_exact`, `rhs_func`).
#[derive(Clone, Copy)]
struct Exact {
    omega: f64,
    prob: Prob,
}

impl Exact {
    /// `acoustics_solution(X)`.
    fn p(&self, x: &[f64]) -> (f64, f64) {
        match self.prob {
            Prob::PlaneWave => {
                let beta = self.omega / (x.len() as f64).sqrt();
                cexp(0.0, beta * x.iter().sum::<f64>())
            }
            Prob::GaussianBeam => {
                let rk = self.omega;
                let alpha = (180.0 + 45.0) * PI / 180.0;
                let (sina, cosa) = (alpha.sin(), alpha.cos());
                let (xprim, yprim) = (x[0] + 0.1, x[1] + 0.1);
                let x = xprim * sina - yprim * cosa;
                let y = xprim * cosa + yprim * sina;
                let rl = 2.0 * PI / rk;
                let w0 = 0.05_f64;
                let fact = rl / PI / (w0 * w0);
                let aux = 1.0 + (fact * y) * (fact * y);
                let w = w0 * aux.sqrt();
                let phi0 = (fact * y).atan();
                let r = y + 1.0 / y / (fact * fact);
                let zi_rk_y = (0.0, -rk * y);
                let zi_pi_x2 = (0.0, -PI * x * x / rl / r);
                let zi_phi0 = (0.0, phi0 / 2.0);
                let ze_re = -x * x / (w * w);
                let ze = cadd(cadd(cadd((ze_re, 0.0), zi_rk_y), zi_pi_x2), zi_phi0);
                let pf = (2.0 / PI / (w * w)).powf(0.25);
                cscale(pf, cexp(ze.0, ze.1))
            }
        }
    }

    /// `acoustics_solution_grad(X)`.
    fn grad_p(&self, x: &[f64]) -> Vec<(f64, f64)> {
        match self.prob {
            Prob::PlaneWave => {
                let beta = self.omega / (x.len() as f64).sqrt();
                let p = self.p(x);
                vec![cmul((0.0, beta), p); x.len()]
            }
            Prob::GaussianBeam => {
                let rk = self.omega;
                let alpha = (180.0 + 45.0) * PI / 180.0;
                let (sina, cosa) = (alpha.sin(), alpha.cos());
                let (xprim, yprim) = (x[0] + 0.1, x[1] + 0.1);
                let x = xprim * sina - yprim * cosa;
                let y = xprim * cosa + yprim * sina;
                let rl = 2.0 * PI / rk;
                let w0 = 0.05_f64;
                let fact = rl / PI / (w0 * w0);
                let aux = 1.0 + (fact * y) * (fact * y);
                let w = w0 * aux.sqrt();
                let dwdy = w0 * fact * fact * y / aux.sqrt();
                let phi0 = (fact * y).atan();
                let dphi0dy = phi0.cos() * phi0.cos() * fact;
                let r = y + 1.0 / y / (fact * fact);
                let drdy = 1.0 - 1.0 / (y * y) / (fact * fact);
                let ze_re = -x * x / (w * w);
                let ze_im = -rk * y - PI * x * x / rl / r + phi0 / 2.0;
                // zdedx = -2x/w² - 2iπx/(rl r)
                let zdedx = (-2.0 * x / (w * w), -2.0 * PI * x / rl / r);
                // zdedy = 2x²/w³ dwdy - i rk + iπx²/(rl r²) drdy + i dphi0dy/2
                let zdedy = (
                    2.0 * x * x / (w * w * w) * dwdy,
                    -rk + PI * x * x / rl / (r * r) * drdy + dphi0dy / 2.0,
                );
                let pf = (2.0 / PI / (w * w)).powf(0.25);
                let dpfdy = -(2.0 / PI / (w * w)).powf(-0.75) / PI / (w * w * w) * dwdy;
                let zp = cscale(pf, cexp(ze_re, ze_im));
                let zdpdx = cmul(zp, zdedx);
                let zdpdy = cadd(cscale(dpfdy, cexp(ze_re, ze_im)), cmul(zp, zdedy));
                // dp/dxprim = zdpdx*sina + zdpdy*cosa
                // dp/dyprim = zdpdx*(-cosa) + zdpdy*sina
                vec![
                    cadd(cscale(sina, zdpdx), cscale(cosa, zdpdy)),
                    cadd(cscale(-cosa, zdpdx), cscale(sina, zdpdy)),
                ]
            }
        }
    }

    /// `acoustics_solution_laplacian(X)` (1:1; note the C++ plane-wave branch
    /// returns `+dim·β²·p`, the sign is irrelevant here because the laplacian
    /// only feeds the RHS, which is zero for the plane-wave problem).
    fn laplacian_p(&self, x: &[f64]) -> (f64, f64) {
        match self.prob {
            Prob::PlaneWave => {
                let beta = self.omega / (x.len() as f64).sqrt();
                cscale(x.len() as f64 * beta * beta, self.p(x))
            }
            Prob::GaussianBeam => {
                let rk = self.omega;
                let alpha = (180.0 + 45.0) * PI / 180.0;
                let (sina, cosa) = (alpha.sin(), alpha.cos());
                let (xprim, yprim) = (x[0] + 0.1, x[1] + 0.1);
                let x = xprim * sina - yprim * cosa;
                let y = xprim * cosa + yprim * sina;
                let rl = 2.0 * PI / rk;
                let w0 = 0.05_f64;
                let fact = rl / PI / (w0 * w0);
                let aux = 1.0 + (fact * y) * (fact * y);
                let w = w0 * aux.sqrt();
                let dwdy = w0 * fact * fact * y / aux.sqrt();
                let d2wdydy = w0 * fact * fact * (1.0 - (fact * y) * (fact * y) / aux) / aux.sqrt();
                let phi0 = (fact * y).atan();
                let dphi0dy = phi0.cos() * phi0.cos() * fact;
                let d2phi0dydy = -2.0 * phi0.cos() * phi0.sin() * fact * dphi0dy;
                let r = y + 1.0 / y / (fact * fact);
                let drdy = 1.0 - 1.0 / (y * y) / (fact * fact);
                let d2rdydy = 2.0 / (y * y * y) / (fact * fact);
                let ze = (-x * x / (w * w), -rk * y - PI * x * x / rl / r + phi0 / 2.0);
                let zdedx = (-2.0 * x / (w * w), -2.0 * PI * x / rl / r);
                let zdedy = (
                    2.0 * x * x / (w * w * w) * dwdy,
                    -rk + PI * x * x / rl / (r * r) * drdy + dphi0dy / 2.0,
                );
                let zd2edxdx = (-2.0 / (w * w), -2.0 * PI / rl / r);
                let zd2edxdy = (
                    4.0 * x / (w * w * w) * dwdy,
                    2.0 * 2.0 * PI * x / rl / (r * r) * drdy,
                );
                let zd2edydy = (
                    -6.0 * x * x / (w * w * w * w) * dwdy * dwdy
                        + 2.0 * x * x / (w * w * w) * d2wdydy,
                    // 1:1 with the C++ `zi/real_t(2.*d2phi0dydy)` term
                    -(2.0 * PI * x * x / rl / (r * r * r) * drdy * drdy)
                        + PI * x * x / rl / (r * r) * d2rdydy
                        + 1.0 / (2.0 * d2phi0dydy),
                );
                let pf = (2.0 / PI / (w * w)).powf(0.25);
                let dpfdy = -(2.0 / PI / (w * w)).powf(-0.75) / PI / (w * w * w) * dwdy;
                let d2pfdydy = -1.0 / PI
                    * (2.0 / PI).powf(-0.75)
                    * (-1.5 * w.powf(-2.5) * dwdy * dwdy + w.powf(-1.5) * d2wdydy);
                let expze = cexp(ze.0, ze.1);
                let zp = cscale(pf, expze);
                let zdpdx = cmul(zp, zdedx);
                let zdpdy = cadd(cscale(dpfdy, expze), cmul(zp, zdedy));
                let zd2pdxdx = cadd(cmul(zdpdx, zdedx), cmul(zp, zd2edxdx));
                let zd2pdxdy = cadd(cmul(zdpdy, zdedx), cmul(zp, zd2edxdy));
                let zd2pdydx = cadd(
                    cadd(cscale(dpfdy, cmul(expze, zdedx)), cmul(zdpdx, zdedy)),
                    cmul(zp, zd2edxdy),
                );
                let zd2pdydy = cadd(
                    cadd(
                        cscale(d2pfdydy, expze),
                        cscale(dpfdy, cmul(expze, zdedy)),
                    ),
                    cadd(cmul(zdpdy, zdedy), cmul(zp, zd2edydy)),
                );
                // (zd2pdxdx*sina + zd2pdydx*cosa)*sina
                // + (zd2pdxdy*sina + zd2pdydy*cosa)*cosa
                // + (zd2pdxdx*(-cosa) + zd2pdydx*sina)*(-cosa)
                // + (zd2pdxdy*(-cosa) + zd2pdydy*sina)*sina
                let t00 = cmul(zd2pdxdx, (sina, 0.0));
                let t00 = cadd(t00, cmul(zd2pdydx, (cosa, 0.0)));
                let t01 = cmul(zd2pdxdy, (sina, 0.0));
                let t01 = cadd(t01, cmul(zd2pdydy, (cosa, 0.0)));
                let t10 = cmul(zd2pdxdx, (-cosa, 0.0));
                let t10 = cadd(t10, cmul(zd2pdydx, (sina, 0.0)));
                let t11 = cmul(zd2pdxdy, (-cosa, 0.0));
                let t11 = cadd(t11, cmul(zd2pdydy, (sina, 0.0)));
                cadd(
                    cadd(cscale(sina, t00), cscale(cosa, t01)),
                    cadd(cscale(-cosa, t10), cscale(sina, t11)),
                )
            }
        }
    }

    /// `u_exact_r/i`: `u = -∇p/(iω)`, i.e. `u_r = -∇p_i/ω`, `u_i = ∇p_r/ω`.
    fn u(&self, x: &[f64]) -> Vec<(f64, f64)> {
        self.grad_p(x)
            .into_iter()
            .map(|(gr, gi)| (-gi / self.omega, gr / self.omega))
            .collect()
    }

    /// `rhs_func_r/i`: `f = ∇·u + iω p` (only used for the Gaussian beam).
    fn f(&self, x: &[f64]) -> (f64, f64) {
        let (pr, pi) = self.p(x);
        let (lr, li) = self.laplacian_p(x);
        let (divur, divui) = (-li / self.omega, lr / self.omega);
        (divur - self.omega * pi, divui + self.omega * pr)
    }
}

/// C++-style `std::scientific` with 3 digits (`8.008e-01`).
fn cpp_sci3(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v:.3e}");
    }
    if v == 0.0 {
        return "0.000e+00".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let mut exp = a.log10().floor() as i32;
    let mut mant = a / 10f64.powi(exp);
    if mant >= 10.0 {
        mant /= 10.0;
        exp += 1;
    }
    if mant < 1.0 {
        mant *= 10.0;
        exp -= 1;
    }
    let mut s = format!("{mant:.3}");
    if s.parse::<f64>().unwrap_or(mant) >= 10.0 {
        s = format!("{:.3}", mant / 10.0);
        exp += 1;
    }
    format!(
        "{}{}e{}{:02}",
        if neg { "-" } else { "" },
        s,
        if exp < 0 { "-" } else { "+" },
        exp.abs()
    )
}

/// One refinement level: assemble, solve, return `(dofs, l2 error, residual,
/// pcg iterations)`.
struct LevelResult {
    dofs: usize,
    l2: f64,
    residual: f64,
    iters: usize,
}

fn solve_level(
    mesh: &Mesh<2>,
    n_workers: usize,
    order: u8,
    delta_order: u8,
    omega: f64,
    prob: Prob,
    static_cond: bool,
) -> LevelResult {
    let p = order;
    let test_order = order + delta_order;
    let p_us = p as usize;
    let result = Arc::new(Mutex::new(None::<LevelResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::new(mesh.clone());

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        // Identity node numbering: the DPG face/edge tables derive their
        // canonical face direction from the local node ids (see pdiffusion).
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();

        let mut a = ParComplexDPGWeakForm::new(local_mesh, partition, comm.clone());
        // The fem-rs weak form uses fixed quadrature orders: volume 2·test_order
        // integrates every DPG integrand of this problem exactly, and — for the
        // Gaussian-beam RHS — matches the C++ `DomainLFIntegrator` default rule
        // `2·el.GetOrder() + 0` (`fem/lininteg.hpp`), which is what pins the
        // non-polynomial `(f, q)` load vector to all printed digits.  Face rule
        // `test_order + p − 1` integrates the trace pairings exactly.
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p - 1);
        a.store_matrices(true);

        let ps = a.add_trial_scalar_space(p - 1);
        let us = a.add_trial_vector_space(p - 1, 2);
        let hatp = a.add_trial_trace_space_h1(p);
        let hatu = a.add_trial_trace_space(p - 1);
        let q = a.add_test_space(VolKind::Scalar, test_order);
        let v = a.add_test_space(VolKind::HDiv, test_order - 1);

        // Trial integrators (C++ pacoustics.cpp block table).
        a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: omega })), ps, q); // iω (p, q)
        a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q); // -(u, ∇q)
        a.add_trial_integrator(
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
            None,
            ps,
            v,
        ); // -(p, ∇·v)
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: omega })),
            us,
            v,
        ); // iω (u, v)
        a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v); // <p̂, v·n>
        a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q); // <û, q>

        // Adjoint graph norm (test integrators).
        a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
        a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorGradientIntegrator {
                q: vec![vec![-omega, 0.0], vec![0.0, -omega]],
            })),
            v,
            q,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
                q: vec![vec![-omega, 0.0], vec![0.0, -omega]],
            })),
            q,
            v,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: omega * omega })),
            None,
            v,
            v,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -omega })),
            q,
            v,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -omega })),
            v,
            q,
        );
        a.add_test_integrator(
            Some(Box::new(DpgMassIntegrator { q: omega * omega })),
            None,
            q,
            q,
        );

        // RHS (f, q) — the Gaussian beam manufactured solution.
        if prob == Prob::GaussianBeam {
            let ex_r = Exact { omega, prob };
            let ex_i = ex_r;
            a.add_domain_lf_integrator(
                Some(Box::new(DpgDomainLFIntegrator {
                    f: move |x: &[f64]| ex_r.f(x).0,
                })),
                Some(Box::new(DpgDomainLFIntegrator {
                    f: move |x: &[f64]| ex_i.f(x).1,
                })),
                q,
            );
        }

        if static_cond {
            a.enable_static_condensation();
        }
        a.assemble();

        // Essential BCs: p̂ = p₀ on every global boundary face (C++
        // `hatp_fes->GetEssentialTrueDofs(ess_bdr, …)` with all boundary
        // attributes).
        let ex = Exact { omega, prob };
        let pairs = a.trace_boundary_dofs(hatp);
        let merged = a.merge_dof_points(&pairs);
        let ess_ids: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();

        let n_local = a.local().size();
        let mut x_local_r = vec![0.0_f64; n_local];
        let mut x_local_i = vec![0.0_f64; n_local];
        a.fill_essential_values(&mut x_local_r, &mut x_local_i, &merged, &|pt: &[f64]| {
            ex.p(pt)
        });

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local_r, &x_local_i);

        let half = x0.len() / 2;
        let exchange = a.ghost_exchange_arc();
        let mut xv = ParComplexVector {
            re: ParVector::from_local_raw(
                x0[..half].to_vec(),
                sys.n_owned,
                exchange.clone(),
                comm.clone(),
            ),
            im: ParVector::from_local_raw(x0[half..].to_vec(), sys.n_owned, exchange, comm.clone()),
        };

        // Block-diagonal preconditioner over the owned diagonal blocks (MFEM
        // `BlockDiagonalPreconditioner` + `ComplexPreconditioner`; the
        // per-block solvers differ from Hypre — see the module docs for the
        // iteration-count caveat).
        let offsets: Vec<usize> = a.owned_block_offsets().to_vec();
        // C++ CGSolver: SetRelTol(1e-6), SetMaxIter(10000); fem-rs converges
        // tighter (1e-12) so the printed L2/residual digits are pinned.
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 10000,
            verbose: false,
            ..SolverConfig::default()
        };
        let precond = ComplexBlockDiagGs::from_diag_block(sys.a.diag_block(), &offsets);
        let pc = move |r: &[f64], ri: &[f64], z: &mut [f64], zi: &mut [f64]| {
            precond.apply(r, ri, z, zi)
        };
        let res = par_solve_complex_pcg(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("pacoustics: complex PCG failed");

        let n_owned = sys.n_owned;
        let mut x_owned = vec![0.0_f64; 2 * n_owned];
        x_owned[..n_owned].copy_from_slice(&xv.re.as_slice()[..n_owned]);
        x_owned[n_owned..].copy_from_slice(&xv.im.as_slice()[..n_owned]);
        let x_full = a.recover_fem_solution(&x_owned);

        // Residual ‖residuals‖₂ over the owned elements (global reduction).
        let residual = a.global_residual_norm(&x_full);

        // L2 error of p (re+im) and u (re+im) over the owned elements.
        let (e_p, e_u) = l2_errors(&a, &x_full, ps, us, p_us.saturating_sub(1) as u8, &ex);
        let l2 = comm.allreduce_sum_f64(e_p + e_u).max(0.0).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pacoustics mutex") = Some(LevelResult {
                dofs: a.n_global_trial_dofs(),
                l2,
                residual,
                iters: res.iterations,
            });
        }
    });

    let out = result
        .lock()
        .expect("pacoustics mutex after launch")
        .take()
        .expect("rank 0 did not publish the pacoustics result");
    out
}

/// Squared L2 errors of the `p` and `u` trial blocks (owned elements only),
/// mirroring MFEM `ParGridFunction::ComputeL2Error` (complex re/im parts).
fn l2_errors(
    a: &ParComplexDPGWeakForm<Mesh<2>>,
    x_full: &[f64],
    p_block: usize,
    u_block: usize,
    order: u8,
    ex: &Exact,
) -> (f64, f64) {
    use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};

    let mesh = a.local().mesh();
    let offsets = a.local().trial_offsets();
    let n_target = x_full.len() / 2;
    let (sol_r, sol_i) = (&x_full[..n_target], &x_full[n_target..]);
    let et = mesh.element_type(0);
    let n = scalar_ref_elem(et, order).n_dofs();
    let (qpts, qwts) = vol_quadrature(et, 2 * order + 3);
    let fe = scalar_ref_elem(et, order);
    let mut phi = vec![0.0_f64; n];
    let simp = matches!(et, ElementType::Tri3);
    let rank = a.comm().rank();
    let part = a.partition_ref();
    let dim = 2usize;

    let (mut sp, mut su) = (0.0_f64, 0.0_f64);
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        for (q, xi) in qpts.iter().enumerate() {
            let (det, xp) = if simp {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, mesh.element_nodes(e));
                (tr.det_j(), tr.map_to_physical(xi))
            } else {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("pacoustics: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
                (det, xp)
            };
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
            let (pr, pi) = ex.p(&xp);
            // p (scalar L2 block).
            let base = offsets[p_block] + e as usize * n;
            let (mut phr, mut phi_) = (0.0, 0.0);
            for (i, &b) in phi.iter().enumerate() {
                phr += sol_r[base + i] * b;
                phi_ += sol_i[base + i] * b;
            }
            sp += w * ((phr - pr) * (phr - pr) + (phi_ - pi) * (phi_ - pi));
            // u (vector L2 block, byNODES ordering).
            let ue = ex.u(&xp);
            let ubase = offsets[u_block] + e as usize * n * dim;
            for c in 0..dim {
                let (mut uhr, mut uhi) = (0.0, 0.0);
                for (i, &b) in phi.iter().enumerate() {
                    uhr += sol_r[ubase + c * n + i] * b;
                    uhi += sol_i[ubase + c * n + i] * b;
                }
                su += w * ((uhr - ue[c].0) * (uhr - ue[c].0) + (uhi - ue[c].1) * (uhi - ue[c].1));
            }
        }
    }
    (sp, su)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flags: &[&str]) -> Option<T> {
    for f in flags {
        if let Some(i) = args.iter().position(|a| a == f) {
            if let Some(v) = args.get(i + 1) {
                if let Ok(v) = v.parse::<T>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn has_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|a| flags.contains(&a.as_str()))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let n_workers: usize = parse_arg(&args, &["--ranks"]).unwrap_or(2);
    let mesh_file: String =
        get("-m").or_else(|| get("--mesh")).unwrap_or_else(|| "data/inline-quad.mesh".into());
    let order: i32 = get("-o").or_else(|| get("--order")).and_then(|v| v.parse().ok()).unwrap_or(1);
    let delta_order: i32 =
        get("-do").or_else(|| get("--delta-order")).and_then(|v| v.parse().ok()).unwrap_or(1);
    let rnum: f64 = get("-rnum")
        .or_else(|| get("--number-of-wavelengths"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let iprob: i32 = get("-prob").or_else(|| get("--problem")).and_then(|v| v.parse().ok()).unwrap_or(0);
    let theta: f64 =
        get("-theta").or_else(|| get("--theta")).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let sref: i32 =
        get("-sref").or_else(|| get("--serial-ref")).and_then(|v| v.parse().ok()).unwrap_or(0);
    let pref: i32 =
        get("-pref").or_else(|| get("--parallel-ref")).and_then(|v| v.parse().ok()).unwrap_or(0);
    let pmg = has_flag(&args, &["-pmg", "--p-refinement-multigrid"]);
    let static_cond = has_flag(&args, &["-sc", "--static-condensation"]);
    let paraview = has_flag(&args, &["-paraview", "--paraview"]);
    let iprob = if iprob > 5 { 0 } else { iprob };
    let prob = match iprob {
        0 => Prob::PlaneWave,
        1 => Prob::GaussianBeam,
        _ => {
            eprintln!(
                "pacoustics: GAP — `-prob {iprob}` (PML) is not ported to fem-rs.  The C++ \
                 miniapp assembles the CartesianPML stretched-map coefficients \
                 (`util/pml.cpp`: α = JᵀJ/|J|, β = |J| as RestrictedCoefficient/PmlMatrix \
                 coefficient pairs) which fem_parallel does not carry; `-prob 3..5` \
                 additionally need the meshes/scatter.mesh + GSLIB point-source machinery."
            );
            exit(3);
        }
    };
    let omega = 2.0 * PI * rnum;

    println!("=== fem-rs pacoustics: parallel ultraweak DPG for the Helmholtz problem ===");
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {order}");
    println!("   --number-of-wavelengths {rnum}");
    println!("   --problem {iprob} ({})", prob.name());
    println!("   --delta-order {delta_order}");
    println!("   --theta {theta}");
    println!("   --serial-ref {sref}");
    println!("   --parallel-ref {pref}");
    println!("   --static-condensation{}", if static_cond { "" } else { "-disabled" });
    println!("   --ranks {n_workers}");
    println!("   --no-visualization");

    if pmg {
        eprintln!(
            "pacoustics: GAP — `-pmg` (PRefinementMultigrid) is not ported to fem-rs.  The \
             C++ miniapp builds a complex p-multigrid preconditioner over the trial spaces \
             (`util/preconditioners.cpp:ComplexPRefinementMultigrid`); fem-rs has no \
             p-prolongation operators for DPG blocks."
        );
        exit(3);
    }
    if pref > 0 {
        eprintln!(
            "pacoustics: GAP — `-pref {pref}` (parallel AMR driven by the DPG residual \
             indicator + `ParComplexDPGWeakForm::Update`) is not ported: \
             `fem_parallel::par_refine_marked*` rebuilds the partition with compact node \
             ids, which breaks the identity node numbering the DPG trace numbering \
             depends on.  Re-run with `-pref 0` (the C++ default) for the verified path."
        );
        exit(3);
    }
    if paraview {
        println!("pacoustics: note -paraview is ignored (no ParaView writer for this path).");
    }

    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("pacoustics: cannot read {mesh_file}: {e}"));
    let mesh: Mesh<2> = match mfem.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "pacoustics: GAP — 3-D H1-trace parallel numbering is missing.  The 3-D H1 \
                 trace has DOFs on the skeleton EDGES (shared by every incident face) as \
                 well as on face interiors, while the parallel numbering implements the 2-D \
                 H1 trace (shared vertices) and face-discontinuous traces only \
                 (crates/parallel/src/par_dpg_numbering.rs panics for the 3-D combination)."
            );
            exit(3);
        }
    };
    let mut mesh = mesh;
    for _ in 0..sref {
        mesh = refine_uniform(&mesh);
    }

    let order = order.max(1) as u8;
    let delta_order = delta_order.max(0) as u8;

    println!("\n  Ref |    Dofs    |    ω    |  L2 Error  |  Rate  |  Residual  |  Rate  | PCG it |");
    println!("{}", "-".repeat(82));

    let mut err0 = 0.0_f64;
    let mut res0 = 0.0_f64;
    let mut dof0 = 0usize;

    for it in 0..=pref.max(0) {
        let r = solve_level(&mesh, n_workers, order, delta_order, omega, prob, static_cond);
        // rate = dim·log(err0/err)/log(dof0/dofs), dim = 2 (C++ formula).
        let rate_err = if it > 0 && err0 > 0.0 && r.dofs != dof0 {
            2.0 * (err0 / r.l2).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        let rate_res = if it > 0 && res0 > 0.0 && r.dofs != dof0 {
            2.0 * (res0 / r.residual).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
        } else {
            0.0
        };
        err0 = r.l2;
        res0 = r.residual;
        dof0 = r.dofs;
        println!(
            "{:>5} | {:>10} | {:>4.1} π  | {:>10} | {:>6.2} | {:>10} | {:>6.2} | {:>6} | ",
            it,
            dof0,
            2.0 * rnum,
            cpp_sci3(err0),
            rate_err,
            cpp_sci3(res0),
            rate_res,
            r.iters
        );
    }
}
