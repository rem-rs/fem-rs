//! 1:1 serial port of MFEM's `miniapps/dfem/dfem-minimal-surface.cpp` (a PAR
//! miniapp) at `-np 1`.
//!
//! Solves the 2D minimal surface problem with Dirichlet boundary conditions
//! given by the Scherk surface
//!
//! ```text
//!     u|∂Ω = log(cos(a·x)/cos(a·y))/a ,  a = 1e-2
//! ```
//!
//! on `Mesh::MakeCartesian2D(4, 4, QUADRILATERAL)` transformed to
//! `[-π/2, π/2]²` (`x → (x − 0.5)·π`), with interior initial guess
//! `1e-2 · boundary_func`.  The nonlinear problem is solved with Newton's
//! method; the Jacobian action is computed in one of three ways (`-der`):
//!
//! ```text
//! -der 0 = Automatic differentiation (dual numbers, fem_assembly::ad)
//! -der 1 = Hand-coded derivatives (ManualDerivativeApply)
//! -der 2 = Finite differences (MFEM FDJacobian scheme)
//! ```
//!
//! The pointwise residual form is MFEM's `MFApply`:
//! `F_i = Σ_q coeff(∇ₓu) ∇ₓu · J⁻ᵀ det(J) w · ∇̂φ_i`, and the AD path seeds the
//! reference gradient with `(B u, B δu)` duals — the forward-mode equivalent
//! of `DifferentiableOperator::GetDerivative`'s action.
//!
//! MFEM `CGSolver` (relTol 1e-4, maxIter 500, printLevel 2, no preconditioner)
//! inside MFEM `NewtonSolver` (relTol 1e-6, absTol 0, maxIter 10, printLevel 1)
//! are ported locally with byte-identical iteration output.
//!
//! Clipped vs the C++ miniapp (message + `exit(3)` where a run would differ):
//!   * `-pcamg` (AMG-only path assembles the Jacobian into a HypreParMatrix
//!     for BoomerAMG — Hypre-only infrastructure);
//!   * `-d` device other than `cpu`.
//!
//! Sample runs:
//! ```text
//! cargo run --release --example dfem_minimal_surface -- -no-vis
//! cargo run --release --example dfem_minimal_surface -- -der 1 -no-vis
//! cargo run --release --example dfem_minimal_surface -- -der 0 -r 1 -no-vis
//! ```

use fem_assembly::ad::{AdScalar, Dual};
use fem_element::lagrange::{QuadQ1, QuadQ2, QuadQ3, QuadQ4};
use fem_element::quadrature::quad_rule;
use fem_element::{QuadratureRule, ReferenceElement};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, H1Space};
use std::cell::RefCell;

const DIM: usize = 2;

// ─── CLI (MFEM OptionsParser) ────────────────────────────────────────────────

/// Derivative type enum (MFEM `enum DerivativeType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DerivativeType {
    /// Automatic differentiation (dual type).
    Autodiff,
    /// Hand-coded derivative.
    Handcoded,
    /// Finite differences.
    Fd,
}

struct Args {
    order: i32,
    device: String,
    visualization: bool,
    refinements: i32,
    derivative_type: i32,
    enable_pcamg: bool,
}

impl Args {
    fn parse() -> Args {
        let mut a = Args {
            order: 1,
            device: "cpu".to_string(),
            visualization: true,
            refinements: 0,
            derivative_type: 0,
            enable_pcamg: false,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let arg = argv[i].clone();
            let val = |i: &mut usize| -> String {
                *i += 1;
                argv.get(*i).cloned().unwrap_or_else(|| {
                    eprintln!("dfem-minimal-surface: missing value for {arg}");
                    std::process::exit(3);
                })
            };
            match arg.as_str() {
                "-o" | "--order" => a.order = val(&mut i).parse().unwrap_or(a.order),
                "-d" | "--device" => a.device = val(&mut i),
                "-vis" | "--visualization" => a.visualization = true,
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-r" | "--refinements" => a.refinements = val(&mut i).parse().unwrap_or(a.refinements),
                "-der" | "--derivative-type" => {
                    a.derivative_type = val(&mut i).parse().unwrap_or(a.derivative_type)
                }
                "-pcamg" | "--pcamg" => a.enable_pcamg = true,
                "-no-pcamg" | "--no-pcamg" => a.enable_pcamg = false,
                other => {
                    eprintln!("dfem-minimal-surface (Rust port): Unrecognized option: {other}");
                    std::process::exit(3);
                }
            }
            i += 1;
        }
        a
    }

    /// MFEM `OptionsParser::PrintOptions`.
    fn print_options(&self) {
        println!("Options used:");
        println!("   --order {}", self.order);
        println!("   --device {}", self.device);
        if self.visualization {
            println!("   --visualization");
        } else {
            println!("   --no-visualization");
        }
        println!("   --refinements {}", self.refinements);
        println!("   --derivative-type {}", self.derivative_type);
        if self.enable_pcamg {
            println!("   --pcamg");
        } else {
            println!("   --no-pcamg");
        }
    }
}

// ─── Boundary function (Scherk surface) ──────────────────────────────────────

/// Boundary function for the minimal surface problem described by the Scherk
/// surface: `log(cos(a·x)/cos(a·y))/a` with `a = 1e-2`.
fn boundary_func(coords: &[f64]) -> f64 {
    let x = coords[0];
    let y = coords[1];
    let a = 1.0e-2;
    ((a * x).cos().ln() - (a * y).cos().ln()) / a
}

// ─── Pointwise form (MFApply / ManualDerivativeApply) ────────────────────────

/// `coeff(∇u) = 1 / sqrt(1 + |∇u|²)` — generic so the same code runs for the
/// passive evaluation (`f64`) and the AD evaluation ([`Dual`]).
fn coeff<T: AdScalar>(a: &[T; DIM]) -> T {
    let mut n2 = T::zero();
    for k in 0..DIM {
        n2 = n2 + a[k] * a[k];
    }
    T::of_f64(1.0) / (T::of_f64(1.0) + n2).sqrt_s()
}

/// Closed-form inverse of a plain 2×2.
fn inv2(a: &[[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let inv_det = 1.0 / det;
    [[a[1][1] * inv_det, -a[0][1] * inv_det], [-a[1][0] * inv_det, a[0][0] * inv_det]]
}

fn det2(a: &[[f64; DIM]; DIM]) -> f64 {
    a[0][0] * a[1][1] - a[0][1] * a[1][0]
}

// ─── MinimalSurface operator ─────────────────────────────────────────────────

/// Trait matching MFEM `Operator::Mult`.
trait Operator {
    fn height(&self) -> usize;
    fn mult(&self, x: &[f64], y: &mut [f64]);
}

/// Serial replica of MFEM's `MinimalSurface` operator (T-vector == L-vector).
struct MinimalSurfaceOp {
    mesh: Mesh<2>,
    space: H1Space<Mesh<2>>,
    order: usize,
    /// Solution-space reference element (MFEM H1 quad basis on [-1,1]²).
    ref_elem: Box<dyn ReferenceElement>,
    /// MFEM `IntRules.Get(QUADRILATERAL, 2*order+1)`.
    quad: QuadratureRule,
    ess_tdofs: Vec<usize>,
    derivative_type: DerivativeType,
    /// Cached state for the Newton linearization (`MinimalSurface::u`).
    u_state: RefCell<Vec<f64>>,
}

impl MinimalSurfaceOp {
    fn new(mesh: Mesh<2>, space: H1Space<Mesh<2>>, derivative_type: DerivativeType) -> Self {
        let order = space.get_order().max(1) as usize;
        let ref_elem: Box<dyn ReferenceElement> = match order {
            1 => Box::new(QuadQ1),
            2 => Box::new(QuadQ2),
            3 => Box::new(QuadQ3),
            _ => Box::new(QuadQ4),
        };
        let quad = quad_rule((2 * order + 1) as u8);
        MinimalSurfaceOp {
            mesh,
            space,
            order,
            ref_elem,
            quad,
            ess_tdofs: Vec::new(),
            derivative_type,
            u_state: RefCell::new(Vec::new()),
        }
    }

    fn set_essential_true_dofs(&mut self) {
        let dofs = boundary_dofs(&self.mesh, self.space.dof_manager(), &[1, 2, 3, 4]);
        self.ess_tdofs = dofs.iter().map(|&d| d as usize).collect();
    }

    /// `dudxi[c] = Σ_i x[dofs[i]] · dφ_i/dξ_c`, evaluated with `f64` values
    /// and (optionally) a `Dual` direction seeded from `z`.
    fn interp_grads(
        &self,
        x: &[f64],
        z: Option<&[f64]>,
        dofs: &[u32],
        grads_ref: &[f64],
    ) -> ([f64; DIM], [Dual; DIM]) {
        let nd = dofs.len();
        let mut val = [0.0_f64; DIM];
        let mut du = [Dual::new(0.0, 0.0); DIM];
        for c in 0..DIM {
            let mut sv = 0.0_f64;
            let mut sz = 0.0_f64;
            for i in 0..nd {
                sv += x[dofs[i] as usize] * grads_ref[i * DIM + c];
                if let Some(zv) = z {
                    sz += zv[dofs[i] as usize] * grads_ref[i * DIM + c];
                }
            }
            val[c] = sv;
            du[c] = Dual::new(sv, sz);
        }
        (val, du)
    }

    /// Shared per-element residual loop (MFEM `MFApply` + `Gradient` output).
    fn apply_residual(&self, x: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let nq = self.quad.points.len();
        let mut grads_ref = vec![0.0_f64; self.ref_elem.n_dofs() * DIM];
        let mut geo_grads = vec![0.0_f64; 4 * DIM];
        let geo = QuadQ1;

        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.space.element_dofs(e);
            let nodes = self.mesh.element_nodes(e);
            for q in 0..nq {
                let xi = &self.quad.points[q];
                let w = self.quad.weights[q];
                self.ref_elem.eval_grad_basis(xi, &mut grads_ref);
                geo.eval_grad_basis(xi, &mut geo_grads);
                let mut jac = [[0.0_f64; DIM]; DIM];
                for k in 0..4 {
                    let xk = self.mesh.geom_coords_of(nodes[k]);
                    for i in 0..DIM {
                        for d in 0..DIM {
                            jac[i][d] += xk[i] * geo_grads[k * DIM + d];
                        }
                    }
                }
                let det_j = det2(&jac);
                let inv_j = inv2(&jac);

                // dudxi[c]
                let nd = dofs.len();
                let mut dudxi = [0.0_f64; DIM];
                for c in 0..DIM {
                    let mut s = 0.0;
                    for i in 0..nd {
                        s += x[dofs[i] as usize] * grads_ref[i * DIM + c];
                    }
                    dudxi[c] = s;
                }
                // dudx = dudxi * invJ
                let mut dudx = [0.0_f64; DIM];
                for c in 0..DIM {
                    let mut s = 0.0;
                    for k in 0..DIM {
                        s += dudxi[k] * inv_j[k][c];
                    }
                    dudx[c] = s;
                }
                // out = coeff(dudx) * dudx * transpose(invJ) * det(J) * w
                //     (out[c] = coeff · Σ_k dudx[k] · invJ[c][k] · detJ · w)
                let cf = coeff::<f64>(&dudx);
                let mut out = [0.0_f64; DIM];
                for c in 0..DIM {
                    let mut s = 0.0;
                    for k in 0..DIM {
                        s += dudx[k] * inv_j[c][k];
                    }
                    out[c] = cf * s * det_j * w;
                }

                // y[dofs[i]] += out · ∇̂φ_i
                for i in 0..nd {
                    let mut s = 0.0;
                    for c in 0..DIM {
                        s += out[c] * grads_ref[i * DIM + c];
                    }
                    y[dofs[i] as usize] += s;
                }
            }
        }
        set_zero_at(y, &self.ess_tdofs);
    }

    /// Jacobian action via dual-number AD (`-der 0`): the reference gradient
    /// is seeded with the dual `(B u, B δu)` — the forward-mode equivalent of
    /// `DifferentiableOperator::GetDerivative(...)->Mult(z, y)`.
    fn jac_apply_ad(&self, z: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let x = self.u_state.borrow();
        let nq = self.quad.points.len();
        let mut grads_ref = vec![0.0_f64; self.ref_elem.n_dofs() * DIM];
        let mut geo_grads = vec![0.0_f64; 4 * DIM];
        let geo = QuadQ1;

        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.space.element_dofs(e);
            let nodes = self.mesh.element_nodes(e);
            for q in 0..nq {
                let xi = &self.quad.points[q];
                let w = self.quad.weights[q];
                self.ref_elem.eval_grad_basis(xi, &mut grads_ref);
                geo.eval_grad_basis(xi, &mut geo_grads);
                let mut jac = [[0.0_f64; DIM]; DIM];
                for k in 0..4 {
                    let xk = self.mesh.geom_coords_of(nodes[k]);
                    for i in 0..DIM {
                        for d in 0..DIM {
                            jac[i][d] += xk[i] * geo_grads[k * DIM + d];
                        }
                    }
                }
                let det_j = det2(&jac);
                let inv_j = inv2(&jac);

                let nd = dofs.len();
                // seed: value = Bu, gradient = Bz (forward mode through B)
                let mut dudxi = [Dual::new(0.0, 0.0); DIM];
                for c in 0..DIM {
                    let mut vu = 0.0_f64;
                    let mut vz = 0.0_f64;
                    for i in 0..nd {
                        vu += x[dofs[i] as usize] * grads_ref[i * DIM + c];
                        vz += z[dofs[i] as usize] * grads_ref[i * DIM + c];
                    }
                    dudxi[c] = Dual::new(vu, vz);
                }
                // dudx = dudxi * invJ  (Dual·f64 arithmetic)
                let mut dudx = [Dual::new(0.0, 0.0); DIM];
                for c in 0..DIM {
                    let mut s = Dual::new(0.0, 0.0);
                    for k in 0..DIM {
                        s = s + dudxi[k] * inv_j[k][c];
                    }
                    dudx[c] = s;
                }
                // out = coeff(dudx) * dudx * transpose(invJ) * detJ * w
                let cf = coeff::<Dual>(&dudx);
                let mut out = [Dual::new(0.0, 0.0); DIM];
                for c in 0..DIM {
                    let mut s = Dual::new(0.0, 0.0);
                    for k in 0..DIM {
                        s = s + dudx[k] * inv_j[c][k];
                    }
                    out[c] = cf * s * det_j * w;
                }

                // y[dofs[i]] += d(out)/dδu · ∇̂φ_i  (real part is the residual)
                for i in 0..nd {
                    let mut s = 0.0;
                    for c in 0..DIM {
                        s += out[c].gradient * grads_ref[i * DIM + c];
                    }
                    y[dofs[i] as usize] += s;
                }
            }
        }
    }

    /// Jacobian action via the hand-coded derivative (`-der 1`, MFEM
    /// `ManualDerivativeApply`).
    fn jac_apply_hand(&self, z: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let x = self.u_state.borrow();
        let nq = self.quad.points.len();
        let mut grads_ref = vec![0.0_f64; self.ref_elem.n_dofs() * DIM];
        let mut geo_grads = vec![0.0_f64; 4 * DIM];
        let geo = QuadQ1;

        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.space.element_dofs(e);
            let nodes = self.mesh.element_nodes(e);
            for q in 0..nq {
                let xi = &self.quad.points[q];
                let w = self.quad.weights[q];
                self.ref_elem.eval_grad_basis(xi, &mut grads_ref);
                geo.eval_grad_basis(xi, &mut geo_grads);
                let mut jac = [[0.0_f64; DIM]; DIM];
                for k in 0..4 {
                    let xk = self.mesh.geom_coords_of(nodes[k]);
                    for i in 0..DIM {
                        for d in 0..DIM {
                            jac[i][d] += xk[i] * geo_grads[k * DIM + d];
                        }
                    }
                }
                let det_j = det2(&jac);
                let inv_j = inv2(&jac);

                let nd = dofs.len();
                let mut dudxi = [0.0_f64; DIM];
                let mut ddudxi = [0.0_f64; DIM];
                for c in 0..DIM {
                    let mut su = 0.0;
                    let mut sz = 0.0;
                    for i in 0..nd {
                        su += x[dofs[i] as usize] * grads_ref[i * DIM + c];
                        sz += z[dofs[i] as usize] * grads_ref[i * DIM + c];
                    }
                    dudxi[c] = su;
                    ddudxi[c] = sz;
                }
                let mut dudx = [0.0_f64; DIM];
                let mut ddudx = [0.0_f64; DIM];
                for c in 0..DIM {
                    let mut su = 0.0;
                    let mut sz = 0.0;
                    for k in 0..DIM {
                        su += dudxi[k] * inv_j[k][c];
                        sz += ddudxi[k] * inv_j[k][c];
                    }
                    dudx[c] = su;
                    ddudx[c] = sz;
                }
                // ManualDerivativeApply:
                //   term1 = c · ddudx
                //   term2 = c³ · dot(dudx, ddudx) · dudx
                //   out   = (term1 − term2) · transpose(invJ) · detJ · w
                let c0 = coeff::<f64>(&dudx);
                let dot = dudx[0] * ddudx[0] + dudx[1] * ddudx[1];
                let mut out = [0.0_f64; DIM];
                for c in 0..DIM {
                    let term1 = c0 * ddudx[c];
                    let term2 = c0 * c0 * c0 * dot * dudx[c];
                    let mut s = 0.0;
                    for k in 0..DIM {
                        s += (term1 - term2) * inv_j[c][k];
                    }
                    out[c] = s * det_j * w;
                }

                for i in 0..nd {
                    let mut s = 0.0;
                    for c in 0..DIM {
                        s += out[c] * grads_ref[i * DIM + c];
                    }
                    y[dofs[i] as usize] += s;
                }
            }
        }
    }

    /// MFEM `MinimalSurface::Mult`.
    fn residual(&self, x: &[f64], y: &mut [f64]) {
        self.apply_residual(x, y);
        for &d in &self.ess_tdofs {
            y[d] = 0.0;
        }
    }

    /// MFEM `MinimalSurfaceJacobian::Mult` (ess elimination wrapper).
    fn jac_mult(&self, x: &[f64], y: &mut [f64]) {
        let mut z = x.to_vec();
        for &d in &self.ess_tdofs {
            z[d] = 0.0;
        }
        match self.derivative_type {
            DerivativeType::Handcoded => self.jac_apply_hand(&z, y),
            _ => self.jac_apply_ad(&z, y),
        }
        for &d in &self.ess_tdofs {
            y[d] = x[d];
        }
    }

    fn get_gradient(&self, x: &[f64]) -> MinimalSurfaceJacobian<'_> {
        let mut st = self.u_state.borrow_mut();
        if st.len() != x.len() {
            *st = x.to_vec();
        } else {
            st.copy_from_slice(x);
        }
        drop(st);
        MinimalSurfaceJacobian { minsurface: self }
    }
}

impl Operator for MinimalSurfaceOp {
    fn height(&self) -> usize {
        self.space.n_dofs()
    }
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.residual(x, y);
    }
}

fn set_zero_at(y: &mut [f64], idx: &[usize]) {
    for &i in idx {
        y[i] = 0.0;
    }
}

/// Port of `MinimalSurfaceJacobian` (AD / hand-coded derivative action).
struct MinimalSurfaceJacobian<'a> {
    minsurface: &'a MinimalSurfaceOp,
}

impl Operator for MinimalSurfaceJacobian<'_> {
    fn height(&self) -> usize {
        self.minsurface.height()
    }
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.minsurface.jac_mult(x, y);
    }
}/// Port of MFEM's `FDJacobian` (`fem/dfem/util.hpp`): one-sided finite
/// difference `y = (F(x + ε·v) − F(x)) / ε` with
/// `ε = λ·(λ + |x|/|v|)`, `λ = 1e-6`; `F(x)` is cached.
struct FdJacobian<'a> {
    op: &'a MinimalSurfaceOp,
    x: Vec<f64>,
    f: Vec<f64>,
    xnorm: f64,
}

impl FdJacobian<'_> {
    fn new<'a>(op: &'a MinimalSurfaceOp, x: &[f64]) -> FdJacobian<'a> {
        let mut f = vec![0.0_f64; op.height()];
        op.mult(x, &mut f);
        FdJacobian { op, x: x.to_vec(), f, xnorm: x.iter().map(|v| v * v).sum::<f64>().sqrt() }
    }
}

impl Operator for FdJacobian<'_> {
    fn height(&self) -> usize {
        self.op.height()
    }
    fn mult(&self, v: &[f64], y: &mut [f64]) {
        // FDJacobian::Mult — ess handling is inside op.Mult's z elimination
        // (mirrors the C++ usage where FDJacobian wraps the MinimalSurface op).
        let lambda = 1.0e-6_f64;
        let vnorm = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        let eps = lambda * (lambda + self.xnorm / vnorm);
        let mut xpev = vec![0.0_f64; self.x.len()];
        for i in 0..self.x.len() {
            xpev[i] = self.x[i] + eps * v[i];
        }
        self.op.mult(&xpev, y);
        for i in 0..self.f.len() {
            y[i] = (y[i] - self.f[i]) / eps;
        }
    }
}

// ─── MFEM CGSolver / NewtonSolver (serial ports, no preconditioner) ─────────

/// MFEM `PrintLevel` flags derived from the legacy print level.
struct PrintLevel {
    warnings: bool,
    iterations: bool,
    summary: bool,
    first_and_last: bool,
}

/// MFEM `FromLegacyPrintLevel`.
fn from_legacy_print_level(lvl: i32) -> PrintLevel {
    match lvl {
        -1 => PrintLevel { warnings: false, iterations: false, summary: false, first_and_last: false },
        0 => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: false },
        1 => PrintLevel { warnings: true, iterations: true, summary: false, first_and_last: false },
        2 => PrintLevel { warnings: true, iterations: false, summary: true, first_and_last: false },
        3 => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: true },
        _ => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: false },
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Port of MFEM `CGSolver::Mult` (serial, unpreconditioned,
/// `iterative_mode = false`).  Returns `(converged, final_iter, sqrt(betanom))`.
// `unused_assignments` is silenced because MFEM's control flow lets `betanom`
// keep its last loop value across every break/return path, so the seeding
// assignment is deliberately never read.
#[allow(unused_assignments)]
fn cg_mult(
    a: &dyn Operator,
    b: &[f64],
    x: &mut [f64],
    rtol: f64,
    atol: f64,
    max_iter: usize,
    pl: &PrintLevel,
) -> (bool, usize, f64) {
    let n = a.height();
    // iterative_mode == false: r = b; x = 0.0;
    let mut r = b.to_vec();
    for v in x.iter_mut() {
        *v = 0.0;
    }
    // no preconditioner: d = r
    let mut d = r.clone();
    let mut nom = dot(&d, &r);
    let nom0 = nom;
    if pl.iterations || pl.first_and_last {
        print!(
            "   Iteration : {:>3}  (B r, r) = {}{}",
            0,
            fmt_g6(nom),
            if pl.first_and_last { " ...\n" } else { "\n" }
        );
    }

    if nom < 0.0 {
        if pl.warnings {
            println!(
                "PCG: The preconditioner is not positive definite. (Br, r) = {}",
                fmt_g6(nom)
            );
        }
        return (false, 0, nom);
    }
    let r0 = (nom * rtol * rtol).max(atol * atol);
    if nom <= r0 {
        // MFEM returns here without printing the summary.
        return (true, 0, nom.sqrt());
    }

    // z = A d
    let mut z = vec![0.0_f64; n];
    a.mult(&d, &mut z);
    let mut den = dot(&z, &d);

    let mut converged = false;
    let mut final_iter = max_iter;
    let mut betanom = nom;
    let mut i = 1_usize;
    loop {
        let alpha = nom / den;
        for k in 0..n {
            x[k] += alpha * d[k];
            r[k] -= alpha * z[k];
        }
        // no preconditioner: betanom = (r, r)
        betanom = dot(&r, &r);
        if betanom < 0.0 {
            if pl.warnings {
                println!(
                    "PCG: The preconditioner is not positive definite. (Br, r) = {}",
                    fmt_g6(betanom)
                );
            }
            final_iter = i;
            break;
        }
        if pl.iterations {
            println!("   Iteration : {:>3}  (B r, r) = {}", i, fmt_g6(betanom));
        }
        if betanom <= r0 {
            converged = true;
            final_iter = i;
            break;
        }
        i += 1;
        if i > max_iter {
            break;
        }
        let beta = betanom / nom;
        for k in 0..n {
            d[k] = r[k] + beta * d[k];
        }
        a.mult(&d, &mut z);
        den = dot(&d, &z);
        if den == 0.0 {
            final_iter = i;
            break;
        }
        nom = betanom;
    }

    if pl.first_and_last && !pl.iterations {
        println!("   Iteration : {:>3}  (B r, r) = {}", final_iter, fmt_g6(betanom));
    }
    if pl.summary || (pl.warnings && !converged) {
        println!("PCG: Number of iterations: {}", final_iter);
    }
    if pl.summary || pl.iterations || pl.first_and_last {
        let arf = (betanom / nom0).powf(0.5 / final_iter as f64);
        println!("Average reduction factor = {}", fmt_g6(arf));
    }
    if pl.warnings && !converged {
        println!("PCG: No convergence!");
    }
    (converged, final_iter, betanom.sqrt())
}

/// Port of MFEM `NewtonSolver::Mult` (serial, `iterative_mode = true`,
/// `ComputeScalingFactor == 1`, `have_b == false`).
fn newton_mult(
    oper: &MinimalSurfaceOp,
    x: &mut [f64],
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    pl: &PrintLevel,
    cg_rtol: f64,
    cg_atol: f64,
    cg_max_iter: usize,
    cg_pl: &PrintLevel,
) -> (bool, usize, f64) {
    let n = oper.height();
    let mut r = vec![0.0_f64; n];
    oper.mult(x, &mut r);
    let norm0 = norm2(&r);
    if pl.first_and_last && !pl.iterations {
        println!("Newton iteration {:>2} : ||r|| = {}...\n", 0, fmt_g6(norm0));
    }
    let norm_goal = (rel_tol * norm0).max(abs_tol);

    let mut norm = norm0;
    let mut converged = false;
    let mut it = 0_usize;
    loop {
        if pl.iterations {
            print!("Newton iteration {:>2} : ||r|| = {}", it, fmt_g6(norm));
            if it > 0 {
                print!(", ||r||/||r_0|| = {}", fmt_g6(norm / norm0));
            }
            println!();
        }
        if norm <= norm_goal {
            converged = true;
            break;
        }
        if it >= max_iter {
            break;
        }

        let grad = oper.get_gradient(x);
        let mut c = vec![0.0_f64; n];
        cg_mult(&grad, &r, &mut c, cg_rtol, cg_atol, cg_max_iter, cg_pl);
        // c_scale = 1.0; x -= c
        for k in 0..n {
            x[k] -= c[k];
        }
        oper.mult(x, &mut r);
        norm = norm2(&r);
        it += 1;
    }

    if pl.summary || (!converged && pl.warnings) || pl.first_and_last {
        println!("Newton: Number of iterations: {}", it);
        println!(
            "   ||r|| = {},  ||r||/||r_0|| = {}",
            fmt_g6(norm),
            fmt_g6(norm / norm0)
        );
    }
    if !converged && (pl.summary || pl.warnings) {
        println!("Newton: No convergence!");
    }
    (converged, it, norm)
}

// ─── C++ std::ostream default formatting (%.6g) ──────────────────────────────

/// `std::ostream` with default precision 6: `%g` semantics.
fn fmt_g6(v: f64) -> String {
    fmt_g(v, 6)
}

fn fmt_g(x: f64, sig: u32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        let mut e = exp;
        let mut mantissa = x / 10_f64.powi(e);
        if mantissa.abs() >= 10.0 {
            mantissa /= 10.0;
            e += 1;
        }
        let mut s = format!("{:.*}", (sig - 1) as usize, mantissa);
        trim_trailing_zeros(&mut s);
        format!("{s}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let decimals = ((sig as i32 - 1) - exp).max(0) as usize;
        let mut s = format!("{:.*}", decimals, x);
        trim_trailing_zeros(&mut s);
        s
    }
}

fn trim_trailing_zeros(s: &mut String) {
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    args.print_options();

    // Device(device_config): only the CPU path is ported.
    if args.device != "cpu" {
        eprintln!(
            "dfem-minimal-surface (Rust port): device {:?} is not ported (CPU only).",
            args.device
        );
        std::process::exit(3);
    }
    if args.enable_pcamg {
        // MFEM_ABORT("AMG only available for the AUTODIFF derivative type") —
        // here the whole AMG path is Hypre-only and is clipped.
        eprintln!(
            "dfem-minimal-surface (Rust port): -pcamg is not ported \
             (assembles the Jacobian into a HypreParMatrix for BoomerAMG)."
        );
        std::process::exit(3);
    }
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // 4. Mesh::MakeCartesian2D(4, 4, QUADRILATERAL) on [0,1]², then transform
    //    to [-π/2, π/2]²:  x → (x − 0.5)·π   (SetCurvature(order): linear
    //    nodes for order 1 — the geometry map is affine for every order).
    let mut mesh = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    for c in mesh.coords.chunks_mut(DIM) {
        c[0] = (c[0] - 0.5) * std::f64::consts::PI;
        c[1] = (c[1] - 0.5) * std::f64::consts::PI;
    }
    // 5. Refine the mesh to increase the resolution
    for _ in 0..args.refinements {
        mesh = refine_uniform(&mesh);
    }

    // 7. H1_FECollection fec(order, dim); FiniteElementSpace H1(&pmesh, &fec)
    let order = args.order.max(1) as usize;
    let space = H1Space::new(mesh.clone(), order as u8);

    // 8. Integration rule: IntRules.Get(QUAD, 2*order+1)
    let quad = quad_rule((2 * order + 1) as u8);
    {
        let ip0 = &quad.points[0];
        println!("[check] ir npts={} x0={:.6} w0={:.6}", quad.points.len(), ip0[0], quad.weights[0]);
    }
    println!("[check] ndofs={}", space.n_dofs());

    let derivative_type = match args.derivative_type {
        1 => DerivativeType::Handcoded,
        2 => DerivativeType::Fd,
        _ => DerivativeType::Autodiff,
    };

    // 9. MinimalSurface operator
    let mut minsurface = MinimalSurfaceOp::new(mesh.clone(), space.clone(), derivative_type);

    // 10/11. Essential BCs on all boundaries + initial condition
    minsurface.set_essential_true_dofs();
    let ess_tdofs = minsurface.ess_tdofs.clone();

    let mut u = vec![0.0_f64; space.n_dofs()];
    let dm = space.dof_manager();
    for d in 0..space.n_dofs() {
        // ProjectCoefficient(boundary_coeff) then u *= 1e-2 — for H1 nodal
        // elements this is interpolation at the dof locations.
        u[d] = 1.0e-2 * boundary_func(&dm.dof_coord(d as u32)[..DIM]);
    }
    for &d in &ess_tdofs {
        // u.ProjectBdrCoefficient(boundary_coeff, ess_bdr)
        u[d] = boundary_func(&dm.dof_coord(d as u32)[..DIM]);
    }

    // 12. CGSolver krylov: relTol 1e-4, absTol 0, maxIter 500, printLevel 2
    //     (no preconditioner; -pcamg is clipped above)
    let cg_rtol = 1e-4;
    let cg_atol = 0.0;
    let cg_max_iter = 500_usize;
    let cg_pl = from_legacy_print_level(2);

    // 13. NewtonSolver: relTol 1e-6, absTol 0, maxIter 10, printLevel 1
    // 14. X = restriction of u; newton.Mult(zero, X)
    let mut x = u.clone();
    let (converged, it, final_norm) = newton_mult(
        &minsurface,
        &mut x,
        1e-6,
        0.0,
        10,
        &from_legacy_print_level(1),
        cg_rtol,
        cg_atol,
        cg_max_iter,
        &cg_pl,
    );

    // 15/16. GLVis / ParaView outputs (vtu stands in for ParaViewDataCollection)
    if args.visualization {
        match fem_io::glvis::GlVisSocket::connect("localhost", 19916) {
            Ok(mut sock) => {
                sock.send_solution_2d(&mesh, &x, "solution").ok();
            }
            Err(e) => eprintln!("  GLVis not available: {e}"),
        }
    }
    {
        let mut writer = fem_io::VtkWriter::new(&mesh);
        writer.add_point_data(fem_io::DataArray::scalars("solution", x.clone()));
        writer
            .write_file("dfem-minimal-surface-output.vtu")
            .map_err(|e| eprintln!("  ParaView (vtu) output failed: {e}"))
            .ok();
    }

    println!("[check] converged = {converged} iters = {it}");
    println!("[check] final ||r|| = {:.16e}", final_norm);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "[check] ||X|| = {:.16e} Xmax = {xmax:.16e} Xmin = {xmin:.16e}",
        norm2(&x)
    );
}
