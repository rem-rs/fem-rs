//! # MFEM Example 25 — PML for Maxwell — 1:1 Rust translation
//!
//! Solves `(1/μ) curl curl E − ω² ε E = f` with PML.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -o 2 -f 1.0 -ref 0 -prob 0
//! cargo run --example mfem_ex25_pml_maxwell -- -o 2 -f 5.0 -ref 1 -prob 4 -m data/inline-quad.mesh
//! ```

use std::f64::consts::PI;
use nalgebra::Complex;
use fem_assembly::{
    VectorAssembler,
    postproc::coefficient::{CoeffCtx, ScalarCoeff, MatrixCoeff},
    standard::{CurlCurlIntegrator, CurlCurlTensorIntegrator,
               VectorMassTensorIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_io::mfem;
use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_solver::{BlockDiagPrecond, GSSmoother};
use fem_linalg::fem_to_linlvo_csr;
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};
use fem_mesh::{Mesh, topology::MeshTopology, element_type::ElementType};
use fem_mesh::refine_uniform;

// ═══════════════════════════════════════════════════════════════════════════
// Bessel functions (J₀, J₁, J₂, Y₀, Y₁, Y₂) — series-based, pure Rust
// ═══════════════════════════════════════════════════════════════════════════

const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243;

fn bessel_j0(x: f64) -> f64 {
    if x <= 0.0 { return 1.0; }
    let x2 = x / 2.0;
    let x2sq = x2 * x2;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1i32;
    loop {
        term *= -x2sq / (k as f64 * k as f64);
        let prev = sum;
        sum += term;
        if sum == prev || term.abs() < 1e-30 { break; }
        k += 1;
    }
    sum
}

fn bessel_j1(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let x2 = x / 2.0;
    let x2sq = x2 * x2;
    let mut term = x2;
    let mut sum = term;
    let mut k = 1i32;
    loop {
        term *= -x2sq / (k as f64 * (k + 1) as f64);
        let prev = sum;
        sum += term;
        if sum == prev || term.abs() < 1e-30 { break; }
        k += 1;
    }
    sum
}

fn bessel_j2(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let j0 = bessel_j0(x); let j1 = bessel_j1(x);
    if x.abs() < 1e-14 { return 0.0; }
    2.0 / x * j1 - j0
}

fn harmonic(k: u32) -> f64 {
    let mut h = 0.0;
    for i in 1..=k { h += 1.0 / i as f64; }
    h
}

fn bessel_y0(x: f64) -> f64 {
    if x <= 0.0 { return -f64::INFINITY; }
    let x2 = x / 2.0; let x2sq = x2 * x2;
    let j0 = bessel_j0(x);
    let ln_term = (x / 2.0).ln() + EULER_GAMMA;
    let mut sum = 0.0; let mut term = 1.0;
    let mut k = 1u32;
    loop {
        term *= -x2sq / (k as f64 * k as f64);
        let add = term * harmonic(k);
        let prev = sum; sum += add;
        if sum == prev || add.abs() < 1e-30 { break; }
        k += 1;
    }
    2.0 / PI * (ln_term * j0 - sum)
}

fn bessel_y1(x: f64) -> f64 {
    if x <= 0.0 { return -f64::INFINITY; }
    let x2 = x / 2.0; let x2sq = x2 * x2;
    let j1 = bessel_j1(x);
    let ln_term = (x / 2.0).ln() + EULER_GAMMA;
    let mut sum = 0.0; let mut term = x2;
    let mut k = 1u32;
    loop {
        let hk = harmonic(k) + harmonic(k + 1);
        let add = term * (1.0 - 0.5 / (k as f64 + 1.0) - hk);
        let prev = sum; sum += add;
        if sum == prev || add.abs() < 1e-30 { break; }
        k += 1;
        term *= -x2sq / (k as f64 * (k + 1) as f64 + 1e-300);
    }
    2.0 / PI * (ln_term * j1 - 1.0 / x - sum)
}

fn bessel_y2(x: f64) -> f64 {
    if x <= 0.0 { return -f64::INFINITY; }
    if x.abs() < 1e-14 { return -f64::INFINITY; }
    2.0 / x * bessel_y1(x) - bessel_y0(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// PML — 1:1 with MFEM ex25
// ═══════════════════════════════════════════════════════════════════════════

struct PmlParams {
    dim: usize,
    comp_domain_bdr: Vec<[f64; 2]>,
    length: Vec<[f64; 2]>,
    k: f64,
}

impl PmlParams {
    fn new(bb_lo: &[f64], bb_hi: &[f64], pml_lo: &[f64; 3], pml_hi: &[f64; 3], omega: f64, dim: usize) -> Self {
        let mut comp = Vec::with_capacity(dim);
        let mut len = Vec::with_capacity(dim);
        for d in 0..dim {
            comp.push([bb_lo[d] + pml_lo[d], bb_hi[d] - pml_hi[d]]);
            len.push([pml_lo[d], pml_hi[d]]);
        }
        PmlParams { dim, comp_domain_bdr: comp, length: len, k: omega }
    }

    fn stretch(&self, x: &[f64]) -> Vec<Complex<f64>> {
        let mut dxs = vec![Complex::new(1.0, 0.0); self.dim];
        if self.k.abs() < 1e-30 { return dxs; }
        for d in 0..self.dim {
            if x[d] >= self.comp_domain_bdr[d][1] {
                let dist = x[d] - self.comp_domain_bdr[d][1];
                let len = self.length[d][1];
                if len > 0.0 {
                    dxs[d] = Complex::new(1.0, 10.0 * dist / (self.k * len * len));
                }
            } else if x[d] <= self.comp_domain_bdr[d][0] {
                let dist = self.comp_domain_bdr[d][0] - x[d];
                let len = self.length[d][0];
                if len > 0.0 {
                    dxs[d] = Complex::new(1.0, 10.0 * dist / (self.k * len * len));
                }
            }
        }
        dxs
    }

    fn det_j(&self, dxs: &[Complex<f64>]) -> Complex<f64> {
        let mut det = Complex::new(1.0, 0.0);
        for d in 0..self.dim { det *= dxs[d]; }
        det
    }

    fn coeffs_at(&self, x: &[f64], dim: usize) -> [[f64; 3]; 6] {
        let dxs = self.stretch(x);
        let det = self.det_j(&dxs);
        let mut c = [[0.0_f64; 3]; 6];
        if dim == 2 {
            // 2D: curl-curl is SCALAR = 1/det
            c[0][0] = (1.0 / det).re; c[1][0] = (1.0 / det).im; c[2][0] = (1.0 / det).norm();
            // 2D: mass is VECTOR = det/dxs[d]²
            for d in 0..2 {
                let v = det / (dxs[d] * dxs[d]);
                c[3][d] = v.re; c[4][d] = v.im; c[5][d] = v.norm();
            }
        } else {
            for d in 0..3 {
                let cc = dxs[d] * dxs[d] / det;
                c[0][d] = cc.re; c[1][d] = cc.im; c[2][d] = cc.norm();
                let m = det / (dxs[d] * dxs[d]);
                c[3][d] = m.re; c[4][d] = m.im; c[5][d] = m.norm();
            }
        }
        c
    }
}

// ─── 2D curl-curl: scalar coefficients ────────────────────────────────────

struct CurlScalarRe { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for CurlScalarRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        self.pml.coeffs_at(ctx.x, ctx.dim)[0][0]
    }
}
struct CurlScalarIm { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for CurlScalarIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        self.pml.coeffs_at(ctx.x, ctx.dim)[1][0]
    }
}
struct CurlScalarAbs { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for CurlScalarAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        self.pml.coeffs_at(ctx.x, ctx.dim)[2][0]
    }
}

// ─── Diagonal matrix coefficients (for mass in 2D/3D and curl in 3D) ───────

macro_rules! diag_coeff {
    ($name:ident, $idx:expr) => {
        struct $name { pml: std::sync::Arc<PmlParams> }
        impl MatrixCoeff for $name {
            fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut[f64]) {
                for v in out.iter_mut() { *v = 0.0; }
                let d = ctx.dim;
                if ctx.elem_tag == 1 {
                    for i in 0..d { out[i*(d+1)] = 1.0; } return;
                }
                let arr = self.pml.coeffs_at(ctx.x, d);
                for i in 0..d { out[i*(d+1)] = arr[$idx][i]; }
            }
        }
    };
}

diag_coeff!(CurlMatRe, 0);
diag_coeff!(CurlMatIm, 1);
diag_coeff!(CurlMatAbs, 2);
diag_coeff!(MassMatRe, 3);
diag_coeff!(MassMatIm, 4);
diag_coeff!(MassMatAbs, 5);

// ═══════════════════════════════════════════════════════════════════════════
// Exact solutions
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum Prob { Beam, Disc, Lshape, Fichera, LoadSrc }
const ZI: Complex<f64> = Complex::new(0.0, 1.0);

fn maxwell_solution(x: &[f64], dim: usize, prob: Prob, k: f64) -> Vec<Complex<f64>> {
    let mut e = vec![Complex::new(0.0, 0.0); dim];
    match prob {
        Prob::Disc | Prob::Lshape | Prob::Fichera => {
            let shift = match prob {
                Prob::Fichera => [1.0, 1.0, 1.0],
                Prob::Disc => [-0.5, -0.5, 0.0],
                Prob::Lshape => [-1.0, -1.0, 0.0],
                _ => [0.0; 3],
            };
            if dim == 2 {
                let x0 = x[0] + shift[0]; let x1 = x[1] + shift[1];
                let r = (x0*x0 + x1*x1).sqrt();
                let beta = k * r;
                if r < 1e-14 { e[0] = -ZI * ZI * 0.25; return e; }
                let (j0,j1,j2,y0,y1,y2) = (bessel_j0(beta),bessel_j1(beta),bessel_j2(beta),
                                            bessel_y0(beta),bessel_y1(beta),bessel_y2(beta));
                let ho = Complex::new(j0, y0);
                let ho_r = -k * Complex::new(j1, y1);
                let ho_rr = -k*k * (Complex::new(j1, y1)/beta - Complex::new(j2, y2));
                let r_x = x0/r; let r_y = x1/r;
                let r_xy = -(r_x/r)*r_y; let r_xx = (1.0/r)*(1.0 - r_x*r_x);
                let val = 0.25*ZI*ho;
                let val_xx = 0.25*ZI*(r_xx*ho_r + r_x*r_x*ho_rr);
                let val_xy = 0.25*ZI*(r_xy*ho_r + r_x*r_y*ho_rr);
                e[0] = ZI/k * (k*k*val + val_xx);
                e[1] = ZI/k * val_xy;
            } else {
                let x0 = x[0]+shift[0]; let x1 = x[1]+shift[1]; let x2 = x[2]+shift[2];
                let r = (x0*x0 + x1*x1 + x2*x2).sqrt();
                if r < 1e-14 { return e; }
                let (rx,ry,rz) = (x0/r, x1/r, x2/r);
                let val = (ZI*k*r).exp()/r;
                let vr = val/r*(ZI*k*r - 1.0);
                let vrr = val/(r*r)*(-k*k*r*r - 2.0*ZI*k*r + 2.0);
                let vxx = vrr*rx*rx + vr*(1.0/r)*(1.0 - rx*rx);
                let vyx = vrr*rx*ry + vr*(-(ry/r)*rx);
                let vzx = vrr*rx*rz + vr*(-(rz/r)*rx);
                let a = ZI*k/(4.0*PI*k*k);
                e[0] = a*(k*k*val + vxx); e[1] = a*vyx; e[2] = a*vzx;
            }
        }
        Prob::Beam => {
            if dim == 3 { e[1] = -ZI*k/PI*(PI*x[2]).sin()*(ZI*(k*k-PI*PI).sqrt()*x[0]).exp(); }
            else { e[1] = -ZI*k/PI*(ZI*k*x[0]).exp(); }
        }
        Prob::LoadSrc => {}
    }
    e
}

fn source_fn(x: &[f64], dim: usize, comp_bdr: &[[f64; 2]], omega: f64) -> Vec<f64> {
    let mut center = vec![0.0; dim];
    for d in 0..dim { center[d] = 0.5*(comp_bdr[d][0] + comp_bdr[d][1]); }
    let r2: f64 = (0..dim).map(|d| (x[d]-center[d]).powi(2)).sum();
    let n = 5.0*omega/PI;
    let mut f = vec![0.0; dim]; f[0] = n*n/PI*(-n*n*r2).exp(); f
}

// ═══════════════════════════════════════════════════════════════════════════
// Main solver
// ═══════════════════════════════════════════════════════════════════════════

fn solve_pml<M: MeshTopology + Clone>(mesh: M,
    args: &Args, prob: Prob, exact_known: bool, pml: std::sync::Arc<PmlParams>,
    bdr_tags: Vec<i32>) {

    let dim = mesh.dim() as usize;
    let omega = 2.0*PI*args.freq;
    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);
    println!("  Mode: {}D", dim);
    let qo = (2*args.order + 1) as u8;
    let omega2 = -omega*omega;

    // ── Essential BC ──────────────────────────────────────────────────────
    let ess_tdofs = boundary_dofs_hcurl(space.mesh(), &space, &bdr_tags);

    // ── Assemble complex system ──────────────────────────────────────────
    // K_re + i·K_im where K = μ⁻¹·curlcurl − ω²·mass (with PML coeffs)
    // Assemble curl-curl and mass separately then combine, matching MFEM.
    let dim_flag = dim;
    let (mut k_re, mut k_im) = if dim_flag == 2 {
        let cc_re = CurlCurlIntegrator { mu: CurlScalarRe { pml: pml.clone() } };
        let cc_im = CurlCurlIntegrator { mu: CurlScalarIm { pml: pml.clone() } };
        let mr  = VectorMassTensorIntegrator { alpha: MassMatRe { pml: pml.clone() } };
        let mi  = VectorMassTensorIntegrator { alpha: MassMatIm { pml: pml.clone() } };
        let ccr = VectorAssembler::assemble_bilinear(&space, &[&cc_re], qo);
        let cci = VectorAssembler::assemble_bilinear(&space, &[&cc_im], qo);
        let mre = VectorAssembler::assemble_bilinear(&space, &[&mr], qo);
        let mim = VectorAssembler::assemble_bilinear(&space, &[&mi], qo);
        (ccr.axpby(1.0, &mre, omega2), cci.axpby(1.0, &mim, omega2))
    } else {
        let cc_re = CurlCurlTensorIntegrator { mu: CurlMatRe { pml: pml.clone() } };
        let cc_im = CurlCurlTensorIntegrator { mu: CurlMatIm { pml: pml.clone() } };
        let mr  = VectorMassTensorIntegrator { alpha: MassMatRe { pml: pml.clone() } };
        let mi  = VectorMassTensorIntegrator { alpha: MassMatIm { pml: pml.clone() } };
        let ccr = VectorAssembler::assemble_bilinear(&space, &[&cc_re], qo);
        let cci = VectorAssembler::assemble_bilinear(&space, &[&cc_im], qo);
        let mre = VectorAssembler::assemble_bilinear(&space, &[&mr], qo);
        let mim = VectorAssembler::assemble_bilinear(&space, &[&mi], qo);
        (ccr.axpby(1.0, &mre, omega2), cci.axpby(1.0, &mim, omega2))
    };

    // Apply BC: K_re = I, K_im = 0 on essential DOFs
    for &d in &ess_tdofs {
        let d = d as usize;
        for p in k_re.row_ptr[d]..k_re.row_ptr[d+1] {
            let c = k_re.col_idx[p] as usize;
            k_re.values[p] = if c == d { 1.0 } else { 0.0 };
        }
        for p in k_im.row_ptr[d]..k_im.row_ptr[d+1] { k_im.values[p] = 0.0; }
    }

    // ── RHS / BC projection ────────────────────────────────────────────────
    let mut rhs_re = vec![0.0; n];
    let mut rhs_im = vec![0.0; n];

    if prob == Prob::LoadSrc {
        let comp_bdr = pml.comp_domain_bdr.clone();
        struct SrcFn { omega: f64, comp_bdr: Vec<[f64; 2]> }
        impl VectorLinearIntegrator for SrcFn {
            fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
                let f = source_fn(qp.x_phys, qp.dim, &self.comp_bdr, self.omega);
                for i in 0..qp.n_dofs {
                    let mut s = 0.0;
                    for d in 0..qp.dim { s += qp.phi_vec[i*qp.dim+d] * f[d]; }
                    fe[i] += qp.weight * s;
                }
            }
        }
        let vec = VectorAssembler::assemble_linear(&space, &[&SrcFn { omega, comp_bdr }], qo);
        rhs_re.copy_from_slice(&vec);
    }

    // Project exact solution onto essential DOFs (BC)
    if !ess_tdofs.is_empty() && exact_known {
        let k = omega;
        let f_re = |x: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.re).collect()
        };
        let f_im = |x: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.im).collect()
        };
        let bc_re = space.interpolate_vector(&f_re);
        let bc_im = space.interpolate_vector(&f_im);
        for &d in &ess_tdofs {
            rhs_re[d as usize] = bc_re[d as usize];
            rhs_im[d as usize] = bc_im[d as usize];
        }
    }

    // Build [K_re -K_im; K_im K_re]
    let mut coo = CooMatrix::new(2*n, 2*n);
    for i in 0..n {
        for p in k_re.row_ptr[i]..k_re.row_ptr[i+1] {
            let j=k_re.col_idx[p]as usize; let v=k_re.values[p];
            if v != 0.0 { coo.add(i,j,v); coo.add(n+i,n+j,v); }
        }
        for p in k_im.row_ptr[i]..k_im.row_ptr[i+1] {
            let j=k_im.col_idx[p]as usize; let v=k_im.values[p];
            if v != 0.0 { coo.add(i,n+j,-v); coo.add(n+i,j,v); }
        }
    }
    let a = coo.into_csr();

    // ── Preconditioner ───────────────────────────────────────────────────
    let prec = if dim == 2 {
        let cc_abs  = VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlIntegrator{mu:CurlScalarAbs{pml:pml.clone()}}], qo);
        let mass_abs = VectorAssembler::assemble_bilinear(&space,
            &[&VectorMassTensorIntegrator{alpha:MassMatAbs{pml:pml.clone()}}], qo);
        cc_abs.axpby(1.0, &mass_abs, omega*omega)
    } else {
        let cc_abs  = VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlTensorIntegrator{mu:CurlMatAbs{pml:pml.clone()}}], qo);
        let mass_abs = VectorAssembler::assemble_bilinear(&space,
            &[&VectorMassTensorIntegrator{alpha:MassMatAbs{pml}}], qo);
        cc_abs.axpby(1.0, &mass_abs, omega*omega)
    };
    let mut shift = CooMatrix::new(n, n);
    for i in 0..n { shift.add(i,i,1e-6); }
    let prec_mat = CsrMatrix::add(&prec, &shift.into_csr());
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&prec_mat), 1.0).expect("GSSmoother");
    let bp = BlockDiagPrecond { inner: gs, n };

    // ── Solve ────────────────────────────────────────────────────────────
    let mut flat_rhs = vec![0.0_f64; 2*n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; }
    for i in 0..n { flat_rhs[n+i] = rhs_im[i]; }
    let mut x = vec![0.0; 2*n];
    let res = fem_solver::solve_gmres_precond(&a, &flat_rhs, &mut x, 200, &bp,
        &SolverConfig { rtol:1e-3, max_iter:2000, verbose:true, ..Default::default() })
        .expect("GMRES");
    println!("  GMRES converged in {} iters, final residual = {:.6e}",
             res.iterations, res.final_residual);

    // ── Error computation ────────────────────────────────────────────────
    if exact_known {
        let qe = std::cmp::max(2, 2*args.order + 1) as u8;
        let total_err2 = compute_l2_error_vector(&space, &x, dim, prob, omega, qe, Some(2));
        let zero = vec![0.0; 2*n];
        let exact_norm2 = compute_l2_error_vector(&space, &zero, dim, prob, omega, qe, Some(2));
        println!("\n Relative Error (L²): || E_h - E || / ||E|| = {:.6e}",
                 total_err2.sqrt() / exact_norm2.sqrt().max(1e-30));
        println!(" Total Error: {:.6e}", total_err2.sqrt());
    }

    let sol_norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}\n", sol_norm);
}

// ─── Error computation helpers ────────────────────────────────────────────

use fem_element::ReferenceElement;
use fem_element::lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3, QuadQ1, QuadQ2, HexQ1, HexQ2};
use nalgebra::DMatrix;

fn ref_elem_for(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) | (ElementType::Quad9, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) | (ElementType::Quad9, 2) => Box::new(QuadQ2),
        (ElementType::Tet4, 1) | (ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) | (ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) | (ElementType::Tet10, 3) => Box::new(TetP3),
        (ElementType::Hex8, 1) | (ElementType::Hex27, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) | (ElementType::Hex27, 2) => Box::new(HexQ2),
        _ => panic!("ref_elem: ({et:?}, order={order})"),
    }
}

fn elem_jacobian<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
    }
    (j.clone(), j.determinant())
}

fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {
    let mut xp = x0.to_vec();
    for i in 0..dim { for k in 0..dim { xp[i] += j[(i,k)] * xi[k]; } }
    xp
}

#[allow(dead_code)]
fn compute_l2_error_vector<M: MeshTopology>(
    space: &HCurlSpace<M>, x: &[f64], dim: usize, prob: Prob, k: f64, qo: u8,
    exclude_tag: Option<i32>) -> f64 {
    let mesh = space.mesh(); let order = space.order();
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        if exclude_tag.map_or(false, |et| mesh.element_tag(e) == et) { continue; }
        let re = ref_elem_for(mesh.element_type(e), order);
        let n_ldofs = re.n_dofs(); let quad = re.quadrature(qo);
        let dofs = space.element_dofs(e); let nodes = mesh.element_nodes(e);
        let (jac, det_j) = elem_jacobian(mesh, nodes, dim);
        let x0 = mesh.node_coords(nodes[0]);
        let mut phi = vec![0.0; n_ldofs];
        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            let mut uh = vec![0.0; dim];
            for i in 0..n_ldofs { for d in 0..dim { uh[d] += x[dofs[i] as usize] * phi[i]; } }
            let xp = phys_coords(x0, &jac, xi, dim);
            let exact = maxwell_solution(&xp, dim, prob, k);
            let diff2: f64 = (0..dim).map(|d| (uh[d] - exact[d].re).powi(2)).sum();
            err2 += w * diff2;
        }
    }
    err2
}



// ═══════════════════════════════════════════════════════════════════════════
// Mesh / PML tagging
// ═══════════════════════════════════════════════════════════════════════════

fn tag_pml<const D: usize>(mesh: &mut Mesh<D>, pml_lo: &[f64; 3], pml_hi: &[f64; 3]) {
    let bb = mesh.bounding_box();
    for e in 0..mesh.n_elems() as u32 {
        let mut in_pml = false;
        for &v in mesh.elem_nodes(e) {
            let c = mesh.node_coords(v);
            for d in 0..D {
                if c[d] < bb.0[d] + pml_lo[d] || c[d] > bb.1[d] - pml_hi[d] { in_pml = true; break; }
            }
            if in_pml { break; }
        }
        if in_pml { mesh.elem_tags[e as usize] = 2; }
    }
}

fn pml_vals(prob: &Prob) -> ([f64; 3], [f64; 3]) {
    match *prob {
        Prob::Beam    => ([0.0,0.0,0.0], [2.0,0.0,0.0]),
        Prob::Disc    => ([0.2,0.2,0.0], [0.2,0.2,0.0]),
        Prob::Lshape  => ([0.1,0.1,0.0], [0.0,0.0,0.0]),
        Prob::Fichera => ([0.0,0.0,0.0], [0.5,0.5,0.5]),
        Prob::LoadSrc => ([0.25,0.25,0.0], [0.25,0.25,0.0]),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

struct Args { mesh: Option<String>, order: i32, ref_levels: i32, iprob: i32, freq: f64 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, ref_levels: 3, iprob: 4, freq: 5.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = Some(it.next().unwrap_or_default()),
            "-o"|"--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-ref"|"--refinements" => a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-prob"|"--problem" => a.iprob = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "-f"|"--frequency" => a.freq = it.next().and_then(|s| s.parse().ok()).unwrap_or(5.0),
            _ => {}
        }
    }
    a
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args = parse_args();
    let prob = match args.iprob.min(4) { 0=>Prob::Beam, 1=>Prob::Disc, 2=>Prob::Lshape, 3=>Prob::Fichera, _=>Prob::LoadSrc };
    let exact_known = matches!(prob, Prob::Beam|Prob::Disc|Prob::Lshape|Prob::Fichera);
    let mesh_file = args.mesh.as_deref().unwrap_or(match prob {
        Prob::Beam=>"data/beam-quad.mesh", Prob::Disc=>"data/square-disc.mesh",
        Prob::Lshape=>"data/l-shape.mesh", Prob::Fichera=>"data/fichera.mesh",
        Prob::LoadSrc=>"data/inline-quad.mesh",
    });
    println!("--mesh {mesh_file} --order {} --prob {} --refinements {} --freq {}",
             args.order, args.iprob.min(4), args.ref_levels, args.freq);
    let mfem_data = mfem::read_mfem_file(mesh_file).expect("mesh");
    let dim = if mfem_data.mesh3d.is_some() { 3 } else { 2 };
    let (pml_lo, pml_hi) = pml_vals(&prob);
    let omega = 2.0*PI*args.freq;

    if dim == 2 {
        let mut mesh: Mesh<2> = mfem_data.mesh2d.expect("2D");
        for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }
        let bb = mesh.bounding_box();
        let bdr_tags = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &pml_lo, &pml_hi);
        let pml = std::sync::Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, omega, 2));
        solve_pml(mesh, &args, prob, exact_known, pml, bdr_tags);
    } else {
        let mut mesh: Mesh<3> = mfem_data.mesh3d.expect("3D");
        for _ in 0..args.ref_levels { mesh = fem_mesh::refine_uniform_3d(&mesh); }
        let bb = mesh.bounding_box();
        let bdr_tags = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &pml_lo, &pml_hi);
        let pml = std::sync::Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, omega, 3));
        solve_pml(mesh, &args, prob, exact_known, pml, bdr_tags);
    }
}
