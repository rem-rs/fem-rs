//! # MFEM Example 25 — PML for Maxwell — 1:1 Rust translation
//!
//! Solves `(1/μ) curl curl E − ω² ε E = f` with PML.
//!
//! Uses core library coefficients (`RestrictedCoefficient`,
//! `ScalarVectorProductCoefficient`) to match C++ coefficient chain.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -o 2 -f 1.0 -ref 2 -prob 0
//! cargo run --example mfem_ex25_pml_maxwell -- -o 2 -f 5.0 -ref 3 -prob 4 -m data/inline-quad.mesh
//! ```

use std::f64::consts::PI;
use std::io::Write;
use std::net::TcpStream;
use nalgebra::Complex;
use fem_assembly::{
    VectorAssembler,
    postproc::coefficient::{CoeffCtx, ScalarCoeff, MatrixCoeff,
                            RestrictedCoefficient, ScalarVectorProductCoefficient,
                            VectorRestrictedCoefficient,
                            ProductCoeff, ScalarMatrixCoeff},
    postproc::grid_function::compute_l2_error_hcurl,
    standard::{CurlCurlIntegrator, CurlCurlTensorIntegrator,
               VectorMassTensorIntegrator},
    vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData},
};
use fem_io::mfem::{self, write_mfem};
use fem_linalg::SolverConfig;
use fem_solver::{GSSmoother, ScaledPrecond, BlockDiagPrecondPair};
use fem_linalg::fem_to_linlvo_csr;
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_mesh::refine_uniform;

// ═══════════════════════════════════════════════════════════════════════════
// Bessel functions (J₀, J₁, J₂, Y₀, Y₁, Y₂)
// ═══════════════════════════════════════════════════════════════════════════

// MSVC CRT Bessel functions (bit-identical to C++ ex25)
#[cfg(windows)]
extern "C" {
    fn _jn(n: i32, x: f64) -> f64;
    fn _yn(n: i32, x: f64) -> f64;
}
fn bessel_j0(x: f64) -> f64 { unsafe { _jn(0, x) } }
fn bessel_j1(x: f64) -> f64 { unsafe { _jn(1, x) } }
fn bessel_j2(x: f64) -> f64 { unsafe { _jn(2, x) } }
fn bessel_y0(x: f64) -> f64 { unsafe { _yn(0, x) } }
fn bessel_y1(x: f64) -> f64 { unsafe { _yn(1, x) } }
fn bessel_y2(x: f64) -> f64 { unsafe { _yn(2, x) } }

// ═══════════════════════════════════════════════════════════════════════════
// PML — 1:1 with MFEM ex25
// ═══════════════════════════════════════════════════════════════════════════

/// PML region parameters (equivalent to MFEM's `PML` class).
struct PmlParams {
    dim: usize,
    comp_domain_bdr: Vec<[f64; 2]>,
    length: Vec<[f64; 2]>,
    k: f64,
}

impl PmlParams {
    fn new(bb_lo: &[f64], bb_hi: &[f64], pml_lo: &[f64; 3], pml_hi: &[f64; 3], k: f64, dim: usize) -> Self {
        let mut comp = Vec::with_capacity(dim);
        let mut len = Vec::with_capacity(dim);
        for d in 0..dim {
            comp.push([bb_lo[d] + pml_lo[d], bb_hi[d] - pml_hi[d]]);
            len.push([pml_lo[d], pml_hi[d]]);
        }
        PmlParams { dim, comp_domain_bdr: comp, length: len, k }
    }

    /// Complex stretching s'(x) = 1 + i·σ(x)/k (equivalent to C++ StretchFunction).
    fn stretch(&self, x: &[f64]) -> Vec<Complex<f64>> {
        let mut dxs = vec![Complex::new(1.0, 0.0); self.dim];
        if self.k.abs() < 1e-30 { return dxs; }
        let n = 2.0;
        let c = 5.0;
        for d in 0..self.dim {
            if x[d] >= self.comp_domain_bdr[d][1] {
                let dist = x[d] - self.comp_domain_bdr[d][1];
                let len = self.length[d][1];
                if len > 0.0 {
                    let coeff = n * c / self.k / len.powf(n);
                    dxs[d] = Complex::new(1.0, coeff * dist.abs().powf(n - 1.0));
                }
            } else if x[d] <= self.comp_domain_bdr[d][0] {
                let dist = self.comp_domain_bdr[d][0] - x[d];
                let len = self.length[d][0];
                if len > 0.0 {
                    let coeff = n * c / self.k / len.powf(n);
                    dxs[d] = Complex::new(1.0, coeff * dist.abs().powf(n - 1.0));
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

    /// Compute PML stretch coefficients at a point.
    /// Returns [Re, Im, Abs] for curl-curl (index 0-2) and mass (index 3-5).
    /// - curl-curl stretch: 2D: 1/det(J); 3D: dxs[d]²/det(J) per component
    /// - mass stretch: det(J)/dxs[d]² per component
    fn coeffs_at(&self, x: &[f64], dim: usize) -> [[f64; 3]; 6] {
        let dxs = self.stretch(x);
        let det = self.det_j(&dxs);
        let mut c = [[0.0_f64; 3]; 6];
        if dim == 2 {
            // 2D: curl-curl is scalar = 1/det
            let inv_det = 1.0 / det;
            c[0][0] = inv_det.re; c[1][0] = inv_det.im; c[2][0] = inv_det.norm();
            // 2D: mass is vector = det/dxs[d]²
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

// ─── PML stretch as ScalarCoeff (2D curl-curl) ────────────────────────────

/// PML stretch for 2D curl-curl: scalar 1/det(J)
struct PmlCurlScalar { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalar {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.pml.coeffs_at(ctx.x, ctx.dim)[0][0]  // Re part
    }
}
struct PmlCurlScalarIm { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalarIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.pml.coeffs_at(ctx.x, ctx.dim)[1][0]  // Im part
    }
}
struct PmlCurlScalarAbs { pml: std::sync::Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalarAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.pml.coeffs_at(ctx.x, ctx.dim)[2][0]  // Abs
    }
}

// ─── PML stretch as MatrixCoeff (3D curl-curl, and mass in 2D/3D) ─────────

macro_rules! pml_matrix_coeff {
    ($name:ident, $idx:expr) => {
        struct $name { pml: std::sync::Arc<PmlParams> }
        impl MatrixCoeff for $name {
            fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut[f64]) {
                for v in out.iter_mut() { *v = 0.0; }
                let d = ctx.dim;
                let arr = self.pml.coeffs_at(ctx.x, d);
                for i in 0..d { out[i*(d+1)] = arr[$idx][i]; }
            }
        }
    };
}
// curl-curl: 3D uses diagonal matrix (index 0-2)
pml_matrix_coeff!(PmlCurlMatRe, 0);
pml_matrix_coeff!(PmlCurlMatIm, 1);
pml_matrix_coeff!(PmlCurlMatAbs, 2);
// mass: diagonal matrix (index 3-5)
pml_matrix_coeff!(PmlMassMatRe, 3);
pml_matrix_coeff!(PmlMassMatIm, 4);
pml_matrix_coeff!(PmlMassMatAbs, 5);

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

fn source_fn(x: &[f64], dim: usize, comp_bdr: &[[f64; 2]], omega: f64, eps: f64, mu: f64) -> Vec<f64> {
    let mut center = vec![0.0; dim];
    for d in 0..dim { center[d] = 0.5*(comp_bdr[d][0] + comp_bdr[d][1]); }
    let r2: f64 = (0..dim).map(|d| (x[d]-center[d]).powi(2)).sum();
    let n = 5.0*omega*(eps*mu).sqrt()/PI;
    let mut f = vec![0.0; dim]; f[0] = n*n/PI*(-n*n*r2).exp(); f
}

// ═══════════════════════════════════════════════════════════════════════════
// Main solver (1:1 with C++ ex25 main)
// ═══════════════════════════════════════════════════════════════════════════

fn solve_pml<M: MeshTopology + Clone>(mesh: M,
    args: &Args, prob: Prob, exact_known: bool, pml: std::sync::Arc<PmlParams>,
    bdr_tags: Vec<i32>, herm_conv: bool, visualization: bool, mesh_data: &[u8]) {

    let dim = mesh.dim() as usize;
    let omega = 2.0*PI*args.freq;
    let k = omega * (args.eps * args.mu).sqrt();  // wave number (matches C++)
    let mu_inv = 1.0 / args.mu;
    let omega2_eps = -omega * omega * args.eps;   // -ω²ε (negative for indefinite Maxwell)
    let abs_omega2_eps = omega * omega * args.eps; // +ω²ε (for preconditioner)

    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);
    println!("  Mode: {}D", dim);
    let qo = (2*args.order + 1) as u8;

    // ── Boundary conditions (1:1 with C++ ess_bdr logic) ─────────────────
    let ess_bdr_tags = if prob == Prob::Lshape || prob == Prob::Fichera {
        let mut ess = Vec::new();
        for f in 0..mesh.n_boundary_faces() as u32 {
            let tag = mesh.face_tag(f);
            if ess.contains(&tag) { continue; }
            let nodes = mesh.face_nodes(f);
            let mut center = vec![0.0; dim];
            for d in 0..dim {
                center[d] = nodes.iter().map(|&n| mesh.node_coords(n)[d]).sum::<f64>()
                    / nodes.len() as f64;
            }
            let should_constrain = match prob {
                Prob::Lshape => {
                    (center[0] - 1.0).abs() < 1e-8 || (center[0] - 0.5).abs() < 1e-8
                        || (center[1] - 0.5).abs() < 1e-8
                }
                Prob::Fichera => {
                    (center[0] + 1.0).abs() < 1e-8 || center[0].abs() < 1e-8
                        || center[1].abs() < 1e-8 || (dim > 2 && center[2].abs() < 1e-8)
                }
                _ => false,
            };
            if should_constrain { ess.push(tag); }
        }
        ess
    } else {
        bdr_tags.clone()
    };

    // ── Essential DOFs ───────────────────────────────────────────────────
    let ess_tdofs = boundary_dofs_hcurl(space.mesh(), &space, &ess_bdr_tags);

    // ── Assemble complex system via SesquilinearForm (1:1 with C++) ──────
    use fem_assembly::complex::{SesquilinearForm, Convention};
        let conv = if herm_conv { Convention::Hermitian } else { Convention::BlockSymmetric };

    let attr     = vec![1];  // computational domain (element tag 1)
    let attr_pml = vec![2];  // PML region (element tag 2)

    // ── Create named integrators (must outlive assemble()) ────────────
    let cc_nonpml = CurlCurlIntegrator {
        mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() }
    };
    let mass_nonpml = VectorMassTensorIntegrator {
        alpha: VectorRestrictedCoefficient {
            inner: ScalarMatrixCoeff(omega2_eps),
            attrs: attr.clone(),
        }
    };

    // PML curl-curl: 2D scalar or 3D tensor — boxed for uniform handling
    let (pml_cc_re, pml_cc_im): (Box<dyn VectorBilinearIntegrator>, Box<dyn VectorBilinearIntegrator>) = if dim == 2 {
        (Box::new(CurlCurlIntegrator {
            mu: RestrictedCoefficient {
                inner: PmlCurlScalar { pml: pml.clone() },
                attrs: attr_pml.clone(),
            }
        }),
        Box::new(CurlCurlIntegrator {
            mu: RestrictedCoefficient {
                inner: PmlCurlScalarIm { pml: pml.clone() },
                attrs: attr_pml.clone(),
            }
        }))
    } else {
        (Box::new(CurlCurlTensorIntegrator {
            mu: ScalarVectorProductCoefficient {
                scalar: mu_inv,
                vector: VectorRestrictedCoefficient {
                    inner: PmlCurlMatRe { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        }),
        Box::new(CurlCurlTensorIntegrator {
            mu: ScalarVectorProductCoefficient {
                scalar: mu_inv,
                vector: VectorRestrictedCoefficient {
                    inner: PmlCurlMatIm { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        }))
    };

    let pml_mass_re = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatRe { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        }
    };
    let pml_mass_im = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatIm { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        }
    };

    // ── Assemble with per-integrator quadrature (matching C++) ───────
    // C++ CurlCurlIntegrator for ND_TriangleElement (Pk):
    //   order = 2*GetOrder() - 2 = 0 (1-point centroid rule)
    // C++ VectorFEMassIntegrator:
    //   order = Trans.OrderW() + 2*GetOrder() = 3 (4-point rule)
    // Rust's tri_rule(3) gives 3-point (same integral as 4-point for polynomials)
    let qo_mass = (2*args.order + 1) as u8;  // matches C++ mass order
    let qo_curl = 0u8;  // matches C++ curl-curl for Pk triangle

    // ── Dump element DOFs and signs ──────────────────────────────────
    {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("elem_dofs_rust.txt").unwrap();
        for e in 0..space.mesh().n_elements() as u32 {
            let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = space.element_signs(e);
            let etag = space.mesh().element_tag(e);
            write!(f, "elem {e} attr={etag} dofs=").unwrap();
            for (i, &d) in dofs.iter().enumerate() {
                let s = if i < signs.len() { signs[i] } else { 1.0 };
                // Mimic C++ signed encoding: positive=+dof, negative=-(dof+1)
                let signed_dof = if s > 0.0 { d as i32 } else { -(d as i32 + 1) };
                write!(f, "{} ", signed_dof).unwrap();
            }
            writeln!(f).unwrap();
        }
    }

    let mass_re = VectorAssembler::assemble_bilinear(&space, &[&mass_nonpml, &pml_mass_re], qo_mass);
    let mass_im = VectorAssembler::assemble_bilinear(&space, &[&pml_mass_im], qo_mass);
    let cc_re = VectorAssembler::assemble_bilinear(&space, &[&cc_nonpml, pml_cc_re.as_ref()], qo_curl);
    let cc_im = VectorAssembler::assemble_bilinear(&space, &[pml_cc_im.as_ref()], qo_curl);

    use fem_linalg::spadd;
    let k_re = spadd(&mass_re, &cc_re);
    let k_im = spadd(&mass_im, &cc_im);
    let mut cs = fem_assembly::complex::ComplexSystem { k_re, k_im, omega: 0.0 };

    // ── RHS ──────────────────────────────────────────────────────────────
    let mut rhs_re = vec![0.0; n];
    let mut rhs_im = vec![0.0; n];
    if prob == Prob::LoadSrc {
        let comp_bdr: Vec<[f64; 2]> = (0..dim).map(|d| pml.comp_domain_bdr[d]).collect();
        let src_fn = |x: &[f64], _ctx: &VectorQpData<'_>| -> Vec<f64> {
            source_fn(x, dim, &comp_bdr, omega, args.eps, args.mu)
        };
        let vec = VectorAssembler::assemble_linear(&space, &[&VectorSrc { f: &src_fn }], qo);
        rhs_im.copy_from_slice(&vec);
    }

    // ── Project BC (1:1 with C++ E_bdr_data_Re/Im + ProjectBdrCoefficientTangent) ──
    let mut bc_re = vec![0.0; n];
    let mut bc_im = vec![0.0; n];
    if !ess_tdofs.is_empty() && exact_known {
        let pml_ref = pml.clone();
        let bc_fn_re = move |x: &[f64]| -> Vec<f64> {
            for d in 0..dim {
                if x[d] > pml_ref.comp_domain_bdr[d][1] || x[d] < pml_ref.comp_domain_bdr[d][0] {
                    return vec![0.0; dim];
                }
            }
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.re).collect()
        };
        let pml_ref = pml.clone();
        let bc_fn_im = move |x: &[f64]| -> Vec<f64> {
            for d in 0..dim {
                if x[d] > pml_ref.comp_domain_bdr[d][1] || x[d] < pml_ref.comp_domain_bdr[d][0] {
                    return vec![0.0; dim];
                }
            }
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.im).collect()
        };
        let q_bdr = std::cmp::max(2, 2*args.order + 1) as u8;
        let bc_re_full = space.interpolate_vector(&bc_fn_re);
        let bc_im_full = space.interpolate_vector(&bc_fn_im);
        bc_re.copy_from_slice(bc_re_full.as_slice());
        bc_im.copy_from_slice(bc_im_full.as_slice());
    }

    let ess_u: Vec<usize> = ess_tdofs.iter().map(|&d| d as usize).collect();
    let bc_re_ess: Vec<f64> = ess_tdofs.iter().map(|&d| bc_re[d as usize]).collect();
    let bc_im_ess: Vec<f64> = ess_tdofs.iter().map(|&d| bc_im[d as usize]).collect();
    // Build flat system then apply BC (C++ FormLinearSystem equivalent)
    let mut flat_rhs_init = vec![0.0_f64; 2*n];
    for i in 0..n { flat_rhs_init[i] = rhs_re[i]; flat_rhs_init[n+i] = rhs_im[i]; }
    cs.apply_dirichlet(&ess_u, &bc_re_ess, &bc_im_ess, &mut flat_rhs_init);
    for i in 0..n { rhs_re[i] = flat_rhs_init[i]; rhs_im[i] = flat_rhs_init[n+i]; }

    // ── TEMP: dump flat matrix and RHS for C++ comparison ────────────
    let a_mat_check = cs.to_flat_csr_with_conv(conv);
    {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("flat_rhs_all.txt").unwrap();
        for i in 0..n.min(5) { writeln!(f, "rhs_re[{i}] = {:.15e}", rhs_re[i]).unwrap(); }
        for i in 0..n.min(5) { writeln!(f, "rhs_im[{i}] = {:.15e}", rhs_im[i]).unwrap(); }
        // Check essential DOFs 58, 61, 63
        for &d in &[58usize, 61, 63] {
            writeln!(f, "rhs_re[{d}] = {:.15e}", rhs_re[d]).unwrap();
            writeln!(f, "rhs_im[{d}] = {:.15e}", rhs_im[d]).unwrap();
            // check flat system rows
            let s = a_mat_check.row_ptr[d]; let e = a_mat_check.row_ptr[d+1];
            for p in s..e {
                writeln!(f, "  flat({}, {}) = {:.15e}", d, a_mat_check.col_idx[p], a_mat_check.values[p]).unwrap();
            }
        }
    }

        // ── Preconditioner — single-pass assembly (1:1 with C++ BilinearForm) ────
    // Non-PML: μ⁻¹·curlcurl + ω²ε·mass (coefficients baked into integrator)
    // PML:     μ⁻¹·|stretch|·curlcurl + ω²ε·|stretch|·mass
    let cc_nonpml_prec = CurlCurlIntegrator {
        mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() }
    };
    let mass_nonpml_prec = VectorMassTensorIntegrator {
        alpha: VectorRestrictedCoefficient {
            inner: ScalarMatrixCoeff(abs_omega2_eps),
            attrs: attr.clone(),
        }
    };
    let pml_mass_abs_prec = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: abs_omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatAbs { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        }
    };

    // PML curl-curl: 2D scalar or 3D tensor
    let pml_cc_abs_prec: Box<dyn VectorBilinearIntegrator> = if dim == 2 {
        Box::new(CurlCurlIntegrator {
            mu: ProductCoeff {
                a: mu_inv,
                b: RestrictedCoefficient {
                    inner: PmlCurlScalarAbs { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        })
    } else {
        Box::new(CurlCurlTensorIntegrator {
            mu: ScalarVectorProductCoefficient {
                scalar: mu_inv,
                vector: VectorRestrictedCoefficient {
                    inner: PmlCurlMatAbs { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        })
    };

    // ── Flat GMRES (converges to machine precision) ────────────────────
    let a_mat = cs.to_flat_csr_with_conv(conv);
    let mut flat_rhs = vec![0.0; 2*n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; flat_rhs[n+i] = rhs_im[i]; }
    let mut x = vec![0.0; 2*n];
    let res = fem_solver::solve_gmres(&a_mat, &flat_rhs, &mut x, 500,
        &SolverConfig { rtol:1e-12, max_iter:10000, verbose:false, ..Default::default() });
    match &res {
        Ok(r) => println!("  GMRES converged in {} iters, final residual = {:.6e}",
                         r.iterations, r.final_residual),
        Err(e) => println!("  GMRES: {e}"),
    }

    // ── Error computation (1:1 with C++ L2 error via ComputeL2Error) ────
    if exact_known {
        let qe = std::cmp::max(2, 2*args.order + 1) as u8;
        let ne = space.mesh().n_elements() as usize;
        let exclude: Vec<bool> = (0..ne)
            .map(|e| space.mesh().element_tag(e as u32) == 2).collect();

        let exact_re = |xp: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(xp, dim, prob, k);
            (0..dim).map(|d| e[d].re).collect()
        };
        let exact_im = |xp: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(xp, dim, prob, k);
            (0..dim).map(|d| e[d].im).collect()
        };

        let l2err_re = compute_l2_error_hcurl(&x[..n], &space, &exact_re, qe, Some(&exclude));
        let l2err_im = compute_l2_error_hcurl(&x[n..], &space, &exact_im, qe, Some(&exclude));

        let zero = vec![0.0; n];
        let norm_re = compute_l2_error_hcurl(&zero, &space, &exact_re, qe, Some(&exclude)).max(1e-30);
        let norm_im = compute_l2_error_hcurl(&zero, &space, &exact_im, qe, Some(&exclude)).max(1e-30);

        println!("\n Relative Error (Re part): || E_h - E || / ||E|| = {:.6e}",
                 l2err_re / norm_re);
        println!(" Relative Error (Im part): || E_h - E || / ||E|| = {:.6e}",
                 l2err_im / norm_im);
        println!(" Total Error: {:.6e}", (l2err_re*l2err_re + l2err_im*l2err_im).sqrt());
    }

    // ── Output ──────────────────────────────────────────────────────────
    let sol_norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}\n", sol_norm);

    let _ = fem_io::mfem::write_gf_file("ex25-sol_r.gf", dim, &x[..n], "ND", args.order as u8, dim);
    let _ = fem_io::mfem::write_gf_file("ex25-sol_i.gf", dim, &x[n..], "ND", args.order as u8, dim);
    println!("  Wrote ex25-sol_r.gf, ex25-sol_i.gf");

    // ── GLVis visualization (1:1 with C++ Section 17) ────────────────────
    if visualization {
        let keys = if dim == 3 {
            if prob == Prob::Beam { "keys macFFiYYYYYYYYYYYYYYYYYY\n" }
            else { "keys macF\n" }
        } else {
            if prob == Prob::Beam { "keys amrRljcUUuuu\n" }
            else { "keys amrRljcUUuu\n" }
        };

        let glvis_send = |stream: &mut TcpStream, dofs: &[f64], title: &str| -> std::io::Result<()> {
            write!(stream, "solution\n")?;
            stream.write_all(mesh_data)?;
            writeln!(stream, "FiniteElementSpace")?;
            writeln!(stream, "FiniteElementCollection: ND_{dim}D_P{}", args.order)?;
            writeln!(stream, "VDim: {dim}")?;
            writeln!(stream, "Ordering: 1")?;
            writeln!(stream)?;
            for v in dofs { writeln!(stream, "{:.7e}", v)?; }
            write!(stream, "{keys}")?;
            writeln!(stream, "window_title '{title}'")?;
            stream.flush()
        };

        if let Ok(mut sock) = TcpStream::connect("localhost:19916") {
            let _ = glvis_send(&mut sock, &x[..n], "Solution real part");
        }
        if let Ok(mut sock) = TcpStream::connect("localhost:19916") {
            let _ = glvis_send(&mut sock, &x[n..], "Solution imag part");
        }

        let mut x_t = vec![0.0; n];
        if let Ok(mut sock) = TcpStream::connect("localhost:19916") {
            for i in 0..n { x_t[i] = x[i]; }
            let _ = glvis_send(&mut sock, &x_t, "Harmonic Solution (t = 0.0 T)");
            let _ = writeln!(sock, "pause\n");
            println!("GLVis visualization paused. Press space (in the GLVis window) to resume it.");
            for i in 0..32 {
                if sock.peer_addr().is_err() { break; }
                let t = (i as f64) / 32.0;
                let ct = (2.0 * PI * t).cos();
                let st = (2.0 * PI * t).sin();
                for j in 0..n { x_t[j] = ct * x[j] + st * x[j + n]; }
                if glvis_send(&mut sock, &x_t, &format!("Harmonic Solution (t = {:.3} T)", t)).is_err() { break; }
            }
        }
    }
}

// ─── Vector source integrator ─────────────────────────────────────────────

struct VectorSrc<'a> {
    f: &'a (dyn Fn(&[f64], &VectorQpData<'_>) -> Vec<f64> + Send + Sync),
}
impl VectorLinearIntegrator for VectorSrc<'_> {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let f_val = (self.f)(qp.x_phys, qp);
        for i in 0..qp.n_dofs {
            for d in 0..qp.dim {
                f_elem[i * qp.dim + d] += qp.weight * f_val[d] * qp.phi_vec[i * qp.dim + d];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mesh / PML tagging (1:1 with C++ PML::SetAttributes)
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
// CLI (1:1 with C++ OptionsParser)
// ═══════════════════════════════════════════════════════════════════════════

struct Args {
    mesh: Option<String>,
    order: i32,
    ref_levels: i32,
    iprob: i32,
    freq: f64,
    mu: f64,
    eps: f64,
    herm_conv: bool,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, ref_levels: 3, iprob: 4, freq: 5.0, mu: 1.0, eps: 1.0, herm_conv: true, visualization: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = Some(it.next().unwrap_or_default()),
            "-o"|"--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-ref"|"--refinements" => a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-prob"|"--problem" => a.iprob = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "-f"|"--frequency" => a.freq = it.next().and_then(|s| s.parse().ok()).unwrap_or(5.0),
            "-mu"|"--permeability" => a.mu = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-eps"|"--permittivity" => a.eps = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-herm"|"--hermitian" => a.herm_conv = true,
            "-no-herm"|"--no-hermitian" => a.herm_conv = false,
            "-vis"|"--visualization" => a.visualization = true,
            "-no-vis"|"--no-visualization" => a.visualization = false,
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
        Prob::Beam=>"data/beam-hex.mesh", Prob::Disc=>"data/square-disc.mesh",
        Prob::Lshape=>"data/l-shape.mesh", Prob::Fichera=>"data/fichera.mesh",
        Prob::LoadSrc=>"data/inline-quad.mesh",
    });
    println!("--mesh {mesh_file} --order {} --prob {} --refinements {} --freq {}",
             args.order, args.iprob.min(4), args.ref_levels, args.freq);
    let mfem_data = mfem::read_mfem_file(mesh_file).expect("mesh");
    let dim = if mfem_data.mesh3d.is_some() { 3 } else { 2 };
    let (pml_lo, pml_hi) = pml_vals(&prob);
    let omega = 2.0*PI*args.freq;
    let k = omega * (args.eps * args.mu).sqrt();

    if dim == 2 {
        let mut mesh: Mesh<2> = mfem_data.mesh2d.expect("2D");
        for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }
        let bb = mesh.bounding_box();
        let bdr_tags = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &pml_lo, &pml_hi);
        let pml = std::sync::Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, k, 2));
        let mut mesh_buf = Vec::new();
        write_mfem(&mut mesh_buf, &mesh, None).unwrap();
        solve_pml(mesh.clone(), &args, prob, exact_known, pml, bdr_tags,
            args.herm_conv, args.visualization, &mesh_buf);
        let _ = fem_io::mfem::write_mfem_file("ex25.mesh", &mesh);
        println!("  Wrote ex25.mesh");
    } else {
        let mut mesh: Mesh<3> = mfem_data.mesh3d.expect("3D");
        for _ in 0..args.ref_levels { mesh = fem_mesh::refine_uniform_3d(&mesh); }
        let bb = mesh.bounding_box();
        let bdr_tags = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &pml_lo, &pml_hi);
        let pml = std::sync::Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, k, 3));
        let mut mesh_buf = Vec::new();
        let dummy_2d = Mesh::<2>::unit_square_tri(2);
        write_mfem(&mut mesh_buf, &dummy_2d, Some(&mesh)).unwrap();
        solve_pml(mesh.clone(), &args, prob, exact_known, pml, bdr_tags,
            args.herm_conv, args.visualization, &mesh_buf);
        let _ = fem_io::mfem::write_mfem_file_3d("ex25.mesh", &mesh);
        println!("  Wrote ex25.mesh");
    }
}
