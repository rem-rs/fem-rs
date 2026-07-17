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
use nalgebra::Complex;
use fem_assembly::{
    VectorAssembler,
    postproc::coefficient::{CoeffCtx, ScalarCoeff, MatrixCoeff,
                            RestrictedCoefficient, ScalarVectorProductCoefficient,
                            VectorRestrictedCoefficient,
                            ProductCoeff, ScalarMatrixCoeff},
    standard::{CurlCurlIntegrator, CurlCurlTensorIntegrator,
               VectorMassTensorIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    project_bdr_coefficient_tangent, project_bdr_coefficient_tangent_2d,
};
use fem_io::mfem;
use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_solver::{GSSmoother, ScaledPrecond, BlockDiagPrecondPair};
use fem_linalg::fem_to_linlvo_csr;
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};
use fem_mesh::{Mesh, topology::MeshTopology, element_type::ElementType};
use fem_mesh::refine_uniform;

// ═══════════════════════════════════════════════════════════════════════════
// Bessel functions (J₀, J₁, J₂, Y₀, Y₁, Y₂)
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
    bdr_tags: Vec<i32>) {

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
        // For lshape/fichera, only constrain specific boundary attributes
        // based on face center coordinates (matching C++ geometric check)
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
    let mut a = SesquilinearForm::new(&space, Convention::Hermitian, qo);

    let attr     = vec![1];  // computational domain (element tag 1)
    let attr_pml = vec![2];  // PML region (element tag 2)

    // Non-PML: CurlCurlIntegrator(μ⁻¹) + VectorFEMassIntegrator(-ω²ε)
    a.add_domain_integrator_pair(
        &CurlCurlIntegrator {
            mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() }
        },
        None,
    );
    a.add_domain_integrator_pair(
        // Non-PML mass: -ω²ε · I  (C++: RestrictedCoefficient(ConstantCoefficient(-ω²ε), attr))
        &VectorMassTensorIntegrator {
            alpha: VectorRestrictedCoefficient {
                inner: ScalarMatrixCoeff(omega2_eps),
                attrs: attr.clone(),
            }
        },
        None,
    );

    // PML curl-curl: μ⁻¹ · stretch
    if dim == 2 {
        a.add_domain_integrator_pair(
            &CurlCurlIntegrator {
                mu: RestrictedCoefficient {
                    inner: PmlCurlScalar { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                }
            },
            Some(&CurlCurlIntegrator {
                mu: RestrictedCoefficient {
                    inner: PmlCurlScalarIm { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                }
            }),
        );
    } else {
        a.add_domain_integrator_pair(
            &CurlCurlTensorIntegrator {
                mu: ScalarVectorProductCoefficient {
                    scalar: mu_inv,
                    vector: VectorRestrictedCoefficient {
                        inner: PmlCurlMatRe { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                }
            },
            Some(&CurlCurlTensorIntegrator {
                mu: ScalarVectorProductCoefficient {
                    scalar: mu_inv,
                    vector: VectorRestrictedCoefficient {
                        inner: PmlCurlMatIm { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                }
            }),
        );
    }

    // PML mass: ω²ε · stretch (diagonal matrix)
    a.add_domain_integrator_pair(
        &VectorMassTensorIntegrator {
            alpha: ScalarVectorProductCoefficient {
                scalar: omega2_eps,
                vector: VectorRestrictedCoefficient {
                    inner: PmlMassMatRe { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        },
        Some(&VectorMassTensorIntegrator {
            alpha: ScalarVectorProductCoefficient {
                scalar: omega2_eps,
                vector: VectorRestrictedCoefficient {
                    inner: PmlMassMatIm { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        }),
    );

    // Assemble → SesquilinearForm → flat 2×2 block
    let cs = a.assemble();
    let a_mat = cs.to_flat_csr_with_conv(Convention::Hermitian);

    // ── RHS ──────────────────────────────────────────────────────────────
    let mut rhs_re = vec![0.0; n];
    let mut rhs_im = vec![0.0; n];
    if prob == Prob::LoadSrc {
        let comp_bdr: Vec<[f64; 2]> = (0..dim).map(|d| pml.comp_domain_bdr[d]).collect();
        let src_fn = |x: &[f64], _ctx: &VectorQpData<'_>| -> Vec<f64> {
            source_fn(x, dim, &comp_bdr, omega, args.eps, args.mu)
        };
        let vec = VectorAssembler::assemble_linear(&space, &[&VectorSrc { f: &src_fn }], qo);
        rhs_re.copy_from_slice(&vec);
    }

    // ── Project BC (1:1 with C++ ProjectBdrCoefficientTangent) ──────────
    // Note: For full 1:1 with C++, use project_bdr_coefficient_tangent on
    // concrete Mesh<2>/Mesh<3>. Here we use interpolate_vector for the
    // generic MeshTopology path (equivalent for HCurl edge DOFs).
    if !ess_tdofs.is_empty() && exact_known {
        let bc_re = space.interpolate_vector(&|x: &[f64]| {
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.re).collect()
        });
        let bc_im = space.interpolate_vector(&|x: &[f64]| {
            let e = maxwell_solution(x, dim, prob, k);
            e.iter().map(|c| c.im).collect()
        });
        for &d in &ess_tdofs {
            rhs_re[d as usize] = bc_re[d as usize];
            rhs_im[d as usize] = bc_im[d as usize];
        }
    }

    // ── Preconditioner (1:1 with C++) ────────────────────────────────────
    // Non-PML: μ⁻¹·curlcurl + ω²ε·mass (with DIAG_ONE for BCs)
    // PML:     μ⁻¹·|stretch|·curlcurl + ω²ε·|stretch|·mass
    let nonpml_cc = if dim == 2 {
        VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlIntegrator {
                mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() }
            }], qo)
    } else {
        VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlTensorIntegrator {
                mu: ScalarMatrixCoeff(
                    RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() }
                )
            }], qo)
    };
    let nonpml_mass = VectorAssembler::assemble_bilinear(&space,
        &[&VectorMassTensorIntegrator {
            alpha: VectorRestrictedCoefficient {
                inner: ScalarMatrixCoeff(1.0_f64),
                attrs: attr.clone(),
            }
        }], qo);
    let mut prec = nonpml_cc.axpby(1.0, &nonpml_mass, abs_omega2_eps);

    // PML contribution to preconditioner
    if dim == 2 {
        let pml_cc_abs = VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlIntegrator {
                mu: ProductCoeff {
                    a: mu_inv,
                    b: RestrictedCoefficient {
                        inner: PmlCurlScalarAbs { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                }
            }], qo);
        prec = prec.axpby(1.0, &pml_cc_abs, 1.0);
    } else {
        let pml_cc_abs = VectorAssembler::assemble_bilinear(&space,
            &[&CurlCurlTensorIntegrator {
                mu: ScalarVectorProductCoefficient {
                    scalar: mu_inv,
                    vector: VectorRestrictedCoefficient {
                        inner: PmlCurlMatAbs { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                }
            }], qo);
        prec = prec.axpby(1.0, &pml_cc_abs, 1.0);
    }

    let pml_mass_abs = VectorAssembler::assemble_bilinear(&space,
        &[&VectorMassTensorIntegrator {
            alpha: ScalarVectorProductCoefficient {
                scalar: abs_omega2_eps,
                vector: VectorRestrictedCoefficient {
                    inner: PmlMassMatAbs { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            }
        }], qo);
    prec = prec.axpby(1.0, &pml_mass_abs, 1.0);

    // Apply symmetric BC elimination to preconditioner (matching C++ FormSystemMatrix + DIAG_ONE)
    let mut prec_mat = prec;
    let mut dummy_rhs = vec![0.0; n];
    for &d in &ess_tdofs {
        let d = d as usize;
        prec_mat.apply_dirichlet_symmetric(d, 1.0, &mut dummy_rhs);
    }
    let la = fem_to_linlvo_csr(&prec_mat);
    let gs = GSSmoother::from_csr(&la).expect("GSSmoother");
    let gs_im = GSSmoother::from_csr(&la).expect("GSSmoother_im");
    let pc_im = ScaledPrecond { inner: gs_im, scale: -1.0 };
    let bp = BlockDiagPrecondPair { pre_re: gs, pre_im: pc_im, n };

    // ── GMRES Solve ─────────────────────────────────────────────────────
    let mut flat_rhs = vec![0.0_f64; 2*n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; }
    for i in 0..n { flat_rhs[n+i] = rhs_im[i]; }
    let mut x = vec![0.0; 2*n];
    let res = fem_solver::solve_gmres_precond(&a_mat, &flat_rhs, &mut x, 200, &bp,
        &SolverConfig { rtol:1e-5, max_iter:2000, verbose:true, ..Default::default() })
        .expect("GMRES");
    println!("  GMRES converged in {} iters, final residual = {:.6e}",
             res.iterations, res.final_residual);

    // ── Error computation (1:1 with C++ output) ─────────────────────────
    if exact_known {
        let qe = std::cmp::max(2, 2*args.order + 1) as u8;
        let l2err2_re = compute_l2_error(&space, &x[..n], dim, prob, k, qe, Some(2), false);
        let l2err2_im = compute_l2_error(&space, &x[n..], dim, prob, k, qe, Some(2), true);
        let zero = vec![0.0; n];
        let norm2_re = compute_l2_error(&space, &zero, dim, prob, k, qe, Some(2), false);
        let norm2_im = compute_l2_error(&space, &zero, dim, prob, k, qe, Some(2), true);
        let l2err_re = l2err2_re.sqrt();
        let l2err_im = l2err2_im.sqrt();
        let norm_re = norm2_re.sqrt().max(1e-30);
        let norm_im = norm2_im.sqrt().max(1e-30);
        println!("\n Relative Error (Re part): || E_h - E || / ||E|| = {:.6e}",
                 l2err_re / norm_re);
        println!(" Relative Error (Im part): || E_h - E || / ||E|| = {:.6e}",
                 l2err_im / norm_im);
        println!(" Total Error: {:.6e}", (l2err2_re + l2err2_im).sqrt());
    }

    // ── Output ──────────────────────────────────────────────────────────
    let sol_norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}\n", sol_norm);

    let _ = fem_io::mfem::write_gf_file("ex25-sol_r.gf", dim, &x[..n], "ND", args.order as u8, dim);
    let _ = fem_io::mfem::write_gf_file("ex25-sol_i.gf", dim, &x[n..], "ND", args.order as u8, dim);
    println!("  Wrote ex25-sol_r.gf, ex25-sol_i.gf");
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

// ─── L² error computation ────────────────────────────────────────────────

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

fn elem_jacobian<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize, et: ElementType, xi: &[f64])
    -> (DMatrix<f64>, f64, Vec<f64>)
{
    match et {
        ElementType::Quad4 => {
            let xc: Vec<Vec<f64>> = (0..4).map(|k| mesh.node_coords(nodes[k]).to_vec()).collect();
            let (xi_v, eta) = (xi[0], xi[1]);
            let dndxi = |k: usize, e: f64| -> f64 { match k {
                0 => -0.25*(1.0-e), 1 => 0.25*(1.0-e),
                2 => 0.25*(1.0+e), 3 => -0.25*(1.0+e), _ => 0.0 }};
            let dndeta = |k: usize, x: f64| -> f64 { match k {
                0 => -0.25*(1.0-x), 1 => -0.25*(1.0+x),
                2 => 0.25*(1.0+x), 3 => 0.25*(1.0-x), _ => 0.0 }};
            let n = |k: usize, x: f64, e: f64| -> f64 { match k {
                0 => 0.25*(1.0-x)*(1.0-e), 1 => 0.25*(1.0+x)*(1.0-e),
                2 => 0.25*(1.0+x)*(1.0+e), 3 => 0.25*(1.0-x)*(1.0+e), _ => 0.0 }};
            let mut j = DMatrix::<f64>::zeros(dim, dim);
            for k in 0..4 {
                j[(0,0)] += dndxi(k,eta)*xc[k][0]; j[(0,1)] += dndeta(k,xi_v)*xc[k][0];
                j[(1,0)] += dndxi(k,eta)*xc[k][1]; j[(1,1)] += dndeta(k,xi_v)*xc[k][1];
            }
            let det = j.determinant();
            let xp = vec![
                n(0,xi_v,eta)*xc[0][0] + n(1,xi_v,eta)*xc[1][0] + n(2,xi_v,eta)*xc[2][0] + n(3,xi_v,eta)*xc[3][0],
                n(0,xi_v,eta)*xc[0][1] + n(1,xi_v,eta)*xc[1][1] + n(2,xi_v,eta)*xc[2][1] + n(3,xi_v,eta)*xc[3][1]];
            (j, det, xp)
        }
        _ => {
            let x0 = mesh.node_coords(nodes[0]);
            let mut j = DMatrix::<f64>::zeros(dim, dim);
            for col in 0..dim {
                let xc = mesh.node_coords(nodes[col + 1]);
                for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
            }
            let det_j = j.determinant();
            let xp: Vec<f64> = (0..dim).map(|i| {
                x0[i] + (0..dim).map(|k| j[(i,k)] * xi[k]).sum::<f64>()
            }).collect();
            (j, det_j, xp)
        }
    }
}

fn compute_l2_error<M: MeshTopology>(
    space: &HCurlSpace<M>, x: &[f64], dim: usize, prob: Prob, k: f64, qo: u8,
    exclude_tag: Option<i32>, use_imag: bool) -> f64 {
    let mesh = space.mesh(); let order = space.order();
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        if exclude_tag.map_or(false, |et| mesh.element_tag(e) == et) { continue; }
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, order);
        let n_ldofs = re.n_dofs(); let quad = re.quadrature(qo);
        let dofs = space.element_dofs(e); let nodes = mesh.element_nodes(e);
        let mut phi = vec![0.0; n_ldofs];
        for (q, xi) in quad.points.iter().enumerate() {
            let (_, det_j_q, xp) = elem_jacobian(mesh, nodes, dim, et, xi);
            let w = quad.weights[q] * det_j_q.abs();
            re.eval_basis(xi, &mut phi);
            let mut uh = vec![0.0; dim];
            for i in 0..n_ldofs { for d in 0..dim { uh[d] += x[dofs[i] as usize] * phi[i]; } }
            let exact = maxwell_solution(&xp, dim, prob, k);
            let diff2: f64 = (0..dim).map(|d| {
                let u_ex = if use_imag { exact[d].im } else { exact[d].re };
                (uh[d] - u_ex).powi(2)
            }).sum();
            err2 += w * diff2;
        }
    }
    err2
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
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, ref_levels: 3, iprob: 4, freq: 5.0, mu: 1.0, eps: 1.0 };
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
        solve_pml(mesh.clone(), &args, prob, exact_known, pml, bdr_tags);
        let _ = fem_io::mfem::write_mfem_file("ex25.mesh", &mesh);
        println!("  Wrote ex25.mesh");
    } else {
        let mut mesh: Mesh<3> = mfem_data.mesh3d.expect("3D");
        for _ in 0..args.ref_levels { mesh = fem_mesh::refine_uniform_3d(&mesh); }
        let bb = mesh.bounding_box();
        let bdr_tags = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &pml_lo, &pml_hi);
        let pml = std::sync::Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, k, 3));
        solve_pml(mesh.clone(), &args, prob, exact_known, pml, bdr_tags);
        let _ = fem_io::mfem::write_mfem_file_3d("ex25.mesh", &mesh);
        println!("  Wrote ex25.mesh");
    }
}
