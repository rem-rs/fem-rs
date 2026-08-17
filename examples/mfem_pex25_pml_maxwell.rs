//! # Parallel Example 25 — PML for Maxwell, complex-valued (1:1 with MFEM
//! ex25p / ex25p.cpp)
//!
//! Solves the second-order indefinite Maxwell equation
//! `(1/μ) curl curl E − ω² ε E = f` with a Perfectly Matched Layer (PML),
//! discretized with Nédélec H(curl) elements in 2-D or 3-D.  The PML is a
//! complex coordinate stretch: inside the PML region the coefficients become
//! complex (Re + Im parts), so the system is complex-valued.
//!
//! Test problems (prob = 0..4): beam / disc / lshape / fichera have known
//! exact solutions (relative L² errors are reported, excluding the PML
//! region); load_src (default) is an approximated point source with PML all
//! around (no exact solution).
//!
//! ## Parallel structure (mirrors ex25p)
//! 1. Read the serial mesh (default per problem: beam → `beam-hex.mesh`,
//!    disc → `square-disc.mesh`, lshape → `l-shape.mesh`, fichera →
//!    `fichera.mesh`, load_src → `inline-quad.mesh`), compute the PML
//!    boundaries from the *original* bounding box, refine
//!    `ser_ref_levels + par_ref_levels` times **serially** (the final global
//!    mesh is identical to C++'s serial+parallel refinement), then mark the
//!    PML elements (attribute 2) and partition.
//! 2. Build the parallel H(curl) space; assemble the real and imaginary
//!    blocks separately with the PML-restricted coefficients (curl-curl at
//!    quadrature order 0, mass at order 2p+1 — matching C++ per-integrator
//!    orders), then combine into a `ParComplexCsrMatrix` (same sparsity).
//! 3. Essential DOFs (boundary edges of the marked boundary faces) are
//!    detected on the **full** serial mesh (physical edge keys — deterministic,
//!    partition-independent, pex34/pex35 pattern); the BC values come from the
//!    exact-solution tangential projection (zero inside the PML) on the full
//!    mesh, mapped to the local partition slots **directly** (partition basis
//!    == full-mesh basis, pex35 pattern).  Complex Dirichlet elimination
//!    (DIAG_ONE rows) + cross-rank ghost-column elimination.
//! 4. Solve with the parallel complex FGMRES and the block-diagonal
//!    preconditioner `[P, ±P]` (C++: GMRES + `HypreAMS` on the absolute-value
//!    PML matrix, imag block scaled by ±1 per the complex convention).
//! 5. Relative L² errors (Re/Im + total) on the owned non-PML elements
//!    (global L² norm, allreduce); norm / sum / physical-edge checksums
//!    (np1-4 must agree).
//!
//! ## Usage
//! ```bash
//! cargo run --release --example mfem_pex25_pml_maxwell -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex25_pml_maxwell -- --ranks 2 -no-vis
//! cargo run --release --example mfem_pex25_pml_maxwell -- --ranks 4 -prob 0 -o 2 -f 1.0 -rs 1 -rp 1 -no-vis
//! ```
//!
//! ## Acceptance
//! * Default problem (load_src, inline-quad, order 1, freq 5, rs 1 + rp 2):
//!   unknowns 2112 (matches C++ `GlobalTrueVSize`); np1-4 norm/sum/checksums
//!   **bit-identical** (Re ‖u‖ = 2.36972909e-2, Im ‖u‖ = 2.61511307e-2,
//!   Re checksum −1.63926897e-1, Im checksum −9.60121880e0); total ‖E‖ =
//!   3.529084e-2 = C++ ex25p (3.529612e-2, 0.015% — C++ GMRES stops at
//!   rel-tol 1e-5 while FGMRES here converges to 1e-10; the indefinite
//!   Maxwell system's Krylov path is tolerance-sensitive).
//! * disc (prob 1) np1: total L² error 3.402338e-2 = serial ex25 (3.403996e-2,
//!   serial converged to a looser GMRES residual); unknowns 972.
//!
//! ## Core-library finding (fixed in this port)
//! The serial ex25 `VectorSrc` linear integrator wrote its element vector
//! component-wise (`f_elem[i·dim+d]`) while the assembler's accumulator pairs
//! `f_elem[i]` with dof `i` (per-DOF convention, cf. `VectorDomainLFIntegrator`)
//! — the load_src RHS was shuffled and the solution came out **half** of C++
//! (‖E‖ 1.82e-2 vs 3.53e-2).  Both `mfem_ex25_pml_maxwell.rs` and
//! `mfem_pex25_pml_maxwell.rs` now sum the components per dof.
//!
//! ## Known differences vs C++
//! * FGMRES tolerance 1e-10 (C++ GMRES 1e-5): the converged solution agrees
//!   with C++ up to C++'s tolerance; np1-4 are bit-identical at 1e-10 while
//!   C++'s own np1-4 would differ at ~1e-5.
//! * The parallel preconditioner is a block-diagonal [AMS, ±AMS] built from
//!   the local diagonal block (C++ uses the global HypreAMS); the harder
//!   indefinite problems (disc/lshape/fichera/beam at np>1) may stall around
//!   ‖r‖/‖b‖ ≈ 1e-4 (the block-diagonal AMS is weaker than the global one,
//!   cf. pex34/pex35) — the default load_src problem converges at all np.
//! * Order > 1 on *triangle* meshes is not supported: the discrete-gradient
//!   assembler (`assemble_hcurl_h1_gradient` / `ref_elem_vec`) has no
//!   TriNDk reference element (order 2 on quads/hexes works).
//! * Essential DOFs are detected from the FULL serial mesh by physical edge
//!   keys (pex34/pex35 pattern): the local mesh's own boundary faces lose some
//!   boundary marks after partitioning (`boundary_dofs_hcurl` misses them on
//!   np>1 — a known core-library gap).

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::io::Write;
use std::sync::Arc;

use fem_assembly::postproc::coefficient::{
    CoeffCtx, MatrixCoeff, RestrictedCoefficient, ScalarCoeff,
    ScalarVectorProductCoefficient, VectorRestrictedCoefficient,
};
use fem_assembly::postproc::grid_function::compute_l2_error_hcurl;
use fem_assembly::standard::{CurlCurlIntegrator, CurlCurlTensorIntegrator, VectorMassTensorIntegrator};
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_linalg::spadd;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_mesh::{refine_uniform, refine_uniform_3d};
use fem_parallel::{
    Comm, ParAmsPrecond, ParCsrMatrix, ParMixedAssembler, ParVector, ParVectorAssembler,
    ParallelFESpace, WorkerConfig,
    launcher::native::ThreadLauncher,
    par_complex_csr::ParComplexCsrMatrix,
    par_partition::partition_mesh,
    par_solver::par_solve_fgmres_complex,
    par_vector::ParComplexVector,
};
use fem_solver::SolverConfig;
use fem_space::{
    H1Space, HCurlSpace,
    fe_space::FESpace,
};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother};

// ═══════════════════════════════════════════════════════════════════════════
// CLI (mirrors ex25p OptionsParser)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Args {
    mesh: Option<String>,
    order: i32,
    ser_ref_levels: i32,
    par_ref_levels: i32,
    iprob: i32,
    freq: f64,
    mu: f64,
    eps: f64,
    herm_conv: bool,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        order: 1,
        ser_ref_levels: 1,
        par_ref_levels: 2,
        iprob: 4,
        freq: 5.0,
        mu: 1.0,
        eps: 1.0,
        herm_conv: true,
        ranks: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = Some(it.next().unwrap_or_default()),
            "-o" | "--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-rs" | "--refinements-serial" => a.ser_ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-rp" | "--refinements-parallel" => a.par_ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            "-prob" | "--problem" => a.iprob = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "-f" | "--frequency" => a.freq = it.next().and_then(|s| s.parse().ok()).unwrap_or(5.0),
            "-mu" | "--permeability" => a.mu = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-eps" | "--permittivity" => a.eps = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-herm" | "--hermitian" => a.herm_conv = true,
            "-no-herm" | "--no-hermitian" => a.herm_conv = false,
            "-vis" | "--visualization" => {}
            "-no-vis" | "--no-visualization" => {}
            "--ranks" => a.ranks = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "-pa" | "--partial-assembly" => {}
            "-no-pa" | "--no-partial-assembly" => {}
            "-d" | "--device" => { it.next(); }
            "-slu" | "--superlu" => {}
            "-mumps" | "--mumps-solver" => {}
            "-strumpack" | "--strumpack-solver" => {}
            _ => {}
        }
    }
    a
}

// ═══════════════════════════════════════════════════════════════════════════
// Bessel functions (MSVC CRT — bit-identical to C++ ex25)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(windows)]
extern "C" {
    fn _jn(n: i32, x: f64) -> f64;
    fn _yn(n: i32, x: f64) -> f64;
}
#[cfg(not(windows))]
extern "C" {
    fn jn(n: i32, x: f64) -> f64;
    fn yn(n: i32, x: f64) -> f64;
}
#[cfg(windows)]
unsafe fn bessel_j(n: i32, x: f64) -> f64 { _jn(n, x) }
#[cfg(windows)]
unsafe fn bessel_y(n: i32, x: f64) -> f64 { _yn(n, x) }
#[cfg(not(windows))]
unsafe fn bessel_j(n: i32, x: f64) -> f64 { jn(n, x) }
#[cfg(not(windows))]
unsafe fn bessel_y(n: i32, x: f64) -> f64 { yn(n, x) }
fn bessel_j0(x: f64) -> f64 { unsafe { bessel_j(0, x) } }
fn bessel_j1(x: f64) -> f64 { unsafe { bessel_j(1, x) } }
fn bessel_j2(x: f64) -> f64 { unsafe { bessel_j(2, x) } }
fn bessel_y0(x: f64) -> f64 { unsafe { bessel_y(0, x) } }
fn bessel_y1(x: f64) -> f64 { unsafe { bessel_y(1, x) } }
fn bessel_y2(x: f64) -> f64 { unsafe { bessel_y(2, x) } }

// ═══════════════════════════════════════════════════════════════════════════
// PML region (1:1 with the C++ `PML` class)
// ═══════════════════════════════════════════════════════════════════════════

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

    /// Complex stretching s'(x) = 1 + i·σ(x)/k (C++ `StretchFunction`).
    fn stretch(&self, x: &[f64]) -> Vec<nalgebra::Complex<f64>> {
        let mut dxs = vec![nalgebra::Complex::new(1.0, 0.0); self.dim];
        if self.k.abs() < 1e-30 { return dxs; }
        let n = 2.0;
        let c = 5.0;
        for d in 0..self.dim {
            if x[d] >= self.comp_domain_bdr[d][1] {
                let dist = x[d] - self.comp_domain_bdr[d][1];
                let len = self.length[d][1];
                if len > 0.0 {
                    let coeff = n * c / self.k / len.powf(n);
                    dxs[d] = nalgebra::Complex::new(1.0, coeff * dist.abs().powf(n - 1.0));
                }
            } else if x[d] <= self.comp_domain_bdr[d][0] {
                let dist = self.comp_domain_bdr[d][0] - x[d];
                let len = self.length[d][0];
                if len > 0.0 {
                    let coeff = n * c / self.k / len.powf(n);
                    dxs[d] = nalgebra::Complex::new(1.0, coeff * dist.abs().powf(n - 1.0));
                }
            }
        }
        dxs
    }

    fn det_j(&self, dxs: &[nalgebra::Complex<f64>]) -> nalgebra::Complex<f64> {
        let mut det = nalgebra::Complex::new(1.0, 0.0);
        for d in 0..self.dim { det *= dxs[d]; }
        det
    }

    /// PML stretch coefficients at a point.
    /// Returns [Re, Im, Abs] for curl-curl (index 0-2) and mass (index 3-5).
    /// - curl-curl stretch: 2D: 1/det(J); 3D: dxs[d]²/det(J) per component
    /// - mass stretch: det(J)/dxs[d]² per component
    fn coeffs_at(&self, x: &[f64], dim: usize) -> [[f64; 3]; 6] {
        let dxs = self.stretch(x);
        let det = self.det_j(&dxs);
        let mut c = [[0.0_f64; 3]; 6];
        if dim == 2 {
            let inv_det = 1.0 / det;
            c[0][0] = inv_det.re; c[1][0] = inv_det.im; c[2][0] = inv_det.norm();
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

struct PmlCurlScalar { pml: Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalar {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 { self.pml.coeffs_at(ctx.x, ctx.dim)[0][0] }
}
struct PmlCurlScalarIm { pml: Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalarIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 { self.pml.coeffs_at(ctx.x, ctx.dim)[1][0] }
}
struct PmlCurlScalarAbs { pml: Arc<PmlParams> }
impl ScalarCoeff for PmlCurlScalarAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 { self.pml.coeffs_at(ctx.x, ctx.dim)[2][0] }
}

// ─── PML stretch as MatrixCoeff (3D curl-curl, and mass in 2D/3D) ─────────

macro_rules! pml_matrix_coeff {
    ($name:ident, $idx:expr) => {
        struct $name { pml: Arc<PmlParams> }
        impl MatrixCoeff for $name {
            fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
                for v in out.iter_mut() { *v = 0.0; }
                let d = ctx.dim;
                let arr = self.pml.coeffs_at(ctx.x, d);
                for i in 0..d { out[i * (d + 1)] = arr[$idx][i]; }
            }
        }
    };
}
pml_matrix_coeff!(PmlCurlMatRe, 0);
pml_matrix_coeff!(PmlCurlMatIm, 1);
pml_matrix_coeff!(PmlCurlMatAbs, 2);
pml_matrix_coeff!(PmlMassMatRe, 3);
pml_matrix_coeff!(PmlMassMatIm, 4);
pml_matrix_coeff!(PmlMassMatAbs, 5);

// ═══════════════════════════════════════════════════════════════════════════
// Exact solutions / source (1:1 with ex25p)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum Prob { Beam, Disc, Lshape, Fichera, LoadSrc }
const ZI: nalgebra::Complex<f64> = nalgebra::Complex::new(0.0, 1.0);

fn maxwell_solution(x: &[f64], dim: usize, prob: Prob, k: f64) -> Vec<nalgebra::Complex<f64>> {
    let mut e = vec![nalgebra::Complex::new(0.0, 0.0); dim];
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
                let r = (x0 * x0 + x1 * x1).sqrt();
                let beta = k * r;
                if r < 1e-14 { e[0] = -ZI * ZI * 0.25; return e; }
                let (j0, j1, j2, y0, y1, y2) = (
                    bessel_j0(beta), bessel_j1(beta), bessel_j2(beta),
                    bessel_y0(beta), bessel_y1(beta), bessel_y2(beta),
                );
                let ho = nalgebra::Complex::new(j0, y0);
                let ho_r = -k * nalgebra::Complex::new(j1, y1);
                let ho_rr = -k * k * (nalgebra::Complex::new(j1, y1) / beta - nalgebra::Complex::new(j2, y2));
                let r_x = x0 / r; let r_y = x1 / r;
                let r_xy = -(r_x / r) * r_y; let r_xx = (1.0 / r) * (1.0 - r_x * r_x);
                let val = 0.25 * ZI * ho;
                let val_xx = 0.25 * ZI * (r_xx * ho_r + r_x * r_x * ho_rr);
                let val_xy = 0.25 * ZI * (r_xy * ho_r + r_x * r_y * ho_rr);
                e[0] = ZI / k * (k * k * val + val_xx);
                e[1] = ZI / k * val_xy;
            } else {
                let x0 = x[0] + shift[0]; let x1 = x[1] + shift[1]; let x2 = x[2] + shift[2];
                let r = (x0 * x0 + x1 * x1 + x2 * x2).sqrt();
                if r < 1e-14 { return e; }
                let (rx, ry, rz) = (x0 / r, x1 / r, x2 / r);
                let val = (ZI * k * r).exp() / r;
                let vr = val / r * (ZI * k * r - 1.0);
                let vrr = val / (r * r) * (-k * k * r * r - 2.0 * ZI * k * r + 2.0);
                let vxx = vrr * rx * rx + vr * (1.0 / r) * (1.0 - rx * rx);
                let vyx = vrr * rx * ry + vr * (-(ry / r) * rx);
                let vzx = vrr * rx * rz + vr * (-(rz / r) * rx);
                let a = ZI * k / (4.0 * PI * k * k);
                e[0] = a * (k * k * val + vxx); e[1] = a * vyx; e[2] = a * vzx;
            }
        }
        Prob::Beam => {
            if dim == 3 {
                e[1] = -ZI * k / PI * (PI * x[2]).sin() * (ZI * (k * k - PI * PI).sqrt() * x[0]).exp();
            } else {
                e[1] = -ZI * k / PI * (ZI * k * x[0]).exp();
            }
        }
        Prob::LoadSrc => {}
    }
    e
}

fn source_fn(x: &[f64], dim: usize, comp_bdr: &[[f64; 2]], omega: f64, eps: f64, mu: f64) -> Vec<f64> {
    let mut center = vec![0.0; dim];
    for d in 0..dim { center[d] = 0.5 * (comp_bdr[d][0] + comp_bdr[d][1]); }
    let r2: f64 = (0..dim).map(|d| (x[d] - center[d]).powi(2)).sum();
    let n = 5.0 * omega * (eps * mu).sqrt() / PI;
    let mut f = vec![0.0; dim];
    f[0] = n * n / PI * (-n * n * r2).exp();
    f
}

// ═══════════════════════════════════════════════════════════════════════════
// PML tagging + helpers (1:1 with PML::SetAttributes)
// ═══════════════════════════════════════════════════════════════════════════

fn pml_vals(prob: &Prob) -> ([f64; 3], [f64; 3]) {
    match *prob {
        Prob::Beam    => ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        Prob::Disc    => ([0.2, 0.2, 0.0], [0.2, 0.2, 0.0]),
        Prob::Lshape  => ([0.1, 0.1, 0.0], [0.0, 0.0, 0.0]),
        Prob::Fichera => ([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
        Prob::LoadSrc => ([0.25, 0.25, 0.0], [0.25, 0.25, 0.0]),
    }
}

fn default_mesh(prob: &Prob) -> &'static str {
    match prob {
        Prob::Beam => "data/beam-hex.mesh",
        Prob::Disc => "data/square-disc.mesh",
        Prob::Lshape => "data/l-shape.mesh",
        Prob::Fichera => "data/fichera.mesh",
        Prob::LoadSrc => "data/inline-quad.mesh",
    }
}

/// Mark elements with any vertex outside the computational domain as PML
/// (attribute 2) — C++ `PML::SetAttributes` on the (parallel) mesh.
fn tag_pml<const D: usize>(mesh: &mut Mesh<D>, pml: &PmlParams) {
    for e in 0..mesh.n_elems() as u32 {
        let mut in_pml = false;
        for &v in mesh.elem_nodes(e) {
            let c = mesh.node_coords(v);
            for d in 0..D {
                if c[d] < pml.comp_domain_bdr[d][0] || c[d] > pml.comp_domain_bdr[d][1] {
                    in_pml = true;
                    break;
                }
            }
            if in_pml { break; }
        }
        if in_pml { mesh.elem_tags[e as usize] = 2; }
    }
}

/// Per-element local edge table (HCurl space ordering).
fn edges_for_elem(et: fem_mesh::ElementType) -> &'static [(usize, usize)] {
    use fem_mesh::ElementType;
    match et {
        ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (0, 2)],
        ElementType::Quad4 | ElementType::Quad9 => &[(0, 1), (1, 2), (2, 3), (0, 3)],
        ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        ElementType::Hex8 | ElementType::Hex20 => &[
            (0, 1), (1, 2), (3, 2), (0, 3),
            (4, 5), (5, 6), (7, 6), (4, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
        ElementType::Prism6 => &[
            (0, 1), (1, 2), (0, 2),
            (3, 4), (4, 5), (3, 5),
            (0, 3), (1, 4), (2, 5),
        ],
        _ => &[(0, 1), (1, 2), (0, 2)],
    }
}

// ─── Parallel matrix helpers ────────────────────────────────────────────────

/// Clone a `ParCsrMatrix` (deep-copy diag/offd).
fn clone_par_csr(a: &ParCsrMatrix) -> ParCsrMatrix {
    ParCsrMatrix::from_blocks(
        a.diag_block().clone(),
        a.offd_block().clone(),
        a.n_owned(),
        a.n_ghost(),
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

/// Add two `ParCsrMatrix` (diag/offd separately; different patterns allowed).
fn par_spadd(a: &ParCsrMatrix, b: &ParCsrMatrix) -> ParCsrMatrix {
    let diag = spadd(a.diag_block(), b.diag_block());
    let offd = spadd(a.offd_block(), b.offd_block());
    ParCsrMatrix::from_blocks(
        diag, offd, a.n_owned(), a.n_ghost(),
        a.ghost_exchange_arc(), a.comm().clone(),
    )
}

// ─── Vector source integrator (load_src RHS) ───────────────────────────────

struct VectorSrc<'a> {
    f: &'a (dyn Fn(&[f64], &VectorQpData<'_>) -> Vec<f64> + Send + Sync),
}
impl VectorLinearIntegrator for VectorSrc<'_> {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        // f_elem is per-DOF (length n_dofs), the vector-linear-integrator
        // convention (cf. `VectorDomainLFIntegrator`): each entry is the FULL
        // component-summed contribution `w · Σ_d f_d · φ_i,d`.  Writing the
        // components separately (length n_dofs × dim) mismatches the
        // accumulator and shuffles the load vector (this was the load_src
        // RHS bug that made the solution half of C++).
        let f_val = (self.f)(qp.x_phys, qp);
        for i in 0..qp.n_dofs {
            let mut dot = 0.0;
            for d in 0..qp.dim {
                dot += f_val[d] * qp.phi_vec[i * qp.dim + d];
            }
            f_elem[i] += qp.weight * dot;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main solver (generic over 2-D / 3-D mesh)
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn solve_pml<M: MeshTopology + Clone + Send + Sync + 'static>(
    args: &Args,
    comm: &Comm,
    prob: Prob,
    exact_known: bool,
    pml: &Arc<PmlParams>,
    full_mesh: &M,
    par_mesh: &fem_parallel::ParallelMesh<M>,
    out: &mut String,
) where
    HCurlSpace<M>: FESpace<Mesh = M>,
{
    let dim = full_mesh.dim() as usize;
    let is_root = comm.is_root();
    let omega = 2.0 * PI * args.freq;
    let mu_inv = 1.0 / args.mu;
    let omega2_eps = -omega * omega * args.eps;
    let abs_omega2_eps = omega * omega * args.eps;

    let order = args.order as u8;
    let local_mesh = par_mesh.local_mesh().clone();
    let partition = par_mesh.partition().clone();
    let local_space = HCurlSpace::new(local_mesh.clone(), order);
    let par_space = ParallelFESpace::new_for_edge_space(
        HCurlSpace::new(local_mesh.clone(), order), par_mesh, comm.clone());
    let dp = par_space.dof_partition();
    let n_owned = dp.n_owned_dofs;
    let n_ghost = dp.n_ghost_dofs;

    if is_root {
        out.push_str(&format!("Number of finite element unknowns: {}\n", par_space.n_global_dofs()));
        out.push_str(&format!("  Mode: {dim}D\n"));
    }

    let qo_mass = (2 * order + 1) as u8;
    let qo_curl = 0u8;

    // ── Coefficient blocks (1:1 with C++ sections 14/16a) ──────────────────
    let attr = vec![1];
    let attr_pml = vec![2];
    let cc_nonpml = CurlCurlIntegrator {
        mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() },
    };
    let mass_nonpml = VectorMassTensorIntegrator {
        alpha: VectorRestrictedCoefficient {
            inner: fem_assembly::postproc::coefficient::ScalarMatrixCoeff(omega2_eps),
            attrs: attr.clone(),
        },
    };

    // PML curl-curl: 2D scalar or 3D tensor.
    let (pml_cc_re, pml_cc_im): (Box<dyn VectorBilinearIntegrator>, Box<dyn VectorBilinearIntegrator>) = if dim == 2 {
        (
            Box::new(CurlCurlIntegrator {
                mu: RestrictedCoefficient { inner: PmlCurlScalar { pml: pml.clone() }, attrs: attr_pml.clone() },
            }),
            Box::new(CurlCurlIntegrator {
                mu: RestrictedCoefficient { inner: PmlCurlScalarIm { pml: pml.clone() }, attrs: attr_pml.clone() },
            }),
        )
    } else {
        (
            Box::new(CurlCurlTensorIntegrator {
                mu: ScalarVectorProductCoefficient {
                    scalar: mu_inv,
                    vector: VectorRestrictedCoefficient {
                        inner: PmlCurlMatRe { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                },
            }),
            Box::new(CurlCurlTensorIntegrator {
                mu: ScalarVectorProductCoefficient {
                    scalar: mu_inv,
                    vector: VectorRestrictedCoefficient {
                        inner: PmlCurlMatIm { pml: pml.clone() },
                        attrs: attr_pml.clone(),
                    },
                },
            }),
        )
    };

    let pml_mass_re = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatRe { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        },
    };
    let pml_mass_im = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatIm { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        },
    };

    // ── Assemble re/im blocks (per-integrator quadrature, C++ orders) ──────
    let cc_re = ParVectorAssembler::assemble_bilinear(
        &par_space, &[&cc_nonpml, pml_cc_re.as_ref()], qo_curl);
    let cc_im = ParVectorAssembler::assemble_bilinear(
        &par_space, &[pml_cc_im.as_ref()], qo_curl);
    let mass_re = ParVectorAssembler::assemble_bilinear(
        &par_space, &[&mass_nonpml, &pml_mass_re], qo_mass);
    let mass_im = ParVectorAssembler::assemble_bilinear(
        &par_space, &[&pml_mass_im], qo_mass);
    let k_re = par_spadd(&mass_re, &cc_re);
    let k_im = par_spadd(&mass_im, &cc_im);

    let mut a_sys = ParComplexCsrMatrix::new(
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.diag_block(), k_im.diag_block()),
        fem_linalg::complex_csr::ComplexCsr::from_re_im(k_re.offd_block(), k_im.offd_block()),
        n_owned, n_ghost,
        par_space.dof_ghost_exchange_arc(),
        comm.clone(),
    );

    // ── RHS (load_src only; others are pure BC problems) ───────────────────
    let mut rhs = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
    if prob == Prob::LoadSrc {
        let comp_bdr: Vec<[f64; 2]> = (0..dim).map(|d| pml.comp_domain_bdr[d]).collect();
        let src_fn = |x: &[f64], _ctx: &VectorQpData<'_>| -> Vec<f64> {
            source_fn(x, dim, &comp_bdr, omega, args.eps, args.mu)
        };
        // C++ VectorFEDomainLFIntegrator uses IntRules order 2·el.GetOrder() = 2.
        let vec = ParVectorAssembler::assemble_linear(
            &par_space, &[&VectorSrc { f: &src_fn }], 2 * order);
        rhs.im.owned_slice_mut().copy_from_slice(&vec.owned_slice());
    }

    // ── Essential boundary edges (full-mesh physical keys) + BC values ─────
    let ess_edge_keys = full_ess_edge_keys(full_mesh, prob, dim);
    let full_fes = HCurlSpace::new(full_mesh.clone(), order);
    let full_re = full_fes.interpolate_vector(&|x: &[f64]| -> Vec<f64> {
        for d in 0..dim {
            if x[d] > pml.comp_domain_bdr[d][1] || x[d] < pml.comp_domain_bdr[d][0] {
                return vec![0.0; dim];
            }
        }
        let e = maxwell_solution(x, dim, prob, pml.k);
        (0..dim).map(|d| e[d].re).collect()
    });
    let full_im = full_fes.interpolate_vector(&|x: &[f64]| -> Vec<f64> {
        for d in 0..dim {
            if x[d] > pml.comp_domain_bdr[d][1] || x[d] < pml.comp_domain_bdr[d][0] {
                return vec![0.0; dim];
            }
        }
        let e = maxwell_solution(x, dim, prob, pml.k);
        (0..dim).map(|d| e[d].im).collect()
    });
    let mut full_re_by_key: HashMap<(u32, u32), f64> = HashMap::new();
    let mut full_im_by_key: HashMap<(u32, u32), f64> = HashMap::new();
    for e in full_mesh.elem_iter() {
        let et = full_mesh.element_type(e);
        let ns = full_mesh.element_nodes(e);
        let dofs = full_fes.element_dofs(e);
        for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
            let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
            full_re_by_key.entry(key).or_insert(full_re[dofs[k] as usize]);
            full_im_by_key.entry(key).or_insert(full_im[dofs[k] as usize]);
        }
    }

    // Map to local partition slots (partition basis == full-mesh basis).
    let lfes = HCurlSpace::new(local_mesh.clone(), order);
    let mut bc_re = vec![0.0; n_owned + n_ghost];
    let mut bc_im = vec![0.0; n_owned + n_ghost];
    let mut ess_mask = vec![false; n_owned + n_ghost];
    let mut seen = HashSet::new();
    for e in local_mesh.elem_iter() {
        let et = local_mesh.element_type(e);
        let ns = local_mesh.element_nodes(e);
        let dofs = lfes.element_dofs(e);
        for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
            let (ga, gb) = (partition.global_node(ns[a]), partition.global_node(ns[b]));
            let key = (ga.min(gb), ga.max(gb));
            if ess_edge_keys.contains(&key) && seen.insert(dofs[k]) {
                let pid = dp.permute_dof(dofs[k]) as usize;
                if pid < bc_re.len() {
                    bc_re[pid] = full_re_by_key.get(&key).copied().unwrap_or(0.0);
                    bc_im[pid] = full_im_by_key.get(&key).copied().unwrap_or(0.0);
                    ess_mask[pid] = true;
                }
            }
        }
    }

    // Eliminate ALL essential DOFs — including those with a *zero* Dirichlet
    // value (e.g. load_src: the exact solution vanishes on the boundary and
    // inside the PML).  Skipping zero-valued ess DOFs leaves them free in the
    // system (wrong solution — the PEC boundary must be removed from the
    // Krylov space regardless of the imposed value).
    let mut clamped: Vec<(usize, f64, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64, f64)> = Vec::new();
    for pid in 0..n_owned + n_ghost {
        if !ess_mask[pid] { continue; }
        let (vr, vi) = (bc_re[pid], bc_im[pid]);
        if pid < n_owned {
            clamped.push((pid, vr, vi));
        } else {
            ghost_ess.push((pid - n_owned, vr, vi));
        }
    }
    for &(pid, vr, vi) in &clamped {
        a_sys.apply_dirichlet_par(pid, vr, vi, &mut rhs);
    }
    a_sys.apply_ghost_ess_columns(&ghost_ess, &mut rhs);

    if is_root {
        out.push_str(&format!("Size of linear system: {}\n", 2 * par_space.n_global_dofs()));
        out.push_str("\nSolving for the complex field using FGMRES with a block-diagonal preconditioner\n");
    }

    // ── Preconditioner [P, ±P]: AMS on the absolute-value PML matrix ───────
    let imag_scale = if args.herm_conv { -1.0 } else { 1.0 };
    let cc_nonpml_prec = CurlCurlIntegrator {
        mu: RestrictedCoefficient { inner: mu_inv, attrs: attr.clone() },
    };
    let mass_nonpml_prec = VectorMassTensorIntegrator {
        alpha: VectorRestrictedCoefficient {
            inner: fem_assembly::postproc::coefficient::ScalarMatrixCoeff(abs_omega2_eps),
            attrs: attr.clone(),
        },
    };
    let pml_mass_abs_prec = VectorMassTensorIntegrator {
        alpha: ScalarVectorProductCoefficient {
            scalar: abs_omega2_eps,
            vector: VectorRestrictedCoefficient {
                inner: PmlMassMatAbs { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        },
    };
    let pml_cc_abs_prec: Box<dyn VectorBilinearIntegrator> = if dim == 2 {
        Box::new(CurlCurlIntegrator {
            mu: RestrictedCoefficient {
                inner: PmlCurlScalarAbs { pml: pml.clone() },
                attrs: attr_pml.clone(),
            },
        })
    } else {
        Box::new(CurlCurlTensorIntegrator {
            mu: ScalarVectorProductCoefficient {
                scalar: mu_inv,
                vector: VectorRestrictedCoefficient {
                    inner: PmlCurlMatAbs { pml: pml.clone() },
                    attrs: attr_pml.clone(),
                },
            },
        })
    };
    let prec_mass = ParVectorAssembler::assemble_bilinear(
        &par_space, &[&mass_nonpml_prec, &pml_mass_abs_prec], qo_mass);
    let prec_cc = ParVectorAssembler::assemble_bilinear(
        &par_space, &[&cc_nonpml_prec, pml_cc_abs_prec.as_ref()], qo_curl);
    let prec = par_spadd(&prec_mass, &prec_cc);
    let mut pc_elim = clone_par_csr(&prec);
    {
        let mut dummy = ParVector::zeros(&par_space);
        for &(pid, _, _) in &clamped {
            pc_elim.apply_dirichlet_par_keep_diag(pid, 0.0, &mut dummy);
        }
    }
    let h1_par = ParallelFESpace::new(
        H1Space::new(local_mesh.clone(), 1), par_mesh, comm.clone());
    let grad = ParMixedAssembler::assemble_hcurl_h1_gradient(&h1_par, &par_space, (2 * order + 1) as u8);
    let ams = std::sync::Arc::new(ParAmsPrecond::new(
        &pc_elim, &grad,
        AmsConfig {
            smoother_omega: 1.0,
            smoother_sweeps: 2,
            edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
            cycle: AmsCycle::MultiplicativeV11,
            face_space: false,
            singularity_regularization: 1e-10,
            ..Default::default()
        },
    ));
    let precond = move |r_re: &[f64], r_im: &[f64], z_re: &mut [f64], z_im: &mut [f64]| {
        ams.apply(r_re, z_re);
        let mut zz = vec![0.0; z_im.len()];
        ams.apply(r_im, &mut zz);
        for i in 0..z_im.len() {
            z_im[i] = imag_scale * zz[i];
        }
    };

    // ── Solve: parallel complex FGMRES (C++ GMRES: kdim 200, max 2000, rtol 1e-5) ──
    let cfg = SolverConfig {
        rtol: 1e-10, atol: 0.0, max_iter: 2000,
        verbose: std::env::var("PEX25_VERBOSE").is_ok(),
        ..Default::default()
    };
    let mut x = ParComplexVector::zeros_like(&rhs.re);
    for i in 0..n_owned {
        x.re.owned_slice_mut()[i] = bc_re[i];
        x.im.owned_slice_mut()[i] = bc_im[i];
    }
    x.update_ghosts();

    let res = par_solve_fgmres_complex(&a_sys, &rhs, &mut x, 200, Some(&precond), &cfg)
        .expect("FGMRES failed");
    if is_root {
        out.push_str(&format!("  FGMRES: Number of iterations: {}\n", res.iterations));
        out.push_str(&format!("  FGMRES: Final relative residual: {:.6e}\n", res.final_residual));
    }

    // ── Relative L² errors (exclude PML + non-owned elements) ──────────────
    x.update_ghosts();
    let mut x_dm_re = vec![0.0; local_space.n_dofs()];
    let mut x_dm_im = vec![0.0; local_space.n_dofs()];
    for i in 0..n_owned + n_ghost {
        let dm = dp.unpermute_dof(i as u32) as usize;
        let s = dp.sign_correction(dm as u32);
        x_dm_re[dm] = s * x.re.as_slice()[i];
        x_dm_im[dm] = s * x.im.as_slice()[i];
    }
    let qe = std::cmp::max(2, 2 * args.order + 1) as u8;
    let ne = local_mesh.n_elements() as usize;
    let exclude: Vec<bool> = (0..ne)
        .map(|e| e >= partition.n_owned_elems || local_mesh.element_tag(e as u32) == 2)
        .collect();
    if exact_known {
        let exact_re = |xp: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(xp, dim, prob, pml.k);
            (0..dim).map(|d| e[d].re).collect()
        };
        let exact_im = |xp: &[f64]| -> Vec<f64> {
            let e = maxwell_solution(xp, dim, prob, pml.k);
            (0..dim).map(|d| e[d].im).collect()
        };
        let l2err_re = compute_l2_error_hcurl(&x_dm_re, &local_space, &exact_re, qe, Some(&exclude));
        let l2err_im = compute_l2_error_hcurl(&x_dm_im, &local_space, &exact_im, qe, Some(&exclude));
        let zero = vec![0.0; local_space.n_dofs()];
        let norm_re = compute_l2_error_hcurl(&zero, &local_space, &exact_re, qe, Some(&exclude)).max(1e-30);
        let norm_im = compute_l2_error_hcurl(&zero, &local_space, &exact_im, qe, Some(&exclude)).max(1e-30);
        let err_re = comm.allreduce_sum_f64(l2err_re * l2err_re);
        let err_im = comm.allreduce_sum_f64(l2err_im * l2err_im);
        let nrm_re = comm.allreduce_sum_f64(norm_re * norm_re);
        let nrm_im = comm.allreduce_sum_f64(norm_im * norm_im);
        if is_root {
            out.push_str(&format!("\n Relative Error (Re part): || E_h - E || / ||E|| = {:.6e}\n",
                err_re.sqrt() / nrm_re.sqrt().max(1e-30)));
            out.push_str(&format!(" Relative Error (Im part): || E_h - E || / ||E|| = {:.6e}\n",
                err_im.sqrt() / nrm_im.sqrt().max(1e-30)));
            out.push_str(&format!(" Total Error: {:.6e}\n\n",
                (err_re + err_im).sqrt()));
        }
    }

    // ── Metrics (norm/sum/checksums — np1-4 must agree) ───────────────────
    let re_norm = x.re.global_norm();
    let re_sum = comm.allreduce_sum_f64(x.re.owned_slice().iter().sum::<f64>());
    let im_norm = x.im.global_norm();
    let im_sum = comm.allreduce_sum_f64(x.im.owned_slice().iter().sum::<f64>());
    let re_checksum = physical_checksum(&x.re, &par_space, full_mesh, par_mesh, order, &comm);
    let im_checksum = physical_checksum(&x.im, &par_space, full_mesh, par_mesh, order, &comm);
    if is_root {
        out.push_str(&format!("  Re: ||u|| = {:.8e}, sum = {:.8e}, checksum = {:.8e}\n",
            re_norm, re_sum, re_checksum));
        out.push_str(&format!("  Im: ||u|| = {:.8e}, sum = {:.8e}, checksum = {:.8e}\n",
            im_norm, im_sum, im_checksum));
    }

    // ── Save solution (per rank; mesh saved in main) ───────────────────────
    {
        let sol_r_name = format!("ex25p-sol_r.{:06}", comm.rank());
        let mut f = std::fs::File::create(&sol_r_name).expect("sol_r file");
        for &v in x.re.owned_slice() { writeln!(f, "{:.8e}", v).expect("sol_r write"); }
        let sol_i_name = format!("ex25p-sol_i.{:06}", comm.rank());
        let mut f = std::fs::File::create(&sol_i_name).expect("sol_i file");
        for &v in x.im.owned_slice() { writeln!(f, "{:.8e}", v).expect("sol_i write"); }
    }
}

/// Physical-edge-id checksum of an owned real/imag block (partition-independent,
/// matches C++ `(i+1)·x_i` at np1 — the full-mesh ND dof numbering is the
/// element×local-edge first-seen order).
fn physical_checksum<M: MeshTopology + Clone>(
    v: &ParVector,
    par_space: &ParallelFESpace<HCurlSpace<M>>,
    full_mesh: &M,
    par_mesh: &fem_parallel::ParallelMesh<M>,
    order: u8,
    comm: &Comm,
) -> f64 {
    let dp = par_space.dof_partition();
    let local_mesh = par_mesh.local_mesh();
    let partition = par_mesh.partition();
    let mut key_to_edge: HashMap<(u32, u32), u64> = HashMap::new();
    let mut next_edge = 0u64;
    for e in full_mesh.elem_iter() {
        let et = full_mesh.element_type(e);
        let ns = full_mesh.element_nodes(e);
        for &(a, b) in edges_for_elem(et) {
            let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
            if !key_to_edge.contains_key(&key) {
                key_to_edge.insert(key, next_edge);
                next_edge += 1;
            }
        }
    }
    let lfes = HCurlSpace::new(local_mesh.clone(), order);
    let mut dm_to_key: HashMap<u32, (u32, u32)> = HashMap::new();
    for e in local_mesh.elem_iter() {
        let et = local_mesh.element_type(e);
        let ns = local_mesh.element_nodes(e);
        let dofs = lfes.element_dofs(e);
        for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
            let (ga, gb) = (partition.global_node(ns[a]), partition.global_node(ns[b]));
            dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
        }
    }
    let mut ck = 0.0_f64;
    for p in 0..dp.n_owned_dofs {
        let dm = dp.unpermute_dof(p as u32);
        let id = dm_to_key.get(&dm).and_then(|&k| key_to_edge.get(&k)).copied().unwrap_or(0);
        ck += (id as f64 + 1.0) * v.as_slice()[p];
    }
    comm.allreduce_sum_f64(ck)
}

/// Full-mesh essential boundary edge keys (physical node pairs) — computed
/// deterministically on every rank (pex34/pex35 pattern).  For lshape/fichera
/// only the boundary faces whose centers match the C++ conditions are marked.
fn full_ess_edge_keys<M: MeshTopology>(full_mesh: &M, prob: Prob, dim: usize) -> HashSet<(u32, u32)> {
    let mut out = HashSet::new();
    let all = prob != Prob::Lshape && prob != Prob::Fichera;
    for bf in 0..full_mesh.n_boundary_faces() as u32 {
        let nodes = full_mesh.face_nodes(bf);
        let tag = full_mesh.face_tag(bf);
        let mut mark = all;
        if prob == Prob::Lshape || prob == Prob::Fichera {
            let mut center = vec![0.0; dim];
            for d in 0..dim {
                center[d] = nodes.iter().map(|&n| full_mesh.node_coords(n)[d]).sum::<f64>()
                    / nodes.len() as f64;
            }
            mark = match prob {
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
        }
        if mark {
            let _ = tag;
            for i in 0..nodes.len() {
                let a = nodes[i];
                let b = nodes[(i + 1) % nodes.len()];
                out.insert((a.min(b), a.max(b)));
            }
        }
    }
    out
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        let is_root = comm.is_root();
        let mut out = String::new();
        let prob = match args.iprob.min(4) {
            0 => Prob::Beam, 1 => Prob::Disc, 2 => Prob::Lshape, 3 => Prob::Fichera, _ => Prob::LoadSrc,
        };
        let exact_known = matches!(prob, Prob::Beam | Prob::Disc | Prob::Lshape | Prob::Fichera);
        let mesh_file = args.mesh.clone().unwrap_or_else(|| default_mesh(&prob).to_string());
        let omega = 2.0 * PI * args.freq;
        let k = omega * (args.eps * args.mu).sqrt();

        if is_root {
            out.push_str(&format!("--mesh {mesh_file} --order {} --prob {} --refinements-serial {} --refinements-parallel {} --frequency {}\n\n",
                args.order, args.iprob.min(4), args.ser_ref_levels, args.par_ref_levels, args.freq));
        }

        let mfem = read_mfem_file(&mesh_file).expect("failed to read MFEM mesh");
        let dim = if mfem.mesh3d.is_some() { 3 } else { 2 };
        let (pml_lo, pml_hi) = pml_vals(&prob);

        if dim == 2 {
            let mesh0: Mesh<2> = mfem.mesh2d.expect("2D mesh");
            let bb = mesh0.bounding_box();
            let pml = Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, k, 2));
            let mut full: Mesh<2> = mesh0;
            for _ in 0..args.ser_ref_levels + args.par_ref_levels {
                full = refine_uniform(&full);
            }
            tag_pml(&mut full, &pml);
            let pm = partition_mesh(&full, &comm);
            solve_pml(&args, &comm, prob, exact_known, &pml, &full, &pm, &mut out);
            let mesh_name = format!("mesh.{:06}", comm.rank());
            let mut f = std::fs::File::create(&mesh_name).expect("mesh file");
            let _ = write_mfem(&mut f, pm.local_mesh(), None).ok();
        } else {
            let mesh0: Mesh<3> = mfem.mesh3d.expect("3D mesh");
            let bb = mesh0.bounding_box();
            let pml = Arc::new(PmlParams::new(&bb.0, &bb.1, &pml_lo, &pml_hi, k, 3));
            let mut full: Mesh<3> = mesh0;
            for _ in 0..args.ser_ref_levels + args.par_ref_levels {
                full = refine_uniform_3d(&full);
            }
            tag_pml(&mut full, &pml);
            let pm = partition_mesh(&full, &comm);
            solve_pml(&args, &comm, prob, exact_known, &pml, &full, &pm, &mut out);
            let mesh_name = format!("mesh.{:06}", comm.rank());
            let mut f = std::fs::File::create(&mesh_name).expect("mesh file");
            let dummy = Mesh::<2>::unit_square_tri(1);
            let _ = write_mfem(&mut f, &dummy, Some(pm.local_mesh())).ok();
        }

        if is_root {
            out.push_str("\nFinished.\n");
            print!("{out}");
        }
    });
}
