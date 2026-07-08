//! # Example 25 — PML for Maxwell (H(curl), analogous to MFEM ex25)
//!
//! Solves the time-harmonic Maxwell equation with a Perfectly Matched Layer:
//!
//! ```text
//!   ∇×(∇×E) − ω²·ε·E = f
//! ```
//!
//! with complex-valued material parameters in the PML region:
//!   ε → ε·(1 + i·σ/ω)  (lossy permittivity in PML)
//!   μ → μ·(1 + i·σ/ω)⁻¹ (lossy permeability in PML)
//!
//! where σ(x) ramps from 0 to σ_max in the outer PML layer.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -o 2 -f 1.0
//! cargo run --example mfem_ex25_pml_maxwell -- -m data/inline-quad.mesh -o 3 -f 8.0
//! ```

use fem_assembly::{
    coefficient::PmlCoeff,
    postproc::coefficient::{CoeffCtx, ScalarCoeff},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    VectorAssembler,
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, SolverConfig};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::solve_gmres;
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::boundary_dofs_hcurl,
};

// ─── PML coefficients ───────────────────────────────────────────────────

/// Real part of 1/μ_cplx = 1/(1 + σ²/ω²) where μ_cplx = 1 + i·σ/ω
struct MuInvRe { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for MuInvRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let s = self.pml.eval(ctx);
        1.0 / (1.0 + s * s / (self.omega * self.omega))
    }
}

/// Imag part of 1/μ_cplx = -σ/(ω·(1 + σ²/ω²))
struct MuInvIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for MuInvIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let s = self.pml.eval(ctx);
        -s / (self.omega * (1.0 + s * s / (self.omega * self.omega)))
    }
}

/// Real part of ε_cplx = 1 (unchanged permittivity in PML)
struct EpsRe;
impl ScalarCoeff for EpsRe {
    fn eval(&self, _ctx: &CoeffCtx<'_>) -> f64 { 1.0 }
}

/// Imag part of ε_cplx = σ/ω
struct EpsIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for EpsIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.pml.eval(ctx) / self.omega
    }
}

// ─── CLI ────────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, n: usize, order: u8, omega: f64 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 8, order: 2, omega: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(8),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "-f" | "--frequency" => a.omega = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            _ => {}
        }
    }
    a
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    println!("=== Example 25: PML Maxwell (H(curl)) ===");
    println!("  ω={}, order={}", args.omega, args.order);

    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_quad(args.n)
    };
    println!("  Mesh: {} elements, type={:?}", mesh.n_elems(), mesh.element_type(0));

    let space = HCurlSpace::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  DOFs: {n}");

    // PML: outer 20% layer, σ_max = 3.0
    let pml = PmlCoeff::new(vec![0.0, 0.0], vec![1.0, 1.0], 0.2, 3.0);
    let qo = (args.order as u8) * 2 + 1;

    // Real part: ∇×(μ_re⁻¹·∇×E) − ω²·ε_re·E
    let k_re_c = VectorAssembler::assemble_bilinear(
        &space, &[&CurlCurlIntegrator { mu: MuInvRe { omega: args.omega, pml: pml.clone() } }], qo);
    let m_re = VectorAssembler::assemble_bilinear(
        &space, &[&VectorMassIntegrator { alpha: EpsRe }], qo);
    let k_re = k_re_c.axpby(1.0, &m_re, -args.omega * args.omega);

    // Imag part: ∇×(μ_im⁻¹·∇×E) − ω²·ε_im·E
    let k_im_c = VectorAssembler::assemble_bilinear(
        &space, &[&CurlCurlIntegrator { mu: MuInvIm { omega: args.omega, pml: pml.clone() } }], qo);
    let m_im = VectorAssembler::assemble_bilinear(
        &space, &[&VectorMassIntegrator { alpha: EpsIm { omega: args.omega, pml } }], qo);
    let k_im = k_im_c.axpby(1.0, &m_im, -args.omega * args.omega);

    // PEC BC on all boundaries
    let bdr = boundary_dofs_hcurl(space.mesh(), &space, &space.mesh().unique_boundary_tags());
    let mut k_re = k_re;
    let mut k_im = k_im;
    let mut rhs = vec![0.0; 2 * n];

    for &d in &bdr {
        let d = d as usize;
        for p in k_re.row_ptr[d]..k_re.row_ptr[d+1] {
            let col = k_re.col_idx[p] as usize;
            k_re.values[p] = if col == d { 1.0 } else { 0.0 };
        }
        for p in k_im.row_ptr[d]..k_im.row_ptr[d+1] {
            k_im.values[p] = 0.0;
        }
        rhs[d] = 0.0;
        rhs[n + d] = 0.0;
    }

    // Build flat system [K_re -K_im; K_im K_re]
    let mut coo = CooMatrix::new(2 * n, 2 * n);
    for i in 0..n {
        for p in k_re.row_ptr[i]..k_re.row_ptr[i+1] {
            let j = k_re.col_idx[p] as usize;
            let v = k_re.values[p];
            coo.add(i, j, v); coo.add(n+i, n+j, v);
        }
        for p in k_im.row_ptr[i]..k_im.row_ptr[i+1] {
            let j = k_im.col_idx[p] as usize;
            let v = k_im.values[p];
            coo.add(i, n+j, -v); coo.add(n+i, j, v);
        }
    }
    let a = coo.into_csr();

    let mut x = vec![0.0; 2 * n];
    let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, verbose: false, ..Default::default() };
    let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES solve");
    println!("  GMRES: {} iters, residual={:.3e}, converged={}",
        res.iterations, res.final_residual, res.converged);

    let sol_norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}", sol_norm);
    println!("  PASS");
}
