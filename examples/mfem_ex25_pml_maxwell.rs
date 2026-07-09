//! # Example 25 — PML for Maxwell  (1:1 with MFEM ex25)
//!
//! ```text
//!   (1/μ)curl curl E − ω²ε E = f    with PML absorbing layers
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -prob 0 -f 1.0
//! cargo run --example mfem_ex25_pml_maxwell -- -prob 4 -o 2
//! ```

use std::f64::consts::PI;
use fem_assembly::{
    VectorAssembler, coefficient::PmlCoeff,
    postproc::coefficient::{CoeffCtx, ScalarCoeff},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_mesh::{Mesh, refine_uniform};
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};

// ─── Problem types ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Prob { Beam, Disc, Lshape, Fichera, LoadSrc }

// ─── PML coefficients (tensor formulation matching ex25) ───────────────────

fn pml_sigma_x(x: &[f64], pml: &PmlCoeff, k: f64) -> f64 {
    let d = x.len();
    let width_x = pml.max[0] - pml.min[0];
    let thick_x = width_x * pml.thickness;
    let inner_lo = pml.min[0] + thick_x;
    let inner_hi = pml.max[0] - thick_x;
    if d >= 1 {
        if x[0] <= inner_lo { return 5.0 * ((inner_lo - x[0])/thick_x.max(1e-16)).powi(2) / k.max(1e-16); }
        if x[0] >= inner_hi { return 5.0 * ((x[0] - inner_hi)/thick_x.max(1e-16)).powi(2) / k.max(1e-16); }
    }
    0.0
}
fn pml_sigma_y(x: &[f64], pml: &PmlCoeff, k: f64) -> f64 {
    let d = x.len();
    let width_y = pml.max[1] - pml.min[1];
    let thick_y = width_y * pml.thickness;
    let inner_lo = pml.min[1] + thick_y;
    let inner_hi = pml.max[1] - thick_y;
    if d >= 2 {
        if x[1] <= inner_lo { return 5.0 * ((inner_lo - x[1])/thick_y.max(1e-16)).powi(2) / k.max(1e-16); }
        if x[1] >= inner_hi { return 5.0 * ((x[1] - inner_hi)/thick_y.max(1e-16)).powi(2) / k.max(1e-16); }
    }
    0.0
}

/// Curl-curl coefficient in PML: 1/det(J) where J = diag(1+iσ/ω)
struct PmlCurlRe { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let k = self.omega * (1.0f64).sqrt(); // k = ω/c, c=1
        let sx = pml_sigma_x(ctx.x, &self.pml, k);
        let sy = pml_sigma_y(ctx.x, &self.pml, k);
        let dx_re = 1.0; let dx_im = sx/self.omega;
        let dy_re = 1.0; let dy_im = sy/self.omega;
        let det_re = dx_re*dy_re - dx_im*dy_im;
        let det_im = dx_re*dy_im + dx_im*dy_re;
        let inv_det_re = det_re / (det_re*det_re + det_im*det_im);
        if ctx.elem_tag == 1 { 1.0 } else { inv_det_re }
    }
}
struct PmlCurlIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let k = self.omega;
        let sx = pml_sigma_x(ctx.x, &self.pml, k);
        let sy = pml_sigma_y(ctx.x, &self.pml, k);
        let dx_im = sx/self.omega; let dy_im = sy/self.omega;
        let det_re = 1.0 - dx_im*dy_im;
        let det_im = dx_im + dy_im;
        let inv_det_im = -det_im / (det_re*det_re + det_im*det_im);
        if ctx.elem_tag == 1 { 0.0 } else { inv_det_im }
    }
}

/// Absolute-value PML coefficients for preconditioner
struct PmlCurlReAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlReAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        let k = self.omega;
        let sx = pml_sigma_x(ctx.x, &self.pml, k);
        let sy = pml_sigma_y(ctx.x, &self.pml, k);
        let dx_im = sx/self.omega; let dy_im = sy/self.omega;
        let det_re = 1.0 - dx_im*dy_im; let det_im = dx_im + dy_im;
        let inv_det_abs = 1.0 / (det_re*det_re + det_im*det_im).sqrt();
        inv_det_abs
    }
}
struct PmlCurlImAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlImAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        let k = self.omega;
        let sx = pml_sigma_x(ctx.x, &self.pml, k);
        let sy = pml_sigma_y(ctx.x, &self.pml, k);
        let dx_im = sx/self.omega; let dy_im = sy/self.omega;
        let det_im = dx_im + dy_im;
        det_im.abs() / (1.0 + dy_im*dy_im).max(1e-16)
    }
}
struct PmlMassAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        let sx = pml_sigma_x(ctx.x, &self.pml, self.omega);
        let sy = pml_sigma_y(ctx.x, &self.pml, self.omega);
        let d_re = 1.0 - (sx/self.omega)*(sy/self.omega);
        let d_im = sx/self.omega + sy/self.omega;
        (d_re*d_re + d_im*d_im).sqrt()
    }
}

/// Mass coefficient in PML: det(J)
struct PmlMassRe { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        let sx = pml_sigma_x(ctx.x, &self.pml, self.omega);
        let sy = pml_sigma_y(ctx.x, &self.pml, self.omega);
        1.0 - (sx/self.omega)*(sy/self.omega)
    }
}
struct PmlMassIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        let sx = pml_sigma_x(ctx.x, &self.pml, self.omega);
        let sy = pml_sigma_y(ctx.x, &self.pml, self.omega);
        sx/self.omega + sy/self.omega
    }
}

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, order: i32, ref_levels: i32, iprob: i32, freq: f64 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, ref_levels: 3, iprob: 4, freq: 5.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = Some(it.next().unwrap_or("".into())),
            "-o"|"--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-ref"|"--refinements" => a.ref_levels = it.next().unwrap_or("3".into()).parse().unwrap_or(3),
            "-prob"|"--problem" => a.iprob = it.next().unwrap_or("4".into()).parse().unwrap_or(4),
            "-f"|"--frequency" => a.freq = it.next().unwrap_or("5.0".into()).parse().unwrap_or(5.0),
            _ => {}
        }
    }
    a
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let prob = match args.iprob.min(4) { 0 => Prob::Beam, 1 => Prob::Disc, 2 => Prob::Lshape, 3 => Prob::Fichera, _ => Prob::LoadSrc };
    let _exact_known = matches!(prob, Prob::Beam | Prob::Disc | Prob::Lshape | Prob::Fichera);

    let mesh_file = args.mesh.as_deref().unwrap_or(match prob {
        Prob::Beam => "data/beam-quad.mesh", Prob::Disc => "data/square-disc.mesh",
        Prob::Lshape => "data/l-shape.mesh", Prob::Fichera => "data/fichera.mesh",
        Prob::LoadSrc => "data/inline-quad.mesh",
    });

    let mut mesh: Mesh<2> = read_mfem_file(mesh_file).expect("mesh").mesh2d.expect("2D");
    for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }

    let omega = 2.0 * PI * args.freq;
    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);

    // Tag PML elements
    let bb = mesh.bounding_box();
    let pml_len = match prob {
        Prob::Disc => 0.2, Prob::Lshape => 0.1, Prob::Fichera => 0.5,
        Prob::Beam => 2.0, Prob::LoadSrc => 0.25,
    };
    let cl = bb.0; let ch = bb.1;
    for e in 0..mesh.n_elems() as u32 {
        let mut in_pml = false;
        for &v in mesh.elem_nodes(e) {
            let c = mesh.coords_of(v);
            if c[0] < cl[0]+pml_len || c[0] > ch[0]-pml_len ||
               c[1] < cl[1]+pml_len || c[1] > ch[1]-pml_len { in_pml = true; break; }
        }
        if in_pml { mesh.elem_tags[e as usize] = 2; }
    }

    let pml = PmlCoeff::new(vec![cl[0],cl[1]], vec![ch[0],ch[1]], pml_len, 5.0);
    let qo = (2 * args.order + 1) as u8;

    // Assemble complex system
    let curl_re = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlRe { omega, pml: pml.clone() } }], qo);
    let mass_re = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassRe { omega, pml: pml.clone() } }], qo);
    let mut k_re = curl_re.axpby(1.0, &mass_re, -omega*omega);

    let curl_im = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlIm { omega, pml: pml.clone() } }], qo);
    let mass_im = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassIm { omega, pml: pml.clone() } }], qo);
    let mut k_im = curl_im.axpby(1.0, &mass_im, -omega*omega);
    let pml_prec = pml.clone(); // saved for preconditioner

    // Source: Gaussian approximation of point source (matching C++ load_src)
    let (cx, cy) = { let (lo, hi) = mesh.bounding_box(); ((lo[0]+hi[0])/2.0, (lo[1]+hi[1])/2.0) };
    let n_sigma = 5.0 * omega;
    let coeff = n_sigma * n_sigma / PI;
    let src_fn = |x: &[f64]| -> [f64; 2] {
        let r2 = (x[0]-cx).powi(2) + (x[1]-cy).powi(2);
        let alpha = -n_sigma * n_sigma * r2;
        [coeff * alpha.exp(), 0.0]
    };

    // BC
    let bdr = boundary_dofs_hcurl(space.mesh(), &space, &space.mesh().unique_boundary_tags());
    // Apply BC to k_re, k_im
    for &d in &bdr { let d = d as usize; for p in k_re.row_ptr[d]..k_re.row_ptr[d+1] { let c = k_re.col_idx[d] as usize; k_re.values[p] = if c==d {1.0} else {0.0}; } }
    for &d in &bdr { let d = d as usize; for p in k_im.row_ptr[d]..k_im.row_ptr[d+1] { k_im.values[p] = 0.0; } }

    // RHS (real part only for load_src)
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    struct Src<F: Fn(&[f64]) -> [f64; 2] + Send + Sync> { f: F }
    impl<F: Fn(&[f64]) -> [f64; 2] + Send + Sync> VectorLinearIntegrator for Src<F> {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
            let f = (self.f)(qp.x_phys);
            for i in 0..qp.n_dofs { fe[i] += qp.weight * (qp.phi_vec[i*2]*f[0] + qp.phi_vec[i*2+1]*f[1]); }
        }
    }
    let rhs_re = VectorAssembler::assemble_linear(&space, &[&Src { f: src_fn }], qo);
    let mut flat_rhs = vec![0.0_f64; 2 * n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; }
    for &d in &bdr { flat_rhs[d as usize] = 0.0; flat_rhs[n + d as usize] = 0.0; }

    // Build [K_re -K_im; K_im K_re]
    let mut coo = CooMatrix::new(2*n, 2*n);
    for i in 0..n {
        for p in k_re.row_ptr[i]..k_re.row_ptr[i+1] { let j=k_re.col_idx[p] as usize; let v=k_re.values[p]; coo.add(i,j,v); coo.add(n+i,n+j,v); }
        for p in k_im.row_ptr[i]..k_im.row_ptr[i+1] { let j=k_im.col_idx[p] as usize; let v=k_im.values[p]; coo.add(i,n+j,-v); coo.add(n+i,j,v); }
    }
    let a = coo.into_csr();

    let mut x = vec![0.0; 2*n];
    use fem_solver::{BlockDiagPrecond, GSSmoother};
    use fem_linalg::fem_to_linlvo_csr;

    let prec_coeff_re = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlReAbs { omega, pml: pml_prec.clone() } }], qo);
    let prec_coeff_im = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlImAbs { omega, pml: pml_prec.clone() } }], qo);
    let prec_mass = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassAbs { omega, pml: pml_prec } }], qo);
    let prec_mat = CsrMatrix::add(&prec_coeff_re, &prec_coeff_im).axpby(1.0, &prec_mass, omega*omega);

    let prec_linlvo = fem_to_linlvo_csr(&prec_mat);
    let gs = GSSmoother::from_csr(&prec_linlvo, 1.0).expect("GSSmoother");
    let block_prec = BlockDiagPrecond { inner: gs, n };

    let _res = fem_solver::solve_gmres_precond(&a, &flat_rhs, &mut x, 200, &block_prec, &SolverConfig { rtol:1e-3, max_iter:2000, verbose:true, ..Default::default() }).expect("GMRES");

    let norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}", norm);
    println!("\nFinished.");
}
