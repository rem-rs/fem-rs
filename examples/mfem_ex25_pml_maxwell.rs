//! # Example 25 — PML for Maxwell  (1:1 with MFEM ex25) — Generic 2D/3D
//!
//! ```text
//!   (1/μ)curl curl E − ω²ε E = f    with PML absorbing layers
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -prob 0 -f 1.0
//! cargo run --example mfem_ex25_pml_maxwell -- -m data/inline-hex.mesh -prob 4 -o 2
//! ```

use std::f64::consts::PI;
use fem_assembly::{
    VectorAssembler, coefficient::PmlCoeff,
    postproc::coefficient::{CoeffCtx, ScalarCoeff},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_mesh::{Mesh, refine_uniform, refine_uniform_3d, topology::MeshTopology};
use fem_solver::{BlockDiagPrecond, GSSmoother};
use fem_linalg::fem_to_linlvo_csr;
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};

// ─── Problem types ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Prob { Beam, Disc, Lshape, Fichera, LoadSrc }

// ─── PML coefficients (dimension-agnostic via x.len()) ─────────────────────

fn pml_sigma(x: &[f64], lo: f64, hi: f64, thick: f64, k: f64) -> f64 {
    if x[0] <= lo + thick { return 5.0 * ((lo + thick - x[0])/thick.max(1e-16)).powi(2) / k.max(1e-16); }
    if x[0] >= hi - thick { return 5.0 * ((x[0] - hi + thick)/thick.max(1e-16)).powi(2) / k.max(1e-16); }
    0.0
}

fn pml_det_re_im(x: &[f64], pml: &PmlCoeff, omega: f64) -> (f64, f64) {
    let k = omega;
    let mut s = vec![0.0; x.len()];
    for d in 0..x.len() {
        let width = pml.max[d] - pml.min[d];
        s[d] = pml_sigma(&[x[d]], pml.min[d], pml.max[d], width * pml.thickness, k);
    }
    let dim = x.len();
    if dim == 2 {
        let det_re = 1.0 - s[0]*s[1]/(omega*omega); let det_im = (s[0]+s[1])/omega;
        (det_re, det_im)
    } else {
        let sx = s[0]/omega; let sy = s[1]/omega; let sz = s[2]/omega;
        let det_re = 1.0 - sx*sy - sx*sz - sy*sz - 2.0*sx*sy*sz;
        let det_im = sx + sy + sz - sx*sy*sz;
        (det_re, det_im)
    }
}

fn inv_det_r(dr: f64, di: f64) -> f64 { dr / (dr*dr + di*di) }
fn inv_det_i(dr: f64, di: f64) -> f64 { -di / (dr*dr + di*di) }
macro_rules! pml_coeff {
    ($name:ident, $expr:expr) => {
        struct $name { omega: f64, pml: PmlCoeff }
        impl ScalarCoeff for $name {
            fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
                if ctx.elem_tag == 1 { return 0.0; }
                let (dr, di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
                $expr(dr, di)
            }
        }
    };
}
pml_coeff!(PmlCurlRe, |dr: f64, di: f64| inv_det_r(dr, di));
pml_coeff!(PmlCurlIm, |_dr: f64, di: f64| inv_det_i(1.0, di));
pml_coeff!(PmlMassRe, |dr: f64, _di: f64| dr);
pml_coeff!(PmlMassIm, |_dr: f64, di: f64| di);
pml_coeff!(PmlCurlReAbs, |dr: f64, di: f64| 1.0 / (dr*dr + di*di).sqrt().max(1e-16));
pml_coeff!(PmlMassAbs, |dr: f64, di: f64| (dr*dr + di*di).sqrt());

struct One; impl ScalarCoeff for One { fn eval(&self, _: &CoeffCtx<'_>) -> f64 { 1.0 } }

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

// ─── Generic solver (works for both Mesh<2> and Mesh<3>) ───────────────────

fn solve_pml<M: MeshTopology + Clone>(mesh: M, args: &Args, prob: Prob, _exact_known: bool, bb: ([f64; 3], [f64; 3]), bdr_tags: Vec<i32>) {
    let dim = mesh.dim() as usize;
    let omega = 2.0 * PI * args.freq;
    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);
    if dim == 2 { println!("  Mode: 2D"); } else { println!("  Mode: 3D (scalar PML)"); }

    let qo = (2 * args.order + 1) as u8;
    let pml_len = match prob {
        Prob::Disc => 0.2, Prob::Lshape => 0.1, Prob::Fichera => 0.5,
        Prob::Beam => 2.0, Prob::LoadSrc => 0.25,
    };

    // Build PML coeff
    let pml_lo: Vec<f64> = (0..dim).map(|d| bb.0[d]).collect();
    let pml_hi: Vec<f64> = (0..dim).map(|d| bb.1[d]).collect();
    let pml = PmlCoeff::new(pml_lo, pml_hi, pml_len, 5.0);

    // Helper: assemble with coeff that dispatches on element tag
    let assemble_re = |use_pml: bool, omega: f64, pml: &PmlCoeff| {
        let curl = if use_pml {
            VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlRe { omega, pml: pml.clone() } }], qo)
        } else {
            VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: One }], qo)
        };
        let mass = if use_pml {
            VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassRe { omega, pml: pml.clone() } }], qo)
        } else {
            VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: One }], qo)
        };
        curl.axpby(1.0, &mass, -omega*omega)
    };

    // Assemble complex system (2×2 block)
    let mut k_re = assemble_re(true, omega, &pml);
    let curl_im = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlIm { omega, pml: pml.clone() } }], qo);
    let mass_im = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassIm { omega, pml: pml.clone() } }], qo);
    let mut k_im = curl_im.axpby(1.0, &mass_im, -omega*omega);
    let pml_prec = pml.clone();

    // BC
    let bdr = boundary_dofs_hcurl(space.mesh(), &space, &bdr_tags);
    for &d in &bdr {
        let d = d as usize;
        for p in k_re.row_ptr[d]..k_re.row_ptr[d+1] {
            let c = k_re.col_idx[p] as usize;
            k_re.values[p] = if c == d { 1.0 } else { 0.0 };
        }
    }
    for &d in &bdr {
        let d = d as usize;
        for p in k_im.row_ptr[d]..k_im.row_ptr[d+1] { k_im.values[p] = 0.0; }
    }

    // Source (real only) — use bounding box center
    let cx = (bb.0[0] + bb.1[0]) / 2.0;
    let cy = if dim >= 2 { (bb.0[1] + bb.1[1]) / 2.0 } else { 0.0 };
    let ns = 5.0 * omega;
    let coeff = ns*ns/PI;
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    struct Src<F: Fn(&[f64])->[f64;2]+Send+Sync>{f:F}
    impl<F: Fn(&[f64])->[f64;2]+Send+Sync> VectorLinearIntegrator for Src<F> {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
            let f = (self.f)(qp.x_phys);
            for i in 0..qp.n_dofs { fe[i] += qp.weight * (qp.phi_vec[i*2]*f[0] + qp.phi_vec[i*2+1]*f[1]); }
        }
    }
    let src_fn = |x:&[f64]|->[f64;2]{ let r2=(x[0]-cx).powi(2)+(x[1]-cy).powi(2); [coeff*(-ns*ns*r2).exp(), 0.0] };
    let rhs_re = VectorAssembler::assemble_linear(&space, &[&Src{f:src_fn}], qo);
    let mut flat_rhs = vec![0.0_f64; 2*n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; }

    // Build [K_re -K_im; K_im K_re]
    let mut coo = CooMatrix::new(2*n, 2*n);
    for i in 0..n {
        for p in k_re.row_ptr[i]..k_re.row_ptr[i+1] { let j=k_re.col_idx[p]as usize; let v=k_re.values[p]; coo.add(i,j,v); coo.add(n+i,n+j,v); }
        for p in k_im.row_ptr[i]..k_im.row_ptr[i+1] { let j=k_im.col_idx[p]as usize; let v=k_im.values[p]; coo.add(i,n+j,-v); coo.add(n+i,j,v); }
    }
    let a = coo.into_csr();

    // Preconditioner
    let prec_c = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlReAbs { omega, pml: pml_prec.clone() } }], qo);
    let prec_m = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassAbs { omega, pml: pml_prec } }], qo);
    let prec_mat = prec_c.axpby(1.0, &prec_m, omega*omega);
    // Add small diagonal shift to prevent zero-diagonal in PML regions
    let mut shift_coo = CooMatrix::new(n, n);
    for i in 0..n { shift_coo.add(i, i, 1e-6); }
    let prec_mat = CsrMatrix::add(&prec_mat, &shift_coo.into_csr());
    let pl = fem_to_linlvo_csr(&prec_mat);
    let gs = GSSmoother::from_csr(&pl, 1.0).expect("GSSmoother");
    let bp = BlockDiagPrecond { inner: gs, n };

    let mut x = vec![0.0; 2*n];
    let _res = fem_solver::solve_gmres_precond(&a, &flat_rhs, &mut x, 200, &bp,
        &SolverConfig { rtol:1e-3, max_iter:2000, verbose:true, ..Default::default() }).expect("GMRES");
    let norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}", norm);
    println!("\nFinished.");
}

// ─── Main (dispatches 2D vs 3D) ───────────────────────────────────────────

fn main() {
    let args = parse_args();
    let prob = match args.iprob.min(4) { 0=>Prob::Beam, 1=>Prob::Disc, 2=>Prob::Lshape, 3=>Prob::Fichera, _=>Prob::LoadSrc };
    let exact_known = matches!(prob, Prob::Beam|Prob::Disc|Prob::Lshape|Prob::Fichera);

    let mesh_file = args.mesh.as_deref().unwrap_or(match prob {
        Prob::Beam=>"data/beam-quad.mesh", Prob::Disc=>"data/square-disc.mesh",
        Prob::Lshape=>"data/l-shape.mesh", Prob::Fichera=>"data/fichera.mesh",
        Prob::LoadSrc=>"data/inline-quad.mesh",
    });

    let mfem = read_mfem_file(mesh_file).expect("mesh");
    let dim = if mfem.mesh3d.is_some() { 3 } else { 2 };

    if dim == 2 {
        let mut mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
        for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }
        let bb = mesh.bounding_box();
        let bt = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &prob);
        solve_pml(mesh, &args, prob, exact_known, ([bb.0[0],bb.0[1],0.0],[bb.1[0],bb.1[1],0.0]), bt);
    } else {
        let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D mesh");
        for _ in 0..args.ref_levels { mesh = refine_uniform_3d(&mesh); }
        let bb = mesh.bounding_box();
        let bt = mesh.unique_boundary_tags();
        tag_pml(&mut mesh, &prob);
        solve_pml(mesh, &args, prob, exact_known, (bb.0, bb.1), bt);
    }
}

fn tag_pml<const D: usize>(mesh: &mut Mesh<D>, prob: &Prob) {
    let bb = mesh.bounding_box();
    let len = match prob { Prob::Disc=>0.2, Prob::Lshape=>0.1, Prob::Fichera=>0.5, Prob::Beam=>2.0, Prob::LoadSrc=>0.25 };
    for e in 0..mesh.n_elems() as u32 {
        let mut in_pml = false;
        for &v in mesh.elem_nodes(e) {
            let c = mesh.coords_of(v);
            if (0..D).any(|d| c[d] < bb.0[d]+len || c[d] > bb.1[d]-len) { in_pml = true; break; }
        }
        if in_pml { mesh.elem_tags[e as usize] = 2; }
    }
}
