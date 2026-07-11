#![allow(dead_code)]
//! # Tesla Mini App: Simple Magnetostatics [serial 1:1 translation]
//!
//! Solves `∇×(ν∇×A) = J` in 3D with PEC BC using AMS preconditioned CG.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -o 1
//! ```

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_assembly::mixed::assemble_hcurl_hdiv_mixed;
use fem_assembly::standard::{CurlCurlIntegrator, VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{fem_to_linlvo_csr, CsrMatrix};
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::div_free::project_divergence_free;
use fem_solver::{solve_cg, solve_pcg_ams, AmsSolverConfig, SolverConfig};
use fem_space::constraints::dirichlet::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct TeslaSolver {
    // Spaces
    h1: H1Space<Mesh<3>>,
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,

    // Bilinear forms
    curl_mu_inv_curl: CsrMatrix<f64>,  // ∇×(ν∇×)
    h_curl_mass: CsrMatrix<f64>,       // ∫u·v on HCurl
    h_div_h_curl_mu_inv: CsrMatrix<f64>, // Mixed HCurl×HDiv mass (ν)

    // Discrete operators
    grad: CsrMatrix<f64>,  // H1 → HCurl
    curl: CsrMatrix<f64>,  // HCurl → HDiv

    // Solution vectors
    a: Vec<f64>,  // Vector potential (HCurl)
    b: Vec<f64>,  // Magnetic flux (HDiv)
    // BCs
    ess_tdofs: Vec<u32>,
}

impl TeslaSolver {
    pub fn new(mesh: Mesh<3>, order: u8) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0); // RT0 for curl_3d compat
        let n_nd = nd.n_dofs();
        let n_rt = rt.n_dofs();
        let ess_tdofs = {
            let tags = mesh.unique_boundary_tags();
            boundary_dofs_hcurl(&mesh, &nd, &tags)
        };
        let nu = 1.0 / MU0;
        let qo = (2 * order + 1).max(4);

        // Assembled in constructor (C++ splits Assemble/Solve)
        let curl_mu_inv_curl = VectorAssembler::assemble_bilinear(
            &nd, &[&CurlCurlIntegrator { mu: nu }], qo);
        let h_curl_mass = VectorAssembler::assemble_bilinear(
            &nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let h_div_h_curl_mu_inv = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, nu);
        let grad = DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let curl = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");

        TeslaSolver {
            h1, nd, rt,
            curl_mu_inv_curl, h_curl_mass, h_div_h_curl_mu_inv,
            grad, curl,
            a: vec![0.0; n_nd],
            b: vec![0.0; n_rt],
            ess_tdofs,
        }
    }

    /// Solve ∇×(ν∇×A) = J with PEC BC.
    ///
    /// Equivalent to MFEM's `TeslaSolver::Solve()`:
    /// 1. Compute divergence-free current j_ from j_src
    /// 2. AMS PCG for A
    /// 3. B = ∇×A
    /// 4. H = solve M_HCurl * h = M_mixed * B
    pub fn solve(
        &mut self,
        j_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
    ) {
        let n_nd = self.nd.n_dofs();

        // ── 1a. RHS from current density J ──────────────────────────────
        let mut j = vec![0.0; n_nd];
        if let Some(j_fn) = j_src {
            let src = VectorDomainLFIntegrator { f: FnVectorCoeff(j_fn) };
            j = VectorAssembler::assemble_linear(&self.nd, &[&src], 15);
        }

        // ── 1b. Divergence-free projection of J ──────────────────────────
        let cfg_h1 = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 200,
            verbose: false, ..Default::default()
        };
        let solve_h1 = |a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]| {
            // Pin first DOF to handle gradient nullspace
            use fem_space::constraints::dirichlet::eliminate_dirichlet;
            let ess = vec![0u32];
            let vals = vec![0.0];
            let (red, rb, free, _) = eliminate_dirichlet(a, b, &ess, &vals);
            let mut xr = vec![0.0; red.nrows];
            solve_cg(&red, &rb, &mut xr, &cfg_h1).expect("H1 coarse solve");
            for (i, &d) in free.iter().enumerate() { x[d as usize] = xr[i]; }
        };
        project_divergence_free(&mut j, &self.grad, &self.h_curl_mass,
                                self.h1.n_dofs(), &solve_h1);

        // ── 1c. jd_ = M * j  (mass-matrix-weighted RHS) ─────────────────
        let mut jd = vec![0.0; n_nd];
        self.h_curl_mass.spmv(&j, &mut jd);

        // ── 2. AMS solve for A ──────────────────────────────────────────
        // Apply PEC BCs (AMS-compatible: zero rows/cols)
        let mut a_ams = self.curl_mu_inv_curl.clone();
        for &d in &self.ess_tdofs {
            a_ams.apply_dirichlet_symmetric(d as usize, 0.0, &mut jd);
            if let Some(k) = a_ams.find_entry(d as usize, d as usize) {
                a_ams.values[k] = 1.0;
            }
        }

        let g_linlvo = fem_to_linlvo_csr(&self.grad);
        let ams_cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-12, atol: 0.0, max_iter: 200,
                verbose: true, ..Default::default()
            },
            ..Default::default()
        };
        let mut a = vec![0.0; n_nd];
        solve_pcg_ams(&a_ams, &g_linlvo, &jd, &mut a, &ams_cfg)
            .expect("AMS PCG");
        self.a = a;

        // ── 3. B = ∇×A ─────────────────────────────────────────────────
        self.curl.spmv(&self.a, &mut self.b);
    }

    pub fn sizes(&self) -> (usize, usize, usize) {
        (self.h1.n_dofs(), self.nd.n_dofs(), self.rt.n_dofs())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/beam-tet.mesh".to_string();
    let mut order: u8 = 1;
    let mut refs = 0usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m"|"--mesh" => { i+=1; if i<args.len() { mesh_file = args[i].clone(); }}
            "-o"|"--order" => { i+=1; if i<args.len() { order = args[i].parse().unwrap_or(1); }}
            "-rs"|"--serial-ref-levels" => { i+=1; if i<args.len() { refs = args[i].parse().unwrap_or(0); }}
            _ => {}
        }
        i += 1;
    }
    let mfem = read_mfem_file(&mesh_file).expect("mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D mesh");
    for _ in 0..refs { mesh = refine_uniform_3d(&mesh); }
    eprintln!("mesh={mesh_file} o={order} r={refs}");

    let mut ts = TeslaSolver::new(mesh, order);
    let (h1, nd, rt) = ts.sizes();
    println!("H1 {h1}  HCurl {nd}  HDiv {rt}");

    let j_coil = Box::new(|x: &[f64], out: &mut [f64]| {
        out[0] = 0.0; out[1] = 0.0;
        out[2] = if x[0].abs() < 0.3 && x[1].abs() < 0.3 { 1e6 } else { 0.0 };
    });
    ts.solve(Some(j_coil));

    let an = ts.a.iter().map(|v| v*v).sum::<f64>().sqrt();
    let bn = ts.b.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("|A|₂ = {an:.6e}  |B|₂ = {bn:.6e}");
}
