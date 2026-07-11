#![allow(dead_code)]
//! # Tesla Mini App: Simple Magnetostatics [serial 1:1 translation]
//!
//! Solves `∇×(ν∇×A) = J` with PEC BC using AMS preconditioned CG.
//! Post-processes `B = ∇×A`.
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
use fem_linalg::{fem_to_linlvo_csr, CooMatrix, CsrMatrix};
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::{solve_cg, solve_pcg_ams, AmsSolverConfig, SolverConfig};
use fem_space::constraints::dirichlet::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// Apply divergence-free projection: `jr = rhs - G·(G^T·M·G)^{-1}·G^T·M·rhs`
/// Removes the irrotational (non-divergence-free) part of the RHS.
fn project_div_free(
    rhs: &mut [f64],
    g: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    h1_space: &H1Space<Mesh<3>>,
) {
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200,
        verbose: false, ..Default::default() };
    let n_h1 = h1_space.n_dofs();
    let n_nd = rhs.len();

    // rhoh = G^T * M * rhs  (project RHS onto H1)
    let mut m_rhs = vec![0.0; n_nd];
    m.spmv(rhs, &mut m_rhs);
    let mut rhoh = vec![0.0; n_h1];
    for j in 0..n_nd {
        for r in g.row_ptr[j]..g.row_ptr[j+1] {
            rhoh[g.col_idx[r] as usize] += g.values[r] * m_rhs[j];
        }
    }

    // Build A_h1 = G^T * M * G (H1 coarse operator)
    let mut coo = CooMatrix::new(n_h1, n_h1);
    for i in 0..n_h1 {
        for r in g.row_ptr[i]..g.row_ptr[i+1] {
            let k = g.col_idx[r] as usize;
            let gik = g.values[r];
            for c in g.row_ptr[k]..g.row_ptr[k+1] {
                let j = g.col_idx[c] as usize;
                let gkj = g.values[c];
                // Find M[k,k] (use full M for consistency)
                if let Some(p) = m.find_entry(k, j) {
                    coo.add(i, j, gik * m.values[p] * gkj);
                }
            }
        }
    }
    let a_h1 = coo.into_csr();

    // Solve A_h1 * x = rhoh
    let mut x_h1 = vec![0.0; n_h1];
    let _ = solve_cg(&a_h1, &rhoh, &mut x_h1, &cfg);

    // rhs -= G * x_h1
    let mut gx = vec![0.0; n_nd];
    g.spmv(&x_h1, &mut gx);
    for i in 0..n_nd { rhs[i] -= gx[i]; }
}

/// Impose essential BC by zeroing rows/cols (AMS-compatible).
fn apply_essential_bc_ams(mat: &mut CsrMatrix<f64>, rhs: &mut [f64], dofs: &[u32]) {
    for &d in dofs {
        let d_us = d as usize;
        mat.apply_dirichlet_symmetric(d_us, 0.0, rhs);
        if let Some(k) = mat.find_entry(d_us, d_us) {
            mat.values[k] = 1.0;
        }
    }
}

pub struct TeslaSolver {
    h1: H1Space<Mesh<3>>,
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,
    stiffness: CsrMatrix<f64>,
    hdiv_mass: CsrMatrix<f64>,
    weak_curl: CsrMatrix<f64>,
    g_fem: CsrMatrix<f64>,
    curl: CsrMatrix<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    ess_dofs: Vec<u32>,
}

impl TeslaSolver {
    pub fn new(mesh: Mesh<3>, order: u8) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let n_nd = nd.n_dofs();
        let rt = HDivSpace::new(mesh.clone(), 0); // RT0 for curl_3d compat
        let n_rt = rt.n_dofs();
        let ess_dofs = boundary_dofs_hcurl(&mesh, &nd, &mesh.unique_boundary_tags());
        let nu = 1.0 / MU0;
        let qo = (2 * order + 1).max(4);

        let stiffness = VectorAssembler::assemble_bilinear(
            &nd, &[&CurlCurlIntegrator { mu: nu },
                   &VectorMassIntegrator { alpha: 1e-12 }], qo);
        let hdiv_mass = VectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let weak_curl = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, nu);
        let g_fem = DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let curl = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");

        TeslaSolver {
            h1, nd, rt, stiffness, hdiv_mass, weak_curl,
            g_fem, curl,
            a: vec![0.0; n_nd],
            b: vec![0.0; n_rt],
            ess_dofs,
        }
    }

    /// Solve ∇×(ν∇×A) = J with PEC BC, using AMS preconditioner.
    /// `j_src`: current density J(x) as optional vector function.
    /// `m_src`: magnetization M(x) as optional vector function.
    pub fn solve(
        &mut self,
        j_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
        _m_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
    ) {
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500,
                verbose: true, ..Default::default() },
            ..Default::default()
        };
        let n = self.nd.n_dofs();
        let mut rhs = vec![0.0; n];

        // J source
        if let Some(j) = j_src {
            let src = VectorDomainLFIntegrator { f: FnVectorCoeff(j) };
            rhs = VectorAssembler::assemble_linear(&self.nd, &[&src], 15);
        }

        // Divergence-free projection of RHS
        project_div_free(&mut rhs, &self.g_fem, &self.stiffness, &self.h1);

        // Apply PEC BCs on the FULL system (AMS-compatible: zero rows/cols)
        let mut a_ams = self.stiffness.clone();
        apply_essential_bc_ams(&mut a_ams, &mut rhs, &self.ess_dofs);

        // AMS preconditioned CG
        let g_linlvo = fem_to_linlvo_csr(&self.g_fem);
        self.a = vec![0.0; n];
        solve_pcg_ams(&a_ams, &g_linlvo, &rhs, &mut self.a, &cfg)
            .expect("AMS PCG");

        // B = ∇×A
        self.curl.spmv(&self.a, &mut self.b);
    }

    pub fn sizes(&self) -> (usize, usize, usize) {
        (self.h1.n_dofs(), self.nd.n_dofs(), self.rt.n_dofs())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/beam-tet.mesh".to_string();
    let mut order: u8 = 1; let mut refs = 0usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m"|"--mesh" => { i+=1; if i<args.len() { mesh_file=args[i].clone(); }}
            "-o"|"--order" => { i+=1; if i<args.len() { order=args[i].parse().unwrap_or(1); }}
            "-rs"|"--serial-ref-levels" => { i+=1; if i<args.len() { refs=args[i].parse().unwrap_or(0); }}
            _ => {}
        }
        i += 1;
    }
    let mfem = read_mfem_file(&mesh_file).expect("mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D");
    for _ in 0..refs { mesh = refine_uniform_3d(&mesh); }
    eprintln!("mesh={mesh_file} o={order} r={refs}");

    let mut s = TeslaSolver::new(mesh, order);
    let (h1, nd, rt) = s.sizes();
    println!("H1 {h1}  HCurl {nd}  HDiv {rt}");

    let j_coil = Box::new(|x: &[f64], out: &mut [f64]| {
        out[0] = 0.0; out[1] = 0.0;
        out[2] = if x[0].abs() < 0.3 && x[1].abs() < 0.3 { 1e6 } else { 0.0 };
    });
    s.solve(Some(j_coil), None);

    let an = s.a.iter().map(|v| v*v).sum::<f64>().sqrt();
    let bn = s.b.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("|A|₂ = {an:.6e}  |B|₂ = {bn:.6e}");
}
