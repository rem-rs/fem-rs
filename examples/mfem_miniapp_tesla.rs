//! # Tesla Mini App: Simple Magnetostatics [serial 1:1 translation]
//!
//! Solves `∇×(ν∇×A) = J` with PEC BC (A×n = 0).
//! Post-processes `B = ∇×A`.
//!
#![allow(dead_code)]
//! ## Usage
//! ```text
//! cargo run --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -o 1 -rs 0
//! ```

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_assembly::mixed::assemble_hcurl_hdiv_mixed;
use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorDomainLFIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::{boundary_dofs_hcurl, eliminate_dirichlet};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct TeslaSolver {
    h1: H1Space<Mesh<3>>,
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,
    stiffness: CsrMatrix<f64>,
    hdiv_mass: CsrMatrix<f64>,
    weak_curl: CsrMatrix<f64>,
    grad: CsrMatrix<f64>,
    curl: CsrMatrix<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    ess_dofs: Vec<u32>,
}

impl TeslaSolver {
    pub fn new(mesh: Mesh<3>, order: u8, bound_all: bool) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0); // RT0 for curl_3d compatibility
        let ess_dofs = if bound_all {
            boundary_dofs_hcurl(&mesh, &nd, &mesh.unique_boundary_tags())
        } else { vec![] };
        let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs();
        let nu = 1.0 / MU0;
        let qo = (2 * order + 1).max(4);

        let stiffness = VectorAssembler::assemble_bilinear(
            &nd, &[&CurlCurlIntegrator { mu: nu },
                   &VectorMassIntegrator { alpha: 1e-12 }], qo);
        let hdiv_mass = VectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let weak_curl = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, nu);
        let grad = DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let curl = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");

        TeslaSolver { h1, nd, rt, stiffness, hdiv_mass, weak_curl, grad, curl,
            a: vec![0.0; n_nd], b: vec![0.0; n_rt], ess_dofs }
    }

    pub fn solve(&mut self, j_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>) {
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500,
            verbose: true, ..Default::default() };
        let n = self.nd.n_dofs();
        let mut rhs = vec![0.0; n];
        if let Some(j) = j_src {
            let src = VectorDomainLFIntegrator {
                f: fem_assembly::coefficient::FnVectorCoeff(j) };
            rhs = VectorAssembler::assemble_linear(&self.nd, &[&src], 15);
        }

        if !self.ess_dofs.is_empty() {
            let zv = vec![0.0; self.ess_dofs.len()];
            let (red, rr, free, _) = eliminate_dirichlet(&self.stiffness, &rhs, &self.ess_dofs, &zv);
            let mut x = vec![0.0; red.nrows];
            solve_cg(&red, &rr, &mut x, &cfg).expect("PCG");
            self.a = vec![0.0; n];
            for (i, &d) in free.iter().enumerate() { self.a[d as usize] = x[i]; }
        } else {
            // Not well-defined without BCs (curl-curl is singular)
            panic!("Tesla requires at least one Dirichlet BC");
        }
        self.curl.spmv(&self.a, &mut self.b);
    }

    pub fn sizes(&self) -> (usize,usize,usize) {
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

    let mut s = TeslaSolver::new(mesh, order, true);
    let (h1, nd, rt) = s.sizes();
    println!("H1 {h1} HCurl {nd} HDiv {rt}");

    s.solve(Some(Box::new(|x: &[f64], out: &mut [f64]| {
        out[0]=0.0; out[1]=0.0;
        out[2]=if x[0].abs()<0.3&&x[1].abs()<0.3{1e6}else{0.0};
    })));
    let an = s.a.iter().map(|v|v*v).sum::<f64>().sqrt();
    let bn = s.b.iter().map(|v|v*v).sum::<f64>().sqrt();
    println!("|A|₂={an:.6e}  |B|₂={bn:.6e}");
}
