//! Volta Mini App: Electrostatics [serial 1:1 translation]
//! cargo run --example mfem_miniapp_volta -- -m data/beam-tri.mesh -o 1

use fem_assembly::assembler::Assembler;
use fem_assembly::form::form_linear_system;
use fem_assembly::form::recover_fem_solution;
use fem_assembly::mixed::assemble_hcurl_hdiv_mixed;
use fem_assembly::standard::*;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

const EPS0: f64 = 8.8541878176e-12;

pub struct VoltaSolver {
    h1: H1Space<Mesh<2>>,
    nd: HCurlSpace<Mesh<2>>,
    rt: HDivSpace<Mesh<2>>,
    l2: L2Space<Mesh<2>>,
    div_eps_grad: CsrMatrix<f64>,
    hdiv_mass: CsrMatrix<f64>,
    hcurl_hdiv_eps: CsrMatrix<f64>,
    grad: CsrMatrix<f64>,
    div: CsrMatrix<f64>,
    phi: Vec<f64>, e: Vec<f64>, d: Vec<f64>, rho: Vec<f64>,
    ess_tdofs: Vec<u32>,
}

impl VoltaSolver {
    pub fn new(mesh: Mesh<2>, order: u8, dbcs: &[i32]) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), order.max(1));
        let l2 = L2Space::new(mesh.clone(), order.max(1));
        let n_h1 = h1.n_dofs(); let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs(); let n_l2 = l2.n_dofs();
        let qo = (2 * order + 1).max(4);
        let ess_tdofs = if !dbcs.is_empty() {
            let dm = h1.dof_manager();
            fem_space::constraints::dirichlet::boundary_dofs(&mesh, dm, dbcs)
        } else { vec![] };

        let div_eps_grad = Assembler::assemble_bilinear(&h1, &[&DiffusionIntegrator { kappa: EPS0 }], qo);
        let hdiv_mass = VectorAssembler::assemble_bilinear(&rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let hcurl_hdiv_eps = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, EPS0);
        let grad = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let div = fem_assembly::discrete_op::DiscreteLinearOperator::divergence(&rt, &l2).expect("divergence");

        Self { h1, nd, rt, l2, div_eps_grad, hdiv_mass, hcurl_hdiv_eps, grad, div,
            phi: vec![0.0; n_h1], e: vec![0.0; n_nd], d: vec![0.0; n_rt], rho: vec![0.0; n_l2],
            ess_tdofs }
    }

    pub fn solve(&mut self) {
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500,
            verbose: true, ..Default::default() };
        let n = self.h1.n_dofs();
        let rhs = vec![0.0; n];
        let bv = vec![0.0; self.ess_tdofs.len()];
        let (red, rr, free, constrained) = form_linear_system(&self.div_eps_grad, &rhs, &self.ess_tdofs, &bv);
        let mut x = vec![0.0; red.nrows];
        solve_cg(&red, &rr, &mut x, &cfg).expect("PCG");
        self.phi = recover_fem_solution(&x, &free, &constrained, &bv, n);

        self.grad.spmv(&self.phi, &mut self.e);
        self.e.iter_mut().for_each(|v| *v = -*v);

        let mut ed = vec![0.0; self.rt.n_dofs()];
        self.hcurl_hdiv_eps.spmv(&self.e, &mut ed);
        let (rd, rrd, fd, cd) = form_linear_system(&self.hdiv_mass, &ed, &[], &[] as &[f64]);
        let mut xd = vec![0.0; rd.nrows];
        solve_cg(&rd, &rrd, &mut xd, &cfg).expect("D solve");
        self.d = recover_fem_solution(&xd, &fd, &cd, &[] as &[f64], self.rt.n_dofs());
        self.div.spmv(&self.d, &mut self.rho);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut mf = "data/beam-tri.mesh".to_string(); let mut o: u8 = 1; let mut r = 0usize;
    let mut db: Vec<i32> = Vec::new(); let mut i = 1;
    while i < a.len() { match a[i].as_str() {
        "-m"|"--mesh" => { i+=1; if i<a.len() { mf=a[i].clone(); }}
        "-o"|"--order" => { i+=1; if i<a.len() { o=a[i].parse().unwrap_or(1); }}
        "-rs"|"--serial-ref-levels" => { i+=1; if i<a.len() { r=a[i].parse().unwrap_or(0); }}
        "-dbcs"|"--dirichlet-bc-surf" => { i+=1; while i<a.len() && !a[i].starts_with('-') { db.push(a[i].parse().unwrap_or(0)); i+=1; } continue; }
        _ => {} } i+=1; }
    let mfem = read_mfem_file(&mf).expect("mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("2D");
    for _ in 0..r { mesh = refine_uniform(&mesh); }
    eprintln!("mesh={mf} o={o} r={r}");

    let mut s = VoltaSolver::new(mesh, o, &db);
    println!("H1 {} HCurl {} HDiv {} L2 {}", s.h1.n_dofs(), s.nd.n_dofs(), s.rt.n_dofs(), s.l2.n_dofs());
    s.solve();
    println!("|φ|={:.6e} |E|={:.6e} |D|={:.6e} |ρ|={:.6e}",
             s.phi.iter().map(|v|v*v).sum::<f64>().sqrt(),
             s.e.iter().map(|v|v*v).sum::<f64>().sqrt(),
             s.d.iter().map(|v|v*v).sum::<f64>().sqrt(),
             s.rho.iter().map(|v|v*v).sum::<f64>().sqrt());
}
