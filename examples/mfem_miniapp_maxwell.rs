//! # Maxwell Mini App: Time-Domain EM [serial 1:1 translation]
//!
//! Solves the first-order Maxwell system with implicit time stepping:
//! ```text
//! ε·∂E/∂t = ∇×(μ⁻¹·B) − σE − J
//!    ∂B/∂t = −∇×E
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_miniapp_maxwell -- -m data/beam-tet.mesh -o 1 -tf 2
//! ```

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_assembly::standard::*;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::{boundary_dofs_hcurl, eliminate_dirichlet};
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, HDivSpace};

const EPS0: f64 = 8.8541878176e-12;
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct MaxwellSolver {
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,

    neg_curl: CsrMatrix<f64>,  // −C: HCurl → HDiv
    curl_t: CsrMatrix<f64>,    // C^T: HDiv → HCurl

    m_eps: CsrMatrix<f64>,     // M_ε (HCurl mass)
    m_mu_inv: CsrMatrix<f64>,  // M_μ⁻¹ (HDiv mass)
    m_loss: Option<CsrMatrix<f64>>, // σ·M_ε + η⁻¹·M_bdr

    e: Vec<f64>,  // Electric field
    b: Vec<f64>,  // Magnetic flux

    dbc_dofs: Vec<u32>,

    pub dt: f64,
    pub t: f64,
}

impl MaxwellSolver {
    pub fn new(mesh: Mesh<3>, order: u8, abc_tags: &[i32], dbc_tags: &[i32]) -> Self {
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0);
        let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs();
        let qo = (2 * order + 1).max(4);

        let dbc_dofs = if !dbc_tags.is_empty() {
            boundary_dofs_hcurl(&mesh, &nd, dbc_tags)
        } else { vec![] };

        // Curl: HCurl → HDiv (store −C and C^T)
        let c = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");
        let neg_curl = scale_csr(&c, -1.0);
        let curl_t = transpose_csr(&c);

        // Mass matrices
        let m_eps = VectorAssembler::assemble_bilinear(
            &nd, &[&VectorMassIntegrator { alpha: EPS0 }], qo);
        let m_mu_inv = VectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 / MU0 }], qo);

        // Loss (absorbing BC)
        let m_loss = if !abc_tags.is_empty() {
            let eta_inv = (EPS0 / MU0).sqrt();
            Some(VectorAssembler::assemble_bilinear(
                &nd, &[&VectorMassIntegrator { alpha: eta_inv }], qo))
        } else { None };

        // CFL estimate
        let bb = mesh.bounding_box();
        let h: f64 = bb.0.iter().zip(bb.1.iter()).map(|(a,b)| b - a).sum::<f64>()
            / (mesh.n_nodes() as f64).powf(1.0/3.0);
        let dt = h / (order as f64 + 1.0) / 3.0e8;

        Self { nd, rt, neg_curl, curl_t, m_eps, m_mu_inv, m_loss,
            e: vec![0.0; n_nd], b: vec![0.0; n_rt],
            dbc_dofs, dt, t: 0.0 }
    }

    /// Advance one time step.
    pub fn step(&mut self, j_fn: &(dyn Fn(&[f64], f64, &mut [f64]) + Send + Sync)) {
        let dt = self.dt;
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200,
            verbose: false, ..Default::default() };

        // 1. rhs = C^T · B − L · E − J
        let mut rhs = vec![0.0; self.nd.n_dofs()];
        self.curl_t.spmv(&self.b, &mut rhs);
        if let Some(ref loss) = self.m_loss {
            let mut le = vec![0.0; self.nd.n_dofs()];
            loss.spmv(&self.e, &mut le);
            for i in 0..self.nd.n_dofs() { rhs[i] -= le[i]; }
        }
        let jt = self.t;
        let j_coef = FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| j_fn(x, jt, out)));
        let j_vec = VectorAssembler::assemble_linear(&self.nd, &[&VectorDomainLFIntegrator { f: j_coef }], 15);
        for i in 0..self.nd.n_dofs() { rhs[i] -= j_vec[i]; }

        // 2. A1 = M_ε + ½dt·L; solve A1·ΔE = rhs
        let mut a1 = self.m_eps.clone();
        if let Some(ref loss) = self.m_loss {
            let hdt = 0.5 * dt;
            for r in 0..self.nd.n_dofs() {
                for ci in loss.row_ptr[r]..loss.row_ptr[r+1] {
                    let j = loss.col_idx[ci] as usize;
                    if let Some(p) = a1.find_entry(r, j) {
                        a1.values[p] += hdt * loss.values[ci];
                    }
                }
            }
        }
        let zv = vec![0.0; self.dbc_dofs.len()];
        let (red, rr, free, _) = eliminate_dirichlet(&a1, &rhs, &self.dbc_dofs, &zv);
        let mut xr = vec![0.0; red.nrows];
        solve_cg(&red, &rr, &mut xr, &cfg).expect("Maxwell implicit step");
        let mut de = vec![0.0; self.nd.n_dofs()];
        for (i, &d) in free.iter().enumerate() { de[d as usize] = xr[i]; }

        // 3. E^{n+1} = E^n + ΔE
        for i in 0..self.nd.n_dofs() { self.e[i] += de[i]; }

        // 4. B^{n+1} = B^n + dt · (−C) · E^{n+1}
        let mut ce = vec![0.0; self.rt.n_dofs()];
        self.neg_curl.spmv(&self.e, &mut ce);
        for i in 0..self.rt.n_dofs() { self.b[i] += dt * ce[i]; }

        self.t += dt;
    }

    pub fn energy(&self) -> f64 {
        let mut me = vec![0.0; self.nd.n_dofs()];
        self.m_eps.spmv(&self.e, &mut me);
        let mut mm = vec![0.0; self.rt.n_dofs()];
        self.m_mu_inv.spmv(&self.b, &mut mm);
        let ep: f64 = self.e.iter().zip(me.iter()).map(|(a,b)| a*b).sum();
        let bp: f64 = self.b.iter().zip(mm.iter()).map(|(a,b)| a*b).sum();
        0.5 * (ep + bp)
    }

    pub fn sizes(&self) -> (usize, usize) {
        (self.nd.n_dofs(), self.rt.n_dofs())
    }
}

fn scale_csr(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    let mut b = a.clone();
    for v in b.values.iter_mut() { *v *= s; }
    b
}

fn transpose_csr(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let nrows = a.ncols;
    let ncols = a.nrows;
    let mut nnz = vec![0usize; nrows];
    for r in 0..a.nrows {
        for c in a.row_ptr[r]..a.row_ptr[r+1] {
            nnz[a.col_idx[c] as usize] += 1;
        }
    }
    let mut row_ptr = vec![0usize; nrows + 1];
    for i in 0..nrows { row_ptr[i+1] = row_ptr[i] + nnz[i]; }
    let mut col_idx = vec![0u32; row_ptr[nrows]];
    let mut values = vec![0.0; row_ptr[nrows]];
    let mut pos = row_ptr[..nrows].to_vec();
    for r in 0..a.nrows {
        for c in a.row_ptr[r]..a.row_ptr[r+1] {
            let ci = a.col_idx[c] as usize;
            let p = pos[ci];
            col_idx[p] = r as u32;
            values[p] = a.values[c];
            pos[ci] += 1;
        }
    }
    CsrMatrix { nrows, ncols, row_ptr, col_idx, values }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/beam-tet.mesh".to_string();
    let mut order: u8 = 1; let mut refs = 0usize;
    let mut tf = 2e-9_f64; let mut ts = 0.5e-9_f64;
    let mut abcs: Vec<i32> = Vec::new();
    let mut dbcs: Vec<i32> = Vec::new();

    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "-m"|"--mesh" => { i+=1; if i<a.len() { mesh_file=a[i].clone(); }}
            "-o"|"--order" => { i+=1; if i<a.len() { order=a[i].parse().unwrap_or(1); }}
            "-rs"|"--serial-ref-levels" => { i+=1; if i<a.len() { refs=a[i].parse().unwrap_or(0); }}
            "-tf"|"--final-time" => { i+=1; if i<a.len() { tf=a[i].parse().unwrap_or(2e-9); }}
            "-ts"|"--snapshot-time" => { i+=1; if i<a.len() { ts=a[i].parse().unwrap_or(0.5e-9); }}
            "-abcs"|"--absorbing-bc-surf" => { i+=1; while i<a.len() && !a[i].starts_with('-') { abcs.push(a[i].parse().unwrap_or(0)); i+=1; } continue; }
            "-dbcs"|"--dirichlet-bc-surf" => { i+=1; while i<a.len() && !a[i].starts_with('-') { dbcs.push(a[i].parse().unwrap_or(0)); i+=1; } continue; }
            _ => {}
        }
        i += 1;
    }
    let mfem = read_mfem_file(&mesh_file).expect("mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D");
    for _ in 0..refs { mesh = refine_uniform_3d(&mesh); }
    eprintln!("mesh={mesh_file} o={order} r={refs}");

    let mut ms = MaxwellSolver::new(mesh, order, &abcs, &dbcs);
    let (nd, rt) = ms.sizes();
    println!("HCurl {nd}  HDiv {rt}  dt={:.3e}", ms.dt);

    let j_fn = |x: &[f64], _t: f64, out: &mut [f64]| {
        out[0]=0.0; out[1]=0.0;
        out[2]=if x[0].abs()<0.2&&x[1].abs()<0.2&&x[2].abs()<0.2{1e6}else{0.0};
    };

    let snap = (ts / ms.dt).max(1.0) as usize;
    let mut n = 0;
    while ms.t < tf {
        ms.step(&j_fn);
        n += 1;
        if n % snap == 0 {
            println!("t={:.3e}s  E={:.3e}J  |E|₂={:.3e}  |B|₂={:.3e}",
                     ms.t, ms.energy(),
                     ms.e.iter().map(|v|v*v).sum::<f64>().sqrt(),
                     ms.b.iter().map(|v|v*v).sum::<f64>().sqrt());
        }
    }
    println!("{n} steps to t={:.3e}s", ms.t);
}
