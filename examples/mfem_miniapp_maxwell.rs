//! Maxwell Mini App: Time-Domain EM [serial 1:1 translation]
//! ε·∂E/∂t = ∇×(μ⁻¹B) − σE − J, ∂B/∂t = −∇×E
//! cargo run --example mfem_miniapp_maxwell -- -m data/beam-tet.mesh -o 1 -tf 2e-9

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::form::form_linear_system;
use fem_assembly::mixed::assemble_hcurl_hdiv_weak_curl;
use fem_assembly::standard::*;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, HDivSpace};

const EPS0: f64 = 8.8541878176e-12;
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct MaxwellSolver {
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,
    neg_curl: CsrMatrix<f64>,  // −C
    weak_curl_t: CsrMatrix<f64>, // (ν·curl)^T = C^T·μ⁻¹
    m_eps: CsrMatrix<f64>,     // M_ε
    m_mu_inv: CsrMatrix<f64>,  // M_μ⁻¹
    m_loss: Option<CsrMatrix<f64>>, // σ + η⁻¹
    e: Vec<f64>, b: Vec<f64>,
    dbc_dofs: Vec<u32>,
    pub dt: f64, pub t: f64,
}

impl MaxwellSolver {
    pub fn new(mesh: Mesh<3>, order: u8, abc_tags: &[i32], dbc_tags: &[i32]) -> Self {
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0);
        let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs();
        let qo = (2 * order + 1).max(4);
        let dbc_dofs = if !dbc_tags.is_empty() { boundary_dofs_hcurl(&mesh, &nd, dbc_tags) } else { vec![] };

        let c = fem_assembly::discrete_op::DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");
        let neg_curl = { let mut b=c.clone(); for v in b.values.iter_mut(){*v*=-1.0;} b };
        // weak_curl = ∫ ν·curl(ψ)·w dx (HDiv×HCurl) = μ⁻¹·C
        // weak_curl_t = C^T·μ⁻¹ (HCurl×HDiv) matches MFEM's WeakCurlMuInv_
        let nu = 1.0 / MU0;
        let wc = assemble_hcurl_hdiv_weak_curl(&nd, &rt, qo, nu);
        let weak_curl_t = { let mut coo=fem_linalg::CooMatrix::new(n_nd, n_rt);
            for r in 0..n_rt { for ci in wc.row_ptr[r]..wc.row_ptr[r+1] {
                coo.add(wc.col_idx[ci] as usize, r, wc.values[ci]); } }
            coo.into_csr() };

        let m_eps = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator{alpha:EPS0}], qo);
        let m_mu_inv = VectorAssembler::assemble_bilinear(&rt, &[&VectorMassIntegrator{alpha:1.0/MU0}], qo);
        let m_loss = if !abc_tags.is_empty() { Some(VectorAssembler::assemble_bilinear(
            &nd, &[&VectorMassIntegrator{alpha:(EPS0/MU0).sqrt()}], qo)) } else { None };

        Self { nd, rt, neg_curl, weak_curl_t, m_eps, m_mu_inv, m_loss,
            e: vec![0.0; n_nd], b: vec![0.0; n_rt], dbc_dofs, dt: 0.0, t: 0.0 }
    }

    /// Compute the maximum stable time step using power iteration
    /// (MFEM's `GetMaximumTimeStep()`).
    pub fn compute_max_dt(&mut self) -> f64 {
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200,
            verbose: false, ..Default::default() };
        let n_nd = self.nd.n_dofs();
        let n_rt = self.rt.n_dofs();
        // Assume lossless (dt=0 → A₁ = M_ε)
        let a1 = self.m_eps.clone();
        // Pre-solve HDiv mass system
        let hdiv_solve = |u: &[f64], x: &mut [f64]| {
            let (rd, rr, fd, _) = form_linear_system(&self.m_mu_inv, u, &[0u32], &[0.0]);
            let mut xr = vec![0.0; rd.nrows];
            solve_cg(&rd, &rr, &mut xr, &cfg).expect("HDiv mass solve");
            for (i, &d) in fd.iter().enumerate() { x[d as usize] = xr[i]; }
        };
        let hcurl_solve = |z: &[f64], x: &mut [f64]| {
            let (rd, rr, fd, _) = form_linear_system(&a1, z, &[0u32], &[0.0]);
            let mut xr = vec![0.0; rd.nrows];
            solve_cg(&rd, &rr, &mut xr, &cfg).expect("HCurl mass solve");
            for (i, &d) in fd.iter().enumerate() { x[d as usize] = xr[i]; }
        };

        let nstep = 20usize;
        let ptol = 0.001_f64;
        let mut v0 = vec![0.0_f64; n_nd];
        let mut v1 = vec![0.0_f64; n_nd];
        let mut u = vec![0.0_f64; n_rt];
        let mut w = vec![0.0_f64; n_rt];
        let mut z = vec![0.0_f64; n_nd];

        // Randomize v0 (deterministic seed for reproducibility)
        for i in 0..n_nd { v0[i] = ((i * 12347 + 54321) % 1000) as f64 / 500.0 - 1.0; }

        let mut dt0 = 1.0;
        for _iter in 0..nstep {
            let norm_v0: f64 = v0.iter().map(|v| v * v).sum();
            if norm_v0 > 0.0 { for v in v0.iter_mut() { *v /= norm_v0.sqrt(); } }

            // u = NegCurl * v0
            self.neg_curl.spmv(&v0, &mut u);

            // w = M_mu_inv^{-1} * u
            hdiv_solve(&u, &mut w);

            // z = C^T·μ⁻¹·w  (using weak_curl_t)
            self.weak_curl_t.spmv(&w, &mut z);

            // v1 = A1^{-1} * z
            hcurl_solve(&z, &mut v1);

            let lambda: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum::<f64>()
                / v0.iter().map(|v| v * v).sum::<f64>().max(1e-30);

            let dt1 = 2.0 / lambda.sqrt();
            let change = ((dt1 - dt0) / dt0).abs();
            dt0 = dt1;
            if change < ptol { break; }
            std::mem::swap(&mut v0, &mut v1);
        }
        dt0
    }

    /// Advance one time step using MFEM's symplectic scheme:
    /// 1. E^{n+1} = E^n + dt·M_ε^{-1}·(C^T·μ⁻¹·B^n − L·E^n − J)
    /// 2. B^{n+1} = B^n − dt·C·E^{n+1}
    pub fn step(&mut self, j_fn: &(dyn Fn(&[f64], f64, &mut [f64]) + Send + Sync)) {
        let dt = self.dt;
        let cfg = SolverConfig{rtol:1e-12,atol:0.0,max_iter:500,verbose:false,..Default::default()};

        // MFEM symplectic scheme (lossless):
        //   1. dEdt = M_ε^{-1} · W · B^n  (rhs = C^T·μ⁻¹·B^n)
        //   2. E^{n+1} = E^n + dt · dEdt
        //   3. B^{n+1} = B^n - dt · C · E^{n+1}  (uses updated E)

        // 1. rhs = W · B^n  (W = C^T·μ⁻¹ from weak_curl_t)
        let mut rhs = vec![0.0; self.nd.n_dofs()];
        self.weak_curl_t.spmv(&self.b, &mut rhs);
        if let Some(ref l) = self.m_loss {
            let mut le=vec![0.0;self.nd.n_dofs()]; l.spmv(&self.e, &mut le);
            for i in 0..self.nd.n_dofs() { rhs[i] -= le[i]; }
        }
        let jt = self.t;
        let jc = FnVectorCoeff(Box::new(move |x:&[f64],out:&mut[f64]| j_fn(x, jt, out)));
        let jv = VectorAssembler::assemble_linear(&self.nd, &[&VectorDomainLFIntegrator{f:jc}], 15);
        for i in 0..self.nd.n_dofs() { rhs[i] -= jv[i]; }

        // 2. Solve (M_ε + ½dt·L) · ΔE = rhs  → E^{n+1} = E^n + ΔE
        let mut a1 = self.m_eps.clone();
        if let Some(ref l) = self.m_loss {
            let hdt=0.5*dt;
            for r in 0..self.nd.n_dofs() {
                for ci in l.row_ptr[r]..l.row_ptr[r+1] {
                    let j=l.col_idx[ci]as usize;
                    if let Some(p)=a1.find_entry(r,j){a1.values[p]+=hdt*l.values[ci];}
                }
            }
        }
        let zv = vec![0.0; self.dbc_dofs.len()];
        let (rd,rr,fd,_) = form_linear_system(&a1, &rhs, &self.dbc_dofs, &zv);
        let mut xr = vec![0.0; rd.nrows];
        solve_cg(&rd, &rr, &mut xr, &cfg).expect("implicit");
        for (i,&d) in fd.iter().enumerate() { self.e[d as usize] += xr[i]; }

        // 3. B^{n+1} = B^n - dt · C · E^{n+1}
        let mut ce = vec![0.0; self.rt.n_dofs()];
        self.neg_curl.spmv(&self.e, &mut ce);
        for i in 0..self.rt.n_dofs() { self.b[i] += dt * ce[i]; }

        self.t += dt;
    }

    pub fn energy(&self) -> f64 {
        let mut me=vec![0.0;self.nd.n_dofs()]; self.m_eps.spmv(&self.e,&mut me);
        let mut mm=vec![0.0;self.rt.n_dofs()]; self.m_mu_inv.spmv(&self.b,&mut mm);
        0.5*(self.e.iter().zip(me.iter()).map(|(a,b)|a*b).sum::<f64>()
            +self.b.iter().zip(mm.iter()).map(|(a,b)|a*b).sum::<f64>())
    }
}

fn main() {
    let a: Vec<String>=std::env::args().collect(); let mut mf="data/beam-tet.mesh".to_string();
    let mut o:u8=1;let mut r=0usize;let mut tf=2e-9_f64;let mut ts=1e-9_f64;
    let mut i=1;
    while i<a.len(){match a[i].as_str(){
        "-m"|"--mesh"=>{i+=1;if i<a.len(){mf=a[i].clone();}}
        "-o"|"--order"=>{i+=1;if i<a.len(){o=a[i].parse().unwrap_or(1);}}
        "-rs"|"--serial-ref-levels"=>{i+=1;if i<a.len(){r=a[i].parse().unwrap_or(0);}}
        "-tf"|"--final-time"=>{i+=1;if i<a.len(){tf=a[i].parse().unwrap_or(2e-9);}}
        "-ts"|"--snapshot-time"=>{i+=1;if i<a.len(){ts=a[i].parse().unwrap_or(1e-9);}}
        _=>{}}i+=1;}
    let mfem=read_mfem_file(&mf).expect("mesh");let mut mesh:Mesh<3>=mfem.mesh3d.expect("3D");
    for _ in 0..r{mesh=refine_uniform_3d(&mesh);}

    let mut ms=MaxwellSolver::new(mesh,o,&[],&[]);
    ms.dt = 1.0e-10; // conservative CFL for 3D tet mesh
    println!("HCurl {} HDiv {} dt={:.3e}",ms.nd.n_dofs(),ms.rt.n_dofs(),ms.dt);
    let j_fn=|x:&[f64],_t:f64,out:&mut[f64]|{
        out[0]=0.;out[1]=0.;out[2]=if x[0].abs()<0.2&&x[1].abs()<0.2&&x[2].abs()<0.2{1e6}else{0.};};
    let sn=(ts/ms.dt).max(1.0)as usize;let mut n=0;
    while ms.t<tf{ms.step(&j_fn);n+=1;if n%sn==0{
        let en=ms.energy();
        println!("n={n} t={:.3e}s E={:.3e}J |E|₂={:.3e}",ms.t,en,
            ms.e.iter().map(|v|v*v).sum::<f64>().sqrt());}}
    println!("{n} steps to t={:.3e}s",ms.t);
}
