//! Tesla Mini App: Magnetostatics [serial 1:1 translation]
//! Solves ∇×(ν∇×A) = J + ∇×M, PEC BC. B = ∇×A, H = νB−M.
//! cargo run --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -o 1

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::form::{form_linear_system, recover_fem_solution};
use fem_assembly::mixed::{assemble_hcurl_hdiv_mixed, assemble_hcurl_hdiv_weak_curl};
use fem_assembly::standard::*;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::div_free::project_divergence_free;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct TeslaSolver {
    h1: H1Space<Mesh<3>>, nd: HCurlSpace<Mesh<3>>, rt: HDivSpace<Mesh<3>>,
    curl_mu_inv_curl: CsrMatrix<f64>,
    h_curl_mass: CsrMatrix<f64>,
    h_div_h_curl_mu_inv_t: CsrMatrix<f64>,
    weak_curl_mu_inv: Option<CsrMatrix<f64>>,
    weak_curl_mu_inv_t: Option<CsrMatrix<f64>>,
    grad: CsrMatrix<f64>, curl: CsrMatrix<f64>,
    a: Vec<f64>, b: Vec<f64>, h: Vec<f64>,
    ess_dofs: Vec<u32>,
}

impl TeslaSolver {
    pub fn new(mesh: Mesh<3>, order: u8) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0);
        let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs();
        let ess_dofs = boundary_dofs_hcurl(&mesh, &nd, &mesh.unique_boundary_tags());
        let nu = 1.0 / MU0;
        let qo = (2 * order + 1).max(4);

        let curl_mu_inv_curl = VectorAssembler::assemble_bilinear(
            &nd, &[&CurlCurlIntegrator { mu: nu }], qo);
        let h_curl_mass = VectorAssembler::assemble_bilinear(
            &nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let h_div_h_curl_mu_inv_t = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, nu).transpose();
        let grad = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let curl = fem_assembly::discrete_op::DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");

        Self { h1, nd, rt, curl_mu_inv_curl, h_curl_mass, h_div_h_curl_mu_inv_t,
            weak_curl_mu_inv: None, weak_curl_mu_inv_t: None, grad, curl,
            a: vec![0.0; n_nd], b: vec![0.0; n_rt], h: vec![0.0; n_nd], ess_dofs }
    }

    pub fn enable_magnetization(&mut self) { let nu = 1.0/MU0; let qo = (2*self.nd.order()+1).max(4);
        let wc = assemble_hcurl_hdiv_weak_curl(&self.nd, &self.rt, qo, nu);
        self.weak_curl_mu_inv_t = Some(wc.transpose());
        self.weak_curl_mu_inv = Some(wc); }

    pub fn solve(&mut self, j_fn: Option<Box<dyn Fn(&[f64],&mut[f64])+Send+Sync>>,
                 m_fn: Option<Box<dyn Fn(&[f64],&mut[f64])+Send+Sync>>) {
        let n_nd = self.nd.n_dofs();
        let mut jd = vec![0.0; n_nd];

        if let Some(j) = j_fn {
            let mut jv = VectorAssembler::assemble_linear(&self.nd, &[&VectorDomainLFIntegrator{f:FnVectorCoeff(j)}], 15);
            let solve_h1 = |a:&CsrMatrix<f64>,b:&[f64],x:&mut[f64]|{
                let e=vec![0u32];let v=vec![0.0];
                let(r,rr,f,_)=form_linear_system(a,b,&e,&v);
                let mut xr=vec![0.0;r.nrows];solve_cg(&r,&rr,&mut xr,&SolverConfig{rtol:1e-12,..Default::default()}).expect("H1");
                for(i,&d)in f.iter().enumerate(){x[d as usize]=xr[i];}
            };
            project_divergence_free(&mut jv, &self.grad, &self.h_curl_mass, self.h1.n_dofs(), &solve_h1);
            self.h_curl_mass.spmv(&jv, &mut jd);
        }

        if let Some(m) = m_fn {
            if let Some(ref wc_t) = self.weak_curl_mu_inv_t {
                let mv = VectorAssembler::assemble_linear(&self.rt, &[&VectorDomainLFIntegrator{f:FnVectorCoeff(m)}], 15);
                // jd += MU0 * weakCurl^T * mv  (using pre-computed transpose)
                wc_t.spmv_add(MU0, &mv, 1.0, &mut jd);
            }
        }

        // Reduced system with CG (reliable for all mesh sizes)
        let bv = vec![0.0; self.ess_dofs.len()];
        let (red, rr, free, _) = form_linear_system(&self.curl_mu_inv_curl, &jd, &self.ess_dofs, &bv);
        let mut xa = vec![0.0; red.nrows];
        let cfg = SolverConfig{rtol:1e-12,atol:0.0,max_iter:2000,verbose:true,..Default::default()};
        solve_cg(&red, &rr, &mut xa, &cfg).expect("PCG");
        self.a = vec![0.0; n_nd];
        for (i, &d) in free.iter().enumerate() { self.a[d as usize] = xa[i]; }

        self.curl.spmv(&self.a, &mut self.b);
        // bd = h_div_h_curl_mu_inv^T * b  (using pre-computed transpose)
        let mut bd = vec![0.0; n_nd];
        self.h_div_h_curl_mu_inv_t.spmv(&self.b, &mut bd);
        let(rd,rrd,fd,cd)=form_linear_system(&self.h_curl_mass,&bd,&[],&[]as&[f64]);
        let mut xh=vec![0.0;rd.nrows];solve_cg(&rd,&rrd,&mut xh,&SolverConfig{rtol:1e-12,..Default::default()}).expect("H");
        self.h=recover_fem_solution(&xh,&fd,&cd,&[]as&[f64],n_nd);
    }
}

fn main() {
    let a: Vec<String>=std::env::args().collect(); let mut mf="data/beam-tet.mesh".to_string();
    let mut o:u8=1;let mut r=0usize;let mut i=1;
    while i<a.len(){match a[i].as_str(){
        "-m"|"--mesh"=>{i+=1;if i<a.len(){mf=a[i].clone();}}
        "-o"|"--order"=>{i+=1;if i<a.len(){o=a[i].parse().unwrap_or(1);}}
        "-rs"|"--serial-ref-levels"=>{i+=1;if i<a.len(){r=a[i].parse().unwrap_or(0);}}
        _=>{}}i+=1;}
    let mfem=read_mfem_file(&mf).expect("mesh");let mut mesh:Mesh<3>=mfem.mesh3d.expect("3D");
    for _ in 0..r{mesh=refine_uniform_3d(&mesh);}eprintln!("mesh={mf} o={o} r={r}");

    let mut s=TeslaSolver::new(mesh,o);s.enable_magnetization();
    println!("H1 {} HCurl {} HDiv {}",s.h1.n_dofs(),s.nd.n_dofs(),s.rt.n_dofs());
    let jc=Box::new(|x:&[f64],out:&mut[f64]|{out[0]=0.;out[1]=0.;out[2]=if x[0].abs()<0.3&&x[1].abs()<0.3{1e6}else{0.};});
    s.solve(Some(jc),None);
    println!("|A|₂={:.6e} |B|₂={:.6e} |H|₂={:.6e}",
             s.a.iter().map(|v|v*v).sum::<f64>().sqrt(),
             s.b.iter().map(|v|v*v).sum::<f64>().sqrt(),
             s.h.iter().map(|v|v*v).sum::<f64>().sqrt());
}
