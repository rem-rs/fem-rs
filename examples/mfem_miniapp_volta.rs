//! # Volta Mini App: Simple Electrostatics [1:1 serial translation]
//!
//! Solves `−∇·(ε ∇V) = ρ`, post-processes `E = −∇V`, `D = εE`, `ρ = ∇·D`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_miniapp_volta -- -m data/beam-tri.mesh -o 1 -rs 1
//! ```

use fem_assembly::assembler::Assembler;
use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_assembly::mixed::ref_elem_vec;
use fem_assembly::standard::{DiffusionIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{refine_uniform, topology::MeshTopology, ElementType, Mesh};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::{boundary_dofs, eliminate_dirichlet};
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

const EPSILON0: f64 = 8.854187817e-12;

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

    phi: Vec<f64>,
    e: Vec<f64>,
    d: Vec<f64>,
    rho: Vec<f64>,

    ess_dofs: Vec<u32>,
}

impl VoltaSolver {
    pub fn new(mesh: Mesh<2>, order: u8, dbcs: &[i32]) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let ess_dofs = if !dbcs.is_empty() {
            boundary_dofs(&mesh, h1.dof_manager(), dbcs)
        } else { vec![] };
        let n_h1 = h1.n_dofs();

        let nd = HCurlSpace::new(mesh.clone(), order);
        let n_nd = nd.n_dofs();
        let rt = HDivSpace::new(mesh.clone(), order.max(1));
        let n_rt = rt.n_dofs();
        let l2 = L2Space::new(mesh.clone(), order.max(1));
        let n_l2 = l2.n_dofs();
        let qo = (2 * order + 1).max(4);

        let div_eps_grad = Assembler::assemble_bilinear(
            &h1, &[&DiffusionIntegrator { kappa: EPSILON0 }], qo);
        let hdiv_mass = VectorAssembler::assemble_bilinear(
            &rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let hcurl_hdiv_eps = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, EPSILON0);
        let grad = DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let div = DiscreteLinearOperator::divergence(&rt, &l2).expect("divergence");

        VoltaSolver {
            h1, nd, rt, l2,
            div_eps_grad, hdiv_mass, hcurl_hdiv_eps,
            grad, div,
            phi: vec![0.0; n_h1],
            e: vec![0.0; n_nd],
            d: vec![0.0; n_rt],
            rho: vec![0.0; n_l2],
            ess_dofs,
        }
    }

    pub fn solve(&mut self) {
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500,
            verbose: true, ..Default::default() };
        let n = self.h1.n_dofs();
        let rhs = vec![0.0; n];

        if !self.ess_dofs.is_empty() {
            let bc_vals = vec![0.0; self.ess_dofs.len()];
            let (red, red_rhs, free, _) =
                eliminate_dirichlet(&self.div_eps_grad, &rhs, &self.ess_dofs, &bc_vals);
            let mut x = vec![0.0; red.nrows];
            solve_cg(&red, &red_rhs, &mut x, &cfg).expect("PCG");
            self.phi = vec![0.0; n];
            for (i, &d) in free.iter().enumerate() { self.phi[d as usize] = x[i]; }
        } else {
            let ess = vec![0u32]; let vals = vec![0.0];
            let (red, red_rhs, free, _) =
                eliminate_dirichlet(&self.div_eps_grad, &rhs, &ess, &vals);
            let mut x = vec![0.0; red.nrows];
            solve_cg(&red, &red_rhs, &mut x, &cfg).expect("PCG");
            self.phi = vec![0.0; n];
            for (i, &d) in free.iter().enumerate() { self.phi[d as usize] = x[i]; }
        }

        // E = -G * phi
        self.grad.spmv(&self.phi, &mut self.e);
        self.e.iter_mut().for_each(|v| *v = -*v);

        // D = M_HDiv^{-1} * M_mixed * e
        let mut ed = vec![0.0; self.rt.n_dofs()];
        self.hcurl_hdiv_eps.spmv(&self.e, &mut ed);
        let (red_d, rhs_d, free_d, _) =
            eliminate_dirichlet(&self.hdiv_mass, &ed, &[], &[] as &[f64]);
        let mut x_d = vec![0.0; red_d.nrows];
        let _ = solve_cg(&red_d, &rhs_d, &mut x_d, &cfg);
        self.d = vec![0.0; self.rt.n_dofs()];
        for (i, &d) in free_d.iter().enumerate() { self.d[d as usize] = x_d[i]; }

        // rho = div(D)
        self.div.spmv(&self.d, &mut self.rho);
    }

    pub fn sizes(&self) -> (usize,usize,usize,usize) {
        (self.h1.n_dofs(), self.nd.n_dofs(), self.rt.n_dofs(), self.l2.n_dofs())
    }
}

fn assemble_hcurl_hdiv_mixed(
    nd: &HCurlSpace<Mesh<2>>, rt: &HDivSpace<Mesh<2>>, qo: u8, eps: f64,
) -> CsrMatrix<f64> {
    let mesh = nd.mesh(); let dim = 2usize;
    let mut coo = CooMatrix::new(rt.n_dofs(), nd.n_dofs());
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let nd_ref = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap();
        let rt_ref = ref_elem_vec(et, rt.order(), SpaceType::HDiv).unwrap();
        let nd_s = nd.element_signs(e); let rt_s = rt.element_signs(e);
        let g_nd: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();
        let g_rt: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nnd=nd_ref.n_dofs(); let nrt=rt_ref.n_dofs();
        let quad=nd_ref.quadrature(qo);
        let mut me=vec![0.0;g_rt.len()*g_nd.len()];
        let mut nb=vec![0.0;nnd*dim]; let mut rb=vec![0.0;nrt*dim];
        let iso=!matches!(et,ElementType::Tri3|ElementType::Tet4|ElementType::Line2);
        let ge=if iso{geo_ref_elem_from_mesh(mesh,e)}else{None};
        let nodes=mesh.element_nodes(e);
        for(qi,xi) in quad.points.iter().enumerate(){
            let(w,jit,det_j)=if iso{
                let g:&dyn ReferenceElement=ge.as_deref().unwrap();
                let(jac,det,_)=isoparametric_jacobian(mesh,&nodes,g,xi,dim);
                (quad.weights[qi]*det.abs(),jac.try_inverse().unwrap().transpose(),det)
            }else{
                let tr=fem_mesh::ElementTransformation::from_simplex_nodes(mesh,nodes);
                (quad.weights[qi]*tr.det_j().abs(),tr.jacobian_inv_t().clone(),tr.det_j())
            };
            nd_ref.eval_basis_vec(xi,&mut nb); rt_ref.eval_basis_vec(xi,&mut rb);
            let jac=jit.clone().try_inverse().map(|m|m.transpose())
                .unwrap_or_else(||nalgebra::DMatrix::identity(2,2));
            for j in 0..g_nd.len(){
                let sj=nd_s.get(j).copied().unwrap_or(1.0);
                let ndx=sj*(jit[(0,0)]*nb[j*dim]+jit[(0,1)]*nb[j*dim+1]);
                let ndy=sj*(jit[(1,0)]*nb[j*dim]+jit[(1,1)]*nb[j*dim+1]);
                for i in 0..g_rt.len(){
                    let sj2=rt_s.get(i).copied().unwrap_or(1.0);
                    let id=1.0/det_j;
                    let rtx=sj2*id*(jac[(0,0)]*rb[i*dim]+jac[(0,1)]*rb[i*dim+1]);
                    let rty=sj2*id*(jac[(1,0)]*rb[i*dim]+jac[(1,1)]*rb[i*dim+1]);
                    me[i*g_nd.len()+j]+=w*eps*(ndx*rtx+ndy*rty);
                }
            }
        }
        for(ir,&r) in g_rt.iter().enumerate(){
            for(ic,&c) in g_nd.iter().enumerate(){
                let v=me[ir*g_nd.len()+ic];
                if v!=0.0{coo.add(r,c,v);}
            }
        }
    }
    coo.into_csr()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order: u8 = 1;
    let mut refs = 0usize;
    let mut dbcs: Vec<i32> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m"|"--mesh" => { i+=1; if i<args.len() { mesh_file=args[i].clone(); }}
            "-o"|"--order" => { i+=1; if i<args.len() { order=args[i].parse().unwrap_or(1); }}
            "-rs"|"--serial-ref-levels" => { i+=1; if i<args.len() { refs=args[i].parse().unwrap_or(0); }}
            "-dbcs"|"--dirichlet-bc-surf" => { i+=1; while i<args.len() && !args[i].starts_with('-') { dbcs.push(args[i].parse().unwrap_or(0)); i+=1; } continue; }
            _ => {}
        }
        i += 1;
    }
    let mfem = read_mfem_file(&mesh_file).expect("mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("2D");
    for _ in 0..refs { mesh = refine_uniform(&mesh); }
    eprintln!("mesh={mesh_file} o={order} r={refs} bdr={:?}", mesh.unique_boundary_tags());
    let use_dbcs = if !dbcs.is_empty() { dbcs.clone() } else { vec![] };
    let mut s = VoltaSolver::new(mesh, order, &use_dbcs);
    let (h,nd,rt,l2)=s.sizes();
    println!("H1 {h} HCurl {nd} HDiv {rt} L2 {l2}");
    s.solve();
    let pn = s.phi.iter().map(|v|v*v).sum::<f64>().sqrt();
    let en = s.e.iter().map(|v|v*v).sum::<f64>().sqrt();
    let dn = s.d.iter().map(|v|v*v).sum::<f64>().sqrt();
    let rn = s.rho.iter().map(|v|v*v).sum::<f64>().sqrt();
    println!("|phi|={pn:.6e}  |E|={en:.6e}  |D|={dn:.6e}  |rho|={rn:.6e}");
}
