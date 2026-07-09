#![allow(dead_code, unused_imports)]
//! MFEM Example 9 — DG Advection (1:1 translation)
//! du/dt + v·∇u = 0, DG upwind. Library matches MFEM sign convention.
//! dudt = M^{-1} · K · u (no negation)
use fem_assembly::{Assembler, dg::dg_advection::{DGAdvectionIntegrator,assemble_dg_interior_faces},
    interior_faces::InteriorFaceList, postproc::coefficient::ConstantVectorCoeff, standard::MassIntegrator};
use fem_mesh::{refine_uniform,topology::MeshTopology,Mesh};
use fem_solver::{SolverConfig,solve_cg,ode::{Rk4,TimeStepper}};
use fem_space::{L2Space,fe_space::FESpace};
use fem_linalg::{CooMatrix,CsrMatrix};
fn main() {
    let a=Args::parse();
    let mesh:Mesh<2>=if!a.mesh.is_empty(){fem_io::mfem::read_mfem_file(&a.mesh).unwrap().mesh2d.unwrap()}else{Mesh::<2>::unit_square_tri(4)};
    let mesh=if a.refine>0{let mut m=mesh;for _ in 0..a.refine{m=refine_uniform(&m);}m}else{mesh};
    let sp=L2Space::new(mesh.clone(),1);let n=sp.n_dofs();println!("Number of unknowns: {n}");
    let mass=Assembler::assemble_bilinear(&sp,&[&MassIntegrator{rho:1.0}],3);
    let vel=[(2.0_f64/3.0).sqrt(),(1.0_f64/3.0).sqrt()];
    let dg=DGAdvectionIntegrator{velocity:ConstantVectorCoeff(vel.to_vec())};
    let k_vol=Assembler::assemble_bilinear(&sp,&[&dg],3);
    let mut coo=CooMatrix::new(n,n);
    for i in 0..n{for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1]{coo.add(i,k_vol.col_idx[p]as usize,k_vol.values[p]);}}
    let ifl=InteriorFaceList::build(sp.mesh());
    assemble_dg_interior_faces(&mut coo,sp.mesh(),&sp,&ifl,1,3,&dg);
    // Outflow boundary (MFEM α=-1): el += -h·vn·φ·φ for vn>0, RHS += +h·vn·g·φ = 0 for vn<0
    for bf in 0..mesh.n_boundary_faces()as u32{let f=mesh.face_nodes(bf);let(el,_)=mesh.face_elements(bf);
        let pa=mesh.node_coords(f[0]);let pb=mesh.node_coords(f[1]);
        let h=((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
        let vn=vel[0]*(pb[1]-pa[1])/h+vel[1]*(-(pb[0]-pa[0]))/h;
        let(dofs,en):(Vec<usize>,_)=(sp.element_dofs(el).iter().map(|&d|d as usize).collect(),mesh.element_nodes(el));
        let find=|n|en.iter().position(|&e|e==n);
        if let(Some(p0),Some(p1))=(find(f[0]),find(f[1])){
            let w=-h*vn;for&pi in&[p0,p1]{for&pj in&[p0,p1]{coo.add(dofs[pi],dofs[pj],w*0.25);}}
        }
    }
    let k_adv=coo.into_csr();
    let mut u:Vec<f64>=sp.interpolate(&|x|(-40.0*((x[0]-0.5).powi(2)+(x[1]-0.5).powi(2))).exp()).as_slice().to_vec();
    let cfg=SolverConfig{rtol:1e-9,max_iter:200,verbose:false,..Default::default()};
    let dt=a.dt.min(a.t_final);let mut t=0.0;
    for _ in 0..(a.t_final/dt).ceil()as usize{let dta=dt.min(a.t_final-t);
        Rk4.step(t,dta,&mut u,|_,u,dudt|{let mut f=vec![0.0;n];k_adv.spmv(u,&mut f);let _=solve_cg(&mass,&f,dudt,&cfg);});
        t+=dta;if t>=a.t_final-1e-14{break;}
    }
    let mut mu=vec![0.0;n];mass.spmv(&u,&mut mu);
    println!("L2={:.10e}",u.iter().zip(mu.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt());
}
struct Args{mesh:String,refine:usize,dt:f64,t_final:f64}
impl Args{fn parse()->Self{let(mut m,mut r,mut dt,mut tf)=(String::new(),2usize,0.005_f64,10.0_f64);let mut it=std::env::args().skip(1);while let Some(a)=it.next(){match a.as_str(){"-m"|"--mesh"=>{m=it.next().unwrap_or(m);}"-r"|"--refine"=>{r=it.next().and_then(|s|s.parse().ok()).unwrap_or(2);}"-dt"|"--time-step"=>{dt=it.next().and_then(|s|s.parse().ok()).unwrap_or(0.005);}"-tf"|"--final-time"=>{tf=it.next().and_then(|s|s.parse().ok()).unwrap_or(10.0);}_=>{}}}Args{mesh:m,refine:r,dt,t_final:tf}}}
