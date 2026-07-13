//! Example 21 — AMR for linear elasticity. 1:1 translation of MFEM ex21.

#![allow(non_snake_case, dead_code)]
use fem_assembly::postproc::error_estimate::zz_estimator;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_linalg::SolverConfig;
use fem_mesh::{amr::HangingNodeConstraint, topology::MeshTopology, Mesh};
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::hanging_2d::{apply_hanging_constraints, recover_hanging_values};
use fem_space::{FESpace, VectorH1Space};

fn bdr_load(mesh:&Mesh<2>,sp:&VectorH1Space<Mesh<2>>,attr:u32,_fx:f64,fy:f64)->Vec<f64>{
    use fem_element::{ReferenceElement,lagrange::SegP1};
    let dim=2;let nd=sp.n_dofs();let mut r=vec![0.0;nd];
    let qf:[[usize;2];4]=[[0,1],[1,2],[2,3],[3,0]];
    for e in 0..mesh.n_elements()as u32{let en=mesh.element_nodes(e);if en.len()<4{continue;}
        for&[a,b]in&qf{let(v0,v1)=(en[a],en[b]);
            for f in 0..mesh.n_boundary_faces()as u32{
                let fn_=mesh.face_nodes(f);if fn_.len()<2{continue;}
                if((fn_[0]==v0&&fn_[1]==v1)||(fn_[0]==v1&&fn_[1]==v0))&&mesh.face_tag(f)==attr as i32{
                    let x0=mesh.node_coords(v0);let x1=mesh.node_coords(v1);
                    let el=((x1[0]-x0[0]).powi(2)+(x1[1]-x0[1]).powi(2)).sqrt();
                    let seg=SegP1;let q=seg.quadrature(3);
                    let pa=en.iter().position(|&n|n==v0).unwrap();
                    let pb=en.iter().position(|&n|n==v1).unwrap();
                    for(qi,xi)in q.points.iter().enumerate(){
                        let(ph0,ph1)=(0.5*(1.0-xi[0]),0.5*(1.0+xi[0]));
                        let w=q.weights[qi]*(el/2.0);
                        if pa*dim+1<nd{r[pa*dim+1]+=w*ph0*fy;}
                        if pb*dim+1<nd{r[pb*dim+1]+=w*ph1*fy;}
                    }
                }
            }
        }
    }
    r
}
fn tef(eta:&[f64],fraction:f64)->Vec<u32>{
    let t:f64=eta.iter().sum();let target=fraction*t;
    let mut idx:Vec<usize>=(0..eta.len()).collect();
    idx.sort_by(|&a,&b|eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum=0.;let mut m=Vec::new();
    for &i in&idx{m.push(i as u32);cum+=eta[i];if cum>=target{break;}}m
}
fn main(){
    let mut mf="data/beam-quad.mesh".to_string();let mut o=1u8;let max_dofs=50000;let max_amr=20;
    let mut a=std::env::args().skip(1);
    while let Some(arg)=a.next(){match arg.as_str(){
        "-h"=>{eprintln!("Usage: ex21 [-m mesh] [-o order]");return;}
        "-m"|"--mesh"=>{mf=a.next().unwrap_or_default();}
        "-o"|"--order"=>{o=a.next().and_then(|v|v.parse().ok()).unwrap_or(1);}_=>{}}
    }
    println!("Options:\n  --mesh {mf}\n  --order {o}");
    let data=fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh:Mesh<2>=data.mesh2d.expect("2D mesh");
    let cfg=SolverConfig{rtol:1e-12,atol:0.,max_iter:2000,..SolverConfig::default()};
    let mut ch:Vec<HangingNodeConstraint>=Vec::new();
    for it in 0..=max_amr{
        let sp=VectorH1Space::new(mesh.clone(),o,2);let ns=sp.n_scalar_dofs();
        let dm=sp.scalar_dof_manager();let bd=boundary_dofs(sp.mesh(),dm,&[1]);
        let ess:Vec<usize>=bd.iter().flat_map(|&d|vec![d as usize,d as usize+ns]).collect();
        let n=sp.n_dofs();println!("\nAMR iteration {it}\nUnknowns: {n}");
        let stiff=Assembler::assemble_bilinear(&sp,&[&ElasticityIntegrator::new(
            fem_assembly::postproc::coefficient::PWConstCoeff::new([(1,50.),(2,1.)]),
            fem_assembly::postproc::coefficient::PWConstCoeff::new([(1,50.),(2,1.)]),
        )],3);
        let mut A=stiff.clone();let mut rhs=bdr_load(&mesh,&sp,2,0.,-0.01);
        if!ch.is_empty(){
            let mut vc:Vec<HangingNodeConstraint>=Vec::with_capacity(ch.len()*2);
            for c in&ch{vc.push(HangingNodeConstraint{constrained:c.constrained,parent_a:c.parent_a,parent_b:c.parent_b});
                vc.push(HangingNodeConstraint{constrained:c.constrained+ns,parent_a:c.parent_a+ns,parent_b:c.parent_b+ns});}
            apply_hanging_constraints(&mut A,&mut rhs,&vc);
            for&d in&ess{A.apply_dirichlet_row_zeroing(d,0.,&mut rhs);}
            let mut X=vec![0.;n];let _=fem_solver::solve_pcg_gssmoother(&A,&rhs,&mut X,&cfg);
            recover_hanging_values(&mut X,&vc);
            let gf=GridFunction::new(&sp,X);let ind=zz_estimator(&gf);
            let me=ind.eta.iter().cloned().fold(0.,f64::max);
            println!("  Max err: {me:.6e}");let marked=tef(&ind.eta,0.7);
            println!("  Marked {}",marked.len());
            if n>max_dofs||marked.is_empty(){break;}
            let(nm,nc)=fem_mesh::amr::refine_nonconforming_quad(&mesh,&marked,None);
            mesh=nm;ch=nc;
        }else{
            for&d in&ess{A.apply_dirichlet_row_zeroing(d,0.,&mut rhs);}
            let mut X=vec![0.;n];let _=fem_solver::solve_pcg_gssmoother(&A,&rhs,&mut X,&cfg);
            let gf=GridFunction::new(&sp,X);let ind=zz_estimator(&gf);
            let me=ind.eta.iter().cloned().fold(0.,f64::max);
            println!("  Max err: {me:.6e}");let marked=tef(&ind.eta,0.7);
            println!("  Marked {}",marked.len());
            if n>max_dofs||marked.is_empty(){break;}
            let(nm,nc)=fem_mesh::amr::refine_nonconforming_quad(&mesh,&marked,None);
            mesh=nm;ch=nc;
        }
    }
}
