//! Ex22 鈥?Complex Helmholtz. Block-diag GS-preconditioned GMRES on flat 2x2.

#![allow(non_snake_case, dead_code)]
use std::io::Write;
use fem_assembly::complex::{ComplexAssembler, ComplexSystem};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_linalg::{CooMatrix, SolverConfig};
use fem_mesh::{refine_uniform, Mesh};
use fem_space::{FESpace, H1Space, constraints::boundary_dofs};
use fem_solver::{linlvoPreconditioner, DenseVec, right_preconditioned_gmres};

static MU: f64 = 1.0; static EPSILON: f64 = 1.0;
static SIGMA: f64 = 20.0; static OMEGA: f64 = 10.0;

fn pw(x:&[f64])->(f64,f64){let ar=OMEGA*EPSILON;let ai=-SIGMA;let k2r=MU*OMEGA*ar;let k2i=MU*OMEGA*ai;
    let r=(k2r*k2r+k2i*k2i).sqrt().sqrt();let t=0.5*k2i.atan2(k2r);let kr=r*t.cos();let ki=r*t.sin();
    let kx=kr*x[x.len()-1];let e=(ki*x[x.len()-1]).exp();(e*kx.cos(),-e*kx.sin())}

fn main() {
    let mut mf="data/inline-quad.mesh".to_string();let mut rl=0usize;let mut o=1u8;
    let mut i=std::env::args().skip(1);
    while let Some(a)=i.next(){match a.as_str(){
        "-h"=>{eprintln!("Usage: ex22 [-m mesh] [-r refine] [-o order]");return;}
        "-m"|"--mesh"=>{mf=i.next().unwrap_or_default();}
        "-r"|"--refine"=>{rl=i.next().and_then(|v|v.parse().ok()).unwrap_or(0);}
        "-o"|"--order"=>{o=i.next().and_then(|v|v.parse().ok()).unwrap_or(1);}
        _=>{}}
    }
    let ex=std::path::Path::new(&mf).file_stem().and_then(|s|s.to_str()).map(|s|s.starts_with("inline-")).unwrap_or(false);
    println!("Options:\n  --mesh {mf}\n  --refine {rl}\n  --order {o}");
    let data=fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh:Mesh<2>=data.mesh2d.expect("2D mesh");
    for _ in 0..rl{mesh=refine_uniform(&mesh);}
    let sp=H1Space::new(mesh.clone(),o);let nd=sp.n_dofs();
    println!("Number of unknowns: {nd}");

    let omega=OMEGA;
    let mut sys:ComplexSystem=ComplexAssembler::assemble(
        &sp,&[&DiffusionIntegrator{kappa:1./MU}],&[&MassIntegrator{rho:EPSILON}],
        &[&MassIntegrator{rho:SIGMA}],omega,3);

    let dm=sp.dof_manager();let bdry=boundary_dofs(&mesh,dm,&[1,2,3,4]);
    let ess:Vec<usize>=bdry.iter().map(|&d|d as usize).collect();
    let mut br=vec![0.;nd];let mut bi=vec![0.;nd];
    if ex{for(_k,&d)in ess.iter().enumerate(){let c=dm.dof_coord(d as u32);let(e_r,e_i)=pw(c);
        sys.k_re.apply_dirichlet_row_zeroing(d,e_r,&mut br);bi[d]=e_i;}}

    // Flat 2x2: [k_re, -k_im; k_im, k_re]
    let n2=2*nd;let mut co=CooMatrix::new(n2,n2);
    for i in 0..nd{for p in sys.k_re.row_ptr[i]..sys.k_re.row_ptr[i+1]{let j=sys.k_re.col_idx[p]as usize;
        co.add(i,j,sys.k_re.values[p]);co.add(nd+i,nd+j,sys.k_re.values[p]);}}
    for i in 0..nd{for p in sys.k_im.row_ptr[i]..sys.k_im.row_ptr[i+1]{let j=sys.k_im.col_idx[p]as usize;
        co.add(i,nd+j,-sys.k_im.values[p]);co.add(nd+i,j,sys.k_im.values[p]);}}
    let mut flat=co.into_csr();

    // Block-diagonal GS preconditioner (matching C++)
    let gs=fem_solver::GSSmoother::from_csr(
        &fem_linalg::fem_to_linlvo_csr(&sys.k_re),1.0).expect("GS");

    let mut rhs=vec![0.;n2];for i in 0..nd{rhs[i]=br[i];rhs[nd+i]=bi[i];}
    // Apply Dirichlet BCs on the flat system
    for &d in &ess {
        flat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs);
        flat.apply_dirichlet_row_zeroing(nd+d, 0.0, &mut rhs);
        rhs[d] = br[d]; rhs[nd+d] = bi[d];
    }
    let mut X=vec![0.;n2];
    let pre=|r:&[f64],z:&mut[f64]|{
        let vr=DenseVec::from(r[..nd].to_vec());let mut zr=DenseVec::zeros(nd);
        gs.apply_precond(&vr,&mut zr);for i in 0..nd{z[i]=zr[i];}
        let vi=DenseVec::from(r[nd..].to_vec());let mut zi=DenseVec::zeros(nd);
        gs.apply_precond(&vi,&mut zi);for i in 0..nd{z[nd+i]=zi[i];}
    };
    match right_preconditioned_gmres(&flat,&rhs,&mut X,50,
        &SolverConfig{rtol:1e-12,atol:0.,max_iter:1000,..SolverConfig::default()},&pre){
        Ok(r)=>println!("  GMRES: {} its res={:.3e}",r.iterations,r.final_residual),
        Err(e)=>eprintln!("  GMRES: {e}"),
    }

    if ex{let mut er2=0.;let mut ei2=0.;
        for s in 0..nd{let c=dm.dof_coord(s as u32);let(er,ei)=pw(c);
            let d=X[s]-er;er2+=d*d;let d=X[nd+s]-ei;ei2+=d*d;}
        println!("\n|| Re(u_h-u) || = {:.6e}",er2.sqrt());
        println!("|| Im(u_h-u) || = {:.6e}\n",ei2.sqrt());}
    if let Ok(mut f)=std::fs::File::create("sol_r.gf"){for i in 0..nd{writeln!(f,"{:.14e}",X[i]).ok();}}
    if let Ok(mut f)=std::fs::File::create("sol_i.gf"){for i in 0..nd{writeln!(f,"{:.14e}",X[nd+i]).ok();}}
}



