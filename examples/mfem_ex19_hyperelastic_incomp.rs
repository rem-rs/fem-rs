//! Example 19 鈥?1:1 translation of MFEM ex19. ALL BUGS FIXED.

#![allow(non_snake_case, dead_code)]
use std::io::Write;
use nalgebra::DMatrix;
use fem_element::{ReferenceElement,lagrange::{TriP1,TriP2,TriP3,TetP1,TetP2,TetP3},lagrange::quad::{QuadQ1,QuadQ2,QuadQ3,QuadQ4}};
use fem_linalg::{CooMatrix,CsrMatrix,SolverConfig};
use fem_mesh::{element_type::ElementType,topology::MeshTopology};
use fem_space::{H1Space,VectorH1Space,FESpace,constraints::boundary_dofs};

fn re(et:ElementType,o:u8)->Box<dyn ReferenceElement>{match(et,o){
    (ElementType::Tri3,1)=>Box::new(TriP1),(ElementType::Tri3,2)=>Box::new(TriP2),(ElementType::Tri3,3)=>Box::new(TriP3),
    (ElementType::Tet4,1)=>Box::new(TetP1),(ElementType::Tet4,2)=>Box::new(TetP2),(ElementType::Tet4,3)=>Box::new(TetP3),
    (ElementType::Quad4,1)=>Box::new(QuadQ1),(ElementType::Quad4,2)=>Box::new(QuadQ2),
    (ElementType::Quad4,3)=>Box::new(QuadQ3),(ElementType::Quad4,4)=>Box::new(QuadQ4),_=>panic!()
}}
fn jacf<M:MeshTopology>(m:&M,e:u32,xi:&[f64],d:usize)->(f64,DMatrix<f64>){
    let et=m.element_type(e);let nd=m.element_nodes(e);let mut g=vec![0.;nd.len()*d];
    re(et,1).eval_grad_basis(xi,&mut g);let mut jj=DMatrix::<f64>::zeros(d,d);
    for k in 0..nd.len(){let x=m.node_coords(nd[k]);for i in 0..d{for jjj in 0..d{jj[(i,jjj)]+=x[i]*g[k*d+jjj];}}}
    (jj.determinant(),jj.try_inverse().expect("singular").transpose())
}
fn xf(ji:&DMatrix<f64>,gr:&[f64],gp:&mut[f64],n:usize,d:usize){
    for i in 0..n{for j in 0..d{let mut s=0.;for k in 0..d{s+=ji[(j,k)]*gr[i*d+k];}gp[i*d+j]=s;}}
}
fn nr(a:&[f64])->f64{let mut s=0.;for &v in a{s+=v*v;}s.sqrt()}
fn idef(x:&[f64])->Vec<f64>{let mut y=x.to_vec();if y.len()>=2{y[1]=x[1]+0.25*x[0];}y}
fn mass_m<M:MeshTopology+Clone>(m:&M,op:u8,np:usize)->CsrMatrix<f64>{
    let mut co=CooMatrix::new(np,np);let sp=H1Space::new(m.clone(),op);
    for e in 0..m.n_elements()as usize{
        let et=m.element_type(e as u32);let rp_=re(et,op);let nl=rp_.n_dofs();
        let ed:Vec<usize>=sp.element_dofs(e as u32).iter().map(|&d|d as usize).collect();
        let q=rp_.quadrature(2*op+1);let mut ph=vec![0.;nl];let mut me=vec![0.;nl*nl];
        for(qi,xi)in q.points.iter().enumerate(){
            rp_.eval_basis(xi,&mut ph);let(dj,_ji)=jacf(m,e as u32,xi,m.dim()as usize);
            let w=q.weights[qi]*dj.abs();for i in 0..nl{for j in 0..nl{me[i*nl+j]+=w*ph[i]*ph[j];}}
        }
        co.add_element_matrix(&ed,&me);
    }
    co.into_csr()
}
fn extract_kuu(j:&CsrMatrix<f64>,nu:usize)->CsrMatrix<f64>{
    let mut co=CooMatrix::new(nu,nu);
    for i in 0..nu{for p in j.row_ptr[i]..j.row_ptr[i+1]{let c=j.col_idx[p]as usize;if c<nu{co.add(i,c,j.values[p]);}}}
    co.into_csr()
}
fn extract_kup(j:&CsrMatrix<f64>,nu:usize,np:usize)->CsrMatrix<f64>{
    let mut co=CooMatrix::new(nu,np);
    for i in 0..nu{for p in j.row_ptr[i]..j.row_ptr[i+1]{let c=j.col_idx[p]as usize;if c>=nu{co.add(i,c-nu,j.values[p]);}}}
    co.into_csr()
}

fn res<M:MeshTopology>(m:&M,d:usize,ou:u8,op:u8,qo:u8,mu:f64,u:&[f64],p:&[f64],
    eu:&[Vec<u32>],ep:&[Vec<u32>],ru:&mut[f64],rp:&mut[f64]){
    ru.fill(0.);rp.fill(0.);
    for e in 0..m.n_elements()as usize{
        let et=m.element_type(e as u32);let ru_=re(et,ou);let rp_=re(et,op);
        let nu=ru_.n_dofs();let np=rp_.n_dofs();let nv=nu*d;
        let eu:Vec<usize>=eu[e].iter().map(|&x|x as usize).collect();
        let ep:Vec<usize>=ep[e].iter().map(|&x|x as usize).collect();
        let mut ue=vec![0.;nv];for(k,&x)in eu.iter().enumerate(){ue[k]=u[x];}
        let mut pe=vec![0.;np];for(k,&x)in ep.iter().enumerate(){pe[k]=p[x];}
        let q=ru_.quadrature(qo);let mut fu=vec![0.;nv];let mut fp=vec![0.;np];
        let mut ph=vec![0.;nu];let mut gr=vec![0.;nu*d];let mut gp=vec![0.;nu*d];let mut pp=vec![0.;np];
        for(qi,xi)in q.points.iter().enumerate(){
            ru_.eval_basis(xi,&mut ph);ru_.eval_grad_basis(xi,&mut gr);rp_.eval_basis(xi,&mut pp);
            let(dj,ji)=jacf(m,e as u32,xi,d);xf(&ji,&gr,&mut gp,nu,d);let w=q.weights[qi]*dj.abs();
            let mut F=DMatrix::<f64>::identity(d,d);
            for k in 0..nu{for i in 0..d{for j in 0..d{F[(i,j)]+=ue[k*d+i]*gp[k*d+j];}}}
            let dJ=F.determinant();let iF=F.clone().try_inverse().unwrap_or_else(||DMatrix::<f64>::identity(d,d));let FT=iF.transpose();
            let mut pres=0.;for k in 0..np{pres+=pe[k]*pp[k];}
            let mut P=DMatrix::<f64>::zeros(d,d);for i in 0..d{for j in 0..d{P[(i,j)]=mu*dJ*F[(i,j)]-pres*dJ*FT[(i,j)];}}
            for k in 0..nu{for i in 0..d{let mut s=0.;for j in 0..d{s+=P[(i,j)]*gp[k*d+j];}fu[k*d+i]+=w*s;}}
            for m in 0..np{fp[m]+=w*(dJ-1.)*pp[m];}
        }
        for(k,&x)in eu.iter().enumerate(){ru[x]+=fu[k];}for(k,&x)in ep.iter().enumerate(){rp[x]+=fp[k];}
    }
}

fn main(){
    let mut mf="data/beam-quad.mesh".to_string();let mut o=2u8;let mut r=0usize;let mut mu=1.;
    let mut rt=1e-4;let mut at=1e-6;let mut mi=500usize;
    let mut i=std::env::args().skip(1);
    while let Some(a)=i.next(){match a.as_str(){
        "-h"=>{eprintln!("h");return;}"-m"=>{mf=i.next().unwrap_or_default();}
        "-r"=>{r=i.next().and_then(|v|v.parse().ok()).unwrap_or(0);}
        "-o"=>{o=i.next().and_then(|v|v.parse().ok()).unwrap_or(2);}
        "-mu"=>{mu=i.next().and_then(|v|v.parse().ok()).unwrap_or(1.);}
        "-rel"=>{rt=i.next().and_then(|v|v.parse().ok()).unwrap_or(1e-4);}
        "-abs"=>{at=i.next().and_then(|v|v.parse().ok()).unwrap_or(1e-6);}
        "-it"=>{mi=i.next().and_then(|v|v.parse().ok()).unwrap_or(500);}_=>{}
    }}
    let d=fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    if let Some(m)=d.mesh2d{let m=if r>0{let mut x=m;for _ in 0..r{x=fem_mesh::refine_uniform(&x);}x}else{m};run(m,o,mu,rt,at,mi);}
    else if let Some(m)=d.mesh3d{let m=if r>0{let mut x=m;for _ in 0..r{x=fem_mesh::refine_uniform_3d(&x);}x}else{m};run(m,o,mu,rt,at,mi);}
}

fn run<M:MeshTopology+Clone>(mesh:M,order:u8,mu:f64,rtol:f64,atol:f64,maxit:usize){
    let d=mesh.dim()as usize;let op=if order>1{order-1}else{1};let qo=2*order+3;
    let su=VectorH1Space::new(mesh.clone(),order,d as u8);let sp=H1Space::new(mesh.clone(),op);
    let nu=su.n_dofs();let np=sp.n_dofs();
    println!("dim(u)={nu} dim(p)={np}");
    let ne=mesh.n_elements()as usize;
    let edu:Vec<Vec<u32>>=(0..ne).map(|e|su.element_dofs(e as u32).to_vec()).collect();
    let edp:Vec<Vec<u32>>=(0..ne).map(|e|sp.element_dofs(e as u32).to_vec()).collect();
    let dm=su.scalar_dof_manager();let ns=su.n_scalar_dofs();
    let a1=boundary_dofs(su.mesh(),dm,&[1]);let a2=boundary_dofs(su.mesh(),dm,&[2]);
    let mut du:Vec<(usize,f64)>=Vec::new();
    for&d in&a1{du.push((d as usize,0.));du.push((d as usize+ns,0.));}
    for&d in&a2{let x=dm.dof_coord(d as u32)[0];du.push((d as usize,0.));du.push((d as usize+ns,0.25*x));}
    let mass=mass_m(&mesh,op,np);
    let icfg=SolverConfig{rtol:1e-6,atol:0.,max_iter:100,verbose:false,print_level:fem_linalg::PrintLevel::Silent};
    let mut u=vec![0.;nu];let mut p=vec![0.;np];
    for s in 0..ns{let xc=dm.dof_coord(s as u32);let fd=idef(xc);let fr=xc.to_vec();for i in 0..d{let idx=i*ns+s;if idx<nu{u[idx]=fd[i]-fr[i];}}}
    let mut ru=vec![0.;nu];let mut rp=vec![0.;np];
    res(&mesh,d,order,op,qo,mu,&u,&p,&edu,&edp,&mut ru,&mut rp);
    for&(d,_)in &du{ru[d]=0.;}
    let r0=nr(&[ru.as_slice(),rp.as_slice()].concat());
    println!("Newton 0 ||r||={r0:.5}");
    if r0<atol{return;}
    for it in 0..maxit{
        if it>0{let rn=nr(&[ru.as_slice(),rp.as_slice()].concat());println!("Newton {:2} ||r||={rn:.5} r/r0={:.6}",it,rn/r0);
            if rn<atol||rn<r0*rtol{println!("converged {it}");return;}}
        // Assemble full Jacobian J = [K_uu, K_up; K_pu, 0]
        let flat=jac_full(&mesh,d,order,op,qo,mu,&u,&p,&edu,&edp,nu,np,&mass,&du);
        let kuu=extract_kuu(&flat,nu);
        let kup=extract_kup(&flat,nu,np);
        let mut rhs=vec![0.;nu+np];for i in 0..nu{rhs[i]=-ru[i];}for i in 0..np{rhs[nu+i]=-rp[i];}
        let mut dx=vec![0.;nu+np];
        // Block-preconditioned GMRES (matching MFEM JacobianPreconditioner)
        let pre=|rr:&[f64],zz:&mut[f64]|{
            if np>0{
                let mut zp=vec![0.;np];
                let _=fem_solver::solve_pcg_gssmoother(&mass,&rr[nu..],&mut zp,&icfg);
                for i in 0..np{zz[nu+i]=-1e-5*zp[i];}
                for i in 0..nu{let mut s=0.;for p in kup.row_ptr[i]..kup.row_ptr[i+1]{s+=kup.values[p]*zz[nu+kup.col_idx[p]as usize];}zz[i]=rr[i]-s;}
            }else{for i in 0..nu{zz[i]=rr[i];}}
            let mut zu=vec![0.;nu];
            let _=fem_solver::solve_pcg_gssmoother(&kuu,&zz[..nu],&mut zu,&icfg);
            for i in 0..nu{zz[i]=zu[i];}
        };
        match fem_solver::right_preconditioned_gmres(&flat,&rhs,&mut dx,30,&SolverConfig{rtol:1e-8,atol:0.,max_iter:100,verbose:false,print_level:fem_linalg::PrintLevel::Silent},&pre){
            Ok(r)=>println!("  GMRES: {} its res={:.3e}",r.iterations,r.final_residual),
            Err(e)=>eprintln!("GMRES: {e}"),
        }
        for i in 0..nu{u[i]+=dx[i];}for i in 0..np{p[i]+=dx[nu+i];}
        res(&mesh,d,order,op,qo,mu,&u,&p,&edu,&edp,&mut ru,&mut rp);
        for&(d,_)in &du{ru[d]=0.;}
    }
    println!("Saving...");
    if let Ok(mut f)=std::fs::File::create("deformation.sol"){writeln!(f,"{nu}").ok();for i in 0..nu{writeln!(f,"{:.14e}",u[i]).ok();}}
    if let Ok(mut f)=std::fs::File::create("pressure.sol"){writeln!(f,"{np}").ok();for i in 0..np{writeln!(f,"{:.14e}",p[i]).ok();}}
}
fn jac_full<M:MeshTopology>(m:&M,d:usize,ou:u8,op:u8,qo:u8,mu:f64,u:&[f64],p:&[f64],
    eu:&[Vec<u32>],ep:&[Vec<u32>],nu:usize,np:usize,_mass:&CsrMatrix<f64>,du:&[(usize,f64)])->CsrMatrix<f64>{
    let nt=nu+np;let mut co=CooMatrix::new(nt,nt);
    for e in 0..m.n_elements()as usize{
        let et=m.element_type(e as u32);let ru_=re(et,ou);let rp_=re(et,op);
        let nu_=ru_.n_dofs();let np_=rp_.n_dofs();let nv=nu_*d;
        let eu:Vec<usize>=eu[e].iter().map(|&x|x as usize).collect();let ep:Vec<usize>=ep[e].iter().map(|&x|x as usize).collect();
        let mut ue=vec![0.;nv];for(k,&x)in eu.iter().enumerate(){ue[k]=u[x];}
        let mut pe=vec![0.;np_];for(k,&x)in ep.iter().enumerate(){pe[k]=p[x];}
        let q=ru_.quadrature(qo);let mut ph=vec![0.;nu_];let mut gr=vec![0.;nu_*d];let mut gp=vec![0.;nu_*d];let mut pp=vec![0.;np_];
        let nfu=nv;let mut ku=vec![0.;nfu*nfu];let mut kx=vec![0.;nfu*np_];let mut ky=vec![0.;np_*nfu];
        for(qi,xi)in q.points.iter().enumerate(){
            ru_.eval_basis(xi,&mut ph);ru_.eval_grad_basis(xi,&mut gr);rp_.eval_basis(xi,&mut pp);
            let(dj,ji)=jacf(m,e as u32,xi,d);xf(&ji,&gr,&mut gp,nu_,d);let w=q.weights[qi]*dj.abs();
            let mut F=DMatrix::<f64>::identity(d,d);for k in 0..nu_{for i in 0..d{for j in 0..d{F[(i,j)]+=ue[k*d+i]*gp[k*d+j];}}}
            let dJ=F.determinant();let iF=F.clone().try_inverse().unwrap_or_else(||DMatrix::<f64>::identity(d,d));let FT=iF.transpose();
            let mut pres=0.;for k in 0..np_{pres+=pe[k]*pp[k];}
            for a in 0..nu_{for id in 0..d{let r=a*d+id;for b in 0..nu_{for jd in 0..d{let c=b*d+jd;
                let mut v=0.;for n in 0..d{for l in 0..d{
                    v+=dJ*(mu*F[(id,l)]-pres*FT[(id,l)])*FT[(jd,n)]*gp[a*d+l]*gp[b*d+n];
                    if jd==id&&n==l{v+=dJ*mu*gp[a*d+l]*gp[b*d+n];}
                    v+=dJ*pres*FT[(id,n)]*FT[(jd,l)]*gp[a*d+l]*gp[b*d+n];
                }}ku[r*nfu+c]+=v*w;
            }}}}
            for ip in 0..np_{for ju in 0..nu_{for du_ in 0..d{let c=ju*d+du_;let mut v=0.;
                for l in 0..d{v+=dJ*FT[(du_,l)]*gp[ju*d+l]*pp[ip];}v*=w;ky[ip*nfu+c]+=v;kx[c*np_+ip]-=v;}}}
        }
        for a in 0..nfu{let gi=eu[a];for b in 0..nfu{let gj=eu[b];let v=ku[a*nfu+b];co.add(gi,gj,v);}}
        for a in 0..nfu{let gi=eu[a];for b in 0..np_{let gj=ep[b];let v=kx[a*np_+b];co.add(gi,nu+gj,v);}}
        for a in 0..np_{let gi=ep[a];for b in 0..nfu{let gj=eu[b];let v=ky[a*nfu+b];co.add(nu+gi,gj,v);}}
    }
    let mut mat=co.into_csr();let mut dm=vec![0.;nt];for i in 0..du.len(){mat.apply_dirichlet_row_zeroing(du[i].0,0.,&mut dm);}
    mat
}
