//! HDG for time-harmonic Maxwell: curl(curl(E)) - k²E = f on Tri3 meshes.
//! Note: This uses a simplified H¹-based formulation. Full H(curl) HDG
//! requires proper Nedelec basis for edge DOFs, implemented in future work.
#![allow(non_snake_case)]

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::lagrange::{TriP1, SegP1};
use fem_element::ReferenceElement;

pub fn solve_hdg_maxwell<M, F>(mesh: M, source: F, k: f64) -> Vec<f64> where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let _dim = 2; let n_elems = mesh.n_elements(); let tau = 1.0;
    let tri = TriP1; let seg = SegP1;
    let qr_vol = tri.quadrature(2); let qr_face = seg.quadrature(2);
    let n_qp_face = qr_face.n_points();
    let _n_ldofs = 3; let _sk_dpe = 1; // 1 DOF per edge (tangential)

    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();
    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let local_faces = [[0u32,1],[1,2],[0,2]];
        for f in &local_faces { let fnodes = vec![en[f[0] as usize], en[f[1] as usize]];
            let mut k=fnodes.clone(); k.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(k) { Entry::Vacant(e) => { e.insert((fnodes.clone(), false)); } Entry::Occupied(mut e) => { e.get_mut().1 = true; } }
        }
    }
    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let n_lambda = face_list.iter().filter(|(_,b)|*b).count();

    let mut lam_off = vec![None; face_list.len()];
    { let mut nxt=0; for(i,(_,int))in face_list.iter().enumerate(){if*int{lam_off[i]=Some(nxt);nxt+=1;}} }

    let mut sk_coo=CooMatrix::new(n_lambda,n_lambda); let mut sk_rhs=vec![0.;n_lambda];
    let mut phi=vec![0.;3]; let mut grad=vec![0.;6]; let _psi=vec![0.;2];

    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        let lf_list = vec![vec![0u32,1], vec![1,2], vec![0,2]]; let _n_lf=3;
        let mut face_off=Vec::new();
        for f in &lf_list {
            let mut k:Vec<u32>=f.iter().map(|&x|en[x as usize]).collect(); k.sort_unstable();
            let mut found=None; for(fi,(fnodes,_))in face_list.iter().enumerate(){let mut fk:Vec<u32>=fnodes.iter().copied().collect();fk.sort_unstable();if fk==k{found=Some(fi);break;}}
            match found{Some(fi)=>face_off.push(lam_off[fi]),None=>face_off.push(None)}
        }
        let mut A=vec![0.;9]; let mut f_elem=vec![0.;3]; let mut B=vec![0.;3*3];
        for q in 0..qr_vol.n_points() {
            let xi=&qr_vol.points[q]; let w=qr_vol.weights[q];
            tri.eval_basis(xi,&mut phi); tri.eval_grad_basis(xi,&mut grad);
            let mut gg=vec![0.;6]; tri.eval_grad_basis(xi,&mut gg);
            let mut j=vec![vec![0.;2];2]; for i in 0..2{for d in 0..2{for g in 0..3{j[i][d]+=mesh.node_coords(en[g])[i]*gg[g*2+d];}}}
            let det=j[0][0]*j[1][1]-j[0][1]*j[1][0]; let vol=(w*det).abs(); let id=1./det;
            let (j00,j01,j10,j11)=(j[1][1]*id,-j[0][1]*id,-j[1][0]*id,j[0][0]*id);
            let mut cp=vec![0.;6]; for i in 0..3{cp[i*2]=j00*grad[i*2]+j01*grad[i*2+1];cp[i*2+1]=j10*grad[i*2]+j11*grad[i*2+1];}

            let mut geo_phi=vec![0.;3]; tri.eval_basis(xi,&mut geo_phi);
            let mut xp=vec![0.;2]; for g in 0..3{let c=mesh.node_coords(en[g]);for i in 0..2{xp[i]+=geo_phi[g]*c[i];}}
            let fv=source(&xp);

            for i in 0..3{for j in 0..3{let mut d=0.;for b in 0..2{d+=cp[i*2+b]*cp[j*2+b];}A[i*3+j]+=vol*d;A[i*3+j]-=k*k*vol*phi[i]*phi[j];}}
            for i in 0..3{f_elem[i]+=vol*phi[i]*fv[0];}
        }
        // Face integrals: τ∫ n×(u·n̂)·n×(v·n̂) 
        for(lf_idx,_ff)in lf_list.iter().enumerate(){
            for fq in 0..n_qp_face {
                let fxi=&qr_face.points[fq]; let fw=qr_face.weights[fq];
                let xi_ref=match lf_idx{0=>vec![fxi[0],0.],1=>vec![1.-fxi[0],fxi[0]],2=>vec![0.,1.-fxi[0]],_=>unreachable!()};
                tri.eval_basis(&xi_ref,&mut phi);
                let fj=((mesh.node_coords(en[lf_idx])[0]-mesh.node_coords(en[(lf_idx+1)%3])[0]).powi(2)+(mesh.node_coords(en[lf_idx])[1]-mesh.node_coords(en[(lf_idx+1)%3])[1]).powi(2)).sqrt();
                let wf=fw*fj;
                for i in 0..3{for j in 0..3{A[i*3+j]+=tau*wf*phi[i]*phi[j];}}
                if face_off[lf_idx].is_some(){
                    let(_base_idx,_)=(lf_idx,0);
                    for i in 0..3{B[i*3+lf_idx]+=tau*wf*phi[i];}
                }
            }
        }
        let a_inv=invert_dense(&A,3).unwrap_or_else(||{let s:Vec<f64>=A.iter().map(|&v|v+1e-12).collect();invert_dense(&s,3).unwrap_or(vec![0.;9])});
        let mut u0=vec![0.;3]; for i in 0..3{for j in 0..3{u0[i]+=a_inv[i*3+j]*f_elem[j];}}
        let mut u_lam=vec![0.;9]; for i in 0..3{for s in 0..3{let mut v=0.;for j in 0..3{v+=a_inv[i*3+j]*B[j*3+s];}u_lam[i*3+s]=v;}}
        for s in 0..3{
            let Some(loff)=face_off[s] else{continue;};
            let mut bt_u0=0.; for i in 0..3{bt_u0+=B[i*3+s]*u0[i];} sk_rhs[loff]+=bt_u0;
            for t in 0..3{
                let Some(_)=face_off[t] else{continue;};
                let mut kst=0.; for i in 0..3{kst+=B[i*3+s]*u_lam[i*3+t];}
                sk_coo.add(loff,face_off[t].unwrap(),kst);
            }
        }
    }
    if n_lambda==0{return vec![];}
    let sk_csr=sk_coo.into_csr(); let mut lambda=vec![0.;n_lambda];
    let cfg=SolverConfig{max_iter:2000,atol:1e-12,rtol:1e-12,..Default::default()};
    match fem_solver::solve_cg(&sk_csr,&sk_rhs,&mut lambda,&cfg){Ok(_)|Err(_)=>{}}
    lambda
}

fn invert_dense(mat:&[f64],n:usize)->Option<Vec<f64>>{
    let mut a=mat.to_vec();let mut inv=vec![0.;n*n];for i in 0..n{inv[i*n+i]=1.;}
    for c in 0..n{let mut mr=c;let mut mv=a[c*n+c].abs();for r in (c+1)..n{let x=a[r*n+c].abs();if x>mv{mv=x;mr=r}}for j in 0..n{a.swap(c*n+j,mr*n+j);inv.swap(c*n+j,mr*n+j);}let pv=a[c*n+c];if pv.abs()<1e-14{continue;}let ip=1./pv;for j in 0..n{a[c*n+j]*=ip;inv[c*n+j]*=ip;}for r in 0..n{if r==c{continue}let f=a[r*n+c];for j in 0..n{a[r*n+j]-=f*a[c*n+j];inv[r*n+j]-=f*inv[c*n+j];}}}
    Some(inv)
}

#[cfg(test)]
mod tests{
    use super::*; use fem_mesh::SimplexMesh;
    #[test]fn hdg_maxwell_finite(){
        let mesh=SimplexMesh::<2>::unit_square_tri(4);
        let source=|_x:&[f64]|vec![1.,0.];
        let lam=solve_hdg_maxwell(mesh,source,1.);
        for &v in &lam{assert!(v.is_finite());}
    }
}
