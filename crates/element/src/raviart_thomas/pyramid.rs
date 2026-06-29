use std::sync::OnceLock;
use crate::quadrature::{tri_rule, seg_rule, pyramid_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// 5 monomials: (1,0,0),(0,1,0),(0,0,1),(x,0,0),(0,y,0)
fn eval_monos(x: f64, y: f64, _z: f64, out: &mut [f64; 15]) {
    out[0]=1.;out[1]=0.;out[2]=0.; out[3]=0.;out[4]=1.;out[5]=0.;
    out[6]=0.;out[7]=0.;out[8]=1.; out[9]=x;out[10]=0.;out[11]=0.;
    out[12]=0.;out[13]=y;out[14]=0.;
}

fn build_vm() -> [[f64;5];5] {
    let mut v=[[0.;5];5];
    let tri=tri_rule(4); let seg=seg_rule(4); let mut m=[0.;15];
    // f0: base z=0, x∈[0,1], y∈[0,1], n̂=(0,0,-1), area=1
    for s in 0..seg.n_points(){let xi=seg.points[s][0];let ws=seg.weights[s];
        for t in 0..seg.n_points(){let eta=seg.points[t][0];let wt=seg.weights[t];
            eval_monos(xi,eta,0.,&mut m);
            for j in 0..5{v[0][j]+=ws*wt*(-m[j*3+2]);}}}
    // f1: x=0, tri (y,z): y∈[0,1-z], z∈[0,1], n̂=(-1,0,0), area=0.5
    for q in 0..tri.n_points(){let(eta,zeta)=(tri.points[q][0],tri.points[q][1]);let w=tri.weights[q];
        eval_monos(0.,eta,zeta,&mut m);
        for j in 0..5{v[1][j]+=w*(-m[j*3]);}}
    // f2: y=0, tri (x,z): x∈[0,1-z], z∈[0,1], n̂=(0,-1,0), area=0.5
    for q in 0..tri.n_points(){let(xi,zeta)=(tri.points[q][0],tri.points[q][1]);let w=tri.weights[q];
        eval_monos(xi,0.,zeta,&mut m);
        for j in 0..5{v[2][j]+=w*(-m[j*3+1]);}}
    // f3: x+z=1, n̂=(1,0,1)/√2, param y∈[0,1], t=z∈[0,1], x=1-t, ds=√2
    let s2=std::f64::consts::FRAC_1_SQRT_2;
    for s in 0..seg.n_points(){let y=seg.points[s][0];let ws=seg.weights[s];
        for t in 0..seg.n_points(){let z=seg.points[t][0];let wt=seg.weights[t];let x=1.-z;
            eval_monos(x,y,z,&mut m);
            for j in 0..5{let ndot=s2*(m[j*3]+m[j*3+2]);v[3][j]+=ws*wt*2f64.sqrt()*ndot;}}}
    // f4: y+z=1, n̂=(0,1,1)/√2, param x∈[0,1], t=z∈[0,1], y=1-t, ds=√2
    for s in 0..seg.n_points(){let x=seg.points[s][0];let ws=seg.weights[s];
        for t in 0..seg.n_points(){let z=seg.points[t][0];let wt=seg.weights[t];let y=1.-z;
            eval_monos(x,y,z,&mut m);
            for j in 0..5{let ndot=s2*(m[j*3+1]+m[j*3+2]);v[4][j]+=ws*wt*2f64.sqrt()*ndot;}}}
    v
}

fn inv5(mut a:[[f64;5];5])->[[f64;5];5]{let n=5;let mut inv=[[0.;5];5];for i in 0..n{inv[i][i]=1.;}for c in 0..n{let(mut mr,mut mv)=(c,a[c][c].abs());for r in(c+1)..n{let v=a[r][c].abs();if v>mv{mv=v;mr=r}}a.swap(c,mr);inv.swap(c,mr);let p=a[c][c];let ip=1./p;for j in 0..n{a[c][j]*=ip;inv[c][j]*=ip;}for r in 0..n{if r==c{continue}let f=a[r][c];for j in 0..n{a[r][j]-=f*a[c][j];inv[r][j]-=f*inv[c][j];}}}inv}

fn coeff()->&'static[[f64;5];5]{static C:OnceLock<[[f64;5];5]>=OnceLock::new();C.get_or_init(||{let vi=inv5(build_vm());let mut c=[[0.;5];5];for i in 0..5{for j in 0..5{c[i][j]=vi[j][i];}}c})}

pub struct PyraRT0;

impl VectorReferenceElement for PyraRT0 {
    fn dim(&self)->u8{3} fn order(&self)->u8{0} fn n_dofs(&self)->usize{5}
    fn eval_basis_vec(&self,xi:&[f64],vals:&mut[f64]){let c=coeff();let mut m=[0.;15];eval_monos(xi[0],xi[1],xi[2],&mut m);for i in 0..5{let(mut vx,mut vy,mut vz)=(0.,0.,0.);for j in 0..5{vx+=c[i][j]*m[j*3];vy+=c[i][j]*m[j*3+1];vz+=c[i][j]*m[j*3+2];}vals[i*3]=vx;vals[i*3+1]=vy;vals[i*3+2]=vz;}}
    fn eval_curl(&self,_xi:&[f64],curl:&mut[f64]){for v in curl.iter_mut(){*v=0.;}}
    fn eval_div(&self,_xi:&[f64],div:&mut[f64]){for i in 0..5{div[i]=3.;}}
    fn quadrature(&self,order:u8)->QuadratureRule{pyramid_rule(order)}
    fn dof_coords(&self)->Vec<Vec<f64>>{vec![vec![0.5,0.5,0.], vec![0.,1./3.,1./3.], vec![1./3.,0.,1./3.], vec![0.5,0.5,0.5], vec![0.5,0.5,0.5]]}
}

#[cfg(test)]
mod tests{use super::*;
    #[test]fn rt0_finite(){let e=PyraRT0;let mut v=vec![0.;15];for p in &e.quadrature(3).points{e.eval_basis_vec(p,&mut v);for x in &v{assert!(x.is_finite());}}}
    #[test]fn rt0_nodal(){let e=PyraRT0;let s2=std::f64::consts::FRAC_1_SQRT_2;
        let faces:[([f64;3],f64);5]=[([0.,0.,-1.],1.),([-1.,0.,0.],0.5),([0.,-1.,0.],0.5),([s2,0.,s2],2f64.sqrt()/2.),([0.,s2,s2],2f64.sqrt()/2.)];
        let tri=tri_rule(4);let seg=seg_rule(4);let mut vals=vec![0.;15];
        for(j,(n,_area))in faces.iter().enumerate(){let mut dofs=[0.;5];
            match j{0=>{for s in 0..seg.n_points(){let x=seg.points[s][0];let ws=seg.weights[s];for t in 0..seg.n_points(){let y=seg.points[t][0];let wt=seg.weights[t];e.eval_basis_vec(&[x,y,0.],&mut vals);for i in 0..5{dofs[i]+=ws*wt*(vals[i*3]*n[0]+vals[i*3+1]*n[1]+vals[i*3+2]*n[2]);}}}}
                1=>{for q in 0..tri.n_points(){let(y,z)=(tri.points[q][0],tri.points[q][1]);let w=tri.weights[q];e.eval_basis_vec(&[0.,y,z],&mut vals);for i in 0..5{dofs[i]+=w*(vals[i*3]*n[0]+vals[i*3+1]*n[1]+vals[i*3+2]*n[2]);}}}
                2=>{for q in 0..tri.n_points(){let(x,z)=(tri.points[q][0],tri.points[q][1]);let w=tri.weights[q];e.eval_basis_vec(&[x,0.,z],&mut vals);for i in 0..5{dofs[i]+=w*(vals[i*3]*n[0]+vals[i*3+1]*n[1]+vals[i*3+2]*n[2]);}}}
                3=>{for s in 0..seg.n_points(){let y=seg.points[s][0];let ws=seg.weights[s];for t in 0..seg.n_points(){let z=seg.points[t][0];let wt=seg.weights[t];e.eval_basis_vec(&[1.-z,y,z],&mut vals);for i in 0..5{dofs[i]+=ws*wt*2f64.sqrt()*(vals[i*3]*n[0]+vals[i*3+1]*n[1]+vals[i*3+2]*n[2]);}}}}
                4=>{for s in 0..seg.n_points(){let x=seg.points[s][0];let ws=seg.weights[s];for t in 0..seg.n_points(){let z=seg.points[t][0];let wt=seg.weights[t];e.eval_basis_vec(&[x,1.-z,z],&mut vals);for i in 0..5{dofs[i]+=ws*wt*2f64.sqrt()*(vals[i*3]*n[0]+vals[i*3+1]*n[1]+vals[i*3+2]*n[2]);}}}}
                _=>unreachable!()}
            for i in 0..5{let exp=if i==j{1.}else{0.};assert!((dofs[i]-exp).abs()<1e-12,"DOF_{j}(Φ_{i})={},exp{exp}",dofs[i]);}}
    }
    #[test]fn rt0_div(){let e=PyraRT0;let qr=e.quadrature(3);let mut dv=vec![0.;5];for i in 0..5{let mut s=0.;for(p,&w)in qr.points.iter().zip(qr.weights.iter()){e.eval_div(p,&mut dv);s+=dv[i]*w;}assert!((s-1.).abs()<1e-12,"∫divΦ_{i}={},exp1",s);}}
}
