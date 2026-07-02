//! Nitsche contact for 3D elasticity (Chouly-Hild gamma-family).
//! Imposes unilateral contact without LM via consistent penalty.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;

pub struct NitscheContactConfig {
    pub contact_tags: Vec<i32>,
    pub gamma: f64,
    pub theta: f64,
    pub lambda: f64,
    pub mu: f64,
    pub gap_function: fn(&[f64]) -> f64,
}

pub fn assemble_nitsche_contact_3d<M: MeshTopology>(
    mesh: &M, cfg: &NitscheContactConfig, u: &[f64],
) -> (Vec<f64>, CsrMatrix<f64>) {
    let n = mesh.n_nodes() as usize; let nd = n*3;
    let mut rhs = vec![0.0; nd]; let mut coo = CooMatrix::new(nd, nd);
    let cs: std::collections::HashSet<i32> = cfg.contact_tags.iter().copied().collect();
    let (tp, tw) = tri3();

    // Map boundary face to owning element
    let mut f2e: Vec<u32> = vec![0; mesh.n_boundary_faces() as usize];
    for f in 0..mesh.n_boundary_faces() as u32 {
        let fn_ = mesh.face_nodes(f);
        for e in 0..mesh.n_elements() as u32 {
            let ns = mesh.element_nodes(e);
            let faces: [(usize,usize,usize);4] = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)];
            let mut found = false;
            for &(a,b,c) in &faces {
                let mut fk = [ns[a],ns[b],ns[c]]; fk.sort_unstable();
                let mut fk2 = [fn_[0],fn_[1],fn_[2]]; fk2.sort_unstable();
                if fk == fk2 { f2e[f as usize] = e; found = true; break; }
            }
            if found { break; }
        }
    }

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tag = mesh.face_tag(f);
        if !cs.contains(&tag) { continue; }
        let fn_ = mesh.face_nodes(f);
        if fn_.len() < 3 { continue; }
        let n3 = [fn_[0]as usize, fn_[1]as usize, fn_[2]as usize];
        let p = [mesh.node_coords(fn_[0]), mesh.node_coords(fn_[1]), mesh.node_coords(fn_[2])];
        let e = f2e[f as usize];
        let ns = mesh.element_nodes(e);

        // Element Jacobian
        let xv: Vec<_> = (0..4).map(|i| mesh.node_coords(ns[i])).collect();
        let j00=xv[1][0]-xv[0][0];let j01=xv[2][0]-xv[0][0];let j02=xv[3][0]-xv[0][0];
        let j10=xv[1][1]-xv[0][1];let j11=xv[2][1]-xv[0][1];let j12=xv[3][1]-xv[0][1];
        let j20=xv[1][2]-xv[0][2];let j21=xv[2][2]-xv[0][2];let j22=xv[3][2]-xv[0][2];
        let dj=j00*(j11*j22-j12*j21)-j01*(j10*j22-j12*j20)+j02*(j10*j21-j11*j20);
        let id=1.0/dj;
        let ja=(j11*j22-j12*j21)*id;let jb=(j02*j21-j01*j22)*id;let jc=(j01*j12-j02*j11)*id;
        let jd=(j12*j20-j10*j22)*id;let je=(j00*j22-j02*j20)*id;let jf=(j02*j10-j00*j12)*id;
        let jg=(j10*j21-j11*j20)*id;let jh=(j01*j20-j00*j21)*id;let ji=(j00*j11-j01*j10)*id;
        let tgx=[-1.0,1.0,0.0,0.0];let tgy=[-1.0,0.0,1.0,0.0];let tgz=[-1.0,0.0,0.0,1.0];
        let mut gx=[0.0;4];let mut gy=[0.0;4];let mut gz=[0.0;4];
        for i in 0..4{gx[i]=ja*tgx[i]+jb*tgy[i]+jc*tgz[i];gy[i]=jd*tgx[i]+je*tgy[i]+jf*tgz[i];gz[i]=jg*tgx[i]+jh*tgy[i]+ji*tgz[i];}

        let e1=[p[1][0]-p[0][0],p[1][1]-p[0][1],p[1][2]-p[0][2]];
        let e2=[p[2][0]-p[0][0],p[2][1]-p[0][1],p[2][2]-p[0][2]];
        let nx=e1[1]*e2[2]-e1[2]*e2[1];let ny=e1[2]*e2[0]-e1[0]*e2[2];let nz=e1[0]*e2[1]-e1[1]*e2[0];
        let al=(nx*nx+ny*ny+nz*nz).sqrt().max(1e-30);let h=(al*0.5*2.0).sqrt();
        let nu=[nx/al,ny/al,nz/al];
        let gam = cfg.gamma * (cfg.lambda + 2.0*cfg.mu) / h;

        for (_ti, (l, wt)) in tp.iter().zip(tw.iter()).enumerate() {
            let (l1,l2,l3)=(l[0],l[1],l[2]);let ph=[l1,l2,l3];
            let xp=[p[0][0]*l1+p[1][0]*l2+p[2][0]*l3,p[0][1]*l1+p[1][1]*l2+p[2][1]*l3,p[0][2]*l1+p[1][2]*l2+p[2][2]*l3];
            let wb = wt * al * 0.5;
            let g0 = (cfg.gap_function)(&xp);
            let ux = u[n3[0]*3]*ph[0]+u[n3[1]*3]*ph[1]+u[n3[2]*3]*ph[2];
            let uy = u[n3[0]*3+1]*ph[0]+u[n3[1]*3+1]*ph[1]+u[n3[2]*3+1]*ph[2];
            let uz = u[n3[0]*3+2]*ph[0]+u[n3[1]*3+2]*ph[1]+u[n3[2]*3+2]*ph[2];
            let un = ux*nu[0]+uy*nu[1]+uz*nu[2];
            let gap = un - g0;
            if gap >= 0.0 { continue; }

            let mut exx=0.0;let mut eyy=0.0;let mut ezz=0.0;
            let mut exy=0.0;let mut exz=0.0;let mut eyz=0.0;
            for i in 0..4{let ux_i=u[ns[i]as usize*3];let uy_i=u[ns[i]as usize*3+1];let uz_i=u[ns[i]as usize*3+2];
                exx+=ux_i*gx[i];eyy+=uy_i*gy[i];ezz+=uz_i*gz[i];
                exy+=0.5*(ux_i*gy[i]+uy_i*gx[i]);exz+=0.5*(ux_i*gz[i]+uz_i*gx[i]);eyz+=0.5*(uy_i*gz[i]+uz_i*gy[i]);}
            let div=exx+eyy+ezz;
            let snx=cfg.lambda*div*nu[0]+2.0*cfg.mu*(exx*nu[0]+exy*nu[1]+exz*nu[2]);
            let sny=cfg.lambda*div*nu[1]+2.0*cfg.mu*(exy*nu[0]+eyy*nu[1]+eyz*nu[2]);
            let snz=cfg.lambda*div*nu[2]+2.0*cfg.mu*(exz*nu[0]+eyz*nu[1]+ezz*nu[2]);

            for li in 0..3 {
                let di=n3[li]*3;let dy=n3[li]*3+1;let dz=n3[li]*3+2;
                let pl = ph[li];
                rhs[di]-=snx*pl*wb;rhs[dy]-=sny*pl*wb;rhs[dz]-=snz*pl*wb;
                let fp = gam*(-gap)*pl*wb;
                rhs[di]+=fp*nu[0];rhs[dy]+=fp*nu[1];rhs[dz]+=fp*nu[2];
                for lj in 0..3 {
                    let plj=ph[lj];let jx=n3[lj]*3;let jy=n3[lj]*3+1;let jz=n3[lj]*3+2;
                    let kp = gam*pl*plj*wb;
                    coo.add(di,jx,kp*nu[0]*nu[0]);coo.add(di,jy,kp*nu[0]*nu[1]);coo.add(di,jz,kp*nu[0]*nu[2]);
                    coo.add(dy,jx,kp*nu[1]*nu[0]);coo.add(dy,jy,kp*nu[1]*nu[1]);coo.add(dy,jz,kp*nu[1]*nu[2]);
                    coo.add(dz,jx,kp*nu[2]*nu[0]);coo.add(dz,jy,kp*nu[2]*nu[1]);coo.add(dz,jz,kp*nu[2]*nu[2]);
                }
            }
        }
    }
    (rhs, coo.into_csr())
}

fn tri3() -> (Vec<[f64;3]>, Vec<f64>) {
    (vec![[1./3.,1./3.,1./3.],[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]],
     vec![-27./48.,25./48.,25./48.,25./48.])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    #[test]
    fn nitsche_contact_3d_finite() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let c = NitscheContactConfig {
            contact_tags: vec![2], gamma: 10.0, theta: -1.0,
            lambda: 100.0, mu: 50.0,
            gap_function: |x| 0.1 - x[1],
        };
        let u = vec![0.0; m.n_nodes()as usize*3];
        let (f,k)=assemble_nitsche_contact_3d(&m,&c,&u);
        assert!(f.iter().all(|v|v.is_finite()));
        assert!(k.nrows>0);
    }
}
