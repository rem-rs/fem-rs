//! Self-contact with KD-tree proximity and radial-return friction.
//!
//! Uses TetPointLocator for proximity queries on boundary faces.
//! Applies penalty/AL normal contact + Coulomb friction with radial return.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, point_locator::TetPointLocator};
use crate::contact::{ContactConfig, FrictionModel};

fn gp(x: f64) -> f64 { if x < 0.0 { -x } else { 0.0 } }
fn gpd(x: f64) -> f64 { if x < 0.0 { -1.0 } else { 0.0 } }
fn gps(x: f64, e: f64) -> f64 {
    if x >= e { 0.0 } else if x <= -e { -x-0.5*e } else { (x-e)*(x-e)/(4.0*e) }
}
fn gpsd(x: f64, e: f64) -> f64 {
    if x >= e { 0.0 } else if x <= -e { -1.0 } else { (x-e)/(2.0*e) }
}
fn tri3() -> (Vec<[f64;3]>, Vec<f64>) {
    (vec![[1./3.,1./3.,1./3.],[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]],
     vec![-27./48.,25./48.,25./48.,25./48.])
}

/// Assemble self-contact for 3D tet meshes.
pub fn assemble_self_contact_3d<M: MeshTopology>(
    mesh: &M, cfg: &ContactConfig, u: &[f64], lam: &[f64],
    locator: &TetPointLocator, gap_max: f64,
) -> (Vec<f64>, CsrMatrix<f64>) {
    let n = mesh.n_nodes() as usize; let nd = n*3;
    let mut rhs = vec![0.0; nd]; let mut coo = CooMatrix::new(nd, nd);
    let pn = cfg.penalty_normal;
    let (pt, mu) = match &cfg.friction {
        FrictionModel::Frictionless => (0.0,0.0),
        FrictionModel::Coulomb{mu,penalty_tangential}=>(*penalty_tangential,*mu),
    };
    let cs: std::collections::HashSet<i32> = cfg.contact_tags.iter().copied().collect();
    let (tp, tw) = tri3(); let hl = !lam.is_empty();

    for f in 0..mesh.n_boundary_faces() as u32 {
        let tg = mesh.face_tag(f); if !cs.contains(&tg) { continue; }
        let fn_ = mesh.face_nodes(f); if fn_.len() < 3 { continue; }
        let n3 = [fn_[0]as usize, fn_[1]as usize, fn_[2]as usize];
        let p = [mesh.node_coords(fn_[0]), mesh.node_coords(fn_[1]), mesh.node_coords(fn_[2])];
        let e1=[p[1][0]-p[0][0],p[1][1]-p[0][1],p[1][2]-p[0][2]];
        let e2=[p[2][0]-p[0][0],p[2][1]-p[0][1],p[2][2]-p[0][2]];
        let nx=e1[1]*e2[2]-e1[2]*e2[1];let ny=e1[2]*e2[0]-e1[0]*e2[2];let nz=e1[0]*e2[1]-e1[1]*e2[0];
        let al=(nx*nx+ny*ny+nz*nz).sqrt().max(1e-30); let fa=al*0.5;
        let nu=[nx/al,ny/al,nz/al];
        let ad=[nu[0].abs(),nu[1].abs(),nu[2].abs()];
        let rd=if ad[0]<=ad[1]&&ad[0]<=ad[2]{[1.,0.,0.]}else if ad[1]<=ad[2]{[0.,1.,0.]}else{[0.,0.,1.]};
        let tx=nu[1]*rd[2]-nu[2]*rd[1];let ty=nu[2]*rd[0]-nu[0]*rd[2];let tz=nu[0]*rd[1]-nu[1]*rd[0];
        let tl=(tx*tx+ty*ty+tz*tz).sqrt().max(1e-30);
        let t1=[tx/tl,ty/tl,tz/tl];
        let t2=[nu[1]*t1[2]-nu[2]*t1[1],nu[2]*t1[0]-nu[0]*t1[2],nu[0]*t1[1]-nu[1]*t1[0]];

        for (ti,(l,wt)) in tp.iter().zip(tw.iter()).enumerate() {
            let(l1,l2,l3)=(l[0],l[1],l[2]);let ph=[l1,l2,l3];
            let xp=[p[0][0]*l1+p[1][0]*l2+p[2][0]*l3,p[0][1]*l1+p[1][1]*l2+p[2][1]*l3,p[0][2]*l1+p[1][2]*l2+p[2][2]*l3];
            let wb = wt*fa;
            let nn = locator.nearest_node(&xp) as usize;
            if nn >= n { continue; }
            let q = mesh.node_coords(nn as u32);
            let dx = xp[0]-q[0]; let dy = xp[1]-q[1]; let dz = xp[2]-q[2];
            if (dx*dx+dy*dy+dz*dz).sqrt() > gap_max { continue; }
            let gap = -(dx*nu[0]+dy*nu[1]+dz*nu[2]);
            if gap > 0.0 { continue; }

            let ux = u[n3[0]*3]*ph[0]+u[n3[1]*3]*ph[1]+u[n3[2]*3]*ph[2];
            let uy = u[n3[0]*3+1]*ph[0]+u[n3[1]*3+1]*ph[1]+u[n3[2]*3+1]*ph[2];
            let uz = u[n3[0]*3+2]*ph[0]+u[n3[1]*3+2]*ph[1]+u[n3[2]*3+2]*ph[2];
            let un = ux*nu[0]+uy*nu[1]+uz*nu[2];
            let u1 = ux*t1[0]+uy*t1[1]+uz*t1[2];
            let u2 = ux*t2[0]+uy*t2[1]+uz*t2[2];
            let gun = un - gap;
            let lv = if hl { lam[ti%lam.len()] } else { 0.0 };
            let act = lv + pn*gun;
            let (np,npd) = if !hl { (gp(gun),gpd(gun)) } else { (gps(act,1e-8),gpsd(act,1e-8)) };
            let fv = if !hl { -pn*np*wb } else { -(lv+pn*np)*wb };

            for li in 0..3 {
                let dx=n3[li]*3;let dy=n3[li]*3+1;let dz=n3[li]*3+2;
                rhs[dx]+=fv*nu[0]*ph[li];rhs[dy]+=fv*nu[1]*ph[li];rhs[dz]+=fv*nu[2]*ph[li];
                rhs[nn*3]-=fv*nu[0]*ph[li];rhs[nn*3+1]-=fv*nu[1]*ph[li];rhs[nn*3+2]-=fv*nu[2]*ph[li];
                for lj in 0..3 {
                    let k = -pn*npd*ph[li]*ph[lj]*wb;let jx=n3[lj]*3;let jy=n3[lj]*3+1;let jz=n3[lj]*3+2;
                    coo.add(dx,jx,k*nu[0]*nu[0]);coo.add(dx,jy,k*nu[0]*nu[1]);coo.add(dx,jz,k*nu[0]*nu[2]);
                    coo.add(dy,jx,k*nu[1]*nu[0]);coo.add(dy,jy,k*nu[1]*nu[1]);coo.add(dy,jz,k*nu[1]*nu[2]);
                    coo.add(dz,jx,k*nu[2]*nu[0]);coo.add(dz,jy,k*nu[2]*nu[1]);coo.add(dz,jz,k*nu[2]*nu[2]);
                }
            }
            if pt > 0.0 && mu > 0.0 {
                let sn = (-fv/wb).max(0.0);
                let s1 = pt*u1; let s2 = pt*u2; let sm = (s1*s1+s2*s2).sqrt().max(1e-30);
                if sm <= mu*sn + 1e-15 {
                    let ks = pt*wb;
                    for li in 0..3 { let dx=n3[li]*3;let dy=n3[li]*3+1;let dz=n3[li]*3+2;
                        rhs[dx]-=s1*t1[0]*ph[li]+s2*t2[0]*ph[li];rhs[dy]-=s1*t1[1]*ph[li]+s2*t2[1]*ph[li];rhs[dz]-=s1*t1[2]*ph[li]+s2*t2[2]*ph[li];
                        for lj in 0..3 { let b=ks*ph[li]*ph[lj];let jx=n3[lj]*3;let jy=n3[lj]*3+1;let jz=n3[lj]*3+2;
                            coo.add(dx,jx,b*(t1[0]*t1[0]+t2[0]*t2[0]));coo.add(dx,jy,b*(t1[0]*t1[1]+t2[0]*t2[1]));coo.add(dx,jz,b*(t1[0]*t1[2]+t2[0]*t2[2]));
                            coo.add(dy,jx,b*(t1[1]*t1[0]+t2[1]*t2[0]));coo.add(dy,jy,b*(t1[1]*t1[1]+t2[1]*t2[1]));coo.add(dy,jz,b*(t1[1]*t1[2]+t2[1]*t2[2]));
                            coo.add(dz,jx,b*(t1[2]*t1[0]+t2[2]*t2[0]));coo.add(dz,jy,b*(t1[2]*t1[1]+t2[2]*t2[1]));coo.add(dz,jz,b*(t1[2]*t1[2]+t2[2]*t2[2]));}
                    }
                } else { let sc=mu*sn/sm; for li in 0..3 { let dx=n3[li]*3;let dy=n3[li]*3+1;let dz=n3[li]*3+2;
                    rhs[dx]-=(sc*s1*t1[0]+sc*s2*t2[0])*ph[li];rhs[dy]-=(sc*s1*t1[1]+sc*s2*t2[1])*ph[li];rhs[dz]-=(sc*s1*t1[2]+sc*s2*t2[2])*ph[li];}
                }
            }
        }
    }
    (rhs, coo.into_csr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test] fn self_contact_3d_finite() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let loc = TetPointLocator::new(&m);
        let cfg = ContactConfig {
            penalty_normal: 1e6, contact_type: crate::contact::ContactType::Penalty,
            friction: FrictionModel::Frictionless, gap_function: |_| 0.0, contact_tags: vec![1,2,3,4,5,6],
        };
        let u = vec![0.0; m.n_nodes() as usize * 3];
        let (f,k)=assemble_self_contact_3d(&m,&cfg,&u,&[],&loc,1.0);
        assert!(f.iter().all(|v|v.is_finite())); assert!(k.nrows>0);
    }
    #[test] fn self_contact_3d_friction_finite() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let loc = TetPointLocator::new(&m);
        let cfg = ContactConfig {
            penalty_normal: 1e6, contact_type: crate::contact::ContactType::Penalty,
            friction: FrictionModel::Coulomb{mu:0.3,penalty_tangential:1e5},
            gap_function: |_| 0.0, contact_tags: vec![1,2,3,4,5,6],
        };
        let u = vec![0.0; m.n_nodes() as usize * 3];
        let (f,k)=assemble_self_contact_3d(&m,&cfg,&u,&[],&loc,1.0);
        assert!(f.iter().all(|v|v.is_finite())); assert!(k.nrows>0);
    }
}
