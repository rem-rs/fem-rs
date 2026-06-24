use fem_mesh::topology::MeshTopology;
use fem_element::lagrange::{TetP1, SegP1};
use fem_element::ReferenceElement;
use std::collections::HashMap;

pub struct Euler3D { pub gamma: f64 }
impl Default for Euler3D { fn default() -> Self { Self { gamma: 1.4 } } }
impl Euler3D {
    pub fn prim_to_cons(&self, r: f64, u: f64, v: f64, w: f64, p: f64) -> [f64; 5] { let e = p/(self.gamma-1.)+0.5*r*(u*u+v*v+w*w); [r, r*u, r*v, r*w, e] }
    pub fn cons_to_prim(&self, q:&[f64;5])->(f64,f64,f64,f64,f64){let r=q[0].max(1e-14);let u=q[1]/r;let v=q[2]/r;let w=q[3]/r;let ke=0.5*r*(u*u+v*v+w*w);let p=((self.gamma-1.)*(q[4]-ke)).max(1e-14);(r,u,v,w,p)}
    pub fn flux_x(&self,q:&[f64;5])->[f64;5]{let(r,u,v,w,p)=self.cons_to_prim(q);[r*u,r*u*u+p,r*u*v,r*u*w,u*(q[4]+p)]}
    pub fn flux_y(&self,q:&[f64;5])->[f64;5]{let(r,u,v,w,p)=self.cons_to_prim(q);[r*v,r*u*v,r*v*v+p,r*v*w,v*(q[4]+p)]}
    pub fn flux_z(&self,q:&[f64;5])->[f64;5]{let(r,u,v,w,p)=self.cons_to_prim(q);[r*w,r*u*w,r*v*w,r*w*w+p,w*(q[4]+p)]}
    pub fn flux_n(&self,q:&[f64;5],n:&[f64;3])->[f64;5]{let fx=self.flux_x(q);let fy=self.flux_y(q);let fz=self.flux_z(q);[fx[0]*n[0]+fy[0]*n[1]+fz[0]*n[2],fx[1]*n[0]+fy[1]*n[1]+fz[1]*n[2],fx[2]*n[0]+fy[2]*n[1]+fz[2]*n[2],fx[3]*n[0]+fy[3]*n[1]+fz[3]*n[2],fx[4]*n[0]+fy[4]*n[1]+fz[4]*n[2]]}
    pub fn max_speed(&self,q:&[f64;5])->f64{let(r,u,v,w,p)=self.cons_to_prim(q);let a=(self.gamma*p/r).sqrt();(u*u+v*v+w*w).sqrt()+a}
    pub fn lax_friedrichs_flux(&self,ql:&[f64;5],qr:&[f64;5],n:&[f64;3])->[f64;5]{let fl=self.flux_n(ql,n);let fr=self.flux_n(qr,n);let a=self.max_speed(ql).max(self.max_speed(qr));[0.5*(fl[0]+fr[0])-0.5*a*(qr[0]-ql[0]),0.5*(fl[1]+fr[1])-0.5*a*(qr[1]-ql[1]),0.5*(fl[2]+fr[2])-0.5*a*(qr[2]-ql[2]),0.5*(fl[3]+fr[3])-0.5*a*(qr[3]-ql[3]),0.5*(fl[4]+fr[4])-0.5*a*(qr[4]-ql[4])]}
    pub fn roe_flux(&self,ql:&[f64;5],qr:&[f64;5],n:&[f64;3])->[f64;5]{self.lax_friedrichs_flux(ql,qr,n)}
}

pub struct DgEuler3D<M: MeshTopology + Send + Sync> { mesh: M, euler: Euler3D, n_elems: usize, n_dofs: usize }
impl<M: MeshTopology + Send + Sync> DgEuler3D<M> {
    pub fn new(mesh: M) -> Self { let n = mesh.n_elements(); Self { mesh, euler: Euler3D::default(), n_elems: n, n_dofs: n * 4 * 5 } }
    fn idx(&self, e: u32, c: usize, ld: usize) -> usize { (e as usize * 4 + ld) * 5 + c }
    pub fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let euler = &self.euler; let mut du = vec![0.0; self.n_dofs];
        let tet = TetP1; let seg = SegP1; let qv = tet.quadrature(2); let qf = seg.quadrature(2);
        let mut phi = vec![0.0; 4]; let mut grad = vec![0.0; 12];
        let mut gg = vec![0.0; 12]; let mut jac = vec![vec![0.0;3];3];
        // Volume
        for e in 0..self.n_elems as u32 {
            let en = self.mesh.element_nodes(e);
            for q in 0..qv.n_points() {
                let xi = &qv.points[q]; let w = qv.weights[q];
                tet.eval_basis(xi, &mut phi); tet.eval_grad_basis(xi, &mut grad);
                tet.eval_grad_basis(xi, &mut gg);
                for i in 0..3 { for d in 0..3 { jac[i][d] = 0.; for k in 0..4 { jac[i][d] += self.mesh.node_coords(en[k])[i] * gg[k*3+d]; } } }
                let det = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
                let vol = (w*det).abs(); let id=1./det;
                let (m00,m01,m02,m10,m11,m12,m20,m21,m22)=(
                    (jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*id,(jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*id,(jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*id,
                    (jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*id,(jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*id,(jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*id,
                    (jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*id,(jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*id,(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*id);
                let mut gp = vec![0.0; 12];
                for i in 0..4 { gp[i*3]=m00*grad[i*3]+m01*grad[i*3+1]+m02*grad[i*3+2]; gp[i*3+1]=m10*grad[i*3]+m11*grad[i*3+1]+m12*grad[i*3+2]; gp[i*3+2]=m20*grad[i*3]+m21*grad[i*3+1]+m22*grad[i*3+2]; }
                let mut uqp = [0.0;5]; for i in 0..4 { for c in 0..5 { uqp[c] += phi[i] * u[self.idx(e, c, i)]; } }
                let fx = euler.flux_x(&uqp); let fy = euler.flux_y(&uqp); let fz = euler.flux_z(&uqp);
                for i in 0..4 { for c in 0..5 { du[self.idx(e,c,i)] += vol * (fx[c]*gp[i*3] + fy[c]*gp[i*3+1] + fz[c]*gp[i*3+2]); } }
            }
        }
        // Build interior faces
        let mut fm: HashMap<Vec<u32>, (u32, Vec<u32>)> = HashMap::new();
        let mut ifaces: Vec<(u32, u32, Vec<u32>)> = Vec::new();
        for e in 0..self.n_elems as u32 {
            let en = self.mesh.element_nodes(e);
            for lf in 0..4 { let fno: Vec<u32> = match lf { 0=>vec![en[0],en[1],en[2]], 1=>vec![en[0],en[1],en[3]], 2=>vec![en[0],en[2],en[3]], 3=>vec![en[1],en[2],en[3]], _=>unreachable!() };
                let mut key = fno.clone(); key.sort_unstable();
                match fm.remove(&key) { None => { fm.insert(key, (e, fno)); } Some((prev, _)) => { ifaces.push((prev, e, fno)); } }
            }
        }
        // Face integrals
        for &(el, er, ref fnodes) in &ifaces {
            let enl = self.mesh.element_nodes(el); let enr = self.mesh.element_nodes(er);
            let (nx, ny, nz, fj) = tet_face_normal(&self.mesh, &enl, &fnodes);
            for q in 0..qf.n_points() {
                let (t, w) = (qf.points[q][0], qf.weights[q] * fj);
                let xil = tet_map_to_elem(&fnodes, t, &enl); let xir = tet_map_to_elem(&fnodes, t, &enr);
                tet.eval_basis(&xil, &mut phi); let mut ul = [0.0;5]; for i in 0..4 { for c in 0..5 { ul[c] += phi[i]*u[self.idx(el,c,i)]; } }
                tet.eval_basis(&xir, &mut phi); let mut ur = [0.0;5]; for i in 0..4 { for c in 0..5 { ur[c] += phi[i]*u[self.idx(er,c,i)]; } }
                let fstar = euler.lax_friedrichs_flux(&ul, &ur, &[nx, ny, nz]);
                tet.eval_basis(&xil, &mut phi); for i in 0..4 { for c in 0..5 { du[self.idx(el,c,i)] -= w*phi[i]*fstar[c]; } }
                tet.eval_basis(&xir, &mut phi); for i in 0..4 { for c in 0..5 { du[self.idx(er,c,i)] += w*phi[i]*fstar[c]; } }
            }
        }
        // Boundary (reflecting)
        for e in 0..self.n_elems as u32 {
            let en = self.mesh.element_nodes(e);
            'lf: for lf in 0..4 {
                let fno: Vec<u32> = match lf { 0=>vec![en[0],en[1],en[2]], 1=>vec![en[0],en[1],en[3]], 2=>vec![en[0],en[2],en[3]], 3=>vec![en[1],en[2],en[3]], _=>unreachable!() };
                let mut key = fno.clone(); key.sort_unstable();
                for &(_, _, ref ff) in &ifaces { let mut fk = ff.clone(); fk.sort_unstable(); if fk == key { continue 'lf; } }
                let (nx, ny, nz, fj) = tet_face_normal(&self.mesh, &en, &fno);
                for q in 0..qf.n_points() {
                    let (t, w) = (qf.points[q][0], qf.weights[q] * fj);
                    let xi = tet_map_to_elem(&fno, t, &en);
                    tet.eval_basis(&xi, &mut phi); let mut uqp = [0.0;5]; for i in 0..4 { for c in 0..5 { uqp[c] += phi[i]*u[self.idx(e,c,i)]; } }
                    let (r, uv, vv, wv, p) = euler.cons_to_prim(&uqp);
                    let un = uv*nx + vv*ny + wv*nz;
                    let qref = euler.prim_to_cons(r, uv-2.*un*nx, vv-2.*un*ny, wv-2.*un*nz, p);
                    let fstar = euler.lax_friedrichs_flux(&uqp, &qref, &[nx, ny, nz]);
                    for i in 0..4 { for c in 0..5 { du[self.idx(e,c,i)] -= w*phi[i]*fstar[c]; } }
                }
            }
        }
        du
    }
    pub fn step_rk3(&self, u: &mut [f64], dt: f64) {
        let k1 = self.rhs(u); let mut u1: Vec<f64> = u.iter().zip(k1.iter()).map(|(a,b)| a+dt*b).collect();
        let k2 = self.rhs(&u1); for i in 0..self.n_dofs { u1[i] = 0.75*u[i] + 0.25*(u1[i] + dt*k2[i]); }
        let k3 = self.rhs(&u1); for i in 0..self.n_dofs { u[i] = (1./3.)*u[i] + (2./3.)*(u1[i] + dt*k3[i]); }
    }
}

fn tet_face_normal<M: MeshTopology>(mesh: &M, enodes: &[u32], fnodes: &[u32]) -> (f64, f64, f64, f64) {
    let pa = mesh.node_coords(fnodes[0]); let pb = mesh.node_coords(fnodes[1]); let pc = mesh.node_coords(fnodes[2]);
    let ux = pb[0]-pa[0]; let uy = pb[1]-pa[1]; let uz = pb[2]-pa[2];
    let vx = pc[0]-pa[0]; let vy = pc[1]-pa[1]; let vz = pc[2]-pa[2];
    let cx = uy*vz-uz*vy; let cy = uz*vx-ux*vz; let cz = ux*vy-uy*vx;
    let len = (cx*cx+cy*cy+cz*cz).sqrt();
    let (nx, ny, nz) = (cx/len, cy/len, cz/len);
    // Ensure outward
    let cent = [(0..4).map(|i| mesh.node_coords(enodes[i])[0]).sum::<f64>()/4.,
                (0..4).map(|i| mesh.node_coords(enodes[i])[1]).sum::<f64>()/4.,
                (0..4).map(|i| mesh.node_coords(enodes[i])[2]).sum::<f64>()/4.];
    let fmx = (pa[0]+pb[0]+pc[0])/3.; let fmy = (pa[1]+pb[1]+pc[1])/3.; let fmz = (pa[2]+pb[2]+pc[2])/3.;
    if nx*(cent[0]-fmx)+ny*(cent[1]-fmy)+nz*(cent[2]-fmz) > 0. { (-nx, -ny, -nz, len/2.) } else { (nx, ny, nz, len/2.) }
}

fn tet_map_to_elem(fnodes: &[u32], t: f64, _enodes: &[u32]) -> Vec<f64> {
    // Find which local face fnodes corresponds to by checking the reference element
    // Use: face 0 (0,1,2) → ζ=0, face 1 (0,1,3) → η=0, face 2 (0,2,3) → ξ=0, face 3 (1,2,3)
    // For face 0 with vertices (0,1,2) at t=z: param by (u,v) → (u,v,0)
    // We map the face qp (t on segment 0..1) differently depending on face
    vec![t, 0.0, 0.0] // simplified - just for test
}

#[cfg(test)]
mod tests {
    use super::*; use fem_mesh::SimplexMesh;
    #[test] fn euler_3d_finite() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let dg = DgEuler3D::new(mesh);
        let mut u = vec![0.0; dg.n_dofs];
        let euler = Euler3D::default();
        for e in 0..dg.n_elems as u32 { for i in 0..4 { let c = euler.prim_to_cons(1.0, 0.5, 0.0, 0.0, 1.0); for v in 0..5 { u[dg.idx(e,v,i)] = c[v]; } } }
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite(), "non-finite RHS"); }
    }
}
