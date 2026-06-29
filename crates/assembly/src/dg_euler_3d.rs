use fem_mesh::topology::MeshTopology;
use fem_element::lagrange::{TetP1, TriP1};
use fem_element::ReferenceElement;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerFluxKind { LaxFriedrichs, Roe, Hllc }
impl Default for EulerFluxKind { fn default() -> Self { EulerFluxKind::LaxFriedrichs } }

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

    pub fn hllc_flux(&self, ql: &[f64; 5], qr: &[f64; 5], n: &[f64; 3]) -> [f64; 5] {
        let fl = self.flux_n(ql, n);
        let fr = self.flux_n(qr, n);
        let (rl, ul, vl, wl, pl) = self.cons_to_prim(ql);
        let (rr, ur, vr, wr, pr) = self.cons_to_prim(qr);
        let unl = ul*n[0] + vl*n[1] + wl*n[2];
        let unr = ur*n[0] + vr*n[1] + wr*n[2];
        let al = (self.gamma*pl/rl).sqrt();
        let ar = (self.gamma*pr/rr).sqrt();
        let sl = (unl - al).min(unr - ar);
        let sr = (unl + al).max(unr + ar);
        if sl >= 0.0 { return fl; }
        if sr <= 0.0 { return fr; }
        let raw = rr*(sr-unr) - rl*(sl-unl);
        let denom = if raw.abs() < 1e-14 { 1e-14_f64.copysign(raw) } else { raw };
        let sm = (rr*unr*(sr-unr) - rl*unl*(sl-unl) + pl - pr) / denom;
        let star = |q: &[f64;5], r: f64, u: f64, v: f64, w: f64, un: f64, p: f64, sk: f64| -> [f64;5] {
            let dsm = sk - sm;
            let safe = if dsm.abs() < 1e-14 { 1e-14_f64.copysign(dsm) } else { dsm };
            let fac = r * (sk - un) / safe;
            let ek = q[4];
            [
                fac,
                fac*(n[0]*sm + u - n[0]*un),
                fac*(n[1]*sm + v - n[1]*un),
                fac*(n[2]*sm + w - n[2]*un),
                fac*(ek/r + (sm-un)*(sm + p/(r*(sk-un)))),
            ]
        };
        if sm >= 0.0 {
            let qst = star(ql, rl, ul, vl, wl, unl, pl, sl);
            [fl[0]+sl*(qst[0]-ql[0]), fl[1]+sl*(qst[1]-ql[1]), fl[2]+sl*(qst[2]-ql[2]), fl[3]+sl*(qst[3]-ql[3]), fl[4]+sl*(qst[4]-ql[4])]
        } else {
            let qst = star(qr, rr, ur, vr, wr, unr, pr, sr);
            [fr[0]+sr*(qst[0]-qr[0]), fr[1]+sr*(qst[1]-qr[1]), fr[2]+sr*(qst[2]-qr[2]), fr[3]+sr*(qst[3]-qr[3]), fr[4]+sr*(qst[4]-qr[4])]
        }
    }

    pub fn numerical_flux(&self, kind: EulerFluxKind, ql: &[f64;5], qr: &[f64;5], n: &[f64;3]) -> [f64;5] {
        match kind {
            EulerFluxKind::LaxFriedrichs => self.lax_friedrichs_flux(ql, qr, n),
            EulerFluxKind::Roe => self.roe_flux(ql, qr, n),
            EulerFluxKind::Hllc => self.hllc_flux(ql, qr, n),
        }
    }
}

pub struct DgEuler3D<M: MeshTopology + Send + Sync> { mesh: M, euler: Euler3D, n_elems: usize, n_dofs: usize, pub flux_kind: EulerFluxKind }
impl<M: MeshTopology + Send + Sync> DgEuler3D<M> {
    pub fn new(mesh: M) -> Self { let n = mesh.n_elements(); Self { mesh, euler: Euler3D::default(), n_elems: n, n_dofs: n * 4 * 5, flux_kind: EulerFluxKind::LaxFriedrichs } }
    pub fn with_flux(mut self, kind: EulerFluxKind) -> Self { self.flux_kind = kind; self }
    fn idx(&self, e: u32, c: usize, ld: usize) -> usize { (e as usize * 4 + ld) * 5 + c }
    pub fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let euler = &self.euler; let mut du = vec![0.0; self.n_dofs];
        let tet = TetP1; let tri = TriP1; let qv = tet.quadrature(2); let qf = tri.quadrature(3);
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
            // Determine which local face index fnodes corresponds to
            let face_idx = |en: &[u32]| -> usize {
                if fnodes[0]==en[0]||fnodes[0]==en[1]||fnodes[0]==en[2] { if fnodes[2]==en[0]||fnodes[2]==en[1]||fnodes[2]==en[2] { return 0; } }
                if fnodes[0]==en[0]||fnodes[0]==en[1]||fnodes[0]==en[3] { if fnodes[2]==en[0]||fnodes[2]==en[1]||fnodes[2]==en[3] { return 1; } }
                if fnodes[0]==en[0]||fnodes[0]==en[2]||fnodes[0]==en[3] { if fnodes[2]==en[0]||fnodes[2]==en[2]||fnodes[2]==en[3] { return 2; } }
                3
            };
            let fl = face_idx(&enl);
            for q in 0..qf.n_points() {
                let (pu, pv) = (qf.points[q][0], qf.points[q][1]);
                let w = qf.weights[q] * fj;
                let xil = match fl {
                    0 => vec![pu, pv, 0.0], 1 => vec![pu, 0.0, pv], 2 => vec![0.0, pu, pv],
                    _ => vec![1.0-pu-pv, pu, pv],
                };
                let fr = face_idx(&enr);
                let xir = match fr {
                    0 => vec![pu, pv, 0.0], 1 => vec![pu, 0.0, pv], 2 => vec![0.0, pu, pv],
                    _ => vec![1.0-pu-pv, pu, pv],
                };
                tet.eval_basis(&xil, &mut phi); let mut ul = [0.0;5]; for i in 0..4 { for c in 0..5 { ul[c] += phi[i]*u[self.idx(el,c,i)]; } }
                tet.eval_basis(&xir, &mut phi); let mut ur = [0.0;5]; for i in 0..4 { for c in 0..5 { ur[c] += phi[i]*u[self.idx(er,c,i)]; } }
                let fstar = euler.numerical_flux(self.flux_kind, &ul, &ur, &[nx, ny, nz]);
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
                    let (pu, pv) = (qf.points[q][0], qf.points[q][1]);
                    let w = qf.weights[q] * fj;
                    let xi = match lf {
                        0 => vec![pu, pv, 0.0], 1 => vec![pu, 0.0, pv], 2 => vec![0.0, pu, pv],
                        _ => vec![1.0-pu-pv, pu, pv],
                    };
                    tet.eval_basis(&xi, &mut phi); let mut uqp = [0.0;5]; for i in 0..4 { for c in 0..5 { uqp[c] += phi[i]*u[self.idx(e,c,i)]; } }
                    let (r, uv, vv, wv, p) = euler.cons_to_prim(&uqp);
                    let un = uv*nx + vv*ny + wv*nz;
                    let qref = euler.prim_to_cons(r, uv-2.*un*nx, vv-2.*un*ny, wv-2.*un*nz, p);
                    let fstar = euler.numerical_flux(self.flux_kind, &uqp, &qref, &[nx, ny, nz]);
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

    #[test]
    fn hllc_consistent_at_uniform_state() {
        let e = Euler3D::default();
        let q = e.prim_to_cons(1.0, 0.3, 0.2, 0.1, 1.0);
        let n = [1.0_f64, 0.0, 0.0];
        let f_hllc = e.hllc_flux(&q, &q, &n);
        let f_lf = e.lax_friedrichs_flux(&q, &q, &n);
        let f_phy = e.flux_n(&q, &n);
        for i in 0..5 {
            assert!((f_hllc[i] - f_phy[i]).abs() < 1e-10, "HLLC consistency violated at i={i}: {:?} vs {:?}", f_hllc[i], f_phy[i]);
            assert!((f_lf[i]  - f_phy[i]).abs() < 1e-10, "LF consistency violated at i={i}");
        }
    }

    #[test]
    fn hllc_sod_shock_returns_finite() {
        // Sod tube initial states (1D, n = x-axis).
        let e = Euler3D::default();
        let ql = e.prim_to_cons(1.0,   0.0, 0.0, 0.0, 1.0);
        let qr = e.prim_to_cons(0.125, 0.0, 0.0, 0.0, 0.1);
        let n = [1.0_f64, 0.0, 0.0];
        let f_hllc = e.hllc_flux(&ql, &qr, &n);
        let f_lf   = e.lax_friedrichs_flux(&ql, &qr, &n);
        for i in 0..5 {
            assert!(f_hllc[i].is_finite(), "HLLC Sod flux non-finite at i={i}");
            assert!(f_lf[i].is_finite(),   "LF Sod flux non-finite at i={i}");
        }
        // HLLC should differ from LF on a strong discontinuity (sanity).
        let diff: f64 = (0..5).map(|i| (f_hllc[i] - f_lf[i]).abs()).sum();
        assert!(diff > 1e-3, "HLLC should differ from LF on a shock; got diff={diff:.3e}");
    }

    #[test]
    fn dg_euler_3d_with_hllc_runs() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let dg = DgEuler3D::new(mesh).with_flux(EulerFluxKind::Hllc);
        let euler = Euler3D::default();
        let mut u = vec![0.0; dg.n_dofs];
        for e in 0..dg.n_elems as u32 {
            for i in 0..4 {
                let c = euler.prim_to_cons(1.0, 0.2, 0.1, 0.0, 1.0);
                for v in 0..5 { u[dg.idx(e,v,i)] = c[v]; }
            }
        }
        let du = dg.rhs(&u);
        for v in &du { assert!(v.is_finite(), "HLLC DG RHS non-finite"); }
    }
}
