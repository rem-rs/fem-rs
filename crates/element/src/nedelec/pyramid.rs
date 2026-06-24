use std::sync::OnceLock;
use crate::quadrature::pyramid_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

const EDGES: [(usize, usize); 8] = [
    (0,1), (1,2), (2,3), (0,3), (0,4), (1,4), (2,4), (3,4),
];

/// Evaluate Whitney forms and curls at (x,y,z) in physical space.
fn whitney_all(x: f64, y: f64, z: f64) -> ([f64; 24], [f64; 24]) {
    if z >= 1. - 1e-14 { return ([0.; 24], [0.; 24]); }
    let a = 1. - z; let r = x / a; let s = y / a;
    let u = 1.-r; let v = 1.-s; let w = 1.-z;
    let th = [u*v*w, r*v*w, r*s*w, u*s*w, z];
    let gth = [[-v*w, -u*w, -u*v], [v*w, -r*w, -r*v], [s*w, r*w, -r*s],
               [-s*w, u*w, -u*s], [0., 0., 1.]];
    let mut vals = [0.; 24]; let mut curls = [0.; 24];
    for (ei, &(i, j)) in EDGES.iter().enumerate() {
        let pr = [th[i]*gth[j][0]-th[j]*gth[i][0],
                  th[i]*gth[j][1]-th[j]*gth[i][1],
                  th[i]*gth[j][2]-th[j]*gth[i][2]];
        vals[ei*3]=pr[0]/a; vals[ei*3+1]=pr[1]/a; vals[ei*3+2]= r*pr[0]/a + s*pr[1]/a + pr[2];
        let cx = gth[i][1]*gth[j][2]-gth[i][2]*gth[j][1];
        let cy = gth[i][2]*gth[j][0]-gth[i][0]*gth[j][2];
        let cz = gth[i][0]*gth[j][1]-gth[i][1]*gth[j][0];
        let cr = [2.*cx, 2.*cy, 2.*cz];
        let a2 = a*a;
        curls[ei*3]=(a*cr[0] - r*cr[2])/a2; curls[ei*3+1]=(a*cr[1] - s*cr[2])/a2; curls[ei*3+2]=cr[2]/a2;
    }
    (vals, curls)
}

/// Build DOF matrix M[8×8] where M[i][j] = DOF_i(Φ̂_j), then invert.
fn build_correction() -> [[f64; 8]; 8] {
    let mut m = [[0.; 8]; 8];
    let ed: [([f64;3],[f64;3]);8] = [
        ([0.,0.,0.],[1.,0.,0.]),([1.,0.,0.],[1.,1.,0.]),([1.,1.,0.],[0.,1.,0.]),
        ([0.,0.,0.],[0.,1.,0.]),([0.,0.,0.],[0.,0.,1.]),([1.,0.,0.],[0.,0.,1.]),
        ([1.,1.,0.],[0.,0.,1.]),([0.,1.,0.],[0.,0.,1.])];
    let gx = [0.0694318442029737,0.3300094782075719,0.6699905217924281,0.9305681557970263];
    let gw = [0.1739274225687269,0.3260725774312731,0.3260725774312731,0.1739274225687269];
    for (ei, (a, b)) in ed.iter().enumerate() {
        let d = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        for k in 0..4 {
            let (t, w) = (gx[k], gw[k]); let p = [a[0]+t*d[0], a[1]+t*d[1], a[2]+t*d[2]];
            let (vals, _) = whitney_all(p[0], p[1], p[2]);
            for j in 0..8 {
                let tang = vals[j*3]*d[0] + vals[j*3+1]*d[1] + vals[j*3+2]*d[2];
                m[ei][j] += w * tang;
            }
        }
    }
    // Invert m
    let mut inv = [[0.; 8]; 8]; for i in 0..8 { inv[i][i] = 1.; }
    let mut a = m;
    for c in 0..8 {
        let mut mr = c; let mut mv = a[c][c].abs();
        for r in (c+1)..8 { let x = a[r][c].abs(); if x > mv { mv = x; mr = r; } }
        if mv < 1e-15 { a.swap(c, mr); inv.swap(c, mr); a[c][c] = 1.; continue; }
        a.swap(c, mr); inv.swap(c, mr);
        let p = a[c][c]; let ip = 1./p;
        for j in 0..8 { a[c][j] *= ip; inv[c][j] *= ip; }
        for r in 0..8 { if r == c { continue; } let f = a[r][c];
            for j in 0..8 { a[r][j] -= f*a[c][j]; inv[r][j] -= f*inv[c][j]; } }
    }
    inv // correction matrix: Φ_i_corrected = Σ_j inv[i][j] · Φ̂_j
}

static CORR: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
fn correction() -> &'static [[f64; 8]; 8] { CORR.get_or_init(build_correction) }

pub struct PyraND1;

impl VectorReferenceElement for PyraND1 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 8 }
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let c = correction(); let (wf, _) = whitney_all(xi[0], xi[1], xi[2]);
        for i in 0..8 { values[i*3]=0.; values[i*3+1]=0.; values[i*3+2]=0.;
            for j in 0..8 {
                values[i*3] += c[i][j] * wf[j*3];
                values[i*3+1] += c[i][j] * wf[j*3+1];
                values[i*3+2] += c[i][j] * wf[j*3+2];
            }
        }
    }
    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let c = correction(); let (_, wc) = whitney_all(xi[0], xi[1], xi[2]);
        for i in 0..8 { curl_vals[i*3]=0.; curl_vals[i*3+1]=0.; curl_vals[i*3+2]=0.;
            for j in 0..8 {
                curl_vals[i*3] += c[i][j] * wc[j*3];
                curl_vals[i*3+1] += c[i][j] * wc[j*3+1];
                curl_vals[i*3+2] += c[i][j] * wc[j*3+2];
            }
        }
    }
    fn eval_div(&self, _xi: &[f64], div: &mut [f64]) { for v in div.iter_mut() { *v = 0.; } }
    fn quadrature(&self, order: u8) -> QuadratureRule { pyramid_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.5,0.,0.],vec![1.,0.5,0.],vec![0.5,1.,0.],vec![0.,0.5,0.],
             vec![0.,0.,0.5],vec![0.5,0.,0.5],vec![0.5,0.5,0.5],vec![0.,0.5,0.5]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn g4() -> ([f64;4],[f64;4]) {([0.0694318442029737,0.3300094782075719,0.6699905217924281,0.9305681557970263],[0.1739274225687269,0.3260725774312731,0.3260725774312731,0.1739274225687269])}
    #[test] fn nd1_finite() { let e = PyraND1; let mut v = vec![0.; 24]; for p in &e.quadrature(3).points { e.eval_basis_vec(p, &mut v); for x in &v { assert!(x.is_finite()); } } }
    #[test] fn nd1_curl_finite() { let e = PyraND1; let mut c = vec![0.; 24]; for p in &e.quadrature(3).points { e.eval_curl(p, &mut c); for x in &c { assert!(x.is_finite()); } } }
    #[test] fn nd1_nodal() {
        let e = PyraND1; let ed: [([f64;3],[f64;3]);8]=[([0.,0.,0.],[1.,0.,0.]),([1.,0.,0.],[1.,1.,0.]),([1.,1.,0.],[0.,1.,0.]),([0.,0.,0.],[0.,1.,0.]),([0.,0.,0.],[0.,0.,1.]),([1.,0.,0.],[0.,0.,1.]),([1.,1.,0.],[0.,0.,1.]),([0.,1.,0.],[0.,0.,1.])];
        let(gx,gw)=g4(); let mut mv=vec![0.;24];
        for(j,(a,b))in ed.iter().enumerate(){let d=[b[0]-a[0],b[1]-a[1],b[2]-a[2]];let mut mom=[0.;8];
            for k in 0..4{let(t,w)=(gx[k],gw[k]);let p=[a[0]+t*d[0],a[1]+t*d[1],a[2]+t*d[2]];
                e.eval_basis_vec(&p,&mut mv);for i in 0..8{let tang=mv[i*3]*d[0]+mv[i*3+1]*d[1]+mv[i*3+2]*d[2];mom[i]+=w*tang;}}
            for i in 0..8{let exp=if i==j{1.}else{0.};assert!((mom[i]-exp).abs()<1e-12,"DOF_{j}(Φ_{i})={},exp{exp}",mom[i]);}}
    }
}
