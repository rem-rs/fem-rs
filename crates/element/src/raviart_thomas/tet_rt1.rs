//! Raviart-Thomas RT1 element on the reference tetrahedron.
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Space
//! `RT₁ = P₁³ ⊕ x P̃₁`  (dim = 12 + 3 = 15)
//!
//! Monomial basis (15 functions):
//! - P₁³ (j=0..11): (1,0,0),(ξ,0,0),(η,0,0),(ζ,0,0),(0,1,0),...,(0,0,ζ)
//! - x P̃₁ (j=12..14): (ξ²,ξη,ξζ),(ξη,η²,ηζ),(ξζ,ηζ,ζ²)
//!
//! # DOF functionals (15 total)
//! 3 normal-flux moments per face (4 faces × 3 = 12) + 3 interior moments.
//!
//! Face ordering:
//! - Face 0: v₁v₂v₃ (ξ+η+ζ=1), outward normal (1,1,1)/√3
//! - Face 1: v₀v₂v₃ (ξ=0),     outward normal (−1,0,0)
//! - Face 2: v₀v₁v₃ (η=0),     outward normal (0,−1,0)
//! - Face 3: v₀v₁v₂ (ζ=0),     outward normal (0,0,−1)

use std::sync::OnceLock;

use crate::quadrature::{tri_rule, tet_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

static COEFF: OnceLock<[[f64; 15]; 15]> = OnceLock::new();

/// Evaluate the 15 RT1 monomials at (x,y,z).
/// vals layout: vals[j*3], vals[j*3+1], vals[j*3+2]
fn eval_monomials(x: f64, y: f64, z: f64, vals: &mut [f64]) {
    // P₁³ (j=0..11)
    vals[0]=1.; vals[1]=0.; vals[2]=0.;   // (1,0,0)
    vals[3]=x;  vals[4]=0.; vals[5]=0.;   // (ξ,0,0)
    vals[6]=y;  vals[7]=0.; vals[8]=0.;   // (η,0,0)
    vals[9]=z;  vals[10]=0.;vals[11]=0.;  // (ζ,0,0)
    vals[12]=0.;vals[13]=1.;vals[14]=0.;  // (0,1,0)
    vals[15]=0.;vals[16]=x; vals[17]=0.;  // (0,ξ,0)
    vals[18]=0.;vals[19]=y; vals[20]=0.;  // (0,η,0)
    vals[21]=0.;vals[22]=z; vals[23]=0.;  // (0,ζ,0)
    vals[24]=0.;vals[25]=0.;vals[26]=1.;  // (0,0,1)
    vals[27]=0.;vals[28]=0.;vals[29]=x;   // (0,0,ξ)
    vals[30]=0.;vals[31]=0.;vals[32]=y;   // (0,0,η)
    vals[33]=0.;vals[34]=0.;vals[35]=z;   // (0,0,ζ)
    // x P̃₁ (j=12..14): x(x,y,z)^T scaled by homogeneous linear
    vals[36]=x*x; vals[37]=x*y; vals[38]=x*z;  // (ξ²,ξη,ξζ)
    vals[39]=x*y; vals[40]=y*y; vals[41]=y*z;  // (ξη,η²,ηζ)
    vals[42]=x*z; vals[43]=y*z; vals[44]=z*z;  // (ξζ,ηζ,ζ²)
}

/// Divergence of each monomial.
fn eval_monomial_divs(x: f64, y: f64, z: f64, divs: &mut [f64]) {
    divs[0]=0.; divs[1]=1.; divs[2]=0.; divs[3]=0.; // (1,0,0),(ξ,0,0),(η,0,0),(ζ,0,0)
    divs[4]=0.; divs[5]=0.; divs[6]=1.; divs[7]=0.; // (0,1,0),...,(0,ζ,0)
    divs[8]=0.; divs[9]=0.; divs[10]=0.; divs[11]=1.; // (0,0,1),...,(0,0,ζ)
    // div(ξ²,ξη,ξζ) = 2ξ + ξ + ξ = 4ξ? wait: ∂ξ²/∂ξ + ∂ξη/∂η + ∂ξζ/∂ζ = 2ξ + ξ + ξ = 4ξ
    divs[12] = 4.0*x;
    // div(ξη,η²,ηζ) = η + 2η + η = 4η
    divs[13] = 4.0*y;
    // div(ξζ,ηζ,ζ²) = ζ + ζ + 2ζ = 4ζ
    divs[14] = 4.0*z;
}

/// Build 15×15 Vandermonde matrix.
fn build_vandermonde() -> [[f64; 15]; 15] {
    let mut v = [[0.0f64; 15]; 15];

    // Faces:
    // Face 0: v₁v₂v₃ (ξ+η+ζ=1), parameterized as v₁+s*(v₂-v₁)+t*(v₃-v₁) = (1-s-t,s,t)
    //   outward normal = (1,1,1), unnormalized; area factor from |ds×dt|
    //   ds = v₂-v₁ = (-1,1,0), dt = v₃-v₁ = (-1,0,1)
    //   ds×dt = (1·1-0·0, 0·(-1)-(-1)·1, (-1)·0-1·(-1)) = (1,1,1), |..| = √3
    //   normal (unnorm) = (1,1,1), dot with (1,1,1) = 3 flux directions
    face_moments(&mut v, 0, [1.,0.,0.], [-1.,1.,0.], [-1.,0.,1.], [1.,1.,1.]);
    // Face 1: v₀v₂v₃ (ξ=0), param s*(v₂-v₀)+t*(v₃-v₀) = (0,s,t)
    //   ds=(0,1,0), dt=(0,0,1), ds×dt=(1,0,0), normal=(-1,0,0)
    face_moments(&mut v, 3, [0.,0.,0.], [0.,1.,0.], [0.,0.,1.], [-1.,0.,0.]);
    // Face 2: v₀v₁v₃ (η=0), param s*(v₁-v₀)+t*(v₃-v₀) = (s,0,t)
    //   ds=(1,0,0), dt=(0,0,1), ds×dt=(0,-1,0), normal=(0,-1,0)
    face_moments(&mut v, 6, [0.,0.,0.], [1.,0.,0.], [0.,0.,1.], [0.,-1.,0.]);
    // Face 3: v₀v₁v₂ (ζ=0), param s*(v₁-v₀)+t*(v₂-v₀) = (s,t,0)
    //   ds=(1,0,0), dt=(0,1,0), ds×dt=(0,0,1), normal=(0,0,-1)
    face_moments(&mut v, 9, [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]);

    // Interior DOFs 12,13,14: ∫_T Φ_x dV, ∫_T Φ_y dV, ∫_T Φ_z dV
    let qr = tet_rule(4);
    let mut mono = vec![0.0f64; 15*3];
    for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
        eval_monomials(xi[0], xi[1], xi[2], &mut mono);
        for j in 0..15 {
            v[12][j] += w * mono[j*3];
            v[13][j] += w * mono[j*3+1];
            v[14][j] += w * mono[j*3+2];
        }
    }

    v
}

/// Add 3 face-DOF rows starting at `row_start`.
/// Face parameterized as v0 + s*ds + t*dt.
/// Normal (unnormalized): normal_dir. Three DOFs:
///   DOF_0: ∫_face Φ·n̂ dσ        (one moment per face is not enough for RT1...)
/// Wait — RT1 has 3 DOFs per face, but these are normal-flux moments against 3 test functions.
/// For a triangle face, the 3 DOF functionals are:
///   ∫_face (Φ·n̂) dσ
///   ∫_face (Φ·n̂) s dσ
///   ∫_face (Φ·n̂) t dσ
/// where (s,t) are face parameters.
fn face_moments(
    v: &mut [[f64; 15]; 15],
    row_start: usize,
    v0: [f64;3], ds: [f64;3], dt: [f64;3],
    normal: [f64;3], // unnormalized outward normal (dot with flux gives sign)
) {
    // |n| is the area scale (Jacobian of the 2D parametrization)
    let n_len = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
    // Unit normal
    let n_unit = [normal[0]/n_len, normal[1]/n_len, normal[2]/n_len];
    // Area Jacobian: the cross product magnitude of ds and dt
    let cx = ds[1]*dt[2]-ds[2]*dt[1];
    let cy = ds[2]*dt[0]-ds[0]*dt[2];
    let cz = ds[0]*dt[1]-ds[1]*dt[0];
    let jac_area = (cx*cx + cy*cy + cz*cz).sqrt();

    // Use triangle quadrature accurate for degree 4
    let qr = tri_rule(4);
    let mut mono = vec![0.0f64; 15*3];

    for (xi2d, w) in qr.points.iter().zip(qr.weights.iter()) {
        let s = xi2d[0]; let t = xi2d[1];
        let pt = [
            v0[0]+s*ds[0]+t*dt[0],
            v0[1]+s*ds[1]+t*dt[1],
            v0[2]+s*ds[2]+t*dt[2],
        ];
        eval_monomials(pt[0], pt[1], pt[2], &mut mono);
        for j in 0..15 {
            let nflux = mono[j*3]*n_unit[0] + mono[j*3+1]*n_unit[1] + mono[j*3+2]*n_unit[2];
            let d_sigma = w * jac_area; // area-weighted
            v[row_start][j] += d_sigma * nflux;
            v[row_start+1][j] += d_sigma * nflux * s;
            v[row_start+2][j] += d_sigma * nflux * t;
        }
    }
}

fn invert_15x15(a: [[f64; 15]; 15]) -> [[f64; 15]; 15] {
    let n = 15usize;
    let mut m = vec![[0.0f64; 30]; n];
    for i in 0..n { for j in 0..n { m[i][j] = a[i][j]; } m[i][n+i] = 1.0; }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col+1)..n { if m[row][col].abs() > max_val { max_val=m[row][col].abs(); max_row=row; } }
        m.swap(col, max_row);
        let inv = 1.0/m[col][col];
        assert!(inv.is_finite(), "TetRT1 Vandermonde singular (col={col})");
        for j in 0..2*n { m[col][j] *= inv; }
        for row in 0..n {
            if row==col { continue; }
            let f = m[row][col];
            for j in 0..2*n { let d=f*m[col][j]; m[row][j]-=d; }
        }
    }
    let mut r = [[0.0f64; 15]; 15];
    for i in 0..n { for j in 0..n { r[i][j] = m[i][n+j]; } }
    r
}

fn transpose_15x15(a: [[f64; 15]; 15]) -> [[f64; 15]; 15] {
    let mut t = [[0.0f64; 15]; 15];
    for i in 0..15 { for j in 0..15 { t[i][j] = a[j][i]; } }
    t
}

fn coeff() -> &'static [[f64; 15]; 15] {
    COEFF.get_or_init(|| transpose_15x15(invert_15x15(build_vandermonde())))
}

// ─── TetRT1 ─────────────────────────────────────────────────────────────────

/// Raviart-Thomas RT1 H(div) element on the reference tetrahedron — 15 DOFs, order 1.
pub struct TetRT1;

impl VectorReferenceElement for TetRT1 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize { 15 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut mono = vec![0.0f64; 15*3];
        eval_monomials(x, y, z, &mut mono);
        for i in 0..15 {
            let mut vx=0.; let mut vy=0.; let mut vz=0.;
            for j in 0..15 { vx+=c[i][j]*mono[j*3]; vy+=c[i][j]*mono[j*3+1]; vz+=c[i][j]*mono[j*3+2]; }
            values[i*3]=vx; values[i*3+1]=vy; values[i*3+2]=vz;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut md = vec![0.0f64; 15];
        eval_monomial_divs(x, y, z, &mut md);
        for i in 0..15 {
            let mut s=0.;
            for j in 0..15 { s += c[i][j]*md[j]; }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut c = Vec::with_capacity(15);
        // Face DOFs: 3 per face, at Gauss points on each face triangle
        // Face 0: v₁v₂v₃
        c.push(vec![2./3.,1./6.,1./6.]); c.push(vec![1./6.,2./3.,1./6.]); c.push(vec![1./6.,1./6.,2./3.]);
        // Face 1: v₀v₂v₃ (ξ=0)
        c.push(vec![0.,2./3.,1./6.]); c.push(vec![0.,1./6.,2./3.]); c.push(vec![0.,1./6.,1./6.]);
        // Face 2: v₀v₁v₃ (η=0)
        c.push(vec![2./3.,0.,1./6.]); c.push(vec![1./6.,0.,2./3.]); c.push(vec![1./6.,0.,1./6.]);
        // Face 3: v₀v₁v₂ (ζ=0)
        c.push(vec![2./3.,1./6.,0.]); c.push(vec![1./6.,2./3.,0.]); c.push(vec![1./6.,1./6.,0.]);
        // Interior
        c.push(vec![0.25,0.25,0.25]); c.push(vec![0.5,0.1,0.1]); c.push(vec![0.1,0.5,0.1]);
        c
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_rt1_coeff_computed() {
        let c = coeff();
        let diag: f64 = (0..15).map(|i| c[i][i].abs()).sum();
        assert!(diag > 0.1, "diagonal sum is {diag}");
    }

    #[test]
    fn tet_rt1_basis_finite() {
        let elem = TetRT1;
        let mut v = vec![0.0; 15*3];
        for xi in &[
            vec![0.,0.,0.], vec![1.,0.,0.], vec![0.,1.,0.], vec![0.,0.,1.],
            vec![0.25,0.25,0.25],
        ] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v { assert!(val.is_finite(), "non-finite at {xi:?}: {val}"); }
        }
    }

    #[test]
    fn tet_rt1_div_finite() {
        let elem = TetRT1;
        let mut div = vec![0.0; 15];
        let qr = elem.quadrature(3);
        for xi in &qr.points {
            elem.eval_div(xi, &mut div);
            for &d in &div { assert!(d.is_finite()); }
        }
    }

    /// Face 0 normal-flux DOFs (rows 0-2) should be identity on Φ_0..Φ_2 and 0 elsewhere.
    #[test]
    fn tet_rt1_face0_dofs_nodal() {
        let elem = TetRT1;
        let qr = tri_rule(5);
        // Face 0: param (1-s-t, s, t), normal (1,1,1)/√3, Jac = √3
        let n_unit = [1./3f64.sqrt(), 1./3f64.sqrt(), 1./3f64.sqrt()];
        let jac_area = 3f64.sqrt() * 0.5; // area of the face triangle * jac

        // Actually: |ds × dt| where ds=(-1,1,0), dt=(-1,0,1):
        // cross = (1·1-0·0, 0·(-1)-(-1)·1, (-1)·0-1·(-1)) = (1,1,1), |(1,1,1)|=√3
        // Quadrature weights in tri_rule sum to 0.5 (area of ref triangle)
        // So area element dσ = |cross| × w = √3 × w
        let j = 3f64.sqrt();

        let mut vals = vec![0.0f64; 15*3];
        let mut mom = [[0.0f64; 15]; 3]; // mom[dof_idx][basis_idx]

        for (xi2d, w) in qr.points.iter().zip(qr.weights.iter()) {
            let (s, t) = (xi2d[0], xi2d[1]);
            let pt = [1.-s-t, s, t];
            elem.eval_basis_vec(&pt, &mut vals);
            for i in 0..15 {
                let nf = vals[i*3]*n_unit[0] + vals[i*3+1]*n_unit[1] + vals[i*3+2]*n_unit[2];
                let ds = w * j;
                mom[0][i] += ds * nf;
                mom[1][i] += ds * nf * s;
                mom[2][i] += ds * nf * t;
            }
        }
        for (d, m) in mom.iter().enumerate() {
            for i in 0..15 {
                let exp = if i == d { 1.0 } else { 0.0 };
                assert!((m[i]-exp).abs() < 1e-9,
                    "Face0 DOF_{d}(Phi_{i}) = {}, expected {exp}", m[i]);
            }
        }
        let _ = jac_area;
    }
}
