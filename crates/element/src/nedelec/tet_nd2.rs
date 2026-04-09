//! Nedelec-I order-2 element on the reference tetrahedron.
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Space
//! `N₂ = P₁³ ⊕ x^⊥ P̃₁`  (dim = 12 + 8 = 20)
//!
//! Monomial basis (20 functions):
//! ```text
//! P₁³: (1,0,0),(ξ,0,0),(η,0,0),(ζ,0,0),
//!      (0,1,0),(0,ξ,0),(0,η,0),(0,ζ,0),
//!      (0,0,1),(0,0,ξ),(0,0,η),(0,0,ζ)
//! x×P̃₁: ξ×{(0,ζ,-η),(0,ξ,-η)} and permutations... see impl
//! ```
//!
//! # DOF functionals (20 total)
//! 2 tangential moments per edge (6 edges × 2 = 12)
//! plus 2 tangential moments per face (4 faces × 2 directions × 1 moment = 8)
//!
//! Implemented via a Vandermonde matrix V[k][j] = DOF_k(m_j), inverted at startup.

use std::sync::OnceLock;

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Coefficient matrix (cached) ────────────────────────────────────────────

static COEFF: OnceLock<[[f64; 20]; 20]> = OnceLock::new();

/// Evaluate the 20 monomials at (x,y,z).
/// Layout: monos[j*3], monos[j*3+1], monos[j*3+2] = components of m_j.
///
/// P₁³ monomials (j=0..11):
///   j=0: (1,0,0)  j=1: (ξ,0,0)  j=2: (η,0,0)  j=3: (ζ,0,0)
///   j=4: (0,1,0)  j=5: (0,ξ,0)  j=6: (0,η,0)  j=7: (0,ζ,0)
///   j=8: (0,0,1)  j=9: (0,0,ξ)  j=10:(0,0,η)  j=11:(0,0,ζ)
/// x^⊥ P̃₁ monomials (j=12..19):
///   These span the complement: x × homogeneous-linear.
///   curl(x × p) = 3p for p homogeneous linear, giving us the 8 extra dims.
///   Concrete choices:
///   j=12: (-ηξ, ξ², 0)    j=13: (-ζξ, 0, ξ²)
///   j=14: (-η², ηξ, 0)    j=15: (0, -ζη, η²)
///   j=16: (-ζ², 0, ζξ)    j=17: (0, -ζ², ζη)
///   j=18: (-ηζ, ζξ, 0)... adjusted
///   Actually using: x^⊥ P̃₁ = {u(x) : u = (-yp, xp, 0) or (-zp, 0, xp) etc}
///   Let's use the 8 explicit: (from Monk, Table 5.2 or MFEM ND_TetrahedronElement)
///   j=12: (-ξη, ξ², 0)   j=13: (-ξζ, 0, ξ²)
///   j=14: (-η², ξη, 0)   j=15: (0, -ηζ, η²)
///   j=16: (-ζ², 0, ξζ)   j=17: (0, -ζ², ηζ)
///   j=18: (-ηζ, 0, ξη) + adjustment
///   j=19: (0, -ηζ, ξη)?
///   → Use the systematic: x^⊥ = { (-y·m, x·m, 0), (-z·m, 0, x·m), (0,-z·m, y·m) }
///     for m in {ξ, η, ζ} / {1,...} with 8 choices matching dim.
fn eval_monomials(x: f64, y: f64, z: f64, vals: &mut [f64]) {
    // P₁³ (j=0..11)
    // j=0
    vals[0] = 1.0; vals[1] = 0.0; vals[2] = 0.0;
    // j=1
    vals[3] = x;   vals[4] = 0.0; vals[5] = 0.0;
    // j=2
    vals[6] = y;   vals[7] = 0.0; vals[8] = 0.0;
    // j=3
    vals[9] = z;   vals[10] = 0.0; vals[11] = 0.0;
    // j=4
    vals[12] = 0.0; vals[13] = 1.0; vals[14] = 0.0;
    // j=5
    vals[15] = 0.0; vals[16] = x; vals[17] = 0.0;
    // j=6
    vals[18] = 0.0; vals[19] = y; vals[20] = 0.0;
    // j=7
    vals[21] = 0.0; vals[22] = z; vals[23] = 0.0;
    // j=8
    vals[24] = 0.0; vals[25] = 0.0; vals[26] = 1.0;
    // j=9
    vals[27] = 0.0; vals[28] = 0.0; vals[29] = x;
    // j=10
    vals[30] = 0.0; vals[31] = 0.0; vals[32] = y;
    // j=11
    vals[33] = 0.0; vals[34] = 0.0; vals[35] = z;

    // x^⊥ P̃₁ (j=12..19): chosen to span complement of P₁³ in N₂.
    // From the identity x^⊥ P̃₁ = {(-yp,xp,0),(−zp,0,xp),(0,−zp,yp)} for p∈{ξ,η,ζ} mod dups
    // 9 candidates minus 1 linear dep = 8:
    // j=12: (-ξη, ξ², 0)
    vals[36] = -x*y; vals[37] = x*x; vals[38] = 0.0;
    // j=13: (-ζξ, 0, ξ²)
    vals[39] = -z*x; vals[40] = 0.0; vals[41] = x*x;
    // j=14: (-η², ξη, 0)
    vals[42] = -y*y; vals[43] = x*y; vals[44] = 0.0;
    // j=15: (0, -ηζ, η²)
    vals[45] = 0.0; vals[46] = -y*z; vals[47] = y*y;
    // j=16: (-ζ², 0, ζξ)
    vals[48] = -z*z; vals[49] = 0.0; vals[50] = z*x;
    // j=17: (0, -ζ², ηζ)
    vals[51] = 0.0; vals[52] = -z*z; vals[53] = y*z;
    // j=18: (-ηζ, ξζ, 0)
    vals[54] = -y*z; vals[55] = x*z; vals[56] = 0.0;
    // j=19: (0, -ζη, ξη) -- need to check linear independence from j=15,j=18...
    // Use instead: (-ζη, 0, ξη) to avoid linear dependence
    vals[57] = -z*y; vals[58] = 0.0; vals[59] = x*y;
}

/// Curl of each monomial: curl(m)_k = ε_{ijk} ∂m_j/∂x_i summed.
/// curl(m) = (∂m_z/∂y - ∂m_y/∂z, ∂m_x/∂z - ∂m_z/∂x, ∂m_y/∂x - ∂m_x/∂y)
fn eval_monomial_curls(x: f64, y: f64, z: f64, curls: &mut [f64]) {
    // Each curl is 3 components: curls[j*3..j*3+3]
    // j=0: (1,0,0) → curl=(0,0,0)
    curls[0]=0.0; curls[1]=0.0; curls[2]=0.0;
    // j=1: (ξ,0,0) → curl=(0-0, 0-0, 0-1)=(0,0,-... wait:
    // curl(u,v,w) = (∂w/∂y-∂v/∂z, ∂u/∂z-∂w/∂x, ∂v/∂x-∂u/∂y)
    // j=1:(ξ,0,0): (0-0,0-0,0-0)=(0,0,0)
    curls[3]=0.0; curls[4]=0.0; curls[5]=0.0;
    // j=2:(η,0,0): u=η,v=0,w=0 → (0-0,0-0,0-1)=(0,0,-1)
    curls[6]=0.0; curls[7]=0.0; curls[8]=-1.0;
    // j=3:(ζ,0,0): (0-0,1-0,0-0)=(0,1,0)... wait ∂u/∂z=1 where u=ζ? No, u=ζ means u=x₃=z.
    // curl(ζ,0,0): (∂0/∂y-∂0/∂z, ∂ζ/∂z-∂0/∂x, ∂0/∂x-∂ζ/∂y)=(0,1,0)
    curls[9]=0.0; curls[10]=1.0; curls[11]=0.0;
    // j=4:(0,1,0): (0,0,0)
    curls[12]=0.0; curls[13]=0.0; curls[14]=0.0;
    // j=5:(0,ξ,0): u=0,v=ξ,w=0 → (0-0,0-0,1-0)=(0,0,1)
    curls[15]=0.0; curls[16]=0.0; curls[17]=1.0;
    // j=6:(0,η,0): (0-0,0-0,0-0)=(0,0,0) [∂v/∂x=0, ∂u/∂y=0 since v=η=y → ∂v/∂x=0]
    curls[18]=0.0; curls[19]=0.0; curls[20]=0.0;
    // j=7:(0,ζ,0): u=0,v=ζ,w=0 → (0-0,0-0,0-0)=(0,0,0)... ∂v/∂x=0, ∂w/∂z=0 hmm
    // curl: (∂0/∂y-∂ζ/∂z, ∂0/∂z-∂0/∂x, ∂ζ/∂x-∂0/∂y)=(-1,0,0)
    curls[21]=-1.0; curls[22]=0.0; curls[23]=0.0;
    // j=8:(0,0,1): (0,0,0)
    curls[24]=0.0; curls[25]=0.0; curls[26]=0.0;
    // j=9:(0,0,ξ): (0-0, ∂ξ/∂z-... wait: (∂w/∂y-∂v/∂z, ∂u/∂z-∂w/∂x, ∂v/∂x-∂u/∂y)
    // u=0,v=0,w=ξ: (0-0,0-1,0-0)=(0,-1,0)
    curls[27]=0.0; curls[28]=-1.0; curls[29]=0.0;
    // j=10:(0,0,η): u=0,v=0,w=η: (1-0,0-0,0-0)=(1,0,0)
    curls[30]=1.0; curls[31]=0.0; curls[32]=0.0;
    // j=11:(0,0,ζ): u=0,v=0,w=ζ: (0-0,0-0,0-0)=(0,0,0)
    curls[33]=0.0; curls[34]=0.0; curls[35]=0.0;

    // j=12: (-ξη, ξ², 0): u=-xy, v=x², w=0
    // curl: (∂0/∂y-∂x²/∂z, ∂(-xy)/∂z-∂0/∂x, ∂x²/∂x-∂(-xy)/∂y)=(0,0,2x+x)=(0,0,3x)
    curls[36]=0.0; curls[37]=0.0; curls[38]=3.0*x;
    // j=13: (-ζξ, 0, ξ²): u=-zx, v=0, w=x²
    // curl: (∂x²/∂y-∂0/∂z, ∂(-zx)/∂z-∂x²/∂x, ∂0/∂x-∂(-zx)/∂y)=(0,-x-2x,0)=(0,-3x,0)
    curls[39]=0.0; curls[40]=-3.0*x; curls[41]=0.0;
    // j=14: (-η², ξη, 0): u=-y², v=xy, w=0
    // curl: (0-0, 0-0, ∂xy/∂x-∂(-y²)/∂y)=(0,0,y+2y)=(0,0,3y)
    curls[42]=0.0; curls[43]=0.0; curls[44]=3.0*y;
    // j=15: (0, -ηζ, η²): u=0, v=-yz, w=y²
    // curl: (∂y²/∂y-∂(-yz)/∂z, ∂0/∂z-∂y²/∂x, ∂(-yz)/∂x-∂0/∂y)=(2y+y,0,0)=(3y,0,0)
    curls[45]=3.0*y; curls[46]=0.0; curls[47]=0.0;
    // j=16: (-ζ², 0, ζξ): u=-z², v=0, w=zx
    // curl: (∂zx/∂y-∂0/∂z, ∂(-z²)/∂z-∂zx/∂x, ∂0/∂x-∂(-z²)/∂y)=(0,-2z-z,0)=(0,-3z,0)
    curls[48]=0.0; curls[49]=-3.0*z; curls[50]=0.0;
    // j=17: (0, -ζ², ηζ): u=0, v=-z², w=yz
    // curl: (∂yz/∂y-∂(-z²)/∂z, ∂0/∂z-∂yz/∂x, ∂(-z²)/∂x-∂0/∂y)=(z+2z,0,0)=(3z,0,0)
    curls[51]=3.0*z; curls[52]=0.0; curls[53]=0.0;
    // j=18: (-ηζ, ξζ, 0): u=-yz, v=xz, w=0
    // curl: (0-∂xz/∂z, ∂(-yz)/∂z-0, ∂xz/∂x-∂(-yz)/∂y)=(−x, −y, z+z)=(−x,−y,2z)
    // Hmm — this isn't simply 3*monomial. That's ok, just record it.
    curls[54]=-x; curls[55]=-y; curls[56]=2.0*z;
    // j=19: (-ζη, 0, ξη): u=-zy, v=0, w=xy
    // curl: (∂xy/∂y-∂0/∂z, ∂(-zy)/∂z-∂xy/∂x, ∂0/∂x-∂(-zy)/∂y)=(x, -y-y, z)=(x,-2y,z)
    // Wait: ∂(-zy)/∂z = -y; ∂xy/∂x = y → ∂u/∂z-∂w/∂x = -y - y = -2y? Let me redo:
    // curl(u,v,w)=(∂w/∂y-∂v/∂z, ∂u/∂z-∂w/∂x, ∂v/∂x-∂u/∂y)
    // u=-zy,v=0,w=xy: (∂xy/∂y - 0, ∂(-zy)/∂z - ∂xy/∂x, 0 - ∂(-zy)/∂y)=(x, -y-y, z)=(x,-2y,z)
    curls[57]=x; curls[58]=-2.0*y; curls[59]=z;
}

/// Build the 20×20 Vandermonde matrix.
/// DOF functionals:
/// Edges (6 × 2 = 12):
///   DOF 0,1: edge e₀₁ (v₀→v₁), tangent t=(1,0,0), moments ∫₀¹ Φ·t dt and ∫₀¹ Φ·t·t dt
///   DOF 2,3: edge e₀₂ (v₀→v₂), tangent t=(0,1,0)
///   DOF 4,5: edge e₀₃ (v₀→v₃), tangent t=(0,0,1)
///   DOF 6,7: edge e₁₂ (v₁→v₂), tangent t=(-1,1,0)
///   DOF 8,9: edge e₁₃ (v₁→v₃), tangent t=(-1,0,1)
///   DOF 10,11: edge e₂₃ (v₂→v₃), tangent t=(0,-1,1)
/// Faces (4 × 2 = 8): DOF 12–19
///   Face 0 (v₁v₂v₃, ξ+η+ζ=1): two tangential integral moments
///   Face 1 (v₀v₂v₃, ξ=0): ...
///   Face 2 (v₀v₁v₃, η=0): ...
///   Face 3 (v₀v₁v₂, ζ=0): ...
fn build_vandermonde() -> [[f64; 20]; 20] {
    let mut v = [[0.0f64; 20]; 20];

    // Helper: tangential moment ∫₀¹ m(γ(t))·tang dt and ∫₀¹ m(γ(t))·tang·t dt
    // where γ(t) = start + t*(end-start).
    let edge_moments = |row: &mut usize, v: &mut [[f64; 20]; 20],
                        start: [f64;3], end: [f64;3], tang: [f64;3]| {
        // Use 4-point Gauss-Legendre on [0,1]
        let gl = gauss_legendre_01_4();
        let mut m0 = [0.0f64; 20*3];
        for (t, w) in gl.0.iter().zip(gl.1.iter()) {
            let pt = [
                start[0] + t*(end[0]-start[0]),
                start[1] + t*(end[1]-start[1]),
                start[2] + t*(end[2]-start[2]),
            ];
            eval_monomials(pt[0], pt[1], pt[2], &mut m0);
            for j in 0..20 {
                let tdot = m0[j*3]*tang[0] + m0[j*3+1]*tang[1] + m0[j*3+2]*tang[2];
                v[*row][j] += w * tdot;
                v[*row+1][j] += w * tdot * t;
            }
        }
        *row += 2;
    };

    let mut row = 0usize;
    // e₀₁: v₀=(0,0,0) → v₁=(1,0,0), tang=(1,0,0)
    edge_moments(&mut row, &mut v, [0.0,0.0,0.0], [1.0,0.0,0.0], [1.0,0.0,0.0]);
    // e₀₂: v₀=(0,0,0) → v₂=(0,1,0), tang=(0,1,0)
    edge_moments(&mut row, &mut v, [0.0,0.0,0.0], [0.0,1.0,0.0], [0.0,1.0,0.0]);
    // e₀₃: v₀=(0,0,0) → v₃=(0,0,1), tang=(0,0,1)
    edge_moments(&mut row, &mut v, [0.0,0.0,0.0], [0.0,0.0,1.0], [0.0,0.0,1.0]);
    // e₁₂: v₁=(1,0,0) → v₂=(0,1,0), tang=(-1,1,0) (unnormalized)
    edge_moments(&mut row, &mut v, [1.0,0.0,0.0], [0.0,1.0,0.0], [-1.0,1.0,0.0]);
    // e₁₃: v₁=(1,0,0) → v₃=(0,0,1), tang=(-1,0,1)
    edge_moments(&mut row, &mut v, [1.0,0.0,0.0], [0.0,0.0,1.0], [-1.0,0.0,1.0]);
    // e₂₃: v₂=(0,1,0) → v₃=(0,0,1), tang=(0,-1,1)
    edge_moments(&mut row, &mut v, [0.0,1.0,0.0], [0.0,0.0,1.0], [0.0,-1.0,1.0]);
    assert_eq!(row, 12);

    // Face DOFs: interior tangential moments using face quadrature.
    // For each face (triangle), pick 2 linearly independent tangent vectors t₁, t₂.
    // DOF = ∫_face Φ·tₖ dA for k=1,2.
    // Face 0: v₁v₂v₃ (ξ+η+ζ=1), normal=(1,1,1)/√3
    //   Param: (s,t) → v₁ + s*(v₂-v₁) + t*(v₃-v₁) = (1,0,0)+s*(-1,1,0)+t*(-1,0,1)
    //   t₁=(-1,1,0), t₂=(-1,0,1), Jacobian area = |t₁×t₂|/2
    face_moments(&mut row, &mut v,
        [1.0,0.0,0.0], [-1.0,1.0,0.0], [-1.0,0.0,1.0],
        [-1.0,1.0,0.0], [-1.0,0.0,1.0]);
    // Face 1: v₀v₂v₃ (ξ=0), param s*(0,1,0)+t*(0,0,1)
    //   t₁=(0,1,0), t₂=(0,0,1)
    face_moments(&mut row, &mut v,
        [0.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0],
        [0.0,1.0,0.0], [0.0,0.0,1.0]);
    // Face 2: v₀v₁v₃ (η=0), param s*(1,0,0)+t*(0,0,1)
    //   t₁=(1,0,0), t₂=(0,0,1)
    face_moments(&mut row, &mut v,
        [0.0,0.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0],
        [1.0,0.0,0.0], [0.0,0.0,1.0]);
    // Face 3: v₀v₁v₂ (ζ=0), param s*(1,0,0)+t*(0,1,0)
    //   t₁=(1,0,0), t₂=(0,1,0)
    face_moments(&mut row, &mut v,
        [0.0,0.0,0.0], [1.0,0.0,0.0], [0.0,1.0,0.0],
        [1.0,0.0,0.0], [0.0,1.0,0.0]);
    assert_eq!(row, 20);

    v
}

/// 4-point Gauss-Legendre on [0,1].
fn gauss_legendre_01_4() -> ([f64;4], [f64;4]) {
    let sq6_5 = (6.0f64 / 5.0).sqrt();
    let ta = ((3.0 - 2.0 * sq6_5) / 7.0).sqrt();
    let tb = ((3.0 + 2.0 * sq6_5) / 7.0).sqrt();
    let wa = (18.0 + 30.0f64.sqrt()) / 36.0;
    let wb = (18.0 - 30.0f64.sqrt()) / 36.0;
    (
        [0.5*(1.0-tb), 0.5*(1.0-ta), 0.5*(1.0+ta), 0.5*(1.0+tb)],
        [0.5*wb, 0.5*wa, 0.5*wa, 0.5*wb],
    )
}

/// Add 2 face moment DOF rows to the Vandermonde matrix.
/// Face is parameterized as v0 + s*ds + t*dt over the reference triangle (s≥0,t≥0,s+t≤1).
/// Two tangential DOFs per face: ∫_face Φ·tang1 dA and ∫_face Φ·tang2 dA.
fn face_moments(
    row: &mut usize,
    v: &mut [[f64; 20]; 20],
    v0: [f64;3], ds: [f64;3], dt: [f64;3],
    tang1: [f64;3], tang2: [f64;3],
) {
    // Reference triangle quadrature (5-point rule, exact for degree 4)
    let qr = tri_rule(4);
    // Area scaling factor: |ds × dt| (which is |J| for the parametric map)
    let cx = ds[1]*dt[2] - ds[2]*dt[1];
    let cy = ds[2]*dt[0] - ds[0]*dt[2];
    let cz = ds[0]*dt[1] - ds[1]*dt[0];
    let area_factor = (cx*cx + cy*cy + cz*cz).sqrt();

    let mut mono = vec![0.0f64; 20*3];
    for (xi2d, w) in qr.points.iter().zip(qr.weights.iter()) {
        let s = xi2d[0];
        let t = xi2d[1];
        let pt = [
            v0[0] + s*ds[0] + t*dt[0],
            v0[1] + s*ds[1] + t*dt[1],
            v0[2] + s*ds[2] + t*dt[2],
        ];
        eval_monomials(pt[0], pt[1], pt[2], &mut mono);
        for j in 0..20 {
            let mx = mono[j*3]; let my = mono[j*3+1]; let mz = mono[j*3+2];
            let d1 = mx*tang1[0] + my*tang1[1] + mz*tang1[2];
            let d2 = mx*tang2[0] + my*tang2[1] + mz*tang2[2];
            v[*row][j] += w * area_factor * d1;
            v[*row+1][j] += w * area_factor * d2;
        }
    }
    *row += 2;
}

/// Invert a 20×20 matrix using Gauss-Jordan elimination.
fn invert_20x20(a: [[f64; 20]; 20]) -> [[f64; 20]; 20] {
    let n = 20usize;
    let mut m = vec![[0.0f64; 40]; n];
    for i in 0..n {
        for j in 0..n { m[i][j] = a[i][j]; }
        m[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col+1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs(); max_row = row;
            }
        }
        m.swap(col, max_row);
        let pivot = m[col][col];
        assert!(pivot.abs() > 1e-14, "TetND2 Vandermonde matrix is singular (col={col})");
        let inv = 1.0/pivot;
        for j in 0..2*n { m[col][j] *= inv; }
        for row in 0..n {
            if row == col { continue; }
            let f = m[row][col];
            for j in 0..2*n { let d=f*m[col][j]; m[row][j]-=d; }
        }
    }
    let mut r = [[0.0f64; 20]; 20];
    for i in 0..n { for j in 0..n { r[i][j] = m[i][n+j]; } }
    r
}

fn transpose_20x20(a: [[f64; 20]; 20]) -> [[f64; 20]; 20] {
    let mut t = [[0.0f64; 20]; 20];
    for i in 0..20 { for j in 0..20 { t[i][j] = a[j][i]; } }
    t
}

fn coeff() -> &'static [[f64; 20]; 20] {
    COEFF.get_or_init(|| transpose_20x20(invert_20x20(build_vandermonde())))
}

// ─── TetND2 ─────────────────────────────────────────────────────────────────

/// Nédélec first-kind H(curl) element on the reference tetrahedron — 20 DOFs, order 2.
pub struct TetND2;

impl VectorReferenceElement for TetND2 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize { 20 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut mono = vec![0.0f64; 20*3];
        eval_monomials(x, y, z, &mut mono);
        for i in 0..20 {
            let mut vx = 0.0; let mut vy = 0.0; let mut vz = 0.0;
            for j in 0..20 {
                vx += c[i][j] * mono[j*3];
                vy += c[i][j] * mono[j*3+1];
                vz += c[i][j] * mono[j*3+2];
            }
            values[i*3]   = vx;
            values[i*3+1] = vy;
            values[i*3+2] = vz;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut mc = vec![0.0f64; 20*3];
        eval_monomial_curls(x, y, z, &mut mc);
        for i in 0..20 {
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for j in 0..20 {
                cx += c[i][j] * mc[j*3];
                cy += c[i][j] * mc[j*3+1];
                cz += c[i][j] * mc[j*3+2];
            }
            curl_vals[i*3]   = cx;
            curl_vals[i*3+1] = cy;
            curl_vals[i*3+2] = cz;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        use crate::quadrature::tet_rule;
        tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut coords = Vec::with_capacity(20);
        // Edge DOFs: 1/3 and 2/3 along each edge
        let edges: [([f64;3],[f64;3]); 6] = [
            ([0.,0.,0.],[1.,0.,0.]), ([0.,0.,0.],[0.,1.,0.]),
            ([0.,0.,0.],[0.,0.,1.]), ([1.,0.,0.],[0.,1.,0.]),
            ([1.,0.,0.],[0.,0.,1.]), ([0.,1.,0.],[0.,0.,1.]),
        ];
        for (a, b) in &edges {
            coords.push(vec![a[0]+1./3.*(b[0]-a[0]), a[1]+1./3.*(b[1]-a[1]), a[2]+1./3.*(b[2]-a[2])]);
            coords.push(vec![a[0]+2./3.*(b[0]-a[0]), a[1]+2./3.*(b[1]-a[1]), a[2]+2./3.*(b[2]-a[2])]);
        }
        // Face DOFs: centroids of each face
        let faces: [[f64;3]; 4] = [
            [1./3.,1./3.,1./3.], [0.,1./3.,1./3.],
            [1./3.,0.,1./3.], [1./3.,1./3.,0.],
        ];
        for f in &faces {
            coords.push(vec![f[0]*0.5, f[1]*0.5, f[2]*0.5]);
            coords.push(vec![f[0]*0.6, f[1]*0.2, f[2]*0.2]);
        }
        coords
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_nd2_coeff_computed() {
        let c = coeff();
        let diag_sum: f64 = (0..20).map(|i| c[i][i].abs()).sum();
        assert!(diag_sum > 0.1, "coefficient matrix diagonal is small");
    }

    #[test]
    fn tet_nd2_basis_finite() {
        let elem = TetND2;
        let mut vals = vec![0.0; 20*3];
        for xi in &[
            vec![0.0,0.0,0.0], vec![1.0,0.0,0.0], vec![0.0,1.0,0.0], vec![0.0,0.0,1.0],
            vec![0.25,0.25,0.25], vec![1./3.,1./3.,0.0],
        ] {
            elem.eval_basis_vec(xi, &mut vals);
            for &v in &vals {
                assert!(v.is_finite(), "non-finite at {xi:?}: {v}");
            }
        }
    }

    #[test]
    fn tet_nd2_curl_finite() {
        let elem = TetND2;
        let mut curl = vec![0.0; 20*3];
        let qr = elem.quadrature(4);
        for xi in &qr.points {
            elem.eval_curl(xi, &mut curl);
            for &v in &curl {
                assert!(v.is_finite(), "non-finite curl at {xi:?}: {v}");
            }
        }
    }

    /// Nodal basis: DOF_k(Φ_i) ≈ δ_{ki}  (tested via edge moment integrals only)
    #[test]
    fn tet_nd2_edge_dofs_nodal() {
        let elem = TetND2;
        let gl = gauss_legendre_01_4();
        let mut vals = vec![0.0f64; 20*3];

        // DOF 0,1: edge e₀₁, tang=(1,0,0)
        let mut mom0 = [0.0f64; 20];
        let mut mom1 = [0.0f64; 20];
        for k in 0..4 {
            let (t, w) = (gl.0[k], gl.1[k]);
            elem.eval_basis_vec(&[t, 0.0, 0.0], &mut vals);
            for i in 0..20 {
                let tang = vals[i*3]; // x-component
                mom0[i] += w * tang;
                mom1[i] += w * tang * t;
            }
        }
        // DOF_0 should be delta_{0,i}, DOF_1 should be delta_{1,i}
        for i in 0..20 {
            let exp0 = if i == 0 { 1.0 } else { 0.0 };
            let exp1 = if i == 1 { 1.0 } else { 0.0 };
            assert!((mom0[i]-exp0).abs() < 1e-9, "DOF_0(Phi_{i})={}, expected {exp0}", mom0[i]);
            assert!((mom1[i]-exp1).abs() < 1e-9, "DOF_1(Phi_{i})={}, expected {exp1}", mom1[i]);
        }
    }
}
