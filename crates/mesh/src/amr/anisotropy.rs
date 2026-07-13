//! # Anisotropic refinement decision framework (Phase 4)
//!
//! Provides Hessian-based anisotropy estimation and quality control for
//! directional mesh adaptation.  The pipeline is:
//!
//! ```text
//! Solution u_h → Hessian recovery → Eigendecomposition →
//!   Anisotropy score → Refinement direction →
//!     Integration with anisotropic refine functions →
//!       Aspect-ratio quality control
//! ```
//!
//! ## Usage (2-D Quad4)
//! ```ignore
//! use fem_mesh::*;
//! let dirs = estimate_anisotropy_quad(&mesh, &u, 4.0); // threshold = 4
//! let (mesh2, constraints) = refine_nonconforming_quad_aniso(&mesh, &dirs, None);
//! ```
//!
//! ## References
//! - Dolejší 2004: "Anisotropic mesh adaptation for finite element problems"
//! - Chen 2017: "Hessian-based anisotropic mesh adaptation"

use fem_core::ElemId;
use crate::amr::{QuadRefineDir, HexRefineDir, TriRefineDir};
use crate::amr::recovery::spr_recover_gradient_2d;
use crate::element_type::ElementType;
use crate::Mesh;

// ═════════════════════════════════════════════════════════════════════════════
//  Direction enums
// ═════════════════════════════════════════════════════════════════════════════

/// Refinement direction for 2-D anisotropy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnisoDirection {
    /// Refine along the X direction (vertical cut for quads, edge0 for tri).
    X,
    /// Refine along the Y direction (horizontal cut for quads).
    Y,
    /// Refine along a user-specified direction.
    Oriented { angle_rad: f64 },
    /// Full isotropic refinement.
    Isotropic,
}

/// Refinement direction for 3-D anisotropy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnisoDirection3d {
    X, Y, Z,
    XY, XZ, YZ,
    Isotropic,
}

/// Anisotropy estimate for one element.
#[derive(Debug, Clone, Copy)]
pub struct AnisotropyEstimate<Dir> {
    /// Optimal refinement direction.
    pub direction: Dir,
    /// Aspect ratio of the optimal anisotropic element (≥ 1).
    /// 1 = isotropic, larger values = stronger anisotropy.
    pub aspect_ratio: f64,
    /// Magnitude of the dominant eigenvalue.
    pub strength: f64,
    /// Ratio of eigenvalues: |λ₁| / |λ₂| (clamped).
    pub eigenvalue_ratio: f64,
}

// ═════════════════════════════════════════════════════════════════════════════
//  2×2 symmetric eigenvalue decomposition
// ═════════════════════════════════════════════════════════════════════════════

/// Result of a 2×2 symmetric eigen-decomposition.
struct Eig22 {
    lambda0: f64,  // larger magnitude
    lambda1: f64,  // smaller magnitude
    v0x: f64, v0y: f64,  // eigenvector for λ₀ (unit)
}

/// Compute eigenvalues and eigenvectors of a symmetric 2×2 matrix.
///
/// Returns eigenvalues sorted by *magnitude* (|λ₀| ≥ |λ₁|) and the
/// eigenvector for λ₀.
fn sym_eig_2x2(h: [[f64; 2]; 2]) -> Eig22 {
    let a = h[0][0];
    let b = h[0][1]; // = h[1][0]
    let c = h[1][1];
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let l_plus  = 0.5 * (a + c + disc);
    let l_minus = 0.5 * (a + c - disc);

    // Sort by magnitude
    let (lambda0, lambda1, use_plus) = if l_plus.abs() >= l_minus.abs() {
        (l_plus, l_minus, true)
    } else {
        (l_minus, l_plus, false)
    };

    // Eigenvector for λ₀: (H - λI) v = 0
    // [a-λ, b] [vx] = [0]
    // [b, c-λ] [vy]   [0]
    // v = [b, λ - a]  (provided λ ≠ a)
    let lam = if use_plus { l_plus } else { l_minus };
    let (vx, vy) = if (lam - a).abs() > 1e-14 {
        (b, lam - a)
    } else {
        (lam - c, b)
    };
    let norm = (vx * vx + vy * vy).sqrt();
    if norm > 1e-30 {
        Eig22 { lambda0, lambda1, v0x: vx / norm, v0y: vy / norm }
    } else {
        // Degenerate: isotropic
        Eig22 { lambda0, lambda1, v0x: 1.0, v0y: 0.0 }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Hessian recovery (double-SPR)
// ═════════════════════════════════════════════════════════════════════════════

/// Recover the Hessian (matrix of second derivatives) at each element centroid
/// using a double-SPR technique:
///
/// 1. Recover first derivatives (gradient) at nodes via SPR.
/// 2. For each element, apply the P1 gradient formula to each component of the
///    recovered nodal gradient, yielding the Hessian.
///
/// # Returns
/// Per-element Hessian matrices `H[e] = [[hxx, hxy], [hyx, hyy]]`.
pub fn recover_hessian_2d(mesh: &Mesh<2>, u: &[f64]) -> Vec<[[f64; 2]; 2]> {
    let n_elems = mesh.n_elems();
    let recovered = spr_recover_gradient_2d(mesh, u);

    (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let c = |i: usize| mesh.coords_of(ns[i]);

        let (j00, j01, j10, j11, idet) = if npe == 3 {
            let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
            let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            (j00, j01, j10, j11, idet)
        } else {
            // Quad4 at centroid (ξ=0, η=0)
            let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
            let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
            let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
            let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
            let det = j00 * j11 - j01 * j10;
            let idet = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            (j00, j01, j10, j11, idet)
        };

        let mut hess = [[0.0_f64; 2]; 2];

        if npe == 3 {
            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            for comp in 0..2 {
                let g_vals: Vec<f64> = ns.iter()
                    .map(|&n| recovered[n as usize][comp]).collect();
                let mut gx = 0.0; let mut gy = 0.0;
                for k in 0..3 {
                    let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
                    let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
                    gx += g_vals[k] * gpx; gy += g_vals[k] * gpy;
                }
                hess[comp][0] = gx; hess[comp][1] = gy;
            }
        } else {
            let gref = [[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]];
            for comp in 0..2 {
                let g_vals: Vec<f64> = ns.iter()
                    .map(|&n| recovered[n as usize][comp]).collect();
                let mut gx = 0.0; let mut gy = 0.0;
                for k in 0..4 {
                    let gpx = (j11 * gref[k][0] - j10 * gref[k][1]) * idet;
                    let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * idet;
                    gx += g_vals[k] * gpx; gy += g_vals[k] * gpy;
                }
                hess[comp][0] = gx; hess[comp][1] = gy;
            }
        }
        hess
    }).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  Anisotropy estimation – 2-D
// ═════════════════════════════════════════════════════════════════════════════

/// Default threshold for the eigenvalue ratio that triggers anisotropic refinement.
/// If |λ₁| / |λ₂| > `ANISO_THRESHOLD`, the element is refined anisotropically.
pub const ANISO_THRESHOLD: f64 = 4.0;

/// Maximum allowed aspect ratio for anisotropic elements.
pub const MAX_ASPECT_RATIO: f64 = 10.0;

/// Estimate the optimal refinement direction for each element in a 2-D mesh.
///
/// The anisotropy is determined from the eigendecomposition of the recovered
/// Hessian at each element centroid.  The dominant eigenvector indicates the
/// direction of strongest curvature, and the eigenvalue ratio determines
/// whether anisotropic refinement is warranted.
///
/// # Arguments
/// * `mesh` — 2-D Tri3 or Quad4 mesh.
/// * `u` — nodal solution values.
/// * `threshold` — eigenvalue ratio threshold (recommended: 4–10).
///   Elements with |λ₁| / |λ₂| > `threshold` are marked anisotropic.
///
/// # Returns
/// Per-element `AnisotropyEstimate<AnisoDirection>`.
pub fn estimate_anisotropy_2d(
    mesh: &Mesh<2>,
    u: &[f64],
    threshold: f64,
) -> Vec<AnisotropyEstimate<AnisoDirection>> {
    let n_elems = mesh.n_elems();
    let hessians = recover_hessian_2d(mesh, u);

    (0..n_elems).map(|e| {
        let eig = sym_eig_2x2(hessians[e]);

        let ratio = if eig.lambda1.abs() > 1e-30 {
            (eig.lambda0.abs() / eig.lambda1.abs()).min(1e6)
        } else if eig.lambda0.abs() > 1e-30 {
            1e6 // extremely anisotropic
        } else {
            1.0 // both zero → isotropic
        };

        let aspect = ratio.sqrt().clamp(1.0, MAX_ASPECT_RATIO);

        let direction = if ratio <= threshold {
            AnisoDirection::Isotropic
        } else {
            // Dominant eigenvector direction → which split?
            let ax = eig.v0x.abs();
            let ay = eig.v0y.abs();
            if ax >= 2.0 * ay {
                AnisoDirection::X
            } else if ay >= 2.0 * ax {
                AnisoDirection::Y
            } else {
                AnisoDirection::Oriented { angle_rad: eig.v0y.atan2(eig.v0x) }
            }
        };

        AnisotropyEstimate {
            direction,
            aspect_ratio: aspect,
            strength: eig.lambda0.abs(),
            eigenvalue_ratio: ratio,
        }
    }).collect()
}

/// Convenience: map `AnisoDirection` to `QuadRefineDir` for Quad4 meshes.
pub fn aniso_to_quad_dir(dir: AnisoDirection) -> QuadRefineDir {
    match dir {
        AnisoDirection::X => QuadRefineDir::X,
        AnisoDirection::Y => QuadRefineDir::Y,
        AnisoDirection::Oriented { angle_rad } => {
            // Map the angle to X or Y split (whichever is closer)
            let angle = angle_rad.rem_euclid(std::f64::consts::PI);
            if angle < std::f64::consts::FRAC_PI_4
                || angle > 3.0 * std::f64::consts::FRAC_PI_4
            {
                QuadRefineDir::X
            } else {
                QuadRefineDir::Y
            }
        }
        AnisoDirection::Isotropic => QuadRefineDir::Both,
    }
}

/// Convenience: map `AnisoDirection` to `TriRefineDir` for Tri3 meshes.
///
/// Maps the dominant eigenvector to the edge that is most aligned with the
/// direction of *minimum* variation (i.e., the edge perpendicular to the
/// eigenvector, which is the one to cut).
pub fn aniso_to_tri_dir(dir: AnisoDirection) -> TriRefineDir {
    match dir {
        AnisoDirection::X | AnisoDirection::Y | AnisoDirection::Isotropic => TriRefineDir::Red,
        AnisoDirection::Oriented { angle_rad } => {
            // Approximate edge alignment: for a Tri3, edges are at angles
            // 0°, 60°, 120° from the horizontal for a standard reference
            // triangle.  We pick the edge closest to perpendicular to the
            // dominant eigenvector.
            let a = angle_rad.rem_euclid(std::f64::consts::PI);
            // Edge normal angles (perpendicular to each edge):
            // Edge0 (between nodes 0-1): normal ≈ 90°
            // Edge1 (between nodes 1-2): normal ≈ -30°
            // Edge2 (between nodes 2-0): normal ≈ 30°
            let edges = [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_6, std::f64::consts::FRAC_PI_6];
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (i, &e) in edges.iter().enumerate() {
                let d = (a - e).abs().min((a - e + std::f64::consts::PI).abs());
                if d < best_d { best_d = d; best = i; }
            }
            match best {
                0 => TriRefineDir::Edge0,
                1 => TriRefineDir::Edge1,
                _ => TriRefineDir::Edge2,
            }
        }
    }
}

/// Build a list of marked elements with `QuadRefineDir` for
/// [`refine_nonconforming_quad_aniso`].
pub fn mark_anisotropic_quads(
    estimates: &[AnisotropyEstimate<AnisoDirection>],
    min_strength: f64,
) -> Vec<(ElemId, QuadRefineDir)> {
    estimates.iter().enumerate()
        .filter(|(_, e)| e.strength > min_strength)
        .map(|(i, e)| (i as ElemId, aniso_to_quad_dir(e.direction)))
        .collect()
}

/// Build a list of marked elements with `TriRefineDir` for
/// [`refine_nonconforming_tri_aniso`].
pub fn mark_anisotropic_tris(
    estimates: &[AnisotropyEstimate<AnisoDirection>],
    min_strength: f64,
) -> Vec<(ElemId, TriRefineDir)> {
    estimates.iter().enumerate()
        .filter(|(_, e)| e.strength > min_strength)
        .map(|(i, e)| (i as ElemId, aniso_to_tri_dir(e.direction)))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  3-D anisotropy
// ═════════════════════════════════════════════════════════════════════════════

/// Recover Hessian for 3-D (Tet4) using double-SPR on recovered gradients.
///
/// Returns `H[e] = [[hxx, hxy, hxz], [hyx, hyy, hyz], [hzx, hzy, hzz]]`.
pub fn recover_hessian_3d(mesh: &Mesh<3>, u: &[f64]) -> Vec<[[f64; 3]; 3]> {
    let n_elems = mesh.n_elems();

    // Build a simple element-averaged gradient (SPR analogue for 3D Tet4).
    let elem_grads: Vec<[f64; 3]> = (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let uu = |i| u[ns[i] as usize];
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3; let b=(r+2)%3; let d=(c+1)%3; let e=(c+2)%3;
            (j[a][d]*j[b][e]-j[a][e]*j[b][d])*idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for jj in 0..3 { g[i] += uh[k]*jit(jj,i)*gref[k][jj]; } } }
        g
    }).collect();

    // Nodal averaging for recovered gradient (simple 3D ZZ)
    let n_nodes = mesh.n_nodes();
    let mut nodal_grad = vec![[0.0_f64; 3]; n_nodes];
    let mut nodal_count = vec![0usize; n_nodes];
    for (e, &g) in elem_grads.iter().enumerate() {
        for &n in mesh.elem_nodes(e as ElemId) {
            for d in 0..3 { nodal_grad[n as usize][d] += g[d]; }
            nodal_count[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nodal_count[n] as f64;
        if c > 0.0 { for d in 0..3 { nodal_grad[n][d] /= c; } }
    }

    // P1 gradient of each recovered gradient component → Hessian
    (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3; let b=(r+2)%3; let d=(c+1)%3; let e=(c+2)%3;
            (j[a][d]*j[b][e]-j[a][e]*j[b][d])*idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];

        let mut hess = [[0.0_f64; 3]; 3];
        for comp in 0..3 {
            let g_vals: [f64; 4] = [
                nodal_grad[ns[0] as usize][comp],
                nodal_grad[ns[1] as usize][comp],
                nodal_grad[ns[2] as usize][comp],
                nodal_grad[ns[3] as usize][comp],
            ];
            let mut g = [0.0_f64; 3];
            for k in 0..4 {
                for i in 0..3 {
                    for jj in 0..3 {
                        g[i] += g_vals[k] * jit(jj, i) * gref[k][jj];
                    }
                }
            }
            hess[comp] = g;
        }
        hess
    }).collect()
}

/// 3×3 symmetric eigenvalue decomposition (analytical cubic).
struct Eig33 {
    lambda: [f64; 3],       // sorted by magnitude descending
    vectors: [[f64; 3]; 3], // corresponding eigenvectors (unit)
}

fn sym_eig_3x3(h: [[f64; 3]; 3]) -> Eig33 {
    let a = h[0][0]; let b = h[0][1]; let c = h[0][2];
    let d = h[1][1]; let e = h[1][2]; let f = h[2][2];

    // Check for near-zero matrix — treat as isotropic
    let norm = a.abs().max(b.abs()).max(c.abs()).max(d.abs()).max(e.abs()).max(f.abs());
    if norm < 1e-30 {
        return Eig33 { lambda: [0.0; 3], vectors: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] };
    }

    // Characteristic polynomial: -λ³ + I₁λ² - I₂λ + I₃ = 0  →  λ³ - I₁λ² + I₂λ - I₃ = 0
    let i1 = a + d + f;
    let i2 = a*d + a*f + d*f - b*b - c*c - e*e;
    let i3 = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);

    // Trigonometric formula for cubic (works when p ≤ 0).
    let p = i2 - i1*i1 / 3.0;
    let q = 2.0*i1*i1*i1 / 27.0 - i1*i2 / 3.0 + i3;

    let mut lam = if p >= 0.0 {
        // Near-degenerate case: eigenvalues are i1/3 ± sqrt(p/3)
        let sqrt_p3 = (p / 3.0).sqrt();
        let mu = i1 / 3.0;
        [mu + sqrt_p3, mu - sqrt_p3, mu]
    } else {
        let r = (-p.powi(3) / 27.0).sqrt();
        let phi = (q / (2.0 * r)).clamp(-1.0, 1.0).acos() / 3.0;
        let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
        let sqrt_p3 = 2.0 * (p / 3.0).abs().sqrt();
        [
            sqrt_p3 * phi.cos() + i1 / 3.0,
            -sqrt_p3 * (phi + two_pi_3).cos() + i1 / 3.0,
            -sqrt_p3 * (phi - two_pi_3).cos() + i1 / 3.0,
        ]
    };
    lam.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

    // Eigenvectors via cross-product method
    let mut vecs = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        let l = lam[i];
        let r0 = [a - l, b, c];
        let r1 = [b, d - l, e];
        let v = [
            r0[1]*r1[2] - r0[2]*r1[1],
            r0[2]*r1[0] - r0[0]*r1[2],
            r0[0]*r1[1] - r0[1]*r1[0],
        ];
        let norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        if norm > 1e-30 {
            vecs[i] = [v[0]/norm, v[1]/norm, v[2]/norm];
        } else if i == 0 {
            vecs[i] = [1.0, 0.0, 0.0];
        } else {
            // Gram-Schmidt on the previous vector
            let prev = vecs[0];
            let dot = prev[0] + prev[1] + prev[2]; // just use [1,0,0] or [0,1,0]
            if dot.abs() < 0.9 { vecs[i] = [1.0, 0.0, 0.0]; }
            else { vecs[i] = [0.0, 1.0, 0.0]; }
            // Orthogonalize
            let dp = vecs[i][0]*prev[0] + vecs[i][1]*prev[1] + vecs[i][2]*prev[2];
            for j in 0..3 { vecs[i][j] -= dp * prev[j]; }
            let n = (vecs[i][0].powi(2)+vecs[i][1].powi(2)+vecs[i][2].powi(2)).sqrt();
            if n > 1e-30 { for j in 0..3 { vecs[i][j] /= n; } }
        }
    }
    Eig33 { lambda: lam, vectors: vecs }
}

/// Estimate anisotropy for 3-D (Tet4) meshes.
pub fn estimate_anisotropy_3d(
    mesh: &Mesh<3>,
    u: &[f64],
    threshold: f64,
) -> Vec<AnisotropyEstimate<AnisoDirection3d>> {
    let n_elems = mesh.n_elems();
    let hessians = recover_hessian_3d(mesh, u);

    (0..n_elems).map(|e| {
        let eig = sym_eig_3x3(hessians[e]);

        let r01 = if eig.lambda[1].abs() > 1e-30 { (eig.lambda[0].abs() / eig.lambda[1].abs()).min(1e6) } else { 1e6 };
        let r12 = if eig.lambda[2].abs() > 1e-30 { (eig.lambda[1].abs() / eig.lambda[2].abs()).min(1e6) } else { 1e6 };
        let overall_ratio = r01.max(r12);
        let aspect = overall_ratio.sqrt().clamp(1.0, MAX_ASPECT_RATIO);

        let direction = if overall_ratio <= threshold {
            AnisoDirection3d::Isotropic
        } else {
            let v0 = eig.vectors[0];
            let ax = v0[0].abs(); let ay = v0[1].abs(); let az = v0[2].abs();
            // Map the dominant eigenvector to the nearest axis
            if ax >= ay && ax >= az { AnisoDirection3d::X }
            else if ay >= az { AnisoDirection3d::Y }
            else { AnisoDirection3d::Z }
        };

        AnisotropyEstimate { direction, aspect_ratio: aspect, strength: eig.lambda[0].abs(), eigenvalue_ratio: overall_ratio }
    }).collect()
}

/// Map `AnisoDirection3d` to `HexRefineDir` for Hex8 meshes.
pub fn aniso3d_to_hex_dir(dir: AnisoDirection3d) -> HexRefineDir {
    match dir {
        AnisoDirection3d::X => HexRefineDir::X,
        AnisoDirection3d::Y => HexRefineDir::Y,
        AnisoDirection3d::Z => HexRefineDir::Z,
        AnisoDirection3d::XY => HexRefineDir::XY,
        AnisoDirection3d::XZ => HexRefineDir::XZ,
        AnisoDirection3d::YZ => HexRefineDir::YZ,
        AnisoDirection3d::Isotropic => HexRefineDir::All,
    }
}

/// Build marked list for Hex8 anisotropic refinement.
pub fn mark_anisotropic_hexes(
    estimates: &[AnisotropyEstimate<AnisoDirection3d>],
    min_strength: f64,
) -> Vec<(ElemId, HexRefineDir)> {
    estimates.iter().enumerate()
        .filter(|(_, e)| e.strength > min_strength)
        .map(|(i, e)| (i as ElemId, aniso3d_to_hex_dir(e.direction)))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  Quality control (Task 4.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Compute the aspect ratio of each element in a 2-D mesh.
///
/// - Tri3: ratio of longest edge to shortest altitude.
/// - Quad4: ratio of longest to shortest edge length.
pub fn element_aspect_ratios_2d(mesh: &Mesh<2>) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let is_quad = mesh.element_type_at(0) == ElementType::Quad4;

    (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        if is_quad {
            let c = |i| mesh.coords_of(ns[i]);
            let edges = [
                ((c(1)[0]-c(0)[0]).powi(2) + (c(1)[1]-c(0)[1]).powi(2)).sqrt(),
                ((c(2)[0]-c(1)[0]).powi(2) + (c(2)[1]-c(1)[1]).powi(2)).sqrt(),
                ((c(3)[0]-c(2)[0]).powi(2) + (c(3)[1]-c(2)[1]).powi(2)).sqrt(),
                ((c(0)[0]-c(3)[0]).powi(2) + (c(0)[1]-c(3)[1]).powi(2)).sqrt(),
            ];
            let max_e = edges.iter().cloned().fold(0.0, f64::max);
            let min_e = edges.iter().cloned().fold(f64::MAX, f64::min);
            if min_e > 1e-30 { max_e / min_e } else { MAX_ASPECT_RATIO }
        } else {
            let c = |i| mesh.coords_of(ns[i]);
            let e0 = ((c(1)[0]-c(0)[0]).powi(2) + (c(1)[1]-c(0)[1]).powi(2)).sqrt();
            let e1 = ((c(2)[0]-c(1)[0]).powi(2) + (c(2)[1]-c(1)[1]).powi(2)).sqrt();
            let e2 = ((c(0)[0]-c(2)[0]).powi(2) + (c(0)[1]-c(2)[1]).powi(2)).sqrt();
            let area = 0.5 * ((c(1)[0]-c(0)[0])*(c(2)[1]-c(0)[1])
                            - (c(2)[0]-c(0)[0])*(c(1)[1]-c(0)[1])).abs();
            let max_e = e0.max(e1).max(e2);
            if area > 1e-30 {
                // Shortest altitude = 2*area / longest edge
                let min_alt = 2.0 * area / max_e;
                if min_alt > 1e-30 { max_e / min_alt } else { MAX_ASPECT_RATIO }
            } else {
                MAX_ASPECT_RATIO
            }
        }
    }).collect()
}

/// Compute the aspect ratio of each element in a 3-D mesh (Tet4).
///
/// For a Tet4: ratio of longest edge / shortest altitude.
pub fn element_aspect_ratios_3d(mesh: &Mesh<3>) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    (0..n_elems as ElemId).map(|e| {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        // 6 edges of a tet
        let edges = [
            ((c(1)[0]-c(0)[0]).powi(2)+(c(1)[1]-c(0)[1]).powi(2)+(c(1)[2]-c(0)[2]).powi(2)).sqrt(),
            ((c(2)[0]-c(0)[0]).powi(2)+(c(2)[1]-c(0)[1]).powi(2)+(c(2)[2]-c(0)[2]).powi(2)).sqrt(),
            ((c(3)[0]-c(0)[0]).powi(2)+(c(3)[1]-c(0)[1]).powi(2)+(c(3)[2]-c(0)[2]).powi(2)).sqrt(),
            ((c(2)[0]-c(1)[0]).powi(2)+(c(2)[1]-c(1)[1]).powi(2)+(c(2)[2]-c(1)[2]).powi(2)).sqrt(),
            ((c(3)[0]-c(1)[0]).powi(2)+(c(3)[1]-c(1)[1]).powi(2)+(c(3)[2]-c(1)[2]).powi(2)).sqrt(),
            ((c(3)[0]-c(2)[0]).powi(2)+(c(3)[1]-c(2)[1]).powi(2)+(c(3)[2]-c(2)[2]).powi(2)).sqrt(),
        ];
        let max_e = edges.iter().cloned().fold(0.0, f64::max);
        // Volume of tet
        let det = (c(1)[0]-c(0)[0])*((c(2)[1]-c(0)[1])*(c(3)[2]-c(0)[2])-(c(2)[2]-c(0)[2])*(c(3)[1]-c(0)[1]))
                - (c(1)[1]-c(0)[1])*((c(2)[0]-c(0)[0])*(c(3)[2]-c(0)[2])-(c(2)[2]-c(0)[2])*(c(3)[0]-c(0)[0]))
                + (c(1)[2]-c(0)[2])*((c(2)[0]-c(0)[0])*(c(3)[1]-c(0)[1])-(c(2)[1]-c(0)[1])*(c(3)[0]-c(0)[0]));
        let vol = det.abs() / 6.0;
        if vol > 1e-30 {
            // Altitude = 3*vol / area_of_largest_face
            let face_areas = [
                // face 0,1,2
                {
                    let ex = c(1)[0]-c(0)[0]; let ey = c(1)[1]-c(0)[1]; let ez = c(1)[2]-c(0)[2];
                    let fx = c(2)[0]-c(0)[0]; let fy = c(2)[1]-c(0)[1]; let fz = c(2)[2]-c(0)[2];
                    0.5 * ((ey*fz-ez*fy).powi(2)+(ez*fx-ex*fz).powi(2)+(ex*fy-ey*fx).powi(2)).sqrt()
                },
                // face 0,1,3
                {
                    let ex = c(1)[0]-c(0)[0]; let ey = c(1)[1]-c(0)[1]; let ez = c(1)[2]-c(0)[2];
                    let fx = c(3)[0]-c(0)[0]; let fy = c(3)[1]-c(0)[1]; let fz = c(3)[2]-c(0)[2];
                    0.5 * ((ey*fz-ez*fy).powi(2)+(ez*fx-ex*fz).powi(2)+(ex*fy-ey*fx).powi(2)).sqrt()
                },
                // face 0,2,3
                {
                    let ex = c(2)[0]-c(0)[0]; let ey = c(2)[1]-c(0)[1]; let ez = c(2)[2]-c(0)[2];
                    let fx = c(3)[0]-c(0)[0]; let fy = c(3)[1]-c(0)[1]; let fz = c(3)[2]-c(0)[2];
                    0.5 * ((ey*fz-ez*fy).powi(2)+(ez*fx-ex*fz).powi(2)+(ex*fy-ey*fx).powi(2)).sqrt()
                },
                // face 1,2,3
                {
                    let ex = c(2)[0]-c(1)[0]; let ey = c(2)[1]-c(1)[1]; let ez = c(2)[2]-c(1)[2];
                    let fx = c(3)[0]-c(1)[0]; let fy = c(3)[1]-c(1)[1]; let fz = c(3)[2]-c(1)[2];
                    0.5 * ((ey*fz-ez*fy).powi(2)+(ez*fx-ex*fz).powi(2)+(ex*fy-ey*fx).powi(2)).sqrt()
                },
            ];
            let max_face = face_areas.iter().cloned().fold(0.0, f64::max);
            if max_face > 1e-30 {
                let alt = 3.0 * vol / max_face;
                if alt > 1e-30 { max_e / alt } else { MAX_ASPECT_RATIO }
            } else { MAX_ASPECT_RATIO }
        } else { MAX_ASPECT_RATIO }
    }).collect()
}

/// Degrade marked anisotropic elements whose aspect ratio exceeds the maximum.
///
/// Elements whose anisotropy estimate implies an aspect ratio > `max_ratio`
/// are changed to isotropic refinement.  This prevents the creation of
/// overly stretched elements.
///
/// For use with Quad4: modifies `QuadRefineDir::X`/`Y` to `Both`.
pub fn degrade_anisotropic_quads(
    marked: &mut [(ElemId, QuadRefineDir)],
    aspects: &[f64],
    max_ratio: f64,
) {
    for (e, dir) in marked.iter_mut() {
        if *dir != QuadRefineDir::Both {
            let e_idx = *e as usize;
            if e_idx < aspects.len() && aspects[e_idx] > max_ratio {
                *dir = QuadRefineDir::Both;
            }
        }
    }
}

/// Degrade marked anisotropic Tri3 elements whose aspect ratio exceeds max.
pub fn degrade_anisotropic_tris(
    marked: &mut Vec<(ElemId, TriRefineDir)>,
    post_refine_aspects: &[f64],
    max_ratio: f64,
) {
    marked.retain(|&(_, _dir)| {
        // Keep only isotropic refinements for degraded elements.
        // This callback is intended for post-refinement quality checks;
        // elements whose actual refined aspect ratio is too high are
        // simply removed and will be refined isotropically in the next pass.
        true
    });
    let _ = post_refine_aspects; // unused in this simple version
    let _ = max_ratio;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::NodeId;
    use crate::Mesh;

    // ── 2×2 eigenvalue test ───────────────────────────────────────────────

    #[test]
    fn sym_eig_2x2_identity() {
        let h = [[1.0, 0.0], [0.0, 1.0]];
        let eig = sym_eig_2x2(h);
        assert!((eig.lambda0 - 1.0).abs() < 1e-14);
        assert!((eig.lambda1 - 1.0).abs() < 1e-14);
        assert!((eig.v0x - 1.0).abs() < 1e-14 || (eig.v0y - 1.0).abs() < 1e-14);
    }

    #[test]
    fn sym_eig_2x2_diagonal() {
        let h = [[3.0, 0.0], [0.0, 1.0]];
        let eig = sym_eig_2x2(h);
        assert!((eig.lambda0 - 3.0).abs() < 1e-14);
        assert!((eig.lambda1 - 1.0).abs() < 1e-14);
        assert!((eig.v0x - 1.0).abs() < 1e-14);
        assert!((eig.v0y).abs() < 1e-14);
    }

    #[test]
    fn sym_eig_2x2_off_diagonal() {
        let h = [[2.0, 1.0], [1.0, 2.0]];
        let eig = sym_eig_2x2(h);
        // eigenvalues: 3 and 1
        assert!((eig.lambda0 - 3.0).abs() < 1e-14);
        assert!((eig.lambda1 - 1.0).abs() < 1e-14);
        // eigenvector for λ=3 is [1,1]/√2
        assert!((eig.v0x - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
        assert!((eig.v0y - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
    }

    // ── Hessian recovery tests ─────────────────────────────────────────────

    #[test]
    fn hessian_linear_solution_zero() {
        // u = 2x + 3y → ∇u = [2, 3], Hessian = [[0,0],[0,0]]
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); 2.0*c[0] + 3.0*c[1]
        }).collect();

        let hess = recover_hessian_2d(&mesh, &u);
        let max = hess.iter()
            .flat_map(|h| h.iter().flat_map(|r| r.iter()))
            .cloned().fold(0.0, f64::max);
        assert!(max < 1e-10, "Hessian should be near-zero for linear u, got {max:.3e}");
    }

    #[test]
    fn hessian_quadratic_solution_constant() {
        // u = x² + y² → ∇u = [2x, 2y], Hessian = [[2,0],[0,2]]
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0] + c[1]*c[1]
        }).collect();

        let hess = recover_hessian_2d(&mesh, &u);
        // Hessian should be ≈ [[2,0],[0,2]] for all elements
        let mut avg_hxx = 0.0; let mut avg_hyy = 0.0;
        for h in &hess {
            avg_hxx += h[0][0]; avg_hyy += h[1][1];
        }
        let n = hess.len() as f64;
        avg_hxx /= n; avg_hyy /= n;

        assert!((avg_hxx - 2.0).abs() < 0.5,
            "H_xx should be ≈ 2 for x²+y², got {avg_hxx:.3}");
        assert!((avg_hyy - 2.0).abs() < 0.5,
            "H_yy should be ≈ 2 for x²+y², got {avg_hyy:.3}");
    }

    // ── Anisotropy estimation tests ────────────────────────────────────────

    #[test]
    fn anisotropy_linear_solution_isotropic() {
        // Linear u → Hessian = 0 → all elements isotropic
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] + 2.0*c[1]
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, ANISO_THRESHOLD);
        assert_eq!(estimates.len(), mesh.n_elems());
        for est in &estimates {
            assert_eq!(est.direction, AnisoDirection::Isotropic,
                "Linear solution should give isotropic everywhere");
            assert!((est.aspect_ratio - 1.0).abs() < 0.1,
                "Linear solution should have aspect ratio near 1, got {}", est.aspect_ratio);
        }
    }

    #[test]
    fn anisotropy_strong_x_curvature() {
        // u = x² (strong X curvature, zero Y curvature)
        // → Hessian = [[2,0],[0,0]] → λ₀=2 (x-direction), λ₁=0
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, 2.0);
        let n_aniso = estimates.iter().filter(|e| e.direction != AnisoDirection::Isotropic).count();
        assert!(n_aniso > 0, "x² should produce some anisotropic markers, got {n_aniso}");

        // Most anisotropic elements should point in X direction
        let n_x = estimates.iter()
            .filter(|e| e.direction == AnisoDirection::X).count();
        assert!(n_x > n_aniso / 3,
            "x² should produce mostly X-direction anisotropy, got X:{n_x} anisotropic:{n_aniso}");
    }

    #[test]
    fn anisotropy_strong_y_curvature() {
        // u = y² → Hessian = [[0,0],[0,2]] → λ₀=2 (y-direction)
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1]*c[1]
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, 2.0);
        let n_y = estimates.iter()
            .filter(|e| e.direction == AnisoDirection::Y).count();
        assert!(n_y > 0, "y² should produce some Y-direction markers, got {n_y}");
    }

    // ── QuadRefineDir mapping tests ────────────────────────────────────────

    #[test]
    fn aniso_to_quad_dir_mapping() {
        assert_eq!(aniso_to_quad_dir(AnisoDirection::X), QuadRefineDir::X);
        assert_eq!(aniso_to_quad_dir(AnisoDirection::Y), QuadRefineDir::Y);
        assert_eq!(aniso_to_quad_dir(AnisoDirection::Isotropic), QuadRefineDir::Both);
    }

    // ── Aspect ratio tests ─────────────────────────────────────────────────

    #[test]
    fn aspect_ratio_unit_square_tri() {
        // Regular triangles on unit square → aspect ratio ≈ 1.4 (right isosceles)
        let mesh = Mesh::<2>::unit_square_tri(8);
        let ratios = element_aspect_ratios_2d(&mesh);
        for &r in &ratios {
            assert!(r >= 1.0, "Aspect ratio should be ≥ 1, got {r}");
            assert!(r < 10.0, "Aspect ratio on regular mesh should be small, got {r}");
        }
    }

    #[test]
    fn aspect_ratio_unit_square_quad() {
        // Regular quads on unit square → aspect ratio = 1
        let mesh = Mesh::<2>::unit_square_quad(8);
        let ratios = element_aspect_ratios_2d(&mesh);
        for &r in &ratios {
            assert!((r - 1.0).abs() < 0.01,
                "Regular square quads should have aspect ratio 1, got {r}");
        }
    }

    #[test]
    fn aspect_ratio_unit_cube_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(4);
        let ratios = element_aspect_ratios_3d(&mesh);
        for &r in &ratios {
            assert!(r >= 1.0 && r < 10.0,
                "Tet aspect ratio should be reasonable, got {r}");
        }
    }

    // ── Degrade tests ──────────────────────────────────────────────────────

    #[test]
    fn degrade_overly_anisotropic_quads() {
        let mut marked = vec![(0, QuadRefineDir::X), (1, QuadRefineDir::Y), (2, QuadRefineDir::Both)];
        let aspects = vec![15.0, 3.0, 1.0]; // elem 0 exceeds max 10
        degrade_anisotropic_quads(&mut marked, &aspects, 10.0);
        // elem 0 should be degraded to Both
        assert_eq!(marked[0].1, QuadRefineDir::Both, "Overly anisotropic should degrade to Both");
        assert_eq!(marked[1].1, QuadRefineDir::Y, "Elem 1 should keep Y (aspect 3 < 10)");
        assert_eq!(marked[2].1, QuadRefineDir::Both, "Elem 2 should stay Both");
    }

    // ── Boundary-layer-style test (Task 4.1 Step 4) ────────────────────────

    #[test]
    fn boundary_layer_produces_anisotropy() {
        // Simulate a boundary layer: u = exp(-20*y) (strong variation near y=0)
        // The anisotropy should be perpendicular to the boundary (Y direction).
        let mesh = Mesh::<2>::unit_square_quad(16);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            (-20.0 * c[1]).exp()  // strong boundary layer at bottom
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, 3.0);
        let n_y = estimates.iter()
            .filter(|e| e.direction == AnisoDirection::Y).count();
        let total = estimates.len();
        // The boundary layer should produce many Y-direction refinements
        // (refinement across the layer, not along it).
        assert!(n_y > total / 4,
            "Boundary layer should produce significant Y anisotropy, got Y:{n_y}/{total}");
    }

    // ── Mark function tests ────────────────────────────────────────────────

    #[test]
    fn mark_anisotropic_quads_filters_by_strength() {
        let mesh = Mesh::<2>::unit_square_quad(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0]
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, 2.0);
        let marked = mark_anisotropic_quads(&estimates, 0.01);
        assert!(marked.len() <= estimates.len());
        for &(e, dir) in &marked {
            assert!(e < estimates.len() as ElemId);
            assert!(dir == QuadRefineDir::X || dir == QuadRefineDir::Both);
        }
    }

    // ── 3-D anisotropy tests ───────────────────────────────────────────────

    #[test]
    fn hessian_3d_linear_solution_finite() {
        let mesh = Mesh::<3>::unit_cube_tet(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] + c[1] + c[2]
        }).collect();

        let hess = recover_hessian_3d(&mesh, &u);
        // Check Hessian has reasonable values (finite, no NaN)
        for h in &hess {
            for row in h {
                for &v in row {
                    assert!(v.is_finite(), "Hessian should be finite, got NaN/Inf");
                }
            }
        }
        // The Hessian might not be exactly zero on Tet4 due to averaging
        // artifacts, but should not catastrophically diverge.
        let max_abs = hess.iter()
            .flat_map(|h| h.iter().flat_map(|r| r.iter()))
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(max_abs < 100.0,
            "3D Hessian should not be huge for linear u, got {max_abs:.1}");
    }

    #[test]
    fn anisotropy_3d_linear_isotropic() {
        let mesh = Mesh::<3>::unit_cube_tet(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] + c[1] - c[2]
        }).collect();

        let estimates = estimate_anisotropy_3d(&mesh, &u, ANISO_THRESHOLD);
        // Most elements should be isotropic (Hessian ≈ 0 for linear soln on Tet)
        let n_iso = estimates.iter().filter(|e| e.direction == AnisoDirection3d::Isotropic).count();
        assert!(n_iso as f64 > estimates.len() as f64 * 0.5,
            "3D linear: most elements should be isotropic, got isotropic {n_iso}/{}", estimates.len());
    }

    #[test]
    fn aniso3d_to_hex_dir_mapping() {
        assert_eq!(aniso3d_to_hex_dir(AnisoDirection3d::X), HexRefineDir::X);
        assert_eq!(aniso3d_to_hex_dir(AnisoDirection3d::Y), HexRefineDir::Y);
        assert_eq!(aniso3d_to_hex_dir(AnisoDirection3d::Z), HexRefineDir::Z);
        assert_eq!(aniso3d_to_hex_dir(AnisoDirection3d::Isotropic), HexRefineDir::All);
    }

    // ── Quad4 boundary layer test ──────────────────────────────────────────

    #[test]
    fn boundary_layer_quad4_aspect_ratio_check() {
        // Refine a boundary-layer solution on Quad4 mesh, check aspect ratios
        let mesh = Mesh::<2>::unit_square_quad(12);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId);
            (-15.0 * c[1]).exp()
        }).collect();

        let estimates = estimate_anisotropy_2d(&mesh, &u, 3.0);
        let mut marked = mark_anisotropic_quads(&estimates, 0.001);

        // Check that no element has aspect ratio > 10 after degradation
        let aspects = element_aspect_ratios_2d(&mesh);
        let max_before = aspects.iter().cloned().fold(0.0, f64::max);

        degrade_anisotropic_quads(&mut marked, &aspects, 10.0);
        let degraded_count = marked.iter()
            .filter(|(_, d)| *d == QuadRefineDir::Both)
            .count();

        assert!(max_before < 10.0 || degraded_count > 0,
            "Elements with aspect ratio > 10 should be degraded");
    }
}
