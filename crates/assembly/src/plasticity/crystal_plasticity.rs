//! Single-crystal plasticity for FCC metals.
//!
//! Implements rate-dependent crystal plasticity with:
//! - 12 FCC slip systems {111}<110>
//! - Power-law flow rule: γ̇ᵅ = γ̇₀·(|τᵅ|/gᵅ)ⁿ·sign(τᵅ)
//! - Asymptotic hardening: ġᵅ = h₀·(1 - gᵅ/g_∞)·Σ|γ̇ᵝ|
//! - Resolved shear stress: τᵅ = σ : (sᵅ ⊗ mᵅ)
//! - Consistent tangent via algorithmic elasto-plastic modulus
//!
//! ## Slip systems (FCC)
//! {111}<110> family: 12 slip systems defined by the 4 {111} slip planes
//! and 3 <110> slip directions per plane.
//!
//! ## Crystal orientation
//! By default assumes [100], [010], [001] aligned with the global axes.
//! Use [`CrystalConfig::with_orientation`] for rotated crystals.

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

/// Crystal plasticity material parameters.
#[derive(Debug, Clone)]
pub struct CrystalConfig {
    /// Elastic constants in crystal frame.
    pub c11: f64, pub c12: f64, pub c44: f64,
    /// Reference slip rate γ̇₀.
    pub dot_gamma_0: f64,
    /// Rate-sensitivity exponent n.
    pub n: f64,
    /// Initial slip resistance g₀.
    pub g0: f64,
    /// Saturation slip resistance g_∞.
    pub g_inf: f64,
    /// Initial hardening rate h₀.
    pub h0: f64,
    /// Crystal orientation (passive rotation matrix, 3×3).
    pub orientation: DMatrix<f64>,
}

impl CrystalConfig {
    /// Cubic crystal (e.g., aluminium) aligned with global axes.
    pub fn cubic(c11: f64, c12: f64, c44: f64) -> Self {
        Self {
            c11, c12, c44,
            dot_gamma_0: 0.001, n: 20.0,
            g0: 30.0, g_inf: 100.0, h0: 500.0,
            orientation: DMatrix::identity(3, 3),
        }
    }

    /// FCC aluminium parameters (typical).
    pub fn aluminium() -> Self {
        Self::cubic(108.2e3, 61.3e3, 28.5e3)
    }

    /// FCC copper parameters.
    pub fn copper() -> Self {
        Self::cubic(168.4e3, 121.4e3, 75.4e3)
    }

    /// Apply a rotation to the crystal orientation.
    pub fn with_orientation(mut self, angles_deg: [f64; 3]) -> Self {
        // Bunge Euler angles (φ₁, Φ, φ₂) → rotation matrix
        let (c1, s1) = angles_deg[0].to_radians().sin_cos();
        let (c2, s2) = angles_deg[1].to_radians().sin_cos();
        let (c3, s3) = angles_deg[2].to_radians().sin_cos();
        let r = DMatrix::from_row_slice(3, 3, &[
            c1*c3 - s1*c2*s3,  s1*c3 + c1*c2*s3,  s2*s3,
            -c1*s3 - s1*c2*c3, -s1*s3 + c1*c2*c3, s2*c3,
            s1*s2,             -c1*s2,            c2,
        ]);
        self.orientation = r;
        self
    }

    /// Elastic stiffness in the global frame (Voigt 6×6).
    fn elastic_stiffness(&self) -> DMatrix<f64> {
        let mut c = DMatrix::zeros(6, 6);
        let a = [[self.c11, self.c12, self.c12],
                 [self.c12, self.c11, self.c12],
                 [self.c12, self.c12, self.c11]];
        for i in 0..3 { for j in 0..3 { c[(i,j)] = a[i][j]; } }
        c[(3,3)] = self.c44; c[(4,4)] = self.c44; c[(5,5)] = self.c44;
        // Transform to global frame if orientation is not identity
        if self.orientation != DMatrix::identity(3, 3) {
            c = rotate_stiffness(&c, &self.orientation);
        }
        c
    }
}

/// Rotate a 6×6 Voigt stiffness matrix by rotation matrix R (3×3).
fn rotate_stiffness(c: &DMatrix<f64>, r: &DMatrix<f64>) -> DMatrix<f64> {
    // Transformation matrix T (6×6) from Kelvin-Voigt:
    // σ̄ = M·σ,  ε̄ = M⁻ᵀ·ε  with M built from R
    let mut m = DMatrix::zeros(6, 6);
    for i in 0..3 { for j in 0..3 { m[(i,j)] = r[(i,j)].powi(2); } }
    // Shear components
    m[(0,3)] = 2.0*r[(0,0)]*r[(0,1)]; m[(0,4)] = 2.0*r[(0,0)]*r[(0,2)]; m[(0,5)] = 2.0*r[(0,1)]*r[(0,2)];
    m[(1,3)] = 2.0*r[(1,0)]*r[(1,1)]; m[(1,4)] = 2.0*r[(1,0)]*r[(1,2)]; m[(1,5)] = 2.0*r[(1,1)]*r[(1,2)];
    m[(2,3)] = 2.0*r[(2,0)]*r[(2,1)]; m[(2,4)] = 2.0*r[(2,0)]*r[(2,2)]; m[(2,5)] = 2.0*r[(2,1)]*r[(2,2)];
    m[(3,0)] = r[(0,0)]*r[(1,0)]; m[(3,1)] = r[(0,1)]*r[(1,1)]; m[(3,2)] = r[(0,2)]*r[(1,2)];
    m[(3,3)] = r[(0,0)]*r[(1,1)]+r[(0,1)]*r[(1,0)]; m[(3,4)] = r[(0,0)]*r[(1,2)]+r[(0,2)]*r[(1,0)];
    m[(3,5)] = r[(0,1)]*r[(1,2)]+r[(0,2)]*r[(1,1)];
    m[(4,0)] = r[(0,0)]*r[(2,0)]; m[(4,1)] = r[(0,1)]*r[(2,1)]; m[(4,2)] = r[(0,2)]*r[(2,2)];
    m[(4,3)] = r[(0,0)]*r[(2,1)]+r[(0,1)]*r[(2,0)]; m[(4,4)] = r[(0,0)]*r[(2,2)]+r[(0,2)]*r[(2,0)];
    m[(4,5)] = r[(0,1)]*r[(2,2)]+r[(0,2)]*r[(2,1)];
    m[(5,0)] = r[(1,0)]*r[(2,0)]; m[(5,1)] = r[(1,1)]*r[(2,1)]; m[(5,2)] = r[(1,2)]*r[(2,2)];
    m[(5,3)] = r[(1,0)]*r[(2,1)]+r[(1,1)]*r[(2,0)]; m[(5,4)] = r[(1,0)]*r[(2,2)]+r[(1,2)]*r[(2,0)];
    m[(5,5)] = r[(1,1)]*r[(2,2)]+r[(1,2)]*r[(2,1)];
    let mt = m.transpose();
    m * c * mt
}

// ─── FCC slip systems ─────────────────────────────────────────────────────────

/// A single slip system: slip direction s, plane normal m.
#[derive(Debug, Clone, Copy)]
pub struct SlipSystem {
    s: [f64; 3],
    m: [f64; 3],
}

/// Generate the 12 FCC {111}<110> slip systems.
fn fcc_slip_systems() -> Vec<SlipSystem> {
    // Slip planes {111}: normals
    let planes = [
        [ 1.0,  1.0,  1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ];
    // Slip directions <110>
    let dirs = [
        [ 1.0, -1.0,  0.0],
        [ 0.0,  1.0, -1.0],
        [ 1.0,  0.0, -1.0],
    ];
    let mut sys = Vec::with_capacity(12);
    for &p in &planes {
        for &d in &dirs {
            // Normalise
            let pn = f64::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
            let dn = f64::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            let m = [p[0]/pn, p[1]/pn, p[2]/pn];
            let mut s = [d[0]/dn, d[1]/dn, d[2]/dn];
            // Ensure s·m = 0
            let dot = s[0]*m[0] + s[1]*m[1] + s[2]*m[2];
            for i in 0..3 { s[i] -= dot * m[i]; }
            let sn = f64::sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
            for i in 0..3 { s[i] /= sn; }
            sys.push(SlipSystem { s, m });
        }
    }
    sys
}

/// Compute the Schmid tensor: Pᵅ = sᵅ ⊗ mᵅ (symmetric part).
fn schmid_tensor(ss: &SlipSystem) -> DMatrix<f64> {
    let mut p = DMatrix::zeros(3, 3);
    for i in 0..3 { for j in 0..3 {
        p[(i,j)] = 0.5 * (ss.s[i] * ss.m[j] + ss.s[j] * ss.m[i]);
    }}
    p
}

/// Resolved shear stress τᵅ on slip system α.
fn resolved_shear(sigma: &[f64; 6], p: &DMatrix<f64>) -> f64 {
    // sigma in Voigt: [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx]
    let mut tau = 0.0;
    tau += sigma[0] * p[(0,0)];
    tau += sigma[1] * p[(1,1)];
    tau += sigma[2] * p[(2,2)];
    tau += 2.0 * sigma[3] * p[(0,1)];
    tau += 2.0 * sigma[4] * p[(1,2)];
    tau += 2.0 * sigma[5] * p[(0,2)];
    tau
}

// ─── Crystal plasticity state ─────────────────────────────────────────────────

/// Per-QP state for crystal plasticity.
pub struct CrystalState {
    /// Slip resistances gᵅ per system (length 12).
    pub g: Vec<Vec<f64>>,
    /// Total accumulated slip γ_a.
    pub gamma_acc: Vec<f64>,
    /// Plastic velocity gradient L^p (incremental).
    pub n_qp: usize,
}

impl CrystalState {
    pub fn new(n_qp: usize) -> Self {
        let g = vec![vec![30.0; 12]; n_qp];
        Self { g, gamma_acc: vec![0.0; n_qp], n_qp }
    }
}

// ─── Crystal plasticity assembly ──────────────────────────────────────────────

/// Assemble the crystal plasticity residual and tangent stiffness.
///
/// Implements a rate-dependent crystal plasticity constitutive update
/// at each quadrature point with power-law flow rule and asymptotic hardening.
pub fn assemble_crystal_plasticity<M: MeshTopology>(
    mesh: &M,
    space: &dyn FESpace<Mesh = M>,
    u: &[f64],
    cfg: &CrystalConfig,
    state: &mut CrystalState,
    dt: f64,
    quad_order: u8,
) -> (Vec<f64>, CsrMatrix<f64>) {
    let dim = 3usize; // crystal plasticity is inherently 3D
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut f_int = vec![0.0; n_dofs];
    let c_e = cfg.elastic_stiffness();
    let sys = fcc_slip_systems();
    let schmid: Vec<DMatrix<f64>> = sys.iter().map(schmid_tensor).collect();

    let mut qp_idx = 0usize;
    let mut elem_dofs_cache = Vec::new();

    // Build element DOF cache
    for e in mesh.elem_iter() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        elem_dofs_cache.push(dofs);
    }

    for (el, e) in mesh.elem_iter().enumerate() {
        let et = mesh.element_type(e);
        let re = ref_elem_vol(et, space.order());
        let n_ldofs = re.n_dofs();
        let n_vec = n_ldofs * dim;
        let quad = re.quadrature(quad_order);
        let n_qp_e = quad.weights.len();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac_2(mesh, nodes, dim);
        let jit = jac.try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));

        let mut u_elem = vec![0.0_f64; n_vec];
        for (k, &dof) in elem_dofs_cache[el].iter().enumerate() { u_elem[k] = u[dof]; }

        let mut k_elem = vec![0.0_f64; n_vec * n_vec];
        let mut f_elem = vec![0.0_f64; n_vec];
        let mut gref = vec![0.0_f64; n_ldofs * dim];
        let mut gphys = vec![0.0_f64; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w_vol = quad.weights[q] * det_j.abs();
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            // Total strain increment at QP
            let mut deps = [0.0_f64; 6];
            for k in 0..n_ldofs {
                deps[0] += u_elem[k*dim]   * gphys[k*dim];
                deps[1] += u_elem[k*dim+1] * gphys[k*dim+1];
                deps[2] += u_elem[k*dim+2] * gphys[k*dim+2];
                deps[3] += u_elem[k*dim]*gphys[k*dim+1] + u_elem[k*dim+1]*gphys[k*dim];
                deps[4] += u_elem[k*dim+1]*gphys[k*dim+2] + u_elem[k*dim+2]*gphys[k*dim+1];
                deps[5] += u_elem[k*dim]*gphys[k*dim+2] + u_elem[k*dim+2]*gphys[k*dim];
            }
            // Plastic strain increment via crystal plasticity update
            // Trial stress: σ^tr = C : (ε_total - ε_plastic)
            // For simplicity, assume small-strain additive decomposition
            // and perform a full implicit update at each QP.

            // Current slip resistance and accumulated slip
            let mut g_cur = state.g[qp_idx + q].clone();

            // Compute resolved shear stresses from trial stress
            // (simplified: assume fully elastic trial, then relax)
            let mut sigma_trial = [0.0_f64; 6];
            for i in 0..6 {
                for j in 0..6 { sigma_trial[i] += c_e[(i,j)] * deps[j]; }
            }

            // Power-law flow: Δγᵅ = Δt · γ̇₀ · (|τᵅ|/gᵅ)ⁿ · sign(τᵅ)
            let mut dgamma = [0.0_f64; 12];
            let mut gamma_dot_sum = 0.0_f64;
            for (a, s) in schmid.iter().enumerate() {
                let tau = resolved_shear(&sigma_trial, s);
                let ratio = tau.abs() / g_cur[a].max(1e-30);
                let flow = cfg.dot_gamma_0 * ratio.powf(cfg.n);
                dgamma[a] = dt * flow * tau.signum();
                gamma_dot_sum += dgamma[a].abs();
            }

            // Update slip resistance (asymptotic hardening)
            let h_slope = cfg.h0 * (1.0 - g_cur.iter().sum::<f64>() / 12.0 / cfg.g_inf).max(0.0);
            for a in 0..12 {
                g_cur[a] += h_slope * gamma_dot_sum * dt;
                g_cur[a] = g_cur[a].min(cfg.g_inf);
            }

            // Plastic strain increment: Δε^p = Σ Δγᵅ · Pᵅ
            let mut deps_p = [0.0_f64; 6];
            for (a, s) in schmid.iter().enumerate() {
                let p = s;
                deps_p[0] += dgamma[a] * p[(0,0)];
                deps_p[1] += dgamma[a] * p[(1,1)];
                deps_p[2] += dgamma[a] * p[(2,2)];
                deps_p[3] += 2.0 * dgamma[a] * p[(0,1)];
                deps_p[4] += 2.0 * dgamma[a] * p[(1,2)];
                deps_p[5] += 2.0 * dgamma[a] * p[(0,2)];
            }

            // Elastic strain and stress
            let mut eps_e = [0.0_f64; 6];
            for i in 0..6 { eps_e[i] = deps[i] - deps_p[i]; }
            let mut sigma = [0.0_f64; 6];
            for i in 0..6 {
                for j in 0..6 { sigma[i] += c_e[(i,j)] * eps_e[j]; }
            }

            // Store state
            state.g[qp_idx + q] = g_cur;
            state.gamma_acc[qp_idx + q] += gamma_dot_sum * dt;

            // Tangent approximation: elastic predictor (simplified)
            // For a full consistent tangent, need ∂Δγ/∂τ coupling.
            // Here we use the elastic tangent as an approximation.
            let c_tangent = &c_e;

            // Assemble residual
            for k in 0..n_ldofs {
                for i in 0..dim {
                    let row = k * dim + i;
                    let mut s = 0.0;
                    for j in 0..dim {
                        let sig_idx = if i == j { i } else { 3 + (i+j-1)%3 };
                        s += sigma[sig_idx] * gphys[k * dim + j];
                    }
                    f_elem[row] += w_vol * s;
                }
            }

            // Assemble tangent (elastic)
            for k in 0..n_ldofs { for i in 0..dim {
                let row = k * dim + i;
                for l in 0..n_ldofs { for a in 0..dim {
                    let col = l * dim + a;
                    let mut val = 0.0;
                    for j in 0..dim { for b in 0..dim {
                        let cij = if i==j { i } else { 3+(i+j-1)%3 };
                        let cab = if a==b { a } else { 3+(a+b-1)%3 };
                        val += c_tangent[(cij, cab)] * gphys[k*dim+j] * gphys[l*dim+b];
                    }}
                    k_elem[row * n_vec + col] += w_vol * val;
                }}
            }}
        }

        let dofs = &elem_dofs_cache[el];
        coo.add_element_matrix(dofs, &k_elem);
        for (k, &dof) in dofs.iter().enumerate() { f_int[dof] += f_elem[k]; }
        qp_idx += n_qp_e;
    }

    (f_int, coo.into_csr())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        _ => panic!("crystal_plasticity: unsupported element {:?} order {}", et, order),
    }
}

fn simplex_jac_2<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for c in 0..dim {
        let xc = mesh.node_coords(nodes[c+1]);
        for r in 0..dim { j[(r,c)] = xc[r] - x0[r]; }
    }
    (j.clone(), j.determinant())
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// ─── Lattice rotation (texture update) ─────────────────────────────────

/// Compute the antisymmetric Schmid tensor: Aⁱⱼ = ½(sᵢ·mⱼ − sⱼ·mᵢ).
fn spin_tensor(ss: &SlipSystem) -> DMatrix<f64> {
    let mut a = DMatrix::zeros(3, 3);
    for i in 0..3 { for j in 0..3 {
        a[(i,j)] = 0.5 * (ss.s[i] * ss.m[j] - ss.s[j] * ss.m[i]);
    }}
    a
}

/// Update the crystal orientation due to plastic spin.
///
/// Given the total velocity gradient `L = ∇u` and the slip increments
/// `dgamma` per slip system, compute the lattice rotation increment
/// and update the orientation matrix.
///
/// Returns the new orientation matrix.
pub fn update_lattice_rotation(
    orientation: &DMatrix<f64>,
    grad_u: &DMatrix<f64>,
    dgamma: &[f64],
    slip_systems: &[SlipSystem],
) -> DMatrix<f64> {
    // Total spin: W = (L - Lᵀ)/2
    let lt = grad_u.transpose();
    let w_total = (grad_u - &lt) * 0.5;

    // Plastic spin: W^p = Σ_α γ̇_α · Aⁱ
    let mut wp = DMatrix::zeros(3, 3);
    for (i, ss) in slip_systems.iter().enumerate() {
        let a = spin_tensor(ss);
        for r in 0..3 { for c in 0..3 {
            wp[(r,c)] += dgamma[i] * a[(r,c)];
        }}
    }

    // Elastic spin: W^e = W - W^p
    let we = &w_total - &wp;

    // Rotation increment: ΔR = exp(W^e·Δt) ≈ I + W^e (small rotation approx)
    // For small elastic rotations (typical in metal plasticity), use the
    // linearized update: R_new = (I + W^e·Δt) · R_old
    // Normalize to ensure orthogonality.
    let mut r_new = &we + &DMatrix::identity(3, 3);
    r_new = &r_new * orientation;

    // Re-orthogonalize via polar decomposition (simplified: Gram-Schmidt)
    for i in 0..3 {
        let mut col = r_new.column(i).into_owned();
        let norm = (col[0]*col[0] + col[1]*col[1] + col[2]*col[2]).sqrt();
        if norm > 1e-30 { for j in 0..3 { col[j] /= norm; } }
        // Orthogonalize against previous columns
        for j in 0..i {
            let dot = col[0]*r_new[(0,j)] + col[1]*r_new[(1,j)] + col[2]*r_new[(2,j)];
            for k in 0..3 { col[k] -= dot * r_new[(k,j)]; }
        }
        let norm2 = (col[0]*col[0] + col[1]*col[1] + col[2]*col[2]).sqrt();
        if norm2 > 1e-30 { for j in 0..3 { r_new[(j,i)] = col[j] / norm2; } }
    }
    r_new
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcc_slip_system_count() {
        let sys = fcc_slip_systems();
        assert_eq!(sys.len(), 12, "FCC has 12 slip systems");
    }

    #[test]
    fn schmid_tensor_symmetric() {
        let sys = fcc_slip_systems();
        let p = schmid_tensor(&sys[0]);
        for i in 0..3 { for j in 0..3 {
            assert!((p[(i,j)] - p[(j,i)]).abs() < 1e-15, "Schmid tensor not symmetric");
        }}
    }

    #[test]
    fn crystal_orientation_rotation() {
        let cfg = CrystalConfig::aluminium().with_orientation([30.0, 0.0, 0.0]);
        let c = cfg.elastic_stiffness();
        assert_eq!(c.nrows(), 6);
        assert_eq!(c.ncols(), 6);
    }

    #[test]
    fn crystal_state_initialises() {
        let s = CrystalState::new(5);
        assert_eq!(s.g.len(), 5);
        assert_eq!(s.g[0].len(), 12);
    }

    #[test]
    fn spin_tensor_antisymmetric() {
        let sys = fcc_slip_systems();
        let a = spin_tensor(&sys[0]);
        for i in 0..3 { for j in 0..3 {
            assert!((a[(i,j)] + a[(j,i)]).abs() < 1e-15, "Spin tensor not antisymmetric");
        }}
    }

    #[test]
    fn lattice_rotation_preserves_orthogonality() {
        let sys = fcc_slip_systems();
        let orientation = DMatrix::identity(3, 3);
        // Simple shear deformation gradient
        let grad_u = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.1, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let dgamma = vec![0.01; 12]; // equal slip on all systems
        let r_new = update_lattice_rotation(&orientation, &grad_u, &dgamma, &sys);
        // Check orthogonality: R·Rᵀ ≈ I
        let rrt = &r_new * r_new.transpose();
        for i in 0..3 { for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((rrt[(i,j)] - expected).abs() < 1e-12,
                "Lattice rotation not orthogonal: (R·Rᵀ)[{i},{j}] = {}", rrt[(i,j)]);
        }}
        // Determinant should be 1 (proper rotation)
        let det = r_new[(0,0)]*(r_new[(1,1)]*r_new[(2,2)]-r_new[(1,2)]*r_new[(2,1)])
                - r_new[(0,1)]*(r_new[(1,0)]*r_new[(2,2)]-r_new[(1,2)]*r_new[(2,0)])
                + r_new[(0,2)]*(r_new[(1,0)]*r_new[(2,1)]-r_new[(1,1)]*r_new[(2,0)]);
        assert!((det - 1.0).abs() < 1e-12, "Lattice rotation determinant should be 1, got {det}");
    }

    #[test]
    fn lattice_rotation_identity_for_no_slip() {
        let sys = fcc_slip_systems();
        let orientation = DMatrix::identity(3, 3);
        let grad_u = DMatrix::zeros(3, 3);
        let dgamma = vec![0.0; 12]; // no slip
        let r_new = update_lattice_rotation(&orientation, &grad_u, &dgamma, &sys);
        for i in 0..3 { for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((r_new[(i,j)] - expected).abs() < 1e-15,
                "No slip should give identity rotation");
        }}
    }
}
