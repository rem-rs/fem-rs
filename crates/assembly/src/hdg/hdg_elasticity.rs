//! HDG for linear elasticity: −μΔu − (λ+μ)∇(∇·u) = f.
//!
//! Supports P1 (6 DOFs/tri) and P2 (12 DOFs/tri) with Pk skeleton trace.
//! The element build is dimension-generic: 2-D (tri/seg) and 3-D (tet/tri)
//! share one code path for volume integrals, ∂K integrals and static
//! condensation (used by both the assembly pass and the reconstruction pass).

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::lagrange::{TriPk, TetPk, SegPk};
use fem_element::{QuadratureRule, ReferenceElement};

fn npe(dim: usize, k: usize) -> usize {
    match dim {
        1 => k+1,
        2 => (k+1)*(k+2)/2,
        3 => (k+1)*(k+2)*(k+3)/6,
        _ => unreachable!()
    }
}

#[derive(Debug)]
pub struct HdgElasticityResult {
    pub u: Vec<f64>,
    pub lambda: Vec<f64>,
}

/// Solve HDG elasticity with P1 (backward compat).
pub fn solve_hdg_elasticity<M, F>(
    mesh: M, source: F, mu: f64, lambda: f64,
) -> HdgElasticityResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    solve_hdg_elasticity_order(mesh, source, mu, lambda, 1)
}

/// Solve HDG elasticity with configurable velocity order (1=P1, 2=P2).
pub fn solve_hdg_elasticity_order<M, F>(
    mesh: M, source: F, mu: f64, lambda: f64, vel_order: u8,
) -> HdgElasticityResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let tau = 2.0 * mu;
    let vo = vel_order as usize;

    let n_vel_b = npe(dim, vo);       // scalar DOFs per velocity component
    let u_dpe = n_vel_b * dim;         // velocity DOFs per element
    let n_u = n_elems * u_dpe;

    let ref_elem: Box<dyn ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(vo)),
        3 => Box::new(TetPk::new(vo)),
        _ => unreachable!(),
    };
    let geo_elem: Box<dyn ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(1)),
        3 => Box::new(TetPk::new(1)),
        _ => unreachable!(),
    };
    let geo_n = geo_elem.n_dofs();

    let (face_ref, sk_dpe, n_sk_b): (Box<dyn ReferenceElement>, usize, usize) = match dim {
        2 => {
            let fr: Box<dyn ReferenceElement> = Box::new(SegPk::new(vo));
            let sd = (vo + 1) * dim;
            let nb = vo + 1;
            (fr, sd, nb)
        }
        3 => {
            let fr: Box<dyn ReferenceElement> = Box::new(TriPk::new(vo));
            let sd = ((vo + 1) * (vo + 2) / 2) * dim;
            let nb = (vo + 1) * (vo + 2) / 2;
            (fr, sd, nb)
        }
        _ => unreachable!(),
    };

    // Local face vertex tables. `face_xi_ref` and `face_size` MUST use the
    // same convention: face basis dof ld peaks at `local_faces[lf][ld]`.
    let local_faces: Vec<Vec<u32>> = match dim {
        2 => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
        3 => vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        _ => unreachable!(),
    };

    let qr_vol = ref_elem.quadrature((2 * vo) as u8);
    let qr_face = face_ref.quadrature((2 * vo) as u8);

    // Build face list
    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();
    for e in 0..n_elems as u32 {
        let en = mesh.element_nodes(e);
        for f in &local_faces {
            let mut k: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect();
            k.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(k) {
                Entry::Vacant(e) => { e.insert((f.clone(), false)); }
                Entry::Occupied(mut e) => { e.get_mut().1 = true; }
            }
        }
    }
    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let n_faces = face_list.len();
    let n_lambda = face_list.iter().filter(|(_, interior)| *interior).count() * sk_dpe;

    let mut lam_off: Vec<Option<usize>> = vec![None; n_faces];
    {
        let mut nxt = 0;
        for (i, (_, interior)) in face_list.iter().enumerate() {
            if *interior {
                lam_off[i] = Some(nxt);
                nxt += sk_dpe;
            }
        }
    }

    let prob = HdgProblem {
        mesh: &mesh,
        dim,
        tau,
        mu,
        lambda,
        n_vel_b,
        u_dpe,
        geo_n,
        sk_dpe,
        n_sk_b,
        local_faces: &local_faces,
        face_list: &face_list,
        lam_off: &lam_off,
        qr_vol: &qr_vol,
        qr_face: &qr_face,
        ref_elem: ref_elem.as_ref(),
        geo_elem: geo_elem.as_ref(),
        face_ref: face_ref.as_ref(),
        source: &source,
    };

    let mut sk_coo = CooMatrix::new(n_lambda, n_lambda);
    let mut sk_rhs = vec![0.0; n_lambda];

    // Pass 1: assemble the skeleton system by static condensation.
    for e in 0..n_elems as u32 {
        let cs = prob.build_condensed(e);

        // Assemble: K = B^T A^{-1} B, rhs g = B^T u0
        for s in 0..cs.ns {
            let lf_idx = s / sk_dpe;
            let ld = s % sk_dpe;
            let Some(loff) = cs.face_off[lf_idx] else { continue; };
            let lam_s = loff + ld;

            // g_s += B^T u0
            let mut bt_u0 = 0.0;
            for i in 0..cs.nu {
                bt_u0 += cs.b_mat[i * cs.ns + s] * cs.u0[i];
            }
            sk_rhs[lam_s] += bt_u0;

            for t in 0..cs.ns {
                let lf_idx2 = t / sk_dpe;
                let ld2 = t % sk_dpe;
                let Some(loff2) = cs.face_off[lf_idx2] else { continue; };
                let lam_t = loff2 + ld2;

                // K_st = B^T u_lam[:, t] = Σ_i B[i,s] * u_lam[i,t]
                let mut kst = 0.0;
                for i in 0..cs.nu {
                    kst += cs.b_mat[i * cs.ns + s] * cs.u_lam[i * cs.ns + t];
                }

                sk_coo.add(lam_s, lam_t, kst);
            }
        }
    }

    // Solve global skeleton system
    if n_lambda == 0 {
        return HdgElasticityResult {
            u: vec![0.0; n_u],
            lambda: vec![],
        };
    }
    let sk_csr = sk_coo.into_csr();
    let mut lambda_sol = vec![0.0; n_lambda];
    let cfg = SolverConfig {
        max_iter: 2000,
        atol: 1e-12,
        rtol: 1e-12,
        ..Default::default()
    };
    match fem_solver::solve_cg(&sk_csr, &sk_rhs, &mut lambda_sol, &cfg) {
        Ok(_) | Err(_) => {}
    }

    // Pass 2: reconstruct the bulk solution u = u0 + u_lam · λ
    let mut u_bulk = vec![0.0; n_u];
    for e in 0..n_elems as u32 {
        let cs = prob.build_condensed(e);
        let base_u = e as usize * u_dpe;
        u_bulk[base_u..base_u + cs.nu].copy_from_slice(&cs.u0[..cs.nu]);

        // u_i = u0_i + Σ_s u_lam[i,s] · λ_s (u_lam = −A⁻¹B was built in pass 1)
        for s in 0..cs.ns {
            let lf_idx = s / sk_dpe;
            let ld = s % sk_dpe;
            let Some(loff) = cs.face_off[lf_idx] else { continue; };
            let lam_val = lambda_sol[loff + ld];
            for i in 0..cs.nu {
                u_bulk[base_u + i] += cs.u_lam[i * cs.ns + s] * lam_val;
            }
        }
    }

    HdgElasticityResult {
        u: u_bulk,
        lambda: lambda_sol,
    }
}

/// Immutable per-problem data shared by every per-element build.
struct HdgProblem<'a, M: MeshTopology, F: Fn(&[f64]) -> Vec<f64>> {
    mesh: &'a M,
    dim: usize,
    tau: f64,
    mu: f64,
    lambda: f64,
    n_vel_b: usize,
    u_dpe: usize,
    geo_n: usize,
    sk_dpe: usize,
    n_sk_b: usize,
    local_faces: &'a [Vec<u32>],
    face_list: &'a [(Vec<u32>, bool)],
    lam_off: &'a [Option<usize>],
    qr_vol: &'a QuadratureRule,
    qr_face: &'a QuadratureRule,
    ref_elem: &'a dyn ReferenceElement,
    geo_elem: &'a dyn ReferenceElement,
    face_ref: &'a dyn ReferenceElement,
    source: &'a F,
}

/// Result of the per-element static condensation.
///
/// Local layout: λ block dof `s = lf_idx * sk_dpe + ld` (global offset via
/// `face_off[lf_idx]`), velocity dofs `i * dim + a`.
struct CondensedElement {
    /// Global λ offset per local face; `None` on mesh-boundary faces.
    face_off: Vec<Option<usize>>,
    nu: usize,
    ns: usize,
    /// u0 = A⁻¹ f.
    u0: Vec<f64>,
    /// u_lam = −A⁻¹ B (velocity response to each local λ dof).
    u_lam: Vec<f64>,
    /// B = τ∫_{∂K} φ ψ_λ (interior faces only; columns in local layout).
    b_mat: Vec<f64>,
}

/// Image of face point `fxi` on the element reference domain for local face
/// `lf_idx`, chosen so that face basis dof `psi[ld]` peaks at the local face
/// vertex `local_faces[lf_idx][ld]` (same convention as `face_size`).
fn face_xi_ref(dim: usize, lf_idx: usize, fxi: &[f64]) -> Vec<f64> {
    let s = fxi[0];
    match (dim, lf_idx) {
        (2, 0) => vec![s, 0.0],                    // face [0,1]: v0↔ψ0, v1↔ψ1
        (2, 1) => vec![1.0 - s, s],                // face [1,2]
        (2, 2) => vec![0.0, 1.0 - s],              // face [0,2]
        (3, 0) => vec![s, fxi[1], 0.0],            // face [0,1,2]
        (3, 1) => vec![s, 0.0, fxi[1]],            // face [0,1,3]
        (3, 2) => vec![0.0, s, fxi[1]],            // face [0,2,3]
        (3, 3) => vec![1.0 - s - fxi[1], s, fxi[1]], // face [1,2,3]
        _ => unreachable!(),
    }
}

/// Determinant of the dim×dim isoparametric Jacobian (row = physical, col = reference).
fn jacobian_det(jac: &[Vec<f64>], dim: usize) -> f64 {
    match dim {
        2 => jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0],
        3 => {
            jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0])
        }
        _ => unreachable!(),
    }
}

/// Row-major J⁻ᵀ = adj(J)/det, so that ∇_x φ = J⁻ᵀ ∇_ξ φ component-wise:
/// ∂φ/∂x_c = Σ_d inv_t[d*dim + c] · ∂φ/∂ξ_d.
fn jacobian_inv_t(jac: &[Vec<f64>], dim: usize, id: f64) -> Vec<f64> {
    match dim {
        2 => {
            let (j00, j01, j10, j11) = (jac[0][0], jac[0][1], jac[1][0], jac[1][1]);
            vec![j11 * id, -j10 * id, -j01 * id, j00 * id]
        }
        3 => {
            let (j00, j01, j02, j10, j11, j12, j20, j21, j22) = (
                jac[0][0], jac[0][1], jac[0][2],
                jac[1][0], jac[1][1], jac[1][2],
                jac[2][0], jac[2][1], jac[2][2],
            );
            vec![
                (j11*j22 - j12*j21) * id,
                (j02*j21 - j01*j22) * id,
                (j01*j12 - j02*j11) * id,
                (j12*j20 - j10*j22) * id,
                (j00*j22 - j02*j20) * id,
                (j02*j10 - j00*j12) * id,
                (j10*j21 - j11*j20) * id,
                (j01*j20 - j00*j21) * id,
                (j00*j11 - j01*j10) * id,
            ]
        }
        _ => unreachable!(),
    }
}

fn face_size<M: MeshTopology>(mesh: &M, enodes: &[u32], lf_idx: usize, dim: usize) -> f64 {
    if dim == 2 {
        let a = enodes[lf_idx];
        let b = enodes[(lf_idx + 1) % 3];
        let pa = mesh.node_coords(a);
        let pb = mesh.node_coords(b);
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        (dx * dx + dy * dy).sqrt()
    } else {
        // Tet4 face (triangular). Same convention as `local_faces` (3-D):
        // lf_idx 0..3 → vertex triples [0,1,2], [0,1,3], [0,2,3], [1,2,3].
        let tri_faces: [(usize, usize, usize); 4] = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];
        let (ai, bi, ci) = tri_faces[lf_idx];
        let a = mesh.node_coords(enodes[ai]);
        let b = mesh.node_coords(enodes[bi]);
        let c = mesh.node_coords(enodes[ci]);
        let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        0.5 * (nx*nx + ny*ny + nz*nz).sqrt()
    }
}

fn invert_dense(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r;
            }
        }
        if mv < 1e-15 {
            return None;
        }
        if mr != c {
            for j in 0..n {
                a.swap(c * n + j, mr * n + j);
                inv.swap(c * n + j, mr * n + j);
            }
        }
        let pv = a[c * n + c];
        let ip = 1.0 / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    Some(inv)
}

impl<'a, M: MeshTopology, F: Fn(&[f64]) -> Vec<f64>> HdgProblem<'a, M, F> {
    /// Map each local face to its entry in the global `face_list`.
    ///
    /// Returns per local face the global λ offset (`None` on boundary faces)
    /// and, for interior faces, the slot permutation: `perm[ld]` = position of
    /// the physical node of local face vertex `ld` inside the canonical
    /// `face_list` node order (fixed by the first element that registered the
    /// face). Both neighbouring elements must write their trace couplings into
    /// the same slot per physical vertex, otherwise the numerical flux is not
    /// single-valued across the face.
    fn locate_faces(&self, en: &[u32]) -> (Vec<Option<usize>>, Vec<Option<Vec<usize>>>) {
        let mut face_off = Vec::with_capacity(self.local_faces.len());
        let mut face_perm: Vec<Option<Vec<usize>>> = Vec::with_capacity(self.local_faces.len());
        for f in self.local_faces {
            let mut key: Vec<u32> = f.iter().map(|&x| en[x as usize]).collect();
            key.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in self.face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.to_vec();
                fk.sort_unstable();
                if fk == key { found = Some(fi); break; }
            }
            match found {
                Some(fi) => {
                    let fnodes = &self.face_list[fi].0;
                    let perm: Vec<usize> = f.iter().map(|&lv| {
                        let node = en[lv as usize];
                        fnodes.iter()
                            .position(|&n| n == node)
                            .expect("face node set must match the located entry")
                    }).collect();
                    face_off.push(self.lam_off[fi]);
                    face_perm.push(Some(perm));
                }
                None => {
                    face_off.push(None);
                    face_perm.push(None);
                }
            }
        }
        (face_off, face_perm)
    }

    /// Build the element volume/∂K integrals and statically condense the
    /// velocity field: A u = f − B λ ⇒ u = u0 + u_lam λ.
    fn build_condensed(&self, e: u32) -> CondensedElement {
        let dim = self.dim;
        let en = self.mesh.element_nodes(e);
        let n_lf = self.local_faces.len();
        let nu = self.u_dpe;
        let ns = n_lf * self.sk_dpe;
        let (face_off, face_perm) = self.locate_faces(en);

        let mut amat = vec![0.0; nu * nu];
        let mut f_u = vec![0.0; nu];
        let mut bmat = vec![0.0; nu * ns];
        let mut phi = vec![0.0; self.n_vel_b];
        let mut grad = vec![0.0; self.n_vel_b * dim];
        let mut psi = vec![0.0; self.n_sk_b];

        // ── Volume integrals ────────────────────────────────────────────
        for q in 0..self.qr_vol.n_points() {
            let xi = &self.qr_vol.points[q];
            let w = self.qr_vol.weights[q];

            self.ref_elem.eval_basis(xi, &mut phi);
            self.ref_elem.eval_grad_basis(xi, &mut grad);

            let mut geo_grad = vec![0.0; self.geo_n * dim];
            self.geo_elem.eval_grad_basis(xi, &mut geo_grad);
            let mut jac = vec![vec![0.0; dim]; dim];
            for i in 0..dim {
                for d in 0..dim {
                    for k in 0..self.geo_n {
                        jac[i][d] += self.mesh.node_coords(en[k])[i] * geo_grad[k * dim + d];
                    }
                }
            }
            let det_j = jacobian_det(&jac, dim);
            let vol = (w * det_j).abs();
            let id = 1.0 / det_j;
            let inv_t = jacobian_inv_t(&jac, dim, id);

            // Physical gradients ∇_x φ = J⁻ᵀ ∇_ξ φ
            let mut gp = vec![0.0; self.n_vel_b * dim];
            for i in 0..self.n_vel_b {
                for c in 0..dim {
                    let mut acc = 0.0;
                    for d in 0..dim {
                        acc += inv_t[d * dim + c] * grad[i * dim + d];
                    }
                    gp[i * dim + c] = acc;
                }
            }

            // Physical coords for source
            let mut geo_phi = vec![0.0; self.geo_n];
            self.geo_elem.eval_basis(xi, &mut geo_phi);
            let mut xp = vec![0.0; dim];
            for k in 0..self.geo_n {
                let c = self.mesh.node_coords(en[k]);
                for i in 0..dim {
                    xp[i] += geo_phi[k] * c[i];
                }
            }
            let fv = (self.source)(&xp);

            // A += μ∫∇u·∇w (component-wise: δ_ab Σ_c ∂φ_i/∂x_c · ∂φ_j/∂x_c)
            for a in 0..dim {
                for i in 0..self.n_vel_b {
                    for j in 0..self.n_vel_b {
                        let mut d = 0.0;
                        for b in 0..dim {
                            d += gp[i * dim + b] * gp[j * dim + b];
                        }
                        amat[(i * dim + a) * nu + (j * dim + a)] += self.mu * vol * d;
                    }
                }
            }

            // A += (λ+μ)∫(∇·u)(∇·w) (grad-div coupling)
            // (∇·u) = Σ_c ∂u_c/∂x_c, (∇·w) = Σ_c ∂w_c/∂x_c
            for a in 0..dim {
                for i in 0..self.n_vel_b {
                    for b in 0..dim {
                        for j in 0..self.n_vel_b {
                            amat[(i * dim + a) * nu + (j * dim + b)] +=
                                (self.lambda + self.mu) * vol * gp[i * dim + a] * gp[j * dim + b];
                        }
                    }
                }
            }

            // f += ∫ f·φ
            for a in 0..dim {
                for i in 0..self.n_vel_b {
                    f_u[i * dim + a] += vol * phi[i] * fv[a];
                }
            }
        }

        // ── ∂K integrals: τ∫φ·φ on every face, τ∫φ·ψ_λ on interior faces ──
        let n_qp_face = self.qr_face.n_points();
        for (lf_idx, _lf) in self.local_faces.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &self.qr_face.points[fq];
                let fw = self.qr_face.weights[fq];
                let xi_ref = face_xi_ref(dim, lf_idx, fxi);
                self.ref_elem.eval_basis(&xi_ref, &mut phi);
                self.face_ref.eval_basis(fxi, &mut psi);
                let fj = face_size(self.mesh, en, lf_idx, dim);
                let wf = fw * fj;

                // τ∫φ·φ on ∂K (boundary faces included)
                for a in 0..dim {
                    for i in 0..self.n_vel_b {
                        for j in 0..self.n_vel_b {
                            amat[(i * dim + a) * nu + (j * dim + a)] += self.tau * wf * phi[i] * phi[j];
                        }
                    }
                }

                // τ∫φ·ψ_λ on interior faces (B, local block columns). The slot
                // permutation keeps λ columns aligned with the canonical
                // face_list vertex order shared by both neighbours.
                if face_off[lf_idx].is_some() {
                    let perm = face_perm[lf_idx]
                        .as_ref()
                        .expect("interior face must carry a slot permutation");
                    let base = lf_idx * self.sk_dpe;
                    for a in 0..dim {
                        for i in 0..self.n_vel_b {
                            let dof_row = i * dim + a;
                            for (ld, &slot) in perm.iter().enumerate() {
                                // ψ[ld] peaks at local face vertex ld; slot is
                                // that vertex's position in the canonical order.
                                let lam_col = base + slot * dim + a;
                                bmat[dof_row * ns + lam_col] += self.tau * wf * phi[i] * psi[ld];
                            }
                        }
                    }
                }
            }
        }

        // ── Static condensation: eliminate u in terms of λ ──────────────
        let a_inv = invert_dense(&amat, nu).unwrap_or_else(|| {
            let s: Vec<f64> = amat.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, nu).unwrap_or(vec![0.0; nu * nu])
        });

        // u0 = A⁻¹ f
        let mut u0 = vec![0.0; nu];
        for i in 0..nu {
            for j in 0..nu {
                u0[i] += a_inv[i * nu + j] * f_u[j];
            }
        }

        // u_lam = −A⁻¹ B (response to each λ dof)
        let mut u_lam = vec![0.0; nu * ns];
        for i in 0..nu {
            for s in 0..ns {
                let mut v = 0.0;
                for j in 0..nu {
                    v += a_inv[i * nu + j] * bmat[j * ns + s];
                }
                u_lam[i * ns + s] = -v;
            }
        }

        CondensedElement { face_off, nu, ns, u0, u_lam, b_mat: bmat }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn hdg_elasticity_2d_finite() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let source = |_: &[f64]| vec![0.0, 0.0];
        let result = solve_hdg_elasticity(mesh, source, 1.0, 1.0);
        for &v in &result.u {
            assert!(v.is_finite());
        }
        for &v in &result.lambda {
            assert!(v.is_finite());
        }
        assert!(result.lambda.len() > 0);
    }

    #[test]
    fn hdg_elasticity_2d_p2_finite() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let result = solve_hdg_elasticity_order(mesh, |_| vec![0.0,0.0], 1.0, 1.0, 2);
        assert!(result.u.iter().all(|v|v.is_finite()) && result.lambda.iter().all(|v|v.is_finite()));
    }

    #[test]
    fn hdg_elasticity_nonzero_source() {
        use std::f64::consts::PI;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let source = |x: &[f64]| vec![
            2.0 * (PI * x[0]).sin() * (PI * x[1]).sin(),
            2.0 * (PI * x[0]).cos() * (PI * x[1]).cos(),
        ];
        let result = solve_hdg_elasticity(mesh, source, 1.0, 1.0);
        for &v in &result.u {
            assert!(v.is_finite());
        }
        assert!(result.u.len() > 0);
    }

    #[test]
    fn hdg_elasticity_3d_finite() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let result = solve_hdg_elasticity_order(mesh, |_| vec![0.0,0.0,0.0], 1.0, 1.0, 1);
        assert!(result.u.iter().all(|v|v.is_finite()), "u has non-finite values");
        assert!(result.lambda.iter().all(|v|v.is_finite()), "lambda has non-finite values");
    }
}
