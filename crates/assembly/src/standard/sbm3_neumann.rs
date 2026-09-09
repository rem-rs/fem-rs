//! Shifted Boundary Method (SBM) Neumann face integrators — 1:1 serial port
//! of the MFEM `miniapps/shifted` second-generation SBM kernels
//! (`sbm_solver.{hpp,cpp}`): [`Sbm3NeumannIntegrator`] (MFEM
//! `SBM2NeumannIntegrator`) and [`Sbm3NeumannLFIntegrator`] (MFEM
//! `SBM2NeumannLFIntegrator`).
//!
//! The Neumann data `t_n = ∇u·n̂` is shifted from the true boundary point
//! `x̂ + D` (D = distance vector to the true boundary, `n̂` the true-boundary
//! normal) onto the surrogate boundary with normal `n` (surface measure folded
//! in, MFEM `CalcOrtho`):
//!
//! ```text
//! A(u,w) = -⟨∇u·n, w⟩ + ⟨(∇u + ∇(∇u)·d + h.o.t.)·n̂ (n̂·n), w⟩
//! L(w)   = ⟨(n̂·n) t_n, w⟩
//! ```
//!
//! The higher-order Taylor terms use the discrete second-derivative operators
//! `D_k D_c` (MFEM `ProjectGrad` products) with `Factorial(i) = (i+1)!`
//! (note the different normalisation w.r.t. the Dirichlet integrators).
//! Neumann conditions require `ho_terms >= 1` (MFEM verifies this in
//! `diffusion.cpp`).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use super::sbm3_dirichlet::{
    calc_ortho, dkphi_dxk, discrete_grad_matrices, dot, dot_row, elem_centroid, sub,
    surrogate_face_list, DistVec, ElemGeometry, FaceGeometry,
};

/// MFEM `Factorial` vector for the Neumann bilinear integrator:
/// `F(0) = 1, F(i) = F(i-1)·(i+1)` → `F(i) = (i+1)!`.
fn factorial_neumann(nterms: usize) -> Vec<f64> {
    let mut f = Vec::with_capacity(nterms);
    if nterms > 0 {
        f.push(1.0);
        for i in 1..nterms {
            f.push(f[i - 1] * (i as f64 + 1.0));
        }
    }
    f
}

// ─── Neumann bilinear form integrator ────────────────────────────────────────

/// SBM Neumann form (MFEM `SBM2NeumannIntegrator`): assembles the face
/// contributions of every surrogate-boundary face onto the DOFs of the active
/// (cut) element.
pub struct Sbm3NeumannIntegrator<'a, M: MeshTopology, S: FESpace<Mesh = M>> {
    /// Solution H¹ space.
    pub space: &'a S,
    /// Distance vector to the true boundary (component-major DOFs, length
    /// `dim · n_dofs`; MFEM `VectorCoefficient vD`).
    pub dist: &'a [f64],
    /// True-boundary normal field evaluated at the *surrogate* point `x`;
    /// the shift to `x + D` is applied internally (MFEM
    /// `ShiftedVectorFunctionCoefficient::Eval(T, ip, D)`).
    pub nhat: &'a (dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    /// Element markers from [`crate::dist_solver::ShiftedFaceMarker::mark_elements`].
    pub elem_marker: &'a [i32],
    /// Include cut cells in the surrogate domain (MFEM `-cut`; the Neumann
    /// miniapp paths run with `include_cut_cell = false`).
    pub include_cut_cell: bool,
    /// Extra Taylor terms beyond the leading `∇(∇u)·d` term (MFEM `-ho`;
    /// Neumann conditions need at least 1).
    pub ho_terms: usize,
}

impl<M, S> Sbm3NeumannIntegrator<'_, M, S>
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    /// Assemble `A += Σ_faces E_activeᵀ K_face E_active`.
    pub fn assemble_bilinear(&self) -> CsrMatrix<f64> {
        use super::sbm3_dirichlet::q_hess_dot_d_generic;
        let mesh = self.space.mesh();
        let n = self.space.n_dofs();
        let dim = mesh.topological_dim() as usize;
        let mut coo = CooMatrix::<f64>::new(n, n);

        let faces = surrogate_face_list(mesh, self.elem_marker, self.include_cut_cell);
        let dist = DistVec { dofs: self.dist, n_scalar: n };

        for face in &faces {
            let e = face.elem;
            let order = self.space.element_order(e);
            let sre = crate::assembler::ref_elem_vol_for_space(
                self.space,
                mesh.element_type(e),
                order,
            );
            let nd = sre.n_dofs();
            let edofs = self.space.element_dofs(e);

            let geo = ElemGeometry::new(mesh, e);
            let fgeo = FaceGeometry::new(face.nodes.len(), dim);
            let quad = fgeo.re.quadrature((4 * order).min(u8::MAX));

            let (dkphi, factorial) = if self.ho_terms > 0 {
                let dm = discrete_grad_matrices(&geo, sre.as_ref());
                let dk = dkphi_dxk(&dm, nd, dim, self.ho_terms);
                (Some(dk), Some(factorial_neumann(self.ho_terms)))
            } else {
                (None, None)
            };

            let mut phi = vec![0.0_f64; nd];
            let mut gref = vec![0.0_f64; nd * dim];
            let mut gphys = vec![0.0_f64; nd * dim];
            let mut k_face = vec![0.0_f64; nd * nd];

            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                let x = fgeo.map(mesh, &face.nodes, xi);
                let jf = fgeo.jacobian(mesh, &face.nodes, xi);
                let mut nor = calc_ortho(&jf, dim);
                let xc = elem_centroid(mesh, mesh.element_nodes(e));
                if dot(&nor, &sub(&x, &xc)) < 0.0 {
                    for nv in nor.iter_mut() {
                        *nv = -*nv;
                    }
                }

                let xi_e = geo.invert(&x);
                let (_je, _det_e, jit) = geo.jacobian(&xi_e);
                sre.eval_basis(&xi_e, &mut phi);
                sre.eval_grad_basis(&xi_e, &mut gref);
                super::sbm3_dirichlet::transform_grads(&jit, &gref, &mut gphys, nd, dim);

                let d_vec = dist.eval(self.space, e, &xi_e);
                let mut x_true = x.clone();
                for (xt, dv) in x_true.iter_mut().zip(d_vec.iter()) {
                    *xt += dv;
                }
                let n_hat = (self.nhat)(&x_true);

                // dshapedn_i = ipw · (∇φ_i · nor): the MFEM chain
                // adjJ·(ipw/detJ)·nor cancels detJ, measure stays in nor.
                let dshapedn: Vec<f64> = (0..nd)
                    .map(|i| ipw * dot_row(&gphys, i, &nor, dim))
                    .collect();

                // Term 2: -⟨∇u·n, w⟩
                for i in 0..nd {
                    for j in 0..nd {
                        k_face[i * nd + j] -= phi[i] * dshapedn[j];
                    }
                }

                // Term 3: +⟨(∇u·n̂)(n̂·n), w⟩
                let n_dot = dot(&nor, &n_hat);
                let dshapedn_hat: Vec<f64> = (0..nd)
                    .map(|i| ipw * dot_row(&gphys, i, &n_hat, dim) * n_dot)
                    .collect();
                for i in 0..nd {
                    for j in 0..nd {
                        k_face[i * nd + j] += phi[i] * dshapedn_hat[j];
                    }
                }

                // Term 4: +⟨(∇(∇u)·d + h.o.t.)·n̂ (n̂·n), w⟩
                // (MFEM AddMult_a_VWt(shape, ipw·n_dot·q_hess)).
                if let (Some(dk), Some(f)) = (&dkphi, &factorial) {
                    let q = q_hess_dot_d_generic(dk, f, &phi, &d_vec, &n_hat, nd, dim);
                    for i in 0..nd {
                        for (j, qj) in q.iter().enumerate() {
                            k_face[i * nd + j] += ipw * n_dot * phi[i] * qj;
                        }
                    }
                }
            }

            for (i, &di) in edofs.iter().enumerate() {
                for (j, &dj) in edofs.iter().enumerate() {
                    coo.add(di as usize, dj as usize, k_face[i * nd + j]);
                }
            }
        }

        coo.into_csr()
    }
}

// ─── Neumann linear form integrator ──────────────────────────────────────────

/// SBM Neumann right-hand side (MFEM `SBM2NeumannLFIntegrator` with a
/// `ShiftedFunctionCoefficient uN` evaluated at the true-boundary point
/// `x + D`).
pub struct Sbm3NeumannLFIntegrator<'a, M: MeshTopology, S: FESpace<Mesh = M>> {
    /// Solution H¹ space.
    pub space: &'a S,
    /// Distance vector to the true boundary (component-major DOFs).
    pub dist: &'a [f64],
    /// True-boundary normal field (shifted evaluation applied internally).
    pub nhat: &'a (dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    /// Neumann traction `t_n` (shifted evaluation applied internally).
    pub tn: &'a (dyn Fn(&[f64]) -> f64 + Send + Sync),
    /// Element markers.
    pub elem_marker: &'a [i32],
    /// Include cut cells in the surrogate domain.
    pub include_cut_cell: bool,
    /// Unused extra terms (kept for API parity with the bilinear integrator;
    /// the LF carries no derivative terms in the C++ implementation).
    pub ho_terms: usize,
}

impl<M, S> Sbm3NeumannLFIntegrator<'_, M, S>
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    /// Assemble `b += Σ_faces E_activeᵀ l_face` with
    /// `l_face = ⟨(n̂·n) t_n, w⟩`.
    pub fn assemble_linear(&self) -> Vec<f64> {
        let mesh = self.space.mesh();
        let n = self.space.n_dofs();
        let dim = mesh.topological_dim() as usize;
        let mut b = vec![0.0_f64; n];

        let faces = surrogate_face_list(mesh, self.elem_marker, self.include_cut_cell);
        let dist = DistVec { dofs: self.dist, n_scalar: n };

        for face in &faces {
            let e = face.elem;
            let order = self.space.element_order(e);
            let sre = crate::assembler::ref_elem_vol_for_space(
                self.space,
                mesh.element_type(e),
                order,
            );
            let nd = sre.n_dofs();
            let edofs = self.space.element_dofs(e);

            let geo = ElemGeometry::new(mesh, e);
            let fgeo = FaceGeometry::new(face.nodes.len(), dim);
            let quad = fgeo.re.quadrature((4 * order).min(u8::MAX));

            let mut phi = vec![0.0_f64; nd];

            for (q, xi) in quad.points.iter().enumerate() {
                let ipw = quad.weights[q];
                let x = fgeo.map(mesh, &face.nodes, xi);
                let jf = fgeo.jacobian(mesh, &face.nodes, xi);
                let mut nor = calc_ortho(&jf, dim);
                let xc = elem_centroid(mesh, mesh.element_nodes(e));
                if dot(&nor, &sub(&x, &xc)) < 0.0 {
                    for nv in nor.iter_mut() {
                        *nv = -*nv;
                    }
                }

                let xi_e = geo.invert(&x);
                let (_je, _det_e, _jit) = geo.jacobian(&xi_e);
                sre.eval_basis(&xi_e, &mut phi);

                let d_vec = dist.eval(self.space, e, &xi_e);
                let mut x_true = x.clone();
                for (xt, dv) in x_true.iter_mut().zip(d_vec.iter()) {
                    *xt += dv;
                }
                let n_hat = (self.nhat)(&x_true);
                let tn = (self.tn)(&x_true);

                let w = ipw * n_dot_ntilde(&nor, &n_hat) * tn;
                for (i, &dof) in edofs.iter().enumerate() {
                    b[dof as usize] += w * phi[i];
                }
            }
        }

        b
    }
}

fn n_dot_ntilde(nor: &[f64], nhat: &[f64]) -> f64 {
    dot(nor, nhat)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// Split marker: {x < 0.5} INSIDE, rest CUT (mirrors the Dirichlet tests;
    /// the Neumann miniapp paths apply the faces on the same surrogate
    /// boundary geometry).
    fn marker_split(mesh: &Mesh<2>) -> Vec<i32> {
        mesh.elem_iter()
            .map(|e| {
                let nodes = mesh.element_nodes(e);
                let cx: f64 = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>()
                    / nodes.len() as f64;
                if cx <= 0.5 {
                    0
                } else {
                    2
                }
            })
            .collect()
    }

    /// Planar-face reduction: with a *zero* distance field and the
    /// true-boundary normal equal to the surrogate normal n, the Neumann
    /// bilinear terms cancel exactly for any ho_terms
    /// (`-⟨∇u·n,w⟩ + ⟨(∇u·n)(n·n),w⟩ + 0 = 0` for D = 0), so the assembled
    /// face matrix must vanish to round-off.
    #[test]
    fn neumann_bilinear_vanishes_on_aligned_face() {
        let space = H1Space::new(Mesh::<2>::unit_square_quad(4), 2);
        let marker = marker_split(space.mesh());
        let nd = space.n_dofs();
        let dm = space.dof_manager();

        let mut dist = vec![0.0_f64; 2 * nd];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            dist[dof as usize] = 0.5 - x[0]; // zero on the surrogate face x=0.5
        }
        // n̂ = +x (the outward normal of the active strip at x = 0.5).
        let nhat = |_x: &[f64]| vec![1.0_f64, 0.0];

        let nb = Sbm3NeumannIntegrator {
            space: &space,
            dist: &dist,
            nhat: &nhat,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 1,
        };
        let a = nb.assemble_bilinear().to_dense();
        let mut max_abs = 0.0_f64;
        for v in a.iter() {
            max_abs = max_abs.max(v.abs());
        }
        println!("SBM Neumann aligned-face residual: {max_abs:.3e}");
        assert!(max_abs < 1e-11, "aligned Neumann face not annihilated: {max_abs}");
    }

    /// Neumann LF with n̂ = n and constant traction t equals the standard
    /// boundary load `∫_Σ t w dγ` — compared against a direct line-integral
    /// assembly.
    #[test]
    fn neumann_lf_matches_boundary_load() {
        let space = H1Space::new(Mesh::<2>::unit_square_quad(5), 1);
        let marker = marker_split(space.mesh());
        let nd = space.n_dofs();
        let dm = space.dof_manager();

        let mut dist = vec![0.0_f64; 2 * nd];
        for dof in 0..nd as u32 {
            let x = dm.dof_coord(dof);
            dist[dof as usize] = 0.5 - x[0];
        }
        let nhat = |_x: &[f64]| vec![1.0_f64, 0.0];
        let tn = |x: &[f64]| 2.0 * x[1] + 1.0; // t(x) = 2y + 1

        let lf = Sbm3NeumannLFIntegrator {
            space: &space,
            dist: &dist,
            nhat: &nhat,
            tn: &tn,
            elem_marker: &marker,
            include_cut_cell: false,
            ho_terms: 0,
        };
        let b = lf.assemble_linear();

        // Direct: for every surrogate face, ∫ t(γ(s)) φ_m(γ(s)) |γ'| ds with
        // 3-point Gauss on each straight segment.
        let faces = surrogate_face_list(space.mesh(), &marker, false);
        let mut b_direct = vec![0.0_f64; nd];
        let gs = [-0.7745966692414834_f64, 0.0, 0.7745966692414834];
        let ws = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
        for face in &faces {
            let p0 = space.mesh().node_coords(face.nodes[0]);
            let p1 = space.mesh().node_coords(face.nodes[1]);
            let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
            let len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
            let sre = crate::assembler::ref_elem_vol_for_space(
                &space,
                space.mesh().element_type(face.elem),
                1,
            );
            let geo = ElemGeometry::new(space.mesh(), face.elem);
            let ndl = sre.n_dofs();
            let mut phi = vec![0.0_f64; ndl];
            for (q, &s) in gs.iter().enumerate() {
                let xi = [0.5 * (1.0 + s)];
                let x = [mid[0] + s * 0.5 * (p1[0] - p0[0]), mid[1] + s * 0.5 * (p1[1] - p0[1])];
                let w = ws[q] * len * 0.5;
                let xi_e = geo.invert(&x);
                sre.eval_basis(&xi_e, &mut phi);
                for (i, &dof) in space.element_dofs(face.elem).iter().enumerate() {
                    b_direct[dof as usize] += w * tn(&x) * phi[i];
                }
            }
        }

        let mut max_diff = 0.0_f64;
        for (a, bv) in b.iter().zip(b_direct.iter()) {
            max_diff = max_diff.max((a - bv).abs());
        }
        println!("SBM Neumann LF vs direct boundary load max diff: {max_diff:.3e}");
        assert!(max_diff < 1e-12, "Neumann LF mismatch: {max_diff}");
        let _ = &dm;
    }
}
