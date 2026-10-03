//! Weak Galerkin (WG) finite element method for Maxwell equations.
//!
//! Reference: Wang & Ye (2013), "A weak Galerkin FEM for Maxwell equations".
//! Bilinear form: a_h(E,v) = (κ ∇_w × E, ∇_w × v)_T + s(E,v)
//!
//! E ∈ V_h = Nédélec order k,  flux Σ_h = [P_{k-1}]^d.
//! Face stabilizer s(·,·) penalizes tangential jumps.
//!
//! The face stabilizer's geometry (measure, physical point, element reference
//! point) comes from the family's shared isoparametric face path
//! (`super::wg_face_point` / `super::wg_face_measure`, i.e.
//! `dg_base::face_point_geom`); see [`super`] for what the pre-D810-1 chord
//! route got wrong.  The **volume** path (D814-3) likewise takes its geometry
//! from the family's shared isoparametric source —
//! `fem_mesh::transformation::element_jacobian_at`, i.e. MFEM
//! `ElementTransformation::Jacobian()` — so body and face terms see the same
//! element on a curved mesh.  Truth: `tmp/d87b/d814_probe.cpp` +
//! `crates/assembly/tests/d831_wg_volume_geometry.rs`.
//!
//! The dof tables are **signed** throughout (D1080): the volume
//! `C_wᵀM_Σ⁻¹C_w` carries the H(curl) `element_signs` on the `C_w` rows and
//! each face-penalty block is scattered with its own element's signs — MFEM's
//! signed `GetElementDofs`/`GetElementVDofs` + `SparseMatrix::AddSubMatrix`
//! product-of-signs scatter — and the volume curl is the *physical* curl of
//! the mapped basis (`curl̂/detJ`, `(J·curl̂)/detJ`; MFEM
//! `CalcPhysCurlShape`).  Truth:
//! `crates/assembly/tests/d109_d1080_wg_maxwell_signs.rs`.

use nalgebra::DMatrix;
use fem_element::{
    ReferenceElement, VectorReferenceElement,
    lagrange::{TriPk, TetPk},
    nedelec::{TriNDk, TetNDk},
};
use fem_element::quadrature::{tri_rule, tet_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::element_jacobian_at;
use fem_space::fe_space::FESpace;

use super::{wg_boundary_face_map, wg_face_measure, wg_face_point, wg_face_rule};

// ─── Weak curl matrix ─────────────────────────────────────────────────────
// C_w[i,j] = ∫_T (∇_w × φ_j) · σ_i dV  where φ_j ∈ V_h (Nédélec), σ_i ∈ Σ_h
// M_Σ[m,n] = ∫_T σ_m · σ_n dV  (flux mass matrix, used to recover curl)
//
// D1080: the rows of `C_w` are the element's H(curl) dofs, so the weak-curl
// coefficients of the *global* basis are the **signed** rows
// `S·C_w` (`S = diag(element_signs)`): MFEM's element scatter multiplies every
// entry by the product of the row/column signs of the signed
// `GetElementDofs` table (`linalg/sparsemat.cpp:2795-2811`; element matrices
// land at `mat->AddSubMatrix(vdofs_, vdofs_, elmat)`, `fem/bilinearform.cpp:420`)
// — signing the rows here makes `C_wᵀM_Σ⁻¹C_w = (SC_w)ᵀM_Σ⁻¹(SC_w)` exactly
// that signed scatter.  D1080 (same stroke, D1051's second disease): the
// physical curl of the covariantly mapped Nédélec basis is `curl̂/detJ`
// (2-D scalar) resp. `(J·curl̂)/detJ` (3-D) — MFEM `CalcPhysCurlShape` —
// paired with the D696 **signed** weight `ip.weight·Trans.Weight()`, so detJ
// cancels exactly on affine elements instead of leaving detJ²-scaled (2-D) /
// J-less (3-D) entries.

fn weak_curl_matrix<M: MeshTopology>(
    mesh: &M, e: u32, dim: usize, order: usize, quad_order: u8,
    signs: Option<&[f64]>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let nd_elem: Box<dyn VectorReferenceElement> = if dim == 2 {
        Box::new(TriNDk::new(order))
    } else {
        Box::new(TetNDk::new(order))
    };
    let n_v = nd_elem.n_dofs();

    let os = if order > 0 { order - 1 } else { 0 };
    let n_ss: usize;
    let mut ref_s: Option<Box<dyn ReferenceElement>> = None;
    if os == 0 {
        n_ss = 1;
    } else {
        let rs: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                              else { Box::new(TetPk::new(os)) };
        n_ss = rs.n_dofs();
        ref_s = Some(rs);
    }
    let n_s = if dim == 2 { n_ss } else { dim * n_ss };

    let qr = if dim == 2 { tri_rule(quad_order) } else { tet_rule(quad_order) };

    let mut Cw = DMatrix::zeros(n_v, n_s);
    let mut Ms = DMatrix::zeros(n_s, n_s);
    // `eval_curl` native layout: 2-D n_v scalars, 3-D n_v × 3; normalized to
    // physical curl per dof below (2-D scalar, 3-D vector).
    let mut nv_curl = vec![0.0_f64; n_v * dim];
    let mut ps = vec![0.0_f64; n_ss];

    for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
        // D814-3: the body path's geometry is the element's own order-`g`
        // isoparametric map at this QP (MFEM `ElementTransformation::
        // Jacobian()`), not the element's vertex chords (pre-fix `local_jac`,
        // now gone; oracle `d831_wg_volume_geometry`).
        let (jac, _xp) = element_jacobian_at(mesh, e, pt, dim);
        let det_j = jac.determinant();
        if det_j.abs() < 1e-30 { continue; }
        // D696 verdict: **signed** weight for the curl pairing (the det in the
        // physical-curl normalization below cancels it); the flux mass
        // matrix keeps the |detJ| measure.
        let wq_signed = w * det_j;
        let wq_abs = w * det_j.abs();
        nd_elem.eval_curl(pt, &mut nv_curl);

        // Sigma basis values
        if os == 0 {
            ps[0] = 1.0;
        } else {
            let rs = ref_s.as_ref().unwrap();
            rs.eval_basis(pt, &mut ps);
        }

        // Weak curl assembly: flux space sigma is
        //   2D: scalar P_{k-1}  (curl is scalar)
        //   3D: vector [P_{k-1}]^d  (curl is vector)
        if dim == 2 {
            for i in 0..n_v { for j in 0..n_s {
                // Physical scalar curl: nv_curl[i] = reference curl̂_i.
                let curl_phys = nv_curl[i] / det_j;
                Cw[(i, j)] -= wq_signed * curl_phys * ps[j];
            }}
            for p in 0..n_s { for q in 0..n_s {
                Ms[(p, q)] += wq_abs * ps[p] * ps[q];
            }}
        } else {
            for i in 0..n_v { for j in 0..n_s {
                let sc = j / n_ss; let sd = j % n_ss;
                // Physical curl component: (J·curl̂_i)_sc/detJ.
                let curl_phys_sc =
                    (0..3).map(|k| jac[(sc, k)] * nv_curl[i * 3 + k]).sum::<f64>() / det_j;
                Cw[(i, j)] -= wq_signed * curl_phys_sc * ps[sd];
            }}
            for p in 0..n_s { let pc = p / n_ss; let pd = p % n_ss;
                for q in 0..n_s { let qc = q / n_ss; let qd = q % n_ss;
                    if pc == qc { Ms[(p, q)] += wq_abs * ps[pd] * ps[qd]; }
                }
            }
        }
    }
    // D1080: per-dof orientation signs on the H(curl) rows — row i *is* dof i
    // (one curl per row), so the sign multiplies the whole row.
    if let Some(signs) = signs {
        for i in 0..n_v {
            let s = signs.get(i).copied().unwrap_or(1.0);
            if s != 1.0 {
                for j in 0..n_s {
                    Cw[(i, j)] *= s;
                }
            }
        }
    }
    (Cw, Ms)
}

// ─── Face stabilizer (tangential jump for H(curl)) ────────────────────────
//
// D1080: both element blocks of the stabilizer scatter through the element's
// **signed** dof table — MFEM's interior-face path concatenates the signed
// `GetElementVDofs` of *both* elements and adds the (block) face matrix at
// `[vdofs; vdofs2]²` (`fem/bilinearform.cpp:683-697`), so each element's own
// block is congruent with that element's signs (`A ← S A S` on the block);
// `SparseMatrix::AddSubMatrix` supplies the per-entry product of the row and
// column signs (`linalg/sparsemat.cpp:2795-2811`).
#[allow(clippy::too_many_arguments)]
fn add_face_penalty_hcurl<M: MeshTopology>(
    coo: &mut CooMatrix<f64>, mesh: &M,
    hcurl_space: &dyn FESpace<Mesh=M>,
    el: u32, er: u32, fnodes: &[u32], alpha: f64, qo: u8,
) {
    let dim = mesh.dim() as usize;
    let order = hcurl_space.order() as usize;
    let nd_elem: Box<dyn VectorReferenceElement> = if dim == 2 {
        Box::new(TriNDk::new(order))
    } else {
        Box::new(TetNDk::new(order))
    };
    let ne = nd_elem.n_dofs();
    let qf = wg_face_rule(dim, qo);
    let signs_l = hcurl_space.element_signs(el);
    let dofs_l: Vec<usize> = hcurl_space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = if el != er {
        hcurl_space.element_dofs(er).iter().map(|&d| d as usize).collect()
    } else { dofs_l.clone() };

    for (qi, xi) in qf.points.iter().enumerate() {
        // D810-1: isoparametric measure + composed reference point.
        let g = wg_face_point(mesh, el, fnodes, xi);
        let w = qf.weights[qi] * g.nor_mag();
        let mut pb = vec![0.0_f64; ne * dim];
        nd_elem.eval_basis_vec(&g.eip, &mut pb);
        for i in 0..ne { for j in 0..ne {
            let s_i = signs_l.and_then(|s| s.get(i)).copied().unwrap_or(1.0);
            let s_j = signs_l.and_then(|s| s.get(j)).copied().unwrap_or(1.0);
            let v = alpha * w * (0..dim).map(|d| pb[i*dim+d] * pb[j*dim+d]).sum::<f64>();
            if v.abs() > 1e-30 { coo.add(dofs_l[i], dofs_l[j], s_i * s_j * v); }
        }}
        if el != er {
            let signs_r = hcurl_space.element_signs(er);
            let gr = wg_face_point(mesh, er, fnodes, xi);
            let mut pb = vec![0.0_f64; ne * dim];
            nd_elem.eval_basis_vec(&gr.eip, &mut pb);
            for i in 0..ne { for j in 0..ne {
                let s_i = signs_r.and_then(|s| s.get(i)).copied().unwrap_or(1.0);
                let s_j = signs_r.and_then(|s| s.get(j)).copied().unwrap_or(1.0);
                let v = alpha * w * (0..dim).map(|d| pb[i*dim+d] * pb[j*dim+d]).sum::<f64>();
                if v.abs() > 1e-30 { coo.add(dofs_r[i], dofs_r[j], s_i * s_j * v); }
            }}
        }
    }
}

// ─── WG Maxwell assembly ──────────────────────────────────────────────────

/// Assemble the WG Maxwell stiffness matrix for κ = 1 (vacuum).
///
/// # Arguments
/// * `hcurl_space` — Nédélec HCurl space of order k
/// * `quad_order` — volume/face quadrature order
/// * `penalty` — face penalty coefficient (h^{-1} scaling applied internally)
/// * `dirichlet_bc` — (dof, value) pairs for essential BC on E_tan
pub fn assemble_wg_maxwell<M, S>(
    hcurl_space: &S,
    quad_order: u8,
    penalty: f64,
    dirichlet_bc: &[(usize, f64)],
) -> (CsrMatrix<f64>, Vec<f64>)
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let mesh = hcurl_space.mesh();
    let dim = mesh.dim() as usize;
    let n = hcurl_space.n_dofs();
    let ne = mesh.n_elements();
    let order = hcurl_space.order() as usize;

    let mut coo = CooMatrix::new(n, n);
    let mut rhs = vec![0.0_f64; n];

    // ── Volume stiffness: A_elem = C_w^T * M_Σ^{-1} * C_w ──────────────────
    for e in 0..ne as u32 {
        // D1080: the element's H(curl) orientation signs — the volume block is
        // scattered through the signed `GetElementDofs` table (MFEM
        // `bilinearform.cpp:420` + `sparsemat.cpp:2795-2811`).
        let signs = hcurl_space.element_signs(e);
        let (Cw, Ms) = weak_curl_matrix(mesh, e, dim, order, quad_order, signs);
        let n_v = Cw.nrows(); let n_s = Cw.ncols();
        let X = Ms.clone().lu().solve(&Cw.transpose()).unwrap_or(DMatrix::zeros(n_s, n_v));
        let A_elem = &Cw * X;
        let dofs: Vec<usize> = hcurl_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for i in 0..n_v { for j in 0..n_v {
            let v = A_elem[(i, j)];
            if v.abs() > 1e-30 { coo.add(dofs[i], dofs[j], v); }
        }}
    }

    // ── Face penalty (tangential jump stabilizer) ──────────────────────────
    let interior_faces = crate::InteriorFaceList::build(mesh);
    for f in &interior_faces.faces {
        let h = wg_face_measure(mesh, f.elem_left, &f.face_nodes, quad_order);
        let alpha = penalty / h.max(1e-14);
        add_face_penalty_hcurl(&mut coo, mesh, hcurl_space, f.elem_left, f.elem_right, &f.face_nodes, alpha, quad_order);
    }
    let fe_map = wg_boundary_face_map(mesh);
    for bf in mesh.face_iter() {
        if let Some(&el) = fe_map.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            let h = wg_face_measure(mesh, el, &fnodes, quad_order);
            let alpha = penalty / h.max(1e-14);
            add_face_penalty_hcurl(&mut coo, mesh, hcurl_space, el, el, &fnodes, alpha, quad_order);
        }
    }

    // ── Dirichlet BC ───────────────────────────────────────────────────────
    for &(dof, val) in dirichlet_bc {
        if dof < n { rhs[dof] = val; }
    }

    (coo.into_csr(), rhs)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::HCurlSpace;

    /// WG Maxwell matrix must be symmetric positive semi-definite.
    #[test]
    fn wg_maxwell_2d_spd() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 1);
        let (k, _) = assemble_wg_maxwell(&hcurl, 3, 10.0, &[]);
        let n = hcurl.n_dofs();

        // Check symmetry
        let mut max_asym: f64 = 0.0;
        for i in 0..n.min(50) {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            for p in s..e {
                let j = k.col_idx[p] as usize;
                if j < n.min(50) {
                    let vij = k.values[p];
                    let s2 = k.row_ptr[j]; let e2 = k.row_ptr[j + 1];
                    for q in s2..e2 {
                        if k.col_idx[q] == i as u32 {
                            max_asym = max_asym.max((vij - k.values[q]).abs());
                        }
                    }
                }
            }
        }
        assert!(max_asym < 1e-12, "A block symmetry violated: {max_asym}");

        // Check no zero diagonals
        for i in 0..n {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            let mut diag = 0.0;
            for p in s..e {
                if k.col_idx[p] == i as u32 { diag = k.values[p]; break; }
            }
            assert!(diag > 0.0, "Zero diagonal at DOF {i}");
        }
    }

    /// WG Maxwell 2D matrix must be invertible (no zero rows).
    #[test]
    fn wg_maxwell_2d_no_zero_rows() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 1);
        let (k, _) = assemble_wg_maxwell(&hcurl, 3, 10.0, &[]);
        let n = hcurl.n_dofs();
        for i in 0..n {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            let mut row_sum = 0.0;
            for p in s..e { row_sum += k.values[p].abs(); }
            assert!(row_sum > 1e-14, "Zero row {i}");
        }
    }
}
