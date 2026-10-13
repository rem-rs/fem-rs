//! Acoustic–structural interface coupling matrix.
//!
//! The coupling matrix referenced by `pro_physics::nvh::acoustic_coupling`
//! (audit 2026-10 priority 6: "doc引用的 fem-assembly 耦合矩阵不存在").
//!
//! # Definition and sign convention (derived, pinned by unit referees)
//!
//! With `n̂` the **outward** unit normal of the acoustic (fluid) domain on
//! the interface Γ (pointing into the structure), the coupling matrix
//!
//! ```text
//! C[(c, s), a] = ρ_f · ∫_Γ (n̂·e_c) · N_s(s) · N_a(a) dΓ
//! ```
//!
//! rows are structural vector DOFs (block layout of `VectorH1Space`:
//! global = `c·n_scalar + s`), columns are acoustic pressure DOFs.
//!
//! The coupled harmonic (e^{iωt}) block system is
//!
//! ```text
//! [ K_s − ω²M_s + iωC_s     −C            ] [u]   [f_s]
//! [ −ρ_f·ω²·Cᵀ             K_a − ω²M_a    ] [p] = [f_a]
//! ```
//!
//! Derivation:
//! * Structure: the fluid pressure loads the structure with the surface
//!   traction `t = p·n̂` (a positive pressure pushes along `n̂` — think of a
//!   pressurised balloon), so the consistent load is `F = ∫ (N_s·n̂) p dΓ
//!   = C·p`, i.e. block (1,2) = `−C` after moving to the left-hand side.
//! * Fluid: linearised Euler `iωρ_f v = −∇p` with the kinematic interface
//!   condition `v·n̂ = iω u_n` gives `∂p/∂n = ρ_f ω² u_n`; the Helmholtz
//!   weak form's boundary term `∫ q ∂p/∂n dΓ = ρ_f ω² (Cᵀu)_q` lands on the
//!   right-hand side, i.e. block (2,1) = `−ρ_f ω² Cᵀ`.
//!
//! Consistency check (the referee's physics pin): a piston of mass m on a
//! spring k closing an air column has, in the low-frequency limit, the
//! characteristic equation `(k_s − ω²m_s)(k_a − ω²m_a) − ρ_f ω² C² = 0`,
//! whose roots *raise* the in-vacuo structural frequency (the trapped air
//! acts as an added spring).  The sign pair `(−C, −ρ_f ω² Cᵀ)` is the one
//! consistent with that limit; the reversed sign on block (2,1) would lower
//! it.  `validate_acoustic_coupling::decoupling_limit…` and the unit
//! referees below pin this.
//!
//! # Scope
//!
//! 2-D straight boundary edges (`Line2` faces) with P1 traces are exact;
//! anything else (3-D faces, curved geometry, higher-order traces) is a hard
//! [`fem_core::FemError`] rather than an approximation.

use fem_core::{FemError, FemResult};
use fem_element::lagrange::SegP1;
use fem_element::reference::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

/// Assemble the vibro-acoustic coupling matrix `C` (rows = structural vector
/// DOFs in block layout, columns = acoustic pressure DOFs).
///
/// # Arguments
/// * `mesh`                — the acoustic-domain mesh; interface faces are the
///   boundary faces whose tag is in `tags` (the face normal is taken with
///   respect to *this* domain, so the mesh must be the fluid side).
/// * `n_struct_dofs`       — total structural vector DOFs.
/// * `struct_face_dofs`    — `face → structural vector DOFs` on that face,
///   block layout: index `c·n_face_scalar + k` is component `c` of scalar
///   face DOF `k` (the face-level mirror of `VectorH1Space`'s `byNODES`
///   global layout).
/// * `struct_dim`          — number of structural components (2 in 2-D).
/// * `acoustic_n_dofs`     — total acoustic pressure DOFs.
/// * `acoustic_face_dofs`  — `face → acoustic DOFs` on that face.
/// * `rho_fluid`           — fluid density ρ_f.
/// * `tags`                — interface boundary tags.
/// * `quad_order`          — quadrature accuracy order on the face element.
pub fn assemble_acoustic_coupling(
    mesh: &(dyn MeshTopology + Sync),
    n_struct_dofs: usize,
    struct_face_dofs: &(dyn Fn(u32) -> Vec<u32> + Sync),
    struct_dim: usize,
    acoustic_n_dofs: usize,
    acoustic_face_dofs: &(dyn Fn(u32) -> Vec<u32> + Sync),
    rho_fluid: f64,
    tags: &[i32],
    quad_order: u8,
) -> FemResult<CsrMatrix<f64>> {
    if mesh.dim() != 2 {
        return Err(FemError::Other(format!(
            "assemble_acoustic_coupling: only 2-D interfaces supported, mesh dim = {}",
            mesh.dim()
        )));
    }
    if mesh.geom_order() > 1 {
        return Err(FemError::Other(
            "assemble_acoustic_coupling: curved (geom_order > 1) interfaces not supported".to_string(),
        ));
    }

    let seg = SegP1;
    let quad = seg.quadrature(quad_order);
    let mut phi = vec![0.0_f64; seg.n_dofs()];
    let mut coo = CooMatrix::<f64>::new(n_struct_dofs, acoustic_n_dofs);

    for f in mesh.face_iter() {
        if !tags.contains(&mesh.face_tag(f)) {
            continue;
        }
        // D817-1 (mesh.cpp:1312): MFEM answers INVALID for a boundary element
        // whose face is true interior and its callers skip it.
        if mesh.bdr_face_true_interior(f) {
            continue;
        }
        let nodes = mesh.face_nodes(f);
        if nodes.len() != 2 {
            return Err(FemError::Other(format!(
                "assemble_acoustic_coupling: interface face {f} has {} nodes; only Line2 (2-D) faces are supported",
                nodes.len()
            )));
        }
        let v_dofs = struct_face_dofs(f);
        let a_dofs = acoustic_face_dofs(f);
        let n_face_scalar = v_dofs.len() / struct_dim;
        if v_dofs.len() % struct_dim != 0 || n_face_scalar != a_dofs.len() {
            return Err(FemError::Other(format!(
                "assemble_acoustic_coupling: face {f} DOF lists mismatch ({} struct vs {} acoustic)",
                v_dofs.len(),
                a_dofs.len()
            )));
        }

        // Outward normal of a straight edge, MFEM `CalcOrtho` (2×1 Jacobian):
        // rotate the tangent t = p1 − p0 by −90° → (t_y, −t_x)/|t|.
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let (tx, ty) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = (tx * tx + ty * ty).sqrt();
        if len < 1e-300 {
            return Err(FemError::Other(format!(
                "assemble_acoustic_coupling: degenerate (zero-length) interface face {f}"
            )));
        }
        let normal = [ty / len, -tx / len];

        let mut xp = [0.0_f64; 1]; // face reference coordinate (1-D)
        for (q, point) in quad.points.iter().enumerate() {
            xp[0] = point[0];
            seg.eval_basis(&xp, &mut phi);
            let w = quad.weights[q] * len;
            for c in 0..struct_dim {
                let n_c = normal[c];
                if n_c.abs() < 1e-300 {
                    continue; // component orthogonal to n̂: zero block, skip
                }
                for (i, &vs) in v_dofs.iter().skip(c * n_face_scalar).take(n_face_scalar).enumerate() {
                    for (j, &a) in a_dofs.iter().enumerate() {
                        let val = w * rho_fluid * n_c * phi[i] * phi[j];
                        if val.abs() > 0.0 {
                            coo.add(vs as usize, a as usize, val);
                        }
                    }
                }
            }
        }
    }

    Ok(coo.into_csr())
}

/// Space-based convenience wrapper.
///
/// Assembles `C` for a [`VectorH1Space`] structural field against an
/// [`H1Space`] acoustic field on the same mesh.  Both spaces must be P1
/// (the trace DOF map used here is the P1 vertex map, matching
/// [`crate::assembler::face_dofs_p1`]).
pub fn assemble_acoustic_coupling_spaces<M>(
    struct_space: &fem_space::vector_h1::VectorH1Space<M>,
    acoustic_space: &fem_space::h1::H1Space<M>,
    rho_fluid: f64,
    tags: &[i32],
    quad_order: u8,
) -> FemResult<CsrMatrix<f64>>
where
    M: MeshTopology + Clone + Send + Sync + 'static,
{
    if struct_space.order() != 1 || acoustic_space.order() != 1 {
        return Err(FemError::Other(format!(
            "assemble_acoustic_coupling_spaces: P1 spaces required, got order {} / {}",
            struct_space.order(),
            acoustic_space.order()
        )));
    }
    let dim = struct_space.mesh().dim() as usize;
    let n_scalar = struct_space.n_scalar_dofs();
    let mesh = struct_space.mesh().clone();

    // P1 traces: scalar face DOF = vertex id; vector DOF = c·n_scalar + vertex.
    let struct_face = move |f: u32| -> Vec<u32> {
        let nodes = mesh.face_nodes(f);
        let mut out = Vec::with_capacity(nodes.len() * dim);
        for c in 0..dim as u32 {
            for &n in nodes.iter() {
                out.push(c * n_scalar as u32 + n);
            }
        }
        out
    };
    let amesh = acoustic_space.mesh().clone();
    let acoustic_face = move |f: u32| amesh.face_nodes(f).to_vec();

    assemble_acoustic_coupling(
        struct_space.mesh(),
        struct_space.n_dofs(),
        &struct_face,
        dim,
        acoustic_space.n_dofs(),
        &acoustic_face,
        rho_fluid,
        tags,
        quad_order,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::h1::H1Space;
    use fem_space::vector_h1::VectorH1Space;

    const RHO: f64 = 5.0;

    fn spaces(mesh: &Mesh<2>) -> (VectorH1Space<Mesh<2>>, H1Space<Mesh<2>>) {
        (
            VectorH1Space::new(mesh.clone(), 1, 2),
            H1Space::new(mesh.clone(), 1),
        )
    }

    /// Closed-form referee on the right edge (tag 2, n̂ = (+1, 0)) of the
    /// 2×2 unit-square quad mesh.
    ///
    /// P1 traces, two segments of length h = 1/2; right-edge vertex dofs are
    /// {2, 5, 8} (`nid = j·3 + i` with i = 2):
    /// `∫ N_i N_j dx = h/3` (diagonal), `h/6` (off-diagonal), so with ρ = 5
    ///
    /// ```text
    ///            a=2    a=5    a=8      (acoustic dofs on x=1)
    /// row s=2  [  5/6,  5/12,   0  ]
    /// row s=5  [ 5/12,  5/3 , 5/12 ]      (s=5 joins both segments)
    /// row s=8  [   0 ,  5/12,  5/6 ]
    /// ```
    /// y-component rows are exactly zero (n̂_y = 0 on a vertical interface).
    #[test]
    fn coupling_matrix_closed_form_right_edge() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (vs, as_) = spaces(&mesh);
        let c = assemble_acoustic_coupling_spaces(&vs, &as_, RHO, &[2], 4)
            .expect("coupling assembly");
        assert_eq!((c.nrows, c.ncols), (18, 9));

        let d = c.to_dense();
        let want = |s: usize, a: usize| -> f64 {
            let diag = RHO / 6.0; // h/3 with h = 1/2
            let off = RHO / 12.0; // h/6
            match (s, a) {
                (2, 2) | (8, 8) => diag,
                (5, 5) => 2.0 * diag,
                (2, 5) | (5, 2) | (5, 8) | (8, 5) => off,
                _ => 0.0,
            }
        };
        for &s in &[2usize, 5, 8] {
            for &a in &[2usize, 5, 8] {
                let got = d[s * 9 + a];
                assert!(
                    (got - want(s, a)).abs() < 1e-14,
                    "C[{s},{a}] = {got:.3e}, want {:.6e}",
                    want(s, a)
                );
                // y-component rows: interface is vertical → n̂_y = 0.
                assert_eq!(d[(9 + s) * 9 + a], 0.0, "y row [{},{}]", 9 + s, a);
            }
        }
        // Every other row must be zero (no DOFs off the interface).
        for i in 0..18usize {
            if ![2, 5, 8, 11, 14, 17].contains(&i) {
                for j in 0..9 {
                    assert_eq!(d[i * 9 + j], 0.0, "row {i} col {j} must vanish");
                }
            }
        }
    }

    /// Sign referee: on the left edge (tag 4) the outward normal of the
    /// acoustic domain is n̂ = (−1, 0), so the x-component rows flip sign.
    /// Left-edge vertex dofs are {0, 3, 6} (`nid = j·3 + i` with i = 0).
    #[test]
    fn coupling_normal_sign_left_edge() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (vs, as_) = spaces(&mesh);
        let c = assemble_acoustic_coupling_spaces(&vs, &as_, RHO, &[4], 4)
            .expect("coupling assembly");
        let d = c.to_dense();
        assert!((d[0 * 9 + 0] - (-RHO / 6.0)).abs() < 1e-14, "C[0,0] = {}", d[0]);
        assert!((d[3 * 9 + 3] - (-2.0 * RHO / 6.0)).abs() < 1e-14, "C[3,3] = {}", d[3 * 9 + 3]);
        assert!((d[6 * 9 + 6] - (-RHO / 6.0)).abs() < 1e-14, "C[6,6] = {}", d[6 * 9 + 6]);
    }

    /// Conservation referee: C·1 over the acoustic columns equals
    /// ρ_f·n̂_x·∫N_a dΓ (the structural traces on the interface sum to the
    /// constant 1, so the column sums are exact one-line integrals).
    #[test]
    fn coupling_column_sums_conservation() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (vs, as_) = spaces(&mesh);
        let c = assemble_acoustic_coupling_spaces(&vs, &as_, RHO, &[2], 4)
            .expect("coupling assembly");
        let ones = vec![1.0_f64; c.ncols];
        let mut y = vec![0.0_f64; c.nrows];
        c.spmv(&ones, &mut y);
        // ρ_f·n̂_x·∫N_a dΓ on the right edge = ρ_f·[1/4, 1/2, 1/4]·n̂_x
        // = [5/4, 5/2, 5/4] at acoustic dofs 2, 5, 8 (all other rows 0).
        for (a, want) in [(2usize, 1.25), (5, 2.5), (8, 1.25)] {
            assert!((y[a] - want).abs() < 1e-12, "col sum [{a}] = {}, want {want}", y[a]);
        }
        for i in 0..c.nrows {
            if ![2, 5, 8].contains(&i) {
                assert!(y[i].abs() < 1e-12, "row {i} must not receive coupling");
            }
        }
    }

    /// Rank referee: rank(C) = number of interface acoustic DOFs (3 on the
    /// 2-segment right edge) — the coupling injects exactly that many
    /// independent constraints into the block system.
    #[test]
    fn coupling_rank_equals_interface_dofs() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (vs, as_) = spaces(&mesh);
        let c = assemble_acoustic_coupling_spaces(&vs, &as_, RHO, &[2], 4)
            .expect("coupling assembly");
        let d = c.to_dense();
        // Gaussian elimination rank of the 18×9 matrix.
        let (rows, cols) = (c.nrows, c.ncols);
        let mut m = d;
        let mut rank = 0usize;
        for col in 0..cols {
            let pivot = (rank..rows).find(|&r| m[r * cols + col].abs() > 1e-12);
            if let Some(p) = pivot {
                // Swap *whole rows* (rank ↔ p), not scalars.
                for cc in 0..cols {
                    m.swap(rank * cols + cc, p * cols + cc);
                }
                let piv = m[rank * cols + col];
                for r in rank + 1..rows {
                    let factor = m[r * cols + col] / piv;
                    if factor != 0.0 {
                        for cc in col..cols {
                            m[r * cols + cc] -= factor * m[rank * cols + cc];
                        }
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(rank, 3, "rank(C) must equal the 3 interface acoustic DOFs");
    }

    /// Scope referee: a 3-D interface is a hard error, not a silent
    /// approximation.
    #[test]
    fn three_d_interface_is_error() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let vs = VectorH1Space::new(mesh.clone(), 1, 3);
        let as_ = H1Space::new(mesh.clone(), 1);
        let res = assemble_acoustic_coupling_spaces(&vs, &as_, RHO, &[1], 3);
        assert!(res.is_err(), "3-D interface must be a hard error");
    }
}
